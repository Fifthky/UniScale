"""Minimal Time-Series-Library DLinear and PatchTST forecasting models.

The implementations are adapted from THUML Time-Series-Library under its MIT
license. They retain the long-term forecasting computation used by the source
models while exposing a small univariate interface for matched-history tasks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel size must be a positive odd integer")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        front = values[:, :1, :].repeat(1, padding, 1)
        end = values[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat([front, values, end], dim=1)
        return self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = _MovingAverage(kernel_size)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(values)
        return values - trend, trend


class DLinear(nn.Module):
    """Time-Series-Library DLinear long-term forecasting path."""

    def __init__(self, input_length: int, horizon: int, moving_average: int = 25):
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.decomposition = _SeriesDecomposition(moving_average)
        self.seasonal = nn.Linear(input_length, horizon)
        self.trend = nn.Linear(input_length, horizon)
        with torch.no_grad():
            self.seasonal.weight.fill_(1.0 / input_length)
            self.trend.weight.fill_(1.0 / input_length)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError(
                "DLinear expects scalar-series batches shaped (batch, time, 1)"
            )
        seasonal, trend = self.decomposition(values)
        seasonal = self.seasonal(seasonal.permute(0, 2, 1))
        trend = self.trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)


class _PositionalEmbedding(nn.Module):
    def __init__(self, width: int, maximum_length: int = 5000):
        super().__init__()
        embedding = torch.zeros(maximum_length, width).float()
        position = torch.arange(maximum_length).float().unsqueeze(1)
        divisor = (
            torch.arange(0, width, 2).float()
            * -(math.log(10000.0) / width)
        ).exp()
        embedding[:, 0::2] = torch.sin(position * divisor)
        embedding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("embedding", embedding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.embedding[:, : values.size(1)]


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        width: int,
        patch_length: int,
        stride: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_length, width, bias=False)
        self.position_embedding = _PositionalEmbedding(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, int]:
        variable_count = values.shape[1]
        values = self.padding(values)
        values = values.unfold(-1, self.patch_length, self.stride)
        values = values.reshape(
            values.shape[0] * values.shape[1], values.shape[2], values.shape[3]
        )
        embedded = self.value_embedding(values) + self.position_embedding(values)
        return self.dropout(embedded), variable_count


class _FullAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attention, values).contiguous()


class _AttentionLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("PatchTST width must be divisible by attention heads")
        head_width = width // heads
        self.heads = heads
        self.query = nn.Linear(width, head_width * heads)
        self.key = nn.Linear(width, head_width * heads)
        self.value = nn.Linear(width, head_width * heads)
        self.output = nn.Linear(head_width * heads, width)
        self.attention = _FullAttention(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        queries = self.query(values).view(batch, length, self.heads, -1)
        keys = self.key(values).view(batch, length, self.heads, -1)
        projected = self.value(values).view(batch, length, self.heads, -1)
        attended = self.attention(queries, keys, projected)
        return self.output(attended.view(batch, length, -1))


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward_width: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.attention = _AttentionLayer(width, heads, dropout)
        self.feedforward_in = nn.Conv1d(width, feedforward_width, kernel_size=1)
        self.feedforward_out = nn.Conv1d(feedforward_width, width, kernel_size=1)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        if activation not in {"relu", "gelu"}:
            raise ValueError("PatchTST activation must be relu or gelu")
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.dropout(self.attention(values))
        residual = values = self.norm1(values)
        residual = self.dropout(
            self.activation(self.feedforward_in(residual.transpose(1, 2)))
        )
        residual = self.dropout(self.feedforward_out(residual).transpose(1, 2))
        return self.norm2(values + residual)


class _Transpose(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.transpose(1, 2)


class PatchTST(nn.Module):
    """Time-Series-Library PatchTST long-term forecasting path."""

    def __init__(
        self,
        input_length: int,
        horizon: int,
        *,
        width: int = 512,
        heads: int = 8,
        encoder_layers: int = 2,
        feedforward_width: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_length: int = 16,
        stride: int = 8,
        internal_normalization: bool = True,
    ):
        super().__init__()
        if input_length < patch_length:
            raise ValueError("PatchTST input length must cover at least one patch")
        self.input_length = input_length
        self.horizon = horizon
        self.internal_normalization = internal_normalization
        self.embedding = _PatchEmbedding(
            width=width,
            patch_length=patch_length,
            stride=stride,
            dropout=dropout,
        )
        self.encoder = nn.ModuleList(
            [
                _EncoderLayer(
                    width=width,
                    heads=heads,
                    feedforward_width=feedforward_width,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.encoder_norm = nn.Sequential(
            _Transpose(), nn.BatchNorm1d(width), _Transpose()
        )
        patch_count = int((input_length - patch_length) / stride + 2)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(width * patch_count, horizon),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError(
                "PatchTST expects scalar-series batches shaped (batch, time, 1)"
            )
        if self.internal_normalization:
            means = values.mean(1, keepdim=True).detach()
            centered = values - means
            standard_deviation = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = centered / standard_deviation
        else:
            means = None
            standard_deviation = None
            normalized = values
        encoded, variable_count = self.embedding(normalized.permute(0, 2, 1))
        for layer in self.encoder:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)
        encoded = encoded.reshape(
            -1, variable_count, encoded.shape[-2], encoded.shape[-1]
        ).permute(0, 1, 3, 2)
        forecast = self.head(encoded).permute(0, 2, 1)
        if self.internal_normalization:
            return forecast * standard_deviation[:, :1, :] + means[:, :1, :]
        return forecast


def build_model(
    name: str,
    input_length: int,
    horizon: int,
    hyperparameters: dict[str, Any] | None = None,
) -> nn.Module:
    """Build an adapted univariate Time-Series-Library model."""
    options = dict(hyperparameters or {})
    normalized = name.lower()
    if normalized == "dlinear":
        allowed = {"moving_average"}
        unexpected = sorted(set(options).difference(allowed))
        if unexpected:
            raise ValueError(f"Unsupported DLinear hyperparameters: {unexpected}")
        return DLinear(input_length, horizon, **options)
    if normalized == "patchtst":
        allowed = {
            "width",
            "heads",
            "encoder_layers",
            "feedforward_width",
            "dropout",
            "activation",
            "patch_length",
            "stride",
            "internal_normalization",
        }
        unexpected = sorted(set(options).difference(allowed))
        if unexpected:
            raise ValueError(f"Unsupported PatchTST hyperparameters: {unexpected}")
        return PatchTST(input_length, horizon, **options)
    raise ValueError(f"Unsupported full-shot model: {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
