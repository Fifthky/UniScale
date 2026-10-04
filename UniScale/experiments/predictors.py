"""Notebook-derived predictors with explicit context and call-length control."""

from __future__ import annotations

import json
import importlib
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast, SampleForecast

from UniScale.model_registry import import_model_module

from .metrics import QUANTILE_LEVELS


def _target_length(entry: dict[str, Any]) -> int:
    target = np.asarray(entry["target"])
    if target.ndim not in (1, 2):
        raise ValueError(f"Expected one- or two-dimensional target, received {target.shape}")
    return int(target.shape[-1])


def _retryable_cuda_batch_error(error: RuntimeError) -> bool:
    """Return whether a CUDA launch can be retried with smaller batching."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error: invalid configuration argument",
            "cuda error: launch out of resources",
        )
    )


class Chronos2Predictor:
    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        batch_size: int,
        device: str,
        torch_dtype: str,
        pipeline: Any | None = None,
    ):
        import torch
        from chronos import BaseChronosPipeline, Chronos2Pipeline

        dtype = getattr(torch, torch_dtype)
        self.pipeline = pipeline or BaseChronosPipeline.from_pretrained(
            checkpoint, device_map=device, torch_dtype=dtype, local_files_only=True
        )
        if not isinstance(self.pipeline, Chronos2Pipeline):
            raise TypeError(f"Checkpoint {checkpoint} did not load as Chronos2Pipeline")
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.effective_batch_size = batch_size

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        inputs = [{"target": entry["target"]} for entry in entries]
        batch_size = self.batch_size
        while True:
            try:
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=inputs,
                    prediction_length=self.prediction_length,
                    batch_size=batch_size,
                    quantile_levels=QUANTILE_LEVELS,
                    predict_batches_jointly=True,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
        self.effective_batch_size = min(self.effective_batch_size, batch_size)

        values = torch.stack(quantiles).permute(0, 3, 2, 1).cpu().numpy()
        if np.asarray(entries[0]["target"]).ndim == 1:
            values = values.squeeze(-1)
        for item, entry in zip(values, entries):
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


def _flowstate_scale_factor(frequency: str, domain: str | None) -> float:
    """Return the frequency scale used by the official GIFT wrapper."""
    normalized = frequency.upper()
    human_weekly = domain in {"Transport", "Healthcare", "Sales"}
    base_season = 24.0
    if normalized == "4S":
        return base_season / 900.0
    if normalized == "10S":
        return base_season / 360.0
    if normalized in {"T", "MIN"}:
        return base_season / (24.0 * 60.0)
    if normalized.endswith("T") and normalized[:-1].isdigit():
        return base_season / (24.0 * 60.0 / int(normalized[:-1]))
    if normalized.endswith("MIN") and normalized[:-3].isdigit():
        return base_season / (24.0 * 60.0 / int(normalized[:-3]))
    if normalized == "H":
        return base_season / 24.0
    if normalized == "6H":
        return base_season / 4.0
    if normalized == "D":
        return base_season / (7.0 if human_weekly else 365.0)
    if normalized.endswith("D") and "WED" not in normalized:
        multiplier = int(normalized[:-1])
        return multiplier * base_season / (7.0 if human_weekly else 365.0)
    if normalized == "W" or normalized.startswith("W-"):
        return base_season / (365.0 / 7.0)
    if normalized == "M" or normalized.startswith("M-"):
        return base_season / 12.0
    if "Q" in normalized:
        return base_season / 4.0
    if "A" in normalized or "Y" in normalized:
        return base_season / 4.0
    raise ValueError(f"FlowState does not define a scale factor for frequency {frequency}")


class FlowStatePredictor:
    """Notebook-aligned GIFT adapter for Granite FlowState."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        device: str,
        frequency: str,
        domain: str | None,
        has_daily_cycle: bool,
    ):
        import torch

        if model.config.prediction_type != "quantile":
            raise ValueError("FlowState GIFT inference requires quantile prediction")
        self.torch = torch
        self.model = model
        self.model.eval()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.quantiles = [float(value) for value in model.config.quantiles]
        if self.quantiles != list(QUANTILE_LEVELS):
            raise ValueError(
                f"FlowState quantiles {self.quantiles} do not match {QUANTILE_LEVELS}"
            )
        scale_factor = _flowstate_scale_factor(frequency, domain)
        self.scale_factor = scale_factor if has_daily_cycle else scale_factor / 7.0
        native_span = max(
            1,
            int(model.config.decoder_patch_len / self.scale_factor + 1e-6),
        )
        future_mask_length = max(0, prediction_length - native_span)
        self.model_context_limit = min(
            16 * 1024,
            int(model.config.context_length / self.scale_factor),
        ) - future_mask_length
        if self.model_context_limit <= 0:
            raise ValueError(
                "FlowState prediction length leaves no room for observed context"
            )
        self.effective_context_limit = min(context_length, self.model_context_limit)
        self.effective_batch_size = batch_size

    @staticmethod
    def _prepare_target(target: Any) -> np.ndarray:
        values = np.asarray(target, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"FlowState UniScale adapter requires univariate targets, received {values.shape}"
            )
        finite = np.flatnonzero(~np.isnan(values))
        if finite.size == 0:
            return np.zeros_like(values)
        return values[int(finite[0]) :]

    def _forecast(self, entries: list[dict[str, Any]], batch_size: int) -> list[np.ndarray]:
        indexed = []
        for index, entry in enumerate(entries):
            target = self._prepare_target(entry["target"])[-self.effective_context_limit :]
            indexed.append((len(target), index, target))
        indexed.sort(key=lambda item: item[0])

        predictions: list[np.ndarray | None] = [None] * len(entries)
        offset = 0
        while offset < len(indexed):
            length = indexed[offset][0]
            group_end = offset
            while group_end < len(indexed) and indexed[group_end][0] == length:
                group_end += 1
            group = indexed[offset:group_end]
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                batch = self.torch.stack(
                    [self.torch.from_numpy(item[2]) for item in chunk], dim=1
                ).unsqueeze(-1)
                with self.torch.no_grad():
                    output = self.model(
                        past_values=batch.to(self.device),
                        scale_factor=self.scale_factor,
                        prediction_length=self.prediction_length,
                        batch_first=False,
                    ).prediction_outputs
                values = output.squeeze(-1).detach().cpu().numpy()
                strictly_positive = self.torch.all(
                    self.torch.nan_to_num(batch.squeeze(-1), nan=1.0) >= 0,
                    dim=0,
                ).cpu().numpy()
                for row, (_, original_index, _) in enumerate(chunk):
                    item = values[row]
                    if strictly_positive[row]:
                        item = np.maximum(item, 0.0)
                    predictions[original_index] = item
            offset = group_end
        if any(item is None for item in predictions):
            raise RuntimeError("FlowState did not produce one forecast per input")
        return [item for item in predictions if item is not None]

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                values = self._forecast(entries, batch_size)
                break
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.torch.cuda.empty_cache()
        self.effective_batch_size = min(self.effective_batch_size, batch_size)
        for item, entry in zip(values, entries):
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in self.quantiles],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


class PatchTSTFMPredictor:
    """Official PatchTST-FM GIFT predictor behind the UniScale interface."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        device: str,
    ):
        import torch

        self.torch = torch
        self.model = model
        self.model.eval()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        internal_forecast = max(
            prediction_length,
            int(model.config.d_patch) * max(int(model.config.pretrain_mask_cont), 2),
        )
        self.model_context_limit = int(model.config.context_length) - internal_forecast
        if self.model_context_limit <= 0:
            raise ValueError(
                "PatchTST-FM prediction length leaves no room for observed context"
            )
        self.effective_context_limit = min(context_length, self.model_context_limit)
        self.effective_batch_size = batch_size

    @staticmethod
    def _prepare_target(target: Any) -> np.ndarray:
        values = np.asarray(target, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"PatchTST-FM UniScale adapter requires univariate targets, received {values.shape}"
            )
        if np.isnan(values).all():
            return np.zeros_like(values)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=float(np.nanmean(values)))
        return values

    def _forecast(self, entries: list[dict[str, Any]], batch_size: int) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for raw in batcher(entries, batch_size=batch_size):
            targets = [
                self.torch.from_numpy(
                    self._prepare_target(entry["target"])[-self.effective_context_limit :]
                ).to(self.device)
                for entry in raw
            ]
            with self.torch.no_grad():
                model_outputs = self.model(
                    past_values=targets,
                    prediction_length=self.prediction_length,
                    quantile_levels=list(QUANTILE_LEVELS),
                )
            outputs.extend(
                item.squeeze(-1).detach().cpu().numpy()
                for item in model_outputs.quantile_outputs
            )
        return outputs

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                values = self._forecast(entries, batch_size)
                break
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.torch.cuda.empty_cache()
        self.effective_batch_size = min(self.effective_batch_size, batch_size)
        for item, entry in zip(values, entries):
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


class ChronosPredictor:
    """Adapter shared by the original Chronos-T5 and Chronos-Bolt models."""

    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        batch_size: int,
        num_samples: int,
        device: str,
        torch_dtype: str,
        pipeline: Any | None = None,
    ):
        import torch
        from chronos import BaseChronosPipeline

        self.pipeline = pipeline or BaseChronosPipeline.from_pretrained(
            checkpoint,
            device_map=device,
            torch_dtype=getattr(torch, torch_dtype),
            local_files_only=True,
        )
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.effective_batch_size = batch_size

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch
        from chronos import ForecastType

        entries = list(test_data_input)
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if self.pipeline.forecast_type == ForecastType.SAMPLES
            else {}
        )
        outputs = []
        batch_size = self.batch_size
        while True:
            try:
                outputs.clear()
                for batch in batcher(entries, batch_size=batch_size):
                    contexts = [torch.as_tensor(entry["target"]) for entry in batch]
                    outputs.append(
                        self.pipeline.predict(
                            contexts,
                            prediction_length=self.prediction_length,
                            **predict_kwargs,
                        ).cpu().numpy()
                    )
                values = np.concatenate(outputs)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
        self.effective_batch_size = min(self.effective_batch_size, batch_size)

        for item, entry in zip(values, entries):
            common = {
                "start_date": entry["start"] + _target_length(entry),
                "item_id": entry.get("item_id"),
            }
            if self.pipeline.forecast_type == ForecastType.SAMPLES:
                yield SampleForecast(samples=item, **common)
            elif self.pipeline.forecast_type == ForecastType.QUANTILES:
                yield QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=[str(level) for level in self.pipeline.quantiles],
                    **common,
                )
            else:
                raise ValueError(f"Unsupported Chronos forecast type: {self.pipeline.forecast_type}")


class MoiraiPredictor:
    """Adapter for Moirai 1.0 and 1.1 checkpoints."""

    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        past_feat_dynamic_real_dim: int,
        batch_size: int,
        num_samples: int,
        device: str,
        module: Any | None = None,
    ):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        self.model = MoiraiForecast(
            module=module or MoiraiModule.from_pretrained(
                checkpoint, local_files_only=True
            ),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=32,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
        self.batch_size = batch_size
        self.device = device
        self._runtime_batch_size = batch_size
        self.effective_batch_size = batch_size
        self.recovery_events: list[dict[str, int | str]] = []

    def _create_predictor(self, batch_size: int):
        return self.model.create_predictor(batch_size=batch_size, device=self.device)

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        batch_size = self._runtime_batch_size
        reductions = 0
        while True:
            predictor = self._create_predictor(batch_size)
            try:
                forecasts = list(predictor.predict(entries))
                break
            except RuntimeError as error:
                if not _retryable_cuda_batch_error(error):
                    raise
                if batch_size == 1:
                    raise
                previous = batch_size
                batch_size = max(1, batch_size // 2)
                reductions += 1
                self.recovery_events.append(
                    {
                        "reason": (
                            "cuda_out_of_memory"
                            if "out of memory" in str(error).lower()
                            else "cuda_launch_configuration"
                        ),
                        "from_batch_size": previous,
                        "to_batch_size": batch_size,
                    }
                )
                del predictor
                torch.cuda.empty_cache()
        self._runtime_batch_size = batch_size
        self.effective_batch_size = min(self.effective_batch_size, batch_size)
        if reductions:
            print(
                f"Moirai batch recovery succeeded at batch_size={batch_size} "
                f"after {reductions} reductions",
                flush=True,
            )
        yield from forecasts


class Moirai2Predictor:
    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        context_length: int,
        past_feat_dynamic_real_dim: int,
        batch_size: int,
        device: str,
        module: Any | None = None,
    ):
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.torch = torch
        self.batch_size = batch_size
        self.model = Moirai2Forecast(
            module=module or Moirai2Module.from_pretrained(checkpoint, local_files_only=True),
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        ).to(torch.device(device))

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                outputs = []
                for batch in batcher(entries, batch_size=batch_size):
                    outputs.append(self.model.predict([entry["target"] for entry in batch]))
                values = np.concatenate(outputs)
                break
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)

        for item, entry in zip(values, entries):
            if item.ndim == 3 and item.shape[-1] == 1:
                item = item.squeeze(-1)
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )




class Toto2Predictor:
    """Official Toto 2 GluonTS adapter shared by all public model sizes."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        past_feat_dynamic_real_dim: int,
        batch_size: int,
        device: str,
    ):
        from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig

        self.torch_model = model
        self.config = Toto2GluonTSModelConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            decode_block_size=None,
        )
        self.gluonts_model = Toto2GluonTSModel(model, self.config).to(device).eval()
        self.batch_size = batch_size
        self.device = device

    def _create_predictor(self, batch_size: int):
        return self.gluonts_model.create_predictor(
            batch_size=batch_size,
            device=self.device,
        )

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                forecasts = list(self._create_predictor(batch_size).predict(entries))
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        yield from forecasts


class TiRex1Predictor:
    """Official TiRex 1.1 GIFT adapter with frequency resampling."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        batch_size: int,
    ):
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                forecasts = list(
                    self.model.forecast_gluon(
                        entries,
                        prediction_length=self.prediction_length,
                        output_type="gluonts",
                        batch_size=batch_size,
                        resample_strategy="frequency",
                    )
                )
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        yield from forecasts


class TiRex2Predictor:
    """Official TiRex 2 GIFT notebook adapter using univariate GluonTS output."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        batch_size: int,
    ):
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        batch_size = self.batch_size
        while True:
            try:
                forecasts = list(
                    self.model.forecast_gluon(
                        entries,
                        prediction_length=self.prediction_length,
                        output_type="gluonts",
                        batch_size=batch_size,
                        multivariate=False,
                    )
                )
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        yield from forecasts


_TIMESFM_LEGACY_PROFILES = {
    "timesfm1": {
        "context_len": 512,
        "num_layers": 20,
        "use_positional_embedding": True,
    },
    "timesfm2": {
        "context_len": 2048,
        "num_layers": 50,
        "use_positional_embedding": False,
    },
}


def _timesfm25_module():
    return import_model_module("timesfm25")


def _load_timesfm25_checkpoint(checkpoint: str):
    """Load TimesFM 2.5 while bypassing incompatible HubMixin kwargs."""
    module = _timesfm25_module()
    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path / "config.json"
    weights_path = checkpoint_path / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "TimesFM 2.5 requires config.json and model.safetensors under "
            f"{checkpoint_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return module.TimesFM_2p5_200M_torch._from_pretrained(
        model_id=str(checkpoint_path),
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        config=config,
        torch_compile=False,
    )


def _rounded_timesfm_horizon(prediction_length: int) -> int:
    output_patch_length = 128
    return (
        (prediction_length + output_patch_length - 1) // output_patch_length
    ) * output_patch_length


def _timesfm_checkpoint_file(checkpoint: str) -> Path:
    path = Path(checkpoint)
    checkpoint_file = path / "torch_model.ckpt" if path.is_dir() else path
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"TimesFM torch_model.ckpt does not exist: {checkpoint_file}"
        )
    return checkpoint_file


class TimesFMLegacyPredictor:
    """GIFT notebook-compatible adapter for TimesFM 1.0 and 2.0."""

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        frequency: str,
        per_core_batch_size: int,
    ):
        from UniScale.vendor.timesfm_legacy import freq_map

        if tuple(model.quantiles) != tuple(QUANTILE_LEVELS):
            raise ValueError(
                f"TimesFM quantiles {model.quantiles} do not match evaluation "
                f"quantiles {QUANTILE_LEVELS}"
            )
        model.horizon_len = _rounded_timesfm_horizon(prediction_length)
        self.model = model
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.frequency = freq_map(frequency)
        self.per_core_batch_size = per_core_batch_size

    def _forecast(self, entries: list[dict[str, Any]], batch_size: int) -> np.ndarray:
        outputs = []
        self.model.global_batch_size = batch_size
        self.model.per_core_batch_size = self.per_core_batch_size
        for batch in batcher(entries, batch_size=batch_size):
            contexts = []
            for entry in batch:
                target = np.asarray(entry["target"])
                if target.ndim != 1:
                    raise ValueError(
                        f"TimesFM requires univariate targets, received {target.shape}"
                    )
                contexts.append(target[-self.context_length :])
            frequencies = [self.frequency] * len(contexts)
            _, full_predictions = self.model.forecast(
                contexts,
                frequencies,
                forecast_context_len=self.context_length,
                normalize=True,
            )
            quantiles = full_predictions[:, : self.prediction_length, 1:]
            outputs.append(quantiles.transpose(0, 2, 1))
        return np.concatenate(outputs)

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        import torch

        entries = list(test_data_input)
        if not entries:
            return
        batch_size = self.batch_size
        while True:
            try:
                values = self._forecast(entries, batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)

        for item, entry in zip(values, entries):
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


def _rounded_timesfm25_context(maximum: int, patch: int) -> int:
    if maximum <= 0 or patch <= 0:
        raise ValueError("maximum and patch must be positive")
    return min(15360, ((maximum + patch - 1) // patch) * patch)


class TimesFM25Predictor:
    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        per_core_batch_size: int,
        model: Any | None = None,
    ):
        timesfm = _timesfm25_module()

        if prediction_length > 1024:
            raise ValueError("TimesFM-2.5 continuous quantile calls are limited to 1024 steps")
        if context_length > 15360:
            raise ValueError("TimesFM-2.5 GIFT inference is limited to 15360 context steps")
        self.configs = timesfm
        self.model = model or _load_timesfm25_checkpoint(checkpoint)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.per_core_batch_size = per_core_batch_size

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        entries = list(test_data_input)
        outputs = []
        for batch in batcher(entries, batch_size=self.batch_size):
            contexts = [np.asarray(entry["target"])[-self.context_length :] for entry in batch]
            maximum = max(len(context) for context in contexts)
            patch = self.model.model.p
            compiled_context = _rounded_timesfm25_context(maximum, patch)
            self.model.compile(
                forecast_config=self.configs.ForecastConfig(
                    max_context=compiled_context,
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=self.per_core_batch_size,
                )
            )
            _, predictions = self.model.forecast(
                horizon=self.prediction_length,
                inputs=contexts,
            )
            outputs.append(predictions[:, : self.prediction_length, 1:].transpose(0, 2, 1))

        for item, entry in zip(np.concatenate(outputs), entries):
            yield QuantileForecast(
                forecast_arrays=item,
                forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


class SundialPredictor:
    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        num_samples: int,
        device: str,
        model: Any | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = torch.device(device)
        self.model = model or AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        ).to(self.device).eval()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_samples = num_samples

    def _left_pad(self, contexts: list[Any]):
        maximum = max(len(context) for context in contexts)
        padded = []
        for context in contexts:
            padding = self.torch.full(
                (maximum - len(context),),
                fill_value=self.torch.nan,
                device=context.device,
            )
            padded.append(self.torch.cat([padding, context]))
        return self.torch.stack(padded)

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        from gluonts.transform import LastValueImputation

        entries = list(test_data_input)
        outputs = []
        for batch in batcher(entries, batch_size=self.batch_size):
            contexts = [
                self.torch.as_tensor(np.asarray(entry["target"])[-self.context_length :], dtype=self.torch.float32)
                for entry in batch
            ]
            stacked = self._left_pad(contexts).cpu().numpy()
            stacked = np.vstack([LastValueImputation()(row) for row in stacked])
            tensor = self.torch.as_tensor(stacked, device=self.device)
            autocast = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                generated = self.model.generate(
                    tensor,
                    max_new_tokens=self.prediction_length,
                    revin=True,
                    num_samples=self.num_samples,
                )
            outputs.append(generated.detach().cpu().numpy())

        for item, entry in zip(np.concatenate(outputs), entries):
            yield SampleForecast(
                samples=item,
                start_date=entry["start"] + _target_length(entry),
                item_id=entry.get("item_id"),
            )


@dataclass(frozen=True)
class PredictorSettings:
    checkpoint: str
    prediction_length: int
    context_length: int
    batch_size: int
    device: str
    torch_dtype: str = "float32"
    num_samples: int = 100
    past_feat_dynamic_real_dim: int = 0
    frequency: str = "D"
    domain: str | None = None
    has_daily_cycle: bool = True
    target_dim: int = 1
    seed: int | None = None
    per_core_batch_size: int | None = None
    deterministic_algorithms: bool | None = None
    float32_matmul_precision: str | None = None


def configure_inference_runtime(settings: PredictorSettings) -> None:
    """Apply notebook-declared reproducibility settings before each forecast run."""
    if (
        settings.seed is None
        and settings.deterministic_algorithms is None
        and settings.float32_matmul_precision is None
    ):
        return

    import torch

    if settings.seed is not None:
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
    if settings.deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(settings.deterministic_algorithms)
    if settings.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(settings.float32_matmul_precision)


def load_backend(adapter: str, settings: PredictorSettings):
    if adapter == "flowstate":
        module = import_model_module(adapter)
        return module.FlowStateForPrediction.from_pretrained(
            settings.checkpoint,
            local_files_only=True,
        ).to(settings.device).eval()
    if adapter == "patchtstfm":
        module = import_model_module(adapter)
        return module.PatchTSTFMForPrediction.from_pretrained(
            settings.checkpoint,
            local_files_only=True,
        ).to(settings.device).eval()
    if adapter == "toto2":
        from toto2 import Toto2Model

        if settings.torch_dtype != "float32":
            raise ValueError("Toto 2 checkpoint inference requires torch_dtype=float32")
        return Toto2Model.from_pretrained(
            settings.checkpoint,
            map_location=settings.device,
            local_files_only=True,
        ).to(settings.device).eval()
    if adapter == "tirex1":
        from tirex.base import PretrainedModel

        if settings.torch_dtype != "float32":
            raise ValueError("TiRex 1.1 checkpoint inference requires torch_dtype=float32")
        if settings.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError(
                "TiRex 1.1 supports device cpu, cuda, or cuda:0; received "
                f"{settings.device}"
            )
        checkpoint = Path(settings.checkpoint)
        checkpoint_file = (
            checkpoint / "model.ckpt"
            if checkpoint.is_dir()
            else checkpoint
        )
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"TiRex 1.1 model.ckpt does not exist: {checkpoint_file}"
            )
        model_class = PretrainedModel.REGISTRY.get("TiRex")
        if model_class is None:
            raise RuntimeError("Installed tirex package does not register TiRex")
        backend = "torch" if settings.device == "cpu" else "cuda"
        return model_class.from_pretrained(
            str(checkpoint_file),
            backend=backend,
            device=settings.device,
            compile=False,
        )
    if adapter == "tirex2":
        from tirex2 import load_model

        if settings.torch_dtype != "float32":
            raise ValueError("TiRex 2 checkpoint inference requires torch_dtype=float32")
        if settings.device == "cuda:0":
            tirex_device = "cuda"
        elif settings.device in {"cuda", "cpu"}:
            tirex_device = settings.device
        else:
            raise ValueError(
                "TiRex 2 supports device cpu, cuda, or cuda:0; received "
                f"{settings.device}"
            )
        return load_model(
            settings.checkpoint,
            device=tirex_device,
            hf_kwargs={"local_files_only": True},
        )
    if adapter in _TIMESFM_LEGACY_PROFILES:
        from UniScale.vendor.timesfm_legacy import (
            TimesFmCheckpoint,
            TimesFmHparams,
            TimesFmTorch,
        )

        if settings.torch_dtype != "float32":
            raise ValueError(
                f"{adapter} uses the official float32 inference path; received "
                f"torch_dtype={settings.torch_dtype}"
            )
        if settings.device == "cpu":
            backend = "cpu"
        elif settings.device in {"cuda", "cuda:0"}:
            backend = "gpu"
        else:
            raise ValueError(
                f"{adapter} supports device cpu, cuda, or cuda:0; received "
                f"{settings.device}"
            )
        profile = _TIMESFM_LEGACY_PROFILES[adapter]
        return TimesFmTorch(
            hparams=TimesFmHparams(
                backend=backend,
                per_core_batch_size=settings.per_core_batch_size or 32,
                num_layers=profile["num_layers"],
                horizon_len=_rounded_timesfm_horizon(settings.prediction_length),
                context_len=profile["context_len"],
                use_positional_embedding=profile["use_positional_embedding"],
                output_patch_len=128,
            ),
            checkpoint=TimesFmCheckpoint(
                version="torch",
                path=str(_timesfm_checkpoint_file(settings.checkpoint)),
            ),
        )
    if adapter == "chronos":
        import torch
        from chronos import BaseChronosPipeline

        return BaseChronosPipeline.from_pretrained(
            settings.checkpoint,
            device_map=settings.device,
            torch_dtype=getattr(torch, settings.torch_dtype),
            local_files_only=True,
        )
    if adapter == "chronos2":
        import torch
        from chronos import BaseChronosPipeline

        return BaseChronosPipeline.from_pretrained(
            settings.checkpoint,
            device_map=settings.device,
            torch_dtype=getattr(torch, settings.torch_dtype),
            local_files_only=True,
        )
    if adapter == "moirai":
        from uni2ts.model.moirai import MoiraiModule

        return MoiraiModule.from_pretrained(settings.checkpoint, local_files_only=True)
    if adapter == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Module

        return Moirai2Module.from_pretrained(settings.checkpoint, local_files_only=True)
    if adapter == "timesfm25":
        return _load_timesfm25_checkpoint(settings.checkpoint)
    if adapter == "sundial":
        import torch
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            settings.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        ).to(torch.device(settings.device)).eval()
    raise ValueError(f"Unsupported adapter: {adapter}")


def create_predictor(adapter: str, settings: PredictorSettings, backend: Any | None = None):
    configure_inference_runtime(settings)
    if adapter == "flowstate":
        if backend is None:
            backend = load_backend(adapter, settings)
        return FlowStatePredictor(
            model=backend,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            device=settings.device,
            frequency=settings.frequency,
            domain=settings.domain,
            has_daily_cycle=settings.has_daily_cycle,
        )
    if adapter == "patchtstfm":
        if backend is None:
            backend = load_backend(adapter, settings)
        return PatchTSTFMPredictor(
            model=backend,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            device=settings.device,
        )
    if adapter == "toto2":
        if backend is None:
            backend = load_backend(adapter, settings)
        return Toto2Predictor(
            model=backend,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            target_dim=settings.target_dim,
            past_feat_dynamic_real_dim=settings.past_feat_dynamic_real_dim,
            batch_size=settings.batch_size,
            device=settings.device,
        )
    if adapter == "tirex1":
        if backend is None:
            backend = load_backend(adapter, settings)
        return TiRex1Predictor(
            model=backend,
            prediction_length=settings.prediction_length,
            batch_size=settings.batch_size,
        )
    if adapter == "tirex2":
        if backend is None:
            backend = load_backend(adapter, settings)
        return TiRex2Predictor(
            model=backend,
            prediction_length=settings.prediction_length,
            batch_size=settings.batch_size,
        )
    if adapter in _TIMESFM_LEGACY_PROFILES:
        if backend is None:
            backend = load_backend(adapter, settings)
        return TimesFMLegacyPredictor(
            model=backend,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            frequency=settings.frequency,
            per_core_batch_size=settings.per_core_batch_size or 32,
        )
    if adapter == "chronos":
        return ChronosPredictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            batch_size=settings.batch_size,
            num_samples=settings.num_samples,
            device=settings.device,
            torch_dtype=settings.torch_dtype,
            pipeline=backend,
        )
    if adapter == "chronos2":
        return Chronos2Predictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            batch_size=settings.batch_size,
            device=settings.device,
            torch_dtype=settings.torch_dtype,
            pipeline=backend,
        )
    if adapter == "moirai":
        return MoiraiPredictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            target_dim=settings.target_dim,
            past_feat_dynamic_real_dim=settings.past_feat_dynamic_real_dim,
            batch_size=settings.batch_size,
            num_samples=settings.num_samples,
            device=settings.device,
            module=backend,
        )
    if adapter == "moirai2":
        return Moirai2Predictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            past_feat_dynamic_real_dim=settings.past_feat_dynamic_real_dim,
            batch_size=settings.batch_size,
            device=settings.device,
            module=backend,
        )
    if adapter == "timesfm25":
        return TimesFM25Predictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            model=backend,
            per_core_batch_size=settings.per_core_batch_size or 128,
        )
    if adapter == "sundial":
        return SundialPredictor(
            checkpoint=settings.checkpoint,
            prediction_length=settings.prediction_length,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            num_samples=settings.num_samples,
            device=settings.device,
            model=backend,
        )
    raise ValueError(f"Unsupported adapter: {adapter}")
