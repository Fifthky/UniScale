"""Dataset-level full-shot training for matched-history experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from UniScale.vendor.timeseries_library import build_model, trainable_parameter_count


class UnsupportedFullShotCell(ValueError):
    """Raised when a declared dataset cell cannot form the training protocol."""


@dataclass(frozen=True)
class FullShotTrainingSettings:
    batch_size: int
    learning_rate: float = 1e-4
    minimum_epochs: int = 5
    maximum_epochs: int = 30
    patience_epochs: int = 5
    validation_origins_per_series: int = 4
    temporal_coverage_stride: int = 64
    learning_rate_decay_epochs: int = 10
    learning_rate_decay: float = 0.5
    gradient_clip_norm: float = 1.0
    minimum_training_windows: int = 16
    seed: int = 2026
    selection_improvement_tolerance: float = 1e-6
    normalization_policy: str = "window_std_v2"
    training_loss: str = "mse"

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
        batch_size: int,
    ) -> "FullShotTrainingSettings":
        settings = cls(
            batch_size=int(batch_size),
            learning_rate=float(payload.get("learning_rate", 1e-4)),
            minimum_epochs=int(payload.get("minimum_epochs", 5)),
            maximum_epochs=int(payload.get("maximum_epochs", 30)),
            patience_epochs=int(payload.get("patience_epochs", 5)),
            validation_origins_per_series=int(
                payload.get("validation_origins_per_series", 4)
            ),
            temporal_coverage_stride=int(payload.get("temporal_coverage_stride", 64)),
            learning_rate_decay_epochs=int(
                payload.get("learning_rate_decay_epochs", 10)
            ),
            learning_rate_decay=float(payload.get("learning_rate_decay", 0.5)),
            gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
            minimum_training_windows=int(payload.get("minimum_training_windows", 16)),
            seed=int(payload.get("seed", 2026)),
            selection_improvement_tolerance=float(
                payload.get("selection_improvement_tolerance", 1e-6)
            ),
            normalization_policy=str(payload.get("normalization_policy", "window_std_v2")),
            training_loss=str(payload.get("training_loss", "mse")),
        )
        if settings.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if settings.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < settings.minimum_epochs <= settings.maximum_epochs:
            raise ValueError("training epoch bounds are invalid")
        if settings.patience_epochs <= 0:
            raise ValueError("patience_epochs must be positive")
        if settings.validation_origins_per_series <= 0:
            raise ValueError("validation_origins_per_series must be positive")
        if settings.temporal_coverage_stride <= 0:
            raise ValueError("temporal_coverage_stride must be positive")
        if settings.learning_rate_decay_epochs <= 0:
            raise ValueError("learning_rate_decay_epochs must be positive")
        if not 0.0 < settings.learning_rate_decay <= 1.0:
            raise ValueError("learning_rate_decay must be in (0, 1]")
        if settings.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if settings.minimum_training_windows <= 0:
            raise ValueError("minimum_training_windows must be positive")
        if settings.selection_improvement_tolerance < 0:
            raise ValueError("selection_improvement_tolerance cannot be negative")
        if settings.normalization_policy not in {"window_std_v2", "series_window_std_v1"}:
            raise ValueError("Unknown full-shot normalization policy")
        if settings.training_loss not in {"mse", "mae"}:
            raise ValueError("Unknown full-shot training loss")
        return settings


@dataclass
class FullShotCheckpoint:
    model: Any
    fallback_values: np.ndarray
    selected_epoch: int
    selection_epochs_run: int
    final_training_epochs: int
    selection_legal_window_count: int
    full_legal_window_count: int
    selection_examples_per_epoch: int
    full_examples_per_epoch: int
    selection_training_examples_seen: int
    full_training_examples_seen: int
    selection_optimizer_steps: int
    final_optimizer_steps: int
    minimum_series_selection_windows: int
    minimum_series_full_training_windows: int
    minimum_series_selection_examples_per_epoch: int
    minimum_series_full_examples_per_epoch: int
    scalar_series_count: int
    validation_example_count: int
    validation_target_count: int
    parameter_count: int
    validation_mae: float | None
    input_length: int
    horizon: int
    target_dim: int
    interpolated_training_values: int
    selection_history: list[dict[str, float | int]]
    normalization_policy: str
    history_scales: np.ndarray
    selection_history_scales: np.ndarray
    refit_history: list[dict[str, float | int]]
    history_record_ids: tuple[str, ...] = ()
    selection_metric: str = "normalized_mae"
    selection_score: float | None = None


@dataclass(frozen=True)
class ChronologicalSelectionLayout:
    training_window_count: int
    validation_target_starts: tuple[int, ...]
    validation_horizon: int

    @property
    def first_validation_target_start(self) -> int:
        return self.validation_target_starts[0]

    @property
    def validation_origin_count(self) -> int:
        return len(self.validation_target_starts)


class TemporalCoverageWindowPool:
    """Draw one stratified sample per history segment for every scalar series."""

    def __init__(
        self,
        histories: Sequence[np.ndarray],
        window_counts: Sequence[int],
        input_length: int,
        horizon: int,
        coverage_stride: int,
        sampling_history_lengths: Sequence[int] | None = None,
    ):
        if not histories or len(histories) != len(window_counts):
            raise ValueError("histories and window_counts must be non-empty and aligned")
        if any(count <= 0 for count in window_counts):
            raise ValueError("every history must contribute at least one window")
        self.histories = tuple(histories)
        self.window_counts = np.asarray(window_counts, dtype=np.int64)
        self.input_length = int(input_length)
        self.horizon = int(horizon)
        self.coverage_stride = int(coverage_stride)
        if self.coverage_stride <= 0:
            raise ValueError("coverage_stride must be positive")
        if any(
            np.asarray(history).ndim != 2 or np.asarray(history).shape[1] != 1
            for history in histories
        ):
            raise ValueError("training histories must be scalar time series")
        self.history_lengths = np.asarray(
            [len(history) for history in histories] if sampling_history_lengths is None
            else sampling_history_lengths, dtype=np.int64
        )
        if len(self.history_lengths) != len(histories) or (self.history_lengths <= 0).any():
            raise ValueError("Sampling history lengths must align with scalar histories")
        self.examples_per_series = np.minimum(
            self.window_counts,
            np.maximum(
                1,
                np.ceil(self.history_lengths / self.coverage_stride).astype(np.int64),
            ),
        )
        self.total_window_count = int(self.window_counts.sum())
        self.examples_per_epoch = int(self.examples_per_series.sum())

    def iter_epoch(
        self,
        batch_size: int,
        random_generator: np.random.Generator,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield a shuffled epoch whose quota scales with series count and history."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        epoch_series: list[np.ndarray] = []
        epoch_starts: list[np.ndarray] = []
        for series_index, (window_count, quota) in enumerate(
            zip(self.window_counts, self.examples_per_series)
        ):
            edges = np.linspace(0, int(window_count), int(quota) + 1, dtype=np.int64)
            starts = random_generator.integers(edges[:-1], edges[1:])
            epoch_series.append(
                np.full(int(quota), series_index, dtype=np.int64)
            )
            epoch_starts.append(starts.astype(np.int64, copy=False))
        series_indices = np.concatenate(epoch_series)
        window_starts = np.concatenate(epoch_starts)
        order = random_generator.permutation(len(series_indices))
        for offset in range(0, len(order), batch_size):
            batch_order = order[offset : offset + batch_size]
            inputs: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for location in batch_order:
                series_index = int(series_indices[location])
                start = int(window_starts[location])
                history = self.histories[series_index]
                split = start + self.input_length
                inputs.append(history[start:split])
                targets.append(history[split : split + self.horizon])
            yield (
                np.stack(inputs).astype(np.float32, copy=False),
                np.stack(targets).astype(np.float32, copy=False),
            )


@dataclass(frozen=True)
class PreparedFullShotData:
    selection_pool: TemporalCoverageWindowPool
    full_pool: TemporalCoverageWindowPool
    validation_inputs: np.ndarray
    validation_targets: np.ndarray
    fallback_values: np.ndarray
    interpolated_values: int
    history_scales: np.ndarray
    selection_history_scales: np.ndarray
    validation_error_scales: np.ndarray | None = None
    validation_observed: np.ndarray | None = None
    validation_series_indices: np.ndarray | None = None


def _history_scales(histories: Sequence[np.ndarray]) -> np.ndarray:
    """Pool within-record variation per variate using training observations only.

    If a variate is constant in every record, use its pooled RMS level. An
    entirely zero variate has an explicitly defined unit scale. These cases
    provide a reversible transform without a near-zero per-window divisor.
    """
    dimensions = {history.shape[1] for history in histories}
    if len(dimensions) != 1:
        raise ValueError("Stable history scaling requires aligned target variates")
    width = dimensions.pop()
    variation = np.zeros(width, dtype=np.float64)
    energy = np.zeros(width, dtype=np.float64)
    count = 0
    for history in histories:
        values = np.asarray(history, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("History scale estimation requires finite training values")
        variation += np.var(values, axis=0) * len(values)
        energy += np.sum(values * values, axis=0)
        count += len(values)
    scales = np.sqrt(variation / count)
    scales = np.where(scales > 0, scales, np.sqrt(energy / count))
    scales = np.where(scales > 0, scales, 1.0)
    if not np.isfinite(scales).all():
        raise ValueError("Non-finite training-history scale")
    return scales


def supervised_window_count(history_length: int, input_length: int, horizon: int) -> int:
    return history_length - input_length - horizon + 1


def chronological_selection_layout(
    history_length: int,
    input_length: int,
    horizon: int,
    validation_origins_per_series: int,
    minimum_training_windows: int,
) -> ChronologicalSelectionLayout:
    """Reserve separated full-horizon origins after target-disjoint training."""
    last_validation_target_start = history_length - horizon
    earliest_validation_target_start = (
        input_length + horizon + minimum_training_windows - 1
    )
    available_origin_count = (
        last_validation_target_start - earliest_validation_target_start + 1
    )
    if available_origin_count <= 0:
        raise UnsupportedFullShotCell(
            "insufficient_target_disjoint_training_validation_support"
        )
    validation_span = min(horizon, (available_origin_count - 1) // 2)
    origin_count = min(validation_origins_per_series, validation_span + 1)
    first_validation_target_start = last_validation_target_start - validation_span
    validation_target_starts = tuple(
        int(value)
        for value in np.linspace(
            first_validation_target_start,
            last_validation_target_start,
            origin_count,
        ).round()
    )
    first_validation_input_start = validation_target_starts[0] - input_length
    training_window_count = first_validation_target_start - input_length - horizon + 1
    if (
        first_validation_input_start < 0
        or training_window_count < minimum_training_windows
    ):
        raise UnsupportedFullShotCell(
            "insufficient_target_disjoint_training_validation_support"
        )
    return ChronologicalSelectionLayout(
        training_window_count=training_window_count,
        validation_target_starts=validation_target_starts,
        validation_horizon=horizon,
    )


def interpolate_missing(
    history: np.ndarray,
    fallback_values: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Interpolate each scalar target using observed pre-origin values only."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected time-by-variable history, received {values.shape}")
    output = np.array(values, copy=True)
    positions = np.arange(len(output), dtype=np.float64)
    interpolated = 0
    fallback = None if fallback_values is None else np.asarray(fallback_values)
    if fallback is not None:
        if fallback.shape == (1,) and output.shape[1] > 1:
            fallback = np.repeat(fallback, output.shape[1])
        if fallback.shape != (output.shape[1],):
            raise ValueError("fallback_values must match the target dimension")
    for dimension in range(output.shape[1]):
        observed = np.isfinite(output[:, dimension])
        if not observed.any():
            if fallback is None or not np.isfinite(fallback[dimension]):
                raise UnsupportedFullShotCell("scalar_series_has_no_observed_history")
            output[:, dimension] = fallback[dimension]
            interpolated += len(output)
            continue
        missing = ~observed
        if missing.any():
            output[missing, dimension] = np.interp(
                positions[missing], positions[observed], output[observed, dimension]
            )
            interpolated += int(missing.sum())
    return output, interpolated


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _scalar_histories(histories: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Unfold dataset records into independent scalar histories."""
    if not histories:
        raise UnsupportedFullShotCell("dataset_has_no_base_series")
    scalar: list[np.ndarray] = []
    for history in histories:
        values = np.asarray(history, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise UnsupportedFullShotCell(
                f"unsupported_training_history_shape={values.shape}"
            )
        scalar.extend(values[:, index : index + 1] for index in range(values.shape[1]))
    return scalar


def _dataset_scalar_fallback(histories: Sequence[np.ndarray]) -> np.ndarray:
    observed = [history[np.isfinite(history)] for history in histories]
    present = [values for values in observed if values.size]
    if not present:
        raise UnsupportedFullShotCell("dataset_has_no_observed_training_values")
    return np.asarray([np.mean(np.concatenate(present))], dtype=np.float32)


def prepare_full_shot_data(
    histories: Sequence[np.ndarray],
    input_length: int,
    horizon: int,
    settings: FullShotTrainingSettings,
    seasonality: int = 1,
) -> PreparedFullShotData:
    """Build scalar-series pools and full-horizon chronological validation."""
    if settings.normalization_policy == "series_window_std_v1":
        return _prepare_series_window_data(histories, input_length, horizon, settings, seasonality)
    scalar_histories = _scalar_histories(histories)
    fallbacks = _dataset_scalar_fallback(scalar_histories)
    prepared: list[np.ndarray] = []
    selection_counts: list[int] = []
    full_counts: list[int] = []
    validation_inputs: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    interpolated_count = 0
    for history in scalar_histories:
        values, interpolated = interpolate_missing(history, fallbacks)
        layout = chronological_selection_layout(
            len(values),
            input_length,
            horizon,
            settings.validation_origins_per_series,
            settings.minimum_training_windows,
        )
        full_count = supervised_window_count(len(values), input_length, horizon)
        if full_count < settings.minimum_training_windows:
            raise UnsupportedFullShotCell("insufficient_full_training_windows")
        prepared.append(values)
        selection_counts.append(layout.training_window_count)
        full_counts.append(full_count)
        for target_start in layout.validation_target_starts:
            input_start = target_start - input_length
            validation_inputs.append(values[input_start:target_start])
            validation_targets.append(
                values[target_start : target_start + horizon]
            )
        interpolated_count += interpolated
    return PreparedFullShotData(
        selection_pool=TemporalCoverageWindowPool(
            prepared,
            selection_counts,
            input_length,
            horizon,
            settings.temporal_coverage_stride,
        ),
        full_pool=TemporalCoverageWindowPool(
            prepared,
            full_counts,
            input_length,
            horizon,
            settings.temporal_coverage_stride,
        ),
        validation_inputs=np.stack(validation_inputs).astype(np.float32, copy=False),
        validation_targets=np.stack(validation_targets).astype(np.float32, copy=False),
        fallback_values=fallbacks,
        interpolated_values=interpolated_count,
        history_scales=np.ones(1),
        selection_history_scales=np.ones(1),
    )


def _prepare_series_window_data(
    histories: Sequence[np.ndarray], input_length: int, horizon: int,
    settings: FullShotTrainingSettings, seasonality: int,
) -> PreparedFullShotData:
    """Fit per-record scales on selection prefixes and final training histories.

    Missing histories use an explicit zero fill. A constant nonzero history uses
    its RMS scale; an entirely zero history uses unit scale. Validation targets
    never set training scales or enter the training-side interpolation.
    """
    if seasonality <= 0:
        raise ValueError("Validation seasonality must be positive")
    selection, full, selection_counts, full_counts = [], [], [], []
    selection_scales, full_scales = [], []
    inputs, targets, error_scales, observed_masks, series_indices = [], [], [], [], []
    interpolated_count = 0
    for history in histories:
        raw = np.asarray(history, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.ndim != 2 or not len(raw):
            raise ValueError("Expected nonempty time-by-variate histories")
        layout = chronological_selection_layout(
            len(raw), input_length, horizon, settings.validation_origins_per_series,
            settings.minimum_training_windows,
        )
        boundary = layout.first_validation_target_start
        fallback = np.zeros(raw.shape[1], dtype=np.float32)
        prefix, _ = interpolate_missing(raw[:boundary], fallback)
        complete, count = interpolate_missing(raw, fallback)
        # Input interpolation at each validation origin is causal with respect
        # to that origin. The scored targets retain an observed-value mask.
        prefix_scale = _history_scales([prefix])
        complete_scale = _history_scales([complete])
        selection_scales.append(prefix_scale)
        full_scales.append(complete_scale)
        interpolated_count += count
        lag = seasonality if len(prefix) > seasonality else 1
        differences = np.abs(prefix[lag:].astype(np.float64) - prefix[:-lag])
        mase_scale = np.maximum(differences.mean(axis=0), 1e-5 * prefix_scale)
        for dimension in range(raw.shape[1]):
            selection.append(prefix[:, dimension:dimension+1] / prefix_scale[dimension])
            full.append(complete[:, dimension:dimension+1] / complete_scale[dimension])
            selection_counts.append(layout.training_window_count)
            full_counts.append(supervised_window_count(len(complete), input_length, horizon))
            for start in layout.validation_target_starts:
                available, _ = interpolate_missing(raw[:start, dimension:dimension+1], np.zeros(1))
                inputs.append(available[-input_length:] / prefix_scale[dimension])
                target = raw[start:start+horizon, dimension:dimension+1]
                observed = np.isfinite(target)
                targets.append(np.where(observed, target, 0.0) / prefix_scale[dimension])
                observed_masks.append(observed)
                error_scales.append(mase_scale[dimension] / prefix_scale[dimension])
                series_indices.append(len(selection) - 1)
    if not selection:
        raise UnsupportedFullShotCell("dataset_has_no_base_series")
    mask = np.asarray(observed_masks, dtype=bool)
    if not mask.any():
        raise UnsupportedFullShotCell("validation_has_no_observed_targets")
    return PreparedFullShotData(
        selection_pool=TemporalCoverageWindowPool(
            selection, selection_counts, input_length, horizon, settings.temporal_coverage_stride,
            [len(value) for value in full],
        ),
        full_pool=TemporalCoverageWindowPool(
            full, full_counts, input_length, horizon, settings.temporal_coverage_stride,
        ),
        validation_inputs=np.asarray(inputs, dtype=np.float32),
        validation_targets=np.asarray(targets, dtype=np.float32),
        fallback_values=np.zeros(1, dtype=np.float32),
        interpolated_values=interpolated_count,
        history_scales=np.asarray(full_scales),
        selection_history_scales=np.asarray(selection_scales),
        validation_error_scales=np.asarray(error_scales, dtype=np.float64),
        validation_observed=mask,
        validation_series_indices=np.asarray(series_indices, dtype=np.int64),
    )


def _normalize_batch(
    inputs: Any, policy: str = "window_std_v2",
) -> tuple[Any, Any, Any]:
    mean = inputs.mean(dim=1, keepdim=True)
    if policy not in {"window_std_v2", "series_window_std_v1"}:
        raise ValueError("Unknown batch normalization policy")
    if policy == "series_window_std_v1":
        # Inputs are already in their own record/variate history units.
        standard_deviation = inputs.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(
            1e-5
        )
    else:
        standard_deviation = inputs.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    return (inputs - mean) / standard_deviation, mean, standard_deviation


def _learning_rate(settings: FullShotTrainingSettings, epoch: int) -> float:
    decay_count = (epoch - 1) // settings.learning_rate_decay_epochs
    return settings.learning_rate * (settings.learning_rate_decay**decay_count)


def _train_epoch(
    model: Any,
    optimizer: Any,
    pool: TemporalCoverageWindowPool,
    settings: FullShotTrainingSettings,
    random_generator: np.random.Generator,
    device: Any,
    epoch: int,
) -> tuple[float, int, int]:
    import torch

    model.train()
    for group in optimizer.param_groups:
        group["lr"] = _learning_rate(settings, epoch)
    objective_sum = 0.0
    target_count = 0
    example_count = 0
    optimizer_steps = 0
    for inputs_array, targets_array in pool.iter_epoch(
        settings.batch_size, random_generator
    ):
        inputs = torch.from_numpy(inputs_array).to(device, non_blocking=True)
        targets = torch.from_numpy(targets_array).to(device, non_blocking=True)
        normalized_inputs, mean, standard_deviation = _normalize_batch(
            inputs, settings.normalization_policy,
        )
        normalized_targets = (targets - mean) / standard_deviation
        optimizer.zero_grad(set_to_none=True)
        forecasts = model(normalized_inputs)
        errors = forecasts - normalized_targets
        if settings.training_loss == "mae":
            losses = errors.abs()
        else:
            losses = errors**2
        loss = torch.mean(losses)
        if not torch.isfinite(loss) or not torch.isfinite(errors).all():
            raise FloatingPointError("Non-finite full-shot training loss or residual")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()
        objective_sum += float(loss.detach().cpu()) * int(errors.numel())
        target_count += int(errors.numel())
        example_count += len(inputs_array)
        optimizer_steps += 1
    if example_count != pool.examples_per_epoch or target_count <= 0:
        raise RuntimeError("Temporal coverage epoch produced an invalid sample count")
    return objective_sum / target_count, example_count, optimizer_steps


def _validation_score(
    model: Any,
    inputs_array: np.ndarray,
    targets_array: np.ndarray,
    batch_size: int,
    device: Any,
    normalization_policy: str = "window_std_v2",
    error_scales: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    series_indices: np.ndarray | None = None,
) -> float:
    """Use normalized MAE by default, or fixed-prefix stabilized MASE."""
    import torch

    model.eval()
    per_example: list[np.ndarray] = []
    target_horizon = targets_array.shape[1]
    with torch.no_grad():
        for start in range(0, len(inputs_array), batch_size):
            inputs = torch.from_numpy(inputs_array[start : start + batch_size]).to(
                device, non_blocking=True
            )
            targets = torch.from_numpy(targets_array[start : start + batch_size]).to(
                device, non_blocking=True
            )
            normalized_inputs, mean, standard_deviation = _normalize_batch(
                inputs, normalization_policy,
            )
            normalized_targets = (targets - mean) / standard_deviation
            normalized = model(normalized_inputs)[:, :target_horizon]
            if error_scales is None:
                errors = torch.mean(torch.abs(normalized - normalized_targets), dim=(1, 2))
            else:
                if observed is None or series_indices is None:
                    raise ValueError("Scaled validation requires observed masks and series identities")
                absolute = torch.abs(normalized * standard_deviation + mean - targets)
                mask = torch.from_numpy(observed[start:start+batch_size]).to(device)
                counts = mask.sum(dim=(1, 2))
                denominator = torch.as_tensor(error_scales[start:start+batch_size], device=device)
                errors = (absolute * mask).sum(dim=(1, 2)) / counts.clamp_min(1) / denominator
            if not torch.isfinite(errors).all():
                raise FloatingPointError("Non-finite full-shot validation error")
            per_example.append(errors.cpu().numpy())
    values = np.concatenate(per_example)
    if error_scales is not None:
        valid = observed.any(axis=(1, 2))
        # Each scalar series gets equal weight, including when some origins
        # have entirely missing targets. Such origins receive no score.
        if not valid.any():
            raise UnsupportedFullShotCell("validation_has_no_observed_targets")
        totals = np.bincount(series_indices[valid], weights=values[valid])
        counts = np.bincount(series_indices[valid])
        present = counts > 0
        return float(np.mean(totals[present] / counts[present]))
    return float(np.mean(values))


def _new_model_and_optimizer(
    model_name: str,
    hyperparameters: dict[str, Any],
    input_length: int,
    horizon: int,
    settings: FullShotTrainingSettings,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    _seed_everything(settings.seed)
    options = dict(hyperparameters)
    if model_name.lower() == "patchtst":
        options["internal_normalization"] = False
    model = build_model(model_name, input_length, horizon, options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    return model, optimizer


def _select_training_epochs(
    model_name: str,
    hyperparameters: dict[str, Any],
    prepared: PreparedFullShotData,
    input_length: int,
    horizon: int,
    settings: FullShotTrainingSettings,
    device: Any,
) -> tuple[int, int, int, float, int, list[dict[str, float | int]]]:
    model, optimizer = _new_model_and_optimizer(
        model_name, hyperparameters, input_length, horizon, settings, device
    )
    parameter_count = trainable_parameter_count(model)
    random_generator = np.random.default_rng(settings.seed + 1)
    best_epoch = 0
    best_mae = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    epochs_run = 0
    optimizer_steps_run = 0
    for epoch in range(1, settings.maximum_epochs + 1):
        training_objective, training_examples, optimizer_steps = _train_epoch(
            model,
            optimizer,
            prepared.selection_pool,
            settings,
            random_generator,
            device,
            epoch,
        )
        epochs_run = epoch
        optimizer_steps_run += optimizer_steps
        if epoch < settings.minimum_epochs:
            continue
        validation_mae = _validation_score(
            model,
            prepared.validation_inputs,
            prepared.validation_targets,
            settings.batch_size,
            device,
            settings.normalization_policy,
            prepared.validation_error_scales,
            prepared.validation_observed,
            prepared.validation_series_indices,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": _learning_rate(settings, epoch),
                "training_objective": training_objective,
                "training_examples": training_examples,
                "optimizer_steps": optimizer_steps,
                ("validation_stabilized_mase" if prepared.validation_error_scales is not None
                 else "validation_mae"): validation_mae,
            }
        )
        if validation_mae < best_mae - settings.selection_improvement_tolerance:
            best_mae = validation_mae
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= settings.patience_epochs:
            break
    if best_epoch <= 0 or not math.isfinite(best_mae):
        raise RuntimeError("Full-shot selection produced no finite validation checkpoint")
    del optimizer, model
    return (
        best_epoch,
        epochs_run,
        optimizer_steps_run,
        best_mae,
        parameter_count,
        history,
    )


def _refit_full_history(
    model_name: str,
    hyperparameters: dict[str, Any],
    prepared: PreparedFullShotData,
    input_length: int,
    horizon: int,
    settings: FullShotTrainingSettings,
    device: Any,
    training_epochs: int,
) -> tuple[Any, int, int, list[dict[str, float | int]]]:
    model, optimizer = _new_model_and_optimizer(
        model_name, hyperparameters, input_length, horizon, settings, device
    )
    random_generator = np.random.default_rng(settings.seed + 2)
    training_examples_seen = 0
    optimizer_steps_run = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_epochs + 1):
        training_objective, training_examples, optimizer_steps = _train_epoch(
            model,
            optimizer,
            prepared.full_pool,
            settings,
            random_generator,
            device,
            epoch,
        )
        training_examples_seen += training_examples
        optimizer_steps_run += optimizer_steps
        history.append({"epoch": epoch,
                        "training_objective": training_objective,
                        "training_examples": training_examples, "optimizer_steps": optimizer_steps})
    model.eval()
    del optimizer
    return model, training_examples_seen, optimizer_steps_run, history


def fit_full_shot_checkpoint(
    *,
    model_name: str,
    hyperparameters: dict[str, Any],
    histories: Sequence[np.ndarray],
    input_length: int,
    horizon: int,
    settings: FullShotTrainingSettings,
    device_name: str,
    record_ids: Sequence[str] | None = None,
    seasonality: int = 1,
) -> FullShotCheckpoint:
    """Select epochs using chronological validation, then refit once."""
    import torch

    series_window = settings.normalization_policy == "series_window_std_v1"
    if series_window and (
        record_ids is None or len(record_ids) != len(histories)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError("Per-record scaling requires unique training record ids")
    prepared = prepare_full_shot_data(histories, input_length, horizon, settings, seasonality)
    device = torch.device(device_name)
    (
        best_epoch,
        epochs_run,
        selection_optimizer_steps,
        best_mae,
        parameter_count,
        selection_history,
    ) = _select_training_epochs(
        model_name, hyperparameters, prepared, input_length, horizon, settings, device,
    )
    model, full_training_examples_seen, final_optimizer_steps, refit_history = _refit_full_history(
        model_name,
        hyperparameters,
        prepared,
        input_length,
        horizon,
        settings,
        device,
        best_epoch,
    )
    return FullShotCheckpoint(
        model=model,
        fallback_values=prepared.fallback_values,
        selected_epoch=best_epoch,
        selection_epochs_run=epochs_run,
        final_training_epochs=best_epoch,
        selection_legal_window_count=prepared.selection_pool.total_window_count,
        full_legal_window_count=prepared.full_pool.total_window_count,
        selection_examples_per_epoch=prepared.selection_pool.examples_per_epoch,
        full_examples_per_epoch=prepared.full_pool.examples_per_epoch,
        selection_training_examples_seen=(
            prepared.selection_pool.examples_per_epoch * epochs_run
        ),
        full_training_examples_seen=full_training_examples_seen,
        selection_optimizer_steps=selection_optimizer_steps,
        final_optimizer_steps=final_optimizer_steps,
        minimum_series_selection_windows=int(
            prepared.selection_pool.window_counts.min()
        ),
        minimum_series_full_training_windows=int(
            prepared.full_pool.window_counts.min()
        ),
        minimum_series_selection_examples_per_epoch=int(
            prepared.selection_pool.examples_per_series.min()
        ),
        minimum_series_full_examples_per_epoch=int(
            prepared.full_pool.examples_per_series.min()
        ),
        scalar_series_count=len(prepared.full_pool.histories),
        validation_example_count=len(prepared.validation_inputs),
        validation_target_count=(int(prepared.validation_observed.sum()) if series_window
                                 else int(prepared.validation_targets.size)),
        parameter_count=parameter_count,
        validation_mae=None if series_window else best_mae,
        input_length=input_length,
        horizon=horizon,
        target_dim=1,
        interpolated_training_values=prepared.interpolated_values,
        selection_history=selection_history,
        normalization_policy=settings.normalization_policy,
        history_scales=prepared.history_scales,
        selection_history_scales=prepared.selection_history_scales,
        refit_history=refit_history,
        history_record_ids=tuple(record_ids) if series_window else (),
        selection_metric="equal_series_train_prefix_stabilized_mase" if series_window else "normalized_mae",
        selection_score=best_mae,
    )


def forecast_with_checkpoint(
    checkpoint: FullShotCheckpoint,
    histories: Sequence[np.ndarray],
    *,
    batch_size: int,
    device_name: str,
    record_ids: Sequence[str] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Forecast scalar series independently and restore each dataset record."""
    import torch

    prepared: list[np.ndarray] = []
    dimensions: list[int] = []
    interpolated_count = 0
    series_window = checkpoint.normalization_policy == "series_window_std_v1"
    scale_lookup = dict(zip(checkpoint.history_record_ids, checkpoint.history_scales)) if series_window else {}
    if series_window and (record_ids is None or len(record_ids) != len(histories)):
        raise ValueError("Forecast histories require their training record ids")
    applied_scales: list[np.ndarray] = []
    for index, history in enumerate(histories):
        values, interpolated = interpolate_missing(history, checkpoint.fallback_values)
        if len(values) < checkpoint.input_length:
            raise ValueError("Evaluation history is shorter than the local input length")
        dimensions.append(values.shape[1])
        recent = values[-checkpoint.input_length :]
        if series_window:
            key = record_ids[index]
            if key not in scale_lookup:
                raise ValueError(f"Forecast record has no fitted history scale: {key}")
            scale = scale_lookup[key]
            if recent.shape[1] != len(scale):
                raise ValueError("Forecast variates differ from fitted record scales")
            recent = recent / scale
            applied_scales.append(scale)
        prepared.extend(
            recent[:, index : index + 1] for index in range(recent.shape[1])
        )
        interpolated_count += interpolated

    scalar_outputs: list[np.ndarray] = []
    device = torch.device(device_name)
    checkpoint.model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            batch = torch.from_numpy(
                np.stack(prepared[start : start + batch_size]).astype(np.float32)
            ).to(device, non_blocking=True)
            normalized_inputs, mean, standard_deviation = _normalize_batch(
                batch, checkpoint.normalization_policy,
            )
            normalized = checkpoint.model(normalized_inputs)
            forecasts = normalized * standard_deviation + mean
            if not torch.isfinite(forecasts).all():
                raise FloatingPointError("Non-finite full-shot forecast")
            scalar_outputs.extend(
                np.asarray(item, dtype=np.float64)
                for item in forecasts.detach().cpu().numpy()
            )
    outputs: list[np.ndarray] = []
    offset = 0
    for index, dimension in enumerate(dimensions):
        forecast = np.concatenate(scalar_outputs[offset : offset + dimension], axis=1)
        if series_window:
            forecast = forecast * applied_scales[index]
        if not np.isfinite(forecast).all():
            raise FloatingPointError("Non-finite restored full-shot forecast")
        outputs.append(forecast)
        offset += dimension
    if offset != len(scalar_outputs):
        raise RuntimeError("Scalar forecast regrouping consumed an invalid output count")
    return outputs, interpolated_count
