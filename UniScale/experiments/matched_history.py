"""Run dataset-level full-shot models under matched controlled histories."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import get_seasonality

from UniScale.io_utils import (
    atomic_append_csv_row,
    atomic_write_csv,
    atomic_write_text,
    ensure_directory,
)
from UniScale.paths import (
    new_run_timestamp,
    resolve_results_path,
    resolve_run_path,
    validate_run_timestamp,
)
from UniScale.statistics.results import normalize_metric_pair

from .full_shot_results import publish_complete_results as _publish_complete_results
from .data import ControlledDataset
from .dataset_scope import dataset_configuration
from .evaluation import evaluation_sufficient_statistics, evaluate_with_isolated_windows
from .full_shot import (
    FullShotCheckpoint,
    FullShotTrainingSettings,
    UnsupportedFullShotCell,
    fit_full_shot_checkpoint,
    forecast_with_checkpoint,
)
from .metrics import metric_record
from .seasonal_naive import ensure_seasonal_naive


EXPECTED_HORIZONS = [96, 192, 336]
EXPECTED_TRAINING_LENGTHS = [1024, 2048, 4096, 8192]
EXPECTED_INPUT_LENGTH = 96
EXPECTED_ORIGIN_HORIZON = 720
EXPECTED_DATASET_COUNT = 23
SUPPORTED_MODELS = {"dlinear", "patchtst"}
FULL_SHOT_PROTOCOL = "dataset_level_scalar_history_learning_in_weights_v4"
SKIPPED_CELL_FIELDS = [
    "run_key",
    "experiment_name",
    "run_timestamp",
    "experiment_protocol",
    "dataset",
    "model",
    "H",
    "L",
    "reason",
]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if "datasets_file" in config:
        if "datasets" in config:
            raise ValueError("Specify datasets or datasets_file, not both")
        config["datasets"] = _read_json(path.parent / config["datasets_file"])
    dataset_terms = config.get("dataset_terms")
    if dataset_terms is not None:
        if (
            not isinstance(dataset_terms, list)
            or not dataset_terms
            or any(not isinstance(term, str) or not term for term in dataset_terms)
        ):
            raise ValueError("dataset_terms must be a non-empty list of term names")
        config["datasets"] = [
            {**dataset, "terms": list(dataset_terms)}
            for dataset in config["datasets"]
        ]
    required = {
        "experiment_name",
        "models",
        "datasets",
        "H",
        "L",
        "local_input_length",
        "origin_horizon",
        "training",
        "execution",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Matched-history config is missing: {', '.join(missing)}")
    if config["experiment_name"] != "matched-history-full-shot":
        raise ValueError("experiment_name must be matched-history-full-shot")
    if config["H"] != EXPECTED_HORIZONS:
        raise ValueError(f"Matched-history H must be {EXPECTED_HORIZONS}")
    if config["L"] != EXPECTED_TRAINING_LENGTHS:
        raise ValueError(f"Matched-history L must be {EXPECTED_TRAINING_LENGTHS}")
    if int(config["local_input_length"]) != EXPECTED_INPUT_LENGTH:
        raise ValueError(f"Full-shot local input length must be {EXPECTED_INPUT_LENGTH}")
    if int(config["origin_horizon"]) != EXPECTED_ORIGIN_HORIZON:
        raise ValueError("Matched-history origins must be anchored at H=720")
    if not isinstance(config["datasets"], list):
        raise ValueError("datasets must be a list")
    expected_dataset_count = int(config.get("expected_dataset_count", EXPECTED_DATASET_COUNT))
    if expected_dataset_count <= 0 or len(config["datasets"]) != expected_dataset_count:
        raise ValueError(
            f"Matched-history datasets must contain {expected_dataset_count} entries"
        )
    if len({item["name"] for item in config["datasets"]}) != expected_dataset_count:
        raise ValueError("Matched-history datasets must have unique names")
    if any(item.get("terms", ["short"]) != ["short"] for item in config["datasets"]):
        raise ValueError("Every matched-history dataset-frequency must appear once")
    _validate_training_overrides(config, config.get("dataset_training_overrides", {}))

    configured_models: set[str] = set()
    for model in config["models"]:
        if not isinstance(model, dict) or "name" not in model:
            raise ValueError("Every full-shot model requires a name")
        name = str(model["name"]).lower()
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported full-shot model: {model['name']}")
        if name in configured_models:
            raise ValueError(f"Duplicate full-shot model: {model['name']}")
        configured_models.add(name)
        _validate_training_overrides(config, model.get("dataset_training_overrides", {}))
        if int(model.get("batch_size", 0)) <= 0:
            raise ValueError(f"{model['name']} requires a positive batch_size")
        if int(model.get("evaluation_batch_size", 0)) <= 0:
            raise ValueError(f"{model['name']} requires a positive evaluation_batch_size")
        if not isinstance(model.get("hyperparameters", {}), dict):
            raise ValueError(f"{model['name']} hyperparameters must be an object")
        if not isinstance(model.get("training", {}), dict):
            raise ValueError(f"{model['name']} training overrides must be an object")
    if not str(config["execution"].get("device", "cuda")):
        raise ValueError("execution.device cannot be empty")
    return config


def _validate_training_overrides(config: dict, overrides: dict) -> None:
    if not isinstance(overrides, dict) or set(overrides).difference(
        item["name"] for item in config["datasets"]
    ):
        raise ValueError("Dataset training overrides must identify configured datasets")
    allowed_overrides = {"normalization_policy", "training_loss"}
    for payload in overrides.values():
        if not isinstance(payload, dict) or set(payload).difference(allowed_overrides):
            raise ValueError("Dataset overrides may only change normalization and loss settings")
        FullShotTrainingSettings.from_config({**config["training"], **payload}, batch_size=1)


def _training_overrides(config: dict, model_spec: dict) -> dict:
    """Resolve configured training policies without dataset-name branches."""
    return {
        **config.get("dataset_training_overrides", {}),
        **model_spec.get("dataset_training_overrides", {}),
    }


def _validate_run_seed(timestamp: str, config: dict) -> None:
    """Keep a named seed consistent with every configured model."""
    if "_seed" in timestamp:
        seed = int(timestamp.rsplit("_seed", 1)[1])
        for model in config["models"]:
            actual = int({**config["training"], **model.get("training", {})}["seed"])
            if actual != seed:
                raise ValueError(f"Run seed {seed} differs from {model['name']} seed {actual}")


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _existing_keys(path: Path) -> set[str]:
    for attempt in range(10):
        if not path.is_file():
            if attempt == 9:
                return set()
            time.sleep(0.2)
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "run_key" not in (reader.fieldnames or []):
                    raise ValueError(f"Existing result file has no run_key: {path}")
                return {row["run_key"] for row in reader}
        except FileNotFoundError:
            if attempt == 9:
                return set()
            time.sleep(0.2)
    raise RuntimeError("Unreachable result-index retry state")


def _target_time_dimensions(entry: dict[str, Any]) -> np.ndarray:
    target = np.asarray(entry["target"], dtype=np.float32)
    if target.ndim == 1:
        return target[:, None]
    if target.ndim == 2:
        return target.T
    raise UnsupportedFullShotCell(f"unsupported_target_shape={target.shape}")


def _earliest_training_histories(
    dataset: ControlledDataset,
    training_length: int,
    record_ids: list[str] | None = None,
) -> tuple[list[np.ndarray], int, int]:
    entries = list(dataset.test_data.input)
    windows = int(dataset.windows)
    if windows <= 0 or not entries or len(entries) % windows:
        raise RuntimeError("GIFT instances are incompatible with the declared window count")
    histories: list[np.ndarray] = []
    available_lengths: list[int] = []
    for base_index in range(len(entries) // windows):
        group = entries[base_index * windows : (base_index + 1) * windows]
        earliest = min(
            group, key=lambda entry: _target_time_dimensions(entry).shape[0]
        )
        if record_ids is not None:
            if earliest.get("item_id") is None:
                raise ValueError("Per-record normalization requires dataset item ids")
            key = str(earliest["item_id"])
            if any(str(entry.get("item_id")) != key for entry in group):
                raise ValueError("Grouped forecast origins must belong to one record")
            record_ids.append(key)
        available = _target_time_dimensions(earliest)
        available_lengths.append(len(available))
        effective_length = min(len(available), training_length)
        histories.append(np.array(available[-effective_length:], copy=True))
    return histories, min(available_lengths), max(available_lengths)


class DatasetCheckpointPredictor:
    """Apply one fitted dataset checkpoint to every later GIFT window."""

    def __init__(
        self,
        checkpoint: FullShotCheckpoint,
        evaluation_batch_size: int,
        device_name: str,
    ):
        self.checkpoint = checkpoint
        self.evaluation_batch_size = evaluation_batch_size
        self.device_name = device_name
        self.interpolated_evaluation_values = 0

    def predict(self, entries: Iterable[dict[str, Any]]):
        materialized = list(entries)
        histories = [_target_time_dimensions(entry) for entry in materialized]
        forecasts, interpolated = forecast_with_checkpoint(
            self.checkpoint,
            histories,
            batch_size=self.evaluation_batch_size,
            device_name=self.device_name,
            record_ids=([str(entry["item_id"]) for entry in materialized]
                        if self.checkpoint.normalization_policy == "series_window_std_v1" else None),
        )
        self.interpolated_evaluation_values += interpolated
        for forecast, entry in zip(forecasts, materialized):
            raw_target = np.asarray(entry["target"])
            samples = forecast[None, :, :]
            if raw_target.ndim == 1:
                samples = samples[:, :, 0]
            yield SampleForecast(
                samples=samples,
                start_date=entry["start"] + raw_target.shape[-1],
                item_id=entry.get("item_id"),
            )


def _model_specs(
    config: dict[str, Any], selected_model: str | None
) -> Iterable[tuple[int, dict[str, Any]]]:
    for model_index, model in enumerate(config["models"]):
        if selected_model is None or str(model["name"]).lower() == selected_model:
            yield model_index, model


def _task_ordinal(
    config: dict[str, Any],
    model_index: int,
    horizon_index: int,
    dataset_index: int,
    length_index: int,
) -> int:
    datasets = len(config["datasets"])
    horizons = len(config["H"])
    return (
        length_index * len(config["models"]) * horizons * datasets
        + model_index * horizons * datasets
        + horizon_index * datasets
        + dataset_index
    )


def _shard_root(model_root: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return model_root
    return (
        model_root
        / "shards"
        / f"shard-{shard_index:03d}-of-{shard_count:03d}"
    )


def _recursive_existing_keys(model_root: Path, filename: str) -> set[str]:
    keys: set[str] = set()
    for path in model_root.rglob(filename):
        keys.update(_existing_keys(path))
    return keys


def _release_resources(checkpoint: FullShotCheckpoint | None = None) -> None:
    if checkpoint is not None:
        del checkpoint.model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _skip_cell(
    path: Path,
    *,
    run_key: str,
    timestamp: str,
    dataset: str,
    model: str,
    horizon: int,
    training_length: int,
    reason: str,
) -> None:
    atomic_append_csv_row(
        path,
        {
            "run_key": run_key,
            "experiment_name": "matched-history-full-shot",
            "run_timestamp": timestamp,
            "experiment_protocol": FULL_SHOT_PROTOCOL,
            "dataset": dataset,
            "model": model,
            "H": horizon,
            "L": training_length,
            "reason": reason,
        },
    )


def run(
    config_path: Path,
    run_timestamp: str | None = None,
    *,
    resume: bool = False,
    selected_model: str | None = None,
    task_shard_index: int = 0,
    task_shard_count: int = 1,
    selected_task_ordinal: int | None = None,
    retry_skipped: bool = False,
    require_result: bool = False,
) -> None:
    config = load_config(config_path)
    if task_shard_count <= 0:
        raise ValueError("task_shard_count must be positive")
    if task_shard_index < 0 or task_shard_index >= task_shard_count:
        raise ValueError("task_shard_index must be within task_shard_count")
    total_tasks = (
        len(config["models"])
        * len(config["H"])
        * len(config["datasets"])
        * len(config["L"])
    )
    if selected_task_ordinal is not None and not 0 <= selected_task_ordinal < total_tasks:
        raise ValueError("selected_task_ordinal is outside the configured task grid")
    if selected_model is not None:
        selected_model = selected_model.lower()
        configured = {str(item["name"]).lower() for item in config["models"]}
        if selected_model not in configured:
            raise ValueError(f"Model is absent from config: {selected_model}")
    configured_timestamp = config.get("run_timestamp")
    if run_timestamp and configured_timestamp and run_timestamp != configured_timestamp:
        raise ValueError("CLI run timestamp does not match config")
    timestamp = validate_run_timestamp(
        run_timestamp or configured_timestamp or f"{new_run_timestamp()}_seed{int(config['training']['seed'])}"
    )
    _validate_run_seed(timestamp, config)
    repository = Path(__file__).resolve().parents[2]
    run_root = resolve_run_path(repository, config["experiment_name"], timestamp)
    if resume and not run_root.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run_root}")
    ensure_directory(run_root)

    execution = config["execution"]
    properties_path = repository / execution.get(
        "dataset_properties", "UniScale/information/dataset_properties.json"
    )
    properties = _read_json(properties_path)
    results_root = resolve_results_path(repository)
    baseline_path = (
        results_root
        / "baselines"
        / "seasonal_naive_statsforecast_native_v1"
        / "all_results.csv"
    )
    revision = _git_revision(repository)
    device = str(execution.get("device", "cuda"))
    input_length = int(config["local_input_length"])

    for model_index, model_spec in _model_specs(config, selected_model):
        model_name = str(model_spec["name"])
        model_root = run_root / model_name.lower()
        worker_root = _shard_root(model_root, task_shard_index, task_shard_count)
        result_path = worker_root / "all_results.csv"
        skip_path = worker_root / "skipped_cells.csv"
        completed = _recursive_existing_keys(model_root, "all_results.csv")
        skipped = (
            set()
            if retry_skipped
            else _recursive_existing_keys(model_root, "skipped_cells.csv")
        )
        assigned_run_keys: set[str] = set()
        training_payload = dict(config["training"])
        training_payload.update(model_spec.get("training", {}))
        overrides = _training_overrides(config, model_spec)
        settings = FullShotTrainingSettings.from_config(
            training_payload, int(model_spec["batch_size"])
        )
        manifest = {
            "experiment_name": config["experiment_name"],
            "experiment_protocol": FULL_SHOT_PROTOCOL,
            "run_timestamp": timestamp,
            "git_commit": revision,
            "model": model_name,
            "model_hyperparameters": model_spec.get("hyperparameters", {}),
            "training": settings.__dict__,
            "dataset_training_overrides": overrides,
            "local_input_length": input_length,
            "training_lengths": config["L"],
            "horizons": config["H"],
            "origin_horizon": config["origin_horizon"],
            "dataset_count": len(config["datasets"]),
            "device": device,
            "result_path": str(result_path),
            "task_shard_index": task_shard_index,
            "task_shard_count": task_shard_count,
            "training_unit": (
                "One checkpoint per architecture, dataset-frequency, H, and L. "
                "Every dataset record is unfolded into scalar series. Each scalar "
                "series contributes a temporally stratified quota proportional to "
                "its effective history length, and all scalar series share one "
                "univariate model."
            ),
            "evaluation_unit": (
                "The fitted checkpoint is reused for every test window of the same "
                "dataset-frequency, H, and L cell."
            ),
            "temporal_coverage_stride": settings.temporal_coverage_stride,
            "coverage_sampling": "one_random_start_per_temporal_stratum_each_epoch",
            "missing_value_policy": (
                "per_cell_policy_recorded_in_results" if overrides
                else "time_interpolation_with_dataset_scalar_fallback_v2"
            ),
            "normalization_policy": (
                "per_cell_policy_recorded_in_results"
                if overrides else "reversible_per_window_per_scalar_series_v2"
            ),
            "selection_metric": (
                "per_cell_metric_recorded_in_results"
                if overrides else "equal_scalar_series_full_horizon_normalized_mae"
            ),
            "selection_validation_horizon": "full_scored_horizon",
            "selection_validation_origins_per_series": (
                settings.validation_origins_per_series
            ),
            "final_refit": "selected_epochs_on_complete_effective_training_history",
        }
        ensure_directory(worker_root)
        atomic_write_text(
            worker_root / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

        for horizon_index, horizon in enumerate(config["H"]):
            for dataset_index, dataset_spec in enumerate(config["datasets"]):
                assigned_lengths = [
                    (length_index, training_length)
                    for length_index, training_length in enumerate(config["L"])
                    if (
                        _task_ordinal(
                            config,
                            model_index,
                            horizon_index,
                            dataset_index,
                            length_index,
                        )
                        == selected_task_ordinal
                        if selected_task_ordinal is not None
                        else _task_ordinal(
                            config,
                            model_index,
                            horizon_index,
                            dataset_index,
                            length_index,
                        )
                        % task_shard_count
                        == task_shard_index
                    )
                ]
                if not assigned_lengths:
                    continue
                name = str(dataset_spec["name"])
                settings = FullShotTrainingSettings.from_config(
                    {**training_payload, **overrides.get(name, {})},
                    int(model_spec["batch_size"]),
                )
                term = str(dataset_spec.get("terms", ["short"])[0])
                dataset_key, property_key = dataset_configuration(name, term, properties)
                source = ControlledDataset(
                    name=name,
                    term=term,
                    to_univariate=False,
                    prediction_length=int(horizon),
                    origin_horizon=int(config["origin_horizon"]),
                )
                baseline_dataset = (
                    ControlledDataset(
                        name=name,
                        term=term,
                        to_univariate=True,
                        prediction_length=int(horizon),
                        origin_horizon=int(config["origin_horizon"]),
                    )
                    if source.target_dim > 1
                    else source
                )
                baseline = ensure_seasonal_naive(
                    dataset=baseline_dataset,
                    dataset_configuration=dataset_key,
                    output_path=baseline_path,
                    origin_policy=baseline_dataset.origin_policy,
                    domain=properties[property_key]["domain"],
                    num_variates=properties[property_key]["num_variates"],
                )
                sufficient_statistics = evaluation_sufficient_statistics(source)

                for _, training_length in assigned_lengths:
                    run_key = (
                        f"{dataset_key}|H={horizon}|L={training_length}"
                        f"|ell={input_length}|origins={source.origin_policy}"
                        f"|protocol={FULL_SHOT_PROTOCOL}"
                    )
                    assigned_run_keys.add(run_key)
                    if run_key in completed or run_key in skipped:
                        continue
                    checkpoint: FullShotCheckpoint | None = None
                    try:
                        record_ids = [] if settings.normalization_policy == "series_window_std_v1" else None
                        histories, available_min, available_max = (
                            _earliest_training_histories(source, int(training_length), record_ids)
                        )
                        checkpoint = fit_full_shot_checkpoint(
                            model_name=model_name,
                            hyperparameters=dict(model_spec.get("hyperparameters", {})),
                            histories=histories,
                            input_length=input_length,
                            horizon=int(horizon),
                            settings=settings,
                            device_name=device,
                            record_ids=record_ids,
                            seasonality=get_seasonality(source.freq),
                        )
                    except UnsupportedFullShotCell as error:
                        _release_resources(checkpoint)
                        if require_result:
                            raise RuntimeError(
                                f"Required full-shot cell is unsupported: {run_key}: {error}"
                            ) from error
                        _skip_cell(
                            skip_path,
                            run_key=run_key,
                            timestamp=timestamp,
                            dataset=dataset_key,
                            model=model_name,
                            horizon=int(horizon),
                            training_length=int(training_length),
                            reason=str(error),
                        )
                        skipped.add(run_key)
                        continue

                    predictor = DatasetCheckpointPredictor(
                        checkpoint,
                        int(model_spec["evaluation_batch_size"]),
                        device,
                    )
                    raw_metrics = metric_record(
                        evaluate_with_isolated_windows(predictor, source)
                    )
                    normalized = normalize_metric_pair(raw_metrics, baseline)
                    effective_history_lengths = np.asarray(
                        [
                            len(history)
                            for history in histories
                            for _ in range(history.shape[1])
                        ],
                        dtype=np.float64,
                    )
                    row = {
                        "run_key": run_key,
                        "experiment_name": config["experiment_name"],
                        "experiment_protocol": FULL_SHOT_PROTOCOL,
                        "run_timestamp": timestamp,
                        "git_commit": revision,
                        "dataset": dataset_key,
                        "domain": properties[property_key]["domain"],
                        "model": model_name,
                        "H": horizon,
                        "origin_policy": source.origin_policy,
                        "origin_horizon": source.origin_horizon,
                        "L": training_length,
                        "effective_training_length": float(
                            np.exp(np.mean(np.log(effective_history_lengths)))
                        ),
                        "effective_training_length_mean": float(
                            np.mean(effective_history_lengths)
                        ),
                        "effective_training_length_min": int(
                            effective_history_lengths.min()
                        ),
                        "effective_training_length_max": int(
                            effective_history_lengths.max()
                        ),
                        "available_context_min": available_min,
                        "available_context_max": available_max,
                        "local_input_length": input_length,
                        "base_series_count": len(histories),
                        "scalar_series_count": checkpoint.scalar_series_count,
                        "target_dim": checkpoint.target_dim,
                        "rolling_windows": source.windows,
                        "evaluation_instance_count": len(list(source.test_data.input)),
                        "selection_legal_window_count": (
                            checkpoint.selection_legal_window_count
                        ),
                        "full_legal_window_count": (
                            checkpoint.full_legal_window_count
                        ),
                        "selection_examples_per_epoch": (
                            checkpoint.selection_examples_per_epoch
                        ),
                        "full_examples_per_epoch": (
                            checkpoint.full_examples_per_epoch
                        ),
                        "selection_training_examples_seen": (
                            checkpoint.selection_training_examples_seen
                        ),
                        "full_training_examples_seen": (
                            checkpoint.full_training_examples_seen
                        ),
                        "selection_optimizer_steps": (
                            checkpoint.selection_optimizer_steps
                        ),
                        "final_optimizer_steps": checkpoint.final_optimizer_steps,
                        "minimum_series_selection_examples_per_epoch": (
                            checkpoint.minimum_series_selection_examples_per_epoch
                        ),
                        "minimum_series_full_examples_per_epoch": (
                            checkpoint.minimum_series_full_examples_per_epoch
                        ),
                        "minimum_series_selection_windows": (
                            checkpoint.minimum_series_selection_windows
                        ),
                        "minimum_series_full_training_windows": (
                            checkpoint.minimum_series_full_training_windows
                        ),
                        "full_training_target_count": (
                            checkpoint.full_training_examples_seen * int(horizon)
                        ),
                        "training_targets_per_parameter": (
                            checkpoint.full_training_examples_seen
                            * int(horizon)
                            / checkpoint.parameter_count
                        ),
                        "validation_example_count": (
                            checkpoint.validation_example_count
                        ),
                        "validation_target_count": (
                            checkpoint.validation_target_count
                        ),
                        "selection_validation_horizon": (
                            int(horizon)
                        ),
                        "selection_validation_origins_per_series": (
                            settings.validation_origins_per_series
                        ),
                        "selected_epoch": checkpoint.selected_epoch,
                        "selection_epochs_run": checkpoint.selection_epochs_run,
                        "final_training_epochs": checkpoint.final_training_epochs,
                        "selection_validation_mae": checkpoint.validation_mae,
                        "selection_history_json": json.dumps(
                            checkpoint.selection_history,
                            separators=(",", ":"),
                        ),
                        "refit_history_json": json.dumps(checkpoint.refit_history, separators=(",", ":")),
                        "training_settings_json": json.dumps(settings.__dict__, sort_keys=True, separators=(",", ":")),
                        "training_loss": settings.training_loss,
                        "selection_history_scales_json": json.dumps(checkpoint.selection_history_scales.tolist()),
                        "refit_history_scales_json": json.dumps(checkpoint.history_scales.tolist()),
                        "history_scale_record_ids_json": json.dumps(checkpoint.history_record_ids),
                        "selection_validation_score": checkpoint.selection_score,
                        "validation_seasonality": get_seasonality(source.freq),
                        "validation_scale_floor_relative": (
                            1e-5 if settings.normalization_policy == "series_window_std_v1" else None
                        ),
                        "seed": settings.seed,
                        "batch_size": settings.batch_size,
                        "evaluation_batch_size": model_spec["evaluation_batch_size"],
                        "learning_rate": settings.learning_rate,
                        "temporal_coverage_stride": (
                            settings.temporal_coverage_stride
                        ),
                        "parameter_count": checkpoint.parameter_count,
                        "missing_value_policy": (
                            "prefix_isolated_interpolation_zero_empty_history_masked_validation_v1"
                            if settings.normalization_policy == "series_window_std_v1"
                            else "time_interpolation_with_dataset_scalar_fallback_v2"
                        ),
                        "normalization_policy": (
                            "reversible_per_window_per_scalar_series_v2"
                            if settings.normalization_policy == "window_std_v2"
                            else "record_variate_window_standardization_v1"
                        ),
                        "selection_metric": (
                            "equal_scalar_series_full_horizon_normalized_mae"
                            if settings.normalization_policy == "window_std_v2"
                            else checkpoint.selection_metric
                        ),
                        "interpolated_training_values": (
                            checkpoint.interpolated_training_values
                        ),
                        "interpolated_evaluation_values": (
                            predictor.interpolated_evaluation_values
                        ),
                        **raw_metrics,
                        **normalized,
                        **sufficient_statistics,
                    }
                    atomic_append_csv_row(result_path, row)
                    completed.add(run_key)
                    print(
                        f"completed {model_name} {run_key}: "
                        f"rel_MASE={normalized['rel_mase']:.6f}",
                        flush=True,
                    )
                    del predictor, histories, row
                    _release_resources(checkpoint)

        terminal_count = len(assigned_run_keys.intersection(completed | skipped))
        assigned_count = sum(
            1
            for horizon_index in range(len(config["H"]))
            for dataset_index in range(len(config["datasets"]))
            for length_index in range(len(config["L"]))
            if (
                _task_ordinal(
                    config,
                    model_index,
                    horizon_index,
                    dataset_index,
                    length_index,
                )
                == selected_task_ordinal
                if selected_task_ordinal is not None
                else _task_ordinal(
                    config,
                    model_index,
                    horizon_index,
                    dataset_index,
                    length_index,
                )
                % task_shard_count
                == task_shard_index
            )
        )
        if selected_task_ordinal is not None and assigned_count == 0:
            continue
        if terminal_count != assigned_count:
            raise RuntimeError(
                f"Shard finished {terminal_count} of {assigned_count} assigned "
                f"{model_name} cells"
            )
        atomic_write_text(
            worker_root / "completed.json",
            json.dumps(
                {
                    "model": model_name,
                    "task_shard_index": task_shard_index,
                    "task_shard_count": task_shard_count,
                    "assigned_cell_count": assigned_count,
                    "terminal_cell_count": terminal_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    if task_shard_count == 1 and selected_task_ordinal is None:
        _publish_complete_results(config, run_root)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_common_fields(rows: list[dict[str, str]], label: str) -> list[str]:
    if not rows:
        return []
    fields = list(rows[0])
    optional_diagnostics = {
        "refit_history_json", "training_settings_json", "training_loss",
        "selection_history_scales_json", "refit_history_scales_json",
        "history_scale_record_ids_json",
        "selection_validation_score", "validation_seasonality", "validation_scale_floor_relative",
    } if label == "result" else set()
    expected = set(fields).difference(optional_diagnostics)
    for row in rows[1:]:
        if set(row).difference(optional_diagnostics) != expected:
            raise ValueError(f"Inconsistent {label} CSV schemas across task shards")
        fields.extend(key for key in row if key not in fields)
    # Historical rows keep missing diagnostics empty, never invented or zeroed.
    return fields


def merge_task_shards(
    config_path: Path,
    run_timestamp: str,
    task_shard_count: int,
) -> None:
    """Validate and merge a complete multi-GPU matched-history run."""
    if task_shard_count <= 1:
        raise ValueError("Merging requires at least two task shards")
    config = load_config(config_path)
    timestamp = validate_run_timestamp(run_timestamp)
    repository = Path(__file__).resolve().parents[2]
    run_root = resolve_run_path(repository, config["experiment_name"], timestamp)
    expected_per_model = len(config["H"]) * len(config["datasets"]) * len(config["L"])

    for model_spec in config["models"]:
        model_name = str(model_spec["name"])
        model_root = run_root / model_name.lower()
        for shard_index in range(task_shard_count):
            worker_root = _shard_root(model_root, shard_index, task_shard_count)
            completion_path = worker_root / "completed.json"
            if not completion_path.is_file():
                raise FileNotFoundError(
                    f"Task shard has no completion marker: {completion_path}"
                )
            completion = _read_json(completion_path)
            if int(completion["task_shard_index"]) != shard_index:
                raise ValueError(f"Task shard index mismatch in {completion_path}")
            if int(completion["task_shard_count"]) != task_shard_count:
                raise ValueError(f"Task shard count mismatch in {completion_path}")
            if int(completion["terminal_cell_count"]) != int(
                completion["assigned_cell_count"]
            ):
                raise ValueError(f"Incomplete task shard: {completion_path}")

        result_rows: list[dict[str, str]] = []
        skipped_rows: list[dict[str, str]] = []
        shards_root = model_root / "shards"
        for path in sorted(shards_root.rglob("all_results.csv")):
            result_rows.extend(_read_csv_rows(path))
        for path in sorted(shards_root.rglob("skipped_cells.csv")):
            skipped_rows.extend(_read_csv_rows(path))

        result_keys = [row["run_key"] for row in result_rows]
        skipped_keys = [row["run_key"] for row in skipped_rows]
        incompatible = [
            row.get("experiment_protocol")
            for row in result_rows + skipped_rows
            if row.get("experiment_protocol") != FULL_SHOT_PROTOCOL
        ]
        if incompatible:
            raise ValueError(
                f"Incompatible full-shot protocol while merging {model_name}: "
                f"{sorted(set(incompatible))}"
            )
        terminal_keys = result_keys + skipped_keys
        if len(terminal_keys) != len(set(terminal_keys)):
            raise ValueError(f"Duplicate terminal cells while merging {model_name}")
        if len(terminal_keys) != expected_per_model:
            raise ValueError(
                f"Merged {len(terminal_keys)} of {expected_per_model} expected "
                f"{model_name} cells"
            )
        result_fields = _validate_common_fields(result_rows, "result")
        skipped_fields = _validate_common_fields(skipped_rows, "skipped-cell")
        if result_rows:
            atomic_write_csv(model_root / "all_results.csv", result_rows, result_fields)
        if skipped_rows:
            atomic_write_csv(
                model_root / "skipped_cells.csv", skipped_rows, skipped_fields
            )
        atomic_write_text(
            model_root / "manifest.json",
            json.dumps(
                {
                    "experiment_name": config["experiment_name"],
                    "experiment_protocol": FULL_SHOT_PROTOCOL,
                    "run_timestamp": timestamp,
                    "model": model_name,
                    "task_shard_count": task_shard_count,
                    "expected_cell_count": expected_per_model,
                    "completed_cell_count": len(result_rows),
                    "unsupported_cell_count": len(skipped_rows),
                    "status": "complete",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        print(
            f"merged {model_name}: {len(result_rows)} completed, "
            f"{len(skipped_rows)} unsupported",
            flush=True,
        )

    _publish_complete_results(config, run_root)


def merge_dynamic_results(
    config_path: Path, run_timestamp: str,
) -> None:
    """Merge a dynamic-pool run only when every configured cell has a result."""

    config = load_config(config_path)
    timestamp = validate_run_timestamp(run_timestamp)
    repository = Path(__file__).resolve().parents[2]
    run_root = resolve_run_path(repository, config["experiment_name"], timestamp)
    properties_path = repository / config["execution"].get(
        "dataset_properties", "UniScale/information/dataset_properties.json"
    )
    properties = _read_json(properties_path)
    expected_datasets = {
        dataset_configuration(
            str(spec["name"]),
            str(spec.get("terms", ["short"])[0]),
            properties,
        )[0]
        for spec in config["datasets"]
    }
    expected_coordinates = {
        (dataset, int(horizon), int(length))
        for dataset in expected_datasets
        for horizon in config["H"]
        for length in config["L"]
    }
    prepared_merges = []

    for model_spec in config["models"]:
        model_name = str(model_spec["name"])
        model_root = run_root / model_name.lower()
        result_rows: list[dict[str, str]] = []
        result_rows.extend(_read_csv_rows(model_root / "all_results.csv"))
        for path in sorted((model_root / "shards").rglob("all_results.csv")):
            result_rows.extend(_read_csv_rows(path))
        by_key: dict[str, dict[str, str]] = {}
        for row in result_rows:
            key = row["run_key"]
            if key in by_key:
                fields = set(row) | set(by_key[key])
                if all(row.get(field, "") == by_key[key].get(field, "") for field in fields):
                    continue
                raise ValueError(f"Conflicting result cell while merging {model_name}: {key}")
            if row.get("experiment_protocol") != FULL_SHOT_PROTOCOL:
                raise ValueError(
                    f"Incompatible full-shot protocol while merging {model_name}: "
                    f"{row.get('experiment_protocol')}"
                )
            by_key[key] = row
        coordinates = [
            (row["dataset"], int(row["H"]), int(row["L"]))
            for row in by_key.values()
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(f"Duplicate logical cells while merging {model_name}")
        coordinate_set = set(coordinates)
        if coordinate_set != expected_coordinates:
            missing = sorted(expected_coordinates.difference(coordinate_set))
            extra = sorted(coordinate_set.difference(expected_coordinates))
            raise ValueError(
                f"Dynamic merge requires all {len(expected_coordinates)} {model_name} "
                f"cells; missing={missing[:10]}, extra={extra[:10]}"
            )
        ordered_rows = sorted(
            by_key.values(),
            key=lambda row: (int(row["H"]), int(row["L"]), row["dataset"]),
        )
        result_fields = _validate_common_fields(ordered_rows, "result")
        prepared_merges.append((model_name, model_root, ordered_rows, result_fields))

    # Validate both complete model lists before replacing either canonical file.
    for model_name, model_root, ordered_rows, result_fields in prepared_merges:
        atomic_write_csv(model_root / "all_results.csv", ordered_rows, result_fields)
        atomic_write_csv(model_root / "skipped_cells.csv", [], SKIPPED_CELL_FIELDS)
        atomic_write_text(
            model_root / "manifest.json",
            json.dumps(
                {
                    "experiment_name": config["experiment_name"],
                    "experiment_protocol": FULL_SHOT_PROTOCOL,
                    "run_timestamp": timestamp,
                    "model": model_name,
                    "expected_cell_count": len(expected_coordinates),
                    "completed_cell_count": len(ordered_rows),
                    "unsupported_cell_count": 0,
                    "status": "complete",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        print(f"merged {model_name}: {len(ordered_rows)} completed", flush=True)

    _publish_complete_results(config, run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--task-ordinal", type=int)
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument("--require-result", action="store_true")
    timestamp_group = parser.add_mutually_exclusive_group()
    timestamp_group.add_argument("--run-timestamp")
    timestamp_group.add_argument("--resume", metavar="TIMESTAMP")
    timestamp_group.add_argument("--merge-shards", metavar="TIMESTAMP")
    timestamp_group.add_argument("--merge-dynamic", metavar="TIMESTAMP")
    args = parser.parse_args()
    if args.merge_shards is not None:
        if args.model is not None or args.task_shard_index != 0:
            parser.error("--merge-shards cannot be combined with --model or shard index")
        merge_task_shards(
            args.config.resolve(), args.merge_shards, args.task_shard_count
        )
        return
    if args.merge_dynamic is not None:
        if args.model is not None or args.task_ordinal is not None:
            parser.error("--merge-dynamic cannot be combined with model or task selection")
        merge_dynamic_results(
            args.config.resolve(), args.merge_dynamic,
        )
        return
    run(
        args.config.resolve(),
        args.resume or args.run_timestamp,
        resume=args.resume is not None,
        selected_model=args.model,
        task_shard_index=args.task_shard_index,
        task_shard_count=args.task_shard_count,
        selected_task_ordinal=args.task_ordinal,
        retry_skipped=args.retry_skipped,
        require_result=args.require_result,
    )


if __name__ == "__main__":
    main()
