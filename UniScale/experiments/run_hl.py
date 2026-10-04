"""Run the primary H/L intervention while freezing each official model profile."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

from UniScale.io_utils import atomic_append_csv_row, atomic_write_text, ensure_directory
from UniScale.runtime_logging import WorkerLogPolicy
from UniScale.experiments.grid_priority import resource_key, result_key


FORBIDDEN_PROFILE_OVERRIDES = {
    "batch_size",
    "batch_size_cap",
    "batch_size_policy",
    "checkpoint",
    "deterministic_algorithms",
    "float32_matmul_precision",
    "forecast_pipeline",
    "num_samples",
    "per_core_batch_size",
    "request_lengths",
    "output_root",
    "seed",
    "torch_dtype",
}


def _length_grid(config: dict[str, Any], name: str) -> list[int | None]:
    values = config.get(name, [None])
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list when provided")
    resolved: list[int | None] = []
    for value in values:
        if value is None:
            resolved.append(None)
            continue
        length = int(value)
        if length <= 0:
            raise ValueError(f"Every controlled {name} value must be positive")
        resolved.append(length)
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = ["experiment_name", "model_id", "datasets"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Experiment config is missing: {', '.join(missing)}")
    if not config["datasets"]:
        raise ValueError("datasets must be non-empty")
    forbidden = sorted(FORBIDDEN_PROFILE_OVERRIDES.intersection(config))
    if forbidden:
        raise ValueError(
            "The primary H/L interface freezes notebook settings; remove: "
            + ", ".join(forbidden)
        )
    execution = config.get("execution", {})
    forbidden_execution = sorted(FORBIDDEN_PROFILE_OVERRIDES.intersection(execution))
    if forbidden_execution:
        raise ValueError(
            "execution cannot override the standard model profile; remove: "
            + ", ".join(forbidden_execution)
        )
    configured_cells = config.get("cells")
    if configured_cells is not None:
        if "H" in config or "L" in config:
            raise ValueError("cells cannot be combined with H or L grids")
        if not isinstance(configured_cells, list) or not configured_cells:
            raise ValueError("cells must be a non-empty list")
        cells: list[dict[str, int]] = []
        for cell in configured_cells:
            if not isinstance(cell, dict) or set(cell) != {"H", "L"}:
                raise ValueError("Every cell must contain exactly H and L")
            H, L = int(cell["H"]), int(cell["L"])
            if H <= 0 or L <= 0:
                raise ValueError("Every cell H and L must be positive")
            cells.append({"H": H, "L": L})
        if len({(cell["H"], cell["L"]) for cell in cells}) != len(cells):
            raise ValueError("cells must be unique")
        config["cells"] = cells
        config["H"] = list(dict.fromkeys(cell["H"] for cell in cells))
        config["L"] = list(dict.fromkeys(cell["L"] for cell in cells))
    else:
        config["H"] = _length_grid(config, "H")
        config["L"] = _length_grid(config, "L")
        config["cells"] = [
            {"H": H, "L": L} for H in config["H"] for L in config["L"]
        ]
    controlled_horizons = [value for value in config["H"] if value is not None]
    if controlled_horizons and len(controlled_horizons) != len(config["H"]):
        raise ValueError("H cannot mix the GIFT default with controlled horizons")
    configured_origin = config.get("origin_horizon")
    if configured_origin is not None:
        configured_origin = int(configured_origin)
        if configured_origin <= 0:
            raise ValueError("origin_horizon must be positive")
        if not controlled_horizons:
            raise ValueError("origin_horizon requires controlled H values")
        if configured_origin < max(controlled_horizons):
            raise ValueError("origin_horizon cannot be smaller than the largest H")
    config["origin_horizon"] = (
        configured_origin
        if configured_origin is not None
        else (max(controlled_horizons) if controlled_horizons else None)
    )
    if controlled_horizons:
        duplicated_terms = [
            spec["name"] for spec in config["datasets"]
            if len(spec.get("terms", ["short"])) != 1
        ]
        if duplicated_terms:
            raise ValueError(
                "Controlled H uses one shared-origin dataset-frequency entry; "
                "provide exactly one term label for: " + ", ".join(duplicated_terms)
            )
    return config


def git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def existing_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "run_key" not in (reader.fieldnames or []):
            raise ValueError(f"Existing result file has no run_key: {path}")
        return {row["run_key"] for row in reader}


def append_row(path: Path, row: dict[str, object]) -> None:
    atomic_append_csv_row(path, row)


def _target_length(entry: dict[str, Any]) -> int:
    return int(entry["target"].shape[-1])


def _release_cell_resources() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lengths_for_h(config: dict[str, Any], H: int | None) -> list[int | None]:
    return [cell["L"] for cell in config["cells"] if cell["H"] == H]


def pending_lengths(
    lengths: list[int | None], standard_model: Any, dataset_config: str,
    H: int | None, scored_horizon: int, origin_policy: str,
    completed_keys: set[str], grid_keys: set[tuple], progress: WorkerLogPolicy,
    dataset: Any = None,
) -> list[int | None]:
    """Apply Grid precedence first, then exact resume keys, before preparation."""
    pending = []
    for L in lengths:
        resolved = standard_model.context_without_backend(L, dataset)
        allocated = standard_model.record["context"].get("gift") if L is None else L
        grid_context = resolved if resolved is not None else allocated
        if grid_context is not None and resource_key(
            standard_model.record["id"], dataset_config, scored_horizon, grid_context
        ) in grid_keys:
            progress.skipped("grid_preferred")
            continue
        if resolved is not None and result_key(
            dataset_config, scored_horizon, resolved,
            "gift_default" if H is None else "controlled",
            "notebook_default" if L is None else "controlled", origin_policy,
        ) in completed_keys:
            progress.skipped("existing_result")
            continue
        pending.append(L)
    return pending


def run(config_path: Path, run_timestamp: str | None = None) -> None:
    with WorkerLogPolicy() as progress:
        _run(config_path, run_timestamp, progress)


def _run(
    config_path: Path,
    run_timestamp: str | None,
    progress: WorkerLogPolicy,
) -> None:
    from UniScale.models.standard import ExecutionOptions, load_standard_model
    from UniScale.paths import (
        new_run_timestamp,
        resolve_results_path,
        resolve_run_path,
        validate_run_timestamp,
    )
    from UniScale.statistics.results import normalize_metric_pair

    from .data import ControlledDataset, DatasetSourceCache, InstanceShardedDataset, native_horizon
    from .dataset_scope import dataset_configuration
    from .evaluation import (
        evaluation_sufficient_statistics,
        evaluate_with_isolated_windows,
    )
    from .metrics import metric_record
    from .seasonal_naive import BaselineCache, ensure_seasonal_naive

    config = load_config(config_path)
    if config["H"] == [None] and config["experiment_name"].startswith("context-scaling") and "grid_reuse_keys" not in config:
        raise ValueError("Native-H workers require a Grid-first scheduler plan; launch the checkpoint scheduler")
    configured_timestamp = config.get("run_timestamp")
    if run_timestamp and configured_timestamp and run_timestamp != configured_timestamp:
        raise ValueError("CLI run_timestamp does not match the config")
    config["run_timestamp"] = validate_run_timestamp(
        run_timestamp or configured_timestamp or new_run_timestamp()
    )
    repository = Path(__file__).resolve().parents[2]
    execution_config = config.get("execution", {})
    standard_model = load_standard_model(
        config["model_id"],
        ExecutionOptions(device=str(execution_config.get("device", "cuda"))),
    )
    model = standard_model.record
    properties_path = repository / execution_config.get(
        "dataset_properties", "UniScale/information/dataset_properties.json"
    )
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    output_root = resolve_results_path(
        repository,
        default_subdir=execution_config.get("results_subdir"),
    )
    baseline_path = (
        output_root
        / "baselines"
        / "seasonal_naive_statsforecast_native_v1"
        / "all_results.csv"
    )
    run_directory = (
        resolve_run_path(
            repository, config["experiment_name"], config["run_timestamp"]
        )
        / model["id"]
    )
    shard_count = int(config.get("instance_shard_count", 1))
    shard_index = int(config.get("instance_shard_index", 0))
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Invalid instance shard index/count")
    result_relative_path = Path(
        str(config.get("result_relative_path", "all_results.csv"))
    )
    if result_relative_path.is_absolute() or ".." in result_relative_path.parts:
        raise ValueError("result_relative_path must stay inside the model run directory")
    result_path = run_directory / result_relative_path
    canonical_result_path = run_directory / "all_results.csv"
    completed_keys = existing_keys(result_path) | existing_keys(canonical_result_path)
    grid_keys = {tuple(key) for key in config.get("grid_reuse_keys", [])}
    source_cache = DatasetSourceCache()
    baseline_cache = BaselineCache(baseline_path)
    revision = git_revision(repository)
    if config["origin_horizon"] is None:
        experiment_protocol = "standard_hl_gift_v1"
        window_policy = (
            "GIFT horizon-specific forecast origins; each window is predicted independently"
        )
        origin_policy = "gift_horizon_specific_v1"
    else:
        experiment_protocol = "standard_hl_shared_origin_v1"
        window_policy = (
            "Shared forecast origins across controlled H; each window is predicted independently"
        )
        origin_policy = f"shared_origin_v1:Hmax={config['origin_horizon']}"

    manifest = {
        "git_commit": revision,
        "config": config,
        "model_record": model,
        "resolved_checkpoint": Path(standard_model.checkpoint).name,
        "resolved_standard_profile": standard_model.inference,
        "scientific_controls": ["H", "L"],
        "frozen_profile_fields": sorted(FORBIDDEN_PROFILE_OVERRIDES),
        "normalization": "Every MASE and CRPS cell is divided by matching seasonal naive",
        "experiment_protocol": experiment_protocol,
        "window_policy": window_policy,
        "origin_policy": origin_policy,
        "instance_shard_index": shard_index,
        "instance_shard_count": shard_count,
        "result_path": str(result_path),
    }
    ensure_directory(result_path.parent)
    atomic_write_text(
        result_path.parent / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    for dataset_spec in config["datasets"]:
        name = dataset_spec["name"]
        for term in dataset_spec.get("terms", ["short"]):
            dataset_config, property_key = dataset_configuration(name, term, properties)
            frequency = dataset_config.split("/", maxsplit=2)[1]
            for H in config["H"]:
                scored_horizon = native_horizon(name, frequency, term) if H is None else H
                lengths = pending_lengths(
                    _lengths_for_h(config, H), standard_model, dataset_config, H,
                    scored_horizon, origin_policy, completed_keys, grid_keys, progress,
                )
                if not lengths:
                    continue
                source = ControlledDataset(
                    name=name,
                    term=term,
                    to_univariate=False,
                    prediction_length=H,
                    origin_horizon=config["origin_horizon"],
                    source=source_cache.get(name),
                )
                use_joint = bool(model["joint_multivariate"])
                source_available = source.available_context_min
                if config["origin_horizon"] is not None and source_available <= 1000:
                    progress.skipped("insufficient_shared_history")
                    continue
                dataset = (
                    ControlledDataset(
                        name=name,
                        term=term,
                        to_univariate=True,
                        prediction_length=H,
                        origin_horizon=config["origin_horizon"],
                        source=source_cache.get(name, True),
                    )
                    if source.target_dim > 1 and not use_joint
                    else source
                )
                lengths = pending_lengths(
                    lengths, standard_model, dataset_config, H, scored_horizon,
                    origin_policy, completed_keys, grid_keys, progress, dataset,
                )
                if not lengths:
                    continue
                evaluation_dataset = (
                    InstanceShardedDataset(dataset, shard_index, shard_count)
                    if shard_count > 1
                    else dataset
                )
                if (
                    shard_count > 1
                    and evaluation_dataset.instance_count == 0
                ):
                    progress.skipped("empty_instance_shard")
                    continue
                baseline_dataset = (
                    ControlledDataset(
                        name=name,
                        term=term,
                        to_univariate=True,
                        prediction_length=H,
                        origin_horizon=config["origin_horizon"],
                        source=source_cache.get(name, True),
                    )
                    if source.target_dim > 1
                    else source
                )
                baseline = ensure_seasonal_naive(
                    dataset=baseline_dataset,
                    dataset_configuration=dataset_config,
                    output_path=baseline_path,
                    origin_policy=baseline_dataset.origin_policy,
                    domain=properties[property_key]["domain"],
                    num_variates=properties[property_key]["num_variates"],
                    cache=baseline_cache,
                )
                available = [
                    _target_length(entry)
                    for entry in evaluation_dataset.test_data.input
                ]
                sufficient_statistics = evaluation_sufficient_statistics(
                    evaluation_dataset
                )

                for L in lengths:
                    prepared = standard_model.prepare(
                        dataset=evaluation_dataset,
                        frequency=frequency,
                        H=H,
                        L=L,
                        domain=properties[property_key].get("domain"),
                        has_daily_cycle=properties[property_key].get(
                            "has_daily_cycle", True
                        ),
                    )
                    run_key = result_key(
                        dataset_config, prepared.H, prepared.L, prepared.H_source,
                        prepared.L_source, evaluation_dataset.origin_policy,
                    )
                    if run_key in completed_keys:
                        progress.skipped("existing_result")
                        del prepared
                        _release_cell_resources()
                        continue
                    raw_metrics = metric_record(
                        evaluate_with_isolated_windows(
                            prepared.predictor, evaluation_dataset
                        )
                    )
                    normalized = normalize_metric_pair(raw_metrics, baseline)
                    output = model["output"]
                    row = {
                        "run_key": run_key,
                        "experiment_name": config["experiment_name"],
                        "run_timestamp": config["run_timestamp"],
                        "experiment_protocol": experiment_protocol,
                        "git_commit": revision,
                        "dataset": dataset_config,
                        "model": model["id"],
                        "adapter": standard_model.adapter,
                        "checkpoint": Path(standard_model.checkpoint).name,
                        "parameters_total_m": model["parameters_m"]["total"],
                        "parameters_active_m": model["parameters_m"]["active"],
                        "H": prepared.H,
                        "H_source": prepared.H_source,
                        "origin_policy": evaluation_dataset.origin_policy,
                        "origin_horizon": evaluation_dataset.origin_horizon or "",
                        "origin_distance": evaluation_dataset.origin_distance,
                        "L": prepared.L,
                        "L_source": prepared.L_source,
                        "available_context_min": min(available),
                        "available_context_max": max(available),
                        "effective_context_min": min(
                            getattr(
                                prepared.base_predictor,
                                "effective_context_limit",
                                prepared.L,
                            ),
                            min(available),
                        ),
                        "effective_context_max": min(
                            getattr(
                                prepared.base_predictor,
                                "effective_context_limit",
                                prepared.L,
                            ),
                            max(available),
                        ),
                        "requested_batch_size": prepared.settings.batch_size,
                        "effective_batch_size": prepared.effective_batch_size,
                        "effective_samples_per_batch": prepared.effective_samples_per_batch,
                        "batch_recovery_events": json.dumps(
                            prepared.batch_recovery_events, sort_keys=True
                        ),
                        "instance_shard_index": shard_index,
                        "instance_shard_count": shard_count,
                        "instance_count": getattr(
                            evaluation_dataset,
                            "instance_count",
                            len(evaluation_dataset.test_data) // evaluation_dataset.windows,
                        ),
                        **sufficient_statistics,
                        "per_core_batch_size": prepared.settings.per_core_batch_size or "",
                        "num_samples": prepared.settings.num_samples,
                        "seed": prepared.settings.seed if prepared.settings.seed is not None else "",
                        "torch_dtype": prepared.settings.torch_dtype,
                        "output_strategy": "native_horizon",
                        "model_call_horizon": prepared.model_call_horizon,
                        "native_output_block": output["native_block"] or "",
                        "native_rollout_mechanism": output["mechanism"],
                        **raw_metrics,
                        **normalized,
                        "domain": properties[property_key]["domain"],
                        "num_variates": properties[property_key]["num_variates"],
                    }
                    append_row(result_path, row)
                    completed_keys.add(run_key)
                    progress.completed(run_key)
                    del row, normalized, raw_metrics, prepared
                    _release_cell_resources()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-timestamp")
    args = parser.parse_args()
    run(args.config.resolve(), args.run_timestamp)


if __name__ == "__main__":
    main()
