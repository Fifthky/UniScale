"""Merge instance-sharded model results into canonical experiment rows."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from UniScale.io_utils import atomic_write_csv, atomic_write_text
from UniScale.experiments.grid_priority import preferred_indices, row_resource_key


MEAN_METRICS = (
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
)
RMSE = "eval_metrics/RMSE[mean]"
NRMSE = "eval_metrics/NRMSE[mean]"
ND = "eval_metrics/ND[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _weighted_mean(
    rows: list[dict[str, str]], field: str, weights: list[float]
) -> float:
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError(f"Cannot merge {field} with a non-positive weight")
    numerator = sum(
        float(row[field]) * weight for row, weight in zip(rows, weights)
    )
    return numerator / denominator


def merge_shard_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Merge rows for one run_key using GIFT-equivalent sufficient statistics."""
    if not rows:
        raise ValueError("Cannot merge an empty shard group")
    run_keys = {row["run_key"] for row in rows}
    if len(run_keys) != 1:
        raise ValueError("Every shard group must contain one run_key")
    shard_indices = [int(row["instance_shard_index"]) for row in rows]
    if len(shard_indices) != len(set(shard_indices)):
        raise ValueError(f"Duplicate instance shard for {next(iter(run_keys))}")

    valid_counts = [float(row["valid_target_count"]) for row in rows]
    absolute_sums = [float(row["sum_absolute_label"]) for row in rows]
    total_count = sum(valid_counts)
    total_absolute = sum(absolute_sums)
    if total_count <= 0 or total_absolute <= 0:
        raise ValueError("Shard sufficient statistics must be positive")

    merged: dict[str, Any] = dict(rows[0])
    for field in MEAN_METRICS:
        merged[field] = _weighted_mean(rows, field, valid_counts)
    merged[RMSE] = math.sqrt(float(merged["eval_metrics/MSE[mean]"]))
    mean_absolute_label = total_absolute / total_count
    merged[NRMSE] = float(merged[RMSE]) / mean_absolute_label
    merged[ND] = (
        float(merged["eval_metrics/MAE[0.5]"]) * total_count / total_absolute
    )
    merged[CRPS] = _weighted_mean(rows, CRPS, absolute_sums)

    baseline_mase = {float(row["seasonal_naive_mase"]) for row in rows}
    baseline_crps = {float(row["seasonal_naive_crps"]) for row in rows}
    if len(baseline_mase) != 1 or len(baseline_crps) != 1:
        raise ValueError("Shard rows disagree on the Seasonal Naive baseline")
    raw_mase = float(merged["eval_metrics/MASE[0.5]"])
    raw_crps = float(merged[CRPS])
    seasonal_mase = baseline_mase.pop()
    seasonal_crps = baseline_crps.pop()
    rel_mase = raw_mase / seasonal_mase
    rel_crps = raw_crps / seasonal_crps
    merged.update(
        {
            "raw_mase": raw_mase,
            "seasonal_naive_mase": seasonal_mase,
            "rel_mase": rel_mase,
            "log_rel_mase": math.log(rel_mase),
            "raw_crps": raw_crps,
            "seasonal_naive_crps": seasonal_crps,
            "rel_crps": rel_crps,
            "log_rel_crps": math.log(rel_crps),
            "available_context_min": min(
                int(row["available_context_min"]) for row in rows
            ),
            "available_context_max": max(
                int(row["available_context_max"]) for row in rows
            ),
            "effective_context_min": min(
                int(row["effective_context_min"]) for row in rows
            ),
            "effective_context_max": max(
                int(row["effective_context_max"]) for row in rows
            ),
            "effective_batch_size": min(
                int(row["effective_batch_size"]) for row in rows
            ),
            "effective_samples_per_batch": min(
                int(row["effective_samples_per_batch"]) for row in rows
            ),
            "instance_shard_index": "merged",
            "instance_shard_count": max(
                int(row["instance_shard_count"]) for row in rows
            ),
            "instance_count": sum(int(row["instance_count"]) for row in rows),
            "valid_target_count": int(total_count),
            "sum_absolute_label": total_absolute,
        }
    )
    recovery_events = []
    for row in rows:
        for event in json.loads(row.get("batch_recovery_events", "[]") or "[]"):
            recovery_events.append(
                {"instance_shard_index": int(row["instance_shard_index"]), **event}
            )
    merged["batch_recovery_events"] = json.dumps(
        recovery_events, sort_keys=True
    )
    return merged


def merge_model_shards(
    result_root: Path,
    model_id: str,
    shard_count: int,
    exclude_resource_keys: list[list[Any]] | None = None,
    allow_empty: bool = False,
) -> Path:
    """Merge shard groups; Grid references remain only in the Grid result table."""
    model_root = result_root / model_id
    canonical_path = model_root / "all_results.csv"
    previous_canonical = _read_rows(canonical_path)
    reused_keys = {tuple(key) for key in (exclude_resource_keys or [])}
    candidates = [row for row in previous_canonical
                  if not reused_keys or row_resource_key(row) not in reused_keys]
    canonical_rows = [candidates[index] for index in preferred_indices(candidates)]
    canonical_keys = {row["run_key"] for row in canonical_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for shard_index in range(shard_count):
        shard_path = (
            model_root
            / "shards"
            / f"part-{shard_index:05d}-of-{shard_count:05d}"
            / "all_results.csv"
        )
        for row in _read_rows(shard_path):
            if (not reused_keys or row_resource_key(row) not in reused_keys) and row["run_key"] not in canonical_keys:
                grouped[row["run_key"]].append(row)

    merged_rows = [merge_shard_rows(grouped[key]) for key in sorted(grouped)]
    all_rows: list[dict[str, Any]] = [*canonical_rows, *merged_rows]
    if not all_rows and not allow_empty:
        raise ValueError(f"No canonical or shard results found for {model_id}")
    fields: list[str] = [] if all_rows else ["model", "dataset", "H", "L", "run_key"]
    for row in all_rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    atomic_write_csv(canonical_path, all_rows, fields)
    manifest_paths = [
        model_root
        / "shards"
        / f"part-{shard_index:05d}-of-{shard_count:05d}"
        / "manifest.json"
        for shard_index in range(shard_count)
    ]
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in manifest_paths
        if path.is_file()
    ]
    atomic_write_text(
        model_root / "manifest.json",
        json.dumps(
            {
                "model_id": model_id,
                "instance_shard_count": shard_count,
                "shard_manifests": [str(path) for path in manifest_paths],
                "git_commits": sorted(
                    {str(manifest.get("git_commit", "")) for manifest in manifests}
                ),
                "canonical_result_path": str(canonical_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return canonical_path
