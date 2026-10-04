"""Shared resource identity and Grid-first selection for Controlled evidence."""

from __future__ import annotations

from typing import Any

GRID_PRIORITY_POLICY = "grid_preferred_v1"


def resource_key(model: str, dataset: str, horizon: float, context: float) -> tuple:
    """Identify a checkpoint/dataset-frequency/H/L independently of term labels."""
    frequency_id, separator, term = dataset.rpartition("/")
    if not separator or term not in {"short", "medium", "long"}:
        frequency_id = dataset
    return str(model), frequency_id, float(horizon), float(context)


def row_resource_key(row: dict[str, Any]) -> tuple:
    return resource_key(
        row.get("model_id", row.get("model", "")), row["dataset"],
        row["horizon"] if "horizon" in row else row["H"],
        row["context_allocated_len"] if "context_allocated_len" in row else row["L"],
    )


def is_grid(row: dict[str, Any]) -> bool:
    source = str(row.get("source_experiment", row.get("experiment_name", "")))
    return (
        str(row.get("origin_policy", "")).startswith("shared_origin_v1:")
        or source == "joint-scaling" or source.startswith("joint-scaling-")
    )


def preferred_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Select Grid first and reject conflicting copies of the same observation."""
    grid_keys = {row_resource_key(row) for row in rows if is_grid(row)}
    if not grid_keys:
        return list(range(len(rows)))
    selected, seen_grid = [], {}
    for index, row in enumerate(rows):
        key = row_resource_key(row)
        if key in grid_keys and not is_grid(row):
            continue
        if is_grid(row):
            if key in seen_grid:
                previous = rows[seen_grid[key]]
                fields = (
                    "rel_mase", "rel_crps", "parameters_active_m", "parameters_total_m",
                    "available_context_max", "actual_context_len", "origin_policy",
                )
                if any(str(row.get(field, "")) != str(previous.get(field, "")) for field in fields):
                    raise ValueError(f"Conflicting Grid observations for {key}; select one source run")
                continue
            seen_grid[key] = index
        selected.append(index)
    return selected


def result_key(dataset: str, H: int, L: int, H_source: str, L_source: str, origins: str) -> str:
    return f"{dataset}|H={H}|L={L}|Hsrc={H_source}|Lsrc={L_source}|origins={origins}"
