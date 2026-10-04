"""Plan native-H work from completed Grid cells before scheduling workers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from UniScale.experiments.dataset_scope import dataset_configuration
from UniScale.experiments.grid_priority import GRID_PRIORITY_POLICY, is_grid, resource_key
from UniScale.io_utils import atomic_write_text
from UniScale.paths import resolve_run_path, validate_experiment_name


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan_grid_reuse(
    repository: Path, config: dict, run_timestamp: str, properties: dict,
    models: dict, expected_cells: dict[str, int],
) -> dict[str, Any]:
    """Freeze available canonical Grid evidence; raw source files remain intact."""
    reference = config.get("grid_reuse")
    if reference is None:
        return {"policy": GRID_PRIORITY_POLICY, "models": {}, "enabled": False}
    frozen_path = resolve_run_path(repository, config["experiment_name"], run_timestamp) / "_scheduler/grid_reuse.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
    if frozen is not None:
        frozen_root = (repository / frozen["source_root"]).resolve()
        if not frozen_root.is_relative_to(repository / "UniScale/results"):
            raise ValueError("The frozen Grid source must remain inside UniScale/results")
        if reference["experiment_name"] != frozen_root.parent.name:
            raise ValueError("Resume cannot change the frozen Grid experiment")
        if reference.get("run_timestamp", frozen_root.name) != frozen_root.name:
            raise ValueError("Resume cannot change the frozen Grid run")
        reference = {**reference, "run_timestamp": frozen_root.name}
    if config.get("H", [None]) != [None] or "cells" in config:
        raise ValueError("Grid reuse applies only to the native-H route")
    source_experiment = validate_experiment_name(str(reference["experiment_name"]))
    if not source_experiment.startswith("joint-scaling"):
        raise ValueError("Grid reuse requires an explicit joint-scaling source")
    source_timestamp = reference.get("run_timestamp")
    source_parent = repository / "UniScale" / "results" / source_experiment
    if source_timestamp is None:
        same_run = source_parent / run_timestamp
        candidates = [same_run] if same_run.is_dir() else sorted(
            (path for path in source_parent.iterdir() if path.is_dir()), reverse=True
        ) if source_parent.is_dir() else []
        source_root = next((path for path in candidates if (path / "_scheduler/summary.json").is_file()
                            and json.loads((path / "_scheduler/summary.json").read_text()).get("status")
                            in {"completed", "completed_with_errors"}), None)
        if source_root is None:
            raise FileNotFoundError("No completed Grid source; run Grid first or specify grid_reuse.run_timestamp")
    else:
        source_root = resolve_run_path(repository, source_experiment, str(source_timestamp))
    if source_root == resolve_run_path(repository, config["experiment_name"], run_timestamp):
        raise ValueError("A native route cannot reuse itself")
    source_summary = json.loads((source_root / "_scheduler/summary.json").read_text())
    if source_summary.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("The Grid source must finish and persist canonical results before native-H planning")
    completed_models = {item["model_id"] for item in source_summary["tasks"] if item["status"] == "completed"}
    # Imported only for a native-H plan, once in the scheduler, never per model worker.
    from UniScale.experiments.data import native_horizon
    from UniScale.models.standard import resolve_standard_inference

    conditions = []
    for spec in config["datasets"]:
        for term in spec.get("terms", ["short"]):
            dataset, _ = dataset_configuration(spec["name"], term, properties)
            frequency = dataset.split("/", maxsplit=2)[1]
            conditions.append((dataset, native_horizon(spec["name"], frequency, term)))
    plan: dict[str, Any] = {
        "policy": GRID_PRIORITY_POLICY, "enabled": True,
        "source_root": str(source_root.relative_to(repository)), "models": {},
    }
    for specification in config["models"]:
        model_id = specification["id"]
        record = models[model_id]
        profile = resolve_standard_inference(record)
        maximum = record["context"].get("maximum")
        lengths = [int(value if value is not None else record["context"]["gift"])
                   for value in config.get("L", [None])
                   if value is None or maximum is None or int(value) <= int(maximum)]
        eligible = {resource_key(model_id, dataset, H, L): dataset for dataset, H in conditions for L in lengths}
        source_path = source_root / model_id / "all_results.csv"
        slots = []
        if model_id in completed_models:
            rows = read_rows(source_path)
            if not rows:
                raise ValueError(f"Completed Grid model has no canonical rows: {model_id}")
            for row in rows:
                key = resource_key(row["model"], row["dataset"], float(row["H"]), float(row["L"]))
                if key not in eligible:
                    continue
                if not is_grid(row):
                    raise ValueError(f"The selected source is not Grid evidence: {source_path}")
                if float(row["parameters_active_m"]) != float(record["parameters_m"]["active"]):
                    raise ValueError(f"Grid capacity differs from the registered checkpoint: {model_id}")
                expected_checkpoint = record.get("local_checkpoint_subdir")
                if expected_checkpoint and Path(row["checkpoint"]).name != Path(expected_checkpoint).name:
                    raise ValueError(f"Grid checkpoint path differs from the registered checkpoint: {model_id}")
                expected_profile = {
                    "adapter": record["adapter"], "num_samples": profile["num_samples"],
                    "seed": profile["seed"] if profile["seed"] is not None else "",
                    "torch_dtype": "float32",
                }
                if any(str(row[field]) != str(value) for field, value in expected_profile.items()):
                    raise ValueError(f"Grid inference profile differs from the frozen native-H profile: {model_id}")
                slots.append({"key": list(key), "native_dataset": eligible[key]})
        unique = {tuple(slot["key"]) for slot in slots}
        if len(unique) != len(slots):
            raise ValueError(f"Duplicate Grid resource cells for {model_id}")
        native_path = resolve_run_path(repository, config["experiment_name"], run_timestamp) / model_id / "all_results.csv"
        native_existing = {
            resource_key(row["model"], row["dataset"], float(row["H"]), float(row["L"]))
            for row in read_rows(native_path)
        }.intersection(eligible).difference(unique)
        plan["models"][model_id] = {
            "protocol_cells": expected_cells[model_id],
            "native_expected_cells": expected_cells[model_id] - len(unique),
            "grid_reused_cells": len(unique),
            "native_canonical_already_complete": len(native_existing),
            "native_cells_to_cover": expected_cells[model_id] - len(unique) - len(native_existing),
            "grid_source_status": "completed" if model_id in completed_models else "no_completed_grid_model",
            "keys": [list(key) for key in sorted(unique)],
            "slots": sorted(slots, key=lambda slot: slot["key"]),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()
            if model_id in completed_models else None,
        }
    if frozen is not None:
        if set(frozen["models"]) != set(plan["models"]):
            raise ValueError("Resume cannot change the frozen checkpoint panel")
        for model_id, record in plan["models"].items():
            previous = frozen["models"][model_id]
            if (previous["keys"] != record["keys"]
                    or previous["source_sha256"] != record["source_sha256"]):
                raise ValueError(f"The frozen Grid evidence has changed for {model_id}")
    return plan


def persist_grid_plan(path: Path, plan: dict) -> None:
    atomic_write_text(path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
