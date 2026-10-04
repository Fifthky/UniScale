"""Publish one complete, seed-identified full-shot result table."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from UniScale.io_utils import atomic_write_csv, atomic_write_text
from UniScale.experiments.dataset_scope import dataset_configuration


def publish_complete_results(config: dict, run_root: Path) -> bool:
    """Publish only after every model has its complete canonical cell table.

    Model tables remain the resumable worker outputs. The run-level table is
    the single input for downstream analysis and is rebuilt by finalization.
    """
    expected_count = len(config["datasets"]) * len(config["H"]) * len(config["L"])
    repository = Path(__file__).resolve().parents[2]
    properties = json.loads((repository / config["execution"].get(
        "dataset_properties", "UniScale/information/dataset_properties.json"
    )).read_text())
    names = {dataset_configuration(d["name"], "short", properties)[0]: d["name"]
             for d in config["datasets"]}
    normalization_labels = {
        "window_std_v2": "reversible_per_window_per_scalar_series_v2",
        "series_window_std_v1": "record_variate_window_standardization_v1",
    }
    tables = []
    for spec in config["models"]:
        path = run_root / spec["name"].lower() / "all_results.csv"
        if not path.is_file():
            return False
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < expected_count:
            return False
        datasets = {row["dataset"] for row in rows}
        expected = {(d, int(h), int(l)) for d in names for h in config["H"] for l in config["L"]}
        coordinates = [(row["dataset"], int(row["H"]), int(row["L"])) for row in rows]
        seed = int({**config["training"], **spec.get("training", {})}["seed"])
        if (len(rows) != expected_count or len(datasets) != len(config["datasets"])
                or len(set(coordinates)) != len(rows) or set(coordinates) != expected):
            raise ValueError(f"Incomplete or duplicate full-shot grid: {path}")
        if "_seed" in run_root.name and seed != int(run_root.name.rsplit("_seed", 1)[1]):
            raise ValueError("Result-directory seed differs from the configuration")
        overrides = {**config.get("dataset_training_overrides", {}),
                     **spec.get("dataset_training_overrides", {})}
        for row in rows:
            settings = {**config["training"], **spec.get("training", {}),
                        **overrides.get(names[row["dataset"]], {})}
            policy = settings.get("normalization_policy", "window_std_v2")
            if (row["normalization_policy"] != normalization_labels[policy]
                    or (row.get("training_loss") or "mse") != settings.get("training_loss", "mse")):
                raise ValueError(f"Training policy mismatch: {path}: {row['run_key']}")
            if row["model"] != spec["name"] or int(row["seed"]) != seed:
                raise ValueError(f"Model or seed mismatch: {path}: {row['run_key']}")
            if not math.isfinite(float(row["rel_mase"])) or float(row["rel_mase"]) <= 0:
                raise ValueError(f"Invalid relative MASE: {path}: {row['run_key']}")
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        manifest.update({
            "experiment_name": config["experiment_name"],
            "experiment_protocol": rows[0]["experiment_protocol"],
            "run_timestamp": run_root.name, "model": spec["name"],
            "expected_cell_count": expected_count, "completed_cell_count": len(rows),
            "unsupported_cell_count": 0, "status": "complete", "seed": seed,
        })
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tables.append((path, rows))
    protocols = {row["experiment_protocol"] for _, table in tables for row in table}
    if len(protocols) != 1:
        raise ValueError("Cannot publish mixed full-shot protocols")
    rows = sorted((row for _, table in tables for row in table),
                  key=lambda row: (row["model"], int(row["H"]), int(row["L"]), row["dataset"]))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    atomic_write_csv(run_root / "all_results.csv", rows, fields)
    resolved_config = {key: value for key, value in config.items() if key != "datasets_file"}
    atomic_write_text(run_root / "config.json", json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    manifest = {
        "run_timestamp": run_root.name,
        "status": "complete",
        "completed_cell_count": len(rows),
        "dataset_count": len(config["datasets"]),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "experiment_protocols": sorted({row["experiment_protocol"] for row in rows}),
        "source_commits": sorted({row["git_commit"] for row in rows}),
        "model_tables_sha256": {
            str(path.relative_to(run_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path, _ in tables
        },
        "all_results_sha256": hashlib.sha256((run_root / "all_results.csv").read_bytes()).hexdigest(),
    }
    atomic_write_text(run_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return True
