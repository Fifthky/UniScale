"""Registered crossed-history experiment and immutable source inputs."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import REPOSITORY, digest_file


def project_file(relative: str) -> Path:
    path = (REPOSITORY / relative).resolve(strict=True)
    if REPOSITORY not in path.parents:
        raise ValueError("Configuration input escapes the project")
    return path


def validate_controls(config: dict) -> None:
    c = config["controls"]
    x = c["crossed"]
    if x["seeds"] != config["training_seeds"] or len(x["seeds"]) != 3:
        raise ValueError("Crossed training requires the registered three seeds")
    if x["input_lengths"] != [96, 512] or x["history_lengths"] != [4096, 8192]:
        raise ValueError("The crossed design fixes two input and two history lengths")
    if x["horizon"] != 96 or x["origin_horizon"] != 720:
        raise ValueError("The crossed design uses H=96 at the published shared origins")
    if x["models"] != ["DLinear", "PatchTST", "PatchTST-compact", "TimesFM"]:
        raise ValueError("All four architecture/pretraining conditions are required")
    if x["batch_size"] % x["microbatch_size"] or x["microbatch_size"] < 2:
        raise ValueError("Microbatches must divide the effective batch and support BatchNorm")
    if x["report_updates"][0] != 0 or x["report_updates"][-1] != x["updates"]:
        raise ValueError("Report the initial and final predictors and common update budgets")
    if any(v < 0 or v > x["updates"] for v in x["report_updates"]) or x["validation_interval"] < 1:
        raise ValueError("Invalid update or validation schedule")
    if any(x["learning_rates"][name] <= 0 for name in x["models"]):
        raise ValueError("All learning rates must be positive")
    panel = json.loads(project_file(x["dataset_panel"]).read_text())
    if len(panel) != 23 or len({item["name"] for item in panel}) != 23:
        raise ValueError("Crossed real-data controls require the complete 23-task panel")
    project_file(x["training_config"])


def controls_plan(config: dict) -> list[dict]:
    c, tasks = config["controls"], []
    x = c["crossed"]
    for index, dataset in enumerate(json.loads(project_file(x["dataset_panel"]).read_text())):
        for length in x["history_lengths"]:
            for seed in x["seeds"]:
                for model in x["models"]:
                    for context in x["input_lengths"]:
                        tasks.append({"id": f"cross-d{index:02d}-{model}-T{length}-C{context}-seed{seed}",
                                      "kind": "crossed_history", "dataset": dataset["name"],
                                      "model": model, "history_length": length, "input_length": context,
                                      "seed": seed, "environment": "TSFM" if model == "TimesFM" else "toto",
                                      "slot_cost": 4 if model == "TimesFM" else 1, "depends": []})
    return tasks


def input_receipt(config: dict) -> dict:
    """Pin shared real-data configuration before scheduling."""
    c = config["controls"]
    files = {name: digest_file(project_file(name)) for name in
             (c["crossed"]["dataset_panel"], c["crossed"]["training_config"])}
    return {"project_files": files}
