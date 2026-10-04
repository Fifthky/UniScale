"""Registered dynamic-process configuration and safe result I/O."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPOSITORY / "UniScale/results/learning-mechanism"
DEFAULT_CONFIG = REPOSITORY / "UniScale/configs/learning_mechanism.json"
SIGNS = (-1, 1)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["protocol"] != "history-learning-v5":
        raise ValueError("Unrecognized mechanism protocol")
    from .controls import validate_controls

    validate_controls(config)
    state = config["state_transfer"]
    if state["method"] != "shared-history-bn-calibrated-parameters":
        raise ValueError("Only shared-history calibrated parameter transfer is supported")
    if state["calibration_windows_per_rule"] != 512 or state["calibration_batch_size"] != 32:
        raise ValueError("The registered calibration uses 512 windows per rule in batches of 32")
    if state["training_fraction"] != 0.8 or state["patchtst_schedule"] != "cosine":
        raise ValueError("Use the registered validation-selected training protocol")
    if not 0 < config["phi_magnitude"] < 1:
        raise ValueError("AR coefficient magnitude must lie in (0, 1)")
    if config["query_length"] != 96 or config["primary_horizon"] != 1:
        raise ValueError("The registered protocol fixes a 96-step query and primary h=1")
    if config["scored_horizons"] != [1, 2, 3]:
        raise ValueError("Horizon endpoints must be stored separately as 1, 2, 3")
    if any(length < 96 or length % 32 for length in config["context_lengths"]):
        raise ValueError("Context lengths must be multiples of 32 and at least 96")
    if config["processes"] != ["ar1", "lag8", "threshold"]:
        raise ValueError("The release protocol requires all three registered processes")
    if config["magnitudes"] != [0.15, 0.225, 0.3] or config["phi_magnitude"] != 0.3:
        raise ValueError("All registered magnitudes and the .3 parameter-training magnitude are required")
    transfer = config["activation_transfer"]
    if transfer != {"query_seed": 2026091701, "donor_seed": 2026091702, "donor_repeats": 2,
                    "candidate": {"method": "residual_short", "layer": 5, "length": 512,
                                  "donor_count": 4, "strength": 1}}:
        raise ValueError("Only the registered shared-site activation transfer is supported")
    registered = json.loads(DEFAULT_CONFIG.read_text())
    if config != registered:
        raise ValueError("Only the final registered scientific configuration is supported")
    return config


def run_path(timestamp: str) -> Path:
    from UniScale.paths import validate_run_timestamp

    path = (RESULT_ROOT / validate_run_timestamp(timestamp)).resolve()
    if path.parent != RESULT_ROOT.resolve() or RESULT_ROOT.is_symlink():
        raise ValueError("Mechanism runs must remain in their isolated result root")
    return path


def output_path(root: Path, name: str) -> Path:
    root = root.resolve()
    if root.parent != RESULT_ROOT.resolve():
        raise ValueError("Refusing to write into a non-mechanism run")
    target = (root / name).resolve()
    if root not in target.parents:
        raise ValueError("Output path escapes the mechanism run")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(root: Path, name: str, payload: Any) -> None:
    path = output_path(root, name)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_arrays(root: Path, name: str, **arrays: np.ndarray) -> None:
    path = output_path(root, name)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_evidence(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "data/evidence.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
