"""Central model catalogue, checkpoint loading, and source resolution."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_CATALOGUE = Path(__file__).resolve().parent / "information" / "models.json"
MODEL_ROOT_ENV = "UNISCALE_MODEL_ROOT"


MODEL_MODULES = {
    "flowstate": "UniScale.vendor.flowstate",
    "patchtstfm": "UniScale.vendor.patchtstfm",
    "timesfm25": "UniScale.vendor.timesfm25",
}


def configured_model_root(explicit: str | Path | None = None) -> Path:
    """Resolve the checkpoint root from an explicit path or environment."""
    if explicit:
        return Path(explicit)
    root = os.environ.get(MODEL_ROOT_ENV)
    if not root:
        raise ValueError(f"Provide a model root or set {MODEL_ROOT_ENV}")
    return Path(root)


def load_catalogue(path: Path = DEFAULT_CATALOGUE) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        catalogue = json.load(handle)

    if catalogue.get("schema_version") != 1:
        raise ValueError(f"Unsupported catalogue schema in {path}")

    models = catalogue.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Catalogue {path} has no model records")

    identifiers: set[str] = set()
    result_directories: set[str] = set()
    for model in models:
        identifier = model.get("id")
        result_directory = model.get("result_directory")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every catalogue record requires a non-empty id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate model id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(result_directory, str) or not result_directory:
            raise ValueError(f"Model {identifier} has no result_directory")
        if result_directory in result_directories:
            raise ValueError(f"Duplicate result_directory: {result_directory}")
        result_directories.add(result_directory)

        if model.get("include_in_scaling"):
            active = model.get("parameters_m", {}).get("active")
            context = model.get("context", {}).get("gift")
            if not isinstance(active, (int, float)) or active <= 0:
                raise ValueError(
                    f"Scaling model {identifier} has no positive active parameter count"
                )
            if not isinstance(context, (int, float)) or context <= 0:
                raise ValueError(
                    f"Scaling model {identifier} has no positive GIFT context length"
                )

    return catalogue


def model_index(catalogue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in catalogue["models"]}


def resolve_checkpoint(
    record: dict[str, Any],
    model_root: Path,
    explicit_checkpoint: str | None,
    allow_remote: bool,
) -> str:
    if explicit_checkpoint:
        return explicit_checkpoint

    local_subdir = record.get("local_checkpoint_subdir")
    if local_subdir:
        local_path = model_root / local_subdir
        if local_path.is_dir():
            return str(local_path)

    checkpoint = record.get("checkpoint")
    if allow_remote and checkpoint:
        return str(checkpoint)

    expected = model_root / str(local_subdir) if local_subdir else model_root
    raise FileNotFoundError(
        f"No local checkpoint found for {record['id']} at {expected}. "
        "Provide --checkpoint or explicitly enable --allow-remote-checkpoint."
    )


def import_model_module(adapter: str) -> ModuleType:
    """Import the pinned repository-local inference implementation."""
    try:
        module_name = MODEL_MODULES[adapter]
    except KeyError as error:
        raise ValueError(f"Adapter {adapter} has no vendored implementation") from error
    return importlib.import_module(module_name)
