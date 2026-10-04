"""Canonical output paths owned by UniScale."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
RUN_TIMESTAMP_PATTERN = re.compile(r"^(?:release|\d{8}T\d{12,15}[+-]\d{4})(?:_seed\d+)?$")


def resolve_results_path(
    repository: Path,
    configured_path: str | Path | None = None,
    default_subdir: str | Path | None = None,
) -> Path:
    """Resolve a project-owned output path inside ``UniScale/results``."""
    root = (repository / "UniScale" / "results").resolve()
    if configured_path is None:
        target = root if default_subdir is None else root / default_subdir
    else:
        candidate = Path(configured_path)
        target = candidate if candidate.is_absolute() else repository / candidate
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ValueError(
            f"UniScale outputs must remain under {root}; resolved output was {target}"
        )
    return target


def new_run_timestamp() -> str:
    """Return a filesystem-safe, high-resolution UTC run timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f%z")


def validate_experiment_name(value: str) -> str:
    """Validate a short phrase suitable for a stable result-directory name."""
    if not EXPERIMENT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "experiment_name must be a lowercase hyphenated phrase, "
            "for example context-scaling"
        )
    return value


def validate_run_timestamp(value: str) -> str:
    """Validate a run timestamp or the bundled result identifier."""
    if not RUN_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(
            "run_timestamp must be release or YYYYMMDDTHHMMSSffffff+HHMM, with an optional _seedNUMBER suffix"
        )
    return value


def resolve_run_path(
    repository: Path,
    experiment_name: str,
    run_timestamp: str,
) -> Path:
    """Resolve ``results/<experiment-name>/<run-timestamp>``."""
    return (
        resolve_results_path(repository)
        / validate_experiment_name(experiment_name)
        / validate_run_timestamp(run_timestamp)
    )
