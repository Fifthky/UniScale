"""Resolve named inference environments without activating or modifying them."""

from __future__ import annotations

import os
from pathlib import Path


ENVIRONMENTS = ("TSFM", "granite", "tirex2", "toto")


def python_for_environment(name: str) -> str:
    """Use an explicitly configured interpreter or environment root."""
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown inference environment: {name}")
    override = os.environ.get(f"UNISCALE_PYTHON_{name.upper()}")
    root = os.environ.get("UNISCALE_ENV_ROOT")
    if not override and not root:
        raise ValueError(f"Set UNISCALE_PYTHON_{name.upper()} or UNISCALE_ENV_ROOT")
    executable = Path(override) if override else Path(root) / name / "bin/python"
    if not executable.is_absolute():
        raise ValueError(f"The {name} interpreter must be an absolute path")
    return str(executable)
