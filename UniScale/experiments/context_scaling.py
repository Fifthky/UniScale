"""Launch the full-checkpoint controlled context-scaling experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from UniScale.orchestration.checkpoint_scheduler import load_batch_config, run_checkpoint_batch


EXPECTED_L = [128, 256, 512, 1024, 2048, 4096, 6144, 8192]


def validate_context_scaling_config(config: dict[str, Any]) -> None:
    if config.get("experiment_name") != "context-scaling":
        raise ValueError("experiment_name must be context-scaling")
    if "H" in config or "origin_horizon" in config or "cells" in config:
        raise ValueError("Context scaling must use each dataset-frequency default H")
    if config.get("L") != EXPECTED_L:
        raise ValueError(f"Context-scaling L must be {EXPECTED_L}")
    if len(config.get("models", [])) != 21:
        raise ValueError("Context scaling requires the full 21-checkpoint panel")
    datasets = config.get("datasets", [])
    configuration_count = sum(
        len(dataset.get("terms", ["short"]))
        for dataset in datasets
    )
    if len(datasets) != 23 or configuration_count != 61:
        raise ValueError(
            "Default-H acquisition requires 61 GIFT configurations "
            "across 23 dataset-frequency entries"
        )
    # load_batch_config validates GPU ids and the configured OOM concurrency ladder.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    timestamp_group = parser.add_mutually_exclusive_group()
    timestamp_group.add_argument("--run-timestamp")
    timestamp_group.add_argument("--resume", metavar="TIMESTAMP")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_batch_config(config_path)
    validate_context_scaling_config(config)
    succeeded = run_checkpoint_batch(
        config_path,
        args.resume or args.run_timestamp,
        resume=args.resume is not None,
    )
    raise SystemExit(0 if succeeded else 1)


if __name__ == "__main__":
    main()
