"""Launch the full controlled H-by-L scaling experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from UniScale.orchestration.checkpoint_scheduler import (
    load_batch_config,
    run_checkpoint_batch,
)


EXPECTED_H = [48, 96, 192, 336, 720]
EXPECTED_L = [128, 256, 512, 1024, 2048, 4096, 6144, 8192]


def validate(config):
    if config.get("experiment_name") != "joint-scaling":
        raise ValueError("experiment_name must be joint-scaling")
    if config.get("H") != EXPECTED_H or config.get("L") != EXPECTED_L:
        raise ValueError(
            f"Joint scaling requires H={EXPECTED_H} and L={EXPECTED_L}"
        )
    if config.get("origin_horizon") != 720:
        raise ValueError("Joint scaling requires shared origins anchored at 720")
    if len(config["models"]) != 21 or len(config["datasets"]) != 23:
        raise ValueError("Joint scaling requires 21 checkpoints and the 23-entry evaluation panel")
    # load_batch_config validates GPU ids and the configured OOM concurrency ladder.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    timestamp_group = parser.add_mutually_exclusive_group()
    timestamp_group.add_argument("--run-timestamp")
    timestamp_group.add_argument("--resume", metavar="TIMESTAMP")
    args = parser.parse_args()
    path = args.config.resolve()
    validate(load_batch_config(path))
    raise SystemExit(
        0
        if run_checkpoint_batch(
            path,
            args.resume or args.run_timestamp,
            resume=args.resume is not None,
        )
        else 1
    )


if __name__ == "__main__":
    main()
