"""One isolated, resumable GPU mechanism task."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from .protocol import (config_digest, digest_file, load_config, output_path, run_path, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = run_path(arguments.run)
    config = load_config(root / "config.json")
    task = arguments.task
    with (root / "plan.json").open() as stream:
        plan = json.load(stream)
    record = next((item for item in plan["tasks"] if item["id"] == task), None)
    if record is None:
        raise ValueError("Task is absent from this run's immutable plan")
    if config_digest(config) != plan["config_sha256"]:
        raise ValueError("Run configuration changed after planning")
    if config["protocol"] == "history-learning-v5":
        from .protocol import REPOSITORY

        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
        if current != plan["git_commit"]:
            raise RuntimeError("Mechanism code changed while tasks were queued")
    if record["kind"] == "crossed_history":
        from .controls import project_file

        for relative, expected in plan["input_receipt"]["project_files"].items():
            if digest_file(project_file(relative)) != expected:
                raise RuntimeError("Pinned crossed-history configuration changed")
    evidence_receipt = json.loads((root / "data/receipt.json").read_text())
    if digest_file(root / "data/evidence.npz") != evidence_receipt["sha256"]:
        raise RuntimeError("Synthetic evidence changed after planning")
    if record["kind"] == "weights" and digest_file(root / "data/state_evidence.npz") != evidence_receipt["state_sha256"]:
        raise RuntimeError("State-transfer evidence changed after planning")
    if record["kind"].startswith("specificity_") and digest_file(root / "data/transfer_evidence.npz") != evidence_receipt["transfer_sha256"]:
        raise RuntimeError("Activation-transfer evidence changed after planning")
    lock_path = output_path(root, f"tasks/{task}/worker.lock")
    inherited_lock = os.environ.get("UNISCALE_TASK_LOCK_FD")
    if inherited_lock is None:
        task_lock = lock_path.open("a")
        fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        task_lock = os.fdopen(int(inherited_lock), "a")
        if os.fstat(task_lock.fileno()).st_ino != lock_path.stat().st_ino:
            raise RuntimeError("Inherited task lock does not match this task")
    started = time.time()
    try:
        import torch

        torch.set_num_threads(config["torch_threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["data_seed"])
        torch.cuda.manual_seed_all(config["data_seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each mechanism worker must see exactly one assigned GPU")
        write_json(root, f"tasks/{task}/started.json", {
            "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_at": started, "task": record, "config_sha256": plan["config_sha256"],
        })
        from .experiment import run_task

        run_task(config, root, record, arguments.model_root)
        artifacts = {
            str(path.relative_to(root)): digest_file(path)
            for path in sorted((root / "tasks" / task).iterdir())
            if path.is_file() and path.name not in {"started.json", "complete.json", "failed.json", "worker.lock"}
            and not path.name.endswith(".tmp")
        }
        if not artifacts:
            raise RuntimeError("Task produced no durable artifacts")
        write_json(root, f"tasks/{task}/complete.json", {
            "config_sha256": plan["config_sha256"], "git_commit": plan["git_commit"],
            "artifacts": artifacts, "elapsed_seconds": time.time() - started,
            "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        print(f"COMPLETE {task}", flush=True)
    except Exception as error:
        write_json(root, f"tasks/{task}/failed.json", {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started,
        })
        raise
    finally:
        task_lock.close()


if __name__ == "__main__":
    main()
