"""Keep every local GPU slot filled from a dynamic full-shot task pool."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from UniScale.experiments.matched_history import (
    FULL_SHOT_PROTOCOL,
    _existing_keys,
    _read_json,
    _recursive_existing_keys,
    _task_ordinal,
    _validate_run_seed,
    load_config,
    merge_dynamic_results,
)
from UniScale.experiments.dataset_scope import dataset_configuration
from UniScale.io_utils import atomic_write_text, ensure_directory
from UniScale.paths import new_run_timestamp, resolve_run_path, validate_run_timestamp


@dataclass(frozen=True)
class PoolTask:
    ordinal: int
    model: str
    run_key: str
    horizon: int
    training_length: int
    seed: int


@dataclass(frozen=True)
class SeedRun:
    config_path: Path
    timestamp: str
    root: Path
    config: dict


def _seed_runs(config_path: Path, timestamp: str, seeds: list[int] | None, resume: bool) -> dict[int, SeedRun]:
    """Resolve isolated seed archives while retaining every training-policy override."""
    config = load_config(config_path)
    repository = Path(__file__).resolve().parents[2]
    validate_run_timestamp(timestamp)
    selected_seeds = seeds is not None
    if seeds is not None:
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError("seeds must be distinct integers in [0, 2**32)")
        if "_seed" in timestamp:
            raise ValueError("With --seeds, supply the common timestamp without a seed suffix")
    else:
        seeds = [int(config["training"]["seed"])]
        _validate_run_seed(timestamp, config)
    explicit_seeds = selected_seeds or ("_seed" not in timestamp and not resume)
    runs = {}
    for seed in seeds:
        resolved = deepcopy(config)
        if selected_seeds:
            resolved["training"]["seed"] = seed
            for spec in resolved["models"]:
                if "seed" in spec.get("training", {}):
                    spec["training"]["seed"] = seed
        run_timestamp = f"{timestamp}_seed{seed}" if explicit_seeds else timestamp
        _validate_run_seed(run_timestamp, resolved)
        root = resolve_run_path(repository, resolved["experiment_name"], run_timestamp)
        if resume:
            if not root.is_dir():
                raise FileNotFoundError(f"Cannot resume missing run directory: {root}")
        elif root.exists():
            raise FileExistsError(f"Fresh run already exists; use --resume: {root}")
        resolved.pop("datasets_file", None)
        if explicit_seeds or not resume:
            path = root / "config.json"
            if resume and (not path.is_file() or _read_json(path) != resolved):
                raise ValueError(f"Resume configuration differs from the archived seed configuration: {path}")
        else:
            path = config_path
        runs[seed] = SeedRun(path, run_timestamp, root, resolved)
    # Validate every destination before writing any fresh run configuration.
    if not resume:
        for run in runs.values():
            run.root.mkdir(parents=True, exist_ok=False)
            atomic_write_text(run.config_path, json.dumps(run.config, indent=2, sort_keys=True) + "\n")
    return runs


@dataclass
class ActiveAttempt:
    task: PoolTask
    process: subprocess.Popen[str]
    log: TextIO
    slot_index: int
    attempt: int
    log_start: int


def _attempt_had_cuda_oom(attempt: ActiveAttempt) -> bool:
    """Inspect only this attempt's bounded log tail after the worker exits."""
    path = Path(attempt.log.name)
    with path.open("rb") as handle:
        handle.seek(max(attempt.log_start, path.stat().st_size - 65536))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return any(marker in tail for marker in (
        "cuda out of memory", "cuda error: out of memory",
        "torch.outofmemoryerror", "torch.cuda.outofmemoryerror",
        "cudnn_status_alloc_failed",
    ))


def _tasks(
    config: dict,
    properties: dict,
    host_index: int,
    host_count: int,
) -> list[PoolTask]:
    tasks: list[PoolTask] = []
    origin_policy = f"shared_origin_v1:Hmax={int(config['origin_horizon'])}"
    for model_index, model_spec in enumerate(config["models"]):
        model = str(model_spec["name"])
        for horizon_index, horizon in enumerate(config["H"]):
            for dataset_index, dataset_spec in enumerate(config["datasets"]):
                name = str(dataset_spec["name"])
                term = str(dataset_spec.get("terms", ["short"])[0])
                dataset = dataset_configuration(name, term, properties)[0]
                for length_index, training_length in enumerate(config["L"]):
                    ordinal = _task_ordinal(
                        config,
                        model_index,
                        horizon_index,
                        dataset_index,
                        length_index,
                    )
                    if ordinal % host_count != host_index:
                        continue
                    tasks.append(
                        PoolTask(
                            ordinal=ordinal,
                            model=model,
                            run_key=(
                                f"{dataset}|H={horizon}|L={training_length}"
                                f"|ell={int(config['local_input_length'])}"
                                f"|origins={origin_policy}"
                                f"|protocol={FULL_SHOT_PROTOCOL}"
                            ),
                            horizon=int(horizon),
                            training_length=int(training_length),
                            seed=int(config["training"]["seed"]),
                        )
                    )
    return tasks


def _pending_tasks(config: dict, run_root: Path, tasks: list[PoolTask]) -> list[PoolTask]:
    completed_by_model = {
        str(model["name"]).lower(): _recursive_existing_keys(
            run_root / str(model["name"]).lower(), "all_results.csv"
        )
        for model in config["models"]
    }
    pending = [
        task
        for task in tasks
        if task.run_key not in completed_by_model[task.model.lower()]
    ]
    return sorted(
        pending,
        key=lambda task: (
            task.model.lower() == "patchtst",
            task.horizon,
            task.training_length,
            task.ordinal,
        ),
        reverse=True,
    )


def _terminate(active: dict[int, ActiveAttempt]) -> None:
    for attempt in active.values():
        if attempt.process.poll() is None:
            attempt.process.terminate()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and any(
        attempt.process.poll() is None for attempt in active.values()
    ):
        time.sleep(0.25)
    for attempt in active.values():
        if attempt.process.poll() is None:
            attempt.process.kill()
        attempt.log.close()


def run_pool(
    config_path: Path,
    timestamp: str,
    *,
    host_index: int,
    host_count: int,
    gpu_count: int,
    workers_per_gpu: int,
    launcher_name: str,
    maximum_attempts: int,
    gpu_ids: list[int] | None = None,
    expected_pending_cells: int | None = None,
    allowed_datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    resume: bool = True,
    finalize: bool = False,
) -> None:
    if host_count <= 0 or not 0 <= host_index < host_count:
        raise ValueError("host_index must be within host_count")
    if gpu_count <= 0 or workers_per_gpu <= 0:
        raise ValueError("GPU and worker counts must be positive")
    if maximum_attempts <= 0:
        raise ValueError("maximum_attempts must be positive")
    if (finalize or not resume or seeds is not None) and host_count != 1:
        raise ValueError("Fresh, multi-seed, and finalizing pools require one coordinated host")
    if not launcher_name or Path(launcher_name).name != launcher_name or launcher_name in (".", ".."):
        raise ValueError("launcher_name must be a single directory name")
    devices = list(range(gpu_count)) if gpu_ids is None else list(gpu_ids)
    if len(devices) != gpu_count or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError("gpu_ids must contain one distinct nonnegative physical index per GPU")

    repository = Path(__file__).resolve().parents[2]
    runs = _seed_runs(config_path, timestamp, seeds, resume)
    coordinator = next(iter(runs.values()))
    config, run_root = coordinator.config, coordinator.root
    slot_count = gpu_count * workers_per_gpu
    total_slots = host_count * slot_count
    properties_path = repository / config["execution"].get(
        "dataset_properties", "UniScale/information/dataset_properties.json"
    )
    properties = _read_json(properties_path)
    pending = sorted(
        (task for run in runs.values()
         for task in _pending_tasks(run.config, run.root, _tasks(run.config, properties, host_index, host_count))),
        key=lambda task: (task.model.lower() == "patchtst", task.horizon,
                          task.training_length, task.ordinal, task.seed),
        reverse=True,
    )
    if expected_pending_cells is not None and len(pending) != expected_pending_cells:
        raise ValueError(f"Expected {expected_pending_cells} pending cells, found {len(pending)}")
    if allowed_datasets is not None:
        allowed = set(allowed_datasets)
        unexpected = [task.run_key for task in pending if task.run_key.split("|", 1)[0] not in allowed]
        if unexpected:
            raise ValueError(f"Pending tasks outside the allowed datasets: {unexpected}")
    attempts: dict[tuple[int, int], int] = {(task.seed, task.ordinal): 0 for task in pending}
    gpu_limits = {gpu: workers_per_gpu for gpu in devices}
    launch_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    slot_order = [
        gpu_position * workers_per_gpu + lane
        for lane in range(workers_per_gpu)
        for gpu_position in (range(gpu_count) if lane % 2 == 0 else reversed(range(gpu_count)))
    ]
    active: dict[int, ActiveAttempt] = {}
    failures: list[str] = []
    log_root = run_root / "launcher_logs" / launcher_name
    status_root = run_root / "launchers"
    ensure_directory(log_root)
    ensure_directory(status_root)
    receipt = {
        "started_at": new_run_timestamp(), "pid": os.getpid(), "git_commit": launch_commit,
        "gpu_ids": devices,
        "workers_per_gpu": workers_per_gpu, "slot_count": slot_count,
        "maximum_attempts": maximum_attempts, "resume": resume,
        "runs": [{"seed": seed, "timestamp": run.timestamp,
                  "pending_cells": sum(task.seed == seed for task in pending)}
                 for seed, run in runs.items()],
        "pending_tasks": [{"seed": task.seed, "ordinal": task.ordinal,
                           "model": task.model, "run_key": task.run_key} for task in pending],
    }
    atomic_write_text(log_root / "launch_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    stopping = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(
        f"dynamic pool {launcher_name}: {len(pending)} pending cells, "
        f"{slot_count} slots on {gpu_count} GPUs",
        flush=True,
    )
    print("seed runs: " + ", ".join(run.timestamp for run in runs.values()), flush=True)
    status_path = status_root / f"{launcher_name}.status"
    atomic_write_text(status_path, "running\n")
    try:
        while pending or active:
            if stopping:
                raise KeyboardInterrupt
            for slot_index in slot_order:
                if not pending or slot_index in active:
                    continue
                gpu = devices[slot_index // workers_per_gpu]
                active_on_gpu = sum(
                    devices[index // workers_per_gpu] == gpu for index in active
                )
                if slot_index % workers_per_gpu >= gpu_limits[gpu] or active_on_gpu >= gpu_limits[gpu]:
                    continue
                task = pending.pop(0)
                run = runs[task.seed]
                task_id = (task.seed, task.ordinal)
                attempts[task_id] += 1
                attempt_number = attempts[task_id]
                global_slot = host_index * slot_count + slot_index
                log_path = log_root / f"gpu-{gpu}-slot-{slot_index % workers_per_gpu}.log"
                log = log_path.open("a", encoding="utf-8")
                log_start = log.tell()
                log.write(
                    f"\nTASK seed={task.seed} ordinal={task.ordinal} model={task.model} "
                    f"H={task.horizon} L={task.training_length} "
                    f"attempt={attempt_number}\n"
                )
                log.flush()
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "UniScale.experiments.matched_history",
                    "--config",
                    str(run.config_path),
                    "--resume",
                    run.timestamp,
                    "--model",
                    task.model.lower(),
                    "--task-ordinal",
                    str(task.ordinal),
                    "--task-shard-index",
                    str(global_slot),
                    "--task-shard-count",
                    str(total_slots),
                    "--retry-skipped",
                    "--require-result",
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=repository,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[slot_index] = ActiveAttempt(
                    task=task,
                    process=process,
                    log=log,
                    slot_index=slot_index,
                    attempt=attempt_number,
                    log_start=log_start,
                )
                print(
                    f"launched seed {task.seed} task {task.ordinal} on GPU {gpu} "
                    f"slot {slot_index % workers_per_gpu} pid={process.pid}",
                    flush=True,
                )

            finished = [
                slot_index
                for slot_index, attempt in active.items()
                if attempt.process.poll() is not None
            ]
            for slot_index in finished:
                attempt = active.pop(slot_index)
                return_code = int(attempt.process.returncode or 0)
                attempt.log.close()
                output_root = (
                    runs[attempt.task.seed].root
                    / attempt.task.model.lower()
                    / "shards"
                    / f"shard-{host_index * slot_count + slot_index:03d}-of-{total_slots:03d}"
                )
                result_keys = _existing_keys(output_root / "all_results.csv")
                if return_code == 0 and attempt.task.run_key in result_keys:
                    print(f"completed seed {attempt.task.seed} task {attempt.task.ordinal}", flush=True)
                    continue
                gpu = devices[slot_index // workers_per_gpu]
                if _attempt_had_cuda_oom(attempt):
                    previous_limit = gpu_limits[gpu]
                    gpu_limits[gpu] = max(1, previous_limit // 2)
                    print(
                        f"CUDA OOM on GPU {gpu}: slot limit {previous_limit} -> {gpu_limits[gpu]}; "
                        "running workers drain normally; training batch sizes are unchanged",
                        flush=True,
                    )
                if attempt.attempt < maximum_attempts:
                    pending.append(attempt.task)
                    print(
                        f"retrying seed {attempt.task.seed} task {attempt.task.ordinal} after return code "
                        f"{return_code}",
                        flush=True,
                    )
                else:
                    failures.append(
                        f"seed {attempt.task.seed} task {attempt.task.ordinal} ({attempt.task.model}, "
                        f"H={attempt.task.horizon}, L={attempt.task.training_length}) "
                        f"failed {attempt.attempt} attempts"
                    )
            if failures:
                raise RuntimeError("; ".join(failures))
            if not finished:
                time.sleep(1.0)
        if finalize or not resume:
            for run in runs.values():
                merge_dynamic_results(run.config_path, run.timestamp)
    except BaseException:
        _terminate(active)
        atomic_write_text(status_path, "1\n")
        raise

    atomic_write_text(status_path, "0\n")
    atomic_write_text(
        status_root / f"{launcher_name}.json",
        json.dumps(
            {
                "host_index": host_index,
                "host_count": host_count,
                "gpu_count": gpu_count,
                "workers_per_gpu": workers_per_gpu,
                "final_gpu_slot_limits": gpu_limits,
                "finalized_complete_runs": finalize or not resume,
                "seeds": list(runs),
                "run_timestamps": [run.timestamp for run in runs.values()],
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"dynamic pool {launcher_name} completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    timestamps = parser.add_mutually_exclusive_group()
    timestamps.add_argument("--resume", help="Existing timestamp; common unsuffixed timestamp with --seeds")
    timestamps.add_argument("--run-timestamp", help="New timestamp; generated automatically when omitted")
    parser.add_argument("--seeds", type=int, nargs="+", help="Train all listed seeds through one shared slot pool")
    parser.add_argument("--host-index", type=int, required=True)
    parser.add_argument("--host-count", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="Physical GPU indices exposed to workers")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--launcher-name", required=True)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--expected-pending-cells", type=int)
    parser.add_argument("--allowed-dataset", action="append", dest="allowed_datasets")
    parser.add_argument("--finalize", action="store_true", help="Publish each complete resumed seed run; automatic for fresh runs")
    args = parser.parse_args()
    run_pool(
        args.config.resolve(),
        args.resume or args.run_timestamp or new_run_timestamp(),
        host_index=args.host_index,
        host_count=args.host_count,
        gpu_count=args.gpu_count,
        workers_per_gpu=args.workers_per_gpu,
        launcher_name=args.launcher_name,
        maximum_attempts=args.maximum_attempts,
        gpu_ids=args.gpu_ids,
        expected_pending_cells=args.expected_pending_cells,
        allowed_datasets=args.allowed_datasets,
        seeds=args.seeds,
        resume=args.resume is not None,
        finalize=args.finalize,
    )


if __name__ == "__main__":
    main()
