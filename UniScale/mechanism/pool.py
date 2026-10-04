"""Launch isolated two-GPU mechanism experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from UniScale.environments import python_for_environment

from .protocol import (DEFAULT_CONFIG, REPOSITORY, config_digest, digest_file,
                       load_config, output_path, run_path, write_json)
from .data import prepare_data
from .experiment import task_plan


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()


def valid_completion(root: Path, task: dict, plan: dict) -> bool:
    path = root / "tasks" / task["id"] / "complete.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if payload["config_sha256"] != plan["config_sha256"] or payload["git_commit"] != plan["git_commit"]:
        raise RuntimeError(f"Incompatible completion marker for {task['id']}")
    for relative, expected in payload["artifacts"].items():
        artifact = (root / relative).resolve()
        if root not in artifact.parents or digest_file(artifact) != expected:
            raise RuntimeError(f"Incomplete or changed task artifact: {relative}")
    return True


def acquire_task_lock(root: Path, task_id: str):
    stream = output_path(root, f"tasks/{task_id}/worker.lock").open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError(f"Task {task_id} is still running; concurrent resume is refused") from error
    return stream


def runtime_environment(root: Path, gpu: int, interpreter: str, threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    cache = output_path(root, "runtime/cache/.anchor").parent
    temporary = output_path(root, "runtime/tmp/.anchor").parent
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPOSITORY),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(cache / "huggingface"),
        "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "extensions"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"), "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"), "TMPDIR": str(temporary),
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "false",
        "CONDA_PREFIX": str(Path(interpreter).parent.parent),
        "PATH": str(Path(interpreter).parent) + os.pathsep + os.environ.get("PATH", ""),
    })
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-kinds", nargs="+",
                        help="Schedule only these registered task kinds; repeat the selection on resume")
    parser.add_argument("--datasets", nargs="+",
                        help="Restrict crossed-history scheduling to registered dataset names")
    parser.add_argument("--gpu-ids", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--server-workspace", type=Path, required=True)
    arguments = parser.parse_args()
    if len(set(arguments.gpu_ids)) != 2 or arguments.workers_per_gpu < 1:
        raise ValueError("Exactly two distinct GPUs and positive slots are required")
    boundary = arguments.server_workspace.resolve(strict=True)
    if boundary not in REPOSITORY.parents or boundary not in arguments.model_root.resolve().parents:
        raise ValueError("Project and read-only checkpoints must stay inside the approved server workspace")
    config = load_config(arguments.config)
    tasks = task_plan(config)
    if arguments.datasets is not None:
        datasets = set(arguments.datasets)
        registered = {task.get("dataset") for task in tasks if task["kind"] == "crossed_history"}
        if not datasets or datasets - registered:
            raise ValueError("Dataset selection is empty or outside the registered panel")
        tasks = [task for task in tasks if task.get("dataset") in datasets]
    if arguments.task_kinds is not None:
        selected = set(arguments.task_kinds)
        unknown = selected - {task["kind"] for task in tasks}
        if unknown:
            raise ValueError(f"Task kinds absent from this configuration: {sorted(unknown)}")
        tasks = [task for task in tasks if task["kind"] in selected]
        selected_ids = {task["id"] for task in tasks}
        if any(set(task["depends"]) - selected_ids for task in tasks):
            raise ValueError("Task selection excludes required dependencies")
    root = run_path(arguments.run)
    if not root.exists():
        if arguments.resume:
            raise FileNotFoundError("Cannot resume a nonexistent run")
        root.mkdir(parents=True, exist_ok=False)
    elif not arguments.resume:
        raise FileExistsError("Existing runs require explicit --resume")
    lock_stream = output_path(root, "coordinator.lock").open("a")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = {"protocol": config["protocol"], "config_sha256": config_digest(config),
            "git_commit": git_commit(), "tasks": tasks,
            "server_workspace": str(boundary), "model_root_read_only": str(arguments.model_root.resolve()),
            "gpu_ids": arguments.gpu_ids, "workers_per_gpu": arguments.workers_per_gpu,
            "maximum_attempts": arguments.maximum_attempts,
            "count_training_fits": 2 * sum(task["kind"] == "weights"
                                           for task in tasks)
                                   + sum(task["kind"] == "crossed_history" for task in tasks),
            "formal_results_or_checkpoints_modified": False}
    if any(task["kind"] == "crossed_history" for task in tasks):
        from .controls import input_receipt

        data_directory = (boundary / "GiftEval_data").resolve(strict=True)
        if boundary not in data_directory.parents or not data_directory.is_dir():
            raise ValueError("Controlled data must remain inside the approved read-only workspace")
        plan["input_receipt"] = input_receipt(config)
        plan["data_directory_read_only"] = str(data_directory)
    if arguments.resume:
        saved = json.loads((root / "plan.json").read_text())
        for key in ("config_sha256", "git_commit", "tasks", "server_workspace", "model_root_read_only"):
            if saved[key] != plan[key]:
                raise ValueError(f"Cannot resume a changed experiment: {key}")
        if saved.get("input_receipt") != plan.get("input_receipt"):
            raise ValueError("Cannot resume changed source checkpoints or project configuration")
        if saved.get("data_directory_read_only") != plan.get("data_directory_read_only"):
            raise ValueError("Cannot resume a changed controlled data directory")
        plan = saved
        generation = json.loads((root / "data/generation.json").read_text())
        if generation["config_sha256"] != config_digest(config):
            raise RuntimeError("Saved synthetic evidence has the wrong protocol")
        receipt = json.loads((root / "data/receipt.json").read_text())
        if digest_file(root / "data/evidence.npz") != receipt["sha256"]:
            raise RuntimeError("Saved synthetic evidence changed")
        for prefix in ("state", "transfer"):
            if digest_file(root / f"data/{prefix}_evidence.npz") != receipt[f"{prefix}_sha256"]:
                raise RuntimeError(f"Saved {prefix} evidence changed")
    else:
        write_json(root, "config.json", config)
        write_json(root, "plan.json", plan)
        prepare_data(config, root)
        receipt = {"sha256": digest_file(root / "data/evidence.npz")}
        for prefix in ("state", "transfer"):
            receipt[f"{prefix}_sha256"] = digest_file(root / f"data/{prefix}_evidence.npz")
        write_json(root, "data/receipt.json", receipt)
    # Only resumed runs can have surviving workers. Probe existing locks without
    # creating every future task directory on a slow mounted volume. Each worker
    # still inherits its locked descriptor before launch, including import time.
    if arguments.resume:
        for task in plan["tasks"]:
            if (root / "tasks" / task["id"] / "worker.lock").exists():
                stream = acquire_task_lock(root, task["id"])
                stream.close()
    completed = ({task["id"] for task in plan["tasks"] if valid_completion(root, task, plan)}
                 if arguments.resume else set())
    pending = [task for task in plan["tasks"] if task["id"] not in completed]
    failed = {}
    retrying = {}
    attempts = {}
    active = []
    slots = {gpu: arguments.workers_per_gpu for gpu in arguments.gpu_ids}
    started = time.time()
    last_report = 0.0
    print(f"Mechanism run {arguments.run}: {len(pending)} tasks, {plan['count_training_fits']} individual fits, GPUs {arguments.gpu_ids}", flush=True)
    try:
        while pending or active:
            changed = False
            for process in list(active):
                code = process["child"].poll()
                if code is None:
                    continue
                active.remove(process)
                process["log_stream"].close()
                task = process["task"]
                if code == 0 and valid_completion(root, task, plan):
                    completed.add(task["id"])
                    retrying.pop(task["id"], None)
                else:
                    failure_path = root / "tasks" / task["id"] / "failed.json"
                    detail = json.loads(failure_path.read_text()) if failure_path.exists() else {"message": f"worker exit {code}"}
                    if "out of memory" in detail.get("message", "").lower():
                        slots[process["gpu"]] = max(1, slots[process["gpu"]] // 2)
                    if attempts[task["id"]] < arguments.maximum_attempts:
                        retrying[task["id"]] = detail
                        pending.append(task)
                    else:
                        retrying.pop(task["id"], None)
                        failed[task["id"]] = detail
                    print(f"FAILED {task['id']} attempt={attempts[task['id']]}: {detail.get('message', code)}", flush=True)
                changed = True
            for task in list(pending):
                if any(dependency in failed for dependency in task["depends"]):
                    failed[task["id"]] = {"message": "A required task failed; dependent experiment not executed"}
                    pending.remove(task)
                    changed = True
                    continue
                if not all(dependency in completed for dependency in task["depends"]):
                    continue
                counts = {gpu: sum(process.get("slot_cost", 1) for process in active if process["gpu"] == gpu)
                          for gpu in slots}
                eligible = [gpu for gpu in slots if counts[gpu] + min(task.get("slot_cost", 1), slots[gpu]) <= slots[gpu]]
                if not eligible:
                    continue
                gpu = min(eligible, key=lambda selected_gpu: counts[selected_gpu])
                interpreter = python_for_environment(task["environment"])
                attempts[task["id"]] = attempts.get(task["id"], 0) + 1
                attempt = attempts[task["id"]]
                log = output_path(root, f"logs/{task['id']}-attempt{attempt}-{time.time_ns()}.log")
                log_stream = log.open("x")
                command = [interpreter, "-B", "-u", "-m", "UniScale.mechanism.worker",
                           "--run", arguments.run, "--task", task["id"],
                           "--model-root", str(arguments.model_root)]
                task_lock = acquire_task_lock(root, task["id"])
                environment = runtime_environment(root, gpu, interpreter, config["torch_threads"])
                if "data_directory_read_only" in plan:
                    environment["GIFT_EVAL"] = plan["data_directory_read_only"]
                environment["UNISCALE_TASK_LOCK_FD"] = str(task_lock.fileno())
                try:
                    child = subprocess.Popen(command, cwd=REPOSITORY, env=environment,
                        stdout=log_stream, stderr=subprocess.STDOUT, pass_fds=(task_lock.fileno(),))
                finally:
                    task_lock.close()
                active.append({"task": task, "child": child, "gpu": gpu,
                               "slot_cost": min(task.get("slot_cost", 1), slots[gpu]),
                               "log_stream": log_stream, "log": str(log.relative_to(root))})
                pending.remove(task)
                print(f"START {task['id']} GPU={gpu} PID={child.pid} attempt={attempt}", flush=True)
                changed = True
            if changed or time.time() - last_report >= 60:
                write_json(root, "status.json", {
                    "pid": os.getpid(), "total": len(plan["tasks"]), "complete": sorted(completed),
                    "pending": [task["id"] for task in pending], "failed": failed,
                    "retrying": retrying,
                    "active": [{"task": process["task"]["id"], "pid": process["child"].pid,
                                "gpu": process["gpu"], "log": process["log"]} for process in active],
                    "slots_per_gpu": slots, "elapsed_seconds": time.time() - started,
                    "state": "running" if pending or active else ("failed" if failed else "complete"),
                })
                last_report = time.time()
            if pending or active:
                time.sleep(2)
    except BaseException:
        # Only this launcher's children are ever stopped. Existing experiments are untouched.
        for process in active:
            process["child"].terminate()
        raise
    finally:
        for process in active:
            process["log_stream"].close()
        lock_stream.close()
    if failed:
        raise RuntimeError(f"Mechanism tasks failed: {sorted(failed)}")
    print(f"COMPLETE {len(completed)} mechanism tasks", flush=True)


if __name__ == "__main__":
    main()
