"""Run instance-sharded checkpoints with a central multi-GPU task scheduler."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from UniScale.experiments.dataset_scope import configured_dataset_ids
from UniScale.io_utils import ResilientLog, atomic_write_csv, atomic_write_text, ensure_directory
from UniScale.model_registry import load_catalogue, model_index
from UniScale.orchestration.shard_results import merge_model_shards
from UniScale.orchestration.grid_reuse import plan_grid_reuse, persist_grid_plan
from UniScale.paths import new_run_timestamp, resolve_run_path, validate_run_timestamp


OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "hip out of memory",
    "outofmemoryerror",
    "resource exhausted: oom",
)
OOM_RETURN_CODES = {137, -9}
STORAGE_FAILURE_MARKERS = (
    "no space left on device",
    "disk quota exceeded",
    "storage write failed after",
)
STATE_FLUSH_SECONDS = 300.0


class IncompleteResultError(ValueError):
    """Raised when a completed model lacks its exact expected result cells."""


def validate_result_file(
    path: Path,
    expected_cells: int,
    allowed_datasets: set[str] | None = None,
) -> None:
    if not path.is_file():
        raise IncompleteResultError(f"Missing result file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_keys = [row.get("run_key", "") for row in rows]
    datasets = [row.get("dataset", "") for row in rows]
    if (
        len(rows) != expected_cells
        or len(set(run_keys)) != expected_cells
        or any(not run_key for run_key in run_keys)
    ):
        raise IncompleteResultError(
            f"Expected {expected_cells} unique cells, found "
            f"{len(rows)} rows and {len(set(run_keys))} unique run keys"
        )
    if allowed_datasets is not None and any(
        dataset not in allowed_datasets for dataset in datasets
    ):
        unexpected = sorted(set(datasets).difference(allowed_datasets))
        raise IncompleteResultError(
            f"Result file contains datasets outside the configured scope: {unexpected}"
        )


@dataclass(frozen=True)
class CheckpointTask:
    model_id: str
    python_executable: str
    shard_index: int
    shard_count: int = 1

    @property
    def task_id(self) -> str:
        return (
            f"{self.model_id}::part-{self.shard_index:05d}-of-{self.shard_count:05d}"
        )


@dataclass
class Attempt:
    task: CheckpointTask
    gpu: int
    concurrency_limit: int
    number: int
    started_at: str
    started_monotonic: float
    log_path: Path
    log_thread: threading.Thread
    log_errors: list[str]
    process: subprocess.Popen[Any]


@dataclass
class GPUState:
    gpu: int
    maximum: int
    active: list[Attempt] = field(default_factory=list)


@dataclass
class TaskState:
    task: CheckpointTask
    phase: str
    next_concurrency_limit: int
    last_gpu: int | None
    ready_sequence: int


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_oom_failure(return_code: int, log_text: str) -> bool:
    lowered = log_text.lower()
    return return_code in OOM_RETURN_CODES or any(marker in lowered for marker in OOM_MARKERS)


def is_storage_failure(log_text: str) -> bool:
    lowered = log_text.lower()
    return any(marker in lowered for marker in STORAGE_FAILURE_MARKERS)


def classify_failure(return_code: int, log_text: str) -> str | None:
    """Keep exhausted storage writes separate from model-memory recovery."""
    if return_code == 0:
        return None
    if is_storage_failure(log_text):
        return "storage"
    if is_oom_failure(return_code, log_text):
        return "oom"
    return "process"


def next_oom_limit(current: int) -> int | None:
    if current <= 1:
        return None
    return max(1, current // 2)


def worker_environment(repository: Path, gpu: int, python_executable: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(repository)
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": (
                f"{source_path}{os.pathsep}{inherited_pythonpath}"
                if inherited_pythonpath
                else source_path
            ),
        }
    )
    if python_executable is not None:
        prefix = Path(python_executable).parent.parent
        environment["CONDA_PREFIX"] = str(prefix)
        environment["PATH"] = str(prefix / "bin") + os.pathsep + environment.get("PATH", "")
    return environment


def assign_instance_shards(
    models: list[dict[str, Any]], gpu_count: int
) -> list[CheckpointTask]:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    tasks: list[CheckpointTask] = []
    for model in models:
        for shard_index in range(gpu_count):
            tasks.append(
                CheckpointTask(
                    model_id=str(model["id"]),
                    python_executable=str(model["python_executable"]),
                    shard_index=shard_index,
                    shard_count=gpu_count,
                )
            )
    return tasks


def load_batch_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("models", "datasets"):
        reference = config.get(f"{key}_file")
        if reference:
            referenced_path = (path.parent / str(reference)).resolve()
            config[key] = json.loads(referenced_path.read_text(encoding="utf-8"))
    dataset_terms = config.get("dataset_terms")
    if dataset_terms is not None:
        if (
            not isinstance(dataset_terms, list)
            or not dataset_terms
            or any(not isinstance(term, str) or not term for term in dataset_terms)
        ):
            raise ValueError("dataset_terms must be a non-empty list of term names")
        config["datasets"] = [
            {**dataset, "terms": list(dataset_terms)}
            for dataset in config.get("datasets", [])
        ]
    required = {"experiment_name", "models", "datasets", "scheduler"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Batch config is missing: {', '.join(missing)}")
    if not config["models"]:
        raise ValueError("models must be non-empty")
    model_ids = [str(model.get("id", "")) for model in config["models"]]
    if any(not model_id for model_id in model_ids):
        raise ValueError("Every model requires a non-empty id")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Model ids must be unique within an experiment")
    for model in config["models"]:
        if "environment" in model:
            from UniScale.environments import python_for_environment
            model["python_executable"] = python_for_environment(model["environment"])
        executable = Path(str(model.get("python_executable", "")))
        if not executable.is_absolute():
            raise ValueError(f"python_executable must be absolute for {model['id']}")
    scheduler = config["scheduler"]
    gpus = [int(value) for value in scheduler.get("gpus", [])]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(
            "scheduler.gpus must contain one or more distinct non-negative GPU ids"
        )
    maximum = int(scheduler.get("max_workers_per_gpu", 8))
    if maximum <= 0:
        raise ValueError("max_workers_per_gpu must be positive")
    levels = [int(value) for value in scheduler.get("oom_concurrency", [8, 4, 2, 1])]
    expected = []
    value = maximum
    while True:
        expected.append(value)
        if value == 1:
            break
        value = max(1, value // 2)
    if levels != expected:
        raise ValueError(f"oom_concurrency must be {expected}")
    config["scheduler"]["gpus"] = gpus
    config["scheduler"]["max_workers_per_gpu"] = maximum
    config["scheduler"]["oom_concurrency"] = levels
    shard_count = int(scheduler.get("instance_shards", len(gpus)))
    if shard_count <= 0:
        raise ValueError("scheduler.instance_shards must be positive")
    config["scheduler"]["instance_shards"] = shard_count
    if config["experiment_name"].startswith("context-scaling"):
        config.setdefault("grid_reuse", {
            "experiment_name": config["experiment_name"].replace("context-scaling", "joint-scaling", 1)
        })
    return config


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in value)


def _write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_task_table(path: Path, records: dict[str, dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "model_id",
        "instance_shard_index",
        "instance_shard_count",
        "sequence",
        "gpu",
        "status",
        "started_at",
        "ended_at",
        "wall_seconds",
        "attempts",
        "oom_retries",
        "last_concurrency_limit",
        "next_concurrency_limit",
        "result_path",
        "error",
    ]
    atomic_write_csv(
        path,
        [{field: records[model_id].get(field, "") for field in fields} for model_id in records],
        fields,
    )


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resume_concurrency_limit(
    previous: dict[str, Any] | None, maximum: int
) -> int:
    if previous is None:
        return maximum
    explicit = previous.get("next_concurrency_limit")
    if explicit not in (None, ""):
        return max(1, min(maximum, int(explicit)))
    candidates = [maximum]
    last_limit = previous.get("last_concurrency_limit")
    if last_limit not in (None, ""):
        candidates.append(int(last_limit))
    oom_attempts = [
        int(attempt["concurrency_limit"])
        for attempt in previous.get("attempt_history", [])
        if attempt.get("oom") and attempt.get("concurrency_limit") not in (None, "")
    ]
    if oom_attempts:
        lower = next_oom_limit(min(oom_attempts))
        candidates.append(lower if lower is not None else 1)
    if previous.get("status") == "retrying_after_oom" and last_limit not in (None, ""):
        lower = next_oom_limit(int(last_limit))
        candidates.append(lower if lower is not None else 1)
    return max(1, min(candidates))


class CheckpointScheduler:
    def __init__(
        self,
        config_path: Path,
        run_timestamp: str | None = None,
        *,
        resume: bool = False,
    ):
        self.config_path = config_path.resolve()
        self.config = load_batch_config(self.config_path)
        self.repository = Path(__file__).resolve().parents[2]
        self.experiment_name = str(self.config["experiment_name"])
        if resume and run_timestamp is None:
            raise ValueError("resume requires an existing run timestamp")
        self.resume = resume
        self.run_timestamp = validate_run_timestamp(
            run_timestamp or new_run_timestamp()
        )
        self.git_commit = _git_revision(self.repository)
        self.result_root = resolve_run_path(
            self.repository, self.experiment_name, self.run_timestamp
        )
        self.scheduler_root = self.result_root / "_scheduler"
        self.config_root = self.scheduler_root / "configs"
        self.log_root = self.scheduler_root / "logs"
        self.summary_path = self.scheduler_root / "summary.json"
        self.table_path = self.scheduler_root / "tasks.csv"
        self._last_persist: float | None = None
        self._model_config_paths: dict[str, Path] = {}
        self._canonical_configs_written: set[str] = set()
        scheduler = self.config["scheduler"]
        self.gpus = scheduler["gpus"]
        self.maximum = scheduler["max_workers_per_gpu"]
        self.shard_count = scheduler["instance_shards"]
        self.models = model_index(load_catalogue())
        properties_path = self.repository / self.config.get("execution", {}).get(
            "dataset_properties", "UniScale/information/dataset_properties.json"
        )
        properties = json.loads(properties_path.read_text(encoding="utf-8"))
        self.allowed_datasets = configured_dataset_ids(
            self.config["datasets"], properties
        )
        self.expected_cells_by_model = {
            str(model["id"]): len(self.allowed_datasets)
            * self._condition_count(str(model["id"]))
            for model in self.config["models"]
        }
        tasks = assign_instance_shards(self.config["models"], self.shard_count)
        previous_summary = self._load_resume_summary(tasks) if resume else None
        self.grid_reuse_plan = plan_grid_reuse(
            self.repository, self.config, self.run_timestamp, properties,
            self.models, self.expected_cells_by_model,
        )
        for model_id, record in self.grid_reuse_plan["models"].items():
            self.expected_cells_by_model[model_id] = record["native_expected_cells"]
        previous_model_records = (
            {
                str(record["model_id"]): record
                for record in previous_summary.get("tasks", [])
            }
            if previous_summary is not None
            else {}
        )
        previous_shard_records = (
            {
                str(record["task_id"]): record
                for record in previous_summary.get("shard_tasks", [])
            }
            if previous_summary is not None
            else {}
        )
        completed_tasks = {
            task.task_id
            for task in tasks
            if self.expected_cells_by_model[task.model_id] == 0
            or self._resume_task_is_complete(
                task, previous_model_records, previous_shard_records
            )
        }
        self.states = [GPUState(gpu=gpu, maximum=self.maximum) for gpu in self.gpus]
        self.records = {
            task.task_id: self._initial_record(
                task,
                previous_shard_records.get(task.task_id),
                task.task_id in completed_tasks,
            )
            for task in tasks
        }
        self.ready_sequence = 0
        self.task_states: dict[str, TaskState] = {}
        self.fresh_queue: deque[str] = deque()
        self.recovery_queue: deque[str] = deque()
        self.active_by_task: dict[str, Attempt] = {}
        self.gpu_cursor = 0
        for task in tasks:
            completed = task.task_id in completed_tasks
            previous = previous_shard_records.get(task.task_id)
            limit = _resume_concurrency_limit(previous, self.maximum)
            if completed:
                phase = "terminal"
            elif limit < self.maximum:
                phase = "recovery"
            else:
                phase = "fresh"
            last_gpu = None
            if previous and previous.get("gpu") not in (None, ""):
                last_gpu = int(previous["gpu"])
            task_state = TaskState(
                task=task,
                phase=phase,
                next_concurrency_limit=limit,
                last_gpu=last_gpu,
                ready_sequence=self._next_ready_sequence(),
            )
            self.task_states[task.task_id] = task_state
            self.records[task.task_id]["next_concurrency_limit"] = limit
            if phase == "fresh":
                self.fresh_queue.append(task.task_id)
            elif phase == "recovery":
                self.records[task.task_id]["status"] = "retrying_after_oom"
                self.recovery_queue.append(task.task_id)
        self.started_at = (
            str(previous_summary.get("started_at", timestamp()))
            if previous_summary is not None
            else timestamp()
        )
        self.resume_events = (
            list(previous_summary.get("resume_events", []))
            if previous_summary is not None
            else []
        )
        if previous_summary is not None:
            self.resume_events.append(
                {
                    "resumed_at": timestamp(),
                    "git_commit": self.git_commit,
                    "previous_git_commit": previous_summary.get("git_commit", ""),
                    "previous_gpus": previous_summary.get("gpus", []),
                    "gpus": self.gpus,
                    "instance_shards": self.shard_count,
                    "panel_expansion": previous_summary.get("panel_expansion", {}),
                    "skipped_completed_shards": sorted(completed_tasks),
                    "rerun_shards": sorted(
                        task.task_id
                        for task in tasks
                        if task.task_id not in completed_tasks
                    ),
                }
            )

    def _next_ready_sequence(self) -> int:
        self.ready_sequence += 1
        return self.ready_sequence

    def _condition_count(self, model_id: str) -> int:
        maximum = self.models[model_id]["context"].get("maximum")
        if "cells" in self.config:
            return sum(
                1
                for cell in self.config["cells"]
                if maximum is None or int(cell["L"]) <= int(maximum)
            )
        horizons = self.config.get("H", [None])
        lengths = [
            value
            for value in self.config.get("L", [None])
            if value is None or maximum is None or int(value) <= int(maximum)
        ]
        return len(horizons) * len(lengths)

    def _load_resume_summary(
        self, tasks: list[CheckpointTask]
    ) -> dict[str, Any]:
        if not self.result_root.is_dir():
            raise FileNotFoundError(
                f"Resume run directory does not exist: {self.result_root}"
            )
        if not self.summary_path.is_file():
            raise FileNotFoundError(
                f"Resume scheduler summary does not exist: {self.summary_path}"
            )
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        if summary.get("experiment_name") != self.experiment_name:
            raise ValueError("Resume summary experiment_name does not match config")
        if summary.get("run_timestamp") != self.run_timestamp:
            raise ValueError("Resume summary run_timestamp does not match CLI timestamp")
        expected_models = {task.model_id for task in tasks}
        recorded_models = {
            str(record.get("model_id", "")) for record in summary.get("tasks", [])
        }
        expected_shards = {task.model_id: task.shard_count for task in tasks}
        for record in summary.get("shard_tasks", []):
            model_id = str(record.get("model_id", ""))
            if (
                model_id in expected_shards
                and int(record["instance_shard_count"]) != expected_shards[model_id]
            ):
                raise ValueError(
                    "Resume must preserve the recorded instance-shard count; "
                    "set scheduler.instance_shards independently of the GPU pool"
                )
        summary["panel_expansion"] = {
            "added_models": sorted(expected_models.difference(recorded_models)),
            "retained_models": sorted(expected_models.intersection(recorded_models)),
            "retired_models": sorted(recorded_models.difference(expected_models)),
        }
        return summary

    def _resume_task_is_complete(
        self,
        task: CheckpointTask,
        previous_model_records: dict[str, dict[str, Any]],
        previous_shard_records: dict[str, dict[str, Any]],
    ) -> bool:
        canonical_path = self.result_root / task.model_id / "all_results.csv"
        if canonical_path.is_file():
            try:
                validate_result_file(
                    canonical_path,
                    self.expected_cells_by_model[task.model_id],
                    self.allowed_datasets,
                )
            except IncompleteResultError:
                return False
            return True
        shard_record = previous_shard_records.get(task.task_id)
        shard_path = self._shard_result_path(task)
        return bool(
            shard_record
            and shard_record.get("status") in {"completed", "failed_merge"}
            and shard_path.is_file()
            and shard_path.stat().st_size > 0
        )

    def _initial_record(
        self,
        task: CheckpointTask,
        previous: dict[str, Any] | None,
        completed: bool,
    ) -> dict[str, Any]:
        result_path = str(self.result_root / task.model_id / "all_results.csv")
        shard_result_path = str(self._shard_result_path(task))
        if previous is None:
            return {
                "task_id": task.task_id,
                "model_id": task.model_id,
                "instance_shard_index": task.shard_index,
                "instance_shard_count": task.shard_count,
                "sequence": task.shard_index + 1,
                "gpu": self.gpus[task.shard_index % len(self.gpus)],
                "status": "completed" if completed else "pending",
                "started_at": "",
                "ended_at": "",
                "wall_seconds": "",
                "attempts": 0,
                "oom_retries": 0,
                "last_concurrency_limit": "",
                "next_concurrency_limit": self.maximum,
                "result_path": shard_result_path,
                "canonical_result_path": result_path,
                "error": "",
                "attempt_history": [],
                "resume_skipped": completed,
            }
        record = dict(previous)
        record["task_id"] = task.task_id
        record["model_id"] = task.model_id
        record["instance_shard_index"] = task.shard_index
        record["instance_shard_count"] = task.shard_count
        record["sequence"] = task.shard_index + 1
        record["gpu"] = self.gpus[task.shard_index % len(self.gpus)]
        record["attempt_history"] = list(previous.get("attempt_history", []))
        record["result_path"] = shard_result_path
        record["canonical_result_path"] = result_path
        record["resume_skipped"] = completed
        if not completed:
            record["status"] = "pending"
            record["ended_at"] = ""
            record["error"] = ""
        return record

    def _shard_result_path(self, task: CheckpointTask) -> Path:
        return (
            self.result_root
            / task.model_id
            / "shards"
            / f"part-{task.shard_index:05d}-of-{task.shard_count:05d}"
            / "all_results.csv"
        )

    def _model_config(self, task: CheckpointTask) -> Path:
        if task.task_id in self._model_config_paths:
            return self._model_config_paths[task.task_id]
        payload = {
            "experiment_name": self.experiment_name,
            "run_timestamp": self.run_timestamp,
            "model_id": task.model_id,
            "datasets": self.config["datasets"],
            "execution": self.config.get("execution", {"device": "cuda"}),
            "instance_shard_index": task.shard_index,
            "instance_shard_count": task.shard_count,
            "result_relative_path": str(
                self._shard_result_path(task).relative_to(
                    self.result_root / task.model_id
                )
            ),
            "grid_reuse_keys": getattr(self, "grid_reuse_plan", {}).get("models", {}).get(task.model_id, {}).get("keys", []),
        }
        maximum = self.models[task.model_id]["context"].get("maximum")
        if "cells" in self.config:
            payload["cells"] = [
                cell for cell in self.config["cells"]
                if maximum is None or int(cell["L"]) <= int(maximum)
            ]
        else:
            if "H" in self.config:
                payload["H"] = self.config["H"]
            if "L" in self.config:
                payload["L"] = [
                    value for value in self.config["L"]
                    if value is None or maximum is None or int(value) <= int(maximum)
                ]
        if "origin_horizon" in self.config:
            payload["origin_horizon"] = self.config["origin_horizon"]
        canonical_payload = dict(payload)
        canonical_payload.pop("instance_shard_index")
        canonical_payload.pop("instance_shard_count")
        canonical_payload.pop("result_relative_path")
        if task.model_id not in self._canonical_configs_written:
            _write_json(
                self.config_root / f"{_safe_name(task.model_id)}.json",
                canonical_payload,
            )
            self._canonical_configs_written.add(task.model_id)
        path = self.config_root / (
            f"{_safe_name(task.model_id)}.part-{task.shard_index:05d}"
            f"-of-{task.shard_count:05d}.json"
        )
        _write_json(path, payload)
        self._model_config_paths[task.task_id] = path
        return path

    def _persist(self, status: str, *, force: bool = False) -> None:
        if (
            status == "running"
            and not force
            and self._last_persist is not None
            and time.monotonic() - self._last_persist < STATE_FLUSH_SECONDS
        ):
            return
        model_records = self._model_records()
        payload = {
            "experiment_name": self.experiment_name,
            "run_timestamp": self.run_timestamp,
            "status": status,
            "started_at": self.started_at,
            "updated_at": timestamp(),
            "git_commit": self.git_commit,
            "resume": self.resume,
            "resume_events": self.resume_events,
            "maximum_workers_per_gpu": self.maximum,
            "gpus": self.gpus,
            "fresh_queue": list(self.fresh_queue),
            "recovery_queue": list(self.recovery_queue),
            "task_phases": {
                task_id: task_state.phase
                for task_id, task_state in self.task_states.items()
            },
            "tasks": list(model_records.values()),
            "shard_tasks": list(self.records.values()),
            "grid_priority": {
                model: {key: value for key, value in record.items() if key not in {"keys", "slots"}}
                for model, record in getattr(self, "grid_reuse_plan", {}).get("models", {}).items()
            },
        }
        _write_json(self.summary_path, payload)
        _write_task_table(self.table_path, self.records)
        self._last_persist = time.monotonic()

    def _model_records(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.records.values():
            grouped.setdefault(str(record["model_id"]), []).append(record)
        output: dict[str, dict[str, Any]] = {}
        terminal_failures = {
            "failed",
            "failed_incomplete",
            "failed_merge",
            "failed_storage",
            "skipped_oom_at_one",
        }
        for model_id, shards in grouped.items():
            statuses = [str(shard["status"]) for shard in shards]
            if all(status == "completed" for status in statuses):
                model_status = "completed"
            elif any(status == "running" for status in statuses):
                model_status = "running"
            elif any(status == "retrying_after_oom" for status in statuses):
                model_status = "retrying_after_oom"
            elif any(status in terminal_failures for status in statuses):
                model_status = "failed"
            else:
                model_status = "pending"
            started = sorted(
                str(shard["started_at"])
                for shard in shards
                if shard.get("started_at")
            )
            ended = sorted(
                str(shard["ended_at"])
                for shard in shards
                if shard.get("ended_at")
            )
            errors = [str(shard["error"]) for shard in shards if shard.get("error")]
            output[model_id] = {
                "model_id": model_id,
                "status": model_status,
                "started_at": started[0] if started else "",
                "ended_at": ended[-1] if len(ended) == len(shards) else "",
                "wall_seconds": round(
                    sum(float(shard.get("wall_seconds") or 0.0) for shard in shards),
                    3,
                ),
                "attempts": sum(int(shard.get("attempts", 0)) for shard in shards),
                "oom_retries": sum(
                    int(shard.get("oom_retries", 0)) for shard in shards
                ),
                "result_path": str(self.result_root / model_id / "all_results.csv"),
                "error": " | ".join(errors),
                "resume_skipped": all(
                    bool(shard.get("resume_skipped")) for shard in shards
                ),
                "shard_statuses": {
                    str(shard["instance_shard_index"]): shard["status"]
                    for shard in shards
                },
            }
        return output

    def _launch(self, state: GPUState, task_state: TaskState) -> None:
        task = task_state.task
        limit = task_state.next_concurrency_limit
        if task.task_id in self.active_by_task:
            raise RuntimeError(f"Task already active: {task.task_id}")
        if task_state.phase not in {"fresh", "recovery"}:
            raise RuntimeError(
                f"Cannot launch {task.task_id} from phase {task_state.phase}"
            )
        record = self.records[task.task_id]
        attempt_number = int(record["attempts"]) + 1
        config_path = self._model_config(task)
        log_path = self.log_root / (
            f"{_safe_name(task.model_id)}.part-{task.shard_index:05d}"
            f"-of-{task.shard_count:05d}.attempt-{attempt_number:02d}"
            f".gpu-{state.gpu}.cap-{limit}.log"
        )
        ensure_directory(log_path.parent)
        log_path.unlink(missing_ok=True)
        sink = ResilientLog(log_path)
        environment = worker_environment(self.repository, state.gpu, task.python_executable)
        command = [
            task.python_executable,
            "-m",
            "UniScale.experiments.run_hl",
            "--config",
            str(config_path),
        ]
        started_at = timestamp()
        started_monotonic = time.monotonic()
        sink.write(
            json.dumps(
                {
                    "event": "start",
                    "time": started_at,
                    "gpu": state.gpu,
                    "concurrency_limit": limit,
                },
                sort_keys=True,
            )
            + "\n"
        )
        sink.flush()
        record["attempts"] = attempt_number
        record["last_concurrency_limit"] = limit
        record["next_concurrency_limit"] = limit
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repository,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as error:
            ended_at = timestamp()
            elapsed = round(time.monotonic() - started_monotonic, 3)
            if not record["started_at"]:
                record["started_at"] = started_at
            record["gpu"] = state.gpu
            record["status"] = "failed"
            record["ended_at"] = ended_at
            record["error"] = f"launch failed: {type(error).__name__}: {error}"
            record["attempt_history"].append(
                {
                    "attempt": attempt_number,
                    "gpu": state.gpu,
                    "concurrency_limit": limit,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "wall_seconds": elapsed,
                    "return_code": "",
                    "oom": False,
                    "storage_failure": False,
                    "launch_failure": True,
                    "log_path": str(log_path),
                    "log_write_errors": [],
                }
            )
            record["wall_seconds"] = round(
                sum(item["wall_seconds"] for item in record["attempt_history"]), 3
            )
            task_state.phase = "terminal"
            task_state.last_gpu = state.gpu
            print(
                f"failed to launch {task.task_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return
        log_errors: list[str] = []

        def drain() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    sink.write(line)
            except Exception as error:
                log_errors.append(f"{type(error).__name__}: {error}")
                discarded_lines = 0
                for _ in process.stdout:
                    discarded_lines += 1
                log_errors.append(
                    f"Discarded {discarded_lines} output lines after the log write failed"
                )
            finally:
                try:
                    sink.flush()
                except Exception as error:
                    log_errors.append(f"Final log flush failed: {type(error).__name__}: {error}")

        log_thread = threading.Thread(target=drain, daemon=True)
        log_thread.start()
        attempt = Attempt(
            task=task,
            gpu=state.gpu,
            concurrency_limit=limit,
            number=attempt_number,
            started_at=started_at,
            started_monotonic=started_monotonic,
            log_path=log_path,
            log_thread=log_thread,
            log_errors=log_errors,
            process=process,
        )
        state.active.append(attempt)
        self.active_by_task[task.task_id] = attempt
        task_state.phase = "running"
        task_state.last_gpu = state.gpu
        if not record["started_at"]:
            record["started_at"] = started_at
        record["gpu"] = state.gpu
        record["status"] = "running"
        print(
            f"started {task.task_id} on GPU {state.gpu} "
            f"(attempt {attempt_number}, concurrency cap {limit})",
            flush=True,
        )

    def _finish(self, state: GPUState, attempt: Attempt, return_code: int) -> None:
        attempt.log_thread.join()
        ended_at = timestamp()
        wall_seconds = time.monotonic() - attempt.started_monotonic
        log_text = attempt.log_path.read_text(encoding="utf-8", errors="replace")
        classification_text = "\n".join([log_text, *attempt.log_errors])
        failure_kind = classify_failure(return_code, classification_text)
        storage_failure = failure_kind == "storage"
        oom = failure_kind == "oom"
        record = self.records[attempt.task.task_id]
        task_state = self.task_states[attempt.task.task_id]
        history = {
            "attempt": attempt.number,
            "gpu": attempt.gpu,
            "concurrency_limit": attempt.concurrency_limit,
            "started_at": attempt.started_at,
            "ended_at": ended_at,
            "wall_seconds": round(wall_seconds, 3),
            "return_code": return_code,
            "oom": oom,
            "storage_failure": storage_failure,
            "log_path": str(attempt.log_path),
            "log_write_errors": attempt.log_errors,
        }
        record["attempt_history"].append(history)
        if return_code == 0:
            task_state.phase = "terminal"
            task_state.last_gpu = state.gpu
            record["status"] = "completed"
            record["ended_at"] = ended_at
            record["wall_seconds"] = round(
                sum(item["wall_seconds"] for item in record["attempt_history"]), 3
            )
            record["error"] = ""
            record["next_concurrency_limit"] = attempt.concurrency_limit
            print(f"completed {attempt.task.task_id} on GPU {state.gpu}", flush=True)
            return
        if oom:
            lower_limit = next_oom_limit(attempt.concurrency_limit)
            if lower_limit is not None:
                task_state.phase = "recovery"
                task_state.next_concurrency_limit = lower_limit
                task_state.last_gpu = state.gpu
                task_state.ready_sequence = self._next_ready_sequence()
                record["status"] = "retrying_after_oom"
                record["oom_retries"] = int(record["oom_retries"]) + 1
                record["error"] = f"OOM at concurrency cap {attempt.concurrency_limit}"
                record["next_concurrency_limit"] = lower_limit
                self.recovery_queue.append(attempt.task.task_id)
                print(
                    f"OOM for {attempt.task.task_id} on GPU {state.gpu}; "
                    f"retry cap reduced to {lower_limit}",
                    flush=True,
                )
                return
            task_state.phase = "terminal"
            task_state.last_gpu = state.gpu
            record["status"] = "skipped_oom_at_one"
            record["ended_at"] = ended_at
            record["wall_seconds"] = round(
                sum(item["wall_seconds"] for item in record["attempt_history"]), 3
            )
            record["error"] = f"OOM persisted at concurrency cap 1; see {attempt.log_path}"
            record["next_concurrency_limit"] = 1
            print(
                f"skipped {attempt.task.task_id} after OOM at cap 1 on GPU {state.gpu}",
                flush=True,
            )
            return
        task_state.phase = "terminal"
        task_state.last_gpu = state.gpu
        record["status"] = "failed_storage" if storage_failure else "failed"
        record["ended_at"] = ended_at
        record["wall_seconds"] = round(
            sum(item["wall_seconds"] for item in record["attempt_history"]), 3
        )
        failure_message = (
            "storage retries exhausted"
            if storage_failure
            else f"exit code {return_code}"
        )
        record["error"] = f"{failure_message}; see {attempt.log_path}"
        record["next_concurrency_limit"] = attempt.concurrency_limit
        print(
            f"failed {attempt.task.task_id} on GPU {state.gpu}: {failure_message}",
            flush=True,
        )

    def _poll_all(self) -> bool:
        finished: list[tuple[GPUState, Attempt, int]] = []
        for state in self.states:
            remaining: list[Attempt] = []
            for attempt in state.active:
                return_code = attempt.process.poll()
                if return_code is None:
                    remaining.append(attempt)
                else:
                    finished.append((state, attempt, return_code))
            state.active = remaining
        for state, attempt, return_code in finished:
            self.active_by_task.pop(attempt.task.task_id, None)
            self._finish(state, attempt, return_code)
        return bool(finished)

    def _effective_limit(
        self, state: GPUState, incoming_limit: int | None = None
    ) -> int:
        limits = [state.maximum]
        limits.extend(attempt.concurrency_limit for attempt in state.active)
        if incoming_limit is not None:
            limits.append(incoming_limit)
        return min(limits)

    def _can_launch(self, state: GPUState, limit: int) -> bool:
        return len(state.active) < self._effective_limit(state, limit)

    def _gpu_rank(self, state: GPUState) -> int:
        index = self.states.index(state)
        return (index - self.gpu_cursor) % len(self.states)

    def _select_gpu(self, task_state: TaskState) -> GPUState | None:
        limit = task_state.next_concurrency_limit
        candidates = [state for state in self.states if self._can_launch(state, limit)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda state: (
                len(state.active),
                int(task_state.last_gpu == state.gpu),
                sum(
                    attempt.task.model_id == task_state.task.model_id
                    for attempt in state.active
                ),
                self._gpu_rank(state),
            ),
        )

    def _ordered_recoveries(self) -> list[str]:
        return sorted(
            self.recovery_queue,
            key=lambda task_id: (
                self.task_states[task_id].next_concurrency_limit,
                self.task_states[task_id].ready_sequence,
            ),
        )

    def _schedule_recoveries(self) -> None:
        while self.recovery_queue:
            launched = False
            for task_id in self._ordered_recoveries():
                task_state = self.task_states[task_id]
                state = self._select_gpu(task_state)
                if state is None:
                    continue
                self.recovery_queue.remove(task_id)
                self._launch(state, task_state)
                self.gpu_cursor = (self.states.index(state) + 1) % len(self.states)
                launched = True
                break
            if not launched:
                return

    def _recovery_reservations(self) -> set[int]:
        reserved: set[int] = set()
        for task_id in self._ordered_recoveries():
            task_state = self.task_states[task_id]
            candidates = [state for state in self.states if state.gpu not in reserved]
            if not candidates:
                break
            target = min(
                candidates,
                key=lambda state: (
                    len(state.active),
                    int(task_state.last_gpu == state.gpu),
                    self._gpu_rank(state),
                ),
            )
            reserved.add(target.gpu)
        return reserved

    def _schedule_fresh(self, reserved_gpus: set[int]) -> None:
        while self.fresh_queue:
            candidates = [
                state
                for state in self.states
                if state.gpu not in reserved_gpus
                and self._can_launch(state, self.maximum)
            ]
            if not candidates:
                return
            state = min(
                candidates,
                key=lambda item: (len(item.active), self._gpu_rank(item)),
            )
            task_id = self.fresh_queue.popleft()
            task_state = self.task_states[task_id]
            self._launch(state, task_state)
            self.gpu_cursor = (self.states.index(state) + 1) % len(self.states)

    def _schedule_all(self) -> None:
        self._schedule_recoveries()
        self._schedule_fresh(self._recovery_reservations())
        self._assert_state_invariants()

    def _assert_state_invariants(self) -> None:
        fresh = list(self.fresh_queue)
        recoveries = list(self.recovery_queue)
        active = [
            attempt.task.task_id
            for state in self.states
            for attempt in state.active
        ]
        if len(fresh) != len(set(fresh)) or len(recoveries) != len(set(recoveries)):
            raise RuntimeError("A task appears more than once in a scheduler queue")
        if len(active) != len(set(active)) or set(active) != set(self.active_by_task):
            raise RuntimeError("Active task ownership is inconsistent")
        if (
            set(fresh) & set(recoveries)
            or set(fresh) & set(active)
            or set(recoveries) & set(active)
        ):
            raise RuntimeError("A task belongs to multiple scheduler phases")
        phase_sets = {
            "fresh": set(fresh),
            "recovery": set(recoveries),
            "running": set(active),
        }
        for task_id, task_state in self.task_states.items():
            if task_state.phase in phase_sets and task_id not in phase_sets[task_state.phase]:
                raise RuntimeError(f"Task phase is inconsistent for {task_id}")
        for phase, task_ids in phase_sets.items():
            for task_id in task_ids:
                if self.task_states[task_id].phase != phase:
                    raise RuntimeError(f"Queue ownership is inconsistent for {task_id}")
        for state in self.states:
            if len(state.active) > self._effective_limit(state):
                raise RuntimeError(f"GPU {state.gpu} exceeds its effective concurrency cap")

    def _unfinished(self) -> bool:
        return any(
            task_state.phase in {"fresh", "recovery", "running"}
            for task_state in self.task_states.values()
        )

    def _merge_completed_models(self) -> None:
        for model in self.config["models"]:
            model_id = str(model["id"])
            shard_records = [
                record
                for record in self.records.values()
                if record["model_id"] == model_id
            ]
            if not shard_records or not all(
                record["status"] == "completed" for record in shard_records
            ):
                continue
            skip_merge = (
                (self.result_root / model_id / "all_results.csv").is_file()
                and all(bool(record.get("resume_skipped")) for record in shard_records)
                and not any(
                    Path(str(record["result_path"])).is_file()
                    for record in shard_records
                )
            )
            try:
                if not skip_merge:
                    merge_model_shards(
                        self.result_root, model_id, self.shard_count,
                        exclude_resource_keys=self.grid_reuse_plan["models"].get(model_id, {}).get("keys", []),
                        allow_empty=self.expected_cells_by_model[model_id] == 0,
                    )
                validate_result_file(
                    self.result_root / model_id / "all_results.csv",
                    self.expected_cells_by_model[model_id],
                    self.allowed_datasets,
                )
            except IncompleteResultError as error:
                for record in shard_records:
                    record["status"] = "failed_incomplete"
                    record["error"] = str(error)
                print(f"incomplete results for {model_id}: {error}", flush=True)
            except Exception as error:
                failed_record = shard_records[0]
                failed_record["status"] = "failed_merge"
                failed_record["error"] = f"{type(error).__name__}: {error}"
                print(
                    f"failed to merge instance shards for {model_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    def run(self) -> bool:
        ensure_directory(self.scheduler_root)
        ensure_directory(self.config_root)
        if self.grid_reuse_plan["enabled"]:
            persist_grid_plan(self.scheduler_root / "grid_reuse.json", self.grid_reuse_plan)
            counts = self.grid_reuse_plan["models"].values()
            print(
                f"Grid-first plan: {sum(item['grid_reused_cells'] for item in counts)} canonical cells reused; "
                f"{sum(item['native_cells_to_cover'] for item in self.grid_reuse_plan['models'].values())} native cells to cover",
                flush=True,
            )
        for task_state in self.task_states.values():
            self._model_config(task_state.task)
        self._persist("running", force=True)
        try:
            while self._unfinished():
                finished_attempts = self._poll_all()
                self._schedule_all()
                self._persist("running", force=finished_attempts)
                if self._unfinished():
                    time.sleep(1.0)
        except BaseException:
            for state in self.states:
                for attempt in state.active:
                    attempt.process.terminate()
            for state in self.states:
                for attempt in state.active:
                    try:
                        attempt.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        attempt.process.kill()
                        attempt.process.wait()
                    attempt.log_thread.join(timeout=5)
            self._persist("interrupted")
            raise
        self._merge_completed_models()
        succeeded = all(record["status"] == "completed" for record in self.records.values())
        self._persist("completed" if succeeded else "completed_with_errors")
        return succeeded


def run_checkpoint_batch(
    config_path: Path,
    run_timestamp: str | None = None,
    *,
    resume: bool = False,
) -> bool:
    return CheckpointScheduler(config_path, run_timestamp, resume=resume).run()
