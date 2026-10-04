"""Low-volume worker summaries without changing evaluation or metric persistence."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from typing import Callable

from UniScale.io_utils import FORECAST_MEAN_WARNING


WORKER_REPORT_SECONDS = 300.0


class WorkerLogPolicy(logging.Filter):
    """Report completed work periodically and coalesce a known warning at source."""

    def __init__(
        self,
        *,
        report_seconds: float = WORKER_REPORT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        if report_seconds <= 0:
            raise ValueError("Worker report interval must be positive")
        self.report_seconds = report_seconds
        self.clock = clock
        self._last_report = clock()
        self._lock = threading.Lock()
        self._interval: Counter[str] = Counter()
        self._totals: Counter[str] = Counter()
        self._last_completed = ""
        self._mean_warning_seen = False
        self._logger = logging.getLogger("gluonts.model.forecast")

    def __enter__(self) -> WorkerLogPolicy:
        self._logger.addFilter(self)
        print(json.dumps({"event": "compact_worker_logging", "report_seconds": self.report_seconds}), flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._logger.removeFilter(self)
        self.report(force=True, status="failed" if exc_type else "completed")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING or record.getMessage() != FORECAST_MEAN_WARNING:
            return True
        with self._lock:
            if not self._mean_warning_seen:
                self._mean_warning_seen = True
                return True
            self._interval["repeated_mean_warnings"] += 1
            self._totals["repeated_mean_warnings"] += 1
        return False

    def completed(self, run_key: str) -> None:
        with self._lock:
            self._last_completed = run_key
            self._interval["completed_cells"] += 1
            self._totals["completed_cells"] += 1
        self.report()

    def skipped(self, reason: str) -> None:
        with self._lock:
            key = f"skipped_{reason}"
            self._interval[key] += 1
            self._totals[key] += 1
        self.report()

    def report(self, *, force: bool = False, status: str = "running") -> None:
        now = self.clock()
        if not force and now - self._last_report < self.report_seconds:
            return
        with self._lock:
            payload = {
                "event": "worker_summary",
                "status": status,
                "interval_seconds": round(now - self._last_report, 1),
                "counts": dict(self._interval),
                "totals": dict(self._totals),
                "last_completed": self._last_completed,
            }
            self._interval.clear()
            self._last_report = now
        print(json.dumps(payload, sort_keys=True), flush=True)
