"""Durable project-owned writes with bounded storage-pressure retries."""

from __future__ import annotations

import csv
import errno
import io
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")
STORAGE_PRESSURE_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", 122)}
STORAGE_RETRIES = 300
STORAGE_RETRY_SECONDS = 120
LOG_FLUSH_SECONDS = 300.0
LOG_BUFFER_BYTES = 1024 * 1024
FORECAST_MEAN_WARNING = (
    "The mean prediction is not stored in the forecast data; the median is "
    "being returned instead. This behaviour may change in the future."
)
PROGRESS_LINE = re.compile(r"^\s*(?:\d+(?:\.\d+)?[kMGT]?it\s*\[|\d{1,3}%\|)")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CRITICAL_LOG_LINE = re.compile(
    r"traceback|error|exception|\boom\b|out of memory|failed|failure|"
    r"no space left|disk quota|recover|retry|interrupt|terminat",
    re.IGNORECASE,
)


def _release_cached_accelerator_memory() -> None:
    """Release cached CUDA blocks without importing a new runtime dependency."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def retry_storage_pressure(
    operation: Callable[[], T],
    *,
    retries: int = STORAGE_RETRIES,
    delay_seconds: float = STORAGE_RETRY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = lambda message: print(message, flush=True),
) -> T:
    """Retry transient storage-pressure writes with one final status message."""
    failures = 0
    last_error: OSError | None = None
    while True:
        try:
            result = operation()
            if failures:
                report(
                    f"storage write recovered after {failures} retries "
                    f"({failures * delay_seconds:g} seconds waited)"
                )
            return result
        except OSError as error:
            if error.errno not in STORAGE_PRESSURE_ERRNOS:
                raise
            failures += 1
            last_error = error
            if failures > retries:
                report(
                    f"storage write failed after {retries} retries "
                    f"({retries * delay_seconds:g} seconds waited): {last_error}"
                )
                raise
            if failures == 1:
                _release_cached_accelerator_memory()
            sleep(delay_seconds)


def ensure_directory(path: Path) -> None:
    retry_storage_pressure(lambda: path.mkdir(parents=True, exist_ok=True))


def atomic_write_text(path: Path, text: str) -> None:
    def operation() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    retry_storage_pressure(operation)


def atomic_write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def atomic_append_csv_row(path: Path, row: dict[str, object]) -> None:
    def operation() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        fields = list(row)
        existing_rows: list[dict[str, str]] = []
        if existing:
            reader = csv.DictReader(io.StringIO(existing))
            existing_fields = reader.fieldnames or []
            if not existing_fields:
                raise ValueError(f"CSV has no header: {path}")
            if not set(existing_fields).issubset(fields):
                removed = sorted(set(existing_fields).difference(fields))
                raise ValueError(
                    f"CSV schema would remove fields in {path}: {removed}"
                )
            existing_rows = list(reader)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    retry_storage_pressure(operation)


class ResilientLog:
    """Compact non-result output while preserving failure and recovery details."""

    def __init__(
        self,
        path: Path,
        *,
        flush_seconds: float = LOG_FLUSH_SECONDS,
        buffer_bytes: int = LOG_BUFFER_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ):
        if flush_seconds <= 0 or buffer_bytes <= 0:
            raise ValueError("Log flush interval and buffer size must be positive")
        self.path = path
        self.flush_seconds = flush_seconds
        self.buffer_bytes = buffer_bytes
        self.clock = clock
        self._last_flush = clock()
        self._chunks: list[str] = []
        self._pending_bytes = 0
        self._mean_warning_seen = False
        self._repeated_mean_warnings = 0
        self._seen_warnings: set[str] = set()
        self._compacted: Counter[str] = Counter()
        self._traceback_started = False

    def write(self, text: str) -> None:
        for line in text.splitlines(keepends=True):
            clean = ANSI_ESCAPE.sub("", line).strip()
            if clean.startswith("Traceback (most recent call last):"):
                self._traceback_started = True
            protected = self._traceback_started or bool(CRITICAL_LOG_LINE.search(clean))
            if not protected:
                if not clean:
                    self._compacted["blank_lines"] += 1
                    continue
                if PROGRESS_LINE.match(clean):
                    self._compacted["progress_frames"] += 1
                    continue
                if clean.startswith(("INFO:", "DEBUG:")):
                    self._compacted["info_debug_lines"] += 1
                    continue
                is_warning = clean.startswith("WARNING:") or any(
                    marker in clean
                    for marker in ("UserWarning:", "FutureWarning:", "DeprecationWarning:")
                )
                if is_warning and FORECAST_MEAN_WARNING not in clean:
                    if clean in self._seen_warnings:
                        self._compacted["repeated_warning_lines"] += 1
                        continue
                    if len(self._seen_warnings) < 128:
                        self._seen_warnings.add(clean)
            message = clean.removeprefix("WARNING:gluonts.model.forecast:")
            if not protected and message == FORECAST_MEAN_WARNING:
                if self._mean_warning_seen:
                    self._repeated_mean_warnings += 1
                    continue
                self._mean_warning_seen = True
            self._chunks.append(line)
            self._pending_bytes += len(line.encode("utf-8"))
        if (
            self._pending_bytes >= self.buffer_bytes
            or self.clock() - self._last_flush >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        if not self._chunks and not self._repeated_mean_warnings and not self._compacted:
            self._last_flush = self.clock()
            return
        text = "".join(self._chunks)
        if self._repeated_mean_warnings:
            text += (
                f"Coalesced {self._repeated_mean_warnings} repeated forecast-mean "
                "warnings; the first occurrence is retained in this log.\n"
            )
        if self._compacted:
            text += "Log compaction counts: " + ", ".join(
                f"{name}={count}" for name, count in sorted(self._compacted.items())
            ) + "\n"

        def operation() -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()

        retry_storage_pressure(operation)
        self._chunks.clear()
        self._pending_bytes = 0
        self._repeated_mean_warnings = 0
        self._compacted.clear()
        self._last_flush = self.clock()
