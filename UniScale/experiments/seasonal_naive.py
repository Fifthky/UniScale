"""GIFT-compatible seasonal-naive baseline for arbitrary horizons."""

from __future__ import annotations

import csv
import fcntl
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from gluonts.transform import LastValueImputation
from scipy.special import ndtri

from UniScale.io_utils import atomic_append_csv_row, ensure_directory, retry_storage_pressure

from .evaluation import evaluate_with_isolated_windows
from .metrics import QUANTILE_LEVELS, metric_record

BASELINE_VERSION = "statsforecast-native-v1"


class BaselineCache:
    """Bounded positive-result cache; misses still use the shared file lock."""

    def __init__(self, path: Path, maximum: int = 2048):
        if maximum <= 0:
            raise ValueError("Baseline cache capacity must be positive")
        self.path = path
        self.maximum = maximum
        self.rows: OrderedDict[tuple[str, int, str], dict[str, object]] = OrderedDict()

    def get(self, key: tuple[str, int, str]) -> dict[str, object] | None:
        if key not in self.rows:
            return None
        self.rows.move_to_end(key)
        return dict(self.rows[key])

    def remember(self, key: tuple[str, int, str], row: dict[str, object]) -> None:
        self.rows[key] = dict(row)
        self.rows.move_to_end(key)
        while len(self.rows) > self.maximum:
            self.rows.popitem(last=False)


class SeasonalNaivePredictor:
    def __init__(self, prediction_length: int, season_length: int):
        if prediction_length <= 0 or season_length <= 0:
            raise ValueError("prediction_length and season_length must be positive")
        self.prediction_length = prediction_length
        self.season_length = season_length

    def predict(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim != 1:
                raise ValueError("SeasonalNaivePredictor requires univariate entries")
            if np.isnan(target).any():
                target = LastValueImputation()(target.copy())
            if len(target) < self.season_length:
                raise ValueError(
                    f"History length {len(target)} is shorter than season length {self.season_length}"
                )
            last_season = target[-self.season_length :]
            repetitions = (self.prediction_length + self.season_length - 1) // self.season_length
            point_forecast = np.tile(last_season, repetitions)[: self.prediction_length]

            # This mirrors StatsForecast's native SeasonalNaive intervals, which
            # are used by GIFT's official baseline. The innovation variance is
            # estimated from one-season residuals. Forecast uncertainty grows
            # once per repeated seasonal cycle: sqrt(floor(step / season) + 1).
            residuals = target[self.season_length :] - target[: -self.season_length]
            sigma = np.sqrt(np.nansum(residuals**2) / residuals.size)
            cycle = np.floor(np.arange(self.prediction_length) / self.season_length)
            sigma_h = sigma * np.sqrt(cycle + 1.0)
            quantiles = np.stack(
                [point_forecast + ndtri(level) * sigma_h for level in QUANTILE_LEVELS],
                axis=0,
            )
            arrays = np.concatenate([point_forecast[None, :], quantiles], axis=0)
            yield QuantileForecast(
                forecast_arrays=arrays,
                forecast_keys=["mean", *[str(level) for level in QUANTILE_LEVELS]],
                start_date=entry["start"] + len(target),
                item_id=entry.get("item_id"),
            )


def _baseline_index(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["dataset"],
            int(row["scored_horizon"]),
            row.get("origin_policy", "gift_horizon_specific_v1"),
        ): row
        for row in rows
    }


def _append(path: Path, row: dict[str, object]) -> None:
    atomic_append_csv_row(path, row)


def ensure_seasonal_naive(
    dataset: Any,
    dataset_configuration: str,
    output_path: Path,
    domain: str,
    num_variates: int,
    origin_policy: str = "gift_horizon_specific_v1",
    cache: BaselineCache | None = None,
) -> dict[str, object]:
    key = (dataset_configuration, int(dataset.prediction_length), origin_policy)
    if cache is not None:
        if cache.path != output_path:
            raise ValueError("Baseline cache belongs to a different result file")
        cached = cache.get(key)
        if cached is not None:
            return cached
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    ensure_directory(lock_path.parent)
    lock_handle = retry_storage_pressure(
        lambda: lock_path.open("a", encoding="utf-8")
    )
    with lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        index = _baseline_index(output_path)
        if cache is not None:
            for indexed_key, row in index.items():
                cache.remember(indexed_key, row)
        existing = index.get(key)
        if existing is not None:
            if cache is not None:
                cache.remember(key, existing)
            return existing

        predictor = SeasonalNaivePredictor(
            prediction_length=dataset.prediction_length,
            season_length=get_seasonality(dataset.freq),
        )
        metrics = metric_record(evaluate_with_isolated_windows(predictor, dataset))
        row: dict[str, object] = {
            "dataset": dataset_configuration,
            "scored_horizon": dataset.prediction_length,
            "origin_policy": origin_policy,
            "model": "UniScale_Seasonal_Naive",
            "baseline_version": BASELINE_VERSION,
            **metrics,
            "domain": domain,
            "num_variates": num_variates,
            "rolling_windows": dataset.windows,
            "scored_values_per_series": dataset.windows * dataset.prediction_length,
            "origin_span_per_series": dataset.windows * getattr(
                dataset, "origin_distance", dataset.prediction_length
            ),
        }
        _append(output_path, row)
        if cache is not None:
            cache.remember(key, row)
        return row
