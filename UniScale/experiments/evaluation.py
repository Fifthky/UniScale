"""Evaluation helpers that isolate rolling windows."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality

from .metrics import gift_metrics


def evaluate_with_isolated_windows(predictor: Any, dataset: Any):
    forecast_windows = []
    number_of_windows = dataset.test_data.windows
    for window_index in range(number_of_windows):
        entries = list(
            itertools.islice(dataset.test_data.input, window_index, None, number_of_windows)
        )
        forecast_windows.append(list(predictor.predict(entries)))
    forecasts = [forecast for group in zip(*forecast_windows) for forecast in group]
    return evaluate_forecasts(
        forecasts,
        test_data=dataset.test_data,
        metrics=gift_metrics(),
        batch_size=256,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(dataset.freq),
    )


def evaluation_sufficient_statistics(dataset: Any) -> dict[str, float | int]:
    """Return additive label statistics needed for exact shard aggregation."""
    valid_target_count = 0
    sum_absolute_label = 0.0
    for label in dataset.test_data.label:
        target = np.asarray(label["target"], dtype=np.float64)
        valid = np.isfinite(target)
        valid_target_count += int(valid.sum())
        sum_absolute_label += float(np.abs(target[valid]).sum())
    if valid_target_count <= 0:
        raise ValueError("Evaluation shard has no finite target values")
    return {
        "valid_target_count": valid_target_count,
        "sum_absolute_label": sum_absolute_label,
    }
