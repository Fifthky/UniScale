"""Normalize evaluation metrics against the matching Seasonal Naive baseline."""

from __future__ import annotations

import math

from typing import Iterable, Mapping

MASE_COLUMN = "eval_metrics/MASE[0.5]"

CRPS_COLUMN = "eval_metrics/mean_weighted_sum_quantile_loss"

def _positive_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive: {value!r}")
    return number

def normalize_metric_pair(
    model_row: Mapping[str, object],
    baseline_row: Mapping[str, object],
) -> dict[str, float]:
    raw_mase = _positive_float(model_row.get(MASE_COLUMN), MASE_COLUMN)
    raw_crps = _positive_float(model_row.get(CRPS_COLUMN), CRPS_COLUMN)
    baseline_mase = _positive_float(baseline_row.get(MASE_COLUMN), f"seasonal naive {MASE_COLUMN}")
    baseline_crps = _positive_float(baseline_row.get(CRPS_COLUMN), f"seasonal naive {CRPS_COLUMN}")
    rel_mase = raw_mase / baseline_mase
    rel_crps = raw_crps / baseline_crps
    return {
        "raw_mase": raw_mase,
        "seasonal_naive_mase": baseline_mase,
        "rel_mase": rel_mase,
        "log_rel_mase": math.log(rel_mase),
        "raw_crps": raw_crps,
        "seasonal_naive_crps": baseline_crps,
        "rel_crps": rel_crps,
        "log_rel_crps": math.log(rel_crps),
    }
