"""GIFT metric definitions shared by every UniScale experiment."""

from __future__ import annotations

from typing import Any

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)


QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def gift_point_metrics() -> list[Any]:
    """GIFT point metrics for an explicitly deterministic point forecast.

    The same point is supplied as both the mean and median. No artificial
    quantiles are constructed for models/interventions returning only a point.
    """
    return [MSE(forecast_type="mean"), MAE(), MASE(),
            RMSE(forecast_type="mean"), NRMSE(forecast_type="mean"), ND()]


def gift_metrics() -> list[Any]:
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(forecast_type="mean"),
        NRMSE(forecast_type="mean"),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]


def metric_record(result: Any) -> dict[str, float]:
    if hasattr(result, "reset_index"):
        records = result.reset_index(drop=True).to_dict(orient="records")
        if len(records) != 1:
            raise ValueError(f"Expected one aggregate metric record, received {len(records)}")
        record = records[0]
    elif isinstance(result, dict):
        record = {}
        for key, value in result.items():
            if hasattr(value, "iloc"):
                record[key] = value.iloc[0]
            elif isinstance(value, (list, tuple)):
                record[key] = value[0]
            else:
                record[key] = value
    else:
        raise TypeError(f"Unsupported GluonTS metric result: {type(result)!r}")
    return {f"eval_metrics/{key}": float(value) for key, value in record.items()}
