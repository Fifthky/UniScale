"""Shared GIFT point scoring and lossless per-variable sufficient statistics."""

from __future__ import annotations

import numpy as np

SCHEMA = "gift-point-statistics-v1"
COLUMNS = ["origin_record", "variable", "observed_targets", "absolute_error_sum",
           "squared_error_sum", "absolute_target_sum", "scaled_absolute_error_sum",
           "scaled_error_count", "baseline_absolute_error_sum", "baseline_squared_error_sum",
           "baseline_scaled_absolute_error_sum", "baseline_scaled_error_count"]
METRICS = {"MAE[0.5]": "mae", "MSE[mean]": "mse", "MASE[0.5]": "mase",
           "RMSE[mean]": "rmse", "NRMSE[mean]": "nrmse", "ND[0.5]": "nd"}


def scoring_scale(histories, seasonality):
    """Match GIFT's masked raw-history seasonal error, before imputation."""
    from gluonts.ev.ts_stats import seasonal_error

    return np.ma.stack([seasonal_error(np.ma.masked_invalid(history), seasonality,
                                      time_axis=-1) for history in histories])


def naive_point(histories, horizon, seasonality):
    """GIFT SeasonalNaive point forecast with causal last-value imputation."""
    output = []
    for history in histories:
        values = np.asarray(history, dtype=np.float32).copy()
        if values.ndim != 1 or values.size == 0 or np.isinf(values).any():
            raise ValueError("SeasonalNaive requires a nonempty scalar history without infinities")
        missing = np.isnan(values)
        if missing.all():
            # GluonTS LastValueImputation delegates all-NaN histories to
            # DummyValueImputation(0). Their raw-history MASE scale stays masked.
            values.fill(0.0)
        elif missing.any():
            # GluonTS LastValueImputation: forward fill and backfill initial NaNs.
            positions = np.where(~missing, np.arange(len(values)), 0)
            np.maximum.accumulate(positions, out=positions)
            values = values[positions]
            values[np.isnan(values)] = values[np.flatnonzero(~np.isnan(values))[0]]
        output.append(np.resize(values[-seasonality:], horizon) if len(values) >= seasonality
                      else np.full(horizon, np.nanmean(values)))
    return np.asarray(output)


def sufficient_rows(record, labels, predictions, scales, baseline):
    labels = np.ma.masked_invalid(np.asarray(labels, dtype=np.float64))
    predictions = np.asarray(predictions, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if predictions.shape != labels.shape or baseline.shape != labels.shape:
        raise ValueError("Point forecast/target shape mismatch")
    if not np.isfinite(predictions).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite point prediction")
    ae, be = np.abs(labels - predictions), np.abs(labels - baseline)
    scaled, bscaled = ae / scales, be / scales
    sums = lambda values: np.ma.sum(values, axis=-1).filled(0)
    return np.column_stack([np.full(len(labels), record), np.arange(len(labels)),
        np.ma.count(labels, axis=-1), sums(ae), sums((labels - predictions) ** 2), sums(np.abs(labels)),
        sums(scaled), np.ma.count(scaled, axis=-1), sums(be), sums((labels - baseline) ** 2),
        sums(bscaled), np.ma.count(bscaled, axis=-1)]).astype(np.float64)


def reduce_statistics(rows):
    """Exact GIFT sum/mean reduction of stored per-variable loss statistics."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(COLUMNS) or not np.isfinite(rows).all():
        raise ValueError("Invalid GIFT sufficient statistics")
    if np.any(rows[:, 2:] < 0):
        raise ValueError("Negative loss sum/count")
    n, ae, se, ay, ase, sn, bae, bse, base, bsn = rows[:, 2:].sum(axis=0)
    if min(n, ay, sn, bsn) <= 0:
        raise ValueError("Undefined GIFT point metric")
    def scores(abs_sum, sq_sum, scaled_sum, scaled_count):
        rmse = float(np.sqrt(sq_sum / n))
        return dict(mae=float(abs_sum / n), mse=float(sq_sum / n), rmse=rmse,
                    nd=float(abs_sum / ay), nrmse=float(rmse / (ay / n)),
                    mase=float(scaled_sum / scaled_count))
    model, baseline = scores(ae, se, ase, sn), scores(bae, bse, base, bsn)
    if any(value <= 0 for value in baseline.values()):
        raise ValueError("Undefined relative metric: zero SeasonalNaive error")
    return {"model": model, "seasonal_naive": baseline,
            "relative": {name: model[name] / baseline[name] for name in model},
            "observed_targets": int(n), "mase_targets": int(sn)}


class GiftPointAccumulator:
    """Call the same GluonTS metric classes as GIFT, retaining reducible inputs."""

    def __init__(self):
        from UniScale.experiments.metrics import gift_point_metrics

        self.model = [metric(axis=None) for metric in gift_point_metrics()]
        self.baseline = [metric(axis=None) for metric in gift_point_metrics()]
        self.rows = []

    def update(self, record, labels, predictions, scales, baseline):
        rows = sufficient_rows(record, labels, predictions, scales, baseline)
        self.rows.append(rows)
        # Entirely masked records carry no observations into the official mean.
        valid = rows[:, 2] > 0
        if not valid.any():
            return
        labels = np.ma.masked_invalid(np.asarray(labels, dtype=np.float64)[valid])
        for evaluators, point in ((self.model, predictions), (self.baseline, baseline)):
            values = np.asarray(point, dtype=np.float64)[valid]
            batch = {"label": labels, "mean": values, "0.5": values,
                     "seasonal_error": np.ma.asarray(scales)[valid]}
            for evaluator in evaluators:
                evaluator.update(batch)

    def finish(self):
        rows = np.concatenate(self.rows)
        result = reduce_statistics(rows)
        for name, evaluators in (("model", self.model), ("seasonal_naive", self.baseline)):
            official = {METRICS[e.name]: float(e.get()) for e in evaluators}
            for metric, value in official.items():
                if not np.isclose(value, result[name][metric], rtol=1e-8, atol=1e-10):
                    raise ValueError(f"GIFT/statistic reduction mismatch: {name}/{metric}")
            result[name] = official
        return rows, result
