"""Model-independent context controls applied around official model calls."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def crop_entry(entry: dict[str, Any], context_length: int) -> dict[str, Any]:
    """Keep the most recent context and align target-indexed past covariates."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    cropped = dict(entry)
    target = np.asarray(entry["target"])
    trim = max(0, target.shape[-1] - context_length)
    if trim:
        cropped["target"] = target[..., trim:]
        cropped["start"] = entry["start"] + trim
    else:
        cropped["target"] = target.copy()
    if "past_feat_dynamic_real" in entry:
        past_features = np.asarray(entry["past_feat_dynamic_real"])
        if past_features.ndim == 0:
            raise ValueError("past_feat_dynamic_real must include a time dimension")
        if past_features.shape[-1] != target.shape[-1]:
            raise ValueError(
                "past_feat_dynamic_real and target must have the same time length "
                f"before context cropping, received {past_features.shape[-1]} and "
                f"{target.shape[-1]}"
            )
        cropped["past_feat_dynamic_real"] = past_features[..., trim:].copy()
    return cropped


class ContextLimitedPredictor:
    """Apply the declared history limit immediately before a model call."""

    def __init__(self, predictor: Any, context_length: int):
        self.predictor = predictor
        self.context_length = context_length

    def predict(self, test_data_input: Iterable[dict[str, Any]]):
        entries = [crop_entry(entry, self.context_length) for entry in test_data_input]
        return self.predictor.predict(entries)
