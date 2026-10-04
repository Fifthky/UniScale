"""Controlled horizons with optional forecast origins shared across H."""

from __future__ import annotations

import math
from copy import copy
from functools import cached_property

from UniScale.vendor.gift_eval.data import (
    M4_PRED_LENGTH_MAP,
    PRED_LENGTH_MAP,
    Dataset,
    MultivariateToUnivariate,
    Term,
    MAX_WINDOW,
    TEST_SPLIT,
    maybe_reconvert_freq,
)
from gluonts.dataset.split import TestData, split
from gluonts.time_feature import norm_freq_str
from pandas.tseries.frequencies import to_offset


def native_horizon(name: str, frequency: str, term: str) -> int:
    """Resolve the official horizon without loading the Arrow dataset."""
    frequency = maybe_reconvert_freq(norm_freq_str(to_offset(frequency).name))
    mapping = M4_PRED_LENGTH_MAP if "m4" in name else PRED_LENGTH_MAP
    return Term(term).multiplier * mapping[frequency]


class DatasetSourceCache:
    """Keep one immutable Arrow source and its lazy univariate view per worker."""

    def __init__(self):
        self.name: str | None = None
        self.views: dict[bool, Dataset] = {}

    def get(self, name: str, to_univariate: bool = False) -> Dataset:
        if name != self.name:
            self.views.clear()
            self.name = name
        if False not in self.views:
            source = Dataset(name, to_univariate=False)
            # These properties depend only on the raw source, never on H or origins.
            for field in ("freq", "target_dim", "past_feat_dynamic_real_dim", "_min_series_length"):
                getattr(source, field)
            self.views[False] = source
        if to_univariate not in self.views:
            view = copy(self.views[False])
            view.gluonts_dataset = MultivariateToUnivariate("target").apply(
                view.gluonts_dataset
            )
            self.views[True] = view
        return self.views[to_univariate]


def gift_window_count(
    minimum_series_length: int,
    horizon: int,
    is_m4: bool = False,
) -> int:
    """Return the GIFT window count evaluated at a declared horizon."""
    if minimum_series_length <= 0 or horizon <= 0:
        raise ValueError("minimum_series_length and horizon must be positive")
    if is_m4:
        return 1
    windows = math.ceil(TEST_SPLIT * minimum_series_length / horizon)
    return min(max(1, windows), MAX_WINDOW)


class ControlledDataset(Dataset):
    """GIFT dataset with controlled H and an optional shared-origin anchor.

    With no ``origin_horizon``, this reproduces GIFT's horizon-specific split.
    With ``origin_horizon=H_max``, every scored horizon uses the cutoffs and
    distance generated at H_max, so forecast origins remain identical across H.
    """

    def __init__(
        self,
        name: str,
        term: str,
        to_univariate: bool,
        prediction_length: int | None = None,
        origin_horizon: int | None = None,
        storage_env_var: str = "GIFT_EVAL",
        source: Dataset | None = None,
    ):
        if prediction_length is not None and prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if origin_horizon is not None and origin_horizon <= 0:
            raise ValueError("origin_horizon must be positive")
        if (
            prediction_length is not None
            and origin_horizon is not None
            and prediction_length > origin_horizon
        ):
            raise ValueError("prediction_length cannot exceed origin_horizon")
        self.controlled_prediction_length = prediction_length
        self.origin_horizon = origin_horizon
        if source is None:
            super().__init__(
                name=name, term=term, to_univariate=to_univariate,
                storage_env_var=storage_env_var,
            )
        else:
            if source.name != name:
                raise ValueError("Cached source does not match the requested dataset")
            self.name, self.term = name, Term(term)
            for field in (
                "hf_dataset", "gluonts_dataset", "freq", "target_dim",
                "past_feat_dynamic_real_dim", "_min_series_length",
            ):
                self.__dict__[field] = getattr(source, field)

    @cached_property
    def prediction_length(self) -> int:
        if self.controlled_prediction_length is not None:
            return self.controlled_prediction_length
        return native_horizon(self.name, self.freq, self.term.value)

    @property
    def origin_distance(self) -> int:
        return self.origin_horizon or self.prediction_length

    @cached_property
    def windows(self) -> int:
        return gift_window_count(
            minimum_series_length=self._min_series_length,
            horizon=self.origin_distance,
            is_m4="m4" in self.name,
        )

    @property
    def origin_policy(self) -> str:
        if self.origin_horizon is None:
            return "gift_horizon_specific_v1"
        return f"shared_origin_v1:Hmax={self.origin_horizon}"

    @property
    def available_context_min(self) -> int:
        """Minimum history visible at the earliest evaluation origin."""
        return self._min_series_length - self.origin_distance * self.windows

    @property
    def test_data(self):
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.origin_distance * self.windows,
        )
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.origin_distance,
        )


class InstanceShardedDataset:
    """Expose one deterministic base-series shard of an evaluation dataset."""

    def __init__(self, source: ControlledDataset, shard_index: int, shard_count: int):
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("shard_index must be between zero and shard_count - 1")
        source_test_data = source.test_data
        entries = [
            entry
            for index, entry in enumerate(source_test_data.dataset)
            if index % shard_count == shard_index
        ]
        self.source = source
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.instance_count = len(entries)
        self._test_data = TestData(
            dataset=entries,
            splitter=source_test_data.splitter,
            prediction_length=source_test_data.prediction_length,
            windows=source_test_data.windows,
            distance=source_test_data.distance,
            max_history=source_test_data.max_history,
        )

    @property
    def test_data(self) -> TestData:
        return self._test_data

    def __getattr__(self, name: str):
        return getattr(self.source, name)
