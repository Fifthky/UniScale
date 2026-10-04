# Migrated from google-research/timesfm, v1/src/timesfm, revision
# 3dae50b20d7a724981e8ea36cda75578f80dd2dc. Integration uses local imports,
# array-based inference only, and explicit abstract-method errors.
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class for TimesFM inference. This will be common to PAX and Pytorch."""

import dataclasses
from typing import Any, Literal, Sequence

import numpy as np


_TOL = 1e-6
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def moving_average(arr, window_size):
  """Calculates the moving average using NumPy's convolution function."""
  # Pad with zeros to handle initial window positions
  arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
  smoothed_arr = (np.convolve(arr_padded, np.ones(window_size), "valid") /
                  window_size)
  return [smoothed_arr, arr - smoothed_arr]


def freq_map(freq: str):
  """Returns the frequency map for the given frequency string."""
  freq = str.upper(freq)
  if freq.endswith("MS"):
    return 1
  elif freq.endswith(("H", "T", "MIN", "D", "B", "U", "S")):
    return 0
  elif (
    freq.endswith(("W", "M"))
    or freq.startswith("W-")
    or (freq.startswith("M") and len(freq) == 2)
  ):
    return 1
  elif (
    freq.endswith(("Y", "Q", "A"))
    or freq.startswith("Y-")
    or freq.startswith("Q-")
    or freq.startswith("A-")
  ):
    return 2
  else:
    raise ValueError(f"Invalid frequency: {freq}")


def strip_leading_nans(arr):
  """
  Removes contiguous NaN values from the beginning of a NumPy array.

  Args:
    arr: The input NumPy array.

  Returns:
    A new NumPy array with leading NaN values removed.
    If the array is all NaNs or empty, returns an empty array.
  """

  isnan = np.isnan(arr)
  first_valid_index = np.argmax(~isnan)
  return arr[first_valid_index:]


def linear_interpolation(arr):
  """
    Performs linear interpolation to fill NaN values in a 1D numpy array.

    Args:
        arr: The 1D numpy array containing NaN values.

    Returns:
        A new numpy array with NaN values filled using linear interpolation,
        or the original array if no NaNs are present.
        Returns None if the input is not a 1D array.
        Returns the original array if there are no NaN values.
    """

  nans = np.isnan(arr)
  if not np.any(nans):  # Check if there are any NaNs
    return arr

  def x(z):
    return z.nonzero()[0]

  nans_indices = x(nans)
  non_nans_indices = x(~nans)
  non_nans_values = arr[~nans]

  try:
    arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
  except ValueError:
    if len(non_nans_values) > 0:
      mu = np.nanmean(arr)
    else:
      mu = 0.0
    arr = np.where(np.isfinite(arr), arr, mu)
  return arr


# Per time series normalization: forward.
def _normalize(batch):
  stats = [
      (np.mean(x), np.where((w := np.std(x)) > _TOL, w, 1.0)) for x in batch
  ]
  new_batch = [(x - stat[0]) / stat[1] for x, stat in zip(batch, stats)]
  return new_batch, stats


@dataclasses.dataclass(kw_only=True)
class TimesFmHparams:
  """Hparams used to initialize a TimesFM model for inference.

  These are the sufficient subset of hparams to configure TimesFM inference
  agnostic to the checkpoint version, and are not necessarily the same as the
  hparams used to train the checkpoint.

  Attributes:
    context_len: Largest context length the model allows for each decode call.
      This technically can be any large, but practically should set to the
      context length the checkpoint was trained with.
    horizon_len: Forecast horizon.
    input_patch_len: Input patch len.
    output_patch_len: Output patch len. How many timepoints is taken from a
      single step of autoregressive decoding. Can be set as the training horizon
      of the checkpoint.
    num_layers: Number of transformer layers in the model.
    model_dims: Model dimension.
    per_core_batch_size: Batch size on each core for data parallelism.
    backend: One of "cpu", "gpu" or "tpu".
    quantiles: Which quantiles are output by the model.
  """

  context_len: int = 512
  horizon_len: int = 128
  input_patch_len: int = 32
  output_patch_len: int = 128
  num_layers: int = 20
  num_heads: int = 16
  model_dims: int = 1280
  per_core_batch_size: int = 32
  backend: Literal["cpu", "gpu", "tpu"] = "cpu"
  quantiles: Sequence[float] | None = DEFAULT_QUANTILES
  use_positional_embedding: bool = True
  # Hparams beyond the model.
  point_forecast_mode: Literal["mean", "median"] = "median"


@dataclasses.dataclass(kw_only=True)
class TimesFmCheckpoint:
  """Checkpoint used to initialize a TimesFM model for inference.

  Attributes:
    version: Version of the checkpoint, e.g. "jax", "torch", "tensorflow", etc.
      The factory will create the corresponding TimesFm inference class based on
      this version.
    path: Path to the checkpoint.
    type: If provided, type of the checkpoint used by the specific checkpoint
      loader per version.
    step: If provided, step of the checkpoint.
  """

  version: str = "jax"
  path: str | None = None
  huggingface_repo_id: str | None = None
  type: Any = None
  step: int | None = None
  local_dir: str | None = None


class TimesFmBase:
  """Base TimesFM forecast API for inference.

  This class is the scaffolding for calling TimesFM forecast. To properly use:
    1. Create an instance with the correct hyperparameters of a TimesFM model.
    2. Call `load_from_checkpoint` to load a compatible checkpoint.
    3. Call `forecast` for inference.
  """

  def _logging(self, s):
    print(s)

  def __post_init__(self) -> None:
    """Additional initialization for subclasses before checkpoint loading."""
    raise NotImplementedError("`__post_init__` is not implemented.")

  def __init__(self, hparams: TimesFmHparams,
               checkpoint: TimesFmCheckpoint) -> None:
    """Initializes the TimesFM forecast API.

    Args:
      hparams: Hyperparameters of the model.
      checkpoint: Checkpoint to load. Notice `checkpoint.version` will decide
        which TimesFM version to use.
    """
    self.hparams = hparams

    # Expand hparams for conciseness within the model code.
    self.context_len = hparams.context_len
    self.horizon_len = hparams.horizon_len
    self.input_patch_len = hparams.input_patch_len
    self.output_patch_len = hparams.output_patch_len
    self.num_layers = hparams.num_layers
    self.model_dims = hparams.model_dims
    self.backend = hparams.backend
    self.quantiles = hparams.quantiles
    self.num_heads = hparams.num_heads
    self.use_pos_emb = hparams.use_positional_embedding

    # Rewrite these values in __post_init__ for SPMD.
    self.num_cores = 1
    self.per_core_batch_size = hparams.per_core_batch_size
    self.global_batch_size = hparams.per_core_batch_size

    self._horizon_start = self.context_len - self.input_patch_len
    self.__post_init__()
    self.load_from_checkpoint(checkpoint)

  def load_from_checkpoint(self, checkpoint: TimesFmCheckpoint) -> None:
    """Loads a checkpoint and compiles the decoder."""
    raise NotImplementedError("`load_from_checkpoint` is not implemented.")

  def _preprocess(
      self, inputs: Sequence[np.ndarray],
      freq: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Formats and pads raw inputs to feed into the model.

    This function both pads each time series to match the context length, and
    pads the inputs to meet the SPMD shape requirement.

    Args:
      inputs: A list of 1d JTensors. Each JTensor is the context time series of
        a single forecast task.
      freq: list of frequencies

    Returns:
    A tuple of:
    - the padded input time series to meet the model required context.
    - the padding indicator.
    - the frequency of each input time series.
    - the number of padded examples for SPMD so that each core has the same
        number (a multiple of `batch_size`) of examples.
    """

    input_ts, input_padding, inp_freq = [], [], []

    pmap_pad = ((len(inputs) - 1) // self.global_batch_size +
                1) * self.global_batch_size - len(inputs)

    for i, ts in enumerate(inputs):
      input_len = ts.shape[0]
      padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
      if input_len < self.context_len:
        num_front_pad = self.context_len - input_len
        ts = np.concatenate([np.zeros(shape=(num_front_pad,), dtype=float), ts],
                            axis=0)
        padding = np.concatenate(
            [np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0)
      elif input_len > self.context_len:
        ts = ts[-self.context_len:]
        padding = padding[-(self.context_len + self.horizon_len):]

      input_ts.append(ts)
      input_padding.append(padding)
      inp_freq.append(freq[i])

    # Padding the remainder batch.
    for _ in range(pmap_pad):
      input_ts.append(input_ts[-1])
      input_padding.append(input_padding[-1])
      inp_freq.append(inp_freq[-1])

    return (
        np.stack(input_ts, axis=0),
        np.stack(input_padding, axis=0),
        np.array(inp_freq).astype(np.int32).reshape(-1, 1),
        pmap_pad,
    )

  def _forecast(
      self,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts on a list of time series.

    Args:
      inputs: list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      return_forecast_on_context: True to return the forecast on the context
        when available, i.e. after the first input patch.

    Returns:
    A tuple for np.array:
    - the mean forecast of size (# inputs, # forecast horizon),
    - the full forecast (mean + quantiles) of size
        (# inputs,  # forecast horizon, 1 + # quantiles).

    Raises:
    ValueError: If the checkpoint is not properly loaded.
    """
    raise NotImplementedError("`_forecast` is not implemented.")

  def forecast(
      self,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
      normalize: bool = False,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts on a list of time series.

    Args:
      inputs: list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      return_forecast_on_context: True to return the forecast on the context
        when available, i.e. after the first input patch.
      normalize: If True, then we normalize the inputs before forecasting and
        the outputs are then renormalized to the original scale.

    Returns:
    A tuple for np.array:
    - the mean forecast of size (# inputs, # forecast horizon),
    - the full forecast (mean + quantiles) of size
        (# inputs,  # forecast horizon, 1 + # quantiles).

    Raises:
    ValueError: If the checkpoint is not properly loaded.
    """
    stats = None

    tmp_inputs = []
    for each_input in inputs:
      arr = np.array(each_input)
      if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = strip_leading_nans(arr)
        arr = linear_interpolation(arr)
      tmp_inputs.append(arr)

    inputs = tmp_inputs
    if normalize:
      inputs, stats = _normalize(inputs)
    mean_forecast, quantile_forecast = self._forecast(
        inputs,
        freq,
        window_size,
        forecast_context_len,
        return_forecast_on_context,
    )
    if stats is not None:
      stats = np.array(stats)
      mu = stats[:, 0]
      sigma = stats[:, 1]
      mean_forecast = mean_forecast * sigma[:, None] + mu[:, None]
      quantile_forecast = (quantile_forecast * sigma[:, None, None] +
                           mu[:, None, None])
    if self.hparams.point_forecast_mode == "mean":
      return mean_forecast, quantile_forecast
    elif self.hparams.point_forecast_mode == "median":
      if self._median_index == -1:
        for i, quantile in enumerate(self.quantiles):
          if quantile == 0.5:
            self._median_index = i
            break
        if self._median_index == -1:
          raise ValueError("Median (0.5) is not found in the model quantiles:"
                           f" {self.quantiles}. Please check the hparams.")
      return (
          quantile_forecast[:, :, 1 + self._median_index],
          quantile_forecast,
      )
    else:
      raise ValueError(
          "Unsupported point forecast mode:"
          f" {self.hparams.point_forecast_mode}. Use 'mean' or 'median'.")
