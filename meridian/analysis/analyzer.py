# Copyright 2025 The Meridian Authors.
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

"""Methods to compute analysis metrics of the model and the data."""

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import numbers
from typing import Any, Optional
import warnings

from meridian import backend
from meridian import constants
from meridian.model import adstock_hill
from meridian.model import model
from meridian.model import transformers
import numpy as np
import pandas as pd
from typing_extensions import Self
import xarray as xr

__all__ = [
    "Analyzer",
    "DataTensors",
    "DistributionTensors",
    "get_central_tendency_and_ci",
]


def _validate_non_media_baseline_values_numbers(
    non_media_baseline_values: Sequence[str | float] | None,
):
  if non_media_baseline_values is None:
    return

  for value in non_media_baseline_values:
    if not isinstance(value, numbers.Number):
      raise ValueError(
          f"Invalid `non_media_baseline_values` value: '{value}'. Only float"
          " numbers are supported."
      )


# TODO: Refactor the related unit tests to be under DataTensors.
@dataclasses.dataclass
class DataTensors(backend.ExtensionType):
  """Container for data variable arguments of Analyzer methods.

  Attributes:
    media: Optional tensor with dimensions `(n_geos, T, n_media_channels)` for
      any time dimension `T`.
    media_spend: Optional tensor with dimensions `(n_media_channels,)` or
      `(n_geos, T, n_media_channels)` for any time dimension `T`. If the object
      includes variables with modified time periods, then this tensor must be
      provided at the geo and time granularity.
    reach: Optional tensor with dimensions `(n_geos, T, n_rf_channels)` for any
      time dimension `T`.
    frequency: Optional tensor with dimensions `(n_geos, T, n_rf_channels)` for
      any time dimension `T`.
    rf_impressions: Optional tensor with dimensions `(n_geos, T, n_rf_channels)`
      for any time dimension `T`.
    rf_spend: Optional tensor with dimensions `(n_rf_channels,)` or `(n_geos, T,
      n_rf_channels)` for any time dimension `T`. If the object includes
      variables with modified time periods, then this tensor must be provided at
      the geo and time granularity.
    organic_media: Optional tensor with dimensions `(n_geos, T,
      n_organic_media_channels)` for any time dimension `T`.
    organic_reach: Optional tensor with dimensions `(n_geos, T,
      n_organic_rf_channels)` for any time dimension `T`.
    organic_frequency: Optional tensor with dimensions `(n_geos, T,
      n_organic_rf_channels)` for any time dimension `T`.
    non_media_treatments: Optional tensor with dimensions `(n_geos, T,
      n_non_media_channels)` for any time dimension `T`.
    controls: Optional tensor with dimensions `(n_geos, n_times, n_controls)`.
    revenue_per_kpi: Optional tensor with dimensions `(n_geos, T)` for any time
      dimension `T`.
    time: Optional tensor of time coordinates in the "YYYY-mm-dd" string format
      for time dimension `T`.
  """

  media: Optional[backend.Tensor]
  media_spend: Optional[backend.Tensor]
  reach: Optional[backend.Tensor]
  frequency: Optional[backend.Tensor]
  rf_impressions: Optional[backend.Tensor]
  rf_spend: Optional[backend.Tensor]
  organic_media: Optional[backend.Tensor]
  organic_reach: Optional[backend.Tensor]
  organic_frequency: Optional[backend.Tensor]
  non_media_treatments: Optional[backend.Tensor]
  controls: Optional[backend.Tensor]
  revenue_per_kpi: Optional[backend.Tensor]
  time: Optional[backend.Tensor]

  def __init__(
      self,
      media: Optional[backend.Tensor] = None,
      media_spend: Optional[backend.Tensor] = None,
      reach: Optional[backend.Tensor] = None,
      frequency: Optional[backend.Tensor] = None,
      rf_impressions: Optional[backend.Tensor] = None,
      rf_spend: Optional[backend.Tensor] = None,
      organic_media: Optional[backend.Tensor] = None,
      organic_reach: Optional[backend.Tensor] = None,
      organic_frequency: Optional[backend.Tensor] = None,
      non_media_treatments: Optional[backend.Tensor] = None,
      controls: Optional[backend.Tensor] = None,
      revenue_per_kpi: Optional[backend.Tensor] = None,
      time: Optional[Sequence[str] | backend.Tensor] = None,
  ):
    self.media = (
        backend.cast(media, backend.float32) if media is not None else None
    )
    self.media_spend = (
        backend.cast(media_spend, backend.float32)
        if media_spend is not None
        else None
    )
    self.reach = (
        backend.cast(reach, backend.float32) if reach is not None else None
    )
    self.frequency = (
        backend.cast(frequency, backend.float32)
        if frequency is not None
        else None
    )
    self.rf_impressions = (
        backend.cast(rf_impressions, backend.float32)
        if rf_impressions is not None
        else None
    )
    self.rf_spend = (
        backend.cast(rf_spend, backend.float32)
        if rf_spend is not None
        else None
    )
    self.organic_media = (
        backend.cast(organic_media, backend.float32)
        if organic_media is not None
        else None
    )
    self.organic_reach = (
        backend.cast(organic_reach, backend.float32)
        if organic_reach is not None
        else None
    )
    self.organic_frequency = (
        backend.cast(organic_frequency, backend.float32)
        if organic_frequency is not None
        else None
    )
    self.non_media_treatments = (
        backend.cast(non_media_treatments, backend.float32)
        if non_media_treatments is not None
        else None
    )
    self.controls = (
        backend.cast(controls, backend.float32)
        if controls is not None
        else None
    )
    self.revenue_per_kpi = (
        backend.cast(revenue_per_kpi, backend.float32)
        if revenue_per_kpi is not None
        else None
    )
    self.time = (
        backend.to_tensor(time, dtype=backend.string)
        if time is not None
        else None
    )
    self._validate_n_dims()

  def __eq__(self, other: Any) -> bool:
    """Provides safe equality comparison for mixed tensor/non-tensor fields."""
    if type(self) is not type(other):
      return NotImplemented
    for field in dataclasses.fields(self):
      a = getattr(self, field.name)
      b = getattr(other, field.name)
      if a is None and b is None:
        continue
      if a is None or b is None:
        return False
      try:
        if not bool(np.all(backend.to_tensor(backend.equal(a, b)))):
          return False
      except (ValueError, TypeError):
        if a != b:
          return False
    return True

  def total_spend(self) -> backend.Tensor | None:
    """Returns the total spend tensor.

    Returns:
      The `media_spend` tensor (if present) concatenated with the `rf_spend`
      tensor (if present), in this order. If both tensors are missing, returns
      `None`.
    """
    spend_tensors = []
    if self.media_spend is not None:
      spend_tensors.append(self.media_spend)
    if self.rf_spend is not None:
      spend_tensors.append(self.rf_spend)
    return (
        backend.concatenate(spend_tensors, axis=-1) if spend_tensors else None
    )

  def get_modified_times(self, meridian: model.Meridian) -> int | None:
    """Returns `n_times` of any tensor where `n_times` has been modified.

    This method compares the time dimensions of the attributes in the
    `DataTensors` object with the corresponding tensors in the `meridian`
    object. If any of the time dimensions are different, then this method
    returns the modified number of time periods of the tensor in the
    `DataTensors` object. If all time dimensions are the same, returns `None`.

    Args:
      meridian: A Meridian object to validate against and get the original data
        tensors from.

    Returns:
      The `n_times` of any tensor where `n_times` is different from the times
      of the corresponding tensor in the `meridian` object. If all time
      dimensions are the same, returns `None`.
    """
    for field in dataclasses.fields(self):
      new_tensor = getattr(self, field.name)
      if field.name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(meridian.rf_tensors, field.name)
      else:
        old_tensor = getattr(meridian.input_data, field.name)
      # The time dimension is always the second dimension, except for when spend
      # data is provided with only one dimension of (n_channels).
      if (
          new_tensor is not None
          and old_tensor is not None
          and new_tensor.ndim > 1
          and old_tensor.ndim > 1
          and new_tensor.shape[1] != old_tensor.shape[1]
      ):
        return new_tensor.shape[1]
    return None

  def filter_fields(self, fields: Sequence[str]) -> Self:
    """Returns a new DataTensors object with only the specified fields."""
    output = {}
    for field in fields:
      output[field] = getattr(self, field)
    return DataTensors(**output)

  def validate_and_fill_missing_data(
      self,
      required_tensors_names: Sequence[str],
      meridian: model.Meridian,
      allow_modified_times: bool = True,
  ) -> Self:
    """Fills missing data tensors with their original values from the model.

    This method uses the collection of data tensors set in the DataTensor class
    and fills in the missing tensors with their original values from the
    Meridian object that is passed in. For example, if `required_tensors_names =
    ["media", "reach", "frequency"]` and only `media` is set in the DataTensors
    class, then this method will output a new DataTensors object with the
    `media` value in this object plus the values of the `reach` and `frequency`
    from the `meridian` object.

    Args:
      required_tensors_names: A sequence of data tensors names to validate and
        fill in with the original values from the `meridian` object.
      meridian: The Meridian object to validate against and get the original
        data tensors from.
      allow_modified_times: A boolean flag indicating whether to allow modified
        time dimensions in the new data tensors. If False, an error will be
        raised if the time dimensions of any tensor is modified.

    Returns:
      A `DataTensors` container with the original values from the Meridian
      object filled in for the missing data tensors.
    """
    self._validate_correct_variables_filled(required_tensors_names, meridian)
    self._validate_geo_dims(required_tensors_names, meridian)
    self._validate_channel_dims(required_tensors_names, meridian)
    if allow_modified_times:
      self._validate_time_dims_flexible_times(required_tensors_names, meridian)
    else:
      self._validate_time_dims(required_tensors_names, meridian)

    return self._fill_default_values(required_tensors_names, meridian)

  def _validate_n_dims(self):
    """Raises an error if the tensors have the wrong number of dimensions."""
    for field in dataclasses.fields(self):
      tensor = getattr(self, field.name)
      if tensor is None:
        continue
      if field.name == constants.REVENUE_PER_KPI:
        _check_n_dims(tensor, field.name, 2)
      elif field.name in [constants.MEDIA_SPEND, constants.RF_SPEND]:
        if tensor.ndim not in [1, 3]:
          raise ValueError(
              f"New `{field.name}` must have 1 or 3 dimensions. Found"
              f" {tensor.ndim} dimensions."
          )
      elif field.name == constants.TIME:
        _check_n_dims(tensor, field.name, 1)
      else:
        _check_n_dims(tensor, field.name, 3)

  def _validate_correct_variables_filled(
      self, required_variables: Sequence[str], meridian: model.Meridian
  ):
    """Validates that the correct variables are filled.

    Args:
      required_variables: A sequence of data tensors names that are required to
        be filled in.
      meridian: The Meridian object to validate against.

    Raises:
      ValueError: If an attribute exists in the `DataTensors` object that is not
        in the `meridian` object, it is not allowed to be used in analysis.
      Warning: If an attribute exists in the `DataTensors` object that is not in
        the `required_variables` list, it will be ignored.
    """
    for field in dataclasses.fields(self):
      tensor = getattr(self, field.name)
      if tensor is None:
        continue
      if field.name not in required_variables:
        warnings.warn(
            f"A `{field.name}` value was passed in the `new_data` argument. "
            "This is not supported and will be ignored."
        )
      if field.name in required_variables:
        if field.name == constants.RF_IMPRESSIONS:
          if meridian.n_rf_channels == 0:
            raise ValueError(
                "New `rf_impressions` is not allowed because there are no R&F"
                " channels in the Meridian model."
            )
        elif getattr(meridian.input_data, field.name) is None:
          raise ValueError(
              f"New `{field.name}` is not allowed because the input data to the"
              f" Meridian model does not contain `{field.name}`."
          )

  def _validate_geo_dims(
      self, required_fields: Sequence[str], meridian: model.Meridian
  ):
    """Validates the geo dimension of the specified data variables."""
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if new_tensor is not None and new_tensor.shape[0] != meridian.n_geos:
        # Skip spend and time data with only 1 dimension.
        if new_tensor.ndim == 1:
          continue
        raise ValueError(
            f"New `{var_name}` is expected to have {meridian.n_geos}"
            f" geos. Found {new_tensor.shape[0]} geos."
        )

  def _validate_channel_dims(
      self, required_fields: Sequence[str], meridian: model.Meridian
  ):
    """Validates the channel dimension of the specified data variables."""
    for var_name in required_fields:
      if var_name in [constants.REVENUE_PER_KPI, constants.TIME]:
        continue
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(meridian.rf_tensors, var_name)
      else:
        old_tensor = getattr(meridian.input_data, var_name)
      if new_tensor is not None:
        assert old_tensor is not None
        if new_tensor.shape[-1] != old_tensor.shape[-1]:
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[-1]}"
              f" channels. Found {new_tensor.shape[-1]} channels."
          )

  def _validate_time_dims(
      self, required_fields: Sequence[str], meridian: model.Meridian
  ):
    """Validates the time dimension of the specified data variables."""
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(meridian.rf_tensors, var_name)
      else:
        old_tensor = getattr(meridian.input_data, var_name)

      # Skip spend data with only 1 dimension of (n_channels).
      if (
          var_name in [constants.MEDIA_SPEND, constants.RF_SPEND]
          and new_tensor is not None
          and new_tensor.ndim == 1
      ):
        continue

      if new_tensor is not None:
        assert old_tensor is not None
        if (
            var_name == constants.TIME
            and new_tensor.shape[0] != old_tensor.shape[0]
        ):
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[0]}"
              f" time periods. Found {new_tensor.shape[0]} time periods."
          )
        elif new_tensor.ndim > 1 and new_tensor.shape[1] != old_tensor.shape[1]:
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[1]}"
              f" time periods. Found {new_tensor.shape[1]} time periods."
          )

  def _validate_time_dims_flexible_times(
      self, required_fields: Sequence[str], meridian: model.Meridian
  ):
    """Validates the time dimension for the flexible times case."""
    new_n_times = self.get_modified_times(meridian)
    # If no times were modified, then there is nothing more to validate.
    if new_n_times is None:
      return

    missing_params = []
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(meridian.rf_tensors, var_name)
      else:
        old_tensor = getattr(meridian.input_data, var_name)

      if old_tensor is None:
        continue

      if new_tensor is None:
        missing_params.append(var_name)
      elif var_name == constants.TIME and new_tensor.shape[0] != new_n_times:
        raise ValueError(
            "If the time dimension of any variable in `new_data` is "
            "modified, then all variables must be provided with the same "
            f"number of time periods. `{var_name}` has {new_tensor.shape[1]} "
            "time periods, which does not match the modified number of time "
            f"periods, {new_n_times}.",
        )
      elif (
          var_name in [constants.MEDIA_SPEND, constants.RF_SPEND]
          and new_tensor.ndim == 1
      ):
        raise ValueError(
            "If the time dimension of any variable in `new_data` is modified, "
            "then spend variables must be provided at the geo and time "
            "granularity with thee same number of time periods as the other "
            f"new data variables. Found `{var_name}` with only 1 dimension."
        )
      elif new_tensor.ndim > 1 and new_tensor.shape[1] != new_n_times:
        raise ValueError(
            "If the time dimension of any variable in `new_data` is "
            "modified, then all variables must be provided with the same "
            f"number of time periods. `{var_name}` has {new_tensor.shape[1]} "
            "time periods, which does not match the modified number of time "
            f"periods, {new_n_times}.",
        )

    if missing_params:
      raise ValueError(
          "If the time dimension of a variable in `new_data` is modified,"
          " then all variables must be provided in `new_data`."
          f" The following variables are missing: `{missing_params}`."
      )

  def _fill_default_values(
      self, required_fields: Sequence[str], meridian: model.Meridian
  ) -> Self:
    """Fills default values and returns a new DataTensors object."""
    output = {}
    for field in dataclasses.fields(self):
      var_name = field.name
      if var_name not in required_fields:
        continue

      if hasattr(meridian.media_tensors, var_name):
        old_tensor = getattr(meridian.media_tensors, var_name)
      elif hasattr(meridian.rf_tensors, var_name):
        old_tensor = getattr(meridian.rf_tensors, var_name)
      elif hasattr(meridian.organic_media_tensors, var_name):
        old_tensor = getattr(meridian.organic_media_tensors, var_name)
      elif hasattr(meridian.organic_rf_tensors, var_name):
        old_tensor = getattr(meridian.organic_rf_tensors, var_name)
      elif var_name == constants.NON_MEDIA_TREATMENTS:
        old_tensor = meridian.non_media_treatments
      elif var_name == constants.CONTROLS:
        old_tensor = meridian.controls
      elif var_name == constants.REVENUE_PER_KPI:
        old_tensor = meridian.revenue_per_kpi
      elif var_name == constants.TIME:
        old_tensor = backend.to_tensor(
            meridian.input_data.time.values.tolist(), dtype=backend.string
        )
      else:
        continue

      new_tensor = getattr(self, var_name)
      output[var_name] = new_tensor if new_tensor is not None else old_tensor

    return DataTensors(**output)


@dataclasses.dataclass
class DistributionTensors(backend.ExtensionType):
  """Container for parameters distributions arguments of Analyzer methods."""

  alpha_m: Optional[backend.Tensor] = None
  alpha_rf: Optional[backend.Tensor] = None
  alpha_om: Optional[backend.Tensor] = None
  alpha_orf: Optional[backend.Tensor] = None
  ec_m: Optional[backend.Tensor] = None
  ec_rf: Optional[backend.Tensor] = None
  ec_om: Optional[backend.Tensor] = None
  ec_orf: Optional[backend.Tensor] = None
  slope_m: Optional[backend.Tensor] = None
  slope_rf: Optional[backend.Tensor] = None
  slope_om: Optional[backend.Tensor] = None
  slope_orf: Optional[backend.Tensor] = None
  beta_gm: Optional[backend.Tensor] = None
  beta_grf: Optional[backend.Tensor] = None
  beta_gom: Optional[backend.Tensor] = None
  beta_gorf: Optional[backend.Tensor] = None
  mu_t: Optional[backend.Tensor] = None
  tau_g: Optional[backend.Tensor] = None
  gamma_gc: Optional[backend.Tensor] = None
  gamma_gn: Optional[backend.Tensor] = None


def get_central_tendency_and_ci(
    data: np.ndarray | backend.Tensor,
    confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
    axis: tuple[int, ...] = (0, 1),
    include_median=False,
) -> np.ndarray:
  """Calculates mean and credible intervals for the given data.

  Args:
    data: Data for the metric.
    confidence_level: Confidence level for computing credible intervals,
      represented as a value between zero and one.
    axis: Axis or axes along which the mean, median, and quantiles are computed.
    include_median: A boolean flag indicating whether to calculate and include
      the median in the output Dataset (default: False).

  Returns:
    A numpy array or backend.Tensor containing the mean and credible intervals
    for the given data. Optionally, it also includes the median.
  """
  mean = np.mean(data, axis=axis, keepdims=False)
  ci_lo = np.quantile(data, (1 - confidence_level) / 2, axis=axis)
  ci_hi = np.quantile(data, (1 + confidence_level) / 2, axis=axis)

  if include_median:
    median = np.median(data, axis=axis, keepdims=False)
    return np.stack([mean, median, ci_lo, ci_hi], axis=-1)
  else:
    return np.stack([mean, ci_lo, ci_hi], axis=-1)


def _transformed_new_or_scaled(
    new_variable: backend.Tensor | None,
    transformer: transformers.TensorTransformer | None,
    scaled_variable: backend.Tensor | None,
) -> backend.Tensor | None:
  """Returns the transformed new variable or the scaled variable.

  If the `new_variable` is present, returns
  `transformer.forward(new_variable)`. Otherwise, returns the
  `scaled_variable`.

  Args:
    new_variable: Optional tensor to be transformed..
    transformer: Optional DataTransformer.
    scaled_variable: Tensor to be returned if `new_variable` is None.

  Returns:
    The transformed new variable (if the new variable is present) or the
    original scaled variable from the input data otherwise.
  """
  if new_variable is None or transformer is None:
    return scaled_variable
  return transformer.forward(new_variable)


def _calc_rsquared(expected, actual):
  """Calculates r-squared between actual and expected outcome."""
  return 1 - backend.nanmean((expected - actual) ** 2) / backend.nanvar(actual)


def _calc_mape(expected, actual):
  """Calculates MAPE between actual and expected outcome."""
  return backend.nanmean(backend.absolute((actual - expected) / actual))


def _calc_weighted_mape(expected, actual):
  """Calculates wMAPE between actual and expected outcome (weighted by actual)."""
  return backend.nansum(backend.absolute(actual - expected)) / backend.nansum(
      actual
  )


def _warn_if_geo_arg_in_kwargs(**kwargs):
  """Raises warning if a geo-level argument is used with national model."""
  for kwarg, value in kwargs.items():
    if (
        kwarg in constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS
        and value != constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]
    ):
      warnings.warn(
          f"The `{kwarg}` argument is ignored in the national model. It will be"
          " reset to"
          f" `{constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]}`."
      )


def _check_n_dims(tensor: backend.Tensor, name: str, n_dims: int):
  """Raises an error if the tensor has the wrong number of dimensions."""
  if tensor.ndim != n_dims:
    raise ValueError(
        f"New `{name}` must have {n_dims} dimension(s). Found"
        f" {tensor.ndim} dimension(s)."
    )


def _is_bool_list(l: Sequence[Any]) -> bool:
  """Returns True if the list contains only booleans."""
  return all(isinstance(item, bool) for item in l)


def _is_str_list(l: Sequence[Any]) -> bool:
  """Returns True if the list contains only strings."""
  return all(isinstance(item, str) for item in l)


def _validate_selected_times(
    selected_times: Sequence[str] | Sequence[bool],
    input_times: xr.DataArray,
    n_times: int,
    arg_name: str,
    comparison_arg_name: str,
):
  """Raises an error if selected_times is invalid.

  This checks that the `selected_times` argument is a list of strings or a list
  of booleans. If it is a list of strings, then each string must match the name
  of a time period coordinate in `input_times`. If it is a list of booleans,
  then it must have the same number of elements as `n_times`.

  Args:
    selected_times: Optional list of times to validate.
    input_times: Time dimension coordinates from `InputData.time` or
      `InputData.media_time`.
    n_times: The number of time periods in the tensor.
    arg_name: The name of the argument being validated.
    comparison_arg_name: The name of the argument being compared to.
  """
  if not selected_times:
    return
  if _is_bool_list(selected_times):
    if len(selected_times) != n_times:
      raise ValueError(
          f"Boolean `{arg_name}` must have the same number of elements as "
          f"there are time period coordinates in {comparison_arg_name}."
      )
  elif _is_str_list(selected_times):
    if any(time not in input_times for time in selected_times):
      raise ValueError(
          f"`{arg_name}` must match the time dimension names from "
          "meridian.InputData."
      )
  else:
    raise ValueError(
        f"`{arg_name}` must be a list of strings or a list of booleans."
    )


def _validate_flexible_selected_times(
    selected_times: Sequence[str] | Sequence[bool] | None,
    media_selected_times: Sequence[str] | Sequence[bool] | None,
    new_n_media_times: int,
    new_time: Sequence[str] | None = None,
):
  """Raises an error if selected times or media selected times is invalid.

  This checks that (1) the `selected_times` and `media_selected_times` arguments
  are lists of booleans with the same number of elements as `new_n_media_times`,
  or (2) the `selected_times` and `media_selected_times` arguments are lists of
  strings and the `new_time` list is provided and `selected_times` and
  `media_selected_times` are subsets of `new_time`. This is only relevant if the
  time dimension of any of the variables in `new_data` used in the analysis is
  modified.

  Args:
    selected_times: Optional list of times to validate.
    media_selected_times: Optional list of media times to validate.
    new_n_media_times: The number of time periods in the new data.
    new_time: The optional time dimension of the new data.
  """
  if selected_times and (
      not (
          _is_bool_list(selected_times)
          and len(selected_times) == new_n_media_times
      )
      and not (
          _is_str_list(selected_times)
          and new_time is not None
          and set(selected_times) <= set(new_time)
      )
  ):
    raise ValueError(
        "If `media`, `reach`, `frequency`, `organic_media`,"
        " `organic_reach`, `organic_frequency`, `non_media_treatments`, or"
        " `revenue_per_kpi` is provided with a different number of time"
        " periods than in `InputData`, then (1) `selected_times` must be a list"
        " of booleans with length equal to the number of time periods in"
        " the new data, or (2) `selected_times` must be a list of strings and"
        " `new_time` must be provided and `selected_times` must be a subset of"
        " `new_time`."
    )

  if media_selected_times and (
      not (
          _is_bool_list(media_selected_times)
          and len(media_selected_times) == new_n_media_times
      )
      and not (
          _is_str_list(media_selected_times)
          and new_time is not None
          and set(media_selected_times) <= set(new_time)
      )
  ):
    raise ValueError(
        "If `media`, `reach`, `frequency`, `organic_media`,"
        " `organic_reach`, `organic_frequency`, `non_media_treatments`, or"
        " `revenue_per_kpi` is provided with a different number of time"
        " periods than in `InputData`, then (1) `media_selected_times` must be"
        " a list of booleans with length equal to the number of time"
        " periods in the new data, or (2) `media_selected_times` must be a list"
        " of strings and `new_time` must be provided and"
        " `media_selected_times` must be a subset of `new_time`."
    )


def _scale_tensors_by_multiplier(
    data: DataTensors,
    multiplier: float,
    by_reach: bool,
) -> DataTensors:
  """Get scaled tensors for incremental outcome calculation.

  Args:
    data: DataTensors object containing the optional tensors to scale. Only
      `media`, `reach`, `frequency`, `organic_media`, `organic_reach`, and
      `organic_frequency` are scaled. The other tensors remain unchanged.
    multiplier: Float indicating the factor to scale tensors by.
    by_reach: Boolean indicating whether to scale reach or frequency when rf
      data is available.

  Returns:
    A `DataTensors` object containing scaled tensor parameters. The original
    tensors that should not be scaled remain unchanged.
  """
  incremented_data = {}
  if data.media is not None:
    incremented_data[constants.MEDIA] = data.media * multiplier
  if data.reach is not None and data.frequency is not None:
    if by_reach:
      incremented_data[constants.REACH] = data.reach * multiplier
      incremented_data[constants.FREQUENCY] = data.frequency
    else:
      incremented_data[constants.REACH] = data.reach
      incremented_data[constants.FREQUENCY] = data.frequency * multiplier
  if data.organic_media is not None:
    incremented_data[constants.ORGANIC_MEDIA] = data.organic_media * multiplier
  if data.organic_reach is not None and data.organic_frequency is not None:
    if by_reach:
      incremented_data[constants.ORGANIC_REACH] = (
          data.organic_reach * multiplier
      )
      incremented_data[constants.ORGANIC_FREQUENCY] = data.organic_frequency
    else:
      incremented_data[constants.ORGANIC_REACH] = data.organic_reach
      incremented_data[constants.ORGANIC_FREQUENCY] = (
          data.organic_frequency * multiplier
      )

  # Include the original data that does not get scaled.
  incremented_data[constants.NON_MEDIA_TREATMENTS] = data.non_media_treatments
  incremented_data[constants.MEDIA_SPEND] = data.media_spend
  incremented_data[constants.RF_SPEND] = data.rf_spend
  incremented_data[constants.CONTROLS] = data.controls
  incremented_data[constants.REVENUE_PER_KPI] = data.revenue_per_kpi

  return DataTensors(**incremented_data)


def _central_tendency_and_ci_by_prior_and_posterior(
    prior: backend.Tensor,
    posterior: backend.Tensor,
    metric_name: str,
    xr_dims: Sequence[str],
    xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
    include_median: bool = False,
) -> xr.Dataset:
  """Calculates central tendency and CI of prior/posterior data for a metric.

  Args:
    prior: A tensor with the prior data for the metric.
    posterior: A tensor with the posterior data for the metric.
    metric_name: The name of the input metric for the computations.
    xr_dims: A list of dimensions for the output dataset.
    xr_coords: A dictionary with the coordinates for the output dataset.
    confidence_level: Confidence level for computing credible intervals,
      represented as a value between zero and one.
    include_median: A boolean flag indicating whether to calculate and include
      the median in the output Dataset (default: False).

  Returns:
    An xarray Dataset containing central tendency and confidence intervals for
    prior and posterior data for the metric.
  """
  metrics = np.stack(
      [
          get_central_tendency_and_ci(
              prior, confidence_level, include_median=include_median
          ),
          get_central_tendency_and_ci(
              posterior, confidence_level, include_median=include_median
          ),
      ],
      axis=-1,
  )
  xr_data = {metric_name: (xr_dims, metrics)}
  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


class Analyzer:
  """Runs calculations to analyze the raw data after fitting the model."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian
    # Make the meridian object ready for methods in this analyzer that create
    # backend.function computation graphs: it should be frozen for no more
    # internal states mutation before those graphs execute.
    self._meridian.populate_cached_properties()

  @backend.function(jit_compile=True)
  def _get_kpi_means(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
  ) -> backend.Tensor:
    """Computes batched KPI means.

    Note that the output array has the same number of time periods as the media
    data (lagged time periods are included).

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `controls`. The `media`,
        `reach`, `organic_media`, `organic_reach` and `non_media_treatments`
        tensors are expected to be scaled by their corresponding transformers.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, organic RF, non-media treatments,
        and controls (if available).

    Returns:
      Tensor representing computed kpi means.
    """
    tau_gt = backend.expand_dims(dist_tensors.tau_g, -1) + backend.expand_dims(
        dist_tensors.mu_t, -2
    )
    combined_media_transformed, combined_beta = (
        self._get_transformed_media_and_beta(
            data_tensors=data_tensors,
            dist_tensors=dist_tensors,
        )
    )

    result = tau_gt + backend.einsum(
        "...gtm,...gm->...gt", combined_media_transformed, combined_beta
    )
    if self._meridian.controls is not None:
      result += backend.einsum(
          "...gtc,...gc->...gt",
          data_tensors.controls,
          dist_tensors.gamma_gc,
      )
    if data_tensors.non_media_treatments is not None:
      result += backend.einsum(
          "...gtm,...gm->...gt",
          data_tensors.non_media_treatments,
          dist_tensors.gamma_gn,
      )
    return result

  def _use_kpi(self, use_kpi: bool = False) -> bool:
    """Checks if KPI analysis should be used.

    If `use_kpi` is `True` but `kpi_type=REVENUE`, then `use_kpi` is ignored.

    If `use_kpi` is `False`, then  `revenue_per_kpi` is required to perform
    the revenue analysis. Setting `use_kpi` to `False` in this case is ignored.

    Args:
      use_kpi: A boolean flag indicating whether KPI analysis should be used.

    Returns:
      A boolean flag indicating whether KPI analysis should be used.
    Raises:
      UserWarning: If the KPI type is revenue and use_kpi is True or if
      `use_kpi=False` but `revenue_per_kpi` is not available.
    """
    if use_kpi and self._meridian.input_data.kpi_type == constants.REVENUE:
      warnings.warn(
          "Setting `use_kpi=True` has no effect when `kpi_type=REVENUE`"
          " since in this case, KPI is equal to revenue."
      )
      return False

    if not use_kpi and self._meridian.input_data.revenue_per_kpi is None:
      warnings.warn(
          "Revenue analysis is not available when `revenue_per_kpi` is"
          " unknown. Defaulting to KPI analysis."
      )

    return use_kpi or self._meridian.input_data.revenue_per_kpi is None

  def _get_adstock_dataframe(
      self,
      channel_type: str,
      l_range: np.ndarray,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> pd.DataFrame:
    """Computes decayed effect means and CIs for media or RF channels.

    Args:
      channel_type: Specifies `media`, `rf`, `organic_media`, or `organic_rf`
        for computing prior and posterior decayed effects.
      l_range: The range of time across which the adstock effect is computed.
      xr_dims: A list of dimensions for the output dataset.
      xr_coords: A dictionary with the coordinates for the output dataset.
      confidence_level: Confidence level for computing credible intervals,
        represented as a value between zero and one.

    Returns:
      Pandas DataFrame containing the channel, time_units, distribution, ci_hi,
      ci_lo, and mean decayed effects for either media or RF channel types.
    """
    window_size = min(
        self._meridian.model_spec.max_lag + 1, self._meridian.n_media_times
    )
    if channel_type == constants.MEDIA:
      prior = self._meridian.inference_data.prior.alpha_m.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_m.values,
          (-1, self._meridian.n_media_channels),
      )
      decay_functions = self._meridian.adstock_decay_spec.media
    elif channel_type == constants.RF:
      prior = self._meridian.inference_data.prior.alpha_rf.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_rf.values,
          (-1, self._meridian.n_rf_channels),
      )
      decay_functions = self._meridian.adstock_decay_spec.rf
    elif channel_type == constants.ORGANIC_MEDIA:
      prior = self._meridian.inference_data.prior.alpha_om.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_om.values,
          (-1, self._meridian.n_organic_media_channels),
      )
      decay_functions = self._meridian.adstock_decay_spec.organic_media
    elif channel_type == constants.ORGANIC_RF:
      prior = self._meridian.inference_data.prior.alpha_orf.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_orf.values,
          (-1, self._meridian.n_organic_rf_channels),
      )
      decay_functions = self._meridian.adstock_decay_spec.organic_rf
    else:
      raise ValueError(
          f"Unsupported channel type for adstock decay: '{channel_type}'. "
      )

    decayed_effect_prior = adstock_hill.compute_decay_weights(
        alpha=backend.to_tensor(
            prior[backend.newaxis, ...], dtype=backend.float32
        ),
        l_range=backend.to_tensor(l_range, dtype=backend.float32),
        window_size=window_size,
        decay_functions=decay_functions,
        normalize=False,
    )
    decayed_effect_posterior = adstock_hill.compute_decay_weights(
        alpha=backend.to_tensor(
            posterior[backend.newaxis, ...], dtype=backend.float32
        ),
        l_range=backend.to_tensor(l_range, dtype=backend.float32),
        window_size=window_size,
        decay_functions=decay_functions,
        normalize=False,
    )

    decayed_effect_prior_transpose = backend.transpose(
        decayed_effect_prior, perm=[0, 1, 3, 2]
    )
    decayed_effect_posterior_transpose = backend.transpose(
        decayed_effect_posterior, perm=[0, 1, 3, 2]
    )
    adstock_dataset = _central_tendency_and_ci_by_prior_and_posterior(
        decayed_effect_prior_transpose,
        decayed_effect_posterior_transpose,
        constants.EFFECT,
        xr_dims,
        xr_coords,
        confidence_level,
    )
    return (
        adstock_dataset[constants.EFFECT]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.TIME_UNITS,
                constants.DISTRIBUTION,
            ],
            columns=constants.METRIC,
            values=constants.EFFECT,
        )
        .reset_index()
    )

  def _get_scaled_data_tensors(
      self,
      new_data: DataTensors | None = None,
      include_non_paid_channels: bool = True,
  ) -> DataTensors:
    """Get scaled tensors using given new data and original data.

    This method returns a new `DataTensors` container with scaled versions of
    `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
    `organic_frequency`, `non_media_treatments`, `controls` and
    `revenue_per_kpi` tensors. For each tensor, if its value is provided in the
    `new_data` argument, the provided tensors are used. Otherwise the original
    tensors from the Meridian model are used. The tensors are then either scaled
    by their corresponding transformers (`media`, `reach`, `organic_media`,
    `organic_reach`, `non_media_treatments`, `controls`), or left as is
    (`frequency`, `organic_frequency`, `revenue_per_kpi`). For example,

    ```
    _get_scaled_data_tensors(
        new_data=DataTensors(media=new_media),
    )
    ```

    returns a `DataTensors` container with `media` set to the scaled version of
    `new_media`, and all other tensors set to their original scaled values from
    the Meridian model.

    Args:
      new_data: An optional `DataTensors` container with optional tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `controls` and
        `revenue_per_kpi`. If `None`, the original scaled tensors from the
        Meridian object are used. If `new_data` is provided, the output contains
        the scaled versions of the tensors in `new_data` and the original scaled
        versions of all the remaining tensors. The new tensors' dimensions must
        match the dimensions of the corresponding original tensors from
        `meridian.input_data`.
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.

    Returns:
      A DataTensors object containing the scaled `media`, `reach`, `frequency`
      `organic_media`, `organic_reach`, `organic_frequency`,
      `non_media_treatments`, `controls` and `revenue_per_kpi` data tensors.
    """
    if new_data is None:
      return DataTensors(
          media=self._meridian.media_tensors.media_scaled,
          reach=self._meridian.rf_tensors.reach_scaled,
          frequency=self._meridian.rf_tensors.frequency,
          organic_media=self._meridian.organic_media_tensors.organic_media_scaled,
          organic_reach=self._meridian.organic_rf_tensors.organic_reach_scaled,
          organic_frequency=self._meridian.organic_rf_tensors.organic_frequency,
          non_media_treatments=self._meridian.non_media_treatments_normalized,
          controls=self._meridian.controls_scaled,
          revenue_per_kpi=self._meridian.revenue_per_kpi,
      )
    media_scaled = _transformed_new_or_scaled(
        new_variable=new_data.media,
        transformer=self._meridian.media_tensors.media_transformer,
        scaled_variable=self._meridian.media_tensors.media_scaled,
    )

    reach_scaled = _transformed_new_or_scaled(
        new_variable=new_data.reach,
        transformer=self._meridian.rf_tensors.reach_transformer,
        scaled_variable=self._meridian.rf_tensors.reach_scaled,
    )

    frequency = (
        new_data.frequency
        if new_data.frequency is not None
        else self._meridian.rf_tensors.frequency
    )

    controls_scaled = _transformed_new_or_scaled(
        new_variable=new_data.controls,
        transformer=self._meridian.controls_transformer,
        scaled_variable=self._meridian.controls_scaled,
    )
    revenue_per_kpi = (
        new_data.revenue_per_kpi
        if new_data.revenue_per_kpi is not None
        else self._meridian.revenue_per_kpi
    )

    if include_non_paid_channels:
      organic_media_scaled = _transformed_new_or_scaled(
          new_variable=new_data.organic_media,
          transformer=self._meridian.organic_media_tensors.organic_media_transformer,
          scaled_variable=self._meridian.organic_media_tensors.organic_media_scaled,
      )
      organic_reach_scaled = _transformed_new_or_scaled(
          new_variable=new_data.organic_reach,
          transformer=self._meridian.organic_rf_tensors.organic_reach_transformer,
          scaled_variable=self._meridian.organic_rf_tensors.organic_reach_scaled,
      )
      organic_frequency = (
          new_data.organic_frequency
          if new_data.organic_frequency is not None
          else self._meridian.organic_rf_tensors.organic_frequency
      )
      non_media_treatments_normalized = _transformed_new_or_scaled(
          new_variable=new_data.non_media_treatments,
          transformer=self._meridian.non_media_transformer,
          scaled_variable=self._meridian.non_media_treatments_normalized,
      )
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=frequency,
          organic_media=organic_media_scaled,
          organic_reach=organic_reach_scaled,
          organic_frequency=organic_frequency,
          non_media_treatments=non_media_treatments_normalized,
          controls=controls_scaled,
          revenue_per_kpi=revenue_per_kpi,
      )
    else:
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=frequency,
          controls=controls_scaled,
          revenue_per_kpi=revenue_per_kpi,
      )

  def _get_causal_param_names(
      self,
      include_non_paid_channels: bool,
  ) -> list[str]:
    """Gets media, RF, non-media, organic media, and organic RF distributions.

    Args:
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.

    Returns:
      A list containing available media, RF, non-media treatments, organic media
      and organic RF parameters names in inference data.
    """
    params = []
    if self._meridian.media_tensors.media is not None:
      params.extend([
          constants.EC_M,
          constants.SLOPE_M,
          constants.ALPHA_M,
          constants.BETA_GM,
      ])
    if self._meridian.rf_tensors.reach is not None:
      params.extend([
          constants.EC_RF,
          constants.SLOPE_RF,
          constants.ALPHA_RF,
          constants.BETA_GRF,
      ])
    if include_non_paid_channels:
      if self._meridian.organic_media_tensors.organic_media is not None:
        params.extend([
            constants.EC_OM,
            constants.SLOPE_OM,
            constants.ALPHA_OM,
            constants.BETA_GOM,
        ])
      if self._meridian.organic_rf_tensors.organic_reach is not None:
        params.extend([
            constants.EC_ORF,
            constants.SLOPE_ORF,
            constants.ALPHA_ORF,
            constants.BETA_GORF,
        ])
      if self._meridian.non_media_treatments is not None:
        params.extend([
            constants.GAMMA_GN,
        ])
    return params

  def _get_transformed_media_and_beta(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
      n_times_output: int | None = None,
  ) -> tuple[backend.Tensor | None, backend.Tensor | None]:
    """Function for transforming media using adstock and hill functions.

    This transforms the media tensor using the adstock and hill functions, in
    the desired order.

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, and organic RF channels.
      n_times_output: Optional number of time periods to output. Defaults to the
        corresponding argument defaults for `adstock_hill_media` and
        `adstock_hill_rf`.

    Returns:
      A tuple `(combined_media_transformed, combined_beta)`.
    """
    combined_medias = []
    combined_betas = []
    if data_tensors.media is not None:
      combined_medias.append(
          self._meridian.adstock_hill_media(
              media=data_tensors.media,
              alpha=dist_tensors.alpha_m,
              ec=dist_tensors.ec_m,
              slope=dist_tensors.slope_m,
              decay_functions=self._meridian.adstock_decay_spec.media,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gm)

    if data_tensors.reach is not None:
      combined_medias.append(
          self._meridian.adstock_hill_rf(
              reach=data_tensors.reach,
              frequency=data_tensors.frequency,
              alpha=dist_tensors.alpha_rf,
              ec=dist_tensors.ec_rf,
              slope=dist_tensors.slope_rf,
              decay_functions=self._meridian.adstock_decay_spec.rf,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_grf)
    if data_tensors.organic_media is not None:
      combined_medias.append(
          self._meridian.adstock_hill_media(
              media=data_tensors.organic_media,
              alpha=dist_tensors.alpha_om,
              ec=dist_tensors.ec_om,
              slope=dist_tensors.slope_om,
              decay_functions=self._meridian.adstock_decay_spec.organic_media,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gom)
    if data_tensors.organic_reach is not None:
      combined_medias.append(
          self._meridian.adstock_hill_rf(
              reach=data_tensors.organic_reach,
              frequency=data_tensors.organic_frequency,
              alpha=dist_tensors.alpha_orf,
              ec=dist_tensors.ec_orf,
              slope=dist_tensors.slope_orf,
              decay_functions=self._meridian.adstock_decay_spec.organic_rf,
              n_times_output=n_times_output,
          )
      )
      combined_betas.append(dist_tensors.beta_gorf)

    combined_media_transformed = backend.concatenate(combined_medias, axis=-1)
    combined_beta = backend.concatenate(combined_betas, axis=-1)
    return combined_media_transformed, combined_beta

  def filter_and_aggregate_geos_and_times(
      self,
      tensor: backend.Tensor,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      flexible_time_dim: bool = False,
      has_media_dim: bool = True,
  ) -> backend.Tensor:
    """Filters and/or aggregates geo and time dimensions of a tensor.

    Args:
      tensor: Tensor with dimensions `[..., n_geos, n_times]` or `[..., n_geos,
        n_times, n_channels]`, where `n_channels` is the number of either media
        channels, RF channels, all paid channels (media and RF), or all channels
        (media, RF, non-media, organic media, organic RF).
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included. The selected geos should match those in
        `InputData.geo`.
      selected_times: Optional list of times to include. This can either be a
        string list containing a subset of time dimension coordinates from
        `InputData.time` or a boolean list with length equal to the time
        dimension of the tensor. By default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the tensor is summed over all geos.
      aggregate_times: Boolean. If `True`, the tensor is summed over all time
        periods.
      flexible_time_dim: Boolean. If `True`, the time dimension of the tensor is
        not required to match the number of time periods in `InputData.time`. In
        this case, if using `selected_times`, it must be a boolean list with
        length equal to the time dimension of the tensor.
      has_media_dim: Boolean. Only used if `flexible_time_dim=True`. Otherwise,
        this is assumed based on the tensor dimensions. If `True`, the tensor is
        assumed to have a media dimension following the time dimension. If
        `False`, the last dimension of the tensor is assumed to be the time
        dimension.

    Returns:
      A tensor with filtered and/or aggregated geo and time dimensions.
    """
    mmm = self._meridian

    # Validate the tensor shape and determine if it has a media dimension.
    if flexible_time_dim:
      n_dim = tensor.ndim
      if (has_media_dim and n_dim < 3) or (not has_media_dim and n_dim < 2):
        raise ValueError(
            "The tensor must have at least 3 dimensions if `has_media_dim=True`"
            " or at least 2 dimensions if `has_media_dim=False`."
        )
      n_times = tensor.shape[-2] if has_media_dim else tensor.shape[-1]
    else:
      n_times = mmm.n_times
    # Allowed subsets of channels: media, RF, media+RF, all channels.
    allowed_n_channels = [
        mmm.n_media_channels,
        mmm.n_rf_channels,
        mmm.n_media_channels + mmm.n_rf_channels,
        mmm.n_media_channels
        + mmm.n_rf_channels
        + mmm.n_non_media_channels
        + mmm.n_organic_media_channels
        + mmm.n_organic_rf_channels,
    ]
    # Allow extra channel if aggregated (All_Channels) value is included.
    allowed_channel_dim = allowed_n_channels + [
        c + 1 for c in allowed_n_channels
    ]
    expected_shapes_w_media = [
        backend.TensorShape(shape)
        for shape in itertools.product(
            [mmm.n_geos], [n_times], allowed_channel_dim
        )
    ]
    expected_shape_wo_media = backend.TensorShape([mmm.n_geos, n_times])
    if not flexible_time_dim:
      if tensor.shape[-3:] in expected_shapes_w_media:
        has_media_dim = True
      elif tensor.shape[-2:] == expected_shape_wo_media:
        has_media_dim = False
      else:
        raise ValueError(
            "The tensor must have shape [..., n_geos, n_times, n_channels] or"
            " [..., n_geos, n_times] if `flexible_time_dim=False`."
        )
    else:
      if has_media_dim and tensor.shape[-3:] not in expected_shapes_w_media:
        raise ValueError(
            "If `has_media_dim=True`, the tensor must have shape "
            "`[..., n_geos, n_times, n_channels]`, where the time dimension is "
            "flexible."
        )
      elif not has_media_dim and tensor.shape[-2:] != expected_shape_wo_media:
        raise ValueError(
            "If `has_media_dim=False`, the tensor must have shape "
            "`[..., n_geos, n_times]`, where the time dimension is flexible."
        )
    geo_dim = tensor.ndim - 2 - (1 if has_media_dim else 0)
    time_dim = tensor.ndim - 1 - (1 if has_media_dim else 0)

    # Validate the selected geo and time dimensions and create a mask.
    if selected_geos is not None:
      if any(geo not in mmm.input_data.geo for geo in selected_geos):
        raise ValueError(
            "`selected_geos` must match the geo dimension names from "
            "meridian.InputData."
        )
      geo_indices = [
          i for i, x in enumerate(mmm.input_data.geo) if x in selected_geos
      ]
      tensor = backend.gather(
          tensor,
          backend.to_tensor(geo_indices, dtype=backend.int32),
          axis=geo_dim,
      )

    if selected_times is not None:
      _validate_selected_times(
          selected_times=selected_times,
          input_times=mmm.input_data.time,
          n_times=tensor.shape[time_dim],
          arg_name="selected_times",
          comparison_arg_name="`tensor`",
      )
      if _is_str_list(selected_times):
        time_indices = [
            i for i, x in enumerate(mmm.input_data.time) if x in selected_times
        ]
        tensor = backend.gather(
            tensor,
            backend.to_tensor(time_indices, dtype=backend.int32),
            axis=time_dim,
        )
      elif _is_bool_list(selected_times):
        time_indices = [i for i, x in enumerate(selected_times) if x]
        tensor = backend.gather(
            tensor,
            backend.to_tensor(time_indices, dtype=backend.int32),
            axis=time_dim,
        )

    tensor_dims = "...gt" + "m" * has_media_dim
    output_dims = (
        "g" * (not aggregate_geos)
        + "t" * (not aggregate_times)
        + "m" * has_media_dim
    )
    return backend.einsum(f"{tensor_dims}->...{output_dims}", tensor)

  def expected_outcome(
      self,
      use_posterior: bool = True,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_outcome: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates either prior or posterior expected outcome.

    This calculates `E(Outcome|Media, RF, Organic media, Organic RF, Non-media
    treatments, Controls)` for each posterior (or prior) parameter draw, where
    `Outcome` refers to either `revenue` if `use_kpi=False`, or `kpi` if
    `use_kpi=True`. When `revenue_per_kpi` is not defined, `use_kpi` cannot
    be `False`.

    If `new_data=None`, this method calculates expected outcome conditional on
    the values of the independent variables that the Meridian object was
    initialized with. The user can also override this historical data through
    the `new_data` argument, as long as the new tensors' dimensions match. For
    example,

    ```python
    new_data=DataTensors(reach=new_reach, frequency=new_frequency)
    ```

    In principle, expected outcome could be calculated with other time
    dimensions (for future predictions, for instance). However, this is not
    allowed with this method because of the additional complexities this
    introduces:

    1.  Corresponding price (revenue per KPI) data would also be needed.
    2.  If the model contains weekly effect parameters, then some method is
        needed to estimate or predict these effects for time periods outside of
        the training data window.

    Args:
      use_posterior: Boolean. If `True`, then the expected outcome posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      new_data: An optional `DataTensors` container with optional new tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `revenue_per_kpi`,
        `controls`. If `None`, expected outcome is calculated conditional on the
        original values of the data tensors that the Meridian object was
        initialized with. If `new_data` argument is used, expected outcome is
        calculated conditional on the values of the tensors passed in `new_data`
        and on the original values of the remaining unset tensors. For example,
        `expected_outcome(new_data=DataTensors(reach=new_reach,
        frequency=new_frequency))` calculates expected outcome conditional on
        the original `media`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments`, `revenue_per_kpi`, and
        `controls` tensors and on the new given values for `reach` and
        `frequency` tensors. The new tensors' dimensions must match the
        dimensions of the corresponding original tensors from `input_data`.
      selected_geos: Optional list of containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list of containing a subset of dates to include.
        The values accepted here must match time dimension coordinates from
        `InputData.time`. By default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected outcome is summed over
        all regions.
      aggregate_times: Boolean. If `True`, the expected outcome is summed over
        all time periods.
      inverse_transform_outcome: Boolean. If `True`, returns the expected
        outcome in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        outcome after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: Boolean. If `use_kpi = True`, the expected KPI is calculated;
        otherwise the expected revenue `(kpi * revenue_per_kpi)` is calculated.
        It is required that `use_kpi = True` if `revenue_per_kpi` is not defined
        or if `inverse_transform_outcome = False`.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of expected outcome (either KPI or revenue, depending on the
      `use_kpi` argument) with dimensions `(n_chains, n_draws, n_geos,
      n_times)`. The `n_geos` and `n_times` dimensions is dropped if
      `aggregate_geos=True` or `aggregate_time=True`, respectively.
    Raises:
      NotFittedModelError: if `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
    """
    use_kpi = self._use_kpi(use_kpi)
    self._check_kpi_transformation(inverse_transform_outcome, use_kpi)
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling"
          " `expected_outcome()`."
      )
    if new_data is None:
      new_data = DataTensors()

    required_fields = (
        constants.PAID_DATA + constants.NON_PAID_DATA + (constants.CONTROLS,)
    )
    filled_tensors = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_fields,
        meridian=self._meridian,
        allow_modified_times=False,
    )

    params = (
        self._meridian.inference_data.posterior
        if use_posterior
        else self._meridian.inference_data.prior
    )
    # We always compute the expected outcome of all channels, including non-paid
    # channels.
    data_tensors = self._get_scaled_data_tensors(
        new_data=filled_tensors,
        include_non_paid_channels=True,
    )

    n_draws = params.draw.size
    n_chains = params.chain.size
    outcome_means = backend.zeros(
        (n_chains, 0, self._meridian.n_geos, self._meridian.n_times)
    )
    batch_starting_indices = np.arange(n_draws, step=batch_size)
    param_list = (
        [
            constants.MU_T,
            constants.TAU_G,
        ]
        + ([constants.GAMMA_GC] if self._meridian.n_controls else [])
        + self._get_causal_param_names(include_non_paid_channels=True)
    )
    outcome_means_temps = []
    for start_index in batch_starting_indices:
      stop_index = np.min([n_draws, start_index + batch_size])
      batch_dists = {
          k: backend.to_tensor(params[k][:, start_index:stop_index, ...])
          for k in param_list
      }
      dist_tensors = DistributionTensors(**batch_dists)

      outcome_means_temps.append(
          self._get_kpi_means(
              data_tensors=data_tensors,
              dist_tensors=dist_tensors,
          )
      )
    outcome_means = backend.concatenate(
        [outcome_means, *outcome_means_temps], axis=1
    )
    if inverse_transform_outcome:
      outcome_means = self._meridian.kpi_transformer.inverse(outcome_means)
      if not use_kpi:
        outcome_means *= filled_tensors.revenue_per_kpi

    return self.filter_and_aggregate_geos_and_times(
        outcome_means,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
    )

  def _check_kpi_transformation(
      self, inverse_transform_outcome: bool, use_kpi: bool
  ):
    """Validates `use_kpi` functionality based on `inverse_transform_outcome`.

    When both `inverse_transform_outcome` and `use_kpi` are `False`, it
    indicates
    that the user wants to calculate "transformed revenue", which is not
    well-defined.

    Args:
      inverse_transform_outcome: Boolean. Indicates whether to inverse the
        transformation done by `KpiTransformer`.
      use_kpi: Boolean. Indicates whether to calculate the expected KPI or
        expected revenue.

    Raises:
      ValueError: If both `inverse_transform_outcome` and `use_kpi` are `False`.
    """
    if not inverse_transform_outcome and not use_kpi:
      raise ValueError(
          "use_kpi=False is only supported when inverse_transform_outcome=True."
      )

  def _get_incremental_kpi(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
      non_media_treatments_baseline_normalized: Sequence[float] | None = None,
  ) -> backend.Tensor:
    """Computes incremental KPI distribution.

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments, `controls` and
        `revenue_per_kpi`. The `media`, `reach`, `organic_media`,
        `organic_reach`, `non_media_treatments` and `controls` tensors are
        expected to be scaled by the corresponding transformers.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, organic RF and non-media
        treatments channels.
      non_media_treatments_baseline_normalized: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float that will be used as
        baseline for the given channel. The values are expected to be scaled by
        population for channels where
        `model_spec.non_media_population_scaling_id` is `True` and normalized by
        centering and scaling using means and standard deviations. This argument
        is required if the data contains non-media treatments.

    Returns:
      Tensor of incremental KPI distribution.
    """
    if (
        data_tensors.non_media_treatments is not None
        and non_media_treatments_baseline_normalized is None
    ):
      raise ValueError(
          "`non_media_treatments_baseline_normalized` must be passed to"
          " `_get_incremental_kpi` when `non_media_treatments` data is"
          " present."
      )
    n_media_times = self._meridian.n_media_times
    if data_tensors.media is not None:
      n_times = data_tensors.media.shape[1]  # pytype: disable=attribute-error
      n_times_output = n_times if n_times != n_media_times else None
    elif data_tensors.reach is not None:
      n_times = data_tensors.reach.shape[1]  # pytype: disable=attribute-error
      n_times_output = n_times if n_times != n_media_times else None
    else:
      raise ValueError("Both media_scaled and reach_scaled cannot be None.")
    combined_media_transformed, combined_beta = (
        self._get_transformed_media_and_beta(
            data_tensors=data_tensors,
            dist_tensors=dist_tensors,
            n_times_output=n_times_output,
        )
    )
    combined_media_kpi = backend.einsum(
        "...gtm,...gm->...gtm",
        combined_media_transformed,
        combined_beta,
    )
    if data_tensors.non_media_treatments is not None:
      non_media_kpi = backend.einsum(
          "gtn,...gn->...gtn",
          data_tensors.non_media_treatments
          - non_media_treatments_baseline_normalized,
          dist_tensors.gamma_gn,
      )
      return backend.concatenate([combined_media_kpi, non_media_kpi], axis=-1)
    else:
      return combined_media_kpi

  def _inverse_outcome(
      self,
      modeled_incremental_outcome: backend.Tensor,
      use_kpi: bool,
      revenue_per_kpi: backend.Tensor | None,
  ) -> backend.Tensor:
    """Inverses incremental outcome (revenue or KPI).

    This method assumes that additive changes on the model kpi scale
    correspond to additive changes on the original kpi scale. In other
    words, the intercept and control effects do not influence the media effects.

    Args:
      modeled_incremental_outcome: Tensor of incremental outcome modeled from
        parameter distributions.
      use_kpi: Boolean. If True, the incremental KPI is calculated. If False,
        incremental outcome `(KPI * revenue_per_kpi)` is calculated. Only used
        if `inverse_transform_outcome=True`. `use_kpi` must be True when
        `revenue_per_kpi` is not defined.
      revenue_per_kpi: Optional tensor of revenue per kpi. Uses
        `revenue_per_kpi` from `InputData` if None.

    Returns:
       Tensor of incremental outcome returned in terms of revenue or KPI.
    """
    use_kpi = self._use_kpi(use_kpi)
    if revenue_per_kpi is None:
      revenue_per_kpi = self._meridian.revenue_per_kpi
    t1 = self._meridian.kpi_transformer.inverse(
        backend.einsum("...m->m...", modeled_incremental_outcome)
    )
    t2 = self._meridian.kpi_transformer.inverse(backend.zeros_like(t1))
    kpi = backend.einsum("m...->...m", t1 - t2)

    if use_kpi:
      return kpi
    return backend.einsum("gt,...gtm->...gtm", revenue_per_kpi, kpi)

  @backend.function(
      jit_compile=True,
      static_argnames=[
          "inverse_transform_outcome",
          "use_kpi",
          "selected_geos",
          "selected_times",
          "aggregate_geos",
          "aggregate_times",
      ],
  )
  def _incremental_outcome_impl(
      self,
      data_tensors: DataTensors,
      dist_tensors: DistributionTensors,
      non_media_treatments_baseline_normalized: Sequence[float] | None = None,
      inverse_transform_outcome: bool | None = None,
      use_kpi: bool | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
  ) -> backend.Tensor:
    """Computes incremental outcome (revenue or KPI) on a batch of data.

    Args:
      data_tensors: A `DataTensors` container with the following tensors:
        `media`: `media` data scaled by the per-geo median, normalized by the
        geo population. Shape (n_geos x T x n_media_channels), for any time
        dimension T. `reach`: `reach` data scaled by the per-geo median,
        normalized by the geo population. Shape (n_geos x T x n_rf_channels),
        for any time dimension T. `frequency`: Contains frequency data with
        shape(n_geos x T x n_rf_channels), for any time dimension T.
        `organic_media`: `organic media data scaled by the per-geo median,
        normalized by the geo population. Shape (n_geos x T x
        n_organic_media_channels), for any time dimension T. `organic_reach`:
        `organic reach` data scaled by the per-geo median, normalized by the geo
        poulation. Shape (n_geos x T x n_organic_rf_channels), for any time
        dimension T. `organic_frequency`: `organic frequency data` with shape
        (n_geos x T x n_organic_rf_channels), for any time dimension T.
        `non_media_treatments`: `non_media_treatments` data scaled by population
        for the selected channels and normalized by means and standard
        deviations with shape (n_geos x T x n_non_media_channels), for any time
        dimension T. `revenue_per_kpi`: Contains revenue per kpi data with shape
        `(n_geos x T)`, for any time dimension `T`.
      dist_tensors: A `DistributionTensors` container with the distribution
        tensors for media, RF, organic media, organic RF and non-media
        treatments channels.
      non_media_treatments_baseline_normalized: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float that will be used as
        baseline for the given channel. The values are expected to be scaled by
        population for channels where
        `model_spec.non_media_population_scaling_id` is `True` and normalized by
        centering and scaling using means and standard deviations. This argument
        is required if the data contains non-media treatments.
      inverse_transform_outcome: Boolean. If `True`, returns the expected
        outcome in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        outcome after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: If True, the incremental KPI is calculated. If False, incremental
        revenue `(KPI * revenue_per_kpi)` is calculated. Only used if
        `inverse_transform_outcome=True`. `use_kpi` must be True when
        `revenue_per_kpi` is not defined.
      selected_geos: Contains a subset of geos to include. By default, all geos
        are included.
      selected_times: An optional string list containing a subset of
        `input_data.time` to include or a boolean list with length equal to the
        number of time periods in `data_tensors` if time is modified in
        `data_tensors`, or `input_data.n_times` otherwise. If time in
        `data_tensors` is modified, then only the boolean list can be used as
        `selected_times`. By default, all time periods are included.
      aggregate_geos: If True, then incremental outcome is summed over all
        regions.
      aggregate_times: If True, then incremental outcome is summed over all time
        periods.

    Returns:
      Tensor containing the incremental outcome distribution.
    """
    use_kpi = self._use_kpi(use_kpi)
    if (
        data_tensors.non_media_treatments is not None
        and non_media_treatments_baseline_normalized is None
    ):
      raise ValueError(
          "`non_media_treatments_baseline_normalized` must be passed to"
          " `_incremental_outcome_impl` when `non_media_treatments` data is"
          " present."
      )

    transformed_outcome = self._get_incremental_kpi(
        data_tensors=data_tensors,
        dist_tensors=dist_tensors,
        non_media_treatments_baseline_normalized=non_media_treatments_baseline_normalized,
    )
    if inverse_transform_outcome:
      incremental_outcome = self._inverse_outcome(
          transformed_outcome,
          use_kpi=use_kpi,
          revenue_per_kpi=data_tensors.revenue_per_kpi,
      )
    else:
      incremental_outcome = transformed_outcome
    return self.filter_and_aggregate_geos_and_times(
        tensor=incremental_outcome,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        flexible_time_dim=True,
        has_media_dim=True,
    )

  # TODO: b/407847021 - Add support for `new_data.time`.
  def incremental_outcome(
      self,
      use_posterior: bool = True,
      new_data: DataTensors | None = None,
      non_media_baseline_values: Sequence[float] | None = None,
      scaling_factor0: float = 0.0,
      scaling_factor1: float = 1.0,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      media_selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_outcome: bool = True,
      use_kpi: bool = False,
      by_reach: bool = True,
      include_non_paid_channels: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates either the posterior or prior incremental outcome.

    This calculates the media outcome of each media channel for each posterior
    or prior parameter draw. Incremental outcome is defined as:

    `E(Outcome|Treatment_1, Controls)` minus `E(Outcome|Treatment_0, Controls)`

    For paid & organic channels (without reach and frequency data),
    `Treatment_1` means that media execution for a given channel is multiplied
    by
    `scaling_factor1` (1.0 by default) for the set of time periods specified
    by `media_selected_times`. Similarly, `Treatment_0` means that media
    execution is multiplied by `scaling_factor0` (0.0 by default) for these time
    periods.

    For paid & organic channels with reach and frequency data, either reach or
    frequency is held fixed while the other is scaled, depending on the
    `by_reach` argument.

    For non-media treatments, `Treatment_1` means that the variable is set to
    historical values. `Treatment_0` means that the variable is set to its
    baseline value for all geos and time periods. Note that the scaling factors
    (`scaling_factor0` and `scaling_factor1`) are not applicable to non-media
    treatments.

    "Outcome" refers to either `revenue` if `use_kpi=False`, or `kpi` if
    `use_kpi=True`. When `revenue_per_kpi` is not defined, `use_kpi` cannot be
    False.

    If `new_data=None`, this method computes incremental outcome using `media`,
    `reach`, `frequency`, `organic_media`, `organic_reach`, `organic_frequency`,
    `non_media_treatments` and `revenue_per_kpi` tensors that the Meridian
    object was initialized with. This behavior can be overridden with the
    `new_data` argument. For example, `new_data=DataTensors(media=new_media)`
    calculates incremental outcome using the `new_media` tensor and the original
    values of `reach`, `frequency`, `organic_media`, `organic_reach`,
    `organic_frequency`, `non_media_treatments` and `revenue_per_kpi` tensors.

    The calculation in this method depends on two key assumptions made in the
    Meridian implementation:

    1.  Additivity of media effects (no interactions).
    2.  Additive changes on the model KPI scale correspond to additive
        changes on the original KPI scale. In other words, the intercept and
        control effects do not influence the media effects. This assumption
        currently holds because the outcome transformation only involves
        centering and scaling, for example, no log transformations.

    Args:
      use_posterior: Boolean. If `True`, then the incremental outcome posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      new_data: Optional `DataTensors` container with optional tensors: `media`,
        `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
        `None`, the incremental outcome is calculated using the `InputData`
        provided to the Meridian object. If `new_data` is provided, the
        incremental outcome is calculated using the new tensors in `new_data`
        and the original values of the remaining tensors. For example,
        `incremental_outcome(new_data=DataTensors(media=new_media)` computes the
        incremental outcome using `new_media` and the original values of
        `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
        any of the tensors in `new_data` is provided with a different number of
        time periods than in `InputData`, then all tensors must be provided with
        the same number of time periods.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      scaling_factor0: Float. The factor by which to scale the counterfactual
        scenario "Media_0" during the time periods specified in
        `media_selected_times`. Must be non-negative and less than
        `scaling_factor1`.
      scaling_factor1: Float. The factor by which to scale "Media_1" during the
        selected time periods specified in `media_selected_times`. Must be
        non-negative and greater than `scaling_factor0`.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        `new_data` if time is modified in `new_data`, or `input_data.n_times`
        otherwise. The incremental outcome corresponds to incremental KPI
        generated during the `selected_times` arg by media executed during the
        `media_selected_times` arg. Note that if `use_kpi=False`, then
        `selected_times` can only include the time periods that have
        `revenue_per_kpi` input data. By default, all time periods are included
        where `revenue_per_kpi` data is available.
      media_selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        KPI data or number of time periods in the `new_data` args, if provided.
        If `new_data` is provided, `media_selected_times` can select any subset
        of time periods in `new_data`. If `new_data` is not provided,
        `media_selected_times` selects from `InputData.time`. The incremental
        outcome corresponds to incremental KPI generated during the
        `selected_times` arg by treatment variables executed during the
        `media_selected_times` arg. For each channel, the incremental outcome is
        defined as the difference between expected KPI when treatment variables
        execution is scaled by `scaling_factor1` and `scaling_factor0` during
        these specified time periods. By default, the difference is between
        treatment variables at historical execution levels, or as provided in
        `new_data`, versus zero execution. Defaults to include all time periods.
      aggregate_geos: Boolean. If `True`, then incremental outcome is summed
        over all regions.
      aggregate_times: Boolean. If `True`, then incremental outcome is summed
        over all time periods.
      inverse_transform_outcome: Boolean. If `True`, returns the expected
        outcome in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        outcome after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: Boolean. If `use_kpi = True`, the expected KPI is calculated;
        otherwise the expected revenue `(kpi * revenue_per_kpi)` is calculated.
        It is required that `use_kpi = True` if `revenue_per_kpi` data is not
        available or if `inverse_transform_outcome = False`.
      by_reach: Boolean. If `True`, then the incremental outcome is calculated
        by scaling the reach and holding the frequency constant. If `False`,
        then the incremental outcome is calculated by scaling the frequency and
        holding the reach constant. Only used for channels with RF data.
      include_non_paid_channels: Boolean. If `True`, then non-media treatments
        and organic effects are included in the calculation. If `False`, then
        only the paid media and RF effects are included.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of incremental outcome (either KPI or revenue, depending on
      `use_kpi` argument) with dimensions `(n_chains, n_draws, n_geos,
      n_times, n_channels)`. If `include_non_paid_channels=True`, then
      `n_channel` is the total number of media, RF, organic media, and organic
      RF and non-media channels. If `include_non_paid_channels=False`, then
      `n_channels` is the total number of media and RF channels. The `n_geos`
      and `n_times` dimensions are dropped if `aggregate_geos=True` or
      `aggregate_times=True`, respectively.
    Raises:
      NotFittedModelError: If `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
      ValueError: If `new_data` argument contains tensors with modified time
        dimension and not all treatment variables are provided in `new_data`
        with matching time dimensions.
    """
    mmm = self._meridian
    use_kpi = self._use_kpi(use_kpi)
    self._check_kpi_transformation(inverse_transform_outcome, use_kpi)
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR

    if dist_type not in mmm.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )

    # Validate scaling factor arguments.
    if scaling_factor1 < 0:
      raise ValueError("scaling_factor1 must be non-negative.")
    if scaling_factor0 < 0:
      raise ValueError("scaling_factor0 must be non-negative.")
    if scaling_factor1 <= scaling_factor0:
      raise ValueError(
          "scaling_factor1 must be greater than scaling_factor0. Got"
          f" {scaling_factor1=} and {scaling_factor0=}."
      )

    if new_data is None:
      new_data = DataTensors()

    required_params = constants.PAID_DATA
    if include_non_paid_channels:
      required_params += constants.NON_PAID_DATA
    data_tensors = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_params, meridian=self._meridian
    )
    new_n_media_times = data_tensors.get_modified_times(self._meridian)

    if new_n_media_times is None:
      new_n_media_times = mmm.n_media_times
      _validate_selected_times(
          selected_times=selected_times,
          input_times=mmm.input_data.time,
          n_times=mmm.n_times,
          arg_name="selected_times",
          comparison_arg_name="the input data",
      )
      _validate_selected_times(
          selected_times=media_selected_times,
          input_times=mmm.input_data.media_time,
          n_times=mmm.n_media_times,
          arg_name="media_selected_times",
          comparison_arg_name="the media tensors",
      )
    else:
      _validate_flexible_selected_times(
          selected_times=selected_times,
          media_selected_times=media_selected_times,
          new_n_media_times=new_n_media_times,
      )
    if media_selected_times is None:
      media_selected_times = [True] * new_n_media_times
    else:
      if all(isinstance(time, str) for time in media_selected_times):
        media_selected_times = [
            x in media_selected_times for x in mmm.input_data.media_time
        ]

    # Set counterfactual tensors based on the scaling factors and the media
    # selected times.
    counterfactual0 = (
        1 + (scaling_factor0 - 1) * np.array(media_selected_times)
    )[:, None]
    counterfactual1 = (
        1 + (scaling_factor1 - 1) * np.array(media_selected_times)
    )[:, None]

    if data_tensors.non_media_treatments is not None:
      non_media_treatments_baseline_scaled = (
          self._meridian.compute_non_media_treatments_baseline(
              non_media_baseline_values=non_media_baseline_values,
          )
      )
      non_media_treatments_baseline_normalized = self._meridian.non_media_transformer.forward(  # pytype: disable=attribute-error
          non_media_treatments_baseline_scaled,
          apply_population_scaling=False,
      )
      non_media_treatments0 = backend.broadcast_to(
          backend.to_tensor(
              non_media_treatments_baseline_normalized, dtype=backend.float32
          )[backend.newaxis, backend.newaxis, :],
          data_tensors.non_media_treatments.shape,  # pytype: disable=attribute-error
      )
    else:
      non_media_treatments_baseline_normalized = None
      non_media_treatments0 = None

    incremented_data0 = _scale_tensors_by_multiplier(
        data=data_tensors,
        multiplier=counterfactual0,
        by_reach=by_reach,
    )
    incremented_data1 = _scale_tensors_by_multiplier(
        data=data_tensors, multiplier=counterfactual1, by_reach=by_reach
    )

    scaled_data0 = self._get_scaled_data_tensors(
        new_data=incremented_data0,
        include_non_paid_channels=include_non_paid_channels,
    )
    # TODO: b/415198977 - Verify the computation of outcome of non-media
    # treatments with `media_selected_times` and scale factors.

    data_tensors0 = DataTensors(
        media=scaled_data0.media,
        reach=scaled_data0.reach,
        frequency=scaled_data0.frequency,
        organic_media=scaled_data0.organic_media,
        organic_reach=scaled_data0.organic_reach,
        organic_frequency=scaled_data0.organic_frequency,
        revenue_per_kpi=scaled_data0.revenue_per_kpi,
        non_media_treatments=non_media_treatments0,
    )

    data_tensors1 = self._get_scaled_data_tensors(
        new_data=incremented_data1,
        include_non_paid_channels=include_non_paid_channels,
    )

    # Calculate incremental outcome in batches.
    params = (
        self._meridian.inference_data.posterior
        if use_posterior
        else self._meridian.inference_data.prior
    )
    n_draws = params.draw.size
    batch_starting_indices = np.arange(n_draws, step=batch_size)
    param_list = self._get_causal_param_names(
        include_non_paid_channels=include_non_paid_channels
    )
    incremental_outcome_temps = [None] * len(batch_starting_indices)
    dim_kwargs = {
        "selected_geos": (
            tuple(selected_geos) if selected_geos is not None else None
        ),
        "selected_times": (
            tuple(selected_times) if selected_times is not None else None
        ),
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    incremental_outcome_kwargs = {
        "inverse_transform_outcome": inverse_transform_outcome,
        "use_kpi": use_kpi,
        "non_media_treatments_baseline_normalized": (
            non_media_treatments_baseline_normalized
        ),
    }
    for i, start_index in enumerate(batch_starting_indices):
      stop_index = np.min([n_draws, start_index + batch_size])
      batch_dists = {
          k: backend.to_tensor(params[k][:, start_index:stop_index, ...])
          for k in param_list
      }
      dist_tensors = DistributionTensors(**batch_dists)
      incremental_outcome_temps[i] = self._incremental_outcome_impl(
          data_tensors=data_tensors1,
          dist_tensors=dist_tensors,
          **dim_kwargs,
          **incremental_outcome_kwargs,
      )
      # Calculate incremental outcome under counterfactual scenario "Media_0".
      if scaling_factor0 != 0 or not all(media_selected_times):
        incremental_outcome_temps[i] -= self._incremental_outcome_impl(
            data_tensors=data_tensors0,
            dist_tensors=dist_tensors,
            **dim_kwargs,
            **incremental_outcome_kwargs,
        )
    return backend.concatenate(incremental_outcome_temps, axis=1)

  def _validate_geo_and_time_granularity(
      self,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
  ):
    """Validates the geo and time granularity arguments for ROI analysis.

    Args:
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional. Contains a subset of times to include or
        booleans with length `input_data.n_times`. By default, all time periods
        are included.
      aggregate_geos: If `True`, then expected revenue is summed over all
        regions.

    Raises:
      ValueError: If the geo or time granularity arguments are not valid for the
        ROI analysis.
    """
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    if selected_geos is not None or not aggregate_geos:
      if (
          self._meridian.media_tensors.media_spend is not None
          and not self._meridian.input_data.media_spend_has_geo_dimension
      ):
        raise ValueError(
            "`selected_geos` and `aggregate_geos=False` are not allowed because"
            " Meridian `media_spend` data does not have a geo dimension."
        )
      if (
          self._meridian.rf_tensors.rf_spend is not None
          and not self._meridian.input_data.rf_spend_has_geo_dimension
      ):
        raise ValueError(
            "`selected_geos` and `aggregate_geos=False` are not allowed because"
            " Meridian `rf_spend` data does not have a geo dimension."
        )

    if selected_times is not None:
      if (
          self._meridian.media_tensors.media_spend is not None
          and not self._meridian.input_data.media_spend_has_time_dimension
      ):
        raise ValueError(
            "`selected_times` is not allowed because Meridian `media_spend`"
            " data does not have a time dimension."
        )
      if (
          self._meridian.rf_tensors.rf_spend is not None
          and not self._meridian.input_data.rf_spend_has_time_dimension
      ):
        raise ValueError(
            "`selected_times` is not allowed because Meridian `rf_spend` data"
            " does not have a time dimension."
        )

  def marginal_roi(
      self,
      incremental_increase: float = 0.01,
      use_posterior: bool = True,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      by_reach: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates the marginal ROI prior or posterior distribution.

    The marginal ROI (mROI) numerator is the change in expected outcome (`kpi`
    or `kpi * revenue_per_kpi`) when one channel's spend is increased by a small
    fraction. The mROI denominator is the corresponding small fraction of the
    channel's total spend.

    If `new_data=None`, this method calculates marginal ROI conditional on the
    values of the paid media variables that the Meridian object was initialized
    with. The user can also override this historical data through the `new_data`
    argument. For example,

    ```python
    new_data = DataTensors(media=new_media, frequency=new_frequency)
    ```

    If `selected_geos` or `selected_times` is specified, then the mROI
    denominator is based on the total spend during the selected geos and time
    periods. An exception will be thrown if the spend of the InputData used to
    train the model does not have geo and time dimensions. (If the
    `new_data.media_spend` and `new_data.rf_spend` arguments are used with
    different dimensions than the InputData spend, then an exception will be
    thrown since this is a likely user error.)

    Args:
      incremental_increase: Small fraction by which each channel's spend is
        increased when calculating its mROI numerator. The mROI denominator is
        this fraction of the channel's total spend. Only used if marginal is
        `True`.
      use_posterior: If `True` then the posterior distribution is calculated.
        Otherwise, the prior distribution is calculated.
      new_data: Optional. DataTensors containing `media`, `media_spend`,
        `reach`, `frequency`, `rf_spend` and `revenue_per_kpi` data. If
        provided, the marginal ROI is calculated using the values of the tensors
        passed in `new_data` and the original values of all the remaining
        tensors. If `None`, the marginal ROI is calculated using the original
        values of all the tensors. If any of the tensors in `new_data` is
        provided with a different number of time periods than in `InputData`,
        then all tensors must be provided with the same number of time periods.
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        the `new_data` args, if provided. By default, all time periods are
        included.
      aggregate_geos: If `True`, the expected revenue is summed over all of the
        regions.
      by_reach: Used for a channel with reach and frequency. If `True`, returns
        the mROI by reach for a given fixed frequency. If `False`, returns the
        mROI by frequency for a given fixed reach.
      use_kpi: If `False`, then revenue is used to calculate the mROI numerator.
        Otherwise, uses KPI to calculate the mROI numerator.
      batch_size: Maximum draws per chain in each batch. The calculation is run
        in batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      Tensor of mROI values with dimensions `(n_chains, n_draws, n_geos,
      (n_media_channels + n_rf_channels))`. The `n_geos` dimension is dropped if
      `aggregate_geos=True`.
    """
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
    }
    use_kpi = self._use_kpi(use_kpi)
    self._validate_geo_and_time_granularity(**dim_kwargs)
    required_values = constants.PERFORMANCE_DATA
    if not new_data:
      new_data = DataTensors()
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_values,
        meridian=self._meridian,
    )
    numerator = self.incremental_outcome(
        new_data=filled_data.filter_fields(constants.PAID_DATA),
        scaling_factor0=1,
        scaling_factor1=1 + incremental_increase,
        inverse_transform_outcome=True,
        use_posterior=use_posterior,
        use_kpi=use_kpi,
        by_reach=by_reach,
        batch_size=batch_size,
        include_non_paid_channels=False,
        aggregate_times=True,
        **dim_kwargs,
    )
    spend_inc = filled_data.total_spend() * incremental_increase
    if spend_inc is not None and spend_inc.ndim == 3:
      denominator = self.filter_and_aggregate_geos_and_times(
          spend_inc,
          aggregate_times=True,
          flexible_time_dim=True,
          has_media_dim=True,
          **dim_kwargs,
      )
    else:
      if not aggregate_geos:
        # This check should not be reachable. It is here to protect against
        # future changes to self._validate_geo_and_time_granularity. If
        # spend_inc.ndim is not 3 and `aggregate_geos` is `False`, then
        # self._validate_geo_and_time_granularity should raise an error.
        raise ValueError(
            "aggregate_geos must be True if spend does not have a geo "
            "dimension."
        )
      denominator = spend_inc
    return backend.divide_no_nan(numerator, denominator)

  def roi(
      self,
      use_posterior: bool = True,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates ROI prior or posterior distribution for each media channel.

    The ROI numerator is the change in expected outcome (`kpi` or `kpi *
    revenue_per_kpi`) when one channel's spend is set to zero, leaving all other
    channels' spend unchanged. The ROI denominator is the total spend of the
    channel.

    If `new_data=None`, this method calculates ROI conditional on the values of
    the paid media variables that the Meridian object was initialized with. The
    user can also override this historical data through the `new_data` argument.
    For example,

    ```python
    new_data = DataTensors(media=new_media, frequency=new_frequency)
    ```

    If `selected_geos` or `selected_times` is specified, then the ROI
    denominator is the total spend during the selected geos and time periods. An
    exception will be thrown if the spend of the InputData used to train the
    model does not have geo and time dimensions. (If the `new_data.media_spend`
    and `new_data.rf_spend` arguments are used with different dimensions than
    the InputData spend, then an exception will be thrown since this is a likely
    user error.)

    Args:
      use_posterior: Boolean. If `True`, then the posterior distribution is
        calculated. Otherwise, the prior distribution is calculated.
      new_data: Optional. DataTensors containing `media`, `media_spend`,
        `reach`, `frequency`, and `rf_spend`, and `revenue_per_kpi` data. If
        provided, the ROI is calculated using the values of the tensors passed
        in `new_data` and the original values of all the remaining tensors. If
        `None`, the ROI is calculated using the original values of all the
        tensors. If any of the tensors in `new_data` is provided with a
        different number of time periods than in `InputData`, then all tensors
        must be provided with the same number of time periods.
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        the `new_data` args, if provided. By default, all time periods are
        included.
      aggregate_geos: Boolean. If `True`, the expected revenue is summed over
        all of the regions.
      use_kpi: If `False`, then revenue is used to calculate the ROI numerator.
        Otherwise, uses KPI to calculate the ROI numerator.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of ROI values with dimensions `(n_chains, n_draws, n_geos,
      (n_media_channels + n_rf_channels))`. The `n_geos` dimension is dropped if
      `aggregate_geos=True`.
    """
    use_kpi = self._use_kpi(use_kpi)
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
    }
    incremental_outcome_kwargs = {
        "inverse_transform_outcome": True,
        "use_posterior": use_posterior,
        "use_kpi": use_kpi,
        "batch_size": batch_size,
        "include_non_paid_channels": False,
        "aggregate_times": True,
    }
    self._validate_geo_and_time_granularity(**dim_kwargs)
    required_values = constants.PERFORMANCE_DATA
    if not new_data:
      new_data = DataTensors()
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_values,
        meridian=self._meridian,
    )
    incremental_outcome = self.incremental_outcome(
        new_data=filled_data.filter_fields(constants.PAID_DATA),
        **incremental_outcome_kwargs,
        **dim_kwargs,
    )

    spend = filled_data.total_spend()
    if spend is not None and spend.ndim == 3:
      denominator = self.filter_and_aggregate_geos_and_times(
          spend,
          aggregate_times=True,
          flexible_time_dim=True,
          has_media_dim=True,
          **dim_kwargs,
      )
    else:
      if not aggregate_geos:
        # This check should not be reachable. It is here to protect against
        # future changes to self._validate_geo_and_time_granularity. If
        # spend_inc.ndim is not 3 and either of `aggregate_geos` or
        # `aggregate_times` is `False`, then
        # self._validate_geo_and_time_granularity should raise an error.
        raise ValueError(
            "aggregate_geos must be True if spend does not have a geo "
            "dimension."
        )
      denominator = spend
    return backend.divide_no_nan(incremental_outcome, denominator)

  def cpik(
      self,
      use_posterior: bool = True,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> backend.Tensor:
    """Calculates the cost per incremental KPI distribution for each channel.

    The CPIK numerator is the total spend on the channel. The CPIK denominator
    is the change in expected KPI when one channel's spend is set to zero,
    leaving all other channels' spend unchanged.

    If `new_data=None`, this method calculates CPIK conditional on the values of
    the paid media variables that the Meridian object was initialized with. The
    user can also override this historical data through the `new_data` argument.
    For example,

    ```python
    new_data = DataTensors(media=new_media, frequency=new_frequency)
    ```

    If `selected_geos` or `selected_times` is specified, then the CPIK
    numerator is the total spend during the selected geos and time periods. An
    exception will be thrown if the spend of the InputData used to train the
    model does not have geo and time dimensions. (If the `new_data.media_spend`
    and `new_data.rf_spend` arguments are used with different dimensions than
    the InputData spend, then an exception will be thrown since this is a likely
    user error.)

    Note that CPIK is simply 1/ROI, where ROI is obtained from a call to the
    `roi` method with `use_kpi=True`.

    Args:
      use_posterior: Boolean. If `True` then the posterior distribution is
        calculated. Otherwise, the prior distribution is calculated.
      new_data: Optional. DataTensors containing `media`, `media_spend`,
        `reach`, `frequency`, `rf_spend` and `revenue_per_kpi` data. If
        provided, the cpik is calculated using the values of the tensors passed
        in `new_data` and the original values of all the remaining tensors. If
        `None`, the ROI is calculated using the original values of all the
        tensors. If any of the tensors in `new_data` is provided with a
        different number of time periods than in `InputData`, then all tensors
        must be provided with the same number of time periods.
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        the `new_data` args, if provided. By default, all time periods are
        included.
      aggregate_geos: Boolean. If `True`, the expected KPI is summed over all of
        the regions.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of CPIK values with dimensions `(n_chains, n_draws, n_geos,
      (n_media_channels + n_rf_channels))`. The `n_geos` dimension is dropped if
      `aggregate_geos=True`.
    """
    roi = self.roi(
        use_kpi=True,
        use_posterior=use_posterior,
        new_data=new_data,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        batch_size=batch_size,
    )
    return backend.divide_no_nan(1.0, roi)

  def _mean_and_ci_by_eval_set(
      self,
      draws: backend.Tensor,
      split_by_holdout: bool,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> np.ndarray:
    """Calculates the mean and CI of `draws`, split by `holdout_id` if needed.

    Args:
      draws: A tensor of a set of draws with dimensions `(n_chains, n_draws,
        n_geos, n_times)`.
      split_by_holdout: Boolean. If `True` and `holdout_id` exists, the data is
        split into `'Train'`, `'Test'`, and `'All Data'` subsections.
      aggregate_geos: If `True`, the draws tensor is summed over all regions.
      aggregate_times: If `True`, the draws tensor is summed over all times.
      confidence_level: Confidence level for computing credible intervals,
        represented as a value between zero and one.

    Returns:
      The mean and CI of the draws with dimensions that could be
       * `(n_geos, n_times, n_metrics, n_evaluation_sets)` if
       `split_by_holdout=True`, and no aggregations.
       * `(n_geos, n_times, n_metrics)` if `split_by_holdout=False`, and no
       aggregations.
       * `(n_metrics, n_evaluation_sets)` if `split_by_holdout=True`, and
        `aggregate_geos=True` or `aggregate_times=True`.
       * `(n_metrics)` if `split_by_holdout=False`, and `aggregate_geos=True` or
        `aggregate_times=True`.
    """

    if not split_by_holdout:
      draws = self.filter_and_aggregate_geos_and_times(
          draws, aggregate_geos=aggregate_geos, aggregate_times=aggregate_times
      )
      return get_central_tendency_and_ci(
          draws, confidence_level=confidence_level
      )

    train_draws = np.where(self._meridian.model_spec.holdout_id, np.nan, draws)
    test_draws = np.where(self._meridian.model_spec.holdout_id, draws, np.nan)
    draws_by_evaluation_set = np.stack(
        [train_draws, test_draws, draws], axis=0
    )  # shape (n_evaluation_sets(=3), n_chains, n_draws, n_geos, n_times)
    draws_by_evaluation_set = self.filter_and_aggregate_geos_and_times(
        draws_by_evaluation_set,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
    )  # shape (n_evaluation_sets(=3), n_chains, n_draws, ...)

    # The shape of the output from `get_central_tendency_and_ci` is,
    # for example, (n_evaluation_sets(=3), n_geos, n_times, n_metrics(=3)) if no
    # aggregations. To get the shape of (n_geos, n_times, n_metrics,
    # n_evaluation_sets), we need to transpose the output.
    mean_and_ci = get_central_tendency_and_ci(
        draws_by_evaluation_set, confidence_level=confidence_level, axis=(1, 2)
    )
    return mean_and_ci.transpose(list(range(1, mean_and_ci.ndim)) + [0])

  def _can_split_by_holdout_id(self, split_by_holdout_id: bool) -> bool:
    """Returns whether the data can be split by holdout_id."""
    if split_by_holdout_id and self._meridian.model_spec.holdout_id is None:
      warnings.warn(
          "`split_by_holdout_id` is True but `holdout_id` is `None`. Data will"
          " not be split."
      )
    return (
        split_by_holdout_id and self._meridian.model_spec.holdout_id is not None
    )

  def expected_vs_actual_data(
      self,
      aggregate_geos: bool = False,
      aggregate_times: bool = False,
      use_kpi: bool = False,
      split_by_holdout_id: bool = False,
      non_media_baseline_values: Sequence[float] | None = None,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> xr.Dataset:
    """Calculates the data for the expected versus actual outcome over time.

    Args:
      aggregate_geos: Boolean. If `True`, the expected, baseline, and actual are
        summed over all of the regions.
      aggregate_times: Boolean. If `True`, the expected, baseline, and actual
        are summed over all of the time periods.
      use_kpi: If `True`, calculate the incremental KPI. Otherwise, calculate
        the incremental revenue using the revenue per KPI (if available).
      split_by_holdout_id: Boolean. If `True` and `holdout_id` exists, the data
        is split into `'Train'`, `'Test'`, and `'All Data'` subsections.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      confidence_level: Confidence level for expected outcome credible
        intervals, represented as a value between zero and one. Default: `0.9`.

    Returns:
      A dataset with the expected, baseline, and actual outcome metrics.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)
    use_kpi = self._use_kpi(use_kpi)
    mmm = self._meridian
    can_split_by_holdout = self._can_split_by_holdout_id(split_by_holdout_id)
    expected_outcome = self.expected_outcome(
        aggregate_geos=False, aggregate_times=False, use_kpi=use_kpi
    )

    expected = self._mean_and_ci_by_eval_set(
        expected_outcome,
        can_split_by_holdout,
        aggregate_geos,
        aggregate_times,
        confidence_level,
    )

    baseline_expected_outcome = self._calculate_baseline_expected_outcome(
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=use_kpi,
        non_media_baseline_values=non_media_baseline_values,
    )
    baseline = self._mean_and_ci_by_eval_set(
        baseline_expected_outcome,
        can_split_by_holdout,
        aggregate_geos,
        aggregate_times,
        confidence_level,
    )
    actual = np.asarray(
        self.filter_and_aggregate_geos_and_times(
            mmm.kpi if use_kpi else mmm.kpi * mmm.revenue_per_kpi,
            aggregate_geos=aggregate_geos,
            aggregate_times=aggregate_times,
        )
    )

    # Set up the coordinates.
    coords = {
        constants.METRIC: [constants.MEAN, constants.CI_LO, constants.CI_HI],
    }

    if not aggregate_geos:
      coords[constants.GEO] = mmm.input_data.geo.data
    if not aggregate_times:
      coords[constants.TIME] = mmm.input_data.time.data
    if can_split_by_holdout:
      coords[constants.EVALUATION_SET_VAR] = list(constants.EVALUATION_SET)

    # Set up the dimensions.
    actual_dims = ((constants.GEO,) if not aggregate_geos else ()) + (
        (constants.TIME,) if not aggregate_times else ()
    )
    expected_and_baseline_dims = (
        actual_dims
        + (constants.METRIC,)
        + ((constants.EVALUATION_SET_VAR,) if can_split_by_holdout else ())
    )

    data_vars = {
        constants.EXPECTED: (expected_and_baseline_dims, expected),
        constants.BASELINE: (expected_and_baseline_dims, baseline),
        constants.ACTUAL: (actual_dims, actual),
    }
    attrs = {constants.CONFIDENCE_LEVEL: confidence_level}

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

  def _calculate_baseline_expected_outcome(
      self,
      non_media_baseline_values: Sequence[float] | None = None,
      **expected_outcome_kwargs,
  ) -> backend.Tensor:
    """Calculates either the posterior or prior expected outcome of baseline.

    This is a wrapper for expected_outcome() that automatically sets the
    following argument values:
      1) `new_media` is set to all zeros
      2) `new_reach` is set to all zeros
      3) `new_organic_media` is set to all zeros
      4) `new_organic_reach` is set to all zeros
      5) `new_non_media_treatments` is set to the counterfactual values
      according to the `non_media_baseline_values` argument
      6) `new_controls` are set to historical values

    All other arguments of `expected_outcome` can be passed to this method.

    Args:
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float denoting a fixed
        value that will be used as the baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      **expected_outcome_kwargs: kwargs to pass to `expected_outcome`, which
        could contain use_posterior, selected_geos, selected_times,
        aggregate_geos, aggregate_times, inverse_transform_outcome, use_kpi,
        batch_size.

    Returns:
      Tensor of expected outcome of baseline with dimensions `(n_chains,
      n_draws, n_geos, n_times)`. The `n_geos` and `n_times` dimensions is
      dropped if `aggregate_geos=True` or `aggregate_time=True`, respectively.
    """
    new_media = (
        backend.zeros_like(self._meridian.media_tensors.media)
        if self._meridian.media_tensors.media is not None
        else None
    )
    # Frequency is not needed because the reach is zero.
    new_reach = (
        backend.zeros_like(self._meridian.rf_tensors.reach)
        if self._meridian.rf_tensors.reach is not None
        else None
    )
    new_organic_media = (
        backend.zeros_like(self._meridian.organic_media_tensors.organic_media)
        if self._meridian.organic_media_tensors.organic_media is not None
        else None
    )
    new_organic_reach = (
        backend.zeros_like(self._meridian.organic_rf_tensors.organic_reach)
        if self._meridian.organic_rf_tensors.organic_reach is not None
        else None
    )
    if self._meridian.non_media_treatments is not None:
      if self._meridian.model_spec.non_media_population_scaling_id is not None:
        scaling_factors = backend.where(
            self._meridian.model_spec.non_media_population_scaling_id,
            self._meridian.population[:, backend.newaxis, backend.newaxis],
            backend.ones_like(self._meridian.population)[
                :, backend.newaxis, backend.newaxis
            ],
        )
      else:
        scaling_factors = backend.ones_like(self._meridian.population)[
            :, backend.newaxis, backend.newaxis
        ]

      baseline = self._meridian.compute_non_media_treatments_baseline(
          non_media_baseline_values=non_media_baseline_values,
      )
      new_non_media_treatments_population_scaled = backend.broadcast_to(
          backend.to_tensor(baseline, dtype=backend.float32)[
              backend.newaxis, backend.newaxis, :
          ],
          self._meridian.non_media_treatments.shape,
      )
      new_non_media_treatments = (
          new_non_media_treatments_population_scaled * scaling_factors
      )
    else:
      new_non_media_treatments = None
    new_controls = self._meridian.controls

    new_data = DataTensors(
        media=new_media,
        reach=new_reach,
        organic_media=new_organic_media,
        organic_reach=new_organic_reach,
        non_media_treatments=new_non_media_treatments,
        controls=new_controls,
    )
    return self.expected_outcome(new_data=new_data, **expected_outcome_kwargs)

  def compute_incremental_outcome_aggregate(
      self,
      use_posterior: bool,
      new_data: DataTensors | None = None,
      use_kpi: bool = False,
      include_non_paid_channels: bool = True,
      non_media_baseline_values: Sequence[float] | None = None,
      **kwargs,
  ) -> backend.Tensor:
    """Aggregates the incremental outcome of the media channels.

    Args:
      use_posterior: Boolean. If `True`, then the incremental outcome posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      new_data: Optional `DataTensors` container with optional tensors: `media`,
        `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
        `None`, the incremental outcome is calculated using the `InputData`
        provided to the Meridian object. If `new_data` is provided, the
        incremental outcome is calculated using the new tensors in `new_data`
        and the original values of the remaining tensors. For example,
        `compute_incremental_outcome_aggregate(new_data=DataTensors(media=new_media))`
        computes the incremental outcome using `new_media` and the original
        values of `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
        any of the tensors in `new_data` is provided with a different number of
        time periods than in `InputData`, then all tensors must be provided with
        the same number of time periods.
      use_kpi: Boolean. If `True`, the summary metrics are calculated using KPI.
        If `False`, the metrics are calculated using revenue.
      include_non_paid_channels: Boolean. If `True`, then non-media treatments
        and organic effects are included in the calculation. If `False`, then
        only the paid media and RF effects are included.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      **kwargs: kwargs to pass to `incremental_outcome`, which could contain
        selected_geos, selected_times, aggregate_geos, aggregate_times,
        batch_size.

    Returns:
      A Tensor with the same dimensions as `incremental_outcome` except the size
      of the channel dimension is incremented by one, with the new component at
      the end containing the total incremental outcome of all channels.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)
    use_kpi = self._use_kpi(use_kpi)
    incremental_outcome_m = self.incremental_outcome(
        use_posterior=use_posterior,
        new_data=new_data,
        use_kpi=use_kpi,
        include_non_paid_channels=include_non_paid_channels,
        non_media_baseline_values=non_media_baseline_values,
        **kwargs,
    )
    incremental_outcome_total = backend.reduce_sum(
        incremental_outcome_m, axis=-1, keepdims=True
    )

    return backend.concatenate(
        [incremental_outcome_m, incremental_outcome_total],
        axis=-1,
    )

  def summary_metrics(
      self,
      new_data: DataTensors | None = None,
      marginal_roi_by_reach: bool = True,
      marginal_roi_incremental_increase: float = 0.01,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      optimal_frequency: Sequence[float] | None = None,
      use_kpi: bool = False,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
      include_non_paid_channels: bool = False,
      non_media_baseline_values: Sequence[float] | None = None,
  ) -> xr.Dataset:
    """Returns summary metrics.

    If `new_data=None`, this method calculates all the metrics conditional on
    the values of the data variables that the Meridian object was initialized
    with. The user can also override this historical data through the `new_data`
    argument. For example, to override the media, frequency, and non-media
    treatments data variables, the user can pass the following `new_data`
    argument:

    ```python
    new_data = DataTensors(
        media=new_media,
        frequency=new_frequency,
        non_media_treatments=new_non_media_treatments)
    ```

    Note that if `new_data` is provided with a different number of time periods
    than in `InputData`, `pct_of_contribution` is not defined because
    `expected_outcome()` is not defined for new time periods.

    Note that `mroi` and `effectiveness` metrics are not defined (`math.nan`)
    for the aggregate `"All Paid Channels"` channel dimension.

    Args:
      new_data: Optional `DataTensors` object with optional new tensors:
        `media`, `media_spend`, `reach`, `frequency`, `rf_spend`,
        `organic_media`, `organic_reach`, `organic_frequency`,
        `non_media_treatments`, `controls`, `revenue_per_kpi`. If provided, the
        summary metrics are calculated using the values of the tensors passed in
        `new_data` and the original values of all the remaining tensors. If
        `None`, the summary metrics are calculated using the original values of
        all the tensors. If `new_data` is provided with a different number of
        time periods than in `InputData`, then all tensors, except `controls`,
        must have the same number of time periods.
      marginal_roi_by_reach: Boolean. Marginal ROI (mROI) is defined as the
        return on the next dollar spent. If this argument is `True`, the
        assumption is that the next dollar spent only impacts reach, holding
        frequency constant. If this argument is `False`, the assumption is that
        the next dollar spent only impacts frequency, holding reach constant.
        Used only when `include_non_paid_channels` is `False`.
      marginal_roi_incremental_increase: Small fraction by which each channel's
        spend is increased when calculating its mROI numerator. The mROI
        denominator is this fraction of the channel's total spend. Used only
        when `include_non_paid_channels` is `False`.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        the tensors in the `new_data` argument, if provided. By default, all
        time periods are included.
      aggregate_geos: Boolean. If `True`, the expected outcome is summed over
        all of the regions.
      aggregate_times: Boolean. If `True`, the expected outcome is summed over
        all of the time periods. Note that if `False`, ROI, mROI, Effectiveness,
        and CPIK are not reported because they do not have a clear
        interpretation by time period.
      optimal_frequency: An optional list with dimension `n_rf_channels`,
        containing the optimal frequency per channel, that maximizes posterior
        mean ROI. Default value is `None`, and historical frequency is used for
        the metrics calculation.
      use_kpi: Boolean. If `True`, the summary metrics are calculated using KPI.
        If `False`, the metrics are calculated using revenue.
      confidence_level: Confidence level for summary metrics credible intervals,
        represented as a value between zero and one.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.
      include_non_paid_channels: Boolean. If `True`, non-paid channels (organic
        media, organic reach and frequency, and non-media treatments) are
        included in the summary but only the metrics independent of spend are
        reported. If `False`, only the paid channels (media, reach and
        frequency) are included but the summary contains also the metrics
        dependent on spend. Default: `False`.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.

    Returns:
      An `xr.Dataset` with coordinates: `channel`, `metric` (`mean`, `median`,
      `ci_low`, `ci_high`), `distribution` (prior, posterior) and contains the
      following non-paid data variables: `incremental_outcome`,
      `pct_of_contribution`, `effectiveness`, and the following paid
      data variables: `impressions`, `pct_of_impressions`, `spend`,
      `pct_of_spend`, `CPM`, `roi`, `mroi`, `cpik`. The paid data variables are
      only included when `include_non_paid_channels` is `False`. Note that
      `roi`, `mroi`, `cpik`, and `effectiveness` metrics are not reported
      when `aggregate_times=False` because they do not have a clear
      interpretation by time period.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)
    use_kpi = self._use_kpi(use_kpi)
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    batched_kwargs = {"batch_size": batch_size}
    new_data = new_data or DataTensors()
    aggregated_impressions = self.get_aggregated_impressions(
        new_data=new_data.filter_fields(constants.IMPRESSIONS_DATA),
        optimal_frequency=optimal_frequency,
        include_non_paid_channels=include_non_paid_channels,
        **dim_kwargs,
    )
    impressions_with_total = backend.concatenate(
        [
            aggregated_impressions,
            backend.reduce_sum(aggregated_impressions, -1, keepdims=True),
        ],
        axis=-1,
    )

    incremental_outcome_fields = list(
        constants.PAID_DATA + constants.NON_PAID_DATA
    )
    incremental_outcome_prior = self.compute_incremental_outcome_aggregate(
        use_posterior=False,
        new_data=new_data.filter_fields(incremental_outcome_fields),
        use_kpi=use_kpi,
        include_non_paid_channels=include_non_paid_channels,
        non_media_baseline_values=non_media_baseline_values,
        **dim_kwargs,
        **batched_kwargs,
    )
    incremental_outcome_posterior = self.compute_incremental_outcome_aggregate(
        use_posterior=True,
        new_data=new_data.filter_fields(incremental_outcome_fields),
        use_kpi=use_kpi,
        include_non_paid_channels=include_non_paid_channels,
        non_media_baseline_values=non_media_baseline_values,
        **dim_kwargs,
        **batched_kwargs,
    )
    incremental_outcome_mroi_prior = self.compute_incremental_outcome_aggregate(
        use_posterior=False,
        new_data=new_data.filter_fields(incremental_outcome_fields),
        use_kpi=use_kpi,
        by_reach=marginal_roi_by_reach,
        scaling_factor0=1,
        scaling_factor1=1 + marginal_roi_incremental_increase,
        include_non_paid_channels=include_non_paid_channels,
        non_media_baseline_values=non_media_baseline_values,
        **dim_kwargs,
        **batched_kwargs,
    )
    incremental_outcome_mroi_posterior = (
        self.compute_incremental_outcome_aggregate(
            use_posterior=True,
            new_data=new_data.filter_fields(incremental_outcome_fields),
            use_kpi=use_kpi,
            by_reach=marginal_roi_by_reach,
            scaling_factor0=1,
            scaling_factor1=1 + marginal_roi_incremental_increase,
            include_non_paid_channels=include_non_paid_channels,
            non_media_baseline_values=non_media_baseline_values,
            **dim_kwargs,
            **batched_kwargs,
        )
    )

    xr_dims = (
        ((constants.GEO,) if not aggregate_geos else ())
        + ((constants.TIME,) if not aggregate_times else ())
        + (constants.CHANNEL,)
    )
    channels = (
        self._meridian.input_data.get_all_channels()
        if include_non_paid_channels
        else self._meridian.input_data.get_all_paid_channels()
    )
    xr_coords = {constants.CHANNEL: list(channels) + [constants.ALL_CHANNELS]}
    if not aggregate_geos:
      geo_dims = (
          self._meridian.input_data.geo.data
          if selected_geos is None
          else selected_geos
      )
      xr_coords[constants.GEO] = geo_dims
    if not aggregate_times:
      # Get the time coordinates for flexible time dimensions.
      modified_times = new_data.get_modified_times(self._meridian)
      if modified_times is None:
        times = self._meridian.input_data.time.data
      else:
        times = np.arange(modified_times)

      if selected_times is None:
        time_dims = times
      elif _is_bool_list(selected_times):
        indices = np.where(selected_times)
        time_dims = times[indices]
      else:
        time_dims = selected_times
      xr_coords[constants.TIME] = time_dims
    xr_dims_with_ci_and_distribution = xr_dims + (
        constants.METRIC,
        constants.DISTRIBUTION,
    )
    xr_coords_with_ci_and_distribution = {
        constants.METRIC: [
            constants.MEAN,
            constants.MEDIAN,
            constants.CI_LO,
            constants.CI_HI,
        ],
        constants.DISTRIBUTION: [constants.PRIOR, constants.POSTERIOR],
        **xr_coords,
    }
    incremental_outcome = _central_tendency_and_ci_by_prior_and_posterior(
        prior=incremental_outcome_prior,
        posterior=incremental_outcome_posterior,
        metric_name=constants.INCREMENTAL_OUTCOME,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
        include_median=True,
    )
    effectiveness = self._compute_effectiveness_aggregate(
        incremental_outcome_prior=incremental_outcome_prior,
        incremental_outcome_posterior=incremental_outcome_posterior,
        impressions_with_total=impressions_with_total,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
        # Drop effectiveness metric values in the Dataset's data_vars for the
        # aggregated "All Paid Channels" channel dimension value. The
        # "Effectiveness" metric has no meaningful interpretation in this case
        # because the media execution metric is generally not consistent across
        # channels.
    ).where(lambda ds: ds.channel != constants.ALL_CHANNELS)

    if new_data.get_modified_times(self._meridian) is None:
      expected_outcome_fields = list(
          constants.PAID_DATA + constants.NON_PAID_DATA + (constants.CONTROLS,)
      )
      expected_outcome_prior = self.expected_outcome(
          use_posterior=False,
          new_data=new_data.filter_fields(expected_outcome_fields),
          use_kpi=use_kpi,
          **dim_kwargs,
          **batched_kwargs,
      )
      expected_outcome_posterior = self.expected_outcome(
          use_posterior=True,
          new_data=new_data.filter_fields(expected_outcome_fields),
          use_kpi=use_kpi,
          **dim_kwargs,
          **batched_kwargs,
      )
      pct_of_contribution = self._compute_pct_of_contribution(
          incremental_outcome_prior=incremental_outcome_prior,
          incremental_outcome_posterior=incremental_outcome_posterior,
          expected_outcome_prior=expected_outcome_prior,
          expected_outcome_posterior=expected_outcome_posterior,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
      )
    else:
      pct_of_contribution = xr.Dataset()

    if include_non_paid_channels:
      # If non-paid channels are included, return only the non-paid metrics.
      if not aggregate_times:
        # Outcome metrics should not be normalized by weekly media metrics,
        # which do not have a clear interpretation due to lagged effects.
        # Therefore, certain metrics are not reported if aggregate_times=False.
        warnings.warn(
            "Effectiveness is not reported because it does not have a clear"
            " interpretation by time period."
        )
        return xr.merge([
            incremental_outcome,
            pct_of_contribution,
        ])
      else:
        return xr.merge([
            incremental_outcome,
            pct_of_contribution,
            effectiveness,
        ])

    # If non-paid channels are not included, return all metrics, paid and
    # non-paid.
    spend_list = []
    new_spend_tensors = new_data.filter_fields(
        constants.SPEND_DATA
    ).validate_and_fill_missing_data(constants.SPEND_DATA, self._meridian)
    if self._meridian.n_media_channels > 0:
      spend_list.append(new_spend_tensors.media_spend)
    if self._meridian.n_rf_channels > 0:
      spend_list.append(new_spend_tensors.rf_spend)
    # TODO Add support for 1-dimensional spend.
    aggregated_spend = self.filter_and_aggregate_geos_and_times(
        tensor=backend.concatenate(spend_list, axis=-1),
        flexible_time_dim=True,
        **dim_kwargs,
    )
    spend_with_total = backend.concatenate(
        [
            aggregated_spend,
            backend.reduce_sum(aggregated_spend, -1, keepdims=True),
        ],
        axis=-1,
    )
    spend_data = self._compute_spend_data_aggregate(
        spend_with_total=spend_with_total,
        impressions_with_total=impressions_with_total,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
    )

    if not aggregate_times:
      # Outcome metrics should not be normalized by weekly media metrics, which
      # do not have a clear interpretation due to lagged effects. Therefore, NaN
      # values are returned for certain metrics if aggregate_times=False.
      warnings.warn(
          "ROI, mROI, Effectiveness, and CPIK are not reported because they "
          "do not have a clear interpretation by time period."
      )
      return xr.merge([
          spend_data,
          incremental_outcome,
          pct_of_contribution,
      ])
    else:
      roi = self._compute_roi_aggregate(
          incremental_outcome_prior=incremental_outcome_prior,
          incremental_outcome_posterior=incremental_outcome_posterior,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
          spend_with_total=spend_with_total,
      )
      mroi = self._compute_roi_aggregate(
          incremental_outcome_prior=incremental_outcome_mroi_prior,
          incremental_outcome_posterior=incremental_outcome_mroi_posterior,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
          spend_with_total=spend_with_total * marginal_roi_incremental_increase,
          metric_name=constants.MROI,
          # Drop mROI metric values in the Dataset's data_vars for the
          # aggregated "All Paid Channels" channel dimension value.
          # "Marginal ROI" calculation must arbitrarily assume how the
          # "next dollar" of spend is allocated across "All Paid Channels" in
          # this case, which may cause confusion in Meridian model and does not
          # have much practical usefulness, anyway.
      ).where(lambda ds: ds.channel != constants.ALL_CHANNELS)
      cpik = self._compute_cpik_aggregate(
          incremental_kpi_prior=self.compute_incremental_outcome_aggregate(
              use_posterior=False,
              new_data=new_data.filter_fields(incremental_outcome_fields),
              use_kpi=True,
              include_non_paid_channels=False,
              **dim_kwargs,
              **batched_kwargs,
          ),
          incremental_kpi_posterior=self.compute_incremental_outcome_aggregate(
              use_posterior=True,
              new_data=new_data.filter_fields(incremental_outcome_fields),
              use_kpi=True,
              include_non_paid_channels=False,
              **dim_kwargs,
              **batched_kwargs,
          ),
          spend_with_total=spend_with_total,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
      )
      return xr.merge([
          spend_data,
          incremental_outcome,
          pct_of_contribution,
          roi,
          effectiveness,
          mroi,
          cpik,
      ])

  def get_aggregated_impressions(
      self,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      optimal_frequency: Sequence[float] | None = None,
      include_non_paid_channels: bool = True,
  ) -> backend.Tensor:
    """Computes aggregated impressions values in the data across all channels.

    Args:
      new_data: An optional `DataTensors` object containing the new `media`,
        `reach`, `frequency`, `organic_media`, `organic_reach`,
        `organic_frequency`, and `non_media_treatments` tensors. If `new_data`
        argument is used, then the aggregated impressions are computed using the
        values of the tensors passed in the `new_data` argument and the original
        values of all the remaining tensors. If `None`, the existing tensors
        from the Meridian object are used.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        the tensors in the `new_data` argument, if provided. By default, all
        time periods are included.
      aggregate_geos: Boolean. If `True`, the expected outcome is summed over
        all of the regions.
      aggregate_times: Boolean. If `True`, the expected outcome is summed over
        all of the time periods.
      optimal_frequency: An optional list with dimension `n_rf_channels`,
        containing the optimal frequency per channel, that maximizes posterior
        mean ROI. Default value is `None`, and historical frequency is used for
        the metrics calculation.
      include_non_paid_channels: Boolean. If `True`, the organic media, organic
        RF, and non-media channels are included in the aggregation.

    Returns:
      A tensor with the shape `(n_selected_geos, n_selected_times, n_channels)`
      (or `(n_channels,)` if geos and times are aggregated) with aggregate
      impression values per channel.
    """
    tensor_names_list = (
        constants.MEDIA,
        constants.REACH,
        constants.FREQUENCY,
    )
    if include_non_paid_channels:
      tensor_names_list += constants.NON_PAID_DATA
    if new_data is None:
      new_data = DataTensors()
    data_tensors = new_data.validate_and_fill_missing_data(
        tensor_names_list, self._meridian
    )
    n_times = (
        data_tensors.get_modified_times(self._meridian)
        or self._meridian.n_times
    )
    impressions_list = []
    if self._meridian.n_media_channels > 0:
      impressions_list.append(data_tensors.media[:, -n_times:, :])

    if self._meridian.n_rf_channels > 0:
      if optimal_frequency is None:
        new_frequency = data_tensors.frequency
      else:
        new_frequency = (
            backend.ones_like(data_tensors.frequency) * optimal_frequency
        )
      impressions_list.append(
          data_tensors.reach[:, -n_times:, :] * new_frequency[:, -n_times:, :]
      )

    if include_non_paid_channels:
      if self._meridian.n_organic_media_channels > 0:
        impressions_list.append(data_tensors.organic_media[:, -n_times:, :])
      if self._meridian.n_organic_rf_channels > 0:
        if optimal_frequency is None:
          new_organic_frequency = data_tensors.organic_frequency
        else:
          new_organic_frequency = (
              backend.ones_like(data_tensors.organic_frequency)
              * optimal_frequency
          )
        impressions_list.append(
            data_tensors.organic_reach[:, -n_times:, :]
            * new_organic_frequency[:, -n_times:, :]
        )
      if self._meridian.n_non_media_channels > 0:
        impressions_list.append(data_tensors.non_media_treatments)

    return self.filter_and_aggregate_geos_and_times(
        tensor=backend.concatenate(impressions_list, axis=-1),
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        flexible_time_dim=True,
    )

  def baseline_summary_metrics(
      self,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      non_media_baseline_values: Sequence[float] | None = None,
      use_kpi: bool = False,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Returns baseline summary metrics.

    Args:
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected outcome is summed over
        all of the regions.
      aggregate_times: Boolean. If `True`, the expected outcome is summed over
        all of the time periods.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      use_kpi: Boolean. If `True`, the baseline summary metrics are calculated
        using KPI. If `False`, the metrics are calculated using revenue.
      confidence_level: Confidence level for media summary metrics credible
        intervals, represented as a value between zero and one.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      An `xr.Dataset` with coordinates: `metric` (`mean`, `median`,
      `ci_low`,`ci_high`),`distribution` (prior, posterior) and contains the
      following data variables: `baseline_outcome`, `pct_of_contribution`.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)
    # TODO: Change "pct_of_contribution" to a more accurate term.

    use_kpi = self._use_kpi(use_kpi)
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    outcome_kwargs = {"batch_size": batch_size, **dim_kwargs}

    xr_dims = (
        ((constants.GEO,) if not aggregate_geos else ())
        + ((constants.TIME,) if not aggregate_times else ())
        + (constants.CHANNEL,)
    )
    xr_coords = {constants.CHANNEL: [constants.BASELINE]}
    if not aggregate_geos:
      geo_dims = (
          self._meridian.input_data.geo.data
          if selected_geos is None
          else selected_geos
      )
      xr_coords[constants.GEO] = geo_dims
    if not aggregate_times:
      time_dims = (
          self._meridian.input_data.time.data
          if selected_times is None
          else selected_times
      )
      xr_coords[constants.TIME] = time_dims
    xr_dims_with_ci_and_distribution = xr_dims + (
        constants.METRIC,
        constants.DISTRIBUTION,
    )
    xr_coords_with_ci_and_distribution = {
        constants.METRIC: [
            constants.MEAN,
            constants.MEDIAN,
            constants.CI_LO,
            constants.CI_HI,
        ],
        constants.DISTRIBUTION: [constants.PRIOR, constants.POSTERIOR],
        **xr_coords,
    }

    expected_outcome_prior = self.expected_outcome(
        use_posterior=False, use_kpi=use_kpi, **outcome_kwargs
    )
    expected_outcome_posterior = self.expected_outcome(
        use_posterior=True, use_kpi=use_kpi, **outcome_kwargs
    )

    baseline_expected_outcome_prior = backend.expand_dims(
        self._calculate_baseline_expected_outcome(
            use_posterior=False,
            use_kpi=use_kpi,
            non_media_baseline_values=non_media_baseline_values,
            **outcome_kwargs,
        ),
        axis=-1,
    )
    baseline_expected_outcome_posterior = backend.expand_dims(
        self._calculate_baseline_expected_outcome(
            use_posterior=True,
            use_kpi=use_kpi,
            non_media_baseline_values=non_media_baseline_values,
            **outcome_kwargs,
        ),
        axis=-1,
    )

    baseline_outcome = _central_tendency_and_ci_by_prior_and_posterior(
        prior=baseline_expected_outcome_prior,
        posterior=baseline_expected_outcome_posterior,
        metric_name=constants.BASELINE_OUTCOME,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
        include_median=True,
    ).sel(channel=constants.BASELINE)

    baseline_pct_of_contribution = self._compute_pct_of_contribution(
        incremental_outcome_prior=baseline_expected_outcome_prior,
        incremental_outcome_posterior=baseline_expected_outcome_posterior,
        expected_outcome_prior=expected_outcome_prior,
        expected_outcome_posterior=expected_outcome_posterior,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
    ).sel(channel=constants.BASELINE)

    return xr.merge([
        baseline_outcome,
        baseline_pct_of_contribution,
    ])

  def optimal_freq(
      self,
      new_data: DataTensors | None = None,
      max_frequency: float | None = None,
      freq_grid: Sequence[float] | None = None,
      use_posterior: bool = True,
      use_kpi: bool = False,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> xr.Dataset:
    """Calculates the optimal frequency that maximizes posterior mean ROI.

    For this optimization, historical spend is used and fixed, and frequency is
    restricted to be constant across all geographic regions and time periods.
    Reach is calculated for each geographic area and time period such that the
    number of impressions remains unchanged as frequency varies. Meridian solves
    for the frequency at which posterior mean ROI is optimized.

    If `new_data=None`, this method calculates the opptimal frequency on the
    values of the paid RF variables that the Meridian object was initialized
    with. The user can override this historical data through the `new_data`
    argument. For example,

    ```python
    new_data = DataTensors(reach=new_reach, frequency=new_frequency)
    ```

    Note: The ROI numerator is revenue if `use_kpi` is `False`, otherwise, the
    ROI numerator is KPI units.

    Args:
      new_data: Optional `DataTensors` object containing `rf_impressions`,
        `rf_spend`, and `revenue_per_kpi`. If provided, the optimal frequency is
        calculated using the values of the tensors passed in `new_data` and the
        original values of all the remaining tensors. If `None`, the historical
        data used to initialize the Meridian object is used. If any of the
        tensors in `new_data` is provided with a different number of time
        periods than in `InputData`, then all tensors must be provided with the
        same number of time periods.
      max_frequency: Maximum frequency value used to calculate the frequency
        grid. If `None`, the maximum frequency value is calculated from the
        historic frequency (maximum value of Meridian.input_data, not
        `new_data`). If `freq_grid` is provided, this argument has no effect.
      freq_grid: List of frequency values. The ROI of each channel is calculated
        for each frequency value in the list. By default, the list includes
        numbers from `1.0` to the maximum frequency in increments of `0.1`.
      use_posterior: Boolean. If `True`, posterior optimal frequencies are
        generated. If `False`, prior optimal frequencies are generated.
      use_kpi: Boolean. If `True`, the counterfactual metrics are calculated
        using KPI. If `False`, the counterfactual metrics are calculated using
        revenue.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        `new_data` if time is modified in `new_data`, or `input_data.n_times`
        otherwise. By default, all time periods are included.
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.

    Returns:
      An xarray Dataset which contains:

      * Coordinates: `frequency`, `rf_channel`, `metric` (`mean`, `median`,
      `ci_lo`, `ci_hi`).
      * Data variables:
        * `optimal_frequency`: The frequency that optimizes the posterior mean
            of ROI.
        * `roi`: The ROI for each frequency value in `freq_grid`.
        * `optimized_incremental_outcome`: The incremental outcome based on the
            optimal frequency.
        * `optimized_effectiveness`: The effectiveness based on the optimal
            frequency.
        * `optimized_roi`: The ROI based on the optimal frequency.
        * `optimized_mroi_by_reach`: The marginal ROI with a small change in
            reach and fixed frequency at the optimal frequency.
        * `optimized_mroi_by_frequency`: The marginal ROI with a small change
            around the optimal frequency and fixed reach.
        * `optimized_cpik`: The CPIK based on the optimal frequency.

    Raises:
      NotFittedModelError: If `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
      ValueError: If there are no channels with reach and frequency data.
    """
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    use_kpi = self._use_kpi(use_kpi)
    new_data = new_data or DataTensors()
    if self._meridian.n_rf_channels == 0:
      raise ValueError(
          "Must have at least one channel with reach and frequency data."
      )
    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )

    filled_data = new_data.validate_and_fill_missing_data(
        [
            constants.RF_IMPRESSIONS,
            constants.RF_SPEND,
            constants.REVENUE_PER_KPI,
        ],
        self._meridian,
    )
    # TODO: Once treatment type filtering is added, remove adding
    # dummy media and media spend to `roi()` and `summary_metrics()`. This is a
    # hack to use `roi()` and `summary_metrics()` for RF only analysis.
    has_media = self._meridian.n_media_channels > 0
    n_media_times = (
        filled_data.get_modified_times(self._meridian)
        or self._meridian.n_media_times
    )
    n_times = (
        filled_data.get_modified_times(self._meridian) or self._meridian.n_times
    )
    dummy_media = backend.ones(
        (self._meridian.n_geos, n_media_times, self._meridian.n_media_channels)
    )
    dummy_media_spend = backend.ones(
        (self._meridian.n_geos, n_times, self._meridian.n_media_channels)
    )

    max_freq = max_frequency or np.max(
        np.array(self._meridian.rf_tensors.frequency)
    )
    if freq_grid is None:
      freq_grid = np.arange(1, max_freq, 0.1)

    # Create a frequency grid for shape (len(freq_grid), n_rf_channels, 4) where
    # the last argument is for the mean, median, lower and upper confidence
    # intervals.
    metric_grid = np.zeros((len(freq_grid), self._meridian.n_rf_channels, 4))

    for i, freq in enumerate(freq_grid):
      new_frequency = backend.ones_like(filled_data.rf_impressions) * freq
      new_reach = filled_data.rf_impressions / new_frequency
      new_roi_data = DataTensors(
          reach=new_reach,
          frequency=new_frequency,
          rf_spend=filled_data.rf_spend,
          revenue_per_kpi=filled_data.revenue_per_kpi,
          media=dummy_media if has_media else None,
          media_spend=dummy_media_spend if has_media else None,
      )
      metric_grid_temp = self.roi(
          new_data=new_roi_data,
          use_posterior=use_posterior,
          selected_geos=selected_geos,
          selected_times=selected_times,
          aggregate_geos=True,
          use_kpi=use_kpi,
      )[..., -self._meridian.n_rf_channels :]
      metric_grid[i, :] = get_central_tendency_and_ci(
          metric_grid_temp, confidence_level, include_median=True
      )

    optimal_freq_idx = np.nanargmax(metric_grid[:, :, 0], axis=0)
    rf_channel_values = (
        self._meridian.input_data.rf_channel.values
        if self._meridian.input_data.rf_channel is not None
        else []
    )

    optimal_frequency = [freq_grid[i] for i in optimal_freq_idx]
    optimal_frequency_values = backend.to_tensor(
        optimal_frequency, dtype=backend.float32
    )
    optimal_frequency_tensor = (
        backend.ones_like(filled_data.rf_impressions) * optimal_frequency_values
    )
    optimal_reach = filled_data.rf_impressions / optimal_frequency_tensor

    new_summary_metrics_data = DataTensors(
        reach=optimal_reach,
        frequency=optimal_frequency_tensor,
        rf_spend=filled_data.rf_spend,
        revenue_per_kpi=filled_data.revenue_per_kpi,
        media=dummy_media if has_media else None,
        media_spend=dummy_media_spend if has_media else None,
    )

    # Compute the optimized metrics based on the optimal frequency.
    optimized_metrics_by_reach = self.summary_metrics(
        new_data=new_summary_metrics_data,
        marginal_roi_by_reach=True,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
    ).sel({
        constants.CHANNEL: rf_channel_values,
        constants.DISTRIBUTION: dist_type,
    })
    optimized_metrics_by_frequency = self.summary_metrics(
        new_data=new_summary_metrics_data,
        marginal_roi_by_reach=False,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
    ).sel({
        constants.CHANNEL: rf_channel_values,
        constants.DISTRIBUTION: dist_type,
    })

    data_vars = {
        constants.ROI: (
            [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
            metric_grid,
        ),
        constants.OPTIMAL_FREQUENCY: (
            [constants.RF_CHANNEL],
            optimal_frequency,
        ),
        constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
            [constants.RF_CHANNEL, constants.METRIC],
            optimized_metrics_by_reach.incremental_outcome.data,
        ),
        constants.OPTIMIZED_ROI: (
            (constants.RF_CHANNEL, constants.METRIC),
            optimized_metrics_by_reach.roi.data,
        ),
        constants.OPTIMIZED_EFFECTIVENESS: (
            [constants.RF_CHANNEL, constants.METRIC],
            optimized_metrics_by_reach.effectiveness.data,
        ),
        constants.OPTIMIZED_MROI_BY_REACH: (
            (constants.RF_CHANNEL, constants.METRIC),
            optimized_metrics_by_reach.mroi.data,
        ),
        constants.OPTIMIZED_MROI_BY_FREQUENCY: (
            (constants.RF_CHANNEL, constants.METRIC),
            optimized_metrics_by_frequency.mroi.data,
        ),
        constants.OPTIMIZED_CPIK: (
            (constants.RF_CHANNEL, constants.METRIC),
            optimized_metrics_by_reach.cpik.data,
        ),
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            constants.FREQUENCY: freq_grid,
            constants.RF_CHANNEL: rf_channel_values,
            constants.METRIC: [
                constants.MEAN,
                constants.MEDIAN,
                constants.CI_LO,
                constants.CI_HI,
            ],
        },
        attrs={
            constants.CONFIDENCE_LEVEL: confidence_level,
            constants.USE_POSTERIOR: use_posterior,
            constants.IS_REVENUE_KPI: not use_kpi,
        },
    )

  def predictive_accuracy(
      self,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Calculates `R-Squared`, `MAPE`, and `wMAPE` goodness of fit metrics.

    `R-Squared`, `MAPE` (mean absolute percentage error), and `wMAPE` (weighted
    absolute percentage error) are calculated on the revenue scale
    (`KPI * revenue_per_kpi`) when `revenue_per_kpi` is specified, or the KPI
    scale when `revenue_per_kpi = None`. This is the same scale as what is used
    in the ROI numerator (incremental outcome).

    Prediction errors in `wMAPE` are weighted by the actual revenue
    (`KPI * revenue_per_kpi`) when `revenue_per_kpi` is specified, or weighted
    by the KPI scale when `revenue_per_kpi = None`. This means that percentage
    errors when revenue is high are weighted more heavily than errors when
    revenue is low.

    `R-Squared`, `MAPE` and `wMAPE` are calculated both at the model-level (one
    observation per geo and time period) and at the national-level (aggregating
    KPI or revenue outcome across geos so there is one observation per time
    period).

    `R-Squared`, `MAPE`, and `wMAPE` are calculated for the full sample. If the
    model object has any holdout observations, then `R-squared`, `MAPE`, and
    `wMAPE` are also calculated for the `Train` and `Test` subsets.

    Args:
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of dates to include. By
        default, all time periods are included.
      use_kpi: Whether to use KPI or revenue scale for the predictive accuracy
        metrics.
      batch_size: Integer representing the maximum draws per chain in each
        batch. By default, `batch_size` is `100`. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      An xarray Dataset containing the computed `R_Squared`, `MAPE`, and `wMAPE`
      values, with coordinates `metric`, `geo_granularity`, `evaluation_set`,
      and accompanying data variable `value`. If `holdout_id` exists, the data
      is split into `'Train'`, `'Test'`, and `'All Data'` subsections, and the
      three metrics are computed for each.
    """
    use_kpi = self._use_kpi(use_kpi)
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          selected_geos=selected_geos,
      )
    dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": False,
        "aggregate_times": False,
    }

    xr_dims = [constants.METRIC, constants.GEO_GRANULARITY]
    xr_coords = {
        constants.METRIC: [
            constants.R_SQUARED,
            constants.MAPE,
            constants.WMAPE,
        ],
        constants.GEO_GRANULARITY: [constants.GEO, constants.NATIONAL],
    }
    if use_kpi:
      input_tensor = self._meridian.kpi
    else:
      input_tensor = self._meridian.kpi * self._meridian.revenue_per_kpi
    actual = np.asarray(
        self.filter_and_aggregate_geos_and_times(
            tensor=input_tensor,
            **dims_kwargs,
        )
    )
    expected = np.mean(
        self.expected_outcome(
            batch_size=batch_size, use_kpi=use_kpi, **dims_kwargs
        ),
        (0, 1),
    )
    rsquared, mape, wmape = self._predictive_accuracy_helper(actual, expected)
    rsquared_national, mape_national, wmape_national = (
        self._predictive_accuracy_helper(np.sum(actual, 0), np.sum(expected, 0))
    )
    if self._meridian.model_spec.holdout_id is None:
      rsquared_arr = [rsquared, rsquared_national]
      mape_arr = [mape, mape_national]
      wmape_arr = [wmape, wmape_national]

      stacked_metric_values = np.stack([rsquared_arr, mape_arr, wmape_arr])

      xr_data = {constants.VALUE: (xr_dims, stacked_metric_values)}
      dataset = xr.Dataset(data_vars=xr_data, coords=xr_coords)
    else:
      xr_dims.append(constants.EVALUATION_SET_VAR)
      xr_coords[constants.EVALUATION_SET_VAR] = list(constants.EVALUATION_SET)

      holdout_id = self._filter_holdout_id_for_selected_geos_and_times(
          self._meridian.model_spec.holdout_id, selected_geos, selected_times
      )

      nansum = lambda x: np.where(
          np.all(np.isnan(x), 0), np.nan, np.nansum(x, 0)
      )
      actual_train = np.where(holdout_id, np.nan, actual)
      actual_test = np.where(holdout_id, actual, np.nan)
      expected_train = np.where(holdout_id, np.nan, expected)
      expected_test = np.where(holdout_id, expected, np.nan)

      geo_train = self._predictive_accuracy_helper(actual_train, expected_train)
      national_train = self._predictive_accuracy_helper(
          nansum(actual_train), nansum(expected_train)
      )
      geo_test = self._predictive_accuracy_helper(actual_test, expected_test)
      national_test = self._predictive_accuracy_helper(
          nansum(actual_test), nansum(expected_test)
      )
      geo_all_data = [rsquared, mape, wmape]
      national_all_data = [rsquared_national, mape_national, wmape_national]

      stacked_train = np.stack([geo_train, national_train], axis=-1)
      stacked_test = np.stack([geo_test, national_test], axis=-1)
      stacked_all_data = np.stack([geo_all_data, national_all_data], axis=-1)
      stacked_total = np.stack(
          [stacked_train, stacked_test, stacked_all_data], axis=-1
      )
      xr_data = {constants.VALUE: (xr_dims, stacked_total)}
      dataset = xr.Dataset(data_vars=xr_data, coords=xr_coords)
    if self._meridian.is_national:
      # Remove the geo-level coordinate.
      dataset = dataset.sel(geo_granularity=[constants.NATIONAL])
    return dataset

  def _predictive_accuracy_helper(
      self,
      actual_eval_set: np.ndarray,
      expected_eval_set: np.ndarray,
  ) -> list[np.floating]:
    """Calculates the predictive accuracy metrics when `holdout_id` exists.

    Args:
      actual_eval_set: An array with filtered and/or aggregated geo and time
        dimensions for the `meridian.kpi * meridian.revenue_per_kpi` calculation
        for either the `'Train'`, `'Test'`, or `'All Data'` evaluation sets.
      expected_eval_set: An array of expected outcome with dimensions
        `(n_chains, n_draws, n_geos, n_times)` for either the `'Train'`,
        `'Test'`, or `'All Data'` evaluation sets.

    Returns:
      A list containing the `geo` or `national` level data for the `R_Squared`,
      `MAPE`, and `wMAPE` metrics computed for either a `'Train'`, `'Test'`, or
      `'All Data'` evaluation set.
    """
    rsquared = _calc_rsquared(expected_eval_set, actual_eval_set)
    mape = _calc_mape(expected_eval_set, actual_eval_set)
    wmape = _calc_weighted_mape(expected_eval_set, actual_eval_set)
    return [rsquared, mape, wmape]

  def _filter_holdout_id_for_selected_geos_and_times(
      self,
      holdout_id: np.ndarray,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
  ) -> np.ndarray:
    """Filters the holdout_id array for selected times and geos."""

    if selected_geos is not None and not self._meridian.is_national:
      geo_mask = [x in selected_geos for x in self._meridian.input_data.geo]
      holdout_id = holdout_id[geo_mask]

    if selected_times is not None:
      time_mask = [x in selected_times for x in self._meridian.input_data.time]
      # If model is national, holdout_id will have only 1 dimension.
      if self._meridian.is_national:
        holdout_id = holdout_id[time_mask]
      else:
        holdout_id = holdout_id[:, time_mask]

    return holdout_id

  def get_rhat(self) -> Mapping[str, backend.Tensor]:
    """Computes the R-hat values for each parameter in the model.

    Returns:
      A dictionary of r-hat values where each parameter is a key and values are
      r-hats corresponding to the parameter.

    Raises:
      NotFittedModelError: If self.sample_posterior() is not called before
        calling this method.
    """
    if constants.POSTERIOR not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          "sample_posterior() must be called prior to calling this method."
      )

    def _transpose_first_two_dims(x: Any) -> backend.Tensor:
      x_tensor = backend.to_tensor(x)
      n_dim = len(x_tensor.shape)
      perm = [1, 0] + list(range(2, n_dim))
      return backend.transpose(x_tensor, perm)

    rhat = backend.mcmc.potential_scale_reduction({
        k: _transpose_first_two_dims(v)
        for k, v in self._meridian.inference_data.posterior.data_vars.items()
    })
    return rhat

  def rhat_summary(self, bad_rhat_threshold: float = 1.2) -> pd.DataFrame:
    """Computes a summary of the R-hat values for each parameter in the model.

    Summarizes the Gelman & Rubin (1992) potential scale reduction for chain
    convergence, commonly referred to as R-hat. It is a convergence diagnostic
    measure that measures the degree to which variance (of the means) between
    chains exceeds what you would expect if the chains were identically
    distributed. Values close to 1.0 indicate convergence. R-hat < 1.2 indicates
    approximate convergence and is a reasonable threshold for many problems
    (Brooks & Gelman, 1998).

    References:
      Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
        Using Multiple Sequences. Statistical Science, 7(4):457-472, 1992.
      Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
        Convergence of Iterative Simulations. Journal of Computational and
        Graphical Statistics, 7(4), 1998.

    Args:
      bad_rhat_threshold: The threshold for determining which R-hat values are
        considered bad.

    Returns:
      A DataFrame with the following columns:

      *   `n_params`: The number of respective parameters in the model.
      *   `avg_rhat`: The average R-hat value for the respective parameter.
      *   `max_rhat`: The maximum R-hat value for the respective parameter.
      *   `percent_bad_rhat`: The percentage of R-hat values for the respective
          parameter that are greater than `bad_rhat_threshold`.
      *   `row_idx_bad_rhat`: The row indices of the R-hat values that are
          greater than `bad_rhat_threshold`.
      *   `col_idx_bad_rhat`: The column indices of the R-hat values that are
          greater than `bad_rhat_threshold`.

    Raises:
      NotFittedModelError: If `self.sample_posterior()` is not called before
        calling this method.
      ValueError: If the number of dimensions of the R-hat array for a parameter
        is not `1` or `2`.
    """
    rhat = self.get_rhat()

    rhat_summary = []
    for param in rhat:
      # Skip if parameter is deterministic according to the prior.
      if self._meridian.prior_broadcast.has_deterministic_param(param):
        continue

      if rhat[param].ndim == 2:
        row_idx, col_idx = np.where(rhat[param] > bad_rhat_threshold)
      elif rhat[param].ndim == 1:
        row_idx = np.where(rhat[param] > bad_rhat_threshold)[0]
        col_idx = []
      elif rhat[param].ndim == 0:
        row_idx = col_idx = []
      else:
        raise ValueError(f"Unexpected dimension for parameter {param}.")

      rhat_summary.append(
          pd.Series({
              constants.PARAM: param,
              constants.N_PARAMS: np.prod(rhat[param].shape),
              constants.AVG_RHAT: np.nanmean(rhat[param]),
              constants.MAX_RHAT: np.nanmax(rhat[param]),
              constants.PERCENT_BAD_RHAT: np.nanmean(
                  rhat[param] > bad_rhat_threshold
              ),
              constants.ROW_IDX_BAD_RHAT: row_idx,
              constants.COL_IDX_BAD_RHAT: col_idx,
          })
      )
    return pd.DataFrame(rhat_summary)

  def response_curves(
      self,
      new_data: DataTensors | None = None,
      spend_multipliers: list[float] | None = None,
      use_posterior: bool = True,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      by_reach: bool = True,
      use_optimal_frequency: bool = False,
      use_kpi: bool = False,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Method to generate a response curves xarray.Dataset.

    Response curves are calculated in aggregate across geos and time periods,
    assuming the historical flighting pattern across geos and time periods for
    each media channel.

    A list of multipliers is applied to each media channel's total historical
    spend within `selected_geos` and `selected_times` to obtain the x-axis
    values. The y-axis values are the incremental outcome generated by each
    channel within `selected_geos` and `selected_times` under the counterfactual
    where media units in each geo and time period are scaled by the
    corresponding multiplier. (Media units for time periods prior to
    `selected_times` are also scaled by the multiplier.)

    Args:
      new_data: Optional `DataTensors` object with optional new tensors:
        `media`, `reach`, `frequency`, `media_spend`, `rf_spend`,
        `revenue_per_kpi`, `times`. If provided, the response curves are
        calculated using the values of the tensors passed in `new_data` and the
        original values of all the remaining tensors. If `None`, the response
        curves are calculated using the original values of all the tensors. If
        any of the tensors in `new_data` is provided with a different number of
        time periods than in `InputData`, then all tensors must be provided with
        the same number of time periods and the `time` tensor must be provided.
      spend_multipliers: List of multipliers. Each channel's total spend is
        multiplied by these factors to obtain the values at which the curve is
        calculated for that channel.
      use_posterior: Boolean. If `True`, posterior response curves are
        generated. If `False`, prior response curves are generated.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of dates to include. If
        `new_data` is provided with modified time periods, then `selected_times`
        must be a subset of `new_data.times`. Otherwise, `selected_times` must
        be a subset of `self._meridian.input_data.time`. By default, all time
        periods are included.
      by_reach: Boolean. For channels with reach and frequency. If `True`, plots
        the response curve by reach. If `False`, plots the response curve by
        frequency.
      use_optimal_frequency: If `True`, uses the optimal frequency to plot the
        response curves. Defaults to `False`.
      use_kpi: A boolean flag indicating whether to use KPI instead of revenue
        to generate the response curves. Defaults to `False`.
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
        An `xarray.Dataset` containing the data needed to visualize response
        curves.
    """
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          selected_geos=selected_geos,
      )
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": True,
        "aggregate_times": True,
    }
    if new_data is None:
      new_data = DataTensors()
    # TODO: b/442920356 - Support flexible time without providing exact dates.
    required_tensors_names = constants.PERFORMANCE_DATA + (constants.TIME,)
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_tensors_names,
        meridian=self._meridian,
        allow_modified_times=True,
    )
    new_n_media_times = filled_data.get_modified_times(self._meridian)

    if new_n_media_times is None:
      _validate_selected_times(
          selected_times=selected_times,
          input_times=self._meridian.input_data.time,
          n_times=self._meridian.n_times,
          arg_name="selected_times",
          comparison_arg_name="the input data",
      )
    else:
      new_time = np.asarray(filled_data.time).astype(str).tolist()
      _validate_flexible_selected_times(
          selected_times=selected_times,
          media_selected_times=None,
          new_n_media_times=new_n_media_times,
          new_time=new_time,
      )
      # TODO: b/407847021 - Switch to Sequence[str] once it is supported.
      if selected_times is not None:
        selected_times = [x in selected_times for x in new_time]
        dim_kwargs["selected_times"] = selected_times

    if self._meridian.n_rf_channels > 0 and use_optimal_frequency:
      opt_freq_data = DataTensors(
          media=filled_data.media,
          rf_impressions=filled_data.reach * filled_data.frequency,
          media_spend=filled_data.media_spend,
          rf_spend=filled_data.rf_spend,
          revenue_per_kpi=filled_data.revenue_per_kpi,
      )
      frequency = backend.ones_like(filled_data.frequency) * backend.to_tensor(
          self.optimal_freq(
              new_data=opt_freq_data,
              selected_geos=selected_geos,
              selected_times=selected_times,
              use_kpi=use_kpi,
          ).optimal_frequency,
          dtype=backend.float32,
      )
      reach = backend.divide_no_nan(
          filled_data.reach * filled_data.frequency,
          frequency,
      )
    else:
      frequency = filled_data.frequency
      reach = filled_data.reach
    if spend_multipliers is None:
      spend_multipliers = list(np.arange(0, 2.2, 0.2))
    incremental_outcome = np.zeros((
        len(spend_multipliers),
        len(self._meridian.input_data.get_all_paid_channels()),
        3,
    ))
    for i, multiplier in enumerate(spend_multipliers):
      if multiplier == 0:
        incremental_outcome[i, :, :] = backend.zeros(
            (len(self._meridian.input_data.get_all_paid_channels()), 3)
        )  # Last dimension = 3 for the mean, ci_lo and ci_hi.
        continue
      scaled_data = _scale_tensors_by_multiplier(
          data=DataTensors(
              media=filled_data.media,
              reach=reach,
              frequency=frequency,
              revenue_per_kpi=filled_data.revenue_per_kpi,
          ),
          multiplier=multiplier,
          by_reach=by_reach,
      )
      inc_outcome_temp = self.incremental_outcome(
          use_posterior=use_posterior,
          new_data=scaled_data.filter_fields(constants.PAID_DATA),
          inverse_transform_outcome=True,
          batch_size=batch_size,
          use_kpi=use_kpi,
          include_non_paid_channels=False,
          **dim_kwargs,
      )
      incremental_outcome[i, :] = get_central_tendency_and_ci(
          inc_outcome_temp, confidence_level
      )

    spend = filled_data.total_spend()
    if spend is not None and spend.ndim == 3:
      spend = self.filter_and_aggregate_geos_and_times(
          tensor=spend,
          flexible_time_dim=True,
          **dim_kwargs,
      )
    spend_einsum = backend.einsum("k,m->km", np.array(spend_multipliers), spend)
    xr_coords = {
        constants.CHANNEL: self._meridian.input_data.get_all_paid_channels(),
        constants.METRIC: [
            constants.MEAN,
            constants.CI_LO,
            constants.CI_HI,
        ],
        constants.SPEND_MULTIPLIER: spend_multipliers,
    }
    xr_data_vars = {
        constants.SPEND: (
            [constants.SPEND_MULTIPLIER, constants.CHANNEL],
            spend_einsum,
        ),
        constants.INCREMENTAL_OUTCOME: (
            [constants.SPEND_MULTIPLIER, constants.CHANNEL, constants.METRIC],
            incremental_outcome,
        ),
    }
    attrs = {constants.CONFIDENCE_LEVEL: confidence_level}
    return xr.Dataset(data_vars=xr_data_vars, coords=xr_coords, attrs=attrs)

  def adstock_decay(
      self, confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL
  ) -> pd.DataFrame:
    """Calculates adstock decay for paid media, RF, and organic media channels.

    Args:
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.

    Returns:
      Pandas DataFrame containing the `channel`, `time_units`, `distribution`,
      `ci_hi`, `ci_lo`, and `mean` for the Adstock function.
    """
    if (
        constants.PRIOR not in self._meridian.inference_data.groups()
        or constants.POSTERIOR not in self._meridian.inference_data.groups()
    ):
      raise model.NotFittedModelError(
          "sample_prior() and sample_posterior() must be called prior to"
          " calling this method."
      )

    # Choose a step_size such that time_unit has consecutive integers defined
    # throughout.
    max_lag = max(self._meridian.model_spec.max_lag, 1)
    steps_per_time_period_max_lag = (
        constants.ADSTOCK_DECAY_MAX_TOTAL_STEPS // max_lag
    )
    steps_per_time_period = min(
        constants.ADSTOCK_DECAY_DEFAULT_STEPS_PER_TIME_PERIOD,
        steps_per_time_period_max_lag,
    )
    step_size = 1 / steps_per_time_period
    l_range = np.arange(0, max_lag, step_size)

    xr_dims = [
        constants.TIME_UNITS,
        constants.CHANNEL,
        constants.METRIC,
        constants.DISTRIBUTION,
    ]
    base_xr_coords = {
        constants.TIME_UNITS: l_range,
        constants.DISTRIBUTION: [constants.PRIOR, constants.POSTERIOR],
        constants.METRIC: [constants.MEAN, constants.CI_LO, constants.CI_HI],
    }
    final_df_list = []

    def _add_adstock_decay_for_channel(
        n_channels: int,
        channel_data: xr.DataArray | None,
        adstock_channel_type: str,
    ) -> None:
      """Helper to compute and append adstock decay data for a channel type."""
      if n_channels > 0:
        channel_values = channel_data.values if channel_data is not None else []
        xr_coords = base_xr_coords | {constants.CHANNEL: channel_values}
        adstock_df = self._get_adstock_dataframe(
            adstock_channel_type,
            l_range,
            xr_dims,
            xr_coords,
            confidence_level,
        )
        if not adstock_df.empty:
          final_df_list.append(adstock_df)

    _add_adstock_decay_for_channel(
        self._meridian.n_media_channels,
        self._meridian.input_data.media_channel,
        constants.MEDIA,
    )
    _add_adstock_decay_for_channel(
        self._meridian.n_rf_channels,
        self._meridian.input_data.rf_channel,
        constants.RF,
    )
    _add_adstock_decay_for_channel(
        self._meridian.n_organic_media_channels,
        self._meridian.input_data.organic_media_channel,
        constants.ORGANIC_MEDIA,
    )
    _add_adstock_decay_for_channel(
        self._meridian.n_organic_rf_channels,
        self._meridian.input_data.organic_rf_channel,
        constants.ORGANIC_RF,
    )

    final_df = pd.concat(final_df_list, ignore_index=True)
    # Adding an extra column that indicates whether time_units is an integer
    # for marking the discrete points on the plot.
    final_df[constants.IS_INT_TIME_UNIT] = final_df[constants.TIME_UNITS].apply(
        lambda x: x.is_integer()
    )
    return final_df

  def _get_hill_curves_dataframe(
      self,
      channel_type: str,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> pd.DataFrame:
    """Computes the point-wise mean and credible intervals for the Hill curves.

    Args:
      channel_type: Type of channel, either `media`, `rf`, or `organic_media`.
      confidence_level: Confidence level for `posterior` and `prior` credible
        intervals, represented as a value between zero and one.

    Returns:
      A DataFrame with data needed to plot the Hill curves, with columns:

      *   `channel`: `media`, `rf`, `organic_media`, or `organic_rf` channel
      name.
      *   `media_units`: Media (for `media` channels) or average frequency (for
          `rf` channels) units.
      *   `distribution`: Indication of `posterior` or `prior` draw.
      *   `ci_hi`: Upper bound of the credible interval of the value of the Hill
          function.
      *   `ci_lo`: Lower bound of the credible interval of the value of the Hill
          function.
      *   `mean`: Point-wise mean of the value of the Hill function per draw.
      *   channel_type: Indication of a `media`, `rf`, `organic_media`
          channel, or `organic_rf`.

    Raises:
      ValueError: If `channel_type` is not one of the recognized constants
      `media`, `rf`, `organic_media`, or `organic_rf`.
    """
    if (
        channel_type == constants.MEDIA
        and self._meridian.input_data.media_channel is not None
    ):
      ec = constants.EC_M
      slope = constants.SLOPE_M
      channels = self._meridian.input_data.media_channel.values
      transformer = self._meridian.media_tensors.media_transformer
      linspace_max_values = np.max(
          np.array(self._meridian.media_tensors.media_scaled), axis=(0, 1)
      )
    elif (
        channel_type == constants.RF
        and self._meridian.input_data.rf_channel is not None
    ):
      ec = constants.EC_RF
      slope = constants.SLOPE_RF
      channels = self._meridian.input_data.rf_channel.values
      transformer = None
      linspace_max_values = np.max(
          np.array(self._meridian.rf_tensors.frequency), axis=(0, 1)
      )
    elif (
        channel_type == constants.ORGANIC_MEDIA
        and self._meridian.input_data.organic_media_channel is not None
    ):
      ec = constants.EC_OM
      slope = constants.SLOPE_OM
      channels = self._meridian.input_data.organic_media_channel.values
      transformer = (
          self._meridian.organic_media_tensors.organic_media_transformer
      )
      linspace_max_values = np.max(
          np.array(self._meridian.organic_media_tensors.organic_media_scaled),
          axis=(0, 1),
      )
    elif (
        channel_type == constants.ORGANIC_RF
        and self._meridian.input_data.organic_rf_channel is not None
    ):
      ec = constants.EC_ORF
      slope = constants.SLOPE_ORF
      channels = self._meridian.input_data.organic_rf_channel.values
      transformer = None
      linspace_max_values = np.max(
          np.array(self._meridian.organic_rf_tensors.organic_frequency),
          axis=(0, 1),
      )
    else:
      raise ValueError(
          f"Unsupported channel type: {channel_type} or the requested type of"
          " channels (`media`, `rf`, `organic_media`, or `organic_rf`) are not"
          " present."
      )
    linspace = np.linspace(
        0,
        linspace_max_values,
        constants.HILL_NUM_STEPS,
    )
    linspace_filler = np.linspace(0, 1, constants.HILL_NUM_STEPS)
    xr_dims = [
        constants.MEDIA_UNITS,
        constants.CHANNEL,
        constants.METRIC,
        constants.DISTRIBUTION,
    ]
    xr_coords = {
        constants.MEDIA_UNITS: linspace_filler,
        constants.CHANNEL: list(channels),
        constants.DISTRIBUTION: [constants.PRIOR, constants.POSTERIOR],
        constants.METRIC: [constants.MEAN, constants.CI_LO, constants.CI_HI],
    }
    # Expanding the linspace by one dimension since the HillTransformer requires
    # 3-dimensional input as (geo, time, channel).
    expanded_linspace = backend.expand_dims(linspace, axis=0)
    # Including [:, :, 0, :, :] in the output of the Hill Function to reduce the
    # tensors by the geo dimension. Original Hill dimension shape is (n_chains,
    # n_draws, n_geos, n_times, n_channels), and we want to plot the
    # dependency on time only.
    hill_vals_prior = adstock_hill.HillTransformer(
        self._meridian.inference_data.prior[ec].values,
        self._meridian.inference_data.prior[slope].values,
    ).forward(expanded_linspace)[:, :, 0, :, :]
    hill_vals_posterior = adstock_hill.HillTransformer(
        self._meridian.inference_data.posterior[ec].values,
        self._meridian.inference_data.posterior[slope].values,
    ).forward(expanded_linspace)[:, :, 0, :, :]

    hill_dataset = _central_tendency_and_ci_by_prior_and_posterior(
        hill_vals_prior,
        hill_vals_posterior,
        constants.HILL_SATURATION_LEVEL,
        xr_dims,
        xr_coords,
        confidence_level,
    )

    df_raw = (
        hill_dataset[constants.HILL_SATURATION_LEVEL]
        .to_dataframe()
        .reset_index()
    )

    # Ensure the channel order matches the tensor order (defined by 'channels')
    # by using a Categorical type before pivoting. This prevents pivot from
    # sorting alphabetically, which can cause misalignment between channel names
    # and the calculated media units derived later from the tensor order.
    df_raw[constants.CHANNEL] = pd.Categorical(
        df_raw[constants.CHANNEL], categories=channels
    )
    df = df_raw.pivot(
        index=[
            constants.CHANNEL,
            constants.MEDIA_UNITS,
            constants.DISTRIBUTION,
        ],
        columns=constants.METRIC,
        values=constants.HILL_SATURATION_LEVEL,
    ).reset_index()

    # Fill media_units or frequency x-axis with the correct range.
    media_units_arr = []
    if transformer is not None:
      population_scaled_median = transformer.population_scaled_median_m
      x_range_full_shape = linspace * backend.transpose(
          population_scaled_median[:, backend.newaxis]
      )
    else:
      x_range_full_shape = linspace

    # Flatten this into a list.
    x_range_list = np.asarray(
        backend.reshape(backend.transpose(x_range_full_shape), [-1])
    ).tolist()

    # Doubles each value in the list to account for alternating prior
    # and posterior.
    x_range_doubled = list(
        itertools.chain.from_iterable(zip(x_range_list, x_range_list))
    )
    media_units_arr.extend(x_range_doubled)

    df[constants.CHANNEL_TYPE] = channel_type
    df[constants.MEDIA_UNITS] = media_units_arr
    return df

  def _get_channel_hill_histogram_dataframe(
      self,
      channel_type: str,
      data_to_histogram: backend.Tensor,
      channel_names: Sequence[str],
      n_bins: int,
  ) -> pd.DataFrame:
    """Calculates hill histogram dataframe for a given channel type's values.

    Args:
      channel_type: The type of channel (e.g., 'rf', 'media', 'organic_media',
        'organic_rf').
      data_to_histogram: The 2D tensor (observations, channels). containing the
        data whose distribution needs to be histogrammed for each channel.
      channel_names: The names corresponding to the channels in
        data_to_histogram.
      n_bins: The number of bins for the histogram.

    Returns:
      A Pandas DataFrame containing the calculated histogram data for all
      channels of the given type. Returns an empty DataFrame if no valid
      data is found for any channel.
    """
    channels_data = {
        constants.CHANNEL: [],
        constants.CHANNEL_TYPE: [],
        constants.SCALED_COUNT_HISTOGRAM: [],
        constants.COUNT_HISTOGRAM: [],
        constants.START_INTERVAL_HISTOGRAM: [],
        constants.END_INTERVAL_HISTOGRAM: [],
    }

    for i, channel_name in enumerate(channel_names):
      channel_data_np = np.asarray(data_to_histogram[:, i])
      channel_data_np = channel_data_np[~np.isnan(channel_data_np)]
      if channel_data_np.size == 0:
        continue

      counts_per_bucket, buckets = np.histogram(
          channel_data_np, bins=n_bins, density=True
      )
      max_counts = (
          np.max(counts_per_bucket) if np.max(counts_per_bucket) > 0 else 1.0
      )

      num_buckets = len(counts_per_bucket)
      channels_data[constants.CHANNEL].extend([channel_name] * num_buckets)
      channels_data[constants.CHANNEL_TYPE].extend([channel_type] * num_buckets)
      channels_data[constants.SCALED_COUNT_HISTOGRAM].extend(
          counts_per_bucket / max_counts
      )
      channels_data[constants.COUNT_HISTOGRAM].extend(counts_per_bucket)
      channels_data[constants.START_INTERVAL_HISTOGRAM].extend(buckets[:-1])
      channels_data[constants.END_INTERVAL_HISTOGRAM].extend(buckets[1:])

    return pd.DataFrame(channels_data)

  def _get_hill_histogram_dataframe(self, n_bins: int) -> pd.DataFrame:
    """Calculates histogram data for a given channel type's values.

      Computes histogram data for the distribution of media units (for media or
      organic media channels) or frequency (for RF channels) across
      observations.

    Args:
      n_bins: Number of equal-width bins to include in the histogram for the
        plotting.

    Returns:
      Pandas DataFrame with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `channel_type`: `media` or `rf` channel type.
      *   `scaled_count_histogram`: Scaled count of media units or average
          frequencies within the bin.
      *   `count_histogram`: True count value of media units or average
          frequencies within the bin.
      *   `start_interval_histogram`: Media unit or average frequency starting
          point for a histogram bin.
      *   `end_interval_histogram`: Media unit or average frequency ending point
          for a histogram bin.

      This DataFrame will be used to plot the histograms showing the relative
      distribution of media units per capita for media channels or average
      frequency for RF channels over weeks and geos for the Hill plots.
    """
    n_geos = self._meridian.n_geos
    n_media_times = self._meridian.n_media_times

    df_list = []

    # RF.
    if self._meridian.input_data.rf_channel is not None:
      frequency = self._meridian.rf_tensors.frequency
      if frequency is not None:
        reshaped_frequency = backend.reshape(
            frequency, (n_geos * n_media_times, self._meridian.n_rf_channels)
        )
        rf_hist_data = self._get_channel_hill_histogram_dataframe(
            channel_type=constants.RF,
            data_to_histogram=reshaped_frequency,
            channel_names=self._meridian.input_data.rf_channel.values,
            n_bins=n_bins,
        )
        df_list.append(pd.DataFrame(rf_hist_data))

    # Media.
    if self._meridian.input_data.media_channel is not None:
      transformer = self._meridian.media_tensors.media_transformer
      scaled = self._meridian.media_tensors.media_scaled
      if transformer is not None and scaled is not None:
        population_scaled_median = transformer.population_scaled_median_m
        scaled_media_units = scaled * population_scaled_median
        reshaped_scaled_media_units = backend.reshape(
            scaled_media_units,
            (n_geos * n_media_times, self._meridian.n_media_channels),
        )
        media_hist_data = self._get_channel_hill_histogram_dataframe(
            channel_type=constants.MEDIA,
            data_to_histogram=reshaped_scaled_media_units,
            channel_names=self._meridian.input_data.media_channel.values,
            n_bins=n_bins,
        )
        df_list.append(pd.DataFrame(media_hist_data))
    # Organic media.
    if self._meridian.input_data.organic_media_channel is not None:
      transformer_om = (
          self._meridian.organic_media_tensors.organic_media_transformer
      )
      scaled_om = self._meridian.organic_media_tensors.organic_media_scaled
      if transformer_om is not None and scaled_om is not None:
        population_scaled_median_om = transformer_om.population_scaled_median_m
        scaled_organic_media_units = scaled_om * population_scaled_median_om
        reshaped_scaled_organic_media_units = backend.reshape(
            scaled_organic_media_units,
            (n_geos * n_media_times, self._meridian.n_organic_media_channels),
        )
        organic_media_hist_data = self._get_channel_hill_histogram_dataframe(
            channel_type=constants.ORGANIC_MEDIA,
            data_to_histogram=reshaped_scaled_organic_media_units,
            channel_names=self._meridian.input_data.organic_media_channel.values,
            n_bins=n_bins,
        )
        df_list.append(pd.DataFrame(organic_media_hist_data))

    # Organic RF.
    if self._meridian.input_data.organic_rf_channel is not None:
      frequency = self._meridian.organic_rf_tensors.organic_frequency
      if frequency is not None:
        reshaped_frequency = backend.reshape(
            frequency,
            (n_geos * n_media_times, self._meridian.n_organic_rf_channels),
        )
        organic_rf_hist_data = self._get_channel_hill_histogram_dataframe(
            channel_type=constants.ORGANIC_RF,
            data_to_histogram=reshaped_frequency,
            channel_names=self._meridian.input_data.organic_rf_channel.values,
            n_bins=n_bins,
        )
        df_list.append(pd.DataFrame(organic_rf_hist_data))
    return pd.concat(df_list, ignore_index=True)

  def hill_curves(
      self,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      n_bins: int = 25,
  ) -> pd.DataFrame:
    """Estimates Hill curve tables used for plotting each channel's curves.

    Args:
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one. Default is
        `0.9`.
      n_bins: Number of equal-width bins to include in the histogram for the
        plotting. Default is `25`.

    Returns:
      Hill curves `pd.DataFrame` with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `media_units`: Media (for `media` channels) or average frequency (for
          `rf` channels) units.
      *   `distribution`: Indication of `posterior` or `prior` draw.
      *   `ci_hi`: Upper bound of the credible interval of the value of the Hill
          function.
      *   `ci_lo`: Lower bound of the credible interval of the value of the Hill
          function.
      *   `mean`: Point-wise mean of the value of the Hill function per draw.
      *   `channel_type`: Indication of a `media` or `rf` channel.
      *   `scaled_count_histogram`: Scaled count of media units or average
          frequencies within the bin.
      *   `count_histogram`: Count value of media units or average
          frequencies within the bin.
      *   `start_interval_histogram`: Media unit or average frequency starting
          point for a histogram bin.
      *   `end_interval_histogram`: Media unit or average frequency ending point
          for a histogram bin.
    """
    if (
        constants.PRIOR not in self._meridian.inference_data.groups()
        or constants.POSTERIOR not in self._meridian.inference_data.groups()
    ):
      raise model.NotFittedModelError(
          "sample_prior() and sample_posterior() must be called prior to"
          " calling this method."
      )

    final_dfs = [pd.DataFrame()]
    for n_channels, channel_type in [
        (self._meridian.n_media_channels, constants.MEDIA),
        (self._meridian.n_rf_channels, constants.RF),
        (self._meridian.n_organic_media_channels, constants.ORGANIC_MEDIA),
        (self._meridian.n_organic_rf_channels, constants.ORGANIC_RF),
    ]:
      if n_channels > 0:
        hill_df = self._get_hill_curves_dataframe(
            channel_type, confidence_level
        )
        final_dfs.append(hill_df)

    final_dfs.append(self._get_hill_histogram_dataframe(n_bins=n_bins))
    return pd.concat(final_dfs)

  def _compute_roi_aggregate(
      self,
      incremental_outcome_prior: backend.Tensor,
      incremental_outcome_posterior: backend.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      spend_with_total: backend.Tensor,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      metric_name: str = constants.ROI,
  ) -> xr.Dataset:
    # TODO: Support calibration_period_bool.
    return _central_tendency_and_ci_by_prior_and_posterior(
        prior=incremental_outcome_prior / spend_with_total,
        posterior=incremental_outcome_posterior / spend_with_total,
        metric_name=metric_name,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
        include_median=True,
    )

  def _compute_spend_data_aggregate(
      self,
      spend_with_total: backend.Tensor,
      impressions_with_total: backend.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
  ) -> xr.Dataset:
    """Computes the MediaSummary metrics involving the input data.

    Returns:
      An xarray Dataset consisting of the following arrays:

      * `impressions`
      * `pct_of_impressions`
      * `spend`
      * `pct_of_spend`
      * `cpm` (spend for every 1,000 impressions)
    """
    pct_of_impressions = (
        impressions_with_total / impressions_with_total[..., -1:] * 100
    )
    pct_of_spend = spend_with_total / spend_with_total[..., -1:] * 100

    return xr.Dataset(
        data_vars={
            constants.IMPRESSIONS: (xr_dims, impressions_with_total),
            constants.PCT_OF_IMPRESSIONS: (xr_dims, pct_of_impressions),
            constants.SPEND: (xr_dims, spend_with_total),
            constants.PCT_OF_SPEND: (xr_dims, pct_of_spend),
            constants.CPM: (
                xr_dims,
                spend_with_total / impressions_with_total * 1000,
            ),
        },
        coords=xr_coords,
    )

  def _compute_effectiveness_aggregate(
      self,
      incremental_outcome_prior: backend.Tensor,
      incremental_outcome_posterior: backend.Tensor,
      impressions_with_total: backend.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> xr.Dataset:
    return _central_tendency_and_ci_by_prior_and_posterior(
        prior=incremental_outcome_prior / impressions_with_total,
        posterior=incremental_outcome_posterior / impressions_with_total,
        metric_name=constants.EFFECTIVENESS,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
        include_median=True,
    )

  def _compute_cpik_aggregate(
      self,
      incremental_kpi_prior: backend.Tensor,
      incremental_kpi_posterior: backend.Tensor,
      spend_with_total: backend.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> xr.Dataset:
    return _central_tendency_and_ci_by_prior_and_posterior(
        prior=spend_with_total / incremental_kpi_prior,
        posterior=spend_with_total / incremental_kpi_posterior,
        metric_name=constants.CPIK,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
        include_median=True,
    )

  def _compute_pct_of_contribution(
      self,
      incremental_outcome_prior: backend.Tensor,
      incremental_outcome_posterior: backend.Tensor,
      expected_outcome_prior: backend.Tensor,
      expected_outcome_posterior: backend.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> xr.Dataset:
    """Computes the parts of `MediaSummary` related to mean expected outcome."""
    mean_expected_outcome_prior = backend.reduce_mean(
        expected_outcome_prior, (0, 1)
    )
    mean_expected_outcome_posterior = backend.reduce_mean(
        expected_outcome_posterior, (0, 1)
    )

    return _central_tendency_and_ci_by_prior_and_posterior(
        prior=(
            incremental_outcome_prior
            / mean_expected_outcome_prior[..., None]
            * 100
        ),
        posterior=(
            incremental_outcome_posterior
            / mean_expected_outcome_posterior[..., None]
            * 100
        ),
        metric_name=constants.PCT_OF_CONTRIBUTION,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
        include_median=True,
    )

  def get_historical_spend(
      self,
      selected_times: Sequence[str] | None = None,
      include_media: bool = True,
      include_rf: bool = True,
  ) -> xr.DataArray:
    """Deprecated. Gets the aggregated historical spend based on the time.

    Args:
      selected_times: The time period to get the historical spends. If None, the
        historical spends will be aggregated over all time points.
      include_media: Whether to include spends for paid media channels that do
        not have R&F data.
      include_rf: Whether to include spends for paid media channels with R&F
        data.

    Returns:
      An `xr.DataArray` with the coordinate `channel` and contains the data
      variable `spend`.

    Raises:
      ValueError: A ValueError is raised when `include_media` and `include_rf`
      are both False.
    """
    warnings.warn(
        "`get_historical_spend` is deprecated. Please use "
        "`get_aggregated_spend` with `new_data=None` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.get_aggregated_spend(
        selected_times=selected_times,
        include_media=include_media,
        include_rf=include_rf,
    )

  def get_aggregated_spend(
      self,
      new_data: DataTensors | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      include_media: bool = True,
      include_rf: bool = True,
  ) -> xr.DataArray:
    """Gets the aggregated spend based on the selected geos and time.

    Args:
      new_data: An optional `DataTensors` object containing the new `media`,
        `media_spend`, `reach`, `frequency`, `rf_spend` tensors. If `None`, the
        existing tensors from the Meridian object are used. If `new_data`
        argument is used, then the aggregated spend is computed using the values
        of the tensors passed in the `new_data` argument and the original values
        of all the remaining tensors.  If any of the tensors in `new_data` is
        provided with a different number of time periods than in `InputData`,
        then all tensors must be provided with the same number of time periods.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included. The selected geos should match those in
        `InputData.geo`.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        KPI data. By default, all time periods are included.
      include_media: Whether to include spends for paid media channels that do
        not have R&F data.
      include_rf: Whether to include spends for paid media channels with R&F
        data.

    Returns:
      An `xr.DataArray` with the coordinate `channel` and contains the data
      variable `spend`.

    Raises:
      ValueError: A ValueError is raised when `include_media` and `include_rf`
      are both False.
    """
    if not include_media and not include_rf:
      raise ValueError(
          "At least one of include_media or include_rf must be True."
      )
    new_data = new_data or DataTensors()
    required_tensors_names = constants.PAID_CHANNELS + constants.SPEND_DATA
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names, self._meridian
    )

    empty_da = xr.DataArray(
        dims=[constants.CHANNEL], coords={constants.CHANNEL: []}
    )

    if not include_media:
      aggregated_media_spend = empty_da
    elif (
        self._meridian.media_tensors.media is None
        or self._meridian.media_tensors.media_spend is None
        or self._meridian.input_data.media_channel is None
    ):
      warnings.warn(
          "Requested spends for paid media channels that do not have R&F"
          " data, but the channels are not available."
      )
      aggregated_media_spend = empty_da
    else:
      aggregated_media_spend = self._impute_and_aggregate_spend(
          selected_geos=selected_geos,
          selected_times=selected_times,
          media_execution_values=filled_data.media,
          channel_spend=filled_data.media_spend,
          channel_names=list(self._meridian.input_data.media_channel.values),
      )

    if not include_rf:
      aggregated_rf_spend = empty_da
    elif (
        self._meridian.input_data.rf_channel is None
        or self._meridian.rf_tensors.reach is None
        or self._meridian.rf_tensors.frequency is None
        or self._meridian.rf_tensors.rf_spend is None
    ):
      warnings.warn(
          "Requested spends for paid media channels with R&F data, but the"
          " channels are not available.",
      )
      aggregated_rf_spend = empty_da
    else:
      rf_execution_values = filled_data.reach * filled_data.frequency
      aggregated_rf_spend = self._impute_and_aggregate_spend(
          selected_geos=selected_geos,
          selected_times=selected_times,
          media_execution_values=rf_execution_values,
          channel_spend=filled_data.rf_spend,
          channel_names=list(self._meridian.input_data.rf_channel.values),
      )

    return xr.concat(
        [aggregated_media_spend, aggregated_rf_spend], dim=constants.CHANNEL
    )

  def _impute_and_aggregate_spend(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | Sequence[bool] | None,
      media_execution_values: backend.Tensor,
      channel_spend: backend.Tensor,
      channel_names: Sequence[str],
  ) -> xr.DataArray:
    """Imputes and aggregates the spend within selected dimensions.

    This function is used to aggregate the spend within selected geos over the
    selected time period. Imputation is required when `channel_spend` has only
    one dimension and the aggregation is applied to only a subset of geos or
    times, as specified by `selected_geos` and `selected_times`. The
    `media_execution_values` argument only serves the purpose of imputation.
    Although `media_execution_values` is a required argument, its values only
    affect the output when imputation is required.

    Args:
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included. The selected geos should match those in
        `InputData.geo`.
      selected_times: Optional list containing either a subset of dates to
        include or booleans with length equal to the number of time periods in
        KPI data. By default, all time periods are included.
      media_execution_values: The media execution values over all time points.
      channel_spend: The spend over all time points. Its shape can be `(n_geos,
        n_times, n_media_channels)` or `(n_media_channels,)` if the data is
        aggregated over `geo` and `time` dimensions.
      channel_names: The channel names.

    Returns:
      An `xr.DataArray` with the coordinate `channel` and contains the data
      variable `spend`.
    """
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": True,
        "aggregate_times": True,
        "flexible_time_dim": True,
    }

    if channel_spend.ndim == 3:
      aggregated_spend = np.asarray(
          self.filter_and_aggregate_geos_and_times(
              channel_spend,
              has_media_dim=True,
              **dim_kwargs,
          )
      )
    # channel_spend.ndim can only be 3 or 1.
    else:
      # media spend can have more time points than the model time points
      if media_execution_values.shape[1] == self._meridian.n_media_times:
        media_exe_values = media_execution_values[
            :, -self._meridian.n_times :, :
        ]
      else:
        media_exe_values = media_execution_values
      # Calculates CPM over all times and geos if the spend does not have time
      # and geo dimensions.
      target_media_exe_values = self.filter_and_aggregate_geos_and_times(
          media_exe_values,
          **dim_kwargs,
      )
      imputed_cpmu = backend.divide_no_nan(
          channel_spend,
          np.sum(media_exe_values, (0, 1)),
      )
      aggregated_spend = np.asarray(target_media_exe_values * imputed_cpmu)

    return xr.DataArray(
        data=aggregated_spend,
        dims=[constants.CHANNEL],
        coords={constants.CHANNEL: channel_names},
    )

  def negative_baseline_probability(
      self,
      non_media_baseline_values: Sequence[float] | None = None,
      use_posterior: bool = True,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> np.floating:
    """Calculates either prior or posterior negative baseline probability.

    This calculates either the prior or posterior probability that the baseline,
    aggregated over the supplied time window, is negative.

    The baseline is calculated by computing `expected_outcome` with the
    following assumptions:
      1) `media` is set to all zeros,
      2) `reach` is set to all zeros,
      3) `organic_media` is set to all zeros,
      4) `organic_reach` is set to all zeros,
      5) `non_media_treatments` is set to the counterfactual values according
      to the `non_media_baseline_values` argument,
      6) `controls` are set to historical values.

    Args:
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float denoting a fixed
        value that will be used as the baseline for the given channel. It is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.
      use_posterior: Boolean. If `True`, then the expected outcome posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      selected_geos: Optional list of containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list of containing a subset of dates to include.
        The values accepted here must match time dimension coordinates from
        `InputData.time`. By default, all time periods are included.
      use_kpi: Boolean. If `use_kpi = True`, the expected KPI is calculated;
        otherwise the expected revenue `(kpi * revenue_per_kpi)` is calculated.
        It is required that `use_kpi = True` if `revenue_per_kpi` is not defined
        or if `inverse_transform_outcome = False`.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      A float representing the prior or posterior negative baseline probability
      over the supplied time window.
    Raises:
      NotFittedModelError: if `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
    """

    baseline_draws = self._calculate_baseline_expected_outcome(
        non_media_baseline_values=non_media_baseline_values,
        use_posterior=use_posterior,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=True,
        aggregate_times=True,
        use_kpi=use_kpi,
        batch_size=batch_size,
    )
    return np.mean(baseline_draws < 0)
