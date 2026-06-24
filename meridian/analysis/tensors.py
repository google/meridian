# Copyright 2026 The Meridian Authors.
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

"""Data structures and tensor preparation utilities for Meridian."""

from collections.abc import Sequence
import dataclasses
import numbers
from typing import Any, Optional, Union
import warnings

from meridian import backend
from meridian import constants
from meridian.model import context
from meridian.model import equations
import numpy as np
from typing_extensions import Self
import xarray as xr

__all__ = (
    "AnalyzerInputs",
    "CounterfactualInputs",
    "DataTensors",
    "DataTensorsBuilder",
    "DistributionTensors",
)


# TODO: Remove this method.
def _get_model_context(
    meridian: Any | None,
    model_context: context.ModelContext | None,
) -> context.ModelContext:
  """Gets `model_context`, handling the deprecated `meridian` argument."""
  if meridian is not None:
    warnings.warn(
        (
            "The `meridian` argument is deprecated and will be removed in a"
            " future version. Use `model_context` instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )
    return meridian.model_context
  if model_context is None:
    raise ValueError("Either `meridian` or `model_context` must be provided.")
  return model_context


def _validate_non_media_baseline_values_numbers(
    non_media_baseline_values: Sequence[str | float] | None,
) -> None:
  if non_media_baseline_values is None:
    return

  for value in non_media_baseline_values:
    if not isinstance(value, numbers.Number):
      raise ValueError(
          f"Invalid `non_media_baseline_values` value: '{value}'. Only float"
          " numbers are supported."
      )


def _check_n_dims(tensor: backend.Tensor, name: str, n_dims: int) -> None:
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
    *,
    arg_name: str,
    comparison_arg_name: str,
) -> None:
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
    *,
    selected_times: Sequence[str] | Sequence[bool] | None,
    media_selected_times: Sequence[str] | Sequence[bool] | None,
    new_n_media_times: int,
    new_time: Sequence[str] | None = None,
) -> None:
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


@dataclasses.dataclass(kw_only=True)
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

  media: Union[backend.Tensor, None]
  media_spend: Union[backend.Tensor, None]
  reach: Union[backend.Tensor, None]
  frequency: Union[backend.Tensor, None]
  rf_impressions: Union[backend.Tensor, None]
  rf_spend: Union[backend.Tensor, None]
  organic_media: Union[backend.Tensor, None]
  organic_reach: Union[backend.Tensor, None]
  organic_frequency: Union[backend.Tensor, None]
  non_media_treatments: Union[backend.Tensor, None]
  controls: Union[backend.Tensor, None]
  revenue_per_kpi: Union[backend.Tensor, None]
  time: Union[backend.Tensor, None]

  def __init__(
      self,
      media: backend.Tensor | None = None,
      media_spend: backend.Tensor | None = None,
      reach: backend.Tensor | None = None,
      frequency: backend.Tensor | None = None,
      rf_impressions: backend.Tensor | None = None,
      rf_spend: backend.Tensor | None = None,
      organic_media: backend.Tensor | None = None,
      organic_reach: backend.Tensor | None = None,
      organic_frequency: backend.Tensor | None = None,
      non_media_treatments: backend.Tensor | None = None,
      controls: backend.Tensor | None = None,
      revenue_per_kpi: backend.Tensor | None = None,
      time: Sequence[str] | backend.Tensor | None = None,
  ):
    """Initializes the instance."""
    self.media = (
        backend.cast(media, backend.float_dtype) if media is not None else None
    )
    self.media_spend = (
        backend.cast(media_spend, backend.float_dtype)
        if media_spend is not None
        else None
    )
    self.reach = (
        backend.cast(reach, backend.float_dtype) if reach is not None else None
    )
    self.frequency = (
        backend.cast(frequency, backend.float_dtype)
        if frequency is not None
        else None
    )
    self.rf_impressions = (
        backend.cast(rf_impressions, backend.float_dtype)
        if rf_impressions is not None
        else None
    )
    self.rf_spend = (
        backend.cast(rf_spend, backend.float_dtype)
        if rf_spend is not None
        else None
    )
    self.organic_media = (
        backend.cast(organic_media, backend.float_dtype)
        if organic_media is not None
        else None
    )
    self.organic_reach = (
        backend.cast(organic_reach, backend.float_dtype)
        if organic_reach is not None
        else None
    )
    self.organic_frequency = (
        backend.cast(organic_frequency, backend.float_dtype)
        if organic_frequency is not None
        else None
    )
    self.non_media_treatments = (
        backend.cast(non_media_treatments, backend.float_dtype)
        if non_media_treatments is not None
        else None
    )
    self.controls = (
        backend.cast(controls, backend.float_dtype)
        if controls is not None
        else None
    )
    self.revenue_per_kpi = (
        backend.cast(revenue_per_kpi, backend.float_dtype)
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

  def get_modified_times(
      self,
      # TODO: Remove this argument.
      meridian: Any | None = None,
      model_context: context.ModelContext | None = None,
  ) -> int | None:
    """Returns `n_times` of any tensor where `n_times` has been modified.

    This method compares the time dimensions of the attributes in the
    `DataTensors` object with the corresponding tensors in the `model_context`
    object. If any of the time dimensions are different, then this method
    returns the modified number of time periods of the tensor in the
    `DataTensors` object. If all time dimensions are the same, returns `None`.

    Args:
      meridian: Deprecated. A Meridian object to validate against and get the
        original data tensors from. This argument is deprecated and will be
        removed in a future version. Use `model_context` instead.
      model_context: A ModelContext object to validate against and get the
        original data tensors from.

    Returns:
      The `n_times` of any tensor where `n_times` is different from the times
      of the corresponding tensor in the `model_context` object. If all time
      dimensions are the same, returns `None`.
    """
    model_context = _get_model_context(meridian, model_context)
    for field in dataclasses.fields(self):
      new_tensor = getattr(self, field.name)
      if field.name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(model_context.rf_tensors, field.name)
      else:
        old_tensor = getattr(model_context.input_data, field.name)
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
      # TODO: Remove this argument.
      meridian: Any | None = None,
      model_context: context.ModelContext | None = None,
      allow_modified_times: bool = True,
  ) -> Self:
    """Fills missing data tensors with their original values from the model.

    This method uses the collection of data tensors set in the DataTensor class
    and fills in the missing tensors with their original values from the
    ModelContext object that is passed in. For example, if
    `required_tensors_names = ["media", "reach", "frequency"]` and only `media`
    is set in the DataTensors class, then this method will output a new
    DataTensors object with the `media` value in this object plus the values of
    the `reach` and `frequency` from the `model_context` object.

    Args:
      required_tensors_names: A sequence of data tensors names to validate and
        fill in with the original values from the `model_context` object.
      meridian: Deprecated. A Meridian object to validate against and get the
        original data tensors from. This argument is deprecated and will be
        removed in a future version. Use `model_context` instead.
      model_context: A ModelContext object to validate against and get the
        original data tensors from.
      allow_modified_times: A boolean flag indicating whether to allow modified
        time dimensions in the new data tensors. If False, an error will be
        raised if the time dimensions of any tensor is modified.

    Returns:
      A `DataTensors` container with the original values from the Meridian
      object filled in for the missing data tensors.
    """
    model_context = _get_model_context(meridian, model_context)
    self._validate_correct_variables_filled(
        required_variables=required_tensors_names,
        model_context=model_context,
    )
    self._validate_geo_dims(
        required_fields=required_tensors_names, model_context=model_context
    )
    self._validate_channel_dims(
        required_fields=required_tensors_names, model_context=model_context
    )
    if allow_modified_times:
      self._validate_time_dims_flexible_times(
          required_fields=required_tensors_names, model_context=model_context
      )
    else:
      self._validate_time_dims(
          required_fields=required_tensors_names, model_context=model_context
      )

    return self._fill_default_values(
        required_fields=required_tensors_names,
        model_context=model_context,
    )

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
      self,
      required_variables: Sequence[str],
      model_context: context.ModelContext,
  ) -> None:
    """Validates that the correct variables are filled.

    Args:
      required_variables: A sequence of data tensors names that are required to
        be filled in.
      model_context: A `ModelContext` object to validate against.

    Raises:
      ValueError: If an attribute exists in the `DataTensors` object that is not
        in the `model_context` object, it is not allowed to be used in analysis.
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
          if model_context.n_rf_channels == 0:
            raise ValueError(
                "New `rf_impressions` is not allowed because there are no R&F"
                " channels in the Meridian model."
            )
        elif getattr(model_context.input_data, field.name) is None:
          raise ValueError(
              f"New `{field.name}` is not allowed because the input data to the"
              f" Meridian model does not contain `{field.name}`."
          )

  def _validate_geo_dims(
      self,
      required_fields: Sequence[str],
      model_context: context.ModelContext,
  ) -> None:
    """Validates the geo dimension of the specified data variables."""
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if new_tensor is not None and new_tensor.shape[0] != model_context.n_geos:
        # Skip spend and time data with only 1 dimension.
        if new_tensor.ndim == 1:
          continue
        raise ValueError(
            f"New `{var_name}` is expected to have {model_context.n_geos}"
            f" geos. Found {new_tensor.shape[0]} geos."
        )

  def _validate_channel_dims(
      self,
      required_fields: Sequence[str],
      model_context: context.ModelContext,
  ) -> None:
    """Validates the channel dimension of the specified data variables."""
    for var_name in required_fields:
      if var_name in [constants.REVENUE_PER_KPI, constants.TIME]:
        continue
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(model_context.input_data, var_name)
      if new_tensor is not None:
        assert old_tensor is not None
        if new_tensor.shape[-1] != old_tensor.shape[-1]:
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[-1]}"
              f" channels. Found {new_tensor.shape[-1]} channels."
          )

  def _validate_time_dims(
      self,
      required_fields: Sequence[str],
      model_context: context.ModelContext,
  ):
    """Validates the time dimension of the specified data variables."""
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(model_context.input_data, var_name)

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
      self,
      required_fields: Sequence[str],
      model_context: context.ModelContext,
  ):
    """Validates the time dimension for the flexible times case."""
    new_n_times = self.get_modified_times(model_context=model_context)
    # If no times were modified, then there is nothing more to validate.
    if new_n_times is None:
      return

    missing_params = []
    for var_name in required_fields:
      new_tensor = getattr(self, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(model_context.input_data, var_name)

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
            "granularity with the same number of time periods as the other "
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
      self,
      required_fields: Sequence[str],
      model_context: context.ModelContext,
  ) -> Self:
    """Fills default values and returns a new DataTensors object."""
    output = {}
    for field in dataclasses.fields(self):
      var_name = field.name
      if var_name not in required_fields:
        continue

      if hasattr(model_context.media_tensors, var_name):
        old_tensor = getattr(model_context.media_tensors, var_name)
      elif hasattr(model_context.rf_tensors, var_name):
        old_tensor = getattr(model_context.rf_tensors, var_name)
      elif hasattr(model_context.organic_media_tensors, var_name):
        old_tensor = getattr(model_context.organic_media_tensors, var_name)
      elif hasattr(model_context.organic_rf_tensors, var_name):
        old_tensor = getattr(model_context.organic_rf_tensors, var_name)
      elif var_name == constants.NON_MEDIA_TREATMENTS:
        old_tensor = model_context.non_media_treatments
      elif var_name == constants.CONTROLS:
        old_tensor = model_context.controls
      elif var_name == constants.REVENUE_PER_KPI:
        old_tensor = model_context.revenue_per_kpi
      elif var_name == constants.TIME:
        old_tensor = backend.to_tensor(
            model_context.input_data.time.values.tolist(), dtype=backend.string
        )
      else:
        continue

      new_tensor = getattr(self, var_name)
      output[var_name] = new_tensor if new_tensor is not None else old_tensor

    return DataTensors(**output)


@dataclasses.dataclass(kw_only=True)
class DistributionTensors(backend.ExtensionType):
  """Container for parameters distributions arguments of Analyzer methods."""

  alpha_m: Union[backend.Tensor, None] = None
  alpha_rf: Union[backend.Tensor, None] = None
  alpha_om: Union[backend.Tensor, None] = None
  alpha_orf: Union[backend.Tensor, None] = None
  ec_m: Union[backend.Tensor, None] = None
  ec_rf: Union[backend.Tensor, None] = None
  ec_om: Union[backend.Tensor, None] = None
  ec_orf: Union[backend.Tensor, None] = None
  slope_m: Union[backend.Tensor, None] = None
  slope_rf: Union[backend.Tensor, None] = None
  slope_om: Union[backend.Tensor, None] = None
  slope_orf: Union[backend.Tensor, None] = None
  beta_gm: Union[backend.Tensor, None] = None
  beta_grf: Union[backend.Tensor, None] = None
  beta_gom: Union[backend.Tensor, None] = None
  beta_gorf: Union[backend.Tensor, None] = None
  mu_t: Union[backend.Tensor, None] = None
  tau_g: Union[backend.Tensor, None] = None
  gamma_gc: Union[backend.Tensor, None] = None
  gamma_gn: Union[backend.Tensor, None] = None


def _scale_tensors_by_multiplier(
    data: DataTensors,
    multiplier: float,
    by_reach: bool,
) -> DataTensors:
  """Gets scaled tensors for incremental outcome calculation.

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


@dataclasses.dataclass(kw_only=True)
class AnalyzerInputs(backend.ExtensionType):
  """Base payload containing DataTensors and resolved indices."""

  tensors: DataTensors
  time_indices: Optional[backend.Tensor] = None
  geo_indices: Optional[backend.Tensor] = None


@dataclasses.dataclass(kw_only=True)
class CounterfactualInputs(AnalyzerInputs):
  """Payload specifically for counterfactual scenarios."""
  non_media_baseline_normalized: Optional[backend.Tensor] = None
  media_selected_times_mask: Optional[tuple[bool, ...]] = None


class DataTensorsBuilder:
  """Translates raw modeling inputs into scaled, execution-ready data tensors.

  Attributes:
    model_context: The Meridian model context.
  """

  def __init__(self, model_context: context.ModelContext):
    """Initializes the instance."""
    self.model_context = model_context

  def _resolve_geo_indices(
      self, selected_geos: Sequence[str] | None
  ) -> backend.Tensor | None:
    """Resolves selected geos to their integer indices.

    Args:
      selected_geos: Sequence of geo names to resolve.

    Returns:
      A tensor of geo indices, or None if selected_geos is None.
    """
    if selected_geos is None:
      return None
    if any(
        geo not in self.model_context.input_data.geo for geo in selected_geos
    ):
      raise ValueError(
          "`selected_geos` must match the geo dimension names from "
          "meridian.InputData."
      )
    geo_indices = [
        i
        for i, x in enumerate(self.model_context.input_data.geo)
        if x in selected_geos
    ]
    return backend.to_tensor(geo_indices, dtype=backend.int32)

  def _resolve_time_indices(
      self,
      selected_times: Sequence[str] | Sequence[bool] | None,
      n_times: int,
      input_times: xr.DataArray,
  ) -> backend.Tensor | None:
    """Resolves selected times to their integer indices.

    Args:
      selected_times: Sequence of time names or booleans to resolve.
      n_times: The number of time periods.
      input_times: The input times to resolve against.

    Returns:
      A tensor of time indices, or None if selected_times is None.
    """
    if selected_times is None:
      return None
    _validate_selected_times(
        selected_times=selected_times,
        input_times=input_times,
        n_times=n_times,
        arg_name="selected_times",
        comparison_arg_name="`tensor`",
    )
    if _is_str_list(selected_times):
      time_indices = [
          i for i, x in enumerate(input_times) if x in selected_times
      ]
    elif _is_bool_list(selected_times):
      time_indices = [i for i, x in enumerate(selected_times) if x]
    else:
      return None
    return backend.to_tensor(time_indices, dtype=backend.int32)

  def _package_inputs(
      self,
      tensors: DataTensors,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      payload_cls: type[AnalyzerInputs] = AnalyzerInputs,
      **kwargs,
  ) -> AnalyzerInputs:
    """Resolves indices and packages tensors into the specified payload class."""
    n_times = (
        tensors.get_modified_times(model_context=self.model_context)
        or self.model_context.n_times
    )

    geo_indices = self._resolve_geo_indices(selected_geos)
    if tensors.time is not None:
      if hasattr(tensors.time, "ndim"):
        input_times = np.asarray(tensors.time).astype(str).tolist()
      else:
        input_times = tensors.time
    else:
      input_times = self.model_context.input_data.time

    time_indices = self._resolve_time_indices(
        selected_times=selected_times,
        n_times=n_times,
        input_times=input_times,
    )

    return payload_cls(
        tensors=tensors,
        time_indices=time_indices,
        geo_indices=geo_indices,
        **kwargs,
    )

  def _build_unscaled_data_tensors(
      self,
      new_data: DataTensors | None = None,
      required_tensors_names: Sequence[str] | None = None,
      optimal_frequency: Sequence[float] | backend.Tensor | float | None = None,
      insert_dummy_media: bool = False,
  ) -> DataTensors:
    """Builds unscaled data tensors, filling missing and applying adjustments."""
    if new_data is None:
      filled_data = DataTensors()
    else:
      filled_data = new_data

    if required_tensors_names is not None:
      filled_data = filled_data.validate_and_fill_missing_data(
          required_tensors_names=required_tensors_names,
          model_context=self.model_context,
      )

    if optimal_frequency is not None:
      optimal_frequency_tensor = backend.to_tensor(
          optimal_frequency, dtype=backend.float_dtype
      )

      new_reach = filled_data.reach
      new_frequency = filled_data.frequency
      new_organic_reach = filled_data.organic_reach
      new_organic_frequency = filled_data.organic_frequency

      if self.model_context.n_rf_channels > 0:
        if filled_data.rf_impressions is not None:
          impressions = filled_data.rf_impressions
        elif (
            filled_data.reach is not None and filled_data.frequency is not None
        ):
          impressions = filled_data.reach * filled_data.frequency
        else:
          impressions = None

        if impressions is not None:
          new_frequency = (
              backend.ones_like(impressions) * optimal_frequency_tensor
          )
          new_reach = impressions / new_frequency

      if self.model_context.n_organic_rf_channels > 0:
        if (
            filled_data.organic_frequency is not None
            and filled_data.organic_reach is not None
        ):
          new_organic_frequency = (
              backend.ones_like(filled_data.organic_frequency)
              * optimal_frequency_tensor
          )
          new_organic_reach = (
              filled_data.organic_reach * filled_data.organic_frequency
          ) / new_organic_frequency

      filled_data = dataclasses.replace(
          filled_data,
          reach=new_reach,
          frequency=new_frequency,
          organic_reach=new_organic_reach,
          organic_frequency=new_organic_frequency,
      )

    if insert_dummy_media and self.model_context.n_media_channels > 0:
      n_media_times = (
          filled_data.get_modified_times(model_context=self.model_context)
          or self.model_context.n_media_times
      )
      n_times = (
          filled_data.get_modified_times(model_context=self.model_context)
          or self.model_context.n_times
      )

      dummy_media = backend.ones(
          (
              self.model_context.n_geos,
              n_media_times,
              self.model_context.n_media_channels,
          ),
          dtype=backend.float_dtype,
      )
      dummy_media_spend = backend.ones(
          (
              self.model_context.n_geos,
              n_times,
              self.model_context.n_media_channels,
          ),
          dtype=backend.float_dtype,
      )

      filled_data = dataclasses.replace(
          filled_data,
          media=dummy_media,
          media_spend=dummy_media_spend,
      )

    filled_data = dataclasses.replace(filled_data, rf_impressions=None)

    return filled_data

  def build_unscaled_inputs(
      self,
      new_data: DataTensors | None = None,
      required_tensors_names: Sequence[str] | None = None,
      optimal_frequency: Sequence[float] | backend.Tensor | float | None = None,
      insert_dummy_media: bool = False,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
  ) -> AnalyzerInputs:
    """Builds unscaled inputs and resolves indices.

    Args:
      new_data: Optional `DataTensors` object.
      required_tensors_names: Optional sequence of tensor names to validate and
        fill.
      optimal_frequency: Optional optimal frequency to scale reach/frequency.
      insert_dummy_media: Whether to insert dummy media and media spend.
      selected_geos: Optional subset of geos to include.
      selected_times: Optional subset of times to include.

    Returns:
      An `AnalyzerInputs` object.
    """
    unscaled = self._build_unscaled_data_tensors(
        new_data=new_data,
        required_tensors_names=required_tensors_names,
        optimal_frequency=optimal_frequency,
        insert_dummy_media=insert_dummy_media,
    )
    return self._package_inputs(
        tensors=unscaled,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

  def build_scaled_inputs(
      self,
      new_data: DataTensors | None = None,
      include_non_paid_channels: bool = True,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
  ) -> AnalyzerInputs:
    """Builds scaled inputs and resolves geo and time indices.

    Args:
      new_data: Optional `DataTensors` object containing new data to scale. If
        `None`, the historical data from the model context is used.
      include_non_paid_channels: Boolean indicating whether to include organic
        media, organic RF, and non-media treatments.
      selected_geos: Optional subset of geos to include.
      selected_times: Optional subset of times to include.

    Returns:
      An `AnalyzerInputs` object.
    """
    required_params = list(constants.PAID_DATA) + [constants.CONTROLS]
    if include_non_paid_channels:
      required_params += list(constants.NON_PAID_DATA)

    unscaled = self._build_unscaled_data_tensors(
        new_data=new_data, required_tensors_names=required_params
    )
    scaled = self._scale_data_tensors(
        unscaled, include_non_paid_channels=include_non_paid_channels
    )
    return self._package_inputs(
        tensors=scaled,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

  def build_counterfactual_inputs(
      self,
      new_data: DataTensors | None = None,
      *,
      scaling_factor: float = 1.0,
      non_media_baseline_values: Sequence[float] | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | Sequence[bool] | None = None,
      media_selected_times: Sequence[str] | Sequence[bool] | None = None,
      by_reach: bool = True,
      include_non_paid_channels: bool = True,
      is_baseline: bool = False,
  ) -> CounterfactualInputs:
    """Builds counterfactual inputs for analyzer.

    Args:
      new_data: Optional `DataTensors` container.
      scaling_factor: Float indicating the factor to scale tensors by.
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel.
      selected_geos: Optional list containing a subset of geos to include.
      selected_times: Optional list containing either a subset of dates to
        include or booleans.
      media_selected_times: Optional list containing either a subset of dates to
        include or booleans.
      by_reach: Boolean indicating whether to scale reach or frequency when rf
        data is available.
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.
      is_baseline: Boolean. If `True`, the non-media treatments are set to their
        baseline values.

    Returns:
      A `CounterfactualInputs` object.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)

    times_modified = False
    if new_data is not None:
      times_modified = (
          new_data.get_modified_times(model_context=self.model_context)
          is not None
      )

    required_params = list(constants.PAID_DATA)
    if include_non_paid_channels:
      required_params += list(constants.NON_PAID_DATA)
    if not times_modified:
      required_params.append(constants.CONTROLS)

    base_unscaled = self._build_unscaled_data_tensors(
        new_data=new_data, required_tensors_names=required_params
    )

    new_n_media_times = base_unscaled.get_modified_times(
        model_context=self.model_context
    )

    if new_n_media_times is None:
      new_n_media_times = self.model_context.n_media_times
      _validate_selected_times(
          selected_times=selected_times,
          input_times=self.model_context.input_data.time,
          n_times=self.model_context.n_times,
          arg_name="selected_times",
          comparison_arg_name="the input data",
      )
      _validate_selected_times(
          selected_times=media_selected_times,
          input_times=self.model_context.input_data.media_time,
          n_times=self.model_context.n_media_times,
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
      resolved_media_selected_times = [True] * new_n_media_times
    else:
      if all(isinstance(time, str) for time in media_selected_times):
        resolved_media_selected_times = [
            x in media_selected_times
            for x in self.model_context.input_data.media_time
        ]
      else:
        resolved_media_selected_times = [bool(x) for x in media_selected_times]

    media_selected_times_mask = tuple(resolved_media_selected_times)

    counterfactual = (
        1 + (scaling_factor - 1) * np.array(resolved_media_selected_times)
    )[:, None]

    if base_unscaled.non_media_treatments is not None:
      if self.model_context.non_media_transformer is None:
        raise ValueError(
            "non_media_transformer is missing in model_context despite "
            "non_media_treatments being present in data."
        )
      non_media_treatments_baseline_scaled = (
          equations.ModelEquations(self.model_context)
          .compute_non_media_treatments_baseline(
              non_media_baseline_values=non_media_baseline_values,
          )
      )
      non_media_treatments_baseline_normalized = (
          self.model_context.non_media_transformer.forward(
              non_media_treatments_baseline_scaled,
              apply_population_scaling=False,
          )
      )
      non_media_treatments_baseline_tensor = backend.broadcast_to(
          backend.to_tensor(
              non_media_treatments_baseline_normalized,
              dtype=backend.float_dtype,
          )[backend.newaxis, backend.newaxis, :],
          base_unscaled.non_media_treatments.shape,
      )
      non_media_baseline_normalized_tensor = backend.to_tensor(
          non_media_treatments_baseline_normalized,
          dtype=backend.float_dtype,
      )
    else:
      non_media_treatments_baseline_tensor = None
      non_media_baseline_normalized_tensor = None

    incremented_unscaled = _scale_tensors_by_multiplier(
        data=base_unscaled,
        multiplier=counterfactual,
        by_reach=by_reach,
    )

    scaled_tensors = self._scale_data_tensors(
        incremented_unscaled,
        include_non_paid_channels=include_non_paid_channels,
    )
    if is_baseline and base_unscaled.non_media_treatments is not None:
      scaled_tensors = dataclasses.replace(
          scaled_tensors,
          non_media_treatments=non_media_treatments_baseline_tensor,
      )

    return self._package_inputs(
        tensors=scaled_tensors,
        selected_geos=selected_geos,
        selected_times=selected_times,
        payload_cls=CounterfactualInputs,
        non_media_baseline_normalized=non_media_baseline_normalized_tensor,
        media_selected_times_mask=media_selected_times_mask,
    )

  def build_baseline_inputs(
      self, non_media_baseline_values: Sequence[float] | None = None
  ) -> AnalyzerInputs:
    """Builds baseline inputs for the analyzer.

    Args:
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is a float which means that the
        fixed value will be used as baseline for the given channel.

    Returns:
      An `AnalyzerInputs` object containing the baseline data tensors.
    """
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)

    ctx = self.model_context
    media = (
        backend.zeros_like(ctx.media_tensors.media)
        if ctx.media_tensors.media is not None
        else None
    )
    reach = (
        backend.zeros_like(ctx.rf_tensors.reach)
        if ctx.rf_tensors.reach is not None
        else None
    )
    organic_media = (
        backend.zeros_like(ctx.organic_media_tensors.organic_media)
        if ctx.organic_media_tensors.organic_media is not None
        else None
    )
    organic_reach = (
        backend.zeros_like(ctx.organic_rf_tensors.organic_reach)
        if ctx.organic_rf_tensors.organic_reach is not None
        else None
    )

    if ctx.non_media_treatments is not None:
      baseline = equations.ModelEquations(
          ctx
      ).compute_non_media_treatments_baseline(non_media_baseline_values)
      baseline_tensor = backend.broadcast_to(
          backend.to_tensor(
              baseline,
              dtype=backend.float_dtype,
          )[backend.newaxis, backend.newaxis, :],
          ctx.non_media_treatments.shape,
      )
      if ctx.model_spec.non_media_population_scaling_id is not None:
        scaling_factors = backend.where(
            ctx.model_spec.non_media_population_scaling_id,
            ctx.population[:, backend.newaxis, backend.newaxis],
            backend.ones_like(ctx.population)[
                :, backend.newaxis, backend.newaxis
            ],
        )
      else:
        scaling_factors = backend.ones_like(ctx.population)[
            :, backend.newaxis, backend.newaxis
        ]
      non_media_treatments = baseline_tensor * scaling_factors
    else:
      non_media_treatments = None

    new_data = DataTensors(
        media=media,
        reach=reach,
        organic_media=organic_media,
        organic_reach=organic_reach,
        non_media_treatments=non_media_treatments,
        controls=ctx.controls,
    )
    return self._package_inputs(tensors=new_data)

  def _scale_data_tensors(
      self, unscaled: DataTensors, include_non_paid_channels: bool = True
  ) -> DataTensors:
    """Gets scaled tensors using given unscaled data.

    Args:
      unscaled: A `DataTensors` container containing unscaled tensors.
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.

    Returns:
      A DataTensors object containing the scaled data tensors.
    """

    def _transform(tensor, transformer):
      return (
          transformer.forward(tensor)
          if tensor is not None and transformer is not None
          else tensor
      )

    media_scaled = _transform(
        unscaled.media, self.model_context.media_tensors.media_transformer
    )
    reach_scaled = _transform(
        unscaled.reach, self.model_context.rf_tensors.reach_transformer
    )
    controls_scaled = _transform(
        unscaled.controls, self.model_context.controls_transformer
    )

    if include_non_paid_channels:
      organic_media_scaled = _transform(
          unscaled.organic_media,
          self.model_context.organic_media_tensors.organic_media_transformer,
      )
      organic_reach_scaled = _transform(
          unscaled.organic_reach,
          self.model_context.organic_rf_tensors.organic_reach_transformer,
      )
      non_media_treatments_normalized = _transform(
          unscaled.non_media_treatments,
          self.model_context.non_media_transformer,
      )
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=unscaled.frequency,
          organic_media=organic_media_scaled,
          organic_reach=organic_reach_scaled,
          organic_frequency=unscaled.organic_frequency,
          non_media_treatments=non_media_treatments_normalized,
          controls=controls_scaled,
          revenue_per_kpi=unscaled.revenue_per_kpi,
          time=unscaled.time,
      )
    else:
      return DataTensors(
          media=media_scaled,
          reach=reach_scaled,
          frequency=unscaled.frequency,
          controls=controls_scaled,
          revenue_per_kpi=unscaled.revenue_per_kpi,
          time=unscaled.time,
      )
