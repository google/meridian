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

from collections.abc import Iterable, Sequence
import dataclasses
import datetime as dt
import numbers
from typing import Any, Optional, Union
import warnings

from meridian import backend
from meridian import constants
from meridian.data import time_coordinates as tc
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
    "get_model_context",
    "normalize_date_str",
    "normalize_times_set",
)


# TODO: Remove this method.
def get_model_context(
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


def normalize_date_str(
    time_val: tc.Date | xr.DataArray | backend.Tensor,
) -> str:
  """Extracts the 'YYYY-MM-DD' prefix string from a date coordinate or string.

  Args:
    time_val: A polymorphic `Date` (`tc.Date`), a 0-D xarray `DataArray`
      coordinate scalar (which may wrap a date string, datetime, or NumPy
      timestamp), a `backend.Tensor` string scalar, or any other date coordinate
      scalar (e.g. `pd.Timestamp`).

  Returns:
    The normalized 'YYYY-MM-DD' date string.
  """
  val = time_val.numpy() if hasattr(time_val, "numpy") else time_val
  val = val.item() if hasattr(val, "item") else val
  if isinstance(val, bytes):
    val = val.decode("utf-8")
  if isinstance(val, str):
    val = val[:10]
  return tc.normalize_date(val).strftime(constants.DATE_FORMAT)  # pyrefly: ignore[bad-argument-type]


def normalize_times_set(times: Iterable[Any]) -> set[str]:
  """Returns a set of normalized 'YYYY-MM-DD' date strings from times."""
  return {normalize_date_str(x) for x in times}


def _is_normalized_subset(
    subset: Sequence[Any],
    superset: Sequence[Any],
) -> bool:
  """Returns True if normalized date strings in subset are in superset."""
  try:
    return normalize_times_set(subset) <= normalize_times_set(superset)
  except (ValueError, TypeError):
    return False


def _validate_selected_times(
    selected_times: Sequence[str] | None,
    input_times: xr.DataArray | Sequence[str],
    *,
    arg_name: str,
) -> None:
  """Raises an error if selected_times is invalid.

  Args:
    selected_times: Optional sequence of time names to resolve.
    input_times: Target time period coordinates.
    arg_name: Name of the `selected_times` argument.

  Raises:
    ValueError: A `ValueError` is raised when coordinates in `selected_times` do
      not match time coordinates in `input_times` or is not a list of strings.
  """
  if not selected_times:
    return
  if not all(isinstance(item, str) for item in selected_times):
    raise ValueError(f"`{arg_name}` must be a list of strings.")
  if not _is_normalized_subset(selected_times, input_times):  # pyrefly: ignore[bad-argument-type]
    start_date = normalize_date_str(input_times[0])
    end_date = normalize_date_str(input_times[-1])
    raise ValueError(
        f"`{arg_name}` must match the time dimension coordinates from"
        f" '{start_date}' to '{end_date}'."
    )


def _validate_time_coordinates(time: Sequence[str] | None) -> None:
  """Validates that time follows the same format and spacing rules as InputData.time."""
  if time is None:
    return
  for t in time:
    try:
      dt.datetime.strptime(t, constants.DATE_FORMAT)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          f"Invalid time label: {t!r}. Expected format:"
          f" '{constants.DATE_FORMAT}'"
      ) from exc

  if len(time) > 1:
    time_coords = tc.TimeCoordinates.from_dates(time)
    try:
      _ = time_coords.interval_days
    except ValueError as exc:
      raise ValueError("Time coordinates must be regularly spaced.") from exc


@dataclasses.dataclass(kw_only=True)
class DataTensors(backend.ExtensionType):  # pyrefly: ignore[invalid-inheritance]
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
    time: Optional sequence or array of date coordinates (strings in
      "YYYY-MM-DD" format, `Date` objects, or datetime objects) corresponding to
      time dimension `T`. Required if any tensor has a modified time dimension
      `T` differing from historical model dimensions. If omitted and tensor
      shapes match the original model dimensions, `time` is automatically
      populated from the model's historical time coordinates.
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
  time: Union[tuple[str, ...], None]

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
        tuple(normalize_date_str(t) for t in time) if time is not None else None
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
        if not bool(np.all(backend.to_tensor(backend.equal(a, b)))):  # pyrefly: ignore[no-matching-overload]
          return False
      except (ValueError, TypeError):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
          if not np.array_equal(a, b):
            return False
        elif a != b:
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

    WARNING: This method is deprecated and will be removed in a future version.
    Use `DataTensorsBuilder.get_modified_times` instead.

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
    warnings.warn(
        "DataTensors.get_modified_times is deprecated and will be removed in a"
        " future version. Use DataTensorsBuilder.get_modified_times instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ctx = get_model_context(meridian, model_context)
    return DataTensorsBuilder(ctx).get_modified_times(self)

  @property
  def time_coordinates(self) -> tc.TimeCoordinates | None:
    """Returns TimeCoordinates instance constructed from self.time."""
    if self.time is None:
      return None
    return tc.TimeCoordinates.from_dates(self.time)

  def expand_selected_time_dims(
      self,
      start_date: tc.Date = None,
      end_date: tc.Date = None,
  ) -> list[str] | None:
    """Validates and returns time dimension values based on the selected times.

    If both `start_date` and `end_date` are None, returns None. If specified,
    both `start_date` and `end_date` are inclusive, and must be present in
    `self.time`.

    Args:
      start_date: Start date of the selected time period. If `None`, implies the
        earliest time dimension value in `self.time`.
      end_date: End date of the selected time period. If `None`, implies the
        latest time dimension value in `self.time`.

    Returns:
      A list of time dimension values (as 'YYYY-MM-DD' strings) in `self.time`
      within the selected time period, or `None` if both arguments are `None`,
      or if `start_date` and `end_date` correspond to the entire time range in
      `self.time`.

    Raises:
      `ValueError` if `start_date` or `end_date` is not in `self.time`.
    """
    if self.time is None or (start_date is None and end_date is None):
      return None
    expanded = self.time_coordinates.expand_selected_time_dims(  # pyrefly: ignore[missing-attribute]
        start_date=start_date, end_date=end_date
    )
    if expanded is None:
      return None
    return [date.strftime(constants.DATE_FORMAT) for date in expanded]

  def filter_fields(self, fields: Sequence[str]) -> Self:
    """Returns a new DataTensors object with only the specified fields."""
    output = {}
    for field in fields:
      output[field] = getattr(self, field)
    if self.time is not None and any(v is not None for v in output.values()):
      output[constants.TIME] = self.time
    return DataTensors(**output)

  def validate_and_fill_missing_data(
      self,
      required_tensors_names: Sequence[str],
      meridian: Any | None = None,
      model_context: context.ModelContext | None = None,
      allow_modified_times: bool = True,
  ) -> Self:
    """Fills missing data tensors with their original values from the model.

    WARNING: This method is deprecated and will be removed in a future version.
    Use `DataTensorsBuilder.build_unscaled_inputs` instead.

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
    warnings.warn(
        "DataTensors.validate_and_fill_missing_data is deprecated and will be"
        " removed in a future version. Use"
        " DataTensorsBuilder.build_unscaled_inputs instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ctx = get_model_context(meridian, model_context)
    # pylint: disable=protected-access
    return DataTensorsBuilder(ctx)._validate_and_fill_missing_data(
        self, required_tensors_names, allow_modified_times
    )
    # pylint: enable=protected-access

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
        if not isinstance(tensor, tuple):
          raise ValueError(f"New `{field.name}` must be a tuple.")
      else:
        _check_n_dims(tensor, field.name, 3)


@dataclasses.dataclass(kw_only=True)
class DistributionTensors(backend.ExtensionType):  # pyrefly: ignore[invalid-inheritance]
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
    incremented_data[constants.MEDIA] = data.media * multiplier  # pyrefly: ignore[unsupported-operation]
  if data.reach is not None and data.frequency is not None:
    if by_reach:
      incremented_data[constants.REACH] = data.reach * multiplier  # pyrefly: ignore[unsupported-operation]
      incremented_data[constants.FREQUENCY] = data.frequency
    else:
      incremented_data[constants.REACH] = data.reach
      incremented_data[constants.FREQUENCY] = data.frequency * multiplier  # pyrefly: ignore[unsupported-operation]
  if data.organic_media is not None:
    incremented_data[constants.ORGANIC_MEDIA] = data.organic_media * multiplier  # pyrefly: ignore[unsupported-operation]
  if data.organic_reach is not None and data.organic_frequency is not None:
    if by_reach:
      incremented_data[constants.ORGANIC_REACH] = (
          data.organic_reach * multiplier  # pyrefly: ignore[unsupported-operation]
      )
      incremented_data[constants.ORGANIC_FREQUENCY] = data.organic_frequency
    else:
      incremented_data[constants.ORGANIC_REACH] = data.organic_reach
      incremented_data[constants.ORGANIC_FREQUENCY] = (
          data.organic_frequency * multiplier  # pyrefly: ignore[unsupported-operation]
      )

  # Include the original data that does not get scaled.
  incremented_data[constants.NON_MEDIA_TREATMENTS] = data.non_media_treatments
  incremented_data[constants.MEDIA_SPEND] = data.media_spend
  incremented_data[constants.RF_SPEND] = data.rf_spend
  incremented_data[constants.CONTROLS] = data.controls
  incremented_data[constants.REVENUE_PER_KPI] = data.revenue_per_kpi

  return dataclasses.replace(data, **incremented_data)


@dataclasses.dataclass(kw_only=True)
class AnalyzerInputs(backend.ExtensionType):  # pyrefly: ignore[invalid-inheritance]
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

  def get_modified_times(self, data: DataTensors) -> int | None:
    """Returns `n_times` of any tensor where `n_times` has been modified.

    This method compares the time dimensions of the attributes in the
    `DataTensors` object with the corresponding tensors in the `model_context`
    object. If any of the time dimensions are different, then this method
    returns the modified number of time periods of the tensor in the
    `DataTensors` object. If all time dimensions are the same, returns `None`.

    Args:
      data: A DataTensors object to check.

    Returns:
      The `n_times` of any tensor where `n_times` is different from the times
      of the corresponding tensor in the `model_context` object. If all time
      dimensions are the same, returns `None`.
    """
    for field in dataclasses.fields(data):
      new_tensor = getattr(data, field.name)
      if field.name == constants.TIME:
        continue
      elif field.name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(self.model_context.rf_tensors, field.name)
      else:
        old_tensor = getattr(self.model_context.input_data, field.name, None)
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

  def _validate_and_fill_missing_data(
      self,
      data: DataTensors,
      required_tensors_names: Sequence[str],
      allow_modified_times: bool = True,
  ) -> DataTensors:
    """Fills missing data tensors with their original values from the model."""
    self._validate_correct_variables_filled(
        data=data,
        required_variables=required_tensors_names,
    )
    self._validate_geo_dims(
        data=data,
        required_fields=required_tensors_names,
    )
    self._validate_channel_dims(
        data=data,
        required_fields=required_tensors_names,
    )
    if allow_modified_times:
      self._validate_time_dims_flexible_times(
          data=data,
          required_fields=required_tensors_names,
      )
    else:
      self._validate_time_dims(
          data=data,
          required_fields=required_tensors_names,
      )

    return self._fill_default_values(
        data=data,
        required_fields=required_tensors_names,
    )

  def _validate_correct_variables_filled(
      self,
      data: DataTensors,
      required_variables: Sequence[str],
  ) -> None:
    """Validates that the correct variables are filled."""
    for field in dataclasses.fields(data):
      tensor = getattr(data, field.name)
      if tensor is None:
        continue
      if field.name == "time":
        continue
      if field.name not in required_variables:
        warnings.warn(
            f"A `{field.name}` value was passed in the `new_data` argument. "
            "This is not supported and will be ignored."
        )
      if field.name in required_variables:
        if field.name == constants.RF_IMPRESSIONS:
          if self.model_context.n_rf_channels == 0:
            raise ValueError(
                "New `rf_impressions` is not allowed because there are no R&F"
                " channels in the Meridian model."
            )
        elif getattr(self.model_context.input_data, field.name) is None:
          raise ValueError(
              f"New `{field.name}` is not allowed because the input data to the"
              f" Meridian model does not contain `{field.name}`."
          )

  def _validate_geo_dims(
      self,
      data: DataTensors,
      required_fields: Sequence[str],
  ) -> None:
    """Validates the geo dimension of the specified data variables."""
    for var_name in required_fields:
      if var_name == constants.TIME:
        continue
      new_tensor = getattr(data, var_name)
      if (
          new_tensor is not None
          and new_tensor.shape[0] != self.model_context.n_geos
      ):
        # Skip spend data with only 1 dimension.
        if new_tensor.ndim == 1:
          continue
        raise ValueError(
            f"New `{var_name}` is expected to have {self.model_context.n_geos}"
            f" geos. Found {new_tensor.shape[0]} geos."
        )

  def _validate_channel_dims(
      self,
      data: DataTensors,
      required_fields: Sequence[str],
  ) -> None:
    """Validates the channel dimension of the specified data variables."""
    for var_name in required_fields:
      if var_name in [constants.REVENUE_PER_KPI, constants.TIME]:
        continue
      new_tensor = getattr(data, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(self.model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(self.model_context.input_data, var_name)
      if new_tensor is not None:
        assert old_tensor is not None
        if new_tensor.shape[-1] != old_tensor.shape[-1]:
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[-1]}"
              f" channels. Found {new_tensor.shape[-1]} channels."
          )

  def _validate_time_dims(
      self,
      data: DataTensors,
      required_fields: Sequence[str],
  ) -> None:
    """Validates the time dimension of the specified data variables."""
    if data.time is not None:
      _validate_time_coordinates(data.time)

    for var_name in required_fields:
      new_tensor = getattr(data, var_name)
      if var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(self.model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(self.model_context.input_data, var_name, None)

      if old_tensor is None:
        continue

      # Skip spend data with only 1 dimension of (n_channels).
      if (
          var_name in [constants.MEDIA_SPEND, constants.RF_SPEND]
          and new_tensor is not None
          and new_tensor.ndim == 1
      ):
        continue

      if new_tensor is not None:
        if var_name == constants.TIME:
          if len(new_tensor) != self.model_context.n_times:
            raise ValueError(
                f"New `{var_name}` is expected to have"
                f" {self.model_context.n_times} time periods. Found"
                f" {len(new_tensor)} time periods."
            )
        elif new_tensor.ndim > 1 and new_tensor.shape[1] != old_tensor.shape[1]:
          raise ValueError(
              f"New `{var_name}` is expected to have {old_tensor.shape[1]}"
              f" time periods. Found {new_tensor.shape[1]} time periods."
          )

  def _validate_time_dims_flexible_times(
      self,
      data: DataTensors,
      required_fields: Sequence[str],
  ) -> None:
    """Validates the time dimension for the flexible times case."""
    new_n_times = self.get_modified_times(data)
    # If no times were modified, validate against historical time dimensions.
    if new_n_times is None:
      self._validate_time_dims(data=data, required_fields=required_fields)
      return

    if data.time is None:
      raise ValueError(
          "`time` must be provided in `new_data` if any time dimension in"
          " `new_data` is modified."
      )

    if len(data.time) != new_n_times:
      raise ValueError(
          "If the time dimension of any variable in `new_data` is "
          "modified, then all variables must be provided with the same "
          f"number of time periods. `time` has {len(data.time)} "
          "time periods, which does not match the modified number of time "
          f"periods, {new_n_times}."
      )

    _validate_time_coordinates(data.time)

    missing_params = []
    for var_name in required_fields:
      new_tensor = getattr(data, var_name)
      if var_name == constants.TIME:
        continue
      elif var_name == constants.RF_IMPRESSIONS:
        old_tensor = getattr(self.model_context.rf_tensors, var_name)
      else:
        old_tensor = getattr(self.model_context.input_data, var_name, None)

      if old_tensor is None:
        continue

      if new_tensor is None:
        missing_params.append(var_name)
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
      data: DataTensors,
      required_fields: Sequence[str],
  ) -> DataTensors:
    """Fills default values and returns a new DataTensors object."""
    output = {}
    if data.time is not None:
      output[constants.TIME] = data.time
    else:
      output[constants.TIME] = tuple(
          normalize_date_str(t)
          for t in self.model_context.input_data.time.values
      )
    for field in dataclasses.fields(data):
      var_name = field.name
      if var_name == constants.TIME:
        continue
      if var_name not in required_fields:
        continue

      if hasattr(self.model_context.media_tensors, var_name):
        old_tensor = getattr(self.model_context.media_tensors, var_name)
      elif hasattr(self.model_context.rf_tensors, var_name):
        old_tensor = getattr(self.model_context.rf_tensors, var_name)
      elif hasattr(self.model_context.organic_media_tensors, var_name):
        old_tensor = getattr(self.model_context.organic_media_tensors, var_name)
      elif hasattr(self.model_context.organic_rf_tensors, var_name):
        old_tensor = getattr(self.model_context.organic_rf_tensors, var_name)
      elif var_name == constants.NON_MEDIA_TREATMENTS:
        old_tensor = self.model_context.non_media_treatments
      elif var_name == constants.CONTROLS:
        old_tensor = self.model_context.controls
      elif var_name == constants.REVENUE_PER_KPI:
        old_tensor = self.model_context.revenue_per_kpi
      else:
        continue

      new_tensor = getattr(data, var_name)
      output[var_name] = new_tensor if new_tensor is not None else old_tensor

    return DataTensors(**output)

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
      selected_times: Sequence[str] | None,
      input_times: xr.DataArray | Sequence[str] | Any,
  ) -> backend.Tensor | None:
    """Resolves selected times to their integer indices.

    Args:
      selected_times: Sequence of time names to resolve.
      input_times: The input times to resolve against.

    Returns:
      A tensor of time indices, or None if selected_times is None.
    """
    if selected_times is None:
      return None
    _validate_selected_times(
        selected_times=selected_times,
        input_times=input_times,
        arg_name="selected_times",
    )
    selected_times_set = normalize_times_set(selected_times)
    time_indices = [
        i
        for i, x in enumerate(input_times)
        if normalize_date_str(x) in selected_times_set
    ]
    return backend.to_tensor(time_indices, dtype=backend.int32)

  def _package_inputs(
      self,
      tensors: DataTensors,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      payload_cls: type[AnalyzerInputs] = AnalyzerInputs,
      **kwargs,
  ) -> AnalyzerInputs:
    """Resolves indices and packages tensors into the specified payload class."""
    geo_indices = self._resolve_geo_indices(selected_geos)
    if tensors.time is not None:
      input_times = tensors.time
    else:
      input_times = self.model_context.input_data.time

    time_indices = self._resolve_time_indices(
        selected_times=selected_times,
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
      filled_data = self._validate_and_fill_missing_data(
          data=filled_data,
          required_tensors_names=required_tensors_names,
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
          impressions = filled_data.reach * filled_data.frequency  # pyrefly: ignore[unsupported-operation]
        else:
          impressions = None

        if impressions is not None:
          new_frequency = (
              backend.ones_like(impressions) * optimal_frequency_tensor  # pyrefly: ignore[unsupported-operation]
          )
          new_reach = impressions / new_frequency  # pyrefly: ignore[unsupported-operation]

      if self.model_context.n_organic_rf_channels > 0:
        if (
            filled_data.organic_frequency is not None
            and filled_data.organic_reach is not None
        ):
          new_organic_frequency = (
              backend.ones_like(filled_data.organic_frequency)  # pyrefly: ignore[unsupported-operation]
              * optimal_frequency_tensor
          )
          new_organic_reach = (
              filled_data.organic_reach * filled_data.organic_frequency  # pyrefly: ignore[unsupported-operation]
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
          self.get_modified_times(filled_data)
          or self.model_context.n_media_times
      )
      n_times = (
          self.get_modified_times(filled_data) or self.model_context.n_times
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
      selected_times: Sequence[str] | None = None,
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
      selected_times: Sequence[str] | None = None,
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

  def _resolve_and_validate_counterfactual_inputs(
      self,
      new_data: DataTensors | None = None,
      non_media_baseline_values: Sequence[float] | None = None,
      selected_times: Sequence[str] | None = None,
      media_selected_times: Sequence[str] | None = None,
      include_non_paid_channels: bool = True,
  ) -> tuple[DataTensors, int | None]:
    """Resolves unscaled tensors, gets modified times, and performs validation checks."""
    _validate_non_media_baseline_values_numbers(non_media_baseline_values)

    times_modified = False
    if new_data is not None:
      times_modified = self.get_modified_times(new_data) is not None

    required_params = list(constants.PAID_DATA)
    if include_non_paid_channels:
      required_params += list(constants.NON_PAID_DATA)
    if not times_modified:
      required_params.append(constants.CONTROLS)

    base_unscaled = self._build_unscaled_data_tensors(
        new_data=new_data, required_tensors_names=required_params
    )

    new_n_media_times = self.get_modified_times(base_unscaled)

    if new_n_media_times is not None:
      assert base_unscaled.time is not None
      time_coords = base_unscaled.time
      media_time_coords = base_unscaled.time
    else:
      time_coords = self.model_context.input_data.time
      media_time_coords = self.model_context.input_data.media_time

    _validate_selected_times(
        selected_times=selected_times,
        input_times=time_coords,
        arg_name="selected_times",
    )
    _validate_selected_times(
        selected_times=media_selected_times,
        input_times=media_time_coords,
        arg_name="media_selected_times",
    )
    return base_unscaled, new_n_media_times

  def build_counterfactual_inputs(
      self,
      new_data: DataTensors | None = None,
      *,
      scaling_factor: float = 1.0,
      non_media_baseline_values: Sequence[float] | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      media_selected_times: Sequence[str] | None = None,
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
      selected_times: Optional list containing a subset of dates to include.
      media_selected_times: Optional list containing a subset of dates to
        include.
      by_reach: Boolean indicating whether to scale reach or frequency when rf
        data is available.
      include_non_paid_channels: Boolean. If `True`, organic media, organic RF
        and non-media treatments data is included in the output.
      is_baseline: Boolean. If `True`, the non-media treatments are set to their
        baseline values.

    Returns:
      A `CounterfactualInputs` object.
    """
    base_unscaled, new_n_media_times = (
        self._resolve_and_validate_counterfactual_inputs(
            new_data=new_data,
            non_media_baseline_values=non_media_baseline_values,
            selected_times=selected_times,
            media_selected_times=media_selected_times,
            include_non_paid_channels=include_non_paid_channels,
        )
    )

    if new_n_media_times is None:
      new_n_media_times = self.model_context.n_media_times
      media_times = self.model_context.input_data.media_time
    else:
      new_time = base_unscaled.time
      media_times = (
          new_time[-new_n_media_times:] if new_time is not None else ()
      )

    if media_selected_times is None:
      resolved_media_selected_times = [True] * new_n_media_times
    else:
      media_selected_set = normalize_times_set(media_selected_times)
      resolved_media_selected_times = [
          normalize_date_str(x) in media_selected_set for x in media_times
      ]

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
      non_media_treatments_baseline_scaled = equations.ModelEquations(
          self.model_context
      ).compute_non_media_treatments_baseline(
          non_media_baseline_values=non_media_baseline_values,
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
        multiplier=counterfactual,  # pyrefly: ignore[bad-argument-type]
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
