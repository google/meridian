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

"""Meridian EDA Engine."""

import functools
from typing import Callable, Dict, Optional, TypeAlias
from meridian import constants
from meridian.model import model
from meridian.model import transformers
import numpy as np
import tensorflow as tf
import xarray as xr


_DEFAULT_DA_VAR_AGG_FUNCTION = np.sum
AggregationMap: TypeAlias = Dict[str, Callable[[xr.DataArray], np.ndarray]]


class EDAEngine:
  """Meridian EDA Engine."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian

  @functools.cached_property
  def controls_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.controls is None:
      return None
    controls_scaled_da = _data_array_like(
        da=self._meridian.input_data.controls,
        values=self._meridian.controls_scaled,
    )
    return controls_scaled_da

  @functools.cached_property
  def media_raw_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    return self._truncate_media_time(self._meridian.input_data.media)

  @functools.cached_property
  def media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    media_scaled_da = _data_array_like(
        da=self._meridian.input_data.media,
        values=self._meridian.media_tensors.media_scaled,
    )
    return self._truncate_media_time(media_scaled_da)

  @functools.cached_property
  def media_spend_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media_spend is None:
      return None
    media_spend_da = _data_array_like(
        da=self._meridian.input_data.media_spend,
        values=self._meridian.media_tensors.media_spend,
    )
    # No need to truncate the media time for media spend.
    return media_spend_da

  @functools.cached_property
  def media_raw_da_national(self) -> xr.DataArray | None:
    if self.media_raw_da is None:
      return None
    if self._meridian.is_national:
      return self.media_raw_da
    else:
      return self._aggregate_and_scale_geo_da(
          self.media_raw_da,
          None,
      )

  @functools.cached_property
  def media_scaled_da_national(self) -> xr.DataArray | None:
    if self.media_scaled_da is None:
      return None
    if self._meridian.is_national:
      return self.media_scaled_da
    else:
      return self._aggregate_and_scale_geo_da(
          self.media_raw_da,
          transformers.MediaTransformer,
      )

  @functools.cached_property
  def organic_media_raw_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.organic_media is None:
      return None
    return self._truncate_media_time(self._meridian.input_data.organic_media)

  @functools.cached_property
  def organic_media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.organic_media is None:
      return None
    organic_media_scaled_da = _data_array_like(
        da=self._meridian.input_data.organic_media,
        values=self._meridian.organic_media_tensors.organic_media_scaled,
    )
    return self._truncate_media_time(organic_media_scaled_da)

  @functools.cached_property
  def organic_media_raw_da_national(self) -> xr.DataArray | None:
    if self.organic_media_raw_da is None:
      return None
    if self._meridian.is_national:
      return self.organic_media_raw_da
    else:
      return self._aggregate_and_scale_geo_da(self.organic_media_raw_da, None)

  @functools.cached_property
  def organic_media_scaled_da_national(self) -> xr.DataArray | None:
    if self.organic_media_scaled_da is None:
      return None
    if self._meridian.is_national:
      return self.organic_media_scaled_da
    else:
      return self._aggregate_and_scale_geo_da(
          self.organic_media_raw_da,
          transformers.MediaTransformer,
      )

  @functools.cached_property
  def non_media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.non_media_treatments is None:
      return None
    non_media_scaled_da = _data_array_like(
        da=self._meridian.input_data.non_media_treatments,
        values=self._meridian.non_media_treatments_normalized,
    )
    return non_media_scaled_da

  @functools.cached_property
  def rf_spend_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.rf_spend is None:
      return None
    rf_spend_da = _data_array_like(
        da=self._meridian.input_data.rf_spend,
        values=self._meridian.rf_tensors.rf_spend,
    )
    return rf_spend_da

  @functools.cached_property
  def rf_spend_da_national(self) -> xr.DataArray | None:
    if self._meridian.input_data.rf_spend is None:
      return None
    if self._meridian.is_national:
      return self.rf_spend_da
    else:
      return self._aggregate_and_scale_geo_da(
          self._meridian.input_data.rf_spend, None
      )

  def _truncate_media_time(self, da: xr.DataArray) -> xr.DataArray:
    """Truncates the first `start` elements of the media time of a variable."""
    # This should not happen. If it does, it means this function is mis-used.
    if constants.MEDIA_TIME not in da.coords:
      raise ValueError(
          f'Variable does not have a media time coordinate: {da.name}.'
      )

    start = self._meridian.n_media_times - self._meridian.n_times
    da = da.copy().isel({constants.MEDIA_TIME: slice(start, None)})
    da = da.rename({constants.MEDIA_TIME: constants.TIME})
    return da

  def _scale_xarray(
      self,
      xarray: xr.DataArray,
      transformer_class: Optional[type[transformers.TensorTransformer]],
      population: tf.Tensor = tf.constant([1.0], dtype=tf.float32),
  ):
    """Scales xarray values with a TensorTransformer."""
    if transformer_class is None:
      return xarray
    elif transformer_class is transformers.CenteringAndScalingTransformer:
      xarray_transformer = transformers.CenteringAndScalingTransformer(
          tensor=xarray.values, population=population
      )
    elif transformer_class is transformers.MediaTransformer:
      xarray_transformer = transformers.MediaTransformer(
          media=xarray.values, population=population
      )
    else:
      raise ValueError(
          'Unknown transformer class: '
          + str(transformer_class)
          + '.\nMust be one of: CenteringAndScalingTransformer or'
          ' MediaTransformer.'
      )
    xarray.values = xarray_transformer.forward(xarray.values)
    return xarray

  def _aggregate_variables(
      self,
      da_geo: xr.DataArray,
      channel_dim: str,
      da_var_agg_map: AggregationMap,
      keepdims: bool = True,
  ) -> xr.DataArray:
    """Aggregates variables within a DataArray based on user-defined functions.

    Args:
      da_geo: The geo-level DataArray containing multiple variables along
        channel_dim.
      channel_dim: The name of the dimension coordinate to aggregate over (e.g.,
        constants.CONTROL_VARIABLE).
      da_var_agg_map: A dictionary mapping dataArray variable names to
        aggregation functions.
      keepdims: Whether to keep the dimensions of the aggregated DataArray.

    Returns:
      An xr.DataArray aggregated to the national level, with each variable
      aggregated according to the da_var_agg_map.
    """
    agg_results = []
    for var_name in da_geo[channel_dim].values:
      var_data = da_geo.sel({channel_dim: var_name})
      agg_func = da_var_agg_map.get(var_name, _DEFAULT_DA_VAR_AGG_FUNCTION)
      # Apply the aggregation function over the GEO dimension
      aggregated_data = var_data.reduce(
          agg_func, dim=constants.GEO, keepdims=keepdims
      )
      agg_results.append(aggregated_data)

    # Combine the aggregated variables back into a single DataArray
    return xr.concat(agg_results, dim=channel_dim).transpose(..., channel_dim)

  def _aggregate_and_scale_geo_da(
      self,
      da_geo: xr.DataArray,
      transformer_class: Optional[type[transformers.TensorTransformer]],
      channel_dim: Optional[str] = None,
      da_var_agg_map: Optional[AggregationMap] = None,
  ) -> xr.DataArray:
    """Aggregate geo-level xr.DataArray to national level and then scale values.

    Args:
      da_geo: The geo-level DataArray to convert.
      transformer_class: The TensorTransformer class to apply after summing to
        national level. Must be None, CenteringAndScalingTransformer, or
        MediaTransformer.
      channel_dim: The name of the dimension coordinate to aggregate over (e.g.,
        constants.CONTROL_VARIABLE). If None, standard sum aggregation is used.
      da_var_agg_map: A dictionary mapping dataArray variable names to
        aggregation functions. Used only if channel_dim is not None.

    Returns:
      An xr.DataArray representing the aggregated and scaled national-level
        data.
    """
    temp_geo_dim = constants.NATIONAL_MODEL_DEFAULT_GEO_NAME

    if da_var_agg_map is None:
      da_var_agg_map = {}

    if channel_dim is not None:
      da_national = self._aggregate_variables(
          da_geo, channel_dim, da_var_agg_map
      )
    else:
      # Default to sum aggregation if no channel dimension is provided
      da_national = da_geo.sum(
          dim=constants.GEO, keepdims=True, skipna=False, keep_attrs=True
      )

    da_national = da_national.assign_coords({constants.GEO: [temp_geo_dim]})
    da_national.values = tf.cast(da_national.values, tf.float32)
    da_national = self._scale_xarray(da_national, transformer_class)

    return da_national.sel({constants.GEO: temp_geo_dim}, drop=True)


def _data_array_like(
    *, da: xr.DataArray, values: np.ndarray | tf.Tensor
) -> xr.DataArray:
  """Returns a DataArray from `values` with the same structure as `da`.

  Args:
    da: The DataArray whose structure (dimensions, coordinates, name, and attrs)
      will be used for the new DataArray.
    values: The numpy array or tensorflow tensor to use as the values for the
      new DataArray.

  Returns:
    A new DataArray with the provided `values` and the same structure as `da`.
  """
  return xr.DataArray(
      values,
      coords=da.coords,
      dims=da.dims,
      name=da.name,
      attrs=da.attrs,
  )
