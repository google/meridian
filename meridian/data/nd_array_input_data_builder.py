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

"""An implementation of `InputDataBuilder` with n-dimensional array primitives."""

import logging
import warnings
from meridian import constants
from meridian.data import input_data_builder
import numpy as np
import xarray as xr


__all__ = [
    'NDArrayInputDataBuilder',
]


class NDArrayInputDataBuilder(input_data_builder.InputDataBuilder):
  """Builds `InputData` from n-dimensional arrays."""

  # Unlike `DataFrameInputDataBuilder`, each piecemeal data has no coordinate
  # information; they're purely data values. It's up to the user to provide
  # coordinates with setter methods from the abstract base class above.
  # Validation is done on each piece w.r.t. dimensional consistency by
  # shape alone.

  def with_kpi(self, nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    """Reads KPI data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos, n_time)`
    -  `(n_time,)` or `(1, n_time)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the KPI data from.

    Returns:
      The `NDArrayInputDataBuilder` with the added KPI data.
    """
    ### Validate ###
    self._validate_coords()
    self._validate_shape(nd)

    ### Transform ###
    self.kpi = xr.DataArray(
        nd,
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
        },
        name=constants.KPI,
    )
    return self

  def with_controls(
      self, nd: np.ndarray, control_names: list[str]
  ) -> 'NDArrayInputDataBuilder':
    """Reads controls data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos, n_time, n_controls)`
    -  `(n_time, n_controls)` or `(1, n_time, n_controls)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the controls data from.
      control_names: The names of the control variables.

    Returns:
      The `NDArrayInputDataBuilder` with the added controls data.
    """
    ### Validate ###
    self._validate_coords()
    self._validate_shape(nd, control_names)

    ### Transform ###
    self.controls = xr.DataArray(
        nd,
        dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
            constants.CONTROL_VARIABLE: control_names,
        },
        name=constants.CONTROLS,
    )
    return self

  def with_population(self, nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    """Reads population data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos,)`

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the population data from.

    Returns:
      The `NDArrayInputDataBuilder` with the added population data.
    """
    ### Validate ###
    self._validate_coords(is_population=True)
    self._validate_shape(nd, is_population=True)
    ### Transform ###
    self.population = xr.DataArray(
        nd,
        dims=[constants.GEO],
        coords={constants.GEO: self.geos},
        name=constants.POPULATION,
    )

    return self

  def with_revenue_per_kpi(self, nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    """Reads Revenue per KPI data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos, n_time)`
    -  `(n_time,)` or `(1, n_time)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the Reevenue per KPI data from.

    Returns:
      The `NDArrayInputDataBuilder` with the added Revenue per KPI data.
    """
    ### Validate ###
    self._validate_coords()
    self._validate_shape(nd)
    revenue_per_kpi_nd = self._check_revenue_per_kpi_defaults(nd)

    ### Transform ###
    self.revenue_per_kpi = xr.DataArray(
        revenue_per_kpi_nd,
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
        },
        name=constants.REVENUE_PER_KPI,
    )
    return self

  def with_media(
      self, m_nd: np.ndarray, ms_nd: np.ndarray, media_channels: list[str]
  ) -> 'NDArrayInputDataBuilder':
    """Reads media and media spend data from the ndarrays.

    `m_nd` must be given with the shape:
    -  `(n_geos, n_media_times, n_media_channels)`
    -  `(n_media_times, n_media_channels)` or `(1, n_media_times,
       n_media_channels)` for national model.

    `ms_nd` must be given with the shape:
    -  `(n_geos, n_times, n_media_channels)`
    -  `(n_times, n_media_channels)` or `(1, n_times,
       n_media_channels)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      m_nd: The ndarray that contains dimensional media data.
      ms_nd: The ndarray that contains dimensional media spend data.
      media_channels: The names of the media channels.

    Returns:
      The `NDArrayInputDataBuilder` with the added media and media spend data.
    """
    ### Validate ###
    self._validate_coords(is_media_time=True)
    self._validate_coords(is_media_time=False)
    self._validate_shape(nd=m_nd, dims=media_channels, is_media_time=True)
    self._validate_shape(nd=ms_nd, dims=media_channels, is_media_time=False)

    ### Transform ###
    self.media = xr.DataArray(
        m_nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.MEDIA_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.MEDIA_CHANNEL: media_channels,
        },
        name=constants.MEDIA,
    )
    self.media_spend = xr.DataArray(
        ms_nd,
        dims=[
            constants.GEO,
            constants.TIME,
            constants.MEDIA_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
            constants.MEDIA_CHANNEL: media_channels,
        },
        name=constants.MEDIA_SPEND,
    )
    return self

  def with_reach(
      self,
      r_nd: np.ndarray,
      f_nd: np.ndarray,
      rfs_nd: np.ndarray,
      rf_channels: list[str],
  ) -> 'NDArrayInputDataBuilder':
    """Reads reach, frequency, and rf_spend data from the ndarrays.

    `r_nd` and `f_nd` must be given with the shape:
    -  `(n_geos, n_media_times, n_rf_channels)`
    -  `(n_media_times, n_rf_channels)` or `(1, n_media_times,
       n_rf_channels)` for national model.

    `rfs_nd` must be given with the shape:
    -  `(n_geos, n_times, n_rf_channels)`
    -  `(n_times, n_rf_channels)` or `(1, n_times,
       n_rf_channels)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      r_nd: The ndarray that contains dimensional reach data.
      f_nd: The ndarray that contains dimensional frequency data.
      rfs_nd: The ndarray that contains dimensional rf_spend data.
      rf_channels: The names of the rf channels.

    Returns:
      The `NDArrayInputDataBuilder` with the added reach, frequency, and
      rf_spend data.
    """
    ### Validate ###
    self._validate_coords(is_media_time=True)
    self._validate_coords(is_media_time=False)
    self._validate_shape(nd=r_nd, dims=rf_channels, is_media_time=True)
    self._validate_shape(nd=f_nd, dims=rf_channels, is_media_time=True)
    self._validate_shape(nd=rfs_nd, dims=rf_channels, is_media_time=False)

    ### Transform ###
    self.reach = xr.DataArray(
        r_nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.RF_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.RF_CHANNEL: rf_channels,
        },
        name=constants.REACH,
    )
    self.frequency = xr.DataArray(
        f_nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.RF_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.RF_CHANNEL: rf_channels,
        },
        name=constants.FREQUENCY,
    )
    self.rf_spend = xr.DataArray(
        rfs_nd,
        dims=[
            constants.GEO,
            constants.TIME,
            constants.RF_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
            constants.RF_CHANNEL: rf_channels,
        },
        name=constants.RF_SPEND,
    )
    return self

  def with_organic_media(
      self, nd: np.ndarray, organic_media_channels: list[str]
  ) -> 'NDArrayInputDataBuilder':
    """Reads organic media data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos, n_media_times, n_organic_media_channels)`
    -  `(n_media_times, n_organic_media_channels)` or `(1, n_media_times,
       n_organic_media_channels)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the organic media data from.
      organic_media_channels: The names of the organic media channels.

    Returns:
      The `NDArrayInputDataBuilder` with the added organic media data.
    """
    ### Validate ###
    self._validate_coords(is_media_time=True)
    self._validate_shape(nd=nd, dims=organic_media_channels, is_media_time=True)

    ### Transform ###
    self.organic_media = xr.DataArray(
        nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_MEDIA_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.ORGANIC_MEDIA_CHANNEL: organic_media_channels,
        },
        name=constants.ORGANIC_MEDIA,
    )
    return self

  def with_organic_reach(
      self, or_nd: np.ndarray, of_nd: np.ndarray, organic_rf_channels: list[str]
  ) -> 'NDArrayInputDataBuilder':
    """Reads organic reach and organic frequency data from the ndarrays.

    `or_nd` and `of_nd` must be given with the shape:
    -  `(n_geos, n_media_times, n_organic_rf_channels)`
    -  `(n_media_times, n_organic_rf_channels)` or `(1, n_media_times,
       n_organic_rf_channels)` for national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      or_nd: The ndarray that contains dimensional reach data.
      of_nd: The ndarray that contains dimensional frequency data.
      organic_rf_channels: The names of the organic rf channels.

    Returns:
      The `NDArrayInputDataBuilder` with the added organic reach and organic
      frequency data.
    """
    ### Validate ###
    self._validate_coords(is_media_time=True)
    self._validate_shape(nd=or_nd, dims=organic_rf_channels, is_media_time=True)
    self._validate_shape(nd=of_nd, dims=organic_rf_channels, is_media_time=True)

    ### Transform ###
    self.organic_reach = xr.DataArray(
        or_nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_RF_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.ORGANIC_RF_CHANNEL: organic_rf_channels,
        },
        name=constants.ORGANIC_REACH,
    )
    self.organic_frequency = xr.DataArray(
        of_nd,
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_RF_CHANNEL,
        ],
        coords={
            constants.GEO: self.geos,
            constants.MEDIA_TIME: self.media_time_coords,
            constants.ORGANIC_RF_CHANNEL: organic_rf_channels,
        },
        name=constants.ORGANIC_REACH,
    )
    return self

  def with_non_media_treatments(
      self, nd: np.ndarray, non_media_channel_names: list[str]
  ) -> 'NDArrayInputDataBuilder':
    """Reads non-media treatments data from a ndarray.

    `nd` must be given with the shape:
    -  `(n_geos, n_time, n_media_channels)`
    -  `(n_time, n_media_channels)` or `(1, n_time, n_media_channels)` for
    national model.

    If called without a call to .geos() first, the data will be
    assumed to be national-level.

    Args:
      nd: The ndarray to read the non-media treatments data from.
      non_media_channel_names: The names of the non-media channels.

    Returns:
      The `NDArrayInputDataBuilder` with the added non-media treatments data.
    """
    ### Validate ###
    self._validate_coords()
    self._validate_shape(nd, non_media_channel_names)

    ### Transform ###
    self.non_media_treatments = xr.DataArray(
        nd,
        dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
        coords={
            constants.GEO: self.geos,
            constants.TIME: self.time_coords,
            constants.NON_MEDIA_CHANNEL: non_media_channel_names,
        },
        name=constants.NON_MEDIA_TREATMENTS,
    )
    return self

  def _validate_coords(
      self, is_population: bool = False, is_media_time: bool = False
  ):
    """Validates that the data has the expected coordinates."""
    if not is_population:
      if is_media_time and self._media_time_coords is None:
        raise ValueError(
            'Media times are required first. Set using .media_time_coords()'
        )
      if not is_media_time and self.time_coords is None:
        raise ValueError(
            'Time coordinates are required first. Set using .time_coords()'
        )
    if self.geos is None:
      logging.warning(
          'No geo coordinates set. Assuming NATIONAL model and geos will be set'
          ' to the default value.'
      )
      self.geos = [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]

  def _validate_shape(
      self,
      nd: np.ndarray,
      dims: list[str] | None = None,
      is_population: bool = False,
      is_media_time: bool = False,
  ):
    """Validates that the data has the expected shape."""
    # Since all data has a geo dimension (even for national data),
    # Expect the first axis to have the shape of the geo dimension.
    expected_shape = (len(self.geos),)
    detailed_info = f'Expected: {len(self.geos)} geos'
    if not is_population:
      if is_media_time:
        expected_shape += (len(self.media_time_coords),)
        detailed_info += f' x {len(self.media_time_coords)} media times'
      else:
        expected_shape += (len(self.time_coords),)
        detailed_info += f' x {len(self.time_coords)} times'

      if dims is not None:
        if len(dims) != len(set(dims)):
          raise ValueError('given dimensions must be unique.')
        expected_shape += (len(dims),)
        detailed_info += f' x {len(dims)} dims'

    if expected_shape != nd.shape:
      raise ValueError(f'{detailed_info}. Got: {nd.shape}.')

  def _check_revenue_per_kpi_defaults(self, nd: np.ndarray):
    """Sets revenue_per_kpi to default if kpi type is revenue and with_revenue_per_kpi is called."""
    if self._kpi_type == constants.REVENUE:
      warnings.warn(
          'with_revenue_per_kpi was called but kpi_type was set to revenue.'
          ' Assuming revenue per kpi with values [1].'
      )
      return np.ones(nd.shape)
    else:
      return nd
