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

"""This module defines a Builder API for Meridian `InputData`.

The Builder API for `InputData` exposes piecewise data ingestion with its own
validation logic and an overall final validation logic before a valid
`InputData` is constructed.
"""

import abc
from collections.abc import Sequence
import datetime
import warnings
from meridian import constants
from meridian.data import input_data
from meridian.data import time_coordinates as tc
import natsort
import numpy as np
import xarray as xr


class InputDataBuilder(abc.ABC):
  """Abstract base class for `InputData` builders."""

  def __init__(self, kpi_type: str):
    self._kpi_type = kpi_type

    # These working attributes are going to be set along the way as the builder
    # is provided piecemeal with the user's input data.
    # In the course of processing each DataFrame piece, dimension coordinates
    # will be discovered and set with, e.g., `self.time_coords = ...`.
    # The setter code will perform basic validation
    # checks, e.g.:
    # * If previous dataframe input already set it, then it should be consistent
    # * If not, set it for the first time.
    # * When setting, make consistency checks against other dimensions
    # * etc...

    # Working dimensions and their coordinates.
    self._time_coords: Sequence[str] = None
    self._media_time_coords: Sequence[str] = None
    self._geos: Sequence[str] = None

    # Working data arrays (components of the final `InputData` object)
    self._kpi: xr.DataArray = None
    self._controls: xr.DataArray = None
    self._population: xr.DataArray = None
    self._revenue_per_kpi: xr.DataArray = None
    self._media: xr.DataArray = None
    self._media_spend: xr.DataArray = None
    self._reach: xr.DataArray = None
    self._frequency: xr.DataArray = None
    self._rf_spend: xr.DataArray = None
    self._organic_media: xr.DataArray = None
    self._organic_reach: xr.DataArray = None
    self._organic_frequency: xr.DataArray = None
    self._non_media_treatments: xr.DataArray = None

  @property
  def time_coords(self) -> Sequence[str]:
    return self._time_coords

  @time_coords.setter
  def time_coords(self, value: Sequence[str]):
    if len(value) != len(set(value)):
      raise ValueError('`times` coords must be unique.')
    if self.time_coords is not None and set(self.time_coords) != set(value):
      raise ValueError(f'`times` coords already set to {self.time_coords}.')
    if self.media_time_coords is not None and not set(value).issubset(
        self.media_time_coords
    ):
      raise ValueError(
          '`times` coords must be subset of previously set `media_times`'
          ' coords.'
      )
    if self.media_time_coords is not None:
      self._validate_lagged_media(
          media_time_coords=self.media_time_coords, time_coords=value
      )
    _ = tc.TimeCoordinates.from_dates(sorted(value)).interval_days
    self._time_coords = value

  @property
  def media_time_coords(self) -> Sequence[str]:
    return self._media_time_coords

  @media_time_coords.setter
  def media_time_coords(self, value: Sequence[str]):
    if len(value) != len(set(value)):
      raise ValueError('`media_times` coords must be unique.')
    if self.media_time_coords is not None and set(
        self.media_time_coords
    ) != set(value):
      raise ValueError(
          f'`media_times` coords already set to {self.media_time_coords}.'
      )
    if self.time_coords is not None and not set(value).issuperset(
        self.time_coords
    ):
      raise ValueError(
          '`media_times` coords must be superset of previously set `times`'
          ' coords.'
      )
    if self.time_coords is not None:
      self._validate_lagged_media(
          media_time_coords=value, time_coords=self.time_coords
      )
    _ = tc.TimeCoordinates.from_dates(sorted(value)).interval_days
    self._media_time_coords = value

  @property
  def geos(self) -> Sequence[str]:
    return self._geos

  @geos.setter
  def geos(self, value: Sequence[str]):
    if len(value) != len(set(value)):
      raise ValueError('Geos must be unique.')
    if self.geos is not None and set(self.geos) != set(value):
      raise ValueError(f'geos already set to {self.geos}.')
    self._geos = value

  @property
  def kpi(self) -> xr.DataArray:
    return self._kpi

  @kpi.setter
  def kpi(self, kpi: xr.DataArray):
    """Sets the `kpi` data array.

    `kpi` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='kpi',
        dims=['geo', 'time'],
        coords={
          'geo': ...,
          'time': ...,
        },
    )
    ```

    Args:
      kpi: Kpi DataArray.
    """
    self._validate_set('KPI', self.kpi)

    self._kpi = self._normalize_coords(kpi, constants.TIME)
    self.geos = self.kpi.coords[constants.GEO].values.tolist()
    self.time_coords = self.kpi.coords[constants.TIME].values.tolist()

  @property
  def controls(self) -> xr.DataArray:
    return self._controls

  @controls.setter
  def controls(self, controls: xr.DataArray):
    """Sets the `controls` data array.

    `controls` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='controls',
        dims=['geo', 'time', 'control_variable'],
        coords={
          'geo': ...,
          'time': ...,
          'control_variable': ...,
        },
    )
    ```

    Args:
      controls: Controls DataArray.
    """
    self._validate_set('Controls', self.controls)

    self._controls = self._normalize_coords(controls, constants.TIME)
    self.geos = self.controls.coords[constants.GEO].values.tolist()
    self.time_coords = self.controls.coords[constants.TIME].values.tolist()

  @property
  def population(self) -> xr.DataArray:
    return self._population

  @population.setter
  def population(self, population: xr.DataArray):
    """Sets the `media` data array.

    `population` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='population',
        dims=['geo'],
        coords={
          'geo': ...,
        },
    )
    ```

    Args:
      population: Population DataArray.
    """
    self._validate_set('Population', self.population)

    self._population = self._normalize_coords(population)
    self.geos = self.population.coords[constants.GEO].values.tolist()

  @property
  def revenue_per_kpi(self) -> xr.DataArray:
    return self._revenue_per_kpi

  @revenue_per_kpi.setter
  def revenue_per_kpi(self, revenue_per_kpi: xr.DataArray):
    """Sets the `revenue_per_kpi` data array.

    `revenue_per_kpi` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='revenue_per_kpi',
        dims=['geo', 'time'],
        coords={
          'geo': ...,
          'time': ...,
        },
    )
    ```
    Args:
      revenue_per_kpi: Revenue per kpi DataArray.
    """
    self._validate_set('Revenue per KPI', self.revenue_per_kpi)

    self._revenue_per_kpi = self._normalize_coords(
        revenue_per_kpi, constants.TIME
    )
    self.geos = self.revenue_per_kpi.coords[constants.GEO].values.tolist()
    self.time_coords = self.revenue_per_kpi.coords[
        constants.TIME
    ].values.tolist()

  @property
  def media(self) -> xr.DataArray:
    return self._media

  @media.setter
  def media(self, media: xr.DataArray):
    """Sets the `media` data array.

    `media` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='media',
        dims=['geo', 'media_time', 'media_channel'],
        coords={
          'geo': ...,
          'media_time': ...,
          'media_channel': ...,
        },
    )
    ```

    Args:
      media: Media DataArray.
    """
    self._validate_set('Media', self.media)
    self._validate_channels_consistency(
        constants.MEDIA_CHANNEL, [media, self.media_spend]
    )

    self._media = self._normalize_coords(media, constants.MEDIA_TIME)
    self.geos = self.media.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.media.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def media_spend(self) -> xr.DataArray:
    return self._media_spend

  @media_spend.setter
  def media_spend(self, media_spend: xr.DataArray):
    """Sets the `media_spend` data array.

    `media_spend` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='media_spend',
        dims=['geo', 'time', 'media_channel'],
        coords={
          'geo': ...,
          'time': ...,
          'media_channel': ...,
        },
    )
    ```

    Args:
      media_spend: Media spend DataArray.
    """
    self._validate_set('Media spend', self.media_spend)
    self._validate_channels_consistency(
        constants.MEDIA_CHANNEL, [media_spend, self.media]
    )

    self._media_spend = self._normalize_coords(media_spend, constants.TIME)
    self.geos = self.media_spend.coords[constants.GEO].values.tolist()
    self.time_coords = self.media_spend.coords[constants.TIME].values.tolist()

  @property
  def reach(self) -> xr.DataArray:
    return self._reach

  @reach.setter
  def reach(self, reach: xr.DataArray):
    """Sets the `reach` data array.

    `reach` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='reach',
        dims=['geo', 'media_time', 'rf_channel'],
        coords={
          'geo': ...,
          'media_time': ...,
          'rf_channel': ...,
        },
    )
    ```

    Args:
      reach: Reach DataArray.
    """
    self._validate_set('Reach', self.reach)
    self._validate_channels_consistency(
        constants.RF_CHANNEL, [reach, self.frequency, self.rf_spend]
    )

    self._reach = self._normalize_coords(reach, constants.MEDIA_TIME)
    self.geos = self.reach.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.reach.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def frequency(self) -> xr.DataArray:
    return self._frequency

  @frequency.setter
  def frequency(self, frequency: xr.DataArray):
    """Sets the `frequency` data array.

    `frequency` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='frequency',
        dims=['geo', 'media_time', 'rf_channel'],
        coords={
          'geo': ...,
          'media_time': ...,
          'rf_channel': ...,
        },
    )
    ```

    Args:
      frequency: Frequency DataArray.
    """
    self._validate_set('Frequency', self.frequency)
    self._validate_channels_consistency(
        constants.RF_CHANNEL, [frequency, self.reach, self.rf_spend]
    )

    self._frequency = self._normalize_coords(frequency, constants.MEDIA_TIME)
    self.geos = self.frequency.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.frequency.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def rf_spend(self) -> xr.DataArray:
    return self._rf_spend

  @rf_spend.setter
  def rf_spend(self, rf_spend: xr.DataArray):
    """Sets the `rf_spend` data array.

    `rf_spend` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='rf_spend',
        dims=['geo', 'time', 'rf_channel'],
        coords={
          'geo': ...,
          'time': ...,
          'rf_channel': ...,
        },
    )
    ```

    Args:
      rf_spend: RF spend DataArray.
    """
    self._validate_set('RF spend', self.rf_spend)
    self._validate_channels_consistency(
        constants.RF_CHANNEL, [rf_spend, self.reach, self.frequency]
    )

    self._rf_spend = self._normalize_coords(rf_spend, constants.TIME)
    self.geos = self.rf_spend.coords[constants.GEO].values.tolist()
    self.time_coords = self.rf_spend.coords[constants.TIME].values.tolist()

  @property
  def organic_media(self) -> xr.DataArray:
    return self._organic_media

  @organic_media.setter
  def organic_media(self, organic_media: xr.DataArray):
    """Sets the `organic_media` data array.

    `organic_media` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='organic_media',
        dims=['geo', 'media_time'],
        coords={
          'geo': ...,
          'media_time': ...,
        },
    )
    ```

    Args:
      organic_media: Organic media DataArray.
    """
    self._validate_set('Organic media', self.organic_media)

    self._organic_media = self._normalize_coords(
        organic_media, constants.MEDIA_TIME
    )
    self.geos = self.organic_media.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.organic_media.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def organic_reach(self) -> xr.DataArray:
    return self._organic_reach

  @organic_reach.setter
  def organic_reach(self, organic_reach: xr.DataArray):
    """Sets the `organic_reach` data array.

    `organic_reach` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='organic_reach',
        dims=['geo', 'media_time', 'organic_rf_channel'],
        coords={
          'geo': ...,
          'media_time': ...,
          'organic_rf_channel': ...,
        },
    )
    ```

    Args:
      organic_reach: Organic reach DataArray.
    """
    self._validate_set('Organic reach', self.organic_reach)
    self._validate_channels_consistency(
        constants.ORGANIC_RF_CHANNEL, [organic_reach, self.organic_frequency]
    )

    self._organic_reach = self._normalize_coords(
        organic_reach, constants.MEDIA_TIME
    )
    self.geos = self.organic_reach.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.organic_reach.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def organic_frequency(self) -> xr.DataArray:
    return self._organic_frequency

  @organic_frequency.setter
  def organic_frequency(self, organic_frequency: xr.DataArray):
    """Sets the `organic_frequency` data array.

    `organic_frequency` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='organic_frequency',
        dims=['geo', 'media_time', 'organic_rf_channel'],
        coords={
          'geo': ...,
          'media_time': ...,
          'organic_rf_channel': ...,
        },
    )
    ```

    Args:
      organic_frequency: Organic frequency DataArray.
    """
    self._validate_set('Organic frequency', self.organic_frequency)
    self._validate_channels_consistency(
        constants.ORGANIC_RF_CHANNEL, [organic_frequency, self.organic_reach]
    )

    self._organic_frequency = self._normalize_coords(
        organic_frequency, constants.MEDIA_TIME
    )
    self.geos = self.organic_frequency.coords[constants.GEO].values.tolist()
    self.media_time_coords = self.organic_frequency.coords[
        constants.MEDIA_TIME
    ].values.tolist()

  @property
  def non_media_treatments(self) -> xr.DataArray:
    return self._non_media_treatments

  @non_media_treatments.setter
  def non_media_treatments(self, non_media_treatments: xr.DataArray):
    """Sets the `non media treatments` data array.

    `non_media_treatments` must have the following `DataArray` signature:

    ```
    xarray.DataArray(
        data=...,
        name='non_media_treatments',
        dims=['geo', 'time', 'non_media_channel'],
        coords={
          'geo': ...,
          'time': ...,
          'non_media_channel': ...,
        },
    )
    ```

    Args:
      non_media_treatments: Non-media treatments DataArray.
    """
    self._validate_set('Non-media treatments', self.non_media_treatments)

    self._non_media_treatments = self._normalize_coords(
        non_media_treatments, constants.TIME
    )
    self.geos = self.non_media_treatments.coords[constants.GEO].values.tolist()
    self.time_coords = self.non_media_treatments.coords[
        constants.TIME
    ].values.tolist()

  def build(self) -> input_data.InputData:
    """Builds an `InputData`.

    Constructs an `InputData` from constituent `DataArray`s given to this
    builder thus far after performing one final validation pass over all data
    arrays for consistency checks.

    Returns:
      A validated `InputData`.
    """
    self._validate_required_components()
    self._validate_nas()

    # TODO: move logic from input_data to here: all channel names
    # should be unique across media channels, rf channels, organic media
    # channels, and organic rf channels.
    sorted_geos = natsort.natsorted(self.geos)
    sorted_times = natsort.natsorted(self.time_coords)
    sorted_media_times = natsort.natsorted(self.media_time_coords)

    def _get_sorted(da: xr.DataArray | None, is_media_time: bool = False):
      """Naturally sorts the DataArray by geo and time/media time."""

      if da is None:
        return None
      else:
        if is_media_time:
          return da.reindex(geo=sorted_geos, media_time=sorted_media_times)
        else:
          return da.reindex(geo=sorted_geos, time=sorted_times)

    return input_data.InputData(
        kpi_type=self._kpi_type,
        kpi=_get_sorted(self.kpi),
        revenue_per_kpi=_get_sorted(self.revenue_per_kpi),
        controls=_get_sorted(self.controls),
        population=self.population.reindex(geo=sorted_geos),
        media=_get_sorted(self.media, True),
        media_spend=_get_sorted(self.media_spend),
        reach=_get_sorted(self.reach, True),
        frequency=_get_sorted(self.frequency, True),
        rf_spend=_get_sorted(self.rf_spend),
        non_media_treatments=_get_sorted(self.non_media_treatments),
        organic_media=_get_sorted(self.organic_media, True),
        organic_reach=_get_sorted(self.organic_reach, True),
        organic_frequency=_get_sorted(self.organic_frequency, True),
    )

  def _normalize_coords(
      self, da: xr.DataArray, time_dimension_name: str | None = None
  ) -> xr.DataArray:
    """Validates that time values are in the conventional Meridian format and geos have national name if national."""
    if time_dimension_name is not None:
      # Time values are expected to be
      # (a) strings formatted in `"yyyy-mm-dd"`
      # or
      # (b) `datetime` values as numpy's `datetime64` types.
      # All other types are not currently supported.

      # If (b), `datetime` coord values will be normalized as formatted strings.

      if da.coords.dtypes[time_dimension_name] == np.dtype('datetime64[ns]'):
        date_strvalues = np.datetime_as_string(
            da.coords[time_dimension_name], unit='D'
        )
        da = da.assign_coords({time_dimension_name: date_strvalues})

      # Assume that the time coordinate labels are date-formatted strings.
      # We don't currently support other, arbitrary object types in the builder.
      for time in da.coords[time_dimension_name].values:
        try:
          _ = datetime.datetime.strptime(time, constants.DATE_FORMAT)
        except ValueError as exc:
          raise ValueError(
              f"Invalid time label: '{time}'. Expected format:"
              f" '{constants.DATE_FORMAT}'"
          ) from exc

    if len(da.coords[constants.GEO].values.tolist()) == 1:
      da = da.assign_coords(
          {constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]},
      )
    return da

  def _validate_set(self, component: str, da: xr.DataArray):
    if da is not None:
      raise ValueError(f'{component} was already set to {da}.')

  def _validate_channels_consistency(
      self, channel_dimension_name: str, da_list: list[xr.DataArray | None]
  ):
    for da in da_list:
      if da is not None and set(
          da.coords[channel_dimension_name].values.tolist()
      ) != set(da_list[0].coords[channel_dimension_name].values.tolist()):
        raise ValueError(
            f'{channel_dimension_name} coordinates must be the same between'
            f' {[da.name for da in da_list if da is not None]}.'
        )

  def _validate_required_components(self):
    """Validates that all required data arrays are defined."""
    if self.kpi is None:
      raise ValueError('KPI is required.')

    if len(self.geos) == 1:
      if self.population is not None:
        warnings.warn(
            'The `population` argument is ignored in a nationally aggregated'
            ' model. It will be reset to [1, 1, ..., 1]'
        )
      self._population = xr.DataArray(
          [constants.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE],
          dims=[constants.GEO],
          coords={
              constants.GEO: self.geos,
          },
          name=constants.POPULATION,
      )
    if self.population is None:
      raise ValueError('Population is required for non national models.')

    if (self.media is None) ^ (self.media_spend is None):
      raise ValueError('Media and media spend must be provided together.')
    if (
        self.reach is not None
        or self.frequency is not None
        or self.rf_spend is not None
    ) and (
        self.reach is None or self.frequency is None or self.rf_spend is None
    ):
      raise ValueError(
          'Reach, frequency, and rf_spend must be provided together.'
      )
    if (self.organic_reach is None) ^ (self.organic_frequency is None):
      raise ValueError(
          'Organic reach and organic frequency must be provided together.'
      )
    if (
        self.reach is None
        and self.frequency is None
        and self.rf_spend is None
        and self.media_spend is None
        and self.media is None
    ):
      raise ValueError(
          'It is required to have at least one of media or reach + frequency.'
      )

  def _validate_nas(self):
    """Check for NAs in all of the DataArrays.

    Since the DataArray components should already distinguish between media time
    and time coords, there are no media times to infer so there should be no
    NAs.
    """
    if self.kpi.isnull().any(axis=None):
      raise ValueError('NA values found in the kpi data.')
    if self.population.isnull().any(axis=None):
      raise ValueError('NA values found in the population data.')
    if self.controls is not None and self.controls.isnull().any(axis=None):
      raise ValueError('NA values found in the controls data.')
    if self.revenue_per_kpi is not None and self.revenue_per_kpi.isnull().any(
        axis=None
    ):
      raise ValueError('NA values found in the revenue per kpi data.')
    if self.media_spend is not None and self.media_spend.isnull().any(
        axis=None
    ):
      raise ValueError('NA values found in the media spend data.')
    if self.rf_spend is not None and self.rf_spend.isnull().any(axis=None):
      raise ValueError('NA values found in the rf spend data.')
    if (
        self.non_media_treatments is not None
        and self.non_media_treatments.isnull().any(axis=None)
    ):
      raise ValueError('NA values found in the non media treatments data.')

    if self.media is not None and self.media.isnull().any(axis=None):
      raise ValueError('NA values found in the media data.')

    if self.reach is not None and self.reach.isnull().any(axis=None):
      raise ValueError('NA values found in the reach data.')
    if self.frequency is not None and self.frequency.isnull().any(axis=None):
      raise ValueError('NA values found in the frequency data.')

    if self.organic_media is not None and self.organic_media.isnull().any(
        axis=None
    ):
      raise ValueError('NA values found in the organic media data.')

    if self.organic_reach is not None and self.organic_reach.isnull().any(
        axis=None
    ):
      raise ValueError('NA values found in the organic reach data.')
    if (
        self.organic_frequency is not None
        and self.organic_frequency.isnull().any(axis=None)
    ):
      raise ValueError('NA values found in the organic frequency data.')

  def _validate_lagged_media(
      self, media_time_coords: Sequence[str], time_coords: Sequence[str]
  ):
    na_period = np.sort(list(set(media_time_coords) - set(time_coords)))
    if not np.all(na_period == np.sort(media_time_coords)[: len(na_period)]):
      raise ValueError(
          "The 'lagged media' period (period with 100% NA values in all"
          f' non-media columns) {na_period} is not a continuous window'
          ' starting from the earliest time period.'
      )
