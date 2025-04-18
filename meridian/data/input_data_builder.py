# Copyright 2024 The Meridian Authors.
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

"""TODO: swijaya - DO NOT SUBMIT without one-line documentation for builder.

TODO: swijaya - DO NOT SUBMIT without a detailed description of builder.
"""

import abc
from collections.abc import Sequence
import datetime
import numpy as np
import pandas as pd
import xarray as xr


class InputDataBuilder(abc.ABC):
  """Abstract base class for `InputData` builders."""

  # These working attributes are going to be set along the way as the builder
  # is provided piecemeal with the user's input data.

  ### Working dimensions and their coordinates. ###
  # Time coordinates.
  @property
  def time_coords(self) -> Sequence[datetime.date] | None:
    return self._time_coords

  @time_coords.setter
  def time_coords(self, value: Sequence[datetime.date]):
    if self.time_coords is not None and self.time_coords != value:
      raise ValueError(f'Time coords already set to {self.time_coords}.')

    # Validate: unique coordinate values, evenly spaced, etc.
    # Validate: If _media_time_coords is set, then this must be a subset.
    # ...
    self._time_coords = value

  # Media time coordinates.
  @property
  def media_time_coords(self) -> Sequence[datetime.date] | None:
    return self._media_time_coords

  @media_time_coords.setter
  def media_time_coords(self, value: Sequence[datetime.date]):
    if self.media_time_coords is not None and self.media_time_coords != value:
      raise ValueError(
          f'Media time coords already set to {self.media_time_coords}.'
      )
    self._media_time_coords = value

  # Geos.
  @property
  def geos(self) -> Sequence[str] | None:
    return self._geos

  @geos.setter
  def geos(self, value: Sequence[str]):
    if self.geos is not None and self.geos != value:
      raise ValueError(f'geos already set to {self.geos}.')
    self._geos = value

  # Media channels.
  @property
  def media_channels(self) -> Sequence[str] | None:
    return self._media_channels

  @media_channels.setter
  def media_channels(self, value: Sequence[str]):
    if self.media_channels is not None and self.media_channels != value:
      raise ValueError(f'media channels already set to {self.media_channels}.')
    self._media_channels = value

  # RF channels.
  @property
  def rf_channels(self) -> Sequence[str] | None:
    return self._rf_channels

  @rf_channels.setter
  def rf_channels(self, value: Sequence[str]):
    if self.rf_channels is not None and self.rf_channels != value:
      raise ValueError(f'rf channels already set to {self.rf_channels}.')
    self._rf_channels = value

  # Organic media channels.
  @property
  def organic_media_channels(self) -> Sequence[str] | None:
    return self._organic_media_channels

  @organic_media_channels.setter
  def organic_media_channels(self, value: Sequence[str]):
    if (
        self.organic_media_channels is not None
        and self.organic_media_channels != value
    ):
      raise ValueError(
          'organic media channels already set to'
          f' {self.organic_media_channels}.'
      )
    self._organic_media_channels = value

  # Organic RF channels.
  @property
  def organic_rf_channels(self) -> Sequence[str] | None:
    return self._organic_rf_channels

  @organic_rf_channels.setter
  def organic_rf_channels(self, value: Sequence[str]):
    if (
        self.organic_rf_channels is not None
        and self.organic_rf_channels != value
    ):
      raise ValueError(
          f'organic rf channels already set to {self.organic_rf_channels}.'
      )
    self._organic_rf_channels = value

  # Non-media channels.
  @property
  def non_media_channels(self) -> Sequence[str] | None:
    return self._non_media_channels

  @non_media_channels.setter
  def non_media_channels(self, value: Sequence[str]):
    if self.non_media_channels is not None and self.non_media_channels != value:
      raise ValueError(
          f'non media channels already set to {self.non_media_channels}.'
      )
    self._non_media_channels = value

  # Control variables.
  @property
  def control_variables(self) -> Sequence[str] | None:
    return self._control_variables

  @control_variables.setter
  def control_variables(self, value: Sequence[str]):
    if self.control_variables is not None and self.control_variables != value:
      raise ValueError(
          f'control variables already set to {self.control_variables}.'
      )
    self._control_variables = value

  ### Working data arrays (components of the final `InputData` object) ###
  # KPI
  @property
  def kpi(self) -> xr.DataArray | None:
    return self._kpi

  @kpi.setter
  def kpi(self, kpi: xr.DataArray):
    # Validate
    if self.kpi is not None:
      raise ValueError(f'KPI was already already set to {self.kpi}.')

    self._kpi = kpi

  # Controls
  @property
  def controls(self) -> xr.DataArray | None:
    return self._kpi

  @controls.setter
  def controls(self, controls: xr.DataArray):
    if self.controls is not None:
      raise ValueError(f'Controls was already already set to {self.controls}.')
    self._controls = controls

  # Population
  @property
  def population(self) -> xr.DataArray | None:
    return self._population

  @population.setter
  def population(self, population: xr.DataArray):
    if self.population is not None:
      raise ValueError(
          f'Population was already already set to {self.population}.'
      )
    self._population = population

  # Revenue per KPI
  @property
  def revenue_per_kpi(self) -> xr.DataArray | None:
    return self._revenue_per_kpi

  @revenue_per_kpi.setter
  def revenue_per_kpi(self, revenue_per_kpi: xr.DataArray):
    if self.revenue_per_kpi is not None:
      raise ValueError(
          f'Revenue per KPI was already already set to {self.revenue_per_kpi}.'
      )
    self._revenue_per_kpi = revenue_per_kpi

  # Media
  @property
  def media(self) -> xr.DataArray | None:
    return self._media

  @media.setter
  def media(self, media: xr.DataArray):
    if self.media is not None:
      raise ValueError(f'Media was already already set to {self.media}.')
    self._media = media

  # Media spend
  @property
  def media_spend(self) -> xr.DataArray | None:
    return self._media_spend

  @media_spend.setter
  def media_spend(self, media_spend: xr.DataArray):
    if self.media_spend is not None:
      raise ValueError(
          f'Media spend was already already set to {self.media_spend}.'
      )
    self._media_spend = media_spend

  # Reach
  @property
  def reach(self) -> xr.DataArray | None:
    return self._reach

  @reach.setter
  def reach(self, reach: xr.DataArray):
    if self.reach is not None:
      raise ValueError(f'Reach was already already set to {self.reach}.')
    self._reach = reach

  # Frequency
  @property
  def frequency(self) -> xr.DataArray | None:
    return self._frequency

  @frequency.setter
  def frequency(self, frequency: xr.DataArray):
    if self.frequency is not None:
      raise ValueError(
          f'Frequency was already already set to {self.frequency}.'
      )
    self._frequency = frequency

  # RF spend
  @property
  def rf_spend(self) -> xr.DataArray | None:
    return self._rf_spend

  @rf_spend.setter
  def rf_spend(self, rf_spend: xr.DataArray):
    if self.rf_spend is not None:
      raise ValueError(f'RF spend was already already set to {self.rf_spend}.')
    self._rf_spend = rf_spend

  # Organic media
  @property
  def organic_media(self) -> xr.DataArray | None:
    return self._organic_media

  @organic_media.setter
  def organic_media(self, organic_media: xr.DataArray):
    if self.organic_media is not None:
      raise ValueError(
          f'Organic media was already already set to {self.organic_media}.'
      )
    self._organic_media = organic_media

  # Organic reach
  @property
  def organic_reach(self) -> xr.DataArray | None:
    return self._organic_reach

  @organic_reach.setter
  def organic_reach(self, organic_reach: xr.DataArray):
    if self.organic_reach is not None:
      raise ValueError(
          f'Organic reach was already already set to {self.organic_reach}.'
      )
    self._organic_reach = organic_reach

  # Organic frequency
  @property
  def organic_frequency(self) -> xr.DataArray | None:
    return self._organic_frequency

  @organic_frequency.setter
  def organic_frequency(self, organic_frequency: xr.DataArray):
    if self.organic_frequency is not None:
      raise ValueError(
          'Organic frequency was already already set to'
          f' {self.organic_frequency}.'
      )
    self._organic_frequency = organic_frequency

  # Non-media treatments
  @property
  def non_media_treatments(self) -> xr.DataArray | None:
    return self._non_media_treatments

  @non_media_treatments.setter
  def non_media_treatments(self, non_media_treatments: xr.DataArray):
    if self.non_media_treatments is not None:
      raise ValueError(
          'Non-media treatments was already already set to'
          f' {self.non_media_treatments}.'
      )
    self._non_media_treatments = non_media_treatments

  def __init__(self, kpi_type: str):
    self._kpi_type = kpi_type

  def build(self):  # -> input_data.InputData:
    """Builds the input data."""
    # Here:
    # * Make one final validation pass for cross-arrays consistency.
    # * Collect all given arrays into a single `InputData` object.
    # * Return.
    # return input_data.InputData(
    #     kpi_type=self._kpi_type,
    #     kpi=self._kpi,
    #     # ...
    # )


class DataFrameInputDataBuilder(InputDataBuilder):
  """Builds `InputData` from DataFrames."""

  def with_kpi(self, kpi_df: pd.DataFrame) -> 'DataFrameInputDataBuilder':
    """Document `kpi_df` requirements."""
    # In the course of processing each DataFrame piece, dimension coordinates
    # will be discovered and set with, e.g., `self.time_coords = ...`.
    # The setter code (in the base abstract class) will perform basic validation
    # checks, e.g.:
    # * If previous dataframe input already set it, then it should be consistent
    # * If not, set it for the first time.
    # * When setting, make consistency checks against other dimensions
    # * etc...
    # self.time_coords = ...

    # Beyond the basic checks that the setter methods can do, here perform
    # KPI-specific checks, as well.
    # ...
    # TODO
    self.kpi = xr.DataArray(kpi_df)
    return self

  def with_controls(
      self, controls_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.controls = xr.DataArray(controls_df)
    return self

  def with_population(
      self, population_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.population = xr.DataArray(population_df)
    return self

  def with_revenue_per_kpi(
      self, revenue_per_kpi_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.revenue_per_kpi = xr.DataArray(revenue_per_kpi_df)
    return self

  def with_media(self, media_df: pd.DataFrame) -> 'DataFrameInputDataBuilder':
    self.media = xr.DataArray(media_df)
    self.media_spend = xr.DataArray(media_df)
    return self

  def with_reach(self, reach_df: pd.DataFrame) -> 'DataFrameInputDataBuilder':
    self.reach = xr.DataArray(reach_df)
    self.frequency = xr.DataArray(reach_df)
    self.rf_spend = xr.DataArray(reach_df)
    return self

  def with_organic_media(
      self, organic_media_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.organic_media = xr.DataArray(organic_media_df)
    return self

  def with_organic_reach(
      self, organic_reach_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.organic_reach = xr.DataArray(organic_reach_df)
    self.organic_frequency = xr.DataArray(organic_reach_df)
    return self

  def with_non_media_treatments(
      self, non_media_treatments_df: pd.DataFrame
  ) -> 'DataFrameInputDataBuilder':
    self.non_media_treatments = xr.DataArray(non_media_treatments_df)
    return self


class NDArrayInputDataBuilder(InputDataBuilder):
  """Builds `InputData` from n-dimensional arrays."""

  def with_kpi(self, kpi_nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    """Document `kpi_nd` requirements."""
    # Unlike `DataFrameInputDataBuilder`, each piecemeal data has no coordinate
    # information; they're purely data values. It's up to the user to provide
    # coordinates with setter methods from the abstract base class above.
    # Validation is done on each piece w.r.t. dimensional consistency by
    # shape alone.
    self.kpi = xr.DataArray(kpi_nd)
    return self

  def with_controls(self, controls_nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    self.controls = xr.DataArray(controls_nd)
    return self

  def with_population(
      self, population_nd: np.ndarray
  ) -> 'NDArrayInputDataBuilder':
    self.population = xr.DataArray(population_nd)
    return self

  def with_revenue_per_kpi(
      self, revenue_per_kpi_nd: np.ndarray
  ) -> 'NDArrayInputDataBuilder':
    self.revenue_per_kpi = xr.DataArray(revenue_per_kpi_nd)
    return self

  def with_media(self, media_nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    self.media = xr.DataArray(media_nd)
    self.media_spend = xr.DataArray(media_nd)
    return self

  def with_reach(self, reach_nd: np.ndarray) -> 'NDArrayInputDataBuilder':
    self.reach = xr.DataArray(reach_nd)
    self.frequency = xr.DataArray(reach_nd)
    self.rf_spend = xr.DataArray(reach_nd)
    return self

  def with_organic_media(
      self, organic_media_nd: np.ndarray
  ) -> 'NDArrayInputDataBuilder':
    self.organic_media = xr.DataArray(organic_media_nd)
    return self

  def with_organic_reach(
      self, organic_reach_nd: np.ndarray
  ) -> 'NDArrayInputDataBuilder':
    self.organic_reach = xr.DataArray(organic_reach_nd)
    self.organic_frequency = xr.DataArray(organic_reach_nd)
    return self

  def with_non_media_treatments(
      self, non_media_treatments_nd: np.ndarray
  ) -> 'NDArrayInputDataBuilder':
    self.non_media_treatments = xr.DataArray(non_media_treatments_nd)
    return self
