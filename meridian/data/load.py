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

"""Contains classes and methods to load input data for Meridian.

The `InputDataLoader` abstract class defines a single method: `load()` which
reads data from any of the supported sources and stores it as an `InputData`
object.
"""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import immutabledict
from meridian import constants
from meridian.data import data_frame_input_data_builder
from meridian.data import input_data
from meridian.data import input_data_builder
import pandas as pd
import xarray as xr


__all__ = [
    'InputDataLoader',
    'XrDatasetDataLoader',
    'DataFrameDataLoader',
]


class InputDataLoader(metaclass=abc.ABCMeta):
  """Loads the data from the specified data format."""

  @abc.abstractmethod
  def load(self) -> input_data.InputData:
    """Reads the data and outputs an `InputData` object."""
    raise NotImplementedError()


class XrDatasetDataLoader(InputDataLoader):
  """Reads data from an `xarray.Dataset` object.

  Attributes:
    dataset: An `xr.Dataset` object containing the input data.
    kpi_type: A string denoting whether the KPI is of a `'revenue'` or
      `'non-revenue'` type. When the `kpi_type` is `'non-revenue'` and
      `revenue_per_kpi` exists, ROI calibration is used and the analysis is run
      on revenue. When `revenue_per_kpi` doesn't exist for the same `kpi_type`,
      custom ROI calibration is used and the analysis is run on KPI.

  Example:

  ```python
    data_loader = XrDatasetDataLoader(pickle.loads('data.pickle'))
    data = data_loader.load()
  ```
  """

  dataset: xr.Dataset

  def __init__(
      self,
      dataset: xr.Dataset,
      kpi_type: str,
      name_mapping: Mapping[str, str] | None = None,
  ):
    """Constructor.

    The coordinates of the input dataset should be: `time`, `media_time`,
    `control_variable` (optional), `geo` (optional for a national model),
    `non_media_channel` (optional), `organic_media_channel` (optional),
    `organic_rf_channel` (optional), and
    either `media_channel`, `rf_channel`, or both.

    Coordinate labels for `time` and `media_time` must be formatted in
    `"yyyy-mm-dd"` date format.

    In a geo model, the dataset should consist of the following arrays of the
    following dimensions. We use `1` to indicate a required dimension with
    length 1:

    *   `kpi`: `(geo, time)`
    *   `revenue_per_kpi`: `(geo, time)`
    *   `controls`: `(geo, time, control_variable)` - optional
    *   `population`: `(geo)`
    *   `media`: `(geo, media_time, media_channel)` - optional
    *   `media_spend`: `(geo, time, media_channel)`, `(1, time, media_channel)`,
        `(geo, 1, media_channel)`, `(media_channel)` - optional
    *   `reach`: `(geo, media_time, rf_channel)` - optional
    *   `frequency`: `(geo, media_time, rf_channel)` - optional
    *   `rf_spend`: `(geo, time, rf_channel)`, `(1, time, rf_channel)`,
        `(geo, 1, rf_channel)`, or `(rf_channel)` - optional
    *   `non_media_treatments`: `(geo, time, non_media_channel)` - optional
    *   `organic_media`: `(geo, media_time, organic_media_channel)` - optional
    *   `organic_reach`: `(geo, media_time, organic_rf_channel)` - optional
    *   `organic_frequency`: `(geo, media_time, organic_rf_channel)` - optional

    In a national model, the dataset should consist of the following arrays of
    the following dimensions. We use `[1,]` to indicate an optional dimension
    with length 1:

    *   `kpi`: `([1,] time)`
    *   `revenue_per_kpi`: `([1,] time)`
    *   `controls`: `([1,] time, control_variable)` - optional
    *   `population`: `([1],)` - this array is optional for national data
    *   `media`: `([1,] media_time, media_channel)` - optional
    *   `media_spend`: `([1,] time, media_channel)` or
        `([1,], [1,], media_channel)` - optional
    *   `reach`: `([1,] media_time, rf_channel)` - optional
    *   `frequency`: `([1,] media_time, rf_channel)` - optional
    *   `rf_spend`: `([1,] time, rf_channel)` or `([1,], [1,], rf_channel)` -
        optional
    *   `non_media_treatments`: `([1,] time, non_media_channel)` - optional
    *   `organic_media`: `([1,] media_time, organic_media_channel)` - optional
    *   `organic_reach`: `([1,] media_time, organic_rf_channel)` - optional
    *   `organic_frequency`: `([1,] media_time, organic_rf_channel)` - optional

    In a national model, the data will be expanded to include a single geo
    dimension.

    The dataset should include at least one of the following metric
    combinations: (1) media and media_spend or (2) reach, frequency, rf_spend.

    If the names of the coordinates or arrays are different, they can be renamed
    using the name_mapping argument. Example:

    ```python
    loader = XrDatasetDataLoader(
        dataset=pickle.loads('data.pickle'),
        name_mapping={'group': 'geo', 'cost': 'media_spend', 'conversions':
        'kpi'},
    )
    ```

    Alternatively to using `media_time`, the `media`, `reach` and `frequency`
    arrays can use the `time` coordinate, the same as the other arrays use.
    In such case, the dimensions will be converted by the loader into `time`
    and `media_time` and the lagged period will be determined by the missing
    values in the other arrays, similarly to `DataFrameDataLoader` and
    `CsvDataLoader`.

    Args:
      dataset: An `xarray.Dataset` object containing the input data.
      kpi_type: A string denoting whether the KPI is of a `'revenue'` or
        `'non-revenue'` type. When the `kpi_type` is `'non-revenue'` and
        `revenue_per_kpi` exists, ROI calibration is used and the analysis is
        run on revenue. When `revenue_per_kpi` doesn't exist for the same
        `kpi_type`, custom ROI calibration is used and the analysis is run on
        KPI.
      name_mapping: An optional dictionary whose keys are the current
        coordinates or array names in the `input` dataset and whose values are
        the desired coordinates (`geo`, `time`, `media_time`, `media_channel`
        and/or `rf_channel`, `control_variable`, `non_media_channel`,
        `organic_media_channel`, `organic_rf_channel`) or array names (`kpi`,
        `revenue_per_kpi`, `media`, `media_spend` and/or `rf_spend`, `controls`,
        `population`, `non_media_treatments`, `organic_media`, `organic_reach`,
        `organic_frequency`). Mapping must be provided if the names in the
        `input` dataset are different from the required ones, otherwise errors
        are thrown.
    """
    self.kpi_type = kpi_type
    if name_mapping is None:
      self.dataset = dataset
    else:
      source_coord_names = tuple(dataset.coords.keys())
      source_array_names = tuple(dataset.data_vars.keys())
      source_coords_and_arrays_set = frozenset(
          source_coord_names + source_array_names
      )
      for name in name_mapping.values():
        if name not in constants.POSSIBLE_INPUT_DATA_COORDS_AND_ARRAYS_SET:
          raise ValueError(
              f"Target name '{name}' from the mapping is none of the target"
              f' coordinate names {constants.POSSIBLE_INPUT_DATA_COORD_NAMES}'
              f' or array names {constants.POSSIBLE_INPUT_DATA_ARRAY_NAMES}.'
          )

      for name in name_mapping.keys():
        if name not in source_coords_and_arrays_set:
          raise ValueError(
              f"Source name '{name}' from the mapping is none of the"
              f' coordinate names {source_coord_names} or array names'
              f' {source_array_names} of the input dataset.'
          )

      self.dataset = dataset.rename(name_mapping)

    # Add a `geo` dimension if it is not already present.
    if (constants.GEO) not in self.dataset.sizes.keys():
      self.dataset = self.dataset.expand_dims(dim=[constants.GEO], axis=0)

    if constants.MEDIA_TIME not in self.dataset.sizes.keys():
      # Arrays in which NAs are expected in the lagged-media period.
      na_arrays = [
          constants.KPI,
      ]

      na_mask = self.dataset[constants.KPI].isnull().any(dim=constants.GEO)

      if constants.CONTROLS in self.dataset.data_vars.keys():
        na_arrays.append(constants.CONTROLS)
        na_mask |= (
            self.dataset[constants.CONTROLS]
            .isnull()
            .any(dim=[constants.GEO, constants.CONTROL_VARIABLE])
        )

      if constants.NON_MEDIA_TREATMENTS in self.dataset.data_vars.keys():
        na_arrays.append(constants.NON_MEDIA_TREATMENTS)
        na_mask |= (
            self.dataset[constants.NON_MEDIA_TREATMENTS]
            .isnull()
            .any(dim=[constants.GEO, constants.NON_MEDIA_CHANNEL])
        )

      if constants.REVENUE_PER_KPI in self.dataset.data_vars.keys():
        na_arrays.append(constants.REVENUE_PER_KPI)
        na_mask |= (
            self.dataset[constants.REVENUE_PER_KPI]
            .isnull()
            .any(dim=constants.GEO)
        )
      if constants.MEDIA_SPEND in self.dataset.data_vars.keys():
        na_arrays.append(constants.MEDIA_SPEND)
        na_mask |= (
            self.dataset[constants.MEDIA_SPEND]
            .isnull()
            .any(dim=[constants.GEO, constants.MEDIA_CHANNEL])
        )
      if constants.RF_SPEND in self.dataset.data_vars.keys():
        na_arrays.append(constants.RF_SPEND)
        na_mask |= (
            self.dataset[constants.RF_SPEND]
            .isnull()
            .any(dim=[constants.GEO, constants.RF_CHANNEL])
        )

      # Dates with at least one non-NA value in non-media columns
      no_na_period = self.dataset[constants.TIME].isel(time=~na_mask).values

      # Create new `time` and `media_time` coordinates.
      new_time = 'new_time'

      new_dataset = self.dataset.assign_coords(
          new_time=(new_time, no_na_period),
      )

      new_dataset[constants.KPI] = (
          new_dataset[constants.KPI]
          .dropna(dim=constants.TIME)
          .rename({constants.TIME: new_time})
      )
      if constants.CONTROLS in new_dataset.data_vars.keys():
        new_dataset[constants.CONTROLS] = (
            new_dataset[constants.CONTROLS]
            .dropna(dim=constants.TIME)
            .rename({constants.TIME: new_time})
        )
      if constants.NON_MEDIA_TREATMENTS in new_dataset.data_vars.keys():
        new_dataset[constants.NON_MEDIA_TREATMENTS] = (
            new_dataset[constants.NON_MEDIA_TREATMENTS]
            .dropna(dim=constants.TIME)
            .rename({constants.TIME: new_time})
        )

      if constants.REVENUE_PER_KPI in new_dataset.data_vars.keys():
        new_dataset[constants.REVENUE_PER_KPI] = (
            new_dataset[constants.REVENUE_PER_KPI]
            .dropna(dim=constants.TIME)
            .rename({constants.TIME: new_time})
        )

      if constants.MEDIA_SPEND in new_dataset.data_vars.keys():
        new_dataset[constants.MEDIA_SPEND] = (
            new_dataset[constants.MEDIA_SPEND]
            .dropna(dim=constants.TIME)
            .rename({constants.TIME: new_time})
        )

      if constants.RF_SPEND in new_dataset.data_vars.keys():
        new_dataset[constants.RF_SPEND] = (
            new_dataset[constants.RF_SPEND]
            .dropna(dim=constants.TIME)
            .rename({constants.TIME: new_time})
        )

      self.dataset = new_dataset.rename(
          {constants.TIME: constants.MEDIA_TIME, new_time: constants.TIME}
      )

  def load(self) -> input_data.InputData:
    """Returns an `InputData` object containing the data from the dataset."""
    builder = input_data_builder.InputDataBuilder(self.kpi_type)
    builder.kpi = self.dataset.kpi
    if constants.POPULATION in self.dataset.data_vars.keys():
      builder.population = self.dataset.population
    if constants.CONTROLS in self.dataset.data_vars.keys():
      builder.controls = self.dataset.controls
    if constants.REVENUE_PER_KPI in self.dataset.data_vars.keys():
      builder.revenue_per_kpi = self.dataset.revenue_per_kpi
    if constants.MEDIA in self.dataset.data_vars.keys():
      builder.media = self.dataset.media
    if constants.MEDIA_SPEND in self.dataset.data_vars.keys():
      builder.media_spend = self.dataset.media_spend
    if constants.REACH in self.dataset.data_vars.keys():
      builder.reach = self.dataset.reach
    if constants.FREQUENCY in self.dataset.data_vars.keys():
      builder.frequency = self.dataset.frequency
    if constants.RF_SPEND in self.dataset.data_vars.keys():
      builder.rf_spend = self.dataset.rf_spend
    if constants.NON_MEDIA_TREATMENTS in self.dataset.data_vars.keys():
      builder.non_media_treatments = self.dataset.non_media_treatments
    if constants.ORGANIC_MEDIA in self.dataset.data_vars.keys():
      builder.organic_media = self.dataset.organic_media
    if constants.ORGANIC_REACH in self.dataset.data_vars.keys():
      builder.organic_reach = self.dataset.organic_reach
    if constants.ORGANIC_FREQUENCY in self.dataset.data_vars.keys():
      builder.organic_frequency = self.dataset.organic_frequency
    return builder.build()


@dataclasses.dataclass(frozen=True)
class CoordToColumns:
  """A mapping between the desired and actual column names in the input data.

  Attributes:
    time: Name of column containing `time` values in the input data.
    geo:  Name of column containing `geo` values in the input data. This field
      is optional for a national model.
    kpi: Name of column containing `kpi` values in the input data.
    controls: List of column names containing `controls` values in the input
      data. Optional.
    revenue_per_kpi: Name of column containing `revenue_per_kpi` values in the
      input data. Optional. Will be overridden if model KPI type is "revenue".
    population: Name of column containing `population` values in the input data.
      This field is optional for a national model.
    media: List of column names containing `media` values in the input data.
    media_spend: List of column names containing `media_spend` values in the
      input data.
    reach: List of column names containing `reach` values in the input data.
    frequency: List of column names containing `frequency` values in the input
      data.
    rf_spend: List of column names containing `rf_spend` values in the input
      data.
    non_media_treatments: List of column names containing `non_media_treatments`
      values in the input data.
    organic_media: List of column names containing `organic_media` values in the
      input data.
    organic_reach: List of column names containing `organic_reach` values in the
      input data.
    organic_frequency: List of column names containing `organic_frequency`
      values in the input data.
  """

  time: str = constants.TIME
  geo: str = constants.GEO
  kpi: str = constants.KPI
  controls: Sequence[str] | None = None
  revenue_per_kpi: str | None = None
  population: str = constants.POPULATION
  # Media data
  media: Sequence[str] | None = None
  media_spend: Sequence[str] | None = None
  # RF data
  reach: Sequence[str] | None = None
  frequency: Sequence[str] | None = None
  rf_spend: Sequence[str] | None = None
  # Non-media treatments data
  non_media_treatments: Sequence[str] | None = None
  # Organic media and RF data
  organic_media: Sequence[str] | None = None
  organic_reach: Sequence[str] | None = None
  organic_frequency: Sequence[str] | None = None

  def __post_init__(self):
    has_media_fields = self.media and self.media_spend
    has_rf_fields = self.reach and self.frequency and self.rf_spend
    if not (has_media_fields or has_rf_fields):
      raise ValueError(
          '`coord_to_columns` should include media data (`media` and'
          ' `media_spend`) or RF data (`reach`, `frequency` and `rf_spend`), or'
          ' both.'
      )


@dataclasses.dataclass
class DataFrameDataLoader(InputDataLoader):
  """Reads data from a Pandas `DataFrame`.

  This class reads input data from a Pandas `DataFrame`. The `coord_to_columns`
  attribute stores a mapping from target `InputData` coordinates and array names
  to the DataFrame column names if they are different. The fields are:

  *   `geo`, `time`, `kpi`, `revenue_per_kpi`, `population` (single column)
  *   `controls` (multiple columns, optional)
  *   (1) `media`, `media_spend` (multiple columns)
  *   (2) `reach`, `frequency`, `rf_spend` (multiple columns)
  *   `non_media_treatments` (multiple columns, optional)
  *   `organic_media` (multiple columns, optional)
  *   `organic_reach`, `organic_frequency` (multiple columns, optional)

  The `DataFrame` must include (1) or (2), but doesn't need to include both.
  Also, each media channel must appear in (1) or (2), but not both.

  Note the following:

  *   Time column values must be formatted in _yyyy-mm-dd_ date format.
  *   In a national model, `geo` and `population` are optional. If the
      `population` is provided, it is reset to a default value of `1.0`.
  *   If `media` data is provided, then `media_to_channel` and
      `media_spend_to_channel` are required. If `reach` and `frequency` data is
      provided, then `reach_to_channel` and `frequency_to_channel` and
      `rf_spend_to_channel` are required.
  *   If `organic_reach` and `organic_frequency` data is provided, then
      `organic_reach_to_channel` and `organic_frequency_to_channel` are
      required.

  Example:

  ```python
  # df = [...]
  coord_to_columns = CoordToColumns(
    geo='dmas',
    time='dates',
    kpi='conversions',
    revenue_per_kpi='revenue_per_conversions',
    controls=['control_income'],
    population='populations',
    media=['impressions_tv', 'impressions_fb', 'impressions_search'],
    media_spend=['spend_tv', 'spend_fb', 'spend_search'],
    reach=['reach_yt'],
    frequency=['frequency_yt'],
    rf_spend=['rf_spend_yt'],
    non_media_treatments=['price', 'discount']
    organic_media=['organic_impressions_blog'],
    organic_reach=['organic_reach_newsletter'],
    organic_frequency=['organic_frequency_newsletter'],
  )
  media_to_channel = {
      'impressions_tv': 'tv',
      'impressions_fb': 'fb',
      'impressions_search': 'search',
  }
  media_spend_to_channel = {
      'spend_tv': 'tv', 'spend_fb': 'fb', 'spend_search': 'search'
  }
  reach_to_channel = {'reach_yt': 'yt'}
  frequency_to_channel = {'frequency_yt': 'yt'}
  rf_spend_to_channel = {'rf_spend_yt': 'yt'}
  organic_reach_to_channel = {'organic_reach_newsletter': 'newsletter'}
  organic_frequency_to_channel = {'organic_frequency_newsletter': 'newsletter'}

  data_loader = DataFrameDataLoader(
      df=df,
      coord_to_columns=coord_to_columns,
      kpi_type='non-revenue',
      media_to_channel=media_to_channel,
      media_spend_to_channel=media_spend_to_channel,
      reach_to_channel=reach_to_channel,
      frequency_to_channel=frequency_to_channel,
      rf_spend_to_channel=rf_spend_to_channel,
      organic_reach_to_channel=organic_reach_to_channel,
      organic_frequency_to_channel=organic_frequency_to_channel,
  )
  data = data_loader.load()
  ```

  Attributes:
    df: The `pd.DataFrame` object to read from. One of the following conditions
      is required:

      *   There are no NAs in the dataframe
      *   For any number of initial periods there is only media data and NAs in
          all of the non-media data columns (`kpi`, `revenue_per_kpi`,
          `media_spend`, `controls`, and `population`).

    coord_to_columns: A `CoordToColumns` object whose fields are the desired
      coordinates of the `InputData` and the values are the current names of
      columns (or lists of columns) in the DataFrame. Example:

      ```
      coord_to_columns = CoordToColumns(
          geo='dmas',
          time='dates',
          kpi='conversions',
          revenue_per_kpi='revenue_per_conversions',
          media=['impressions_tv', 'impressions_yt', 'impressions_search'],
          spend=['spend_tv', 'spend_yt', 'spend_search'],
          controls=['control_income'],
          population=population,
      )
      ```

    kpi_type: A string denoting whether the KPI is of a `'revenue'` or
      `'non-revenue'` type. When the `kpi_type` is `'non-revenue'` and there
      exists a `revenue_per_kpi`, ROI calibration is used and the analysis is
      run on revenue. When the `revenue_per_kpi` doesn't exist for the same
      `kpi_type`, custom ROI calibration is used and the analysis is run on KPI.
    media_to_channel: A dictionary whose keys are the actual column names for
      `media` data in the dataframe, and the values are the desired channel
      names. These are the same as for the `media_spend` data. Example:

      ```
      media_to_channel = {'media_tv': 'tv', 'media_yt': 'yt', 'media_fb': 'fb'}
      ```

    media_spend_to_channel: A dictionary whose keys are the actual column names
      for `media_spend` data in the dataframe, and the values are the desired
      channel names. These are same as for the `media` data. Example:

      ```
      media_spend_to_channel = {
          'spend_tv': 'tv', 'spend_yt': 'yt', 'spend_fb': 'fb'
      }
      ```

    reach_to_channel: A dictionary whose keys are the actual column names for
      `reach` data in the dataframe, and the values are the desired channel
      names. These are the same as for the `rf_spend` data. Example:

      ```
      reach_to_channel = {'reach_tv': 'tv', 'reach_yt': 'yt', 'reach_fb': 'fb'}
      ```

    frequency_to_channel: A dictionary whose keys are the actual column names
      for `frequency` data in the dataframe, and the values are the desired
      channel names. These are the same as for the `rf_spend` data. Example:

      ```
      frequency_to_channel = {
          'frequency_tv': 'tv', 'frequency_yt': 'yt', 'frequency_fb': 'fb'
      }
      ```

    rf_spend_to_channel: A dictionary whose keys are the actual column names for
      `rf_spend` data in the dataframe, and values are the desired channel
      names. These are the same as for the `reach` and `frequency` data.
      Example:

      ```
      rf_spend_to_channel = {
          'rf_spend_tv': 'tv', 'rf_spend_yt': 'yt', 'rf_spend_fb': 'fb'
      }
      ```

    organic_reach_to_channel: A dictionary whose keys are the actual column names
      for `organic_reach` data in the dataframe, and the values are the desired
      channel names. These are the same as for the `organic_frequency` data.
      Example:

      ```
      organic_reach_to_channel = {
          'organic_reach_newsletter': 'newsletter',
      }
      ```

    organic_frequency_to_channel: A dictionary whose keys are the actual column
      names for `organic_frequency` data in the dataframe, and the values are
      the desired channel names. These are the same as for the `organic_reach`
      data. Example:

      ```
      organic_frequency_to_channel = {
          'organic_frequency_newsletter': 'newsletter',
      }
      ```
  """  # pyformat: disable

  df: pd.DataFrame
  coord_to_columns: CoordToColumns
  kpi_type: str
  media_to_channel: Mapping[str, str] | None = None
  media_spend_to_channel: Mapping[str, str] | None = None
  reach_to_channel: Mapping[str, str] | None = None
  frequency_to_channel: Mapping[str, str] | None = None
  rf_spend_to_channel: Mapping[str, str] | None = None
  organic_reach_to_channel: Mapping[str, str] | None = None
  organic_frequency_to_channel: Mapping[str, str] | None = None

  def __post_init__(self):
    # If [key] in the following dict exists as an attribute in
    # `coord_to_columns`, then the corresponding attribute must exist in this
    # loader instance.
    required_mappings = immutabledict.immutabledict({
        'media': 'media_to_channel',
        'media_spend': 'media_spend_to_channel',
        'reach': 'reach_to_channel',
        'frequency': 'frequency_to_channel',
        'rf_spend': 'rf_spend_to_channel',
        'organic_reach': 'organic_reach_to_channel',
        'organic_frequency': 'organic_frequency_to_channel',
    })
    for coord_name, channel_dict in required_mappings.items():
      if (
          getattr(self.coord_to_columns, coord_name, None) is not None
          and getattr(self, channel_dict, None) is None
      ):
        raise ValueError(
            f"When {coord_name} data is provided, '{channel_dict}' is required."
        )

  def load(self) -> input_data.InputData:
    """Reads data from a dataframe and returns an InputData object."""

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=self.kpi_type
    ).with_kpi(
        self.df,
        self.coord_to_columns.kpi,
        self.coord_to_columns.time,
        self.coord_to_columns.geo,
    )
    if self.coord_to_columns.population in self.df.columns:
      builder.with_population(
          self.df, self.coord_to_columns.population, self.coord_to_columns.geo
      )
    if self.coord_to_columns.controls is not None:
      builder.with_controls(
          self.df,
          list(self.coord_to_columns.controls),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    if self.coord_to_columns.non_media_treatments is not None:
      builder.with_non_media_treatments(
          self.df,
          list(self.coord_to_columns.non_media_treatments),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    if self.coord_to_columns.revenue_per_kpi is not None:
      builder.with_revenue_per_kpi(
          self.df,
          self.coord_to_columns.revenue_per_kpi,
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    if (
        self.coord_to_columns.media is not None
        and self.media_to_channel is not None
    ):
      builder.with_media(
          self.df,
          list(self.coord_to_columns.media),
          list(self.coord_to_columns.media_spend),
          list(self.media_to_channel.values()),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )

    if (
        self.coord_to_columns.reach is not None
        and self.reach_to_channel is not None
    ):
      builder.with_reach(
          self.df,
          list(self.coord_to_columns.reach),
          list(self.coord_to_columns.frequency),
          list(self.coord_to_columns.rf_spend),
          list(self.reach_to_channel.values()),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    if self.coord_to_columns.organic_media is not None:
      builder.with_organic_media(
          self.df,
          list(self.coord_to_columns.organic_media),
          list(self.coord_to_columns.organic_media),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    if (
        self.coord_to_columns.organic_reach is not None
        and self.organic_reach_to_channel is not None
    ):
      builder.with_organic_reach(
          self.df,
          list(self.coord_to_columns.organic_reach),
          list(self.coord_to_columns.organic_frequency),
          list(self.organic_reach_to_channel.values()),
          self.coord_to_columns.time,
          self.coord_to_columns.geo,
      )
    return builder.build()


class CsvDataLoader(InputDataLoader):
  """Reads data from a CSV file.

  This class reads input data from a CSV file. The `coord_to_columns` attribute
  stores a mapping from target `InputData` coordinates and array names to the
  CSV column names, if they are different. The fields are:

  *   `geo`, `time`, `kpi`, `revenue_per_kpi`, `population` (single column)
  *   `controls` (multiple columns, optional)
  *   (1) `media`, `media_spend` (multiple columns)
  *   (2) `reach`, `frequency`, `rf_spend` (multiple columns)
  *   `non_media_treatments` (multiple columns, optional)
  *   `organic_media` (multiple columns, optional)
  *   `organic_reach`, `organic_frequency` (multiple columns, optional)

  The DataFrame must include either (1) or (2), but doesn't need to include
  both.

  Note: Time column values must be formatted using the _yyyy-mm-dd_ date format.

  Internally, this class reads the CSV file into a Pandas DataFrame and then
  loads the data using `DataFrameDataLoader`.

  Note: In a national model, `geo` and `population` are optional. If
  `population` is provided, it is reset to a default value of `1.0`.
  """

  def __init__(
      self,
      csv_path: str,
      coord_to_columns: CoordToColumns,
      kpi_type: str,
      media_to_channel: Mapping[str, str] | None = None,
      media_spend_to_channel: Mapping[str, str] | None = None,
      reach_to_channel: Mapping[str, str] | None = None,
      frequency_to_channel: Mapping[str, str] | None = None,
      rf_spend_to_channel: Mapping[str, str] | None = None,
      organic_reach_to_channel: Mapping[str, str] | None = None,
      organic_frequency_to_channel: Mapping[str, str] | None = None,
  ):
    """Constructor.

    Reads CSV file into a Pandas DataFrame and uses it to create a
    `DataFrameDataLoader`.

    Args:
      csv_path: The path to the CSV file to read from. One of the following
        conditions is required:

        *   There are no gaps in the data.
        *   For up to `max_lag` initial periods there is only media data and
            empty cells in all the data columns different from `media`, `reach`,
            `frequency`, `organic_media`, `organic_reach` and
            `organic_frequency` (`kpi`, `revenue_per_kpi`, `media_spend`,
            `rf_spend`, `controls`, `population` and `non_media_treatments`).

      coord_to_columns: A `CoordToColumns` object whose fields are the desired
        coordinates of the `InputData` and the values are the current names of
        columns (or lists of columns) in the CSV file. Example:

        ```
        coord_to_columns = CoordToColumns(
            geo='dmas',
            time='dates',
            kpi='revenue',
            revenue_per_kpi='revenue_per_conversions',
            media=['impressions_tv', impressions_yt', 'impressions_search'],
            spend=['spend_tv', 'spend_yt', 'spend_search'],
            reach=['reach_fb'],
            frequency=['frequency_fb'],
            rf_spend=['rf_spend_fb'],
            controls=['control_income'],
            population='population',
            non_media_treatments=['price', 'discount'],
            organic_media=['organic_impressions_blog'],
            organic_reach=['organic_reach_newsletter'],
            organic_frequency=['organic_frequency_newsletter'],
        )
        ```

      kpi_type: A string denoting whether the KPI is of a `'revenue'` or
        `'non-revenue'` type. When the `kpi_type` is `'non-revenue'` and there
        exists a `revenue_per_kpi`, ROI calibration is used and the analysis is
        run on revenue. When the `revenue_per_kpi` doesn't exist for the same
        `kpi_type`, custom ROI calibration is used and the analysis is run on
        KPI.
      media_to_channel: A dictionary whose keys are the actual column names for
        media data in the CSV file and values are the desired channel names, the
        same as for the `media_spend` data. Example:

        ```
        media_to_channel = {
            'media_tv': 'tv', 'media_yt': 'yt', 'media_fb': 'fb'
        }
        ```

      media_spend_to_channel: A dictionary whose keys are the actual column
        names for `media_spend` data in the CSV file and values are the desired
        channel names, the same as for the `media` data. Example:

        ```
        `media_spend_to_channel = {
            'spend_tv': 'tv', 'spend_yt': 'yt', 'spend_fb': 'fb'
        }
        ```

      reach_to_channel: A dictionary whose keys are the actual column names for
        `reach` data in the dataframe and values are the desired channel names,
        the same as for the `rf_spend` data. Example:

        ```
        reach_to_channel = {
            'reach_tv': 'tv', 'reach_yt': 'yt', 'reach_fb': 'fb'
        }
        ```

      frequency_to_channel: A dictionary whose keys are the actual column names
        for `frequency` data in the dataframe and values are the desired channel
        names, the same as for the `rf_spend` data. Example:

        ```
        frequency_to_channel = {
            'frequency_tv': 'tv', 'frequency_yt': 'yt', 'frequency_fb': 'fb'
        }
        ```

      rf_spend_to_channel: A dictionary whose keys are the actual column names
        for `rf_spend` data in the dataframe and values are the desired channel
        names, the same as for the `reach` and `frequency` data. Example:

        ```
        rf_spend_to_channel = {
            'rf_spend_tv': 'tv', 'rf_spend_yt': 'yt', 'rf_spend_fb': 'fb'
        }
        ```

      organic_reach_to_channel: A dictionary whose keys are the actual column
        names for `organic_reach` data in the dataframe and values are the
        desired channel names, the same as for the `organic_frequency`. Example:

        ```
        organic_reach_to_channel = {
            'organic_reach_newsletter': 'newsletter',
        }
        ```

      organic_frequency_to_channel: A dictionary whose keys are the actual
        column names for `organic_frequency` data in the dataframe and values
        are the desired channel names, the same as for the `organic_reach`
        data. Example:

        ```
        organic_frequency_to_channel = {
            'organic_frequency_newsletter': 'newsletter',
        }
        ```

    Note: In a national model, `geo` and `population` are optional. If
    `population` is provided, it is reset to a default value of `1.0`.

    Note: If `media` data is provided, then `media_to_channel` and
    `media_spend_to_channel` are required. If `reach` and `frequency` data is
    provided, then `reach_to_channel`, `frequency_to_channel`, and
    `rf_spend_to_channel` are required.
    """  # pyformat: disable
    df = pd.read_csv(csv_path)
    self._df_loader = DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=kpi_type,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
        organic_reach_to_channel=organic_reach_to_channel,
        organic_frequency_to_channel=organic_frequency_to_channel,
    )

  def load(self) -> input_data.InputData:
    """Reads data from a CSV file and returns an `InputData` object."""

    return self._df_loader.load()
