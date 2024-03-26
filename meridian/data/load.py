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

"""Contains classes and methods to load input data for Meridian.

The `InputDataLoader` abstract class defines a single method: `load()` which
reads data from any of the supported sources and stores it as an `InputData`
object.
"""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import datetime as dt
import warnings

import immutabledict
from meridian import constants
from meridian.data import input_data
import numpy as np
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
      on revenue. When `revenue_per_kpi` doesn't exist for the same
      `kpi_type`, custom ROI calibration is used and the analysis is run on KPI.

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
    `control_variable`, `geo` (optional for a national model) and either
    `media_channel`, `rf_channel`, or both.

    Coordinate labels for `time` and `media_time` must be formatted in
    `"yyyy-mm-dd"` date format.

    In a geo model, the dataset should consist of the following arrays of the
    following dimensions. We use `1` to indicate a required dimension with
    length 1:

    *   `kpi`: `(geo, time)`
    *   `revenue_per_kpi`: `(geo, time)`
    *   `controls`: `(geo, time, control_variable)`
    *   `population`: `(geo)`
    *   `media`: `(geo, media_time, media_channel)` - optional
    *   `media_spend`: `(geo, time, media_channel)`, `(1, time, media_channel)`,
        `(geo, 1, media_channel)`, `(media_channel)` - optional
    *   `reach`: `(geo, media_time, rf_channel)` - optional
    *   `frequency`: `(geo, media_time, rf_channel)` - optional
    *   `rf_spend`: `(geo, time, rf_channel)`, `(1, time, rf_channel)`,
        `(geo, 1, rf_channel)`, or `(rf_channel)` - optional

    In a national model, the dataset should consist of the following arrays of
    the following dimensions. We use `[1,]` to indicate an optional dimension
    with length 1:

    *   `kpi`: `([1,] time)`
    *   `revenue_per_kpi`: `([1,] time)`
    *   `controls`: `([1,] time, control_variable)`
    *   `population`: `([1],)` - this array is optional for national data
    *   `media`: `([1,] media_time, media_channel)` - optional
    *   `media_spend`: `([1,] time, media_channel)` or
        `([1,], [1,], media_channel)` - optional
    *   `reach`: `([1,] media_time, rf_channel)` - optional
    *   `frequency`: `([1,] media_time, rf_channel)` - optional
    *   `rf_spend`: `([1,] time, rf_channel)` or `([1,], [1,], rf_channel)` -
        optional

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
        and/or `rf_channel`, `control_variable`) or array names (`kpi`,
        `revenue_per_kpi`, `media`, `media_spend` and/or `rf_spend`,
        `controls`, `population`). Mapping must be provided if the names in the
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
    if (constants.GEO) not in self.dataset.dims.keys():
      self.dataset = self.dataset.expand_dims(dim=[constants.GEO], axis=0)

    if len(self.dataset.coords[constants.GEO]) == 1:
      if constants.POPULATION in self.dataset.data_vars.keys():
        warnings.warn(
            'The `population` argument is ignored in a nationally aggregated'
            ' model. It will be reset to [1]'
        )
        self.dataset = self.dataset.drop_vars(names=[constants.POPULATION])

      # Add a default `population` [1].
      national_population_darray = xr.DataArray(
          [constants.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE],
          dims=[constants.GEO],
          coords={
              constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          },
          name=constants.POPULATION,
      )
      self.dataset = xr.combine_by_coords(
          [
              national_population_darray,
              self.dataset.assign_coords(
                  {constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]}
              ),
          ],
          compat='override',
      )

    self._normalize_time_coordinates(constants.TIME)
    self._normalize_time_coordinates(constants.MEDIA_TIME)
    self._validate_dataset()

  def _normalize_time_coordinates(self, dim: str):
    if self.dataset.coords.dtypes[dim] == np.dtype('datetime64[ns]'):
      date_strvalues = np.datetime_as_string(self.dataset.coords[dim], unit='D')
      self.dataset = self.dataset.assign_coords({dim: date_strvalues})

    # Assume that the time coordinate labels are date-formatted strings.
    # We don't currently support other, arbitrary object types in the loaders.
    for time in self.dataset.coords[dim].values:
      try:
        _ = dt.datetime.strptime(time, constants.DATE_FORMAT)
      except ValueError as exc:
        raise ValueError(
            f"Invalid time label: '{time}'. Expected format:"
            f" '{constants.DATE_FORMAT}'"
        ) from exc

  def _validate_dataset(self):
    for coord_name in constants.REQUIRED_INPUT_DATA_COORD_NAMES:
      if coord_name not in self.dataset.coords:
        raise ValueError(
            f"Coordinate '{coord_name}' not found in dataset's coordinates."
            " Please use the 'name_mapping' argument to rename the coordinates."
        )

    for array_name in constants.REQUIRED_INPUT_DATA_ARRAY_NAMES:
      if array_name not in self.dataset.data_vars:
        raise ValueError(
            f"Array '{array_name}' not found in dataset's arrays."
            " Please use the 'name_mapping' argument to rename the arrays."
        )

    # Check for media.
    missing_media_input = []
    for coord_name in constants.MEDIA_INPUT_DATA_COORD_NAMES:
      if coord_name not in self.dataset.coords:
        missing_media_input.append(coord_name)
    for array_name in constants.MEDIA_INPUT_DATA_ARRAY_NAMES:
      if array_name not in self.dataset.data_vars:
        missing_media_input.append(array_name)

    # Check for RF.
    missing_rf_input = []
    for coord_name in constants.RF_INPUT_DATA_COORD_NAMES:
      if coord_name not in self.dataset.coords:
        missing_rf_input.append(coord_name)
    for array_name in constants.RF_INPUT_DATA_ARRAY_NAMES:
      if array_name not in self.dataset.data_vars:
        missing_rf_input.append(array_name)

    if missing_media_input and missing_rf_input:
      raise ValueError(
          "Some required data is missing. Please use the 'name_mapping'"
          ' argument to rename the coordinates/arrays. It is required to have'
          ' at least one of media or reach and frequency.'
      )

    if missing_media_input and len(missing_media_input) != len(
        constants.MEDIA_INPUT_DATA_COORD_NAMES
    ) + len(constants.MEDIA_INPUT_DATA_ARRAY_NAMES):
      raise ValueError(
          f"Media data is partially missing. '{missing_media_input}' not found"
          " in dataset's coordinates/arrays. Please use the 'name_mapping'"
          ' argument to rename the coordinates/arrays.'
      )

    if missing_rf_input and len(missing_rf_input) != len(
        constants.RF_INPUT_DATA_COORD_NAMES
    ) + len(constants.RF_INPUT_DATA_ARRAY_NAMES):
      raise ValueError(
          f"RF data is partially missing. '{missing_rf_input}' not found in"
          " dataset's coordinates/arrays. Please use the 'name_mapping'"
          ' argument to rename the coordinates/arrays.'
      )

  def load(self) -> input_data.InputData:
    """Returns an `InputData` object containing the data from the dataset."""
    revenue_per_kpi = (
        self.dataset.revenue_per_kpi
        if constants.REVENUE_PER_KPI in self.dataset.data_vars.keys()
        else None
    )
    media = (
        self.dataset.media
        if constants.MEDIA in self.dataset.data_vars.keys()
        else None
    )
    media_spend = (
        self.dataset.media_spend
        if constants.MEDIA in self.dataset.data_vars.keys()
        else None
    )
    reach = (
        self.dataset.reach
        if constants.REACH in self.dataset.data_vars.keys()
        else None
    )
    frequency = (
        self.dataset.frequency
        if constants.FREQUENCY in self.dataset.data_vars.keys()
        else None
    )
    rf_spend = (
        self.dataset.rf_spend
        if constants.RF_SPEND in self.dataset.data_vars.keys()
        else None
    )
    return input_data.InputData(
        kpi=self.dataset.kpi,
        kpi_type=self.kpi_type,
        revenue_per_kpi=revenue_per_kpi,
        controls=self.dataset.controls,
        population=self.dataset.population,
        media=media,
        media_spend=media_spend,
        reach=reach,
        frequency=frequency,
        rf_spend=rf_spend,
    )


@dataclasses.dataclass(frozen=True)
class CoordToColumns:
  """A mapping between the desired and actual column names in the input data.

  Attributes:
    controls: List of column names containing `controls` values in the input
      data.
    time: Name of column containing `time` values in the input data.
    kpi: Name of column containing `kpi` values in the input data.
    revenue_per_kpi: Name of column containing `revenue_per_kpi` values in the
      input data.
    geo:  Name of column containing `geo` values in the input data. This field
      is optional for a national model.
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
  """

  controls: Sequence[str]
  time: str = constants.TIME
  kpi: str = constants.KPI
  revenue_per_kpi: str | None = None
  geo: str = constants.GEO
  population: str = constants.POPULATION
  # Media data
  media: Sequence[str] | None = None
  media_spend: Sequence[str] | None = None
  # RF data
  reach: Sequence[str] | None = None
  frequency: Sequence[str] | None = None
  rf_spend: Sequence[str] | None = None

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
  *   `controls` (multiple columns)
  *   (1) `media`, `media_spend` (multiple columns)
  *   (2) `reach`, `frequency`, `rf_spend` (multiple columns)

  The `DataFrame` must include (1) or (2), but doesn't need to include both.
  Also, each media channel must appear in (1) or (2), but not both.

  Note: Time column values must be formatted in _yyyy-mm-dd_ date format.

  Note: In a national model, `geo` and `population` are optional. If the
  `population` is provided, it is reset to a default value of `1.0`.

  Note: If `media` data is provided, then `media_to_channel` and
  `media_spend_to_channel` are required. If `reach` and `frequency` data is
  provided, then `reach_to_channel` and `frequency_to_channel` and
  `rf_spend_to_channel` are required.

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

  data_loader = DataFrameDataLoader(
      df=df,
      coord_to_columns=coord_to_columns,
      kpi_type='non-revenue',
      media_to_channel=media_to_channel,
      media_spend_to_channel=media_spend_to_channel,
      reach_to_channel=reach_to_channel,
      frequency_to_channel=frequency_to_channel,
      rf_spend_to_channel=rf_spend_to_channel
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
  """

  df: pd.DataFrame
  coord_to_columns: CoordToColumns
  kpi_type: str
  media_to_channel: Mapping[str, str] | None = None
  media_spend_to_channel: Mapping[str, str] | None = None
  reach_to_channel: Mapping[str, str] | None = None
  frequency_to_channel: Mapping[str, str] | None = None
  rf_spend_to_channel: Mapping[str, str] | None = None

  # If [key] in the following dict exists as an attribute in `coord_to_columns`,
  # then the corresponding attribute must exist in this loader instance.
  _required_mappings = immutabledict.immutabledict({
      'media': 'media_to_channel',
      'media_spend': 'media_spend_to_channel',
      'reach': 'reach_to_channel',
      'frequency': 'frequency_to_channel',
      'rf_spend': 'rf_spend_to_channel',
  })

  def __post_init__(self):
    self._validate_and_normalize_time_values()
    self._expand_if_national()
    self._validate_column_names()
    self._validate_required_mappings()
    self._validate_geo_and_time()
    self._validate_nas()

  def _validate_and_normalize_time_values(self):
    """Validates that time values are in the conventional Meridian format.

    Time values are expected to be (a) strings formatted in `"yyyy-mm-dd"` or
    (b) `datetime` values as numpy's `datetime64` types. All other types are
    not currently supported.

    In (b) case, `datetime` coordinate values will be normalized as formatted
    strings.
    """
    time_column_name = self.coord_to_columns.time

    if self.df.dtypes[time_column_name] == np.dtype('datetime64[ns]'):
      self.df[time_column_name] = self.df[time_column_name].map(
          lambda time: time.strftime(constants.DATE_FORMAT)
      )
    else:
      # Assume that the `time` column values are strings formatted as dates.
      for _, time in self.df[time_column_name].items():
        try:
          _ = dt.datetime.strptime(time, constants.DATE_FORMAT)
        except ValueError as exc:
          raise ValueError(
              f"Invalid time label: '{time}'. Expected format:"
              f" '{constants.DATE_FORMAT}'"
          ) from exc

  def _validate_column_names(self):
    """Validates the column names in `df` and `coord_to_columns`."""

    desired_columns = []
    for field in dataclasses.fields(self.coord_to_columns):
      value = getattr(self.coord_to_columns, field.name)
      if isinstance(value, str):
        desired_columns.append(value)
      elif isinstance(value, Sequence):
        for column in value:
          desired_columns.append(column)
    desired_columns = sorted(desired_columns)

    actual_columns = sorted(self.df.columns.to_list())
    if any(d not in actual_columns for d in desired_columns):
      raise ValueError(
          f'Values of the `coord_to_columns` object {desired_columns}'
          f' should map to the DataFrame column names {actual_columns}.'
      )

  def _expand_if_national(self):
    """Adds geo/population columns in a national model if necessary."""

    geo_column_name = self.coord_to_columns.geo
    population_column_name = self.coord_to_columns.population

    if geo_column_name not in self.df.columns:
      self.df[geo_column_name] = constants.NATIONAL_MODEL_DEFAULT_GEO_NAME

    if len(set(self.df[geo_column_name].values)) == 1:
      self.df[geo_column_name] = constants.NATIONAL_MODEL_DEFAULT_GEO_NAME
      if population_column_name in self.df.columns:
        warnings.warn(
            'The `population` argument is ignored in a nationally aggregated'
            ' model. It will be reset to [1, 1, ..., 1]'
        )
        self.df[population_column_name] = (
            constants.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE
        )

    if population_column_name not in self.df.columns:
      self.df[population_column_name] = (
          constants.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE
      )

  def _validate_required_mappings(self):
    """Validates required mappings in `coord_to_columns`."""
    for coord_name, channel_dict in self._required_mappings.items():
      if (
          getattr(self.coord_to_columns, coord_name, None) is not None
          and getattr(self, channel_dict, None) is None
      ):
        raise ValueError(
            f"When {coord_name} data is provided, '{channel_dict}' is required."
        )

  def _validate_geo_and_time(self):
    """Validates that for every geo the list of `time`s is the same."""
    geo_column_name = self.coord_to_columns.geo
    time_column_name = self.coord_to_columns.time

    df_grouped = self.df.sort_values(time_column_name).groupby(
        geo_column_name, sort=False
    )[time_column_name]
    if any(df_grouped.count() != df_grouped.nunique()):
      raise ValueError("Duplicate entries found in the 'time' column.")

    times_by_geo = df_grouped.apply(list)
    if any(t != times_by_geo[0] for t in times_by_geo[1:]):
      raise ValueError(
          "Values in the 'time' column not consistent across different geos."
      )

  def _validate_nas(self):
    """Validates that the only NAs are in the lagged-media period."""
    # Check if there are no NAs in media.
    if self.coord_to_columns.media is not None:
      if self.df[self.coord_to_columns.media].isna().any(axis=None):
        raise ValueError('NA values found in the media columns.')

    # Check if there are no NAs in reach & frequency.
    if self.coord_to_columns.reach is not None:
      if self.df[self.coord_to_columns.reach].isna().any(axis=None):
        raise ValueError('NA values found in the reach columns.')
    if self.coord_to_columns.frequency is not None:
      if self.df[self.coord_to_columns.frequency].isna().any(axis=None):
        raise ValueError('NA values found in the frequency columns.')

    # Determine columns in which NAs are expected in the lagged-media period.
    na_columns = []
    coords = [
        constants.KPI,
        constants.CONTROLS,
        constants.POPULATION,
    ]
    if self.coord_to_columns.revenue_per_kpi is not None:
      coords.append(constants.REVENUE_PER_KPI)
    if self.coord_to_columns.media_spend is not None:
      coords.append(constants.MEDIA_SPEND)
    if self.coord_to_columns.rf_spend is not None:
      coords.append(constants.RF_SPEND)
    for coord in coords:
      columns = getattr(self.coord_to_columns, coord)
      columns = [columns] if isinstance(columns, str) else columns
      na_columns.extend(columns)

    # Dates with at least one non-NA value in non-media columns
    time_column_name = self.coord_to_columns.time
    no_na_period = self.df[(~self.df[na_columns].isna()).any(axis=1)][
        time_column_name
    ].unique()

    # Dates with 100% NA values in all non-media columns.
    na_period = [
        t for t in self.df[time_column_name].unique() if t not in no_na_period
    ]

    # Check if na_period is a continuous window starting from the earliest time
    # period.
    if not np.all(
        np.sort(na_period)
        == np.sort(self.df[time_column_name].unique())[: len(na_period)]
    ):
      raise ValueError(
          "The 'lagged media' period (period with 100% NA values in all"
          f' non-media columns) {na_period} is not a continuous window starting'
          ' from the earliest time period.'
      )

    # Check if for the non-lagged period, there are no NAs in non-media data.
    not_lagged_data = self.df.loc[
        self.df[time_column_name].isin(no_na_period),
        na_columns,
    ]
    if not_lagged_data.isna().any(axis=None):
      raise ValueError(
          'NA values found in non-media columns outside the lagged-media'
          f' period {na_period} (continuous window of 100% NA values in all'
          ' non-media columns).'
      )

  def load(self) -> input_data.InputData:
    """Reads data from a dataframe and returns an InputData object."""

    # Change geo strings to numbers to keep the order of geos. The .to_xarray()
    # method from Pandas sorts lexicographically by the key columns, so if the
    # geos were unsorted strings, it would change their order.
    geo_column_name = self.coord_to_columns.geo
    time_column_name = self.coord_to_columns.time
    geo_names = self.df[geo_column_name].unique()
    self.df[geo_column_name] = self.df[geo_column_name].replace(
        dict(zip(geo_names, np.arange(len(geo_names))))
    )
    df_indexed = self.df.set_index([geo_column_name, time_column_name])

    kpi_xr = (
        df_indexed[self.coord_to_columns.kpi]
        .dropna()
        .rename(constants.KPI)
        .rename_axis([constants.GEO, constants.TIME])
        .to_frame()
        .to_xarray()
    )
    population_xr = (
        df_indexed[self.coord_to_columns.population]
        .groupby(geo_column_name)
        .mean()
        .rename(constants.POPULATION)
        .rename_axis([constants.GEO])
        .to_frame()
        .to_xarray()
    )
    controls_xr = (
        df_indexed[self.coord_to_columns.controls]
        .stack()
        .rename(constants.CONTROLS)
        .rename_axis(
            [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE]
        )
        .to_frame()
        .to_xarray()
    )
    dataset = xr.combine_by_coords([kpi_xr, population_xr, controls_xr])

    if self.coord_to_columns.revenue_per_kpi is not None:
      revenue_per_kpi_xr = (
          df_indexed[self.coord_to_columns.revenue_per_kpi]
          .dropna()
          .rename(constants.REVENUE_PER_KPI)
          .rename_axis([constants.GEO, constants.TIME])
          .to_frame()
          .to_xarray()
      )
      dataset = xr.combine_by_coords([dataset, revenue_per_kpi_xr])
    if self.coord_to_columns.media is not None:
      media_xr = (
          df_indexed[self.coord_to_columns.media]
          .stack()
          .rename(constants.MEDIA)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      media_xr.coords[constants.MEDIA_CHANNEL] = [
          self.media_to_channel[x]
          for x in media_xr.coords[constants.MEDIA_CHANNEL].values
      ]

      media_spend_xr = (
          df_indexed[self.coord_to_columns.media_spend]
          .stack()
          .rename(constants.MEDIA_SPEND)
          .rename_axis([constants.GEO, constants.TIME, constants.MEDIA_CHANNEL])
          .to_frame()
          .to_xarray()
      )
      media_spend_xr.coords[constants.MEDIA_CHANNEL] = [
          self.media_spend_to_channel[x]
          for x in media_spend_xr.coords[constants.MEDIA_CHANNEL].values
      ]
      dataset = xr.combine_by_coords([dataset, media_xr, media_spend_xr])

    if self.coord_to_columns.reach is not None:
      reach_xr = (
          df_indexed[self.coord_to_columns.reach]
          .stack()
          .rename(constants.REACH)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      reach_xr.coords[constants.RF_CHANNEL] = [
          self.reach_to_channel[x]
          for x in reach_xr.coords[constants.RF_CHANNEL].values
      ]

      frequency_xr = (
          df_indexed[self.coord_to_columns.frequency]
          .stack()
          .rename(constants.FREQUENCY)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      frequency_xr.coords[constants.RF_CHANNEL] = [
          self.frequency_to_channel[x]
          for x in frequency_xr.coords[constants.RF_CHANNEL].values
      ]

      rf_spend_xr = (
          df_indexed[self.coord_to_columns.rf_spend]
          .stack()
          .rename(constants.RF_SPEND)
          .rename_axis([constants.GEO, constants.TIME, constants.RF_CHANNEL])
          .to_frame()
          .to_xarray()
      )
      rf_spend_xr.coords[constants.RF_CHANNEL] = [
          self.rf_spend_to_channel[x]
          for x in rf_spend_xr.coords[constants.RF_CHANNEL].values
      ]
      dataset = xr.combine_by_coords(
          [dataset, reach_xr, frequency_xr, rf_spend_xr]
      )

    # Change back to geo names
    self.df[geo_column_name] = self.df[geo_column_name].replace(
        dict(zip(np.arange(len(geo_names)), geo_names))
    )
    dataset.coords[constants.GEO] = geo_names
    return XrDatasetDataLoader(dataset, kpi_type=self.kpi_type).load()


class CsvDataLoader(InputDataLoader):
  """Reads data from a CSV file.

  This class reads input data from a CSV file. The `coord_to_columns` attribute
  stores a mapping from target `InputData` coordinates and array names to the
  CSV column names, if they are different. The fields are:

  *   `geo`, `time`, `kpi`, `revenue_per_kpi`, `population` (single column)
  *   `controls` (multiple columns)
  *   (1) `media`, `media_spend` (multiple columns)
  *   (2) `reach`, `frequency`, `rf_spend` (multiple columns)

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
  ):
    """Constructor.

    Reads CSV file into a Pandas DataFrame and uses it to create a
    `DataFrameDataLoader`.

    Args:
      csv_path: The path to the CSV file to read from. One of the following
        conditions is required:

        *   There are no gaps in the data.
        *   For up to `max_lag` initial periods there is only media data and
            empty cells in all the non-media data columns (`kpi`,
            `revenue_per_kpi`, `media_spend`, `controls`, and `population`).

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
            controls=['control_income'],
            population='population'
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

    Note: In a national model, `geo` and `population` are optional. If
    `population` is provided, it is reset to a default value of `1.0`.

    Note: If `media` data is provided, then `media_to_channel` and
    `media_spend_to_channel` are required. If `reach` and `frequency` data is
    provided, then `reach_to_channel`, `frequency_to_channel`, and
    `rf_spend_to_channel` are required.

    """
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
    )

  def load(self) -> input_data.InputData:
    """Reads data from a CSV file and returns an `InputData` object."""

    return self._df_loader.load()
