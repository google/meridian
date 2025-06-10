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

"""An implementation of `InputDataBuilder` with DataFrame primitives."""

import logging
import warnings

from meridian import constants
from meridian.data import input_data_builder
import pandas as pd


class DataFrameInputDataBuilder(input_data_builder.InputDataBuilder):
  """Builds `InputData` from DataFrames."""

  def with_kpi(
      self,
      df: pd.DataFrame,
      kpi_col: str = constants.KPI,
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads KPI data from a DataFrame.

    Args:
      df: The DataFrame to read the KPI data from.
      kpi_col: The name of the column containing the KPI values. If not
        provided, the default name is `kpi`.
      time_col: The name of the column containing the time coordinates. If not
        provided, the default name is `time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added KPI data.
    """
    kpi_df = df.copy()

    ### Validate ###
    self._validate_cols(kpi_df, [kpi_col, time_col], [geo_col])
    self._validate_coords(kpi_df, geo_col, time_col)

    ### Transform ###
    data = kpi_df.set_index([geo_col, time_col])[kpi_col].dropna()
    self.kpi = (
        data.rename(constants.KPI)
        .rename_axis([constants.GEO, constants.TIME])
        .to_xarray()
    )
    return self

  def with_controls(
      self,
      df: pd.DataFrame,
      control_cols: list[str],
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads controls data from a DataFrame.

    Args:
      df: The DataFrame to read the controls data from.
      control_cols: The names of the columns containing the controls values.
      time_col: The name of the column containing the time coordinates. If not
        provided, the default name is `time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added controls data.
    """
    controls_df = df.copy()

    ### Validate ###
    self._validate_cols(
        controls_df,
        control_cols + [time_col],
        [geo_col],
    )
    self._validate_coords(controls_df, geo_col, time_col)

    ### Transform ###
    data = controls_df.set_index([geo_col, time_col])[control_cols].stack()
    self.controls = (
        data.rename(constants.CONTROLS)
        .rename_axis(
            [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE]
        )
        .to_xarray()
    )
    return self

  def with_population(
      self,
      df: pd.DataFrame,
      population_col: str = constants.POPULATION,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads population data from a DataFrame.

    Args:
      df: The DataFrame to read the population data from.
      population_col: The name of the column containing the population values.
        If not provided, the default name is `population`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added population data.
    """
    population_df = df.copy()

    ### Validate ###
    self._validate_cols(population_df, [population_col], [geo_col])
    self._validate_coords(population_df, geo_col)

    ### Transform ###
    data = (
        population_df.set_index([geo_col])[population_col]
        .groupby(geo_col)
        .mean()
    )
    self.population = (
        data.rename(constants.POPULATION)
        .rename_axis([constants.GEO])
        .to_xarray()
    )

    return self

  def with_revenue_per_kpi(
      self,
      df: pd.DataFrame,
      revenue_per_kpi_col: str = constants.REVENUE_PER_KPI,
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads revenue per KPI data from a DataFrame.

    Args:
      df: The DataFrame to read the revenue per KPI data from.
      revenue_per_kpi_col: The name of the column containing the revenue per KPI
        values. If not provided, the default name is `revenue_per_kpi`.
      time_col: The name of the column containing the time coordinates. If not
        provided, the default name is `time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added revenue per KPI data.
    """
    revenue_per_kpi_df = df.copy()

    ### Validate ###
    self._validate_cols(
        revenue_per_kpi_df,
        [revenue_per_kpi_col, time_col],
        [geo_col],
    )
    self._check_revenue_per_kpi_defaults(
        revenue_per_kpi_df, revenue_per_kpi_col
    )
    self._validate_coords(revenue_per_kpi_df, geo_col, time_col)

    ### Transform ###
    data = revenue_per_kpi_df.set_index([geo_col, time_col])[
        revenue_per_kpi_col
    ].dropna()
    self.revenue_per_kpi = (
        data.rename(constants.REVENUE_PER_KPI)
        .rename_axis([constants.GEO, constants.TIME])
        .to_xarray()
    )

    return self

  def with_media(
      self,
      df: pd.DataFrame,
      media_cols: list[str],
      media_spend_cols: list[str],
      media_channels: list[str],
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads media and media spend data from a DataFrame.

    Args:
      df: The DataFrame to read the media and media spend data from.
      media_cols: The name of the columns containing the media values.
      media_spend_cols: The name of the columns containing the media spend
        values.
      media_channels: The desired media channel coordinate names. Must match
        `media_cols` and `media_spend_cols` in length. These are also index
        mapped.
      time_col: The name of the column containing the time coordinates for media
        spend and media time coordinates for media. If not provided, the default
        name is `time`. Media time coordinates will be shorter than time
        coordinates if media spend values are missing (NaN) for some t in
        `time`. Media time must be equal or a subset of time.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added media and media spend data.
    """
    media_df = df.copy()

    ### Validate ###
    # For with_media, media and media spend columns can be the same, so we
    # validate separate when checking for duplicates.
    self._validate_cols(media_df, media_cols + [time_col], [geo_col])
    self._validate_cols(media_df, media_spend_cols + [time_col], [geo_col])
    self._validate_coords(media_df, geo_col, time_col)
    self._validate_channel_cols(media_channels, [media_cols, media_spend_cols])
    ### Transform ###
    media_data = media_df.set_index([geo_col, time_col])[media_cols]
    media_data.columns = media_channels
    self.media = (
        media_data.stack()
        .rename(constants.MEDIA)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.MEDIA_CHANNEL,
        ])
        .to_xarray()
    )
    media_spend_data = media_df.set_index([geo_col, time_col])[media_spend_cols]
    media_spend_data.columns = media_channels
    self.media_spend = (
        media_spend_data.stack()
        .rename(constants.MEDIA_SPEND)
        .rename_axis([
            constants.GEO,
            constants.TIME,
            constants.MEDIA_CHANNEL,
        ])
        .to_xarray()
    )
    return self

  def with_reach(
      self,
      df: pd.DataFrame,
      reach_cols: list[str],
      frequency_cols: list[str],
      rf_spend_cols: list[str],
      rf_channels: list[str],
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads reach, frequency, and rf spend data from a DataFrame.

    Args:
      df: The DataFrame to read the reach, frequency, and rf spend data from.
      reach_cols: The name of the columns containing the reach values.
      frequency_cols: The name of the columns containing the frequency values.
      rf_spend_cols: The name of the columns containing the rf spend values.
      rf_channels: The desired rf channel coordinate names. Must match
        `reach_cols`, `frequency_cols`, and `rf_spend_cols` in length. These are
        also index mapped.
      time_col: The name of the column containing the time coordinates for rf
        spend and media time coordinates for reach and frequency. If not
        provided, the default name is `time`. Media time coordinates will be
        shorter than time coordinates if media spend values are missing (NaN)
        for some t in `time`. Media time must be equal or a subset of time.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added reach, frequency, and rf
      spend data.
    """
    reach_df = df.copy()

    ### Validate ###
    self._validate_cols(
        reach_df,
        reach_cols + frequency_cols + rf_spend_cols + [time_col],
        [geo_col],
    )
    self._validate_coords(reach_df, geo_col, time_col)
    self._validate_channel_cols(
        rf_channels,
        [reach_cols, frequency_cols, rf_spend_cols],
    )

    ### Transform ###
    reach_data = reach_df.set_index([geo_col, time_col])[reach_cols]
    reach_data.columns = rf_channels
    self.reach = (
        reach_data.stack()
        .rename(constants.REACH)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.RF_CHANNEL,
        ])
        .to_xarray()
    )

    frequency_data = reach_df.set_index([geo_col, time_col])[frequency_cols]
    frequency_data.columns = rf_channels
    self.frequency = (
        frequency_data.stack()
        .rename(constants.FREQUENCY)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.RF_CHANNEL,
        ])
        .to_xarray()
    )

    rf_spend_data = reach_df.set_index([geo_col, time_col])[rf_spend_cols]
    rf_spend_data.columns = rf_channels
    self.rf_spend = (
        rf_spend_data.stack()
        .rename(constants.RF_SPEND)
        .rename_axis([
            constants.GEO,
            constants.TIME,
            constants.RF_CHANNEL,
        ])
        .to_xarray()
    )
    return self

  def with_organic_media(
      self,
      df: pd.DataFrame,
      organic_media_cols: list[str],
      organic_media_channels: list[str] | None = None,
      media_time_col: str = constants.MEDIA_TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads organic media data from a DataFrame.

    Args:
      df: The DataFrame to read the organic media data from.
      organic_media_cols: The name of the columns containing the organic media
        values.
      organic_media_channels: The desired organic media channel coordinate
        names. Will default to the organic media columns if not given. If
        provided, must match `organic_media_cols` in length. This is index
        mapped.
      media_time_col: The name of the column containing the media time
        coordinates. If not provided, the default name is `media_time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added organic media data.
    """
    organic_media_df = df.copy()

    ### Validate ###
    if not organic_media_channels:
      organic_media_channels = organic_media_cols
    self._validate_cols(
        organic_media_df,
        organic_media_cols + [media_time_col],
        [geo_col],
    )
    self._validate_coords(organic_media_df, geo_col, media_time_col)
    self._validate_channel_cols(
        organic_media_channels,
        [organic_media_cols],
    )

    ### Transform ###
    data = organic_media_df.set_index([geo_col, media_time_col])[
        organic_media_cols
    ]
    data.columns = organic_media_channels
    self.organic_media = (
        data.stack()
        .rename(constants.ORGANIC_MEDIA)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_MEDIA_CHANNEL,
        ])
        .to_xarray()
    )

    return self

  def with_organic_reach(
      self,
      df: pd.DataFrame,
      organic_reach_cols: list[str],
      organic_frequency_cols: list[str],
      organic_rf_channels: list[str],
      media_time_col: str = constants.MEDIA_TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads organic reach and organic frequency data from a DataFrame.

    Args:
      df: The DataFrame to read the organic reach and frequency data from.
      organic_reach_cols: The name of the columns containing the organic reach
        values.
      organic_frequency_cols: The name of the columns containing the organic
        frequency values.
      organic_rf_channels: The desired organic rf channel coordinate names. Must
        match `organic_reach_cols` and `organic_frequency_cols` in length. These
        are also index mapped.
      media_time_col: The name of the column containing the media time
        coordinates. If not provided, the default name is `media_time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added organic reach and organic
      frequency data.
    """
    organic_reach_frequency_df = df.copy()

    ### Validate ###
    self._validate_cols(
        organic_reach_frequency_df,
        organic_reach_cols + organic_frequency_cols + [media_time_col],
        [geo_col],
    )
    self._validate_coords(organic_reach_frequency_df, geo_col, media_time_col)
    self._validate_channel_cols(
        organic_rf_channels,
        [organic_reach_cols, organic_frequency_cols],
    )
    ### Transform ###
    organic_reach_data = organic_reach_frequency_df.set_index(
        [geo_col, media_time_col]
    )[organic_reach_cols]
    organic_reach_data.columns = organic_rf_channels
    self.organic_reach = (
        organic_reach_data.stack()
        .rename(constants.ORGANIC_REACH)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_RF_CHANNEL,
        ])
        .to_xarray()
    )
    organic_frequency_data = organic_reach_frequency_df.set_index(
        [geo_col, media_time_col]
    )[organic_frequency_cols]
    organic_frequency_data.columns = organic_rf_channels
    self.organic_frequency = (
        organic_frequency_data.stack()
        .rename(constants.ORGANIC_FREQUENCY)
        .rename_axis([
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_RF_CHANNEL,
        ])
        .to_xarray()
    )
    return self

  def with_non_media_treatments(
      self,
      df: pd.DataFrame,
      non_media_treatment_cols: list[str],
      time_col: str = constants.TIME,
      geo_col: str = constants.GEO,
  ) -> 'DataFrameInputDataBuilder':
    """Reads non-media treatments data from a DataFrame.

    Args:
      df: The DataFrame to read the non-media treatments data from.
      non_media_treatment_cols: The names of the columns containing the
        non-media treatments values.
      time_col: The name of the column containing the time coordinates. If not
        provided, the default name is `time`.
      geo_col: (Optional) The name of the column containing the geo coordinates.
        If not provided, the default name is `geo`. If the DataFrame provided
        has no geo column, a national model data is assumed and a geo dimension
        will be created internally with a single coordinate value
        `national_geo`.

    Returns:
      The `DataFrameInputDataBuilder` with the added non-media treatments data.
    """
    non_media_treatments_df = df.copy()

    ### Validate ###
    self._validate_cols(
        non_media_treatments_df,
        non_media_treatment_cols + [time_col],
        [geo_col],
    )
    self._validate_coords(non_media_treatments_df, geo_col, time_col)

    ### Transform ###
    data = non_media_treatments_df.set_index([geo_col, time_col])[
        non_media_treatment_cols
    ].stack()
    self.non_media_treatments = (
        data.rename(constants.NON_MEDIA_TREATMENTS)
        .rename_axis(
            [constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL]
        )
        .to_xarray()
    )
    return self

  def _validate_cols(
      self, df: pd.DataFrame, required_cols: list[str], optional_cols: list[str]
  ):
    """Validates that the DataFrame has all the expected columns and there are no duplicates."""
    if len(required_cols + optional_cols) != len(
        set(required_cols + optional_cols)
    ):
      raise ValueError(
          'DataFrame has duplicate columns from'
          f' {required_cols + optional_cols}'
      )

    if not all(column in df.columns for column in required_cols):
      raise ValueError(
          f'DataFrame is missing one or more columns from {required_cols}'
      )

  def _validate_coords(
      self,
      df: pd.DataFrame,
      geo_col: str,
      time_col: str | None = None,
  ):
    """Adds geo columns in a national model if necessary and validates that for every geo the list of `time`s is the same for non-population dfs."""
    if geo_col not in df.columns:
      df[geo_col] = constants.NATIONAL_MODEL_DEFAULT_GEO_NAME
      logging.info('DataFrame has no geo column. Assuming "National".')

    if time_col is not None:
      df_grouped = df.sort_values(time_col).groupby(geo_col)[time_col]
      if any(df_grouped.count() != df_grouped.nunique()):
        # Currently we raise errors for all duplicate geo time entries. Might
        # want to consider silently dropping dupes for column values that are
        # the same (e.g. {geo: ['a', 'a'], 'time': ['1', '1'], kpi: [120, 120]})
        raise ValueError("Duplicate entries found in the 'time' column.")

      times_by_geo = df_grouped.apply(list).reset_index(drop=True)
      if any(t != times_by_geo[0] for t in times_by_geo[1:]):
        raise ValueError(
            "Values in the 'time' column not consistent across different geos."
        )

  def _check_revenue_per_kpi_defaults(
      self, df: pd.DataFrame, revenue_per_kpi_col: str
  ):
    """Sets revenue_per_kpi to default if kpi type is revenue and with_revenue_per_kpi is called."""
    if self._kpi_type == constants.REVENUE:
      df[revenue_per_kpi_col] = 1.0
      warnings.warn(
          'with_revenue_per_kpi was called but kpi_type was set to revenue.'
          ' Assuming revenue per kpi with values [1].'
      )

  def _validate_channel_cols(
      self, channel_names: list[str], all_channel_cols: list[list[str]]
  ):
    if len(channel_names) != len(set(channel_names)):
      raise ValueError('Channel names must be unique.')
    for channel_cols in all_channel_cols:
      if len(channel_cols) != len(channel_names):
        raise ValueError(
            'Given channel columns must have same length as channel names.'
        )
