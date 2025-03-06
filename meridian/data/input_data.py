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

"""This module contains classes to store data for Meridian.

The `InputData` class is used to store all the input data to the model.
"""

from collections import abc
import dataclasses
import datetime as dt
import functools
import warnings

from meridian import constants
from meridian.data import arg_builder
from meridian.data import time_coordinates as tc
import numpy as np
import xarray as xr


__all__ = [
    "InputData",
]


def _check_dim_collection(
    array: xr.DataArray | None, dims: abc.Collection[abc.Collection[str]]
):
  """Checks if a DataArray has the correct collection of dimensions.

  Arguments:
    array: A DataArray to be verified.
    dims: A collection of acceptable collections of dimensions. In case of the
      `media_spend` and `rf_spend` arrays, there are two acceptable collections
      of dimensions: `(channel)` and `(geo, time, channel)` where `channel` is
      either `'media_channel'` or `'rf_channel'`. In case of other arrays, there
      is only one acceptable collection of dimensions.

  Raises:
    ValueError if the number of dimensions or dimension names are incorrect.
  """

  if array is not None and not any(set(array.dims) == set(d) for d in dims):
    raise ValueError(
        f"The dimension list of array '{array.name}' doesn't match any of"
        f" the following dimension lists: {dims}."
    )


def _check_dim_match(dim, arrays):
  """Verifies that the dimensions of the appropriate arrays match."""
  lengths = [len(array.coords[dim]) for array in arrays if array is not None]
  names = [array.name for array in arrays if array is not None]
  if any(l != lengths[0] for l in lengths):
    raise ValueError(
        f"'{dim}' dimensions {lengths} of arrays {names} don't match."
    )


@dataclasses.dataclass
class InputData:
  """A data container for advertising data in a format supported by Meridian.

  Attributes:
    kpi: A DataArray of dimensions `(n_geos, n_times)` containing the
      non-negative dependent variable. Typically this is the number of units
      sold, but it can be any metric, such as revenue or conversions.
    kpi_type: A string denoting whether the KPI is of a `'revenue'` or
      `'non-revenue'` type. When the `kpi_type` is `'non-revenue'` and
      `revenue_per_kpi` exists, ROI calibration is used and the analysis is run
      on revenue. When the `revenue_per_kpi` doesn't exist for the same
      `kpi_type`, custom ROI calibration is used and the analysis is run on KPI.
    controls: A DataArray of dimensions `(n_geos, n_times, n_controls)`
      containing control variable values.
    population: A DataArray of dimensions `(n_geos,)` containing the population
      of each group. This variable is used to scale the KPI and media for
      modeling.
    revenue_per_kpi: An optional DataArray of dimensions `(n_geos, n_times)`
      containing the average revenue amount per KPI unit. Although modeling is
      done on `kpi`, model analysis and optimization are done on `KPI *
      revenue_per_kpi` (revenue), if this value is available. If `kpi`
      corresponds to revenue, then an array of ones is passed automatically.
    media: An optional DataArray of dimensions `(n_geos, n_media_times,
      n_media_channels)` containing non-negative media execution values.
      Typically these are impressions, but it can be any metric, such as cost or
      clicks. `n_media_times` ≥ `n_times` is required, and the final `n_times`
      time periods must align with the time window of `kpi` and `controls`. Due
      to lagged effects, we recommend that the time window for media includes up
      to `max_lag` additional periods prior to this window. If `n_media_times` <
      `n_times` + `max_lag`, the model effectively imputes media history as zero
      (no media execution). If `n_media_times` > `n_times` + `max_lag`, then
      only the final `n_times` + `max_lag` periods are used to fit the model.
      `media` and `media_spend` must contain the same number of media channels
      in the same order. If either of these arguments is passed, then the other
      is not optional.
    media_spend: An optional `DataArray` containing the cost of each media
      channel. This is used as the denominator for ROI calculations. The
      DataArray shape can be `(n_geos, n_times, n_media_channels)` or
      `(n_media_channels,)` if the data is aggregated over `geo` and `time`
      dimensions. We recommend that the spend total aligns with the time window
      of the `kpi` and `controls` data, which is the time window over which
      incremental outcome of the ROI numerator is calculated. However, note that
      incremental outcome is influenced by media execution prior to this time
      window, through lagged effects, and excludes lagged effects beyond the
      time window of media executed during the time window. `media` and
      `media_spend` must contain the same number of media channels in the same
      order. If either of these arguments is passed, then the other is not
      optional.
    reach: An optional `DataArray` of dimensions `(n_geos, n_media_times,
      n_rf_channels)` containing non-negative `reach` values. It is required
      that `n_media_times` ≥ `n_times`, and the final `n_times` time periods
      must align with the time window of `kpi` and `controls`. The time window
      must include the time window of the `kpi` and `controls` data, but it is
      optional to include lagged time periods prior to the time window of the
      `kpi` and `controls` data. If lagged reach is not included, or if the
      lagged reach includes fewer than `max_lag` time periods, then the model
      calculates Adstock assuming that reach execution is zero prior to the
      first observed time period. We recommend including `n_times` + `max_lag`
      time periods, unless the value of `max_lag` is prohibitively large. If
      only `media` data is used, then `reach` will be `None`. `reach`,
      `frequency`, and `rf_spend` must contain the same number of media channels
      in the same order. If any of these arguments is passed, then the others
      are not optional.
    frequency: An optional `DataArray` of dimensions `(n_geos, n_media_times,
      n_rf_channels)` containing non-negative `frequency` values. It is required
      that `n_media_times` ≥ `n_times`, and the final `n_times` time periods
      must align with the time window of `kpi` and `controls`. The time window
      must include the time window of the `kpi` and `controls` data, but it is
      optional to include lagged time periods prior to the time window of the
      `kpi` and `controls` data. If lagged frequency is not included, or if the
      lagged frequency includes fewer than `max_lag` time periods, then the
      model calculates Adstock assuming that frequency execution is zero prior
      to the first observed time period. We recommend including `n_times` +
      `max_lag` time periods, unless the value of `max_lag` is prohibitively
      large. If only `media` data is used, then `frequency` will be `None`.
      `reach`, `frequency`, and `rf_spend` must contain the same number of media
      channels in the same order. If any of these arguments is passed, then the
      others are not optional.
    rf_spend: An optional `DataArray` containing the cost of each reach and
      frequency channel. This is used as the denominator for ROI calculations.
      The DataArray shape can be `(n_rf_channels,)`, `(n_geos, n_times,
      n_rf_channels)`, or `(n_geos, n_rf_channels)`. The spend should be
      aggregated over geo and/or time dimensions that are not represented. We
      recommend that the spend total aligns with the time window of the `kpi`
      and `controls` data, which is the time window over which incremental
      outcome of the ROI numerator is calculated. However, note that incremental
      outcome is influenced by media execution prior to this time window,
      through lagged effects, and excludes lagged effects beyond the time window
      of media executed during the time window. If only `media` data is used,
      `rf_spend` will be `None`. `reach`, `frequency`, and `rf_spend` must
      contain the same number of media channels in the same order. If any of
      these arguments is passed, then the others are not optional.
    organic_media: An optional `DataArray` of dimensions `(n_geos,
      n_media_times, n_organic_media_channels)` containing non-negative organic
      media values. Organic media variables are media activities that have no
      direct cost. These may include impressions from newsletters, a blog post,
      social media activity or email campaigns but it can be any metric, such as
      clicks. `n_media_times` ≥ `n_times` is required, and the final `n_times`
      time periods must align with the time window of `kpi` and `controls`. Due
      to lagged effects, we recommend that the time window for organic media
      includes up to `max_lag` additional periods prior to this window. If
      `n_organic_media_times` < `n_times` + `max_lag`, the model effectively
      imputes organic media history. If `n_organic_media_times` > `n_times` +
      `max_lag`, then only the final `n_times` + `max_lag` periods are used to
      fit the model.
    organic_reach: An optional `DataArray` of dimensions `(n_geos,
      n_media_times, n_organic_rf_channels)` containing non-negative organic
      reach values. It is required that `n_media_times` ≥ `n_times`, and the
      final `n_times` time periods must align with the time window of `kpi` and
      `controls`. The time window must include the time window of the `kpi` and
      `controls` data, but it is optional to include lagged time periods prior
      to the time window of the `kpi` and `controls` data. If lagged reach is
      not included, or if the lagged reach includes fewer than `max_lag` time
      periods, then the model calculates Adstock assuming that reach execution
      is zero prior to the first observed time period. We recommend including
      `n_times` + `max_lag` time periods, unless the value of `max_lag` is
      prohibitively large. If no organic reach and frequency data is used, then
      `organic_reach` and `organic_frequency` will be `None`. `organic_reach`,
      and `organic_frequency` must contain the same number of channels in the
      same order. If any of these arguments is passed, then the other is not
      optional.
    organic_frequency: An optional `DataArray` of dimensions `(n_geos,
      n_media_times, n_organic_rf_channels)` containing non-negative organic
      frequency values. It is required that `n_media_times` ≥ `n_times`, and the
      final `n_times` time periods must align with the time window of `kpi` and
      `controls`. The time window must include the time window of the `kpi` and
      `controls` data, but it is optional to include lagged time periods prior
      to the time window of the `kpi` and `controls` data. If lagged frequency
      is not included, or if the lagged frequency includes fewer than `max_lag`
      time periods, then the model calculates Adstock assuming that frequency
      execution is zero prior to the first observed time period. We recommend
      including `n_times` + `max_lag` time periods, unless the value of
      `max_lag` is prohibitively large. If no organic reach and frequency data
      is used, then `organic_frequency` will be `None`. `organic_reach` and
      `organic_frequency` must contain the same number of channels in the same
      order. If any of these arguments is passed, then the other is not
      optional.
    non_media_treatments: An optional DataArray of dimensions `(n_geos, n_times,
      n_non_media_channels)` containing non-media treatment variables values.
      Non-media treatment variables are marketing activities taken by the
      advertiser not directly related to media. They have no direct marketing
      cost associated with them but unlike organic media variables there are no
      Adstock and Hill effects. They differ from control variables as they are
      considered to be intervenable and hence are treatment variables under the
      causal model. Some examples include running a promotion, the price of a
      product and a change in a product's packaging and/or design.
  """

  kpi: xr.DataArray
  kpi_type: str
  controls: xr.DataArray
  population: xr.DataArray
  revenue_per_kpi: xr.DataArray | None = None
  media: xr.DataArray | None = None
  media_spend: xr.DataArray | None = None
  reach: xr.DataArray | None = None
  frequency: xr.DataArray | None = None
  rf_spend: xr.DataArray | None = None
  organic_media: xr.DataArray | None = None
  organic_reach: xr.DataArray | None = None
  organic_frequency: xr.DataArray | None = None
  non_media_treatments: xr.DataArray | None = None

  def __post_init__(self):
    self._convert_geos_to_strings()
    self._validate_kpi()
    self._validate_scenarios()
    self._validate_names()
    self._validate_dimensions()
    self._validate_media_channels()
    self._validate_time_formats()
    self._validate_times()

  def _convert_geos_to_strings(self):
    """Converts geo coordinates to strings in all relevant DataArrays."""
    for field in dataclasses.fields(self):
      array = getattr(self, field.name)
      if isinstance(array, xr.DataArray) and constants.GEO in array.dims:
        array.coords[constants.GEO] = array.coords[constants.GEO].astype(str)

  @property
  def geo(self) -> xr.DataArray:
    """Returns the geo dimension."""
    return self.kpi[constants.GEO]

  @property
  def time(self) -> xr.DataArray:
    """Returns the time dimension coordinates."""
    return self.kpi[constants.TIME]

  @functools.cached_property
  def time_coordinates(self) -> tc.TimeCoordinates:
    """Returns the (KPI) time dimension in a `TimeCoordinates` wrapper."""
    return tc.TimeCoordinates.from_dates(self.time)

  @property
  def media_time(self) -> xr.DataArray:
    """Returns the media time dimension coordinates."""
    if self.media is not None:
      return self.media[constants.MEDIA_TIME]
    else:
      return self.reach[constants.MEDIA_TIME]

  @functools.cached_property
  def media_time_coordinates(self) -> tc.TimeCoordinates:
    """Returns the media time dimension in a `TimeCoordinates` wrapper."""
    return tc.TimeCoordinates.from_dates(self.media_time)

  @property
  def media_channel(self) -> xr.DataArray | None:
    """Returns the media channel dimension."""
    if self.media is not None:
      return self.media[constants.MEDIA_CHANNEL]
    else:
      return None

  @property
  def rf_channel(self) -> xr.DataArray | None:
    """Returns the RF channel dimension."""
    if self.reach is not None:
      return self.reach[constants.RF_CHANNEL]
    else:
      return None

  @property
  def organic_media_channel(self) -> xr.DataArray | None:
    """Returns the organic media channel dimension."""
    if self.organic_media is not None:
      return self.organic_media[constants.ORGANIC_MEDIA_CHANNEL]
    else:
      return None

  @property
  def organic_rf_channel(self) -> xr.DataArray | None:
    """Returns the organic RF channel dimension."""
    if self.organic_reach is not None:
      return self.organic_reach[constants.ORGANIC_RF_CHANNEL]
    else:
      return None

  @property
  def non_media_channel(self) -> xr.DataArray | None:
    """Returns the non-media treatments channel dimension."""
    if self.non_media_treatments is not None:
      return self.non_media_treatments[constants.NON_MEDIA_CHANNEL]
    else:
      return None

  @property
  def control_variable(self) -> xr.DataArray:
    """Returns the control variable dimension."""
    return self.controls[constants.CONTROL_VARIABLE]

  @property
  def media_spend_has_geo_dimension(self) -> bool:
    """Checks whether the `media_spend` array has a geo dimension."""
    return (
        self.media_spend is not None
        and constants.GEO in self.media_spend.coords
    )

  @property
  def media_spend_has_time_dimension(self) -> bool:
    """Checks whether the `media_spend` array has a time dimension."""
    return (
        self.media_spend is not None
        and constants.TIME in self.media_spend.coords
    )

  @property
  def rf_spend_has_geo_dimension(self) -> bool:
    """Checks whether the `rf_spend` array has a geo dimension."""
    return self.rf_spend is not None and constants.GEO in self.rf_spend.coords

  @property
  def rf_spend_has_time_dimension(self) -> bool:
    """Checks whether the `rf_spend` array has a time dimension."""
    return self.rf_spend is not None and constants.TIME in self.rf_spend.coords

  def _validate_scenarios(self):
    """Verifies that calibration and analysis is set correctly."""
    n_geos = len(self.kpi.coords[constants.GEO])
    n_times = len(self.kpi.coords[constants.TIME])
    if self.kpi_type == constants.REVENUE:
      ones = np.ones((n_geos, n_times))
      revenue_per_kpi = xr.DataArray(
          ones,
          dims=[constants.GEO, constants.TIME],
          coords={
              constants.GEO: self.geo,
              constants.TIME: self.time,
          },
          name=constants.REVENUE_PER_KPI,
      )
      if not revenue_per_kpi.equals(
          self.revenue_per_kpi
      ):  # Not equal to all ones.
        warnings.warn(
            "Revenue from the `kpi` data is used when `kpi_type`=`revenue`."
            " `revenue_per_kpi` is ignored.",
            UserWarning,
        )
      self.revenue_per_kpi = revenue_per_kpi
    else:
      if self.revenue_per_kpi is None:
        warnings.warn(
            "Consider setting custom priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with"
            f" `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}`."
            " Further documentation available at"
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior",
            UserWarning,
        )

  def _validate_kpi(self):
    if (
        self.kpi_type != constants.REVENUE
        and self.kpi_type != constants.NON_REVENUE
    ):
      raise ValueError(
          f"Invalid kpi_type: `{self.kpi_type}`; must be one of"
          f" `{constants.REVENUE}` or `{constants.NON_REVENUE}`."
      )

    if (self.kpi.values < 0).any():
      raise ValueError("KPI values must be non-negative.")

  def _validate_names(self):
    """Verifies that the names of the data arrays are correct."""
    arrays = [
        self.kpi,
        self.controls,
        self.population,
        self.revenue_per_kpi,
        self.organic_media,
        self.organic_reach,
        self.organic_frequency,
        self.non_media_treatments,
        self.media,
        self.media_spend,
        self.reach,
        self.frequency,
        self.rf_spend,
    ]

    for array, name in zip(arrays, constants.POSSIBLE_INPUT_DATA_ARRAY_NAMES):
      if array is not None and array.name != name:
        raise ValueError(f"Array '{array.name}' should have name '{name}'")

  def _validate_dimensions(self):
    """Verifies that the data array dimmensions are correct."""
    _check_dim_collection(self.kpi, [[constants.GEO, constants.TIME]])
    _check_dim_collection(
        self.revenue_per_kpi, [[constants.GEO, constants.TIME]]
    )
    _check_dim_collection(
        self.media,
        [[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL]],
    )
    _check_dim_collection(
        self.controls,
        [[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE]],
    )
    _check_dim_collection(
        self.media_spend,
        [
            [constants.MEDIA_CHANNEL],
            [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
        ],
    )
    _check_dim_collection(self.population, [[constants.GEO]])
    _check_dim_collection(
        self.reach,
        [[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]],
    )
    _check_dim_collection(
        self.frequency,
        [[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]],
    )
    _check_dim_collection(
        self.rf_spend,
        [
            [constants.RF_CHANNEL],
            [constants.GEO, constants.TIME, constants.RF_CHANNEL],
            [constants.GEO, constants.RF_CHANNEL],
        ],
    )
    _check_dim_collection(
        self.organic_media,
        [[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.ORGANIC_MEDIA_CHANNEL,
        ]],
    )
    _check_dim_collection(
        self.organic_reach,
        [[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL]],
    )
    _check_dim_collection(
        self.organic_frequency,
        [[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL]],
    )
    _check_dim_collection(
        self.non_media_treatments,
        [[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL]],
    )

    _check_dim_match(
        constants.GEO,
        [
            self.kpi,
            self.revenue_per_kpi,
            self.media,
            self.controls,
            self.population,
            self.reach,
            self.frequency,
            self.organic_media,
            self.organic_reach,
            self.organic_frequency,
            self.non_media_treatments,
        ],
    )
    _check_dim_match(
        constants.TIME,
        [
            self.kpi,
            self.revenue_per_kpi,
            self.controls,
            self.non_media_treatments,
        ],
    )
    _check_dim_match(constants.MEDIA_CHANNEL, [self.media, self.media_spend])
    _check_dim_match(
        constants.RF_CHANNEL, [self.reach, self.frequency, self.rf_spend]
    )
    _check_dim_match(constants.ORGANIC_MEDIA_CHANNEL, [self.organic_media])
    _check_dim_match(
        constants.ORGANIC_RF_CHANNEL,
        [self.organic_reach, self.organic_frequency],
    )
    _check_dim_match(constants.NON_MEDIA_CHANNEL, [self.non_media_treatments])
    _check_dim_match(constants.CONTROL_VARIABLE, [self.controls])

  def _validate_media_channels(self):
    """Verifies Meridian media channel names invariants.

    In the input data, media channel names across `media_channel` and
    `rf_channel` must be unique.
    """
    all_channels = self.get_all_channels()
    if len(np.unique(all_channels)) != all_channels.size:
      raise ValueError(
          "Media channel names across `media_channel` and `rf_channel` must be"
          " unique."
      )

  def _validate_times(self):
    """Validates time coordinate values."""
    self._validate_time(self.media)
    self._validate_time(self.reach)
    self._validate_time(self.frequency)
    self._validate_time(self.organic_media)
    self._validate_time(self.organic_reach)
    self._validate_time(self.organic_frequency)

    # Time coordinates must be evenly spaced.
    try:
      _ = self.time_coordinates.interval_days
    except ValueError as exc:
      raise ValueError("Time coordinates must be regularly spaced.") from exc
    try:
      _ = self.media_time_coordinates.interval_days
    except ValueError as exc:
      raise ValueError(
          "Media time coordinates must be regularly spaced."
      ) from exc

  def _validate_time(self, array: xr.DataArray | None):
    """Validates the `time` dimension of the given `DataArray`.

    The `time` dimension of the selected array cannot be smaller than the
    `time` dimension of the `kpi` array."

    Args:
      array: The `DataArray` containing time coordinates to validate.
    """
    if array is None:
      return

    if len(array[constants.MEDIA_TIME]) < len(self.kpi.coords[constants.TIME]):
      raise ValueError(
          f"The '{constants.MEDIA_TIME}' dimension of the '{array.name}' array"
          f" ({len(array[constants.MEDIA_TIME])}) cannot be smaller"
          " than the 'time' dimension of the 'kpi' array"
          f" ({len(self.kpi.coords[constants.TIME])})"
      )

  def _validate_time_formats(self):
    """Validates the time coordinate format for all variables."""
    self._validate_time_coord_format(self.kpi)
    self._validate_time_coord_format(self.revenue_per_kpi)
    self._validate_time_coord_format(self.controls)
    self._validate_time_coord_format(self.media)
    self._validate_time_coord_format(self.media_spend)
    self._validate_time_coord_format(self.reach)
    self._validate_time_coord_format(self.frequency)
    self._validate_time_coord_format(self.rf_spend)
    self._validate_time_coord_format(self.organic_media)
    self._validate_time_coord_format(self.organic_reach)
    self._validate_time_coord_format(self.organic_frequency)
    self._validate_time_coord_format(self.non_media_treatments)

  def _validate_time_coord_format(self, array: xr.DataArray | None):
    """Validates the `time` dimensions format of the selected DataArray.

    The `time` dimension of the selected array must have labels that are
    formatted in the Meridian conventional `"yyyy-mm-dd"` format.

    Args:
      array: An optional DataArray to validate.
    """
    if array is None:
      return

    time_values = array.coords.get(constants.TIME, None)
    if time_values is not None:
      for time in time_values:
        try:
          _ = dt.datetime.strptime(time.item(), constants.DATE_FORMAT)
        except (TypeError, ValueError) as exc:
          raise ValueError(
              f"Invalid time label: {time.item()}. Expected format:"
              f" {constants.DATE_FORMAT}"
          ) from exc

    media_time_values = array.coords.get(constants.MEDIA_TIME, None)
    if media_time_values is not None:
      for time in media_time_values:
        try:
          _ = dt.datetime.strptime(time.item(), constants.DATE_FORMAT)
        except (TypeError, ValueError) as exc:
          raise ValueError(
              f"Invalid media_time label: {time.item()}. Expected format:"
              f" {constants.DATE_FORMAT}"
          ) from exc

  def as_dataset(self) -> xr.Dataset:
    """Returns data as a single `xarray.Dataset` object."""
    data = [
        self.kpi,
        self.controls,
        self.population,
    ]
    if self.revenue_per_kpi is not None:
      data.append(self.revenue_per_kpi)
    if self.media is not None:
      data.append(self.media)
      data.append(self.media_spend)
    if self.reach is not None:
      data.append(self.reach)
      data.append(self.frequency)
      data.append(self.rf_spend)
    if self.organic_media is not None:
      data.append(self.organic_media)
    if self.organic_reach is not None:
      data.append(self.organic_reach)
      data.append(self.organic_frequency)
    if self.non_media_treatments is not None:
      data.append(self.non_media_treatments)

    return xr.combine_by_coords(data)

  def get_n_top_largest_geos(self, num_geos: int) -> list[str]:
    """Finds the specified number of the largest geos by population.

    Args:
      num_geos: The number of top largest geos to return based on population.

    Returns:
      A list of the specified number of top largest geos.
    """
    geo_by_population = (
        self.population.to_dataframe()
        .reset_index()
        .sort_values(by=constants.POPULATION, ascending=False)
    )
    return geo_by_population[constants.GEO][:num_geos].tolist()

  def get_all_paid_channels(self) -> np.ndarray:
    """Returns all the paid channel dimensions, including both media and RF.

    If both media and RF channels are present, then the RF channels are
    concatenated to the end of the media channels.
    """
    # pytype: disable=attribute-error
    if self.media_channel is not None and self.rf_channel is not None:
      return np.concatenate(
          [self.media_channel.values, self.rf_channel.values],
          axis=None,
      )
    elif self.rf_channel is not None:
      return self.rf_channel.values
    elif self.media_channel is not None:
      return self.media_channel.values
    else:
      raise ValueError("Both RF and media channel values are missing.")
    # pytype: enable=attribute-error

  def get_paid_channels_argument_builder(
      self,
  ) -> arg_builder.OrderedListArgumentBuilder:
    """Returns an argument builder for all *paid* channels."""
    return arg_builder.OrderedListArgumentBuilder(self.get_all_paid_channels())

  def get_paid_media_channels_argument_builder(
      self,
  ) -> arg_builder.OrderedListArgumentBuilder:
    """Returns an argument builder for *paid* media channels *only*."""
    if self.media_channel is None:
      raise ValueError("There are no media channels in the input data.")
    return arg_builder.OrderedListArgumentBuilder(self.media_channel.values)

  def get_paid_rf_channels_argument_builder(
      self,
  ) -> arg_builder.OrderedListArgumentBuilder:
    """Returns an argument builder for *paid* RF channels *only*."""
    if self.rf_channel is None:
      raise ValueError("There are no RF channels in the input data.")
    return arg_builder.OrderedListArgumentBuilder(self.rf_channel.values)

  def get_all_channels(self) -> np.ndarray:
    """Returns all the channel dimensions.

    This method returns media, RF, organic media, organic RF and non-media
    channel names, concatenated into a single array in that order.
    """
    channels = [self.get_all_paid_channels()]
    # pytype: disable=attribute-error
    if self.organic_media_channel is not None:
      channels.append(self.organic_media_channel.values)
    if self.organic_rf_channel is not None:
      channels.append(self.organic_rf_channel.values)
    if self.non_media_channel is not None:
      channels.append(self.non_media_channel.values)
    # pytype: enable=attribute-error

    return np.concatenate(channels)

  def get_all_media_and_rf(self) -> np.ndarray:
    """Returns all of the media execution values, including both media and RF.

    If media, reach, and frequency were used for modeling, reach * frequency
    is concatenated to the end of media.

    Returns:
      `np.ndarray` with dimensions `(n_geos, n_media_times, n_channels)`
      containing media or reach * frequency for each `media_channel` or
      `rf_channel`.
    """
    if (
        self.media is not None
        and self.reach is not None
        and self.frequency is not None
    ):
      return np.concatenate(
          [self.media.values, self.reach.values * self.frequency.values],
          axis=-1,
      )
    elif self.reach is not None and self.frequency is not None:
      return self.reach.values * self.frequency.values
    elif self.media is not None:
      return self.media.values
    else:
      raise ValueError("Both RF and Media are missing.")

  def get_total_spend(self) -> np.ndarray:
    """Returns total spend, including `media_spend` and `rf_spend`."""
    if self.media_spend is not None and self.rf_spend is not None:
      return np.concatenate(
          [self.media_spend.values, self.rf_spend.values], axis=-1
      )
    elif self.rf_spend is not None:
      return self.rf_spend.values
    elif self.media_spend is not None:
      return self.media_spend.values
    else:
      raise ValueError("Both RF and Media are missing.")
