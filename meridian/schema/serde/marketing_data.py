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

"""Serialization and deserialization of `InputData` for Meridian models."""

from collections.abc import Mapping
import dataclasses
import datetime as dt
import functools
import itertools
from typing import Sequence

from meridian import constants as c
from meridian.data import input_data as meridian_input_data
from mmm.v1.common import date_interval_pb2
from mmm.v1.marketing import marketing_data_pb2 as marketing_pb
from meridian.schema.serde import constants as sc
from meridian.schema.serde import serde
from meridian.schema.utils import time_record
import numpy as np
import xarray as xr

from google.type import date_pb2

# Mapping from DataArray names to coordinate names
_COORD_NAME_MAP = {
    c.MEDIA: c.MEDIA_CHANNEL,
    c.REACH: c.RF_CHANNEL,
    c.FREQUENCY: c.RF_CHANNEL,
    c.ORGANIC_MEDIA: c.ORGANIC_MEDIA_CHANNEL,
    c.ORGANIC_REACH: c.ORGANIC_RF_CHANNEL,
    c.ORGANIC_FREQUENCY: c.ORGANIC_RF_CHANNEL,
    c.NON_MEDIA_TREATMENTS: c.NON_MEDIA_CHANNEL,
}


@dataclasses.dataclass(frozen=True)
class _DeserializedTimeDimension:
  """Wrapper class for `TimeDimension` proto to provide utility methods during deserialization."""

  _time_dimension: marketing_pb.MarketingDataMetadata.TimeDimension

  def __post_init__(self):
    if not self._time_dimension.dates:
      raise ValueError("TimeDimension proto must have at least one date.")

  @functools.cached_property
  def date_coordinates(self) -> list[dt.date]:
    """Returns a list of date coordinates in this time dimension."""
    return [dt.date(d.year, d.month, d.day) for d in self._time_dimension.dates]

  @functools.cached_property
  def time_dimension_interval(self) -> date_interval_pb2.DateInterval:
    """Returns the `[start, end)` interval that spans this time dimension.

    This date interval spans all of the date coordinates in this time dimension.
    """
    date_intervals = time_record.convert_times_to_date_intervals(
        self.date_coordinates
    )
    return _get_date_interval_from_date_intervals(list(date_intervals.values()))


@dataclasses.dataclass(frozen=True)
class _DeserializedMetadata:
  """A container for parsed metadata from the `MarketingData` proto.

  Attributes:
    _metadata: The `MarketingDataMetadata` proto.
  """

  _metadata: marketing_pb.MarketingDataMetadata

  def __post_init__(self):
    # Evaluate the properties to trigger validation
    _ = self.time_dimension
    _ = self.media_time_dimension

  def _get_time_dimension(self, name: str) -> _DeserializedTimeDimension:
    """Helper method to get a specific TimeDimension proto by name."""
    for time_dimension in self._metadata.time_dimensions:
      if time_dimension.name == name:
        return _DeserializedTimeDimension(time_dimension)
    raise ValueError(f"No TimeDimension found with name '{name}' in metadata.")

  @functools.cached_property
  def time_dimension(self) -> _DeserializedTimeDimension:
    """Returns the TimeDimension with name 'time'."""
    return self._get_time_dimension(c.TIME)

  @functools.cached_property
  def media_time_dimension(self) -> _DeserializedTimeDimension:
    """Returns the TimeDimension with name 'media_time'."""
    return self._get_time_dimension(c.MEDIA_TIME)

  @functools.cached_property
  def channel_dimensions(self) -> Mapping[str, list[str]]:
    """Returns a mapping of channel dimension names to their corresponding channel coordinate names."""
    return {
        cd.name: list(cd.channels) for cd in self._metadata.channel_dimensions
    }

  @functools.cached_property
  def channel_types(self) -> Mapping[str, str | None]:
    """Returns a mapping of individual channel names to their types."""
    channel_coord_map = {}
    for name, channels in self.channel_dimensions.items():
      for channel in channels:
        channel_coord_map[channel] = _COORD_NAME_MAP.get(
            name,
        )
    return channel_coord_map


def _extract_data_array(
    serialized_data_points: Sequence[marketing_pb.MarketingDataPoint],
    data_extractor_fn,
    data_name,
) -> xr.DataArray | None:
  """Helper function to extract data into an `xr.DataArray`.

  Args:
    serialized_data_points: A Sequence of MarketingDataPoint protos.
    data_extractor_fn: A function that takes a data point and returns either a
      tuple of `(geo_id, time_str, value)`, or `None` if the data point should
      be skipped.
    data_name: The desired name for the `xr.DataArray`.

  Returns:
    An `xr.DataArray` containing the extracted data, or `None` if no data is
    found.
  """
  data_dict = {}  # (geo_id, time_str) -> value
  geo_ids = []
  times = []

  for data_point in serialized_data_points:
    extraction_result = data_extractor_fn(data_point)
    if extraction_result is None:
      continue

    geo_id, time_str, value = extraction_result

    # TODO: Enforce dimension uniqueness in Meridian.
    if geo_id not in geo_ids:
      geo_ids.append(geo_id)
    if time_str not in times:
      times.append(time_str)

    data_dict[(geo_id, time_str)] = value

  if not data_dict:
    return None

  data_values = np.array([
      [data_dict.get((geo_id, time), np.nan) for time in times]
      for geo_id in geo_ids
  ])

  return xr.DataArray(
      data=data_values,
      coords={
          c.GEO: geo_ids,
          c.TIME: times,
      },
      dims=(c.GEO, c.TIME),
      name=data_name,
  )


def _extract_3d_data_array(
    serialized_data_points: Sequence[marketing_pb.MarketingDataPoint],
    data_extractor_fn,
    data_name,
    third_dim_name,
    time_dim_name=c.TIME,
) -> xr.DataArray | None:
  """Helper function to extract data with 3 dimensions into an `xr.DataArray`.

  The first dimension is always `GEO`, and the second is the time dimension
  (default: `TIME`).

  Args:
      serialized_data_points: A sequence of MarketingDataPoint protos.
      data_extractor_fn: A function that takes a data point and returns either a
        tuple of `(geo_id, time_str, third_dim_key, value)`, or `None` if the
        data point should be skipped.
      data_name: The desired name for the `xr.DataArray`.
      third_dim_name: The name of the third dimension.
      time_dim_name: The name of the time dimension. Default is `TIME`.

  Returns:
      An `xr.DataArray` containing the extracted data, or `None` if no data is
      found.
  """
  data_dict = {}  # (geo_id, time_str, third_dim_key) -> value
  geo_ids = []
  times = []
  third_dim_keys = []

  for data_point in serialized_data_points:
    for extraction_result in data_extractor_fn(data_point):
      geo_id, time_str, third_dim_key, value = extraction_result

      if geo_id not in geo_ids:
        geo_ids.append(geo_id)
      if time_str not in times:
        times.append(time_str)
      if third_dim_key not in third_dim_keys:
        third_dim_keys.append(third_dim_key)

      # TODO: Enforce dimension uniqueness in Meridian.
      data_dict[(geo_id, time_str, third_dim_key)] = value

  if not data_dict:
    return None

  data_values = np.array([
      [
          [
              data_dict.get((geo_id, time, third_dim_key), np.nan)
              for third_dim_key in third_dim_keys
          ]
          for time in times
      ]
      for geo_id in geo_ids
  ])

  return xr.DataArray(
      data=data_values,
      coords={
          c.GEO: geo_ids,
          time_dim_name: times,
          third_dim_name: third_dim_keys,
      },
      dims=(c.GEO, time_dim_name, third_dim_name),
      name=data_name,
  )


def _get_date_interval_from_date_intervals(
    date_intervals: Sequence[date_interval_pb2.DateInterval],
) -> date_interval_pb2.DateInterval:
  """Gets the date interval based on the earliest start date and latest end date.

  Args:
      date_intervals: A list of DateInterval protos.

  Returns:
      A DateInterval representing the earliest start date and latest end date.
  """
  get_start_date = lambda interval: dt.date(
      interval.start_date.year,
      interval.start_date.month,
      interval.start_date.day,
  )
  get_end_date = lambda interval: dt.date(
      interval.end_date.year, interval.end_date.month, interval.end_date.day
  )

  min_start_date_interval = min(date_intervals, key=get_start_date)
  max_end_date_interval = max(date_intervals, key=get_end_date)

  return date_interval_pb2.DateInterval(
      start_date=date_pb2.Date(
          year=min_start_date_interval.start_date.year,
          month=min_start_date_interval.start_date.month,
          day=min_start_date_interval.start_date.day,
      ),
      end_date=date_pb2.Date(
          year=max_end_date_interval.end_date.year,
          month=max_end_date_interval.end_date.month,
          day=max_end_date_interval.end_date.day,
      ),
  )


class _InputDataSerializer:
  """Serializes an `InputData` container in Meridian model."""

  def __init__(self, input_data: meridian_input_data.InputData):
    self._input_data = input_data

  @property
  def _n_geos(self) -> int:
    return len(self._input_data.geo)

  @property
  def _n_times(self) -> int:
    return len(self._input_data.time)

  def __call__(self) -> marketing_pb.MarketingData:
    """Serializes the input data into a MarketingData proto."""
    marketing_proto = marketing_pb.MarketingData()
    # Use media_time since it covers larger range.
    times_to_date_intervals = time_record.convert_times_to_date_intervals(
        self._input_data.media_time.data
    )
    geos_and_times = itertools.product(
        self._input_data.geo.data, self._input_data.media_time.data
    )

    for geo, time in geos_and_times:
      data_point = self._serialize_data_point(
          geo,
          time,
          times_to_date_intervals,
      )
      marketing_proto.marketing_data_points.append(data_point)

    if self._input_data.media_spend is not None:
      if (
          not self._input_data.media_spend_has_geo_dimension
          and not self._input_data.media_spend_has_time_dimension
      ):
        marketing_proto.marketing_data_points.append(
            self._serialize_aggregated_media_spend_data_point(
                self._input_data.media_spend,
                times_to_date_intervals,
            )
        )
      elif (
          self._input_data.media_spend_has_geo_dimension
          != self._input_data.media_spend_has_time_dimension
      ):
        raise AssertionError(
            "Invalid input data: media_spend must either be fully granular"
            " (both geo and time dimensions) or fully aggregated (neither geo"
            " nor time dimensions)."
        )

    if self._input_data.rf_spend is not None:
      if (
          not self._input_data.rf_spend_has_geo_dimension
          and not self._input_data.rf_spend_has_time_dimension
      ):
        marketing_proto.marketing_data_points.append(
            self._serialize_aggregated_rf_spend_data_point(
                self._input_data.rf_spend, times_to_date_intervals
            )
        )
      elif (
          self._input_data.rf_spend_has_geo_dimension
          != self._input_data.rf_spend_has_time_dimension
      ):
        raise AssertionError(
            "Invalid input data: rf_spend must either be fully granular (both"
            " geo and time dimensions) or fully aggregated (neither geo nor"
            " time dimensions)."
        )

    marketing_proto.metadata.CopyFrom(self._serialize_metadata())

    return marketing_proto

  def _serialize_media_variables(
      self,
      geo: str,
      time: str,
      channel_dim_name: str,
      impressions_data_array: xr.DataArray,
      spend_data_array: xr.DataArray | None = None,
  ) -> list[marketing_pb.MediaVariable]:
    """Serializes media variables for a given geo and time.

    Args:
      geo: The geo ID.
      time: The time string.
      channel_dim_name: The name of the channel dimension.
      impressions_data_array: The DataArray containing impressions data.
        Expected dimensions: `(n_geos, n_media_times, n_channels)`.
      spend_data_array: The optional DataArray containing spend data. Expected
        dimensions are `(n_geos, n_times, n_media_channels)`.

    Returns:
      A list of MediaVariable protos.
    """
    media_variables = []
    for media_data in impressions_data_array.sel(geo=geo, media_time=time):
      channel = media_data[channel_dim_name].item()
      media_variable = marketing_pb.MediaVariable(
          channel_name=channel,
          scalar_metric=marketing_pb.ScalarMetric(
              name=c.IMPRESSIONS, value=media_data.item()
          ),
      )
      if spend_data_array is not None and time in spend_data_array.time:
        media_variable.media_spend = spend_data_array.sel(
            geo=geo, time=time, **{channel_dim_name: channel}
        ).item()
      media_variables.append(media_variable)
    return media_variables

  def _serialize_reach_frequency_variables(
      self,
      geo: str,
      time: str,
      channel_dim_name: str,
      reach_data_array: xr.DataArray,
      frequency_data_array: xr.DataArray,
      spend_data_array: xr.DataArray | None = None,
  ) -> list[marketing_pb.ReachFrequencyVariable]:
    """Serializes reach and frequency variables for a given geo and time.

    Iterates through the R&F channels separately, creating a MediaVariable
    for each. It's safe to assume that Meridian media channel names are
    unique across `media_data` and `reach_data`. This assumption is
    checked when an `InputData` is created in model training.

    Dimensions of `reach_data_array` and `frequency_data_array` are expected
    to be `(n_geos, n_media_times, n_rf_channels)`.

    Args:
      geo: The geo ID.
      time: The time string.
      channel_dim_name: The name of the channel dimension (e.g., 'rf_channel').
      reach_data_array: The DataArray containing reach data.
      frequency_data_array: The DataArray containing frequency data.
      spend_data_array: The optional DataArray containing spend data.

    Returns:
      A list of MediaVariable protos.
    """
    rf_variables = []
    for reach_data in reach_data_array.sel(geo=geo, media_time=time):
      reach_value = reach_data.item()
      channel = reach_data[channel_dim_name].item()
      frequency_value = frequency_data_array.sel(
          geo=geo,
          media_time=time,
          **{channel_dim_name: channel},
      ).item()
      rf_variable = marketing_pb.ReachFrequencyVariable(
          channel_name=channel,
          reach=int(reach_value),
          average_frequency=frequency_value,
      )
      if spend_data_array is not None and time in spend_data_array.time:
        rf_variable.spend = spend_data_array.sel(
            geo=geo, time=time, **{channel_dim_name: channel}
        ).item()
      rf_variables.append(rf_variable)
    return rf_variables

  def _serialize_non_media_treatment_variables(
      self, geo: str, time: str
  ) -> list[marketing_pb.NonMediaTreatmentVariable]:
    """Serializes non-media treatment variables for a given geo and time.

    Args:
      geo: The geo ID.
      time: The time string.

    Returns:
      A list of NonMediaTreatmentVariable protos.
    """
    non_media_treatment_variables = []
    if (
        self._input_data.non_media_treatments is not None
        and geo in self._input_data.non_media_treatments.geo
        and time in self._input_data.non_media_treatments.time
    ):
      for non_media_treatment_data in self._input_data.non_media_treatments.sel(
          geo=geo, time=time
      ):
        non_media_treatment_variables.append(
            marketing_pb.NonMediaTreatmentVariable(
                name=non_media_treatment_data[c.NON_MEDIA_CHANNEL].item(),
                value=non_media_treatment_data.item(),
            )
        )
    return non_media_treatment_variables

  def _serialize_data_point(
      self,
      geo: str,
      time: str,
      times_to_date_intervals: Mapping[str, date_interval_pb2.DateInterval],
  ) -> marketing_pb.MarketingDataPoint:
    """Serializes a MarketingDataPoint proto for a given geo and time."""
    data_point = marketing_pb.MarketingDataPoint(
        geo_info=marketing_pb.GeoInfo(
            geo_id=geo,
            population=round(self._input_data.population.sel(geo=geo).item()),
        ),
        date_interval=times_to_date_intervals.get(time),
    )

    if self._input_data.controls is not None:
      if time in self._input_data.controls.time:
        for control_data in self._input_data.controls.sel(geo=geo, time=time):
          data_point.control_variables.add(
              name=control_data.control_variable.item(),
              value=control_data.item(),
          )

    if self._input_data.media is not None:
      if (
          self._input_data.media_spend_has_geo_dimension
          and self._input_data.media_spend_has_time_dimension
      ):
        spend_data_array = self._input_data.media_spend
      else:
        # Aggregated spend data is serialized in a separate data point.
        spend_data_array = None
      media_variables = self._serialize_media_variables(
          geo,
          time,
          c.MEDIA_CHANNEL,
          self._input_data.media,
          spend_data_array,
      )
      data_point.media_variables.extend(media_variables)

    if (
        self._input_data.reach is not None
        and self._input_data.frequency is not None
    ):
      if (
          self._input_data.rf_spend_has_geo_dimension
          and self._input_data.rf_spend_has_time_dimension
      ):
        rf_spend_data_array = self._input_data.rf_spend
      else:
        # Aggregated spend data is serialized in a separate data point.
        rf_spend_data_array = None
      rf_variables = self._serialize_reach_frequency_variables(
          geo,
          time,
          c.RF_CHANNEL,
          self._input_data.reach,
          self._input_data.frequency,
          rf_spend_data_array,
      )
      data_point.reach_frequency_variables.extend(rf_variables)

    if self._input_data.organic_media is not None:
      organic_media_variables = self._serialize_media_variables(
          geo, time, c.ORGANIC_MEDIA_CHANNEL, self._input_data.organic_media
      )
      data_point.media_variables.extend(organic_media_variables)

    if (
        self._input_data.organic_reach is not None
        and self._input_data.organic_frequency is not None
    ):
      organic_rf_variables = self._serialize_reach_frequency_variables(
          geo,
          time,
          c.ORGANIC_RF_CHANNEL,
          self._input_data.organic_reach,
          self._input_data.organic_frequency,
      )
      data_point.reach_frequency_variables.extend(organic_rf_variables)

    non_media_treatment_variables = (
        self._serialize_non_media_treatment_variables(geo, time)
    )
    data_point.non_media_treatment_variables.extend(
        non_media_treatment_variables
    )

    if time in self._input_data.kpi.time:
      kpi_proto = self._make_kpi_proto(geo, time)
      data_point.kpi.CopyFrom(kpi_proto)

    return data_point

  def _serialize_aggregated_media_spend_data_point(
      self,
      spend_data_array: xr.DataArray,
      times_to_date_intervals: Mapping[str, date_interval_pb2.DateInterval],
  ) -> marketing_pb.MarketingDataPoint:
    """Serializes and appends a data point for aggregated spend."""
    spend_data_point = marketing_pb.MarketingDataPoint()
    date_interval = _get_date_interval_from_date_intervals(
        list(times_to_date_intervals.values())
    )
    spend_data_point.date_interval.CopyFrom(date_interval)

    for channel_name in spend_data_array.coords[c.MEDIA_CHANNEL].values:
      spend_value = spend_data_array.sel(
          **{c.MEDIA_CHANNEL: channel_name}
      ).item()
      spend_data_point.media_variables.add(
          channel_name=channel_name, media_spend=spend_value
      )

    return spend_data_point

  def _serialize_aggregated_rf_spend_data_point(
      self,
      spend_data_array: xr.DataArray,
      times_to_date_intervals: Mapping[str, date_interval_pb2.DateInterval],
  ) -> marketing_pb.MarketingDataPoint:
    """Serializes and appends a data point for aggregated spend."""
    spend_data_point = marketing_pb.MarketingDataPoint()
    date_interval = _get_date_interval_from_date_intervals(
        list(times_to_date_intervals.values())
    )
    spend_data_point.date_interval.CopyFrom(date_interval)

    for channel_name in spend_data_array.coords[c.RF_CHANNEL].values:
      spend_value = spend_data_array.sel(**{c.RF_CHANNEL: channel_name}).item()
      spend_data_point.reach_frequency_variables.add(
          channel_name=channel_name, spend=spend_value
      )

    return spend_data_point

  def _serialize_time_dimensions(
      self, name: str, time_data: xr.DataArray
  ) -> marketing_pb.MarketingDataMetadata.TimeDimension:
    """Creates a TimeDimension message."""
    time_dimensions = marketing_pb.MarketingDataMetadata.TimeDimension(
        name=name
    )
    for date in time_data.values:
      date_obj = dt.datetime.strptime(date, c.DATE_FORMAT).date()
      time_dimensions.dates.add(
          year=date_obj.year, month=date_obj.month, day=date_obj.day
      )
    return time_dimensions

  def _serialize_channel_dimensions(
      self, channel_data: xr.DataArray | None
  ) -> marketing_pb.MarketingDataMetadata.ChannelDimension | None:
    """Creates a ChannelDimension message if the corresponding attribute exists."""
    if channel_data is None:
      return None

    coord_name = _COORD_NAME_MAP.get(channel_data.name)
    if coord_name:
      return marketing_pb.MarketingDataMetadata.ChannelDimension(
          name=channel_data.name,
          channels=channel_data.coords[coord_name].values.tolist(),
      )
    else:
      # Make sure that all channel dimensions are handled.
      raise ValueError(f"Unknown channel data name: {channel_data.name}. ")

  def _serialize_metadata(self) -> marketing_pb.MarketingDataMetadata:
    """Serializes metadata from InputData to MarketingDataMetadata."""
    metadata = marketing_pb.MarketingDataMetadata()

    metadata.time_dimensions.append(
        self._serialize_time_dimensions(c.TIME, self._input_data.time)
    )
    metadata.time_dimensions.append(
        self._serialize_time_dimensions(
            c.MEDIA_TIME, self._input_data.media_time
        )
    )

    channel_data_arrays = [
        self._input_data.media,
        self._input_data.reach,
        self._input_data.frequency,
        self._input_data.organic_media,
        self._input_data.organic_reach,
        self._input_data.organic_frequency,
    ]

    for channel_data_array in channel_data_arrays:
      channel_names_message = self._serialize_channel_dimensions(
          channel_data_array
      )
      if channel_names_message:
        metadata.channel_dimensions.append(channel_names_message)

    if self._input_data.controls is not None:
      metadata.control_names.extend(
          self._input_data.controls.control_variable.values
      )

    if self._input_data.non_media_treatments is not None:
      metadata.non_media_treatment_names.extend(
          self._input_data.non_media_treatments.non_media_channel.values
      )

    metadata.kpi_type = self._input_data.kpi_type

    return metadata

  def _make_kpi_proto(self, geo: str, time: str) -> marketing_pb.Kpi:
    """Constructs a Kpi proto from the TrainedModel."""
    kpi_proto = marketing_pb.Kpi(name=self._input_data.kpi_type)
    # `kpi` and `revenue_per_kpi` dimensions: `(n_geos, n_times)`.
    if self._input_data.kpi_type == c.REVENUE:
      kpi_proto.revenue.CopyFrom(
          marketing_pb.Kpi.Revenue(
              value=self._input_data.kpi.sel(geo=geo, time=time).item()
          )
      )
    else:
      kpi_proto.non_revenue.CopyFrom(
          marketing_pb.Kpi.NonRevenue(
              value=self._input_data.kpi.sel(geo=geo, time=time).item()
          )
      )
      if self._input_data.revenue_per_kpi is not None:
        kpi_proto.non_revenue.revenue_per_kpi = (
            self._input_data.revenue_per_kpi.sel(geo=geo, time=time).item()
        )
    return kpi_proto


class _InputDataDeserializer:
  """Deserializes a `MarketingData` proto into a Meridian `InputData`."""

  def __init__(self, serialized: marketing_pb.MarketingData):
    self._serialized = serialized

  def __post_init__(self):
    if not self._serialized.HasField(sc.METADATA):
      raise ValueError(
          f"MarketingData proto is missing the '{sc.METADATA}' field."
      )

  @functools.cached_property
  def _metadata(self) -> _DeserializedMetadata:
    """Parses metadata and extracts time dimensions, channel dimensions, and channel type map."""
    return _DeserializedMetadata(self._serialized.metadata)

  def _extract_population(self) -> xr.DataArray:
    """Extracts population data from the serialized proto."""
    geo_populations = {}

    for data_point in self._serialized.marketing_data_points:
      geo_id = data_point.geo_info.geo_id
      if not geo_id:
        continue

      geo_populations[geo_id] = data_point.geo_info.population

    return xr.DataArray(
        coords={c.GEO: list(geo_populations.keys())},
        data=np.array(list(geo_populations.values())),
        name=c.POPULATION,
    )

  def _extract_kpi_type(self) -> str:
    """Extracts the kpi_type from the serialized proto."""
    kpi_type = None
    for data_point in self._serialized.marketing_data_points:
      if data_point.HasField(c.KPI):
        current_kpi_type = data_point.kpi.WhichOneof(c.TYPE)

        if kpi_type is None:
          kpi_type = current_kpi_type
        elif kpi_type != current_kpi_type:
          raise ValueError(
              "Inconsistent kpi_type found in the data. "
              f"Expected {kpi_type}, found {current_kpi_type}"
          )

    if kpi_type is None:
      raise ValueError("kpi_type not found in the data.")
    return kpi_type

  def _extract_geo_and_time(self, data_point) -> tuple[str | None, str]:
    """Extracts geo_id and time_str from a data_point."""
    geo_id = data_point.geo_info.geo_id
    start_date = data_point.date_interval.start_date
    time_str = dt.datetime(
        start_date.year, start_date.month, start_date.day
    ).strftime(c.DATE_FORMAT)
    return geo_id, time_str

  def _extract_kpi(self, kpi_type: str) -> xr.DataArray:
    """Extracts KPI data from the serialized proto."""

    def _kpi_extractor(data_point):
      if not data_point.HasField(c.KPI):
        return None

      geo_id, time_str = self._extract_geo_and_time(data_point)

      if data_point.kpi.WhichOneof(c.TYPE) != kpi_type:
        raise ValueError(
            "Inconsistent kpi_type found in the data. "
            f"Expected {kpi_type}, found"
            f" {data_point.kpi.WhichOneof(c.TYPE)}"
        )

      kpi_value = (
          data_point.kpi.revenue.value
          if kpi_type == c.REVENUE
          else data_point.kpi.non_revenue.value
      )
      return geo_id, time_str, kpi_value

    kpi = _extract_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_kpi_extractor,
        data_name=c.KPI,
    )

    if kpi is None:
      raise ValueError(f"{c.KPI} is not found in the data.")

    return kpi

  def _extract_revenue_per_kpi(self, kpi_type: str) -> xr.DataArray | None:
    """Extracts revenue per KPI data from the serialized proto."""

    if kpi_type == c.REVENUE:
      raise ValueError(
          f"{c.REVENUE_PER_KPI} is not applicable when kpi_type is {c.REVENUE}."
      )

    def _revenue_per_kpi_extractor(data_point):
      if not data_point.HasField(c.KPI):
        return None

      if not data_point.kpi.non_revenue.HasField(c.REVENUE_PER_KPI):
        return None

      geo_id, time_str = self._extract_geo_and_time(data_point)

      if data_point.kpi.WhichOneof(c.TYPE) != kpi_type:
        raise ValueError(
            "Inconsistent kpi_type found in the data. "
            f"Expected {kpi_type}, found"
            f" {data_point.kpi.WhichOneof(c.TYPE)}"
        )

      return geo_id, time_str, data_point.kpi.non_revenue.revenue_per_kpi

    return _extract_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_revenue_per_kpi_extractor,
        data_name=c.REVENUE_PER_KPI,
    )

  def _extract_controls(self) -> xr.DataArray | None:
    """Extracts control variables data from the serialized proto, if any."""

    def _controls_extractor(data_point):
      if not data_point.control_variables:
        return None

      geo_id, time_str = self._extract_geo_and_time(data_point)

      for control_variable in data_point.control_variables:
        control_name = control_variable.name
        control_value = control_variable.value
        yield geo_id, time_str, control_name, control_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_controls_extractor,
        data_name=c.CONTROLS,
        third_dim_name=c.CONTROL_VARIABLE,
    )

  def _extract_media(self) -> xr.DataArray | None:
    """Extracts media variables data from the serialized proto."""

    def _media_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for media_variable in data_point.media_variables:
        channel_name = media_variable.channel_name
        if self._metadata.channel_types.get(channel_name) != c.MEDIA_CHANNEL:
          continue

        media_value = media_variable.scalar_metric.value
        yield geo_id, time_str, channel_name, media_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_media_extractor,
        data_name=c.MEDIA,
        third_dim_name=c.MEDIA_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_reach(self) -> xr.DataArray | None:
    """Extracts reach data from the serialized proto."""

    def _reach_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for rf_variable in data_point.reach_frequency_variables:
        channel_name = rf_variable.channel_name
        if self._metadata.channel_types.get(channel_name) != c.RF_CHANNEL:
          continue

        reach_value = rf_variable.reach
        yield geo_id, time_str, channel_name, reach_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_reach_extractor,
        data_name=c.REACH,
        third_dim_name=c.RF_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_frequency(self) -> xr.DataArray | None:
    """Extracts frequency data from the serialized proto."""

    def _frequency_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for rf_variable in data_point.reach_frequency_variables:
        channel_name = rf_variable.channel_name
        if self._metadata.channel_types.get(channel_name) != c.RF_CHANNEL:
          continue

        frequency_value = rf_variable.average_frequency
        yield geo_id, time_str, channel_name, frequency_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_frequency_extractor,
        data_name=c.FREQUENCY,
        third_dim_name=c.RF_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_organic_media(self) -> xr.DataArray | None:
    """Extracts organic media variables data from the serialized proto."""

    def _organic_media_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for media_variable in data_point.media_variables:
        channel_name = media_variable.channel_name
        if (
            self._metadata.channel_types.get(channel_name)
            != c.ORGANIC_MEDIA_CHANNEL
        ):
          continue

        media_value = media_variable.scalar_metric.value
        yield geo_id, time_str, channel_name, media_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_organic_media_extractor,
        data_name=c.ORGANIC_MEDIA,
        third_dim_name=c.ORGANIC_MEDIA_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_organic_reach(self) -> xr.DataArray | None:
    """Extracts organic reach data from the serialized proto."""

    def _organic_reach_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for rf_variable in data_point.reach_frequency_variables:
        channel_name = rf_variable.channel_name
        if (
            self._metadata.channel_types.get(channel_name)
            != c.ORGANIC_RF_CHANNEL
        ):
          continue

        reach_value = rf_variable.reach
        yield geo_id, time_str, channel_name, reach_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_organic_reach_extractor,
        data_name=c.ORGANIC_REACH,
        third_dim_name=c.ORGANIC_RF_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_organic_frequency(self) -> xr.DataArray | None:
    """Extracts organic frequency data from the serialized proto."""

    def _organic_frequency_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)

      if not geo_id:
        return None

      for rf_variable in data_point.reach_frequency_variables:
        channel_name = rf_variable.channel_name
        if (
            self._metadata.channel_types.get(channel_name)
            != c.ORGANIC_RF_CHANNEL
        ):
          continue

        frequency_value = rf_variable.average_frequency
        yield geo_id, time_str, channel_name, frequency_value

    return _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_organic_frequency_extractor,
        data_name=c.ORGANIC_FREQUENCY,
        third_dim_name=c.ORGANIC_RF_CHANNEL,
        time_dim_name=c.MEDIA_TIME,
    )

  def _extract_granular_media_spend(
      self,
      data_points_with_spend: list[marketing_pb.MarketingDataPoint],
  ) -> xr.DataArray | None:
    """Extracts granular spend data.

    Args:
      data_points_with_spend: List of MarketingDataPoint protos with spend data.

    Returns:
      An xr.DataArray with granular spend data or None if no data found.
    """

    def _granular_spend_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)
      for media_variable in data_point.media_variables:
        if (
            media_variable.HasField(c.MEDIA_SPEND)
            and self._metadata.channel_types.get(media_variable.channel_name)
            == c.MEDIA_CHANNEL
        ):
          yield geo_id, time_str, media_variable.channel_name, media_variable.media_spend

    return _extract_3d_data_array(
        serialized_data_points=data_points_with_spend,
        data_extractor_fn=_granular_spend_extractor,
        data_name=c.MEDIA_SPEND,
        third_dim_name=c.MEDIA_CHANNEL,
        time_dim_name=c.TIME,
    )

  def _extract_granular_rf_spend(
      self,
      data_points_with_spend: list[marketing_pb.MarketingDataPoint],
  ) -> xr.DataArray | None:
    """Extracts granular spend data.

    Args:
      data_points_with_spend: List of MarketingDataPoint protos with spend data.

    Returns:
      An xr.DataArray with granular spend data or None if no data found.
    """

    def _granular_spend_extractor(data_point):
      geo_id, time_str = self._extract_geo_and_time(data_point)
      for rf_variable in data_point.reach_frequency_variables:
        if (
            rf_variable.HasField(c.SPEND)
            and self._metadata.channel_types.get(rf_variable.channel_name)
            == c.RF_CHANNEL
        ):
          yield geo_id, time_str, rf_variable.channel_name, rf_variable.spend

    return _extract_3d_data_array(
        serialized_data_points=data_points_with_spend,
        data_extractor_fn=_granular_spend_extractor,
        data_name=c.RF_SPEND,
        third_dim_name=c.RF_CHANNEL,
        time_dim_name=c.TIME,
    )

  def _extract_aggregated_media_spend(
      self,
      data_points_with_spend: list[marketing_pb.MarketingDataPoint],
  ) -> xr.DataArray | None:
    """Extracts aggregated spend data.

    Args:
      data_points_with_spend: List of MarketingDataPoint protos with spend data.

    Returns:
      An xr.DataArray with aggregated spend data or None if no data found.
    """
    channel_names = self._metadata.channel_dimensions.get(c.MEDIA, [])
    channel_spend_map = {}

    for spend_data_point in data_points_with_spend:
      for media_variable in spend_data_point.media_variables:
        if (
            media_variable.channel_name in channel_names
            and media_variable.HasField(c.MEDIA_SPEND)
        ):
          channel_spend_map[media_variable.channel_name] = (
              media_variable.media_spend
          )

    if not channel_spend_map:
      return None

    return xr.DataArray(
        data=list(channel_spend_map.values()),
        coords={c.MEDIA_CHANNEL: list(channel_spend_map.keys())},
        dims=[c.MEDIA_CHANNEL],
        name=c.MEDIA_SPEND,
    )

  def _extract_aggregated_rf_spend(
      self,
      data_points_with_spend: list[marketing_pb.MarketingDataPoint],
  ) -> xr.DataArray | None:
    """Extracts aggregated spend data.

    Args:
      data_points_with_spend: List of MarketingDataPoint protos with spend data.

    Returns:
      An xr.DataArray with aggregated spend data or None if no data found.
    """
    channel_names = self._metadata.channel_dimensions.get(c.REACH, [])
    channel_spend_map = {}

    for spend_data_point in data_points_with_spend:
      for rf_variable in spend_data_point.reach_frequency_variables:
        if rf_variable.channel_name in channel_names and rf_variable.HasField(
            c.SPEND
        ):
          channel_spend_map[rf_variable.channel_name] = rf_variable.spend

    if not channel_spend_map:
      return None

    return xr.DataArray(
        data=list(channel_spend_map.values()),
        coords={c.RF_CHANNEL: list(channel_spend_map.keys())},
        dims=[c.RF_CHANNEL],
        name=c.RF_SPEND,
    )

  def _is_aggregated_spend_data_point(
      self, dp: marketing_pb.MarketingDataPoint
  ) -> bool:
    """Checks if a MarketingDataPoint with spend represents aggregated spend data.

    Args:
      dp: A marketing_pb.MarketingDataPoint representing a spend data point.

    Returns:
      True if the data point represents aggregated spend, False otherwise.
    """
    if not dp.HasField(sc.GEO_INFO) and self._metadata.media_time_dimension:
      media_time_interval = (
          self._metadata.media_time_dimension.time_dimension_interval
      )
      return (
          media_time_interval.start_date == dp.date_interval.start_date
          and media_time_interval.end_date == dp.date_interval.end_date
      )
    return False

  def _extract_media_spend(self) -> xr.DataArray | None:
    """Extracts media spend data from the serialized proto.

    Returns:
      An xr.DataArray with spend data or None if no data found.
    """
    # Filter data points relevant to spend based on channel type map
    media_channels = {
        channel
        for channel, metadata_channel_type in self._metadata.channel_types.items()
        if metadata_channel_type == c.MEDIA_CHANNEL
    }
    spend_data_points = [
        dp
        for dp in self._serialized.marketing_data_points
        if any(
            mv.HasField(c.MEDIA_SPEND) and mv.channel_name in media_channels
            for mv in dp.media_variables
        )
    ]

    if not spend_data_points:
      return None

    aggregated_spend_data_points = [
        dp
        for dp in spend_data_points
        if self._is_aggregated_spend_data_point(dp)
    ]

    if aggregated_spend_data_points:
      return self._extract_aggregated_media_spend(aggregated_spend_data_points)

    return self._extract_granular_media_spend(spend_data_points)

  def _extract_rf_spend(self) -> xr.DataArray | None:
    """Extracts reach and frequency spend data from the serialized proto.

    Returns:
      An xr.DataArray with spend data or None if no data found.
    """
    # Filter data points relevant to spend based on channel type map
    rf_channels = {
        channel
        for channel, metadata_channel_type in self._metadata.channel_types.items()
        if metadata_channel_type == c.RF_CHANNEL
    }
    spend_data_points = [
        dp
        for dp in self._serialized.marketing_data_points
        if any(
            mv.HasField(c.SPEND) and mv.channel_name in rf_channels
            for mv in dp.reach_frequency_variables
        )
    ]

    if not spend_data_points:
      return None

    aggregated_spend_data_points = [
        dp
        for dp in spend_data_points
        if self._is_aggregated_spend_data_point(dp)
    ]

    if aggregated_spend_data_points:
      return self._extract_aggregated_rf_spend(aggregated_spend_data_points)

    return self._extract_granular_rf_spend(spend_data_points)

  def _extract_non_media_treatments(self) -> xr.DataArray | None:
    """Extracts non-media treatment variables data from the serialized proto."""

    def _non_media_treatments_extractor(data_point):
      if not data_point.non_media_treatment_variables:
        return None

      geo_id, time_str = self._extract_geo_and_time(data_point)

      for (
          non_media_treatment_variable
      ) in data_point.non_media_treatment_variables:
        treatment_name = non_media_treatment_variable.name
        treatment_value = non_media_treatment_variable.value
        yield geo_id, time_str, treatment_name, treatment_value

    non_media_treatments_data_array = _extract_3d_data_array(
        serialized_data_points=self._serialized.marketing_data_points,
        data_extractor_fn=_non_media_treatments_extractor,
        data_name=c.NON_MEDIA_TREATMENTS,
        third_dim_name=c.NON_MEDIA_CHANNEL,
    )

    return non_media_treatments_data_array

  def __call__(self) -> meridian_input_data.InputData:
    """Converts the `MarketingData` proto to a Meridian `InputData`."""
    kpi_type = self._extract_kpi_type()
    return meridian_input_data.InputData(
        kpi=self._extract_kpi(kpi_type),
        kpi_type=kpi_type,
        controls=self._extract_controls(),
        population=self._extract_population(),
        revenue_per_kpi=(
            self._extract_revenue_per_kpi(kpi_type)
            if kpi_type == c.NON_REVENUE
            else None
        ),
        media=self._extract_media(),
        media_spend=self._extract_media_spend(),
        reach=self._extract_reach(),
        frequency=self._extract_frequency(),
        rf_spend=self._extract_rf_spend(),
        organic_media=self._extract_organic_media(),
        organic_reach=self._extract_organic_reach(),
        organic_frequency=self._extract_organic_frequency(),
        non_media_treatments=self._extract_non_media_treatments(),
    )


class MarketingDataSerde(
    serde.Serde[marketing_pb.MarketingData, meridian_input_data.InputData]
):
  """Serializes and deserializes an `InputData` container in Meridian."""

  def serialize(
      self, obj: meridian_input_data.InputData
  ) -> marketing_pb.MarketingData:
    """Serializes the given Meridian input data into a `MarketingData` proto."""
    return _InputDataSerializer(obj)()

  def deserialize(
      self, serialized: marketing_pb.MarketingData, serialized_version: str = ""
  ) -> meridian_input_data.InputData:
    """Deserializes the given `MarketingData` proto.

    Args:
      serialized: The serialized `MarketingData` proto.
      serialized_version: The version of the serialized model. This is used to
        handle changes in deserialization logic across different versions.

    Returns:
      A Meridian input data container.
    """
    return _InputDataDeserializer(serialized)()
