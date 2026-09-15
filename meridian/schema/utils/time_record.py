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

"""Helper functions for time-related operations."""

from collections.abc import Mapping, MutableMapping, Sequence
import datetime

from meridian import constants
from meridian.data import time_coordinates
from mmm.v1.common import date_interval_pb2
import pandas as pd

from google.type import date_pb2


__all__ = [
    "convert_times_to_date_intervals",
    "create_date_interval_pb",
    "dates_from_date_interval_proto",
    "from_date_proto",
    "to_date_proto",
]


def convert_times_to_date_intervals(
    times: Sequence[str] | Sequence[datetime.date] | pd.DatetimeIndex,
) -> Mapping[str, date_interval_pb2.DateInterval]:
  """Creates a date interval for each time in `times` as dict values.

  Args:
    times: Sequence of date strings in YYYY-MM-DD format, datetime.date objects,
      or a pandas DatetimeIndex.

  Returns:
    Mapping that maps each time in `times` (string form) to the corresponding
    date interval.

  Raises:
    ValueError: If `times` has fewer than 2 elements or if the time coordinates
      are not regularly spaced.
  """
  if len(times) < 2:
    raise ValueError("There must be at least 2 time points.")

  coords = time_coordinates.TimeCoordinates.from_dates(times)
  time_to_date_interval: MutableMapping[str, date_interval_pb2.DateInterval] = (
      {}
  )

  for start_date, end_date in coords.period_ends.items():
    date_interval = create_date_interval_pb(start_date, end_date)
    time_to_date_interval[start_date.strftime(constants.DATE_FORMAT)] = (
        date_interval
    )

  return time_to_date_interval


def to_date_proto(date: datetime.date) -> date_pb2.Date:
  """Converts a `datetime.date` into a `google.type.Date` proto."""
  return date_pb2.Date(year=date.year, month=date.month, day=date.day)


def from_date_proto(date_proto: date_pb2.Date) -> datetime.date:
  """Converts a `google.type.Date` proto into a `datetime.date`."""
  return datetime.date(date_proto.year, date_proto.month, date_proto.day)


def create_date_interval_pb(
    start_date: datetime.date, end_date: datetime.date, tag: str = ""
) -> date_interval_pb2.DateInterval:
  """Creates a `DateInterval` proto for the given start and end dates.

  Args:
    start_date: A datetime object representing the start date.
    end_date: A datetime object representing the end date.
    tag: An optional tag to identify the date interval.

  Returns:
    Returns a date interval proto wrapping the start/end dates.
  """
  return date_interval_pb2.DateInterval(
      start_date=to_date_proto(start_date),
      end_date=to_date_proto(end_date),
      tag=tag,
  )


def dates_from_date_interval_proto(
    date_interval: date_interval_pb2.DateInterval,
) -> tuple[datetime.date, datetime.date]:
  """Returns a tuple of `[start, end)` date range from a `DateInterval` proto."""
  return (
      from_date_proto(date_interval.start_date),
      from_date_proto(date_interval.end_date),
  )
