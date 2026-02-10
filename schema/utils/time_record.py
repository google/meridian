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
from mmm.v1.common import date_interval_pb2
import pandas as pd

from google.type import date_pb2


__all__ = [
    "convert_times_to_date_intervals",
    "create_date_interval_pb",
    "dates_from_date_interval_proto",
]


def convert_times_to_date_intervals(
    times: Sequence[str] | Sequence[datetime.date] | pd.DatetimeIndex,
) -> Mapping[str, date_interval_pb2.DateInterval]:
  """Creates a date interval for each time in `times` as dict values.

  Args:
    times: Sequence of date strings in YYYY-MM-DD format.

  Returns:
    Mapping that maps each time in `times` (string form) to the corresponding
    date interval.

  Raises:
    ValueError: If `times` has fewer than 2 elements or if the interval length
    between each time is not consistent.
  """
  if len(times) < 2:
    raise ValueError("There must be at least 2 time points.")

  if isinstance(times, pd.DatetimeIndex):
    datetimes = [
        datetime.datetime(year=time.year, month=time.month, day=time.day)
        for time in times
    ]
  else:
    datetimes = [
        datetime.datetime.strptime(time, constants.DATE_FORMAT)
        if isinstance(time, str)
        else time
        for time in times
    ]

  interval_length = _compute_interval_length(
      start_date=datetimes[0],
      end_date=datetimes[1],
  )
  time_to_date_interval: MutableMapping[str, date_interval_pb2.DateInterval] = (
      {}
  )

  for i, start_date in enumerate(datetimes):
    if i == len(datetimes) - 1:
      end_date = start_date + datetime.timedelta(days=interval_length)
    else:
      end_date = datetimes[i + 1]
      current_interval_length = _compute_interval_length(
          start_date=start_date,
          end_date=end_date,
      )

      if current_interval_length != interval_length:
        raise ValueError(
            "Interval length between selected times must be consistent."
        )

    date_interval = create_date_interval_pb(start_date, end_date)
    time_to_date_interval[start_date.strftime(constants.DATE_FORMAT)] = (
        date_interval
    )

  return time_to_date_interval


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
  start_date_proto = date_pb2.Date(
      year=start_date.year,
      month=start_date.month,
      day=start_date.day,
  )
  end_date_proto = date_pb2.Date(
      year=end_date.year,
      month=end_date.month,
      day=end_date.day,
  )
  return date_interval_pb2.DateInterval(
      start_date=start_date_proto,
      end_date=end_date_proto,
      tag=tag,
  )


def dates_from_date_interval_proto(
    date_interval: date_interval_pb2.DateInterval,
) -> tuple[datetime.date, datetime.date]:
  """Returns a tuple of `[start, end)` date range from a `DateInterval` proto."""
  start_date = datetime.date(
      date_interval.start_date.year,
      date_interval.start_date.month,
      date_interval.start_date.day,
  )
  end_date = datetime.date(
      date_interval.end_date.year,
      date_interval.end_date.month,
      date_interval.end_date.day,
  )
  return start_date, end_date


def _compute_interval_length(
    start_date: datetime.datetime, end_date: datetime.datetime
) -> int:
  """Computes the number of days between `start_date` and `end_date`.

  Args:
    start_date: A datetime object representing the start date.
    end_date: A datetime object representing the end date.

  Returns:
    The number of days between the given dates.
  """
  return end_date.toordinal() - start_date.toordinal()
