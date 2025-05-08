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

"""Deals with coordinate values in the time dimensions of input data."""

from collections.abc import Sequence
import dataclasses
import datetime
import functools
from typing import TypeAlias

from meridian import constants
import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    "Date",
    "DateInterval",
    "TimeCoordinates",
    "normalize_date",
    "normalize_date_interval",
]


# A type alias for a polymorphic "date" type.
Date: TypeAlias = str | datetime.datetime | datetime.date | np.datetime64 | None

# A type alias for a polymorphic "date interval" type. In all variants it is
# always a tuple of (start_date, end_date).
DateInterval: TypeAlias = tuple[Date, Date]

# Time coordinates are string labels in an xarray data array in `InputData`, but
# they can also be parsed into numpy/pandas `DatetimeIndex` internally here.
_TimeCoordinateValues: TypeAlias = (
    pd.DatetimeIndex | xr.DataArray | Sequence[Date]
)


def _to_pandas_datetime_index(times: _TimeCoordinateValues) -> pd.DatetimeIndex:
  """Normalizes the given values into a pandas `DatetimeIndex`."""
  # Coerce into pandas `DatetimeIndex` in all possible cases.
  return pd.to_datetime(times)


def normalize_date(date: Date) -> datetime.date:
  """Normalizes the given date value into a `datetime.date`."""
  if isinstance(date, str):
    return datetime.datetime.strptime(date, constants.DATE_FORMAT).date()
  elif isinstance(date, datetime.datetime):
    return date.date()
  elif isinstance(date, datetime.date):
    return date
  elif isinstance(date, np.datetime64):
    return date.astype(datetime.date)
  else:
    raise ValueError(f"Unsupported date value type: {type(date)} for {date}")


def normalize_date_interval(
    date_interval: DateInterval,
) -> tuple[datetime.date, datetime.date]:
  """Normalizes representations of a date interval into a tuple of `date`s.

  A date interval here is a tuple of `[start_date, end_date)` where:

  * `start_date` is inclusive and `end_date` is exclusive.
  * Both are polymorphic, taking the form of either:
    * `datetime.datetime` (only the date component will be used)
    * `datetime.date` (the normalized form)
    * `np.datetime64` (only the date component will be used)
    * `str` (will be parsed as "YYYY-mm-dd" or else throws)

  In all instances, the given date interval will be normalized as a tuple of
  `datetime.date`s.

  Args:
    date_interval: a polymorphic date interval to normalize.

  Returns:
    A tuple of `date`s representing a `[start_date, end_date)` date interval.
  """
  start, end = date_interval
  start = normalize_date(start)
  end = normalize_date(end)
  return (start, end)


def _to_dates_list(times: pd.DatetimeIndex) -> list[datetime.date]:
  return [dt.date() for dt in times]


@dataclasses.dataclass(frozen=True)
class TimeCoordinates:
  """A wrapper around time coordinates in Meridian's input data.

  Meridian models store time coordinates as untyped strings. It treats them as
  labels, and they have no intrinsic meaning to the model other than the
  assumption that they represent some linearly increasing time coordinates.

  This wrapper object performs some additional validation and methods for
  extracting values out of these time coordinates which are treated as numeric
  "date" values.

  Attributes:
    datetime_index: The given time coordinates, parsed as indexable
      `DatetimeIndex`.
    all_dates: The given time coordinates, as a list of Pythonic `datetime.date`
      objects.
    all_dates_str: The given time coordinates, as a list of Meridian-formatted
      date strings. This can be used for the model internals, which treat time
      coordinates as simple labels.
  """

  datetime_index: pd.DatetimeIndex

  @classmethod
  def from_dates(
      cls,
      dates: _TimeCoordinateValues,
  ) -> "TimeCoordinates":
    """Creates a `TimeCoordinates` from a polymorphic series of dates.

    Args:
      dates: A polymorphic series of dates; it can either be a Pandas
        `DatetimeIndex` or an Xarray `DataArray` with "YYYY-mm-dd" string
        labels.

    Returns:
      A normalized `TimeCoordinates` dataclass.
    """
    return cls(datetime_index=_to_pandas_datetime_index(dates))

  def __post_init__(self):
    if len(self.datetime_index) <= 1:
      raise ValueError(
          "There must be more than one date index in the time coordinates."
      )
    if not self.datetime_index.is_monotonic_increasing:
      raise ValueError(
          "Time coordinates must be strictly monotonically increasing."
      )

  @property
  def all_dates(self) -> list[datetime.date]:
    return _to_dates_list(self.datetime_index)

  @property
  def all_dates_str(self) -> list[str]:
    return [
        time.strftime(constants.DATE_FORMAT) for time in self.datetime_index
    ]

  @functools.cached_property
  def interval_days(self) -> int:
    """Returns the *mean* interval between two neighboring dates in `all_dates`.

    Raises:
      ValueError if the date index is not "regularly spaced".
    """
    if not self._is_regular_time_index():
      raise ValueError("Time coordinates are not regularly spaced!")

    # Calculate the difference between consecutive dates, in days.
    diffs = self._interval_days
    # Return the rounded mean interval.
    return int(np.round(np.mean(diffs)))

  @property
  def _timedelta_index(self) -> pd.TimedeltaIndex:
    """Returns the timedeltas between consecutive dates in `datetime_index`."""
    return self.datetime_index.diff().dropna()

  @property
  def _interval_days(self) -> Sequence[int]:
    """Converts `_timedelta_index` to a sequence of days for easier compute."""
    return self._timedelta_index.days.to_numpy()

  def _is_regular_time_index(self) -> bool:
    """Returns True if the time index is "regularly spaced"."""
    if np.all(self._interval_days == self._interval_days[0]):
      # All intervals are regular. Base case.
      return True
    # Special cases:
    # * Monthly cadences
    if np.all(np.isin(self._interval_days, [28, 29, 30, 31])):
      return True
    # * Quarterly cadences
    if np.all(np.isin(self._interval_days, [90, 91, 92])):
      return True
    # * Yearly cadences
    if np.all(np.isin(self._interval_days, [365, 366])):
      return True

    return False

  def get_selected_dates(
      self,
      selected_interval: DateInterval | None = None,
  ) -> list[datetime.date]:
    """Creates a sequence of dates containing all points in selected interval.

    Args:
      selected_interval: Tuple of the start and end times, or a `DateInterval`
        proto. If `None`, then `all_dates` is returned.

    Returns:
      A sequence of dates representing the subset of `all_dates` between the
      given start and end dates, as Python's builtin `datetime.date` objects.

    Raises:
      ValueError: If `selected_interval` is not a subset of `all_dates`.
    """
    if selected_interval is None:
      return self.all_dates

    selected_dates = normalize_date_interval(selected_interval)
    expanded = self.expand_selected_time_dims(
        selected_dates[0], selected_dates[1]
    )
    if expanded is None:
      return self.all_dates
    return expanded

  def expand_selected_time_dims(
      self,
      start_date: Date = None,
      end_date: Date = None,
  ) -> list[datetime.date] | None:
    """Validates and returns time dimension values based on the selected times.

    If both `start_date` and `end_date` are None, returns None. If specified,
    both `start_date` and `end_date` are inclusive, and must be present in the
    time coordinates of the input data.

    Args:
      start_date: Start date of the selected time period. If `None`, implies the
        earliest time dimension value in the input data.
      end_date: End date of the selected time period. If `None`, implies the
        latest time dimension value in the input data.

    Returns:
      A list of time dimension values (as `datetime.date` objects) in the input
      data within the selected time period, or do nothing and pass through
      `None` if both arguments are `None`, or if `start_date` and `end_date`
      correspond to the entire time range in the input data.

    Raises:
      `ValueError` if `start_date` or `end_date` is not in the input data's time
      dimension coordinates.
    """
    if start_date is None and end_date is None:
      return None

    if start_date is None:
      start_date = min(self.all_dates)
    else:
      start_date = normalize_date(start_date)
      if start_date not in self.all_dates:
        raise ValueError(
            f"start_date ({start_date}) must be in the time coordinates!"
        )

    if end_date is None:
      end_date = max(self.all_dates)
    else:
      end_date = normalize_date(end_date)
      if end_date not in self.all_dates:
        raise ValueError(
            f"end_date ({end_date}) must be in the time coordinates!"
        )

    if start_date > end_date:
      raise ValueError(
          f"start_date ({start_date}) must be less than or equal to end_date"
          f" ({end_date})!"
      )

    if start_date == min(self.all_dates) and end_date == max(self.all_dates):
      return None

    return [date for date in self.all_dates if start_date <= date <= end_date]
