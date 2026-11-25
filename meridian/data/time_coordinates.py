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

"""Deals with coordinate values in the time dimensions of input data."""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import datetime
import functools
import re
from typing import Any, TypeAlias, overload

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
    "normalize_date_str",
    "normalize_times",
    "normalize_times_set",
    "validate_selected_times",
    "validate_time_coordinates",
]


_NonNoneDate: TypeAlias = (
    str | datetime.datetime | datetime.date | np.datetime64
)

# A type alias for a polymorphic "date" type.
Date: TypeAlias = _NonNoneDate | None

# A type alias for a polymorphic "date interval" type. In all variants it is
# always a tuple of (start_date, end_date).
DateInterval: TypeAlias = tuple[Date, Date]

# Time coordinates are string labels in an xarray data array in `InputData`, but
# they can also be parsed into numpy/pandas `DatetimeIndex` internally here.
_TimeCoordinateValues: TypeAlias = (
    pd.DatetimeIndex | xr.DataArray | Sequence[Date]
)

# Matches date strings in `constants.DATE_FORMAT`.
_DATE_FORMAT_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@overload
def normalize_date(date: _NonNoneDate) -> datetime.date:
  ...


@overload
def normalize_date(date: None) -> None:
  ...


@overload
def normalize_date(date: Date) -> datetime.date | None:
  ...


def normalize_date(date: Date) -> datetime.date | None:
  """Normalizes the given date value into a `datetime.date` or `None`."""
  if date is None:
    return None
  elif isinstance(date, str):
    return datetime.datetime.strptime(date, constants.DATE_FORMAT).date()
  elif isinstance(date, datetime.datetime):
    return date.date()
  elif isinstance(date, datetime.date):
    return date
  elif isinstance(date, np.datetime64):
    # Convert to day precision first: at finer precisions,
    # `astype(datetime.date)` returns a `datetime.datetime` or an int.
    return date.astype("datetime64[D]").astype(datetime.date)
  else:
    raise ValueError(f"Unsupported date value type: {type(date)} for {date}")


def normalize_date_interval(
    date_interval: DateInterval,
) -> tuple[datetime.date, datetime.date]:
  """Normalizes representations of a date interval into a tuple of `date`s.

  A date interval here is a `(start_date, end_date)` tuple. Both dates are
  polymorphic, taking the form of either:

  * `datetime.datetime` (only the date component will be used)
  * `datetime.date` (the normalized form)
  * `np.datetime64` (only the date component will be used)
  * `str` (will be parsed as "YYYY-mm-dd" or else throws)

  In all instances, the given date interval will be normalized as a tuple of
  `datetime.date`s.

  Args:
    date_interval: a polymorphic date interval to normalize.

  Returns:
    A `(start_date, end_date)` tuple of `date`s.
  """
  start, end = date_interval
  if start is None or end is None:
    raise ValueError(
        f"Date interval bounds cannot be None, got: {date_interval}"
    )
  start = normalize_date(start)
  end = normalize_date(end)
  return (start, end)


def normalize_date_str(time_val: Date | xr.DataArray | Any) -> str:
  """Extracts the 'YYYY-MM-DD' prefix string from a date coordinate or string.

  Args:
    time_val: A polymorphic `Date`, a 0-D xarray `DataArray` coordinate scalar
      (which may wrap a date string, datetime, or NumPy timestamp), a string
      tensor scalar, or any other date coordinate scalar (e.g. `pd.Timestamp`).

  Returns:
    The normalized 'YYYY-MM-DD' date string.
  """
  val = time_val.numpy() if hasattr(time_val, "numpy") else time_val
  val = val.item() if hasattr(val, "item") else val
  if isinstance(val, bytes):
    val = val.decode("utf-8")
  if isinstance(val, str):
    val = val[:10]
    # Dates that are already normalized, such as `DataTensors.time`, are
    # returned as is, because parsing them again is slow.
    if _is_normalized_date_str(val):
      return val
  normalized = normalize_date(val)  # pyrefly: ignore[no-matching-overload]
  if normalized is None:
    raise ValueError("`time_val` cannot be None.")
  return normalized.strftime(constants.DATE_FORMAT)


def normalize_times(times: Iterable[Any]) -> list[str]:
  """Returns a list of normalized 'YYYY-MM-DD' date strings from times."""
  return [normalize_date_str(x) for x in _time_values(times)]


def normalize_times_set(times: Iterable[Any]) -> set[str]:
  """Returns a set of normalized 'YYYY-MM-DD' date strings from times."""
  return set(normalize_times(times))


def validate_selected_times(
    selected_times: Sequence[str] | None,
    input_times: xr.DataArray | Sequence[str],
    *,
    arg_name: str,
) -> None:
  """Raises an error if selected_times is invalid.

  Args:
    selected_times: Optional sequence of time names to resolve.
    input_times: Target time period coordinates.
    arg_name: Name of the `selected_times` argument.

  Raises:
    ValueError: A `ValueError` is raised when coordinates in `selected_times` do
      not match time coordinates in `input_times` or is not a list of strings.
  """
  if not selected_times:
    return
  if not all(isinstance(item, str) for item in selected_times):
    raise ValueError(f"`{arg_name}` must be a list of strings.")
  if not _is_normalized_subset(selected_times, input_times):  # pyrefly: ignore[bad-argument-type]
    start_date = normalize_date_str(input_times[0])
    end_date = normalize_date_str(input_times[-1])
    raise ValueError(
        f"`{arg_name}` must match the time dimension coordinates from"
        f" '{start_date}' to '{end_date}'."
    )


def validate_time_coordinates(time: Sequence[str] | None) -> None:
  """Validates that time follows the same rules as `InputData.time`.

  Args:
    time: Optional time coordinates, such as `DataTensors.time`.

  Raises:
    ValueError: If `time` isn't in `constants.DATE_FORMAT` or isn't regularly
      spaced.
  """
  if time is None:
    return
  for t in time:
    if isinstance(t, str) and _is_normalized_date_str(t):
      continue
    try:
      datetime.datetime.strptime(t, constants.DATE_FORMAT)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          f"Invalid time label: {t!r}. Expected format:"
          f" '{constants.DATE_FORMAT}'"
      ) from exc

  if len(time) > 1:
    _validate_time_spacing(tuple(time))


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
    if (
        not self.datetime_index.is_monotonic_increasing
        or not self.datetime_index.is_unique
    ):
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
    return self.datetime_index.diff().dropna()  # pyrefly: ignore[bad-return]

  @property
  def _interval_days(self) -> Sequence[int]:
    """Converts `_timedelta_index` to a sequence of days for easier compute."""
    return self._timedelta_index.days.to_numpy()  # pyrefly: ignore[missing-attribute]

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
      selected_interval: Start and end dates of the interval to select. If
        `None`, then `all_dates` is returned.

    Returns:
      A sequence of dates representing the subset of `all_dates` between the
      given start and end dates, as Python's builtin `datetime.date` objects.

    Raises:
      ValueError: If `selected_interval` is not a subset of `all_dates`.
    """
    if selected_interval is None:
      return self.all_dates

    start_date, end_date = selected_interval
    expanded = self.expand_selected_time_dims(start_date, end_date)
    if expanded is None:
      return self.all_dates
    return expanded

  def get_selected_dates_str(
      self,
      selected_interval: DateInterval | None = None,
  ) -> list[str]:
    """Creates a sequence of date strings containing all points in selected interval.

    Args:
      selected_interval: Start and end dates of the interval to select. If
        `None`, then `all_dates_str` is returned.

    Returns:
      A sequence of date strings representing the subset of `all_dates_str`
      between the given start and end dates.

    Raises:
      ValueError: If `selected_interval` is not a subset of `all_dates`.
    """
    selected_dates = self.get_selected_dates(selected_interval)
    return [d.strftime(constants.DATE_FORMAT) for d in selected_dates]

  @functools.cached_property
  def _period_offset(self) -> pd.DateOffset:
    """One calendar period of this cadence."""
    if np.all(np.isin(self._interval_days, [28, 29, 30, 31])):
      months = 1
    elif np.all(np.isin(self._interval_days, [90, 91, 92])):
      months = 3
    elif np.all(np.isin(self._interval_days, [365, 366])):
      months = 12
    else:
      return pd.DateOffset(days=self.interval_days)
    if self.datetime_index.is_month_end.all():  # pyrefly: ignore[missing-attribute]
      return pd.offsets.MonthEnd(months)
    return pd.DateOffset(months=months)

  def get_period_bounds(
      self,
      selected_interval: DateInterval | None = None,
  ) -> list[tuple[datetime.date, datetime.date]]:
    """Returns contiguous `(start_date, end_date)` pairs for selected dates.

    This method generates contiguous period intervals across the dataset. Each
    returned tuple represents a period `[start_date, end_date)` spanning from a
    given time coordinate up to the start of the next contiguous time
    coordinate.

    For the final time coordinate in the sequence, where no subsequent point
    exists, this method uses the dataset's identified temporal cadence (e.g.,
    monthly, quarterly, yearly, or fixed daily/weekly intervals) to step forward
    by exactly one calendar period. This prevents date-truncation bugs that
    arise when extrapolating calendar months or quarters using fixed timedelta
    day additions.

    Args:
      selected_interval: Start and end dates of the interval to select. If
        `None`, intervals are generated for all dates in the coordinates.

    Returns:
      A list of `(start_date, end_date)` tuples as `datetime.date` objects.

    Raises:
      ValueError: If the time coordinates are not regularly spaced, or if fewer
        than 2 time coordinates are selected.
    """
    if not self._is_regular_time_index():
      raise ValueError("Time coordinates are not regularly spaced!")

    selected_dates = self.get_selected_dates(selected_interval)
    if len(selected_dates) < 2:
      raise ValueError("There must be at least 2 time points.")

    bounds: list[tuple[datetime.date, datetime.date]] = []
    for i, start_date in enumerate(selected_dates):
      if i < len(selected_dates) - 1:
        # Contiguous boundary: period ends exactly when the next period begins.
        end_date = selected_dates[i + 1]
      else:
        # Final period boundary: step forward by exactly one period according
        # to the identified temporal cadence of the dataset.
        end_date = (pd.Timestamp(start_date) + self._period_offset).date()

      bounds.append((start_date, end_date))

    return bounds

  @functools.cached_property
  def period_ends(self) -> Mapping[datetime.date, datetime.date]:
    """Maps each time coordinate to the exclusive end of the period it starts.

    This is `get_period_bounds()` over the whole range, keyed by start date, for
    callers that need to look a single coordinate's period end up rather than
    iterate the sequence.

    Raises:
      ValueError: If the time coordinates are not regularly spaced.
    """
    return dict(self.get_period_bounds())

  def period_end(self, start_date: Date) -> datetime.date:
    """Returns the exclusive end date for a period starting on `start_date`."""
    if start_date is None:
      raise ValueError("`start_date` cannot be None.")
    start_date = normalize_date(start_date)
    if start_date in self.period_ends:
      return self.period_ends[start_date]
    return (pd.Timestamp(start_date) + self._period_offset).date()

  def period_start(self, end_date: Date) -> datetime.date:
    """Returns the start date for a period ending on `end_date`."""
    if end_date is None:
      raise ValueError("`end_date` cannot be None.")
    end_date = normalize_date(end_date)
    period_starts_by_end = {v: k for k, v in self.period_ends.items()}
    if end_date in period_starts_by_end:
      return period_starts_by_end[end_date]
    return (pd.Timestamp(end_date) - self._period_offset).date()

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


def _is_normalized_date_str(date: str) -> bool:
  """Checks whether a string is a valid date in `constants.DATE_FORMAT`.

  This is much faster than parsing the string with `datetime.strptime`, so
  dates that are already normalized, such as `DataTensors.time`, can be checked
  each time they are normalized or validated again.

  Args:
    date: The string to check.

  Returns:
    `True` if `date` is a valid 'YYYY-MM-DD' date string, and `False` otherwise.
  """
  if _DATE_FORMAT_PATTERN.fullmatch(date) is None:
    return False
  try:
    datetime.date.fromisoformat(date)
  except ValueError:
    return False
  return True


def _time_values(times: Iterable[Any]) -> Iterable[Any]:
  """Returns the values of a `DataArray` of times, or the times themselves.

  Iterating over an `xr.DataArray` creates a `DataArray` for each element,
  which is much slower than iterating over its values.

  Args:
    times: An iterable of date coordinates, such as `InputData.time`.

  Returns:
    `times.values` if `times` is an `xr.DataArray`, and `times` otherwise.
  """
  return times.values if isinstance(times, xr.DataArray) else times


def _is_normalized_subset(
    subset: Sequence[Any],
    superset: Sequence[Any],
) -> bool:
  """Returns True if normalized date strings in subset are in superset."""
  try:
    return normalize_times_set(subset) <= normalize_times_set(superset)
  except (ValueError, TypeError):
    return False


@functools.lru_cache(maxsize=16)
def _validate_time_spacing(time: tuple[str, ...]) -> None:
  """Raises an error if time coordinates aren't regularly spaced.

  The result is cached, because `Analyzer` methods validate the same time
  coordinates on every call, for example, for each row of an optimization grid,
  and parsing them with pandas is slow.

  Args:
    time: Time coordinates in `constants.DATE_FORMAT`.
  """
  time_coords = TimeCoordinates.from_dates(time)
  try:
    _ = time_coords.interval_days
  except ValueError as exc:
    raise ValueError("Time coordinates must be regularly spaced.") from exc


def _to_pandas_datetime_index(times: _TimeCoordinateValues) -> pd.DatetimeIndex:
  """Normalizes the given values into a pandas `DatetimeIndex`."""
  # Coerce into pandas `DatetimeIndex` in all possible cases.
  return pd.to_datetime(times)  # pyrefly: ignore[no-matching-overload]


def _to_dates_list(times: pd.DatetimeIndex) -> list[datetime.date]:
  return [dt.date() for dt in times]
