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

"""Which effect budget optimization uses for channels with changepoints.

A channel with changepoints has a different effect in each interval between
its changepoints. Budget optimization has to assume one effect for each time
period it optimizes. The classes in this module say which one:

* `Latest()`: the effect of each channel's latest interval, for every time
  period. This is the default of `BudgetOptimizer.optimize()` and
  `BudgetOptimizer.create_optimization_grid()`, since budget optimization
  usually plans for the future.
* `History()`: each time period's own effect, the same calculation as
  `Analyzer.roi()`. Only defined for time periods in the training data.
  `Analyzer` methods, such as `Analyzer.roi()` and
  `Analyzer.incremental_outcome()`, always use each time period's own effect,
  so with `History()` the optimizer's results for the original budget match the
  Analyzer's.
* `Interval(date=...)`: the effect of the interval that contains `date`, for
  every time period.

To use a separate interval for each channel, pass a mapping from channel names
to `Interval(date=...)` instead, for example:

    {'TV': Interval(date='2025-03-03'), 'YouTube': Interval(date='2025-09-01')}

Each channel in the mapping uses the effect of its own interval, for every time
period. Channels with changepoints that the mapping leaves out use `Latest()`.

Channels without changepoints have one effect, so every option gives them the
same result.
"""

import bisect
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import datetime
from typing import TypeAlias, cast

from meridian.data import time_coordinates as tc
import numpy as np

__all__ = [
    "ChangepointEffect",
    "History",
    "Interval",
    "Latest",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Latest:
  """Uses the effect of each channel's latest interval for every time period.

  This assumes that the effect of the latest interval continues, so it can be
  used for any time periods, including time periods after the training data.
  """


@dataclasses.dataclass(frozen=True, kw_only=True)
class History:
  """Uses each time period's own effect.

  Each time period uses the effect of the interval it falls in, the same
  calculation as `Analyzer.roi()`. Time periods are matched by date, so this is
  only defined for time periods in the training data.
  """


@dataclasses.dataclass(frozen=True, kw_only=True)
class Interval:
  """Uses the effect of the interval containing `date` for every time period.

  Attributes:
    date: A date in the training data, for example `'2025-03-03'`. For each
      channel with changepoints, the effect of the interval that contains this
      date is used. Channels can have different changepoints, so the date can
      fall in a different interval for each channel. Can be a 'YYYY-MM-DD'
      string, `datetime.date`, `datetime.datetime` or `np.datetime64`, and is
      stored as a `datetime.date`.

  To use a separate interval for each channel, map channel names to `Interval`
  instances, for example `{'TV': Interval(date='2025-03-03')}`. Channels with
  changepoints that the mapping leaves out use `Latest()`.
  """

  date: tc.Date

  def __post_init__(self):
    message = (
        "`Interval.date` must be a date, for example '2025-03-03', got"
        f" {self.date!r}."
    )
    try:
      date = tc.normalize_date(self.date)
    except (TypeError, ValueError) as e:
      raise ValueError(message) from e
    # Missing dates, such as `np.datetime64('NaT')` and `pd.NaT`, normalize to
    # `None` or `NaT` rather than a `datetime.date`.
    if not isinstance(date, datetime.date) or isinstance(
        date, datetime.datetime
    ):
      raise ValueError(message)
    object.__setattr__(self, "date", date)


ChangepointEffect: TypeAlias = Latest | History | Interval


class _Intervals(Mapping[str, Interval]):
  """A read-only mapping from channel names to `Interval` instances.

  It equals any mapping with the same items, through the `__eq__` of
  `collections.abc.Mapping`.
  """

  def __init__(self, intervals: Mapping[str, Interval]):
    self._intervals = dict(intervals)

  def __getitem__(self, channel: str) -> Interval:
    return self._intervals[channel]

  def __iter__(self) -> Iterator[str]:
    return iter(self._intervals)

  def __len__(self) -> int:
    return len(self._intervals)

  def __repr__(self) -> str:
    return repr(self._intervals)


# A changepoint effect, or a mapping from channel names to `Interval` instances
# that gives each channel its own interval.
_EffectOrIntervals: TypeAlias = ChangepointEffect | Mapping[str, Interval]

_EFFECT_NAMES = {
    Latest: "changepoint_effects.Latest()",
    History: "changepoint_effects.History()",
    Interval: "changepoint_effects.Interval(date=...)",
}
_EFFECTS_BY_STRING = {
    "latest": Latest,
    "history": History,
    "interval": Interval,
}
# The most dates that an error message lists.
_MAX_LISTED = 5


def _check_effect(effect: object) -> _EffectOrIntervals:
  """Returns `effect` if it's a changepoint effect or a mapping of intervals.

  Args:
    effect: The value to check.

  Returns:
    `effect`, or for a mapping from channel names to `Interval` instances, a
    read-only copy of it with the channel names as strings, so that later
    changes to `effect` don't change the copy.

  Raises:
    TypeError: If `effect` isn't a `Latest`, `History` or `Interval` instance
      or a mapping from channel names to `Interval` instances. For a class or a
      string such as `'latest'`, the message suggests the right value.
  """
  if isinstance(effect, ChangepointEffect):
    return effect
  if isinstance(effect, Mapping):
    # Channel names from NumPy arrays are `np.str_`, which messages would
    # write as `np.str_('TV')`.
    intervals = {str(channel): interval for channel, interval in effect.items()}
    for channel, interval in intervals.items():
      if not isinstance(interval, Interval):
        raise TypeError(
            f"`changepoint_effect[{channel!r}]` must be"
            f" `{_EFFECT_NAMES[Interval]}`, got {interval!r}."
        )
    return _Intervals(intervals)
  if isinstance(effect, type) and effect in _EFFECT_NAMES:
    suggestion = effect
  elif isinstance(effect, str):
    suggestion = _EFFECTS_BY_STRING.get(effect.strip().lower())
  else:
    suggestion = None
  hint = (
      f" Did you mean `{_EFFECT_NAMES[suggestion]}`?"
      if suggestion is not None
      else ""
  )
  raise TypeError(
      f"`changepoint_effect` must be `{_EFFECT_NAMES[Latest]}`,"
      f" `{_EFFECT_NAMES[History]}` or `{_EFFECT_NAMES[Interval]}`, or a"
      f" mapping from channel names to `{_EFFECT_NAMES[Interval]}`, got"
      f" {effect!r}.{hint}"
  )


def _check_channels(
    effect: Mapping[str, Interval], channels: Sequence[str]
) -> None:
  """Checks that the channels of a mapping of intervals have changepoints.

  Args:
    effect: Maps channel names to `Interval` instances.
    channels: The names of the channels with changepoints, in the model's
      channel order.

  Raises:
    ValueError: If some channels of `effect` aren't in `channels`. The message
      lists them in the order of `effect`.
  """
  others = [channel for channel in effect if channel not in channels]
  if others:
    raise ValueError(
        "The `changepoint_effect` mapping has channels without changepoints:"
        f" {others}. Its keys must be channels with changepoints:"
        f" {list(channels)}."
    )


def _describe(effect: _EffectOrIntervals | None) -> str:
  """Writes `effect` the way a user creates it, for messages.

  Args:
    effect: The changepoint effect, a mapping from channel names to `Interval`
      instances, or `None`.

  Returns:
    For example `changepoint_effects.Interval(date='2025-03-03')`,
    `{'TV': changepoint_effects.Interval(date='2025-03-03')}`, or `None`.
  """
  if isinstance(effect, Mapping):
    items = ", ".join(
        f"{channel!r}: {_describe(interval)}"
        for channel, interval in effect.items()
    )
    return f"{{{items}}}"
  if isinstance(effect, Interval):
    # `Interval.__post_init__` normalizes the date to a `datetime.date`.
    date = cast(datetime.date, effect.date)
    return f"changepoint_effects.Interval(date={date.isoformat()!r})"
  if effect is None:
    return "None"
  return _EFFECT_NAMES[type(effect)]


def _channel_time_indices(
    effect: Mapping[str, Interval],
    *,
    time_coordinates: tc.TimeCoordinates,
    channels: Sequence[str],
    n_dates: int,
) -> np.ndarray:
  """For each time period and channel, the training time period it uses.

  Args:
    effect: Maps channel names to `Interval` instances. Its channels must be
      checked with `_check_channels` first.
    time_coordinates: The time coordinates of the training data.
    channels: The channels to look up, in order.
    n_dates: The number of time periods to look up.

  Returns:
    The indices, shape `(n_dates, len(channels))`, with a column for each
    channel: the index of the training time period that contains the date of
    the channel's `Interval`, or the last one for channels that `effect` leaves
    out, as for `Latest`.

  Raises:
    ValueError: If the date of an `Interval` isn't in the training data.
  """
  latest = len(time_coordinates.all_dates) - 1
  by_channel = [
      _interval_time_index(effect[channel], time_coordinates, channel)
      if channel in effect
      else latest
      for channel in channels
  ]
  return np.tile(np.array(by_channel, dtype=int), (n_dates, 1))


def _time_indices(
    effect: ChangepointEffect,
    *,
    time_coordinates: tc.TimeCoordinates,
    dates: Sequence[datetime.date] | None = None,
) -> np.ndarray:
  """For each time period, the training time period whose effect it uses.

  Args:
    effect: The changepoint effect.
    time_coordinates: The time coordinates of the training data.
    dates: The time periods to look up, as their start dates. Default: the
      time periods of the training data.

  Returns:
    For each date in `dates`, the index of the training time period whose
    effect it uses: the last one for `Latest`, the one that contains the date
    of `Interval`, and the one with the same date for `History`.

  Raises:
    ValueError: If the date of an `Interval` isn't in the training data, or
      `History` gets dates that aren't time periods of the training data.
  """
  training_dates = time_coordinates.all_dates
  if dates is None:
    dates = training_dates
  if isinstance(effect, Latest):
    return np.full(len(dates), len(training_dates) - 1, dtype=int)
  if isinstance(effect, Interval):
    index = _interval_time_index(effect, time_coordinates)
    return np.full(len(dates), index, dtype=int)
  index_of = {date: index for index, date in enumerate(training_dates)}
  missing = sorted({date for date in dates if date not in index_of})
  if missing:
    first, last_day = _training_range(time_coordinates)
    listed = ", ".join(date.isoformat() for date in missing[:_MAX_LISTED])
    if len(missing) > _MAX_LISTED:
      listed += f" and {len(missing) - _MAX_LISTED} more"
    raise ValueError(
        f"`{_EFFECT_NAMES[History]}` uses each time period's own effect, so it"
        " is only defined for the time periods of the training data"
        f" ({first.isoformat()} to {last_day.isoformat()}). These dates aren't"
        f" the start of one: {listed}. Use `{_EFFECT_NAMES[Latest]}` or"
        f" `{_EFFECT_NAMES[Interval]}` for them."
    )
  return np.array([index_of[date] for date in dates], dtype=int)


def _interval_time_index(
    effect: Interval,
    time_coordinates: tc.TimeCoordinates,
    channel: str | None = None,
) -> int:
  """The index of the training time period that contains `effect.date`.

  Args:
    effect: The `Interval` effect.
    time_coordinates: The time coordinates of the training data.
    channel: The channel that `effect` belongs to in a mapping from channel
      names to `Interval` instances, for messages. `None` if `effect` isn't in
      a mapping.

  Returns:
    The index of the time period that contains the date.

  Raises:
    ValueError: If the date is before the first time period or after the last
      one ends.
  """
  training_dates = time_coordinates.all_dates
  first, last_day = _training_range(time_coordinates)
  # `Interval.__post_init__` normalizes the date to a `datetime.date`.
  date = cast(datetime.date, effect.date)
  if first <= date <= last_day:
    return bisect.bisect_right(training_dates, date) - 1
  subject = f"`{_describe(effect)}`"
  if channel is not None:
    subject = f"`changepoint_effect[{channel!r}]`, {subject},"
  message = (
      f"{subject} must be a date in the training data"
      f" ({first.isoformat()} to {last_day.isoformat()})."
  )
  if date > last_day:
    if channel is None:
      message += (
          " To use the effect of each channel's latest interval, use"
          f" `{_EFFECT_NAMES[Latest]}`."
      )
    else:
      message += (
          " To use the effect of the channel's latest interval, leave"
          f" {channel!r} out of the mapping."
      )
  raise ValueError(message)


def _training_range(
    time_coordinates: tc.TimeCoordinates,
) -> tuple[datetime.date, datetime.date]:
  """The first and last days of the training data.

  Args:
    time_coordinates: The time coordinates of the training data.

  Returns:
    The start of the first time period, and the last day of the last one.
  """
  training_dates = time_coordinates.all_dates
  last_day = time_coordinates.period_end(training_dates[-1]) - (
      datetime.timedelta(days=1)
  )
  return training_dates[0], last_day
