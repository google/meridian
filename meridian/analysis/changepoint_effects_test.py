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

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis import changepoint_effects
from meridian.data import time_coordinates as tc
import numpy as np
import pandas as pd

# pylint: disable=protected-access


def _weekly(n_times: int = 10) -> tc.TimeCoordinates:
  """Weekly time coordinates from 2024-01-01."""
  return tc.TimeCoordinates.from_dates(
      pd.date_range("2024-01-01", periods=n_times, freq="W-MON")
  )


def _monthly() -> tc.TimeCoordinates:
  """Monthly time coordinates from 2024-01-01 to 2024-06-01."""
  return tc.TimeCoordinates.from_dates(
      pd.date_range("2024-01-01", periods=6, freq="MS")
  )


def _date(text: str) -> datetime.date:
  return datetime.date.fromisoformat(text)


class IntervalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("string", "2024-03-04"),
      ("date", datetime.date(2024, 3, 4)),
      ("datetime", datetime.datetime(2024, 3, 4, 12, 30)),
      ("numpy", np.datetime64("2024-03-04")),
  )
  def test_date_is_stored_as_date(self, date):
    effect = changepoint_effects.Interval(date=date)
    self.assertEqual(effect.date, datetime.date(2024, 3, 4))
    self.assertEqual(
        effect, changepoint_effects.Interval(date=datetime.date(2024, 3, 4))
    )

  @parameterized.named_parameters(
      ("bad_string", "2024-13-01"),
      ("position", 10),
      ("none", None),
      ("numpy_missing_date", np.datetime64("NaT")),
      ("pandas_missing_date", pd.NaT),
  )
  def test_non_date_raises(self, date):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`Interval.date` must be a date, for example '2025-03-03', got"
        f" {date!r}.",
    ):
      changepoint_effects.Interval(date=date)


class CheckEffectTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("latest", changepoint_effects.Latest()),
      ("history", changepoint_effects.History()),
      ("interval", changepoint_effects.Interval(date="2024-03-04")),
  )
  def test_effect_is_returned(self, effect):
    self.assertIs(changepoint_effects._check_effect(effect), effect)

  @parameterized.named_parameters(
      ("string", "latest", "changepoint_effects.Latest()"),
      ("capitalized_string", " History ", "changepoint_effects.History()"),
      ("interval_string", "interval", "changepoint_effects.Interval(date=...)"),
      ("class", changepoint_effects.Latest, "changepoint_effects.Latest()"),
  )
  def test_hint(self, effect, suggestion):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        "`changepoint_effect` must be `changepoint_effects.Latest()`,"
        " `changepoint_effects.History()` or"
        " `changepoint_effects.Interval(date=...)`, got"
        f" {effect!r}. Did you mean `{suggestion}`?",
    ):
      changepoint_effects._check_effect(effect)

  @parameterized.named_parameters(
      ("other_string", "all"),
      ("none", None),
      # Unhashable values can't be looked up as a class or a string.
      ("list", ["latest"]),
  )
  def test_no_hint(self, effect):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        "`changepoint_effect` must be `changepoint_effects.Latest()`,"
        " `changepoint_effects.History()` or"
        f" `changepoint_effects.Interval(date=...)`, got {effect!r}.",
    ):
      changepoint_effects._check_effect(effect)


class DescribeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("latest", changepoint_effects.Latest(), "changepoint_effects.Latest()"),
      (
          "history",
          changepoint_effects.History(),
          "changepoint_effects.History()",
      ),
      (
          "interval",
          changepoint_effects.Interval(date=datetime.date(2024, 1, 8)),
          "changepoint_effects.Interval(date='2024-01-08')",
      ),
      ("none", None, "None"),
  )
  def test_describe(self, effect, expected):
    self.assertEqual(changepoint_effects._describe(effect), expected)


class TimeIndicesTest(parameterized.TestCase):

  def test_latest_uses_the_last_time_period(self):
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.Latest(), time_coordinates=_weekly()
        ),
        np.full(10, 9),
    )

  def test_latest_works_for_time_periods_after_the_training_data(self):
    dates = [_date("2024-03-11"), _date("2024-03-18"), _date("2024-03-25")]
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.Latest(),
            time_coordinates=_weekly(),
            dates=dates,
        ),
        [9, 9, 9],
    )

  @parameterized.named_parameters(
      ("first_time_period", "2024-01-01", 0),
      ("start_of_a_time_period", "2024-02-05", 5),
      ("inside_a_time_period", "2024-02-08", 5),
      ("last_day_of_the_training_data", "2024-03-10", 9),
  )
  def test_interval_uses_the_time_period_that_contains_the_date(
      self, date: str, expected: int
  ):
    dates = [_date("2024-03-11"), _date("2024-03-18")]
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.Interval(date=date),
            time_coordinates=_weekly(),
            dates=dates,
        ),
        [expected, expected],
    )

  def test_interval_in_the_last_month(self):
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.Interval(date="2024-06-30"),
            time_coordinates=_monthly(),
        ),
        np.full(6, 5),
    )

  def test_interval_after_the_last_month_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`changepoint_effects.Interval(date='2024-07-01')` must be a date in"
        " the training data (2024-01-01 to 2024-06-30). To use the effect of"
        " each channel's latest interval, use `changepoint_effects.Latest()`.",
    ):
      changepoint_effects._time_indices(
          changepoint_effects.Interval(date="2024-07-01"),
          time_coordinates=_monthly(),
      )

  def test_interval_before_the_training_data_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`changepoint_effects.Interval(date='2023-12-31')` must be a date in"
        " the training data (2024-01-01 to 2024-03-10).",
    ):
      changepoint_effects._time_indices(
          changepoint_effects.Interval(date="2023-12-31"),
          time_coordinates=_weekly(),
      )

  def test_interval_after_the_training_data_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`changepoint_effects.Interval(date='2024-03-11')` must be a date in"
        " the training data (2024-01-01 to 2024-03-10). To use the effect of"
        " each channel's latest interval, use `changepoint_effects.Latest()`.",
    ):
      changepoint_effects._time_indices(
          changepoint_effects.Interval(date="2024-03-11"),
          time_coordinates=_weekly(),
      )

  @parameterized.named_parameters(
      ("latest", changepoint_effects.Latest()),
      ("history", changepoint_effects.History()),
      ("interval", changepoint_effects.Interval(date="2024-02-05")),
  )
  def test_no_dates_gives_empty_indices(self, effect):
    indices = changepoint_effects._time_indices(
        effect, time_coordinates=_weekly(), dates=[]
    )
    self.assertEqual(indices.shape, (0,))
    self.assertEqual(indices.dtype, int)

  def test_history_uses_each_time_period(self):
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.History(), time_coordinates=_weekly()
        ),
        np.arange(10),
    )

  def test_history_matches_time_periods_by_date(self):
    dates = [_date("2024-02-05"), _date("2024-01-08"), _date("2024-03-04")]
    np.testing.assert_array_equal(
        changepoint_effects._time_indices(
            changepoint_effects.History(),
            time_coordinates=_weekly(),
            dates=dates,
        ),
        [5, 1, 9],
    )

  @parameterized.named_parameters(
      ("one", ["2024-03-04", "2024-03-11"], "2024-03-11"),
      (
          "unsorted_and_repeated",
          ["2024-03-25", "2024-01-01", "2024-03-11", "2024-03-25"],
          "2024-03-11, 2024-03-25",
      ),
      (
          "more_than_five",
          [f"2024-04-{day:02d}" for day in (1, 8, 15, 22, 29)]
          + ["2024-05-06", "2024-05-13"],
          (
              "2024-04-01, 2024-04-08, 2024-04-15, 2024-04-22, 2024-04-29 and"
              " 2 more"
          ),
      ),
  )
  def test_history_of_other_time_periods_raises(
      self, dates: list[str], listed: str
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`changepoint_effects.History()` uses each time period's own effect,"
        " so it is only defined for the time periods of the training data"
        " (2024-01-01 to 2024-03-10). These dates aren't the start of one:"
        f" {listed}. Use `changepoint_effects.Latest()` or"
        " `changepoint_effects.Interval(date=...)` for them.",
    ):
      changepoint_effects._time_indices(
          changepoint_effects.History(),
          time_coordinates=_weekly(),
          dates=[_date(date) for date in dates],
      )


if __name__ == "__main__":
  absltest.main()
