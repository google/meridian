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

import datetime as dt
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import time_coordinates
import numpy as np
import pandas as pd
import xarray as xr

_ALL_DATES = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
    "2024-01-29",
    "2024-02-05",
    "2024-02-12",
    "2024-02-19",
]


class TimeCoordinatesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.all_dates = xr.DataArray(
        data=np.array(_ALL_DATES),
        dims=[constants.TIME],
        coords={
            constants.TIME: (
                [constants.TIME],
                _ALL_DATES,
            ),
        },
    )
    self.coordinates = time_coordinates.TimeCoordinates.from_dates(
        self.all_dates
    )

  def test_constructor_must_have_more_than_one_date(self):
    with self.assertRaisesRegex(
        ValueError,
        "There must be more than one date index",
    ):
      time_coordinates.TimeCoordinates.from_dates(
          pd.DatetimeIndex([np.datetime64("2024-01-01")])
      )

  def test_property_all_dates(self):
    expected_dates = [
        dt.datetime.strptime(date, constants.DATE_FORMAT).date()
        for date in _ALL_DATES
    ]
    self.assertEqual(self.coordinates.all_dates, expected_dates)

  def test_property_all_dates_str(self):
    self.assertEqual(self.coordinates.all_dates_str, _ALL_DATES)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_ascending_times_date_strings",
          all_dates=xr.DataArray(
              data=np.array(["2024-01-01", "2024-01-08", "2024-01-07"]),
              dims=[constants.TIME],
              coords={
                  constants.TIME: (
                      [constants.TIME],
                      ["2024-01-01", "2024-01-08", "2024-01-07"],
                  ),
              },
          ),
      ),
      dict(
          testcase_name="non_ascending_times_datetime_index",
          all_dates=pd.DatetimeIndex([
              np.datetime64("2024-01-01"),
              np.datetime64("2024-01-08"),
              np.datetime64("2024-01-15"),
              np.datetime64("2024-01-08"),
              np.datetime64("2024-01-01"),
              np.datetime64("2024-01-05"),
          ]),
      ),
  )
  def test_init_raises_on_non_ascending_times(
      self, all_dates: time_coordinates.TimeCoordinates
  ):
    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates must be strictly monotonically increasing.",
    ):
      time_coordinates.TimeCoordinates.from_dates(all_dates)  # pyrefly: ignore[bad-argument-type]

  def test_property_interval_days_weekly(self):
    self.assertEqual(self.coordinates.interval_days, 7)

  def test_property_interval_days_daily(self):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
            np.datetime64("2024-01-04"),
            np.datetime64("2024-01-05"),
        ]),
    )
    self.assertEqual(coordinates.interval_days, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly_interval_days",
          dates=[
              dt.datetime(2024, 1, 1) + dt.timedelta(days=7 * i)
              for i in range(14)
          ],
          expected_interval_days=7,
      ),
      dict(
          testcase_name="thirty_day_interval_days",
          dates=[
              dt.datetime(2024, 1, 1) + dt.timedelta(days=30 * i)
              for i in range(24)
          ],
          expected_interval_days=30,
      ),
  )
  def test_property_interval_days_regular(
      self, dates: list[dt.date], expected_interval_days
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    self.assertEqual(coordinates.interval_days, expected_interval_days)

  @parameterized.named_parameters(
      dict(
          testcase_name="monthly_on_the_first_day_of_each_month_one_year",
          dates=[dt.datetime(2024, m, 1) for m in range(1, 13)],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="monthly_on_the_last_day_of_each_month_two_years",
          dates=[
              dt.datetime(2024 + y, m, 1) - dt.timedelta(days=1)  # pylint: disable=g-complex-comprehension
              for y in range(2)
              for m in range(1, 13)
          ],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="monthly_on_the_fifteenth_each_month_four_years",
          dates=[
              dt.datetime(2024 + y, m, 15)  # pylint: disable=g-complex-comprehension
              for y in range(4)
              for m in range(1, 13)
          ],
          expected_interval_days=30,
      ),
      dict(
          testcase_name="quarterly_on_the_first_day_of_each_quarter_one_year",
          dates=[dt.datetime(2024, q * 3 + 1, 1) for q in range(0, 4)],
          expected_interval_days=91,
      ),
      dict(
          testcase_name="yearly_with_leap_day",
          dates=[dt.datetime(2023 + y, 1, 1) for y in range(5)],
          expected_interval_days=365,
      ),
  )
  def test_property_interval_days_fuzzy_regular(
      self,
      dates: list[dt.date],
      expected_interval_days: int,
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    self.assertEqual(coordinates.interval_days, expected_interval_days)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly_misses_one_entire_week",
          dates=[
              dt.datetime(2024, 1, 1)
              + dt.timedelta(days=7 * (i + 1 if i >= 4 else i))
              for i in range(20)
          ],
      ),
      dict(
          testcase_name="weekly_one_week_is_half_week",
          dates=[
              dt.datetime(2024, 1, 1)
              + dt.timedelta(days=7 * i + (4 if i == 10 else 0))
              for i in range(25)
          ],
      ),
      dict(
          testcase_name="monthly_one_month_is_off_by_ten_days",
          dates=[
              # 2024/3/10 on March, 15th for the rest.
              dt.date(2024, m, 15) - dt.timedelta(days=(10 if m == 2 else 0))
              for m in range(1, 13)
          ],
      ),
  )
  def test_property_interval_days_fuzzy_irregular(
      self,
      dates: list[dt.date],
  ):
    coordinates = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates are not regularly spaced!",
    ):
      _ = coordinates.interval_days

  def test_property_irregular_interval_days(self):
    dates = ["2024-01-01", "2024-01-03", "2024-01-10"]
    all_dates = xr.DataArray(
        data=np.array(dates),
        dims=[constants.TIME],
        coords={
            constants.TIME: ([constants.TIME], dates),
        },
    )
    coordinates = time_coordinates.TimeCoordinates.from_dates(all_dates)

    with self.assertRaisesRegex(
        ValueError,
        "Time coordinates are not regularly spaced!",
    ):
      _ = coordinates.interval_days

  def test_get_selected_dates_selected_interval_is_none(self):
    times = self.coordinates.get_selected_dates(
        selected_interval=None,
    )
    self.assertSameElements(
        [t.strftime(constants.DATE_FORMAT) for t in times], self.all_dates.data
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="selected_interval_tuple_str",
          selected_interval=("2024-01-01", "2024-02-19"),
      ),
      dict(
          testcase_name="selected_interval_tuple_datetime",
          selected_interval=(
              dt.datetime(year=2024, month=1, day=1),
              dt.datetime(year=2024, month=2, day=19),
          ),
      ),
      dict(
          testcase_name="selected_interval_tuple_np_datetime64",
          selected_interval=(
              np.datetime64("2024-01-01"),
              np.datetime64("2024-02-19"),
          ),
      ),
      dict(
          testcase_name="selected_interval_date_interval",
          selected_interval=(
              dt.date(year=2024, month=1, day=1),
              dt.date(year=2024, month=2, day=19),
          ),
      ),
  )
  def test_get_selected_dates_selected_interval_matches_range_of_all_dates(
      self, selected_interval: time_coordinates.DateInterval
  ):
    times = self.coordinates.get_selected_dates(
        selected_interval=selected_interval
    )
    self.assertSameElements(
        [t.strftime(constants.DATE_FORMAT) for t in times], self.all_dates.data
    )

  def test_get_selected_dates_selected_interval_is_not_subset_of_all_dates(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"end_date \(2024-02-26\) must be in the time coordinates!",
    ):
      self.coordinates.get_selected_dates(
          selected_interval=("2024-01-01", "2024-02-26"),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="first_half_of_all_dates",
          selected_interval=("2024-01-01", "2024-01-15"),
          expected_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
      ),
      dict(
          testcase_name="second_half_of_all_dates",
          selected_interval=("2024-02-05", "2024-02-19"),
          expected_dates=["2024-02-05", "2024-02-12", "2024-02-19"],
      ),
      dict(
          testcase_name="middle_of_all_dates",
          selected_interval=("2024-01-22", "2024-02-05"),
          expected_dates=["2024-01-22", "2024-01-29", "2024-02-05"],
      ),
  )
  def test_get_selected_dates_converts_selected_interval_into_list_of_dates(
      self, selected_interval: tuple[str, str], expected_dates: list[str]
  ):
    dates = self.coordinates.get_selected_dates(
        selected_interval=selected_interval,
    )
    expected_dates = [  # pyrefly: ignore[bad-assignment]
        dt.datetime.strptime(date, constants.DATE_FORMAT).date()
        for date in expected_dates
    ]
    self.assertEqual(dates, expected_dates)

  def test_get_selected_dates_with_none_boundaries(self):
    dates_start_only = self.coordinates.get_selected_dates(
        selected_interval=("2024-02-05", None)
    )
    self.assertEqual(
        dates_start_only,
        [
            dt.date(2024, 2, 5),
            dt.date(2024, 2, 12),
            dt.date(2024, 2, 19),
        ],
    )
    dates_end_only = self.coordinates.get_selected_dates(
        selected_interval=(None, "2024-01-15")
    )
    self.assertEqual(
        dates_end_only,
        [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 8),
            dt.date(2024, 1, 15),
        ],
    )

  def test_get_selected_dates_str_selected_interval_is_none(self):
    self.assertEqual(
        self.coordinates.get_selected_dates_str(),
        self.coordinates.all_dates_str,
    )
    self.assertEqual(
        self.coordinates.get_selected_dates_str(selected_interval=None),
        self.coordinates.all_dates_str,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="selected_interval_tuple_str",
          selected_interval=("2024-01-01", "2024-02-19"),
      ),
      dict(
          testcase_name="selected_interval_tuple_datetime",
          selected_interval=(
              dt.datetime(year=2024, month=1, day=1),
              dt.datetime(year=2024, month=2, day=19),
          ),
      ),
      dict(
          testcase_name="selected_interval_tuple_np_datetime64",
          selected_interval=(
              np.datetime64("2024-01-01"),
              np.datetime64("2024-02-19"),
          ),
      ),
      dict(
          testcase_name="selected_interval_date_interval",
          selected_interval=(
              dt.date(year=2024, month=1, day=1),
              dt.date(year=2024, month=2, day=19),
          ),
      ),
  )
  def test_get_selected_dates_str_selected_interval_matches_range_of_all_dates(
      self, selected_interval: time_coordinates.DateInterval
  ):
    times = self.coordinates.get_selected_dates_str(
        selected_interval=selected_interval
    )
    self.assertEqual(times, self.coordinates.all_dates_str)

  def test_get_selected_dates_str_selected_interval_is_not_subset_of_all_dates(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"end_date \(2024-02-26\) must be in the time coordinates!",
    ):
      self.coordinates.get_selected_dates_str(
          selected_interval=("2024-01-01", "2024-02-26"),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="first_half_of_all_dates",
          selected_interval=("2024-01-01", "2024-01-15"),
          expected_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
      ),
      dict(
          testcase_name="second_half_of_all_dates",
          selected_interval=("2024-02-05", "2024-02-19"),
          expected_dates=["2024-02-05", "2024-02-12", "2024-02-19"],
      ),
      dict(
          testcase_name="middle_of_all_dates",
          selected_interval=("2024-01-22", "2024-02-05"),
          expected_dates=["2024-01-22", "2024-01-29", "2024-02-05"],
      ),
      dict(
          testcase_name="start_only",
          selected_interval=("2024-02-05", None),
          expected_dates=["2024-02-05", "2024-02-12", "2024-02-19"],
      ),
      dict(
          testcase_name="end_only",
          selected_interval=(None, "2024-01-15"),
          expected_dates=["2024-01-01", "2024-01-08", "2024-01-15"],
      ),
  )
  def test_get_selected_dates_str_converts_selected_interval_into_list_of_strings(
      self,
      selected_interval: tuple[str | None, str | None],
      expected_dates: list[str],
  ):
    dates = self.coordinates.get_selected_dates_str(
        selected_interval=selected_interval,
    )
    self.assertEqual(dates, expected_dates)

  @parameterized.named_parameters(
      dict(
          testcase_name="start_and_end",
          start_date=dt.datetime(2024, 1, 8).date(),
          end_date=dt.datetime(2024, 2, 5),
          expected_time_dims=[
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
          ],
      ),
      dict(
          testcase_name="start_only",
          start_date=dt.datetime(2024, 1, 8).date(),
          end_date=None,
          expected_time_dims=[
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
              dt.datetime(2024, 2, 12).date(),
              dt.datetime(2024, 2, 19).date(),
          ],
      ),
      dict(
          testcase_name="end_only",
          start_date=None,
          end_date=dt.datetime(2024, 2, 5).date(),
          expected_time_dims=[
              dt.datetime(2024, 1, 1).date(),
              dt.datetime(2024, 1, 8).date(),
              dt.datetime(2024, 1, 15).date(),
              dt.datetime(2024, 1, 22).date(),
              dt.datetime(2024, 1, 29).date(),
              dt.datetime(2024, 2, 5).date(),
          ],
      ),
      dict(
          testcase_name="none",
          start_date=None,
          end_date=None,
          expected_time_dims=None,
      ),
      dict(
          testcase_name="start_and_end_are_entire_range",
          start_date=dt.datetime(2024, 1, 1),
          end_date=dt.datetime(2024, 2, 19),
          expected_time_dims=None,
      ),
  )
  def test_expand_selected_time_dims(
      self, start_date, end_date, expected_time_dims
  ):
    self.assertEqual(
        self.coordinates.expand_selected_time_dims(start_date, end_date),
        expected_time_dims,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="start_not_in_data",
          start_date=dt.datetime(2023, 12, 15).date(),
          end_date=dt.datetime(2024, 1, 8).date(),
          expected_error_message=(
              r"start_date \(2023-12-15\) must be in the time coordinates!"
          ),
      ),
      dict(
          testcase_name="end_not_in_data",
          start_date=dt.datetime(2024, 1, 1),
          end_date=dt.datetime(2024, 3, 11),
          expected_error_message=(
              r"end_date \(2024-03-11\) must be in the time coordinates!"
          ),
      ),
      dict(
          testcase_name="start_after_end",
          start_date="2024-01-29",
          end_date="2024-01-01",
          expected_error_message=(
              r"start_date \(2024-01-29\) must be less than or equal to"
              r" end_date \(2024-01-01\)!"
          ),
      ),
  )
  def test_expand_selected_time_dims_fails(
      self, start_date, end_date, expected_error_message
  ):
    with self.assertRaisesRegex(ValueError, expected_error_message):
      self.coordinates.expand_selected_time_dims(start_date, end_date)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly",
          dates=[
              dt.date(2024, 1, 1),
              dt.date(2024, 1, 8),
              dt.date(2024, 1, 15),
          ],
          expected_bounds=[
              (dt.date(2024, 1, 1), dt.date(2024, 1, 8)),
              (dt.date(2024, 1, 8), dt.date(2024, 1, 15)),
              (dt.date(2024, 1, 15), dt.date(2024, 1, 22)),
          ],
      ),
      dict(
          testcase_name="monthly_leap_year",
          dates=[
              dt.date(2024, 1, 1),
              dt.date(2024, 2, 1),
              dt.date(2024, 3, 1),
          ],
          expected_bounds=[
              (dt.date(2024, 1, 1), dt.date(2024, 2, 1)),
              (dt.date(2024, 2, 1), dt.date(2024, 3, 1)),
              (dt.date(2024, 3, 1), dt.date(2024, 4, 1)),
          ],
      ),
      dict(
          testcase_name="quarterly",
          dates=[
              dt.date(2024, 1, 1),
              dt.date(2024, 4, 1),
          ],
          expected_bounds=[
              (dt.date(2024, 1, 1), dt.date(2024, 4, 1)),
              (dt.date(2024, 4, 1), dt.date(2024, 7, 1)),
          ],
      ),
      dict(
          testcase_name="yearly",
          dates=[
              dt.date(2023, 1, 1),
              dt.date(2024, 1, 1),
          ],
          expected_bounds=[
              (dt.date(2023, 1, 1), dt.date(2024, 1, 1)),
              (dt.date(2024, 1, 1), dt.date(2025, 1, 1)),
          ],
      ),
      dict(
          testcase_name="monthly_month_end_thirty_to_thirty_one_days",
          dates=[
              dt.date(2024, 9, 30),
              dt.date(2024, 10, 31),
              dt.date(2024, 11, 30),
          ],
          expected_bounds=[
              (dt.date(2024, 9, 30), dt.date(2024, 10, 31)),
              (dt.date(2024, 10, 31), dt.date(2024, 11, 30)),
              (dt.date(2024, 11, 30), dt.date(2024, 12, 31)),
          ],
      ),
      dict(
          testcase_name="monthly_month_end_january_to_leap_february",
          dates=[
              dt.date(2023, 12, 31),
              dt.date(2024, 1, 31),
          ],
          expected_bounds=[
              (dt.date(2023, 12, 31), dt.date(2024, 1, 31)),
              (dt.date(2024, 1, 31), dt.date(2024, 2, 29)),
          ],
      ),
      dict(
          testcase_name="quarterly_month_end",
          dates=[
              dt.date(2024, 3, 31),
              dt.date(2024, 6, 30),
              dt.date(2024, 9, 30),
          ],
          expected_bounds=[
              (dt.date(2024, 3, 31), dt.date(2024, 6, 30)),
              (dt.date(2024, 6, 30), dt.date(2024, 9, 30)),
              (dt.date(2024, 9, 30), dt.date(2024, 12, 31)),
          ],
      ),
      dict(
          testcase_name="yearly_month_end_leap_february",
          dates=[
              dt.date(2023, 2, 28),
              dt.date(2024, 2, 29),
          ],
          expected_bounds=[
              (dt.date(2023, 2, 28), dt.date(2024, 2, 29)),
              (dt.date(2024, 2, 29), dt.date(2025, 2, 28)),
          ],
      ),
  )
  def test_get_period_bounds_regular_cadences(self, dates, expected_bounds):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex(dates)
    )
    self.assertEqual(coords.get_period_bounds(), expected_bounds)

  def test_get_period_bounds_fewer_than_two_points_raises(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([dt.date(2024, 1, 1), dt.date(2024, 1, 8)])
    )
    with self.assertRaisesRegex(
        ValueError, "There must be at least 2 time points."
    ):
      coords.get_period_bounds(selected_interval=("2024-01-01", "2024-01-01"))

  def test_get_period_bounds_irregular_raises(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 10),
        ])
    )
    with self.assertRaisesRegex(
        ValueError, "Time coordinates are not regularly spaced!"
    ):
      coords.get_period_bounds()

  def test_period_ends_keys_the_bounds_by_start_date(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
            dt.date(2024, 3, 1),
        ])
    )

    self.assertEqual(
        coords.period_ends,
        {
            dt.date(2024, 1, 1): dt.date(2024, 2, 1),
            dt.date(2024, 2, 1): dt.date(2024, 3, 1),
            dt.date(2024, 3, 1): dt.date(2024, 4, 1),
        },
    )

  def test_period_ends_is_cached(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([dt.date(2024, 1, 1), dt.date(2024, 1, 8)])
    )

    self.assertIs(coords.period_ends, coords.period_ends)

  def test_period_ends_irregular_raises(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 10),
        ])
    )
    with self.assertRaisesRegex(
        ValueError, "Time coordinates are not regularly spaced!"
    ):
      _ = coords.period_ends

  def test_period_end_in_sample_and_polymorphic(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
            dt.date(2024, 3, 1),
        ])
    )
    self.assertEqual(coords.period_end("2024-01-01"), dt.date(2024, 2, 1))
    self.assertEqual(
        coords.period_end(dt.date(2024, 2, 1)), dt.date(2024, 3, 1)
    )
    self.assertEqual(
        coords.period_end(dt.datetime(2024, 3, 1, 12, 0)),
        dt.date(2024, 4, 1),
    )

  def test_period_end_out_of_sample_uses_period_offset(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
        ])
    )
    self.assertEqual(coords.period_end("2025-05-01"), dt.date(2025, 6, 1))

    month_end_coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 9, 30),
            dt.date(2024, 10, 31),
        ])
    )
    self.assertEqual(
        month_end_coords.period_end("2024-11-30"), dt.date(2024, 12, 31)
    )

  def test_period_end_none_raises(self):
    with self.assertRaisesRegex(ValueError, "`start_date` cannot be None."):
      self.coordinates.period_end(None)

  def test_period_start_in_sample_and_polymorphic(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
            dt.date(2024, 3, 1),
        ])
    )
    self.assertEqual(coords.period_start("2024-02-01"), dt.date(2024, 1, 1))
    self.assertEqual(
        coords.period_start(dt.date(2024, 3, 1)), dt.date(2024, 2, 1)
    )
    self.assertEqual(
        coords.period_start(dt.datetime(2024, 4, 1, 10, 0)),
        dt.date(2024, 3, 1),
    )

  def test_period_start_out_of_sample_uses_period_offset(self):
    coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
        ])
    )
    self.assertEqual(coords.period_start("2025-06-01"), dt.date(2025, 5, 1))

    month_end_coords = time_coordinates.TimeCoordinates.from_dates(
        pd.DatetimeIndex([
            dt.date(2024, 9, 30),
            dt.date(2024, 10, 31),
        ])
    )
    self.assertEqual(
        month_end_coords.period_start("2024-12-31"), dt.date(2024, 11, 30)
    )

  def test_period_start_none_raises(self):
    with self.assertRaisesRegex(ValueError, "`end_date` cannot be None."):
      self.coordinates.period_start(None)


class NormalizeDateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="string", date="2024-01-08"),
      dict(testcase_name="date", date=dt.date(2024, 1, 8)),
      dict(testcase_name="datetime", date=dt.datetime(2024, 1, 8, 12, 30)),
      dict(testcase_name="datetime64_day", date=np.datetime64("2024-01-08")),
      dict(
          testcase_name="datetime64_second",
          date=np.datetime64("2024-01-08T12:30:00"),
      ),
      dict(
          testcase_name="datetime64_nanosecond",
          date=np.datetime64("2024-01-08T12:30:00", "ns"),
      ),
  )
  def test_returns_date(self, date):
    normalized = time_coordinates.normalize_date(date)
    self.assertIs(type(normalized), dt.date)
    self.assertEqual(normalized, dt.date(2024, 1, 8))

  def test_none_returns_none(self):
    self.assertIsNone(time_coordinates.normalize_date(None))

  @parameterized.named_parameters(
      dict(testcase_name="integer", date=2),
      dict(testcase_name="wrong_format", date="01/08/2024"),
  )
  def test_invalid_date_raises(self, date):
    with self.assertRaises(ValueError):
      time_coordinates.normalize_date(date)

  def test_normalize_date_interval_returns_dates(self):
    self.assertEqual(
        time_coordinates.normalize_date_interval(("2024-01-01", "2024-01-08")),
        (dt.date(2024, 1, 1), dt.date(2024, 1, 8)),
    )

  @parameterized.named_parameters(
      dict(testcase_name="start_none", interval=(None, "2024-01-08")),
      dict(testcase_name="end_none", interval=("2024-01-01", None)),
  )
  def test_normalize_date_interval_none_raises(self, interval):
    with self.assertRaisesRegex(
        ValueError, "Date interval bounds cannot be None"
    ):
      time_coordinates.normalize_date_interval(interval)


class NormalizeDateStrTest(parameterized.TestCase):

  def test_normalize_date_str_with_datetime64_and_strings(self):
    self.assertEqual(
        time_coordinates.normalize_date_str("2021-01-25T12:34:56"),
        "2021-01-25",
    )
    self.assertEqual(
        time_coordinates.normalize_date_str(np.datetime64("2021-01-25")),
        "2021-01-25",
    )

  def test_normalize_date_str_none_raises(self):
    with self.assertRaisesRegex(ValueError, "`time_val` cannot be None."):
      time_coordinates.normalize_date_str(None)

  def test_normalize_date_str_normalized_date_is_not_parsed(self):
    with mock.patch.object(
        time_coordinates, "normalize_date", autospec=True
    ) as mock_normalize_date:
      self.assertEqual(
          time_coordinates.normalize_date_str("2021-01-25"), "2021-01-25"
      )

    mock_normalize_date.assert_not_called()

  def test_normalize_date_str_pads_month_and_day(self):
    self.assertEqual(
        time_coordinates.normalize_date_str("2021-1-5"), "2021-01-05"
    )

  @parameterized.named_parameters(
      dict(testcase_name="day_out_of_range", date="2021-02-30", regex="day"),
      dict(
          testcase_name="month_out_of_range",
          date="2021-13-01",
          regex="does not match format",
      ),
      dict(
          testcase_name="week_date",
          date="2021-W04-1",
          regex="does not match format",
      ),
  )
  def test_normalize_date_str_invalid_date_raises(self, date: str, regex: str):
    with self.assertRaisesRegex(ValueError, regex):
      time_coordinates.normalize_date_str(date)

  def test_normalize_times_keeps_order(self):
    self.assertEqual(
        time_coordinates.normalize_times(
            ["2021-02-01", "2021-01-25T12:34:56", np.datetime64("2021-01-18")]
        ),
        ["2021-02-01", "2021-01-25", "2021-01-18"],
    )

  def test_normalize_times_data_array(self):
    times = xr.DataArray(
        data=np.array(_ALL_DATES),
        dims=[constants.TIME],
        coords={constants.TIME: _ALL_DATES},
    )
    self.assertEqual(time_coordinates.normalize_times(times), _ALL_DATES)

  def test_normalize_times_set(self):
    self.assertEqual(
        time_coordinates.normalize_times_set(
            ["2021-01-25T12:34:56", np.datetime64("2021-01-25")]
        ),
        {"2021-01-25"},
    )

  def test_normalize_times_set_data_array(self):
    times = xr.DataArray(
        data=np.array(_ALL_DATES),
        dims=[constants.TIME],
        coords={constants.TIME: _ALL_DATES},
    )
    self.assertEqual(
        time_coordinates.normalize_times_set(times), set(_ALL_DATES)
    )


class ValidateTimesTest(absltest.TestCase):

  def test_validate_time_coordinates_invalid_label_raises(self):
    with self.assertRaisesRegex(ValueError, "Invalid time label: '2031-W02'"):
      time_coordinates.validate_time_coordinates(("2031-01-06", "2031-W02"))

  def test_validate_time_coordinates_parses_same_times_once(self):
    time_coordinates._validate_time_spacing.cache_clear()
    time = ("2031-01-06", "2031-01-13", "2031-01-20")
    from_dates = time_coordinates.TimeCoordinates.from_dates
    with mock.patch.object(
        time_coordinates.TimeCoordinates,
        "from_dates",
        autospec=True,
        side_effect=from_dates,
    ) as mock_from_dates:
      time_coordinates.validate_time_coordinates(time)
      time_coordinates.validate_time_coordinates(list(time))

    mock_from_dates.assert_called_once()

  def test_validate_time_coordinates_irregular_spacing_raises_each_time(self):
    time = ("2031-01-06", "2031-01-08", "2031-01-20")
    for _ in range(2):
      with self.assertRaisesRegex(
          ValueError, "Time coordinates must be regularly spaced"
      ):
        time_coordinates.validate_time_coordinates(time)

  def test_validate_selected_times_normalized_subset_passes(self):
    time_coordinates.validate_selected_times(
        ["2024-01-08T00:00:00", "2024-01-15"],
        _ALL_DATES,
        arg_name="selected_times",
    )

  def test_validate_selected_times_non_string_raises(self):
    with self.assertRaisesRegex(
        ValueError, "`selected_times` must be a list of strings."
    ):
      time_coordinates.validate_selected_times(
          [np.datetime64("2024-01-08")],  # pyrefly: ignore[bad-argument-type]
          _ALL_DATES,
          arg_name="selected_times",
      )

  def test_validate_selected_times_not_in_input_times_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "`media_selected_times` must match the time dimension coordinates from"
        " '2024-01-01' to '2024-02-19'.",
    ):
      time_coordinates.validate_selected_times(
          ["2024-02-26"], _ALL_DATES, arg_name="media_selected_times"
      )


if __name__ == "__main__":
  absltest.main()
