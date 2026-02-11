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
from meridian.schema.utils import date_range_bucketing


class MonthlyDateRangeGeneratorTest(absltest.TestCase):

  def test_generate_date_intervals_skips_first_interval_if_not_start_of_month(
      self,
  ):
    input_dates = [
        datetime.date(2023, 1, 15),
        datetime.date(2023, 1, 22),
        datetime.date(2023, 1, 29),
        datetime.date(2023, 2, 6),
        datetime.date(2023, 2, 13),
        datetime.date(2023, 2, 20),
        datetime.date(2023, 2, 27),
        datetime.date(2023, 3, 6),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 2, 6), datetime.date(2023, 3, 6)),
    ]

    date_intervals = list(
        date_range_bucketing.MonthlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)

  def test_generate_date_intervals_skips_last_interval(
      self,
  ):
    input_dates = [
        datetime.date(2023, 1, 1),
        datetime.date(2023, 1, 8),
        datetime.date(2023, 1, 15),
        datetime.date(2023, 1, 22),
        datetime.date(2023, 1, 29),
        datetime.date(2023, 2, 6),
        datetime.date(2023, 2, 13),
        datetime.date(2023, 2, 20),
        datetime.date(2023, 2, 27),
        datetime.date(2023, 3, 6),
        datetime.date(2023, 3, 13),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 1, 1), datetime.date(2023, 2, 6)),
        (datetime.date(2023, 2, 6), datetime.date(2023, 3, 6)),
    ]

    date_intervals = list(
        date_range_bucketing.MonthlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)


class QuarterlyDateRangeGeneratorTest(absltest.TestCase):

  def test_generate_date_intervals_skips_first_interval_if_not_start_of_qtr(
      self,
  ):
    input_dates = [
        datetime.date(2023, 2, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 7, 1),
        datetime.date(2023, 8, 1),
        datetime.date(2023, 9, 1),
        datetime.date(2023, 10, 1),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 4, 1), datetime.date(2023, 7, 1)),
        (datetime.date(2023, 7, 1), datetime.date(2023, 10, 1)),
    ]

    date_intervals = list(
        date_range_bucketing.QuarterlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)

  def test_generate_date_intervals_skips_last_interval(
      self,
  ):
    input_dates = [
        datetime.date(2023, 1, 1),
        datetime.date(2023, 2, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 7, 1),
        datetime.date(2023, 8, 1),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 1, 1), datetime.date(2023, 4, 1)),
        (datetime.date(2023, 4, 1), datetime.date(2023, 7, 1)),
    ]

    date_intervals = list(
        date_range_bucketing.QuarterlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)


class YearlyDateRangeGeneratorTest(absltest.TestCase):

  def test_generate_date_intervals_returns_empty_when_full_year_is_not_covered(
      self,
  ):
    input_dates = [
        datetime.date(2022, 11, 1),
        datetime.date(2022, 12, 1),
        datetime.date(2023, 1, 1),
    ]

    date_intervals = list(
        date_range_bucketing.YearlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertEmpty(date_intervals)

  def test_generate_date_intervals_skips_first_interval_if_not_first_month(
      self,
  ):
    input_dates = [
        datetime.date(2022, 12, 1),
        datetime.date(2023, 1, 1),
        datetime.date(2023, 2, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 7, 1),
        datetime.date(2023, 8, 1),
        datetime.date(2023, 9, 1),
        datetime.date(2023, 10, 1),
        datetime.date(2023, 11, 1),
        datetime.date(2023, 12, 1),
        datetime.date(2024, 1, 1),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 1, 1), datetime.date(2024, 1, 1)),
    ]

    date_intervals = list(
        date_range_bucketing.YearlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)

  def test_generate_date_intervals_skips_last_interval(
      self,
  ):
    input_dates = [
        datetime.date(2023, 1, 1),
        datetime.date(2023, 2, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 7, 1),
        datetime.date(2023, 8, 1),
        datetime.date(2023, 9, 1),
        datetime.date(2023, 10, 1),
        datetime.date(2023, 11, 1),
        datetime.date(2023, 12, 1),
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 1, 1), datetime.date(2024, 1, 1)),
    ]

    date_intervals = list(
        date_range_bucketing.YearlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)

  def test_generate_date_intervals_produces_two_intervals_for_two_full_years(
      self,
  ):
    input_dates = [
        datetime.date(2023, 1, 1),
        datetime.date(2023, 2, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 7, 1),
        datetime.date(2023, 8, 1),
        datetime.date(2023, 9, 1),
        datetime.date(2023, 10, 1),
        datetime.date(2023, 11, 1),
        datetime.date(2023, 12, 1),
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
        datetime.date(2024, 4, 1),
        datetime.date(2024, 5, 1),
        datetime.date(2024, 6, 1),
        datetime.date(2024, 7, 1),
        datetime.date(2024, 8, 1),
        datetime.date(2024, 9, 1),
        datetime.date(2024, 10, 1),
        datetime.date(2024, 11, 1),
        datetime.date(2024, 12, 1),
        datetime.date(2025, 1, 1),
        datetime.date(2025, 2, 1),
    ]
    expected_date_intervals = [
        (datetime.date(2023, 1, 1), datetime.date(2024, 1, 1)),
        (datetime.date(2024, 1, 1), datetime.date(2025, 1, 1)),
    ]

    date_intervals = list(
        date_range_bucketing.YearlyDateRangeGenerator(
            input_dates
        ).generate_date_intervals()
    )

    self.assertSequenceEqual(date_intervals, expected_date_intervals)


if __name__ == "__main__":
  absltest.main()
