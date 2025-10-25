# Copyright 2025 The Meridian Authors.
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

"""Helper classes for generating date intervals for various time buckets."""

import abc
from collections.abc import Iterator, Sequence
import datetime
from typing import TypeAlias


__all__ = [
    "DateRangeBucketer",
    "MonthlyDateRangeGenerator",
    "QuarterlyDateRangeGenerator",
    "YearlyDateRangeGenerator",
]


DateInterval: TypeAlias = tuple[datetime.date, datetime.date]


class DateRangeBucketer(abc.ABC):
  """Generates `DateInterval` protos over a range of dates."""

  def __init__(
      self,
      input_dates: Sequence[datetime.date],
  ):
    """Initializes the DateRangeBucketer with a sequence of dates.

    Args:
      input_dates: A sequence of `datetime.date` objects representing the range
        of dates to generate intervals for.
    """
    if not all(
        input_dates[i] < input_dates[i + 1] for i in range(len(input_dates) - 1)
    ):
      raise ValueError("`input_dates` must be strictly ascending dates.")

    self._input_dates = input_dates

  @abc.abstractmethod
  def generate_date_intervals(self) -> Iterator[DateInterval]:
    """Generates `DateInterval` protos for the class's input dates.

    Each interval represents a month, quarter, or year, depending on the
    instance of this class. An interval is excluded if the start date is not the
    first available date (in `self._input_dates`) for the time bucket. The last
    interval in `self._input_dates` is excluded in all cases.

    Returns:
      An iterator over generated `TimeInterval`s.
    """
    raise NotImplementedError()


class MonthlyDateRangeGenerator(DateRangeBucketer):
  """Generates monthly date intervals."""

  def generate_date_intervals(self) -> Iterator[DateInterval]:
    start_date = self._input_dates[0]

    for date in self._input_dates:
      if date.month != start_date.month:
        if start_date.day <= 7:
          yield (start_date, date)

        start_date = date


class QuarterlyDateRangeGenerator(DateRangeBucketer):
  """Generates quarterly date intervals."""

  def generate_date_intervals(self) -> Iterator[DateInterval]:
    start_date = self._input_dates[0]
    for date in self._input_dates:
      start_date_quarter_number = (start_date.month - 1) // 3 + 1
      current_date_quarter_number = (date.month - 1) // 3 + 1

      if start_date_quarter_number != current_date_quarter_number:
        # The interval is only included if the start date is the first date of
        # the quarter that's present in `self._input_dates`. We can detect this
        # date by checking whether it's in the first month of the quarter and
        # falls in the first seven days of the month.
        if (
            start_date.day <= 7
            and start_date.month == ((start_date_quarter_number - 1) * 3) + 1
        ):
          yield (start_date, date)

        start_date = date


class YearlyDateRangeGenerator(DateRangeBucketer):
  """Generates yearly date intervals."""

  def generate_date_intervals(self) -> Iterator[DateInterval]:
    start_date = self._input_dates[0]

    for date in self._input_dates:
      if date.year != start_date.year:
        if start_date.day <= 7 and start_date.month == 1:
          yield (start_date, date)

        start_date = date
