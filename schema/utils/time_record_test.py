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

from absl.testing import absltest
from absl.testing import parameterized
from mmm.v1.common import date_interval_pb2
from schema.utils import time_record
import numpy as np
import pandas as pd

from google.type import date_pb2
from tensorflow.python.util.protobuf import compare


class TimeRecordTest(parameterized.TestCase):

  def test_convert_times_to_date_intervals_fewer_than_two_times(self):
    with self.assertRaisesRegex(
        ValueError,
        "There must be at least 2 time points.",
    ):
      time_record.convert_times_to_date_intervals(
          times=["2024-01-01"],
      )

  def test_convert_times_to_date_intervals_enforces_iso_format(self):
    with self.assertRaisesRegex(
        ValueError,
        "",
    ):
      time_record.convert_times_to_date_intervals(
          times=["2024-01-01", "01-08-2024", "15-01-2024"],
      )

  def test_convert_times_to_date_interval_length_is_not_consistent(self):
    with self.assertRaisesRegex(
        ValueError,
        "Interval length between selected times must be consistent.",
    ):
      time_record.convert_times_to_date_intervals(
          times=["2024-01-01", "2024-01-07", "2024-01-15"],
      )

  def test_convert_times_to_date_intervals_creates_date_intervals(self):
    time_to_date_interval = time_record.convert_times_to_date_intervals(
        times=["2024-01-01", "2024-01-08", "2024-01-15"],
    )
    expected_date_interval_1 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=1),
        end_date=date_pb2.Date(year=2024, month=1, day=8),
    )
    expected_date_interval_2 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=8),
        end_date=date_pb2.Date(year=2024, month=1, day=15),
    )
    expected_date_interval_3 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=15),
        end_date=date_pb2.Date(year=2024, month=1, day=22),
    )
    expected_time_to_date_interval = {
        "2024-01-01": expected_date_interval_1,
        "2024-01-08": expected_date_interval_2,
        "2024-01-15": expected_date_interval_3,
    }
    self.assertEqual(time_to_date_interval, expected_time_to_date_interval)

  def test_convert_times_to_date_intervals_datetime_index_input(self):
    time_to_date_interval = time_record.convert_times_to_date_intervals(
        times=pd.DatetimeIndex([
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-08"),
            np.datetime64("2024-01-15"),
        ]),
    )
    expected_date_interval_1 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=1),
        end_date=date_pb2.Date(year=2024, month=1, day=8),
    )
    expected_date_interval_2 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=8),
        end_date=date_pb2.Date(year=2024, month=1, day=15),
    )
    expected_date_interval_3 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=15),
        end_date=date_pb2.Date(year=2024, month=1, day=22),
    )
    expected_time_to_date_interval = {
        "2024-01-01": expected_date_interval_1,
        "2024-01-08": expected_date_interval_2,
        "2024-01-15": expected_date_interval_3,
    }
    self.assertEqual(time_to_date_interval, expected_time_to_date_interval)

  def test_convert_times_to_date_intervals_date_objects_input(self):
    time_to_date_interval = time_record.convert_times_to_date_intervals(
        times=[
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 8),
            dt.date(2024, 1, 15),
        ],
    )
    expected_date_interval_1 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=1),
        end_date=date_pb2.Date(year=2024, month=1, day=8),
    )
    expected_date_interval_2 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=8),
        end_date=date_pb2.Date(year=2024, month=1, day=15),
    )
    expected_date_interval_3 = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=15),
        end_date=date_pb2.Date(year=2024, month=1, day=22),
    )
    expected_time_to_date_interval = {
        "2024-01-01": expected_date_interval_1,
        "2024-01-08": expected_date_interval_2,
        "2024-01-15": expected_date_interval_3,
    }
    self.assertEqual(time_to_date_interval, expected_time_to_date_interval)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_day",
          start_date=dt.datetime(year=2024, month=1, day=1),
          end_date=dt.datetime(year=2024, month=1, day=1),
          tag="",
          expected=date_interval_pb2.DateInterval(
              start_date=date_pb2.Date(year=2024, month=1, day=1),
              end_date=date_pb2.Date(year=2024, month=1, day=1),
          ),
      ),
      dict(
          testcase_name="multiple_days",
          start_date=dt.datetime(year=2024, month=1, day=1),
          end_date=dt.datetime(year=2024, month=1, day=8),
          tag="",
          expected=date_interval_pb2.DateInterval(
              start_date=date_pb2.Date(year=2024, month=1, day=1),
              end_date=date_pb2.Date(year=2024, month=1, day=8),
          ),
      ),
      dict(
          testcase_name="single_day_with_tag",
          start_date=dt.datetime(year=2024, month=1, day=1),
          end_date=dt.datetime(year=2024, month=1, day=1),
          tag="tag",
          expected=date_interval_pb2.DateInterval(
              start_date=date_pb2.Date(year=2024, month=1, day=1),
              end_date=date_pb2.Date(year=2024, month=1, day=1),
              tag="tag",
          ),
      ),
      dict(
          testcase_name="multiple_days_with_tag",
          start_date=dt.datetime(year=2024, month=1, day=1),
          end_date=dt.datetime(year=2024, month=1, day=8),
          tag="tag",
          expected=date_interval_pb2.DateInterval(
              start_date=date_pb2.Date(year=2024, month=1, day=1),
              end_date=date_pb2.Date(year=2024, month=1, day=8),
              tag="tag",
          ),
      ),
  )
  def test_create_date_interval_pb(self, start_date, end_date, tag, expected):
    actual = time_record.create_date_interval_pb(
        start_date=start_date, end_date=end_date, tag=tag
    )
    compare.assertProtoEqual(self, actual, expected)

  def test_dates_from_date_interval_proto(self):
    date_interval = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=1, day=1),
        end_date=date_pb2.Date(year=2025, month=2, day=8),
    )
    start_date, end_date = time_record.dates_from_date_interval_proto(
        date_interval
    )
    self.assertEqual(start_date, dt.date(year=2024, month=1, day=1))
    self.assertEqual(end_date, dt.date(year=2025, month=2, day=8))


if __name__ == "__main__":
  absltest.main()
