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

import datetime

from absl.testing import absltest
from meridian import constants as c
from lookerstudio.converters import mmm
from lookerstudio.converters import test_data as td
from lookerstudio.converters.dataframe import constants as dc
from lookerstudio.converters.dataframe import marketing_analyses_converters as converters
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
import pandas as pd


_DEFAULT_MMM_PROTO = mmm_pb.Mmm(
    model_fit=fit_pb.ModelFit(
        results=[
            td.MODEL_FIT_RESULT_TRAIN,
            td.MODEL_FIT_RESULT_TEST,
            td.MODEL_FIT_RESULT_ALL_DATA,
        ]
    ),
    marketing_analysis_list=td.MARKETING_ANALYSIS_LIST_BOTH_OUTCOMES,
)

_NONREVENUE_MMM_PROTO = mmm_pb.Mmm(
    model_fit=fit_pb.ModelFit(
        results=[td.MODEL_FIT_RESULT_TRAIN, td.MODEL_FIT_RESULT_TEST]
    ),
    marketing_analysis_list=td.MARKETING_ANALYSIS_LIST_NONREVENUE,
)


class ModelDiagnosticsConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.ModelDiagnosticsConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.ModelDiagnosticsConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MODEL_DIAGNOSTICS)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    c.TRAIN,
                    0.91,
                    60.6,
                    55.5,
                ],
                [
                    c.TEST,
                    0.99,
                    67.7,
                    59.8,
                ],
                [
                    c.ALL_DATA,
                    0.94,
                    60.0,
                    55.4,
                ],
            ],
            columns=[
                dc.MODEL_DIAGNOSTICS_DATASET_COLUMN,
                dc.MODEL_DIAGNOSTICS_R_SQUARED_COLUMN,
                dc.MODEL_DIAGNOSTICS_MAPE_COLUMN,
                dc.MODEL_DIAGNOSTICS_WMAPE_COLUMN,
            ],
        ),
    )


class ModelFitConverterTest(absltest.TestCase):

  def test_call_missing_result(self):
    conv = converters.ModelFitConverter(
        mmm_wrapper=mmm.Mmm(
            mmm_proto=mmm_pb.Mmm(model_fit=fit_pb.ModelFit(results=[]))
        )
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.ModelFitConverter(mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO))

    name, output_df = next(conv())

    self.assertEqual(name, dc.MODEL_FIT)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    "2024-01-01",
                    90.0,
                    110.0,
                    100.0,
                    90.0,
                    105.0,
                ],
                [
                    "2024-01-08",
                    100.0,
                    120.0,
                    110.0,
                    109.0,
                    115.0,
                ],
            ],
            columns=[
                dc.MODEL_FIT_TIME_COLUMN,
                dc.MODEL_FIT_EXPECTED_CI_LOW_COLUMN,
                dc.MODEL_FIT_EXPECTED_CI_HIGH_COLUMN,
                dc.MODEL_FIT_EXPECTED_COLUMN,
                dc.MODEL_FIT_BASELINE_COLUMN,
                dc.MODEL_FIT_ACTUAL_COLUMN,
            ],
        ),
    )

  def test_model_fit_result_no_all_data_result(self):
    conv = converters.ModelFitConverter(
        mmm_wrapper=mmm.Mmm(
            mmm_proto=mmm_pb.Mmm(
                model_fit=fit_pb.ModelFit(
                    results=[
                        td.MODEL_FIT_RESULT_TRAIN,
                        td.MODEL_FIT_RESULT_TEST,
                    ]
                )
            )
        )
    )

    with self.assertWarnsRegex(
        UserWarning,
        expected_regex="Using a model fit `Result` with name: 'Train'",
    ):
      _ = next(conv())


class MediaOutcomeConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.MediaOutcomeConverter(mmm_wrapper=mmm.Mmm(mmm_pb.Mmm()))

    self.assertEmpty(list(conv()))

  def test_call_revenue_kpi(self):
    conv = converters.MediaOutcomeConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_OUTCOME)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          0,
          c.BASELINE,
          50.0,
          0.05,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "Channel 1",
          100.0,
          0.1,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "Channel 2",
          200.0,
          0.2,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "RF Channel 1",
          100.0,
          0.1,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "RF Channel 2",
          200.0,
          0.2,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          2,
          c.ALL_CHANNELS,
          1000.0,
          1.0,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_OUTCOME_CHANNEL_INDEX_COLUMN,
                dc.MEDIA_OUTCOME_CHANNEL_COLUMN,
                dc.MEDIA_OUTCOME_INCREMENTAL_OUTCOME_COLUMN,
                dc.MEDIA_OUTCOME_CONTRIBUTION_SHARE_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )

  def test_call_nonrevenue_kpi(self):
    conv = converters.MediaOutcomeConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto=_NONREVENUE_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_OUTCOME)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          0,
          c.BASELINE,
          40.0,
          0.04,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "Channel 1",
          120.0,
          0.12,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "Channel 2",
          220.0,
          0.22,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "RF Channel 1",
          120.0,
          0.12,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          1,
          "RF Channel 2",
          220.0,
          0.22,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          2,
          c.ALL_CHANNELS,
          1000.0,
          1.0,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_OUTCOME_CHANNEL_INDEX_COLUMN,
                dc.MEDIA_OUTCOME_CHANNEL_COLUMN,
                dc.MEDIA_OUTCOME_INCREMENTAL_OUTCOME_COLUMN,
                dc.MEDIA_OUTCOME_CONTRIBUTION_SHARE_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


class MediaSpendConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.MediaSpendConverter(mmm_wrapper=mmm.Mmm(mmm_pb.Mmm()))

    self.assertEmpty(list(conv()))

  def test_call_revenue_kpi(self):
    conv = converters.MediaSpendConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_SPEND)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          "Channel 1",
          0.5,
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 1",
          0.1 / (0.1 + 0.2 + 0.1 + 0.2),
          dc.MEDIA_SPEND_LABEL_REVENUE_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          (1.0 / 6.0),
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          0.2 / (0.1 + 0.2 + 0.1 + 0.2),
          dc.MEDIA_SPEND_LABEL_REVENUE_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          0.2,
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          0.1 / (0.1 + 0.2 + 0.1 + 0.2),
          dc.MEDIA_SPEND_LABEL_REVENUE_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          (2.0 / 15.0),
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          0.2 / (0.1 + 0.2 + 0.1 + 0.2),
          dc.MEDIA_SPEND_LABEL_REVENUE_SHARE,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_SPEND_CHANNEL_COLUMN,
                dc.MEDIA_SPEND_SHARE_VALUE_COLUMN,
                dc.MEDIA_SPEND_LABEL_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )

  def test_call_nonrevenue_kpi(self):
    conv = converters.MediaSpendConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto=_NONREVENUE_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_SPEND)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          "Channel 1",
          0.5,
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 1",
          0.12 / (0.12 + 0.22 + 0.12 + 0.22),
          dc.MEDIA_SPEND_LABEL_KPI_SHARE,  # NOT "revenue"!
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          (1.0 / 6.0),
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          0.22 / (0.12 + 0.22 + 0.12 + 0.22),
          dc.MEDIA_SPEND_LABEL_KPI_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          0.2,
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          0.12 / (0.12 + 0.22 + 0.12 + 0.22),
          dc.MEDIA_SPEND_LABEL_KPI_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          (2.0 / 15.0),
          dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          0.22 / (0.12 + 0.22 + 0.12 + 0.22),
          dc.MEDIA_SPEND_LABEL_KPI_SHARE,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_SPEND_CHANNEL_COLUMN,
                dc.MEDIA_SPEND_SHARE_VALUE_COLUMN,
                dc.MEDIA_SPEND_LABEL_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


class MediaRoiConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.MediaRoiConverter(mmm_wrapper=mmm.Mmm(mmm_pb.Mmm()))

    self.assertEmpty(list(conv()))

  def test_call_revenue_kpi(self):
    conv = converters.MediaRoiConverter(mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO))

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_ROI)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          "Channel 1",
          75_000.0,
          2.2,
          1.0,
          0.9,
          1.1,
          10.0,
          True,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          25_000.0,
          4.4,
          2.0,
          1.8,
          2.2,
          20.0,
          True,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          30_000.0,
          2.2,
          1.0,
          0.9,
          1.1,
          10.0,
          True,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          20_000.0,
          4.4,
          2.0,
          1.8,
          2.2,
          20.0,
          True,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_ROI_CHANNEL_COLUMN,
                dc.MEDIA_ROI_SPEND_COLUMN,
                dc.MEDIA_ROI_EFFECTIVENESS_COLUMN,
                dc.MEDIA_ROI_ROI_COLUMN,
                dc.MEDIA_ROI_ROI_CI_LOW_COLUMN,
                dc.MEDIA_ROI_ROI_CI_HIGH_COLUMN,
                dc.MEDIA_ROI_MARGINAL_ROI_COLUMN,
                dc.MEDIA_ROI_IS_REVENUE_KPI_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )

  def test_call_nonrevenue_kpi(self):
    conv = converters.MediaRoiConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto=_NONREVENUE_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.MEDIA_ROI)

    expected_data_frame_data = []
    for date_interval in [td.ALL_DATE_INTERVAL] + td.DATE_INTERVALS:
      date_start = datetime.datetime(
          year=date_interval.start_date.year,
          month=date_interval.start_date.month,
          day=date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)
      date_end = datetime.datetime(
          year=date_interval.end_date.year,
          month=date_interval.end_date.month,
          day=date_interval.end_date.day,
      ).strftime(c.DATE_FORMAT)
      tag = date_interval.tag

      expected_data_frame_data.append([
          "Channel 1",
          75_000.0,
          6.6,
          12.0,
          10.8,
          13.2,
          120.0,
          False,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "Channel 2",
          25_000.0,
          12.1,
          22.0,
          19.8,
          24.2,
          220.0,
          False,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 1",
          30_000.0,
          6.6,
          12.0,
          10.8,
          13.2,
          120.0,
          False,
          tag,
          date_start,
          date_end,
      ])
      expected_data_frame_data.append([
          "RF Channel 2",
          20_000.0,
          12.1,
          22.0,
          19.8,
          24.2,
          220.0,
          False,
          tag,
          date_start,
          date_end,
      ])

    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            expected_data_frame_data,
            columns=[
                dc.MEDIA_ROI_CHANNEL_COLUMN,
                dc.MEDIA_ROI_SPEND_COLUMN,
                dc.MEDIA_ROI_EFFECTIVENESS_COLUMN,
                dc.MEDIA_ROI_ROI_COLUMN,
                dc.MEDIA_ROI_ROI_CI_LOW_COLUMN,
                dc.MEDIA_ROI_ROI_CI_HIGH_COLUMN,
                dc.MEDIA_ROI_MARGINAL_ROI_COLUMN,
                dc.MEDIA_ROI_IS_REVENUE_KPI_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


if __name__ == "__main__":
  absltest.main()
