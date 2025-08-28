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

from absl.testing import absltest
from lookerstudio.converters import mmm
from lookerstudio.converters import test_data as td
from lookerstudio.converters.dataframe import constants as dc
from lookerstudio.converters.dataframe import rf_opt_converters as converters
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.marketing.optimization import marketing_optimization_pb2 as optimization_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
import pandas as pd


mock = absltest.mock


_DEFAULT_MMM_PROTO = mmm_pb.Mmm(
    mmm_kernel=kernel_pb.MmmKernel(
        marketing_data=td.MARKETING_DATA,
    ),
    marketing_optimization=optimization_pb.MarketingOptimization(
        reach_frequency_optimization=rf_pb.ReachFrequencyOptimization(
            results=[
                td.RF_OPTIMIZATION_RESULT_FOO,
            ]
        ),
    ),
)

_GID = td.RF_OPTIMIZATION_RESULT_FOO.group_id


class NamedRfOptimizationGridConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.NamedRfOptimizationGridConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.NamedRfOptimizationGridConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    dataframes = list(conv())

    self.assertLen(dataframes, 1)
    foo_grid_name, foo_grid_df = dataframes[0]

    expected_foo_grid_name = "_".join(
        [dc.RF_OPTIMIZATION_GRID_NAME_PREFIX, "frequency_outcome_grid_foo"]
    )

    self.assertEqual(foo_grid_name, expected_foo_grid_name)

    pd.testing.assert_frame_equal(
        foo_grid_df,
        pd.DataFrame(
            [
                [
                    _GID,
                    "RF Channel 1",
                    1.0,
                    100.0,
                ],
                [
                    _GID,
                    "RF Channel 1",
                    2.0,
                    200.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    1.0,
                    100.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    2.0,
                    200.0,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.RF_OPTIMIZATION_GRID_FREQ_COLUMN,
                dc.RF_OPTIMIZATION_GRID_ROI_OUTCOME_COLUMN,
            ],
        ),
    )


class RfOptimizationSpecsConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.RfOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.RfOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.RF_OPTIMIZATION_SPECS)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    _GID,
                    "2024-01-01",
                    "2024-01-15",
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_KPI,
                    440.0,
                    "RF Channel 1",
                    1.0,
                    5.0,
                ],
                [
                    _GID,
                    "2024-01-01",
                    "2024-01-15",
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_KPI,
                    440.0,
                    "RF Channel 2",
                    1.3,
                    6.6,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_SPEC_DATE_INTERVAL_START_COLUMN,
                dc.OPTIMIZATION_SPEC_DATE_INTERVAL_END_COLUMN,
                dc.OPTIMIZATION_SPEC_OBJECTIVE_COLUMN,
                dc.OPTIMIZATION_SPEC_INITIAL_CHANNEL_SPEND_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.RF_OPTIMIZATION_SPEC_CHANNEL_FREQUENCY_MIN_COLUMN,
                dc.RF_OPTIMIZATION_SPEC_CHANNEL_FREQUENCY_MAX_COLUMN,
            ],
        ),
    )

  def test_call_no_rf_channel_constraints(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    mmm_proto.marketing_optimization.reach_frequency_optimization.results[
        0
    ].spec.ClearField("rf_channel_constraints")

    conv = converters.RfOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto)
    )

    with self.assertRaisesRegex(
        ValueError,
        "R&F optimization spec must have channel constraints specified.",
    ):
      next(conv())

  def test_call_missing_an_rf_channel_constraint(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    mmm_proto.marketing_optimization.reach_frequency_optimization.results[
        0
    ].spec.rf_channel_constraints.pop()

    conv = converters.RfOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto)
    )

    with self.assertRaisesRegex(
        ValueError,
        "R&F optimization spec must have channel constraints specified for all"
        " R&F channels.",
    ):
      next(conv())

  def test_call_missing_max_frequency_constraint(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    mmm_proto.marketing_optimization.reach_frequency_optimization.results[
        0
    ].spec.rf_channel_constraints[1].frequency_constraint.ClearField(
        "max_frequency"
    )

    conv = converters.RfOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto)
    )

    with self.assertRaisesRegex(
        ValueError,
        "Channel constraint in R&F optimization spec must have max frequency"
        " specified. Missing for channel: RF Channel 2",
    ):
      next(conv())


class RfOptimizationResultsConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.RfOptimizationResultsConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call_duplicate_group_id(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    mmm_proto.marketing_optimization.reach_frequency_optimization.results.append(
        td.RF_OPTIMIZATION_RESULT_FOO
    )

    with self.assertRaisesRegex(
        ValueError, "Specified group_id must be unique"
    ):
      conv = converters.RfOptimizationResultsConverter(
          mmm_wrapper=mmm.Mmm(mmm_proto)
      )
      next(conv())

  def test_call(self):
    conv = converters.RfOptimizationResultsConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.RF_OPTIMIZATION_RESULTS)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    _GID,
                    "RF Channel 1",
                    True,
                    440.0,
                    3.3,
                    2.2,
                    1.0,
                    10.0,
                    5.0,
                ],
                [
                    _GID,
                    "RF Channel 1",
                    False,
                    440.0,
                    3.3,
                    5.5,
                    10.0,
                    100.0,
                    100.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    True,
                    440.0,
                    5.6,
                    4.4,
                    2.0,
                    20.0,
                    10.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    False,
                    440.0,
                    5.6,
                    11.0,
                    20.0,
                    200.0,
                    200.0,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_RESULT_IS_REVENUE_KPI_COLUMN,
                dc.RF_OPTIMIZATION_RESULT_INITIAL_SPEND_COLUMN,
                dc.RF_OPTIMIZATION_RESULT_AVG_FREQ_COLUMN,
                dc.OPTIMIZATION_RESULT_EFFECTIVENESS_COLUMN,
                dc.OPTIMIZATION_RESULT_ROI_COLUMN,
                dc.OPTIMIZATION_RESULT_MROI_COLUMN,
                dc.OPTIMIZATION_RESULT_CPC_COLUMN,
            ],
        ),
    )

  def test_call_no_revenue_baseline_outcome(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    # Remove revenue-type outcomes from baseline analyses.
    rf_opt = mmm_proto.marketing_optimization.reach_frequency_optimization
    for rf_result in rf_opt.results:
      opt_marketing_analysis = rf_result.optimized_marketing_analysis
      for non_media_analysis in opt_marketing_analysis.non_media_analyses:
        if non_media_analysis.non_media_name != "baseline":
          continue
        revenue_outcome_index = None
        for i, outcome in enumerate(non_media_analysis.non_media_outcomes):
          if outcome.kpi_type == kpi_type_pb.REVENUE:
            revenue_outcome_index = i
            break
        if revenue_outcome_index is not None:
          non_media_analysis.non_media_outcomes.pop(revenue_outcome_index)

    conv = converters.RfOptimizationResultsConverter(
        mmm_wrapper=mmm.Mmm(mmm_proto)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.RF_OPTIMIZATION_RESULTS)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    _GID,
                    "RF Channel 1",
                    True,
                    440.0,
                    3.3,
                    2.2,
                    1.0,
                    10.0,
                    5.0,
                ],
                [
                    _GID,
                    "RF Channel 1",
                    False,
                    440.0,
                    3.3,
                    5.5,
                    10.0,
                    100.0,
                    100.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    True,
                    440.0,
                    5.6,
                    4.4,
                    2.0,
                    20.0,
                    10.0,
                ],
                [
                    _GID,
                    "RF Channel 2",
                    False,
                    440.0,
                    5.6,
                    11.0,
                    20.0,
                    200.0,
                    200.0,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_RESULT_IS_REVENUE_KPI_COLUMN,
                dc.RF_OPTIMIZATION_RESULT_INITIAL_SPEND_COLUMN,
                dc.RF_OPTIMIZATION_RESULT_AVG_FREQ_COLUMN,
                dc.OPTIMIZATION_RESULT_EFFECTIVENESS_COLUMN,
                dc.OPTIMIZATION_RESULT_ROI_COLUMN,
                dc.OPTIMIZATION_RESULT_MROI_COLUMN,
                dc.OPTIMIZATION_RESULT_CPC_COLUMN,
            ],
        ),
    )


if __name__ == "__main__":
  absltest.main()
