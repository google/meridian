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
from lookerstudio.converters import constants as cc
from lookerstudio.converters import mmm
from lookerstudio.converters import test_data as td
from lookerstudio.converters.dataframe import budget_opt_converters as converters
from lookerstudio.converters.dataframe import constants as dc
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import marketing_optimization_pb2 as optimization_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
import pandas as pd


mock = absltest.mock


_DEFAULT_MMM_PROTO = mmm_pb.Mmm(
    mmm_kernel=kernel_pb.MmmKernel(
        marketing_data=td.MARKETING_DATA,
    ),
    marketing_optimization=optimization_pb.MarketingOptimization(
        budget_optimization=budget_pb.BudgetOptimization(
            results=[
                td.BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES,
                td.BUDGET_OPTIMIZATION_RESULT_FLEX_NONREV,
            ]
        ),
    ),
)

_GID1 = td.BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES.group_id
_GID2 = td.BUDGET_OPTIMIZATION_RESULT_FLEX_NONREV.group_id


class NamedOptimizationGridConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.NamedOptimizationGridConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.NamedOptimizationGridConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    dataframes = list(conv())

    self.assertLen(dataframes, 2)
    (foo_grid_name, foo_grid_df), (bar_grid_name, bar_grid_df) = dataframes

    expected_foo_grid_name = "_".join(
        [dc.OPTIMIZATION_GRID_NAME_PREFIX, "incremental_outcome_grid_foo"]
    )
    expected_bar_grid_name = "_".join(
        [dc.OPTIMIZATION_GRID_NAME_PREFIX, "incremental_outcome_grid_bar"]
    )

    self.assertEqual(foo_grid_name, expected_foo_grid_name)
    self.assertEqual(bar_grid_name, expected_bar_grid_name)

    expected_columns = [
        dc.OPTIMIZATION_GROUP_ID_COLUMN,
        dc.OPTIMIZATION_CHANNEL_COLUMN,
        dc.OPTIMIZATION_GRID_SPEND_COLUMN,
        dc.OPTIMIZATION_GRID_INCREMENTAL_OUTCOME_COLUMN,
    ]
    pd.testing.assert_frame_equal(
        foo_grid_df,
        pd.DataFrame(
            [
                [
                    _GID1,
                    "Channel 1",
                    10000.0,
                    100.0,
                ],
                [
                    _GID1,
                    "Channel 1",
                    20000.0,
                    200.0,
                ],
                [
                    _GID1,
                    "Channel 2",
                    10000.0,
                    100.0,
                ],
                [
                    _GID1,
                    "Channel 2",
                    20000.0,
                    200.0,
                ],
            ],
            columns=expected_columns,
        ),
    )
    pd.testing.assert_frame_equal(
        bar_grid_df,
        pd.DataFrame(
            [
                [
                    _GID2,
                    "Channel 1",
                    1000.0,
                    10.0,
                ],
                [
                    _GID2,
                    "Channel 1",
                    2000.0,
                    20.0,
                ],
                [
                    _GID2,
                    "Channel 2",
                    1000.0,
                    10.0,
                ],
                [
                    _GID2,
                    "Channel 2",
                    2000.0,
                    20.0,
                ],
            ],
            columns=expected_columns,
        ),
    )


class BudgetOptimizationSpecsConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.BudgetOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.BudgetOptimizationSpecsConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.OPTIMIZATION_SPECS)
    # Expect two specs: one for each result.
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                # These corresponds to the fixed spec FOO:
                [
                    _GID1,
                    "2024-01-01",
                    "2024-01-15",
                    cc.ANALYSIS_TAG_ALL,
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_ROI,
                    dc.OPTIMIZATION_SPEC_SCENARIO_FIXED,
                    400.0,
                    None,
                    None,
                    "Channel 1",
                    0.0,
                    100000.0,
                ],
                [
                    _GID1,
                    "2024-01-01",
                    "2024-01-15",
                    cc.ANALYSIS_TAG_ALL,
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_ROI,
                    dc.OPTIMIZATION_SPEC_SCENARIO_FIXED,
                    400.0,
                    None,
                    None,
                    "Channel 2",
                    0.0,
                    100000.0,
                ],
                # These corresponds to the flexible spec BAR:
                [
                    _GID2,
                    "2024-01-08",
                    "2024-01-15",
                    "Week2",
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_KPI,
                    dc.OPTIMIZATION_SPEC_SCENARIO_FLEXIBLE,
                    200.0,
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_CPIK,
                    10.0,
                    "Channel 1",
                    1100.0,
                    1500.0,
                ],
                [
                    _GID2,
                    "2024-01-08",
                    "2024-01-15",
                    "Week2",
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_KPI,
                    dc.OPTIMIZATION_SPEC_SCENARIO_FLEXIBLE,
                    200.0,
                    dc.OPTIMIZATION_SPEC_TARGET_METRIC_CPIK,
                    10.0,
                    "Channel 2",
                    1000.0,
                    1800.0,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_SPEC_DATE_INTERVAL_START_COLUMN,
                dc.OPTIMIZATION_SPEC_DATE_INTERVAL_END_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.OPTIMIZATION_SPEC_OBJECTIVE_COLUMN,
                dc.OPTIMIZATION_SPEC_SCENARIO_TYPE_COLUMN,
                dc.OPTIMIZATION_SPEC_INITIAL_CHANNEL_SPEND_COLUMN,
                dc.OPTIMIZATION_SPEC_TARGET_METRIC_CONSTRAINT_COLUMN,
                dc.OPTIMIZATION_SPEC_TARGET_METRIC_VALUE_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_SPEC_CHANNEL_SPEND_MIN_COLUMN,
                dc.OPTIMIZATION_SPEC_CHANNEL_SPEND_MAX_COLUMN,
            ],
        ),
    )


class BudgetOptimizationResultsConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.BudgetOptimizationResultsConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call_duplicate_group_id(self):
    mmm_proto = mmm_pb.Mmm()
    mmm_proto.CopyFrom(_DEFAULT_MMM_PROTO)
    mmm_proto.marketing_optimization.budget_optimization.results.append(
        td.BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES,
    )

    with self.assertRaisesRegex(
        ValueError, "Specified group_id must be unique"
    ):
      conv = converters.BudgetOptimizationResultsConverter(
          mmm_wrapper=mmm.Mmm(mmm_proto)
      )
      next(conv())

  def test_call(self):
    conv = converters.BudgetOptimizationResultsConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.OPTIMIZATION_RESULTS)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [
                    _GID1,
                    "Channel 1",
                    True,
                    75000.0,
                    0.5,
                    2.2,
                    1.0,
                    10.0,
                    5.0,
                ],
                [
                    _GID1,
                    "Channel 1",
                    False,
                    75000.0,
                    0.5,
                    5.5,
                    10.0,
                    100.0,
                    100.0,
                ],
                [
                    _GID1,
                    "Channel 2",
                    True,
                    25000.0,
                    (1.0 / 6.0),
                    4.4,
                    2.0,
                    20.0,
                    10.0,
                ],
                [
                    _GID1,
                    "Channel 2",
                    False,
                    25000.0,
                    (1.0 / 6.0),
                    11.0,
                    20.0,
                    200.0,
                    200.0,
                ],
                [
                    _GID1,
                    "RF Channel 1",
                    True,
                    30000.0,
                    0.2,
                    2.2,
                    1.0,
                    10.0,
                    5.0,
                ],
                [
                    _GID1,
                    "RF Channel 1",
                    False,
                    30000.0,
                    0.2,
                    5.5,
                    10.0,
                    100.0,
                    100.0,
                ],
                [
                    _GID1,
                    "RF Channel 2",
                    True,
                    20000.0,
                    (2.0 / 15.0),
                    4.4,
                    2.0,
                    20.0,
                    10.0,
                ],
                [
                    _GID1,
                    "RF Channel 2",
                    False,
                    20000.0,
                    (2.0 / 15.0),
                    11.0,
                    20.0,
                    200.0,
                    200.0,
                ],
                [
                    _GID2,
                    "Channel 1",
                    False,
                    75000.0,
                    0.5,
                    6.6,
                    12.0,
                    120.0,
                    120.0,
                ],
                [
                    _GID2,
                    "Channel 2",
                    False,
                    25000.0,
                    (1.0 / 6.0),
                    12.1,
                    22.0,
                    220.0,
                    220.0,
                ],
                [
                    _GID2,
                    "RF Channel 1",
                    False,
                    30000.0,
                    0.2,
                    6.6,
                    12.0,
                    120.0,
                    120.0,
                ],
                [
                    _GID2,
                    "RF Channel 2",
                    False,
                    20000.0,
                    (2.0 / 15.0),
                    12.1,
                    22.0,
                    220.0,
                    220.0,
                ],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_RESULT_IS_REVENUE_KPI_COLUMN,
                dc.OPTIMIZATION_RESULT_SPEND_COLUMN,
                dc.OPTIMIZATION_RESULT_SPEND_SHARE_COLUMN,
                dc.OPTIMIZATION_RESULT_EFFECTIVENESS_COLUMN,
                dc.OPTIMIZATION_RESULT_ROI_COLUMN,
                dc.OPTIMIZATION_RESULT_MROI_COLUMN,
                dc.OPTIMIZATION_RESULT_CPC_COLUMN,
            ],
        ),
    )


class BudgetOptimizationResponseCurvesConverterTest(absltest.TestCase):

  def test_call_no_results(self):
    conv = converters.BudgetOptimizationResponseCurvesConverter(
        mmm_wrapper=mmm.Mmm(mmm_pb.Mmm())
    )

    self.assertEmpty(list(conv()))

  def test_call(self):
    conv = converters.BudgetOptimizationResponseCurvesConverter(
        mmm_wrapper=mmm.Mmm(_DEFAULT_MMM_PROTO)
    )

    name, output_df = next(conv())

    self.assertEqual(name, dc.OPTIMIZATION_RESPONSE_CURVES)
    pd.testing.assert_frame_equal(
        output_df,
        pd.DataFrame(
            [
                [_GID1, "Channel 1", 1.0, 100.0],
                [_GID1, "Channel 1", 2.0, 200.0],
                [_GID1, "Channel 2", 2.0, 200.0],
                [_GID1, "Channel 2", 4.0, 400.0],
                [_GID1, "RF Channel 1", 1.0, 100.0],
                [_GID1, "RF Channel 1", 2.0, 200.0],
                [_GID1, "RF Channel 2", 2.0, 200.0],
                [_GID1, "RF Channel 2", 4.0, 400.0],
                [_GID1, "All Channels", 10.0, 1000.0],
                [_GID1, "All Channels", 20.0, 2000.0],
                [_GID2, "Channel 1", 1.2, 120.0],
                [_GID2, "Channel 1", 2.4, 240.0],
                [_GID2, "Channel 2", 2.2, 220.0],
                [_GID2, "Channel 2", 4.4, 440.0],
                [_GID2, "RF Channel 1", 1.2, 120.0],
                [_GID2, "RF Channel 1", 2.4, 240.0],
                [_GID2, "RF Channel 2", 2.2, 220.0],
                [_GID2, "RF Channel 2", 4.4, 440.0],
                [_GID2, "All Channels", 10.0, 1000.0],
                [_GID2, "All Channels", 20.0, 2000.0],
            ],
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_GRID_SPEND_COLUMN,
                dc.OPTIMIZATION_GRID_INCREMENTAL_OUTCOME_COLUMN,
            ],
        ),
    )


if __name__ == "__main__":
  absltest.main()
