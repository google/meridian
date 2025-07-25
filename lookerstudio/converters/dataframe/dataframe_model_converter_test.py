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
from lookerstudio.converters import test_data as td
from lookerstudio.converters.dataframe import constants as dc
from lookerstudio.converters.dataframe import dataframe_model_converter as converter
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import marketing_optimization_pb2 as optimization_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb


_DEFAULT_MMM_PROTO = mmm_pb.Mmm(
    mmm_kernel=kernel_pb.MmmKernel(
        marketing_data=td.MARKETING_DATA,
    ),
    model_fit=fit_pb.ModelFit(
        results=[
            td.MODEL_FIT_RESULT_TRAIN,
            td.MODEL_FIT_RESULT_TEST,
            td.MODEL_FIT_RESULT_ALL_DATA,
        ]
    ),
    marketing_analysis_list=td.MARKETING_ANALYSIS_LIST_BOTH_OUTCOMES,
    marketing_optimization=optimization_pb.MarketingOptimization(
        budget_optimization=budget_pb.BudgetOptimization(
            results=[
                td.BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES,
                td.BUDGET_OPTIMIZATION_RESULT_FLEX_NONREV,
            ]
        ),
        reach_frequency_optimization=rf_pb.ReachFrequencyOptimization(
            results=[
                td.RF_OPTIMIZATION_RESULT_FOO,
            ]
        ),
    ),
)


class DataFrameModelConverterTest(absltest.TestCase):

  def test_call(self):
    conv = converter.DataFrameModelConverter(mmm_proto=_DEFAULT_MMM_PROTO)

    output = conv()

    expected_budget_opt_grid_name1 = "_".join(
        [dc.OPTIMIZATION_GRID_NAME_PREFIX, "incremental_outcome_grid_foo"]
    )

    expected_budget_opt_grid_name2 = "_".join(
        [dc.OPTIMIZATION_GRID_NAME_PREFIX, "incremental_outcome_grid_bar"]
    )

    expected_rf_opt_grid_name = "_".join(
        [dc.RF_OPTIMIZATION_GRID_NAME_PREFIX, "frequency_outcome_grid_foo"]
    )

    for expected_table_name in [
        dc.MODEL_DIAGNOSTICS,
        dc.MODEL_FIT,
        dc.MEDIA_OUTCOME,
        dc.MEDIA_SPEND,
        dc.MEDIA_ROI,
        expected_budget_opt_grid_name1,
        expected_budget_opt_grid_name2,
        dc.OPTIMIZATION_SPECS,
        dc.OPTIMIZATION_RESULTS,
        dc.OPTIMIZATION_RESPONSE_CURVES,
        expected_rf_opt_grid_name,
        dc.RF_OPTIMIZATION_SPECS,
        dc.RF_OPTIMIZATION_RESULTS,
    ]:
      self.assertIn(expected_table_name, output.keys())


if __name__ == "__main__":
  absltest.main()
