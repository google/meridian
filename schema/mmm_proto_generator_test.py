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
from unittest import mock

from absl.testing import absltest
from meridian.analysis import optimizer
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from schema import mmm_proto_generator
from schema import test_data as td
from schema.processors import budget_optimization_processor
from schema.processors import marketing_processor
from schema.processors import model_fit_processor
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor as rf_opt_processor
from schema.serde import meridian_serde

from tensorflow.python.util.protobuf import compare


_STUBBED_PROCESSORS = [
    td.FakeModelFitProcessor,
    td.FakeBudgetOptimizationProcessor,
    td.FakeReachFrequencyOptimizationProcessor,
    td.FakeMarketingProcessor,
]


class MmmProtoGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_mmm = mock.MagicMock()

    # Patch the trained model class.
    self.mock_trained_model = self.enter_context(
        mock.patch.object(model_processor, 'TrainedModel', autospec=True)
    )(mmm=self.mock_mmm)
    self.enter_context(
        mock.patch.object(
            meridian_serde.MeridianSerde,
            'serialize',
            autospec=True,
            return_value=kernel_pb.MmmKernel(),
        )
    )

    self.enter_context(
        mock.patch.object(
            mmm_proto_generator,
            '_TYPES',
            new=_STUBBED_PROCESSORS,
        )
    )

  def test_create_mmm_proto_populates_model_kernel(self):
    output = mmm_proto_generator.create_mmm_proto(self.mock_mmm, [])
    self.assertTrue(output.HasField('mmm_kernel'))

  def test_create_mmm_proto_populates_each_processor_output(self):
    output = mmm_proto_generator.create_mmm_proto(
        self.mock_mmm,
        [
            model_fit_processor.ModelFitSpec(),
            budget_optimization_processor.BudgetOptimizationSpec(
                start_date=datetime.date(2023, 1, 2),
                end_date=datetime.date(2023, 2, 6),
                optimization_name='budget optimization',
                grid_name='youtube_campaign',
                scenario=optimizer.FixedBudgetScenario(1),
            ),
            rf_opt_processor.ReachFrequencyOptimizationSpec(
                start_date=datetime.date(2023, 1, 2),
                end_date=datetime.date(2023, 2, 6),
                optimization_name='RF optimization',
                grid_name='RF optimization grid',
            ),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2023, 1, 2),
                end_date=datetime.date(2023, 2, 6),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )
    with self.subTest('ModelFit'):
      self.assertTrue(output.HasField('model_fit'))
    with self.subTest('MarketingAnalysisList'):
      self.assertTrue(output.HasField('marketing_analysis_list'))
    with self.subTest('MarketingOptimization'):
      self.assertTrue(output.HasField('marketing_optimization'))
    with self.subTest('BudgetOptimization'):
      self.assertTrue(
          output.marketing_optimization.HasField('budget_optimization')
      )
    with self.subTest('ReachFrequencyOptimization'):
      self.assertTrue(
          output.marketing_optimization.HasField('reach_frequency_optimization')
      )

  def test_create_mmm_proto_processes_multiple_specs_of_the_same_type(self):
    expected_marketing_analysis_specs = [
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2022, 11, 21),
            end_date=datetime.date(2023, 1, 16),
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2023, 3, 27),
            end_date=datetime.date(2023, 4, 10),
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
    ]
    output = mmm_proto_generator.create_mmm_proto(
        self.mock_mmm,
        [
            model_fit_processor.ModelFitSpec(),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2022, 11, 21),
                end_date=datetime.date(2023, 1, 16),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2023, 3, 27),
                end_date=datetime.date(2023, 4, 10),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )
    self.assertTrue(output.HasField('model_fit'))
    self.assertTrue(output.HasField('marketing_analysis_list'))
    compare.assertProtoEqual(
        self,
        output.marketing_analysis_list,
        td.FakeMarketingProcessor(self.mock_trained_model).execute(
            expected_marketing_analysis_specs
        ),
    )


if __name__ == '__main__':
  absltest.main()
