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

import dataclasses
import datetime
from unittest import mock
import uuid

from absl.testing import absltest
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.data import time_coordinates as tc
from lookerstudio import mmm_ui_proto_generator
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from schema import mmm_proto_generator
from schema import test_data as td
from schema.processors import budget_optimization_processor
from schema.processors import marketing_processor
from schema.processors import model_fit_processor
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor as rf_opt_processor
from schema.serde import meridian_serde
from schema.utils import date_range_bucketing
import numpy as np
import xarray as xr

from tensorflow.python.util.protobuf import compare


_STUBBED_PROCESSORS = (
    td.FakeModelFitProcessor,
    td.FakeBudgetOptimizationProcessor,
    td.FakeReachFrequencyOptimizationProcessor,
    td.FakeMarketingProcessor,
)


class MmmUiProtoGeneratorHelperFunctionsTest(absltest.TestCase):

  def test_create_tag_monthly(self):
    tag = mmm_ui_proto_generator.create_tag(
        date_range_bucketing.MonthlyDateRangeGenerator,
        datetime.date(2024, 1, 1),
    )
    self.assertEqual(tag, 'Y2024 Jan')

  def test_create_tag_quarterly(self):
    tag = mmm_ui_proto_generator.create_tag(
        date_range_bucketing.QuarterlyDateRangeGenerator,
        datetime.date(2024, 1, 1),
    )
    self.assertEqual(tag, 'Y2024 Q1')

  def test_create_tag_yearly(self):
    tag = mmm_ui_proto_generator.create_tag(
        date_range_bucketing.YearlyDateRangeGenerator,
        datetime.date(2024, 1, 1),
    )
    self.assertEqual(tag, 'Y2024')


class MmmUiDataProtoGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_mmm = mock.MagicMock()

    all_times_data_array = xr.DataArray(
        data=np.array(td.ALL_TIMES_IN_MERIDIAN),
        dims=[constants.TIME],
        coords={
            constants.TIME: (
                [constants.TIME],
                list(td.ALL_TIMES_IN_MERIDIAN),
            ),
        },
    )
    self.mock_mmm.input_data.time = all_times_data_array
    self.mock_mmm.input_data.time_coordinates = tc.TimeCoordinates.from_dates(
        td.ALL_TIMES_IN_MERIDIAN
    )
    self.mock_mmm.input_data.rf_channel = xr.DataArray(['rf_ch_0', 'rf_ch_1'])

    self.mock_mmm_analyzer = self.enter_context(
        mock.patch.object(analyzer, 'Analyzer', autospec=True)
    )(self.mock_mmm)

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

  def test_create_mmm_ui_data_proto_populates_model_kernel(self):
    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(self.mock_mmm, [])
    self.assertTrue(output.HasField('mmm_kernel'))

  def test_create_mmm_ui_data_proto_populates_each_processor_output(self):
    self.mock_mmm.expand_selected_time_dims.return_value = [
        datetime.date(2023, 1, 2),
        datetime.date(2023, 1, 9),
        datetime.date(2023, 1, 16),
    ]
    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            model_fit_processor.ModelFitSpec(),
            budget_optimization_processor.BudgetOptimizationSpec(
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 1, 23),
                optimization_name='budget optimization',
                grid_name='youtube_campaign',
                scenario=optimizer.FixedBudgetScenario(1),
            ),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 1, 23),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
        time_breakdown_generators={},
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

  def test_create_mmm_ui_data_proto_fails_when_requesting_unsupported_spec_type(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        'Unsupported spec type: ReachFrequencyOptimizationSpec',
    ):
      mmm_ui_proto_generator.create_mmm_ui_data_proto(
          self.mock_mmm,
          [
              rf_opt_processor.ReachFrequencyOptimizationSpec(
                  start_date=datetime.date(2023, 1, 1),
                  end_date=datetime.date(2023, 1, 23),
                  optimization_name='RF optimization',
                  grid_name='RF optimization grid',
              ),
          ],
      )

  def test_create_mmm_ui_data_proto_when_creating_subspecs_using_sub_times(
      self,
  ):
    self.mock_mmm.expand_selected_time_dims.return_value = [
        '2022-12-05',
        '2022-12-12',
        '2022-12-19',
        '2022-12-26',
        '2023-01-02',
        '2023-01-09',
        '2023-01-16',
        '2023-01-23',
        '2023-01-30',
    ]

    expected_specs = [
        # Original spec
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2022, 12, 5),
            end_date=datetime.date(2023, 2, 6),
            date_interval_tag='ALL',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        # Monthly buckets
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2022, 12, 5),
            end_date=datetime.date(2023, 1, 2),
            date_interval_tag='Y2022 Dec',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2023, 1, 2),
            end_date=datetime.date(2023, 2, 6),
            date_interval_tag='Y2023 Jan',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
    ]

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2022, 12, 5),
                end_date=datetime.date(2023, 2, 6),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )

    compare.assertProto2SameElements(
        self,
        output.marketing_analysis_list,
        td.FakeMarketingProcessor(self.mock_trained_model).execute(
            expected_specs
        ),
    )

  def test_create_mmm_ui_data_proto_when_creating_subspecs_using_last_possible_end_date(
      self,
  ):
    self.mock_mmm.expand_selected_time_dims.return_value = [
        '2023-01-02',
        '2023-01-09',
        '2023-01-16',
        '2023-01-23',
        '2023-01-30',
    ]

    expected_specs = [
        # Original spec
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2023, 1, 2),
            end_date=datetime.date(2023, 2, 6),
            date_interval_tag='ALL',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        # Monthly buckets
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2023, 1, 2),
            end_date=datetime.date(2023, 2, 6),
            date_interval_tag='Y2023 Jan',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
    ]

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2023, 1, 2),
                end_date=datetime.date(2023, 2, 6),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )

    compare.assertProto2SameElements(
        self,
        output.marketing_analysis_list,
        td.FakeMarketingProcessor(self.mock_trained_model).execute(
            expected_specs
        ),
    )

  def test_create_mmm_ui_data_proto_generating_marketing_analysis_subspecs(
      self,
  ):
    expected_specs = [
        marketing_processor.MarketingAnalysisSpec(
            start_date=dated_spec.start_date,
            end_date=dated_spec.end_date,
            date_interval_tag=dated_spec.date_interval_tag,
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        )
        for dated_spec in td.ALL_TIME_BUCKET_DATED_SPECS
    ]

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2022, 11, 21),
                end_date=datetime.date(2024, 1, 8),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )
    self.assertTrue(output.HasField('marketing_analysis_list'))
    compare.assertProto2SameElements(
        self,
        output.marketing_analysis_list,
        td.FakeMarketingProcessor(self.mock_trained_model).execute(
            expected_specs
        ),
    )

  def test_create_mmm_ui_data_proto_when_budget_optimization_no_rf_data(
      self,
  ):
    self.mock_mmm.input_data.rf_channel = None
    spec = budget_optimization_processor.BudgetOptimizationSpec(
        start_date=datetime.date(2023, 1, 2),
        end_date=datetime.date(2023, 2, 6),
        optimization_name='budget optimization',
        grid_name='youtube_campaign',
        scenario=optimizer.FixedBudgetScenario(1),
    )

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [spec],
        time_breakdown_generators={},
    )

    self.assertTrue(output.HasField('marketing_optimization'))
    self.assertFalse(
        output.marketing_optimization.HasField('reach_frequency_optimization')
    )

  @mock.patch.object(uuid, 'uuid4', return_value='group-id')
  def test_create_mmm_ui_data_proto_generating_budget_and_rf_optimization(
      self,
      mock_uuid,
  ):
    all_time_dated_spec = td.ALL_TIME_BUCKET_DATED_SPECS[0]

    budget_opt_name = 'budget optimization'
    budget_opt_spec = budget_optimization_processor.BudgetOptimizationSpec(
        start_date=all_time_dated_spec.start_date,
        end_date=all_time_dated_spec.end_date,
        date_interval_tag=all_time_dated_spec.date_interval_tag,
        optimization_name=budget_opt_name,
        grid_name='youtube_campaign',
        scenario=optimizer.FixedBudgetScenario(10),
    )

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [budget_opt_spec],
        time_breakdown_generators={},
    )

    self.assertTrue(output.HasField('marketing_optimization'))
    mock_uuid.assert_called_once()

    expected_budget_opt_specs = [
        dataclasses.replace(
            budget_opt_spec,
            group_id='group-id',
        )
    ]
    expected_rf_opt_specs = [
        rf_opt_processor.ReachFrequencyOptimizationSpec(
            start_date=all_time_dated_spec.start_date,
            end_date=all_time_dated_spec.end_date,
            date_interval_tag=all_time_dated_spec.date_interval_tag,
            optimization_name=f'derived RF optimization from {budget_opt_name}',
            grid_name=f'derived_from_{budget_opt_name}',
            group_id='group-id',
        )
    ]

    with self.subTest('BudgetOptimization'):
      self.assertTrue(
          output.marketing_optimization.HasField('budget_optimization')
      )
      compare.assertProto2SameElements(
          self,
          output.marketing_optimization.budget_optimization,
          td.FakeBudgetOptimizationProcessor(self.mock_trained_model).execute(
              expected_budget_opt_specs
          ),
      )

    with self.subTest('ReachFrequencyOptimization'):
      self.assertTrue(
          output.marketing_optimization.HasField('reach_frequency_optimization')
      )
      compare.assertProto2SameElements(
          self,
          output.marketing_optimization.reach_frequency_optimization,
          td.FakeReachFrequencyOptimizationProcessor(
              self.mock_trained_model
          ).execute(expected_rf_opt_specs),
      )

  def test_create_mmm_ui_data_proto_generating_budget_optimization_subspecs(
      self,
  ):
    total_budget = 1
    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            budget_optimization_processor.BudgetOptimizationSpec(
                start_date=datetime.date(2022, 11, 21),
                end_date=datetime.date(2024, 1, 8),
                optimization_name='budget optimization',
                grid_name='youtube_campaign',
                group_id='fixed-group-id',
                scenario=optimizer.FixedBudgetScenario(total_budget),
            ),
        ],
    )

    expected_specs = []
    for dated_spec in td.ALL_TIME_BUCKET_DATED_SPECS:
      subspec_budget = (
          total_budget if dated_spec.date_interval_tag == 'ALL' else None
      )
      group_id = f'fixed-group-id:{dated_spec.date_interval_tag}'
      optimization_name = (
          f'budget optimization for {dated_spec.date_interval_tag}'
      )
      grid_name = (
          f'youtube_campaign_{dated_spec.date_interval_tag.replace(" ", "_")}'
      )

      expected_specs.append(
          budget_optimization_processor.BudgetOptimizationSpec(
              start_date=dated_spec.start_date,
              end_date=dated_spec.end_date,
              date_interval_tag=dated_spec.date_interval_tag,
              optimization_name=optimization_name,
              grid_name=grid_name,
              group_id=group_id,
              scenario=optimizer.FixedBudgetScenario(subspec_budget),
          )
      )

    self.assertTrue(output.HasField('marketing_optimization'))
    self.assertTrue(
        output.marketing_optimization.HasField('budget_optimization')
    )
    compare.assertProto2SameElements(
        self,
        output.marketing_optimization.budget_optimization,
        td.FakeBudgetOptimizationProcessor(self.mock_trained_model).execute(
            expected_specs
        ),
    )

  def test_create_mmm_ui_data_proto_generating_rf_optimization_subspecs(
      self,
  ):
    group_id = 'groupId'
    optimization_name = 'budget optimization'
    grid_name = 'youtube_campaign'

    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            budget_optimization_processor.BudgetOptimizationSpec(
                start_date=datetime.date(2022, 11, 21),
                end_date=datetime.date(2024, 1, 8),
                optimization_name=optimization_name,
                grid_name=grid_name,
                group_id=group_id,
            ),
        ],
    )

    expected_rf_opt_name = f'derived RF optimization from {optimization_name}'
    expected_rf_opt_grid_name = f'derived_from_{optimization_name}'

    expected_specs = []
    for dated_spec in td.ALL_TIME_BUCKET_DATED_SPECS:
      subspec_group_id = f'{group_id}:{dated_spec.date_interval_tag}'
      subspec_rf_opt_name = (
          f'{expected_rf_opt_name} for {dated_spec.date_interval_tag}'
      )
      subspec_rf_opt_grid_name = (
          f'{expected_rf_opt_grid_name}_'
          f'{dated_spec.date_interval_tag.replace(" ", "_")}'
      )

      expected_specs.append(
          rf_opt_processor.ReachFrequencyOptimizationSpec(
              start_date=dated_spec.start_date,
              end_date=dated_spec.end_date,
              date_interval_tag=dated_spec.date_interval_tag,
              optimization_name=subspec_rf_opt_name,
              grid_name=subspec_rf_opt_grid_name,
              group_id=subspec_group_id,
          )
      )

    self.assertTrue(output.HasField('marketing_optimization'))
    self.assertTrue(
        output.marketing_optimization.HasField('reach_frequency_optimization')
    )
    compare.assertProto2SameElements(
        self,
        output.marketing_optimization.reach_frequency_optimization,
        td.FakeReachFrequencyOptimizationProcessor(
            self.mock_trained_model
        ).execute(expected_specs),
    )

  def test_create_mmm_ui_data_proto_warns_when_budget_optimization_subspecs_use_historical_spend(
      self,
  ):
    with self.assertWarns(UserWarning):
      mmm_ui_proto_generator.create_mmm_ui_data_proto(
          self.mock_mmm,
          [
              budget_optimization_processor.BudgetOptimizationSpec(
                  start_date=datetime.date(2022, 11, 21),
                  end_date=datetime.date(2024, 1, 8),
                  optimization_name='budget optimization',
                  grid_name='youtube_campaign',
                  group_id='fixed-group-id',
                  scenario=optimizer.FixedBudgetScenario(1),
              ),
          ],
      )

  def test_create_mmm_ui_data_proto_processes_multiple_specs_of_the_same_type(
      self,
  ):
    self.mock_mmm.expand_selected_time_dims.side_effect = [
        # from 2022-11-21 to 2022-12-26
        [tc.normalize_date(date) for date in td.ALL_TIMES_IN_MERIDIAN[:6]],
        # from 2023-12-25 to 2024-01-01
        [tc.normalize_date(date) for date in td.ALL_TIMES_IN_MERIDIAN[-2:]],
    ]

    expected_specs = [
        # Original spec 1
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2022, 11, 21),
            end_date=datetime.date(2023, 1, 2),
            date_interval_tag='ALL',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        # Monthly buckets spec 1
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2022, 12, 5),
            end_date=datetime.date(2023, 1, 2),
            date_interval_tag='Y2022 Dec',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
        # Original spec 2
        marketing_processor.MarketingAnalysisSpec(
            start_date=datetime.date(2023, 12, 25),
            end_date=datetime.date(2024, 1, 8),
            date_interval_tag='ALL',
            media_summary_spec=marketing_processor.MediaSummarySpec(),
        ),
    ]
    output = mmm_ui_proto_generator.create_mmm_ui_data_proto(
        self.mock_mmm,
        [
            model_fit_processor.ModelFitSpec(),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2022, 11, 21),
                end_date=datetime.date(2023, 1, 2),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
            marketing_processor.MarketingAnalysisSpec(
                start_date=datetime.date(2023, 12, 25),
                end_date=datetime.date(2024, 1, 8),
                media_summary_spec=marketing_processor.MediaSummarySpec(),
            ),
        ],
    )
    self.assertTrue(output.HasField('model_fit'))
    self.assertTrue(output.HasField('marketing_analysis_list'))
    compare.assertProto2SameElements(
        self,
        output.marketing_analysis_list,
        td.FakeMarketingProcessor(self.mock_trained_model).execute(
            expected_specs
        ),
    )

  def test_create_mmm_ui_data_proto_duplicate_budget_optimization_group_ids(
      self,
  ):
    with self.assertRaisesRegex(ValueError, 'Duplicate group ID'):
      mmm_ui_proto_generator.create_mmm_ui_data_proto(
          self.mock_mmm,
          [
              budget_optimization_processor.BudgetOptimizationSpec(
                  start_date=datetime.date(2022, 11, 21),
                  end_date=datetime.date(2024, 1, 8),
                  optimization_name='budget optimization',
                  grid_name='youtube_campaign',
                  group_id='fixed-group-id',
                  scenario=optimizer.FixedBudgetScenario(1),
              ),
              budget_optimization_processor.BudgetOptimizationSpec(
                  start_date=datetime.date(2022, 11, 21),
                  end_date=datetime.date(2024, 1, 8),
                  optimization_name='budget optimization',
                  grid_name='youtube_campaign',
                  group_id='fixed-group-id',
                  scenario=optimizer.FixedBudgetScenario(1),
              ),
          ],
      )


if __name__ == '__main__':
  absltest.main()
