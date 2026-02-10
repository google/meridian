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

"""Test data for MMM proto generator."""

from collections.abc import Sequence
import datetime

from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.common import date_interval_pb2 as date_interval_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from schema.processors import budget_optimization_processor
from schema.processors import marketing_processor
from schema.processors import model_fit_processor
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor as rf_opt_processor

from google.type import date_pb2

# Weekly dates from 2022-11-21 to 2024-01-01.
ALL_TIMES_IN_MERIDIAN = (
    '2022-11-21',
    '2022-11-28',
    '2022-12-05',
    '2022-12-12',
    '2022-12-19',
    '2022-12-26',
    '2023-01-02',
    '2023-01-09',
    '2023-01-16',
    '2023-01-23',
    '2023-01-30',
    '2023-02-06',
    '2023-02-13',
    '2023-02-20',
    '2023-02-27',
    '2023-03-06',
    '2023-03-13',
    '2023-03-20',
    '2023-03-27',
    '2023-04-03',
    '2023-04-10',
    '2023-04-17',
    '2023-04-24',
    '2023-05-01',
    '2023-05-08',
    '2023-05-15',
    '2023-05-22',
    '2023-05-29',
    '2023-06-05',
    '2023-06-12',
    '2023-06-19',
    '2023-06-26',
    '2023-07-03',
    '2023-07-10',
    '2023-07-17',
    '2023-07-24',
    '2023-07-31',
    '2023-08-07',
    '2023-08-14',
    '2023-08-21',
    '2023-08-28',
    '2023-09-04',
    '2023-09-11',
    '2023-09-18',
    '2023-09-25',
    '2023-10-02',
    '2023-10-09',
    '2023-10-16',
    '2023-10-23',
    '2023-10-30',
    '2023-11-06',
    '2023-11-13',
    '2023-11-20',
    '2023-11-27',
    '2023-12-04',
    '2023-12-11',
    '2023-12-18',
    '2023-12-25',
    '2024-01-01',
)

ALL_TIME_BUCKET_DATED_SPECS = (
    # All
    model_processor.DatedSpec(
        start_date=datetime.date(2022, 11, 21),
        end_date=datetime.date(2024, 1, 8),
        date_interval_tag='ALL',
    ),
    # Monthly buckets
    model_processor.DatedSpec(
        start_date=datetime.date(2022, 12, 5),
        end_date=datetime.date(2023, 1, 2),
        date_interval_tag='Y2022 Dec',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 1, 2),
        end_date=datetime.date(2023, 2, 6),
        date_interval_tag='Y2023 Jan',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 2, 6),
        end_date=datetime.date(2023, 3, 6),
        date_interval_tag='Y2023 Feb',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 3, 6),
        end_date=datetime.date(2023, 4, 3),
        date_interval_tag='Y2023 Mar',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 4, 3),
        end_date=datetime.date(2023, 5, 1),
        date_interval_tag='Y2023 Apr',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 5, 1),
        end_date=datetime.date(2023, 6, 5),
        date_interval_tag='Y2023 May',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 6, 5),
        end_date=datetime.date(2023, 7, 3),
        date_interval_tag='Y2023 Jun',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 7, 3),
        end_date=datetime.date(2023, 8, 7),
        date_interval_tag='Y2023 Jul',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 8, 7),
        end_date=datetime.date(2023, 9, 4),
        date_interval_tag='Y2023 Aug',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 9, 4),
        end_date=datetime.date(2023, 10, 2),
        date_interval_tag='Y2023 Sep',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 10, 2),
        end_date=datetime.date(2023, 11, 6),
        date_interval_tag='Y2023 Oct',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 11, 6),
        end_date=datetime.date(2023, 12, 4),
        date_interval_tag='Y2023 Nov',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 12, 4),
        end_date=datetime.date(2024, 1, 1),
        date_interval_tag='Y2023 Dec',
    ),
    # Quarterly buckets
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 1, 2),
        end_date=datetime.date(2023, 4, 3),
        date_interval_tag='Y2023 Q1',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 4, 3),
        end_date=datetime.date(2023, 7, 3),
        date_interval_tag='Y2023 Q2',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 7, 3),
        end_date=datetime.date(2023, 10, 2),
        date_interval_tag='Y2023 Q3',
    ),
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 10, 2),
        end_date=datetime.date(2024, 1, 1),
        date_interval_tag='Y2023 Q4',
    ),
    # Yearly buckets
    model_processor.DatedSpec(
        start_date=datetime.date(2023, 1, 2),
        end_date=datetime.date(2024, 1, 1),
        date_interval_tag='Y2023',
    ),
)


def _dated_spec_to_date_interval(
    spec: model_processor.DatedSpec,
) -> date_interval_pb.DateInterval:
  if spec.start_date is None or spec.end_date is None:
    raise ValueError('Start date or end date is None.')

  return date_interval_pb.DateInterval(
      start_date=date_pb2.Date(
          year=spec.start_date.year,
          month=spec.start_date.month,
          day=spec.start_date.day,
      ),
      end_date=date_pb2.Date(
          year=spec.end_date.year,
          month=spec.end_date.month,
          day=spec.end_date.day,
      ),
      tag=spec.date_interval_tag,
  )


class FakeModelFitProcessor(
    model_processor.ModelProcessor[
        model_fit_processor.ModelFitSpec, fit_pb.ModelFit
    ]
):
  """Fake ModelFitProcessor for testing."""

  def __init__(self, trained_model: model_processor.TrainedModel):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return model_fit_processor.ModelFitSpec

  @classmethod
  def output_type(cls):
    return fit_pb.ModelFit

  def execute(
      self, specs: Sequence[model_fit_processor.ModelFitSpec]
  ) -> fit_pb.ModelFit:
    return fit_pb.ModelFit()

  def _set_output(self, output: mmm_pb.Mmm, result: fit_pb.ModelFit):
    output.model_fit.CopyFrom(result)


class FakeBudgetOptimizationProcessor(
    model_processor.ModelProcessor[
        budget_optimization_processor.BudgetOptimizationSpec,
        budget_pb.BudgetOptimization,
    ]
):
  """Fake BudgetOptimizationProcessor for testing."""

  def __init__(self, trained_model: model_processor.TrainedModel):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return budget_optimization_processor.BudgetOptimizationSpec

  @classmethod
  def output_type(cls):
    return budget_pb.BudgetOptimization

  def execute(
      self,
      specs: Sequence[budget_optimization_processor.BudgetOptimizationSpec],
  ) -> budget_pb.BudgetOptimization:
    results = []
    for spec in specs:
      result = budget_pb.BudgetOptimizationResult(
          name=spec.optimization_name,
          spec=budget_pb.BudgetOptimizationSpec(
              date_interval=_dated_spec_to_date_interval(spec)
          ),
          incremental_outcome_grid=budget_pb.IncrementalOutcomeGrid(
              name=spec.grid_name
          ),
      )
      if spec.group_id:
        result.group_id = spec.group_id
      results.append(result)

    return budget_pb.BudgetOptimization(results=results)

  def _set_output(
      self, output: mmm_pb.Mmm, result: budget_pb.BudgetOptimization
  ):
    output.marketing_optimization.budget_optimization.CopyFrom(result)


class FakeReachFrequencyOptimizationProcessor(
    model_processor.ModelProcessor[
        rf_opt_processor.ReachFrequencyOptimizationSpec,
        rf_pb.ReachFrequencyOptimization,
    ]
):
  """Fake ReachFrequencyOptimizationProcessor for testing."""

  def __init__(self, trained_model: model_processor.TrainedModel):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return rf_opt_processor.ReachFrequencyOptimizationSpec

  @classmethod
  def output_type(cls):
    return rf_pb.ReachFrequencyOptimization

  def execute(
      self,
      specs: Sequence[rf_opt_processor.ReachFrequencyOptimizationSpec],
  ) -> rf_pb.ReachFrequencyOptimization:
    results = []
    for spec in specs:
      result = rf_pb.ReachFrequencyOptimizationResult(
          name=spec.optimization_name,
          spec=rf_pb.ReachFrequencyOptimizationSpec(
              date_interval=_dated_spec_to_date_interval(spec)
          ),
          frequency_outcome_grid=rf_pb.FrequencyOutcomeGrid(
              name=spec.grid_name
          ),
      )
      if spec.group_id:
        result.group_id = spec.group_id
      results.append(result)

    return rf_pb.ReachFrequencyOptimization(results=results)

  def _set_output(
      self,
      output: mmm_pb.Mmm,
      result: rf_pb.ReachFrequencyOptimization,
  ):
    output.marketing_optimization.reach_frequency_optimization.CopyFrom(result)


class FakeMarketingProcessor(
    model_processor.ModelProcessor[
        marketing_processor.MarketingAnalysisSpec,
        marketing_analysis_pb2.MarketingAnalysisList,
    ]
):
  """Fake MarketingProcessor for testing."""

  def __init__(self, trained_model: model_processor.TrainedModel):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return marketing_processor.MarketingAnalysisSpec

  @classmethod
  def output_type(cls):
    return marketing_analysis_pb2.MarketingAnalysisList

  def execute(
      self, specs: Sequence[marketing_processor.MarketingAnalysisSpec]
  ) -> marketing_analysis_pb2.MarketingAnalysisList:
    marketing_analyses = []
    for spec in specs:
      marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
          date_interval=_dated_spec_to_date_interval(spec)
      )
      marketing_analyses.append(marketing_analysis)

    return marketing_analysis_pb2.MarketingAnalysisList(
        marketing_analyses=marketing_analyses
    )

  def _set_output(
      self,
      output: mmm_pb.Mmm,
      result: marketing_analysis_pb2.MarketingAnalysisList,
  ):
    output.marketing_analysis_list.CopyFrom(result)
