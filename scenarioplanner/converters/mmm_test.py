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
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.common import date_interval_pb2 as date_interval_pb
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import target_metric_pb2 as target_metric_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2 as marketing_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_pb
from mmm.v1.marketing.analysis import non_media_analysis_pb2 as non_media_pb
from mmm.v1.marketing.analysis import response_curve_pb2 as response_curve_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from mmm.v1.marketing.optimization import marketing_optimization_pb2 as optimization_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from scenarioplanner.converters import mmm
from scenarioplanner.converters import test_data as td

from google.type import date_pb2 as date_pb


class MmmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._mmm_proto = mmm_pb.Mmm(
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

  def test_marketing_data(self):
    output = mmm.Mmm(self._mmm_proto)
    self.assertEqual(
        output.marketing_data.marketing_data_proto,
        td.MARKETING_DATA,
    )

  def test_model_fit(self):
    output = mmm.Mmm(self._mmm_proto)
    self.assertEqual(output.model_fit, self._mmm_proto.model_fit)

  def test_model_fit_mapped_results(self):
    output = mmm.Mmm(self._mmm_proto)
    self.assertEqual(
        output.model_fit_results,
        {
            c.TRAIN: td.MODEL_FIT_RESULT_TRAIN,
            c.TEST: td.MODEL_FIT_RESULT_TEST,
            c.ALL_DATA: td.MODEL_FIT_RESULT_ALL_DATA,
        },
    )

  def test_budget_optimization_results(self):
    output = mmm.Mmm(self._mmm_proto)
    self.assertEqual(
        output.budget_optimization_results,
        [
            mmm.BudgetOptimizationResult(
                td.BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES
            ),
            mmm.BudgetOptimizationResult(
                td.BUDGET_OPTIMIZATION_RESULT_FLEX_NONREV
            ),
        ],
    )

  def test_reach_frequency_optimization_results(self):
    output = mmm.Mmm(self._mmm_proto)
    self.assertEqual(
        output.reach_frequency_optimization_results,
        [
            mmm.ReachFrequencyOptimizationResult(td.RF_OPTIMIZATION_RESULT_FOO),
        ],
    )


class MarketingDataTest(absltest.TestCase):

  def test_media_channels(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    self.assertEqual(
        marketing_data.media_channels,
        ["Channel 1", "Channel 2"],
    )

  def test_rf_channels(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    self.assertEqual(
        marketing_data.rf_channels,
        ["RF Channel 1", "RF Channel 2"],
    )

  def test_date_intervals(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    self.assertEqual(
        marketing_data.date_intervals,
        [
            mmm.DateInterval(
                (datetime.date(2024, 1, 1), datetime.date(2024, 1, 8))
            ),
            mmm.DateInterval(
                (datetime.date(2024, 1, 8), datetime.date(2024, 1, 15))
            ),
        ],
    )

  def test_media_channel_spends(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    date_interval = ("2024-01-08", "2024-01-15")
    self.assertEqual(
        marketing_data.media_channel_spends(date_interval),
        {
            "Channel 1": td.BASE_MEDIA_SPEND * 2 * 1,  # x geo x time
            "Channel 2": td.BASE_MEDIA_SPEND * 2 * 1,
        },
    )

  def test_media_channel_spends_outside_given_interval(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    date_interval = ("2024-01-15", "2024-01-30")
    self.assertEqual(
        marketing_data.media_channel_spends(date_interval),
        {
            "Channel 1": 0.0,
            "Channel 2": 0.0,
        },
    )

  def test_rf_channel_spends(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    date_interval = ("2024-01-08", "2024-01-15")
    self.assertEqual(
        marketing_data.rf_channel_spends(date_interval),
        {
            "RF Channel 1": td.BASE_RF_MEDIA_SPEND * 2 * 1,  # x geo x time
            "RF Channel 2": td.BASE_RF_MEDIA_SPEND * 2 * 1,
        },
    )

  def test_rf_channel_spends_outside_given_interval(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    date_interval = ("2024-01-15", "2024-01-30")
    self.assertEqual(
        marketing_data.rf_channel_spends(date_interval),
        {
            "RF Channel 1": 0.0,
            "RF Channel 2": 0.0,
        },
    )

  def test_all_channel_spends(self):
    marketing_data = mmm.MarketingData(td.MARKETING_DATA)
    date_interval = ("2024-01-08", "2024-01-15")
    self.assertEqual(
        marketing_data.all_channel_spends(date_interval),
        {
            "Channel 1": td.BASE_MEDIA_SPEND * 2 * 1,
            "Channel 2": td.BASE_MEDIA_SPEND * 2 * 1,
            "RF Channel 1": td.BASE_RF_MEDIA_SPEND * 2 * 1,
            "RF Channel 2": td.BASE_RF_MEDIA_SPEND * 2 * 1,
        },
    )


class MarketingAnalysisTest(absltest.TestCase):

  def test_channel_mapped_media_analyses(self):
    analysis_1 = media_pb.MediaAnalysis(channel_name="Channel 1")
    analysis_2 = media_pb.MediaAnalysis(channel_name="Channel 2")
    analysis_3 = media_pb.MediaAnalysis(channel_name="Channel 3")

    wrapper = mmm.MarketingAnalysis(
        marketing_pb.MarketingAnalysis(
            media_analyses=[
                analysis_1,
                analysis_2,
                analysis_3,
            ]
        )
    )

    self.assertEqual(
        {
            channel: media_analysis.analysis_proto
            for channel, media_analysis in wrapper.channel_mapped_media_analyses.items()
        },
        {
            "Channel 1": analysis_1,
            "Channel 2": analysis_2,
            "Channel 3": analysis_3,
        },
    )

  def test_get_baseline_analysis(self):
    analysis_1 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 1")
    analysis_2 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 2")
    analysis_3 = non_media_pb.NonMediaAnalysis(non_media_name=c.BASELINE)

    wrapper = mmm.MarketingAnalysis(
        marketing_pb.MarketingAnalysis(
            non_media_analyses=[
                analysis_1,
                analysis_2,
                analysis_3,
            ]
        )
    )

    self.assertEqual(
        wrapper.baseline_analysis.analysis_proto,
        analysis_3,
    )

  def test_get_channel_mapped_non_media_analyses(self):
    analysis_1 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 1")
    analysis_2 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 2")
    analysis_3 = non_media_pb.NonMediaAnalysis(non_media_name=c.BASELINE)

    wrapper = mmm.MarketingAnalysis(
        marketing_pb.MarketingAnalysis(
            non_media_analyses=[
                analysis_1,
                analysis_2,
                analysis_3,
            ]
        )
    )

    self.assertEqual(
        {
            channel: non_media_analysis.analysis_proto
            for channel, non_media_analysis in wrapper.channel_mapped_non_media_analyses.items()
        },
        {
            "Channel 1": analysis_1,
            "Channel 2": analysis_2,
            c.BASELINE: analysis_3,
        },
    )

  def test_get_baseline_analysis_missing(self):
    analysis_1 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 1")
    analysis_2 = non_media_pb.NonMediaAnalysis(non_media_name="Channel 2")

    wrapper = mmm.MarketingAnalysis(
        marketing_pb.MarketingAnalysis(
            non_media_analyses=[
                analysis_1,
                analysis_2,
            ]
        )
    )

    with self.assertRaises(ValueError):
      _ = wrapper.baseline_analysis

  def test_response_curves(self):
    curve_1 = response_curve_pb.ResponseCurve(input_name="Spend 1")
    curve_2 = response_curve_pb.ResponseCurve(input_name="Spend 2")
    media_analysis_1 = media_pb.MediaAnalysis(
        channel_name="Channel 1",
        response_curve=curve_1,
    )
    media_analysis_2 = media_pb.MediaAnalysis(
        channel_name="Channel 2",
        response_curve=curve_2,
    )
    marketing_analysis = marketing_pb.MarketingAnalysis(
        media_analyses=[
            media_analysis_1,
            media_analysis_2,
        ]
    )
    wrapper = mmm.MarketingAnalysis(marketing_analysis)
    self.assertEqual(
        [curve.response_curve_proto for curve in wrapper.response_curves],
        [curve_1, curve_2],
    )


class MediaAnalysisTest(absltest.TestCase):

  def test_channel_name(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(channel_name="Channel 1")
    )
    self.assertEqual(analysis.channel_name, "Channel 1")

  def test_spend_info_pb(self):
    spend_info_pb = media_pb.SpendInfo(
        spend=1000.0,
        spend_share=0.5,
    )
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(spend_info=spend_info_pb)
    )
    self.assertEqual(
        analysis.spend_info_pb,
        spend_info_pb,
    )

  def test_maybe_revenue_kpi_outcome_has_no_revenue_outcome(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertIsNone(analysis.maybe_revenue_kpi_outcome)

  def test_maybe_revenue_kpi_outcome_has_both_outcomes(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.maybe_revenue_kpi_outcome,
        mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME),
    )

  def test_revenue_kpi_outcome(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.revenue_kpi_outcome,
        mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME),
    )

  def test_revenue_kpi_outcome_missing(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    with self.assertRaises(ValueError):
      _ = analysis.revenue_kpi_outcome

  def test_maybe_non_revenue_kpi_outcome_has_no_non_revenue_outcome(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertIsNone(analysis.maybe_non_revenue_kpi_outcome)

  def test_maybe_non_revenue_kpi_outcome_has_both_outcomes(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.maybe_non_revenue_kpi_outcome,
        mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME),
    )

  def test_non_revenue_kpi_outcome(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.non_revenue_kpi_outcome,
        mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME),
    )

  def test_non_revenue_kpi_outcome_missing(self):
    analysis = mmm.MediaAnalysis(
        media_pb.MediaAnalysis(
            media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    with self.assertRaises(ValueError):
      _ = analysis.non_revenue_kpi_outcome


class NonMediaAnalysisTest(absltest.TestCase):

  def test_non_media_name(self):
    analysis = mmm.NonMediaAnalysis(
        non_media_pb.NonMediaAnalysis(non_media_name="Baseline")
    )
    self.assertEqual(analysis.non_media_name, "Baseline")

  def test_revenue_kpi_outcome(self):
    analysis = mmm.NonMediaAnalysis(
        non_media_pb.NonMediaAnalysis(
            non_media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.revenue_kpi_outcome,
        mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME),
    )

  def test_revenue_kpi_outcome_missing(self):
    analysis = mmm.NonMediaAnalysis(
        non_media_pb.NonMediaAnalysis(
            non_media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    with self.assertRaisesRegex(
        ValueError, "No revenue-type `KpiOutcome` found"
    ):
      _ = analysis.revenue_kpi_outcome

  def test_non_revenue_kpi_outcome(self):
    analysis = mmm.NonMediaAnalysis(
        non_media_pb.NonMediaAnalysis(
            non_media_outcomes=[
                td.NON_REVENUE_KPI_OUTCOME,
            ]
        )
    )
    self.assertEqual(
        analysis.non_revenue_kpi_outcome,
        mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME),
    )

  def test_non_revenue_kpi_outcome_missing(self):
    analysis = mmm.NonMediaAnalysis(
        non_media_pb.NonMediaAnalysis(
            non_media_outcomes=[
                td.REVENUE_KPI_OUTCOME,
            ]
        )
    )
    with self.assertRaisesRegex(
        ValueError, "No nonrevenue-type `KpiOutcome` found"
    ):
      _ = analysis.non_revenue_kpi_outcome


class ResponseCurveTest(absltest.TestCase):

  def test_input_name(self):
    curve = mmm.ResponseCurve(
        "Channel 1",
        response_curve_pb.ResponseCurve(input_name="Spend"),
    )
    self.assertEqual(curve.input_name, "Spend")

  def test_response_points(self):
    curve = mmm.ResponseCurve(
        "Channel 1",
        response_curve_pb.ResponseCurve(
            response_points=[
                response_curve_pb.ResponsePoint(
                    input_value=1000.0,
                    incremental_kpi=10.0,
                ),
                response_curve_pb.ResponsePoint(
                    input_value=2000.0,
                    incremental_kpi=20.0,
                ),
            ]
        ),
    )
    self.assertEqual(
        curve.response_points,
        [(1000.0, 10.0), (2000.0, 20.0)],
    )


class RevenueOutcomeTest(absltest.TestCase):

  def test_is_revenue_kpi(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertTrue(outcome.is_revenue_kpi)

  def test_is_nonrevenue_kpi(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertFalse(outcome.is_nonrevenue_kpi)

  def test_contribution_pb(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.contribution_pb,
        td.REVENUE_KPI_OUTCOME.contribution,
    )

  def test_effectiveness_pb(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.effectiveness_pb,
        td.REVENUE_KPI_OUTCOME.effectiveness,
    )

  def test_roi_pb(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertEqual(outcome.roi_pb, td.REVENUE_KPI_OUTCOME.roi)

  def test_marginal_roi_pb(self):
    outcome = mmm.KpiOutcome(td.REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.marginal_roi_pb,
        td.REVENUE_KPI_OUTCOME.marginal_roi,
    )


class NonRevenueOutcomeTest(absltest.TestCase):

  def test_is_revenue_kpi(self):
    outcome = mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME)
    self.assertFalse(outcome.is_revenue_kpi)

  def test_is_nonrevenue_kpi(self):
    outcome = mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME)
    self.assertTrue(outcome.is_nonrevenue_kpi)

  def test_contribution_pb(self):
    outcome = mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.contribution_pb,
        td.NON_REVENUE_KPI_OUTCOME.contribution,
    )

  def test_effectiveness_pb(self):
    outcome = mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.effectiveness_pb,
        td.NON_REVENUE_KPI_OUTCOME.effectiveness,
    )

  def test_cost_per_contribution_pb(self):
    outcome = mmm.KpiOutcome(td.NON_REVENUE_KPI_OUTCOME)
    self.assertEqual(
        outcome.cost_per_contribution_pb,
        td.NON_REVENUE_KPI_OUTCOME.cost_per_contribution,
    )


class IncrementalOutcomeGridTest(absltest.TestCase):

  def test_name(self):
    grid = mmm.IncrementalOutcomeGrid(
        budget_pb.IncrementalOutcomeGrid(name="Test")
    )
    self.assertEqual(grid.name, "Test")

  def test_channel_spend_grids(self):
    grid = mmm.IncrementalOutcomeGrid(
        budget_pb.IncrementalOutcomeGrid(
            channel_cells=[
                budget_pb.IncrementalOutcomeGrid.ChannelCells(
                    channel_name="Channel 1",
                    cells=[
                        budget_pb.IncrementalOutcomeGrid.Cell(
                            spend=1000.0,
                            incremental_outcome=estimate_pb.Estimate(
                                value=10.0
                            ),
                        ),
                        budget_pb.IncrementalOutcomeGrid.Cell(
                            spend=2000.0,
                            incremental_outcome=estimate_pb.Estimate(
                                value=20.0
                            ),
                        ),
                    ],
                ),
                budget_pb.IncrementalOutcomeGrid.ChannelCells(
                    channel_name="Channel 2",
                    cells=[
                        budget_pb.IncrementalOutcomeGrid.Cell(
                            spend=1000.0,
                            incremental_outcome=estimate_pb.Estimate(
                                value=10.0
                            ),
                        ),
                        budget_pb.IncrementalOutcomeGrid.Cell(
                            spend=2000.0,
                            incremental_outcome=estimate_pb.Estimate(
                                value=20.0
                            ),
                        ),
                    ],
                ),
            ]
        )
    )
    self.assertEqual(
        grid.channel_spend_grids,
        {
            "Channel 1": [
                (1000.0, 10.0),
                (2000.0, 20.0),
            ],
            "Channel 2": [
                (1000.0, 10.0),
                (2000.0, 20.0),
            ],
        },
    )


class BudgetOptimizationSpecTest(absltest.TestCase):

  def test_date_intervals(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            date_interval=date_interval_pb.DateInterval(
                start_date=date_pb.Date(year=2024, month=1, day=1),
                end_date=date_pb.Date(year=2024, month=1, day=8),
            ),
        )
    )
    self.assertEqual(
        spec.date_interval.date_interval,
        (datetime.date(2024, 1, 1), datetime.date(2024, 1, 8)),
    )

  def test_objective(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            objective=target_metric_pb.TargetMetric.ROI,
        )
    )
    self.assertEqual(spec.objective, target_metric_pb.TargetMetric.ROI)

  def test_is_fixed_scenario_fixed(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            fixed_budget_scenario=budget_pb.FixedBudgetScenario(
                total_budget=1000.0
            ),
        )
    )
    self.assertTrue(spec.is_fixed_scenario)

  def test_is_fixed_scenario_flexible(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            flexible_budget_scenario=budget_pb.FlexibleBudgetScenario(
                total_budget_constraint=constraints_pb.BudgetConstraint(
                    min_budget=100.0,
                    max_budget=1000.0,
                ),
            ),
        )
    )
    self.assertFalse(spec.is_fixed_scenario)

  def test_max_budget_fixed(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            fixed_budget_scenario=budget_pb.FixedBudgetScenario(
                total_budget=1000.0
            ),
        )
    )
    self.assertEqual(spec.max_budget, 1000.0)

  def test_max_budget_flexible(self):
    spec = mmm.BudgetOptimizationSpec(
        budget_pb.BudgetOptimizationSpec(
            flexible_budget_scenario=budget_pb.FlexibleBudgetScenario(
                total_budget_constraint=constraints_pb.BudgetConstraint(
                    min_budget=100.0,
                    max_budget=1000.0,
                ),
            ),
        )
    )
    self.assertEqual(spec.max_budget, 1000.0)


class RfOptimizationSpecTest(absltest.TestCase):

  def test_date_intervals(self):
    spec = mmm.RfOptimizationSpec(
        rf_pb.ReachFrequencyOptimizationSpec(
            date_interval=date_interval_pb.DateInterval(
                start_date=date_pb.Date(year=2024, month=1, day=1),
                end_date=date_pb.Date(year=2024, month=1, day=8),
            ),
        )
    )
    self.assertEqual(
        spec.date_interval.date_interval,
        (datetime.date(2024, 1, 1), datetime.date(2024, 1, 8)),
    )

  def test_objective(self):
    spec = mmm.RfOptimizationSpec(
        rf_pb.ReachFrequencyOptimizationSpec(
            objective=target_metric_pb.TargetMetric.ROI,
        )
    )
    self.assertEqual(spec.objective, target_metric_pb.TargetMetric.ROI)

  def test_total_budget_constraint(self):
    budget_constraint_proto = constraints_pb.BudgetConstraint(
        min_budget=1000.0,
        max_budget=1000.0,
    )
    spec = mmm.RfOptimizationSpec(
        rf_pb.ReachFrequencyOptimizationSpec(
            total_budget_constraint=budget_constraint_proto,
        )
    )
    self.assertEqual(spec.total_budget_constraint, budget_constraint_proto)

  def test_channel_constraints(self):
    channel_constraints = [
        rf_pb.RfChannelConstraint(
            channel_name="Channel 1",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=1000.0,
                max_budget=1000.0,
            ),
            frequency_constraint=constraints_pb.FrequencyConstraint(
                min_frequency=1.0,
                max_frequency=10.0,
            ),
        ),
        rf_pb.RfChannelConstraint(
            channel_name="Channel 2",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=1000.0,
                max_budget=1000.0,
            ),
            frequency_constraint=constraints_pb.FrequencyConstraint(
                min_frequency=2.0,
                max_frequency=12.0,
            ),
        ),
    ]
    spec = mmm.RfOptimizationSpec(
        rf_pb.ReachFrequencyOptimizationSpec(
            rf_channel_constraints=channel_constraints,
        )
    )
    self.assertEqual(spec.channel_constraints, channel_constraints)


class BudgetOptimizationResultTest(absltest.TestCase):

  def test_group_id(self):
    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(name="Test", group_id="group")
    )
    self.assertEqual(result.group_id, "group")

  def test_name(self):
    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(name="Test")
    )
    self.assertEqual(result.name, "Test")

  def test_spec(self):
    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(
            name="Test",
            spec=td.BUDGET_OPTIMIZATION_SPEC_FIXED_ALL_DATES,
        )
    )
    self.assertEqual(
        result.spec.budget_optimization_spec_proto,
        td.BUDGET_OPTIMIZATION_SPEC_FIXED_ALL_DATES,
    )

  def test_optimized_marketing_analysis(self):
    analysis_proto = marketing_pb.MarketingAnalysis(
        date_interval=date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            end_date=date_pb.Date(year=2024, month=1, day=8),
            tag="Tag",
        ),
    )

    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(
            name="Test",
            optimized_marketing_analysis=analysis_proto,
        )
    )
    self.assertEqual(
        result.optimized_marketing_analysis,
        mmm.MarketingAnalysis(analysis_proto),
    )

  def test_incremental_outcome_grid(self):
    grid_proto = budget_pb.IncrementalOutcomeGrid(name="Test")

    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(
            name="Test",
            incremental_outcome_grid=grid_proto,
        )
    )
    self.assertEqual(
        result.incremental_outcome_grid,
        mmm.IncrementalOutcomeGrid(grid_proto),
    )

  def test_response_curves(self):
    curve1 = response_curve_pb.ResponseCurve(
        input_name="Spend",
        response_points=[
            response_curve_pb.ResponsePoint(
                input_value=1000.0,
                incremental_kpi=10.0,
            ),
            response_curve_pb.ResponsePoint(
                input_value=2000.0,
                incremental_kpi=20.0,
            ),
        ],
    )
    curve2 = response_curve_pb.ResponseCurve(
        input_name="Spend",
        response_points=[
            response_curve_pb.ResponsePoint(
                input_value=1010.0,
                incremental_kpi=11.0,
            ),
            response_curve_pb.ResponsePoint(
                input_value=2010.0,
                incremental_kpi=21.0,
            ),
        ],
    )
    result = mmm.BudgetOptimizationResult(
        budget_pb.BudgetOptimizationResult(
            name="Test",
            optimized_marketing_analysis=marketing_pb.MarketingAnalysis(
                media_analyses=[
                    media_pb.MediaAnalysis(
                        channel_name="Channel 1", response_curve=curve1
                    ),
                    media_pb.MediaAnalysis(
                        channel_name="Channel 2", response_curve=curve2
                    ),
                ],
            ),
        )
    )
    self.assertEqual(
        result.response_curves,
        [
            mmm.ResponseCurve("Channel 1", curve1),
            mmm.ResponseCurve("Channel 2", curve2),
        ],
    )


class FrequencyOutcomeGridTest(absltest.TestCase):

  def test_name(self):
    grid = mmm.FrequencyOutcomeGrid(rf_pb.FrequencyOutcomeGrid(name="Test"))
    self.assertEqual(grid.name, "Test")

  def test_channel_frequency_grids(self):
    grid = mmm.FrequencyOutcomeGrid(td.FREQUENCY_OUTCOME_GRID_FOO)
    self.assertEqual(
        grid.channel_frequency_grids,
        {
            "RF Channel 1": [
                (1.0, 100.0),
                (2.0, 200.0),
            ],
            "RF Channel 2": [
                (1.0, 100.0),
                (2.0, 200.0),
            ],
        },
    )


class ReachFrequencyOptimizationResultTest(absltest.TestCase):

  def test_name(self):
    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(name="Test")
    )
    self.assertEqual(result.name, "Test")

  def test_group_id(self):
    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(name="Test", group_id="group")
    )
    self.assertEqual(result.group_id, "group")

  def test_spec(self):
    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(
            name="Test",
            spec=td.RF_OPTIMIZATION_SPEC_ALL_DATES,
        )
    )
    self.assertEqual(
        result.spec.rf_optimization_spec_proto,
        td.RF_OPTIMIZATION_SPEC_ALL_DATES,
    )

  def test_channel_mapped_optimized_frequencies(self):
    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(
            name="Test",
            optimized_channel_frequencies=[
                rf_pb.OptimizedChannelFrequency(
                    channel_name="Channel 1",
                    optimal_average_frequency=1.0,
                ),
                rf_pb.OptimizedChannelFrequency(
                    channel_name="Channel 2",
                    optimal_average_frequency=2.0,
                ),
            ],
        )
    )

    self.assertEqual(
        result.channel_mapped_optimized_frequencies,
        {
            "Channel 1": 1.0,
            "Channel 2": 2.0,
        },
    )

  def test_optimized_marketing_analysis(self):
    analysis_proto = marketing_pb.MarketingAnalysis(
        date_interval=date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            end_date=date_pb.Date(year=2024, month=1, day=8),
            tag="Tag",
        ),
    )

    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(
            name="Test",
            optimized_marketing_analysis=analysis_proto,
        )
    )
    self.assertEqual(
        result.optimized_marketing_analysis,
        mmm.MarketingAnalysis(analysis_proto),
    )

  def test_frequency_outcome_grid(self):
    grid_proto = rf_pb.FrequencyOutcomeGrid(name="Test")
    result = mmm.ReachFrequencyOptimizationResult(
        rf_pb.ReachFrequencyOptimizationResult(
            name="Test",
            frequency_outcome_grid=grid_proto,
        )
    )
    self.assertEqual(
        result.frequency_outcome_grid,
        mmm.FrequencyOutcomeGrid(grid_proto),
    )


if __name__ == "__main__":
  absltest.main()
