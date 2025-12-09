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

"""Budget optimization converters.

This module defines various classes that convert `BudgetOptimizationResult`s
into flat dataframes.
"""

import abc
from collections.abc import Iterator, Sequence

from meridian import constants as c
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from scenarioplanner.converters import mmm
from scenarioplanner.converters.dataframe import common
from scenarioplanner.converters.dataframe import constants as dc
from scenarioplanner.converters.dataframe import converter
import pandas as pd


__all__ = [
    "NamedOptimizationGridConverter",
    "BudgetOptimizationSpecsConverter",
    "BudgetOptimizationResultsConverter",
    "BudgetOptimizationResponseCurvesConverter",
]


class _BudgetOptimizationConverter(converter.Converter, abc.ABC):
  """An abstract class for dealing with `BudgetOptimizationResult`s."""

  def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
    results = self._mmm.budget_optimization_results
    if not results:
      return

    # Validate group IDs.
    group_ids = [result.group_id for result in results if result.group_id]
    if len(set(group_ids)) != len(group_ids):
      raise ValueError(
          "Specified group_id must be unique or unset among the given group of"
          " results."
      )

    yield from self._handle_budget_optimization_results(
        self._mmm.budget_optimization_results
    )

  def _handle_budget_optimization_results(
      self, results: Sequence[mmm.BudgetOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    raise NotImplementedError()


class NamedOptimizationGridConverter(_BudgetOptimizationConverter):
  """Outputs named tables for budget optimization grids.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for each named incremental outcome grid.
  *   "Channel"
  *   "Spend"
  *   "Incremental Outcome"
  For each named budget optimization result in the MMM output proto.
  """

  def _handle_budget_optimization_results(
      self, results: Sequence[mmm.BudgetOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    for budget_opt_result in results:
      # There should be one unique ID for each result.
      group_id = (
          str(budget_opt_result.group_id) if budget_opt_result.group_id else ""
      )
      grid = budget_opt_result.incremental_outcome_grid

      # Each grid yields its own data frame table.
      optimization_grid_data = []
      for channel, cells in grid.channel_spend_grids.items():
        for spend, incremental_outcome in cells:
          optimization_grid_data.append([
              group_id,
              channel,
              spend,
              incremental_outcome,
          ])

      yield (
          common.create_grid_sheet_name(
              dc.OPTIMIZATION_GRID_NAME_PREFIX, grid.name
          ),
          pd.DataFrame(
              optimization_grid_data,
              columns=[
                  dc.OPTIMIZATION_GROUP_ID_COLUMN,
                  dc.OPTIMIZATION_CHANNEL_COLUMN,
                  dc.OPTIMIZATION_GRID_SPEND_COLUMN,
                  dc.OPTIMIZATION_GRID_INCREMENTAL_OUTCOME_COLUMN,
              ],
          ),
      )


class BudgetOptimizationSpecsConverter(_BudgetOptimizationConverter):
  """Outputs a table of budget optimization specs.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for an incremental outcome grid present in the output.
  *   "Date Interval Start"
  *   "Date Interval End"
  *   "Analysis Period"
  *   "Objective"
  *   "Scenario Type"
  *   "Initial Channel Spend"
  *   "Target Metric Constraint"
      None if scenario type is "Fixed"
  *   "Target Metric Value"
      None if scenario type is "Fixed"
  *   "Channel"
  *   "Channel Spend Min"
  *   "Channel Spend Max"
  """

  def _handle_budget_optimization_results(
      self, results: Sequence[mmm.BudgetOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    spec_data = []
    for budget_opt_result in results:
      # There should be one unique ID for each result.
      group_id = (
          str(budget_opt_result.group_id) if budget_opt_result.group_id else ""
      )
      spec = budget_opt_result.spec

      objective = common.map_target_metric_str(spec.objective)
      # These are the start and end dates for the requested budget optimization
      # in this spec.
      date_interval_start, date_interval_end = (
          d.strftime(c.DATE_FORMAT) for d in spec.date_interval.date_interval
      )
      budget_date_interval = (date_interval_start, date_interval_end)

      # aka historical spend from marketing data in the model kernel
      initial_channel_spends = self._mmm.marketing_data.all_channel_spends(
          budget_date_interval
      )

      scenario = (
          dc.OPTIMIZATION_SPEC_SCENARIO_FIXED
          if spec.is_fixed_scenario
          else dc.OPTIMIZATION_SPEC_SCENARIO_FLEXIBLE
      )

      if spec.is_fixed_scenario:
        target_metric_constraint = None
        target_metric_value = None
      else:
        flexible_scenario = (
            spec.budget_optimization_spec_proto.flexible_budget_scenario
        )
        # Meridian flexible budget spec only has one target metric constraint.
        target_metric_constraint_pb = (
            flexible_scenario.target_metric_constraints[0]
        )
        target_metric_constraint = common.map_target_metric_str(
            target_metric_constraint_pb.target_metric
        )
        target_metric_value = target_metric_constraint_pb.target_value

      # When the constraint of a channel is not specified, that channel will
      # have a constraint of `[0, max_budget]` which is equivalent to no
      # constraint.
      #
      # Here, `max_budget` is the total budget for a fixed scenario spec, or the
      # max budget upper bound for a flexible scenario spec.
      #
      # NOTE: This assumption must be in line with what the budget optimization
      # processor does with an empty channel constraints list.
      channel_constraints = spec.channel_constraints
      if not channel_constraints:
        # Implicit channel constraints; synthesize them first before proceeding.
        channel_constraints = [
            budget_pb.ChannelConstraint(
                channel_name=channel_name,
                budget_constraint=constraints_pb.BudgetConstraint(
                    min_budget=0.0,
                    max_budget=spec.max_budget,
                ),
            )
            for channel_name in self._mmm.marketing_data.media_channels
        ]

      for channel_constraint in channel_constraints:
        spec_data.append([
            group_id,
            date_interval_start,
            date_interval_end,
            spec.date_interval_tag,
            objective,
            scenario,
            initial_channel_spends.get(channel_constraint.channel_name, 0.0),
            target_metric_constraint,
            target_metric_value,
            channel_constraint.channel_name,
            channel_constraint.budget_constraint.min_budget,
            channel_constraint.budget_constraint.max_budget,
        ])

    yield (
        dc.OPTIMIZATION_SPECS,
        pd.DataFrame(
            spec_data,
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


class BudgetOptimizationResultsConverter(_BudgetOptimizationConverter):
  """Outputs a table of budget optimization results objectives.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for a budget optimization result present in the output.
  *   "Channel"
  *   "Is Revenue KPI"
      Whether the KPI is revenue or not.
  *   "Optimal Spend"
  *   "Optimal Spend Share"
  *   "Optimal Impression Effectiveness"
  *   "Optimal ROI"
  *   "Optimal mROI"
  *   "Optimal CPC"
  """

  def _handle_budget_optimization_results(
      self, results: Sequence[mmm.BudgetOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    data = []

    for budget_opt_result in results:
      group_id = (
          str(budget_opt_result.group_id) if budget_opt_result.group_id else ""
      )
      marketing_analysis = budget_opt_result.optimized_marketing_analysis

      media_channel_analyses = marketing_analysis.channel_mapped_media_analyses
      for channel, media_analysis in media_channel_analyses.items():
        # Skip "All Channels" pseudo-channel.
        if channel == c.ALL_CHANNELS:
          continue

        spend = media_analysis.spend_info_pb.spend
        spend_share = media_analysis.spend_info_pb.spend_share

        revenue_outcome = media_analysis.maybe_revenue_outcome
        nonrevenue_outcome = media_analysis.maybe_non_revenue_outcome

        # pylint: disable=cell-var-from-loop
        def _append_outcome_data(
            outcome: mmm.Outcome | None,
            is_revenue_kpi: bool,
        ) -> None:
          if outcome is None:
            return
          effectiveness = outcome.effectiveness_pb.value.value
          roi = outcome.roi_pb.value
          mroi = outcome.marginal_roi_pb.value
          cpc = outcome.cost_per_contribution_pb.value
          data.append([
              group_id,
              channel,
              is_revenue_kpi,
              spend,
              spend_share,
              effectiveness,
              roi,
              mroi,
              cpc,
          ])

        _append_outcome_data(revenue_outcome, True)
        _append_outcome_data(nonrevenue_outcome, False)
        # pylint: enable=cell-var-from-loop

    yield (
        dc.OPTIMIZATION_RESULTS,
        pd.DataFrame(
            data,
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


class BudgetOptimizationResponseCurvesConverter(_BudgetOptimizationConverter):
  """Outputs a table of budget optimization response curves.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for a budget optimization result present in the output.
  *   "Channel"
  *   "Spend"
  *   "Incremental Outcome"
  """

  def _handle_budget_optimization_results(
      self, results: Sequence[mmm.BudgetOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    response_curve_data = []
    for budget_opt_result in results:
      group_id = (
          str(budget_opt_result.group_id) if budget_opt_result.group_id else ""
      )
      curves = budget_opt_result.response_curves
      for curve in curves:
        for spend, incremental_outcome in curve.response_points:
          response_curve_data.append([
              group_id,
              curve.channel_name,
              spend,
              incremental_outcome,
          ])

    yield (
        dc.OPTIMIZATION_RESPONSE_CURVES,
        pd.DataFrame(
            response_curve_data,
            columns=[
                dc.OPTIMIZATION_GROUP_ID_COLUMN,
                dc.OPTIMIZATION_CHANNEL_COLUMN,
                dc.OPTIMIZATION_GRID_SPEND_COLUMN,
                dc.OPTIMIZATION_GRID_INCREMENTAL_OUTCOME_COLUMN,
            ],
        ),
    )


CONVERTERS = [
    NamedOptimizationGridConverter,
    BudgetOptimizationSpecsConverter,
    BudgetOptimizationResultsConverter,
    BudgetOptimizationResponseCurvesConverter,
]
