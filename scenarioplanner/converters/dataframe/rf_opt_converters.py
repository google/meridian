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

"""Reach and frequency optimization output converters.

This module defines various classes that convert
`ReachFrequencyOptimizationResult`s into flat dataframes.
"""

import abc
from collections.abc import Iterator, Sequence

from meridian import constants as c
from scenarioplanner.converters import mmm
from scenarioplanner.converters.dataframe import common
from scenarioplanner.converters.dataframe import constants as dc
from scenarioplanner.converters.dataframe import converter
import pandas as pd


__all__ = [
    "NamedRfOptimizationGridConverter",
    "RfOptimizationSpecsConverter",
    "RfOptimizationResultsConverter",
]


class _RfOptimizationConverter(converter.Converter, abc.ABC):
  """An abstract class for dealing with `ReachFrequencyOptimizationResult`s."""

  def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
    results = self._mmm.reach_frequency_optimization_results
    if not results:
      return

    # Validate group IDs.
    group_ids = [result.group_id for result in results if result.group_id]
    if len(set(group_ids)) != len(group_ids):
      raise ValueError(
          "Specified group_id must be unique or unset among the given group of"
          " results."
      )

    yield from self._handle_rf_optimization_results(
        self._mmm.reach_frequency_optimization_results
    )

  def _handle_rf_optimization_results(
      self, results: Sequence[mmm.ReachFrequencyOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    raise NotImplementedError()


class NamedRfOptimizationGridConverter(_RfOptimizationConverter):
  """Outputs named tables for Reach & Frequency optimization grids.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for each named incremental outcome grid.
  *   "Channel"
  *   "Frequency"
  *   "ROI"
  For each named R&F optimization result in the MMM output proto.
  """

  def _handle_rf_optimization_results(
      self, results: Sequence[mmm.ReachFrequencyOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    for rf_opt_result in results:
      # There should be one unique ID for each result.
      group_id = str(rf_opt_result.group_id) if rf_opt_result.group_id else ""
      grid = rf_opt_result.frequency_outcome_grid

      # Each grid yields its own data frame table.
      rf_optimization_grid_data = []
      for channel, cells in grid.channel_frequency_grids.items():
        for frequency, outcome in cells:
          rf_optimization_grid_data.append([
              group_id,
              channel,
              frequency,
              outcome,
          ])

      yield (
          common.create_grid_sheet_name(
              dc.RF_OPTIMIZATION_GRID_NAME_PREFIX, grid.name
          ),
          pd.DataFrame(
              rf_optimization_grid_data,
              columns=[
                  dc.OPTIMIZATION_GROUP_ID_COLUMN,
                  dc.OPTIMIZATION_CHANNEL_COLUMN,
                  dc.RF_OPTIMIZATION_GRID_FREQ_COLUMN,
                  dc.RF_OPTIMIZATION_GRID_ROI_OUTCOME_COLUMN,
              ],
          ),
      )


class RfOptimizationSpecsConverter(_RfOptimizationConverter):
  """Outputs a table of R&F optimization specs.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for an R&F frequency outcome grid present in the output.
  *   "Date Interval Start"
  *   "Date Interval End"
  *   "Objective"
  *   "Initial Channel Spend"
  *   "Channel"
  *   "Channel Frequency Min"
  *   "Channel Frequency Max"
  """

  def _handle_rf_optimization_results(
      self, results: Sequence[mmm.ReachFrequencyOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    spec_data = []
    for rf_opt_result in results:
      # There should be one unique ID for each result.
      group_id = str(rf_opt_result.group_id) if rf_opt_result.group_id else ""
      spec = rf_opt_result.spec

      objective = common.map_target_metric_str(spec.objective)
      # These are the start and end dates for the requested R&F optimization in
      # this spec.
      date_interval_start, date_interval_end = (
          d.strftime(c.DATE_FORMAT) for d in spec.date_interval.date_interval
      )
      rf_date_interval = (date_interval_start, date_interval_end)

      # aka historical spend from marketing data in the model kernel
      initial_channel_spends = self._mmm.marketing_data.rf_channel_spends(
          rf_date_interval
      )

      # When the constraint of a channel is not specified, that channel will
      # have a default frequency constraint of `[1.0, max_freq]`.
      #
      # NOTE: We assume that the processor has already done this max_freq
      # computation. And so we can assert here that channel constraints are
      # always fully specified for R&F channels.
      channel_constraints = spec.channel_constraints
      if not channel_constraints:
        raise ValueError(
            "R&F optimization spec must have channel constraints specified."
        )
      if set([
          channel_constraint.channel_name
          for channel_constraint in channel_constraints
      ]) != set(self._mmm.marketing_data.rf_channels):
        raise ValueError(
            "R&F optimization spec must have channel constraints specified for"
            " all R&F channels."
        )

      for channel_constraint in channel_constraints:
        min_freq = channel_constraint.frequency_constraint.min_frequency or 1.0
        max_freq = channel_constraint.frequency_constraint.max_frequency
        if not max_freq:
          raise ValueError(
              "Channel constraint in R&F optimization spec must have max"
              " frequency specified. Missing for channel:"
              f" {channel_constraint.channel_name}"
          )
        spec_data.append([
            group_id,
            date_interval_start,
            date_interval_end,
            objective,
            initial_channel_spends.get(channel_constraint.channel_name, 0.0),
            channel_constraint.channel_name,
            min_freq,
            max_freq,
        ])

    yield (
        dc.RF_OPTIMIZATION_SPECS,
        pd.DataFrame(
            spec_data,
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


class RfOptimizationResultsConverter(_RfOptimizationConverter):
  """Outputs a table of R&F optimization results.

  When called, this converter returns a data frame with the columns:

  *   "Group ID"
      A UUID generated for a budget optimization result present in the output.
  *   "Channel"
  *   "Is Revenue KPI"
      Whether the KPI is revenue or not.
  *   "Initial Spend"
  *   "Optimal Avg Frequency"
  *   "Optimal Impression Effectiveness"
  *   "Optimal ROI"
  *   "Optimal mROI"
  *   "Optimal CPC"
  """

  def _handle_rf_optimization_results(
      self, results: Sequence[mmm.ReachFrequencyOptimizationResult]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    data = []
    for rf_opt_result in results:
      group_id = str(rf_opt_result.group_id) if rf_opt_result.group_id else ""
      marketing_analysis = rf_opt_result.optimized_marketing_analysis

      spec = rf_opt_result.spec
      # These are the start and end dates for the requested R&F optimization in
      # this spec.
      date_interval_start, date_interval_end = (
          d.strftime(c.DATE_FORMAT) for d in spec.date_interval.date_interval
      )
      rf_date_interval = (date_interval_start, date_interval_end)
      # aka historical spend from marketing data in the model kernel
      initial_budget = self._mmm.marketing_data.rf_channel_spends(
          rf_date_interval
      )

      media_channel_analyses = marketing_analysis.channel_mapped_media_analyses
      for channel, media_analysis in media_channel_analyses.items():
        # Skip "All Channels" pseudo-channel.
        if channel == c.ALL_CHANNELS:
          continue
        # Skip non-R&F channels.
        if channel not in self._mmm.marketing_data.rf_channels:
          continue

        initial_spend = initial_budget[channel]
        optimal_avg_freq = rf_opt_result.channel_mapped_optimized_frequencies[
            channel
        ]

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
              initial_spend,
              optimal_avg_freq,
              effectiveness,
              roi,
              mroi,
              cpc,
          ])

        _append_outcome_data(revenue_outcome, True)
        _append_outcome_data(nonrevenue_outcome, False)
        # pylint: enable=cell-var-from-loop

    yield (
        dc.RF_OPTIMIZATION_RESULTS,
        pd.DataFrame(
            data,
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


CONVERTERS = [
    NamedRfOptimizationGridConverter,
    RfOptimizationSpecsConverter,
    RfOptimizationResultsConverter,
]
