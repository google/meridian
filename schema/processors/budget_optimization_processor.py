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

"""Defines a processor for budget optimization inference on a Meridian model.

This module provides the `BudgetOptimizationProcessor` class, which is used to
perform marketing budget optimization based on a trained Meridian model. The
processor takes a trained model and a `BudgetOptimizationSpec` object,
which defines the optimization parameters, constraints, and scenarios.

The optimization process aims to find the optimal allocation of budget across
different media channels to maximize a specified objective, such as Key
Performance Indicator (KPI) or Revenue, subject to various constraints.

Key Features:

-   Supports both fixed and flexible budget scenarios.
-   Allows setting channel-level budget constraints, either as absolute values
    or relative to historical spend.
-   Generates detailed optimization results, including optimal spends, expected
    outcomes, and response curves.
-   Outputs results in a structured protobuf format (`BudgetOptimization`).

Key Classes:

-   `BudgetOptimizationSpec`: Dataclass to specify optimization parameters and
    constraints.
-   `BudgetOptimizationProcessor`: The main processor class to execute budget
    optimization.

Example Usage:

1.  **Fixed Budget Optimization:**
    Optimize budget allocation for a fixed total budget, aiming to maximize KPI.

    ```python
    from schema.processors import budget_optimization_processor
    from meridian.analysis import optimizer
    from schema.processors import common

    # Assuming 'trained_model' is a loaded Meridian model object

    spec = budget_optimization_processor.BudgetOptimizationSpec(
        optimization_name="fixed_budget_scenario_1",
        scenario=optimizer.FixedBudgetScenario(total_budget=1000000),
        kpi_type=common.KpiType.REVENUE, # Or common.KpiType.NON_REVENUE
        # Optional: Add channel constraints
        constraints=[
            budget_optimization_processor.ChannelConstraintRel(
                channel_name="channel_a",
                spend_constraint_lower=0.1, # Allow 10% decrease
                spend_constraint_upper=0.5  # Allow 50% increase
            ),
            budget_optimization_processor.ChannelConstraintRel(
                channel_name="channel_b",
                spend_constraint_lower=0.0, # No decrease
                spend_constraint_upper=1.0  # Allow 100% increase
            )
        ],
        include_response_curves=True,
    )

    processor = budget_optimization_processor.BudgetOptimizationProcessor(
        trained_model
    )
    # result is a `budget_pb.BudgetOptimization` proto
    result = processor.execute([spec])
    ```

2.  **Flexible Budget Optimization:**
    Optimize budget to achieve a target Return on Investment (ROI).

    ```python
    from schema.processors import budget_optimization_processor
    from meridian.analysis import optimizer
    from schema.processors import common
    import meridian.constants as c

    # Assuming 'trained_model' is a loaded Meridian model object

    spec = budget_optimization_processor.BudgetOptimizationSpec(
        optimization_name="flexible_roi_target",
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.ROI,
            target_value=3.5  # Target ROI of 3.5
        ),
        kpi_type=common.KpiType.REVENUE,
        date_interval_tag="optimization_period",
        # Skip response curves for faster computation.
        include_response_curves=False,
    )

    processor = budget_optimization_processor.BudgetOptimizationProcessor(
        trained_model
    )
    result = processor.execute([spec])
    ```

Note: You can provide the processor with multiple specs. This would result in
a `BudgetOptimization` output with multiple results therein.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import TypeAlias

from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.data import time_coordinates as tc
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2 as analysis_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_analysis_pb
from mmm.v1.marketing.analysis import outcome_pb2 as outcome_pb
from mmm.v1.marketing.analysis import response_curve_pb2 as response_curve_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from schema.processors import common
from schema.processors import model_processor
from schema.utils import time_record
import numpy as np
from typing_extensions import override
import xarray as xr

__all__ = [
    'BudgetOptimizationProcessor',
    'BudgetOptimizationSpec',
    'ChannelConstraintAbs',
    'ChannelConstraintRel',
]


# Default lower and upper bounds (as _relative_ ratios) for channel constraints.
CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO = 1
CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO = 2


@dataclasses.dataclass(frozen=True)
class ChannelConstraintAbs:
  """A budget constraint on a channel.

  Constraint attributes in this dataclass are absolute values. Useful to
  represent resolved absolute constraint values in an output spec metadata.

  Attributes:
    channel_name: The name of the channel.
    abs_lowerbound: A simple absolute lower bound value for a channel's spend.
    abs_upperbound: A simple absolute upper bound value for a channel's spend.
  """

  channel_name: str
  abs_lowerbound: float
  abs_upperbound: float

  def to_proto(self) -> budget_pb.ChannelConstraint:
    return budget_pb.ChannelConstraint(
        channel_name=self.channel_name,
        budget_constraint=constraints_pb.BudgetConstraint(
            min_budget=self.abs_lowerbound,
            max_budget=self.abs_upperbound,
        ),
    )


@dataclasses.dataclass(frozen=True)
class ChannelConstraintRel:
  """A budget constraint on a channel.

  Constraint attributes in this dataclass are relative ratios. Useful for user
  input spec.

  Attributes:
    channel_name: The name of the channel.
    spend_constraint_lower: The spend constraint lower of a channel is the
      change in ratio w.r.t. the channel's historical spend. The absolute lower
      bound value is equal to `(1 - spend_constraint_lower) *
      hist_channel_spend)`. The value must be between `[0, 1]`.
    spend_constraint_upper: The spend constraint upper of a channel is the
      change in ratio w.r.t. the channel's historical spend. The absolute upper
      bound value is equal to `(1 + spend_constraint_upper) *
      hist_channel_spend)`. The value must be non-negative.
  """

  channel_name: str
  spend_constraint_lower: float
  spend_constraint_upper: float

  def __post_init__(self):
    if self.spend_constraint_lower < 0:
      raise ValueError('Spend constraint lower must be non-negative.')
    if self.spend_constraint_lower > 1:
      raise ValueError('Spend constraint lower must not be greater than 1.')
    if self.spend_constraint_upper < 0:
      raise ValueError('Spend constraint upper must be non-negative.')


ChannelConstraint: TypeAlias = ChannelConstraintAbs | ChannelConstraintRel


@dataclasses.dataclass(frozen=True, kw_only=True)
class BudgetOptimizationSpec(model_processor.OptimizationSpec):
  """Spec dataclass for marketing budget optimization processor.

  This spec is used both as user input to inform the budget optimization
  processor of its constraints and parameters, as well as an output structure
  that is serializable to a `BudgetOptimizationSpec` proto. The latter serves
  as a metadata embedded in a `BudgetOptimizationResult`.

  Attributes:
    objective: Always defined as KPI.
    scenario: The optimization scenario (whether fixed or flexible).
    constraints: Per-channel budget constraints. Defaults to relative
      constraints `[1, 2]` for spend_constraint_lower and spend_constraint_upper
      if not specified.
    kpi_type: A `common.KpiType` enum denoting whether the optimized KPI is of a
      `'revenue'` or `'non-revenue'` type.
    grid: The optimization grid to use for the optimization. If None, a new grid
      will be created within the optimizer.
    include_response_curves: Whether to include response curves in the output.
      Setting this to `False` improves performance if only optimization result
      is needed.
    new_data: The new data to use for the optimization. If None, the training
      data will be used.
  """

  scenario: optimizer.FixedBudgetScenario | optimizer.FlexibleBudgetScenario = (
      dataclasses.field(default_factory=optimizer.FixedBudgetScenario)
  )
  constraints: Sequence[ChannelConstraint] = dataclasses.field(
      default_factory=list
  )
  kpi_type: common.KpiType = common.KpiType.REVENUE
  grid: optimizer.OptimizationGrid | None = None
  include_response_curves: bool = True
  new_data: analyzer.DataTensors | None = None

  @property
  def objective(self) -> common.TargetMetric:
    """A Meridian budget optimization objective is always KPI."""
    return common.TargetMetric.KPI

  @override
  def validate(self):
    super().validate()
    if (self.new_data is not None) and (self.new_data.time is None):
      raise ValueError('`time` must be provided in `new_data`.')

  # TODO: Populate `new_marketing_data`.
  def to_proto(self) -> budget_pb.BudgetOptimizationSpec:
    # When invoked as an output proto, the spec should have been fully resolved.
    if self.start_date is None or self.end_date is None:
      raise ValueError(
          'Start and end dates must be resolved before this spec can be'
          ' serialized.'
      )

    proto = budget_pb.BudgetOptimizationSpec(
        date_interval=time_record.create_date_interval_pb(
            self.start_date, self.end_date, tag=self.date_interval_tag
        ),
        objective=self.objective.value,
        kpi_type=(
            kpi_type_pb.KpiType.REVENUE
            if self.kpi_type == common.KpiType.REVENUE
            else kpi_type_pb.KpiType.NON_REVENUE
        ),
    )

    match self.scenario:
      case optimizer.FixedBudgetScenario(total_budget):
        if total_budget is None:
          raise ValueError(
              'Total budget must be resolved before this spec can be serialized'
          )
        proto.fixed_budget_scenario.total_budget = total_budget
      case optimizer.FlexibleBudgetScenario(target_metric, target_value):
        proto.flexible_budget_scenario.target_metric_constraints.append(
            constraints_pb.TargetMetricConstraint(
                target_metric=_target_metric_to_proto(target_metric),
                target_value=target_value,
            )
        )
      case _:
        raise ValueError('Unsupported scenario type.')

    for channel_constraint in self.constraints:
      # When invoked as an output proto, the spec's constraints must have been
      # resolved to absolute values.
      if not isinstance(channel_constraint, ChannelConstraintAbs):
        raise ValueError(
            'Channel constraints must be resolved to absolute values before'
            ' this spec can be serialized.'
        )

      proto.channel_constraints.append(
          budget_pb.ChannelConstraint(
              channel_name=channel_constraint.channel_name,
              budget_constraint=constraints_pb.BudgetConstraint(
                  min_budget=channel_constraint.abs_lowerbound,
                  max_budget=channel_constraint.abs_upperbound,
              ),
          )
      )

    return proto


class BudgetOptimizationProcessor(
    model_processor.ModelProcessor[
        BudgetOptimizationSpec, budget_pb.BudgetOptimization
    ],
):
  """A Processor for marketing budget optimization."""

  def __init__(
      self,
      trained_model: model_processor.ModelType,
  ):
    self._trained_model = model_processor.ensure_trained_model(trained_model)
    self._internal_analyzer = self._trained_model.internal_analyzer
    self._internal_optimizer = self._trained_model.internal_optimizer

  @classmethod
  def spec_type(cls) -> type[BudgetOptimizationSpec]:
    return BudgetOptimizationSpec

  @classmethod
  def output_type(cls) -> type[budget_pb.BudgetOptimization]:
    return budget_pb.BudgetOptimization

  def _set_output(self, output: pb.Mmm, result: budget_pb.BudgetOptimization):
    output.marketing_optimization.budget_optimization.CopyFrom(result)

  def execute(
      self, specs: Sequence[BudgetOptimizationSpec]
  ) -> budget_pb.BudgetOptimization:
    output = budget_pb.BudgetOptimization()

    group_ids = [spec.group_id for spec in specs if spec.group_id]
    if len(set(group_ids)) != len(group_ids):
      raise ValueError(
          'Specified group_id must be unique among the given group of specs.'
      )

    # For each given spec:
    # 1. Run optimize, which computes channel outcomes and their optimal spends.
    # 2. Run _create_grids, which creates incremental spend outcome grids.
    # 3. Compile the final BudgetOptimization proto.
    for spec in specs:
      kwargs = build_scenario_kwargs(spec.scenario)
      constraints_kwargs = build_constraints_kwargs(
          spec.constraints,
          self._trained_model.mmm.input_data.get_all_paid_channels(),
      )
      kwargs.update(constraints_kwargs)
      if spec.new_data is not None and spec.new_data.time is not None:
        time_coords = tc.TimeCoordinates.from_dates(
            [s.decode() for s in np.asarray(spec.new_data.time)]
        )
      else:
        time_coords = self._trained_model.time_coordinates
      resolver = spec.resolver(time_coords)
      start_date, end_date = resolver.to_closed_date_interval_tuple()

      # Note that `optimize()` maximises KPI if the input data is non-revenue
      # and the user selected `use_kpi=True`. Otherwise, it maximizes revenue.
      opt_result = self._internal_optimizer.optimize(
          start_date=start_date,
          end_date=end_date,
          fixed_budget=isinstance(spec.scenario, optimizer.FixedBudgetScenario),
          confidence_level=spec.confidence_level,
          use_kpi=(spec.kpi_type == common.KpiType.NON_REVENUE),
          optimization_grid=spec.grid,
          new_data=spec.new_data,
          **kwargs,
      )

      output.results.append(
          self._to_budget_optimization_result(
              spec, opt_result, resolver, **constraints_kwargs
          )
      )

    return output

  def _to_budget_optimization_result(
      self,
      spec: BudgetOptimizationSpec,
      opt_result: optimizer.OptimizationResults,
      resolver: model_processor.DatedSpecResolver,
      spend_constraint_lower: Sequence[float],
      spend_constraint_upper: Sequence[float],
  ) -> budget_pb.BudgetOptimizationResult:
    """Converts an optimizer result to a BudgetOptimizationResult proto.

    Args:
      spec: The spec used to generate the oiptimization result..
      opt_result: The result of the optimization.
      resolver: A DatedSpecResolver instance.
      spend_constraint_lower: A sequence of lower bound constraints for each
        channel, in relative terms.
      spend_constraint_upper: A sequence of upper bound constraints for each
        channel, in relative terms.

    Returns:
      A BudgetOptimizationResult proto.
    """
    # Copy the current spec, and resolve its date interval.
    start, end = resolver.resolve_to_date_interval_open_end()

    # Resolve the given (input) spec to an (output) spec: the latter features
    # dates and absolute channel constraints resolution.
    spec = dataclasses.replace(
        spec,
        start_date=start,
        end_date=end,
        constraints=_get_channel_constraints_abs(
            opt_result=opt_result,
            constraint_lower=spend_constraint_lower,
            constraint_upper=spend_constraint_upper,
        ),
    )

    # If the spec is a fixed budget scenario, but the total budget is not
    # specified, then set it to the budget amount used in the optimization.
    resolve_historical_budget = (
        isinstance(spec.scenario, optimizer.FixedBudgetScenario)
        and spec.scenario.total_budget is None
    )
    if resolve_historical_budget:
      spec = dataclasses.replace(
          spec,
          scenario=optimizer.FixedBudgetScenario(
              total_budget=opt_result.optimized_data.attrs[c.BUDGET]
          ),
      )

    xr_response_curves = (
        opt_result.get_response_curves()
        if spec.include_response_curves
        else None
    )
    optimized_marketing_analysis = to_marketing_analysis(
        spec=spec,
        xr_data=opt_result.optimized_data,
        xr_response_curves=xr_response_curves,
    )
    nonoptimized_marketing_analysis = to_marketing_analysis(
        spec=spec,
        xr_data=opt_result.nonoptimized_data,
        xr_response_curves=xr_response_curves,
    )
    result = budget_pb.BudgetOptimizationResult(
        name=spec.optimization_name,
        spec=spec.to_proto(),
        optimized_marketing_analysis=optimized_marketing_analysis,
        nonoptimized_marketing_analysis=nonoptimized_marketing_analysis,
        incremental_outcome_grid=_to_incremental_outcome_grid(
            opt_result.optimization_grid.grid_dataset,
            grid_name=spec.grid_name,
        ),
    )

    if spec.group_id:
      result.group_id = spec.group_id
    return result


def to_marketing_analysis(
    spec: model_processor.DatedSpec,
    xr_data: xr.Dataset,
    xr_response_curves: xr.Dataset | None,
) -> analysis_pb.MarketingAnalysis:
  """Converts OptimizationResults to MarketingAnalysis protos.

  Args:
    spec: The spec to build MarketingAnalysis protos for.
    xr_data: The xr.Dataset to convert into MarketingAnalysis proto.
    xr_response_curves: The xr.Dataset to convert into response curves.

  Returns:
    A MarketingAnalysis proto.
  """
  # `spec` should have been resolved with concrete date interval parameters.
  assert spec.start_date is not None and spec.end_date is not None
  marketing_analysis = analysis_pb.MarketingAnalysis(
      date_interval=time_record.create_date_interval_pb(
          start_date=spec.start_date,
          end_date=spec.end_date,
          tag=spec.date_interval_tag,
      ),
  )
  # Include the response curves data for all channels at the optimized freq.
  channel_response_curve_protos = _to_channel_response_curve_protos(
      xr_response_curves
  )

  # Create a per-channel MediaAnalysis.
  for channel in xr_data.channel.values:
    channel_data = xr_data.sel(channel=channel)
    spend = channel_data.spend.item()
    # TODO: Resolve conflict definition of spend share.
    spend_share = channel_data.pct_of_spend.item()
    channel_media_analysis = media_analysis_pb.MediaAnalysis(
        channel_name=channel,
        spend_info=media_analysis_pb.SpendInfo(
            spend=spend,
            spend_share=spend_share,
        ),
    )
    # Output one outcome per channel: either revenue or non-revenue,
    # but not both.
    channel_media_analysis.media_outcomes.append(_to_outcome(channel_data))
    if xr_response_curves is not None:
      channel_media_analysis.response_curve.CopyFrom(
          channel_response_curve_protos[channel]
      )
    marketing_analysis.media_analyses.append(channel_media_analysis)

  return marketing_analysis


def _get_channel_constraints_abs(
    opt_result: optimizer.OptimizationResults,
    constraint_lower: Sequence[float],
    constraint_upper: Sequence[float],
) -> list[ChannelConstraintAbs]:
  """Converts a sequence of channel constraints in relative terms to absolute ones.

  Args:
    opt_result: The optimization result.
    constraint_lower: A sequence of lower bound constraints for each channel, in
      relative terms.
    constraint_upper: A sequence of upper bound constraints for each channel, in
      relative terms.

  Returns:
    A list of channel constraints in absolute terms.
  """
  round_factor = opt_result.optimization_grid.round_factor
  channels = opt_result.optimized_data.channel.values
  (optimization_lower_bound, optimization_upper_bound) = (
      optimizer.get_optimization_bounds(
          n_channels=len(channels),
          spend=opt_result.nonoptimized_data.spend.data,
          round_factor=round_factor,
          spend_constraint_lower=constraint_lower,
          spend_constraint_upper=constraint_upper,
      )
  )

  abs_constraints: list[ChannelConstraintAbs] = []
  for i, channel in enumerate(channels):
    constraint = ChannelConstraintAbs(
        channel_name=channel,
        abs_lowerbound=optimization_lower_bound[i],
        abs_upperbound=optimization_upper_bound[i],
    )
    abs_constraints.append(constraint)
  return abs_constraints


def build_scenario_kwargs(
    scenario: optimizer.FixedBudgetScenario | optimizer.FlexibleBudgetScenario,
) -> dict[str, float]:
  """Returns keyword arguments for an optimizer, given a spec's scenario.

  The keys in the returned kwargs are a subset of the parameters in
  `optimizer.BudgetOptimizer.optimize()` method.

  Args:
    scenario: The scenario to build kwargs for.

  Raises:
    ValueError: If no scenario is specified in the spec, or if for a given
    scenario type, its values are invalid.
  """
  kwargs = {}
  match scenario:
    case optimizer.FixedBudgetScenario(total_budget):
      if total_budget is not None:  # if not specified => historical spend
        kwargs['budget'] = total_budget
    case optimizer.FlexibleBudgetScenario(target_metric, target_value):
      match target_metric:
        case c.ROI:
          key = 'target_roi'
        case c.MROI:
          key = 'target_mroi'
        case _:
          # Technically dead code, since this is already checked in `validate()`
          raise ValueError(
              f'Unsupported target metric: {target_metric} for flexible'
              ' budget scenario.'
          )
      kwargs[key] = target_value
    case _:
      # Technically dead code.
      raise ValueError('Unsupported scenario type.')
  return kwargs


def build_constraints_kwargs(
    constraints: Sequence[ChannelConstraint],
    model_channels: Sequence[str],
) -> dict[str, list[float]]:
  """Returns `spend_constraint_**` kwargs for given channel constraints.

  If a media channel is not present in the spec's channel constraints, then
  its spend constraint is implied to be the max budget of the spec's scenario.

  Args:
    constraints: The channel constraints from the spec.
    model_channels: The list of channels in the model.

  Raises:
    ValueError: If the channel constraints are invalid (e.g. channel names are
      not matched with the internal model data, etc).
  """
  # Validate user-configured channel constraints in the spec.
  constraints_by_channel_name = {c.channel_name: c for c in constraints}
  constraint_channel_names = set(constraints_by_channel_name.keys())
  if not (constraint_channel_names <= set(model_channels)):
    raise ValueError(
        'Channel constraints must have channel names that are in the model'
        f' data. Expected {model_channels}, got {constraint_channel_names}.'
    )

  spend_constraint_lower = []
  spend_constraint_upper = []
  for channel in model_channels:
    if channel in constraints_by_channel_name:
      constraint = constraints_by_channel_name[channel]
      if not isinstance(constraint, ChannelConstraintRel):
        raise ValueError(
            'Channel constraints in user input must be expressed in relative'
            ' ratio terms.'
        )
      lowerbound = constraint.spend_constraint_lower
      upperbound = constraint.spend_constraint_upper
    else:
      lowerbound = CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO
      upperbound = CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO

    spend_constraint_lower.append(lowerbound)
    spend_constraint_upper.append(upperbound)

  return {
      'spend_constraint_lower': spend_constraint_lower,
      'spend_constraint_upper': spend_constraint_upper,
  }


def _to_channel_response_curve_protos(
    optimized_response_curves: xr.Dataset | None,
) -> Mapping[str, response_curve_pb.ResponseCurve]:
  """Converts a response curve dataframe to a map of channel to ResponseCurve.

  Args:
    optimized_response_curves: A dataframe containing the response curve data.
      This is the output of `OptimizationResults.get_response_curves()`.

  Returns:
    A map of channel to ResponseCurve proto.
  """
  if optimized_response_curves is None:
    return {}
  channels = optimized_response_curves.channel.values
  # Flatten the dataset into a tabular dataframe so we can iterate over it.
  df = (
      optimized_response_curves.to_dataframe()
      .reset_index()
      .pivot(
          index=[c.CHANNEL, c.SPEND, c.SPEND_MULTIPLIER],
          columns=c.METRIC,
          values=c.INCREMENTAL_OUTCOME,
      )
      .reset_index()
  ).sort_values(by=[c.CHANNEL, c.SPEND])

  channel_response_curves = {
      channel: response_curve_pb.ResponseCurve(input_name=c.SPEND)
      for channel in channels
  }

  for _, row in df.iterrows():
    channel = row[c.CHANNEL]
    response_point = response_curve_pb.ResponsePoint(
        input_value=row[c.SPEND],
        incremental_kpi=row[c.MEAN],
    )
    channel_response_curves[channel].response_points.append(response_point)

  return channel_response_curves


def _to_outcome(channel_data: xr.Dataset) -> outcome_pb.Outcome:
  """Returns an Outcome value for a given channel's media analysis.

  Args:
    channel_data: A channel-selected dataset from `OptimizationResults`.
  """
  confidence_level = channel_data.attrs[c.CONFIDENCE_LEVEL]
  is_revenue_kpi = channel_data.attrs[c.IS_REVENUE_KPI]

  return outcome_pb.Outcome(
      kpi_type=(
          kpi_type_pb.REVENUE if is_revenue_kpi else kpi_type_pb.NON_REVENUE
      ),
      roi=_to_estimate(channel_data.roi, confidence_level),
      marginal_roi=_to_estimate(channel_data.mroi, confidence_level),
      cost_per_contribution=_to_estimate(
          channel_data.cpik,
          confidence_level=confidence_level,
      ),
      contribution=outcome_pb.Contribution(
          value=_to_estimate(
              channel_data.incremental_outcome, confidence_level
          ),
      ),
      effectiveness=outcome_pb.Effectiveness(
          media_unit=c.IMPRESSIONS,
          value=_to_estimate(channel_data.effectiveness, confidence_level),
      ),
  )


def _to_incremental_outcome_grid(
    optimization_grid: xr.Dataset,
    grid_name: str | None,
) -> budget_pb.IncrementalOutcomeGrid:
  """Converts an optimization grid to an `IncrementalOutcomeGrid` proto.

  Args:
    optimization_grid: The optimization grid dataset in
      `OptimizationResults.optimization_grid`.
    grid_name: A user-given name for this grid.

  Returns:
    An `IncrementalOutcomeGrid` proto.
  """
  grid = budget_pb.IncrementalOutcomeGrid(
      name=(grid_name or ''),
      spend_step_size=optimization_grid.spend_step_size,
  )
  for channel in optimization_grid.channel.values:
    channel_grid = optimization_grid.sel(channel=channel)
    spend_grid = channel_grid.spend_grid.dropna(dim=c.GRID_SPEND_INDEX)
    incremental_outcome_grid = channel_grid.incremental_outcome_grid.dropna(
        dim=c.GRID_SPEND_INDEX
    )
    if len(spend_grid) != len(incremental_outcome_grid):
      raise ValueError(
          f'Spend grid and incremental outcome grid for channel "{channel}" do'
          ' not agree.'
      )
    channel_cells = budget_pb.IncrementalOutcomeGrid.ChannelCells(
        channel_name=channel,
        cells=[
            budget_pb.IncrementalOutcomeGrid.Cell(
                spend=spend.item(),
                incremental_outcome=estimate_pb.Estimate(
                    value=incr_outcome.item()
                ),
            )
            for (spend, incr_outcome) in zip(
                spend_grid, incremental_outcome_grid
            )
        ],
    )
    grid.channel_cells.append(channel_cells)
  return grid


def _to_estimate(
    dataarray: xr.DataArray,
    confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
) -> estimate_pb.Estimate:
  """Converts a DataArray with (mean, ci_lo, ci_hi) `metric` datavars."""
  estimate = estimate_pb.Estimate(value=dataarray.sel(metric=c.MEAN).item())
  uncertainty = estimate_pb.Estimate.Uncertainty(
      probability=confidence_level,
      lowerbound=dataarray.sel(metric=c.CI_LO).item(),
      upperbound=dataarray.sel(metric=c.CI_HI).item(),
  )
  estimate.uncertainties.append(uncertainty)
  return estimate


def _target_metric_to_proto(
    target_metric: str,
) -> target_pb.TargetMetric:
  """Converts a TargetMetric enum to a TargetMetric proto."""
  match target_metric:
    case c.ROI:
      return target_pb.TargetMetric.ROI
    case c.MROI:
      return target_pb.TargetMetric.MARGINAL_ROI
    case _:
      raise ValueError(f'Unsupported target metric: {target_metric}')
