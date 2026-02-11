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

"""Generates an `Mmm` (Marketing Mix Model) proto for Meridian UI.

The MMM proto schema contains parts collected from the core model as well as
analysis results from trained model processors.
"""

import abc
from collections.abc import Collection, Sequence
import dataclasses
import datetime
from typing import TypeVar
import uuid
import warnings

from meridian.analysis import optimizer
from meridian.data import time_coordinates as tc
from meridian.model import model
from mmm.v1 import mmm_pb2 as mmm_pb
from scenarioplanner.converters.dataframe import constants as converter_constants
from meridian.schema import mmm_proto_generator
from meridian.schema.processors import budget_optimization_processor as bop
from meridian.schema.processors import marketing_processor
from meridian.schema.processors import model_fit_processor
from meridian.schema.processors import model_processor
from meridian.schema.processors import reach_frequency_optimization_processor as rfop
from meridian.schema.utils import date_range_bucketing


__all__ = [
    "MmmUiProtoGenerator",
    "create_mmm_ui_data_proto",
    "create_tag",
]


_ALLOWED_SPEC_TYPES_FOR_UI = frozenset({
    model_fit_processor.ModelFitSpec,
    marketing_processor.MarketingAnalysisSpec,
    bop.BudgetOptimizationSpec,
})

_SPEC_TYPES_CREATE_SUBSPECS = frozenset({
    marketing_processor.MarketingAnalysisSpec,
    bop.BudgetOptimizationSpec,
    rfop.ReachFrequencyOptimizationSpec,
})

_DATE_RANGE_GENERATORS = frozenset({
    date_range_bucketing.MonthlyDateRangeGenerator,
    date_range_bucketing.QuarterlyDateRangeGenerator,
    date_range_bucketing.YearlyDateRangeGenerator,
})

SpecType = TypeVar("SpecType", bound=model_processor.Spec)
DatedSpecType = TypeVar("DatedSpecType", bound=model_processor.DatedSpec)
OptimizationSpecType = TypeVar(
    "OptimizationSpecType", bound=model_processor.OptimizationSpec
)

_DERIVED_RF_OPT_NAME_PREFIX = "derived RF optimization from "
_DERIVED_RF_OPT_GRID_NAME_PREFIX = "derived_from_"


class MmmUiProtoGenerator:
  """Creates `Mmm` proto for the Meridian Scenario Planner UI (Looker Studio).

  Currently, it only accepts specs for Model Fit, Marketing Analysis, and Budget
  Optimization, but not stand-alone Reach Frequency Optimization specs.
  Reach Frequency Optimization spec will be derived from the Budget Optimization
  spec; this is done so that we can structurally pair them.

  Attributes:
    mmm: A trained Meridian model. A trained model has its posterior
      distributions already sampled.
    specs: A sequence of specs that specify the analyses to run on the model.
    model_id: An optional model identifier.
    time_breakdown_generators: A list of generators that break down the given
      specs by automatically generated time buckets. Currently, this time period
      breakdown is only done on Marketing Analysis specs and Budget Optimization
      specs. All other specs are processed in their original forms. The set of
      default bucketers break down sub-specs with the following time periods:
      [All (original spec's time period), Yearly, Quarterly, Monthly]
  """

  def __init__(
      self,
      mmm: model.Meridian,
      specs: Sequence[SpecType],
      model_id: str = "",
      time_breakdown_generators: Collection[
          type[date_range_bucketing.DateRangeBucketer]
      ] = _DATE_RANGE_GENERATORS,
  ):
    self._mmm = mmm
    self._input_specs = specs
    self._model_id = model_id
    self._time_breakdown_generators = time_breakdown_generators

  @property
  def _time_coordinates(self) -> tc.TimeCoordinates:
    return self._mmm.input_data.time_coordinates

  def __call__(self) -> mmm_pb.Mmm:
    """Creates `Mmm` proto for the Meridian Scenario Planner UI (Looker Studio).

    Returns:
      A proto containing the model kernel at rest and its analysis results given
      user specs.
    """
    seen_group_ids = set()

    copy_specs = []
    for spec in self._input_specs:
      if not any(isinstance(spec, t) for t in _ALLOWED_SPEC_TYPES_FOR_UI):
        raise ValueError(f"Unsupported spec type: {spec.__class__.__name__}")

      if isinstance(spec, bop.BudgetOptimizationSpec):
        group_id = spec.group_id
        if not group_id:
          group_id = str(uuid.uuid4())
          copy_specs.append(dataclasses.replace(spec, group_id=group_id))
        else:
          if group_id in seen_group_ids:
            raise ValueError(
                f"Duplicate group ID found: {group_id}. Please provide a unique"
                " group ID for each Budget Optimization spec."
            )
          seen_group_ids.add(group_id)
          copy_specs.append(spec)

        # If there are RF channels, derive a RF optimization spec from the
        # Budget Optimization spec.
        if self._mmm.input_data.rf_channel is not None:
          copy_specs.append(
              self._derive_rf_opt_spec_from_budget_opt_spec(copy_specs[-1])
          )
      else:
        copy_specs.append(spec)

    sub_specs = []
    for spec in copy_specs:
      to_create_subspecs = self._time_breakdown_generators and any(
          isinstance(spec, t) for t in _SPEC_TYPES_CREATE_SUBSPECS
      )

      if to_create_subspecs:
        dates = self._enumerate_dates_open_end(spec)
        sub_specs.extend(
            _create_subspecs(spec, dates, self._time_breakdown_generators)
        )
      else:
        sub_specs.append(spec)

    return mmm_proto_generator.create_mmm_proto(
        self._mmm,
        sub_specs,
        model_id=self._model_id,
    )

  def _derive_rf_opt_spec_from_budget_opt_spec(
      self,
      budget_opt_spec: bop.BudgetOptimizationSpec,
  ) -> rfop.ReachFrequencyOptimizationSpec:
    """Derives a ReachFrequencyOptimizationSpec from a BudgetOptimizationSpec."""
    rf_opt_name = (
        f"{_DERIVED_RF_OPT_NAME_PREFIX}{budget_opt_spec.optimization_name}"
    )
    rf_opt_grid_name = (
        f"{_DERIVED_RF_OPT_GRID_NAME_PREFIX}{budget_opt_spec.optimization_name}"
    )

    return rfop.ReachFrequencyOptimizationSpec(
        start_date=budget_opt_spec.start_date,
        end_date=budget_opt_spec.end_date,
        date_interval_tag=budget_opt_spec.date_interval_tag,
        optimization_name=rf_opt_name,
        grid_name=rf_opt_grid_name,
        group_id=budget_opt_spec.group_id,
        confidence_level=budget_opt_spec.confidence_level,
        max_frequency=budget_opt_spec.max_frequency,
    )

  def _enumerate_dates_open_end(
      self, spec: DatedSpecType
  ) -> list[datetime.date]:
    """Enumerates date points with an open end date.

    The date points are enumerated from the data's time coordinates based on the
    spec's start and end dates. The last date point is the exclusive end date as
    same as the spec's end date, if specified.

    Args:
      spec: A dated spec.

    Returns:
      A list of date points.
    """
    inclusive_date_strs = spec.resolver(
        self._mmm.input_data.time_coordinates
    ).resolve_to_enumerated_selected_times()

    if inclusive_date_strs is None:
      dates = self._time_coordinates.all_dates
    else:
      dates = [tc.normalize_date(date_str) for date_str in inclusive_date_strs]

    # If the end date is not specified, compute the exclusive end date based on
    # the last date in the time coordinates.
    exclusive_end_date = spec.end_date or dates[-1] + datetime.timedelta(
        days=self._time_coordinates.interval_days
    )

    dates.append(exclusive_end_date)

    return dates


def create_mmm_ui_data_proto(
    mmm: model.Meridian,
    specs: Sequence[SpecType],
    model_id: str = "",
    time_breakdown_generators: Collection[
        type[date_range_bucketing.DateRangeBucketer]
    ] = _DATE_RANGE_GENERATORS,
) -> mmm_pb.Mmm:
  """Creates `Mmm` proto for the Meridian Scenario Planner UI (Looker Studio).

  Currently, it only accepts specs for Model Fit, Marketing Analysis, and Budget
  Optimization, but not stand-alone Reach Frequency Optimization specs.
  Reach Frequency Optimization spec will be derived from the Budget Optimization
  spec; this is done so that we can structurally pair them.

  Args:
    mmm: A trained Meridian model. A trained model has its posterior
      distributions already sampled.
    specs: A sequence of specs that specify the analyses to run on the model.
    model_id: An optional model identifier.
    time_breakdown_generators: A list of generators that break down the given
      specs by automatically generated time buckets. Currently, this time period
      breakdown is only done on Marketing Analysis specs and Budget Optimization
      specs. All other specs are processed in their original forms. The set of
      default bucketers break down sub-specs with the following time periods:
      [All (original spec's time period), Yearly, Quarterly, Monthly]

  Returns:
    A proto containing the model kernel at rest and its analysis results given
    user specs.
  """
  return MmmUiProtoGenerator(
      mmm,
      specs,
      model_id,
      time_breakdown_generators,
  )()


def create_tag(
    generator_class: type[abc.ABC], start_date: datetime.date
) -> str:
  """Creates a human-readable tag for a spec."""
  if generator_class == date_range_bucketing.YearlyDateRangeGenerator:
    return f"Y{start_date.year}"
  elif generator_class == date_range_bucketing.QuarterlyDateRangeGenerator:
    return f"Y{start_date.year} Q{(start_date.month - 1) // 3 + 1}"
  elif generator_class == date_range_bucketing.MonthlyDateRangeGenerator:
    return f"Y{start_date.year} {start_date.strftime('%b')}"
  else:
    raise ValueError(f"Unsupported generator class: {generator_class}")


def _normalize_optimization_spec_time_info(
    spec: OptimizationSpecType,
    date_interval_tag: str,
) -> OptimizationSpecType:
  """Adds time info to an optimization spec."""
  formatted_date_interval_tag = date_interval_tag.replace(r" ", "_")
  return dataclasses.replace(
      spec,
      group_id=f"{spec.group_id}:{date_interval_tag}",
      optimization_name=f"{spec.optimization_name} for {date_interval_tag}",
      grid_name=f"{spec.grid_name}_{formatted_date_interval_tag}",
  )


def _create_subspecs(
    spec: DatedSpecType,
    date_range: list[datetime.date],
    time_breakdown_generators: Collection[
        type[date_range_bucketing.DateRangeBucketer]
    ],
) -> list[DatedSpecType]:
  """Breaks down a spec into sub-specs for each time bucket."""
  specs = []

  all_period_spec = dataclasses.replace(
      spec,
      date_interval_tag=converter_constants.ANALYSIS_TAG_ALL,
  )
  if isinstance(all_period_spec, model_processor.OptimizationSpec):
    all_period_spec = _normalize_optimization_spec_time_info(
        all_period_spec, converter_constants.ANALYSIS_TAG_ALL
    )
  specs.append(all_period_spec)

  for generator_class in time_breakdown_generators:
    generator = generator_class(date_range)  # pytype: disable=not-instantiable
    date_intervals = generator.generate_date_intervals()
    for start_date, end_date in date_intervals:
      date_interval_tag = create_tag(generator_class, start_date)
      new_spec = dataclasses.replace(
          spec,
          start_date=start_date,
          end_date=end_date,
          date_interval_tag=date_interval_tag,
      )

      if isinstance(new_spec, model_processor.OptimizationSpec):
        new_spec = _normalize_optimization_spec_time_info(
            new_spec, date_interval_tag
        )

        if (
            isinstance(new_spec, bop.BudgetOptimizationSpec)
            and isinstance(new_spec.scenario, optimizer.FixedBudgetScenario)
            and new_spec.scenario.total_budget is not None
        ):
          # TODO: The budget amount should be adjusted based on the
          # budget specified in the `all_period_spec` and the historical spend
          # at the time period.
          new_spec = dataclasses.replace(
              new_spec,
              scenario=optimizer.FixedBudgetScenario(total_budget=None),
          )
          warnings.warn(
              "Using historical spend for budget optimization spec at the"
              f" period of {date_interval_tag}",
          )

      specs.append(new_spec)

  return specs
