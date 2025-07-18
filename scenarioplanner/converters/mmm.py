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

"""Provides wrappers for the `Mmm` proto.

This module defines a set of dataclasses that act as high-level wrappers around
the `Mmm` protocol buffer and its nested messages. The primary goal is to offer
a more intuitive API for accessing and manipulating MMM data, abstracting away
the verbosity of the raw protobuf structures.

The main entry point is the `Mmm` class, which wraps the top-level `mmm_pb2.Mmm`
proto. From an instance of this class, you can navigate through the model's
different components, such as marketing data, model fit results, and various
analyses, using simple properties and methods.

Typical Usage:

```python
from mmm.v1 import mmm_pb2
from lookerstudio.converters import mmm

# Assume `mmm_proto` is a populated instance of the Mmm proto
mmm_proto = mmm_pb2.Mmm()
# ...

# Create the main wrapper instance
mmm_wrapper = mmm.Mmm(mmm_proto)

# Access marketing data and calculate total spends for a given period
marketing_data = mmm_wrapper.marketing_data
total_spends = marketing_data.all_channel_spends(
    date_interval=('2025-01-01', '2025-03-31')
)

# Access budget optimization results
for budget_result in mmm_wrapper.budget_optimization_results:
  print(f"Name: {budget_result.name}, Max: {budget_result.spec.max_budget}")
```
"""

import abc
import dataclasses
import datetime
import functools
from typing import TypeAlias

from meridian import constants as c
from meridian.data import time_coordinates as tc
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.common import date_interval_pb2 as date_interval_pb
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_metric_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing import marketing_data_pb2 as marketing_data_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2 as marketing_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_pb
from mmm.v1.marketing.analysis import non_media_analysis_pb2 as non_media_pb
from mmm.v1.marketing.analysis import outcome_pb2 as outcome_pb
from mmm.v1.marketing.analysis import response_curve_pb2 as response_curve_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb

from google.type import date_pb2 as date_pb


_DateIntervalTuple: TypeAlias = tuple[datetime.date, datetime.date]


@dataclasses.dataclass(frozen=True)
class DateInterval:
  """A dataclass wrapper around a tuple of `(start, end)` dates."""

  date_interval: _DateIntervalTuple

  @property
  def start(self) -> datetime.date:
    return self.date_interval[0]

  @property
  def end(self) -> datetime.date:
    return self.date_interval[1]

  def __contains__(self, date: datetime.date) -> bool:
    """Returns whether this date interval contains the given date."""
    return self.start <= date < self.end

  def __lt__(self, other: "DateInterval") -> bool:
    return self.start < other.start


def _to_datetime_date(
    date_proto: date_pb.Date,
) -> datetime.date:
  """Converts a `Date` proto into a `datetime.date`."""
  return datetime.date(
      year=date_proto.year, month=date_proto.month, day=date_proto.day
  )


def _to_date_interval_dc(
    date_interval: date_interval_pb.DateInterval,
) -> DateInterval:
  """Converts a `DateInterval` proto into `DateInterval` dataclass."""
  return DateInterval((
      _to_datetime_date(date_interval.start_date),
      _to_datetime_date(date_interval.end_date),
  ))


@dataclasses.dataclass(frozen=True)
class Outcome:
  """A wrapper for `Outcome` proto with derived properties."""

  outcome_proto: outcome_pb.Outcome

  @property
  def is_revenue_kpi(self) -> bool:
    return self.outcome_proto.kpi_type == kpi_type_pb.REVENUE

  @property
  def is_nonrevenue_kpi(self) -> bool:
    return self.outcome_proto.kpi_type == kpi_type_pb.NON_REVENUE

  @property
  def contribution_pb(self) -> outcome_pb.Contribution:
    return self.outcome_proto.contribution

  @property
  def effectiveness_pb(self) -> outcome_pb.Effectiveness:
    return self.outcome_proto.effectiveness

  @property
  def roi_pb(self) -> estimate_pb.Estimate:
    return self.outcome_proto.roi

  @property
  def marginal_roi_pb(self) -> estimate_pb.Estimate:
    return self.outcome_proto.marginal_roi

  @property
  def cost_per_contribution_pb(self) -> estimate_pb.Estimate:
    return self.outcome_proto.cost_per_contribution


class _OutcomeMixin(abc.ABC):
  """Mixin for (non-)media analysis with typed KPI outcome property getters.

  A `MediaAnalysis` or `NonMediaAnalysis` proto is configured with multiple
  polymorphic `Outcome`s. In Meridian processors, both types (revenue and
  non-revenue) may be present in the analysis container. However, for each type
  there should be at most one `Outcome` value.

  This mixin provides both `MediaAnalysis` and `NonMediaAnalysis` dataclasses
  with property getters to retrieve typed `Outcome` values.
  """

  @property
  @abc.abstractmethod
  def _outcome_pbs(self) -> list[outcome_pb.Outcome]:
    """Returns a list of `Outcome` protos."""
    raise NotImplementedError()

  @functools.cached_property
  def maybe_revenue_outcome(self) -> Outcome | None:
    """Returns the revenue-type `Outcome`, or None if it does not exist."""
    for outcome_proto in self._outcome_pbs:
      outcome = Outcome(outcome_proto)
      if outcome.is_revenue_kpi:
        return outcome
    return None

  @property
  def revenue_outcome(self) -> Outcome:
    """Returns the revenue-type `Outcome`, or raises an error if it does not exist."""
    outcome = self.maybe_revenue_outcome
    if outcome is None:
      raise ValueError(
          "No revenue-type `Outcome` found in an expected analysis proto."
      )
    return outcome

  @functools.cached_property
  def maybe_non_revenue_outcome(self) -> Outcome | None:
    """Returns the nonrevenue-type `Outcome`, or None if it does not exist."""
    for outcome_proto in self._outcome_pbs:
      outcome = Outcome(outcome_proto)
      if outcome.is_nonrevenue_kpi:
        return outcome
    return None

  @property
  def non_revenue_outcome(self) -> Outcome:
    """Returns the nonrevenue-type `Outcome`, or raises an error if it does not exist."""
    outcome = self.maybe_non_revenue_outcome
    if outcome is None:
      raise ValueError(
          "No nonrevenue-type `Outcome` found in an expected analysis proto."
      )
    return outcome


@dataclasses.dataclass(frozen=True)
class MediaAnalysis(_OutcomeMixin):
  """A wrapper for `MediaAnalysis` proto with derived properties."""

  analysis_proto: media_pb.MediaAnalysis

  @property
  def channel_name(self) -> str:
    return self.analysis_proto.channel_name

  @property
  def spend_info_pb(self) -> media_pb.SpendInfo:
    return self.analysis_proto.spend_info

  @property
  def _outcome_pbs(self) -> list[outcome_pb.Outcome]:
    return list(self.analysis_proto.media_outcomes)


@dataclasses.dataclass(frozen=True)
class NonMediaAnalysis(_OutcomeMixin):
  """A wrapper for `NonMediaAnalysis` proto with derived properties."""

  analysis_proto: non_media_pb.NonMediaAnalysis

  @property
  def non_media_name(self) -> str:
    return self.analysis_proto.non_media_name

  @property
  def _outcome_pbs(self) -> list[outcome_pb.Outcome]:
    return list(self.analysis_proto.non_media_outcomes)


@dataclasses.dataclass(frozen=True)
class ResponseCurve:
  """A wrapper for `ResponseCurve` proto with derived properties."""

  channel_name: str
  response_curve_proto: response_curve_pb.ResponseCurve

  @property
  def input_name(self) -> str:
    return self.response_curve_proto.input_name

  @property
  def response_points(self) -> list[tuple[float, float]]:
    """Returns `(spend, incremental outcome)` tuples for this channel's curve."""
    return [
        (point.input_value, point.incremental_kpi)
        for point in self.response_curve_proto.response_points
    ]


@dataclasses.dataclass(frozen=True)
class MarketingAnalysis:
  """A wrapper for `MarketingAnalysis` proto with derived properties."""

  marketing_analysis_proto: marketing_pb.MarketingAnalysis

  @property
  def tag(self) -> str:
    return self.marketing_analysis_proto.date_interval.tag

  @functools.cached_property
  def analysis_date_interval(
      self,
  ) -> DateInterval:
    return _to_date_interval_dc(self.marketing_analysis_proto.date_interval)

  @property
  def analysis_date_interval_str(self) -> tuple[str, str]:
    """Returns a tuple of `(date_start, date_end)` as strings."""
    return (
        self.analysis_date_interval.start.strftime(c.DATE_FORMAT),
        self.analysis_date_interval.end.strftime(c.DATE_FORMAT),
    )

  @functools.cached_property
  def channel_mapped_media_analyses(self) -> dict[str, MediaAnalysis]:
    """Returns media analyses mapped to their channel names."""
    return {
        analysis.channel_name: MediaAnalysis(analysis)
        for analysis in self.marketing_analysis_proto.media_analyses
    }

  @functools.cached_property
  def channel_mapped_non_media_analyses(self) -> dict[str, NonMediaAnalysis]:
    """Returns non-media analyses mapped to their non-media names."""
    return {
        analysis.non_media_name: NonMediaAnalysis(analysis)
        for analysis in self.marketing_analysis_proto.non_media_analyses
    }

  @functools.cached_property
  def baseline_analysis(self) -> NonMediaAnalysis:
    """Returns a "baseline" non media analysis among the given values.

    Raises:
      ValueError: if there is no "baseline" analysis
    """
    for non_media_analysis in self.marketing_analysis_proto.non_media_analyses:
      if non_media_analysis.non_media_name == c.BASELINE:
        return NonMediaAnalysis(non_media_analysis)
    else:
      raise ValueError(
          f"No '{c.BASELINE}' found in the set of `NonMediaAnalysis` for this"
          " `MarketingAnalysis`."
      )

  @functools.cached_property
  def response_curves(self) -> list[ResponseCurve]:
    """Returns a list of `ResponseCurve`s."""
    return [
        ResponseCurve(m_analysis.channel_name, m_analysis.response_curve)
        for m_analysis in self.marketing_analysis_proto.media_analyses
    ]


@dataclasses.dataclass(frozen=True)
class IncrementalOutcomeGrid:
  """A wrapper for `IncrementalOutcomeGrid` proto with derived properties."""

  incremental_outcome_grid_proto: budget_pb.IncrementalOutcomeGrid

  @property
  def name(self) -> str:
    return self.incremental_outcome_grid_proto.name

  @property
  def channel_spend_grids(self) -> dict[str, list[tuple[float, float]]]:
    """Returns channels mapped to (spend, incremental outcome) tuples."""
    grid = {}
    for channel_cells in self.incremental_outcome_grid_proto.channel_cells:
      grid[channel_cells.channel_name] = [
          (cell.spend, cell.incremental_outcome.value)
          for cell in channel_cells.cells
      ]
    return grid


class _SpecMixin(abc.ABC):
  """Mixin for both budget and R&F optimization specs."""

  @property
  @abc.abstractmethod
  def _date_interval_proto(self) -> date_interval_pb.DateInterval:
    """Returns the date interval proto."""
    raise NotImplementedError()

  @functools.cached_property
  def date_interval(self) -> DateInterval:
    """Returns the spec's date interval."""
    date_interval_proto = self._date_interval_proto
    return DateInterval((
        datetime.date(
            year=date_interval_proto.start_date.year,
            month=date_interval_proto.start_date.month,
            day=date_interval_proto.start_date.day,
        ),
        datetime.date(
            year=date_interval_proto.end_date.year,
            month=date_interval_proto.end_date.month,
            day=date_interval_proto.end_date.day,
        ),
    ))


@dataclasses.dataclass(frozen=True)
class BudgetOptimizationSpec(_SpecMixin):
  """A wrapper for `BudgetOptimizationSpec` proto with derived properties."""

  budget_optimization_spec_proto: budget_pb.BudgetOptimizationSpec

  @property
  def _date_interval_proto(self) -> date_interval_pb.DateInterval:
    return self.budget_optimization_spec_proto.date_interval

  @property
  def date_interval_tag(self) -> str:
    return self._date_interval_proto.tag

  @property
  def objective(self) -> target_metric_pb.TargetMetric:
    return self.budget_optimization_spec_proto.objective

  @property
  def is_fixed_scenario(self) -> bool:
    return (
        self.budget_optimization_spec_proto.WhichOneof("scenario")
        == "fixed_budget_scenario"
    )

  @property
  def max_budget(self) -> float:
    """Returns the maximum budget for this spec.

    Max budget is the total budget for a fixed scenario spec, or the max budget
    upper bound for a flexible scenario spec.
    """
    if self.is_fixed_scenario:
      return (
          self.budget_optimization_spec_proto.fixed_budget_scenario.total_budget
      )
    else:
      return (
          self.budget_optimization_spec_proto.flexible_budget_scenario.total_budget_constraint.max_budget
      )

  @functools.cached_property
  def channel_constraints(self) -> list[budget_pb.ChannelConstraint]:
    """Returns a list of `ChannelConstraint`s.

    If the underlying spec proto has no channel constraints, then it is implied
    that this spec's maximum budget is applied to them. Returns an empty list in
    this case, and it is up to the caller to handle.
    """
    return list(self.budget_optimization_spec_proto.channel_constraints)


@dataclasses.dataclass(frozen=True)
class RfOptimizationSpec(_SpecMixin):
  """A wrapper for `ReachFrequencyOptimizationSpec` proto with derived properties."""

  rf_optimization_spec_proto: rf_pb.ReachFrequencyOptimizationSpec

  @property
  def _date_interval_proto(self) -> date_interval_pb.DateInterval:
    return self.rf_optimization_spec_proto.date_interval

  @property
  def objective(self) -> target_metric_pb.TargetMetric:
    return self.rf_optimization_spec_proto.objective

  @property
  def total_budget_constraint(self) -> constraints_pb.BudgetConstraint:
    return self.rf_optimization_spec_proto.total_budget_constraint

  @functools.cached_property
  def channel_constraints(self) -> list[rf_pb.RfChannelConstraint]:
    """Returns a list of `RfChannelConstraint`s."""
    return list(self.rf_optimization_spec_proto.rf_channel_constraints)


class _NamedResultMixin(abc.ABC):
  """Mixin for named optimization results with assigned group ID."""

  @property
  @abc.abstractmethod
  def group_id(self) -> str:
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def name(self) -> str:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class BudgetOptimizationResult(_NamedResultMixin):
  """A wrapper for `BudgetOptimizationResult` proto with derived properties."""

  budget_optimization_result_proto: budget_pb.BudgetOptimizationResult

  @property
  def name(self) -> str:
    return self.budget_optimization_result_proto.name

  @property
  def group_id(self) -> str:
    return self.budget_optimization_result_proto.group_id

  @functools.cached_property
  def spec(self) -> BudgetOptimizationSpec:
    return BudgetOptimizationSpec(self.budget_optimization_result_proto.spec)

  @functools.cached_property
  def optimized_marketing_analysis(self) -> MarketingAnalysis:
    return MarketingAnalysis(
        self.budget_optimization_result_proto.optimized_marketing_analysis
    )

  @functools.cached_property
  def incremental_outcome_grid(self) -> IncrementalOutcomeGrid:
    return IncrementalOutcomeGrid(
        self.budget_optimization_result_proto.incremental_outcome_grid
    )

  @functools.cached_property
  def response_curves(self) -> list[ResponseCurve]:
    return MarketingAnalysis(
        self.budget_optimization_result_proto.optimized_marketing_analysis
    ).response_curves


@dataclasses.dataclass(frozen=True)
class FrequencyOutcomeGrid:
  """A wrapper for `FrequencyOutcomeGrid` proto with derived properties."""

  frequency_outcome_grid_proto: rf_pb.FrequencyOutcomeGrid

  @property
  def name(self) -> str:
    return self.frequency_outcome_grid_proto.name

  @property
  def channel_frequency_grids(self) -> dict[str, list[tuple[float, float]]]:
    """Returns channels mapped to (frequency, outcome) tuples."""
    grid = {}
    for channel_cells in self.frequency_outcome_grid_proto.channel_cells:
      grid[channel_cells.channel_name] = [
          (cell.reach_frequency.average_frequency, cell.outcome.value)
          for cell in channel_cells.cells
      ]
    return grid


@dataclasses.dataclass(frozen=True)
class ReachFrequencyOptimizationResult(_NamedResultMixin):
  """A wrapper for `ReachFrequencyOptimizationResult` proto with derived properties."""

  rf_optimization_result_proto: rf_pb.ReachFrequencyOptimizationResult

  @property
  def name(self) -> str:
    return self.rf_optimization_result_proto.name

  @property
  def group_id(self) -> str:
    return self.rf_optimization_result_proto.group_id

  @functools.cached_property
  def spec(self) -> RfOptimizationSpec:
    return RfOptimizationSpec(self.rf_optimization_result_proto.spec)

  @functools.cached_property
  def channel_mapped_optimized_frequencies(self) -> dict[str, float]:
    """Returns optimized frequencies mapped to their channel names."""
    return {
        optimized_channel_frequency.channel_name: (
            optimized_channel_frequency.optimal_average_frequency
        )
        for optimized_channel_frequency in self.rf_optimization_result_proto.optimized_channel_frequencies
    }

  @functools.cached_property
  def optimized_marketing_analysis(self) -> MarketingAnalysis:
    return MarketingAnalysis(
        self.rf_optimization_result_proto.optimized_marketing_analysis
    )

  @functools.cached_property
  def frequency_outcome_grid(self) -> FrequencyOutcomeGrid:
    return FrequencyOutcomeGrid(
        self.rf_optimization_result_proto.frequency_outcome_grid
    )


@dataclasses.dataclass(frozen=True)
class MarketingData:
  """A wrapper for `MarketingData` proto with derived properties."""

  marketing_data_proto: marketing_data_pb.MarketingData

  @property
  def _marketing_data_points(
      self,
  ) -> list[marketing_data_pb.MarketingDataPoint]:
    """Returns a list of `MarketingDataPoint`s."""
    return list(self.marketing_data_proto.marketing_data_points)

  @functools.cached_property
  def media_channels(self) -> list[str]:
    """Returns unique (non-R&F) media channel names in the marketing data."""
    channels = set()
    for data_point in self._marketing_data_points:
      for var in data_point.media_variables:
        channels.add(var.channel_name)
    return sorted(channels)  # For deterministic order in iterating.

  @functools.cached_property
  def rf_channels(self) -> list[str]:
    """Returns unique R&F channel names in the marketing data."""
    channels = set()
    for data_point in self._marketing_data_points:
      for var in data_point.reach_frequency_variables:
        channels.add(var.channel_name)
    return sorted(channels)  # For deterministic order in iterating.

  @functools.cached_property
  def date_intervals(self) -> list[DateInterval]:
    """Returns all date intervals in the marketing data."""
    date_intervals = set()
    for data_point in self._marketing_data_points:
      date_intervals.add(_to_date_interval_dc(data_point.date_interval))
    return sorted(date_intervals)

  def media_channel_spends(
      self, date_interval: tc.DateInterval
  ) -> dict[str, float]:
    """Returns non-RF media channel names mapped to their total spend values, for the given date interval.

    All channel spends in time coordinates between `[start, end)` of the given
    date interval are summed up.

    Args:
      date_interval: the date interval to query for

    Returns:
      A dict of channel names mapped to their total spend values, for the given
      date interval.
    """
    date_interval = DateInterval(tc.normalize_date_interval(date_interval))
    channel_spends = {channel: 0.0 for channel in self.media_channels}
    for data_point in self._marketing_data_points:
      # The time coordinate for a marketing data point is the start date of its
      # date interval field: test that it is contained within the given interval
      data_point_date = _to_date_interval_dc(data_point.date_interval).start
      if data_point_date not in date_interval:
        continue
      for var in data_point.media_variables:
        channel_spends[var.channel_name] = (
            channel_spends[var.channel_name] + var.media_spend
        )
    return channel_spends

  def rf_channel_spends(
      self, date_interval: tc.DateInterval
  ) -> dict[str, float]:
    """Returns *Reach and Frequency* channel names mapped to their total spend values, for the given date interval.

    All channel spends in time coordinates between `[start, end)` of the given
    date interval are summed up.

    Args:
      date_interval: the date interval to query for

    Returns:
      A dict of channel names mapped to their total spend values, for the given
      date interval.
    """
    date_interval = DateInterval(tc.normalize_date_interval(date_interval))
    channel_spends = {channel: 0.0 for channel in self.rf_channels}
    for data_point in self._marketing_data_points:
      # The time coordinate for a marketing data point is the start date of its
      # date interval field: test that it is contained within the given interval
      data_point_date = _to_date_interval_dc(data_point.date_interval).start
      if data_point_date not in date_interval:
        continue
      for var in data_point.reach_frequency_variables:
        channel_spends[var.channel_name] = (
            channel_spends[var.channel_name] + var.spend
        )
    return channel_spends

  def all_channel_spends(
      self, date_interval: tc.DateInterval
  ) -> dict[str, float]:
    """Returns *all* channel names mapped to their total spend values, for the given date interval.

    All channel spends in time coordinates between `[start, end)` of the given
    date interval are summed up.

    Args:
      date_interval: the date interval to query for

    Returns:
      A dict of channel names mapped to their total spend values, for the given
      date interval.
    """
    spends = self.rf_channel_spends(date_interval)
    spends.update(self.media_channel_spends(date_interval))
    return spends


@dataclasses.dataclass(frozen=True)
class Mmm:
  """A wrapper for `Mmm` proto with derived properties."""

  mmm_proto: mmm_pb.Mmm

  @functools.cached_property
  def marketing_data(self) -> MarketingData:
    """Returns marketing data inside the MMM model kernel."""
    return MarketingData(self.mmm_proto.mmm_kernel.marketing_data)

  @property
  def model_fit(self) -> fit_pb.ModelFit:
    return self.mmm_proto.model_fit

  @functools.cached_property
  def model_fit_results(self) -> dict[str, fit_pb.Result]:
    """Returns each model fit `Result`, mapped to its dataset name."""
    return {result.name: result for result in self.model_fit.results}

  @functools.cached_property
  def marketing_analyses(self) -> list[MarketingAnalysis]:
    """Returns a list of `MarketingAnalysis` wrappers."""
    return [
        MarketingAnalysis(analysis)
        for analysis in self.mmm_proto.marketing_analysis_list.marketing_analyses
    ]

  @functools.cached_property
  def tagged_marketing_analyses(
      self,
  ) -> dict[str, MarketingAnalysis]:
    """Returns each marketing analysis, mapped to its tag name."""
    return {analysis.tag: analysis for analysis in self.marketing_analyses}

  @functools.cached_property
  def budget_optimization_results(
      self,
  ) -> list[BudgetOptimizationResult]:
    """Returns a list of `BudgetOptimizationResult` wrappers."""
    return [
        BudgetOptimizationResult(result)
        for result in self.mmm_proto.marketing_optimization.budget_optimization.results
    ]

  @functools.cached_property
  def reach_frequency_optimization_results(
      self,
  ) -> list[ReachFrequencyOptimizationResult]:
    """Returns a list of `ReachFrequencyOptimizationResult` wrappers."""
    return [
        ReachFrequencyOptimizationResult(result)
        for result in self.mmm_proto.marketing_optimization.reach_frequency_optimization.results
    ]
