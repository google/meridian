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

"""Defines common and base classes for processing trained Meridian model to an MMM schema."""

import abc
from collections.abc import Sequence
import dataclasses
import datetime
import functools
from typing import Generic, TypeVar

from google.protobuf import message
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.analysis import visualizer
from meridian.data import time_coordinates as tc
from meridian.model import model
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.common import date_interval_pb2
from schema.utils import time_record
from typing_extensions import override


class TrainedModel(abc.ABC):
  """Encapsulates a trained MMM model."""

  def __init__(self, mmm: model.Meridian):
    """Initializes the TrainedModel with a Meridian model.

    Args:
      mmm: A Meridian model that has been fitted (posterior samples drawn).

    Raises:
      ValueError: If the model has not been fitted (posterior samples drawn).
    """
    # Ideally, this could be encoded in the model type itself, and we won't need
    # this extra runtime check.
    if mmm.inference_data.prior is None or mmm.inference_data.posterior is None:  # pytype: disable=attribute-error
      raise ValueError('MMM model has not been fitted.')
    self._mmm = mmm

  @property
  def mmm(self) -> model.Meridian:
    return self._mmm

  @property
  def time_coordinates(self) -> tc.TimeCoordinates:
    return self._mmm.input_data.time_coordinates

  @functools.cached_property
  def internal_analyzer(self) -> analyzer.Analyzer:
    """Returns an internal `Analyzer`  bound to this trained model."""
    return analyzer.Analyzer(self.mmm)

  @functools.cached_property
  def internal_optimizer(self) -> optimizer.BudgetOptimizer:
    """Returns an internal `BudgetOptimizer` bound to this trained model."""
    return optimizer.BudgetOptimizer(self.mmm)

  @functools.cached_property
  def internal_model_diagnostics(self) -> visualizer.ModelDiagnostics:
    """Returns an internal `ModelDiagnostics` bound to this trained model."""
    return visualizer.ModelDiagnostics(self.mmm)


ModelType = model.Meridian | TrainedModel


def ensure_trained_model(model_input: ModelType) -> TrainedModel:
  """Ensure the given model is a trained model, and wrap it in a TrainedModel."""
  if isinstance(model_input, TrainedModel):
    return model_input
  return TrainedModel(model_input)


class Spec(abc.ABC):
  """Contains parameters needed for model-based analysis/optimization."""

  @abc.abstractmethod
  def validate(self):
    """Checks whether each parameter in the Spec has a valid value.

    Raises:
      ValueError: If any parameter in the Spec has an invalid value.
    """

  def __post_init__(self):
    self.validate()


@dataclasses.dataclass(frozen=True)
class DatedSpec(Spec):
  """A spec with a `[start_date, end_date)` closed-open date range semantic.

  Attrs:
    start_date: The start date of the analysis/optimization. If left as `None`,
      then this will eventually resolve to a model's first time coordinate.
    end_date: The end date of the analysis/optimization. If left as `None`, then
      this will eventually resolve to a model's last time coordinate. When
      specified, this end date is exclusive.
    date_interval_tag: An optional tag that identifies the date interval.
  """

  start_date: datetime.date | None = None
  end_date: datetime.date | None = None
  date_interval_tag: str = ''

  @override
  def validate(self):
    """Overrides the Spec.validate() method to check that dates are valid."""
    if (
        self.start_date is not None
        and self.end_date is not None
        and self.start_date > self.end_date
    ):
      raise ValueError('Start date must be before end date.')

  def resolver(
      self, time_coordinates: tc.TimeCoordinates
  ) -> 'DatedSpecResolver':
    """Returns a date resolver for this spec, with the given Meridian model."""
    return DatedSpecResolver(self, time_coordinates)


class DatedSpecResolver:
  """Resolves date parameters in specs based on a model's time coordinates."""

  def __init__(self, spec: DatedSpec, time_coordinates: tc.TimeCoordinates):
    self._spec = spec
    self._time_coordinates = time_coordinates

  @property
  def _interval_days(self) -> int:
    return self._time_coordinates.interval_days

  @property
  def time_coordinates(self) -> tc.TimeCoordinates:
    return self._time_coordinates

  def to_closed_date_interval_tuple(
      self,
  ) -> tuple[str | None, str | None]:
    """Transforms given spec into a closed `[start, end]` date interval tuple.

    For each of the bookends in the tuple, `None` value indicates a time
    coordinate default (first or last time coordinate, respectively).

    Returns:
      A **closed** `[start, end]` date interval tuple.
    """
    start, end = (None, None)

    if self._spec.start_date is not None:
      start = self._spec.start_date.strftime(c.DATE_FORMAT)
    if self._spec.end_date is not None:
      inclusive_end_date = self._spec.end_date - datetime.timedelta(
          days=self._interval_days
      )
      end = inclusive_end_date.strftime(c.DATE_FORMAT)

    return (start, end)

  def resolve_to_enumerated_selected_times(self) -> list[str] | None:
    """Resolves the given spec into an enumerated list of time coordinates.

    Returns:
      An enumerated list of time coordinates, or None (semantic "All") if the
      bound spec is also None.
    """
    start, end = self.to_closed_date_interval_tuple()
    expanded = self._time_coordinates.expand_selected_time_dims(
        start_date=start, end_date=end
    )
    if expanded is None:
      return None
    return [date.strftime(c.DATE_FORMAT) for date in expanded]

  def resolve_to_bool_selected_times(self) -> list[bool] | None:
    """Resolves the given spec into a list of booleans indicating selected times.

    Returns:
      A list of booleans indicating selected times, or None (semantic "All") if
      the bound spec is also None.
    """
    selected_times = self.resolve_to_enumerated_selected_times()
    if selected_times is None:
      return None
    return [
        time in selected_times for time in self._time_coordinates.all_dates_str
    ]

  def collapse_to_date_interval_proto(self) -> date_interval_pb2.DateInterval:
    """Collapses the given spec into a `DateInterval` proto.

    If the spec's date range is unbounded, then the DateInterval proto will have
    the semantic "All", and we resolve it by consulting the time coordinates of
    the model bound to this resolver.

    Note that the exclusive end date semantic will be preserved in the returned
    proto.

    Returns:
      A `DateInterval` proto the represents the date interval specified by the
      spec.
    """
    selected_times = self.resolve_to_enumerated_selected_times()
    if selected_times is None:
      start_date = self._time_coordinates.all_dates[0]
      end_date = self._time_coordinates.all_dates[-1]
    else:
      normalized_selected_times = [
          tc.normalize_date(date) for date in selected_times
      ]
      start_date = normalized_selected_times[0]
      end_date = normalized_selected_times[-1]

    # Adjust end_date to make it exclusive.
    end_date += datetime.timedelta(days=self._interval_days)

    return time_record.create_date_interval_pb(
        start_date, end_date, tag=self._spec.date_interval_tag
    )

  def transform_to_date_interval_protos(
      self,
  ) -> list[date_interval_pb2.DateInterval]:
    """Transforms the given spec into `DateInterval` protos.

    If the spec's date range is unbounded, then the DateInterval proto will have
    the semantic "All", and we resolve it by consulting the time coordinates of
    the model bound to this resolver.

    Note that the exclusive end date semantic will be preserved in the returned
    proto.

    Returns:
      A list of `DateInterval` protos the represents the date intervals
      specified by the spec.
    """
    selected_times = self.resolve_to_enumerated_selected_times()
    if selected_times is None:
      times_list = self._time_coordinates.all_dates
    else:
      times_list = [tc.normalize_date(date) for date in selected_times]

    date_intervals = []
    for start_date in times_list:
      date_interval = time_record.create_date_interval_pb(
          start_date=start_date,
          end_date=start_date + datetime.timedelta(days=self._interval_days),
          tag=self._spec.date_interval_tag,
      )
      date_intervals.append(date_interval)

    return date_intervals

  def resolve_to_date_interval_open_end(
      self,
  ) -> tuple[datetime.date, datetime.date]:
    """Resolves given spec into an open-ended `[start, end)` date interval."""
    start = self._spec.start_date or self._time_coordinates.all_dates[0]
    end = self._spec.end_date
    if end is None:
      end = self._time_coordinates.all_dates[-1]
      # Adjust `end` to make it exclusive, but only if we pulled it from the
      # time coordinates.
      end += datetime.timedelta(days=self._interval_days)
    return (start, end)

  def resolve_to_date_interval_proto(self) -> date_interval_pb2.DateInterval:
    """Resolves the given spec into a fully specified `DateInterval` proto.

    If either `start_date` or `end_date` is None in the bound spec, then we
    resolve it by consulting the time coordinates of the model bound to this
    resolver. They are resolved to the first and last time coordinates (plus
    interval length), respectively.

    Note that the exclusive end date semantic will be preserved in the returned
    proto.

    Returns:
      A resolved `DateInterval` proto the represents the date interval specified
      by the bound spec.
    """
    start_date, end_date = self.resolve_to_date_interval_open_end()
    return time_record.create_date_interval_pb(
        start_date, end_date, tag=self._spec.date_interval_tag
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptimizationSpec(DatedSpec):
  """A dated spec for optimization.

  Attrs:
    optimization_name: The name of the optimization in this spec.
    grid_name: The name of the optimization grid.
    group_id: An optional group ID for linking related optimizations.
    confidence_level: The threshold for computing confidence intervals. Defaults
      to 0.9. Must be a number between 0 and 1.
  """

  optimization_name: str
  grid_name: str
  group_id: str | None = None
  confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL

  @override
  def validate(self):
    """Check optimization parameters are valid."""
    super().validate()

    if not self.optimization_name or self.optimization_name.isspace():
      raise ValueError('Optimization name must not be empty or blank.')

    if not self.grid_name or self.grid_name.isspace():
      raise ValueError('Grid name must not be empty or blank.')

    if self.confidence_level < 0 or self.confidence_level > 1:
      raise ValueError('Confidence level must be between 0 and 1.')


S = TypeVar('S', bound=Spec)
M = TypeVar('M', bound=message.Message)


class ModelProcessor(abc.ABC, Generic[S, M]):
  """Performs model-based analysis or optimization."""

  @classmethod
  @abc.abstractmethod
  def spec_type(cls) -> type[S]:
    """Returns the concrete Spec type that this ModelProcessor operates on."""
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def output_type(cls) -> type[M]:
    """Returns the concrete output type that this ModelProcessor produces."""
    raise NotImplementedError()

  @abc.abstractmethod
  def execute(self, specs: Sequence[S]) -> M:
    """Runs an analysis/optimization on the model using the given specs.

    Args:
      specs: Sequence of Specs containing parameters needed for the
        analysis/optimization. The specs must all be of the same type as
        `self.spec_type()` for this processor

    Returns:
      A proto containing the results of the analysis/optimization.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _set_output(self, output: pb.Mmm, result: M):
    """Sets the output field in the given `MmmOutput` proto.

    A model consumer that orchestrated this processor will indirectly call this
    method (via `__call__`) to attach the output of `execute()` (a
    processor-defined message `M`) into a partially built `MmmOutput` proto that
    the model consumer manages.

    Args:
      output: The container output proto to which the given result message
        should be attached.
      result: An output of `execute()`.
    """
    raise NotImplementedError()

  def __call__(self, specs: Sequence[S], output: pb.Mmm):
    """Runs an analysis/optimization on the model using the given specs.

    This also sets the appropriate output field in the given MmmOutput proto.

    Args:
      specs: Sequence of Specs containing parameters needed for the
        analysis/optimization. The specs must all be of the same type as
        `self.spec_type()` for this processor
      output: The output proto to which the results of the analysis/optimization
        should be attached.

    Raises:
      ValueError: If any spec is not of the same type as `self.spec_type()`.
    """
    if not all([isinstance(spec, self.spec_type()) for spec in specs]):
      raise ValueError('Not all specs are of type %s' % self.spec_type())
    self._set_output(output, self.execute(specs))
