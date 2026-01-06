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

"""Defines a processor for reach and frequency optimization inference on a Meridian model.

This module provides the `ReachFrequencyOptimizationProcessor`, which optimizes
the average frequency for reach and frequency (R&F) media channels in a trained
Meridian model to maximize ROI.

The processor takes a trained model and a `ReachFrequencyOptimizationSpec`
object. The spec defines the constraints for the optimization, such as the
minimum and maximum average frequency to consider for each channel.

Key Features:

-   Optimizes average frequency for all R&F channels simultaneously.
-   Allows setting minimum and maximum frequency constraints.
-   Generates detailed results, including the optimal average frequency for
    each channel, the expected outcomes at this optimal frequency, and
    response curves showing KPI/Revenue as a function of spend.
-   Outputs results in a structured protobuf format
    (`ReachFrequencyOptimization`).

Key Classes:

-   `ReachFrequencyOptimizationSpec`: Dataclass to specify optimization
    parameters and constraints.
-   `ReachFrequencyOptimizationProcessor`: The main processor class to execute
    the R&F optimization.

Example Usage:

```python
from schema.processors import reach_frequency_optimization_processor
from schema.processors import common
from schema.processors import model_processor
import datetime

# Assuming 'mmm' is a trained Meridian model object with R&F channels
trained_model = model_processor.TrainedModel(mmm)

spec = reach_frequency_optimization_processor.ReachFrequencyOptimizationSpec(
    optimization_name="rf_optimize_q1",
    start_date=datetime.date(2023, 1, 1),
    end_date=datetime.date(2023, 4, 1),
    min_frequency=1.0,
    max_frequency=10.0,  # Optional, defaults to model's max frequency
    kpi_type=common.KpiType.REVENUE,
)

processor = (
    reach_frequency_optimization_processor.ReachFrequencyOptimizationProcessor(
        trained_model
    )
)
# result is a rf_pb.ReachFrequencyOptimization proto
result = processor.execute([spec])

print(f"R&F Optimization results for {spec.optimization_name}:")
# Access results from the proto, e.g.:
# result.results[0].optimized_channel_frequencies
# result.results[0].optimized_marketing_analysis
# result.results[0].frequency_outcome_grid
```

Note: You can provide the processor with multiple specs. This would result in
a `ReachFrequencyOptimization` output with multiple results therein.
"""

from collections.abc import Sequence
import dataclasses

from meridian import backend
from meridian import constants
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2 as analysis_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_analysis_pb
from mmm.v1.marketing.analysis import outcome_pb2 as outcome_pb
from mmm.v1.marketing.analysis import response_curve_pb2
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from schema.processors import common
from schema.processors import model_processor
from schema.utils import time_record
import numpy as np
import xarray as xr


__all__ = [
    "ReachFrequencyOptimizationSpec",
    "ReachFrequencyOptimizationProcessor",
]


_STEP_SIZE_DECIMAL_PRECISION = 1
_STEP_SIZE = _STEP_SIZE_DECIMAL_PRECISION / 10
_TOL = 1e-6


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReachFrequencyOptimizationSpec(model_processor.OptimizationSpec):
  """Spec dataclass for marketing reach and frequency optimization processor.

  A frequency grid is generated using the range `[rounded_min_frequency,
  rounded_max_frequency]` and a step size of `STEP_SIZE=0.1`.
  `rounded_min_frequency` and `rounded_max_frequency` are rounded to the
  nearest multiple of `STEP_SIZE`.

  This spec is used both as user input to inform the R&F optimization processor
  of its constraints and parameters, as well as an output structure that is
  serializable to a `ReachFrequencyOptimizationSpec` proto. The latter serves as
  a metadata embedded in a `ReachFrequencyOptimizationResult`. The output spec
  in the proto reflects the actual numbers used to generate the reach and
  frequency optimization result.

  Attributes:
    min_frequency: The minimum frequency constraint for each channel. Must be
      greater than or equal to `1.0`. Defaults to `1.0`.
    max_frequency: The maximum frequency constraint for each channel. Must be
      greater than min_frequency. Defaults to None. If this value is set to
      None, the model's max frequency will be used.
    rf_channels: The R&F media channels in the model. When resolved with a
      model, the model's R&F channels will be present here. Ignored when used as
      input.
    kpi_type: A `common.KpiType` enum denoting whether the optimized KPI is of a
      `'revenue'` or `'non-revenue'` type.
  """

  min_frequency: float = 1.0
  max_frequency: float | None = None
  rf_channels: Sequence[str] = dataclasses.field(default_factory=list)
  kpi_type: common.KpiType = common.KpiType.REVENUE

  @property
  def selected_times(self) -> tuple[str | None, str | None] | None:
    """The start and end dates, as a tuple of date strings."""
    start, end = (None, None)
    if self.start_date is not None:
      start = self.start_date.strftime(constants.DATE_FORMAT)
    if self.end_date is not None:
      end = self.end_date.strftime(constants.DATE_FORMAT)

    if start or end:
      return (start, end)
    return None

  @property
  def objective(self) -> common.TargetMetric:
    """A Meridian budget optimization objective is always ROI."""
    return common.TargetMetric.ROI

  def validate(self):
    super().validate()
    if self.min_frequency < 0:
      raise ValueError("Min frequency must be non-negative.")
    if (
        self.max_frequency is not None
        and self.max_frequency < self.min_frequency
    ):
      raise ValueError("Max frequency must be greater than min frequency.")

  def to_proto(self) -> rf_pb.ReachFrequencyOptimizationSpec:
    # When invoked as an output proto, the spec should have been fully resolved.
    if self.start_date is None or self.end_date is None:
      raise ValueError(
          "Start and end dates must be resolved before this spec can be"
          " serialized."
      )

    return rf_pb.ReachFrequencyOptimizationSpec(
        date_interval=time_record.create_date_interval_pb(
            self.start_date, self.end_date, tag=self.date_interval_tag
        ),
        rf_channel_constraints=[
            rf_pb.RfChannelConstraint(
                channel_name=channel,
                frequency_constraint=constraints_pb.FrequencyConstraint(
                    min_frequency=self.min_frequency,
                    max_frequency=self.max_frequency,
                ),
            )
            for channel in self.rf_channels
        ],
        objective=self.objective.value,
        kpi_type=(
            kpi_type_pb.KpiType.REVENUE
            if self.kpi_type == common.KpiType.REVENUE
            else kpi_type_pb.KpiType.NON_REVENUE
        ),
    )


class ReachFrequencyOptimizationProcessor(
    model_processor.ModelProcessor[
        ReachFrequencyOptimizationSpec, rf_pb.ReachFrequencyOptimization
    ],
):
  """A Processor for marketing reach and frequency optimization."""

  def __init__(
      self,
      trained_model: model_processor.ModelType,
  ):
    trained_model = model_processor.ensure_trained_model(trained_model)
    self._internal_analyzer = trained_model.internal_analyzer
    self._meridian = trained_model.mmm

    if trained_model.mmm.input_data.rf_channel is None:
      raise ValueError("RF channels must be set in the model.")

    self._all_rf_channels = trained_model.mmm.input_data.rf_channel.data

  @classmethod
  def spec_type(cls) -> type[ReachFrequencyOptimizationSpec]:
    return ReachFrequencyOptimizationSpec

  @classmethod
  def output_type(cls) -> type[rf_pb.ReachFrequencyOptimization]:
    return rf_pb.ReachFrequencyOptimization

  def _to_target_precision(self, value: float) -> float:
    return round(value, _STEP_SIZE_DECIMAL_PRECISION)

  def _set_output(
      self, output: pb.Mmm, result: rf_pb.ReachFrequencyOptimization
  ):
    output.marketing_optimization.reach_frequency_optimization.CopyFrom(result)

  def execute(
      self, specs: Sequence[ReachFrequencyOptimizationSpec]
  ) -> rf_pb.ReachFrequencyOptimization:
    output = rf_pb.ReachFrequencyOptimization()

    group_ids = [spec.group_id for spec in specs if spec.group_id]
    if len(set(group_ids)) != len(group_ids):
      raise ValueError(
          "Specified group_id must be unique among the given group of specs."
      )

    for spec in specs:
      selected_times = spec.resolver(
          self._meridian.input_data.time_coordinates
      ).resolve_to_enumerated_selected_times()

      grid_min_freq = self._to_target_precision(spec.min_frequency)
      # If the max frequency is not set, use the model's max frequency.
      grid_max_freq = self._to_target_precision(
          spec.max_frequency or np.max(self._meridian.rf_tensors.frequency)
      )
      grid = [
          self._to_target_precision(f)
          for f in np.arange(grid_min_freq, grid_max_freq + _TOL, _STEP_SIZE)
      ]

      # Note that the internal analyzer, like the budget optimizer, maximizes
      # non-revenue KPI if input data is of non-revenue and the user selects
      # `use_kpi=True`. Otherwise, it maximizes revenue KPI.
      optimal_frequency = self._internal_analyzer.optimal_freq(
          selected_times=selected_times,
          confidence_level=spec.confidence_level,
          freq_grid=grid,
          use_kpi=(spec.kpi_type == common.KpiType.NON_REVENUE),
          max_frequency=spec.max_frequency,
      )
      response_curve = self._internal_analyzer.response_curves(
          confidence_level=spec.confidence_level,
          selected_times=selected_times,
          by_reach=False,
          use_kpi=(spec.kpi_type == common.KpiType.NON_REVENUE),
          use_optimal_frequency=True,
      )

      spend_data = self._compute_spend_data(selected_times=selected_times)

      # Obtain the output spec.
      start, end = spec.resolver(
          self._meridian.input_data.time_coordinates
      ).resolve_to_date_interval_open_end()

      # Copy the current spec, and resolve its date interval as well as model-
      # dependent parameters.
      output_spec = dataclasses.replace(
          spec,
          rf_channels=self._all_rf_channels,
          min_frequency=grid_min_freq,
          max_frequency=grid_max_freq,
          start_date=start,
          end_date=end,
      )

      output.results.append(
          self._to_reach_frequency_optimization_result(
              output_spec,
              optimal_frequency,
              response_curve,
              spend_data,
          )
      )
    return output

  def _compute_spend_data(
      self, selected_times: list[str] | None = None
  ) -> xr.Dataset:
    aggregated_spends = self._internal_analyzer.get_historical_spend(
        selected_times
    )
    aggregated_rf_spend = aggregated_spends.sel(
        {constants.CHANNEL: self._all_rf_channels}
    ).data
    total_spend = np.sum(aggregated_spends.data)
    pct_of_spend = 100.0 * aggregated_rf_spend / total_spend

    xr_dims = (constants.RF_CHANNEL,)
    xr_coords = {
        constants.RF_CHANNEL: (
            [constants.RF_CHANNEL],
            list(self._all_rf_channels),
        ),
    }
    xr_data_vars = {
        constants.SPEND: (xr_dims, aggregated_rf_spend),
        constants.PCT_OF_SPEND: (xr_dims, pct_of_spend),
    }

    return xr.Dataset(
        data_vars=xr_data_vars,
        coords=xr_coords,
    )

  def _to_reach_frequency_optimization_result(
      self,
      spec: ReachFrequencyOptimizationSpec,
      optimal_frequency: xr.Dataset,
      response_curve: xr.Dataset,
      spend_data: xr.Dataset,
  ) -> rf_pb.ReachFrequencyOptimizationResult:
    """Converts given optimal frequency dataset to protobuf form."""
    result = rf_pb.ReachFrequencyOptimizationResult(
        name=spec.optimization_name,
        spec=spec.to_proto(),
        optimized_channel_frequencies=_create_optimized_channel_frequencies(
            optimal_frequency
        ),
        optimized_marketing_analysis=self._to_marketing_analysis(
            spec,
            optimal_frequency,
            response_curve,
            spend_data,
        ),
        frequency_outcome_grid=self._create_frequency_outcome_grid(
            optimal_frequency,
            spec,
        ),
    )
    if spec.group_id:
      result.group_id = spec.group_id
    return result

  def _to_marketing_analysis(
      self,
      spec: ReachFrequencyOptimizationSpec,
      optimal_frequency: xr.Dataset,
      response_curve: xr.Dataset,
      spend_data: xr.Dataset,
  ) -> analysis_pb.MarketingAnalysis:
    """Converts an optimal frequency dataset to a `MarketingAnalysis` proto."""
    # `spec` should have been resolved with concrete date interval parameters.
    assert spec.start_date is not None and spec.end_date is not None

    optimized_marketing_analysis = analysis_pb.MarketingAnalysis(
        date_interval=time_record.create_date_interval_pb(
            start_date=spec.start_date,
            end_date=spec.end_date,
        ),
    )

    # Create a per-channel MediaAnalysis.
    channels = optimal_frequency.coords[constants.RF_CHANNEL].data
    for channel in channels:
      channel_optimal_frequency = optimal_frequency.sel(rf_channel=channel)
      channel_spend_data = spend_data.sel(rf_channel=channel)

      # TODO(b/360928000) Add non-media analyses.
      channel_media_analysis = media_analysis_pb.MediaAnalysis(
          channel_name=channel,
          response_curve=_compute_response_curve(
              response_curve,
              channel,
          ),
          spend_info=media_analysis_pb.SpendInfo(
              spend=channel_spend_data[constants.SPEND].data.item(),
              spend_share=(
                  channel_spend_data[constants.PCT_OF_SPEND].data.item()
              ),
          ),
      )

      # Output one outcome per channel: either revenue or non-revenue.
      channel_media_analysis.media_outcomes.append(
          _to_outcome(
              channel_optimal_frequency,
              is_revenue_kpi=optimal_frequency.attrs[constants.IS_REVENUE_KPI],
          )
      )

      optimized_marketing_analysis.media_analyses.append(channel_media_analysis)

    return optimized_marketing_analysis

  def _create_frequency_outcome_grid(
      self,
      optimal_frequency_dataset: xr.Dataset,
      spec: ReachFrequencyOptimizationSpec,
  ) -> rf_pb.FrequencyOutcomeGrid:
    """Creates a FrequencyOutcomeGrid proto."""
    channel_cells = []
    frequencies = optimal_frequency_dataset.coords[constants.FREQUENCY].data
    channels = optimal_frequency_dataset.coords[constants.RF_CHANNEL].data
    input_tensor_dims = "gtc"
    output_tensor_dims = "c"

    for channel in channels:
      cells = []
      for frequency in frequencies:
        new_frequency = (
            backend.ones_like(self._meridian.rf_tensors.frequency) * frequency
        )
        new_reach = (
            self._meridian.rf_tensors.frequency
            * self._meridian.rf_tensors.reach
            / new_frequency
        )
        channel_mask = [c == channel for c in channels]
        filtered_reach = backend.boolean_mask(new_reach, channel_mask, axis=2)
        aggregated_reach = backend.einsum(
            f"{input_tensor_dims}->...{output_tensor_dims}", filtered_reach
        )
        reach = aggregated_reach.numpy()[-1]

        metric_data_array = optimal_frequency_dataset[constants.ROI].sel(
            frequency=frequency, rf_channel=channel
        )
        outcome = common.to_estimate(metric_data_array, spec.confidence_level)

        cell = rf_pb.FrequencyOutcomeGrid.Cell(
            outcome=outcome,
            reach_frequency=rf_pb.ReachFrequency(
                reach=int(reach),
                average_frequency=frequency,
            ),
        )
        cells.append(cell)

      channel_cell = rf_pb.FrequencyOutcomeGrid.ChannelCells(
          channel_name=channel,
          cells=cells,
      )
      channel_cells.append(channel_cell)

    return rf_pb.FrequencyOutcomeGrid(
        name=spec.grid_name,
        frequency_step_size=_STEP_SIZE,
        channel_cells=channel_cells,
    )


def _create_optimized_channel_frequencies(
    optimal_frequency_dataset: xr.Dataset,
) -> list[rf_pb.OptimizedChannelFrequency]:
  """Creates an OptimizedChannelFrequency proto for each channel in the dataset."""
  optimal_frequency_protos = []
  optimal_frequency = optimal_frequency_dataset[constants.OPTIMAL_FREQUENCY]
  channels = optimal_frequency.coords[constants.RF_CHANNEL].data

  for channel in channels:
    optimal_frequency_protos.append(
        rf_pb.OptimizedChannelFrequency(
            channel_name=channel,
            optimal_average_frequency=optimal_frequency.sel(
                rf_channel=channel
            ).item(),
        )
    )
  return optimal_frequency_protos


def _to_outcome(
    channel_optimal_frequency: xr.Dataset,
    is_revenue_kpi: bool,
) -> outcome_pb.Outcome:
  """Returns an `Outcome` value for a given channel's optimized media analysis.

  Args:
    channel_optimal_frequency: A channel-selected dataset from
      `Analyzer.optimal_freq()`.
    is_revenue_kpi: Whether the KPI is revenue-based.
  """
  confidence_level = channel_optimal_frequency.attrs[constants.CONFIDENCE_LEVEL]
  return outcome_pb.Outcome(
      kpi_type=(
          kpi_type_pb.REVENUE if is_revenue_kpi else kpi_type_pb.NON_REVENUE
      ),
      roi=common.to_estimate(
          channel_optimal_frequency.optimized_roi, confidence_level
      ),
      marginal_roi=common.to_estimate(
          channel_optimal_frequency.optimized_mroi_by_frequency,
          confidence_level,
      ),
      cost_per_contribution=common.to_estimate(
          channel_optimal_frequency.optimized_cpik,
          confidence_level=confidence_level,
      ),
      contribution=outcome_pb.Contribution(
          value=common.to_estimate(
              channel_optimal_frequency.optimized_incremental_outcome,
              confidence_level,
          ),
      ),
      effectiveness=outcome_pb.Effectiveness(
          media_unit=constants.IMPRESSIONS,
          value=common.to_estimate(
              channel_optimal_frequency.optimized_effectiveness,
              confidence_level,
          ),
      ),
  )


def _compute_response_curve(
    response_curve_dataset: xr.Dataset,
    channel_name: str,
) -> response_curve_pb2.ResponseCurve:
  """Returns a ResponseCurve proto for the given channel.

  Args:
    response_curve_dataset: A dataset containing the data needed to generate a
      response curve.
    channel_name: The name of the channel to analyze.
  """

  spend_multiplier_list = response_curve_dataset.coords[
      constants.SPEND_MULTIPLIER
  ].data
  response_points: list[response_curve_pb2.ResponsePoint] = []

  for spend_multiplier in spend_multiplier_list:
    spend = (
        response_curve_dataset[constants.SPEND]
        .sel(spend_multiplier=spend_multiplier, channel=channel_name)
        .data.item()
    )
    incremental_outcome = (
        response_curve_dataset[constants.INCREMENTAL_OUTCOME]
        .sel(
            spend_multiplier=spend_multiplier,
            channel=channel_name,
            metric=constants.MEAN,
        )
        .data.item()
    )

    response_point = response_curve_pb2.ResponsePoint(
        input_value=spend,
        incremental_kpi=incremental_outcome,
    )
    response_points.append(response_point)

  return response_curve_pb2.ResponseCurve(
      input_name=constants.SPEND,
      response_points=response_points,
  )
