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

"""Meridian module for analyzing marketing data in a Meridian model.

This module provides a `MarketingProcessor`, designed to extract key marketing
insights from a trained Meridian model. It allows users to understand the impact
of different marketing channels, calculate return on investment (ROI), and
generate response curves.

The processor uses specifications defined in `MarketingAnalysisSpec` to control
the analysis. Users can request:

1.  **Media Summary Metrics:** Aggregated performance metrics for each media
    channel, including spend, contribution, ROI, and effectiveness.
2.  **Incremental Outcomes:** The additional KPI or revenue driven by marketing
    activities, calculated by comparing against a baseline scenario (e.g., zero
    spend).
3.  **Response Curves:** Visualizations of how the predicted KPI or revenue
    changes as spend on a particular channel increases, helping to identify
    diminishing returns.

The results are output as a `MarketingAnalysisList` protobuf message, containing
detailed breakdowns per channel and for the baseline.

Key Classes:

-   `MediaSummarySpec`: Configures the calculation of summary metrics like ROI.
-   `IncrementalOutcomeSpec`: Configures the calculation of incremental impact.
-   `ResponseCurveSpec`: Configures response curve generation.
-   `MarketingAnalysisSpec`: The main specification to combine the above,
    define date ranges, and set confidence levels.
-   `MarketingProcessor`: The processor class that executes the analysis based
    on the provided specs.

Example Usage:

1.  **Get Media Summary Metrics for a specific period:**

    ```python
    from schema.processors import marketing_processor
    import datetime

    # Assuming 'trained_model' is a loaded Meridian model object

    spec = marketing_processor.MarketingAnalysisSpec(
        analysis_name="q1_summary",
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 3, 31),
        media_summary_spec=marketing_processor.MediaSummarySpec(
            aggregate_times=True
        ),
        response_curve_spec=marketing_processor.ResponseCurveSpec(),
        confidence_level=0.9,
    )

    processor = marketing_processor.MarketingProcessor(trained_model)
    # `result` is a `marketing_analysis_pb2.MarketingAnalysisList` proto
    result = processor.execute([spec])
    ```

2.  **Calculate Incremental Outcome with new spend data:**

    ```python
    from schema.processors import marketing_processor
    from meridian.analysis import analyzer
    import datetime
    import numpy as np

    # Assuming 'trained_model' is a loaded Meridian model object
    # Assuming 'new_media_spend' is a numpy array with shape (time, channels)

    # Create DataTensors for the new data
    # Example:
    # new_data = analyzer.DataTensors(
    #     media=new_media_spend,
    #     time=new_time_index,
    # )

    spec = marketing_processor.MarketingAnalysisSpec(
        analysis_name="what_if_scenario",
        # NOTE: Dates must align with `new_data.time`
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
        incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(
            new_data=new_data,
            aggregate_times=True,
        ),
    )

    processor = marketing_processor.MarketingProcessor(trained_model)
    result = processor.execute([spec])

    print(f"Incremental Outcome for {spec.analysis_name}:")
    # Process results from result.marketing_analyses
    ```

Note: You can provide the processor with multiple specs. This would result in
multiple marketing analysis results in the output.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import functools
import warnings

from meridian import constants
from meridian.analysis import analyzer
from meridian.data import time_coordinates
from mmm.v1 import mmm_pb2
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import kpi_type_pb2
from mmm.v1.marketing.analysis import marketing_analysis_pb2
from mmm.v1.marketing.analysis import media_analysis_pb2
from mmm.v1.marketing.analysis import non_media_analysis_pb2
from mmm.v1.marketing.analysis import outcome_pb2
from mmm.v1.marketing.analysis import response_curve_pb2
from schema.processors import common
from schema.processors import model_processor
import numpy as np
import xarray as xr

__all__ = [
    "MediaSummarySpec",
    "IncrementalOutcomeSpec",
    "ResponseCurveSpec",
    "MarketingAnalysisSpec",
    "MarketingProcessor",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class MediaSummarySpec(model_processor.Spec):
  """Stores parameters needed for creating media summary metrics.

  Attributes:
    aggregate_times: Boolean. If `True`, the media summary metrics are
      aggregated over time. Defaults to `True`.
    marginal_roi_by_reach: Boolean. Marginal ROI (mROI) is defined as the return
      on the next dollar spent. If this argument is `True`, the assumption is
      that the next dollar spent only impacts reach, holding frequency constant.
      If this argument is `False`, the assumption is that the next dollar spent
      only impacts frequency, holding reach constant. Defaults to `True`.
    include_non_paid_channels: Boolean. If `True`, the media summary metrics
      include non-paid channels. Defaults to `False`.
    new_data: Optional `DataTensors` container with optional tensors: `media`,
      `reach`, `frequency`, `organic_media`, `organic_reach`,
      `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
      `None`, the metrics are calculated using the `InputData` provided to the
      Meridian object. If `new_data` is provided, the metrics are calculated
      using the new tensors in `new_data` and the original values of the
      remaining tensors.
    media_selected_times: Optional list containing booleans with length equal to
      the number of time periods in `new_data`, if provided. If `new_data` is
      provided, `media_selected_times` can select any subset of time periods in
      `new_data`.  If `new_data` is not provided, `media_selected_times` selects
      from model's original media data.
  """

  aggregate_times: bool = True
  marginal_roi_by_reach: bool = True
  include_non_paid_channels: bool = False
  new_data: analyzer.DataTensors | None = None
  media_selected_times: Sequence[bool] | None = None

  def validate(self):
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class IncrementalOutcomeSpec(model_processor.Spec):
  """Stores parameters needed for processing a model into `MarketingAnalysis`s.

  Attributes:
    aggregate_times: Boolean. If `True`, the media summary metrics are
      aggregated over time. Defaults to `True`.
    new_data: Optional `DataTensors` container with optional tensors: `media`,
      `reach`, `frequency`, `organic_media`, `organic_reach`,
      `organic_frequency`, `non_media_treatments` and `revenue_per_kpi`. If
      `None`, the incremental outcome is calculated using the `InputData`
      provided to the Meridian object. If `new_data` is provided, the
      incremental outcome is calculated using the new tensors in `new_data` and
      the original values of the remaining tensors. For example,
      `incremental_outcome(new_data=DataTensors(media=new_media)` computes the
      incremental outcome using `new_media` and the original values of `reach`,
      `frequency`, `organic_media`, `organic_reach`, `organic_frequency`,
      `non_media_treatments` and `revenue_per_kpi`. If any of the tensors in
      `new_data` is provided with a different number of time periods than in
      `InputData`, then all tensors must be provided with the same number of
      time periods.
    media_selected_times: Optional list containing booleans with length equal to
      the number of time periods in `new_data`, if provided. If `new_data` is
      provided, `media_selected_times` can select any subset of time periods in
      `new_data`.  If `new_data` is not provided, `media_selected_times` selects
      from model's original media data and its length must be equal to the
      number of time periods in the model's original media data.
    include_non_paid_channels: Boolean. If `True`, the incremental outcome
      includes non-paid channels. Defaults to `False`.
  """

  aggregate_times: bool = True
  new_data: analyzer.DataTensors | None = None
  media_selected_times: Sequence[bool] | None = None
  include_non_paid_channels: bool = False

  def validate(self):
    super().validate()
    if (self.new_data is not None) and (self.new_data.time is None):
      raise ValueError("`time` must be provided in `new_data`.")


@dataclasses.dataclass(frozen=True)
class ResponseCurveSpec(model_processor.Spec):
  """Stores parameters needed for creating response curves.

  Attributes:
    by_reach: Boolean. For channels with reach and frequency. If `True`, plots
      the response curve by reach. If `False`, plots the response curve by
      frequency.
  """

  by_reach: bool = True

  def validate(self):
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class MarketingAnalysisSpec(model_processor.DatedSpec):
  """Stores parameters needed for processing a model into `MarketingAnalysis`s.

  Either `media_summary_spec` or `incremental_outcome_spec` must be provided,
  but not both.

  Attributes:
    media_summary_spec: Parameters for creating media summary metrics. Mutually
      exclusive with `incremental_outcome_spec`.
    incremental_outcome_spec: Parameters for creating incremental outcome.
      Mutually exclusive with `media_summary_spec`. If `new_data` is provided,
      then the start and end dates of this `MarketingAnalysisSpec` must be
      within the `new_data.time`.
    response_curve_spec: Parameters for creating response curves. Response
      curves are only computed for specs that aggregate times and have a
      `media_summary_spec` selected.
    confidence_level: Confidence level for credible intervals, represented as a
      value between zero and one. Defaults to 0.9.
  """

  media_summary_spec: MediaSummarySpec | None = None
  incremental_outcome_spec: IncrementalOutcomeSpec | None = None
  response_curve_spec: ResponseCurveSpec = dataclasses.field(
      default_factory=ResponseCurveSpec
  )
  confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL

  def validate(self):
    super().validate()
    if self.confidence_level <= 0 or self.confidence_level >= 1:
      raise ValueError(
          "Confidence level must be greater than 0 and less than 1."
      )
    if (
        self.media_summary_spec is None
        and self.incremental_outcome_spec is None
    ):
      raise ValueError(
          "At least one of `media_summary_spec` or `incremental_outcome_spec`"
          " must be provided."
      )
    if (
        self.media_summary_spec is not None
        and self.incremental_outcome_spec is not None
    ):
      raise ValueError(
          "Only one of `media_summary_spec` or `incremental_outcome_spec` can"
          " be provided."
      )


class MarketingProcessor(
    model_processor.ModelProcessor[
        MarketingAnalysisSpec, marketing_analysis_pb2.MarketingAnalysisList
    ]
):
  """Generates `MarketingAnalysis` protos for a given trained Meridian model.

  A `MarketingAnalysis` proto is generated for each spec supplied to
  `execute()`.  Within each `MarketingAnalysis` proto, a `MediaAnalysis` proto
  is created for each channel in the model. One `NonMediaAnalysis` proto is also
  created for the model's baseline data.
  """

  def __init__(
      self,
      trained_model: model_processor.ModelType,
  ):
    trained_model = model_processor.ensure_trained_model(trained_model)
    self._analyzer = trained_model.internal_analyzer
    self._meridian = trained_model.mmm
    self._model_time_coordinates = trained_model.time_coordinates
    self._interval_length = self._model_time_coordinates.interval_days

    # If the input data KPI type is "revenue", then the `revenue_per_kpi` tensor
    # must exist, and general-KPI type outcomes should not be defined.
    self._revenue_kpi_type = (
        trained_model.mmm.input_data.kpi_type == constants.REVENUE
    )
    # `_kpi_only` is TRUE iff the input data KPI type is "non-revenue" AND the
    # `revenue_per_kpi` tensor is None.
    self._kpi_only = trained_model.mmm.input_data.revenue_per_kpi is None

  @classmethod
  def spec_type(cls) -> type[MarketingAnalysisSpec]:
    return MarketingAnalysisSpec

  @classmethod
  def output_type(cls) -> type[marketing_analysis_pb2.MarketingAnalysisList]:
    return marketing_analysis_pb2.MarketingAnalysisList

  def _set_output(
      self,
      output: mmm_pb2.Mmm,
      result: marketing_analysis_pb2.MarketingAnalysisList,
  ):
    output.marketing_analysis_list.CopyFrom(result)

  def execute(
      self, marketing_analysis_specs: Sequence[MarketingAnalysisSpec]
  ) -> marketing_analysis_pb2.MarketingAnalysisList:
    """Runs a marketing analysis on the model based on the given specs.

    A `MarketingAnalysis` proto is created for each of the given specs. Each
    `MarketingAnalysis` proto contains a list of `MediaAnalysis` protos and a
    singleton `NonMediaAnalysis` proto for the baseline analysis. The analysis
    covers the time period bounded by the spec's start and end dates.

    The singleton non-media analysis is performed on the model's baseline data,
    and contains metrics such as incremental outcome and baseline percent of
    contribution across media and non-media.

    A media analysis is performed for each channel in the model, plus an
    "All Channels" synthetic channel. The media analysis contains metrics such
    as spend, percent of spend, incremental outcome, percent of contribution,
    and effectiveness. Depending on the type of data (revenue-based or
    non-revenue-based) in the model, the analysis also contains CPIK
    (non-revenue-based) or ROI and MROI (revenue-based).

    Args:
      marketing_analysis_specs: A sequence of MarketingAnalysisSpec objects.

    Returns:
      A MarketingAnalysisList proto containing the results of the marketing
      analysis for each spec.
    """
    marketing_analysis_list: list[marketing_analysis_pb2.MarketingAnalysis] = []

    for spec in marketing_analysis_specs:
      if spec.incremental_outcome_spec is not None:
        new_data = spec.incremental_outcome_spec.new_data
      elif spec.media_summary_spec is not None:
        new_data = spec.media_summary_spec.new_data
      else:
        new_data = None

      if new_data is not None and new_data.time is not None:
        new_time_coords = time_coordinates.TimeCoordinates.from_dates(
            np.asarray(new_data.time).astype(str).tolist()
        )
        resolver = spec.resolver(new_time_coords)
      else:
        resolver = spec.resolver(self._model_time_coordinates)
      media_summary_marketing_analyses = (
          self._generate_marketing_analyses_for_media_summary_spec(
              spec, resolver
          )
      )
      incremental_outcome_marketing_analyses = (
          self._generate_marketing_analyses_for_incremental_outcome_spec(
              spec, resolver
          )
      )
      marketing_analysis_list.extend(
          media_summary_marketing_analyses
          + incremental_outcome_marketing_analyses
      )

    return marketing_analysis_pb2.MarketingAnalysisList(
        marketing_analyses=marketing_analysis_list
    )

  def _generate_marketing_analyses_for_media_summary_spec(
      self,
      marketing_analysis_spec: MarketingAnalysisSpec,
      resolver: model_processor.DatedSpecResolver,
  ) -> list[marketing_analysis_pb2.MarketingAnalysis]:
    """Creates a list of MarketingAnalysis protos based on the given spec.

    If spec's `aggregate_times` is True, then only one MarketingAnalysis proto
    is created. Otherwise, one MarketingAnalysis proto is created for each date
    interval in the spec.

    Args:
      marketing_analysis_spec: An instance of MarketingAnalysisSpec.
      resolver: A DatedSpecResolver instance.

    Returns:
      A list of `MarketingAnalysis` protos containing the results of the
      marketing analysis for the given spec.
    """
    media_summary_spec = marketing_analysis_spec.media_summary_spec
    if media_summary_spec is None:
      return []

    selected_times = resolver.resolve_to_enumerated_selected_times()
    # This contains either a revenue-based KPI or a non-revenue KPI analysis.
    media_summary_metrics, non_media_summary_metrics = (
        self._generate_media_and_non_media_summary_metrics(
            media_summary_spec,
            selected_times,
            marketing_analysis_spec.confidence_level,
            self._kpi_only,
        )
    )

    secondary_non_revenue_kpi_metrics = None
    secondary_non_revenue_kpi_non_media_metrics = None
    # If the input data KPI type is "non-revenue", and we calculated its
    # revenue-based KPI outcomes above, then we should also compute its
    # non-revenue KPI outcomes.
    if not self._revenue_kpi_type and not self._kpi_only:
      (
          secondary_non_revenue_kpi_metrics,
          secondary_non_revenue_kpi_non_media_metrics,
      ) = self._generate_media_and_non_media_summary_metrics(
          media_summary_spec,
          selected_times,
          marketing_analysis_spec.confidence_level,
          use_kpi=True,
      )

    # Note: baseline_summary_metrics() prefers computing revenue (scaled from
    # generic KPI with `revenue_per_kpi` when defined) baseline outcome here.
    # TODO: Baseline outcomes for both revenue and non-revenue
    # KPI types should be computed, when possible.
    baseline_outcome = self._analyzer.baseline_summary_metrics(
        confidence_level=marketing_analysis_spec.confidence_level,
        aggregate_times=media_summary_spec.aggregate_times,
        selected_times=selected_times,
    ).sel(distribution=constants.POSTERIOR)

    # Response curves are only computed for specs that aggregate times.
    if media_summary_spec.aggregate_times:
      response_curve_spec = marketing_analysis_spec.response_curve_spec
      response_curves = self._analyzer.response_curves(
          confidence_level=marketing_analysis_spec.confidence_level,
          use_posterior=True,
          selected_times=selected_times,
          use_kpi=self._kpi_only,
          by_reach=response_curve_spec.by_reach,
      )
    else:
      response_curves = None
      warnings.warn(
          "Response curves are not computed for non-aggregated time periods."
      )

    date_intervals = self._build_time_intervals(
        aggregate_times=media_summary_spec.aggregate_times,
        resolver=resolver,
    )

    return self._marketing_metrics_to_protos(
        media_summary_metrics,
        non_media_summary_metrics,
        baseline_outcome,
        secondary_non_revenue_kpi_metrics,
        secondary_non_revenue_kpi_non_media_metrics,
        response_curves,
        marketing_analysis_spec,
        date_intervals,
    )

  def _generate_media_and_non_media_summary_metrics(
      self,
      media_summary_spec: MediaSummarySpec,
      selected_times: list[str] | None,
      confidence_level: float,
      use_kpi: bool,
  ) -> tuple[xr.Dataset | None, xr.Dataset | None]:
    if media_summary_spec is None:
      return (None, None)
    compute_media_summary_metrics = functools.partial(
        self._analyzer.summary_metrics,
        marginal_roi_by_reach=media_summary_spec.marginal_roi_by_reach,
        selected_times=selected_times,
        aggregate_geos=True,
        aggregate_times=media_summary_spec.aggregate_times,
        new_data=media_summary_spec.new_data,
        confidence_level=confidence_level,
    )

    media_summary_metrics = compute_media_summary_metrics(
        use_kpi=use_kpi,
        include_non_paid_channels=False,
    ).sel(distribution=constants.POSTERIOR)
    # TODO:Produce one metrics for both paid and non-paid channels.
    non_media_summary_metrics = None
    if media_summary_spec.include_non_paid_channels:
      media_summary_metrics = media_summary_metrics.drop_sel(
          channel=constants.ALL_CHANNELS
      )
      non_media_summary_metrics = (
          compute_media_summary_metrics(
              use_kpi=use_kpi,
              include_non_paid_channels=True,
          )
          .sel(distribution=constants.POSTERIOR)
          .drop_sel(
              channel=media_summary_metrics.coords[constants.CHANNEL].data
          )
      )
    return media_summary_metrics, non_media_summary_metrics

  def _generate_marketing_analyses_for_incremental_outcome_spec(
      self,
      marketing_analysis_spec: MarketingAnalysisSpec,
      resolver: model_processor.DatedSpecResolver,
  ) -> list[marketing_analysis_pb2.MarketingAnalysis]:
    """Creates a list of `MarketingAnalysis` protos based on the given spec.

    If the spec's `aggregate_times` is True, then only one `MarketingAnalysis`
    proto is created. Otherwise, one `MarketingAnalysis` proto is created for
    each date interval in the spec.

    Args:
      marketing_analysis_spec: An instance of MarketingAnalysisSpec.
      resolver: A DatedSpecResolver instance.

    Returns:
      A list of `MarketingAnalysis` protos containing the results of the
      marketing analysis for the given spec.
    """
    incremental_outcome_spec = marketing_analysis_spec.incremental_outcome_spec
    if incremental_outcome_spec is None:
      return []

    compute_incremental_outcome = functools.partial(
        self._incremental_outcome_dataset,
        resolver=resolver,
        new_data=incremental_outcome_spec.new_data,
        media_selected_times=incremental_outcome_spec.media_selected_times,
        aggregate_geos=True,
        aggregate_times=incremental_outcome_spec.aggregate_times,
        confidence_level=marketing_analysis_spec.confidence_level,
        include_non_paid_channels=False,
    )
    # This contains either a revenue-based KPI or a non-revenue KPI analysis.
    incremental_outcome = compute_incremental_outcome(use_kpi=self._kpi_only)

    secondary_non_revenue_kpi_metrics = None
    # If the input data KPI type is "non-revenue", and we calculated its
    # revenue-based KPI outcomes above, then we should also compute its
    # non-revenue KPI outcomes.
    if not self._revenue_kpi_type and not self._kpi_only:
      secondary_non_revenue_kpi_metrics = compute_incremental_outcome(
          use_kpi=True
      )

    date_intervals = self._build_time_intervals(
        aggregate_times=incremental_outcome_spec.aggregate_times,
        resolver=resolver,
    )

    return self._marketing_metrics_to_protos(
        metrics=incremental_outcome,
        non_media_metrics=None,
        baseline_outcome=None,
        secondary_non_revenue_kpi_metrics=secondary_non_revenue_kpi_metrics,
        secondary_non_revenue_kpi_non_media_metrics=None,
        response_curves=None,
        marketing_analysis_spec=marketing_analysis_spec,
        date_intervals=date_intervals,
    )

  def _build_time_intervals(
      self,
      aggregate_times: bool,
      resolver: model_processor.DatedSpecResolver,
  ) -> list[date_interval_pb2.DateInterval]:
    """Creates a list of `DateInterval` protos for the given spec.

    Args:
      aggregate_times: Whether to aggregate times.
      resolver: A DatedSpecResolver instance.

    Returns:
      A list of `DateInterval` protos for the given spec.
    """
    if aggregate_times:
      date_interval = resolver.collapse_to_date_interval_proto()
      # This means metrics are aggregated over time, only one date interval is
      # needed.
      return [date_interval]

    # This list will contain all date intervals for the given spec. All dates
    # in this list will share a common tag.
    return resolver.transform_to_date_interval_protos()

  def _marketing_metrics_to_protos(
      self,
      metrics: xr.Dataset,
      non_media_metrics: xr.Dataset | None,
      baseline_outcome: xr.Dataset | None,
      secondary_non_revenue_kpi_metrics: xr.Dataset | None,
      secondary_non_revenue_kpi_non_media_metrics: xr.Dataset | None,
      response_curves: xr.Dataset | None,
      marketing_analysis_spec: MarketingAnalysisSpec,
      date_intervals: Sequence[date_interval_pb2.DateInterval],
  ) -> list[marketing_analysis_pb2.MarketingAnalysis]:
    """Creates a list of MarketingAnalysis protos from datasets."""
    if metrics is None:
      raise ValueError("metrics is None")

    media_channels = list(metrics.coords[constants.CHANNEL].data)
    non_media_channels = (
        list(non_media_metrics.coords[constants.CHANNEL].data)
        if non_media_metrics
        else []
    )
    channels = media_channels + non_media_channels
    channels_with_response_curve = (
        response_curves.coords[constants.CHANNEL].data
        if response_curves
        else []
    )
    marketing_analyses = []
    for date_interval in date_intervals:
      start_date = date_interval.start_date
      start_date_str = datetime.date(
          start_date.year, start_date.month, start_date.day
      ).strftime(constants.DATE_FORMAT)
      media_analyses: list[media_analysis_pb2.MediaAnalysis] = []
      non_media_analyses: list[non_media_analysis_pb2.NonMediaAnalysis] = []

      # For all channels reported in the media summary metrics
      for channel_name in channels:
        channel_response_curve = None
        if response_curves and (channel_name in channels_with_response_curve):
          channel_response_curve = response_curves.sel(
              {constants.CHANNEL: channel_name}
          )
        is_media_channel = channel_name in media_channels

        channel_analysis = self._get_channel_metrics(
            marketing_analysis_spec,
            channel_name,
            start_date_str,
            metrics if is_media_channel else non_media_metrics,
            secondary_non_revenue_kpi_metrics
            if is_media_channel
            else secondary_non_revenue_kpi_non_media_metrics,
            channel_response_curve,
            is_media_channel,
        )
        if isinstance(channel_analysis, media_analysis_pb2.MediaAnalysis):
          media_analyses.append(channel_analysis)

        if isinstance(
            channel_analysis, non_media_analysis_pb2.NonMediaAnalysis
        ):
          non_media_analyses.append(channel_analysis)

      marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
          date_interval=date_interval,
          media_analyses=media_analyses,
          non_media_analyses=non_media_analyses,
      )
      if baseline_outcome is not None:
        baseline_analysis = self._get_baseline_metrics(
            marketing_analysis_spec=marketing_analysis_spec,
            baseline_outcome=baseline_outcome,
            start_date=start_date_str,
        )
        marketing_analysis.non_media_analyses.append(baseline_analysis)

      marketing_analyses.append(marketing_analysis)

    return marketing_analyses

  def _get_channel_metrics(
      self,
      marketing_analysis_spec: MarketingAnalysisSpec,
      channel_name: str,
      start_date_str: str,
      metrics: xr.Dataset,
      secondary_metrics: xr.Dataset | None,
      channel_response_curves: xr.Dataset | None,
      is_media_channel: bool,
  ) -> (
      media_analysis_pb2.MediaAnalysis | non_media_analysis_pb2.NonMediaAnalysis
  ):
    """Returns a MediaAnalysis proto for the given channel."""
    if constants.TIME in metrics.coords:
      sel = {
          constants.CHANNEL: channel_name,
          constants.TIME: start_date_str,
      }
    else:
      sel = {constants.CHANNEL: channel_name}

    channel_metrics = metrics.sel(sel)
    if secondary_metrics is not None:
      channel_secondary_metrics = secondary_metrics.sel(sel)
    else:
      channel_secondary_metrics = None

    return self._channel_metrics_to_proto(
        channel_metrics,
        channel_secondary_metrics,
        channel_response_curves,
        channel_name,
        is_media_channel,
        marketing_analysis_spec.confidence_level,
    )

  def _channel_metrics_to_proto(
      self,
      channel_media_summary_metrics: xr.Dataset,
      channel_secondary_non_revenue_metrics: xr.Dataset | None,
      channel_response_curve: xr.Dataset | None,
      channel_name: str,
      is_media_channel: bool,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> (
      media_analysis_pb2.MediaAnalysis | non_media_analysis_pb2.NonMediaAnalysis
  ):
    """Creates a MediaAnalysis proto for the given channel from datasets.

    Args:
      channel_media_summary_metrics: A dataset containing the model's media
        summary metrics. This dataset is pre-filtered to `channel_name`. This
        dataset contains revenue-based metrics if the model's input data is
        revenue-based, or if `revenue_per_kpi` is defined. Otherwise, it
        contains non-revenue generic KPI metrics.
      channel_secondary_non_revenue_metrics: A dataset containing the model's
        non-revenue-based media summary metrics. This is only defined iff the
        input data is non-revenue type AND `revenue_per_kpi` is available. In
        this case, `channel_media_summary_metrics` contains revenue-based
        metrics computed from `KPI * revenue_per_kpi`, and this dataset contains
        media summary metrics based on the model's generic KPI alone. In all
        other cases, this is `None`.
      channel_response_curve: A dataset containing the data needed to generate a
        response curve. This dataset is pre-filtered to `channel_name`.
      channel_name: The name of the channel to analyze.
      is_media_channel: Whether the channel is a media channel.
      confidence_level: Confidence level for credible intervals, represented as
        a value between zero and one.

    Returns:
      A proto containing the media analysis results for the given channel.
    """

    spend_info = _compute_spend(channel_media_summary_metrics)
    is_all_channels = channel_name == constants.ALL_CHANNELS

    compute_outcome = functools.partial(
        self._compute_outcome,
        is_all_channels=is_all_channels,
        confidence_level=confidence_level,
    )

    outcomes = [
        compute_outcome(
            channel_media_summary_metrics,
            is_revenue_type=(not self._kpi_only),
        )
    ]
    # If `channel_media_summary_metrics` represented non-revenue data with
    # revenue-type outcome (i.e. `is_revenue_type_kpi` is defined), then we
    # should also have been provided with media summary metrics for their
    # generic KPI counterparts, as well.
    if channel_secondary_non_revenue_metrics is not None:
      outcomes.append(
          compute_outcome(
              channel_secondary_non_revenue_metrics,
              is_revenue_type=False,
          )
      )

    if not is_media_channel:
      return non_media_analysis_pb2.NonMediaAnalysis(
          non_media_name=channel_name,
          non_media_outcomes=outcomes,
      )

    media_analysis = media_analysis_pb2.MediaAnalysis(
        channel_name=channel_name,
        media_outcomes=outcomes,
    )

    if spend_info is not None:
      media_analysis.spend_info.CopyFrom(spend_info)

    if channel_response_curve is not None:
      media_analysis.response_curve.CopyFrom(
          self._compute_response_curve(
              channel_response_curve,
          )
      )

    return media_analysis

  def _get_baseline_metrics(
      self,
      marketing_analysis_spec: MarketingAnalysisSpec,
      baseline_outcome: xr.Dataset,
      start_date: str,
  ) -> non_media_analysis_pb2.NonMediaAnalysis:
    """Analyzes "baseline" pseudo-channel outcomes over the given time points.

    Args:
      marketing_analysis_spec: A user input parameter specs for this analysis.
      baseline_outcome: A dataset containing the model's baseline summary
        metrics.
      start_date: The date of the analysis.

    Returns:
      A `NonMediaAnalysis` representing baseline analysis.
    """
    if constants.TIME in baseline_outcome.coords:
      baseline_outcome = baseline_outcome.sel(
          time=start_date,
      )
    incremental_outcome = baseline_outcome[constants.BASELINE_OUTCOME]
    # Convert percentage to decimal.
    contribution_share = baseline_outcome[constants.PCT_OF_CONTRIBUTION] / 100

    contribution = outcome_pb2.Contribution(
        value=common.to_estimate(
            incremental_outcome, marketing_analysis_spec.confidence_level
        ),
        share=common.to_estimate(
            contribution_share, marketing_analysis_spec.confidence_level
        ),
    )
    baseline_analysis = non_media_analysis_pb2.NonMediaAnalysis(
        non_media_name=constants.BASELINE,
    )
    baseline_outcome = outcome_pb2.Outcome(
        contribution=contribution,
        # Baseline outcome is always revenue-based, unless `revenue_per_kpi`
        # is undefined.
        # TODO: kpi_type here is synced with what is used inside
        # `baseline_summary_metrics()`. Ideally, really, we should inject this
        # value into that function rather than re-deriving it here.
        kpi_type=(
            kpi_type_pb2.KpiType.NON_REVENUE
            if self._kpi_only
            else kpi_type_pb2.KpiType.REVENUE
        ),
    )
    baseline_analysis.non_media_outcomes.append(baseline_outcome)

    return baseline_analysis

  def _compute_outcome(
      self,
      media_summary_metrics: xr.Dataset,
      is_revenue_type: bool,
      is_all_channels: bool,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
  ) -> outcome_pb2.Outcome:
    """Returns an `Outcome` proto for the given channel's media analysis.

    Args:
      media_summary_metrics: A dataset containing the model's media summary
        metrics.
      is_revenue_type: Whether the media summary metrics above are revenue
        based.
      is_all_channels: If True, the given media summary represents the aggregate
        "All Channels". Omit `effectiveness` and `mroi` in this case.
      confidence_level: Confidence level for credible intervals, represented as
        a value between zero and one.
    """
    data_vars = media_summary_metrics.data_vars

    effectiveness = roi = mroi = cpik = None
    if not is_all_channels and constants.EFFECTIVENESS in data_vars:
      effectiveness = outcome_pb2.Effectiveness(
          media_unit=constants.IMPRESSIONS,
          value=common.to_estimate(
              media_summary_metrics[constants.EFFECTIVENESS],
              confidence_level,
          ),
      )
    if not is_all_channels and constants.MROI in data_vars:
      mroi = common.to_estimate(
          media_summary_metrics[constants.MROI],
          confidence_level,
      )

    contribution_value = media_summary_metrics[constants.INCREMENTAL_OUTCOME]
    contribution = outcome_pb2.Contribution(
        value=common.to_estimate(
            contribution_value,
            confidence_level,
        ),
    )
    # Convert percentage to decimal.
    if constants.PCT_OF_CONTRIBUTION in data_vars:
      contribution_share = (
          media_summary_metrics[constants.PCT_OF_CONTRIBUTION] / 100
      )
      contribution.share.CopyFrom(
          common.to_estimate(
              contribution_share,
              confidence_level,
          )
      )

    if constants.CPIK in data_vars:
      cpik = common.to_estimate(
          media_summary_metrics[constants.CPIK],
          confidence_level,
          metric=constants.MEDIAN,
      )

    if constants.ROI in data_vars:
      roi = common.to_estimate(
          media_summary_metrics[constants.ROI],
          confidence_level,
      )

    return outcome_pb2.Outcome(
        kpi_type=(
            kpi_type_pb2.KpiType.REVENUE
            if is_revenue_type
            else kpi_type_pb2.KpiType.NON_REVENUE
        ),
        contribution=contribution,
        effectiveness=effectiveness,
        cost_per_contribution=cpik,
        roi=roi,
        marginal_roi=mroi,
    )

  def _compute_response_curve(
      self,
      response_curve_dataset: xr.Dataset,
  ) -> response_curve_pb2.ResponseCurve:
    """Returns a `ResponseCurve` proto for the given channel.

    Args:
      response_curve_dataset: A dataset containing the data needed to generate a
        response curve.
    """

    spend_multiplier_list = response_curve_dataset.coords[
        constants.SPEND_MULTIPLIER
    ].data
    response_points: list[response_curve_pb2.ResponsePoint] = []

    for spend_multiplier in spend_multiplier_list:
      spend = (
          response_curve_dataset[constants.SPEND]
          .sel(spend_multiplier=spend_multiplier)
          .data.item()
      )
      incremental_outcome = (
          response_curve_dataset[constants.INCREMENTAL_OUTCOME]
          .sel(
              spend_multiplier=spend_multiplier,
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

  # TODO: Create an abstraction/container around these inference
  # parameters.
  def _incremental_outcome_dataset(
      self,
      resolver: model_processor.DatedSpecResolver,
      new_data: analyzer.DataTensors | None = None,
      media_selected_times: Sequence[bool] | None = None,
      selected_geos: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      use_kpi: bool = False,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
      include_non_paid_channels: bool = False,
  ) -> xr.Dataset:
    """Returns incremental outcome for each channel with dimensions.

    Args:
      resolver: A `DatedSpecResolver` instance.
      new_data: A dataset containing the new data to use in the analysis.
      media_selected_times: A boolean array of length `n_times` indicating which
        time periods are media-active.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      aggregate_geos: Boolean. If `True`, the expected outcome is summed over
        all of the regions.
      aggregate_times: Boolean. If `True`, the expected outcome is summed over
        all of the time periods.
      use_kpi: Boolean. If `True`, the summary metrics are calculated using KPI.
        If `False`, the metrics are calculated using revenue.
      confidence_level: Confidence level for summary metrics credible intervals,
        represented as a value between zero and one.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.
      include_non_paid_channels: Boolean. If `True`, non-paid channels (organic
        media, organic reach and frequency, and non-media treatments) are
        included in the summary but only the metrics independent of spend are
        reported. If `False`, only the paid channels (media, reach and
        frequency) are included but the summary contains also the metrics
        dependent on spend. Default: `False`.

    Returns:
      An `xr.Dataset` and containing `incremental_outcome` for each channel. The
      coordinates are: `channel` and `metric` (`mean`, `median`, `ci_low`,
      `ci_high`)
    """
    # Selected times in boolean form are supported by the analyzer with and
    # without the new data.
    selected_times_bool = resolver.resolve_to_bool_selected_times()
    kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times_bool,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
        "batch_size": batch_size,
    }
    incremental_outcome_posterior = (
        self._analyzer.compute_incremental_outcome_aggregate(
            new_data=new_data,
            media_selected_times=media_selected_times,
            use_posterior=True,
            use_kpi=use_kpi,
            include_non_paid_channels=include_non_paid_channels,
            **kwargs,
        )
    )

    xr_dims = (
        ((constants.GEO,) if not aggregate_geos else ())
        + ((constants.TIME,) if not aggregate_times else ())
        + (constants.CHANNEL, constants.METRIC)
    )
    channels = (
        self._meridian.input_data.get_all_channels()
        if include_non_paid_channels
        else self._meridian.input_data.get_all_paid_channels()
    )
    xr_coords = {
        constants.CHANNEL: (
            [constants.CHANNEL],
            list(channels) + [constants.ALL_CHANNELS],
        ),
    }
    if not aggregate_geos:
      geo_dims = (
          self._meridian.input_data.geo.data
          if selected_geos is None
          else selected_geos
      )
      xr_coords[constants.GEO] = ([constants.GEO], geo_dims)
    if not aggregate_times:
      selected_times_str = resolver.resolve_to_enumerated_selected_times()
      if selected_times_str is not None:
        time_dims = selected_times_str
      else:
        time_dims = resolver.time_coordinates.all_dates_str
      xr_coords[constants.TIME] = ([constants.TIME], time_dims)
    xr_coords_with_ci = {
        constants.METRIC: (
            [constants.METRIC],
            [
                constants.MEAN,
                constants.MEDIAN,
                constants.CI_LO,
                constants.CI_HI,
            ],
        ),
        **xr_coords,
    }
    metrics = analyzer.get_central_tendency_and_ci(
        incremental_outcome_posterior, confidence_level, include_median=True
    )
    xr_data = {constants.INCREMENTAL_OUTCOME: (xr_dims, metrics)}
    return xr.Dataset(data_vars=xr_data, coords=xr_coords_with_ci)


def _compute_spend(
    media_summary_metrics: xr.Dataset,
) -> media_analysis_pb2.SpendInfo | None:
  """Returns a `SpendInfo` proto with spend information for the given channel.

  Args:
    media_summary_metrics: A dataset containing the model's media summary
      metrics.
  """
  if constants.SPEND not in media_summary_metrics.data_vars:
    return None

  spend = media_summary_metrics[constants.SPEND].item()
  spend_share = media_summary_metrics[constants.PCT_OF_SPEND].data.item() / 100

  return media_analysis_pb2.SpendInfo(
      spend=spend,
      spend_share=spend_share,
  )
