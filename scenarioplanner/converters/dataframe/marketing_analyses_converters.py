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

"""Marketing analyses converters.

This module defines various classes that convert `MarketingAnalysis`s into flat
dataframes.
"""

import abc
from collections.abc import Iterator, Sequence
import datetime
import functools
import math
import warnings

from meridian import constants as c
from mmm.v1.fit import model_fit_pb2 as fit_pb
from scenarioplanner.converters import mmm
from scenarioplanner.converters.dataframe import constants as dc
from scenarioplanner.converters.dataframe import converter
import pandas as pd


__all__ = [
    "ModelDiagnosticsConverter",
    "ModelFitConverter",
    "MediaOutcomeConverter",
    "MediaSpendConverter",
    "MediaRoiConverter",
]


class ModelDiagnosticsConverter(converter.Converter):
  """Outputs a "ModelDiagnostics" table.

  When called, this converter yields a data frame with the columns:

  *   "Dataset"
  *   "R Squared"
  *   "MAPE"
  *   "wMAPE"
  """

  def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
    if not self._mmm.model_fit_results:
      return

    model_diagnostics_data = []
    for name, result in self._mmm.model_fit_results.items():
      model_diagnostics_data.append((
          name,
          result.performance.r_squared,
          result.performance.mape,
          result.performance.weighted_mape,
      ))
    yield (
        dc.MODEL_DIAGNOSTICS,
        pd.DataFrame(
            model_diagnostics_data,
            columns=[
                dc.MODEL_DIAGNOSTICS_DATASET_COLUMN,
                dc.MODEL_DIAGNOSTICS_R_SQUARED_COLUMN,
                dc.MODEL_DIAGNOSTICS_MAPE_COLUMN,
                dc.MODEL_DIAGNOSTICS_WMAPE_COLUMN,
            ],
        ),
    )


class ModelFitConverter(converter.Converter):
  """Outputs a "ModelFit" table from an "All Data" (*) `Result` dataset.

  Note: If there is no such result dataset, the first one available is used,
  instead.

  When called, this converter yields a data frame with the columns:

  *   "Time"
      A string formatted with Meridian date format: YYYY-mm-dd
  *   "Expected CI Low"
  *   "Expected CI High"
  *   "Expected"
  *   "Baseline"
  *   "Actual"
  """

  def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
    if not self._mmm.model_fit_results:
      return

    model_fit_data = []
    for prediction in self._select_model_fit_result().predictions:
      time = datetime.datetime(
          year=prediction.date_interval.start_date.year,
          month=prediction.date_interval.start_date.month,
          day=prediction.date_interval.start_date.day,
      ).strftime(c.DATE_FORMAT)

      if not prediction.predicted_outcome.uncertainties:
        (expected_ci_lo, expected_ci_hi) = (math.nan, math.nan)
      else:
        if len(prediction.predicted_outcome.uncertainties) > 1:
          warnings.warn(
              "More than one `Estimate.uncertainties` found in a"
              " `Prediction.predicted_outcome` in `ModelFit`; processing only"
              " the first confidence interval value."
          )
        uncertainty = prediction.predicted_outcome.uncertainties[0]
        expected_ci_lo = uncertainty.lowerbound
        expected_ci_hi = uncertainty.upperbound
      expected = prediction.predicted_outcome.value
      actual = prediction.actual_value

      baseline = prediction.predicted_baseline.value

      model_fit_data.append(
          (time, expected_ci_lo, expected_ci_hi, expected, baseline, actual)
      )

    yield (
        dc.MODEL_FIT,
        pd.DataFrame(
            model_fit_data,
            columns=[
                dc.MODEL_FIT_TIME_COLUMN,
                dc.MODEL_FIT_EXPECTED_CI_LOW_COLUMN,
                dc.MODEL_FIT_EXPECTED_CI_HIGH_COLUMN,
                dc.MODEL_FIT_EXPECTED_COLUMN,
                dc.MODEL_FIT_BASELINE_COLUMN,
                dc.MODEL_FIT_ACTUAL_COLUMN,
            ],
        ),
    )

  def _select_model_fit_result(self) -> fit_pb.Result:
    """Returns the model fit `Result` dataset with name "All Data".

    Or else, first available.
    """
    model_fit_results = self._mmm.model_fit_results
    if not model_fit_results:
      raise ValueError("Must have at least one `ModelFit.results` value.")
    if c.ALL_DATA in model_fit_results:
      result = model_fit_results[c.ALL_DATA]
    else:
      result = self._mmm.model_fit.results[0]
      warnings.warn(f"Using a model fit `Result` with name: '{result.name}'")

    return result


class _MarketingAnalysisConverter(converter.Converter, abc.ABC):
  """An abstract class for dealing with `MarketingAnalysis`."""

  @functools.cached_property
  def _is_revenue_kpi(self) -> bool:
    """Returns true if analyses are using revenue KPI.

    This is done heuristically: by looking at the (presumed existing) "baseline"
    `NonMediaAnalysis` proto and seeing if `revenue_kpi` field is defined. If it
    is, we assume that all other media analyses must have their `revenue_kpi`
    fields defined, too.

    Likewise: if the baseline analysis defines `non_revenue_kpi` and it does not
    define `revenue_kpi`, we assume that all other media analyses are based on
    a non-revenue KPI.

    Note: This means that this output converter can only work with one type of
    KPI as a whole. If a channel's media analysis has both revenue- and
    nonrevenue-type KPI defined, for example, only the former will be outputted.
    """
    baseline_analysis = self._mmm.tagged_marketing_analyses[
        dc.ANALYSIS_TAG_ALL
    ].baseline_analysis
    return baseline_analysis.maybe_revenue_kpi_outcome is not None

  def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
    if not self._mmm.marketing_analyses:
      return

    yield from self._handle_marketing_analyses(self._mmm.marketing_analyses)

  def _handle_marketing_analyses(
      self, analyses: Sequence[mmm.MarketingAnalysis]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    raise NotImplementedError()


class MediaOutcomeConverter(_MarketingAnalysisConverter):
  """Outputs a "MediaOutcome" table.

  When called, this converter yields a data frame with the columns:

  *   "Channel Index"
      This is to ensure "baseline" and "All Channels" can be sorted to appear
      first and last, respectively, in LS dashboard.
  *   "Channel"
  *   "Incremental Outcome"
  *   "Contribution Share"
  *   "Analysis Period"
      A human-readable analysis period.
  *   "Analysis Date Start"
      A string formatted with Meridian date format: YYYY-mm-dd
  *   "Analysis Date End"
      A string formatted with Meridian date format: YYYY-mm-dd

  Note: If the underlying model analysis works with a revenue-type KPI (i.e.
  dollar value), then all values in the columns of the output table should be
  interpreted the same. Likewise, for non-revenue type KPI. While some
  channels may define their KPI outcome analyses in terms of both revenue- and
  nonrevenue-type semantics, the output table here remains uniform.
  """

  def _handle_marketing_analyses(
      self, analyses: Sequence[mmm.MarketingAnalysis]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    media_outcome_data = []
    for analysis in analyses:
      date_start, date_end = analysis.analysis_date_interval_str

      baseline_kpi_outcome: mmm.KpiOutcome = (
          analysis.baseline_analysis.revenue_kpi_outcome
          if self._is_revenue_kpi
          else analysis.baseline_analysis.non_revenue_kpi_outcome
      )
      # "contribution" == incremental outcome
      baseline_contrib = baseline_kpi_outcome.contribution_pb.value.value
      baseline_contrib_share = baseline_kpi_outcome.contribution_pb.share.value

      media_outcome_data.append((
          0,  # "baseline" pseudo-channel should be indexed first
          c.BASELINE,
          baseline_contrib,
          baseline_contrib_share,
          analysis.tag,
          date_start,
          date_end,
      ))

      for (
          channel,
          media_analysis,
      ) in analysis.channel_mapped_media_analyses.items():
        channel_index = 2 if channel == c.ALL_CHANNELS else 1
        # Note: use the same revenue- or nonrevenue-type KPI outcome analysis
        # as the baseline's.
        try:
          channel_kpi_outcome: mmm.KpiOutcome = (
              media_analysis.revenue_kpi_outcome
              if self._is_revenue_kpi
              else media_analysis.non_revenue_kpi_outcome
          )
        except ValueError:
          warnings.warn(
              f"No {'' if self._is_revenue_kpi else 'non'}revenue-type"
              " `KpiOutcome` found in the channel media analysis for"
              f' "{channel}"'
          )
          channel_contrib = math.nan
          channel_contrib_share = math.nan
        else:
          channel_contrib = channel_kpi_outcome.contribution_pb.value.value
          channel_contrib_share = (
              channel_kpi_outcome.contribution_pb.share.value
          )

        media_outcome_data.append((
            channel_index,
            channel,
            channel_contrib,
            channel_contrib_share,
            analysis.tag,
            date_start,
            date_end,
        ))

    yield (
        dc.MEDIA_OUTCOME,
        pd.DataFrame(
            media_outcome_data,
            columns=[
                dc.MEDIA_OUTCOME_CHANNEL_INDEX_COLUMN,
                dc.MEDIA_OUTCOME_CHANNEL_COLUMN,
                dc.MEDIA_OUTCOME_INCREMENTAL_OUTCOME_COLUMN,
                dc.MEDIA_OUTCOME_CONTRIBUTION_SHARE_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,  # using the `tag` field
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


class MediaSpendConverter(_MarketingAnalysisConverter):
  """Outputs a "MediaSpend" table.

  When called, this converter yields a data frame with the columns:

  *   "Channel"
  *   "Value"
  *   "Label"
      A human-readable label on what "Value" represents
  *   "Analysis Period"
      A human-readable analysis period.
  *   "Analysis Date Start"
      A string formatted with Meridian date format: YYYY-mm-dd
  *   "Analysis Date End"
      A string formatted with Meridian date format: YYYY-mm-dd

  Note: If the underlying model analysis works with a revenue-type KPI (i.e.
  dollar value), then all values in the columns of the output table should
  be interpreted the same. Likewise, for non-revenue type KPI. While some
  channels may define their KPI outcome analyses in terms of both revenue-
  and nonrevenue-type semantics, the output table here remains uniform.
  """
  _share_value_column_index = 1
  _label_column_index = 2

  def _handle_marketing_analyses(
      self, analyses: Sequence[mmm.MarketingAnalysis]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    media_spend_data = []

    for analysis in analyses:
      date_start, date_end = analysis.analysis_date_interval_str
      data = []
      outcome_share_norm_term = 0.0

      for (
          channel,
          media_analysis,
      ) in analysis.channel_mapped_media_analyses.items():
        # Ignore the "All Channels" pseudo-channel.
        if channel == c.ALL_CHANNELS:
          continue

        spend_share = media_analysis.spend_info_pb.spend_share

        try:
          channel_kpi_outcome: mmm.KpiOutcome = (
              media_analysis.revenue_kpi_outcome
              if self._is_revenue_kpi
              else media_analysis.non_revenue_kpi_outcome
          )
        except ValueError:
          warnings.warn(
              f"No {'' if self._is_revenue_kpi else 'non'}revenue-type"
              " `KpiOutcome` found in the channel media analysis for"
              f' "{channel}"'
          )
          outcome_share = math.nan
        else:
          outcome_share = channel_kpi_outcome.contribution_pb.share.value
          outcome_share_norm_term += outcome_share

        data.append([
            channel,
            spend_share,
            dc.MEDIA_SPEND_LABEL_SPEND_SHARE,
            analysis.tag,
            date_start,
            date_end,
        ])
        data.append([
            channel,
            outcome_share,
            (
                dc.MEDIA_SPEND_LABEL_REVENUE_SHARE
                if self._is_revenue_kpi
                else dc.MEDIA_SPEND_LABEL_KPI_SHARE
            ),
            analysis.tag,
            date_start,
            date_end,
        ])

      # Looker Studio media spend/revenue share charts expect the "revenue
      # share" values to be normalized to 100%. This normaliztion provides
      # additional information to what the contribution waterfall chart already
      # provides.
      for d in data:
        if d[self._label_column_index] == dc.MEDIA_SPEND_LABEL_SPEND_SHARE:
          continue
        d[self._share_value_column_index] /= outcome_share_norm_term

      media_spend_data.extend(data)

    yield (
        dc.MEDIA_SPEND,
        pd.DataFrame(
            media_spend_data,
            columns=[
                dc.MEDIA_SPEND_CHANNEL_COLUMN,
                dc.MEDIA_SPEND_SHARE_VALUE_COLUMN,
                dc.MEDIA_SPEND_LABEL_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,  # using the `tag` field
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


class MediaRoiConverter(_MarketingAnalysisConverter):
  """Outputs a "MediaROI" table.

  When called, this converter yields a data frame with the columns:

    *   "Channel"
    *   "Spend"
    *   "Effectiveness"
    *   "ROI"
    *   "ROI CI Low"
        The confidence interval (low) of "ROI" above.
    *   "ROI CI High"
        The confidence interval (high) of "ROI" above.
    *   "Marginal ROI"
    *   "Is Revenue KPI"
        A boolean indicating whether "ROI" refers to revenue or generic KPI.
    *   "Analysis Period"
        A human-readable analysis period.
    *   "Analysis Date Start"
        A string formatted with Meridian date format: YYYY-mm-dd
    *   "Analysis Date End"
        A string formatted with Meridian date format: YYYY-mm-dd

  Note: If the underlying model analysis works with a revenue-type KPI (i.e.
  dollar value), then all values in the columns of the output table should
  be interpreted the same. Likewise, for non-revenue type KPI. While some
  channels may define their KPI outcome analyses in terms of both revenue-
  and nonrevenue-type semantics, the output table here remains uniform.
  """

  def _handle_marketing_analyses(
      self, analyses: Sequence[mmm.MarketingAnalysis]
  ) -> Iterator[tuple[str, pd.DataFrame]]:
    media_roi_data = []
    for analysis in analyses:
      date_start, date_end = analysis.analysis_date_interval_str

      for (
          channel,
          media_analysis,
      ) in analysis.channel_mapped_media_analyses.items():
        # Ignore the "All Channels" pseudo-channel.
        if channel == c.ALL_CHANNELS:
          continue

        spend = media_analysis.spend_info_pb.spend

        try:
          channel_kpi_outcome: mmm.KpiOutcome = (
              media_analysis.revenue_kpi_outcome
              if self._is_revenue_kpi
              else media_analysis.non_revenue_kpi_outcome
          )
        except ValueError as exc:
          raise ValueError(
              f"No {'' if self._is_revenue_kpi else 'non'}revenue-type"
              " `KpiOutcome` found in the channel media analysis for"
              f' "{channel}"'
          ) from exc
        else:
          effectiveness = channel_kpi_outcome.effectiveness_pb.value.value
          roi_estimate = channel_kpi_outcome.roi_pb
          if not roi_estimate.uncertainties:
            (roi_ci_lo, roi_ci_hi) = (math.nan, math.nan)
          else:
            if len(roi_estimate.uncertainties) > 1:
              warnings.warn(
                  "More than one `Estimate.uncertainties` found in a"
                  ' `KpiOutcome.revenue_outcome.roi` in channel "{channel}".'
                  " Using the first confidence interval value."
              )
            uncertainty = roi_estimate.uncertainties[0]
            roi_ci_lo = uncertainty.lowerbound
            roi_ci_hi = uncertainty.upperbound
          roi = roi_estimate.value
          marginal_roi = channel_kpi_outcome.marginal_roi_pb.value
          is_revenue_kpi = channel_kpi_outcome.is_revenue_kpi

        media_roi_data.append([
            channel,
            spend,
            effectiveness,
            roi,
            roi_ci_lo,
            roi_ci_hi,
            marginal_roi,
            is_revenue_kpi,
            analysis.tag,
            date_start,
            date_end,
        ])

    yield (
        dc.MEDIA_ROI,
        pd.DataFrame(
            media_roi_data,
            columns=[
                dc.MEDIA_ROI_CHANNEL_COLUMN,
                dc.MEDIA_ROI_SPEND_COLUMN,
                dc.MEDIA_ROI_EFFECTIVENESS_COLUMN,
                dc.MEDIA_ROI_ROI_COLUMN,
                dc.MEDIA_ROI_ROI_CI_LOW_COLUMN,
                dc.MEDIA_ROI_ROI_CI_HIGH_COLUMN,
                dc.MEDIA_ROI_MARGINAL_ROI_COLUMN,
                dc.MEDIA_ROI_IS_REVENUE_KPI_COLUMN,
                dc.ANALYSIS_PERIOD_COLUMN,
                dc.ANALYSIS_DATE_START_COLUMN,
                dc.ANALYSIS_DATE_END_COLUMN,
            ],
        ),
    )


CONVERTERS = [
    # These converters create tables for the model analysis charts to use:
    ModelDiagnosticsConverter,
    ModelFitConverter,
    MediaOutcomeConverter,
    MediaSpendConverter,
    MediaRoiConverter,
]
