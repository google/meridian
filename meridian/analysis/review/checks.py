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

"""Implementation of the Model Quality Checks."""

import abc
from collections.abc import Sequence
import dataclasses
from typing import Generic, TypeVar
import warnings

from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
from meridian.model import model
import numpy as np

ConfigType = TypeVar("ConfigType", bound=configs.BaseConfig)
ResultType = TypeVar("ResultType", bound=results.CheckResult)


class BaseCheck(abc.ABC, Generic[ConfigType, ResultType]):
  """A generic, abstract base class for a single, runnable quality check."""

  def __init__(
      self,
      meridian: model.Meridian,
      analyzer: analyzer_module.Analyzer,
      config: ConfigType,
  ):
    self._meridian = meridian
    self._analyzer = analyzer
    self._config = config

  @abc.abstractmethod
  def run(self) -> ResultType:
    """Executes the check.

    The return type uses the generic ResultType, making it specific for each
    subclass.
    """
    raise NotImplementedError()


# ==============================================================================
# Check: Convergence
# ==============================================================================
class ConvergenceCheck(
    BaseCheck[configs.ConvergenceConfig, results.ConvergenceCheckResult]
):
  """Checks for model convergence."""

  def run(self) -> results.ConvergenceCheckResult:
    rhats = self._analyzer.get_rhat()
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=RuntimeWarning)
      max_rhats = {k: np.nanmax(v) for k, v in rhats.items()}

    valid_rhat_items = [
        item for item in max_rhats.items() if not np.isnan(item[1])
    ]
    if not valid_rhat_items:
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          details={
              review_constants.RHAT: np.nan,
              review_constants.PARAMETER: np.nan,
              review_constants.CONVERGENCE_THRESHOLD: (
                  self._config.convergence_threshold
              ),
          },
      )

    max_parameter, max_rhat = max(max_rhats.items(), key=lambda item: item[1])

    details = {
        review_constants.RHAT: max_rhat,
        review_constants.PARAMETER: max_parameter,
        review_constants.CONVERGENCE_THRESHOLD: (
            self._config.convergence_threshold
        ),
    }

    # Case 1: Converged.
    if max_rhat < self._config.convergence_threshold:
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          details=details,
      )

    # Case 2: Not fully converged, but potentially acceptable.
    elif (
        self._config.convergence_threshold
        <= max_rhat
        < self._config.not_fully_convergence_threshold
    ):
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
          details=details,
      )

    # Case 3: Not converged and unacceptable.
    else:  # max_rhat >= divergence_threshold
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_CONVERGED,
          details=details,
      )


# ==============================================================================
# Check: Baseline
# ==============================================================================
class BaselineCheck(
    BaseCheck[configs.BaselineConfig, results.BaselineCheckResult]
):
  """Checks for negative baseline probability."""

  def run(self) -> results.BaselineCheckResult:
    prob = self._analyzer.negative_baseline_probability()
    details = {
        review_constants.NEGATIVE_BASELINE_PROB: prob,
        review_constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD: (
            self._config.negative_baseline_prob_fail_threshold
        ),
        review_constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD: (
            self._config.negative_baseline_prob_review_threshold
        ),
    }
    # Case 1: FAIL
    if prob > self._config.negative_baseline_prob_fail_threshold:
      return results.BaselineCheckResult(
          case=results.BaselineCases.FAIL,
          details=details,
      )
    # Case 2: REVIEW
    elif prob >= self._config.negative_baseline_prob_review_threshold:
      return results.BaselineCheckResult(
          case=results.BaselineCases.REVIEW,
          details=details,
      )
    # Case 3: PASS
    else:
      return results.BaselineCheckResult(
          case=results.BaselineCases.PASS, details=details
      )


# ==============================================================================
# Check: Bayesian Posterior Predictive P-value
# ==============================================================================
class BayesianPPPCheck(
    BaseCheck[configs.BayesianPPPConfig, results.BayesianPPPCheckResult]
):
  """Checks for Bayesian Posterior Predictive P-value."""

  def run(self) -> results.BayesianPPPCheckResult:
    mmm = self._meridian
    analyzer = self._analyzer

    outcome = mmm.kpi
    if mmm.revenue_per_kpi is not None:
      outcome *= mmm.revenue_per_kpi
    total_outcome_actual = np.sum(outcome)

    total_outcome_posterior = analyzer.expected_outcome(
        aggregate_times=True, aggregate_geos=True
    )
    total_outcome_expected = np.asarray(total_outcome_posterior).flatten()

    total_outcome_expected_mean = np.mean(total_outcome_expected)

    bayesian_ppp = np.mean(
        np.abs(total_outcome_expected - total_outcome_expected_mean)
        >= np.abs(total_outcome_actual - total_outcome_expected_mean)
    )

    details = {
        review_constants.BAYESIAN_PPP: bayesian_ppp,
    }

    if bayesian_ppp >= self._config.ppp_threshold:
      return results.BayesianPPPCheckResult(
          case=results.BayesianPPPCases.PASS,
          details=details,
      )
    else:
      return results.BayesianPPPCheckResult(
          case=results.BayesianPPPCases.FAIL,
          details=details,
      )


# ==============================================================================
# Check: Goodness of Fit
# ==============================================================================
class GoodnessOfFitCheck(
    BaseCheck[configs.GoodnessOfFitConfig, results.GoodnessOfFitCheckResult]
):
  """Checks for goodness of fit of the model."""

  def run(self) -> results.GoodnessOfFitCheckResult:
    gof_ds = self._analyzer.predictive_accuracy()
    gof_df = gof_ds.to_dataframe().reset_index()

    geo_granularity = (
        constants.NATIONAL if self._meridian.n_geos == 1 else constants.GEO
    )

    gof_metrics = gof_df[gof_df[constants.GEO_GRANULARITY] == geo_granularity]
    if constants.EVALUATION_SET_VAR in gof_df.columns:
      gof_metrics = gof_metrics[
          gof_metrics[constants.EVALUATION_SET_VAR] == constants.ALL_DATA
      ]

    gof_metrics_pivoted = gof_metrics.pivot(
        index=constants.GEO_GRANULARITY,
        columns=constants.METRIC,
        values=constants.VALUE,
    )
    gof_metrics_series = gof_metrics_pivoted.loc[geo_granularity]

    r_squared = gof_metrics_series[constants.R_SQUARED]
    mape = gof_metrics_series[constants.MAPE]
    wmape = gof_metrics_series[constants.WMAPE]

    details = {
        review_constants.R_SQUARED: r_squared,
        review_constants.MAPE: mape,
        review_constants.WMAPE: wmape,
    }

    if r_squared > 0:
      return results.GoodnessOfFitCheckResult(
          case=results.GoodnessOfFitCases.PASS,
          details=details,
      )
    else:  # r_squared <= 0
      return results.GoodnessOfFitCheckResult(
          case=results.GoodnessOfFitCases.REVIEW,
          details=details,
      )


# ==============================================================================
# Check: ROI Consistency
# ==============================================================================
def _format_roi_channels_msg(channels: np.ndarray, direction: str) -> str:
  if channels.size == 0:
    return ""
  plural = "s" if channels.size > 1 else ""
  return (
      f"an unusually {direction} ROI estimate (for channel{plural} "
      f"{', '.join(f'`{channel}`' for channel in channels)})"
  )


def _inf_prior_quantiles_channels(
    channels: np.ndarray,
    lo_roi_quantiles: np.ndarray,
    hi_roi_quantiles: np.ndarray,
) -> np.ndarray:
  """Returns channels with infinite prior quantiles.

  Args:
    channels: The names of the channels.
    lo_roi_quantiles: The lower quantiles of the ROI prior.
    hi_roi_quantiles: The upper quantiles of the ROI prior.

  Returns:
    An array of channel names with infinite prior quantiles.
  """
  inf_mask = np.isinf(lo_roi_quantiles) | np.isinf(hi_roi_quantiles)
  return channels[inf_mask]


@dataclasses.dataclass
class _ROIConsistencyChannelData:
  """A data structure for auxiliary data for the ROI Consistency Check.

  Attributes:
    prior_roi_los: Lower quantile values from ROI priors, corresponding to the
      channels in `all_channels`.
    prior_roi_his: Upper quantile values from ROI priors, corresponding to the
      channels in `all_channels`.
    posterior_means: Mean values of ROI posteriors, corresponding to the
      channels in `all_channels`.
    all_channels: Channel names for which quantile computations were successful;
      channels for which quantiles could not be computed are skipped. They are
      ordered with media channels (`roi_m`) followed by reach and frequency (RF)
      channels (`roi_rf`).
    inf_channels: Channels with infinite prior quantiles.
    low_roi_channels: Channels with posterior means below their prior's lower
      quantile.
    high_roi_channels: Channels with posterior means above their prior's upper
      quantile.
    quantile_not_defined_channels: Channel names for which quantiles could not
      be computed.
    quantile_not_defined_parameters: Parameters for which the quantile method is
      not implemented.
  """

  prior_roi_los: np.ndarray
  prior_roi_his: np.ndarray
  posterior_means: np.ndarray
  all_channels: np.ndarray
  inf_channels: np.ndarray
  low_roi_channels: np.ndarray
  high_roi_channels: np.ndarray
  quantile_not_defined_channels: np.ndarray
  quantile_not_defined_parameters: list[backend.tfd.Distribution] = (
      dataclasses.field(default_factory=list)
  )


def _get_roi_consistency_channel_data(
    prior_rois: Sequence[backend.tfd.Distribution],
    posterior_rois: Sequence[backend.tfd.Distribution],
    channels_names: Sequence[Sequence[str]],
    prior_lower_quantile: float,
    prior_upper_quantile: float,
) -> _ROIConsistencyChannelData:
  """Returns the channel-level data for the ROI Consistency Check.

  Args:
    prior_rois: The ROI priors for all channels, in the same order as
      `channels_names`.
    posterior_rois: The ROI posteriors for all channels, in the same order as
      `channels_names`.
    channels_names: The names of all channels, with media channels (`roi_m`)
      followed by any reach and frequency (RF) channels (`roi_rf`).
    prior_lower_quantile: The lower quantile of the ROI prior.
    prior_upper_quantile: The upper quantile of the ROI prior.

  Returns:
    A _ROIConsistencyChannelData object containing the channel-level data for
    the ROI Consistency Check.
  """

  prior_roi_los_parts = []
  prior_roi_his_parts = []
  posterior_means_parts = []
  all_channels_parts = []
  quantile_not_defined_parameters = []
  quantile_not_defined_channels = []

  for prior_roi, posterior_roi, channels in zip(
      prior_rois, posterior_rois, channels_names
  ):
    try:
      prior_roi_lo = prior_roi.quantile(
          prior_lower_quantile,
      )
      prior_roi_hi = prior_roi.quantile(
          prior_upper_quantile,
      )
      posterior_mean = np.mean(posterior_roi, axis=(0, 1))

      n_channels = len(channels)
      prior_roi_lo = np.broadcast_to(prior_roi_lo, shape=(n_channels,))
      prior_roi_hi = np.broadcast_to(prior_roi_hi, shape=(n_channels,))

      prior_roi_los_parts.append(prior_roi_lo)
      prior_roi_his_parts.append(prior_roi_hi)
      posterior_means_parts.append(posterior_mean)
      all_channels_parts.append(channels)
    except NotImplementedError:
      quantile_not_defined_parameters.append(prior_roi)
      quantile_not_defined_channels.extend(channels)

  if prior_roi_los_parts:
    prior_roi_los = np.concatenate(prior_roi_los_parts)
    prior_roi_his = np.concatenate(prior_roi_his_parts)
    posterior_means = np.concatenate(posterior_means_parts)
    all_channels = np.concatenate(all_channels_parts)
  else:
    prior_roi_los = np.array([])
    prior_roi_his = np.array([])
    posterior_means = np.array([])
    all_channels = np.array([])

  inf_channels = _inf_prior_quantiles_channels(
      channels=all_channels,
      lo_roi_quantiles=prior_roi_los,
      hi_roi_quantiles=prior_roi_his,
  )
  low_roi_channels = all_channels[posterior_means < prior_roi_los]
  high_roi_channels = all_channels[posterior_means > prior_roi_his]

  return _ROIConsistencyChannelData(
      prior_roi_los=prior_roi_los,
      prior_roi_his=prior_roi_his,
      posterior_means=posterior_means,
      all_channels=all_channels,
      inf_channels=inf_channels,
      low_roi_channels=low_roi_channels,
      high_roi_channels=high_roi_channels,
      quantile_not_defined_parameters=quantile_not_defined_parameters,
      quantile_not_defined_channels=np.array(quantile_not_defined_channels),
  )


def _compute_channel_results(
    channel_data: _ROIConsistencyChannelData,
) -> list[results.ROIConsistencyChannelResult]:
  """Returns the channel-level results for the ROI Consistency Check."""

  channel_results = []
  for channel in channel_data.quantile_not_defined_channels:
    case = results.ROIConsistencyChannelCases.QUANTILE_NOT_DEFINED
    channel_results.append(
        results.ROIConsistencyChannelResult(
            case=case,
            details={},
            channel_name=channel,
        )
    )
  for i, channel in enumerate(channel_data.all_channels):
    if channel in channel_data.inf_channels:
      case = results.ROIConsistencyChannelCases.PRIOR_ROI_QUANTILE_INF
    elif channel in channel_data.low_roi_channels:
      case = results.ROIConsistencyChannelCases.ROI_LOW
    elif channel in channel_data.high_roi_channels:
      case = results.ROIConsistencyChannelCases.ROI_HIGH
    else:
      case = results.ROIConsistencyChannelCases.ROI_PASS
    channel_results.append(
        results.ROIConsistencyChannelResult(
            case=case,
            details={
                review_constants.PRIOR_ROI_LO: channel_data.prior_roi_los[i],
                review_constants.PRIOR_ROI_HI: channel_data.prior_roi_his[i],
                review_constants.POSTERIOR_ROI_MEAN: (
                    channel_data.posterior_means[i]
                ),
            },
            channel_name=channel,
        )
    )
  return channel_results


def _compute_aggregate_result(
    channel_data: _ROIConsistencyChannelData,
) -> results.ROIConsistencyCheckResult:
  """Returns the aggregate result for the ROI Consistency Check."""
  channel_results = _compute_channel_results(channel_data=channel_data)

  aggregate_details = {}

  # Channel Case 5: QUANTILE_NOT_DEFINED
  if channel_data.quantile_not_defined_parameters:
    aggregate_details[review_constants.QUANTILE_NOT_DEFINED_MSG] = (
        "The quantile method is not defined for the following parameters:"
        f" {channel_data.quantile_not_defined_parameters}. The ROI"
        " Consistency check cannot be performed for these parameters."
    )
  else:
    aggregate_details[review_constants.QUANTILE_NOT_DEFINED_MSG] = ""

  # Channel Case 4: PRIOR_ROI_QUANTILE_INF
  if channel_data.inf_channels.size > 0:
    aggregate_details[review_constants.INF_CHANNELS_MSG] = (
        "Prior ROI quantiles are infinite for channels:"
        f" {', '.join(channel_data.inf_channels)}"
    )
  else:
    aggregate_details[review_constants.INF_CHANNELS_MSG] = ""

  # Channel Cases 2-3: ROI_LOW, ROI_HIGH
  if (
      channel_data.low_roi_channels.size > 0
      or channel_data.high_roi_channels.size > 0
  ):
    low_msg = _format_roi_channels_msg(channel_data.low_roi_channels, "low")
    high_msg = _format_roi_channels_msg(channel_data.high_roi_channels, "high")

    channels_low_high = " and ".join(filter(None, [low_msg, high_msg]))
    aggregate_details[review_constants.LOW_HIGH_CHANNELS_MSG] = (
        f"We've detected {channels_low_high} where the posterior point"
        " estimate falls into the extreme tail of your custom prior."
    )
  else:
    aggregate_details[review_constants.LOW_HIGH_CHANNELS_MSG] = ""

  if (
      aggregate_details[review_constants.QUANTILE_NOT_DEFINED_MSG]
      or aggregate_details[review_constants.INF_CHANNELS_MSG]
      or aggregate_details[review_constants.LOW_HIGH_CHANNELS_MSG]
  ):
    aggregate_case = results.ROIConsistencyAggregateCases.REVIEW
  else:
    aggregate_case = results.ROIConsistencyAggregateCases.PASS

  return results.ROIConsistencyCheckResult(
      case=aggregate_case,
      details=aggregate_details,
      channel_results=channel_results,
  )


class ROIConsistencyCheck(
    BaseCheck[configs.ROIConsistencyConfig, results.ROIConsistencyCheckResult]
):
  """Checks if ROI posterior mean is in tails of ROI prior."""

  def run(self) -> results.ROIConsistencyCheckResult:
    prior_rois = []
    posterior_rois = []
    channel_names = []
    if (
        constants.MEDIA_CHANNEL
        in self._meridian.inference_data.posterior.coords
    ):
      prior_rois.append(self._meridian.model_spec.prior.roi_m)
      posterior_rois.append(self._meridian.inference_data.posterior.roi_m)
      channel_names.append(
          self._meridian.inference_data.posterior.media_channel.values
      )
    if constants.RF_CHANNEL in self._meridian.inference_data.posterior.coords:
      prior_rois.append(self._meridian.model_spec.prior.roi_rf)
      posterior_rois.append(self._meridian.inference_data.posterior.roi_rf)
      channel_names.append(
          self._meridian.inference_data.posterior.rf_channel.values
      )

    channel_data = _get_roi_consistency_channel_data(
        prior_rois=prior_rois,
        posterior_rois=posterior_rois,
        channels_names=channel_names,
        prior_lower_quantile=self._config.prior_lower_quantile,
        prior_upper_quantile=self._config.prior_upper_quantile,
    )

    return _compute_aggregate_result(channel_data=channel_data)


# ==============================================================================
# Check: Prior-Posterior Shift
# ==============================================================================
def _bootstrap(x: np.ndarray, n_bootstraps: int) -> np.ndarray:
  """Performs non-parametric bootstrap resampling on the columns of x."""
  n_rows, n_cols = x.shape
  x_bs = np.empty((n_bootstraps, n_rows, n_cols))
  for i in range(n_bootstraps):
    col_indices = np.random.choice(n_cols, n_cols, replace=True)
    x_bs[i, :, :] = x[:, col_indices]
  return x_bs


def _calculate_new_statistics_from_samples(
    mmm: model.Meridian, n_bootstraps: int, var_name: str, n_channels: int
) -> dict[str, np.ndarray]:
  """Calculate Mean, Median, Q1, and Q3 from posterior samples."""
  n_chains = len(mmm.inference_data.posterior.coords[constants.CHAIN])
  n_draws = len(mmm.inference_data.posterior.coords[constants.DRAW])
  n_posterior_samples = n_chains * n_draws

  posterior_samples = np.transpose(
      np.reshape(
          mmm.inference_data.posterior.variables[var_name].values,
          (n_posterior_samples, n_channels),
      )
  )
  x = _bootstrap(
      posterior_samples, n_bootstraps
  )  # x is (bootstraps, channels, samples)

  mean = np.mean(x, axis=-1)
  median = np.quantile(x, q=0.5, axis=-1)
  q1 = np.quantile(x, q=0.25, axis=-1)
  q3 = np.quantile(x, q=0.75, axis=-1)

  return {
      review_constants.MEAN: mean,
      review_constants.MEDIAN: median,
      review_constants.Q1: q1,
      review_constants.Q3: q3,
  }


def _get_shifted_mask(
    posterior_stat: np.ndarray, prior_stat: np.ndarray, alpha: float
) -> np.ndarray:
  """Returns a boolean mask indicating which channels have a significant shift."""
  prior_stat_b = prior_stat[np.newaxis, ...]
  shift_1 = np.mean(posterior_stat > prior_stat_b, axis=0) < alpha
  shift_2 = np.mean(posterior_stat < prior_stat_b, axis=0) < alpha
  return shift_1 | shift_2


class PriorPosteriorShiftCheck(
    BaseCheck[
        configs.PriorPosteriorShiftConfig,
        results.PriorPosteriorShiftCheckResult,
    ]
):
  """Checks for a significant shift between prior and posterior of ROI."""

  # Tuple of (channel_results, no_shift_channels)
  _CHANNEL_TYPE_RESULT = tuple[
      list[results.PriorPosteriorShiftChannelResult],
      list[str],
  ]

  def _run_for_channel_type(self, channel_type: str) -> _CHANNEL_TYPE_RESULT:
    """Runs the prior-posterior shift check for a given channel type.

    Args:
      channel_type: The channel type ('media_channel' or 'rf_channel') to run
        the check for.

    Returns:
      A tuple of (`channel_results`, `no_shift_channels`).
    """
    if channel_type not in self._meridian.inference_data.posterior.coords:
      return [], []

    channel_results = []
    no_shift_channels = []

    n_channels = len(
        self._meridian.inference_data.posterior[channel_type].values
    )
    if channel_type == constants.MEDIA_CHANNEL:
      var_name = constants.ROI_M
      prior_dist = self._meridian.model_spec.prior.roi_m
    else:
      var_name = constants.ROI_RF
      prior_dist = self._meridian.model_spec.prior.roi_rf
    prior_stats = {}
    try:
      prior_stats[review_constants.MEAN] = prior_dist.mean()
    except NotImplementedError:
      pass
    try:
      prior_stats[review_constants.MEDIAN] = prior_dist.quantile(0.5)
    except NotImplementedError:
      pass
    try:
      prior_stats[review_constants.Q1] = prior_dist.quantile(0.25)
    except NotImplementedError:
      pass
    try:
      prior_stats[review_constants.Q3] = prior_dist.quantile(0.75)
    except NotImplementedError:
      pass

    post_stats = _calculate_new_statistics_from_samples(
        self._meridian, self._config.n_bootstraps, var_name, n_channels
    )

    alpha = self._config.alpha
    any_shift = np.zeros(n_channels, dtype=bool)
    for key in prior_stats:
      prior_stat = prior_stats[key]
      post_stat = post_stats[key]
      current_shift = _get_shifted_mask(post_stat, prior_stat, alpha)
      any_shift = any_shift | current_shift

    channel_names = self._meridian.inference_data.posterior[channel_type].values
    for i, channel_name in enumerate(channel_names):
      shifted = any_shift[i]
      case = (
          results.PriorPosteriorShiftChannelCases.SHIFT
          if shifted
          else results.PriorPosteriorShiftChannelCases.NO_SHIFT
      )
      if not shifted:
        no_shift_channels.append(channel_name)
      channel_results.append(
          results.PriorPosteriorShiftChannelResult(
              case=case, details={}, channel_name=channel_name
          )
      )
    return channel_results, no_shift_channels

  def _aggregate(
      self,
      *channel_type_results: _CHANNEL_TYPE_RESULT,
  ) -> results.PriorPosteriorShiftCheckResult:
    """Aggregates results from multiple channel types."""
    channel_results = []
    no_shift_channels = []
    for results_part, channels_part in channel_type_results:
      channel_results.extend(results_part)
      no_shift_channels.extend(channels_part)

    if no_shift_channels:
      agg_case = results.PriorPosteriorShiftAggregateCases.REVIEW
      final_details = {
          "channels_str": ", ".join(
              f"`{channel}`" for channel in no_shift_channels
          )
      }
    else:
      agg_case = results.PriorPosteriorShiftAggregateCases.PASS
      final_details = {}

    return results.PriorPosteriorShiftCheckResult(
        case=agg_case, details=final_details, channel_results=channel_results
    )

  def run(self) -> results.PriorPosteriorShiftCheckResult:
    np.random.seed(self._config.seed)
    media_results = self._run_for_channel_type(constants.MEDIA_CHANNEL)
    rf_results = self._run_for_channel_type(constants.RF_CHANNEL)
    return self._aggregate(media_results, rf_results)
