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

"""Implementation of the Model Quality Checks."""

import abc
from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any, Generic, TypeVar
import warnings

import arviz as az
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
from meridian.common import errors
from meridian.model import context
from meridian.model import prior_distribution
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing_extensions import override
import xarray as xr

__all__ = [
    "BaseCheck",
    "BaseROICheck",
    "BaselineCheck",
    "BayesianPPPCheck",
    "ConvergenceCheck",
    "GoodnessOfFitCheck",
    "HighVarianceCheck",
    "ImplausibleROICheck",
    "PotentialBiasCheck",
    "PriorPosteriorShiftCheck",
    "ROIConsistencyCheck",
]

ConfigType = TypeVar("ConfigType", bound=configs.BaseConfig)
ResultType = TypeVar("ResultType", bound=results.CheckResult)


class BaseCheck(abc.ABC, Generic[ConfigType, ResultType]):
  """A generic, abstract base class for a single, runnable quality check."""

  def __init__(
      self,
      *,
      # TODO: Remove this argument.
      meridian: Any | None = None,
      model_context: context.ModelContext | None = None,
      inference_data: az.InferenceData | None = None,
      analyzer: analyzer_module.Analyzer,
      config: ConfigType,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
  ):
    if meridian is not None:
      warnings.warn(
          "The `meridian` argument is deprecated. "
          "Please use `model_context` and `inference_data` instead.",
          category=DeprecationWarning,
          stacklevel=2,
      )
      model_context = meridian.model_context
      inference_data = meridian.inference_data
    if model_context is None or inference_data is None:
      raise ValueError(
          "BaseCheck requires either (model_context AND inference_data) "
          "or the deprecated (meridian) object."
      )
    self._model_context = model_context
    self._inference_data = inference_data
    self._analyzer = analyzer
    self._config = config
    self._selected_geos = selected_geos
    self._selected_times = selected_times

  @abc.abstractmethod
  def run(self) -> ResultType:
    """Executes the check.

    The return type uses the generic ResultType, making it specific for each
    subclass.
    """
    raise NotImplementedError()

  @classmethod
  def is_relevant(
      cls,
      model_context: context.ModelContext,
      inference_data: az.InferenceData | None = None,
  ) -> bool:
    """Checks if the check is relevant for the given model context."""
    del model_context, inference_data
    return True


class BaseROICheck(BaseCheck[ConfigType, ResultType]):
  """Base class for quality checks that rely on ROI priors and posterior estimates."""

  @classmethod
  def _uses_roi_priors(cls, model_context: context.ModelContext) -> bool:
    """Checks if the model uses ROI priors."""
    return (
        model_context.n_media_channels > 0
        and model_context.model_spec.effective_media_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ) or (
        model_context.n_rf_channels > 0
        and model_context.model_spec.effective_rf_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    )

  @classmethod
  @override
  def is_relevant(
      cls,
      model_context: context.ModelContext,
      inference_data: az.InferenceData | None = None,
  ) -> bool:
    """Checks if the check is relevant for the given model context."""
    return cls._uses_roi_priors(model_context)

  def _roi_channel_types(self) -> list[tuple[str, str]]:
    """Returns the channel types whose ROI posterior is available.

    ROI posteriors (`roi_m`, `roi_rf`) only exist for channel types that use
    ROI priors. E.g. a model with ROI priors for media channels but mROI priors
    for RF channels has `roi_m` but no `roi_rf` in its posterior.

    Returns:
      A list of `(channel_coord, roi_var_name)` tuples, media channels first.
    """
    model_spec = self._model_context.model_spec
    candidates = (
        (
            constants.MEDIA_CHANNEL,
            constants.ROI_M,
            model_spec.effective_media_prior_type,
        ),
        (
            constants.RF_CHANNEL,
            constants.ROI_RF,
            model_spec.effective_rf_prior_type,
        ),
    )
    posterior_coords = self._inference_data.posterior.coords  # pyrefly: ignore[missing-attribute]
    return [
        (channel_coord, roi_var)
        for channel_coord, roi_var, prior_type in candidates
        if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI
        and channel_coord in posterior_coords
    ]

  def _roi_channel_spend_share(self, spend_share: np.ndarray) -> np.ndarray:
    """Selects the spend share of the channels in `_roi_channel_types()`.

    Args:
      spend_share: Spend share of all paid channels, media channels first.

    Returns:
      The spend share entries for channels whose ROI posterior is available,
      in the same order as the channels returned by `_roi_channel_types()`.
    """
    posterior = self._inference_data.posterior  # pyrefly: ignore[missing-attribute]
    roi_channel_coords = {coord for coord, _ in self._roi_channel_types()}
    offset = 0
    parts = []
    for channel_coord in (constants.MEDIA_CHANNEL, constants.RF_CHANNEL):
      if channel_coord not in posterior.coords:
        continue
      n_channels = len(getattr(posterior, channel_coord).values)
      if channel_coord in roi_channel_coords:
        parts.append(spend_share[offset : offset + n_channels])
      offset += n_channels
    return np.concatenate(parts) if parts else spend_share[:0]


# ==============================================================================
# Check: Convergence
# ==============================================================================
class ConvergenceCheck(
    BaseCheck[configs.ConvergenceConfig, results.ConvergenceCheckResult]
):
  """Checks for model convergence."""

  def run(self) -> results.ConvergenceCheckResult:
    """Runs the R-hat convergence check.

    Returns:
      The convergence check result. Its case is `NOT_ASSESSED` when the
      posterior has fewer than 2 chains or fewer than 2 draws per chain,
      because R-hat can't be computed for such a posterior.

    Raises:
      NotFittedModelError: If the model hasn't been fitted.
    """
    if not hasattr(self._inference_data, constants.POSTERIOR):
      raise errors.NotFittedModelError(
          "The model must be fitted before running the convergence check."
      )
    posterior = self._inference_data.posterior  # pyrefly: ignore[missing-attribute]
    n_chains = posterior.sizes[constants.CHAIN]
    n_draws = posterior.sizes[constants.DRAW]
    if not analyzer_module.has_assessable_rhat_dims(posterior):
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_ASSESSED,
          config=self._config,
          max_r_hat=np.nan,
          max_parameter=None,
          n_chains=n_chains,
          n_draws=n_draws,
      )

    rhats = self._analyzer.get_rhat()
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=RuntimeWarning)
      max_r_hats = {k: np.nanmax(v) for k, v in rhats.items()}

    valid_rhat_items = [
        item for item in max_r_hats.items() if not np.isnan(item[1])
    ]
    if not valid_rhat_items:
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          config=self._config,
          max_r_hat=np.nan,
          max_parameter=np.nan,  # pyrefly: ignore[bad-argument-type]
          n_chains=n_chains,
          n_draws=n_draws,
      )

    max_parameter, max_r_hat = max(max_r_hats.items(), key=lambda item: item[1])

    # Case 1: Converged.
    if max_r_hat < self._config.convergence_threshold:
      case = results.ConvergenceCases.CONVERGED

    # Case 2: Not fully converged, but potentially acceptable.
    elif (
        self._config.convergence_threshold
        <= max_r_hat
        < self._config.not_fully_converged_threshold
    ):
      case = results.ConvergenceCases.NOT_FULLY_CONVERGED

    # Case 3: Not converged and unacceptable.
    else:  # max_r_hat >= divergence_threshold
      case = results.ConvergenceCases.NOT_CONVERGED

    return results.ConvergenceCheckResult(
        case=case,
        config=self._config,
        max_r_hat=max_r_hat,
        max_parameter=max_parameter,
        n_chains=n_chains,
        n_draws=n_draws,
    )


# ==============================================================================
# Check: Baseline
# ==============================================================================
class BaselineCheck(
    BaseCheck[configs.BaselineConfig, results.BaselineCheckResult]
):
  """Checks for negative baseline probability."""

  def run(self) -> results.BaselineCheckResult:
    prob = float(
        self._analyzer.negative_baseline_probability(
            selected_geos=self._selected_geos,
            selected_times=self._selected_times,
        )
    )

    # Case 1: FAIL
    if prob > self._config.negative_baseline_prob_fail_threshold:
      case = results.BaselineCases.FAIL

    # Case 2: REVIEW
    elif prob >= self._config.negative_baseline_prob_review_threshold:
      case = results.BaselineCases.REVIEW

    # Case 3: PASS
    else:
      case = results.BaselineCases.PASS

    return results.BaselineCheckResult(
        case=case,
        config=self._config,
        negative_baseline_prob=prob,
    )


# ==============================================================================
# Check: Bayesian Posterior Predictive P-value
# ==============================================================================
def _two_sided_normal_mixture_p_value(
    observed: float, means: np.ndarray, stddevs: np.ndarray
) -> float:
  """Computes a two-sided p-value under an equal-weight mixture of normals.

  Each mixture component corresponds to one posterior draw `s` and is normal
  with mean `means[s]` and standard deviation `stddevs[s]`. With
  `center = mean(means)` and `distance = |observed - center|`, the p-value is
  the probability that a draw from the mixture lands at least `distance` away
  from `center`:

    mean_s(Phi((center - distance - means[s]) / stddevs[s])
           + Phi((means[s] - center - distance) / stddevs[s])),

  where `Phi` is the standard normal CDF. Computing this probability exactly,
  rather than estimating it from random draws, makes the result deterministic.
  A component with a zero standard deviation is a point mass, and contributes 1
  if `|means[s] - center| >= distance` and 0 otherwise. A NaN standard deviation
  makes the result NaN.

  Args:
    observed: The observed value of the test statistic.
    means: The mean of each mixture component.
    stddevs: The non-negative standard deviation of each mixture component, with
      the same shape as `means`.

  Returns:
    The two-sided p-value.

  Raises:
    ValueError: If `means` and `stddevs` have different shapes.
  """
  if means.shape != stddevs.shape:
    raise ValueError(
        "`means` and `stddevs` must have the same shape, got"
        f" {means.shape} and {stddevs.shape}."
    )
  center = np.mean(means)
  distance = np.abs(observed - center)
  with np.errstate(divide="ignore", invalid="ignore"):
    tail_probability = scipy_stats.norm.cdf(
        (center - distance - means) / stddevs
    ) + scipy_stats.norm.cdf((means - center - distance) / stddevs)
  point_mass_tail_probability = np.abs(means - center) >= distance
  # Select on `stddevs == 0` rather than `stddevs > 0` so that a NaN standard
  # deviation propagates to the result instead of being read as a point mass.
  tail_probability = np.where(
      stddevs == 0, point_mass_tail_probability, tail_probability
  )
  return float(np.mean(tail_probability))


class BayesianPPPCheck(
    BaseCheck[configs.BayesianPPPConfig, results.BayesianPPPCheckResult]
):
  """Checks the Bayesian posterior predictive p-value of the total outcome.

  The test statistic is the outcome summed over the selected geos and times.
  For each posterior draw, the model's predictive distribution of this total is
  normal, with mean equal to the expected total outcome and variance equal to
  the summed residual variance of the selected geo and time cells. The check
  computes the two-sided posterior predictive p-value in closed form: the
  average, over posterior draws, of the probability that a predictive total
  lands at least as far from the posterior predictive mean as the observed
  total. The check fails if the p-value is below `ppp_threshold`.
  """

  def _calculate_total_sigma(self) -> np.ndarray:
    """Calculates the standard deviation of the aggregate total outcome noise.

    In Meridian, each cell (g, t) has independent residual error on the
    normalized scale:

    y_{g, t}^scaled ~ N(mu_{g, t}^scaled, sigma_g^2)

    Converting normalized KPI to raw KPI/revenue scales each cell by:
    weight_{g, t} = stdev * pop_g * revenue_per_kpi_{g, t}

    where stdev = kpi_transformer.population_scaled_stdev and
    pop_g = population_g. Note: revenue_per_kpi is set to 1.0 in the weight
    calculation if it's passed as None.

    Since the sum of independent normal random variables is itself normal:
    sum_{g, t} y_{g, t} ~ N(sum_{g, t} mu_{g, t}, sum_{g, t} Var(y_{g, t}))

    the aggregate variance is computed analytically:
    total_var_s = sum_{g, t} (sigma_{s, g} * weight_{g, t})^2

    Working with the aggregate sum, distributed as N(total_outcome_expected,
    total_var), is equivalent to working with each individual cell and summing
    them, and is more efficient.

    Returns:
      A 1D numpy array representing the total residual standard deviation across
      all posterior draws.
    """
    stdev = float(self._model_context.kpi_transformer.population_scaled_stdev)
    kpi_shape = np.shape(self._model_context.input_data.kpi)
    pop = np.asarray(self._model_context.population)
    if len(kpi_shape) > pop.ndim:
      pop = pop.reshape(pop.shape + (1,) * (len(kpi_shape) - pop.ndim))

    revenue_per_kpi = self._model_context.input_data.revenue_per_kpi
    if revenue_per_kpi is not None:
      weight_sq = (pop * np.asarray(revenue_per_kpi)) ** 2
    else:
      weight_sq = np.broadcast_to(pop, kpi_shape) ** 2

    weight_sq_filtered = np.asarray(
        self._analyzer.filter_and_aggregate_geos_and_times(
            backend.to_tensor(weight_sq),
            selected_geos=self._selected_geos,
            selected_times=self._selected_times,
            aggregate_geos=False,
            aggregate_times=True,
        )
    )

    sigma = np.asarray(
        self._inference_data.posterior[constants.SIGMA]  # pyrefly: ignore[missing-attribute]
    )
    if sigma.ndim == 2:
      return sigma.flatten() * (
          stdev * np.sqrt(np.sum(weight_sq_filtered))
      )
    if self._selected_geos is not None:
      sigma_selected = np.asarray(
          self._inference_data.posterior[constants.SIGMA].sel(  # pyrefly: ignore[missing-attribute]
              geo=list(self._selected_geos)
          )
      )
    else:
      sigma_selected = sigma
    return np.sqrt(
        np.sum(sigma_selected**2 * weight_sq_filtered, axis=-1) * (stdev**2)
    ).flatten()

  def run(self) -> results.BayesianPPPCheckResult:
    analyzer = self._analyzer

    revenue_per_kpi = self._model_context.input_data.revenue_per_kpi
    outcome = backend.to_tensor(
        self._model_context.input_data.kpi * revenue_per_kpi
        if revenue_per_kpi is not None
        else self._model_context.input_data.kpi
    )
    total_actual_outcome_filtered = (
        self._analyzer.filter_and_aggregate_geos_and_times(
            outcome,
            selected_geos=self._selected_geos,
            selected_times=self._selected_times,
            aggregate_geos=False,
            aggregate_times=False,
        )
    )
    total_outcome_actual = float(
        np.sum(np.asarray(total_actual_outcome_filtered, dtype=np.float64))
    )

    total_outcome_posterior = analyzer.expected_outcome(
        aggregate_times=True,
        aggregate_geos=True,
        selected_geos=self._selected_geos,
        selected_times=self._selected_times,
    )
    total_outcome_expected = np.asarray(
        total_outcome_posterior, dtype=np.float64
    ).flatten()
    total_sigma = np.asarray(self._calculate_total_sigma(), dtype=np.float64)
    bayesian_ppp = _two_sided_normal_mixture_p_value(
        observed=total_outcome_actual,
        means=total_outcome_expected,
        stddevs=total_sigma,
    )

    if bayesian_ppp >= self._config.ppp_threshold:
      case = results.BayesianPPPCases.PASS
    else:
      case = results.BayesianPPPCases.FAIL

    return results.BayesianPPPCheckResult(
        case=case,
        config=self._config,
        bayesian_ppp=bayesian_ppp,
    )


# ==============================================================================
# Check: Goodness of Fit
# ==============================================================================
def _set_metrics_from_gof_dataframe(
    metrics: MutableMapping[str, float],
    gof_df: pd.DataFrame,
    geo_granularity: str,
    suffix: str,
) -> None:
  """Sets the `metrics` variable of the GoodnessOfFitCheckResult.

  This method takes a DataFrame containing goodness of fit metrics and pivots it
  to a Series, which is then added to the `metrics` variable of the
  `GoodnessOfFitCheckResult`.

  Args:
    metrics: A dictionary to store the goodness of fit metrics in.
    gof_df: A DataFrame containing predictive accuracy of the whole data (if
      holdout set is not used) of filtered to a single evaluation set ("all",
      "train", or "test").
    geo_granularity: The geo granularity of the data ("geo" or "national").
    suffix: A suffix to add to the metric names (e.g., "_train", "_test").
  """
  gof_metrics_pivoted = gof_df.pivot(
      index=constants.GEO_GRANULARITY,
      columns=constants.METRIC,
      values=constants.VALUE,
  )
  gof_metrics_series = gof_metrics_pivoted.loc[geo_granularity]
  metrics[f"{review_constants.R_SQUARED}{suffix}"] = gof_metrics_series[
      constants.R_SQUARED
  ]
  metrics[f"{review_constants.MAPE}{suffix}"] = gof_metrics_series[
      constants.MAPE
  ]
  metrics[f"{review_constants.WMAPE}{suffix}"] = gof_metrics_series[
      constants.WMAPE
  ]


class GoodnessOfFitCheck(
    BaseCheck[configs.GoodnessOfFitConfig, results.GoodnessOfFitCheckResult]
):
  """Checks for goodness of fit of the model."""

  def run(self) -> results.GoodnessOfFitCheckResult:
    gof_ds = self._analyzer.predictive_accuracy(
        selected_geos=self._selected_geos,
        selected_times=self._selected_times,
    )
    gof_df = gof_ds.to_dataframe().reset_index()

    gof_metrics = gof_df[
        gof_df[constants.GEO_GRANULARITY] == constants.NATIONAL
    ]
    is_holdout = constants.EVALUATION_SET_VAR in gof_df.columns

    metrics_dict = {}
    case = results.GoodnessOfFitCases.PASS

    if is_holdout:
      for evaluation_set, suffix in [
          (constants.ALL_DATA, review_constants.ALL_SUFFIX),
          (constants.TRAIN, review_constants.TRAIN_SUFFIX),
          (constants.TEST, review_constants.TEST_SUFFIX),
      ]:
        set_metrics = gof_metrics[
            gof_metrics[constants.EVALUATION_SET_VAR] == evaluation_set
        ]
        _set_metrics_from_gof_dataframe(
            metrics=metrics_dict,
            gof_df=set_metrics,
            geo_granularity=constants.NATIONAL,
            suffix=suffix,
        )
        if (
            metrics_dict[f"{review_constants.R_SQUARED}{suffix}"]
            <= self._config.r_squared_threshold
        ):
          case = results.GoodnessOfFitCases.REVIEW
      return results.GoodnessOfFitCheckResult(
          case=case,
          metrics=results.GoodnessOfFitMetrics(
              r_squared=metrics_dict[
                  f"{review_constants.R_SQUARED}{review_constants.ALL_SUFFIX}"
              ],
              mape=metrics_dict[
                  f"{review_constants.MAPE}{review_constants.ALL_SUFFIX}"
              ],
              wmape=metrics_dict[
                  f"{review_constants.WMAPE}{review_constants.ALL_SUFFIX}"
              ],
              r_squared_train=metrics_dict[
                  f"{review_constants.R_SQUARED}{review_constants.TRAIN_SUFFIX}"
              ],
              mape_train=metrics_dict[
                  f"{review_constants.MAPE}{review_constants.TRAIN_SUFFIX}"
              ],
              wmape_train=metrics_dict[
                  f"{review_constants.WMAPE}{review_constants.TRAIN_SUFFIX}"
              ],
              r_squared_test=metrics_dict[
                  f"{review_constants.R_SQUARED}{review_constants.TEST_SUFFIX}"
              ],
              mape_test=metrics_dict[
                  f"{review_constants.MAPE}{review_constants.TEST_SUFFIX}"
              ],
              wmape_test=metrics_dict[
                  f"{review_constants.WMAPE}{review_constants.TEST_SUFFIX}"
              ],
          ),
          is_holdout=is_holdout,
      )
    else:
      _set_metrics_from_gof_dataframe(
          metrics=metrics_dict,
          gof_df=gof_metrics,
          geo_granularity=constants.NATIONAL,
          suffix=review_constants.ALL_SUFFIX,
      )
      if (
          metrics_dict[review_constants.R_SQUARED]
          <= self._config.r_squared_threshold
      ):
        case = results.GoodnessOfFitCases.REVIEW
      return results.GoodnessOfFitCheckResult(
          case=case,
          metrics=results.GoodnessOfFitMetrics(
              r_squared=metrics_dict[
                  f"{review_constants.R_SQUARED}{review_constants.ALL_SUFFIX}"
              ],
              mape=metrics_dict[
                  f"{review_constants.MAPE}{review_constants.ALL_SUFFIX}"
              ],
              wmape=metrics_dict[
                  f"{review_constants.WMAPE}{review_constants.ALL_SUFFIX}"
              ],
          ),
          is_holdout=is_holdout,
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
            channel_name=channel,
            prior_roi_lo=np.nan,
            prior_roi_hi=np.nan,
            posterior_roi_mean=np.nan,
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
            channel_name=channel,
            prior_roi_lo=channel_data.prior_roi_los[i],
            prior_roi_hi=channel_data.prior_roi_his[i],
            posterior_roi_mean=channel_data.posterior_means[i],
        )
    )
  return channel_results


def _compute_aggregate_result(
    channel_data: _ROIConsistencyChannelData,
    config: configs.ROIConsistencyConfig,
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
  ):
    aggregate_case = results.ROIConsistencyAggregateCases.REVIEW
  elif aggregate_details[review_constants.LOW_HIGH_CHANNELS_MSG]:
    n_failing = (
        channel_data.low_roi_channels.size + channel_data.high_roi_channels.size
    )
    if not config.is_failing_channels_within_threshold(
        n_failing=n_failing, n_total=channel_data.all_channels.size
    ):
      aggregate_case = results.ROIConsistencyAggregateCases.REVIEW
    else:
      aggregate_case = results.ROIConsistencyAggregateCases.PASS
  else:
    aggregate_case = results.ROIConsistencyAggregateCases.PASS

  return results.ROIConsistencyCheckResult(
      case=aggregate_case,
      aggregate_details=aggregate_details,
      channel_results=channel_results,
  )


class ROIConsistencyCheck(
    BaseROICheck[
        configs.ROIConsistencyConfig, results.ROIConsistencyCheckResult
    ]
):
  """Checks if ROI posterior mean is in tails of ROI prior."""

  @classmethod
  def _has_custom_roi_priors(cls, model_context: context.ModelContext) -> bool:
    """Checks if the model uses custom ROI priors."""
    default_distribution = prior_distribution.PriorDistribution()
    if (
        model_context.n_media_channels > 0
        and model_context.model_spec.effective_media_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ):
      if not prior_distribution.distributions_are_equal(
          model_context.model_spec.prior.roi_m, default_distribution.roi_m
      ):
        return True
    if (
        model_context.n_rf_channels > 0
        and model_context.model_spec.effective_rf_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ):
      if not prior_distribution.distributions_are_equal(
          model_context.model_spec.prior.roi_rf,
          default_distribution.roi_rf,
      ):
        return True
    return False

  @classmethod
  @override
  def is_relevant(
      cls,
      model_context: context.ModelContext,
      inference_data: az.InferenceData | None = None,
  ) -> bool:
    """Checks if the check is relevant for the given model context."""
    if not super().is_relevant(model_context, inference_data):
      return False
    return cls._has_custom_roi_priors(model_context)

  def run(self) -> results.ROIConsistencyCheckResult:
    prior_rois = []
    posterior_rois = []
    channel_names = []
    posterior = self._inference_data.posterior  # pyrefly: ignore[missing-attribute]
    for channel_coord, roi_var in self._roi_channel_types():
      prior_rois.append(getattr(self._model_context.model_spec.prior, roi_var))
      posterior_rois.append(getattr(posterior, roi_var))
      channel_names.append(getattr(posterior, channel_coord).values)

    channel_data = _get_roi_consistency_channel_data(
        prior_rois=prior_rois,
        posterior_rois=posterior_rois,
        channels_names=channel_names,
        prior_lower_quantile=self._config.prior_lower_quantile,
        prior_upper_quantile=self._config.prior_upper_quantile,
    )

    return _compute_aggregate_result(
        channel_data=channel_data, config=self._config
    )


# ==============================================================================
# Check: Prior-Posterior Shift
# ==============================================================================
def _bootstrap(
    x: np.ndarray,
    n_bootstraps: int,
    n_samples_per_bootstrap: int | None = None,
) -> np.ndarray:
  """Performs non-parametric bootstrap resampling on the columns of x.

  Args:
    x: A 2D array of shape `(n_channels, n_posterior_samples)` containing the
      posterior samples per channel.
    n_bootstraps: Number of bootstrap samples to use for calculating posterior
      statistics.
    n_samples_per_bootstrap: Number of posterior samples to draw with
      replacement for each bootstrap iteration. If None, uses all available
      posterior samples.

  Returns:
    A 3D array of shape `(n_bootstraps, n_channels, n_samples_per_bootstrap)`
    with the resampled data.
  """
  n_rows, n_cols = x.shape
  if n_samples_per_bootstrap is None:
    n_samples_per_bootstrap = n_cols
  x_bs = np.empty((n_bootstraps, n_rows, n_samples_per_bootstrap))
  for i in range(n_bootstraps):
    col_indices = np.random.choice(
        n_cols, n_samples_per_bootstrap, replace=True
    )
    x_bs[i, :, :] = x[:, col_indices]
  return x_bs


def _calculate_new_statistics_from_samples(
    inference_data: az.InferenceData,
    n_bootstraps: int,
    var_name: str,
    n_channels: int,
    n_samples_per_bootstrap: int | None = None,
) -> dict[str, np.ndarray]:
  """Calculate Mean, Median, Q1, and Q3 from posterior samples.

  Args:
    inference_data: The InferenceData object containing posterior draws.
    n_bootstraps: Number of bootstrap samples to use for calculating posterior
      statistics.
    var_name: The variable name in posterior samples to analyze.
    n_channels: Number of channels for the given variable.
    n_samples_per_bootstrap: Number of posterior samples to draw with
      replacement for each bootstrap iteration. If None, uses all available
      posterior samples.

  Returns:
    A dictionary mapping summary statistic names to arrays of shape
    `(n_bootstraps, n_channels)`.
  """
  n_chains = len(inference_data.posterior.coords[constants.CHAIN])  # pyrefly: ignore[missing-attribute]
  n_draws = len(inference_data.posterior.coords[constants.DRAW])  # pyrefly: ignore[missing-attribute]
  n_posterior_samples = n_chains * n_draws

  posterior_samples = np.transpose(
      np.reshape(
          inference_data.posterior.variables[var_name].values,  # pyrefly: ignore[missing-attribute]
          (n_posterior_samples, n_channels),
      )
  )
  x = _bootstrap(
      posterior_samples, n_bootstraps, n_samples_per_bootstrap
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
    BaseROICheck[
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
    if channel_type not in dict(self._roi_channel_types()):
      return [], []

    channel_results = []
    no_shift_channels = []

    n_channels = len(self._inference_data.posterior[channel_type].values)  # pyrefly: ignore[missing-attribute]
    if channel_type == constants.MEDIA_CHANNEL:
      var_name = constants.ROI_M
      prior_dist = self._model_context.model_spec.prior.roi_m
    else:
      var_name = constants.ROI_RF
      prior_dist = self._model_context.model_spec.prior.roi_rf
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
        self._inference_data,
        self._config.n_bootstraps,
        var_name,
        n_channels,
        self._config.n_samples_per_bootstrap,
    )

    alpha = self._config.alpha
    any_shift = np.zeros(n_channels, dtype=bool)
    for key in prior_stats:
      prior_stat = prior_stats[key]
      post_stat = post_stats[key]
      current_shift = _get_shifted_mask(post_stat, prior_stat, alpha)
      any_shift = any_shift | current_shift

    channel_names = self._inference_data.posterior[channel_type].values  # pyrefly: ignore[missing-attribute]
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
              case=case, channel_name=channel_name
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

    if not self._config.is_failing_channels_within_threshold(
        n_failing=len(no_shift_channels), n_total=len(channel_results)
    ):
      agg_case = results.PriorPosteriorShiftAggregateCases.REVIEW
    else:
      agg_case = results.PriorPosteriorShiftAggregateCases.PASS

    return results.PriorPosteriorShiftCheckResult(
        case=agg_case,
        channel_results=channel_results,
        no_shift_channels=no_shift_channels,
    )

  def run(self) -> results.PriorPosteriorShiftCheckResult:
    np.random.seed(self._config.seed)
    media_results = self._run_for_channel_type(constants.MEDIA_CHANNEL)
    rf_results = self._run_for_channel_type(constants.RF_CHANNEL)
    return self._aggregate(media_results, rf_results)


# ==============================================================================
# Check: Implausible ROI
# ==============================================================================
def _calculate_spend_share(model_context: context.ModelContext) -> np.ndarray:
  """Calculates the spend share for all paid channels.

  Args:
    model_context: The ModelContext of the Meridian model.

  Returns:
    A 1D NumPy array of shape `(n_channels,)` containing the spend share for
    each paid channel, or all zeros if the total spend sum is zero.
  """
  initial_spend = model_context.input_data.get_total_spend()
  # TODO: Verify if we really support 1D spend
  spend = (
      np.sum(initial_spend, axis=(0, 1))
      if initial_spend.ndim == 3
      else initial_spend
  )

  total_spend_sum = np.sum(spend)
  if total_spend_sum > 0:
    return spend / total_spend_sum
  else:
    return np.zeros_like(spend)


class ImplausibleROICheck(
    BaseROICheck[
        configs.ImplausibleROIConfig, results.ImplausibleROICheckResult
    ]
):
  """A check for paid channels with implausible posterior ROI estimates."""

  @override
  def run(self) -> results.ImplausibleROICheckResult:
    # 1. Get spend and calculate spend share
    spend_share = _calculate_spend_share(self._model_context)

    # 2. Get posterior ROI and channels
    posterior_rois = []
    channels = []
    posterior = self._inference_data.posterior  # pyrefly: ignore[missing-attribute]
    for channel_coord, roi_var in self._roi_channel_types():
      posterior_rois.append(getattr(posterior, roi_var).values)
      channels.extend(getattr(posterior, channel_coord).values.tolist())

    if not posterior_rois:
      raise ValueError("No posterior ROI data found in inference_data.")

    posterior_roi_concat = np.concatenate(posterior_rois, axis=-1)
    spend_share = self._roi_channel_spend_share(spend_share)
    roi_means = np.mean(posterior_roi_concat, axis=(0, 1))

    # 3. Evaluate checks
    channel_results = []
    high_roi_channels = []
    low_roi_channels = []

    spend_weighted_roi_all = roi_means * spend_share
    reciprocal_spend_weighted_roi_all = np.divide(
        roi_means,
        spend_share,
        out=np.full_like(roi_means, np.nan),
        where=(spend_share != 0),
    )

    for i, channel in enumerate(channels):
      mean = roi_means[i]
      share = spend_share[i]

      spend_weighted_roi = spend_weighted_roi_all[i]
      reciprocal_spend_weighted_roi = reciprocal_spend_weighted_roi_all[i]

      if spend_weighted_roi > self._config.roi_upper_bound:
        case = results.ImplausibleROIChannelCases.ROI_HIGH
        high_roi_channels.append(channel)
      elif reciprocal_spend_weighted_roi < self._config.roi_lower_bound:
        case = results.ImplausibleROIChannelCases.ROI_LOW
        low_roi_channels.append(channel)
      else:
        case = results.ImplausibleROIChannelCases.ROI_PASS

      channel_results.append(
          results.ImplausibleROIChannelResult(
              case=case,
              channel_name=channel,
              spend_share=share,
              roi_mean=mean,
              spend_weighted_roi=spend_weighted_roi,
          )
      )

    if high_roi_channels or low_roi_channels:
      msg_parts = []
      if high_roi_channels:
        msg_parts.append(
            "high ROI estimates (for channel(s) "
            f"{', '.join(f'`{c}`' for c in high_roi_channels)})"
        )
      if low_roi_channels:
        msg_parts.append(
            "low ROI estimates (for channel(s) "
            f"{', '.join(f'`{c}`' for c in low_roi_channels)})"
        )
      implausible_roi_msg = (
          "We've detected implausibly " + " and ".join(msg_parts) + "."
      )
      aggregate_details = {"implausible_roi_msg": implausible_roi_msg}
    else:
      aggregate_details = {"implausible_roi_msg": ""}

    if high_roi_channels or low_roi_channels:
      agg_case = results.ImplausibleROIAggregateCases.REVIEW
    else:
      agg_case = results.ImplausibleROIAggregateCases.PASS

    return results.ImplausibleROICheckResult(
        case=agg_case,
        channel_results=channel_results,
        high_roi_channels=high_roi_channels,
        low_roi_channels=low_roi_channels,
        aggregate_details=aggregate_details,
        roi_upper_bound=self._config.roi_upper_bound,
        roi_lower_bound=self._config.roi_lower_bound,
    )


# ==============================================================================
# Check: High Variance ROI
# ==============================================================================
class HighVarianceCheck(
    BaseROICheck[configs.HighVarianceConfig, results.HighVarianceCheckResult]
):
  """A check for paid channels with high variance in posterior ROI."""

  @override
  def run(self) -> results.HighVarianceCheckResult:
    # 1. Get spend and calculate spend share
    spend_share = _calculate_spend_share(self._model_context)

    # 2. Get posterior ROI and channels
    posterior_rois = []
    channels = []

    posterior = self._inference_data.posterior  # pyrefly: ignore[missing-attribute]
    for channel_coord, roi_var in self._roi_channel_types():
      posterior_rois.append(getattr(posterior, roi_var).values)
      channels.extend(getattr(posterior, channel_coord).values.tolist())

    if not posterior_rois:
      raise ValueError("No posterior ROI data found in inference_data.")

    posterior_roi_concat = np.concatenate(posterior_rois, axis=-1)
    spend_share = self._roi_channel_spend_share(spend_share)
    roi_medians = np.median(posterior_roi_concat, axis=(0, 1))

    # 3. Compute credible intervals using az.hdi
    hdi = az.hdi(posterior_roi_concat, hdi_prob=self._config.hdi_prob)
    hdi_lower, hdi_upper = hdi.T

    rel_width_post = np.divide(
        hdi_upper - hdi_lower,
        np.abs(roi_medians),
        out=np.zeros_like(roi_medians, dtype=float),
        where=(roi_medians != 0),
    )

    # 4. Compute high variance check
    relative_width_ratio = np.divide(
        rel_width_post,
        self._config.prior_relative_hdi_width,
        out=np.zeros_like(rel_width_post, dtype=float),
        where=(self._config.prior_relative_hdi_width != 0),
    )
    spend_weighted_ratio = relative_width_ratio * spend_share

    channel_results = []
    high_variance_channels = []

    for channel, share, ratio, weighted_ratio in zip(
        channels, spend_share, relative_width_ratio, spend_weighted_ratio
    ):
      if weighted_ratio > self._config.high_variance_threshold:
        case = results.HighVarianceChannelCases.HIGH_VARIANCE
        high_variance_channels.append(channel)
      else:
        case = results.HighVarianceChannelCases.ROI_PASS

      channel_results.append(
          results.HighVarianceChannelResult(
              case=case,
              channel_name=channel,
              spend_share=share,
              relative_width_ratio=ratio,
          )
      )

    if high_variance_channels:
      agg_case = results.HighVarianceAggregateCases.REVIEW
    else:
      agg_case = results.HighVarianceAggregateCases.PASS

    return results.HighVarianceCheckResult(
        case=agg_case,
        channel_results=channel_results,
        high_variance_channels=high_variance_channels,
        prior_relative_hdi_width=self._config.prior_relative_hdi_width,
        high_variance_threshold=self._config.high_variance_threshold,
    )


# ==============================================================================
# Check: Potential Bias
# ==============================================================================
class PotentialBiasCheck(
    BaseCheck[configs.PotentialBiasConfig, results.PotentialBiasCheckResult]
):
  """A check for correlation between paid channels and control variables to flag potential confounding."""

  @override
  def run(self) -> results.PotentialBiasCheckResult:
    """Runs the potential bias check.

    This check computes the Pearson correlation between each paid channel's
    media/RF data and all control variables. It flags channels where the
    maximum absolute correlation with any control variable exceeds a
    predefined threshold.

    Returns:
      A results.PotentialBiasCheckResult object containing the results of the
      check.
    """
    channels = self._model_context.input_data.get_all_paid_channels().tolist()

    controls = self._model_context.input_data.controls
    if controls is None or self._model_context.n_controls == 0:
      correlation_matrix = xr.DataArray(
          np.zeros((self._model_context.n_geos, len(channels), 0)),
          coords={
              constants.GEO: self._model_context.input_data.geo.values,
              constants.CHANNEL: channels,
              constants.CONTROL_VARIABLE: [],
          },
          dims=[constants.GEO, constants.CHANNEL, constants.CONTROL_VARIABLE],
      )
      return results.PotentialBiasCheckResult(
          case=results.PotentialBiasAggregateCases.NO_CONTROLS,
          channel_results=[],
          low_correlation_channels=[],
          correlation_matrix=correlation_matrix,
          correlation_threshold=self._config.correlation_threshold,
      )

    media_data = self._model_context.input_data.get_all_media_and_rf()
    n_times = self._model_context.n_times
    media_aligned = media_data[:, -n_times:, :]

    controls_data = controls.values

    # media_aligned: (n_geos, n_times, n_channels)
    # controls_data: (n_geos, n_times, n_controls)
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=RuntimeWarning)
      correlation = scipy_stats.pearsonr(
          media_aligned[..., np.newaxis],
          controls_data[:, :, np.newaxis, :],
          axis=1,
      ).statistic
      abs_correlation = np.abs(correlation)
      max_abs_correlations = np.nanmax(abs_correlation, axis=(0, 2))

    channel_results = []
    low_correlation_channels = []

    for channel, max_corr in zip(channels, max_abs_correlations):
      if max_corr < self._config.correlation_threshold:
        case = results.PotentialBiasChannelCases.LOW_CORRELATION
        low_correlation_channels.append(channel)
      else:
        case = results.PotentialBiasChannelCases.ROI_PASS

      channel_results.append(
          results.PotentialBiasChannelResult(
              case=case,
              channel_name=channel,
              max_abs_correlation=max_corr,
          )
      )

    correlation_matrix = xr.DataArray(
        correlation,
        coords={
            constants.GEO: self._model_context.input_data.geo.values,
            constants.CHANNEL: channels,
            constants.CONTROL_VARIABLE: (
                controls.coords[constants.CONTROL_VARIABLE].values
            ),
        },
        dims=[constants.GEO, constants.CHANNEL, constants.CONTROL_VARIABLE],
    )

    if low_correlation_channels:
      agg_case = results.PotentialBiasAggregateCases.REVIEW
    else:
      agg_case = results.PotentialBiasAggregateCases.PASS

    return results.PotentialBiasCheckResult(
        case=agg_case,
        channel_results=channel_results,
        low_correlation_channels=low_correlation_channels,
        correlation_matrix=correlation_matrix,
        correlation_threshold=self._config.correlation_threshold,
    )
