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

"""Meridian Sampling EDA Engine."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import TYPE_CHECKING

import arviz as az
from meridian import backend
from meridian import constants
from meridian.model import prior_distribution
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
import numpy as np
import xarray as xr

if TYPE_CHECKING:
  from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-bad-import-order,g-import-not-at-top


def _calculate_unadjusted_raw(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Unadjusted (Raw) [Mean, SE] for an experiment.

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Unadjusted (Raw) stage.
  """
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.UNADJUSTED_RAW,
      point_estimate=experiment.raw_experiment_result.point_estimate,
      standard_error=experiment.raw_experiment_result.standard_error,
  )


def _calculate_spend_adjusted(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Spend Adjusted [Mean, SE] for an experiment.

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Spend Adjusted stage.

  Raises:
    ValueError: If `1.0 + tau_spend` is negative.
  """
  sigma_raw = experiment.raw_experiment_result.standard_error
  tau_s = experiment.tau_spend
  if tau_s < -1.0:
    raise ValueError(f"`tau_spend` must be >= -1.0, got {tau_s}.")

  se2 = sigma_raw * math.sqrt(1.0 + tau_s)
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_ADJUSTED,
      point_estimate=experiment.raw_experiment_result.point_estimate,
      standard_error=se2,
  )


def _calculate_spend_duration_adjusted(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Spend + Duration Adjusted [Mean, SE] for an experiment.

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Spend + Duration Adjusted stage.

  Raises:
    ValueError: If `1.0 + tau_spend + tau_duration` is negative.
  """
  mu_raw = experiment.raw_experiment_result.point_estimate
  sigma_raw = experiment.raw_experiment_result.standard_error
  gamma_d = experiment.gamma_duration
  tau_s = experiment.tau_spend
  tau_d = experiment.tau_duration

  if 1.0 + tau_s + tau_d < 0.0:
    raise ValueError(
        "`1.0 + tau_spend + tau_duration` must be >= 0, got"
        f" {1.0 + tau_s + tau_d}."
    )

  mean3 = gamma_d * mu_raw
  se3 = sigma_raw * math.sqrt(1.0 + tau_s + tau_d)
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_ADJUSTED,
      point_estimate=mean3,
      standard_error=se3,
  )


def _calculate_spend_duration_recency_adjusted(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Spend + Duration + Recency Adjusted [Mean, SE] for an experiment.

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Spend + Duration + Recency Adjusted stage.

  Raises:
    ValueError: If `1.0 + tau_spend + tau_duration + tau_recency` is negative.
  """
  mu_raw = experiment.raw_experiment_result.point_estimate
  sigma_raw = experiment.raw_experiment_result.standard_error
  gamma_d = experiment.gamma_duration
  tau_s = experiment.tau_spend
  tau_d = experiment.tau_duration
  tau_r = experiment.tau_recency

  if 1.0 + tau_s + tau_d + tau_r < 0.0:
    raise ValueError(
        "`1.0 + tau_spend + tau_duration + tau_recency` must be >= 0, got"
        f" {1.0 + tau_s + tau_d + tau_r}."
    )

  mean4 = gamma_d * mu_raw
  se4 = sigma_raw * math.sqrt(1.0 + tau_s + tau_d + tau_r)
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_RECENCY_ADJUSTED,
      point_estimate=mean4,
      standard_error=se4,
  )


def _calculate_spend_duration_recency_user_adjusted(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Stage 5 (Optional User): Spend + Duration + Recency + User Adjusted [Mean, SE].

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Spend + Duration + Recency + User Adjusted
    stage.

  Raises:
    ValueError: If `1.0 + tau_spend + tau_duration + tau_recency +
    user_standard_error_adjustment` is negative.
  """
  mu_raw = experiment.raw_experiment_result.point_estimate
  sigma_raw = experiment.raw_experiment_result.standard_error
  gamma_d = experiment.gamma_duration
  tau_s = experiment.tau_spend
  tau_d = experiment.tau_duration
  tau_r = experiment.tau_recency
  gamma_user = (
      experiment.user_point_estimate_adjustment
      if experiment.user_point_estimate_adjustment is not None
      else 0.0
  )
  tau_user = (
      experiment.user_standard_error_adjustment
      if experiment.user_standard_error_adjustment is not None
      else 0.0
  )

  sum_tau = 1.0 + tau_s + tau_d + tau_r + tau_user
  if sum_tau < 0.0:
    raise ValueError(
        "`1.0 + tau_spend + tau_duration + tau_recency +"
        f" user_standard_error_adjustment` must be >= 0, got {sum_tau}."
    )

  mean5 = (gamma_d + gamma_user) * mu_raw
  se5 = sigma_raw * math.sqrt(sum_tau)
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_RECENCY_USER_ADJUSTED,
      point_estimate=mean5,
      standard_error=se5,
  )


def _calculate_final_adjusted(
    experiment: calibration_base.CalibratedExperiment,
) -> eda_outcome.ExperimentAdjustmentStageData:
  """Calculates Adjusted (Final) [Mean, SE] for an experiment.

  Args:
    experiment: CalibratedExperiment instance.

  Returns:
    ExperimentAdjustmentStageData for Final Adjusted stage.
  """
  return eda_outcome.ExperimentAdjustmentStageData(
      stage=eda_outcome.CalibrationExperimentAdjustmentStage.FINAL_ADJUSTED,
      point_estimate=experiment.adjusted_experiment_result.point_estimate,
      standard_error=experiment.adjusted_experiment_result.standard_error,
  )


def _calculate_median_scaled_hdi_width(
    dist: backend.tfd.Distribution,
    percentage: float = 0.8,
    num_samples: int = 100_000,
    seed: int | None = 42,
) -> np.ndarray:
  """Calculates the median-scaled highest density interval (HDI) width.

  Defined as: (hdi_upper - hdi_lower) / median.

  Args:
    dist: Distribution instance to evaluate (1D univariate or 2D batched).
    percentage: Highest density interval probability mass in (0.0, 1.0).
    num_samples: Number of sample draws to estimate HDI.
    seed: Random seed for reproducible sampling.

  Returns:
    A 1D numpy array of shape (n_channels,) containing the median-scaled HDI
    widths for each channel.

  Raises:
    ValueError: If percentage is not in (0.0, 1.0).
  """
  if not 0.0 < percentage < 1.0:
    raise ValueError(
        f"Percentage must be strictly between 0.0 and 1.0, got {percentage}."
    )
  rng_handler = backend.RNGHandler(seed=seed)
  raw_samples = np.asarray(
      dist.sample(num_samples, seed=rng_handler.get_next_seed())
  )
  samples = raw_samples.reshape((num_samples, -1))
  medians = np.median(samples, axis=0)
  # Compute HDI per 1D channel slice rather than passing a 2D array to avoid
  # ArviZ FutureWarning and breaking changes in future ArviZ releases (which
  # will change 2D interpretation from (draw, shape) to (chain, draw)).
  hdi = np.asarray([
      az.hdi(samples[:, c], hdi_prob=percentage)
      for c in range(samples.shape[1])
  ])
  safe_medians = np.where(
      (medians != 0) & np.isfinite(medians), np.abs(medians), 1.0
  )
  is_invalid = (medians == 0) | (~np.isfinite(medians))
  return np.where(
      is_invalid,
      0.0,
      (hdi[:, 1] - hdi[:, 0]) / safe_medians,
  )


@functools.lru_cache(maxsize=16)
def _get_default_prior_median_scaled_hdi_width(
    param_name: str,
    percentage: float = 0.8,
) -> float:
  """Calculates default median-scaled HDI width for supported parameters.

  Args:
    param_name: The name of the prior parameter (e.g. 'roi_m', 'roi_rf').
    percentage: Highest density interval probability mass. Default is 0.8.

  Returns:
    The default median-scaled HDI width as a float.

  Raises:
    ValueError: If param_name is not supported or default distribution is None.
  """
  if param_name not in eda_constants.CALIBRATED_PRIOR_PARAMETERS:
    raise ValueError(
        f"Unsupported calibrated prior parameter '{param_name}'. Must be one of"
        f" {eda_constants.CALIBRATED_PRIOR_PARAMETERS}."
    )
  default_priors = prior_distribution.PriorDistribution()
  default_dist = getattr(default_priors, param_name, None)
  if default_dist is None:
    raise ValueError(
        f"Default prior distribution not found for parameter '{param_name}'."
    )
  return float(
      _calculate_median_scaled_hdi_width(
          default_dist, percentage=percentage, seed=42
      )[0]
  )


def _calculate_prior_width_ratio(
    dist: backend.tfd.Distribution,
    param_name: str,
    percentage: float = 0.8,
) -> np.ndarray:
  """Calculates ratio of prior HDI width to default prior HDI width.

  Args:
    dist: Distribution instance representing the channel prior (or batched
      priors).
    param_name: The name of the prior parameter (e.g. 'roi_m', 'roi_rf').
    percentage: Highest density interval probability mass. Default is 0.8.

  Returns:
    A 1D numpy array of shape (n_channels,) containing the prior width
    ratios.

  Raises:
    ValueError: If default width is non-positive.
  """
  default_width = _get_default_prior_median_scaled_hdi_width(
      param_name, percentage=percentage
  )
  if default_width <= 0:
    raise ValueError(
        f"Default prior relative width for '{param_name}' must be positive, got"
        f" {default_width}."
    )
  actual_width = _calculate_median_scaled_hdi_width(dist, percentage=percentage)
  return actual_width / default_width


def _check_analytical_log_concavity(
    prior_dist: backend.tfd.Distribution,
    bounds: tuple[float, float] = eda_constants.LOG_CONCAVITY_EVALUATION_BOUNDS,
    num_points: int = eda_constants.LOG_CONCAVITY_NUM_POINTS,
) -> bool:
  """Evaluates if a prior distribution is log-concave over specified bounds.

  A prior distribution is log-concave if d^2/dx^2 log(P(x)) <= 0. Log-concave
  priors mathematically guarantee unimodality when combined with Gaussian
  experiment likelihoods.

  Args:
    prior_dist: Distribution instance to evaluate.
    bounds: Interval (min, max) over which to check log-concavity.
    num_points: Number of grid evaluation points.

  Returns:
    True if log-concave over the grid, False otherwise.
  """
  x_grid = np.linspace(bounds[0], bounds[1], num_points)
  log_y = np.asarray(prior_dist.log_prob(x_grid))
  dx = x_grid[1] - x_grid[0]
  d2_np = np.gradient(np.gradient(log_y, dx), dx)

  mask = np.isfinite(d2_np)
  if not np.any(mask):
    return False
  return bool(
      np.all(d2_np[mask] <= eda_constants.LOG_CONCAVITY_NUMERICAL_TOLERANCE)
  )


def _combine_normal_means_ses(
    means: Sequence[float], ses: Sequence[float]
) -> tuple[float, float]:
  """Combines multiple independent normal likelihoods using precision weighting.

  Args:
    means: Sequence of experimental point estimates.
    ses: Sequence of experimental standard errors.

  Returns:
    Tuple of (combined_mean, combined_std).
  """
  means_arr = np.asarray(means)
  ses_arr = np.asarray(ses)

  precisions = 1.0 / (ses_arr**2)
  combined_precision = np.sum(precisions)
  combined_std = float(np.sqrt(1.0 / combined_precision))
  combined_mean = float(np.sum(means_arr * precisions) / combined_precision)
  return combined_mean, combined_std


def _is_lognormal_distribution(dist: backend.tfd.Distribution) -> bool:
  """Checks if a distribution is a LogNormal distribution."""
  return (
      isinstance(dist, backend.tfd.LogNormal)
      or dist.__class__.__name__ == "LogNormal"
  )


def _is_lognormal_posterior_bimodal(
    prior_lognormal: backend.tfd.Distribution,
    likelihood_mean: float,
    likelihood_std: float,
) -> bool:
  """Analytically determines if LogNormal prior + Normal likelihood is bimodal.

  Args:
    prior_lognormal: LogNormal distribution instance representing baseline
      prior.
    likelihood_mean: Pooled experimental likelihood mean.
    likelihood_std: Pooled experimental likelihood standard error.

  Returns:
    True if the posterior is bimodal, False otherwise.
  """
  mu_p = float(prior_lognormal.loc)
  sigma_p = float(prior_lognormal.scale)

  sigma_p_sq = sigma_p**2
  sigma_l_sq = likelihood_std**2
  mu_l = likelihood_mean

  # Condition 1: The Variance Threshold (Ensures the curve can turn)
  if mu_l == 0:
    return False
  variance_threshold = (8.0 * sigma_l_sq) / (mu_l**2)
  if sigma_p_sq <= variance_threshold:
    return False

  # Calculate turning points (roots of quadratic)
  discriminant = (mu_l**2) - (8.0 * sigma_l_sq / sigma_p_sq)
  if discriminant < 0:
    return False

  x1 = (mu_l - np.sqrt(discriminant)) / 4.0
  x2 = (mu_l + np.sqrt(discriminant)) / 4.0

  if x1 <= 0 or x2 <= 0:
    return False

  # Define boundary function B(x)
  def b_func(x: float) -> float:
    return float(np.log(x) + sigma_p_sq * (1.0 - (x * (mu_l - x)) / sigma_l_sq))

  # Condition 2: Location Threshold
  lower_bound = b_func(x2)
  upper_bound = b_func(x1)

  return bool(lower_bound < mu_p < upper_bound)


def _compute_lognormal_bimodality_statistic(
    output: calibration_base.CalibrationOutput,
) -> float:
  """Computes analytical bimodality statistic for a LogNormal baseline prior.

  Args:
    output: CalibrationOutput instance containing experiments and baseline
      prior.

  Returns:
    1.0 if LogNormal posterior is analytically bimodal, 0.0 otherwise.
  """
  if not output.experiments or output.baseline_prior is None:
    return 0.0
  if not _is_lognormal_distribution(output.baseline_prior):
    return 0.0
  exp_means = [
      exp.adjusted_experiment_result.point_estimate
      for exp in output.experiments
  ]
  exp_ses = [
      exp.adjusted_experiment_result.standard_error
      for exp in output.experiments
  ]
  combined_mean, combined_std = _combine_normal_means_ses(exp_means, exp_ses)
  is_bimodal = _is_lognormal_posterior_bimodal(
      output.baseline_prior, combined_mean, combined_std
  )
  return 1.0 if is_bimodal else 0.0


def _run_hartigans_dip(
    samples: np.ndarray,
) -> tuple[float | None, float | None]:
  """Runs Hartigan's Dip Test for unimodality (returns 0.0 for MVP)."""
  # TODO: Implement Hartigan's Dip Test when diptest is added.
  del samples
  return 0.0, None


def _compute_channel_bimodality_statistic(
    output: calibration_base.CalibrationOutput,
) -> float | None:
  """Computes bimodality statistic for a calibrated channel output.

  Args:
    output: CalibrationOutput instance containing baseline and intermediary
      priors and adjusted experiments.

  Returns:
    A float representing the bimodality statistic (1.0 if bimodal, 0.0
    otherwise), or None if the baseline prior is neither log-concave nor
    LogNormal (statistic not available for MVP).
  """
  if not output.experiments or output.baseline_prior is None:
    return 0.0

  if _check_analytical_log_concavity(output.baseline_prior):
    return 0.0

  if not _is_lognormal_distribution(output.baseline_prior):
    return None

  log_normal_stat = _compute_lognormal_bimodality_statistic(output)
  dip_stat, _ = _run_hartigans_dip(np.array([]))
  return max(log_normal_stat, float(dip_stat or 0.0))


def _calculate_distribution_overlap(
    output: calibration_base.CalibrationOutput,
    channel_dist: backend.tfd.Distribution,
) -> float:
  """Calculates overlap between intermediary and parameterized distributions.

  Args:
    output: CalibrationOutput instance containing experiments and intermediary
      prior.
    channel_dist: The parameterized distribution.

  Returns:
    The overlapping coefficient (Riemann sum integral of min(f_emp, f_param)),
    or 1.0 if there are no experiments.
  """
  if not output.experiments:
    return 1.0

  grid = output.intermediary_prior.parameters[eda_constants.GRID]
  pdf = output.intermediary_prior.parameters[eda_constants.PDF]
  dx = output.intermediary_prior.parameters[eda_constants.DX]

  param_pdf = np.asarray(channel_dist.prob(grid))
  param_pdf = np.where(np.isfinite(param_pdf), param_pdf, 0.0)
  shared_mass = np.minimum(pdf, param_pdf)
  return float(np.clip(np.sum(shared_mass) * dx, 0.0, 1.0))


def _evaluate_channel_prior_quality(
    output: calibration_base.CalibrationOutput,
    channel_dist: backend.tfd.Distribution,
    prior_width_ratio: float,
) -> eda_outcome.PriorQualityData:
  """Evaluates prior quality data for a single calibrated channel output."""
  neg_exp_count = sum(
      1
      for exp in output.experiments
      if exp.adjusted_experiment_result.point_estimate < 0
  )
  baseline_prior_type = (
      "None"
      if output.baseline_prior is None
      else type(output.baseline_prior).__name__
  )
  bimodal_stat = _compute_channel_bimodality_statistic(output)
  overlap = _calculate_distribution_overlap(output, channel_dist)

  return eda_outcome.PriorQualityData(
      channel_name=output.channel_name,
      prior_width_ratio=prior_width_ratio,
      baseline_prior_type=baseline_prior_type,
      bimodal_statistic=bimodal_stat,
      overlap_percentage=overlap,
      n_negative_experiments=neg_exp_count,
  )


# TODO: Remove this class once EDAEngine can use Analyzer without
# circular dependencies.
class SamplingEDAEngine(eda_engine.EDAEngine):
  """EDA engine for sampling-based checks."""

  def __init__(
      self,
      analyzer: "analyzer_module.Analyzer",
      spec: eda_spec_module.EDASpec | None = None,
  ):
    """Initializes the instance.

    Args:
      analyzer: The Analyzer instance to use for sampling-based checks. It must
        contain prior samples in its inference_data.
      spec: The EDASpec for configuration.

    Raises:
      ValueError: If the analyzer instance does not have 'prior' in its
        inference_data.
    """

    if spec is None:
      spec = eda_spec_module.EDASpec()

    super().__init__(model_context=analyzer.model_context, spec=spec)
    if constants.PRIOR not in analyzer.inference_data.groups():
      raise ValueError(
          f"Analyzer instance must have '{constants.PRIOR}' in its"
          " inference_data."
      )
    self._analyzer = analyzer

  def get_named_calibrated_priors(
      self,
  ) -> dict[str, calibration_base.CalibratedDistribution]:
    """Returns a mapping from prior parameter names to CalibratedDistribution objects.

    Returns:
      A dictionary mapping prior parameter names (e.g. 'roi_m', 'roi_rf') to
      CalibratedDistribution instances.
    """
    prior = self._model_context.model_spec.prior
    named_priors = {}
    for field in dataclasses.fields(prior):
      dist = getattr(prior, field.name, None)
      if isinstance(dist, calibration_base.CalibratedDistribution):
        named_priors[field.name] = dist
    return named_priors

  def get_calibrated_priors(
      self,
  ) -> list[calibration_base.CalibratedDistribution]:
    """Returns a list of CalibratedDistribution objects from the model's priors.

    Returns:
      A list of CalibratedDistribution objects present in model priors.
    """
    return list(self.get_named_calibrated_priors().values())

  def get_calibration_outputs(self) -> list[calibration_base.CalibrationOutput]:
    """Extracts all non-None CalibrationOutput objects from the model's priors.

    Returns:
      A list of non-None CalibrationOutput objects.
    """
    outputs = []
    for dist in self.get_calibrated_priors():
      outputs.extend(dist.calibration_outputs)
    return [out for out in outputs if out is not None]

  def check_experiment_adjustment(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.ExperimentAdjustmentArtifact]:
    """Calculates step-by-step experiment parameter adjustments for all calibrated channels.

    Returns:
      An EDAOutcome object containing an ExperimentAdjustmentArtifact.
    """
    adjustment_data = {}
    for output in self.get_calibration_outputs():
      exp_list = []
      for exp in output.experiments:
        stages = [
            _calculate_unadjusted_raw(exp),
            _calculate_spend_adjusted(exp),
            _calculate_spend_duration_adjusted(exp),
            _calculate_spend_duration_recency_adjusted(exp),
        ]
        if (
            exp.user_point_estimate_adjustment is not None
            or exp.user_standard_error_adjustment is not None
        ):
          stages.append(_calculate_spend_duration_recency_user_adjusted(exp))
        stages.append(_calculate_final_adjusted(exp))
        exp_list.append(
            eda_outcome.ExperimentAdjustmentData(
                source_type=str(exp.source_type),
                stages=stages,
            )
        )
      adjustment_data[output.channel_name] = exp_list

    artifact = eda_outcome.ExperimentAdjustmentArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        adjustment_data=adjustment_data,
    )
    findings = [
        eda_outcome.EDAFinding(
            severity=eda_outcome.EDASeverity.INFO,
            explanation=eda_constants.PRIOR_CALIBRATION_ADJUSTMENT_GRID_INFO,
            finding_cause=eda_outcome.FindingCause.NONE,
            associated_artifact=artifact,
        )
    ]
    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.EXPERIMENT_ADJUSTMENT,
        findings=findings,
        analysis_artifacts=[artifact],
    )

  def check_prior_quality(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PriorQualityArtifact]:
    """Checks prior quality metrics for all calibrated channels.

    Returns:
      An EDAOutcome containing a PriorQualityArtifact.
    """
    quality_data = []
    for param_name, dist in self.get_named_calibrated_priors().items():
      ratios = _calculate_prior_width_ratio(dist, param_name)
      for i, output in enumerate(dist.calibration_outputs):
        if output is None:
          continue
        ratio = float(ratios[i])
        quality_data.append(
            _evaluate_channel_prior_quality(
                output,
                channel_dist=dist.distributions[i],
                prior_width_ratio=ratio,
            )
        )
    any_flagged = any(
        data.n_negative_experiments > 0
        or data.prior_width_ratio > eda_constants.HIGH_VARIANCE_THRESHOLD
        or data.bimodal_statistic is None
        or data.bimodal_statistic > eda_constants.BIMODALITY_STATISTIC_THRESHOLD
        or data.overlap_percentage < eda_constants.OVERLAP_PERCENTAGE_THRESHOLD
        for data in quality_data
    )
    artifact = eda_outcome.PriorQualityArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        prior_quality_data=quality_data,
    )
    severity = (
        eda_outcome.EDASeverity.REVIEW
        if any_flagged
        else eda_outcome.EDASeverity.INFO
    )
    findings = [
        eda_outcome.EDAFinding(
            severity=severity,
            explanation=eda_constants.PRIOR_QUALITY_TABLE_INFO,
            finding_cause=eda_outcome.FindingCause.NONE,
            associated_artifact=artifact,
        )
    ]
    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PRIOR_QUALITY,
        findings=findings,
        analysis_artifacts=[artifact],
    )

  def check_prior_probability(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PriorProbabilityArtifact]:
    """Checks prior probabilities of negative baseline and contributions.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    prior_negative_baseline_prob = self._analyzer.negative_baseline_probability(
        use_posterior=False
    )

    outcome = (
        self._input_data.kpi * self._input_data.revenue_per_kpi
        if self._input_data.revenue_per_kpi is not None
        else self._input_data.kpi
    )
    total_outcome = outcome.sum()

    n_channels = len(self._model_context.input_data.get_all_channels())
    # Shape = (n_chains, n_draws, n_channels)
    incremental_outcome = self._analyzer.incremental_outcome(
        use_posterior=False
    )

    if total_outcome == 0:
      # If total_outcome is zero, division would result in inf.
      # We set mean_prior_contribution to np.inf to indicate this.
      mean_prior_contribution = np.full(n_channels, np.inf)
    else:
      prior_contribution_samples = (
          np.array(incremental_outcome) / total_outcome.values
      )
      # Shape = (n_channels,)
      mean_prior_contribution = np.mean(prior_contribution_samples, axis=(0, 1))

    mean_prior_contribution_da = xr.DataArray(
        mean_prior_contribution,
        coords={
            constants.CHANNEL: self._model_context.input_data.get_all_channels()
        },
        dims=[constants.CHANNEL],
    )

    artifact = eda_outcome.PriorProbabilityArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        prior_negative_baseline_prob=float(prior_negative_baseline_prob),
        mean_prior_contribution_da=mean_prior_contribution_da,
    )

    findings = [
        eda_outcome.EDAFinding(
            severity=eda_outcome.EDASeverity.INFO,
            explanation=eda_constants.PRIOR_PROBABILITY_INFO,
            finding_cause=eda_outcome.FindingCause.NONE,
            associated_artifact=artifact,
        )
    ]

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PRIOR_PROBABILITY,
        findings=findings,
        analysis_artifacts=[artifact],
    )
