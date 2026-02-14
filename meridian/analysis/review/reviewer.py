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

"""Implementation of the runner of the Model Quality Checks."""

from collections.abc import MutableMapping
import dataclasses
import typing

import immutabledict
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
from meridian.model import prior_distribution
import numpy as np

CheckType = typing.Type[checks.BaseCheck]
ConfigInstance = configs.BaseConfig
ChecksBattery = immutabledict.immutabledict[CheckType, ConfigInstance]

_POST_CONVERGENCE_CHECKS: ChecksBattery = immutabledict.immutabledict({
    checks.BaselineCheck: configs.BaselineConfig(),
    checks.BayesianPPPCheck: configs.BayesianPPPConfig(),
    checks.GoodnessOfFitCheck: configs.GoodnessOfFitConfig(),
    checks.PriorPosteriorShiftCheck: configs.PriorPosteriorShiftConfig(),
    checks.ROIConsistencyCheck: configs.ROIConsistencyConfig(),
})


def _get_baseline_score(
    baseline_check_result: results.BaselineCheckResult,
) -> float:
  """Returns the score of the Baseline check."""
  negative_baseline_prob = baseline_check_result.negative_baseline_prob
  baseline_config = baseline_check_result.config
  review_threshold = baseline_config.negative_baseline_prob_review_threshold
  fail_threshold = baseline_config.negative_baseline_prob_fail_threshold

  return 100.0 * (
      1.0
      - np.clip(
          (negative_baseline_prob - review_threshold)
          / (fail_threshold - review_threshold),
          0,
          1,
      )
  )


def _get_bayesian_ppp_score(
    bayesian_ppp_check_result: results.BayesianPPPCheckResult,
) -> float:
  """Returns the score of the Bayesian PPP check."""
  bayesian_ppp = bayesian_ppp_check_result.bayesian_ppp
  bayesian_ppp_config = bayesian_ppp_check_result.config
  ppp_threshold = bayesian_ppp_config.ppp_threshold
  return 100.0 if bayesian_ppp > ppp_threshold else 0.0


def _get_gof_score(
    goodness_of_fit_check_result: results.GoodnessOfFitCheckResult,
) -> float:
  """Returns the score of the Goodness of Fit check."""
  r_squared = goodness_of_fit_check_result.metrics.r_squared
  return 100.0 / (
      1
      + np.exp(
          -review_constants.R2_STEEPNESS
          * (r_squared - review_constants.R2_MIDPOINT)
      )
  )


def _get_pps_score(
    prior_posterior_shift_check_result: results.PriorPosteriorShiftCheckResult,
) -> float:
  """Returns the score of the Prior-Posterior Shift check."""
  prior_posterior_shift_ratio = len(
      prior_posterior_shift_check_result.no_shift_channels
  ) / len(prior_posterior_shift_check_result.channel_results)
  return (
      100.0
      * (1.0 - np.clip(prior_posterior_shift_ratio, 0, 1))
      ** review_constants.FAIL_RATIO_POWER
  )


def _get_roi_consistency_score(
    roi_consistency_check_result: results.ROIConsistencyCheckResult,
) -> float:
  """Returns the score of the ROI Consistency check."""
  roi_consistency_failure_ratio = sum(
      1
      for r in roi_consistency_check_result.channel_results
      if r.case.status != results.Status.PASS
  ) / len(roi_consistency_check_result.channel_results)
  return (
      100.0
      * (1.0 - np.clip(roi_consistency_failure_ratio, 0, 1))
      ** review_constants.FAIL_RATIO_POWER
  )


@dataclasses.dataclass(frozen=True)
class _HealthScoreComponent:
  """A component used in the calculation of the overall health score.

  Attributes:
    check_type: The class of the check this component represents.
    score_function: A callable that takes the check result and returns a float
      score.
    result_type: The expected type of the result object for this check.
    weight: The weight of this component in the overall health score
      calculation.
    is_required: Whether this check is required to be present for the health
      score to be computed.
  """

  check_type: CheckType
  score_function: typing.Callable[[typing.Any], float]
  result_type: typing.Type[results.CheckResult]
  weight: float
  is_required: bool


_HEALTH_SCORE_COMPONENTS = (
    _HealthScoreComponent(
        check_type=checks.BaselineCheck,
        score_function=_get_baseline_score,
        result_type=results.BaselineCheckResult,
        weight=review_constants.HEALTH_SCORE_WEIGHT_BASELINE,
        is_required=True,
    ),
    _HealthScoreComponent(
        check_type=checks.BayesianPPPCheck,
        score_function=_get_bayesian_ppp_score,
        result_type=results.BayesianPPPCheckResult,
        weight=review_constants.HEALTH_SCORE_WEIGHT_BAYESIAN_PPP,
        is_required=True,
    ),
    _HealthScoreComponent(
        check_type=checks.GoodnessOfFitCheck,
        score_function=_get_gof_score,
        result_type=results.GoodnessOfFitCheckResult,
        weight=review_constants.HEALTH_SCORE_WEIGHT_GOF,
        is_required=True,
    ),
    _HealthScoreComponent(
        check_type=checks.PriorPosteriorShiftCheck,
        score_function=_get_pps_score,
        result_type=results.PriorPosteriorShiftCheckResult,
        weight=review_constants.HEALTH_SCORE_WEIGHT_PRIOR_POSTERIOR_SHIFT,
        is_required=False,
    ),
    _HealthScoreComponent(
        check_type=checks.ROIConsistencyCheck,
        score_function=_get_roi_consistency_score,
        result_type=results.ROIConsistencyCheckResult,
        weight=review_constants.HEALTH_SCORE_WEIGHT_ROI_CONSISTENCY,
        is_required=False,
    ),
)


class ModelReviewer:
  """A tool for executing a series of quality checks on a Meridian model.

  The reviewer first runs a convergence check. If the model has converged, it
  proceeds to run a battery of post-convergence checks.

  The battery of post-convergence checks includes:
    - BaselineCheck
    - BayesianPPPCheck
    - GoodnessOfFitCheck
    - PriorPosteriorShiftCheck
    - ROIConsistencyCheck
  """

  def __init__(
      self,
      meridian,
  ):
    self._meridian = meridian
    self._results: MutableMapping[CheckType, results.CheckResult] = {}
    self._analyzer = analyzer_module.Analyzer(
        model_context=meridian.model_context,
        inference_data=meridian.inference_data,
    )

  def _run_and_handle(self, check_class: CheckType, config: configs.BaseConfig):
    instance: checks.BaseCheck = check_class(self._meridian, self._analyzer, config)  # pytype: disable=not-instantiable
    self._results[check_class] = instance.run()

  def _uses_roi_priors(self):
    """Checks if the model uses ROI priors."""
    return (
        self._meridian.n_media_channels > 0
        and self._meridian.model_spec.effective_media_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ) or (
        self._meridian.n_rf_channels > 0
        and self._meridian.model_spec.effective_rf_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    )

  def _has_custom_roi_priors(self):
    """Checks if the model uses custom ROI priors."""
    default_distribution = prior_distribution.PriorDistribution()
    if (
        self._meridian.n_media_channels > 0
        and self._meridian.model_spec.effective_media_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ):
      if not prior_distribution.distributions_are_equal(
          self._meridian.model_spec.prior.roi_m, default_distribution.roi_m
      ):
        return True
    if (
        self._meridian.n_rf_channels > 0
        and self._meridian.model_spec.effective_rf_prior_type
        == constants.TREATMENT_PRIOR_TYPE_ROI
    ):
      if not prior_distribution.distributions_are_equal(
          self._meridian.model_spec.prior.roi_rf, default_distribution.roi_rf
      ):
        return True
    return False

  def _compute_health_score(self) -> float:
    """Computes the health score of the model.

    Raises:
      ValueError: If any required checks are missing from the results.

    Returns:
      The computed health score.
    """
    missing_checks = [
        comp.check_type.__name__
        for comp in _HEALTH_SCORE_COMPONENTS
        if comp.is_required and comp.check_type not in self._results
    ]
    if missing_checks:
      raise ValueError(
          "The following required checks results are missing:"
          f" {missing_checks}."
      )

    scores_and_weights = [
        (
            comp.score_function(
                typing.cast(comp.result_type, self._results[comp.check_type])
            ),
            comp.weight,
        )
        for comp in _HEALTH_SCORE_COMPONENTS
        if comp.check_type in self._results
    ]

    sum_score = sum(score * weight for score, weight in scores_and_weights)
    total_weight = sum(weight for _, weight in scores_and_weights)

    return sum_score / total_weight if total_weight else 0.0

  def run(self) -> results.ReviewSummary:
    """Executes all checks and generates the final summary."""
    self._results = {}
    self._run_and_handle(checks.ConvergenceCheck, configs.ConvergenceConfig())

    # Stop if the model did not converge.
    if (
        self._results
        and self._results[checks.ConvergenceCheck].case
        is results.ConvergenceCases.NOT_CONVERGED
    ):
      return results.ReviewSummary(
          overall_status=results.Status.FAIL,
          summary_message=(
              "Failed: Model did not converge. Other checks were skipped."
          ),
          results=list(self._results.values()),
          health_score=0.0,
      )

    # Run all other checks in sequence.
    for check_class, config in _POST_CONVERGENCE_CHECKS.items():
      if (
          check_class == checks.PriorPosteriorShiftCheck
          and not self._uses_roi_priors()
      ):
        # Skip the Prior-Posterior Shift check if no ROI priors are used.
        continue
      if (
          check_class == checks.ROIConsistencyCheck
          and not self._has_custom_roi_priors()
      ):
        # Skip the ROI Consistency check if no custom ROI priors are provided.
        continue
      self._run_and_handle(check_class, config)

    # Determine the final overall status.
    has_failures = any(
        res.case.status is results.Status.FAIL for res in self._results.values()
    )
    has_reviews = any(
        res.case.status is results.Status.REVIEW
        for res in self._results.values()
    )

    if has_failures and has_reviews:
      overall_status = results.Status.FAIL
      summary_message = (
          "Failed: Quality issues were detected in your model. Follow"
          " recommendations to address any failed checks and review"
          " results to determine if further action is needed."
      )
    elif has_failures:
      overall_status = results.Status.FAIL
      summary_message = (
          "Failed: Quality issues were detected in your model. Address failed"
          " checks before proceeding."
      )
    elif has_reviews:
      overall_status = results.Status.PASS
      summary_message = "Passed with reviews: Review is needed."
    else:
      overall_status = results.Status.PASS
      summary_message = "Passed: No major quality issues were identified."

    return results.ReviewSummary(
        overall_status=overall_status,
        summary_message=summary_message,
        results=list(self._results.values()),
        health_score=self._compute_health_score(),
    )
