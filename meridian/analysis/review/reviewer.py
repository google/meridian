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

"""Implementation of the runner of the Model Quality Checks."""

import typing

import immutabledict
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import results
from meridian.model import prior_distribution


CheckType = typing.Type[checks.BaseCheck]
ConfigInstance = configs.BaseConfig
ChecksBattery = immutabledict.immutabledict[CheckType, ConfigInstance]

_DEFAULT_POST_CONVERGENCE_CHECKS: ChecksBattery = immutabledict.immutabledict({
    checks.BaselineCheck: configs.BaselineConfig(),
    checks.BayesianPPPCheck: configs.BayesianPPPConfig(),
    checks.GoodnessOfFitCheck: configs.GoodnessOfFitConfig(),
    checks.PriorPosteriorShiftCheck: configs.PriorPosteriorShiftConfig(),
    checks.ROIConsistencyCheck: configs.ROIConsistencyConfig(),
})


class ModelReviewer:
  """Executes a series of quality checks on a Meridian model.

  The reviewer first runs a convergence check. If the model has converged, it
  proceeds to run a battery of post-convergence checks.

  The default battery of post-convergence checks includes:
    - BaselineCheck
    - BayesianPPPCheck
    - GoodnessOfFitCheck
    - PriorPosteriorShiftCheck
    - ROIConsistencyCheck
  Each with its default configuration.

  This battery of checks can be customized by passing a dictionary to the
  `post_convergence_checks` argument of the constructor, mapping check
  classes to their configuration instances. For example, to run only the
  BaselineCheck with a non-default configuration:

  ```python
    my_checks = {
        checks.BaselineCheck: configs.BaselineConfig(
            negative_baseline_prob_review_threshold=0.1,
            negative_baseline_prob_fail_threshold=0.5,
        )
    }
    reviewer = ModelReviewer(meridian_model, post_convergence_checks=my_checks)
  ```
  """

  def __init__(
      self,
      meridian,
      post_convergence_checks: ChecksBattery = _DEFAULT_POST_CONVERGENCE_CHECKS,
  ):
    self._meridian = meridian
    self._results: list[results.CheckResult] = []
    self._analyzer = analyzer_module.Analyzer(meridian)
    self._post_convergence_checks = post_convergence_checks

  def _run_and_handle(self, check_class, config):
    instance = check_class(self._meridian, self._analyzer, config)  # pytype: disable=not-instantiable
    self._results.append(instance.run())

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

  def run(self) -> results.ReviewSummary:
    """Executes all checks and generates the final summary."""
    self._results.clear()
    self._run_and_handle(checks.ConvergenceCheck, configs.ConvergenceConfig())

    # Stop if the model did not converge.
    if (
        self._results
        and self._results[0].case is results.ConvergenceCases.NOT_CONVERGED
    ):
      return results.ReviewSummary(
          overall_status=results.Status.FAIL,
          summary_message=(
              "Failed: Model did not converge. Other checks were skipped."
          ),
          results=self._results,
      )

    # Run all other checks in sequence.
    for check_class, config in self._post_convergence_checks.items():
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
        res.case.status is results.Status.FAIL for res in self._results
    )
    has_reviews = any(
        res.case.status is results.Status.REVIEW for res in self._results
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
        results=self._results,
    )
