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

from collections.abc import MutableMapping, Sequence
import dataclasses
import typing
from typing import Any
import warnings
import arviz as az
import immutabledict
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import plots
from meridian.analysis.review import results
from meridian.model import context
from meridian.model.calibration import base as calibration_base
import numpy as np

CheckType = typing.Type[checks.BaseCheck]
ConfigInstance = configs.BaseConfig
ChecksBattery = typing.Mapping[CheckType, ConfigInstance]

_CALIBRATION_CHECKS = (
    checks.ImplausibleROICheck,
    checks.HighVarianceCheck,
    checks.PotentialBiasCheck,
)


_POST_CONVERGENCE_CHECKS: ChecksBattery = immutabledict.immutabledict({
    checks.BaselineCheck: configs.BaselineConfig(),
    checks.BayesianPPPCheck: configs.BayesianPPPConfig(),
    checks.GoodnessOfFitCheck: configs.GoodnessOfFitConfig(),
    checks.PriorPosteriorShiftCheck: configs.PriorPosteriorShiftConfig(),
    checks.ROIConsistencyCheck: configs.ROIConsistencyConfig(),
    checks.ImplausibleROICheck: configs.ImplausibleROIConfig(),
    checks.HighVarianceCheck: configs.HighVarianceConfig(),
    checks.PotentialBiasCheck: configs.PotentialBiasConfig(),
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
  """Returns the score of the Prior-Posterior Shift check.

  Returns a perfect score (100.0) if `channel_results` is empty (e.g. when no
  media or RF channels are present) to prevent a `ZeroDivisionError`.

  Args:
    prior_posterior_shift_check_result: Result of the Prior-Posterior Shift
      check.
  """
  if not prior_posterior_shift_check_result.channel_results:
    return 100.0
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
  """Returns the score of the ROI Consistency check.

  Returns a perfect score (100.0) if `channel_results` is empty (e.g. when no
  media or RF channels are present) to prevent a `ZeroDivisionError`.

  Args:
    roi_consistency_check_result: Result of the ROI Consistency check.
  """
  if not roi_consistency_check_result.channel_results:
    return 100.0
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
      # TODO: Remove.
      meridian: Any | None = None,
      model_context: context.ModelContext | None = None,
      inference_data: az.InferenceData | None = None,
      convergence_check_config: configs.ConvergenceConfig | None = None,
      post_convergence_checks: ChecksBattery | None = None,
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
          "ModelReviewer requires either (model_context AND inference_data) "
          "or the deprecated (meridian) object."
      )

    self._model_context = model_context
    self._inference_data = inference_data
    self._convergence_check_config = (
        convergence_check_config
        if convergence_check_config is not None
        else configs.ConvergenceConfig()
    )
    self._post_convergence_checks = (
        post_convergence_checks
        if post_convergence_checks is not None
        else _POST_CONVERGENCE_CHECKS
    )
    self._results: MutableMapping[CheckType, results.CheckResult] = {}
    self._analyzer = analyzer_module.Analyzer(
        model_context=self._model_context, inference_data=self._inference_data
    )

  def _run_and_handle(
      self,
      check_class: CheckType,
      config: configs.BaseConfig,
      *,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
  ) -> None:
    """Runs a single check and stores its result.

    Args:
      check_class: The class of the check to run.
      config: The configuration for the check.
      selected_geos: Optional sequence of geos to filter the analysis.
      selected_times: Optional sequence of times to filter the analysis.
    """
    instance: checks.BaseCheck = check_class(
        model_context=self._model_context,
        inference_data=self._inference_data,
        analyzer=self._analyzer,
        config=config,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )  # pyrefly: ignore[abstract-instance]
    self._results[check_class] = instance.run()

  def _is_relevant(self, check_class: CheckType) -> bool:
    """Checks if a check class is relevant for this model."""
    if not check_class.is_relevant(self._model_context, self._inference_data):
      return False
    if check_class in _CALIBRATION_CHECKS:
      if self._should_skip_calibration_checks(check_class):
        return False
    return True

  def _should_skip_calibration_checks(self, check_class: CheckType) -> bool:
    """Checks if calibration checks should be skipped."""
    if (
        self._model_context.n_media_channels == 0
        and self._model_context.n_rf_channels == 0
    ):
      return True
    if self._inference_data is None:
      return True
    if not hasattr(self._inference_data, constants.POSTERIOR):
      return True
    if (
        constants.MEDIA_CHANNEL not in self._inference_data.posterior.coords  # pyrefly: ignore[missing-attribute]
        and constants.RF_CHANNEL not in self._inference_data.posterior.coords  # pyrefly: ignore[missing-attribute]
    ):
      return True

    if (
        check_class in _CALIBRATION_CHECKS
        and self._model_context.input_data.revenue_per_kpi is None
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

  def run(
      self,
      *,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
  ) -> results.ReviewSummary:
    """Executes all checks and generates the final summary."""
    self._results = {}
    self._run_and_handle(
        checks.ConvergenceCheck, self._convergence_check_config
    )

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
    for check_class, config in self._post_convergence_checks.items():
      if not self._is_relevant(check_class):
        continue
      self._run_and_handle(
          check_class,
          config,
          selected_geos=selected_geos,
          selected_times=selected_times,
      )

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

    implausible_roi_result = next(
        (
            r
            for r in self._results.values()
            if isinstance(r, results.ImplausibleROICheckResult)
        ),
        None,
    )
    high_variance_result = next(
        (
            r
            for r in self._results.values()
            if isinstance(r, results.HighVarianceCheckResult)
        ),
        None,
    )
    potential_bias_result = next(
        (
            r
            for r in self._results.values()
            if isinstance(r, results.PotentialBiasCheckResult)
        ),
        None,
    )

    implausible_roi_chart_json = plots.generate_implausible_roi_chart_json(
        implausible_roi_result
    )
    high_variance_chart_json = plots.generate_high_variance_chart_json(
        high_variance_result
    )
    potential_bias_chart_json = plots.generate_potential_bias_chart_json(
        potential_bias_result
    )

    return results.ReviewSummary(
        overall_status=overall_status,
        summary_message=summary_message,
        results=list(self._results.values()),
        health_score=self._compute_health_score(),
        channel_calibration_status=self._get_calibration_status_by_channel(),
        calibrated_channel_names=self._get_calibrated_channels_with_experiments(),
        implausible_roi_chart_json=implausible_roi_chart_json,
        high_variance_chart_json=high_variance_chart_json,
        potential_bias_chart_json=potential_bias_chart_json,
        calibration_overview_data=self._get_calibration_overview_data(),
    )

  def _get_calibration_status_by_channel(self) -> dict[str, bool]:
    """Returns a mapping of channel name to whether it is calibrated."""
    status_map = {}
    input_data = self._model_context.input_data

    def _update_status_map(channel_coord: str, roi_attr: str):
      if (coord_data := getattr(input_data, channel_coord, None)) is not None:
        channels = coord_data.values.tolist()
        prior = getattr(self._model_context.model_spec, constants.PRIOR, None)
        roi_dist = getattr(prior, roi_attr, None) if prior is not None else None
        if isinstance(roi_dist, calibration_base.CalibratedDistribution):
          calib_status = roi_dist.get_calibration_status()
        else:
          calib_status = [False] * len(channels)
        for ch, is_calib in zip(channels, calib_status):
          status_map[ch] = bool(is_calib)

    _update_status_map(constants.MEDIA_CHANNEL, constants.ROI_M)
    _update_status_map(constants.RF_CHANNEL, constants.ROI_RF)

    return status_map

  def _get_calibrated_channels_with_experiments(self) -> list[str]:
    """Returns a list of channel names that have calibration outputs."""
    input_data = self._model_context.input_data
    prior = getattr(self._model_context.model_spec, constants.PRIOR, None)
    if prior is None:
      return []

    channels_with_experiments = []
    for coord, attr in [
        (constants.MEDIA_CHANNEL, constants.ROI_M),
        (constants.RF_CHANNEL, constants.ROI_RF),
    ]:
      if getattr(input_data, coord, None) is None:
        continue

      roi_dist = getattr(prior, attr, None)
      if isinstance(roi_dist, calibration_base.CalibratedDistribution):
        channels_with_experiments.extend(
            out.channel_name for out in roi_dist.calibration_outputs if out
        )

    return channels_with_experiments

  def _get_calibration_overview_data(
      self,
  ) -> list[results.CalibrationOverviewChannelData]:
    """Returns calibration overview data for all calibrated channels with experiments."""
    posterior = getattr(self._inference_data, constants.POSTERIOR, None)
    if posterior is None:
      return []

    input_data = self._model_context.input_data
    prior = self._model_context.model_spec.prior

    overview_data = []
    for coord_name, param_name, coord_data, spend_data in [
        (
            constants.MEDIA_CHANNEL,
            constants.ROI_M,
            input_data.media_channel,
            input_data.media_spend,
        ),
        (
            constants.RF_CHANNEL,
            constants.ROI_RF,
            input_data.rf_channel,
            input_data.rf_spend,
        ),
    ]:
      roi_dist = getattr(prior, param_name, None)
      if (
          coord_data is None
          or spend_data is None
          or not isinstance(roi_dist, calibration_base.CalibratedDistribution)
      ):
        continue

      channel_names = list(coord_data.values)
      total_spend = spend_data.sum(
          dim=[d for d in spend_data.dims if d != coord_name]
      )
      for idx, out in enumerate(roi_dist.calibration_outputs):
        if out is None:
          continue
        coord_val = channel_names[idx]
        ch_name = getattr(out, review_constants.CHANNEL_NAME, None) or coord_val
        ch_data = results.CalibrationOverviewChannelData(
            channel_name=ch_name,
            spend=float(total_spend.sel({coord_name: coord_val}).values),
            calibrated_output=out,
            calibrated_prior_dist=(
                roi_dist.distributions[idx] if roi_dist.distributions else None
            ),
            posterior_samples=posterior[param_name]
            .sel({coord_name: coord_val})
            .values.flatten(),
        )
        ch_data = dataclasses.replace(
            ch_data,
            chart_json=plots.generate_calibration_overview_chart_json(ch_data),
            details_chart_json=plots.generate_calibration_details_chart_json(
                ch_data
            ),
        )
        overview_data.append(ch_data)

    return sorted(overview_data, key=lambda d: d.spend, reverse=True)

