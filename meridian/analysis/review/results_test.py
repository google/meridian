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

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from meridian import backend
from meridian import constants
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants

from meridian.analysis.review import plots

from meridian.analysis.review import results

from meridian.model.calibration import base as calibration_base

import numpy as np
import xarray as xr


def setUpModule():
  flags.FLAGS.mark_as_parsed()


class ConvergenceCheckResultTest(parameterized.TestCase):

  def test_convergence_check_result_converged(self):
    config = configs.ConvergenceConfig(convergence_threshold=2.0)
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        config=config,
        max_r_hat=1.0,
        max_parameter="mock_var",
    )
    self.assertEqual(result.max_r_hat, 1.0)
    self.assertEqual(result.max_rhat, 1.0)
    self.assertEqual(result.case.status, results.Status.PASS)
    self.assertEqual(
        result.recommendation,
        "The model has likely converged, as all parameters have R-hat values"
        " < 2.0.",
    )

  def test_convergence_check_result_needs_review(self):
    config = configs.ConvergenceConfig(convergence_threshold=2.0)
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
        config=config,
        max_r_hat=3.0,
        max_parameter="mock_var",
    )
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.recommendation,
        "The model hasn't fully converged, and the `max_r_hat` for parameter"
        " `mock_var` is 3.00. "
        f"{results.NOT_FULLY_CONVERGED_RECOMMENDATION}",
    )

  def test_convergence_check_result_not_converged(self):
    config = configs.ConvergenceConfig(convergence_threshold=2.0)
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_CONVERGED,
        config=config,
        max_r_hat=11.0,
        max_parameter="mock_var",
    )
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.recommendation,
        "The model hasn't converged, and the `max_r_hat` for parameter"
        " `mock_var` is 11.00. "
        f"{results.NOT_CONVERGED_RECOMMENDATION}",
    )


class BaselineCheckResultTest(parameterized.TestCase):

  def test_baseline_check_result_pass(self):
    config = configs.BaselineConfig(
        negative_baseline_prob_fail_threshold=0.2,
        negative_baseline_prob_review_threshold=0.1,
    )
    result = results.BaselineCheckResult(
        case=results.BaselineCases.PASS,
        config=config,
        negative_baseline_prob=0.01,
    )
    self.assertEqual(
        result.recommendation,
        "The posterior probability that the baseline is negative is 0.01. "
        f"{results._BASELINE_PASS_RECOMMENDATION}",
    )

  def test_baseline_check_result_review(self):
    config = configs.BaselineConfig(
        negative_baseline_prob_fail_threshold=0.2,
        negative_baseline_prob_review_threshold=0.1,
    )
    result = results.BaselineCheckResult(
        case=results.BaselineCases.REVIEW,
        config=config,
        negative_baseline_prob=0.15,
    )
    self.assertEqual(
        result.recommendation,
        "The posterior probability that the baseline is negative is 0.15. "
        f"{results._BASELINE_REVIEW_RECOMMENDATION}",
    )

  def test_baseline_check_result_fail(self):
    config = configs.BaselineConfig(
        negative_baseline_prob_fail_threshold=0.2,
        negative_baseline_prob_review_threshold=0.1,
    )
    result = results.BaselineCheckResult(
        case=results.BaselineCases.FAIL,
        config=config,
        negative_baseline_prob=0.25,
    )
    self.assertEqual(
        result.recommendation,
        "The posterior probability that the baseline is negative is 0.25. "
        f"{results._BASELINE_FAIL_RECOMMENDATION}",
    )


class ROIConsistencyResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="all_pass",
          case=results.ROIConsistencyAggregateCases.PASS,
          details={},
          expected_recommendation=(
              "The posterior distribution of the ROI is within a reasonable"
              " range, aligning with the custom priors you provided."
          ),
      ),
      dict(
          testcase_name="has_reviews",
          case=results.ROIConsistencyAggregateCases.REVIEW,
          details={
              review_constants.QUANTILE_NOT_DEFINED_MSG: "msg1",
              review_constants.INF_CHANNELS_MSG: "msg2",
              review_constants.LOW_HIGH_CHANNELS_MSG: "msg3",
          },
          expected_recommendation=(
              f"msg1msg2msg3 {results._ROI_CONSISTENCY_RECOMMENDATION}"
          ),
      ),
  )
  def test_roi_consistency_result_recommendation(
      self,
      case: results.ROIConsistencyAggregateCases,
      details: dict[str, Any],
      expected_recommendation: str | None,
  ):
    result = results.ROIConsistencyCheckResult(
        case=case,
        aggregate_details=details,
        channel_results=[],
    )
    self.assertEqual(result.recommendation, expected_recommendation)


class BayesianPPPCheckResultTest(parameterized.TestCase):

  def test_bayesian_ppp_check_result_pass(self):
    config = configs.BayesianPPPConfig()
    result = results.BayesianPPPCheckResult(
        case=results.BayesianPPPCases.PASS,
        config=config,
        bayesian_ppp=0.06,
    )
    self.assertEqual(
        result.recommendation,
        "The Bayesian posterior predictive p-value is 0.06. "
        f"{results._BAYESIAN_PPP_PASS_RECOMMENDATION}",
    )

  def test_bayesian_ppp_check_result_fail(self):
    config = configs.BayesianPPPConfig()
    result = results.BayesianPPPCheckResult(
        case=results.BayesianPPPCases.FAIL,
        config=config,
        bayesian_ppp=0.04,
    )
    self.assertEqual(
        result.recommendation,
        "The Bayesian posterior predictive p-value is 0.04. "
        f"{results._BAYESIAN_PPP_FAIL_RECOMMENDATION}",
    )


class GoodnessOfFitCheckResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no_r_squared_train",
          metrics=results.GoodnessOfFitMetrics(
              r_squared=0.1,
              mape=0.1,
              wmape=0.1,
              mape_train=0.1,
              wmape_train=0.1,
              r_squared_test=0.1,
              mape_test=0.1,
              wmape_test=0.1,
          ),
          details_str=(
              "r_squared=0.1, mape=0.1, wmape=0.1, r_squared_train=None,"
              " mape_train=0.1, wmape_train=0.1, r_squared_test=0.1,"
              " mape_test=0.1, wmape_test=0.1"
          ),
      ),
      dict(
          testcase_name="no_mape_test",
          metrics=results.GoodnessOfFitMetrics(
              r_squared=0.1,
              mape=0.1,
              wmape=0.1,
              r_squared_train=0.1,
              mape_train=0.1,
              wmape_train=0.1,
              r_squared_test=0.1,
              wmape_test=0.1,
          ),
          details_str=(
              "r_squared=0.1, mape=0.1, wmape=0.1, r_squared_train=0.1,"
              " mape_train=0.1, wmape_train=0.1, r_squared_test=0.1,"
              " mape_test=None, wmape_test=0.1"
          ),
      ),
  )
  def test_goodness_of_fit_check_result_raises_error(
      self,
      metrics: results.GoodnessOfFitMetrics,
      details_str: str,
  ):
    expected_error_message = (
        "The message template is missing required formatting arguments for"
        " holdout case. Required keys: r_squared_train, mape_train,"
        " wmape_train, r_squared_test, mape_test, wmape_test. Metrics:"
        f" GoodnessOfFitMetrics({details_str})."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.GoodnessOfFitCheckResult(
          case=results.GoodnessOfFitCases.PASS,
          metrics=metrics,
          is_holdout=True,
      )

  def test_goodness_of_fit_check_result_pass(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.PASS,
        metrics=results.GoodnessOfFitMetrics(
            r_squared=0.5,
            mape=0.1,
            wmape=0.2,
        ),
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = 0.5000, MAPE = 0.1000, and wMAPE = 0.2000. "
        f"{results._GOODNESS_OF_FIT_PASS_RECOMMENDATION}",
    )

  def test_goodness_of_fit_check_result_pass_holdout(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.PASS,
        metrics=results.GoodnessOfFitMetrics(
            r_squared=0.5,
            mape=0.1,
            wmape=0.2,
            r_squared_train=0.6,
            mape_train=0.09,
            wmape_train=0.19,
            r_squared_test=0.4,
            mape_test=0.11,
            wmape_test=0.21,
        ),
        is_holdout=True,
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = 0.5000 (All), 0.6000 (Train), 0.4000 (Test); MAPE ="
        " 0.1000 (All), 0.0900 (Train), 0.1100 (Test); wMAPE = 0.2000 (All),"
        " 0.1900 (Train), 0.2100 (Test)."
        f" {results._GOODNESS_OF_FIT_PASS_RECOMMENDATION}",
    )

  def test_goodness_of_fit_check_result_review(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.REVIEW,
        metrics=results.GoodnessOfFitMetrics(
            r_squared=-0.5,
            mape=0.1,
            wmape=0.2,
        ),
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = -0.5000, MAPE = 0.1000, and wMAPE = 0.2000. "
        f"{results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION}",
    )

  def test_goodness_of_fit_check_result_review_holdout(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.REVIEW,
        metrics=results.GoodnessOfFitMetrics(
            r_squared=-0.5,
            mape=0.1,
            wmape=0.2,
            r_squared_train=0.6,
            mape_train=0.09,
            wmape_train=0.19,
            r_squared_test=0.4,
            mape_test=0.11,
            wmape_test=0.21,
        ),
        is_holdout=True,
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = -0.5000 (All), 0.6000 (Train), 0.4000 (Test); MAPE ="
        " 0.1000 (All), 0.0900 (Train), 0.1100 (Test); wMAPE = 0.2000 (All),"
        " 0.1900 (Train), 0.2100 (Test)."
        f" {results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION}",
    )


class PriorPosteriorShiftCheckResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          case=results.PriorPosteriorShiftAggregateCases.PASS,
          no_shift_channels=[],
          expected_recommendation=(
              "The model has successfully learned from the data. This is a"
              " positive sign that your data was informative."
          ),
      ),
      dict(
          testcase_name="review",
          case=results.PriorPosteriorShiftAggregateCases.REVIEW,
          no_shift_channels=["channel1", "channel2"],
          expected_recommendation=(
              "We've detected channel(s) `channel1`, `channel2` where the"
              " posterior distribution did not significantly shift from the"
              " prior. This suggests the data signal for these channels was"
              " not strong enough to update the model's beliefs."
              f" {results._PPS_REVIEW_RECOMMENDATION}"
          ),
      ),
  )
  def test_prior_posterior_shift_result_recommendation(
      self,
      case: results.PriorPosteriorShiftAggregateCases,
      no_shift_channels: list[str],
      expected_recommendation: str | None,
  ):
    result = results.PriorPosteriorShiftCheckResult(
        case=case,
        no_shift_channels=no_shift_channels,
        channel_results=[],
    )
    self.assertEqual(result.recommendation, expected_recommendation)


def _mock_check_results(
    high_roi_channels: Sequence[str] = (),
    low_roi_channels: Sequence[str] = (),
    high_variance_channels: Sequence[str] = (),
    low_correlation_channels: Sequence[str] = (),
    include_passing_checks: bool = False,
    high_roi_mean: float = 30.0,
    high_roi_spend_share: float = 0.5,
    relative_width_ratio: float = 2.0,
    max_abs_correlation: float = 0.05,
) -> list[results.CheckResult]:
  res = []
  if high_roi_channels or low_roi_channels:
    channel_results = [
        results.ImplausibleROIChannelResult(
            case=results.ImplausibleROIChannelCases.ROI_HIGH,
            channel_name=ch,
            spend_share=high_roi_spend_share,
            roi_mean=high_roi_mean,
            spend_weighted_roi=high_roi_spend_share * high_roi_mean,
        )
        for ch in high_roi_channels
    ] + [
        results.ImplausibleROIChannelResult(
            case=results.ImplausibleROIChannelCases.ROI_LOW,
            channel_name=ch,
            spend_share=0.5,
            roi_mean=0.1,
            spend_weighted_roi=0.05,
        )
        for ch in low_roi_channels
    ]
    res.append(
        results.ImplausibleROICheckResult(
            case=results.ImplausibleROIAggregateCases.REVIEW,
            channel_results=channel_results,
            high_roi_channels=list(high_roi_channels),
            low_roi_channels=list(low_roi_channels),
            aggregate_details={},
        )
    )
  elif include_passing_checks:
    res.append(
        results.ImplausibleROICheckResult(
            case=results.ImplausibleROIAggregateCases.PASS,
            channel_results=[],
            high_roi_channels=[],
            low_roi_channels=[],
            aggregate_details={},
        )
    )
  if high_variance_channels:
    channel_results = [
        results.HighVarianceChannelResult(
            channel_name=ch,
            case=results.HighVarianceChannelCases.HIGH_VARIANCE,
            spend_share=0.5,
            relative_width_ratio=relative_width_ratio,
        )
        for ch in high_variance_channels
    ]
    res.append(
        results.HighVarianceCheckResult(
            case=results.HighVarianceAggregateCases.REVIEW,
            channel_results=channel_results,
            high_variance_channels=list(high_variance_channels),
        )
    )
  elif include_passing_checks:
    res.append(
        results.HighVarianceCheckResult(
            case=results.HighVarianceAggregateCases.PASS,
            channel_results=[],
            high_variance_channels=[],
        )
    )
  if low_correlation_channels:
    channel_results = [
        results.PotentialBiasChannelResult(
            channel_name=ch,
            case=results.PotentialBiasChannelCases.LOW_CORRELATION,
            max_abs_correlation=max_abs_correlation,
        )
        for ch in low_correlation_channels
    ]
    res.append(
        results.PotentialBiasCheckResult(
            case=results.PotentialBiasAggregateCases.REVIEW,
            channel_results=channel_results,
            low_correlation_channels=list(low_correlation_channels),
            correlation_matrix=xr.DataArray(),
        )
    )
  elif include_passing_checks:
    res.append(
        results.PotentialBiasCheckResult(
            case=results.PotentialBiasAggregateCases.PASS,
            channel_results=[],
            low_correlation_channels=[],
            correlation_matrix=xr.DataArray(),
        )
    )
  return res


def _create_test_summary(
    overall_status: results.Status = results.Status.PASS,
    summary_message: str = "Passed",
    results_list: Sequence[results.CheckResult] | None = None,
    health_score: float = 90.0,
    channel_calibration_status: Mapping[str, bool] | None = None,
    channel_scores: Mapping[str, float] | None = None,
    calibrated_channel_names: Sequence[str] | None = None,
    include_passing_checks: bool = False,
    **kwargs: Any,
) -> results.ReviewSummary:
  if results_list is None:
    results_list = (
        _mock_check_results(include_passing_checks=True)
        if include_passing_checks
        else ()
    )
  if channel_calibration_status is None and channel_scores is not None:
    channel_calibration_status = {
        ch: ch in (calibrated_channel_names or ()) for ch in channel_scores
    }
  status_dict = dict(channel_calibration_status or {})
  if calibrated_channel_names is None and status_dict:
    calibrated_channel_names = [
        ch for ch, is_cal in status_dict.items() if is_cal
    ]
  summary = results.ReviewSummary(
      overall_status=overall_status,
      summary_message=summary_message,
      results=list(results_list),
      health_score=health_score,
      channel_calibration_status=status_dict,
      calibrated_channel_names=list(calibrated_channel_names or ()),
      **kwargs,
  )
  if channel_scores is not None:
    summary.channel_calibration_scores = dict(channel_scores)
  return summary


def _mock_channel_data(
    channel_name: str = "ch",
    spend: float = 100.0,
    num_experiments: int = 0,
    chart_json: str | None = '{"test": "json"}',
    details_chart_json: str | None = None,
    with_prior_dist: bool = False,
) -> results.CalibrationOverviewChannelData:
  prior_dist = None
  if with_prior_dist:
    prior_dist = backend.tfd.Normal(
        backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
    )
  experiments = [
      calibration_base.CalibratedExperiment(
          source_type=calibration_base.SourceType.MERIDIAN_GEOX,
          raw_experiment_result=calibration_base.ExperimentResult(
              point_estimate=1.0, standard_error=0.2
          ),
          adjusted_experiment_result=calibration_base.ExperimentResult(
              point_estimate=1.0, standard_error=0.2
          ),
          tau_spend=0.0,
          tau_recency=0.0,
          tau_duration=0.0,
          gamma_duration=1.0,
      )
      for _ in range(num_experiments)
  ]
  if prior_dist is not None:
    intermediary_prior = prior_dist
  else:
    intermediary_prior = backend.tfd.Normal(
        backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
    )
  mock_output = calibration_base.CalibrationOutput(
      channel_name=channel_name,
      experiments=experiments,
      intermediary_prior=intermediary_prior,
  )
  return results.CalibrationOverviewChannelData(
      channel_name=channel_name,
      spend=spend,
      calibrated_output=mock_output,
      calibrated_prior_dist=prior_dist,
      chart_json=chart_json,
      details_chart_json=details_chart_json,
  )


class ReviewSummaryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  def test_review_summary_repr(self):
    mock_result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        config=configs.ConvergenceConfig(),
        max_r_hat=1.0,
        max_parameter="mock_var",
    )
    summary = results.ReviewSummary(
        overall_status=results.Status.PASS,
        summary_message="summary",
        results=[mock_result],
        health_score=95.2,
    )
    expected_repr = """========================================
Model Quality Checks
========================================
Overall Status: PASS
Summary: summary
Health Score: 95.2

Check Results:
----------------------------------------
Convergence Check:
  Status: PASS
  Recommendation: The model has likely converged, as all parameters have R-hat values < 1.2."""
    self.assertMultiLineEqual(str(summary), expected_repr)

  @parameterized.named_parameters(
      dict(
          testcase_name="pass_no_banner",
          overall_status=results.Status.PASS,
          summary_message="Passed: No major quality issues were identified.",
          expected_html_snippet="""<div class="metrics-section">""",
      ),
      dict(
          testcase_name="pass_reviews_info_banner",
          overall_status=results.Status.PASS,
          summary_message="Passed with reviews: Review is needed.",
          expected_html_snippet=(
              '<div class="status-banner-strip info">\n'
              '     <span class="material-icons-outlined">check_circle</span>\n'
              "     <span>Passed with reviews: Review is needed.</span>\n"
              "  </div>"
          ),
      ),
      dict(
          testcase_name="fail_banner",
          overall_status=results.Status.FAIL,
          summary_message="Failed: Quality issues were detected in your model.",
          expected_html_snippet=(
              '<div class="status-banner-strip fail">\n     <span'
              ' class="material-icons-outlined">error_outline</span>\n    '
              " <span>Failed: Quality issues were detected in your"
              " model.</span>\n  </div>"
          ),
      ),
  )
  def test_health_card_html_banner(
      self,
      overall_status: results.Status,
      summary_message: str,
      expected_html_snippet: str,
  ):
    summary = results.ReviewSummary(
        overall_status=overall_status,
        summary_message=summary_message,
        results=[],
        health_score=80.0,
    )

    html_output = summary._create_health_card_html()
    self.assertIn(expected_html_snippet, html_output)

  def test_health_card_html_content(self):
    mock_result_model = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.PASS,
        metrics=results.GoodnessOfFitMetrics(
            r_squared=0.5,
            mape=0.1,
            wmape=0.2,
        ),
    )
    mock_result_channel = results.PriorPosteriorShiftCheckResult(
        case=results.PriorPosteriorShiftAggregateCases.REVIEW,
        channel_results=[
            results.PriorPosteriorShiftChannelResult(
                case=results.PriorPosteriorShiftChannelCases.SHIFT,
                channel_name="mock_channel1",
            ),
            results.PriorPosteriorShiftChannelResult(
                case=results.PriorPosteriorShiftChannelCases.NO_SHIFT,
                channel_name="mock_channel2",
            ),
        ],
        no_shift_channels=["mock_channel2"],
    )

    summary = results.ReviewSummary(
        overall_status=results.Status.REVIEW,
        summary_message="Review is needed.",
        results=[mock_result_model, mock_result_channel],
        health_score=85.2,
    )

    html_output = summary._create_health_card_html()

    # 1. Validate health score number
    self.assertIn('<div class="score-value">85.2</div>', html_output)

    # 2. Validate health score graph
    self.assertIn(
        '<div class="health-score-chart" style="--score: 85.2">', html_output
    )

    # 3. Validate metrics check table
    # Model-level check (Goodness of Fit)
    self.assertIn("<td>Goodness of fit</td>", html_output)
    self.assertIn('<chip class="pass">Pass</chip>', html_output)
    self.assertIn(
        "R-squared = 0.5000, MAPE = 0.1000, and wMAPE = 0.2000.", html_output
    )

    # Channel-level check (Prior-Posterior Shift)
    self.assertIn("<td>Prior-posterior shift</td>", html_output)
    self.assertIn('<chip class="review">Review</chip>', html_output)
    self.assertIn(
        '<div class="stats-text">1/2 channels passed</div>', html_output
    )
    self.assertIn(
        "We've detected channel(s) `mock_channel2` where the posterior",
        html_output,
    )

    self.assertNotIn("Calibration score", html_output)

  def test_channel_calibration_recommendations_property(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(high_roi_channels=["ch1"]),
        channel_calibration_status={"ch1": False, "ch2": True},
    )

    recs = summary.channel_calibration_recommendations
    self.assertLen(recs, 2)
    self.assertEqual(recs[0][review_constants.CHANNEL_NAME], "ch1")
    self.assertFalse(recs[0][review_constants.IS_CALIBRATED])
    self.assertAlmostEqual(
        recs[0][review_constants.CALIBRATION_SCORE], 79.0, places=1
    )
    self.assertEqual(
        recs[0][review_constants.HIGH_ROI_STATUS], results.Status.REVIEW
    )
    self.assertEqual(
        recs[0][review_constants.LOW_ROI_STATUS], results.Status.PASS
    )
    self.assertEqual(
        recs[0][review_constants.HIGH_VARIANCE_STATUS], results.Status.PASS
    )
    self.assertEqual(
        recs[0][review_constants.POTENTIAL_BIAS_STATUS], results.Status.PASS
    )

    self.assertEqual(recs[1][review_constants.CHANNEL_NAME], "ch2")
    self.assertTrue(recs[1][review_constants.IS_CALIBRATED])
    self.assertEqual(
        recs[1][review_constants.CALIBRATION_SCORE],
        review_constants.CALIBRATED_CHANNEL_SCORE,
    )

  def test_channel_calibration_recommendations_property_many_channels(self):
    summary = _create_test_summary(
        channel_calibration_status={f"ch{i}": i % 2 == 0 for i in range(25)},
        include_passing_checks=True,
    )

    recs = summary.channel_calibration_recommendations
    # There should only be 12 recommendations (uncalibrated ones, i=1,3...23)
    self.assertLen(recs, 12)
    # Calibrated ones (ch0, ch2...) should be missing
    for rec in recs:
      self.assertFalse(rec[review_constants.IS_CALIBRATED])
      self.assertEqual(
          int(rec[review_constants.CHANNEL_NAME].replace("ch", "")) % 2, 1
      )

  def test_channel_calibration_recommendations_many_uncalibrated_channels(
      self,
  ):
    summary = _create_test_summary(
        channel_calibration_status={f"ch{i}": False for i in range(25)},
        include_passing_checks=True,
    )

    recs = summary.channel_calibration_recommendations
    # Should NOT be limited to 20, should show all 25
    self.assertLen(recs, 25)

    # Should be in original order
    for i in range(25):
      self.assertEqual(recs[i][review_constants.CHANNEL_NAME], f"ch{i}")

  def test_channel_calibration_recommendations_order_few_channels(self):
    summary = _create_test_summary(
        channel_calibration_status={"chA": False, "chB": False, "chC": True},
        include_passing_checks=True,
    )

    recs = summary.channel_calibration_recommendations
    self.assertLen(recs, 3)

    # Should be in original order: chA, chB, chC
    self.assertEqual(recs[0][review_constants.CHANNEL_NAME], "chA")
    self.assertEqual(recs[1][review_constants.CHANNEL_NAME], "chB")
    self.assertEqual(recs[2][review_constants.CHANNEL_NAME], "chC")

  def test_review_summary_repr_with_calibration(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(high_roi_channels=["ch1"]),
        channel_calibration_status={"ch1": False, "ch2": True},
    )
    repr_output = repr(summary)

    self.assertIn("Channel Calibration Recommendation", repr_output)
    self.assertNotIn("Implausible ROI", repr_output)
    self.assertIn("High ROI", repr_output)
    self.assertIn("Low ROI", repr_output)
    self.assertIn("Calibration Score", repr_output)
    expected_row = (
        f"{'ch1':<20} | {'79.0':<18} | {review_constants.DRIVER:<15} |"
        f" {review_constants.NON_DRIVER:<15} |"
        f" {review_constants.NON_DRIVER:<17} |"
        f" {review_constants.NON_DRIVER:<14}"
    )
    self.assertIn(expected_row, repr_output)
    expected_calibrated = (
        f"{'ch2':<20} | {'100.0':<18} | {'-' * 29} Calibrated {'-' * 29}"
    )
    self.assertIn(expected_calibrated, repr_output)

  def test_output_model_health_card(self):
    summary = _create_test_summary(summary_message="summary", health_score=95.2)
    temp_dir = self.create_tempdir().full_path
    summary.output_model_health_card(
        filename="health_card.html", filepath=temp_dir
    )
    output_filepath = os.path.join(temp_dir, "health_card.html")
    self.assertTrue(os.path.exists(output_filepath))
    with open(output_filepath, "r", encoding="utf-8") as f:
      content = f.read()
    self.assertIn("Model health summary card", content)

  def test_channel_recommendation_card_html(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            high_roi_channels=["uncalibrated_channel"]
        ),
        channel_calibration_status={
            "calibrated_channel": True,
            "uncalibrated_channel": False,
        },
        channel_scores={
            "uncalibrated_channel": 60.6,
            "calibrated_channel": 100.0,
        },
    )
    rec_card = summary._create_channel_recommendation_card_html()
    for s in (
        "uncalibrated_channel",
        "calibrated_channel",
        "Calibrated",
        f'<chip class="driver">{review_constants.DRIVER}</chip>',
        f'<chip class="non-driver">{review_constants.NON_DRIVER}</chip>',
        "Score",
        "channel-score-chart",
        "channel-score-chart warning",
        (
            '<td class="calibration-score-cell" style="white-space: nowrap'
            ' !important; text-align: center !important;">'
        ),
        (
            '<div class="score-container" style="display: inline-flex'
            " !important; flex-direction: row !important; align-items: center"
            " !important; justify-content: center !important; gap: 8px"
            ' !important; white-space: nowrap !important;">'
        ),
        (
            "display: inline-flex !important; align-items: center !important;"
            " justify-content: center !important; flex-shrink: 0 !important;"
            " white-space: nowrap !important; box-sizing: border-box"
            " !important;"
        ),
        (
            '<span class="score-value" style="font-size: 14px !important;'
            " font-weight: 500 !important; display: inline-block !important;"
            ' white-space: nowrap !important;">60.6</span>'
        ),
        "High ROI",
        "Low ROI",
        "status-banner-strip warning",
        (
            "We recommend incrementality experiments to improve prior accuracy"
            " for 'uncalibrated_channel': 'uncalibrated_channel' shows issues"
            " with high ROI. See Channel calibration recommendation below for"
            " more details."
        ),
    ):
      self.assertIn(s, rec_card)
    self.assertNotIn("Implausible ROI", rec_card)

    health_card = summary._create_health_card_html()
    for s in (
        "Calibration score",
        "80.3/100",
        "1/2 channels recommended",
        (
            "The overall calibration score is 80.3/100. We recommend"
            " incrementality experiments to improve prior accuracy for"
            " 'uncalibrated_channel'."
        ),
    ):
      self.assertIn(s, health_card)
    self.assertNotIn("Channel recommendation", health_card)

    calib_summary_card = summary._create_calibration_summary_card_html()
    for s in (
        "Meridian GeoX calibration summary",
        "Calibration score",
        "calibration-score-chart",
        '<div class="score-value">80.3</div>',
        "Prior calibration",
        '<chip class="pass">1 channel(s)</chip>',
        (
            "Calibration has been completed for 'calibrated_channel' using"
            " incrementality experiments"
        ),
        "Channel recommendation",
        (
            "We recommend incrementality experiments to improve prior accuracy"
            " for 'uncalibrated_channel': 'uncalibrated_channel' shows issues"
            " with high ROI. See Channel calibration recommendation below for"
            " more details."
        ),
        '<chip class="review">1 channel(s)</chip>',
    ):
      self.assertIn(s, calib_summary_card)
    self.assertNotIn("calibration-score-chart warning", calib_summary_card)

  def test_channel_recommendation_card_html_custom_driver_constants(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            high_roi_channels=["uncalibrated_channel"]
        ),
        channel_calibration_status={
            "calibrated_channel": True,
            "uncalibrated_channel": False,
        },
    )
    with mock.patch.object(
        review_constants, "DRIVER", "CustomDriver"
    ), mock.patch.object(review_constants, "NON_DRIVER", "CustomNonDriver"):
      rec_card = summary._create_channel_recommendation_card_html()
      self.assertIn('<chip class="driver">CustomDriver</chip>', rec_card)
      self.assertIn('<chip class="non-driver">CustomNonDriver</chip>', rec_card)

  def test_channel_recommendation_card_html_with_charts(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            high_roi_channels=["ch1"],
            high_variance_channels=["ch1"],
            low_correlation_channels=["ch1"],
        ),
        channel_calibration_status={"ch1": False},
        implausible_roi_chart_json='{"spec": "implausible"}',
        high_variance_chart_json='{"spec": "variance"}',
        potential_bias_chart_json='{"spec": "bias"}',
    )
    rec_card = summary._create_channel_recommendation_card_html()
    for s in (
        "implausible-roi-chart",
        "high-variance-chart",
        "potential-bias-chart",
        "advisory-warning-banner",
        "We recommend reviewing 'ch1' for having high ROI.",
        "We recommend reviewing 'ch1' for having high variance ROI.",
        "'ch1' shows potential bias",
    ):
      self.assertIn(s, rec_card)

  def test_health_card_html_calibration(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            high_roi_channels=["uncalibrated_channel"]
        ),
        channel_calibration_status={
            "calibrated_channel": True,
            "uncalibrated_channel": False,
        },
        channel_scores={
            "uncalibrated_channel": 60.6,
            "calibrated_channel": 100.0,
        },
    )

    html = summary._create_health_card_html()

    for s in (
        "<td>Calibration score</td>",
        "80.3/100",
        (
            "The overall calibration score is 80.3/100. We recommend"
            " incrementality experiments to improve prior accuracy for"
            " 'uncalibrated_channel'."
        ),
        '<chip class="review">Review</chip>',
        (
            '<div class="stats-text">1/2 channels recommended for'
            " calibration</div>"
        ),
    ):
      self.assertIn(s, html)
    for s in (
        '<div class="score-value">85.0</div>',
        "<th>High ROI</th>",
        "<th>Low ROI</th>",
    ):
      self.assertNotIn(s, html)
    self.assertNotIn(
        "calibrated_channel", html.replace("uncalibrated_channel", "")
    )

  @parameterized.named_parameters(
      (
          "two_channels",
          ["search", "pmax"],
          (
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'search' and 'pmax'."
          ),
          "2/2 channels recommended for calibration",
      ),
      (
          "three_channels",
          ["search", "pmax", "social"],
          (
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'search', 'pmax', and 'social'."
          ),
          "3/3 channels recommended for calibration",
      ),
  )
  def test_health_card_html_calibration_recommended_channels(
      self,
      channels: list[str],
      expected_text: str,
      expected_stats: str,
  ):
    summary = _create_test_summary(
        results_list=_mock_check_results(include_passing_checks=True),
        channel_calibration_status={ch: False for ch in channels},
        channel_scores={ch: 50.0 for ch in channels},
    )
    html = summary._create_health_card_html()
    self.assertIn(expected_text, html)
    self.assertIn(f'<div class="stats-text">{expected_stats}</div>', html)

  def test_health_card_html_calibration_no_checks_run(self):
    summary = _create_test_summary(
        channel_calibration_status={
            "calibrated_channel": True,
            "uncalibrated_channel": False,
        },
    )

    html = summary._create_health_card_html()
    self.assertNotIn("<td>Calibration score</td>", html)

  @parameterized.named_parameters(
      (
          "all_three_cards",
          True,
          True,
          [
              "Model health summary card",
              "Meridian GeoX calibration summary",
              "Calibration overview",
              "Channel calibration recommendation",
          ],
          [],
          True,
      ),
      (
          "without_overview_data",
          True,
          False,
          [
              "Model health summary card",
              "Model health score",
              "Channel calibration recommendation",
          ],
          ['id="calibration-overview"'],
          False,
      ),
      (
          "no_recommendations",
          False,
          False,
          ["Model health summary card", "Model health score"],
          [
              'id="summary"',
              'id="calibration-overview"',
              'class="channel-recommendation-card"',
          ],
          False,
      ),
  )
  def test_gen_model_health_card_calibration_cards(
      self,
      has_recommendations: bool,
      has_overview_data: bool,
      expected_strings: list[str],
      unexpected_strings: list[str],
      check_order: bool,
  ):
    check_results = (
        _mock_check_results(high_roi_channels=["uncalibrated_channel"])
        if has_recommendations
        else []
    )
    overview_data = (
        [_mock_channel_data("calibrated_channel", with_prior_dist=True)]
        if has_overview_data
        else []
    )
    channel_calibration_status = (
        {"calibrated_channel": True, "uncalibrated_channel": False}
        if has_recommendations
        else {}
    )
    summary = _create_test_summary(
        results_list=check_results,
        channel_calibration_status=channel_calibration_status,
        calibration_overview_data=overview_data,
    )

    html = summary._gen_model_health_card()
    for s in expected_strings:
      self.assertIn(s, html)
    for s in unexpected_strings:
      self.assertNotIn(s, html)

    if check_order:
      idx_summary = html.index('id="summary"')
      idx_overview = html.index('id="calibration-overview"')
      idx_recommendation = html.index('class="channel-recommendation-card"')
      self.assertLess(idx_summary, idx_overview)
      self.assertLess(idx_overview, idx_recommendation)

  def test_calibration_summary_card_html_potential_bias(self):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            low_correlation_channels=["biased_channel"]
        ),
        channel_calibration_status={
            "calibrated_channel": True,
            "biased_channel": False,
        },
        channel_scores={"calibrated_channel": 100.0, "biased_channel": 60.0},
    )
    calib_summary_card = summary._create_calibration_summary_card_html()
    self.assertIn(
        "We recommend incrementality experiments to improve prior accuracy for"
        " 'biased_channel': 'biased_channel' shows potential bias. See Channel"
        " calibration recommendation below for more details.",
        calib_summary_card,
    )

  @parameterized.named_parameters(
      (
          "zero_calibrated",
          [],
          '<chip class="neutral">0 channel(s)</chip>',
          (
              "No prior calibration has been completed using incrementality"
              " experiments."
          ),
      ),
      (
          "single_calibrated",
          ["calibrated_channel"],
          '<chip class="pass">1 channel(s)</chip>',
          (
              "Calibration has been completed for 'calibrated_channel' using"
              " incrementality experiments"
          ),
      ),
      (
          "two_channels",
          ["channel1", "channel2"],
          '<chip class="pass">2 channel(s)</chip>',
          (
              "Calibration has been completed for 'channel1' and 'channel2'"
              " using incrementality experiments"
          ),
      ),
      (
          "three_channels",
          ["channel1", "channel2", "channel3"],
          '<chip class="pass">3 channel(s)</chip>',
          (
              "Calibration has been completed for 'channel1', 'channel2', and"
              " 'channel3' using incrementality experiments"
          ),
      ),
  )
  def test_calibration_summary_card_html_calibrated_channels(
      self,
      calibrated: list[str],
      expected_pass_chip: str,
      expected_calibrated_text: str,
  ):
    summary = _create_test_summary(
        channel_calibration_status={ch: True for ch in calibrated}
        or {"uncalibrated_channel": False},
        calibrated_channel_names=calibrated,
    )
    card = summary._create_calibration_summary_card_html()
    for s in (
        "No channels require calibration.",
        '<chip class="neutral">0 channel(s)</chip>',
        expected_pass_chip,
        expected_calibrated_text,
    ):
      self.assertIn(s, card)

  def test_calibration_summary_card_html_calibrated_channels_preserves_status_order(
      self,
  ):
    summary = _create_test_summary(
        channel_calibration_status={"youtube": True, "demandgen": True},
        calibrated_channel_names=["demandgen", "youtube"],
    )
    card = summary._create_calibration_summary_card_html()
    self.assertIn(
        "Calibration has been completed for 'youtube' and 'demandgen' using"
        " incrementality experiments",
        card,
    )

  @parameterized.named_parameters(
      ("below_threshold_yellow", 60.6, "warning", "#fcc934"),
      ("at_threshold_blue", 67.5, None, "#6ba4ff"),
  )
  def test_calibration_score_chart_styling(
      self,
      score: float,
      expected_class: str | None,
      expected_color: str,
  ):
    summary = _create_test_summary(
        include_passing_checks=True,
        channel_scores={"ch": score},
    )
    rec_card = summary._create_channel_recommendation_card_html()
    calib_card = summary._create_calibration_summary_card_html()

    for card, chart_class in [
        (rec_card, "channel-score-chart"),
        (calib_card, "calibration-score-chart"),
    ]:
      self.assertIn(chart_class, card)
      self.assertIn(expected_color, card)
      if expected_class:
        self.assertIn(f"{chart_class} {expected_class}", card)
      else:
        self.assertNotIn(f"{chart_class} warning", card)

  @parameterized.named_parameters(
      (
          "pass_status_renders_info_banner",
          results.Status.PASS,
          "Passed",
          {"low_score_channel": 50.0},
          "status-banner-strip info",
          "check_circle",
      ),
      (
          "fail_status_renders_fail_banner",
          results.Status.FAIL,
          "Failed",
          {},
          "status-banner-strip fail",
          "error_outline",
      ),
  )
  def test_health_card_html_banner_status(
      self,
      overall_status: results.Status,
      summary_message: str,
      channel_scores: dict[str, float],
      expected_banner_class: str,
      expected_icon: str,
  ):
    summary = _create_test_summary(
        overall_status=overall_status,
        summary_message=summary_message,
        health_score=50.0 if overall_status == results.Status.FAIL else 90.0,
        channel_calibration_status={ch: False for ch in channel_scores},
        channel_scores=channel_scores,
    )
    html = summary._create_health_card_html()
    self.assertIn(f'class="{expected_banner_class}"', html)
    self.assertNotIn("status-banner-strip warning", html)
    self.assertIn(
        f'<span class="material-icons-outlined">{expected_icon}</span>', html
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="score_below_threshold_with_driver",
          check_results=_mock_check_results(
              high_roi_channels=["driver_channel"]
          ),
          channel_scores={"driver_channel": 60.0},
          expected_warning_banner=True,
          expected_warning_message=(
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'driver_channel': 'driver_channel' shows issues"
              " with high ROI. See Channel calibration recommendation below"
              " for more details."
          ),
      ),
      dict(
          testcase_name="score_below_threshold_without_driver",
          check_results=[],
          channel_scores={"low_score_channel": 50.0},
          expected_warning_banner=True,
          expected_warning_message=(
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'low_score_channel'. See Channel calibration"
              " recommendation below for more details."
          ),
      ),
      dict(
          testcase_name="score_above_threshold_with_driver",
          check_results=_mock_check_results(
              high_roi_channels=["driver_channel"]
          ),
          channel_scores={"driver_channel": 85.0},
          expected_warning_banner=False,
          expected_warning_message=(
              "No channels require calibration. However, 'driver_channel' shows"
              " issues with high ROI. We recommend reviewing the table and"
              " plots below to check if this channel may be a good candidate"
              " for calibration via an incrementality experiment such as those"
              " run with Meridian GeoX."
          ),
      ),
      dict(
          testcase_name="score_above_threshold_without_driver",
          check_results=[],
          channel_scores={"clean_channel": 85.0},
          expected_warning_banner=False,
          expected_warning_message=None,
      ),
      dict(
          testcase_name="calibrated_driver_and_low_score_ignored",
          check_results=_mock_check_results(
              high_roi_channels=["calibrated_driver_ch"]
          ),
          calibrated_channels=[
              "calibrated_driver_ch",
              "calibrated_low_score_ch",
          ],
          channel_scores={
              "calibrated_driver_ch": 100.0,
              "calibrated_low_score_ch": 50.0,
              "uncalibrated_clean_ch": 95.0,
          },
          expected_warning_banner=False,
          expected_warning_message=None,
      ),
      dict(
          testcase_name=(
              "multiple_channels_table_order_preserved_in_warning_banner"
          ),
          check_results=_mock_check_results(
              high_roi_channels=["search"],
              high_variance_channels=["display"],
          ),
          calibrated_channels=["youtube"],
          channel_scores={
              "youtube": 100.0,
              "search": 55.0,
              "display": 60.0,
              "pmax": 50.0,
              "demandgen": 90.0,
          },
          expected_warning_banner=True,
          expected_warning_message=(
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'search', 'display', and 'pmax': 'search' shows"
              " issues with high ROI and 'display' shows issues with high"
              " variance. See Channel calibration recommendation below for"
              " more details."
          ),
      ),
      dict(
          testcase_name=(
              "multiple_drivers_above_threshold_table_order_preserved"
          ),
          check_results=_mock_check_results(
              high_roi_channels=["search"],
              high_variance_channels=["display"],
          ),
          channel_scores={
              "search": 85.0,
              "display": 90.0,
          },
          expected_warning_banner=False,
          expected_warning_message=(
              "No channels require calibration. However, 'search' shows"
              " issues with high ROI and 'display' shows issues with high"
              " variance. We recommend reviewing the table and plots below"
              " to check if these channels may be good candidates for"
              " calibration via an incrementality experiment such as those run"
              " with Meridian GeoX."
          ),
      ),
  )
  def test_channel_recommendation_card_html_banner_status(
      self,
      check_results: list[results.CheckResult],
      channel_scores: dict[str, float],
      expected_warning_banner: bool,
      calibrated_channels: Sequence[str] = (),
      expected_warning_message: str | None = None,
  ):
    summary = _create_test_summary(
        results_list=check_results,
        health_score=100.0,
        calibrated_channel_names=calibrated_channels,
        channel_scores=channel_scores,
    )
    rec_card = summary._create_channel_recommendation_card_html()
    expected_banner = "warning" if expected_warning_banner else "info"
    unexpected_banner = "info" if expected_warning_banner else "warning"
    self.assertEqual(summary.has_calibration_warning, expected_warning_banner)
    self.assertIn(f"status-banner-strip {expected_banner}", rec_card)
    self.assertNotIn(f"status-banner-strip {unexpected_banner}", rec_card)
    self.assertIn(
        f'<span class="material-icons-outlined">{expected_banner}</span>',
        rec_card,
    )
    self.assertIn(
        expected_warning_message
        or results.NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION,
        rec_card,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name=(
              "uncalibrated_issues_render_warning_banners_and_ignore_calibrated"
          ),
          check_results=_mock_check_results(
              high_roi_channels=["calibrated_ch", "uncal_high"],
              low_roi_channels=["calibrated_ch_2", "uncal_low"],
              high_variance_channels=["calibrated_ch", "uncal_var"],
              low_correlation_channels=[
                  "calibrated_ch",
                  "uncal_bias_1",
                  "uncal_bias_2",
              ],
          ),
          calibrated_channels=["calibrated_ch", "calibrated_ch_2"],
          uncalibrated_channels=[
              "uncal_high",
              "uncal_low",
              "uncal_var",
              "uncal_bias_1",
              "uncal_bias_2",
          ],
          expected_warning_banners_count=3,
          expected_info_banners_count=0,
          expected_in_output=[
              (
                  "We recommend reviewing 'uncal_high' for having high ROI and"
                  " 'uncal_low' for having low ROI."
              ),
              (
                  "We recommend reviewing 'uncal_var' for having high variance"
                  " ROI."
              ),
              (
                  "'uncal_bias_1' and 'uncal_bias_2' show potential bias as"
                  " they have low correlation with all controls and thus may be"
                  " missing relevant controls."
              ),
          ],
          unexpected_in_output=[
              "'calibrated_ch'",
              "'calibrated_ch_2'",
          ],
      ),
      dict(
          testcase_name="all_calibrated_or_passed_renders_info_banners",
          check_results=_mock_check_results(
              high_roi_channels=["calibrated_ch"],
              low_roi_channels=["calibrated_ch_2"],
              include_passing_checks=True,
          ),
          calibrated_channels=["calibrated_ch", "calibrated_ch_2"],
          expected_warning_banners_count=0,
          expected_info_banners_count=3,
          expected_in_output=[
              "Channels closer to the boundaries of the Implausible High ROI",
              "Channels close to the boundary of the High Variance ROI region",
              (
                  "This plot displays your media and reach and frequency"
                  " channels along with their correlation with your available"
                  " controls."
              ),
          ],
          unexpected_in_output=[
              "for having high ROI",
              "for having low ROI",
              "for having high variance ROI",
              "show potential bias",
              "shows potential bias",
          ],
      ),
      dict(
          testcase_name="mixed_warning_and_info_banners",
          check_results=_mock_check_results(
              high_roi_channels=["uncal_high"],
              include_passing_checks=True,
          ),
          uncalibrated_channels=["uncal_high"],
          expected_warning_banners_count=1,
          expected_info_banners_count=2,
          expected_in_output=[
              "We recommend reviewing 'uncal_high' for having high ROI.",
              "Channels close to the boundary of the High Variance ROI region",
              (
                  "This plot displays your media and reach and frequency"
                  " channels along with their correlation with your available"
                  " controls."
              ),
          ],
          unexpected_in_output=[],
      ),
      dict(
          testcase_name="default_calibration_status_none_flags_all",
          check_results=_mock_check_results(
              high_roi_channels=["ch_high"],
              low_roi_channels=["ch_low"],
              high_variance_channels=["ch_var"],
              low_correlation_channels=["ch_bias"],
          ),
          expected_warning_banners_count=3,
          expected_info_banners_count=0,
          expected_in_output=[
              (
                  "We recommend reviewing 'ch_high' for having high ROI and"
                  " 'ch_low' for having low ROI."
              ),
              "We recommend reviewing 'ch_var' for having high variance ROI.",
              (
                  "'ch_bias' shows potential bias as it has low correlation"
                  " with all controls and thus may be missing relevant"
                  " controls."
              ),
          ],
          unexpected_in_output=[],
      ),
      dict(
          testcase_name="plot_banners_channel_order_preserves_table_order",
          check_results=_mock_check_results(
              high_roi_channels=["search", "display"],
              low_roi_channels=["pmax", "demandgen"],
              high_variance_channels=["search", "display"],
              low_correlation_channels=["search", "display"],
          ),
          uncalibrated_channels=["search", "display", "pmax", "demandgen"],
          expected_warning_banners_count=3,
          expected_info_banners_count=0,
          expected_in_output=[
              (
                  "We recommend reviewing 'search' and 'display' for having"
                  " high ROI and 'pmax' and 'demandgen' for having low ROI."
              ),
              (
                  "We recommend reviewing 'search' and 'display' for having"
                  " high variance ROI."
              ),
              (
                  "'search' and 'display' show potential bias as they have low"
                  " correlation with all controls and thus may be missing"
                  " relevant controls."
              ),
          ],
          unexpected_in_output=[
              "'display' and 'search'",
              "'demandgen' and 'pmax'",
          ],
      ),
  )
  def test_channel_recommendation_card_html_plot_descriptions_and_banners(
      self,
      check_results: list[results.CheckResult],
      expected_warning_banners_count: int,
      expected_info_banners_count: int,
      expected_in_output: list[str],
      unexpected_in_output: list[str],
      calibrated_channels: Sequence[str] = (),
      uncalibrated_channels: Sequence[str] = (),
      channel_calibration_status: Mapping[str, bool] | None = None,
  ):
    if channel_calibration_status is None:
      channel_calibration_status = {ch: True for ch in calibrated_channels} | {
          ch: False for ch in uncalibrated_channels
      }
    summary = _create_test_summary(
        results_list=check_results,
        implausible_roi_chart_json='{"spec": "implausible"}',
        high_variance_chart_json='{"spec": "variance"}',
        potential_bias_chart_json='{"spec": "bias"}',
        channel_calibration_status=channel_calibration_status,
    )
    rec_card = summary._create_channel_recommendation_card_html()

    self.assertEqual(
        rec_card.count("status-banner-strip warning advisory-warning-banner"),
        expected_warning_banners_count,
    )
    self.assertEqual(
        rec_card.count("status-banner-strip info advisory-warning-banner"),
        expected_info_banners_count,
    )
    for expected_str in expected_in_output:
      self.assertIn(expected_str, rec_card)
    for unexpected_str in unexpected_in_output:
      self.assertNotIn(unexpected_str, rec_card)

  @parameterized.named_parameters(
      (
          "low_roi",
          _mock_check_results(low_roi_channels=["channel_name"]),
          review_constants.LOW_ROI,
      ),
      (
          "high_roi",
          _mock_check_results(high_roi_channels=["channel_name"]),
          review_constants.HIGH_ROI,
      ),
      (
          "high_variance",
          _mock_check_results(high_variance_channels=["channel_name"]),
          review_constants.HIGH_VARIANCE,
      ),
      (
          "potential_bias",
          _mock_check_results(low_correlation_channels=["channel_name"]),
          review_constants.POTENTIAL_BIAS,
      ),
  )
  def test_get_recommended_channels_with_issues(
      self, check_results, expected_issue_type
  ):
    summary = _create_test_summary(results_list=check_results)
    issues = summary._get_recommended_channels_with_issues()
    self.assertEqual(issues, {"channel_name": [expected_issue_type]})
    self.assertEqual(
        summary._uncalibrated_channels_with_driver_issues(), issues
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="ignores_calibrated_channels",
          channel_calibration_status={
              "cal_1": True,
              "cal_2": True,
              "cal_3": True,
          },
          high_roi_channels=["cal_1"],
          high_variance_channels=["cal_2"],
          low_correlation_channels=["cal_3"],
          expected_channels=[],
      ),
      dict(
          testcase_name="empty_status_treats_all_as_uncalibrated",
          channel_calibration_status={},
          high_roi_channels=["channel_1"],
          high_variance_channels=[],
          low_correlation_channels=[],
          expected_channels=["channel_1"],
      ),
      dict(
          testcase_name="preserves_table_channel_order",
          channel_calibration_status={
              "youtube": True,
              "search": False,
              "display": False,
              "pmax": False,
          },
          high_roi_channels=["pmax"],
          high_variance_channels=["search"],
          low_correlation_channels=[],
          expected_channels=["search", "pmax"],
      ),
  )
  def test_get_recommended_channels_with_issues_calibration_filtering(
      self,
      channel_calibration_status,
      high_roi_channels,
      high_variance_channels,
      low_correlation_channels,
      expected_channels,
  ):
    summary = _create_test_summary(
        results_list=_mock_check_results(
            high_roi_channels=high_roi_channels,
            high_variance_channels=high_variance_channels,
            low_correlation_channels=low_correlation_channels,
        ),
        channel_calibration_status=channel_calibration_status,
    )
    issues = summary._get_recommended_channels_with_issues()
    self.assertEqual(list(issues.keys()), expected_channels)
    self.assertEqual(
        summary._uncalibrated_channels_with_driver_issues(), issues
    )


class ImplausibleROICheckResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          case=results.ImplausibleROIAggregateCases.PASS,
          high_roi_channels=[],
          low_roi_channels=[],
          aggregate_details={"implausible_roi_msg": ""},
          expected_status=results.Status.PASS,
          expected_recommendation="All channels have plausible ROI estimates.",
      ),
      dict(
          testcase_name="review_high",
          case=results.ImplausibleROIAggregateCases.REVIEW,
          high_roi_channels=["channel1"],
          low_roi_channels=[],
          aggregate_details={
              "implausible_roi_msg": (
                  "We've detected implausibly high ROI estimates (for"
                  " channel(s) `channel1`)."
              )
          },
          expected_status=results.Status.REVIEW,
          expected_recommendation=(
              "We've detected implausibly high ROI estimates (for channel(s)"
              f" `channel1`). {review_constants.IMPLAUSIBLE_ROI_RECOMMENDATION}"
          ),
      ),
      dict(
          testcase_name="review_low",
          case=results.ImplausibleROIAggregateCases.REVIEW,
          high_roi_channels=[],
          low_roi_channels=["channel1"],
          aggregate_details={
              "implausible_roi_msg": (
                  "We've detected implausibly low ROI estimates (for"
                  " channel(s) `channel1`)."
              )
          },
          expected_status=results.Status.REVIEW,
          expected_recommendation=(
              "We've detected implausibly low ROI estimates (for channel(s)"
              f" `channel1`). {review_constants.IMPLAUSIBLE_ROI_RECOMMENDATION}"
          ),
      ),
      dict(
          testcase_name="review_both",
          case=results.ImplausibleROIAggregateCases.REVIEW,
          high_roi_channels=["channel1"],
          low_roi_channels=["channel2"],
          aggregate_details={
              "implausible_roi_msg": (
                  "We've detected implausibly high ROI estimates (for"
                  " channel(s) `channel1`) and low ROI estimates (for"
                  " channel(s) `channel2`)."
              )
          },
          expected_status=results.Status.REVIEW,
          expected_recommendation=(
              "We've detected implausibly high ROI estimates (for channel(s)"
              " `channel1`) and low ROI estimates (for channel(s) `channel2`)."
              f" {review_constants.IMPLAUSIBLE_ROI_RECOMMENDATION}"
          ),
      ),
  )
  def test_implausible_roi_check_result(
      self,
      case: results.ImplausibleROIAggregateCases,
      high_roi_channels: list[str],
      low_roi_channels: list[str],
      aggregate_details: dict[str, Any],
      expected_status: results.Status,
      expected_recommendation: str,
  ):
    result = results.ImplausibleROICheckResult(
        case=case,
        channel_results=[],
        high_roi_channels=high_roi_channels,
        low_roi_channels=low_roi_channels,
        aggregate_details=aggregate_details,
    )
    self.assertEqual(result.case.status, expected_status)
    self.assertEqual(result.recommendation, expected_recommendation)


class HighVarianceCheckResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          case=results.HighVarianceAggregateCases.PASS,
          high_variance_channels=[],
          expected_status=results.Status.PASS,
          expected_recommendation="All channels have acceptable ROI variance.",
      ),
      dict(
          testcase_name="review",
          case=results.HighVarianceAggregateCases.REVIEW,
          high_variance_channels=["channel1", "channel2"],
          expected_status=results.Status.REVIEW,
          expected_recommendation=(
              "We've detected channel(s) `channel1`, `channel2` with highly"
              " uncertain ROI estimates (wide posterior intervals)."
              f" {review_constants.HIGH_VARIANCE_ROI_RECOMMENDATION}"
          ),
      ),
  )
  def test_high_variance_check_result(
      self,
      case,
      high_variance_channels,
      expected_status,
      expected_recommendation,
  ):
    result = results.HighVarianceCheckResult(
        case=case,
        channel_results=[],
        high_variance_channels=high_variance_channels,
    )
    self.assertEqual(result.case.status, expected_status)
    self.assertEqual(result.recommendation, expected_recommendation)


class PotentialBiasCheckResultTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          case=results.PotentialBiasAggregateCases.PASS,
          low_correlation_channels=[],
          expected_status=results.Status.PASS,
          expected_recommendation=(
              "All channels have sufficient correlation with control variables."
          ),
          corr_value=0.5,
      ),
      dict(
          testcase_name="review",
          case=results.PotentialBiasAggregateCases.REVIEW,
          low_correlation_channels=["channel1"],
          expected_status=results.Status.REVIEW,
          expected_recommendation=(
              "We've detected channel(s) `channel1` with very low correlation"
              " with all included control variables."
              f" {review_constants.POTENTIAL_BIAS_RECOMMENDATION}"
          ),
          corr_value=0.0,
      ),
  )
  def test_potential_bias_check_result(
      self,
      case,
      low_correlation_channels,
      expected_status,
      expected_recommendation,
      corr_value,
  ):
    corr_matrix = xr.DataArray(
        np.array([[[corr_value]]]),
        coords={
            constants.GEO: ["geo1"],
            constants.CHANNEL: ["channel1"],
            constants.CONTROL_VARIABLE: ["control1"],
        },
        dims=[
            constants.GEO,
            constants.CHANNEL,
            constants.CONTROL_VARIABLE,
        ],
    )
    result = results.PotentialBiasCheckResult(
        case=case,
        channel_results=[],
        low_correlation_channels=low_correlation_channels,
        correlation_matrix=corr_matrix,
    )
    self.assertEqual(result.case.status, expected_status)
    self.assertEqual(result.recommendation, expected_recommendation)
    xr.testing.assert_equal(
        result.details[review_constants.CORRELATION_MATRIX], corr_matrix
    )

  def test_generate_potential_bias_chart_json(self):
    corr_matrix = xr.DataArray(
        np.array([
            [[0.05, 0.15], [0.08, 0.02]],
            [[0.12, 0.04], [0.01, 0.09]],
            [[0.02, 0.18], [0.06, 0.11]],
        ]),
        coords={
            constants.GEO: ["geo1", "geo2", "geo3"],
            constants.CHANNEL: ["channel1", "channel2"],
            constants.CONTROL_VARIABLE: ["control1", "control2"],
        },
        dims=[
            constants.GEO,
            constants.CHANNEL,
            constants.CONTROL_VARIABLE,
        ],
    )
    mock_potential_bias = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.REVIEW,
        channel_results=[
            results.PotentialBiasChannelResult(
                case=results.PotentialBiasChannelCases.LOW_CORRELATION,
                channel_name="channel1",
                max_abs_correlation=0.18,
            ),
            results.PotentialBiasChannelResult(
                case=results.PotentialBiasChannelCases.LOW_CORRELATION,
                channel_name="channel2",
                max_abs_correlation=0.11,
            ),
        ],
        low_correlation_channels=["channel1", "channel2"],
        correlation_matrix=corr_matrix,
    )
    chart_json = plots.generate_potential_bias_chart_json(mock_potential_bias)
    summary = _create_test_summary(
        results_list=[mock_potential_bias],
        channel_calibration_status={"channel1": False, "channel2": False},
        potential_bias_chart_json=chart_json,
    )
    temp_dir = self.create_tempdir().full_path
    summary.output_model_health_card("summary.html", temp_dir)
    with open(os.path.join(temp_dir, "summary.html")) as f:
      html_content = f.read()
    self.assertIn("channel1 - control1", html_content)
    self.assertIsNotNone(chart_json)
    chart_dict = json.loads(chart_json)

    self.assertIn("layer", chart_dict)
    layers = chart_dict["layer"]

    geos_layer = None
    max_layer = None
    for layer in layers:
      if layer.get("mark", {}).get("type") == "point":
        if layer.get("mark", {}).get("shape") == "diamond":
          max_layer = layer
        else:
          geos_layer = layer

    self.assertIsNotNone(geos_layer)
    self.assertIsNotNone(max_layer)

    self.assertEqual(
        geos_layer["encoding"]["y"]["field"], review_constants.PAIR
    )
    self.assertEqual(max_layer["encoding"]["y"]["field"], review_constants.PAIR)

    self.assertIn("channel1 - control1", chart_json)
    self.assertNotIn("channel1 vs. control1", chart_json)

    all_pairs = []
    if "datasets" in chart_dict:
      for dataset in chart_dict["datasets"].values():
        for row in dataset:
          if isinstance(row, dict) and review_constants.PAIR in row:
            all_pairs.append(row[review_constants.PAIR])
    self.assertIn("channel1 - control1", all_pairs)

    geos_domain = geos_layer["encoding"]["x"]["scale"]["domain"]
    max_domain = max_layer["encoding"]["x"]["scale"]["domain"]

    self.assertAlmostEqual(geos_domain[0], -0.23)
    self.assertAlmostEqual(geos_domain[1], 0.23)
    self.assertAlmostEqual(max_domain[0], -0.23)
    self.assertAlmostEqual(max_domain[1], 0.23)


class FormatListWithAndTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="empty", items=[], expected=""),
      dict(testcase_name="one", items=["a"], expected="a"),
      dict(testcase_name="two", items=["a", "b"], expected="a and b"),
      dict(
          testcase_name="three", items=["a", "b", "c"], expected="a, b, and c"
      ),
  )
  def test_format_list_with_and(self, items, expected):
    self.assertEqual(results._format_list_with_and(items), expected)


class ReviewConstantsTest(parameterized.TestCase):

  def test_channel_colors_constant(self):
    self.assertLen(review_constants.CHANNEL_COLORS, 20)
    self.assertLen(set(review_constants.CHANNEL_COLORS), 20)
    for color in review_constants.CHANNEL_COLORS:
      self.assertTrue(
          color.startswith("#") and len(color) == 7,
          f"Invalid hex color: {color}",
      )
    self.assertEqual(review_constants.CHANNEL_COLORS[0], "#185abc")
    self.assertEqual(review_constants.CHANNEL_COLORS[-1], "#8d0053")

  def test_implausible_roi_constants(self):
    self.assertEqual(review_constants.IMPLAUSIBLE_ROI_GAP_PLOTTED, 19.0)
    self.assertAlmostEqual(
        review_constants.IMPLAUSIBLE_ROI_THRESHOLD_LOWER
        * review_constants.IMPLAUSIBLE_ROI_SCALE_FACTOR,
        19.0,
    )

  def test_driver_constants(self):
    self.assertEqual(review_constants.DRIVER, "Driver")
    self.assertEqual(review_constants.NON_DRIVER, "Non-Driver")


_CALIBRATION_LIMIT_MESSAGE = (
    " The five highest-spend channels with experiments are plotted here"
    " along with their five experiments with the smallest adjusted standard"
    " error."
)
_OVERVIEW_LIMIT_MESSAGE = _CALIBRATION_LIMIT_MESSAGE
_DETAILS_LIMIT_MESSAGE = _CALIBRATION_LIMIT_MESSAGE


class CalibrationOverviewCardTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", 0, [], []),
      (
          "single_channel",
          1,
          [
              "Calibration overview",
              "ch_0 calibration",
              "spec_0 = JSON.parse",
              "for 'ch_0'.",
          ],
          [_OVERVIEW_LIMIT_MESSAGE],
      ),
      (
          "at_threshold",
          5,
          [
              "Calibration overview",
              "ch_0 calibration",
              "ch_4 calibration",
          ],
          [_OVERVIEW_LIMIT_MESSAGE],
          5,
      ),
      (
          "exceeds_channels_threshold",
          6,
          [
              "Calibration overview",
              "ch_0 calibration",
              "ch_4 calibration",
              "for 'ch_0', 'ch_1', 'ch_2', 'ch_3', and 'ch_4'.",
              _OVERVIEW_LIMIT_MESSAGE,
          ],
          ["ch_5 calibration"],
      ),
      (
          "exceeds_experiments_threshold",
          1,
          [
              "Calibration overview",
              "ch_0 calibration",
              _OVERVIEW_LIMIT_MESSAGE,
          ],
          [],
          6,
      ),
      (
          "exceeds_both_thresholds",
          6,
          [
              "Calibration overview",
              "ch_0 calibration",
              "ch_4 calibration",
              _OVERVIEW_LIMIT_MESSAGE,
          ],
          ["ch_5 calibration"],
          6,
      ),
  )
  def test_create_calibration_overview_card_html(
      self,
      num_channels: int,
      expected_strings: list[str],
      unexpected_strings: list[str],
      num_experiments: int = 0,
  ):
    ch_list = [
        _mock_channel_data(
            f"ch_{i}",
            float(100 - i),
            num_experiments=num_experiments,
            chart_json=f'{{"test": "json_{i}"}}',
            with_prior_dist=True,
        )
        for i in range(num_channels)
    ]
    summary = _create_test_summary(calibration_overview_data=ch_list)
    html = summary._create_calibration_overview_card_html()
    if num_channels == 0:
      self.assertEqual(html, "")
    for s in expected_strings:
      self.assertIn(s, html)
    for s in unexpected_strings:
      self.assertNotIn(s, html)

  def test_create_calibration_overview_card_html_channel_order_follows_status(
      self,
  ):
    ch_a = _mock_channel_data("ch_a", 200.0, chart_json='{"test": "json_a"}')
    ch_b = _mock_channel_data("ch_b", 100.0, chart_json='{"test": "json_b"}')
    summary = _create_test_summary(
        health_score=1.0,
        channel_calibration_status={"ch_b": True, "ch_a": True},
        calibration_overview_data=[ch_a, ch_b],
    )
    html = summary._create_calibration_overview_card_html()
    self.assertIn("for 'ch_b' and 'ch_a'.", html)

  def test_create_calibration_overview_card_html_none_calibrated_output(self):
    ch_list = [
        results.CalibrationOverviewChannelData(
            channel_name="ch_none",
            spend=100.0,
            calibrated_output=None,
            chart_json='{"test": "json"}',
        ),
    ]
    summary = results.ReviewSummary(
        overall_status=results.Status.PASS,
        summary_message="Passed",
        results=[],
        health_score=1.0,
        calibration_overview_data=ch_list,
    )
    html = summary._create_calibration_overview_card_html()
    self.assertIn("Calibration overview", html)
    self.assertIn("ch_none calibration", html)
    self.assertNotIn(_OVERVIEW_LIMIT_MESSAGE, html)

  def test_create_calibration_overview_card_html_experiments_none(self):
    mock_output = calibration_base.CalibrationOutput(
        channel_name="ch_0",
        experiments=None,  # pyrefly: ignore[bad-argument-type]
        intermediary_prior=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
    )
    ch_list = [
        results.CalibrationOverviewChannelData(
            channel_name="ch_0",
            spend=100.0,
            calibrated_output=mock_output,
            chart_json='{"test": "json"}',
        ),
    ]
    summary = results.ReviewSummary(
        overall_status=results.Status.PASS,
        summary_message="Passed",
        results=[],
        health_score=1.0,
        calibration_overview_data=ch_list,
    )
    html = summary._create_calibration_overview_card_html()
    self.assertIn("Calibration overview", html)
    self.assertNotIn(_OVERVIEW_LIMIT_MESSAGE, html)


class CalibrationDetailsCardTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="single_channel",
          channel_configs=[("test_channel", 100.0, 1)],
          expected_in=(
              "Calibration details",
              "calibration-details-chart-0",
              "spec_0 = JSON.parse",
              "vegaEmbed('#calibration-details-chart-0'",
              "These plots display your incrementality experiments",
          ),
          expected_not_in=(_DETAILS_LIMIT_MESSAGE,),
      ),
      dict(
          testcase_name="exceeds_channels_threshold",
          channel_configs=[(f"ch_{i}", float(100 - i), 1) for i in range(6)],
          expected_in=(
              "Calibration details",
              "calibration-details-chart-0",
              "calibration-details-chart-4",
              _DETAILS_LIMIT_MESSAGE,
          ),
          expected_not_in=("calibration-details-chart-5",),
      ),
      dict(
          testcase_name="exceeds_experiments_threshold",
          channel_configs=[("ch_0", 100.0, 6)],
          expected_in=(
              "Calibration details",
              "calibration-details-chart-0",
              _DETAILS_LIMIT_MESSAGE,
          ),
          expected_not_in=(),
      ),
      dict(
          testcase_name="exceeds_both_thresholds",
          channel_configs=[(f"ch_{i}", float(100 - i), 6) for i in range(6)],
          expected_in=(
              "Calibration details",
              "calibration-details-chart-0",
              "calibration-details-chart-4",
              _DETAILS_LIMIT_MESSAGE,
          ),
          expected_not_in=("calibration-details-chart-5",),
      ),
  )
  def test_create_calibration_details_card_html(
      self, channel_configs, expected_in, expected_not_in
  ):
    ch_list = [
        _mock_channel_data(
            name, spend, num_exps, details_chart_json='{"test_details": "json"}'
        )
        for name, spend, num_exps in channel_configs
    ]
    summary = _create_test_summary(
        health_score=1.0,
        calibration_overview_data=ch_list,
    )
    html = summary._create_calibration_details_card_html()
    for item in expected_in:
      self.assertIn(item, html)
    for item in expected_not_in:
      self.assertNotIn(item, html)

  @parameterized.named_parameters(
      ("empty_data", False, None),
      ("none_chart_json", True, None),
      ("empty_chart_json", True, ""),
  )
  def test_create_calibration_details_card_html_empty(
      self, has_channel_data: bool, chart_json: str | None
  ):
    calibration_overview_data = (
        [_mock_channel_data("ch0", chart_json=chart_json)]
        if has_channel_data
        else []
    )
    summary = _create_test_summary(
        health_score=1.0,
        calibration_overview_data=calibration_overview_data,
    )
    self.assertEqual(summary._create_calibration_details_card_html(), "")

  def test_gen_model_health_card_full_card_ordering(self):
    details_ch = _mock_channel_data(
        "ch1", 100.0, num_experiments=1, details_chart_json='{"details": true}'
    )
    summary = _create_test_summary(
        overall_status=results.Status.REVIEW,
        summary_message="Review needed",
        results_list=_mock_check_results(high_roi_channels=["ch1"]),
        health_score=85.0,
        channel_calibration_status={"ch1": False},
        calibration_overview_data=[details_ch],
    )
    html = summary._gen_model_health_card()
    self.assertIn('id="calibration-details"', html)
    self.assertIn('class="channel-recommendation-card"', html)
    self.assertIn("calibration-details-chart-0", html)
    self.assertIn("spec_0 = JSON.parse", html)
    self.assertIn("vegaEmbed('#calibration-details-chart-0'", html)
    details_idx = html.find('id="calibration-details"')
    rec_idx = html.find('class="channel-recommendation-card"')
    self.assertNotEqual(details_idx, -1)
    self.assertNotEqual(rec_idx, -1)
    self.assertLess(details_idx, rec_idx)

  @parameterized.named_parameters(
      (
          "uses_details_chart_json",
          '{"overview_spec": true}',
          '{"details_spec": true}',
          ("details_spec",),
          ("overview_spec",),
          False,
      ),
      (
          "ignores_overview_chart_json_when_details_none",
          '{"overview_spec": true}',
          None,
          (),
          ("overview_spec",),
          True,
      ),
  )
  def test_create_calibration_details_card_html_chart_json_filtering(
      self,
      chart_json: str | None,
      details_chart_json: str | None,
      expected_in: tuple[str, ...],
      expected_not_in: tuple[str, ...],
      expected_empty: bool,
  ):
    channel_data = _mock_channel_data(
        "ch1",
        chart_json=chart_json,
        details_chart_json=details_chart_json,
    )
    summary = _create_test_summary(
        health_score=1.0,
        calibration_overview_data=[channel_data],
    )
    html = summary._create_calibration_details_card_html()
    if expected_empty:
      self.assertEqual(html, "")
    for item in expected_in:
      self.assertIn(item, html)
    for item in expected_not_in:
      self.assertNotIn(item, html)


class CalibrationScoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="center_returns_100",
          roi=np.sqrt(0.5 * 20.0),
          spend_share=0.5,
          roi_lower_bound=0.5,
          roi_upper_bound=20.0,
          expected_score=100.0,
      ),
      dict(
          testcase_name="boundary_returns_50",
          roi=20.0,
          spend_share=1.0,
          roi_lower_bound=0.5,
          roi_upper_bound=20.0,
          expected_score=50.0,
      ),
      dict(
          testcase_name="zero_spend_returns_100",
          roi=10.0,
          spend_share=0.0,
          roi_lower_bound=0.5,
          roi_upper_bound=20.0,
          expected_score=100.0,
      ),
      dict(
          testcase_name="negative_spend_returns_100",
          roi=10.0,
          spend_share=-0.5,
          roi_lower_bound=0.5,
          roi_upper_bound=20.0,
          expected_score=100.0,
      ),
  )
  def test_normalized_center_bowl(
      self,
      roi: float,
      spend_share: float,
      roi_lower_bound: float,
      roi_upper_bound: float,
      expected_score: float,
  ):
    score = results._normalized_center_bowl(
        roi=roi,
        spend_share=spend_share,
        roi_lower_bound=roi_lower_bound,
        roi_upper_bound=roi_upper_bound,
    )
    self.assertAlmostEqual(score, expected_score, places=5)

  @parameterized.named_parameters(
      dict(testcase_name="zero_roi", roi=0.0, spend_share=0.5),
      dict(testcase_name="negative_roi", roi=-1.0, spend_share=0.5),
      dict(testcase_name="low_roi", roi=0.0001, spend_share=0.5),
      dict(testcase_name="high_roi", roi=1000.0, spend_share=0.5),
  )
  def test_normalized_center_bowl_extreme_roi(
      self,
      roi: float,
      spend_share: float,
  ):
    score = results._normalized_center_bowl(
        roi=roi,
        spend_share=spend_share,
    )
    self.assertGreaterEqual(score, 0.0)
    self.assertLess(score, 50.0)

  def test_normalized_center_bowl_equal_bounds(self):
    # Tests epsilon guard when lower == upper bound and spend_share == 1.0
    score_center = results._normalized_center_bowl(
        roi=1.0,
        spend_share=1.0,
        roi_lower_bound=1.0,
        roi_upper_bound=1.0,
    )
    self.assertAlmostEqual(score_center, 100.0, places=5)

  @parameterized.named_parameters(
      dict(
          testcase_name="below_ideal_returns_100",
          relative_width_ratio=1.0,
          spend_share=0.5,
          high_variance_threshold=1.0,
          ideal_threshold=0.5,
          expected_score=100.0,
      ),
      dict(
          testcase_name="at_threshold_returns_50",
          relative_width_ratio=2.0,
          spend_share=0.5,
          high_variance_threshold=1.0,
          ideal_threshold=0.5,
          expected_score=50.0,
      ),
      dict(
          testcase_name="extreme_high_variance_returns_near_zero",
          relative_width_ratio=100.0,
          spend_share=1.0,
          high_variance_threshold=1.0,
          ideal_threshold=0.5,
          expected_score=0.0,
      ),
  )
  def test_normalized_half_bowl(
      self,
      relative_width_ratio: float,
      spend_share: float,
      high_variance_threshold: float,
      ideal_threshold: float,
      expected_score: float,
  ):
    score = results._normalized_half_bowl(
        relative_width_ratio=relative_width_ratio,
        spend_share=spend_share,
        high_variance_threshold=high_variance_threshold,
        ideal_threshold=ideal_threshold,
    )
    self.assertAlmostEqual(score, expected_score, places=5)

  def test_normalized_half_bowl_default_threshold(self):
    # Default high_variance_threshold is 1.0, ideal_threshold is 0.5.
    # relative_width_ratio=2.0, spend_share=0.5 -> x = 1.0 -> score = 50.0
    score = results._normalized_half_bowl(
        relative_width_ratio=2.0,
        spend_share=0.5,
    )
    self.assertAlmostEqual(score, 50.0, places=5)

  def test_normalized_half_bowl_equal_thresholds(self):
    # Tests epsilon guard when high_variance_threshold <= ideal_threshold
    score = results._normalized_half_bowl(
        relative_width_ratio=2.0,
        spend_share=0.5,
        high_variance_threshold=0.5,
        ideal_threshold=0.5,
    )
    self.assertAlmostEqual(score, 0.0, places=5)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_returns_zero",
          max_abs_correlation=0.0,
          correlation_threshold=0.1,
          expected_score=0.0,
      ),
      dict(
          testcase_name="negative_returns_zero",
          max_abs_correlation=-0.5,
          correlation_threshold=0.1,
          expected_score=0.0,
      ),
      dict(
          testcase_name="at_threshold_returns_50",
          max_abs_correlation=0.1,
          correlation_threshold=0.1,
          expected_score=50.0,
      ),
      dict(
          testcase_name="max_returns_100",
          max_abs_correlation=1.0,
          correlation_threshold=0.1,
          expected_score=100.0,
      ),
      dict(
          testcase_name="greater_than_one_returns_100",
          max_abs_correlation=1.5,
          correlation_threshold=0.1,
          expected_score=100.0,
      ),
      dict(
          testcase_name="zero_threshold_linear",
          max_abs_correlation=0.4,
          correlation_threshold=0.0,
          expected_score=40.0,
      ),
      dict(
          testcase_name="negative_threshold_linear",
          max_abs_correlation=0.4,
          correlation_threshold=-0.1,
          expected_score=40.0,
      ),
      dict(
          testcase_name="one_threshold_linear",
          max_abs_correlation=0.4,
          correlation_threshold=1.0,
          expected_score=40.0,
      ),
      dict(
          testcase_name="greater_than_one_threshold_linear",
          max_abs_correlation=0.4,
          correlation_threshold=1.5,
          expected_score=40.0,
      ),
  )
  def test_potential_bias_score(
      self,
      max_abs_correlation: float,
      correlation_threshold: float,
      expected_score: float,
  ):
    score = results._potential_bias_score(
        max_abs_correlation=max_abs_correlation,
        correlation_threshold=correlation_threshold,
    )
    self.assertAlmostEqual(score, expected_score, places=5)

  def test_potential_bias_score_default_threshold(self):
    score = results._potential_bias_score(max_abs_correlation=0.1)
    self.assertAlmostEqual(score, 50.0, places=5)

  def test_compute_channel_calibration_score(self):
    score = results._compute_channel_calibration_score(
        implausible_roi_score=80.0,
        high_variance_roi_score=60.0,
        potential_bias_score=40.0,
    )
    self.assertAlmostEqual(score, 65.0, places=5)

  def test_channel_calibration_scores_all_calibrated(self):
    summary = _create_test_summary(
        health_score=100.0,
        channel_calibration_status={"ch1": True, "ch2": True},
    )
    self.assertEqual(
        summary.channel_calibration_scores,
        {"ch1": 100.0, "ch2": 100.0},
    )
    self.assertEqual(summary.calibration_score, 100.0)

  def test_channel_calibration_scores_uncalibrated_composite(self):
    summary = _create_test_summary(
        overall_status=results.Status.REVIEW,
        summary_message="Review needed",
        results_list=_mock_check_results(
            high_roi_channels=["ch1"],
            high_variance_channels=["ch1"],
            low_correlation_channels=["ch1"],
            high_roi_mean=20.0,
            high_roi_spend_share=1.0,
            max_abs_correlation=0.1,
        ),
        health_score=50.0,
        channel_calibration_status={"ch1": False, "ch2": True},
    )

    # ch1 scores:
    # implausible_roi: 50.0 (weight 0.5 -> 25.0)
    # high_variance: 50.0 (weight 0.25 -> 12.5)
    # potential_bias: 50.0 (weight 0.25 -> 12.5)
    # ch1 composite = 25.0 + 12.5 + 12.5 = 50.0
    # ch2 composite = 100.0
    scores = summary.channel_calibration_scores
    self.assertAlmostEqual(scores["ch1"], 50.0, places=5)
    self.assertEqual(scores["ch2"], 100.0)
    self.assertAlmostEqual(summary.calibration_score, 75.0, places=5)

  def test_channel_calibration_scores_warning_without_driver_issues(self):
    mock_implausible = results.ImplausibleROICheckResult(
        case=results.ImplausibleROIAggregateCases.PASS,
        channel_results=[
            results.ImplausibleROIChannelResult(
                case=results.ImplausibleROIChannelCases.ROI_PASS,
                channel_name="search",
                spend_share=0.5,
                roi_mean=19.5,
                spend_weighted_roi=9.75,
            ),
        ],
        high_roi_channels=[],
        low_roi_channels=[],
        aggregate_details={},
    )
    mock_high_var = results.HighVarianceCheckResult(
        case=results.HighVarianceAggregateCases.PASS,
        channel_results=[
            results.HighVarianceChannelResult(
                channel_name="search",
                case=results.HighVarianceChannelCases.ROI_PASS,
                spend_share=0.5,
                relative_width_ratio=1.8,
            ),
        ],
        high_variance_channels=[],
    )
    mock_bias = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.PASS,
        channel_results=[
            results.PotentialBiasChannelResult(
                channel_name="search",
                case=results.PotentialBiasChannelCases.ROI_PASS,
                max_abs_correlation=0.11,
            ),
        ],
        low_correlation_channels=[],
        correlation_matrix=xr.DataArray([]),
    )

    summary = _create_test_summary(
        results_list=[mock_implausible, mock_high_var, mock_bias],
        health_score=100.0,
        channel_calibration_status={"search": False},
    )

    scores = summary.channel_calibration_scores
    self.assertAlmostEqual(scores["search"], 63.067, places=3)
    self.assertTrue(summary.has_calibration_warning)
    self.assertEmpty(summary._get_recommended_channels_with_issues())

  def test_channel_calibration_scores_no_controls(self):
    mock_bias = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.NO_CONTROLS,
        channel_results=[],
        low_correlation_channels=[],
        correlation_matrix=xr.DataArray([]),
    )
    summary = _create_test_summary(
        overall_status=results.Status.REVIEW,
        summary_message="No controls",
        results_list=[mock_bias],
        health_score=80.0,
        channel_calibration_status={"ch1": False},
    )
    # ch1 scores:
    # implausible_roi: 100.0 (weight 0.5 -> 50.0)
    # high_variance: 100.0 (weight 0.25 -> 25.0)
    # potential_bias: 0.0 (weight 0.25 -> 0.0)
    # total = 75.0
    scores = summary.channel_calibration_scores
    self.assertAlmostEqual(scores["ch1"], 75.0, places=5)
    self.assertAlmostEqual(summary.calibration_score, 75.0, places=5)

  def test_channel_calibration_scores_empty_status_returns_100(self):
    summary = _create_test_summary(
        health_score=100.0, channel_calibration_status={}
    )
    self.assertEqual(summary.channel_calibration_scores, {})
    self.assertEqual(summary.calibration_score, 100.0)

  @parameterized.named_parameters(
      ("score_lt_threshold_with_driver_issues", {"ch1": 60.0}, True, True),
      ("score_lt_threshold_without_driver_issues", {"ch1": 60.0}, False, True),
      ("score_ge_threshold_with_driver_issues", {"ch1": 80.0}, True, False),
      ("score_ge_threshold_without_driver_issues", {"ch1": 80.0}, False, False),
      (
          "calibrated_channel_with_low_score_ignored",
          {"ch1": 50.0},
          True,
          False,
          {"ch1": True},
      ),
      ("empty_status_with_low_score", {"ch1": 50.0}, False, True, {}),
      ("empty_status_no_channels", {}, False, False, {}),
  )
  def test_has_calibration_warning(
      self,
      channel_scores: dict[str, float],
      has_driver_issue: bool,
      expected_warning: bool,
      channel_calibration_status: dict[str, bool] | None = None,
  ):
    if channel_calibration_status is None:
      channel_calibration_status = {ch: False for ch in channel_scores}
    driver_channels = (
        list(channel_calibration_status.keys()) or ["ch1"]
        if has_driver_issue
        else ()
    )
    summary = _create_test_summary(
        results_list=_mock_check_results(high_roi_channels=driver_channels),
        health_score=85.0,
        channel_calibration_status=channel_calibration_status,
        channel_scores=channel_scores,
    )
    self.assertEqual(summary.has_calibration_warning, expected_warning)

  def test_channels_recommended_for_calibration(self):
    summary = _create_test_summary(
        health_score=85.0,
        calibrated_channel_names=["calibrated_low"],
        channel_scores={
            "calibrated_low": 50.0,
            "uncalibrated_low": 60.0,
            "uncalibrated_high": 80.0,
        },
    )
    self.assertEqual(
        summary.channels_recommended_for_calibration,
        ["uncalibrated_low"],
    )
    self.assertTrue(summary.has_calibration_warning)

  def test_channels_recommended_for_calibration_table_order(self):
    summary = _create_test_summary(
        health_score=85.0,
        calibrated_channel_names=["youtube"],
        channel_scores={
            "youtube": 100.0,
            "search": 55.0,
            "display": 60.0,
            "pmax": 50.0,
        },
    )
    self.assertEqual(
        summary.channels_recommended_for_calibration,
        ["search", "display", "pmax"],
    )

  @parameterized.named_parameters(
      (
          "scores_and_drivers",
          ["pmax", "search"],
          {
              "pmax": [review_constants.HIGH_ROI],
              "search": [review_constants.HIGH_VARIANCE],
          },
          (
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'pmax' and 'search': 'pmax' shows issues with"
              " high ROI and 'search' shows issues with high variance. See"
              " Channel calibration recommendation below for more details."
          ),
      ),
      (
          "no_scores_no_drivers",
          [],
          {},
          results.NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION,
      ),
      (
          "scores_no_drivers",
          ["pmax"],
          {},
          (
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'pmax'. See Channel calibration recommendation"
              " below for more details."
          ),
      ),
      (
          "no_scores_single_driver",
          [],
          {"search": [review_constants.HIGH_VARIANCE]},
          (
              "No channels require calibration. However, 'search' shows issues"
              " with high variance. We recommend reviewing the table and plots"
              " below to check if this channel may be a good candidate for"
              " calibration via an incrementality experiment such as those run"
              " with Meridian GeoX."
          ),
      ),
      (
          "no_scores_multi_channel_drivers",
          [],
          {
              "pmax": [review_constants.HIGH_ROI],
              "search": [review_constants.HIGH_VARIANCE],
          },
          (
              "No channels require calibration. However, 'pmax' shows issues"
              " with high ROI and 'search' shows issues with high variance. We"
              " recommend reviewing the table and plots below to check if these"
              " channels may be good candidates for calibration via an"
              " incrementality experiment such as those run with Meridian"
              " GeoX."
          ),
      ),
      (
          "no_scores_single_channel_multi_drivers",
          [],
          {
              "pmax": [
                  review_constants.HIGH_ROI,
                  review_constants.HIGH_VARIANCE,
              ],
          },
          (
              "No channels require calibration. However, 'pmax' shows issues"
              " with high ROI and high variance. We recommend reviewing the"
              " table and plots below to check if this channel may be a good"
              " candidate for calibration via an incrementality experiment"
              " such as those run with Meridian GeoX."
          ),
      ),
      (
          "scores_and_drivers_order_preserved",
          ["search", "display"],
          {
              "search": [review_constants.HIGH_ROI],
              "pmax": [review_constants.POTENTIAL_BIAS],
          },
          (
              "We recommend incrementality experiments to improve prior"
              " accuracy for 'search' and 'display': 'search' shows issues"
              " with high ROI and 'pmax' shows potential bias. See Channel"
              " calibration recommendation below for more details."
          ),
      ),
  )
  def test_build_calibration_recommendation_text(
      self,
      recommended_channels: list[str],
      driver_issues_by_channel: dict[str, list[str]],
      expected_recommendation: str,
  ):
    # Location (c): channel recommendation
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=recommended_channels,
            driver_issues_by_channel=driver_issues_by_channel,
            location=review_constants.CALIBRATION_TEXT_CHANNEL_RECOMMENDATION,
        ),
        expected_recommendation,
    )
    # Location (b): calibration summary
    expected_summary = (
        expected_recommendation
        if recommended_channels
        else review_constants.NO_CHANNELS_REQUIRE_CALIBRATION
    )
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=recommended_channels,
            driver_issues_by_channel=driver_issues_by_channel,
            location=review_constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY,
        ),
        expected_summary,
    )
    # Location (a): metrics check
    if recommended_channels:
      channels_text = results._format_list_with_and(
          [f"'{c}'" for c in dict.fromkeys(recommended_channels)]
      )
      rec_part = (
          "We recommend incrementality experiments to improve prior accuracy"
          f" for {channels_text}."
      )
    else:
      rec_part = review_constants.NO_CHANNELS_REQUIRE_CALIBRATION
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=recommended_channels,
            driver_issues_by_channel=driver_issues_by_channel,
            location=review_constants.CALIBRATION_TEXT_METRICS_CHECK,
            calibration_score=80.0,
        ),
        f"The overall calibration score is 80.0/100. {rec_part}",
    )

  def test_build_calibration_recommendation_text_defaults_and_dedup(self):
    self.assertEqual(
        results.build_calibration_recommendation_text(),
        results.NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION,
    )
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=["pmax", "pmax", "search"],
            location=review_constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY,
        ),
        "We recommend incrementality experiments to improve prior accuracy for"
        " 'pmax' and 'search'. See Channel calibration recommendation below"
        " for more details.",
    )
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=["search", "pmax", "search"],
            location=review_constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY,
        ),
        "We recommend incrementality experiments to improve prior accuracy for"
        " 'search' and 'pmax'. See Channel calibration recommendation below"
        " for more details.",
    )
    self.assertEqual(
        results.build_calibration_recommendation_text(
            recommended_channels=["search"],
            driver_issues_by_channel={
                "search": [
                    review_constants.HIGH_ROI,
                    review_constants.HIGH_ROI,
                ]
            },
            location=review_constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY,
        ),
        "We recommend incrementality experiments to improve prior accuracy for"
        " 'search': 'search' shows issues with high ROI. See Channel"
        " calibration recommendation below for more details.",
    )


if __name__ == "__main__":
  absltest.main()
