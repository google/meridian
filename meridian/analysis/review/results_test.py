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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
import numpy as np
import xarray as xr


class ConvergenceCheckResultTest(parameterized.TestCase):

  def test_convergence_check_result_converged(self):
    config = configs.ConvergenceConfig(convergence_threshold=2.0)
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        config=config,
        max_rhat=1.0,
        max_parameter="mock_var",
    )
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
        max_rhat=3.0,
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
        max_rhat=11.0,
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


class ReviewSummaryTest(parameterized.TestCase):

  def test_review_summary_repr(self):
    mock_result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        config=configs.ConvergenceConfig(),
        max_rhat=1.0,
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

  def test_high_variance_check_result_pass(self):
    result = results.HighVarianceCheckResult(
        case=results.HighVarianceAggregateCases.PASS,
        channel_results=[],
        high_variance_channels=[],
    )
    self.assertEqual(result.case.status, results.Status.PASS)
    self.assertEqual(
        result.recommendation,
        "All channels have acceptable ROI variance.",
    )

  def test_high_variance_check_result_review(self):
    result = results.HighVarianceCheckResult(
        case=results.HighVarianceAggregateCases.REVIEW,
        channel_results=[],
        high_variance_channels=["channel1", "channel2"],
    )
    self.assertEqual(result.case.status, results.Status.REVIEW)
    self.assertEqual(
        result.recommendation,
        "We've detected channel(s) `channel1`, `channel2` with highly uncertain"
        " ROI estimates (wide posterior intervals). "
        f"{review_constants.HIGH_VARIANCE_ROI_RECOMMENDATION}",
    )


class PotentialBiasCheckResultTest(parameterized.TestCase):

  def test_potential_bias_check_result_pass(self):
    corr_matrix = xr.DataArray(
        np.array([[[0.5]]]),
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
        case=results.PotentialBiasAggregateCases.PASS,
        channel_results=[],
        low_correlation_channels=[],
        correlation_matrix=corr_matrix,
    )
    self.assertEqual(result.case.status, results.Status.PASS)
    self.assertEqual(
        result.recommendation,
        "All channels have sufficient correlation with control variables.",
    )
    xr.testing.assert_equal(
        result.details[review_constants.CORRELATION_MATRIX], corr_matrix
    )

  def test_potential_bias_check_result_review(self):
    corr_matrix = xr.DataArray(
        np.array([[[0.0]]]),
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
        case=results.PotentialBiasAggregateCases.REVIEW,
        channel_results=[],
        low_correlation_channels=["channel1"],
        correlation_matrix=corr_matrix,
    )
    self.assertEqual(result.case.status, results.Status.REVIEW)
    self.assertEqual(
        result.recommendation,
        "We've detected channel(s) `channel1` with very low correlation with"
        " all included control variables."
        f" {review_constants.POTENTIAL_BIAS_RECOMMENDATION}",
    )
    xr.testing.assert_equal(
        result.details[review_constants.CORRELATION_MATRIX], corr_matrix
    )


if __name__ == "__main__":
  absltest.main()
