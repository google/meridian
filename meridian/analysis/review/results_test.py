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
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results


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
        + results.NOT_FULLY_CONVERGED_RECOMMENDATION,
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
        + results.NOT_CONVERGED_RECOMMENDATION,
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
        + results._BASELINE_PASS_RECOMMENDATION,
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
        + results._BASELINE_REVIEW_RECOMMENDATION,
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
        + results._BASELINE_FAIL_RECOMMENDATION,
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
              "msg1msg2msg3 " + results._ROI_CONSISTENCY_RECOMMENDATION
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
        + results._BAYESIAN_PPP_PASS_RECOMMENDATION,
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
        + results._BAYESIAN_PPP_FAIL_RECOMMENDATION,
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
        + results._GOODNESS_OF_FIT_PASS_RECOMMENDATION,
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
        "R-squared = 0.5000 (All), 0.6000 (Train), 0.4000 (Test); MAPE = 0.1000"
        " (All), 0.0900 (Train), 0.1100 (Test); wMAPE = 0.2000 (All), 0.1900"
        " (Train), 0.2100 (Test). "
        + results._GOODNESS_OF_FIT_PASS_RECOMMENDATION,
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
        + results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION,
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
        " 0.1900 (Train), 0.2100 (Test). "
        + results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION,
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
              " prior. This suggests the data signal for these channels was not"
              " strong enough to update the model's beliefs."
              " "
              + results._PPS_REVIEW_RECOMMENDATION
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


if __name__ == "__main__":
  absltest.main()
