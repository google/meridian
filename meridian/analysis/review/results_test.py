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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results


class ConvergenceCheckResultTest(parameterized.TestCase):

  def test_convergence_check_result_raises_error(self):
    expected_error_message = (
        "The message template 'The model has likely converged, as all"
        " parameters have R-hat values < {convergence_threshold}'. is missing"
        " required formatting arguments: convergence_threshold."
        " Details: {}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          details={},
      )

  def test_convergence_check_result_converged(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        details={
            review_constants.RHAT: 1.0,
            review_constants.PARAMETER: "mock_var",
            review_constants.CONVERGENCE_THRESHOLD: 2.0,
        },
    )
    self.assertEqual(result.case.status, results.Status.PASS)
    self.assertEqual(
        result.recommendation,
        "The model has likely converged, as all parameters have R-hat values"
        " < 2.0.",
    )

  def test_convergence_check_result_needs_review(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
        details={
            review_constants.RHAT: 3.0,
            review_constants.PARAMETER: "mock_var",
            review_constants.CONVERGENCE_THRESHOLD: 2.0,
        },
    )
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.recommendation,
        "The model hasn't fully converged, and the `max_r_hat` for parameter"
        " `mock_var` is 3.00. "
        + results.NOT_FULLY_CONVERGED_RECOMMENDATION,
    )

  def test_convergence_check_result_not_converged(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_CONVERGED,
        details={
            review_constants.RHAT: 11.0,
            review_constants.PARAMETER: "mock_var",
            review_constants.CONVERGENCE_THRESHOLD: 2.0,
        },
    )
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.recommendation,
        "The model hasn't converged, and the `max_r_hat` for parameter"
        " `mock_var` is 11.00. "
        + results.NOT_CONVERGED_RECOMMENDATION,
    )


class BaselineCheckResultTest(parameterized.TestCase):

  def test_baseline_check_result_raises_error_with_fail_case(self):
    expected_error_message = (
        "The message template is missing required formatting arguments:"
        " negative_baseline_prob, negative_baseline_prob_fail_threshold,"
        " negative_baseline_prob_review_threshold. Details:"
        " {'mock': 1}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.BaselineCheckResult(
          case=results.BaselineCases.FAIL,
          details={"mock": 1},
      )

  def test_baseline_check_result_raises_error_with_review_case(self):
    expected_error_message = (
        "The message template is missing required formatting arguments:"
        " negative_baseline_prob, negative_baseline_prob_fail_threshold,"
        " negative_baseline_prob_review_threshold. Details:"
        " {'mock': 1}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.BaselineCheckResult(
          case=results.BaselineCases.REVIEW,
          details={"mock": 1},
      )

  def test_baseline_check_result_pass(self):
    result = results.BaselineCheckResult(
        case=results.BaselineCases.PASS,
        details={
            review_constants.NEGATIVE_BASELINE_PROB: 0.01,
            review_constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD: 0.2,
            review_constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD: 0.1,
        },
    )
    self.assertEqual(
        result.recommendation,
        "The posterior probability that the baseline is negative is 0.01. "
        + results._BASELINE_PASS_RECOMMENDATION,
    )

  def test_baseline_check_result_review(self):
    result = results.BaselineCheckResult(
        case=results.BaselineCases.REVIEW,
        details={
            review_constants.NEGATIVE_BASELINE_PROB: 0.15,
            review_constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD: 0.2,
            review_constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD: 0.1,
        },
    )
    self.assertEqual(
        result.recommendation,
        "The posterior probability that the baseline is negative is 0.15. "
        + results._BASELINE_REVIEW_RECOMMENDATION,
    )

  def test_baseline_check_result_fail(self):
    result = results.BaselineCheckResult(
        case=results.BaselineCases.FAIL,
        details={
            review_constants.NEGATIVE_BASELINE_PROB: 0.25,
            review_constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD: 0.2,
            review_constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD: 0.1,
        },
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
        details=details,
        channel_results=[],
    )
    self.assertEqual(result.recommendation, expected_recommendation)


class BayesianPPPCheckResultTest(parameterized.TestCase):

  def test_bayesian_ppp_check_result_raises_error(self):
    expected_error_message = (
        "The message template is missing required formatting arguments:"
        " bayesian_ppp. Details: {'svet': 1}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.BayesianPPPCheckResult(
          case=results.BayesianPPPCases.PASS,
          details={"svet": 1},
      )

  def test_bayesian_ppp_check_result_pass(self):
    result = results.BayesianPPPCheckResult(
        case=results.BayesianPPPCases.PASS,
        details={
            review_constants.BAYESIAN_PPP: 0.06,
        },
    )
    self.assertEqual(
        result.recommendation,
        "The Bayesian posterior predictive p-value is 0.06. "
        + results._BAYESIAN_PPP_PASS_RECOMMENDATION,
    )

  def test_bayesian_ppp_check_result_fail(self):
    result = results.BayesianPPPCheckResult(
        case=results.BayesianPPPCases.FAIL,
        details={
            review_constants.BAYESIAN_PPP: 0.04,
        },
    )
    self.assertEqual(
        result.recommendation,
        "The Bayesian posterior predictive p-value is 0.04. "
        + results._BAYESIAN_PPP_FAIL_RECOMMENDATION,
    )


class GoodnessOfFitCheckResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no_r_squared",
          details={
              review_constants.MAPE: 0.1,
              review_constants.WMAPE: 0.1,
          },
          details_str="{'mape': 0.1, 'wmape': 0.1}",
      ),
      dict(
          testcase_name="no_mape",
          details={
              review_constants.R_SQUARED: 0.1,
              review_constants.WMAPE: 0.1,
          },
          details_str="{'r_squared': 0.1, 'wmape': 0.1}",
      ),
      dict(
          testcase_name="no_wmape",
          details={
              review_constants.R_SQUARED: 0.1,
              review_constants.MAPE: 0.1,
          },
          details_str="{'r_squared': 0.1, 'mape': 0.1}",
      ),
  )
  def test_goodness_of_fit_check_result_raises_error(
      self,
      details: dict[str, Any],
      details_str: str,
  ):
    expected_error_message = (
        "The message template is missing required formatting arguments:"
        f" r_squared, mape, wmape. Details: {details_str}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.GoodnessOfFitCheckResult(
          case=results.GoodnessOfFitCases.PASS,
          details=details,
      )

  def test_goodness_of_fit_check_result_pass(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.PASS,
        details={
            review_constants.R_SQUARED: 0.5,
            review_constants.MAPE: 0.1,
            review_constants.WMAPE: 0.2,
        },
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = 0.5000, MAPE = 0.1000, and wMAPE = 0.2000. "
        + results._GOODNESS_OF_FIT_PASS_RECOMMENDATION,
    )

  def test_goodness_of_fit_check_result_pass_holdout(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.PASS,
        details={
            f"{review_constants.R_SQUARED}_all": 0.5,
            f"{review_constants.MAPE}_all": 0.1,
            f"{review_constants.WMAPE}_all": 0.2,
            f"{review_constants.R_SQUARED}_train": 0.6,
            f"{review_constants.MAPE}_train": 0.09,
            f"{review_constants.WMAPE}_train": 0.19,
            f"{review_constants.R_SQUARED}_test": 0.4,
            f"{review_constants.MAPE}_test": 0.11,
            f"{review_constants.WMAPE}_test": 0.21,
        },
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
        details={
            review_constants.R_SQUARED: -0.5,
            review_constants.MAPE: 0.1,
            review_constants.WMAPE: 0.2,
        },
    )
    self.assertEqual(
        result.recommendation,
        "R-squared = -0.5000, MAPE = 0.1000, and wMAPE = 0.2000. "
        + results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION,
    )

  def test_goodness_of_fit_check_result_review_holdout(self):
    result = results.GoodnessOfFitCheckResult(
        case=results.GoodnessOfFitCases.REVIEW,
        details={
            f"{review_constants.R_SQUARED}_all": -0.5,
            f"{review_constants.MAPE}_all": 0.1,
            f"{review_constants.WMAPE}_all": 0.2,
            f"{review_constants.R_SQUARED}_train": 0.6,
            f"{review_constants.MAPE}_train": 0.09,
            f"{review_constants.WMAPE}_train": 0.19,
            f"{review_constants.R_SQUARED}_test": 0.4,
            f"{review_constants.MAPE}_test": 0.11,
            f"{review_constants.WMAPE}_test": 0.21,
        },
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
          details={},
          expected_recommendation=(
              "The model has successfully learned from the data. This is a"
              " positive sign that your data was informative."
          ),
      ),
      dict(
          testcase_name="review",
          case=results.PriorPosteriorShiftAggregateCases.REVIEW,
          details={"channels_str": "'channel1', 'channel2'"},
          expected_recommendation=(
              "We've detected channel(s) 'channel1', 'channel2' where the"
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
      details: dict[str, Any],
      expected_recommendation: str | None,
  ):
    result = results.PriorPosteriorShiftCheckResult(
        case=case,
        details=details,
        channel_results=[],
    )
    self.assertEqual(result.recommendation, expected_recommendation)


if __name__ == "__main__":
  absltest.main()
