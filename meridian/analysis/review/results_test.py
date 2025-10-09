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

"""Tests for results.py."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis.review import results


class ConvergenceCheckResultTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_details",
          details={},
          details_str="{}",
      ),
      dict(
          testcase_name="no_rhat",
          details={"parameter": "mock_var", "convergence_threshold": 2.0},
          details_str="{'parameter': 'mock_var', 'convergence_threshold': 2.0}",
      ),
      dict(
          testcase_name="no_parameter",
          details={"rhat": 11.0, "convergence_threshold": 2.0},
          details_str="{'rhat': 11.0, 'convergence_threshold': 2.0}",
      ),
      dict(
          testcase_name="no_convergence_threshold",
          details={"rhat": 11.0, "parameter": "mock_var"},
          details_str="{'rhat': 11.0, 'parameter': 'mock_var'}",
      ),
  )
  def test_convergence_check_result_raises_error(
      self,
      details: dict[str, Any],
      details_str: str,
  ):
    expected_error_message = (
        "The message template 'The model is not converged with max rhat ="
        " {rhat} (for parameter '{parameter}').' is missing required"
        " formatting arguments: rhat, parameter, convergence_threshold."
        f" Details: {details_str}."
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      _ = results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_CONVERGED,
          details=details,
      )

  def test_convergence_check_result_converged(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.CONVERGED,
        details={
            "rhat": 1.0,
            "parameter": "mock_var",
            "convergence_threshold": 2.0,
        },
    )
    self.assertEqual(result.case, results.ConvergenceCases.CONVERGED)
    self.assertEqual(result.case.status, results.Status.PASS)
    self.assertEqual(
        result.reporting,
        "The model has likely converged, as all parameters have R-hat values"
        " < 2.0.",
    )
    self.assertIsNone(result.recommendation)

  def test_convergence_check_result_needs_review(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
        details={
            "rhat": 3.0,
            "parameter": "mock_var",
            "convergence_threshold": 2.0,
        },
    )
    self.assertEqual(result.case, results.ConvergenceCases.NOT_FULLY_CONVERGED)
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.reporting,
        "The model is not converged, and the max rhat is 3.0 (for parameter"
        " 'mock_var'). This might be acceptable.",
    )
    self.assertEqual(result.recommendation, results._CONVERGENCE_RECOMMENDATION)

  def test_convergence_check_result_not_converged(self):
    result = results.ConvergenceCheckResult(
        case=results.ConvergenceCases.NOT_CONVERGED,
        details={
            "rhat": 11.0,
            "parameter": "mock_var",
            "convergence_threshold": 2.0,
        },
    )
    self.assertEqual(result.case, results.ConvergenceCases.NOT_CONVERGED)
    self.assertEqual(result.case.status, results.Status.FAIL)
    self.assertEqual(
        result.reporting,
        "The model is not converged with max rhat = 11.0 (for parameter"
        " 'mock_var').",
    )
    self.assertEqual(result.recommendation, results._CONVERGENCE_RECOMMENDATION)


if __name__ == "__main__":
  absltest.main()
