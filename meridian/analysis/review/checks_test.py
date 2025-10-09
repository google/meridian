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

"""Tests for checks.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import results
import numpy as np


class ConvergenceCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.analyzer = mock.create_autospec(
        analyzer_module.Analyzer, instance=True
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="not_converged_high_rhat",
          rhat_mock_value=11.0,
          expected_case=results.ConvergenceCases.NOT_CONVERGED,
      ),
      dict(
          testcase_name="needs_review_medium_rhat",
          rhat_mock_value=9.0,
          expected_case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
      ),
      dict(
          testcase_name="converged_low_rhat",
          rhat_mock_value=1.1,
          expected_case=results.ConvergenceCases.CONVERGED,
      ),
  )
  def test_convergence_check(
      self, rhat_mock_value: float, expected_case: results.ConvergenceCases
  ):
    self.analyzer.get_rhat.return_value = {
        "mock_var": np.array([rhat_mock_value])
    }

    config = configs.ConvergenceConfig()
    convergence_check = checks.ConvergenceCheck(
        analyzer=self.analyzer,
        config=config,
    )
    result = convergence_check.run()
    self.assertEqual(result.case, expected_case)

    reporting = result.reporting

    if result.case == results.ConvergenceCases.CONVERGED:
      self.assertIsNone(result.recommendation)
      self.assertEqual(
          reporting,
          "The model has likely converged, as all parameters have R-hat values"
          " < 1.2.",
      )
    elif result.case == results.ConvergenceCases.NOT_FULLY_CONVERGED:
      self.assertEqual(
          result.recommendation, results._CONVERGENCE_RECOMMENDATION
      )
      self.assertEqual(
          reporting,
          "The model is not converged, and the max rhat is"
          f" {rhat_mock_value} (for parameter 'mock_var'). This might be"
          " acceptable.",
      )
    elif result.case == results.ConvergenceCases.NOT_CONVERGED:
      self.assertEqual(
          result.recommendation, results._CONVERGENCE_RECOMMENDATION
      )
      self.assertEqual(
          reporting,
          f"The model is not converged with max rhat = {rhat_mock_value} (for"
          " parameter 'mock_var').",
      )


if __name__ == "__main__":
  absltest.main()
