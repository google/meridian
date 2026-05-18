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

"""Tests for health_summary serialization and deserialization (Convergence Only)."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis.review import configs
from meridian.analysis.review import results
from mmm.v1.model.meridian.review import results_pb2 as results_pb
from meridian.schema.serde import health_summary as health_summary_serde


def _make_convergence_check_result(
    case: results.ConvergenceCases,
    rhat: float,
    max_parameter: str,
) -> results.ConvergenceCheckResult:
  """Helper to create ConvergenceCheckResult."""
  return results.ConvergenceCheckResult(
      case=case,
      config=configs.ConvergenceConfig(
          convergence_threshold=1.1,
          not_fully_convergence_threshold=9.5,
      ),
      max_rhat=rhat,
      max_parameter=max_parameter,
  )


class HealthSummarySerdeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serde = health_summary_serde.ReviewSummarySerde()

  @parameterized.named_parameters(
      dict(
          testcase_name="convergence_success",
          review_summary=results.ReviewSummary(
              overall_status=results.Status.PASS,
              summary_message="Convergence check passed.",
              health_score=100.0,
              results=[
                  _make_convergence_check_result(
                      results.ConvergenceCases.CONVERGED, 1.05, "beta_m"
                  ),
              ],
          ),
      ),
      dict(
          testcase_name="convergence_failure_subset",
          review_summary=results.ReviewSummary(
              overall_status=results.Status.FAIL,
              summary_message="Convergence check failed early.",
              health_score=0.0,
              results=[
                  _make_convergence_check_result(
                      results.ConvergenceCases.NOT_CONVERGED, 12.5, "beta_m"
                  ),
              ],
          ),
      ),
      dict(
          testcase_name="convergence_not_fully_converged",
          review_summary=results.ReviewSummary(
              overall_status=results.Status.FAIL,
              summary_message="Convergence check not fully converged.",
              health_score=50.0,
              results=[
                  _make_convergence_check_result(
                      results.ConvergenceCases.NOT_FULLY_CONVERGED,
                      1.08,
                      "beta_m",
                  ),
              ],
          ),
      ),
  )
  def test_serialize_deserialize_roundtrip(
      self, review_summary: results.ReviewSummary
  ):
    proto = self.serde.serialize(review_summary)
    self.assertIsInstance(proto, results_pb.ReviewSummaryResults)

    deserialized = self.serde.deserialize(proto)
    self.assertEqual(deserialized.overall_status, review_summary.overall_status)
    self.assertEqual(
        deserialized.summary_message, review_summary.summary_message
    )
    self.assertEqual(deserialized.health_score, review_summary.health_score)
    self.assertEqual(len(deserialized.results), len(review_summary.results))

    for r_act, r_exp in zip(deserialized.results, review_summary.results):
      self.assertEqual(type(r_act), type(r_exp))
      self.assertEqual(r_act.case, r_exp.case)
      self.assertEqual(r_act.details, r_exp.details)

      if isinstance(r_act, results.ConvergenceCheckResult):
        self.assertEqual(r_act.max_rhat, r_exp.max_rhat)
        self.assertEqual(r_act.max_parameter, r_exp.max_parameter)
        self.assertEqual(
            r_act.config.convergence_threshold,
            r_exp.config.convergence_threshold,
        )
        self.assertEqual(
            r_act.config.not_fully_convergence_threshold,
            r_exp.config.not_fully_convergence_threshold,
        )


if __name__ == "__main__":
  absltest.main()
