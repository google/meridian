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

"""Serialization and deserialization of `ReviewSummary` objects (Convergence Only)."""

from __future__ import annotations

from typing import Any

from meridian.analysis.review import configs
from meridian.analysis.review import results
from mmm.v1.model.meridian.review import configs_pb2 as configs_pb
from mmm.v1.model.meridian.review import results_pb2 as results_pb
from meridian.schema.serde import serde

# Proto field names for ReviewSummaryResults
_CONVERGENCE_CHECK_RESULT = "convergence_check_result"


class ReviewSummarySerde(
    serde.Serde[results_pb.ReviewSummaryResults, results.ReviewSummary]
):
  """Serializes and deserializes a `ReviewSummary` object (Convergence Only)."""

  def serialize(
      self, obj: results.ReviewSummary
  ) -> results_pb.ReviewSummaryResults:
    """Serializes a `ReviewSummary` object into a `ReviewSummaryResults` proto."""
    proto = results_pb.ReviewSummaryResults(
        overall_status=self._serialize_status(obj.overall_status),
        summary_message=obj.summary_message,
        health_score=obj.health_score,
    )

    for result in obj.results:
      if isinstance(result, results.ConvergenceCheckResult):
        proto.convergence_check_result.CopyFrom(
            self._serialize_convergence(result)
        )
      else:
        raise ValueError(
            f"Unsupported check result type in CL1: {type(result)}"
        )

    return proto

  def deserialize(
      self,
      serialized: results_pb.ReviewSummaryResults,
      serialized_version: str = "",
  ) -> results.ReviewSummary:
    """Deserializes a `ReviewSummaryResults` proto into a `ReviewSummary` object."""
    res_list = []

    if serialized.HasField(_CONVERGENCE_CHECK_RESULT):
      res_list.append(
          self._deserialize_convergence(serialized.convergence_check_result)
      )

    return results.ReviewSummary(
        overall_status=self._deserialize_status(serialized.overall_status),
        summary_message=serialized.summary_message,
        results=res_list,
        health_score=serialized.health_score,
    )

  # Status mapping
  def _serialize_status(self, status: results.Status) -> Any:
    mapping = {
        results.Status.PASS: results_pb.HealthStatus.HEALTH_STATUS_PASS,
        results.Status.REVIEW: results_pb.HealthStatus.HEALTH_STATUS_REVIEW,
        results.Status.FAIL: results_pb.HealthStatus.HEALTH_STATUS_FAIL,
    }
    return mapping.get(
        status, results_pb.HealthStatus.HEALTH_STATUS_UNSPECIFIED
    )

  def _deserialize_status(self, status: Any) -> results.Status:
    mapping = {
        results_pb.HealthStatus.HEALTH_STATUS_PASS: results.Status.PASS,
        results_pb.HealthStatus.HEALTH_STATUS_REVIEW: results.Status.REVIEW,
        results_pb.HealthStatus.HEALTH_STATUS_FAIL: results.Status.FAIL,
    }
    status_val = mapping.get(status)
    if status_val is None:
      raise ValueError(f"Unsupported or unspecified health status: {status}")
    return status_val

  # Convergence Check Result
  def _serialize_convergence(
      self, result: results.ConvergenceCheckResult
  ) -> results_pb.ConvergenceCheckResult:
    case_mapping = {
        results.ConvergenceCases.CONVERGED: (
            results_pb.CONVERGENCE_CASE_PASS_CONVERGED
        ),
        results.ConvergenceCases.NOT_FULLY_CONVERGED: (
            results_pb.CONVERGENCE_CASE_FAIL_NOT_FULLY_CONVERGED
        ),
        results.ConvergenceCases.NOT_CONVERGED: (
            results_pb.CONVERGENCE_CASE_FAIL_NOT_CONVERGED
        ),
    }
    return results_pb.ConvergenceCheckResult(
        max_r_hat=result.max_rhat,
        max_parameter=result.max_parameter,
        convergence_case=case_mapping[result.case],
        convergence_config=configs_pb.ConvergenceConfig(
            convergence_threshold=result.config.convergence_threshold,
            not_fully_converged_threshold=(
                result.config.not_fully_convergence_threshold
            ),
        ),
    )

  def _deserialize_convergence(
      self, proto: results_pb.ConvergenceCheckResult
  ) -> results.ConvergenceCheckResult:
    case_mapping = {
        results_pb.CONVERGENCE_CASE_PASS_CONVERGED: (
            results.ConvergenceCases.CONVERGED
        ),
        results_pb.CONVERGENCE_CASE_FAIL_NOT_FULLY_CONVERGED: (
            results.ConvergenceCases.NOT_FULLY_CONVERGED
        ),
        results_pb.CONVERGENCE_CASE_FAIL_NOT_CONVERGED: (
            results.ConvergenceCases.NOT_CONVERGED
        ),
    }
    case_val = case_mapping.get(proto.convergence_case)
    if case_val is None:
      raise ValueError(
          "Unsupported or unspecified convergence case: "
          f"{proto.convergence_case}"
      )
    return results.ConvergenceCheckResult(
        case=case_val,
        config=configs.ConvergenceConfig(
            convergence_threshold=(
                proto.convergence_config.convergence_threshold
            ),
            not_fully_convergence_threshold=(
                proto.convergence_config.not_fully_converged_threshold
            ),
        ),
        max_rhat=proto.max_r_hat,
        max_parameter=proto.max_parameter,
    )
