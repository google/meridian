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

"""Serialization and deserialization of ReviewSummary objects."""

from meridian.analysis.review import results as review_results
from mmm.v1.model.meridian.review import results_pb2


def save_review_summary(
    review_summary: review_results.ReviewSummary,
) -> results_pb2.ReviewSummaryResults:
  """Converts a ReviewSummary object to a ReviewSummaryResults proto."""
  # TODO: Implement serialization logic.
  raise NotImplementedError("save_review_summary is not yet implemented.")


def load_review_summary(
    review_summary_proto: results_pb2.ReviewSummaryResults,
) -> review_results.ReviewSummary:
  """Converts a ReviewSummaryResults proto to a ReviewSummary object."""
  # TODO: Implement deserialization logic.
  raise NotImplementedError("load_review_summary is not yet implemented.")
