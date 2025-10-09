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

"""Data structures for the Model Quality Checks results."""

import dataclasses
import enum
from typing import Any


class Status(enum.Enum):
  PASS = enum.auto()
  WARNING = enum.auto()
  FAIL = enum.auto()


@dataclasses.dataclass(frozen=True)
class BaseCase:
  """Base class for all check-specific cases."""

  status: Status
  message_template: str
  action_items: str | None = None


CONVERGENCE_ACTION_ITEMS = (
    "To get your chains to converge, try the following recommendations: first,"
    " increase the number of MCMC iterations, as the model may simply need"
    " more time to explore the posterior distribution and reach a stable"
    " state. If the convergence issue persists after increasing iterations,"
    " you should then investigate potential model misspecification by carefully"
    " re-examining your priors and checking for high multicollinearity between"
    " your predictors. For more information, see Getting MCMC Convergence."
)


class ConvergenceCases(BaseCase, enum.Enum):
  """Cases for the Convergence Check."""

  CONVERGED = (
      Status.PASS,
      (
          "The model has likely converged, as all parameters have R-hat values"
          " < {convergence_threshold}."
      ),
      None,
  )
  NEEDS_REVIEW = (
      Status.FAIL,
      (
          "The model is not converged, and the max rhat is {rhat} (for"
          " parameter '{parameter}'). This might be acceptable."
      ),
      CONVERGENCE_ACTION_ITEMS,
  )
  NOT_CONVERGED = (
      Status.FAIL,
      (
          "The model is not converged with max rhat = {rhat} (for parameter"
          " '{parameter}')."
      ),
      CONVERGENCE_ACTION_ITEMS,
  )


# TODO: Add cases for the other checks.


@dataclasses.dataclass(frozen=True)
class CheckResult:
  """The base class for the immutable result of a single quality check."""

  case: BaseCase
  details: dict[str, Any] | None = None

  @property
  def reporting(self) -> str:
    """Returns the formatted reporting string."""
    return self.case.message_template.format(**self.details)

  @property
  def action_items(self) -> str | None:
    """Returns the action items for this result's case."""
    return self.case.action_items


@dataclasses.dataclass(frozen=True)
class ConvergenceCheckResult(CheckResult):
  case: ConvergenceCases


# TODO: Add results for the other checks.
