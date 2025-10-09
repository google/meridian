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
from meridian.analysis.review import constants


@enum.unique
class Status(enum.Enum):
  PASS = enum.auto()
  WARNING = enum.auto()
  FAIL = enum.auto()


@dataclasses.dataclass(frozen=True)
class BaseCase:
  """Base class for all check-specific cases."""

  status: Status
  message_template: str
  recommendation: str | None = None


_CONVERGENCE_RECOMMENDATION = (
    "To get your chains to converge, try the following recommendations: first,"
    " increase the number of MCMC iterations, as the model may simply need"
    " more time to explore the posterior distribution and reach a stable"
    " state. If the convergence issue persists after increasing iterations,"
    " you should then investigate potential model misspecification by carefully"
    " re-examining your priors and checking for high multicollinearity between"
    " your predictors. For more information, see"
    " https://developers.google.com/meridian/docs/post-modeling/model-debugging#getting-mcmc-convergence."
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
  NOT_FULLY_CONVERGED = (
      Status.FAIL,
      (
          "The model is not converged, and the max rhat is {rhat} (for"
          " parameter '{parameter}'). This might be acceptable."
      ),
      _CONVERGENCE_RECOMMENDATION,
  )
  NOT_CONVERGED = (
      Status.FAIL,
      (
          "The model is not converged with max rhat = {rhat} (for parameter"
          " '{parameter}')."
      ),
      _CONVERGENCE_RECOMMENDATION,
  )


# TODO: Add cases for the other checks.


@dataclasses.dataclass(frozen=True)
class CheckResult:
  """The base class for the immutable result of a single quality check."""

  case: BaseCase
  details: dict[str, Any]

  @property
  def reporting(self) -> str:
    """Returns the formatted reporting string."""
    return self.case.message_template.format(**self.details)

  @property
  def recommendation(self) -> str | None:
    """Returns the recommendation for this result's case, if any."""
    return self.case.recommendation


@dataclasses.dataclass(frozen=True)
class ConvergenceCheckResult(CheckResult):
  """The immutable result of the Convergence Check."""

  case: ConvergenceCases

  def __post_init__(self):
    if any(
        key not in self.details
        for key in (
            constants.RHAT,
            constants.PARAMETER,
            constants.CONVERGENCE_THRESHOLD,
        )
    ):
      raise ValueError(
          "The message template 'The model is not converged with max rhat ="
          " {rhat} (for parameter '{parameter}').' is missing required"
          " formatting arguments: rhat, parameter, convergence_threshold."
          f" Details: {self.details}."
      )


# TODO: Add results for the other checks.
