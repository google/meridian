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

"""Implementation of the Model Quality Checks."""

import abc
from typing import Generic, TypeVar

from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
import numpy as np

ConfigType = TypeVar("ConfigType")
ResultType = TypeVar("ResultType", bound=results.CheckResult)


class BaseCheck(abc.ABC, Generic[ConfigType, ResultType]):
  """A generic, abstract base class for a single, runnable quality check."""

  def __init__(self, analyzer: analyzer_module.Analyzer, config: ConfigType):
    self._analyzer = analyzer
    self._config = config

  @abc.abstractmethod
  def run(self) -> ResultType:
    """Executes the check.

    The return type uses the generic ResultType, making it specific for each
    subclass.
    """
    raise NotImplementedError()


class ConvergenceCheck(
    BaseCheck[configs.ConvergenceConfig, results.ConvergenceCheckResult]
):
  """Checks for model convergence."""

  def run(self) -> results.ConvergenceCheckResult:
    rhats = self._analyzer.get_rhat()
    max_rhats = {k: np.nanmax(v) for k, v in rhats.items()}
    max_parameter, max_rhat = max(max_rhats.items(), key=lambda item: item[1])

    details = {
        review_constants.RHAT: max_rhat,
        review_constants.PARAMETER: max_parameter,
        review_constants.CONVERGENCE_THRESHOLD: (
            self._config.convergence_threshold
        ),
    }

    # Case 1: Converged.
    if max_rhat < self._config.convergence_threshold:
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          details=details,
      )

    # Case 2: Not fully converged, but potentially acceptable.
    elif (
        self._config.convergence_threshold
        <= max_rhat
        < self._config.not_fully_convergence_threshold
    ):
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
          details=details,
      )

    # Case 3: Not converged and unacceptable.
    else:  # max_rhat >= divergence_threshold
      return results.ConvergenceCheckResult(
          case=results.ConvergenceCases.NOT_CONVERGED,
          details=details,
      )
