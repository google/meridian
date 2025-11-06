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

"""Configurations for the Model Quality Checks."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class BaseConfig:
  """Base class for all check configurations."""


@dataclasses.dataclass(frozen=True)
class ConvergenceConfig(BaseConfig):
  """Configuration for the Convergence Check.

  Attributes:
    convergence_threshold: The threshold for the R-hat statistic to determine if
      the model has converged. R-hat values below this are considered converged.
    not_fully_convergence_threshold: The threshold for the R-hat statistic to
      determine if the model is not fully converged but potentially acceptable.
      R-hat values between `convergence_threshold` and this value are considered
      not fully converged. R-hat values above this threshold are considered not
      converged.
  """

  convergence_threshold: float = 1.2
  not_fully_convergence_threshold: float = 10.0


@dataclasses.dataclass(frozen=True)
class ROIConsistencyConfig(BaseConfig):
  """Configuration for the ROI Consistency Check.

  This check verifies if the posterior median of the ROI falls within a
  reasonable range of the prior distribution.

  Attributes:
    prior_lower_quantile: The lower quantile of the ROI prior distribution to
      define the lower bound of the reasonable range.
    prior_upper_quantile: The upper quantile of the ROI prior distribution to
      define the upper bound of the reasonable range.
  """

  prior_lower_quantile: float = 0.01
  prior_upper_quantile: float = 0.99


@dataclasses.dataclass(frozen=True)
class BaselineConfig(BaseConfig):
  """Configuration for the Baseline Check.

  This check warns if there is a high probability of a negative baseline.

  Attributes:
    negative_baseline_prob_review_threshold: Probability threshold for a
      review. If the probability of a negative baseline is above this value, a
      review is issued.
    negative_baseline_prob_fail_threshold: Probability threshold for a failure.
      If the probability of a negative baseline is above this value, the check
      fails.
  """

  negative_baseline_prob_review_threshold: float = 0.2
  negative_baseline_prob_fail_threshold: float = 0.8


@dataclasses.dataclass(frozen=True)
class BayesianPPPConfig(BaseConfig):
  """Configuration for the Bayesian Posterior Predictive P-value Check.

  Attributes:
    ppp_threshold: P-value threshold for posterior predictive check.
  """

  ppp_threshold: float = 0.05


@dataclasses.dataclass(frozen=True)
class GoodnessOfFitConfig(BaseConfig):
  """An empty config for the Goodness of Fit Check."""


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftConfig(BaseConfig):
  """Configuration for the Prior-Posterior Shift Check.

  Attributes:
    n_bootstraps: Number of bootstrap samples to use for calculating posterior
      statistics.
    alpha: Significance level for detecting a shift between prior and posterior
      distributions.
    seed: Random seed for reproducibility of bootstrap sampling.
  """

  n_bootstraps: int = 1000
  alpha: float = 0.05
  seed: int = 42
