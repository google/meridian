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

"""Configurations for the Model Quality Checks."""

import dataclasses

from meridian.analysis.review import constants as review_constants


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
  # TODO: Rename to not_fully_converged_threshold.
  not_fully_convergence_threshold: float = 10.0


@dataclasses.dataclass(frozen=True)
class ChannelCheckConfig(BaseConfig):
  """Base configuration class for channel-level health checks.

  If both `failing_channels_threshold` and `failing_channels_ratio_threshold`
  are set, the check passes if EITHER condition is satisfied (or both).

  Attributes:
    failing_channels_threshold: Absolute threshold for the number of failing
      channels allowed before triggering a review status. If None, absolute
      count tolerance is not applied. If both failing_channels_threshold and
      failing_channels_ratio_threshold are specified, the check passes if EITHER
      condition is satisfied (or both).
    failing_channels_ratio_threshold: The maximum ratio (between 0.0 and 1.0) of
      failing channels allowed before triggering a review status. If None, ratio
      tolerance is not applied. If both failing_channels_threshold and
      failing_channels_ratio_threshold are specified, the check passes if EITHER
      condition is satisfied (or both).
  """

  failing_channels_threshold: int | None = None
  failing_channels_ratio_threshold: float | None = None

  def __post_init__(self):
    if (
        self.failing_channels_threshold is not None
        and self.failing_channels_threshold < 0
    ):
      raise ValueError(
          "failing_channels_threshold must be non-negative, got"
          f" {self.failing_channels_threshold}."
      )
    if self.failing_channels_ratio_threshold is not None and not (
        0.0 <= self.failing_channels_ratio_threshold <= 1.0
    ):
      raise ValueError(
          "failing_channels_ratio_threshold must be between 0.0 and 1.0, got"
          f" {self.failing_channels_ratio_threshold}."
      )

  def is_failing_channels_within_threshold(
      self, n_failing: int, n_total: int
  ) -> bool:
    """Checks if the number and ratio of failing channels are within threshold.

    If both thresholds are specified, the check passes if EITHER condition is
    satisfied (or both). If no threshold is set, returns True only when
    n_failing == 0.

    Args:
      n_failing: The number of failing channels.
      n_total: The total number of channels evaluated.

    Returns:
      True if the failing channels are within the tolerance threshold(s), False
      otherwise.
    """
    if n_failing == 0:
      return True
    if n_total <= 0:
      return False

    abs_pass = (
        n_failing <= self.failing_channels_threshold
        if self.failing_channels_threshold is not None
        else None
    )
    rel_pass = (
        (n_failing / n_total) <= self.failing_channels_ratio_threshold
        if self.failing_channels_ratio_threshold is not None
        else None
    )

    if abs_pass is not None and rel_pass is not None:
      return abs_pass or rel_pass
    elif abs_pass is not None:
      return abs_pass
    elif rel_pass is not None:
      return rel_pass
    else:
      return False


@dataclasses.dataclass(frozen=True)
class ROIConsistencyConfig(ChannelCheckConfig):
  """Configuration for the ROI Consistency Check.

  This check verifies if the posterior mean of the ROI falls within a
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
    negative_baseline_prob_review_threshold: Probability threshold for a review.
      If the probability of a negative baseline is above this value, a review is
      issued.
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
  """Configuration for the Goodness of Fit Check.

  Attributes:
    r_squared_threshold: The threshold for R-squared. If R-squared is less than
      or equal to this threshold, a review is issued.
  """

  r_squared_threshold: float = 0.0


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftConfig(ChannelCheckConfig):
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


@dataclasses.dataclass(frozen=True)
class ImplausibleROIConfig(BaseConfig):
  """Configuration for the Implausible ROI Check.

  Attributes:
    roi_upper_bound: The upper bound threshold for spend-weighted posterior mean
      ROI.
    roi_lower_bound: The lower bound threshold for reciprocal-spend-weighted
      posterior mean ROI.
  """

  roi_upper_bound: float = 20.0
  roi_lower_bound: float = 0.5


@dataclasses.dataclass(frozen=True)
class HighVarianceConfig(BaseConfig):
  """Configuration for the High Variance Check.

  Attributes:
    high_variance_threshold: The threshold for spend-weighted relative width
      ratio.
    prior_relative_hdi_width: The relative width of the prior highest density
      interval (HDI) benchmark.
    hdi_prob: The probability for the highest density interval.
  """

  high_variance_threshold: float = 1.0
  prior_relative_hdi_width: float = (
      review_constants.PRIOR_RELATIVE_HDI_WIDTH_FOR_80_PERCENT
  )
  hdi_prob: float = 0.8


@dataclasses.dataclass(frozen=True)
class PotentialBiasConfig(BaseConfig):
  """Configuration for the Potential Bias Check.

  Attributes:
    correlation_threshold: The threshold for maximum absolute Pearson
      correlation.
  """

  correlation_threshold: float = 0.1
