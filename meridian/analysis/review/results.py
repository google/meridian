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

"""Data structures for the Model Quality Checks results."""

import abc
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import enum
import functools
import os
from typing import Any

import jinja2
from meridian.analysis import summary_text
from meridian.analysis.review import configs
from meridian.analysis.review import constants
from meridian.model.calibration import base as calibration_base
from meridian.templates import formatter
import numpy as np
import xarray as xr

__all__ = [
    "BaseCase",
    "BaseResultData",
    "BaselineCases",
    "BaselineCheckResult",
    "BayesianPPPCases",
    "BayesianPPPCheckResult",
    "CalibrationOverviewChannelData",
    "ChannelResult",
    "CheckResult",
    "ConvergenceCases",
    "ConvergenceCheckResult",
    "GoodnessOfFitCases",
    "GoodnessOfFitCheckResult",
    "GoodnessOfFitMetrics",
    "HighVarianceAggregateCases",
    "HighVarianceChannelCases",
    "HighVarianceChannelResult",
    "HighVarianceCheckResult",
    "ImplausibleROIAggregateCases",
    "ImplausibleROIChannelCases",
    "ImplausibleROIChannelResult",
    "ImplausibleROICheckResult",
    "ModelCheckCase",
    "PotentialBiasAggregateCases",
    "PotentialBiasChannelCases",
    "PotentialBiasChannelResult",
    "PotentialBiasCheckResult",
    "PriorPosteriorShiftAggregateCases",
    "PriorPosteriorShiftChannelCases",
    "PriorPosteriorShiftChannelResult",
    "PriorPosteriorShiftCheckResult",
    "ROIConsistencyAggregateCases",
    "ROIConsistencyChannelCases",
    "ROIConsistencyChannelResult",
    "ROIConsistencyCheckResult",
    "ReviewSummary",
    "Status",
    "build_calibration_recommendation_text",
]


# ==============================================================================
# Base classes
# ==============================================================================
@enum.unique
class Status(enum.Enum):
  PASS = enum.auto()
  REVIEW = enum.auto()
  FAIL = enum.auto()


class BaseCase:
  """Base class for all check cases."""

  status: Status

  def __init__(self, status: Status):
    """Initializes the base case with a status."""
    self.status = status


class ModelCheckCase(BaseCase):
  """Base class for all model-level check cases."""

  message_template: str
  recommendation: str | None = None

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None = None,
  ):
    """Initializes the instance."""
    super().__init__(status)
    self.message_template = message_template
    self.recommendation = recommendation


@dataclasses.dataclass(frozen=True)
class BaseResultData(abc.ABC):
  """Base class for check result data."""

  case: BaseCase

  @property
  @abc.abstractmethod
  def details(self) -> Mapping[str, Any]:
    """The details for message formatting."""
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ChannelResult(BaseResultData):
  """Base class for channel-level check results."""

  channel_name: str


@dataclasses.dataclass(frozen=True)
class CheckResult(BaseResultData):
  """Base class for model-level check results."""

  case: ModelCheckCase

  @property
  def recommendation(self) -> str:
    """The check result message."""
    report_str = self.case.message_template.format(**self.details)
    if self.case.recommendation:
      return f"{report_str} {self.case.recommendation}"
    return report_str


# ==============================================================================
# Check: Convergence
# ==============================================================================
# TODO: Move to constants.
NOT_FULLY_CONVERGED_RECOMMENDATION = (
    "Manually inspect the parameters with high R-hat values to determine if the"
    " results are acceptable for your use case, and consider increasing MCMC"
    " iterations or investigating model misspecification."
)

NOT_CONVERGED_RECOMMENDATION = (
    "We recommend increasing MCMC iterations or investigating model"
    " misspecification (e.g., priors, multicollinearity) before proceeding."
)


@enum.unique
class ConvergenceCases(ModelCheckCase, enum.Enum):
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
          "The model hasn't fully converged, and the `max_r_hat` for parameter"
          " `{parameter}` is {rhat:.2f}."
      ),
      NOT_FULLY_CONVERGED_RECOMMENDATION,
  )
  NOT_CONVERGED = (
      Status.FAIL,
      (
          "The model hasn't converged, and the `max_r_hat` for parameter"
          " `{parameter}` is {rhat:.2f}."
      ),
      NOT_CONVERGED_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class ConvergenceCheckResult(CheckResult):
  """The immutable result of the Convergence Check."""

  case: ConvergenceCases
  config: configs.ConvergenceConfig
  max_r_hat: float
  max_parameter: str

  @property
  def max_rhat(self) -> float:
    """Deprecated: Use max_r_hat instead."""
    return self.max_r_hat

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.RHAT: self.max_r_hat,
        constants.PARAMETER: self.max_parameter,
        constants.CONVERGENCE_THRESHOLD: self.config.convergence_threshold,
    }


# ==============================================================================
# Check: Baseline
# ==============================================================================
_BASELINE_FAIL_RECOMMENDATION = (
    "This high probability points to a statistical error and is a clear signal"
    " that the model requires adjustment. The model is likely over-crediting"
    " your treatments. Consider adjusting the model's settings, data, or priors"
    " to correct this issue."
)
_BASELINE_REVIEW_RECOMMENDATION = (
    "This indicates that the baseline time series occasionally dips into"
    " negative values. We recommend visually inspecting the baseline time"
    " series in the Model Fit charts, but don't be overly concerned. An"
    " occasional, small dip may indicate minor statistical error, which is"
    " inherent in any model."
)
_BASELINE_PASS_RECOMMENDATION = (
    "We recommend visually inspecting the baseline time series in the Model "
    "Fit charts to confirm this."
)


@enum.unique
class BaselineCases(ModelCheckCase, enum.Enum):
  """Cases for the Baseline Check."""

  PASS = (
      Status.PASS,
      (
          "The posterior probability that the baseline is negative is"
          " {negative_baseline_prob:.2f}."
      ),
      _BASELINE_PASS_RECOMMENDATION,
  )
  REVIEW = (
      Status.REVIEW,
      (
          "The posterior probability that the baseline is negative is"
          " {negative_baseline_prob:.2f}."
      ),
      _BASELINE_REVIEW_RECOMMENDATION,
  )
  FAIL = (
      Status.FAIL,
      (
          "The posterior probability that the baseline is negative is"
          " {negative_baseline_prob:.2f}."
      ),
      _BASELINE_FAIL_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class BaselineCheckResult(CheckResult):
  """The immutable result of the Baseline Check."""

  case: BaselineCases
  config: configs.BaselineConfig
  negative_baseline_prob: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.NEGATIVE_BASELINE_PROB: self.negative_baseline_prob,
        constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD: (
            self.config.negative_baseline_prob_fail_threshold
        ),
        constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD: (
            self.config.negative_baseline_prob_review_threshold
        ),
    }


# ==============================================================================
# Check: Bayesian Posterior Predictive P-value
# ==============================================================================
_BAYESIAN_PPP_FAIL_RECOMMENDATION = (
    "The observed total outcome is an extreme outlier compared to the model's"
    " expected total outcomes, which suggests a systematic lack of fit. We"
    " recommend reviewing input data quality and re-examining the model"
    " specification (e.g., priors, transformations) to resolve this issue."
)
_BAYESIAN_PPP_PASS_RECOMMENDATION = (
    "The observed total outcome is consistent with the model's posterior"
    " predictive distribution."
)


@enum.unique
class BayesianPPPCases(ModelCheckCase, enum.Enum):
  """Cases for the Bayesian Posterior Predictive P-value Check."""

  PASS = (
      Status.PASS,
      "The Bayesian posterior predictive p-value is {bayesian_ppp:.2f}.",
      _BAYESIAN_PPP_PASS_RECOMMENDATION,
  )
  FAIL = (
      Status.FAIL,
      "The Bayesian posterior predictive p-value is {bayesian_ppp:.2f}.",
      _BAYESIAN_PPP_FAIL_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class BayesianPPPCheckResult(CheckResult):
  """The immutable result of the Bayesian Posterior Predictive P-value Check."""

  case: BayesianPPPCases
  config: configs.BayesianPPPConfig
  bayesian_ppp: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.BAYESIAN_PPP: self.bayesian_ppp,
    }


# ==============================================================================
# Check: Goodness of Fit
# ==============================================================================
_GOODNESS_OF_FIT_REVIEW_RECOMMENDATION = (
    "A negative R-squared signals a potential conflict between your priors and"
    " the data, and it warrants investigation. If this conflict is intentional"
    " (due to an informative prior), no further action is needed. If it's"
    " unintentional, we recommend relaxing your priors to be less restrictive."
)

_GOODNESS_OF_FIT_PASS_RECOMMENDATION = (
    "These goodness-of-fit metrics are intended for guidance and relative"
    " comparison."
)


@enum.unique
class GoodnessOfFitCases(ModelCheckCase, enum.Enum):
  """Cases for the Goodness of Fit Check."""

  PASS = (
      Status.PASS,
      "R-squared = {r_squared:.4f}, MAPE = {mape:.4f}, and wMAPE = {wmape:.4f}",
      _GOODNESS_OF_FIT_PASS_RECOMMENDATION,
  )
  REVIEW = (
      Status.REVIEW,
      "R-squared = {r_squared:.4f}, MAPE = {mape:.4f}, and wMAPE = {wmape:.4f}",
      _GOODNESS_OF_FIT_REVIEW_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class GoodnessOfFitMetrics:
  """The metrics for the Goodness of Fit Check."""

  r_squared: float
  mape: float
  wmape: float
  r_squared_train: float | None = None
  mape_train: float | None = None
  wmape_train: float | None = None
  r_squared_test: float | None = None
  mape_test: float | None = None
  wmape_test: float | None = None


@dataclasses.dataclass(frozen=True)
class GoodnessOfFitCheckResult(CheckResult):
  """The immutable result of the Goodness of Fit Check."""

  case: GoodnessOfFitCases
  metrics: GoodnessOfFitMetrics
  is_holdout: bool = False

  def __post_init__(self):
    if self.is_holdout:
      if any(
          metric is None
          for metric in (
              self.metrics.r_squared_train,
              self.metrics.mape_train,
              self.metrics.wmape_train,
              self.metrics.r_squared_test,
              self.metrics.mape_test,
              self.metrics.wmape_test,
          )
      ):
        raise ValueError(
            "The message template is missing required formatting arguments for"
            " holdout case. Required keys: r_squared_train, mape_train,"
            " wmape_train, r_squared_test, mape_test, wmape_test. Metrics:"
            f" {self.metrics}."
        )

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        f"{constants.R_SQUARED}{constants.ALL_SUFFIX}": self.metrics.r_squared,
        f"{constants.MAPE}{constants.ALL_SUFFIX}": self.metrics.mape,
        f"{constants.WMAPE}{constants.ALL_SUFFIX}": self.metrics.wmape,
        f"{constants.R_SQUARED}{constants.TRAIN_SUFFIX}": (
            self.metrics.r_squared_train
        ),
        f"{constants.MAPE}{constants.TRAIN_SUFFIX}": self.metrics.mape_train,
        f"{constants.WMAPE}{constants.TRAIN_SUFFIX}": self.metrics.wmape_train,
        f"{constants.R_SQUARED}{constants.TEST_SUFFIX}": (
            self.metrics.r_squared_test
        ),
        f"{constants.MAPE}{constants.TEST_SUFFIX}": self.metrics.mape_test,
        f"{constants.WMAPE}{constants.TEST_SUFFIX}": self.metrics.wmape_test,
    }

  @property
  def recommendation(self) -> str:
    """The check result message."""
    if self.is_holdout:
      report_str = (
          "R-squared = {r_squared:.4f} (All),"
          " {r_squared_train:.4f} (Train), {r_squared_test:.4f} (Test); MAPE"
          " = {mape:.4f} (All), {mape_train:.4f} (Train),"
          " {mape_test:.4f} (Test); wMAPE = {wmape:.4f} (All),"
          " {wmape_train:.4f} (Train), {wmape_test:.4f} (Test)".format(
              **self.details
          )
      )
    else:
      report_str = self.case.message_template.format(**self.details)

    return f"{report_str}. {self.case.recommendation}"


# ==============================================================================
# Check: ROI Consistency
# ==============================================================================
_ROI_CONSISTENCY_RECOMMENDATION = (
    "Please review this result to determine if it is reasonable within your"
    " business context."
)


@enum.unique
class ROIConsistencyChannelCases(BaseCase, enum.Enum):
  """Cases for ROI Consistency Check per channel."""

  ROI_PASS = (Status.PASS, enum.auto())
  ROI_LOW = (Status.REVIEW, enum.auto())
  ROI_HIGH = (Status.REVIEW, enum.auto())
  PRIOR_ROI_QUANTILE_INF = (Status.REVIEW, enum.auto())
  QUANTILE_NOT_DEFINED = (Status.REVIEW, enum.auto())

  def __init__(self, status: Status, unique_id: Any):
    """Initializes the instance."""
    super().__init__(status)


class ROIConsistencyAggregateCases(ModelCheckCase, enum.Enum):
  """Cases for ROI Consistency Check aggregate result."""

  PASS = (
      Status.PASS,
      (
          "The posterior distribution of the ROI is within a reasonable range,"
          " aligning with the custom priors you provided."
      ),
      None,
  )
  REVIEW = (
      Status.REVIEW,
      "{quantile_not_defined_msg}{inf_channels_msg}{low_high_channels_msg}",
      _ROI_CONSISTENCY_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class ROIConsistencyChannelResult(ChannelResult):
  """The immutable result of ROI Consistency Check for a single channel."""

  case: ROIConsistencyChannelCases
  prior_roi_lo: float
  prior_roi_hi: float
  posterior_roi_mean: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.PRIOR_ROI_LO: self.prior_roi_lo,
        constants.PRIOR_ROI_HI: self.prior_roi_hi,
        constants.POSTERIOR_ROI_MEAN: self.posterior_roi_mean,
    }


@dataclasses.dataclass(frozen=True)
class ROIConsistencyCheckResult(CheckResult):
  """The immutable result of model-level ROI Consistency Check."""

  case: ROIConsistencyAggregateCases
  channel_results: list[ROIConsistencyChannelResult]
  aggregate_details: Mapping[str, Any]

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return self.aggregate_details


# ==============================================================================
# Check: Prior-Posterior Shift
# ==============================================================================
_PPS_REVIEW_RECOMMENDATION = (
    "Please review these channels to see if this is expected (due to a strong"
    " priors) or problematic (due to a weak signal)."
)


@enum.unique
class PriorPosteriorShiftChannelCases(BaseCase, enum.Enum):
  """Cases for Prior-Posterior Shift Check per channel."""

  SHIFT = (Status.PASS, enum.auto())
  NO_SHIFT = (Status.REVIEW, enum.auto())

  def __init__(self, status: Status, unique_id: Any):
    """Initializes the instance."""
    super().__init__(status)


class PriorPosteriorShiftAggregateCases(ModelCheckCase, enum.Enum):
  """Cases for Prior-Posterior Shift Check aggregate result."""

  PASS = (
      Status.PASS,
      (
          "The model has successfully learned from the data. This is a positive"
          " sign that your data was informative."
      ),
      None,
  )
  REVIEW = (
      Status.REVIEW,
      (
          "We've detected channel(s) {channels_str} where the posterior"
          " distribution did not significantly shift from the prior. This"
          " suggests the data signal for these channels was not strong enough"
          " to update the model's beliefs."
      ),
      _PPS_REVIEW_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftChannelResult(ChannelResult):
  """The result of Prior-Posterior Shift Check for a single channel."""

  case: PriorPosteriorShiftChannelCases

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {}


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftCheckResult(CheckResult):
  """The immutable result of model-level Prior-Posterior Shift Check."""

  case: PriorPosteriorShiftAggregateCases
  channel_results: list[PriorPosteriorShiftChannelResult]
  no_shift_channels: list[str]

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.CHANNELS_STR: ", ".join(
            f"`{channel}`" for channel in self.no_shift_channels
        )
    }


# ==============================================================================
# Check: Implausible ROI
# ==============================================================================
@enum.unique
class ImplausibleROIChannelCases(BaseCase, enum.Enum):
  """Cases for Implausible ROI Check per channel."""

  ROI_PASS = (Status.PASS, enum.auto())
  ROI_HIGH = (Status.REVIEW, enum.auto())
  ROI_LOW = (Status.REVIEW, enum.auto())

  # TODO: Remove unused unique_id argument, here and elsewhere.
  def __init__(self, status: Status, unique_id: Any):
    """Initializes the instance."""
    super().__init__(status)


class ImplausibleROIAggregateCases(ModelCheckCase, enum.Enum):
  """Cases for Implausible ROI Check aggregate result."""

  PASS = (
      Status.PASS,
      "All channels have plausible ROI estimates.",
      None,
  )
  REVIEW = (
      Status.REVIEW,
      "{implausible_roi_msg}",
      constants.IMPLAUSIBLE_ROI_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    """Initializes the instance."""
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class ImplausibleROIChannelResult(ChannelResult):
  """The immutable result of Implausible ROI Check for a single channel.

  Attributes:
    case: The specific case for this channel's implausible ROI check.
    spend_share: The proportion of total spend for this channel.
    roi_mean: The posterior mean of the ROI for this channel.
    spend_weighted_roi: The spend-weighted ROI for this channel.
  """

  case: ImplausibleROIChannelCases
  spend_share: float
  roi_mean: float
  spend_weighted_roi: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        constants.SPEND_SHARE: self.spend_share,
        constants.ROI_MEAN: self.roi_mean,
        constants.SPEND_WEIGHTED_ROI: self.spend_weighted_roi,
    }


@dataclasses.dataclass(frozen=True)
class ImplausibleROICheckResult(CheckResult):
  """The immutable result of model-level Implausible ROI Check.

  Attributes:
    case: The aggregate case for the implausible ROI check across all channels.
    channel_results: A list of `ImplausibleROIChannelResult` for each channel.
    high_roi_channels: A list of channel names flagged as having high ROI.
    low_roi_channels: A list of channel names flagged as having low ROI.
    aggregate_details: Additional details of the aggregate check result.
    roi_upper_bound: The upper bound for plausible ROI.
    roi_lower_bound: The lower bound for plausible ROI.
  """

  case: ImplausibleROIAggregateCases
  channel_results: list[ImplausibleROIChannelResult]
  high_roi_channels: list[str]
  low_roi_channels: list[str]
  aggregate_details: Mapping[str, Any]
  roi_upper_bound: float = configs.ImplausibleROIConfig.roi_upper_bound
  roi_lower_bound: float = configs.ImplausibleROIConfig.roi_lower_bound

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return self.aggregate_details


# ==============================================================================
# Check: High Variance
# ==============================================================================
@enum.unique
class HighVarianceChannelCases(BaseCase, enum.Enum):
  """Cases for High Variance Check per channel."""

  ROI_PASS = (Status.PASS, enum.auto())
  HIGH_VARIANCE = (Status.REVIEW, enum.auto())

  # TODO: Remove unused unique_id argument, here and elsewhere.
  def __init__(self, status: Status, unique_id: Any):
    super().__init__(status)


class HighVarianceAggregateCases(ModelCheckCase, enum.Enum):
  """Cases for High Variance Check aggregate result."""

  PASS = (
      Status.PASS,
      "All channels have acceptable ROI variance.",
      None,
  )
  REVIEW = (
      Status.REVIEW,
      (
          "We've detected channel(s) {high_variance_channels_str} with highly"
          " uncertain ROI estimates (wide posterior intervals)."
      ),
      constants.HIGH_VARIANCE_ROI_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class HighVarianceChannelResult(ChannelResult):
  """The immutable result of High Variance Check for a single channel.

  Attributes:
    case: The specific case for this channel's high variance check.
    spend_share: The proportion of total spend for this channel.
    relative_width_ratio: The ratio of the posterior ROI credible interval width
      to the prior width.
  """

  case: HighVarianceChannelCases
  spend_share: float
  relative_width_ratio: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        "spend_share": self.spend_share,
        "relative_width_ratio": self.relative_width_ratio,
    }


@dataclasses.dataclass(frozen=True)
class HighVarianceCheckResult(CheckResult):
  """The immutable result of model-level High Variance Check.

  Attributes:
    case: The aggregate case for the high variance check across all channels.
    channel_results: A list of `HighVarianceChannelResult` for each channel.
    high_variance_channels: A list of channel names flagged as having high ROI
      variance.
    prior_relative_hdi_width: The prior relative HDI width threshold.
  """

  case: HighVarianceAggregateCases
  channel_results: list[HighVarianceChannelResult]
  high_variance_channels: list[str]
  prior_relative_hdi_width: float = (
      constants.PRIOR_RELATIVE_HDI_WIDTH_FOR_80_PERCENT
  )

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        "high_variance_channels_str": ", ".join(
            f"`{c}`" for c in self.high_variance_channels
        )
    }


# ==============================================================================
# Check: Potential Bias
# ==============================================================================
@enum.unique
class PotentialBiasChannelCases(BaseCase, enum.Enum):
  """Cases for Potential Bias Check per channel."""

  ROI_PASS = (Status.PASS, enum.auto())
  LOW_CORRELATION = (Status.REVIEW, enum.auto())

  def __init__(self, status: Status, unique_id: Any):
    super().__init__(status)


class PotentialBiasAggregateCases(ModelCheckCase, enum.Enum):
  """Cases for Potential Bias Check aggregate result."""

  PASS = (
      Status.PASS,
      "All channels have sufficient correlation with control variables.",
      None,
  )
  REVIEW = (
      Status.REVIEW,
      (
          "We've detected channel(s) {low_correlation_channels_str} with very"
          " low correlation with all included control variables."
      ),
      constants.POTENTIAL_BIAS_RECOMMENDATION,
  )
  NO_CONTROLS = (
      Status.REVIEW,
      (
          "No control variables are included in the model. Consider adding"
          " control variables to control for potential confounding bias."
      ),
      constants.POTENTIAL_BIAS_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class PotentialBiasChannelResult(ChannelResult):
  """The immutable result of Potential Bias Check for a single channel."""

  case: PotentialBiasChannelCases
  max_abs_correlation: float

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        "max_abs_correlation": self.max_abs_correlation,
    }


@dataclasses.dataclass(frozen=True)
class PotentialBiasCheckResult(CheckResult):
  """The immutable result of model-level Potential Bias Check."""

  case: PotentialBiasAggregateCases
  channel_results: list[PotentialBiasChannelResult]
  low_correlation_channels: list[str]
  correlation_matrix: xr.DataArray
  correlation_threshold: float = (
      configs.PotentialBiasConfig.correlation_threshold
  )

  @property
  def details(self) -> Mapping[str, Any]:
    """The check result details."""
    return {
        "low_correlation_channels_str": ", ".join(
            f"`{c}`" for c in self.low_correlation_channels
        ),
        constants.CORRELATION_MATRIX: self.correlation_matrix,
    }


# ==============================================================================
# Valid subsets of result types for health summary validation.
# ==============================================================================
CONVERGENCE_ONLY_SET = frozenset([ConvergenceCheckResult])
MODEL_LEVEL_SET = frozenset([
    ConvergenceCheckResult,
    BaselineCheckResult,
    GoodnessOfFitCheckResult,
    BayesianPPPCheckResult,
])
PPS_SET = MODEL_LEVEL_SET | frozenset([PriorPosteriorShiftCheckResult])
ROI_SET = PPS_SET | frozenset([ROIConsistencyCheckResult])


_CALIBRATION_CHECK_RESULTS = (
    ImplausibleROICheckResult,
    HighVarianceCheckResult,
    PotentialBiasCheckResult,
)


@dataclasses.dataclass(frozen=True)
class CalibrationOverviewChannelData:
  """Container for calibration overview data for a single channel.

  Attributes:
    channel_name: Name of the channel.
    spend: Total spend of the channel.
    calibrated_output: The calibration output container for the channel.
    calibrated_prior_dist: The 1D calibrated prior distribution for the channel.
    posterior_samples: 1D array of posterior ROI samples for the channel.
    chart_json: Serialized JSON string for the Altair overview chart.
    details_chart_json: Serialized JSON string for the Altair details chart.
  """

  channel_name: str
  spend: float
  calibrated_output: calibration_base.CalibrationOutput | None = None
  calibrated_prior_dist: Any = None
  posterior_samples: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([])
  )
  chart_json: str | None = None
  details_chart_json: str | None = None


def _normalized_center_bowl(
    roi: float,
    spend_share: float,
    roi_lower_bound: float = configs.ImplausibleROIConfig.roi_lower_bound,
    roi_upper_bound: float = configs.ImplausibleROIConfig.roi_upper_bound,
) -> float:
  """Computes the Implausible ROI component score (Normalized Center Bowl).

  Args:
    roi: Posterior mean ROI for the channel.
    spend_share: Proportion of total spend for the channel.
    roi_lower_bound: Lower threshold parameter for implausible ROI.
    roi_upper_bound: Upper threshold parameter for implausible ROI.

  Returns:
    Implausible ROI score bounded in [0.0, 100.0].
  """
  if spend_share <= 0:
    return 100.0
  roi_safe = max(roi, constants.EPSILON)
  spend_share_safe = max(spend_share, constants.EPSILON)

  center = np.sqrt(roi_lower_bound * roi_upper_bound)
  z = np.log(roi_safe / center)
  d = np.log(
      (1.0 / spend_share_safe) * np.sqrt(roi_upper_bound / roi_lower_bound)
  )
  d = max(d, constants.EPSILON)
  v = (z / d) ** 2
  return float(np.clip(100.0 * np.exp(-np.log(2) * v), 0.0, 100.0))


def _normalized_half_bowl(
    relative_width_ratio: float,
    spend_share: float,
    high_variance_threshold: float = configs.HighVarianceConfig.high_variance_threshold,
    ideal_threshold: float = constants.HIGH_VARIANCE_IDEAL_THRESHOLD,
) -> float:
  """Computes the High Variance ROI component score (Normalized Half Bowl).

  Args:
    relative_width_ratio: Ratio of posterior HDI width to prior HDI width.
    spend_share: Proportion of total spend for the channel.
    high_variance_threshold: Threshold parameter for high variance ROI.
    ideal_threshold: Benchmark ideal threshold below which score is 100.

  Returns:
    High variance ROI score bounded in [0.0, 100.0].
  """
  x = relative_width_ratio * spend_share
  if x <= ideal_threshold:
    return 100.0
  x_safe = max(x, constants.EPSILON)
  z = np.log(x_safe / ideal_threshold)
  d = np.log(high_variance_threshold / ideal_threshold)
  d = max(d, constants.EPSILON)
  v = (z / d) ** 2
  penalty = 1.0 - np.exp(-np.log(2) * v)
  return float(np.clip(100.0 * (1.0 - penalty), 0.0, 100.0))


def _potential_bias_score(
    max_abs_correlation: float,
    correlation_threshold: float = configs.PotentialBiasConfig.correlation_threshold,
) -> float:
  """Computes the Potential Bias component score.

  Args:
    max_abs_correlation: Maximum absolute Pearson correlation with any control
      variable across geos.
    correlation_threshold: Threshold for potential bias check.

  Returns:
    Potential bias score bounded in [0.0, 100.0].
  """
  max_bias = float(np.clip(max_abs_correlation, 0.0, 1.0))
  if max_bias <= 0.0:
    return 0.0
  if correlation_threshold <= 0.0 or correlation_threshold >= 1.0:
    exponent = 1.0
  else:
    exponent = np.log(0.5) / np.log(correlation_threshold)
  return float(np.clip(100.0 * (max_bias**exponent), 0.0, 100.0))


def _compute_channel_calibration_score(
    implausible_roi_score: float,
    high_variance_roi_score: float,
    potential_bias_score: float,
) -> float:
  """Computes composite calibration score for an uncalibrated channel."""
  return (
      constants.CALIBRATION_IMPLAUSIBLE_ROI_WEIGHT * implausible_roi_score
      + constants.CALIBRATION_HIGH_VARIANCE_WEIGHT * high_variance_roi_score
      + constants.CALIBRATION_POTENTIAL_BIAS_WEIGHT * potential_bias_score
  )


def _order_channels_by_status(
    channels: Iterable[str],
    status: Mapping[str, bool] | None,
) -> list[str]:
  """Orders channels by their appearance in channel_calibration_status."""
  if not status:
    return list(dict.fromkeys(channels))
  channel_order = {ch: i for i, ch in enumerate(status)}
  unique_channels = list(dict.fromkeys(channels))
  return sorted(
      unique_channels, key=lambda c: channel_order.get(c, float("inf"))
  )


# ==============================================================================
# Review Summary
# ==============================================================================
@dataclasses.dataclass(frozen=False)
class ReviewSummary:
  """The final summary of all model quality checks.

  Attributes:
    overall_status: The overall status of all checks.
    summary_message: A summary message of all checks.
    results: A list of all check results.
    health_score: The health score of the model.
    channel_calibration_status: Mapping of channel name to calibration status.
    calibrated_channel_names: Sequence of calibrated channel names.
    implausible_roi_chart_json: Chart JSON for implausible ROI recommendation
      plot.
    high_variance_chart_json: Chart JSON for high variance recommendation plot.
    potential_bias_chart_json: Chart JSON for potential bias recommendation
      plot.
    calibration_overview_data: Sequence of calibration overview channel data.
  """  # fmt: skip

  overall_status: Status
  summary_message: str
  results: list[CheckResult]
  health_score: float
  channel_calibration_status: Mapping[str, bool] = dataclasses.field(
      default_factory=dict
  )
  calibrated_channel_names: Sequence[str] = dataclasses.field(
      default_factory=list
  )
  implausible_roi_chart_json: str | None = None
  high_variance_chart_json: str | None = None
  potential_bias_chart_json: str | None = None
  calibration_overview_data: Sequence[CalibrationOverviewChannelData] = ()

  @property
  def channel_calibration_recommendations(self) -> list[dict[str, Any]]:
    """Computes per-channel calibration recommendation data.

    The recommendations are filtered based on the total number of channels:
    - If the total number of channels is less than or equal to
      `MAX_CHANNELS_FOR_CALIBRATED_DISPLAY`, all channels (both calibrated and
      uncalibrated) are included in their original order.
    - If the total number of channels exceeds this threshold, calibrated
      channels are excluded from the output, and all uncalibrated channels are
      shown in their original order.

    Returns:
      A list of dictionaries containing channel recommendation data.
    """
    if not any(isinstance(r, _CALIBRATION_CHECK_RESULTS) for r in self.results):
      return []

    high_roi_map = {}
    low_roi_map = {}
    high_variance_map = {}
    potential_bias_map = {}

    for r in self.results:
      if isinstance(r, ImplausibleROICheckResult):
        for cr in r.channel_results:
          if cr.case == ImplausibleROIChannelCases.ROI_HIGH:
            high_roi_map[cr.channel_name] = Status.REVIEW
            low_roi_map[cr.channel_name] = Status.PASS
          elif cr.case == ImplausibleROIChannelCases.ROI_LOW:
            high_roi_map[cr.channel_name] = Status.PASS
            low_roi_map[cr.channel_name] = Status.REVIEW
          else:
            high_roi_map[cr.channel_name] = Status.PASS
            low_roi_map[cr.channel_name] = Status.PASS
      elif isinstance(r, HighVarianceCheckResult):
        # If `HighVarianceCheck` occurs in the list of the post convergence
        # checks more than once, the entire `high_variance_map` will be
        # overwritten.
        high_variance_map = {
            cr.channel_name: cr.case.status for cr in r.channel_results
        }
      elif isinstance(r, PotentialBiasCheckResult):
        # If `PotentialBiasCheck` occurs in the list of the post convergence
        # checks more than once, the entire `potential_bias_map` will be
        # overwritten.
        potential_bias_map = {
            cr.channel_name: cr.case.status for cr in r.channel_results
        }

    recs = []
    n_channels = len(self.channel_calibration_status)
    scores = self.channel_calibration_scores
    for (
        channel_name,
        is_calibrated,
    ) in self.channel_calibration_status.items():
      if is_calibrated:
        if n_channels <= constants.MAX_CHANNELS_FOR_CALIBRATED_DISPLAY:
          recs.append({
              constants.CHANNEL_NAME: channel_name,
              constants.IS_CALIBRATED: True,
              constants.CALIBRATION_SCORE: scores.get(
                  channel_name, constants.CALIBRATED_CHANNEL_SCORE
              ),
          })
      else:
        recs.append({
            constants.CHANNEL_NAME: channel_name,
            constants.IS_CALIBRATED: False,
            constants.CALIBRATION_SCORE: scores.get(
                channel_name, constants.CALIBRATED_CHANNEL_SCORE
            ),
            constants.HIGH_ROI_STATUS: high_roi_map.get(
                channel_name, Status.PASS
            ),
            constants.LOW_ROI_STATUS: low_roi_map.get(
                channel_name, Status.PASS
            ),
            constants.HIGH_VARIANCE_STATUS: high_variance_map.get(
                channel_name, Status.PASS
            ),
            constants.POTENTIAL_BIAS_STATUS: potential_bias_map.get(
                channel_name, Status.PASS
            ),
        })

    return recs

  @functools.cached_property
  def channel_calibration_scores(self) -> Mapping[str, float]:
    """Computes per-channel calibration scores."""
    if not self.channel_calibration_status:
      return {}

    implausible_roi_map = {}
    implausible_roi_result = next(
        (r for r in self.results if isinstance(r, ImplausibleROICheckResult)),
        None,
    )
    if implausible_roi_result is not None:
      for cr in implausible_roi_result.channel_results:
        implausible_roi_map[cr.channel_name] = _normalized_center_bowl(
            roi=cr.roi_mean,
            spend_share=cr.spend_share,
        )

    high_variance_map = {}
    high_variance_result = next(
        (r for r in self.results if isinstance(r, HighVarianceCheckResult)),
        None,
    )
    if high_variance_result is not None:
      for cr in high_variance_result.channel_results:
        high_variance_map[cr.channel_name] = _normalized_half_bowl(
            relative_width_ratio=cr.relative_width_ratio,
            spend_share=cr.spend_share,
        )

    potential_bias_map = {}
    potential_bias_result = next(
        (r for r in self.results if isinstance(r, PotentialBiasCheckResult)),
        None,
    )
    if potential_bias_result is not None:
      if potential_bias_result.case == PotentialBiasAggregateCases.NO_CONTROLS:
        for ch in self.channel_calibration_status:
          potential_bias_map[ch] = 0.0
      else:
        for cr in potential_bias_result.channel_results:
          potential_bias_map[cr.channel_name] = _potential_bias_score(
              max_abs_correlation=cr.max_abs_correlation,
          )

    scores = {}
    for (
        channel_name,
        is_calibrated,
    ) in self.channel_calibration_status.items():
      if is_calibrated:
        scores[channel_name] = constants.CALIBRATED_CHANNEL_SCORE
      else:
        imp_score = implausible_roi_map.get(
            channel_name, constants.CALIBRATED_CHANNEL_SCORE
        )
        hv_score = high_variance_map.get(
            channel_name, constants.CALIBRATED_CHANNEL_SCORE
        )
        pb_score = potential_bias_map.get(
            channel_name, constants.CALIBRATED_CHANNEL_SCORE
        )
        scores[channel_name] = _compute_channel_calibration_score(
            implausible_roi_score=imp_score,
            high_variance_roi_score=hv_score,
            potential_bias_score=pb_score,
        )

    return scores

  @property
  def calibration_score(self) -> float:
    """Computes the overall model-level calibration score (arithmetic average)."""
    scores = self.channel_calibration_scores
    if not scores:
      return constants.CALIBRATED_CHANNEL_SCORE
    return float(np.mean(list(scores.values())))

  @property
  def channels_recommended_for_calibration(self) -> list[str]:
    """Returns uncalibrated channels with calibration score below threshold."""
    status = self.channel_calibration_status or {}
    uncalibrated = [
        ch
        for ch, score in self.channel_calibration_scores.items()
        if not status.get(ch, False)
        and score < constants.CALIBRATION_SCORE_THRESHOLD
    ]
    return _order_channels_by_status(uncalibrated, status)

  _channels_recommended_for_calibration = channels_recommended_for_calibration

  @property
  def has_calibration_warning(self) -> bool:
    """Returns True if any uncalibrated channel has score below threshold."""
    return bool(self.channels_recommended_for_calibration)

  def __repr__(self) -> str:
    report = []
    report.append("=" * 40)
    report.append("Model Quality Checks")
    report.append("=" * 40)
    report.append(f"Overall Status: {self.overall_status.name}")
    report.append(f"Summary: {self.summary_message}")
    report.append(f"Health Score: {self.health_score:.1f}")
    report.append("\nCheck Results:")

    for result in self.results:
      if isinstance(result, _CALIBRATION_CHECK_RESULTS):
        continue
      name = result.__class__.__name__
      if name.endswith("CheckResult"):
        title = name[: -len("CheckResult")]
      else:
        title = name

      report.append("-" * 40)
      report.append(f"{title} Check:")
      report.append(f"  Status: {result.case.status.name}")
      report.append(f"  Recommendation: {result.recommendation}")

    if (
        self.channel_calibration_recommendations
        and self.channel_calibration_status
    ):
      report.append("\n" + "=" * 115)
      report.append("Channel Calibration Recommendation")
      report.append("=" * 115)
      report.append(
          f"{'Channel':<20} | {'Calibration Score':<18} | {'High ROI':<15} |"
          f" {'Low ROI':<15} | {'High Variance ROI':<17} |"
          f" {'Potential Bias':<14}"
      )
      report.append("-" * 115)

      for rec in self.channel_calibration_recommendations:
        channel_name = rec[constants.CHANNEL_NAME]
        score = rec[constants.CALIBRATION_SCORE]
        score_str = f"{score:.1f}"
        if rec[constants.IS_CALIBRATED]:
          report.append(
              f"{channel_name:<20} | {score_str:<18} |"
              f" {'-' * 29} Calibrated {'-' * 29}"
          )
        else:
          high_roi_str = (
              constants.DRIVER
              if rec[constants.HIGH_ROI_STATUS] == Status.REVIEW
              else constants.NON_DRIVER
          )
          low_roi_str = (
              constants.DRIVER
              if rec[constants.LOW_ROI_STATUS] == Status.REVIEW
              else constants.NON_DRIVER
          )
          hv_str = (
              constants.DRIVER
              if rec[constants.HIGH_VARIANCE_STATUS] == Status.REVIEW
              else constants.NON_DRIVER
          )
          pb_str = (
              constants.DRIVER
              if rec[constants.POTENTIAL_BIAS_STATUS] == Status.REVIEW
              else constants.NON_DRIVER
          )
          report.append(
              f"{channel_name:<20} | {score_str:<18} | {high_roi_str:<15}"
              f" | {low_roi_str:<15} | {hv_str:<17} | {pb_str:<14}"
          )
      report.append("-" * 115)

    return "\n".join(report)

  @functools.cached_property
  def _template_env(self) -> jinja2.Environment:
    """A shared template environment bound to this summary."""
    return formatter.create_template_env()

  @property
  def checks_status(self) -> Mapping[str, str]:
    """A dictionary of check names and statuses."""
    return {
        result.__class__.__name__: result.case.status.name
        for result in self.results
    }

  def output_model_health_card(
      self,
      filename: str,
      filepath: str,
  ):
    """Generates and saves the HTML output for the model health card.

    Args:
      filename: The name of the file to save the HTML output to.
      filepath: The path to the directory to save the HTML output to.
    """
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), "w") as f:
      f.write(self._gen_model_health_card())

  def _gen_model_health_card(self) -> str:
    """Generates the HTML model health card (as sanitized content str)."""
    html_template = self._template_env.get_template("summary.html.jinja")
    cards = [self._create_health_card_html()]
    if self.channel_calibration_recommendations:
      cards.append(self._create_calibration_summary_card_html())
      cards.append(self._create_calibration_overview_card_html())
      cards.append(self._create_calibration_details_card_html())
      cards.append(self._create_channel_recommendation_card_html())
    return html_template.render(
        title=summary_text.MODEL_HEALTH_CARD_TITLE,
        cards=cards,
    )

  def _create_health_card_html(self) -> str:
    """Creates the HTML snippet for the Model Health Card."""
    model_checks = []
    channel_checks = []

    for result in self.results:
      if isinstance(
          result,
          (
              ImplausibleROICheckResult,
              HighVarianceCheckResult,
              PotentialBiasCheckResult,
          ),
      ):
        continue
      check_data = self._get_check_data(result)
      if isinstance(
          result,
          (
              PriorPosteriorShiftCheckResult,
              ROIConsistencyCheckResult,
          ),
      ):
        channel_checks.append(check_data)
      else:
        model_checks.append(check_data)

    template = self._template_env.get_template(
        "model_health_summary_card.html.jinja"
    )

    calibration_score = self.calibration_score
    recommended_channels = self.channels_recommended_for_calibration
    recommended_channels_text = _format_list_with_and(
        [f"'{c}'" for c in recommended_channels]
    )
    calibration_recommendation_text = build_calibration_recommendation_text(
        recommended_channels=recommended_channels,
        driver_issues_by_channel=(
            self._uncalibrated_channels_with_driver_issues()
        ),
        location=constants.CALIBRATION_TEXT_METRICS_CHECK,
        calibration_score=calibration_score,
    )

    return template.render(
        health_score=self.health_score,
        overall_status=self.overall_status.name,
        summary_message=self.summary_message,
        metrics_checks=model_checks,
        advanced_checks=channel_checks,
        has_calibration_recommendations=bool(
            self.channel_calibration_recommendations
        ),
        n_recommended=len(recommended_channels),
        n_total_channels=len(self.channel_calibration_status),
        calibration_score=calibration_score,
        recommended_channels_text=recommended_channels_text,
        calibration_recommendation_text=calibration_recommendation_text,
    )

  def _create_channel_recommendation_card_html(self) -> str:
    """Creates the HTML snippet for the Channel Calibration Recommendation Card."""
    status = self.channel_calibration_status or {}
    recommended_channels = self.channels_recommended_for_calibration
    issues_by_channel = self._uncalibrated_channels_with_driver_issues()

    banner_text = build_calibration_recommendation_text(
        recommended_channels=recommended_channels,
        driver_issues_by_channel=issues_by_channel,
        location=constants.CALIBRATION_TEXT_CHANNEL_RECOMMENDATION,
    )

    recommendation_warning = None
    recommendation_info = None
    if self.has_calibration_warning:
      recommendation_warning = banner_text
    else:
      recommendation_info = banner_text

    implausible_roi_result = next(
        (r for r in self.results if isinstance(r, ImplausibleROICheckResult)),
        None,
    )
    implausible_roi_description = None
    implausible_roi_is_warning = False
    if self.implausible_roi_chart_json or implausible_roi_result is not None:
      desc = (
          "This plot displays your media and reach and frequency channels"
          " according to their spend and ROI."
      )
      if implausible_roi_result is not None:
        flagged = []
        high_roi_channels = _order_channels_by_status(
            [
                ch
                for ch in implausible_roi_result.high_roi_channels
                if not status.get(ch, False)
            ],
            status,
        )
        low_roi_channels = _order_channels_by_status(
            [
                ch
                for ch in implausible_roi_result.low_roi_channels
                if not status.get(ch, False)
            ],
            status,
        )
        if high_roi_channels:
          ch_text = _format_list_with_and(
              [f"'{ch}'" for ch in high_roi_channels]
          )
          flagged.append(f"{ch_text} for having high ROI")
        if low_roi_channels:
          ch_text = _format_list_with_and(
              [f"'{ch}'" for ch in low_roi_channels]
          )
          flagged.append(f"{ch_text} for having low ROI")
        if flagged:
          desc += f" We recommend reviewing {_format_list_with_and(flagged)}."
          implausible_roi_is_warning = True
      if implausible_roi_is_warning:
        desc += (
            " In general, the deeper the channels are into their respective"
            " regions, the greater the concern may be and the more value you"
            " may gain from an incrementality experiment for that channel."
            " Conversely, channels outside of the regions but close to the"
            " boundary may also be strong candidates for calibration. For"
            " readability, ROIs between 0.6 and 19 are clustered together on"
            " this plot. Please hover over points or use <a"
            ' href="https://developers.google.com/meridian/reference/api/meridian/analysis/analyzer/MeridianAnalyzer#roi"'
            ' target="_blank">MeridianAnalyzer.roi</a> to view the exact ROI'
            " for specific channels."
        )
      else:
        desc += (
            " Channels closer to the boundaries of the Implausible High ROI and"
            " Implausible Low ROI regions may be strong candidates for"
            " calibration. For readability, ROIs between 0.6 and 19 are"
            " clustered together on this plot. Please hover over points or use"
            ' <a href="https://developers.google.com/meridian/reference/api/meridian/analysis/analyzer/MeridianAnalyzer#roi"'
            ' target="_blank">MeridianAnalyzer.roi</a> to view the exact ROI'
            " for specific channels."
        )
      implausible_roi_description = desc

    high_variance_result = next(
        (r for r in self.results if isinstance(r, HighVarianceCheckResult)),
        None,
    )
    high_variance_description = None
    high_variance_is_warning = False
    if self.high_variance_chart_json or high_variance_result is not None:
      desc = (
          "This plot displays your media and reach and frequency channels"
          " according to their spend and Relative Credible Interval (RCI), a"
          " measure of their variance."
      )
      if high_variance_result is not None:
        high_variance_channels = _order_channels_by_status(
            [
                ch
                for ch in high_variance_result.high_variance_channels
                if not status.get(ch, False)
            ],
            status,
        )
        if high_variance_channels:
          ch_text = _format_list_with_and(
              [f"'{ch}'" for ch in high_variance_channels]
          )
          desc += (
              f" We recommend reviewing {ch_text} for having high variance"
              " ROI. In general, the deeper the channels are into the blue"
              " region, the greater the concern may be and the more value you"
              " may gain from an incrementality experiment for that channel."
              " Conversely, channels outside of the region but close to the"
              " boundary may also be strong candidates for calibration."
          )
          high_variance_is_warning = True
      if not high_variance_is_warning:
        desc += (
            " Channels close to the boundary of the High Variance ROI region"
            " may be strong candidates for calibration."
        )
      high_variance_description = desc

    potential_bias_result = next(
        (r for r in self.results if isinstance(r, PotentialBiasCheckResult)),
        None,
    )
    potential_bias_description = None
    potential_bias_is_warning = False
    if self.potential_bias_chart_json or potential_bias_result is not None:
      desc = (
          "This plot displays your media and reach and frequency channels"
          " along with their correlation with your available controls."
      )
      if potential_bias_result is not None:
        low_correlation_channels = _order_channels_by_status(
            [
                ch
                for ch in potential_bias_result.low_correlation_channels
                if not status.get(ch, False)
            ],
            status,
        )
        if low_correlation_channels:
          ch_text = _format_list_with_and(
              [f"'{ch}'" for ch in low_correlation_channels]
          )
          if len(low_correlation_channels) > 1:
            desc += (
                f" {ch_text} show potential bias as they have low correlation"
                " with all controls and thus may be missing relevant controls."
            )
          else:
            desc += (
                f" {ch_text} shows potential bias as it has low correlation"
                " with all controls and thus may be missing relevant controls."
            )
          potential_bias_is_warning = True
      potential_bias_description = desc

    template = self._template_env.get_template(
        "channel_recommendation_card.html.jinja"
    )
    return template.render(
        recommendations=self.channel_calibration_recommendations,
        has_calibration_warning=self.has_calibration_warning,
        recommendation_warning=recommendation_warning,
        recommendation_info=recommendation_info,
        calibration_score_threshold=constants.CALIBRATION_SCORE_THRESHOLD,
        calibration_score_yellow_color=constants.CALIBRATION_SCORE_YELLOW_COLOR,
        driver_text=constants.DRIVER,
        non_driver_text=constants.NON_DRIVER,
        implausible_roi_chart_json=self.implausible_roi_chart_json,
        high_variance_chart_json=self.high_variance_chart_json,
        potential_bias_chart_json=self.potential_bias_chart_json,
        implausible_roi_description=implausible_roi_description,
        high_variance_description=high_variance_description,
        potential_bias_description=potential_bias_description,
        implausible_roi_is_warning=implausible_roi_is_warning,
        high_variance_is_warning=high_variance_is_warning,
        potential_bias_is_warning=potential_bias_is_warning,
        implausible_roi_has_warning=implausible_roi_is_warning,
        high_variance_has_warning=high_variance_is_warning,
        potential_bias_has_warning=potential_bias_is_warning,
        implausible_roi_has_flagged_channels=implausible_roi_is_warning,
        high_variance_has_flagged_channels=high_variance_is_warning,
        potential_bias_has_flagged_channels=potential_bias_is_warning,
    )

  def _create_calibration_summary_card_html(self) -> str:
    """Creates the HTML snippet for the Meridian GeoX Calibration Summary Card."""
    status = self.channel_calibration_status or {}
    recommended_channels = self.channels_recommended_for_calibration
    issues_by_channel = self._uncalibrated_channels_with_driver_issues()

    recommendation_text = build_calibration_recommendation_text(
        recommended_channels=recommended_channels,
        driver_issues_by_channel=issues_by_channel,
        location=constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY,
    )

    template = self._template_env.get_template(
        "calibration_summary_card.html.jinja"
    )

    calibration_score = self.calibration_score
    calibrated_channels = _order_channels_by_status(
        self.calibrated_channel_names, status
    )

    return template.render(
        calibration_score=calibration_score,
        calibration_score_threshold=constants.CALIBRATION_SCORE_THRESHOLD,
        calibration_score_yellow_color=constants.CALIBRATION_SCORE_YELLOW_COLOR,
        n_calibrated=len(calibrated_channels),
        calibrated_channels_text=_format_list_with_and(
            [f"'{c}'" for c in calibrated_channels]
        ),
        n_recommended=len(recommended_channels),
        recommendation_text=recommendation_text,
    )

  def _uncalibrated_channels_with_driver_issues(
      self,
  ) -> Mapping[str, list[str]]:
    """Returns a mapping of uncalibrated channel name to list of flagged issues."""
    status = self.channel_calibration_status or {}
    issues_by_channel = collections.defaultdict(list)
    for r in self.results:
      if isinstance(r, ImplausibleROICheckResult):
        for ch in r.high_roi_channels:
          if not status.get(ch, False):
            issues_by_channel[ch].append(constants.HIGH_ROI)
        for ch in r.low_roi_channels:
          if not status.get(ch, False):
            issues_by_channel[ch].append(constants.LOW_ROI)
      elif isinstance(r, HighVarianceCheckResult):
        for ch in r.high_variance_channels:
          if not status.get(ch, False):
            issues_by_channel[ch].append(constants.HIGH_VARIANCE)
      elif isinstance(r, PotentialBiasCheckResult):
        for ch in r.low_correlation_channels:
          if not status.get(ch, False):
            issues_by_channel[ch].append(constants.POTENTIAL_BIAS)
    ordered_channels = _order_channels_by_status(
        issues_by_channel.keys(), status
    )
    return {
        ch: list(dict.fromkeys(issues_by_channel[ch]))
        for ch in ordered_channels
    }

  _get_recommended_channels_with_issues = (
      _uncalibrated_channels_with_driver_issues
  )

  def _create_calibration_overview_card_html(self) -> str:
    """Creates the HTML snippet for the Calibration Overview Card."""
    if not self.calibration_overview_data:
      return ""
    sorted_data = sorted(
        self.calibration_overview_data,
        key=lambda ch: ch.spend or 0.0,
        reverse=True,
    )
    plotted_channels_data = sorted_data[
        : constants.MAX_CHANNELS_FOR_OVERVIEW_CARD
    ]
    n_total_channels = len(sorted_data)
    status = self.channel_calibration_status or {}
    ordered_channel_names = _order_channels_by_status(
        [ch.channel_name for ch in plotted_channels_data], status
    )
    channels_text = _format_list_with_and(
        [f"'{c}'" for c in ordered_channel_names]
    )

    overview_description = (
        "These plots display your incrementality experiments and their impact"
        f" on your Meridian priors for {channels_text}. The left column"
        " figure(s) display incrementality experiments and the intermediary"
        " prior for each channel. The middle column figure(s) display the"
        " intermediary prior and the parameterized prior that it produced. The"
        " right column figure(s) display the prior for each channel along with"
        " the trained Meridian posterior, which is a combination of your"
        " experiment-informed prior and your available data."
    )
    has_more_channels = (
        n_total_channels > constants.MAX_CHANNELS_FOR_OVERVIEW_CARD
    )
    has_more_experiments = any(
        len(ch.calibrated_output.experiments)
        > constants.MAX_EXPERIMENTS_FOR_OVERVIEW_CARD
        for ch in plotted_channels_data
        if ch.calibrated_output and ch.calibrated_output.experiments
    )

    if has_more_channels or has_more_experiments:
      # TODO: Add reference to how to plot more experiments.
      overview_description += (
          " The five highest-spend channels with experiments are plotted here"
          " along with their five experiments with the smallest adjusted"
          " standard error."
      )

    plotted_channels = []
    for idx, ch_data in enumerate(plotted_channels_data):
      plotted_channels.append({
          constants.CHANNEL_NAME: ch_data.channel_name,
          constants.CHART_ID: str(idx),
          constants.CHART_JSON: ch_data.chart_json or "",
      })

    template = self._template_env.get_template(
        "calibration_overview_card.html.jinja"
    )
    return template.render(
        overview_description=overview_description,
        plotted_channels=plotted_channels,
    )

  def _create_calibration_details_card_html(self) -> str:
    """Creates the HTML snippet for the Calibration Details Card."""
    if not self.calibration_overview_data:
      return ""
    details_data = list(self.calibration_overview_data)

    sorted_data = sorted(
        details_data, key=lambda ch: ch.spend or 0.0, reverse=True
    )
    valid_data = [ch for ch in sorted_data if ch.details_chart_json]
    if not valid_data:
      return ""
    plotted_channels_data = valid_data[
        : constants.MAX_CHANNELS_FOR_DETAILS_CARD
    ]
    n_total_channels = len(valid_data)

    details_description = (
        "These plots display your incrementality experiments and the"
        " adjustments we made for each experiment's spend, duration and"
        " recency, as well as the final mean and standard error used to"
        " inform the Meridian prior. Default adjustments are plotted instead of"
        " any values you didn't provide. The Meridian prior for each channel"
        " is a combination of the adjusted experiments for that channel and"
        " your baseline prior, if applicable."
    )

    has_more_channels = (
        n_total_channels > constants.MAX_CHANNELS_FOR_DETAILS_CARD
    )
    has_more_experiments = any(
        len(ch.calibrated_output.experiments)
        > constants.MAX_EXPERIMENTS_FOR_DETAILS_CARD
        for ch in plotted_channels_data
        if ch.calibrated_output and ch.calibrated_output.experiments
    )

    if has_more_channels or has_more_experiments:
      # TODO: Add reference to how to plot more experiments.
      details_description += (
          " The five highest-spend channels with experiments are plotted here"
          " along with their five experiments with the smallest adjusted"
          " standard error."
      )

    plotted_channels = []
    for idx, ch_data in enumerate(plotted_channels_data):
      plotted_channels.append({
          constants.CHANNEL_NAME: ch_data.channel_name,
          constants.CHART_ID: str(idx),
          constants.CHART_JSON: ch_data.details_chart_json or "",
      })

    template = self._template_env.get_template(
        "calibration_details_card.html.jinja"
    )
    return template.render(
        details_description=details_description,
        plotted_channels=plotted_channels,
    )

  def _get_check_data(self, result: CheckResult) -> Mapping[str, Any]:
    """Returns data for a health check."""
    check_data = {
        constants.NAME: self._get_check_name(result),
        constants.STATUS: result.case.status.name,
        constants.RECOMMENDATION: result.recommendation,
    }

    if isinstance(result, PriorPosteriorShiftCheckResult) or isinstance(
        result, ROIConsistencyCheckResult
    ):
      check_data[constants.TOTAL_CHANNELS] = len(result.channel_results)  # pyrefly: ignore[unsupported-operation]
      check_data[constants.PASSED_CHANNELS] = sum(  # pyrefly: ignore[unsupported-operation]
          1 for r in result.channel_results if r.case.status == Status.PASS
      )

    return check_data

  def _get_check_name(self, result: CheckResult) -> str:
    """Returns a readable name for the check."""
    name = result.__class__.__name__
    if name not in constants.CHECK_RESULT_NAME_MAP:
      raise ValueError(
          f"Check result {name} not found in CHECK_RESULT_NAME_MAP."
      )
    return constants.CHECK_RESULT_NAME_MAP[name]


NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION = (
    constants.NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION
)


def _format_channel_issue(channel: str, issues: Sequence[str]) -> str:
  """Formats flagged issues for a single channel."""
  unique_issues = list(dict.fromkeys(issues))
  if unique_issues == [constants.POTENTIAL_BIAS]:
    return f"'{channel}' shows {constants.POTENTIAL_BIAS}"
  return f"'{channel}' shows issues with {_format_list_with_and(unique_issues)}"


def _format_list_with_and(items: Sequence[str]) -> str:
  """Formats a list of strings into a natural language list with 'and'."""
  if not items:
    return ""
  if len(items) == 1:
    return items[0]
  if len(items) == 2:
    return f"{items[0]} and {items[1]}"
  return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_calibration_recommendation_text(
    recommended_channels: Sequence[str] | None = None,
    driver_issues_by_channel: Mapping[str, Sequence[str]] | None = None,
    location: str = constants.CALIBRATION_TEXT_CHANNEL_RECOMMENDATION,
    calibration_score: float | None = None,
) -> str:
  """Constructs the calibration recommendation message for a given location.

  Args:
    recommended_channels: List of uncalibrated channel names with score below
      threshold.
    driver_issues_by_channel: Mapping of uncalibrated channel names to their
      flagged driver issue names.
    location: One of constants.CALIBRATION_TEXT_METRICS_CHECK,
      constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY, or
      constants.CALIBRATION_TEXT_CHANNEL_RECOMMENDATION.
    calibration_score: Overall calibration score (used for metrics_check).

  Returns:
    The formatted recommendation string for the user.
  """
  channels = list(dict.fromkeys(recommended_channels or []))
  has_recommended = bool(channels)
  drivers_dict = driver_issues_by_channel or {}
  active_drivers = {
      ch: issues for ch, issues in drivers_dict.items() if issues
  }
  has_drivers = bool(active_drivers)

  rec_clause = None
  rec_sentence = None
  if has_recommended:
    rec_channels_formatted = _format_list_with_and([f"'{c}'" for c in channels])
    rec_clause = (
        "We recommend incrementality experiments to improve prior accuracy"
        f" for {rec_channels_formatted}"
    )
    rec_sentence = f"{rec_clause}."
  else:
    rec_sentence = constants.NO_CHANNELS_REQUIRE_CALIBRATION

  drivers_text = None
  if has_drivers:
    driver_sentences = [
        _format_channel_issue(ch, issues)
        for ch, issues in active_drivers.items()
    ]
    drivers_text = _format_list_with_and(driver_sentences)

  if location == constants.CALIBRATION_TEXT_METRICS_CHECK:
    if calibration_score is not None:
      score_prefix = (
          f"The overall calibration score is {calibration_score:.1f}/100."
      )
      return f"{score_prefix} {rec_sentence}"
    return rec_sentence

  if location == constants.CALIBRATION_TEXT_CALIBRATION_SUMMARY:
    if has_recommended:
      if has_drivers:
        return (
            f"{rec_clause}: {drivers_text}."
            f" {constants.SEE_CHANNEL_CALIBRATION_RECOMMENDATION_BELOW}"
        )
      return (
          f"{rec_clause}."
          f" {constants.SEE_CHANNEL_CALIBRATION_RECOMMENDATION_BELOW}"
      )
    return constants.NO_CHANNELS_REQUIRE_CALIBRATION

  if location == constants.CALIBRATION_TEXT_CHANNEL_RECOMMENDATION:
    if has_recommended:
      if has_drivers:
        return (
            f"{rec_clause}: {drivers_text}."
            f" {constants.SEE_CHANNEL_CALIBRATION_RECOMMENDATION_BELOW}"
        )
      return (
          f"{rec_clause}."
          f" {constants.SEE_CHANNEL_CALIBRATION_RECOMMENDATION_BELOW}"
      )
    if has_drivers:
      candidate_phrase = (
          "this channel may be a good candidate"
          if len(active_drivers) == 1
          else "these channels may be good candidates"
      )
      return (
          f"{constants.NO_CHANNELS_REQUIRE_CALIBRATION} However,"
          f" {drivers_text}. We recommend reviewing the table and plots below"
          f" to check if {candidate_phrase} for calibration via an"
          " incrementality experiment such as those run with Meridian GeoX."
      )
    return (
        f"{constants.NO_CHANNELS_REQUIRE_CALIBRATION}"
        f" {constants.REVIEW_BOUNDARIES_INFO_TEXT}"
    )

  raise ValueError(f"Unknown location: {location}")

