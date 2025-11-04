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
    super().__init__(status)
    self.message_template = message_template
    self.recommendation = recommendation


@dataclasses.dataclass(frozen=True)
class BaseResultData:
  """Base class for check result data."""

  case: BaseCase
  details: dict[str, Any]


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
    """Returns the check result message."""
    report_str = self.case.message_template.format(**self.details)
    if self.case.recommendation:
      return f"{report_str} {self.case.recommendation}"
    return report_str


# ==============================================================================
# Check: Convergence
# ==============================================================================
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
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class ConvergenceCheckResult(CheckResult):
  """The immutable result of the Convergence Check."""

  case: ConvergenceCases

  def __post_init__(self):
    if self.case == ConvergenceCases.CONVERGED and (
        constants.CONVERGENCE_THRESHOLD not in self.details
    ):
      raise ValueError(
          "The message template 'The model has likely converged, as all"
          " parameters have R-hat values < {convergence_threshold}'. is"
          " missing required formatting arguments: convergence_threshold."
          f" Details: {self.details}."
      )


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
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class BaselineCheckResult(CheckResult):
  """The immutable result of the Baseline Check."""

  case: BaselineCases

  def __post_init__(self):
    if self.case is BaselineCases.PASS:
      return
    if any(
        key not in self.details
        for key in (
            constants.NEGATIVE_BASELINE_PROB,
            constants.NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD,
            constants.NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD,
        )
    ):
      raise ValueError(
          "The message template is missing required formatting arguments:"
          " negative_baseline_prob, negative_baseline_prob_fail_threshold,"
          " negative_baseline_prob_review_threshold. Details:"
          f" {self.details}."
      )


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
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class BayesianPPPCheckResult(CheckResult):
  """The immutable result of the Bayesian Posterior Predictive P-value Check."""

  case: BayesianPPPCases

  def __post_init__(self):
    if constants.BAYESIAN_PPP not in self.details:
      raise ValueError(
          "The message template is missing required formatting arguments:"
          " bayesian_ppp. Details:"
          f" {self.details}."
      )


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
      (
          "R-squared = {r_squared:.4f}, MAPE = {mape:.4f}, and wMAPE ="
          " {wmape:.4f}."
      ),
      _GOODNESS_OF_FIT_PASS_RECOMMENDATION,
  )
  REVIEW = (
      Status.REVIEW,
      (
          "R-squared = {r_squared:.4f}, MAPE = {mape:.4f}, and wMAPE ="
          " {wmape:.4f}."
      ),
      _GOODNESS_OF_FIT_REVIEW_RECOMMENDATION,
  )

  def __init__(
      self,
      status: Status,
      message_template: str,
      recommendation: str | None,
  ):
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class GoodnessOfFitCheckResult(CheckResult):
  """The immutable result of the Goodness of Fit Check."""

  case: GoodnessOfFitCases

  def __post_init__(self):
    if any(
        key not in self.details
        for key in (
            constants.R_SQUARED,
            constants.MAPE,
            constants.WMAPE,
        )
    ):
      raise ValueError(
          "The message template is missing required formatting arguments:"
          " r_squared, mape, wmape. Details:"
          f" {self.details}."
      )


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
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class ROIConsistencyChannelResult(ChannelResult):
  """The immutable result of ROI Consistency Check for a single channel."""

  case: ROIConsistencyChannelCases


@dataclasses.dataclass(frozen=True)
class ROIConsistencyCheckResult(CheckResult):
  """The immutable result of model-level ROI Consistency Check."""

  case: ROIConsistencyAggregateCases
  channel_results: list[ROIConsistencyChannelResult]


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
    super().__init__(status, message_template, recommendation)


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftChannelResult(ChannelResult):
  """The result of Prior-Posterior Shift Check for a single channel."""

  case: PriorPosteriorShiftChannelCases


@dataclasses.dataclass(frozen=True)
class PriorPosteriorShiftCheckResult(CheckResult):
  """The immutable result of model-level Prior-Posterior Shift Check."""

  case: PriorPosteriorShiftAggregateCases
  channel_results: list[PriorPosteriorShiftChannelResult]


# ==============================================================================
# Review Summary
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class ReviewSummary:
  """The final summary of all model quality checks.

  Attributes:
    overall_status: The overall status of all checks.
    summary_message: A summary message of all checks.
    results: A list of all check results.
  """

  overall_status: Status
  summary_message: str
  results: list[CheckResult]

  def __repr__(self) -> str:
    report = []
    report.append("=" * 40)
    report.append("Model Quality Checks")
    report.append("=" * 40)
    report.append(f"Overall Status: {self.overall_status.name}")
    report.append(f"Summary: {self.summary_message}")
    report.append("\nCheck Results:")

    for result in self.results:
      name = result.__class__.__name__
      if name.endswith("CheckResult"):
        title = name[: -len("CheckResult")]
      else:
        title = name

      report.append("-" * 40)
      report.append(f"{title} Check:")
      report.append(f"  Status: {result.case.status.name}")
      report.append(f"  Recommendation: {result.recommendation}")

    return "\n".join(report)

  @property
  def checks_status(self) -> dict[str, str]:
    """Returns a dictionary of check names and statuses."""
    return {
        result.__class__.__name__: result.case.status.name
        for result in self.results
    }
