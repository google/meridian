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

"""Base structures for Meridian's calibration framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
import warnings

from meridian import backend
from meridian import constants
from meridian.model import prior_distribution
from meridian.model.calibration import constants as calibration_constants
from meridian.model.calibration import roi
import tensorflow as tf


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentResult:
  """The experiment result used for calibration.

  Attributes:
    point_estimate: The point estimate of the incrementality experiment result
      (e.g. ROI or IKPC).
    standard_error: The uncertainty of the incrementality experiment result.
      Must be positive.

  Raises:
    ValueError: If the experiment shows statistically significant negative lift
      as determined by the 95% confidence interval having a negative upper
      bound or if the standard error is not positive.
  """

  point_estimate: float
  standard_error: float

  def __post_init__(self) -> None:
    if self.standard_error is None or self.standard_error <= 0.0:
      raise ValueError(
          f"Standard error must be positive. Got: {self.standard_error}"
      )

    ci_upper_bound = (
        self.point_estimate
        + calibration_constants.CONFIDENCE_LEVEL_Z_SCORE_95
        * self.standard_error
    )
    if ci_upper_bound < 0.0:
      raise ValueError(
          "The experiment shows statistically significant negative lift:"
          " the 95% confidence interval has a negative upper bound "
          f" ({ci_upper_bound:.4f})."
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentInfo:
  """Incrementality experiment metadata for calibration.

  Attributes:
    total_spend: The total spend covered by the incrementality experiment.
    experiment_start_date: The start date of the incrementality experiment. The
      first day the experiment was active.
    experiment_end_date: The end date of the incrementality experiment. The last
      day the experiment was active.
  """

  total_spend: float
  experiment_start_date: datetime.date
  experiment_end_date: datetime.date

  def __post_init__(self) -> None:
    if self.total_spend is None or self.total_spend <= 0.0:
      raise ValueError(f"Total spend must be positive. Got: {self.total_spend}")
    if self.experiment_start_date >= self.experiment_end_date:
      raise ValueError(
          "Experiment start date must be before the experiment end date."
      )
    if self.experiment_end_date > datetime.date.today():
      warnings.warn(
          f"Experiment end date ({self.experiment_end_date}) is in the future. "
          "Calibration results using incomplete experiments may be unreliable "
          "or incorrect.",
          UserWarning,
      )

  @property
  def avg_daily_spend(self) -> float:
    """Returns the average spend per day of the experiment."""
    days = (self.experiment_end_date - self.experiment_start_date).days
    if days <= 0:
      raise ValueError(
          "Experiment duration must be at least 1 day. Got starting date"
          f" {self.experiment_start_date} and ending date"
          f" {self.experiment_end_date}."
      )
    return self.total_spend / days


class SourceType(enum.StrEnum):
  """Source type representing the origin of the calibration experiment."""

  MERIDIAN_GEOX = "MeridianGeoX"
  GENERIC = "Generic"


@dataclasses.dataclass(kw_only=True)
class CalibrationData:
  """Calibration input data structure for all incrementality experiment sources.

  Attributes:
    experiment_result: The `ExperimentResult` containing point estimate and
      standard error.
    experiment_info: The `ExperimentInfo` containing experiment details for
      calibration.
    point_estimate_adjustment: The optional point estimate (gamma) adjustment.
    standard_error_adjustment: The optional standard error (tau) adjustment.
    source_type: Source type representing the origin of the calibration
      experiment. Default is `SourceType.GENERIC`.
  """

  experiment_result: ExperimentResult
  experiment_info: ExperimentInfo
  point_estimate_adjustment: float | None = None
  standard_error_adjustment: float | None = None
  source_type: SourceType = SourceType.GENERIC


@dataclasses.dataclass(kw_only=True)
class CalibrationInput:
  """Channel-level container composing one or more incrementality experiment sources.

  Attributes:
    channel_name: Name of the channel for which the calibration information is
      associated.
    total_spend: The total spend of the channel in the Meridian model.
    baseline_prior: The `tfd.Distribution` representing the baseline prior for
      the KPI. If provided, the baseline prior will regularize the
      incrementality experiment results. Default is None. If None, an improper
      uniform distribution over the positive real numbers is used.
    adstock_decay_spec: Adstock decay specification used in duration
      adjustments. This can be either `'geometric'` or `'binomial'`, or a
      Mapping from channel name to the decay function string. Default is
      `'geometric'`.
    alpha: The decay rate parameter (between 0 and 1 inclusive) used for
      duration adjustments. It determines the proportion of the total media
      effect captured during the experiment window. This can be either a float
      or a Mapping from channel name to float. Default is `0.5`.
  """

  channel_name: str
  total_spend: float
  baseline_prior: backend.tfd.Distribution | None = None
  adstock_decay_spec: str = constants.GEOMETRIC_DECAY
  alpha: float = calibration_constants.DEFAULT_ALPHA
  _configs: list[CalibrationData] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )

  def __post_init__(self) -> None:
    if self.total_spend is None:
      raise ValueError("Total channel spend is required.")
    if self.total_spend <= 0.0:
      raise ValueError(
          f"Total channel spend must be positive. Got: {self.total_spend}"
          f" for channel {self.channel_name!r}."
      )

    if self.baseline_prior is not None:
      try:
        mean_val = self.baseline_prior.mean()
        var_val = self.baseline_prior.variance()
        log_prob_val = self.baseline_prior.log_prob(tf.cast([0.0], tf.float32))
        if not tf.math.reduce_all(tf.math.is_finite(mean_val)):
          raise ValueError(
              f"The baseline prior for channel {self.channel_name!r} is"
              f" invalid: mean is non-finite (got: {mean_val})."
          )
        if not tf.math.reduce_all(tf.math.is_finite(var_val)):
          raise ValueError(
              f"The baseline prior for channel {self.channel_name!r} is"
              f" invalid: variance is non-finite (got: {var_val})."
          )
        if tf.math.reduce_any(tf.math.is_nan(log_prob_val)):
          raise ValueError(
              f"The baseline prior for channel {self.channel_name!r} is"
              f" invalid: log_prob returns NaN (got: {log_prob_val})."
          )
      except (AttributeError, NotImplementedError, ValueError) as e:
        raise ValueError(
            f"The baseline prior for channel {self.channel_name!r} is invalid:"
            " failed to evaluate mean, variance, or log_prob on inputs."
        ) from e

  def add_calibration_data(self, config: CalibrationData) -> None:
    """Attaches a resolved `CalibrationData` to this channel container.

    Args:
      config: The `CalibrationData` to attach.

    Raises:
      ValueError: If the experiment spend exceeds the total channel spend.
    """
    if (
        self.total_spend is not None
        and config.experiment_info.total_spend is not None
        and config.experiment_info.total_spend > self.total_spend
    ):
      raise ValueError(
          f"Experiment spend ({config.experiment_info.total_spend}) cannot"
          f" exceed total channel spend ({self.total_spend}) for channel"
          f" {self.channel_name!r}."
      )
    self._configs.append(config)

  @property
  def configs(self) -> Sequence[CalibrationData]:
    """The registered calibration configs."""
    return self._configs


@dataclasses.dataclass(frozen=True, kw_only=True)
class CalibrationRegistryResult:
  """Contains the calibrated distributions and diagnostic outputs from the registry."""

  distributions: Mapping[str, backend.tfd.Distribution]
  outputs: Mapping[str, CalibrationOutput]


@dataclasses.dataclass(kw_only=True)
class CalibrationRegistry:
  """Central registry for managing calibration.

  This registry stores channel-level incrementality experiment results and
  information for calibration and delegates calibration calls to the appropriate
  calibration strategy.
  """

  _inputs: list[CalibrationInput] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )

  def add_input(self, data: CalibrationInput) -> None:
    """Registers calibration input for a channel.

    Args:
      data: The CalibrationInput to register.
    """
    self._inputs.append(data)

  def get_roi_distributions_by_channel(
      self,
      last_modeled_date: datetime.date,
      *,
      max_lag: int,
      interval_days: int,
      model_duration_days: int,
  ) -> CalibrationRegistryResult:
    """Retrieves calibrated ROI distributions for media and reach & frequency channels.

    Args:
      last_modeled_date: The last date of the modeled period in the Meridian
        model.
      max_lag: The maximum lag value in model intervals.
      interval_days: The interval size of the Meridian model time coordinates in
        days.
      model_duration_days: The number of days in the modeled period of the
        Meridian model.

    Returns:
      A `CalibrationRegistryResult` containing the mappings of channel names
      to ROI distributions and calibration outputs.

    Raises:
      ValueError: If a baseline prior was provided for a channel, but no
        experiments were found.
    """
    distributions = {}
    outputs = {}
    for cal_input in self._inputs:
      configs = cal_input.configs
      if cal_input.baseline_prior is not None and not configs:
        raise ValueError(
            "Baseline prior was provided for channel"
            f" {cal_input.channel_name!r}, but no experiments were found. A"
            " baseline prior can only be used to regularize active experiment"
            " results."
        )
      if configs:
        calibrated_prior, calibration_output = roi.get_calibrated_roi_prior(
            calibration_data=configs,
            channel_name=cal_input.channel_name,
            total_channel_spend=cal_input.total_spend,
            last_modeled_date=last_modeled_date,
            baseline_prior=cal_input.baseline_prior,
            adstock_decay_function=cal_input.adstock_decay_spec,
            alpha=cal_input.alpha,
            max_lag=max_lag,
            interval_days=interval_days,
            model_duration_days=model_duration_days,
        )
        distributions[cal_input.channel_name] = calibrated_prior
        outputs[cal_input.channel_name] = calibration_output
    return CalibrationRegistryResult(
        distributions=distributions, outputs=outputs
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CalibratedExperiment:
  """Calibration metrics and adjustments for a single experiment.

  Attributes:
    source_type: Source type representing the origin of the calibration
      experiment.
    raw_experiment_result: Experiment result before adjustments.
    adjusted_experiment_result: Experiment result after adjustments.
    tau_spend: Spend standard error adjustment.
    tau_recency: Recency standard error adjustment.
    tau_duration: Duration standard error adjustment.
    gamma_duration: Duration point estimate adjustment.
    user_point_estimate_adjustment: User-specified point estimate adjustment, or
      None.
    user_standard_error_adjustment: User-specified standard error adjustment, or
      None.
  """

  source_type: SourceType
  raw_experiment_result: ExperimentResult
  adjusted_experiment_result: ExperimentResult
  tau_spend: float
  tau_recency: float
  tau_duration: float
  gamma_duration: float
  user_point_estimate_adjustment: float | None = None
  user_standard_error_adjustment: float | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class CalibrationOutput:
  """Outputs and diagnostics for a channel's prior calibration.

  Attributes:
    channel_name: Name of the channel.
    experiments: A sequence of calibrated experiments.
    baseline_prior: The baseline prior distribution, or None.
    intermediary_prior: The prior distribution before parameterization.
    adstock_decay_spec: The adstock decay specification used during calibration.
      Default is 'geometric'.
    max_lag: The maximum lag value used during calibration. Default is 8.
  """

  channel_name: str
  experiments: Sequence[CalibratedExperiment] = dataclasses.field(
      default_factory=list
  )
  baseline_prior: backend.tfd.Distribution | None = None
  intermediary_prior: backend.tfd.Distribution
  adstock_decay_spec: str = constants.GEOMETRIC_DECAY
  max_lag: int = constants.DEFAULT_MAX_LAG


# TODO: Add serde support to this class.
class CalibratedDistribution(
    prior_distribution.IndependentMultivariateDistribution
):
  """Container for a joint distribution supporting calibration information.

  This class extends `IndependentMultivariateDistribution` to store calibration
  information for each channel in the joint distribution.

  Note: Not all channels in the joint distribution are necessarily calibrated.
  """

  def __init__(
      self,
      distributions: (
          Sequence[backend.tfd.Distribution] | backend.tfd.Distribution
      ),
      is_calibrated: Sequence[bool],
      calibration_outputs: Sequence[CalibrationOutput | None] | None = None,
      validate_args: bool = False,
      allow_nan_stats: bool = True,
      name: str | None = None,
  ):
    """Initializes a batch of independent distributions with calibration info.

    Args:
      distributions: List of `tfd.Distribution` from which to construct a
        multivariate distribution, or a single `tfd.Distribution`. If a single
        distribution is provided and it is scalar, it will be broadcasted to
        match the number of channels (length of `is_calibrated`). The
        distributions must have scalar or one dimensional batch shapes; the
        resulting batch shape will be the sum of the underlying batch shapes.
      is_calibrated: A sequence of booleans indicating if the corresponding
        channel is calibrated. Must be the same length as the total number of
        channels.
      calibration_outputs: A sequence of optional calibration outputs for each
        channel. If a channel is calibrated without using `CalibrationBuilder`,
        `is_calibrated` will be `True` and the calibration output will be
        `None`. In this case, no visualizations will be shown for calibration
        experiments (as for non-calibrated channels), but no channel calibration
        will be recommended (as for calibrated channels).
      validate_args: Python `bool`. When `True` distribution parameters are
        checked for validity despite possibly degrading runtime performance.
        When `False` invalid inputs may silently render incorrect outputs.
        Default value is `False`.
      allow_nan_stats: Python `bool`. When `True`, statistics (e.g., mean, mode,
        variance) use the value "`NaN`" to indicate the result is undefined.
        When `False`, an exception is raised if one or more of the statistic's
        batch members are undefined. Default value is `True`.
      name: Python `str` name prefixed to Ops created by this class. Default
        value is 'Calibrated' followed by the names of the underlying
        distributions.

    Raises:
      ValueError: Under the following conditions:
        - The length of `is_calibrated` does not match the total number of
          channels.
        - The length of `calibration_outputs` does not match the total number
          of channels.
        - A channel has a non-None calibration output but `is_calibrated` is
          False.
    """
    if isinstance(distributions, backend.tfd.Distribution):
      if distributions.is_scalar_batch():
        n_channels = len(is_calibrated)
        distributions = backend.tfd.BatchBroadcast(distributions, (n_channels,))
      distributions = [distributions]

    if name is None:
      name = "-".join([constants.CALIBRATED] + [d.name for d in distributions])

    super().__init__(
        distributions=distributions,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name,
    )

    total_channels = self.batch_shape[0]

    if len(is_calibrated) != total_channels:
      raise ValueError(
          f"is_calibrated length ({len(is_calibrated)}) must match total"
          f" number of channels ({total_channels})."
      )

    if calibration_outputs is None:
      calibration_outputs = [None] * total_channels
    elif len(calibration_outputs) != total_channels:
      raise ValueError(
          f"calibration_outputs length ({len(calibration_outputs)}) must match"
          f" total number of channels ({total_channels})."
      )

    for i, (is_cal, output) in enumerate(
        zip(is_calibrated, calibration_outputs)
    ):
      if output is not None and not is_cal:
        raise ValueError(
            f"Channel {i} has a non-None calibration output but is_calibrated"
            " is False."
        )

    self._is_calibrated = tuple(is_calibrated)
    self._calibration_outputs = tuple(calibration_outputs)
    self._parameters.update({
        constants.IS_CALIBRATED: self._is_calibrated,
        constants.CALIBRATION_OUTPUTS: self._calibration_outputs,
    })

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distributions=backend.util.BatchedComponentProperties(
            event_ndims=lambda self: [0 for _ in self.distributions]
        )
    )

  def get_calibration_status(self) -> tuple[bool, ...]:
    """Returns a tuple of booleans indicating if each channel is calibrated."""
    return self._is_calibrated

  @property
  def calibration_outputs(self) -> tuple[CalibrationOutput | None, ...]:
    """Returns a tuple of calibration outputs for each channel."""
    return self._calibration_outputs


@dataclasses.dataclass(frozen=True, kw_only=True)
class CalibratedPriors:
  """Calibrated prior distributions.

  Attributes:
    roi_m: The calibrated prior distribution for paid media channels.
    roi_rf: The calibrated prior distribution for reach & frequency channels.
  """

  roi_m: CalibratedDistribution | None = None
  roi_rf: CalibratedDistribution | None = None
