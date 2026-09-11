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

"""Defines model specification parameters for Meridian."""

from collections.abc import Collection, Mapping
import dataclasses
import datetime
import enum
from typing import Sequence
import warnings
from meridian import constants
from meridian.data import time_coordinates
from meridian.model import prior_distribution
from meridian.model.calibration import base as calibration_base
import numpy as np

__all__ = [
    "CalibrationSpec",
    "ChannelCalibrationSpec",
    "DateRange",
    "GeoHoldoutSpec",
    "HoldoutSpec",
    "ModelSpec",
    "RandomHoldoutSpec",
    "SaturationType",
]


class SaturationType(enum.Enum):
  HILL = "hill"
  NONE = "none"


@dataclasses.dataclass(frozen=True)
class DateRange:
  """A date range `[start_date, end_date]` with both bounds inclusive.

  This matches the convention used throughout Meridian's other date selection
  APIs -- notably `TimeCoordinates.get_selected_dates` and
  `expand_selected_time_dims` -- so that the same pair of dates selects the
  same time periods wherever it is used.

  Either bound may be omitted to leave that side open. An omitted bound is
  resolved against the input data when the range is compiled, not at
  construction time, so the same `DateRange` may cover different dates for
  different datasets:

  - `DateRange(start, None)`: from `start` through the last date in the data.
  - `DateRange(None, end)`: from the first date in the data through `end`.
  - `DateRange()`: the entire date range of the data. This is the identity
    range; specifying it is equivalent to applying no date restriction at all.

  A range whose bounds are equal is valid and selects that single date.

  A bound that is present must name one of the input data's time coordinates
  exactly; a bound falling between two coordinates is rejected. Because it
  depends on the data, the check happens when the range is compiled against an
  `InputData`, not at construction time.

  Note:
    The `mmm.v1.common.DateInterval` proto that these specs serialize to uses
    the opposite, *half-open* `[start_date, end_date)` convention. The
    conversion between the two is performed in the serialization layer, which
    is the only place the two conventions meet.

  Attributes:
    start_date: The start date (inclusive). Can be a `datetime.date`,
      `datetime.datetime`, `np.datetime64`, or an ISO-formatted string
      ('YYYY-MM-DD'). If None, implies all dates up to `end_date`.
    end_date: The end date (inclusive). Can be a `datetime.date`,
      `datetime.datetime`, `np.datetime64`, or an ISO-formatted string
      ('YYYY-MM-DD'). If None, implies all dates from `start_date` onwards.
  """

  start_date: time_coordinates.Date | None = None
  end_date: time_coordinates.Date | None = None

  def __post_init__(self) -> None:
    start = (
        time_coordinates.normalize_date(self.start_date)
        if self.start_date is not None
        else None
    )
    end = (
        time_coordinates.normalize_date(self.end_date)
        if self.end_date is not None
        else None
    )
    if start is not None and end is not None and start > end:
      raise ValueError(f"`start_date` ({start}) must be <= `end_date` ({end}).")
    object.__setattr__(self, "start_date", start)
    object.__setattr__(self, "end_date", end)

  @classmethod
  def from_date_interval(
      cls, date_interval: time_coordinates.DateInterval
  ) -> "DateRange":
    """Creates a `DateRange` from a `time_coordinates.DateInterval` tuple.

    The tuple is interpreted with both bounds *inclusive*, the convention
    `TimeCoordinates.get_selected_dates` uses. It is therefore not
    interchangeable with the half-open `mmm.v1.common.DateInterval` proto of
    the same name.

    Args:
      date_interval: An inclusive `(start_date, end_date)` tuple.

    Returns:
      The equivalent `DateRange`.
    """
    return cls(start_date=date_interval[0], end_date=date_interval[1])

  @property
  def date_interval(self) -> tuple[datetime.date | None, datetime.date | None]:
    """Returns a normalized, inclusive `(start_date, end_date)` tuple."""
    return (self.start_date, self.end_date)  # pyrefly: ignore[bad-return]


@dataclasses.dataclass(frozen=True)
class ChannelCalibrationSpec:
  """Calibration specification for specific media channels.

  Attributes:
    channels: Sequence of channel names to calibrate.
    date_ranges: Sequence of `DateRange`s during which calibration applies.
  """

  channels: Sequence[str]
  date_ranges: Sequence[DateRange]

  def __post_init__(self) -> None:
    if isinstance(self.channels, str):
      raise ValueError(
          "`channels` must be a sequence of strings, not a single string."
      )
    if not self.channels:
      raise ValueError("`channels` cannot be empty.")
    if not self.date_ranges:
      raise ValueError("`date_ranges` cannot be empty.")
    object.__setattr__(self, "channels", tuple(self.channels))
    object.__setattr__(self, "date_ranges", tuple(self.date_ranges))


@dataclasses.dataclass(frozen=True)
class CalibrationSpec:
  """Specifies ROI calibration periods for media channels.

  Can be either:
  - A sequence of `DateRange` objects applying globally to all channels.
  - A sequence of `ChannelCalibrationSpec` objects specifying date ranges per
    channel.

  The two forms are mutually exclusive: a single `CalibrationSpec` cannot mix
  global and per-channel entries, because the scope of the resulting
  calibration would be ambiguous.

  Attributes:
    spec: A sequence of `DateRange` (global) or `ChannelCalibrationSpec`
      (per-channel).
  """

  spec: Sequence[DateRange] | Sequence[ChannelCalibrationSpec]

  def __post_init__(self) -> None:
    if not self.spec:
      raise ValueError("`spec` cannot be empty.")
    spec = tuple(self.spec)
    if not (
        all(isinstance(s, DateRange) for s in spec)
        or all(isinstance(s, ChannelCalibrationSpec) for s in spec)
    ):
      raise ValueError(
          "`spec` must be either a sequence of `DateRange` (applying globally"
          " to all channels) or a sequence of `ChannelCalibrationSpec`"
          " (per-channel); the two cannot be mixed."
      )
    object.__setattr__(self, "spec", spec)


@dataclasses.dataclass(frozen=True)
class GeoHoldoutSpec:
  """Holdout specification for specific geos.

  Attributes:
    geos: Sequence of geo names/identifiers to hold out.
    date_ranges: Sequence of `DateRange`s during which the holdout applies.
  """

  geos: Sequence[str]
  date_ranges: Sequence[DateRange]

  def __post_init__(self) -> None:
    if isinstance(self.geos, str):
      raise ValueError(
          "`geos` must be a sequence of strings, not a single string."
      )
    if not self.geos:
      raise ValueError("`geos` cannot be empty.")
    if not self.date_ranges:
      raise ValueError("`date_ranges` cannot be empty.")
    object.__setattr__(self, "geos", tuple(self.geos))
    object.__setattr__(self, "date_ranges", tuple(self.date_ranges))


@dataclasses.dataclass(frozen=True)
class RandomHoldoutSpec:
  """Random holdout specification.

  This specification is *input only*: it declares the intent to hold out a
  random subset of observations, not the subset itself. The framework draws the
  random sample exactly once, when the model is compiled against its input
  data, producing a holdout mask over `(n_geos, n_times)` that becomes part of
  the model's state. All downstream computation reads that resolved holdout,
  never this specification.

  Both this specification and the resolved draw are serialized: the former as
  provenance recording how the holdout was requested, the latter as the
  authoritative record of what was actually held out. The resolved draw is
  stored declaratively in `HoldoutSpec.resolved`, as the geos and date ranges
  that were selected, rather than as a raw mask. On deserialization it always
  takes precedence and is never re-drawn, because `seed` alone does not
  reproduce a draw across computational backends (TensorFlow vs. JAX) or
  library versions.

  If a serialized model carries only this specification and no resolved draw,
  the draw is performed on load and a warning is emitted. The resulting holdout
  will not match the one used during the original fit. Such a model remains
  valid for inference, since the holdout does not participate in inference, but
  its train/test predictive accuracy metrics are not meaningful.

  Attributes:
    ratio: The fraction of observations to hold out. Must be strictly between
      0.0 and 1.0.
    seed: Optional random seed for deterministic holdout generation.
  """

  ratio: float
  seed: int | None = None

  def __post_init__(self) -> None:
    ratio = float(self.ratio)
    if not 0.0 < ratio < 1.0:
      raise ValueError(
          f"`ratio` must be strictly between 0.0 and 1.0, got: {self.ratio}."
      )
    object.__setattr__(self, "ratio", ratio)
    if self.seed is not None:
      object.__setattr__(self, "seed", int(self.seed))


@dataclasses.dataclass(frozen=True)
class HoldoutSpec:
  """Holdout specification for training/evaluation split.

  `spec` declares the holdout *intent* and can be one of:

  - Sequence of `DateRange`s (global holdout for all geos during those dates).
  - Sequence of `GeoHoldoutSpec`s (per-geo holdout date ranges).
  - `RandomHoldoutSpec` (random holdout ratio and optional seed).

  The first two are deterministic: they resolve to the same holdout every time
  they are applied to the same input data. `RandomHoldoutSpec` is not, so the
  draw it produces is recorded in `resolved`. See `RandomHoldoutSpec` for the
  full lifecycle.

  The two sequence forms are mutually exclusive: a single `HoldoutSpec` cannot
  mix global and per-geo entries, because the scope of the resulting holdout
  would be ambiguous.

  Attributes:
    spec: `Sequence[DateRange]`, `Sequence[GeoHoldoutSpec]`, or
      `RandomHoldoutSpec`.
    resolved: The materialized result of a non-deterministic `spec`, expressed
      declaratively as per-geo date ranges. Set only when `spec` is a
      `RandomHoldoutSpec`; the deterministic variants reproduce their holdout
      exactly from `spec`, so this must be left as `None` for them. When set,
      this is authoritative and is never re-drawn. Default: `None`.
  """

  spec: Sequence[DateRange] | Sequence[GeoHoldoutSpec] | RandomHoldoutSpec
  resolved: Sequence[GeoHoldoutSpec] | None = None

  def __post_init__(self) -> None:
    if self.resolved is not None:
      if not isinstance(self.spec, RandomHoldoutSpec):
        raise ValueError(
            "`resolved` can only be set when `spec` is a `RandomHoldoutSpec`;"
            " deterministic holdout specifications reproduce their holdout"
            " from `spec` alone."
        )
      if not self.resolved:
        raise ValueError("`resolved` cannot be empty.")
      object.__setattr__(self, "resolved", tuple(self.resolved))
    if isinstance(self.spec, RandomHoldoutSpec):
      return
    if not self.spec:
      raise ValueError("`spec` cannot be empty.")
    spec = tuple(self.spec)
    if not (
        all(isinstance(s, DateRange) for s in spec)
        or all(isinstance(s, GeoHoldoutSpec) for s in spec)
    ):
      raise ValueError(
          "`spec` must be either a sequence of `DateRange` (a global holdout"
          " applying to all geos) or a sequence of `GeoHoldoutSpec` (per-geo"
          " holdouts); the two cannot be mixed."
      )
    object.__setattr__(self, "spec", spec)


def _validate_roi_calibration_period(
    array: np.ndarray | None,
    array_name: str,
    channel_dim_name: str,
    prior_type: str,
    prior_type_name: str,
) -> None:
  """Validates the ROI calibration period array."""
  if array is None:
    return
  if prior_type != constants.TREATMENT_PRIOR_TYPE_ROI:
    raise ValueError(
        f"The `{array_name}` should be `None` unless `{prior_type_name}` is"
        f" '{constants.TREATMENT_PRIOR_TYPE_ROI}'."
    )
  if len(array.shape) != 2:
    raise ValueError(
        f"The shape of the `{array_name}` array {array.shape} should be"
        f" 2-dimensional (`n_media_times` x `{channel_dim_name}`)."
    )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Model specification parameters for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.

  Attributes:
    prior: A `PriorDistribution` object specifying the prior distribution of
      each set of model parameters. The distribution for a vector of parameters
      (for example, `alpha_m`) can be passed as either a scalar distribution or
      a vector distribution. If a scalar distribution is passed, it is broadcast
      to the actual shape of the parameter vector. See `paid_media_prior_type`
      for related details.
    media_effects_dist: A string to specify the distribution of media random
      effects across geos. This attribute is not used with a national-level
      model. Allowed values: `'normal'` or `'log_normal'`. Default:
      `'log_normal'`.
    hill_before_adstock: A boolean indicating whether to apply the Hill function
      before the Adstock function, instead of the default order of Adstock
      before Hill. This argument does not apply to RF channels. Default:
      `False`.
    max_lag: An integer indicating the maximum number of lag periods (≥ `0`) to
      include in the Adstock calculation. Default: `8`.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo. If `False`, then a single residual
      variance is used for all geos. Default: `False`.
    media_prior_type: A string to specify the prior type for the media
      coefficients. Allowed values: `'roi'`, `'mroi'`, `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `roi_m`, `mroi_m`,
      `contribution_m`, and `beta_m`, but only one of these is used depending on
      the `media_prior_type`. When `media_prior_type` is `'roi'`, the
      `PriorDistribution.roi_m` parameter is used to specify a prior on the ROI.
      When `media_prior_type` is `'mroi'`, the `PriorDistribution.mroi_m`
      parameter is used to specify a prior on the mROI. When `media_prior_type`
      is `'contribution'`, the `PriorDistribution.contribution_m` parameter is
      used to specify a prior on the contribution. When `media_prior_type` is
      `'coefficient'`, the `PriorDistribution.beta_m` parameter is used to
      specify a prior on the coefficient mean parameters. Default: `'roi'`.
    rf_prior_type: A string to specify the prior type for the RF coefficients.
      Allowed values: `'roi'`, `'mroi'`, `'contribution'`, `'coefficient'`. The
      `PriorDistribution` contains distributions `roi_rf`, `mroi_rf`,
      `contribution_rf`, and`beta_rf`, but only one of these is used depending
      on the `rf_prior_type`. When `rf_prior_type` is `'roi'`, the
      `PriorDistribution.roi_rf` parameter is used to specify a prior on the
      ROI. When `rf_media_prior_type` is `'mroi'`, the
      `PriorDistribution.mroi_rf` parameter is used to specify a prior on the
      mROI. When `rf_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_rf` parameter is used to specify a prior
      on the contribution. When `rf_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_rf` parameter is used to specify a prior on the
      coefficient mean parameters. Default: `'roi'`.
    paid_media_prior_type: Deprecated. Use `media_prior_type` and
      `rf_prior_type` instead. A string to specify the prior type for media and
      RF treatments at the same time. Ignored when `media_prior_type` or
      `rf_prior_type` are set. Default: `'roi'`.
    roi_calibration: An optional `CalibrationSpec` specifying the subset of time
      that the ROI value of the `roi_m` prior applies to. Only used if
      `media_prior_type` is `'roi'`. Default: `None`.
    roi_calibration_period: Deprecated. Use `roi_calibration` instead. If both
      are set, this field takes precedence and a warning is emitted. An optional
      boolean array of shape `(n_media_times, n_media_channels)` indicating the
      subset of `time` that the ROI value of the `roi_m` prior applies to. The
      ROI numerator is the incremental outcome generated by media executed
      during the calibration period. More precisely, it is the difference in
      expected outcome between the counterfactual where media is set to
      historical values versus the counterfactual where media is set to zero
      during the calibration period and set to historical values for all other
      time periods. The denominator is the channel spend during calibration
      period (excluding any calibration time periods prior to the first KPI time
      period). Spend data by time period is required. If `None`, all times are
      used. Only used if `media_prior_type` is `'roi'`. Default: `None`.
    rf_roi_calibration: An optional `CalibrationSpec` specifying the subset of
      time that the ROI value of the `roi_rf` prior applies to. Only used if
      `rf_prior_type` is `'roi'`. Default: `None`.
    rf_roi_calibration_period: Deprecated. Use `rf_roi_calibration` instead. If
      both are set, this field takes precedence and a warning is emitted. An
      optional boolean array of shape `(n_media_times, n_rf_channels)`
      indicating the subset of `time` that the ROI value of the `roi_rf` prior
      applies to. The ROI numerator is the incremental outcome generated by
      media executed during the calibration period. More precisely, it is the
      difference in expected outcome between the counterfactual where reach and
      frequency is set to historical values versus the counterfactual where
      reach is set to zero during the calibration period and set to historical
      values for all other time periods. The denominator is the channel spend
      during calibration period (excluding any calibration time periods prior to
      the first KPI time period). Spend data by time period is required. If
      `None`, all times are used. Only used if `rf_prior_type` is `'roi'`.
      Default: `None`.
    organic_media_prior_type: A string to specify the prior type for the organic
      media coefficients. Allowed values: `'contribution'`, `'coefficient'`.
      `PriorDistribution` contains `contribution_om` and `beta_om`, but only one
      of these is used depending on the `organic_media_prior_type`. When
      `organic_media_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_om` parameter is used to specify a prior
      on the contribution. When `organic_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.beta_om` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    organic_rf_prior_type: A string to specify the prior type for the organic
      reach and frequency coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains distributions
      `contribution_orf`, and `beta_orf`, but only one of these is used
      depending on the `organic_rf_prior_type`. When `organic_rf_prior_type` is
      `'contribution'`, the `PriorDistribution.contribution_orf` parameter is
      used to specify a prior on the contribution. When `organic_rf_prior_type`
      is `'coefficient'`, the `PriorDistribution.beta_orf` parameter is used to
      specify a prior on the coefficient mean parameters. Default:
      `'contribution'`.
    non_media_treatments_prior_type: A string to specify the prior type for the
      non-media treatment coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. `PriorDistribution` contains `contribution_n` and
      `gamma_n`, but only one of these is used depending on the
      `non_media_prior_type`. When `non_media_prior_type` is `'contribution'`,
      the `PriorDistribution.contribution_n` parameter is used to specify a
      prior on the contribution. When `non_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.gamma_n` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    non_media_baseline_values: Optional mapping from non-media channel names to
      baseline values (`Mapping[str, float | str]`), or (deprecated) list with
      the shape `(n_non_media_channels,)`. Each element is either a float (which
      means that the fixed value will be used as baseline for the given channel)
      or one of the strings `"min"` or `"max"` (which mean that the global
      minimum or maximum value will be used as baseline for the scaled values of
      the given non_media treatments channel). If `None`, the minimum value is
      used as baseline for each non-media treatments channel. This attribute is
      used as the default value for the corresponding argument to `Analyzer`
      methods.
    knots: An optional integer or collection of integers indicating the knots
      used to estimate time effects. When `knots` is a collection of integers,
      the knot locations are provided by that list. Zero corresponds to a knot
      at the first time period, one corresponds to a knot at the second time
      period, ..., and `(n_times - 1)` corresponds to a knot at the last time
      period). Typically, we recommend including knots at `0` and `(n_times -
      1)`, but this is not required. When `knots` is an integer, then there are
      knots with locations equally spaced across the time periods, (including
      knots at zero and `(n_times - 1)`. When `knots` is` 1`, there is a single
      common regression coefficient used for all time periods. If `knots` is set
      to `None`, then the numbers of knots used is equal to the number of time
      periods in the case of a geo model. This is equivalent to each time period
      having its own regression coefficient. If `knots` is set to `None` in the
      case of a national model, then the number of knots used is `1`. Default:
      `None`.
    baseline_geo: An optional integer or a string for the baseline geo. The
      baseline geo is treated as the reference geo in the dummy encoding of
      geos. Non-baseline geos have a corresponding `tau_g` indicator variable,
      meaning that they have a higher prior variance than the baseline geo. When
      set to `None`, the geo with the biggest population is used as the
      baseline. Default: `None`.
    holdout: An optional `HoldoutSpec` indicating which observations are part of
      the holdout sample, which are excluded from the training sample. Only KPI
      data is excluded from the training sample. Media data is still included as
      it can affect Adstock for subsequent weeks. If "ROI priors" are used, then
      the `roi_m` parameters correspond to the ROI of all geos and times, even
      those in the holdout sample. Default: `None`.
    holdout_id: Deprecated. Use `holdout` instead. If both are set, this field
      takes precedence and a warning is emitted. Optional boolean tensor of
      dimensions `(n_geos, n_times)` for a geo-level model or `(n_times,)` for a
      national model, indicating which observations are part of the holdout
      sample, which are excluded from the training sample. Only KPI data is
      excluded from the training sample. Media data is still included as it can
      affect Adstock for subsequent weeks. If "ROI priors" are used, then the
      `roi_m` parameters correspond to the ROI of all geos and times, even those
      in the holdout sample.
    population_scaled_controls: An optional sequence of control variable names
      for which the control value will be scaled by population. If `None`, no
      control variables are scaled by population. Default: `None`.
    control_population_scaling_id: Deprecated. Use `population_scaled_controls`
      instead. If both are set, this field takes precedence and a warning is
      emitted. An optional boolean tensor of dimension `(n_controls,)`
      indicating the control variables for which the control value will be
      scaled by population. If `None`, no control variables are scaled by
      population. Default: `None`.
    population_scaled_non_media_channels: An optional sequence of non-media
      channel names for which the non-media value will be scaled by population.
      If `None`, then no non-media variables are scaled by population. Default:
      `None`.
    non_media_population_scaling_id: Deprecated. Use
      `population_scaled_non_media_channels` instead. If both are set, this
      field takes precedence and a warning is emitted. An optional boolean
      tensor of dimension `(n_non_media_channels,)` indicating the non-media
      variables for which the non-media value will be scaled by population. If
      `None`, then no non-media variables are scaled by population. Default:
      `None`.
    adstock_decay_spec: A string or mapping specifying the adstock decay
      function for each media, RF, organic media and organic RF channel. If a
      string, must be either `'geometric'` or `'binomial'`, specifying that
      decay function for all channels. If a mapping, keys should be channel
      names and values should be `'geometric'` or `'binomial'`, with each
      key-value pair denoting the adstock decay function to use for that
      channel. Channels that are not specified in the mapping default to using
      'geometric'. Default: `'geometric'`.
    saturation_spec: A string or mapping specifying the saturation function for
      each media, RF, organic media and organic RF channel. If a string, must be
      either `'hill'` or `'none'`, specifying that saturation function for all
      channels. If a mapping, keys should be channel names and values should be
      `'hill'` or `'none'`, with each key-value pair denoting the saturation
      function to use for that channel. Channels that are not specified in the
      mapping default to using `'hill'`. Default: `'hill'`.
    enable_aks: A boolean indicating whether to use the Automatic Knot Selection
      algorithm to select an optimal number of knots for running the model
      instead of the default 1 for national models and n_times for geo models.
      If this is set to `True` and the `knots` arg is provided, then an error
      will be raised. Default: `False`.
  """

  prior: prior_distribution.PriorDistribution = dataclasses.field(
      default_factory=prior_distribution.PriorDistribution,
  )
  media_effects_dist: str = constants.MEDIA_EFFECTS_LOG_NORMAL
  hill_before_adstock: bool = False
  max_lag: int = constants.DEFAULT_MAX_LAG
  unique_sigma_for_each_geo: bool = False
  media_prior_type: str | None = None
  rf_prior_type: str | None = None
  paid_media_prior_type: str | None = None
  roi_calibration: CalibrationSpec | None = None
  roi_calibration_period: np.ndarray | None = None
  rf_roi_calibration: CalibrationSpec | None = None
  rf_roi_calibration_period: np.ndarray | None = None
  organic_media_prior_type: str = constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  organic_rf_prior_type: str = constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  non_media_treatments_prior_type: str = (
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  )
  non_media_baseline_values: (
      Mapping[str, float | str] | Sequence[float | str] | None
  ) = None
  knots: int | Collection[int] | None = None
  baseline_geo: int | str | None = None
  holdout: HoldoutSpec | None = None
  holdout_id: np.ndarray | None = None
  population_scaled_controls: Sequence[str] | None = None
  control_population_scaling_id: np.ndarray | None = None
  population_scaled_non_media_channels: Sequence[str] | None = None
  non_media_population_scaling_id: np.ndarray | None = None
  adstock_decay_spec: str | Mapping[str, str] = constants.GEOMETRIC_DECAY
  saturation_spec: str | Mapping[str, str] = constants.HILL
  enable_aks: bool = False

  def __post_init__(self) -> None:
    # Validate media_effects_dist.
    if self.media_effects_dist not in constants.MEDIA_EFFECTS_DISTRIBUTIONS:
      raise ValueError(
          f"The `media_effects_dist` parameter '{self.media_effects_dist}' must"
          f" be one of {sorted(constants.MEDIA_EFFECTS_DISTRIBUTIONS)}."
      )
    # Support paid_media_prior_type for backwards compatibility.
    if self.paid_media_prior_type is not None:
      if self.media_prior_type is not None or self.rf_prior_type is not None:
        raise ValueError(
            "The deprecated `paid_media_prior_type` parameter cannot be used"
            " with `media_prior_type` or `rf_prior_type`. Use"
            " `media_prior_type` and `rf_prior_type` instead."
        )
      else:
        warnings.warn(
            "Using `paid_media_prior_type` parameter will set prior types for"
            " media and RF at the same time. This is deprecated and will be"
            " removed in a future version of Meridian. Use `media_prior_type`"
            " and `rf_prior_type` instead."
        )
    # Validate prior_type.
    if (
        self.effective_media_prior_type
        not in constants.PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `media_prior_type` parameter"
          f" '{self.effective_media_prior_type}' must be one of"
          f" {sorted(constants.PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.effective_rf_prior_type not in constants.PAID_TREATMENT_PRIOR_TYPES:
      raise ValueError(
          "The `rf_prior_type` parameter"
          f" '{self.effective_rf_prior_type}' must be one of"
          f" {sorted(constants.PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.organic_media_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_media_prior_type` parameter"
          f" '{self.organic_media_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.organic_rf_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_rf_prior_type` parameter"
          f" '{self.organic_rf_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.non_media_treatments_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `non_media_treatments_prior_type` parameter"
          f" '{self.non_media_treatments_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )

    if constants.TREATMENT_PRIOR_TYPE_COEFFICIENT in (
        self.effective_media_prior_type,
        self.effective_rf_prior_type,
        self.organic_media_prior_type,
        self.organic_rf_prior_type,
        self.non_media_treatments_prior_type,
    ):
      warnings.warn(
          "Using coefficient priors"
          f" (`{constants.TREATMENT_PRIOR_TYPE_COEFFICIENT}`) is not"
          " recommended. Coefficient priors can lead to noisy estimates, lack"
          " of channel parity, and are harder to interpret than ROI or"
          " contribution priors. Consider using ROI priors"
          f" (`{constants.TREATMENT_PRIOR_TYPE_ROI}`), mROI priors"
          f" (`{constants.TREATMENT_PRIOR_TYPE_MROI}`), or contribution priors"
          f" (`{constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION}`) instead."
      )

    # Validate roi_calibration vs roi_calibration_period.
    if (
        self.roi_calibration is not None
        and self.roi_calibration_period is not None
    ):
      warnings.warn(
          "Both `roi_calibration` and the deprecated `roi_calibration_period`"
          " were specified. `roi_calibration_period` takes precedence for"
          " backward compatibility; a future version of Meridian will ignore"
          " it in favor of `roi_calibration`.",
          UserWarning,
          stacklevel=2,
      )
    if self.roi_calibration_period is not None:
      warnings.warn(
          "`roi_calibration_period` is deprecated and will be removed in a"
          " future version of Meridian. Use `roi_calibration` instead.",
          DeprecationWarning,
          stacklevel=2,
      )
    if self.roi_calibration is not None:
      if self.effective_media_prior_type != constants.TREATMENT_PRIOR_TYPE_ROI:
        raise ValueError(
            "The `roi_calibration` should be `None` unless `media_prior_type`"
            f" is '{constants.TREATMENT_PRIOR_TYPE_ROI}'."
        )
    _validate_roi_calibration_period(
        array=self.roi_calibration_period,
        array_name="roi_calibration_period",
        channel_dim_name="n_media_channels",
        prior_type=self.effective_media_prior_type,
        prior_type_name="media_prior_type",
    )

    # Validate rf_roi_calibration vs rf_roi_calibration_period.
    if (
        self.rf_roi_calibration is not None
        and self.rf_roi_calibration_period is not None
    ):
      warnings.warn(
          "Both `rf_roi_calibration` and the deprecated"
          " `rf_roi_calibration_period` were specified."
          " `rf_roi_calibration_period` takes precedence for backward"
          " compatibility; a future version of Meridian will ignore it in"
          " favor of `rf_roi_calibration`.",
          UserWarning,
          stacklevel=2,
      )
    if self.rf_roi_calibration_period is not None:
      warnings.warn(
          "`rf_roi_calibration_period` is deprecated and will be removed in a"
          " future version of Meridian. Use `rf_roi_calibration` instead.",
          DeprecationWarning,
          stacklevel=2,
      )
    if self.rf_roi_calibration is not None:
      if self.effective_rf_prior_type != constants.TREATMENT_PRIOR_TYPE_ROI:
        raise ValueError(
            "The `rf_roi_calibration` should be `None` unless `rf_prior_type`"
            f" is '{constants.TREATMENT_PRIOR_TYPE_ROI}'."
        )
    _validate_roi_calibration_period(
        array=self.rf_roi_calibration_period,
        array_name="rf_roi_calibration_period",
        channel_dim_name="n_rf_channels",
        prior_type=self.effective_rf_prior_type,
        prior_type_name="rf_prior_type",
    )

    # Validate holdout vs holdout_id.
    if self.holdout is not None and self.holdout_id is not None:
      warnings.warn(
          "Both `holdout` and the deprecated `holdout_id` were specified."
          " `holdout_id` takes precedence for backward compatibility; a future"
          " version of Meridian will ignore it in favor of `holdout`.",
          UserWarning,
          stacklevel=2,
      )
    if self.holdout_id is not None:
      warnings.warn(
          "`holdout_id` is deprecated and will be removed in a future version"
          " of Meridian. Use `holdout` instead.",
          DeprecationWarning,
          stacklevel=2,
      )

    # Validate population_scaled_controls vs control_population_scaling_id.
    if (
        self.population_scaled_controls is not None
        and self.control_population_scaling_id is not None
    ):
      warnings.warn(
          "Both `population_scaled_controls` and the deprecated"
          " `control_population_scaling_id` were specified."
          " `control_population_scaling_id` takes precedence for backward"
          " compatibility; a future version of Meridian will ignore it in"
          " favor of `population_scaled_controls`.",
          UserWarning,
          stacklevel=2,
      )
    if self.control_population_scaling_id is not None:
      warnings.warn(
          "`control_population_scaling_id` is deprecated and will be removed"
          " in a future version of Meridian. Use `population_scaled_controls`"
          " instead.",
          DeprecationWarning,
          stacklevel=2,
      )
    if self.population_scaled_controls is not None:
      if isinstance(self.population_scaled_controls, str):
        raise ValueError(
            "`population_scaled_controls` must be a sequence of strings, not a"
            " single string."
        )
      object.__setattr__(
          self,
          "population_scaled_controls",
          tuple(self.population_scaled_controls),
      )

    # Validate population_scaled_non_media_channels vs non_media_population_scaling_id.
    if (
        self.population_scaled_non_media_channels is not None
        and self.non_media_population_scaling_id is not None
    ):
      warnings.warn(
          "Both `population_scaled_non_media_channels` and the deprecated"
          " `non_media_population_scaling_id` were specified."
          " `non_media_population_scaling_id` takes precedence for backward"
          " compatibility; a future version of Meridian will ignore it in"
          " favor of `population_scaled_non_media_channels`.",
          UserWarning,
          stacklevel=2,
      )
    if self.non_media_population_scaling_id is not None:
      warnings.warn(
          "`non_media_population_scaling_id` is deprecated and will be removed"
          " in a future version of Meridian. Use"
          " `population_scaled_non_media_channels` instead.",
          DeprecationWarning,
          stacklevel=2,
      )
    if self.population_scaled_non_media_channels is not None:
      if isinstance(self.population_scaled_non_media_channels, str):
        raise ValueError(
            "`population_scaled_non_media_channels` must be a sequence of"
            " strings, not a single string."
        )
      object.__setattr__(
          self,
          "population_scaled_non_media_channels",
          tuple(self.population_scaled_non_media_channels),
      )

    # Validate non_media_baseline_values.
    #
    # NOTE: only the mapping form validates its values. The legacy sequence
    # form is intentionally left unvalidated so that this change cannot reject
    # any input that callers pass successfully today. The result is an
    # asymmetry: the new API is strict about `float | 'min' | 'max'`, the old
    # one is not. Reviewers: flag this if you would rather validate both forms
    # here and accept the (small) risk of breaking an existing caller.
    if self.non_media_baseline_values is not None:
      if isinstance(self.non_media_baseline_values, Mapping):
        for k, v in self.non_media_baseline_values.items():
          if not (
              isinstance(v, (int, float))
              or (isinstance(v, str) and v in ("min", "max"))
          ):
            raise ValueError(
                f"Invalid value for non-media channel '{k}' in"
                f" `non_media_baseline_values`: {v!r}. Must be a float or"
                " 'min'/'max'."
            )
      elif isinstance(self.non_media_baseline_values, Sequence):
        warnings.warn(
            "Passing a sequence for `non_media_baseline_values` is deprecated"
            " and will be removed in a future version of Meridian. Use a"
            " mapping from channel names to baseline values"
            " (`Mapping[str, float | str]`) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(self.knots, Collection):
      knots_list = list(self.knots)
      if not all(isinstance(x, (int, np.integer)) for x in knots_list):
        raise ValueError("`knots` must be a sequence of integers.")
      object.__setattr__(self, "knots", [int(x) for x in knots_list])

    # Validate knots.
    if isinstance(self.knots, list) and not self.knots:
      raise ValueError("The `knots` parameter cannot be an empty list.")
    if isinstance(self.knots, int) and self.knots == 0:
      raise ValueError("The `knots` parameter cannot be zero.")
    if self.knots is not None and self.enable_aks:
      raise ValueError(
          "The `knots` parameter cannot be set when `enable_aks` is True."
      )
    if not (self.knots is None or isinstance(self.knots, (int, list))):
      raise ValueError(
          f"Unsupported type for `knots` parameter: {type(self.knots)}."
      )

    if (
        not isinstance(self.max_lag, int)
        or isinstance(self.max_lag, bool)
        or self.max_lag < 0
    ):
      raise ValueError(
          f"'max_lag' must be a non-negative integer. Got: {self.max_lag}."
      )

    valid_decays = set(constants.ADSTOCK_DECAY_FUNCTIONS)
    if isinstance(self.adstock_decay_spec, str):
      if self.adstock_decay_spec not in valid_decays:
        raise ValueError(
            f"The `adstock_decay_spec` parameter {self.adstock_decay_spec!r}"
            f" must be one of {sorted(valid_decays)}."
        )
    elif isinstance(self.adstock_decay_spec, Mapping):
      for channel, decay in self.adstock_decay_spec.items():
        if decay not in valid_decays:
          raise ValueError(
              f"The `adstock_decay_spec` for channel {channel!r} must be"
              f" one of {sorted(valid_decays)}, but got {decay!r}."
          )
    else:
      raise ValueError(
          "Unsupported type for `adstock_decay_spec` parameter:"
          f" {type(self.adstock_decay_spec)}."
      )

    valid_saturations = {e.value for e in SaturationType}
    if isinstance(self.saturation_spec, str):
      if self.saturation_spec not in valid_saturations:
        raise ValueError(
            f"The `saturation_spec` parameter {self.saturation_spec!r} must be"
            f" one of {sorted(valid_saturations)}."
        )
    elif isinstance(self.saturation_spec, Mapping):
      for channel, saturation in self.saturation_spec.items():
        if saturation not in valid_saturations:
          raise ValueError(
              f"The `saturation_spec` for channel {channel!r} must be"
              f" one of {sorted(valid_saturations)}, but got {saturation!r}."
          )
    else:
      raise ValueError(
          "Unsupported type for `saturation_spec` parameter:"
          f" {type(self.saturation_spec)}."
      )
    self._validate_calibrated_priors()

  def _validate_calibrated_priors(self) -> None:
    """Validates that calibrated distribution metadata matches ModelSpec settings."""
    if self.prior is None:
      return

    priors_to_check = []
    if dataclasses.is_dataclass(self.prior):
      for field in dataclasses.fields(self.prior):
        priors_to_check.append(getattr(self.prior, field.name))
    elif isinstance(self.prior, Mapping):
      priors_to_check.extend(self.prior.values())
    else:
      for attr in (
          constants.ROI_M,
          constants.ROI_RF,
          constants.MROI_M,
          constants.MROI_RF,
          constants.BETA_M,
          constants.BETA_RF,
          constants.CONTRIBUTION_M,
          constants.CONTRIBUTION_RF,
      ):
        if hasattr(self.prior, attr):
          priors_to_check.append(getattr(self.prior, attr))

    for dist in priors_to_check:
      while hasattr(dist, "distribution"):
        dist = dist.distribution
      if not isinstance(dist, calibration_base.CalibratedDistribution):
        continue

      for output in dist.calibration_outputs:
        if output is None:
          continue

        if output.max_lag != self.max_lag:
          raise ValueError(
              f"The `max_lag` for calibrated channel '{output.channel_name}'"
              f" ({output.max_lag}) does not match the ModelSpec `max_lag`"
              f" ({self.max_lag}). `max_lag` is used to calculate the"
              " duration adjustment during prior calibration. To fix this, set"
              " `ModelSpec(max_lag=...)` to match the value used during prior"
              " calibration, or recalibrate the prior using the desired"
              " `max_lag`."
          )

        if isinstance(self.adstock_decay_spec, str):
          expected_decay = self.adstock_decay_spec
        elif isinstance(self.adstock_decay_spec, Mapping):
          expected_decay = self.adstock_decay_spec.get(
              output.channel_name, constants.GEOMETRIC_DECAY
          )
        else:
          expected_decay = constants.GEOMETRIC_DECAY

        if output.adstock_decay_spec != expected_decay:
          raise ValueError(
              "The `adstock_decay_spec` for calibrated channel"
              f" '{output.channel_name}' ('{output.adstock_decay_spec}') does"
              " not match the ModelSpec `adstock_decay_spec`"
              f" ('{expected_decay}'). `adstock_decay_spec` is used to"
              " calculate the duration adjustment during prior calibration. To"
              " fix this, set `ModelSpec(adstock_decay_spec=...)` to match the"
              " value used during prior calibration, or recalibrate the prior"
              " using the desired `adstock_decay_spec`."
          )

  @property
  def effective_media_prior_type(self) -> str:
    """Returns the effective media prior type.

    The recommended way to set prior types is to use `media_prior_type` and
    `rf_prior_type` directly. If both `media_prior_type` and `rf_prior_type`
    are not set, the deprecated `paid_media_prior_type` is used for both media
    and RF channels. If none of them are set, the default is `roi`.
    """
    if self.media_prior_type is not None:
      return self.media_prior_type
    elif self.paid_media_prior_type is not None:
      return self.paid_media_prior_type
    else:
      return constants.TREATMENT_PRIOR_TYPE_ROI

  @property
  def effective_rf_prior_type(self) -> str:
    """Returns the effective rf prior type.

    The recommended way to set prior types is to use `media_prior_type` and
    `rf_prior_type` directly. If both `media_prior_type` and `rf_prior_type`
    are not set, the deprecated `paid_media_prior_type` is used for both media
    and RF channels. If none of them are set, the default is `roi`.
    """
    if self.rf_prior_type is not None:
      return self.rf_prior_type
    elif self.paid_media_prior_type is not None:
      return self.paid_media_prior_type
    else:
      return constants.TREATMENT_PRIOR_TYPE_ROI
