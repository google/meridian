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

"""Defines ModelContext class for Meridian."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import functools
from typing import Any, cast
import warnings

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.data import time_coordinates as tc
from meridian.model import adstock_hill
from meridian.model import changepoints
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers
import numpy as np

__all__ = [
    "AggregateBaselineConstraint",
    "ChannelParameters",
    "ModelContext",
    "SaturationSpec",
]


@dataclasses.dataclass(frozen=True)
class ChannelParameters:
  """Parameters for a specific channel.

  Attributes:
    index: The index of the channel within its type.
    prefix: The distribution prefix (e.g., 'm', 'rf', 'om', 'orf').
    decay_spec: The model's decay spec for this channel.
    is_rf: Whether the channel is a Reach & Frequency (RF) channel.
  """

  index: int
  prefix: str
  decay_spec: str
  is_rf: bool


# NUTS samples the first rotated knot through the sigmoid bijector of
# `TruncatedNormal`, which needs a finite upper bound. The bound is at least
# this many knot prior scales above both zero and the lower bound, so it leaves
# out a negligible amount of prior probability.
_ROTATED_KNOT_0_UPPER_BOUND_SCALES = 50.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class AggregateBaselineConstraint:
  """Precomputed terms for the aggregate baseline constraint.

  Used to truncate the aggregate baseline across geos and time periods at zero
  when `ModelSpec.allows_negative_aggregate_baseline` is `False`. See
  `ModelContext.aggregate_baseline_constraint`.

  Attributes:
    outcome_weights: Normalized `(n_geos, n_times)` outcome weights summing to
      `n_times`. Weights are proportional to `population * revenue_per_kpi`
      when `revenue_per_kpi` is provided, or `population` otherwise.
    geo_weights: Normalized `(n_geos,)` outcome weights per geo summing to `1`.
    controls_weights: Outcome-weighted `(n_geos, n_controls)` sum of scaled
      controls over time, or `None` if the model has no control variables.
    knot_rotation_matrix: Orthonormal `(n_knots, n_knots)` rotation matrix for
      knots, computed from `knot_info.weights` weighted by the outcome weights
      summed over geos.
    knot_norm_w: Euclidean L2 norm of the outcome-weighted knot weight sums.
    threshold_l: Scalar threshold `-n_times * mean / stdev` on the scaled KPI
      scale. Corresponds to a zero aggregate baseline on the original outcome
      scale.
    knot_loc: Scalar `loc` of the `knot_values` prior, in the prior's dtype.
    knot_scale: Scalar `scale` of the `knot_values` prior, in the prior's dtype.
  """

  outcome_weights: backend.Tensor
  geo_weights: backend.Tensor
  controls_weights: backend.Tensor | None
  knot_rotation_matrix: np.ndarray
  knot_norm_w: float
  threshold_l: backend.Tensor
  knot_loc: backend.Tensor
  knot_scale: backend.Tensor

  def offset(
      self,
      *,
      tau_g: backend.Tensor,
      gamma_gc: backend.Tensor | None = None,
      gamma_gn: backend.Tensor | None = None,
      non_media_baseline_normalized: backend.Tensor | None = None,
  ) -> backend.Tensor:
    """Computes the terms of the aggregate baseline that don't depend on knots.

    On the scaled KPI scale, the aggregate baseline is this offset plus
    `knot_norm_w` times the first coordinate of the knots rotated by
    `knot_rotation_matrix`.

    Args:
      tau_g: Geo effects of shape `(..., n_geos)`.
      gamma_gc: Geo-level control coefficients of shape `(..., n_geos,
        n_controls)`, if the model has control variables.
      gamma_gn: Geo-level non-media treatment coefficients of shape `(...,
        n_geos, n_non_media_channels)`, if the model has non-media treatments.
      non_media_baseline_normalized: The non-media treatment baseline values,
        normalized like the non-media treatments and broadcastable to
        `(n_geos, n_times, n_non_media_channels)`. Required with `gamma_gn`.

    Returns:
      The outcome-weighted sum of the geo, control, and non-media treatment
      terms of the baseline, of shape `(...)`.

    Raises:
      ValueError: If `gamma_gn` is passed without
        `non_media_baseline_normalized`.
    """
    n_geos, n_times = self.outcome_weights.shape
    offset = float(n_times) * backend.einsum(
        "g,...g->...", self.geo_weights, tau_g
    )
    if gamma_gc is not None:
      offset += backend.einsum("gc,...gc->...", self.controls_weights, gamma_gc)
    if gamma_gn is not None:
      if non_media_baseline_normalized is None:
        raise ValueError(
            "`non_media_baseline_normalized` is required with `gamma_gn`."
        )
      broadcast_non_media = backend.broadcast_to(
          non_media_baseline_normalized,  # pyrefly: ignore[bad-argument-type]
          [n_geos, n_times, gamma_gn.shape[-1]],
      )
      offset += backend.einsum(
          "gt,gtn,...gn->...",
          self.outcome_weights,
          broadcast_non_media,
          gamma_gn,
      )
    return offset

  def rotated_knot_0_prior(
      self, offset: backend.Tensor
  ) -> backend.tfd.Distribution:
    """Returns the prior of the first rotated knot.

    This is the knot prior, truncated from below where the aggregate baseline
    equals `threshold_l`.

    Args:
      offset: The output of `offset`, of shape `(...)`.

    Returns:
      A `TruncatedNormal` distribution with batch shape `(...)`.
    """
    knot_dtype = self.knot_scale.dtype
    low = backend.cast(
        backend.divide(self.threshold_l - offset, self.knot_norm_w),
        dtype=knot_dtype,
    )
    upper_margin = _ROTATED_KNOT_0_UPPER_BOUND_SCALES * self.knot_scale
    high = backend.cast(
        backend.maximum(low + upper_margin, upper_margin), dtype=knot_dtype
    )
    return backend.tfd.TruncatedNormal(
        loc=self.knot_loc,
        scale=self.knot_scale,
        low=low,
        high=high,
        name=constants.ROTATED_KNOT_0,
    )

  def rotated_knot_rest_prior(self) -> backend.tfd.Distribution | None:
    """Returns the prior of the rotated knots after the first one.

    Returns:
      The knot prior for each of the other `n_knots - 1` rotated knots, or
      `None` if there is only one knot.
    """
    n_knots = self.knot_rotation_matrix.shape[0]
    if n_knots == 1:
      return None
    return backend.tfd.Sample(
        backend.tfd.Normal(loc=self.knot_loc, scale=self.knot_scale),
        [n_knots - 1],
        name=constants.ROTATED_KNOT_REST,
    )

  def knot_values(
      self,
      rotated_knot_0: backend.Tensor,
      rotated_knot_rest: backend.Tensor | None = None,
  ) -> backend.Tensor:
    """Rotates the rotated knots back to knot values.

    Args:
      rotated_knot_0: The first rotated knot, of shape `(...)`.
      rotated_knot_rest: The other rotated knots, of shape `(..., n_knots -
        1)`. Required if and only if there is more than one knot.

    Returns:
      The knot values, of shape `(..., n_knots)`.

    Raises:
      ValueError: If `rotated_knot_rest` is passed with one knot, or is missing
        with more than one knot.
    """
    n_knots = self.knot_rotation_matrix.shape[0]
    if (rotated_knot_rest is None) != (n_knots == 1):
      raise ValueError(
          "`rotated_knot_rest` must be passed if and only if there is more"
          f" than one knot, but there are {n_knots} knots."
      )
    rotated_knots = backend.expand_dims(rotated_knot_0, -1)  # pyrefly: ignore[bad-argument-type]
    if rotated_knot_rest is None:
      return rotated_knots
    rotated_knots = backend.concatenate(
        [rotated_knots, rotated_knot_rest],  # pyrefly: ignore[bad-argument-type]
        axis=-1,
    )
    return backend.einsum(
        "...k,kl->...l",
        rotated_knots,
        backend.to_tensor(
            self.knot_rotation_matrix, dtype=self.knot_scale.dtype
        ),
    )


def _get_decay(decay_spec: str | Sequence[str], index: int) -> str:
  return decay_spec[index] if not isinstance(decay_spec, str) else decay_spec


def _nearest_coordinates_hint(
    target: datetime.date, dates: Sequence[datetime.date]
) -> str:
  """Describes the time coordinates bracketing `target`, for error messages."""
  earlier = [date for date in dates if date < target]
  later = [date for date in dates if date > target]
  nearest = [str(d) for d in earlier[-1:] + later[:1]]
  if not nearest:
    return ""
  return f" The nearest time coordinates are {' and '.join(nearest)}."


def _compile_date_range_mask(
    date_ranges: Sequence[spec.DateRange],
    dates: Sequence[datetime.date],
    *,
    spec_name: str,
) -> np.ndarray:
  """Compiles a sequence of `DateRange`s into a boolean mask over `dates`.

  Each `DateRange` is the closed interval `[start_date, end_date]`, matching
  the semantics documented on `spec.DateRange`: both bounds are inclusive. An
  omitted bound leaves that side open, so `DateRange()` selects every date. The
  union of all the given ranges is taken.

  Note: Every bound that is present must be one of `dates`.

  Args:
    date_ranges: The date ranges to compile.
    dates: The date coordinates to compile against, in order.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.

  Returns:
    A boolean array of shape `(len(dates),)`, `True` wherever the date falls
    inside at least one of `date_ranges`.

  Raises:
    ValueError: If a bound is not one of `dates`.
  """
  known_dates = set(dates)
  mask = np.zeros(len(dates), dtype=bool)
  for date_range in date_ranges:
    # `DateRange.__post_init__` already normalizes these, but the declared
    # attribute type stays polymorphic (`Date | None`), so normalize again to
    # compare `date` against `date` rather than against `str`.
    start = (
        tc.normalize_date(date_range.start_date)
        if date_range.start_date is not None
        else None
    )
    end = (
        tc.normalize_date(date_range.end_date)
        if date_range.end_date is not None
        else None
    )
    for bound_name, bound in (("start_date", start), ("end_date", end)):
      if bound is not None and bound not in known_dates:
        raise ValueError(
            f"`{spec_name}` has a `DateRange` whose `{bound_name}` ({bound}) is"
            " not one of the input data's time coordinates. Date range bounds"
            " must name an exact time coordinate."
            + _nearest_coordinates_hint(bound, dates)
        )
    mask |= np.array(
        [
            (start is None or date >= start) and (end is None or date <= end)
            for date in dates
        ],
        dtype=bool,
    )
  return mask


def _resolve_name_indices(
    names: Sequence[str],
    universe: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> list[int]:
  """Resolves coordinate names to their positional indices.

  Args:
    names: The names to resolve.
    universe: The ordered coordinate values to resolve against.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The input data dimension being resolved against, used in error
      messages.

  Returns:
    The index of each name in `universe`, in the order given.

  Raises:
    ValueError: If any name is absent from `universe`.
  """
  index_of = {name: index for index, name in enumerate(universe)}
  unknown = [name for name in names if name not in index_of]
  if unknown:
    raise ValueError(
        f"`{spec_name}` refers to {dim_name} that are not in the input data:"
        f" {sorted(unknown)}. Available {dim_name}: {sorted(universe)}."
    )
  return [index_of[name] for name in names]


def _compile_name_mask(
    names: Sequence[str],
    universe: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> np.ndarray:
  """Compiles a selection of coordinate names into a boolean mask.

  Args:
    names: The selected names.
    universe: The ordered coordinate values to compile against.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The input data dimension being resolved against, used in error
      messages.

  Returns:
    A boolean array of shape `(len(universe),)`, `True` at the selected names.

  Raises:
    ValueError: If any name is absent from `universe`.
  """
  mask = np.zeros(len(universe), dtype=bool)
  mask[
      _resolve_name_indices(
          names, universe, spec_name=spec_name, dim_name=dim_name
      )
  ] = True
  return mask


def _compile_calibration_spec(
    calibration: spec.CalibrationSpec,
    dates: Sequence[datetime.date],
    channels: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> np.ndarray:
  """Compiles a `CalibrationSpec` into a boolean calibration period array.

  Args:
    calibration: The declarative calibration specification.
    dates: The media time coordinates to compile against, in order.
    channels: The channel coordinates to compile against, in order.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The channel dimension being resolved against, used in error
      messages.

  Returns:
    A boolean array of shape `(len(dates), len(channels))`.

  Raises:
    ValueError: If a channel name is absent from `channels`, or a date range
      bound is not one of `dates`.
  """
  entries = calibration.spec
  n_channels = len(channels)

  # `CalibrationSpec.__post_init__` guarantees the sequence is homogeneous and
  # non-empty, so the first element determines the scope of the whole spec.
  # The type checker cannot carry that guarantee across the sequence, hence the
  # casts.
  if isinstance(entries[0], spec.DateRange):
    global_mask = _compile_date_range_mask(
        cast(Sequence[spec.DateRange], entries), dates, spec_name=spec_name
    )
    return np.tile(global_mask[:, np.newaxis], (1, n_channels))

  # Per-channel scope. A channel that no entry mentions is left *unrestricted*
  # (all `True`), matching the documented meaning of an unset
  # `roi_calibration_period`: "If `None`, all times are used." Defaulting an
  # unmentioned channel to all `False` would instead zero out its aggregated
  # spend, making the denominator of its ROI prior zero.
  compiled = np.zeros((len(dates), n_channels), dtype=bool)
  is_mentioned = np.zeros(n_channels, dtype=bool)
  for entry in cast(Sequence[spec.ChannelCalibrationSpec], entries):
    mask = _compile_date_range_mask(
        entry.date_ranges, dates, spec_name=spec_name
    )
    for index in _resolve_name_indices(
        entry.channels, channels, spec_name=spec_name, dim_name=dim_name
    ):
      compiled[:, index] |= mask
      is_mentioned[index] = True
  compiled[:, ~is_mentioned] = True
  return compiled


def _compile_geo_holdout_specs(
    geo_specs: Sequence[spec.GeoHoldoutSpec],
    dates: Sequence[datetime.date],
    geos: Sequence[str],
) -> np.ndarray:
  """Compiles per-geo holdout specs into a boolean holdout mask.

  Unlike calibration, a geo that no entry mentions is simply not held out, so
  it compiles to all `False`.

  Args:
    geo_specs: The per-geo holdout specifications.
    dates: The time coordinates to compile against, in order.
    geos: The geo coordinates to compile against, in order.

  Returns:
    A boolean array of shape `(len(geos), len(dates))`.

  Raises:
    ValueError: If a geo name is absent from `geos`, or a date range bound is
      not one of `dates`.
  """
  compiled = np.zeros((len(geos), len(dates)), dtype=bool)
  for geo_spec in geo_specs:
    mask = _compile_date_range_mask(
        geo_spec.date_ranges, dates, spec_name="holdout"
    )
    for index in _resolve_name_indices(
        geo_spec.geos, geos, spec_name="holdout", dim_name="geos"
    ):
      compiled[index, :] |= mask
  return compiled


def _draw_random_holdout(
    random_spec: spec.RandomHoldoutSpec,
    n_geos: int,
    n_times: int,
) -> np.ndarray:
  """Draws a random holdout mask, stratified by geo.

  Each geo independently holds out exactly `round(ratio * n_times)` time
  periods, sampled without replacement. Stratifying by geo -- rather than
  drawing over the flattened `n_geos * n_times` cell space -- guarantees that
  every geo contributes both training and test rows.

  Note that random samples are drawn sequentially in the order geos appear in
  the input data coordinates; permuting geo order will produce different samples
  for the same seed.

  Args:
    random_spec: The random holdout specification.
    n_geos: The number of geos.
    n_times: The number of time periods.

  Returns:
    A boolean array of shape `(n_geos, n_times)`.

  Raises:
    ValueError: If the ratio rounds to zero time periods.
  """
  n_holdout = min(int(round(random_spec.ratio * n_times)), n_times)
  if n_holdout == 0:
    raise ValueError(
        f"`holdout` requests a random holdout ratio of {random_spec.ratio},"
        f" which rounds to zero of the input data's {n_times} time periods, so"
        " nothing would be held out. Use a larger ratio, or omit `holdout`"
        " entirely if no holdout is intended."
    )
  rng = np.random.default_rng(random_spec.seed)
  mask = np.zeros((n_geos, n_times), dtype=bool)
  for geo_index in range(n_geos):
    mask[geo_index, rng.choice(n_times, size=n_holdout, replace=False)] = True
  return mask


def _contiguous_date_ranges(
    mask: np.ndarray, dates: Sequence[datetime.date]
) -> list[spec.DateRange]:
  """Groups a boolean mask over `dates` into one closed range per run.

  Each maximal run of `True` becomes a single closed `DateRange`. Runs
  separated by even one `False` stay separate ranges.

  Args:
    mask: A boolean array of shape `(len(dates),)`.
    dates: The time coordinates the mask is over, in order.

  Returns:
    The selected date ranges, in ascending order. Empty if nothing is selected.
  """
  ranges = []
  run_start = None
  for index, selected in enumerate(mask):
    if selected and run_start is None:
      run_start = index
    elif not selected and run_start is not None:
      ranges.append(spec.DateRange(dates[run_start], dates[index - 1]))
      run_start = None
  if run_start is not None:
    ranges.append(spec.DateRange(dates[run_start], dates[len(mask) - 1]))
  return ranges


def _mask_to_geo_holdout_specs(
    mask: np.ndarray,
    dates: Sequence[datetime.date],
    geos: Sequence[str],
) -> list[spec.GeoHoldoutSpec]:
  """Inverts a holdout mask into per-geo declarative holdout specs.

  This is the inverse of `_compile_geo_holdout_specs`.

  A geo with nothing held out is omitted entirely rather than given an empty
  entry, matching the rule that a geo no entry mentions is not held out.

  Args:
    mask: A boolean array of shape `(n_geos, n_times)`.
    dates: The time coordinates, in order.
    geos: The geo coordinates, in order.

  Returns:
    One `GeoHoldoutSpec` per geo that has at least one held-out period.
  """
  geo_specs = []
  for geo_index, geo in enumerate(geos):
    date_ranges = _contiguous_date_ranges(mask[geo_index], dates)
    if not date_ranges:
      continue
    geo_specs.append(spec.GeoHoldoutSpec(geos=[geo], date_ranges=date_ranges))
  return geo_specs


def _allows_negative_values(
    prior: backend.tfd.Distribution, n_channels: int
) -> np.ndarray | None:
  """Returns whether a prior allows negative values, for each channel.

  Args:
    prior: A prior shared by all channels of a type, or with one value per
      channel.
    n_channels: The number of channels of the type.

  Returns:
    One boolean per channel, or `None` if the prior has a different number of
    values than channels. Meridian reports that error when it sets up the
    priors.
  """
  allows_negative = np.asarray(prior.cdf(0)) > 0
  if np.ndim(allows_negative) == 0 or np.shape(allows_negative) == (
      n_channels,
  ):
    return np.broadcast_to(allows_negative, (n_channels,))
  return None


def _roi_setting(prior_types: Mapping[str, str]) -> str:
  """The first prior type setting in `prior_types` that is `'roi'`."""
  return next(
      name
      for name, prior_type in prior_types.items()
      if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI
  )


def _changepoint_calibration_error(
    channels: Sequence[str], attribute: str
) -> ValueError:
  """Returns the error for ROI calibration of channels with changepoints."""
  channel_list = list(channels)
  example = channels[0]
  verb = "has" if len(channels) == 1 else "have"
  return ValueError(
      "ROI calibration is not supported for channels with `changepoints`."
      f" `{attribute}` sets a calibration period for {channel_list}, which also"
      f" {verb} `changepoints`.\n"
      "To calibrate ROI for channels with changepoints, first, partition the"
      " data manually by dividing the channel into distinct periods and"
      " modeling each as a separate channel without changepoints. For example,"
      f" replace {example!r} with '{example}_1' and '{example}_2', each having"
      " spend and media only during its respective period. Then, calibrate"
      " each resulting channel with the experiment corresponding to that time"
      " period.\n"
      "Alternatively, if you want to model the channel with time-varying"
      " media effectiveness without ROI calibration, keep `changepoints` and"
      f" remove {channel_list} from `{attribute}`."
  )


def _merge_prior_tensors(
    tensor: backend.Tensor | None,
    changepoint_tensor: backend.Tensor | None,
    changepoint_info: changepoints.ChangepointInfo,
    fill: backend.Tensor | None = None,
) -> backend.Tensor | None:
  """Uses `changepoint_tensor` for the channels with changepoints.

  Args:
    tensor: A prior tensor of all channels of a type, for their prior type.
    changepoint_tensor: The same tensor for the prior type of the channels with
      changepoints.
    changepoint_info: The `ChangepointInfo` of the channel type.
    fill: Stands in for a `None` tensor. Default: the other tensor, for tensors
      that are `None` when their prior type doesn't use them.

  Returns:
    The merged tensor, or `None` if both tensors are `None`.
  """
  if tensor is None and changepoint_tensor is None:
    return None
  if tensor is None:
    tensor = changepoint_tensor if fill is None else fill
  if changepoint_tensor is None:
    changepoint_tensor = tensor if fill is None else fill
  assert tensor is not None and changepoint_tensor is not None
  return changepoints.merge_channels(
      x=tensor,
      sub=backend.gather(
          changepoint_tensor,
          backend.to_tensor(changepoint_info.channel_indices),
          axis=-1,
      ),
      info=changepoint_info,
  )


_NATIONAL_KNOTS_WARNING = (
    "You have enabled the time-varying feature with more than one knot in a"
    " national model. If the model doesn't converge, consider using"
    " `knots=1` and/or replacing knots with control variables."
)
_NATIONAL_KNOT_OVERLAP_ERROR = (
    "There are overlapping knots (for mu_t) and changepoints (for beta) in a"
    " national model. The model is unidentifiable and hurts convergence."
    " Consider dropping either one of the overlapping knots or changepoints,"
    " or replacing the knots with control variables."
)
_GEO_KNOT_OVERLAP_WARNING = (
    "There are overlapping knots (for mu_t) and changepoints (for beta) in a"
    " geo model. If convergence is an issue, consider dropping either one of"
    " the overlapping knots or changepoints, or replacing the knots with"
    " control variables."
)


def _knot_overlap_message(
    message: str, overlaps: Mapping[str, Sequence[datetime.date]]
) -> str:
  """Returns `message` with a line that lists the changepoints on a knot."""
  listed = "; ".join(
      f"{channel}: {', '.join(date.isoformat() for date in dates)}"
      for channel, dates in overlaps.items()
  )
  return f"{message}\nChangepoints on a knot: {listed}."


_NATIONAL_SHARED_CHANGEPOINTS_ERROR = (
    "There are overlapping changepoints from two (or more) media channels in a"
    " national model. The model is unidentifiable and hurts convergence."
    " Consider dropping either one of the overlapping changepoints."
)
_GEO_SHARED_CHANGEPOINTS_WARNING = (
    "There are overlapping changepoints from two (or more) media channels in a"
    " geo model. If convergence is an issue, consider dropping either one of"
    " the overlapping changepoints."
)


def _shared_changepoints_message(
    message: str, shared: Mapping[datetime.date, Sequence[str]]
) -> str:
  """Returns `message` with a line that lists the shared changepoint dates."""
  listed = "; ".join(
      f"{date.isoformat()}: {', '.join(channels)}"
      for date, channels in shared.items()
  )
  return f"{message}\nChangepoints on the same date: {listed}."


def _compile_changepoints(
    changepoints_spec: Mapping[str, Sequence[datetime.date]],
    *,
    dates: Sequence[datetime.date],
    paid_channels: Sequence[str],
    other_channels: Sequence[str],
) -> changepoints.ChangepointIndices:
  """Compiles `changepoints` into time period indices.

  Args:
    changepoints_spec: Maps a paid media or RF channel name to its changepoint
      dates, as normalized by `spec._ModelSpecWithChangepoints`.
    dates: The time coordinates of the input data, in order.
    paid_channels: The names of the paid media and RF channels.
    other_channels: The names of the organic media, organic RF and non-media
      channels. Only used for error messages.

  Returns:
    Maps each channel in `changepoints_spec` to the sorted indices of its
    changepoints in `dates`, in the order of `paid_channels`.

  Raises:
    ValueError: If a channel is not a paid media or RF channel, or a
      changepoint is not one of `dates`, is the first date, or is listed twice,
      or if a channel has a changepoint at every time period after the first.
  """
  for channel in changepoints_spec:
    if channel in paid_channels:
      continue
    if channel in other_channels:
      raise ValueError(
          f"Channel {channel!r} is an organic media or a non-media treatment."
          " Only paid media and RF channels can have changepoints."
      )
    raise ValueError(
        f"Channel {channel!r} in `changepoints` is not a paid media or RF"
        f" channel of the input data. Paid channels: {sorted(paid_channels)}."
    )

  index_of = {date: index for index, date in enumerate(dates)}
  compiled = {}
  for channel in paid_channels:
    if channel not in changepoints_spec:
      continue
    indices = []
    for date in changepoints_spec[channel]:
      if date not in index_of:
        if date < dates[0] or date > dates[-1]:
          hint = (
              f" The time periods in the data are between {dates[0]} and"
              f" {dates[-1]} (inclusive)."
          )
        else:
          hint = _nearest_coordinates_hint(date, dates)
        raise ValueError(
            f"The changepoint {date} of channel {channel!r} is not one of the"
            " input data's time coordinates."
            + hint
        )
      if index_of[date] == 0:
        raise ValueError(
            f"The changepoint {date} of channel {channel!r} is the first time"
            " coordinate. The first time period always starts the first"
            " interval, so remove it."
        )
      indices.append(index_of[date])
    if len(set(indices)) != len(indices):
      raise ValueError(
          f"Channel {channel!r} lists the same changepoint more than once."
      )
    if len(indices) > len(dates) - 2:
      raise ValueError(
          f"Channel {channel!r} has {len(indices)} changepoints, so every time"
          " period would start its own interval. List at most"
          f" {len(dates) - 2} changepoints."
      )
    compiled[channel] = tuple(sorted(indices))
  return compiled


@dataclasses.dataclass(frozen=True)
class SaturationSpec:
  """Specification for each channel's saturation function.

  Attributes:
    media: A string or sequence of strings specifying the saturation function to
      use for media channels.
    rf: A string or sequence of strings specifying the saturation function to
      use for reach and frequency channels.
    organic_media: A string or sequence of strings specifying the saturation
      function to use for organic media channels.
    organic_rf: A string or sequence of strings specifying the saturation
      function to use for organic reach and frequency channels.
  """

  media: str | Sequence[str] = constants.HILL
  rf: str | Sequence[str] = constants.HILL
  organic_media: str | Sequence[str] = constants.HILL
  organic_rf: str | Sequence[str] = constants.HILL


class ModelContext:
  """Model context for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec,
  ):
    self._input_data = input_data
    self._model_spec = model_spec

    self._model_spec.validate_knot_values_prior()
    self._validate_data_dependent_model_spec()
    self._validate_model_spec_shapes()
    self._resolve_declarative_model_spec()
    self._validate_changepoint_calibration()

    self._set_total_media_contribution_prior = False
    self._warn_setting_ignored_priors()
    self._validate_mroi_priors_non_revenue()
    self._validate_roi_priors_non_revenue()
    self._check_media_prior_support()
    # The time check runs first so that a variable that is constant everywhere
    # gets its message rather than the geo check's, which says it varies over
    # time.
    self._validate_time_invariants()
    self._validate_geo_invariants()
    self._validate_media_spend_for_paid_channels()
    self._validate_rf_spend_for_paid_channels()
    self._warn_changepoint_knots()
    self._warn_shared_changepoints()

  def _resolve_declarative_model_spec(self) -> None:
    """Resolves every declarative `ModelSpec` attribute against the input data.

    The legacy array attributes are validated eagerly, in `__init__`. Resolving
    the declarative attributes here gives them the same failure timing, so an
    unknown channel, geo or control name -- or a date range bound that is not a
    time coordinate -- is reported while the caller is still holding the spec
    that caused it, rather than part-way through sampling.

    Every property below is a `functools.cached_property`, so this only moves
    the work earlier; it does not repeat it. In particular a `RandomHoldoutSpec`
    is drawn exactly once, here, and every later reader sees that same draw.

    Raises:
      ValueError: If any declarative attribute cannot be resolved against the
        input data.
    """
    _ = self.compiled_roi_calibration_period
    _ = self.compiled_rf_roi_calibration_period
    _ = self.compiled_holdout_id
    _ = self.compiled_control_population_scaling_id
    _ = self.compiled_non_media_population_scaling_id
    _ = self.compiled_non_media_baseline_values
    _ = self._media_changepoint_info
    _ = self._rf_changepoint_info

  # TODO: Deduplicate with `_validate_model_spec_shapes`. Both
  # methods run from `__init__` and validate the same legacy `ModelSpec`
  # arrays (`roi_calibration_period`, `rf_roi_calibration_period`,
  # `holdout_id`, `control_population_scaling_id`) against the same shapes,
  # with near-identical error messages.
  def _validate_data_dependent_model_spec(self):
    """Validates that the data dependent model specs have correct shapes."""

    if self._model_spec.roi_calibration_period is not None and (
        self._model_spec.roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_media_channels,
        )
    ):
      raise ValueError(
          "The shape of `roi_calibration_period`"
          f" {self._model_spec.roi_calibration_period.shape} is different from"
          f" `(n_media_times, n_media_channels) = ({self.n_media_times},"
          f" {self.n_media_channels})`."
      )

    if self._model_spec.rf_roi_calibration_period is not None and (
        self._model_spec.rf_roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_rf_channels,
        )
    ):
      raise ValueError(
          "The shape of `rf_roi_calibration_period`"
          f" {self._model_spec.rf_roi_calibration_period.shape} is different"
          f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
          f" {self.n_rf_channels})`."
      )

    if self._model_spec.holdout_id is not None:
      if self.is_national and (
          self._model_spec.holdout_id.shape != (self.n_times,)
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_times,) = ({self.n_times},)`."
        )
      elif not self.is_national and (
          self._model_spec.holdout_id.shape
          != (
              self.n_geos,
              self.n_times,
          )
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_geos, n_times) = ({self.n_geos},"
            f" {self.n_times})`."
        )

    if self._model_spec.control_population_scaling_id is not None and (
        self._model_spec.control_population_scaling_id.shape
        != (self.n_controls,)
    ):
      raise ValueError(
          "The shape of `control_population_scaling_id`"
          f" {self._model_spec.control_population_scaling_id.shape} is"
          f" different from `(n_controls,) = ({self.n_controls},)`."
      )

    if self._model_spec.non_media_population_scaling_id is not None and (
        self._model_spec.non_media_population_scaling_id.shape
        != (self.n_non_media_channels,)
    ):
      raise ValueError(
          "The shape of `non_media_population_scaling_id`"
          f" {self._model_spec.non_media_population_scaling_id.shape} is"
          " different from `(n_non_media_channels,) ="
          f" ({self.n_non_media_channels},)`."
      )

  # TODO: Deduplicate with `_validate_data_dependent_model_spec`.
  def _validate_model_spec_shapes(self):
    """Validate shapes of model_spec attributes."""
    if self._model_spec.roi_calibration_period is not None:
      if self._model_spec.roi_calibration_period.shape != (
          self.n_media_times,
          self.n_media_channels,
      ):
        raise ValueError(
            "The shape of `roi_calibration_period`"
            f" {self._model_spec.roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_media_channels) = ({self.n_media_times},"
            f" {self.n_media_channels})`."
        )

    if self._model_spec.rf_roi_calibration_period is not None:
      if self._model_spec.rf_roi_calibration_period.shape != (
          self.n_media_times,
          self.n_rf_channels,
      ):
        raise ValueError(
            "The shape of `rf_roi_calibration_period`"
            f" {self._model_spec.rf_roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
            f" {self.n_rf_channels})`."
        )

    if self._model_spec.holdout_id is not None:
      expected_shape = (
          (self.n_times,) if self.is_national else (self.n_geos, self.n_times)
      )
      if self._model_spec.holdout_id.shape != expected_shape:
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            " different from"
            f" {'`(n_times,)`' if self.is_national else '`(n_geos, n_times)`'}"
            f" = {expected_shape}."
        )

    if self._model_spec.control_population_scaling_id is not None:
      if self._model_spec.control_population_scaling_id.shape != (
          self.n_controls,
      ):
        raise ValueError(
            "The shape of `control_population_scaling_id`"
            f" {self._model_spec.control_population_scaling_id.shape} is"
            f" different from `(n_controls,) = ({self.n_controls},)`."
        )

  def _validate_geo_invariants(self):
    """Validates non-national model invariants."""
    if self.is_national:
      return

    if self._input_data.controls is not None:
      self._check_if_no_geo_variation(
          self.controls_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_geo_variation(
          self.non_media_treatments_normalized,  # pyrefly: ignore[bad-argument-type]
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_geo_variation(
          self.media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_geo_variation(
          self.rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_geo_variation(
          self.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
          "organic_media",
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_geo_variation(
          self.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
          "organic_reach",
          self._input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
      )

  def _check_if_no_geo_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if `n_knots == n_time` and data lacks geo variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=0) < epsilon  # pyrefly: ignore[bad-argument-type]
    )[:, 1]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    # We use the shape of scaled_data (instead of `n_time`) because the data may
    # be padded to account for lagged effects.
    data_n_time = scaled_data.shape[1]
    mask = backend.equal(counts, data_n_time)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)

    if col_idx_bad.shape[0] and self.knot_info.n_knots == self.n_times:
      raise ValueError(
          f"The following {data_name} variables do not vary across geos, making"
          f" a model with n_knots=n_time unidentifiable: {dims_bad}. This can"
          " lead to poor model convergence. Since these variables only vary"
          " across time and not across geo, they are collinear with time and"
          " redundant in a model with a parameter for each time period.  To"
          " address this, you can either: (1) decrease the number of knots"
          " (n_knots < n_time), or (2) drop the listed variables that do not"
          " vary across geos."
      )

  def _validate_time_invariants(self):
    """Validates model time invariants."""
    if self._input_data.controls is not None:
      self._check_if_no_time_variation(
          self.controls_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_time_variation(
          self.non_media_treatments_normalized,  # pyrefly: ignore[bad-argument-type]
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_time_variation(
          self.media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_time_variation(
          self.rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_time_variation(
          self.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.ORGANIC_MEDIA,
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_time_variation(
          self.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.ORGANIC_REACH,
          self._input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
      )

  def _validate_media_spend_for_paid_channels(self) -> None:
    self._validate_spend_for_paid_channels(
        self.input_data.aggregate_media_spend(), constants.MEDIA_CHANNEL
    )

  def _validate_rf_spend_for_paid_channels(self) -> None:
    self._validate_spend_for_paid_channels(
        self.input_data.aggregate_rf_spend(), constants.RF_CHANNEL
    )

  def _validate_spend_for_paid_channels(
      self,
      spend: np.ndarray | None,
      dim: str,
  ) -> None:
    """Validates non-zero media spend for paid media channels.

    Args:
      spend: The media spend data to validate.
      dim: The dimension name of the spend data.

    Raises:
      ValueError if any paid media channel has zero total spend.
    """
    if spend is None:
      return
    zero_spend_channels = spend.coords[dim].where(spend == 0, drop=True).values  # pyrefly: ignore[missing-attribute]

    if zero_spend_channels.size > 0:
      raise ValueError(
          "Zero total spend detected for paid channels:"
          f" {', '.join(zero_spend_channels)}. If data is correct and this is"
          " expected, please consider modeling the data as organic media."
      )

  def _check_if_no_time_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if data lacks time variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=1) < epsilon  # pyrefly: ignore[bad-argument-type]
    )[:, 1]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    mask = backend.equal(counts, self.n_geos)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)
    if col_idx_bad.shape[0]:
      if self.is_national:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            " which is equivalent to no signal at all in a national model:"
            f" {dims_bad}.  This can lead to poor model convergence. To address"
            " this, drop the listed variables that do not vary across time."
        )
      else:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            f" making a model with geo main effects unidentifiable: {dims_bad}."
            " This can lead to poor model convergence. A variable that doesn't"
            " vary across time is either constant or only varies across geos,"
            " so it's redundant with the geo main effects. To address this,"
            " drop the listed variables that do not vary across time."
        )

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    """The media tensors.

    If the channels with changepoints use another prior type, their columns of
    the prior tensors are for that prior type.
    """
    build = functools.partial(
        media._build_media_tensors,  # pylint: disable=protected-access
        self._input_data,
        calibration_period=self.compiled_roi_calibration_period,
    )
    prior_types = list(self._paid_prior_types(is_rf=False).values())
    tensors = build(prior_type=prior_types[0])
    info = self._media_changepoint_info
    if info is None or len(set(prior_types)) == 1:
      return tensors
    changepoint_tensors = build(prior_type=prior_types[1])
    assert tensors.media_scaled is not None
    return dataclasses.replace(
        tensors,
        prior_media_scaled_counterfactual=_merge_prior_tensors(
            tensors.prior_media_scaled_counterfactual,
            changepoint_tensors.prior_media_scaled_counterfactual,
            info,
            fill=backend.zeros_like(tensors.media_scaled),
        ),
        prior_denominator=_merge_prior_tensors(
            tensors.prior_denominator,
            changepoint_tensors.prior_denominator,
            info,
        ),
    )

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    """The RF tensors, with prior tensors like `media_tensors`."""
    build = functools.partial(
        media._build_rf_tensors,  # pylint: disable=protected-access
        self._input_data,
        calibration_period=self.compiled_rf_roi_calibration_period,
    )
    prior_types = list(self._paid_prior_types(is_rf=True).values())
    tensors = build(prior_type=prior_types[0])
    info = self._rf_changepoint_info
    if info is None or len(set(prior_types)) == 1:
      return tensors
    changepoint_tensors = build(prior_type=prior_types[1])
    assert tensors.reach_scaled is not None
    return dataclasses.replace(
        tensors,
        prior_reach_scaled_counterfactual=_merge_prior_tensors(
            tensors.prior_reach_scaled_counterfactual,
            changepoint_tensors.prior_reach_scaled_counterfactual,
            info,
            fill=backend.zeros_like(tensors.reach_scaled),
        ),
        prior_denominator=_merge_prior_tensors(
            tensors.prior_denominator,
            changepoint_tensors.prior_denominator,
            info,
        ),
    )

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self._input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self._input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.kpi, dtype=backend.float_dtype)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self._input_data.revenue_per_kpi is None:
      return None
    return backend.to_tensor(
        self._input_data.revenue_per_kpi, dtype=backend.float_dtype
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    if self._input_data.controls is None:
      return None
    return backend.to_tensor(
        self._input_data.controls, dtype=backend.float_dtype
    )

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self._input_data.non_media_treatments is None:
      return None
    return backend.to_tensor(
        self._input_data.non_media_treatments, dtype=backend.float_dtype
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.population, dtype=backend.float_dtype
    )

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_spend(), dtype=backend.float_dtype
    )

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_outcome(), dtype=backend.float_dtype
    )

  @property
  def n_geos(self) -> int:
    return len(self._input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self._input_data.media_channel is None:
      return 0
    return len(self._input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self._input_data.rf_channel is None:
      return 0
    return len(self._input_data.rf_channel)

  @property
  def n_organic_media_channels(self) -> int:
    if self._input_data.organic_media_channel is None:
      return 0
    return len(self._input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self._input_data.organic_rf_channel is None:
      return 0
    return len(self._input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    if self._input_data.control_variable is None:
      return 0
    return len(self._input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self._input_data.non_media_channel is None:
      return 0
    return len(self._input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self._input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self._input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self._model_spec.knots,
        enable_aks=self._model_spec.enable_aks,
        data=self._input_data,
        is_national=self.is_national,
    )

  def _inject_legacy_knot_info_for_serde(self, legacy_knots: list[int]):
    """Injects legacy knots by bypassing the AKS check in get_knot_info."""
    legacy_knots_arr = np.array(legacy_knots, dtype=int)
    n_knots = len(legacy_knots_arr)
    if n_knots == 1:
      weights = np.ones((1, self.n_times), dtype=backend.np_float_dtype)
    else:
      weights = knots.l1_distance_weights(self.n_times, legacy_knots_arr)  # pyrefly: ignore[bad-argument-type]
    self.__dict__["knot_info"] = knots.KnotInfo(
        n_knots=n_knots,
        knot_locations=legacy_knots_arr,  # pyrefly: ignore[bad-argument-type]
        weights=weights,  # pyrefly: ignore[bad-argument-type]
    )

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for controls, if it exists."""
    if self.controls is None:
      return None

    if self.compiled_control_population_scaling_id is not None:
      controls_population_scaling_id = backend.to_tensor(
          self.compiled_control_population_scaling_id, dtype=backend.bool_
      )
    else:
      controls_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.controls,
        population=self.population,
        population_scaling_id=controls_population_scaling_id,
    )

  @functools.cached_property
  def non_media_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for non-media treatments."""
    if self.non_media_treatments is None:
      return None
    if self.compiled_non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.to_tensor(
          self.compiled_non_media_population_scaling_id, dtype=backend.bool_
      )
    else:
      non_media_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.non_media_treatments,
        population=self.population,
        population_scaling_id=non_media_population_scaling_id,
    )

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return transformers.KpiTransformer(self.kpi, self.population)

  @functools.cached_property
  def controls_scaled(self) -> backend.Tensor | None:
    if self.controls is not None:
      # If `controls` is defined, then `controls_transformer` is also defined.
      return self.controls_transformer.forward(self.controls)  # pyrefly: ignore[missing-attribute]
    else:
      return None

  @functools.cached_property
  def non_media_treatments_normalized(self) -> backend.Tensor | None:
    """Normalized non-media treatments.

    The non-media treatments values are scaled by population (for channels where
    `non_media_population_scaling_id` is `True`) and normalized by centering and
    scaling with means and standard deviations.
    """
    if self.non_media_transformer is not None:
      return self.non_media_transformer.forward(
          self.non_media_treatments
      )
    else:
      return None

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
    return self.kpi_transformer.forward(self.kpi)

  @functools.cached_property
  def aggregate_baseline_constraint(
      self,
  ) -> AggregateBaselineConstraint | None:
    """Precomputed terms for the aggregate baseline constraint.

    These terms are only needed when `allows_negative_aggregate_baseline` is
    `False`. Otherwise, they are neither computed nor validated, so models that
    don't constrain the aggregate baseline are unaffected by them.

    Returns:
      An `AggregateBaselineConstraint`, or `None` if
      `allows_negative_aggregate_baseline` is `True`.

    Raises:
      ValueError: If the outcome weights are negative or don't have a strictly
        positive sum, or if the population-scaled KPI is constant.
    """
    if self._model_spec.allows_negative_aggregate_baseline:
      return None

    pop_col = backend.expand_dims(self.population, axis=-1)  # pyrefly: ignore[bad-argument-type]
    if self.revenue_per_kpi is not None:
      raw_weights = pop_col * self.revenue_per_kpi
    else:
      raw_weights = backend.broadcast_to(pop_col, [self.n_geos, self.n_times])
    raw_weights_np = np.asarray(raw_weights)
    total_weight = float(np.sum(raw_weights_np))
    if (
        np.any(raw_weights_np < 0.0)
        or total_weight <= 0.0
        or not np.isfinite(total_weight)
    ):
      raise ValueError(
          "Outcome weights (`population * revenue_per_kpi` or `population`)"
          " must be non-negative and have a strictly positive sum across geos"
          " and time periods when `allows_negative_aggregate_baseline=False`."
          " Please ensure `population` contains positive values and has"
          " non-zero overlap with `revenue_per_kpi`, or set"
          " `allows_negative_aggregate_baseline=True`."
      )
    outcome_weights = backend.divide(
        float(self.n_times) * raw_weights,
        backend.reduce_sum(raw_weights),  # pyrefly: ignore[bad-argument-type]
    )

    stdev = float(np.asarray(self.kpi_transformer.population_scaled_stdev))
    if stdev <= 0.0 or not np.isfinite(stdev):
      kpi = self._input_data.kpi.name
      raise ValueError(
          f"`{kpi}` cannot be constant when"
          " `allows_negative_aggregate_baseline=False` because standardizing"
          " the baseline threshold requires non-zero KPI variability"
          " (`population_scaled_stdev > 0`). Please verify that `kpi` varies"
          " across geos or time periods, or set"
          " `allows_negative_aggregate_baseline=True`."
      )
    threshold_l = backend.to_tensor(
        -float(self.n_times)
        * self.kpi_transformer.population_scaled_mean
        / self.kpi_transformer.population_scaled_stdev,
        dtype=backend.float_dtype,
    )

    time_weights = backend.reduce_sum(outcome_weights, axis=0)  # pyrefly: ignore[bad-argument-type]
    knot_rotation_matrix, knot_norm_w = knots.compute_knot_rotation_matrix(
        self.knot_info.weights,  # pyrefly: ignore[bad-argument-type]
        np.asarray(time_weights),
    )

    if self.controls_scaled is None:
      controls_weights = None
    else:
      controls_weights = backend.einsum(
          "gt,gtc->gc", outcome_weights, self.controls_scaled
      )

    # `validate_knot_values_prior` checks that the knot prior is a `Normal`
    # distribution with the same `loc` and `scale` for all knots.
    knot_prior = self.prior_broadcast.knot_values
    while isinstance(knot_prior, backend.tfd.BatchBroadcast):
      knot_prior = knot_prior.distribution
    assert isinstance(knot_prior, backend.tfd.Normal)
    knot_loc = float(np.reshape(np.asarray(knot_prior.loc), -1)[0])
    knot_scale = float(np.reshape(np.asarray(knot_prior.scale), -1)[0])

    return AggregateBaselineConstraint(
        outcome_weights=outcome_weights,
        geo_weights=backend.divide(
            backend.reduce_sum(outcome_weights, axis=-1),  # pyrefly: ignore[bad-argument-type]
            float(self.n_times),
        ),
        controls_weights=controls_weights,
        knot_rotation_matrix=knot_rotation_matrix,
        knot_norm_w=knot_norm_w,
        threshold_l=threshold_l,
        knot_loc=backend.to_tensor(knot_loc, dtype=knot_prior.dtype),
        knot_scale=backend.to_tensor(knot_scale, dtype=knot_prior.dtype),
    )

  @functools.cached_property
  def media_effects_dist(self) -> str:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[constants.MEDIA_EFFECTS_DIST]  # pyrefly: ignore[bad-return]
    else:
      return self._model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      # Should evaluate to False.
      return constants.NATIONAL_MODEL_SPEC_ARGS[  # pyrefly: ignore[bad-return]
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self._model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self._model_spec.baseline_geo, int):
      if (
          self._model_spec.baseline_geo < 0
          or self._model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self._model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self._model_spec.baseline_geo
    elif isinstance(self._model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self._input_data.geo == self._model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self._model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    compiled = self.compiled_holdout_id
    if compiled is None:
      return None
    tensor = backend.to_tensor(compiled, dtype=backend.bool_)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  # --------------------------------------------------------------------------
  # Compiled model spec properties.
  #
  # These resolve `ModelSpec`'s declarative attributes against this context's
  # `InputData` coordinates, producing the positional boolean arrays the model
  # engine consumes. They are the single place where a channel name becomes a
  # column index and a date range becomes a row mask.
  #
  # Precedence is *legacy first*: when both a declarative attribute and its
  # deprecated array counterpart are set, the deprecated array wins. This
  # matches the contract `ModelSpec.__post_init__` currently advertises in its
  # conflict warning ("<legacy> takes precedence for backward compatibility").
  # --------------------------------------------------------------------------

  def _coordinate_names(self, coordinate: Any) -> list[str]:
    """Returns an input data coordinate's values as a list of strings."""
    if coordinate is None:
      return []
    return [str(value) for value in coordinate.values]

  @functools.cached_property
  def compiled_roi_calibration_period(self) -> np.ndarray | None:
    """The effective ROI calibration period for media channels.

    Resolved from the declarative `ModelSpec.roi_calibration` against the input
    data's media time and media channel coordinates, or taken as-is from the
    deprecated `ModelSpec.roi_calibration_period`.

    Returns:
      A boolean array of shape `(n_media_times, n_media_channels)`, or `None`
      if neither attribute is set.

    Raises:
      ValueError: If the spec names a media channel not in the input data, or
        a date range bound that is not a time coordinate.
    """
    if self._model_spec.roi_calibration_period is not None:
      return self._model_spec.roi_calibration_period
    if self._model_spec.roi_calibration is None:
      return None
    return _compile_calibration_spec(
        self._model_spec.roi_calibration,
        self._input_data.media_time_coordinates.all_dates,
        self._coordinate_names(self._input_data.media_channel),
        spec_name="roi_calibration",
        dim_name="media channels",
    )

  @functools.cached_property
  def compiled_rf_roi_calibration_period(self) -> np.ndarray | None:
    """The effective ROI calibration period for reach & frequency channels.

    Resolved from the declarative `ModelSpec.rf_roi_calibration` against the
    input data's media time and RF channel coordinates, or taken as-is from the
    deprecated `ModelSpec.rf_roi_calibration_period`.

    Returns:
      A boolean array of shape `(n_media_times, n_rf_channels)`, or `None` if
      neither attribute is set.

    Raises:
      ValueError: If the spec names an RF channel not in the input data, or a
        date range bound that is not a time coordinate.
    """
    if self._model_spec.rf_roi_calibration_period is not None:
      return self._model_spec.rf_roi_calibration_period
    if self._model_spec.rf_roi_calibration is None:
      return None
    return _compile_calibration_spec(
        self._model_spec.rf_roi_calibration,
        self._input_data.media_time_coordinates.all_dates,
        self._coordinate_names(self._input_data.rf_channel),
        spec_name="rf_roi_calibration",
        dim_name="RF channels",
    )

  @functools.cached_property
  def compiled_holdout_id(self) -> np.ndarray | None:
    """The effective holdout mask.

    Resolved from the declarative `ModelSpec.holdout` against the input data's
    time and geo coordinates, or taken as-is from the deprecated
    `ModelSpec.holdout_id`.

    For a declarative holdout, a `resolved` draw always wins and is never
    re-drawn; see `spec.RandomHoldoutSpec` for why a seed alone cannot
    reproduce a draw. Only when a `RandomHoldoutSpec` carries no `resolved`
    draw is one made here, once, and memoized for the lifetime of this context.

    Returns:
      A boolean array of shape `(n_times,)` for a national model or
      `(n_geos, n_times)` otherwise -- the same convention as the deprecated
      `ModelSpec.holdout_id` -- or `None` if neither attribute is set.

    Raises:
      ValueError: If the spec names a geo not in the input data, a date range
        bound that is not a time coordinate, or a random holdout ratio that
        rounds to zero time periods.
    """
    if self._model_spec.holdout_id is not None:
      return self._model_spec.holdout_id
    holdout = self._model_spec.holdout
    if holdout is None:
      return None

    dates = self._input_data.time_coordinates.all_dates
    geos = self._coordinate_names(self._input_data.geo)

    if holdout.resolved is not None:
      compiled = _compile_geo_holdout_specs(holdout.resolved, dates, geos)
    elif isinstance(holdout.spec, spec.RandomHoldoutSpec):
      compiled = _draw_random_holdout(holdout.spec, len(geos), len(dates))
    elif isinstance(holdout.spec[0], spec.DateRange):
      # A global holdout applies the same date mask to every geo.
      global_mask = _compile_date_range_mask(
          cast(Sequence[spec.DateRange], holdout.spec),
          dates,
          spec_name="holdout",
      )
      compiled = np.tile(global_mask[np.newaxis, :], (len(geos), 1))
    else:
      compiled = _compile_geo_holdout_specs(
          cast(Sequence[spec.GeoHoldoutSpec], holdout.spec), dates, geos
      )

    # National models carry a 1-D holdout, matching the legacy convention that
    # `_validate_model_spec_shapes` enforces.
    return compiled[0] if self.is_national else compiled

  @functools.cached_property
  def resolved_random_holdout(self) -> Sequence[spec.GeoHoldoutSpec] | None:
    """The draw to record in `HoldoutSpec.resolved`, or `None` if there is none.

    A `RandomHoldoutSpec` does not determine a holdout by itself, so the draw
    the model actually used has to be recorded alongside it; see
    `spec.RandomHoldoutSpec` for why a seed alone cannot reproduce one. This
    inverts `compiled_holdout_id` back into the declarative form that
    `HoldoutSpec.resolved` holds.

    There is a draw to record only when the random specification is the one
    that governed. If the deprecated `holdout_id` is also set it takes
    precedence and this returns `None`.

    Returns:
      The draw as per-geo holdout specs, or `None` if the model has no random
      holdout, or has one that did not govern.

    Raises:
      ValueError: If the spec names a geo not in the input data, a date range
        bound that is not a time coordinate, or a random holdout ratio that
        rounds to zero time periods.
    """
    holdout = self._model_spec.holdout
    if holdout is None or not isinstance(holdout.spec, spec.RandomHoldoutSpec):
      return None
    if holdout.resolved is not None:
      return holdout.resolved
    if self._model_spec.holdout_id is not None:
      return None

    mask = self.compiled_holdout_id
    if mask is None:
      return None
    mask = np.asarray(mask)
    # A national model carries a 1-D mask over time only.
    if mask.ndim == 1:
      mask = mask[np.newaxis, :]

    # The draw holds out at least one period per geo, so this is never empty.
    return (
        _mask_to_geo_holdout_specs(
            mask,
            self._input_data.time_coordinates.all_dates,
            self._coordinate_names(self._input_data.geo),
        )
        or None
    )

  @functools.cached_property
  def compiled_control_population_scaling_id(self) -> np.ndarray | None:
    """The effective population-scaling selection for control variables.

    Resolved from the declarative `ModelSpec.population_scaled_controls`
    against the input data's control variable coordinates, or taken as-is from
    the deprecated `ModelSpec.control_population_scaling_id`.

    Returns:
      A boolean array of shape `(n_controls,)`, or `None` if neither attribute
      is set.

    Raises:
      ValueError: If the spec names a control variable not in the input data.
    """
    if self._model_spec.control_population_scaling_id is not None:
      return self._model_spec.control_population_scaling_id
    if self._model_spec.population_scaled_controls is None:
      return None
    return _compile_name_mask(
        self._model_spec.population_scaled_controls,
        self._coordinate_names(self._input_data.control_variable),
        spec_name="population_scaled_controls",
        dim_name="control variables",
    )

  @functools.cached_property
  def compiled_non_media_population_scaling_id(self) -> np.ndarray | None:
    """The effective population-scaling selection for non-media channels.

    Resolved from the declarative
    `ModelSpec.population_scaled_non_media_channels` against the input data's
    non-media channel coordinates, or taken as-is from the deprecated
    `ModelSpec.non_media_population_scaling_id`.

    Returns:
      A boolean array of shape `(n_non_media_channels,)`, or `None` if neither
      attribute is set.

    Raises:
      ValueError: If the spec names a non-media channel not in the input data.
    """
    if self._model_spec.non_media_population_scaling_id is not None:
      return self._model_spec.non_media_population_scaling_id
    if self._model_spec.population_scaled_non_media_channels is None:
      return None
    return _compile_name_mask(
        self._model_spec.population_scaled_non_media_channels,
        self._coordinate_names(self._input_data.non_media_channel),
        spec_name="population_scaled_non_media_channels",
        dim_name="non-media channels",
    )

  def resolve_non_media_baseline_values(
      self,
      values: Mapping[str, float | str] | Sequence[float | str] | None,
  ) -> list[float | str] | None:
    """Resolves non-media baseline values into positional channel order.

    A mapping only needs to name the channels whose baseline differs from the
    default; any channel it omits falls back to `'min'`.

    Args:
      values: A mapping from non-media channel name to baseline value, a
        sequence already in channel order, or `None`.

    Returns:
      A list of length `n_non_media_channels` in channel order, or `None` if
      `values` is `None`.

    Raises:
      ValueError: If a mapping key is not a known non-media channel.
    """
    if values is None:
      return None
    if not isinstance(values, Mapping):
      return list(values)
    channels = self._coordinate_names(self._input_data.non_media_channel)
    _resolve_name_indices(
        list(values.keys()),
        channels,
        spec_name="non_media_baseline_values",
        dim_name="non-media channels",
    )
    return [
        values.get(channel, constants.NON_MEDIA_BASELINE_MIN)
        for channel in channels
    ]

  @functools.cached_property
  def compiled_non_media_baseline_values(self) -> list[float | str] | None:
    """`ModelSpec.non_media_baseline_values`, in positional channel order.

    Returns:
      A list of length `n_non_media_channels`, or `None` if the attribute is
      unset.

    Raises:
      ValueError: If the attribute is a mapping naming an unknown channel.
    """
    return self.resolve_non_media_baseline_values(
        self._model_spec.non_media_baseline_values
    )

  def _warn_setting_ignored_priors(self):
    """Raises a warning if ignored priors are set."""
    default_distribution = prior_distribution.PriorDistribution()
    for ignored_priors_dict, is_rf in (
        (constants.IGNORED_PRIORS_MEDIA, False),
        (constants.IGNORED_PRIORS_RF, True),
    ):
      prior_types = self._paid_prior_types(is_rf)
      # With changepoints, a prior is ignored only if no channel uses it.
      ignored_priors = [
          prior
          for prior in ignored_priors_dict.get(
              next(iter(prior_types.values())), []
          )
          if all(
              prior in ignored_priors_dict.get(prior_type, [])
              for prior_type in prior_types.values()
          )
      ]
      ignored_custom_priors = []
      for prior in ignored_priors:
        self_prior = getattr(self._model_spec.prior, prior)
        default_prior = getattr(default_distribution, prior)
        if not prior_distribution.distributions_are_equal(
            self_prior, default_prior
        ):
          ignored_custom_priors.append(prior)
      if ignored_custom_priors:
        ignored_priors_str = ", ".join(ignored_custom_priors)
        settings = " and ".join(
            f'`{prior_type_name}` is set to "{prior_type}"'
            for prior_type_name, prior_type in prior_types.items()
        )
        warnings.warn(
            f"Custom prior(s) `{ignored_priors_str}` are ignored when"
            f" {settings}."
        )

  def _validate_mroi_priors_non_revenue(self):
    """Validates mroi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      for is_rf, n_channels, field in (
          (False, self.n_media_channels, constants.MROI_M),
          (True, self.n_rf_channels, constants.MROI_RF),
      ):
        mroi_settings = [
            prior_type_name
            for prior_type_name, prior_type in self._paid_prior_types(
                is_rf
            ).items()
            if prior_type == constants.TREATMENT_PRIOR_TYPE_MROI
        ]
        if (
            n_channels > 0
            and mroi_settings
            and prior_distribution.distributions_are_equal(
                getattr(self._model_spec.prior, field),
                getattr(default_distribution, field),
            )
        ):
          raise ValueError(
              f"Custom priors should be set on `{field}` when"
              f' `{mroi_settings[0]}` is "mroi", KPI is non-revenue and'
              " revenue per kpi data is missing."
          )

  def _validate_roi_priors_non_revenue(self):
    """Validates roi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      roi = constants.TREATMENT_PRIOR_TYPE_ROI
      media_prior_types = self._paid_prior_types(is_rf=False).values()
      rf_prior_types = self._paid_prior_types(is_rf=True).values()
      default_roi_m = prior_distribution.distributions_are_equal(
          self._model_spec.prior.roi_m, default_distribution.roi_m
      )
      default_roi_rf = prior_distribution.distributions_are_equal(
          self._model_spec.prior.roi_rf, default_distribution.roi_rf
      )
      # Some channels use the default ROI prior.
      default_roi_m_used = roi in media_prior_types and default_roi_m
      default_roi_rf_used = roi in rf_prior_types and default_roi_rf
      # All channels of the type use the default ROI prior.
      default_roi_m_only = (
          all(prior_type == roi for prior_type in media_prior_types)
          and default_roi_m
      )
      default_roi_rf_only = (
          all(prior_type == roi for prior_type in rf_prior_types)
          and default_roi_rf
      )
      # If ROI priors are used with the default prior distribution for all paid
      # channels (media and RF), then use the "total paid media contribution
      # prior" procedure.
      if (
          (default_roi_m_only and default_roi_rf_only)
          or (self.n_media_channels == 0 and default_roi_rf_only)
          or (self.n_rf_channels == 0 and default_roi_m_only)
      ):
        self._set_total_media_contribution_prior = True
        warnings.warn(
            "Consider setting custom ROI priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with"
            f" `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}`. Further"
            " documentation available at "
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior",
        )
      elif self.n_media_channels > 0 and default_roi_m_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_M}` when"
            f' `{_roi_setting(self._paid_prior_types(is_rf=False))}` is "roi",'
            " custom priors are assigned on"
            f' `{constants.ROI_RF}` or `rf_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )
      elif self.n_rf_channels > 0 and default_roi_rf_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_RF}` when"
            f' `{_roi_setting(self._paid_prior_types(is_rf=True))}` is "roi",'
            " custom priors are assigned on"
            f' `{constants.ROI_M}` or `media_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )

  def _check_media_prior_support(self) -> None:
    """Checks ROI, mROI, and Contribution prior support when random effects are log-normal.

    Priors for ROI, mROI, and Contribution can only have negative support if the
    random effects follow a normal distribution. This check enforces that priors
    have non-negative support when random effects follow a log-normal
    distribution. This check only applies to geo-level models with log-normal
    random effects since national models do not have random effects. Channels
    with changepoints are checked with `changepoint_media_effects_dist`, which
    also applies to national models.
    """
    prior = self._model_spec.prior
    if self.n_media_channels > 0:
      self._check_for_negative_support(
          prior.roi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
          self._media_changepoint_info,
      )
      self._check_for_negative_support(
          prior.mroi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          self._media_changepoint_info,
      )
      self._check_for_negative_support(
          prior.contribution_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          self._media_changepoint_info,
      )
    if self.n_rf_channels > 0:
      self._check_for_negative_support(
          prior.roi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
          self._rf_changepoint_info,
      )
      self._check_for_negative_support(
          prior.mroi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          self._rf_changepoint_info,
      )
      self._check_for_negative_support(
          prior.contribution_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          self._rf_changepoint_info,
      )
    if self.n_organic_media_channels > 0:
      self._check_for_negative_support(
          prior.contribution_om,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_organic_rf_channels > 0:
      self._check_for_negative_support(
          prior.contribution_orf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )

  def _check_for_negative_support(
      self,
      dist: backend.tfd.Distribution,
      media_effects_dist: str,
      prior_type: str,
      changepoint_info: changepoints.ChangepointInfo | None = None,
  ) -> None:
    """Checks for negative support in prior distributions.

    When `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`, prior distributions
    for media effects must be non-negative. This function raises a ValueError if
    any part of the distribution's CDF is greater than 0 at 0, indicating some
    probability mass below zero.

    Args:
      dist: The distribution to check.
      media_effects_dist: The type of media effects distribution.
      prior_type: The prior type that corresponds with current prior under test.
      changepoint_info: The channels of the type with changepoints, if any.
        They are checked the same way, with `changepoint_media_effects_dist`
        instead of `media_effects_dist`.

    Raises:
      ValueError: If the prior distribution has negative support when
      `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`.
    """
    # Each check: the channels it applies to (all if `None`), their random
    # effects distribution and its setting, and the prior type setting that
    # turns the check on.
    checks = [(
        None,
        media_effects_dist,
        "media_effects_dist",
        self._model_spec.media_prior_type,
    )]
    if changepoint_info is not None:
      is_changepoint_channel = changepoint_info.is_changepoint_channel
      checks = [
          (~is_changepoint_channel, *checks[0][1:]),
          (
              is_changepoint_channel,
              self._changepoint_media_effects_dist,
              "changepoint_media_effects_dist",
              # Like `media_prior_type` above, also for RF channels.
              # TODO: Read the field directly once it moves to
              # `ModelSpec`.
              getattr(self._model_spec, "changepoint_media_prior_type", None),
          ),
      ]
    for channels, effects_dist, setting, set_prior_type in checks:
      if (
          prior_type != set_prior_type
          or effects_dist != constants.MEDIA_EFFECTS_LOG_NORMAL
      ):
        continue
      if channels is None:
        negative = np.asarray(dist.cdf(0)) > 0
      else:
        negative = _allows_negative_values(dist, len(channels))
        if negative is None:
          return
        negative = negative[channels]
      if np.any(negative):
        raise ValueError(
            "Media priors must have non-negative support when"
            f' `{setting}`="{effects_dist}". Found negative prior'
            f" distribution support for {dist.name}."
        )

  # --------------------------------------------------------------------------
  # Time-varying media effects.
  #
  # Only `spec._ModelSpecWithChangepoints` has `changepoints`, so it is read
  # with `getattr`. These members are private until the feature is released.
  # --------------------------------------------------------------------------

  @functools.cached_property
  def _changepoints(self) -> changepoints.ChangepointIndices:
    """The model spec's changepoints, as time period indices.

    Returns:
      Maps each paid media or RF channel with changepoints to the sorted
      indices of its changepoints in the time coordinates. Empty if the model
      has no changepoints.

    Raises:
      ValueError: If a channel is not a paid media or RF channel, or a
        changepoint is not a valid date for the input data.
    """
    # TODO: Read `self._model_spec.changepoints` directly once
    # `changepoints` moves to `ModelSpec`.
    changepoints_spec = getattr(self._model_spec, "changepoints", None)
    if not changepoints_spec:
      return {}
    return _compile_changepoints(
        changepoints_spec,
        dates=self._input_data.time_coordinates.all_dates,
        paid_channels=(
            self._coordinate_names(self._input_data.media_channel)
            + self._coordinate_names(self._input_data.rf_channel)
        ),
        other_channels=(
            self._coordinate_names(self._input_data.organic_media_channel)
            + self._coordinate_names(self._input_data.organic_rf_channel)
            + self._coordinate_names(self._input_data.non_media_channel)
        ),
    )

  def _validate_changepoint_calibration(self) -> None:
    """Rejects ROI calibration periods for channels with changepoints.

    Raises:
      ValueError: If an ROI calibration period restricts a channel with
        changepoints to part of the media time periods.
    """
    if not self._changepoints:
      return
    for calibration_period, channel_coordinate, legacy, declarative in (
        (
            self.compiled_roi_calibration_period,
            self._input_data.media_channel,
            "roi_calibration_period",
            "roi_calibration",
        ),
        (
            self.compiled_rf_roi_calibration_period,
            self._input_data.rf_channel,
            "rf_roi_calibration_period",
            "rf_roi_calibration",
        ),
    ):
      if calibration_period is None:
        continue
      channels = [
          name
          for index, name in enumerate(
              self._coordinate_names(channel_coordinate)
          )
          if name in self._changepoints
          and not np.all(calibration_period[:, index])
      ]
      if channels:
        attribute = (
            legacy
            if getattr(self._model_spec, legacy) is not None
            else declarative
        )
        raise _changepoint_calibration_error(channels, attribute)

  def _changepoint_knot_overlaps(self) -> dict[str, list[datetime.date]]:
    """The changepoints that are on a knot.

    With a changepoint on a knot, the model has difficulty telling a change in
    the baseline apart from a change in the channel's effect. This reads
    `knot_info` on every call, so it checks the final knots, wherever they come
    from: a list, a number of knots, the default, automatic knot selection, or a
    loaded model.

    Returns:
      Maps each channel with a changepoint on a knot to the dates of those
      changepoints. Empty if no changepoint is on a knot.
    """
    if not self._changepoints:
      return {}
    dates = self._input_data.time_coordinates.all_dates
    overlaps = changepoints.changepoint_knot_overlaps(
        self.knot_info.knot_locations, self._changepoints
    )
    return {
        channel: [dates[index] for index in indices]
        for channel, indices in overlaps.items()
    }

  def _warn_changepoint_knots(self) -> None:
    """Warns about knots that can hurt the convergence of changepoints.

    A national model with changepoints and more than one knot gets a warning.
    A geo model with a changepoint on a knot gets a warning. In a national
    model, a changepoint on a knot makes the model unidentifiable, so
    `Meridian.sample_posterior` raises an error
    (`_national_changepoint_knot_error`). The model build also warns about it,
    so users see it before sampling.
    """
    if not self._changepoints:
      return
    if self.is_national:
      if self.knot_info.n_knots > 1:
        warnings.warn(_NATIONAL_KNOTS_WARNING)
      knot_error = self._national_changepoint_knot_error()
      if knot_error is not None:
        warnings.warn(
            f"{knot_error}\n`sample_posterior` raises this as an error until"
            " no changepoint is on a knot."
        )
      return
    overlaps = self._changepoint_knot_overlaps()
    if overlaps:
      warnings.warn(_knot_overlap_message(_GEO_KNOT_OVERLAP_WARNING, overlaps))

  def _national_changepoint_knot_error(self) -> str | None:
    """The error message for a national model with a changepoint on a knot.

    Returns:
      The message, or `None` if the model is a geo model or no changepoint is
      on a knot.
    """
    if not self.is_national:
      return None
    overlaps = self._changepoint_knot_overlaps()
    if not overlaps:
      return None
    return _knot_overlap_message(_NATIONAL_KNOT_OVERLAP_ERROR, overlaps)

  def _shared_changepoints(self) -> dict[datetime.date, tuple[str, ...]]:
    """The changepoint dates that two or more channels have in common.

    When two channels' effects change on the same date, the model can't tell
    which channel's change it sees. Paid media and RF channels are compared
    with each other.

    Returns:
      Maps each date on which two or more channels have a changepoint to those
      channels, in the order of the paid media and RF channels. The dates are
      in order. Empty if no two channels have a changepoint on the same date.
    """
    if not self._changepoints:
      return {}
    shared = changepoints.shared_changepoints(self._changepoints)
    if not shared:
      return {}
    dates = self._input_data.time_coordinates.all_dates
    return {dates[index]: channels for index, channels in shared.items()}

  def _warn_shared_changepoints(self) -> None:
    """Warns when two or more channels have a changepoint on the same date.

    A geo model gets a warning. In a national model, the shared changepoints
    make the model unidentifiable, so `Meridian.sample_posterior` raises an
    error (`_national_shared_changepoints_error`). The model build also warns
    about it, so users see it before sampling.
    """
    if not self._changepoints:
      return
    if self.is_national:
      shared_error = self._national_shared_changepoints_error()
      if shared_error is not None:
        warnings.warn(
            f"{shared_error}\n`sample_posterior` raises this as an error until"
            " no two channels have a changepoint on the same date."
        )
      return
    shared = self._shared_changepoints()
    if shared:
      warnings.warn(
          _shared_changepoints_message(_GEO_SHARED_CHANGEPOINTS_WARNING, shared)
      )

  def _national_shared_changepoints_error(self) -> str | None:
    """The error message for a national model with shared changepoint dates.

    Returns:
      The message, or `None` if the model is a geo model or no two channels
      have a changepoint on the same date.
    """
    if not self.is_national:
      return None
    shared = self._shared_changepoints()
    if not shared:
      return None
    return _shared_changepoints_message(
        _NATIONAL_SHARED_CHANGEPOINTS_ERROR, shared
    )

  def _national_changepoint_error(self) -> str | None:
    """The changepoint error that stops posterior sampling of a national model.

    Returns:
      The messages of `_national_changepoint_knot_error` and
      `_national_shared_changepoints_error`, separated by a blank line, or
      `None` if neither applies.
    """
    messages = [
        message
        for message in (
            self._national_changepoint_knot_error(),
            self._national_shared_changepoints_error(),
        )
        if message is not None
    ]
    if not messages:
      return None
    return "\n\n".join(messages)

  @functools.cached_property
  def _changepoint_media_effects_dist(self) -> str:
    """The random effects distribution of the channels with changepoints.

    Unlike `media_effects_dist`, it also applies to national models.
    """
    # TODO: Read `self._model_spec.changepoint_media_effects_dist`
    # directly once it moves to `ModelSpec`.
    return getattr(
        self._model_spec,
        "changepoint_media_effects_dist",
        constants.MEDIA_EFFECTS_LOG_NORMAL,
    )

  def _changepoint_prior_type(self, is_rf: bool) -> str:
    """The prior type of the paid media or RF channels with changepoints."""
    # TODO: Read the fields of `self._model_spec` directly once
    # they move to `ModelSpec`.
    return getattr(
        self._model_spec,
        "changepoint_rf_prior_type"
        if is_rf
        else "changepoint_media_prior_type",
        constants.TREATMENT_PRIOR_TYPE_ROI,
    )

  def _paid_prior_types(self, is_rf: bool) -> dict[str, str]:
    """The prior types that the paid media or RF channels use.

    Args:
      is_rf: Whether to use the RF channels rather than the media channels.

    Returns:
      Maps the name of each prior type setting that some channel of the type
      uses to its value. The first is `media_prior_type` or `rf_prior_type`,
      unless every channel of the type has changepoints. The other is
      `changepoint_media_prior_type` or `changepoint_rf_prior_type`, if some
      channel of the type has changepoints.
    """
    if is_rf:
      name, n_channels = "rf", self.n_rf_channels
      changepoint_info = self._rf_changepoint_info
      prior_type = self._model_spec.effective_rf_prior_type
    else:
      name, n_channels = "media", self.n_media_channels
      changepoint_info = self._media_changepoint_info
      prior_type = self._model_spec.effective_media_prior_type
    prior_types = {}
    if changepoint_info is None or changepoint_info.n_channels < n_channels:
      prior_types[f"{name}_prior_type"] = prior_type
    if changepoint_info is not None:
      prior_types[f"changepoint_{name}_prior_type"] = (
          self._changepoint_prior_type(is_rf)
      )
    return prior_types

  def _check_changepoint_prior_types(self) -> None:
    """Checks that the posterior can be sampled with these prior types.

    Raises:
      NotImplementedError: If the paid media or RF channels with changepoints
        use another prior type than `media_prior_type` or `rf_prior_type`.
    """
    # TODO: Posterior sample models whose channels with
    # changepoints use another prior type.
    for is_rf, changepoint_info, prior_type in (
        (
            False,
            self._media_changepoint_info,
            self._model_spec.effective_media_prior_type,
        ),
        (
            True,
            self._rf_changepoint_info,
            self._model_spec.effective_rf_prior_type,
        ),
    ):
      changepoint_prior_type = self._changepoint_prior_type(is_rf)
      if changepoint_info is not None and changepoint_prior_type != prior_type:
        name = "rf" if is_rf else "media"
        raise NotImplementedError(
            "Posterior sampling is not supported yet when"
            f" `changepoint_{name}_prior_type` ({changepoint_prior_type!r})"
            f" differs from `{name}_prior_type` ({prior_type!r}). Prior"
            " sampling is supported."
        )

  @functools.cached_property
  def _changepoint_max_intervals(self) -> int:
    """The length of the interval axis shared by media and RF channels."""
    if not self._changepoints:
      return 0
    return 1 + max(len(indices) for indices in self._changepoints.values())

  def _get_changepoint_info(
      self, channel_coordinate: Any
  ) -> changepoints.ChangepointInfo | None:
    """Returns the `ChangepointInfo` of the channels in `channel_coordinate`."""
    if not self._changepoints or channel_coordinate is None:
      return None
    return changepoints.get_changepoint_info(
        n_times=self.n_times,
        changepoints=self._changepoints,
        channel_names=self._coordinate_names(channel_coordinate),
        max_intervals=self._changepoint_max_intervals,
    )

  @functools.cached_property
  def _media_changepoint_info(self) -> changepoints.ChangepointInfo | None:
    """The changepoints of the media channels, or `None` if they have none."""
    return self._get_changepoint_info(self._input_data.media_channel)

  @functools.cached_property
  def _rf_changepoint_info(self) -> changepoints.ChangepointInfo | None:
    """The changepoints of the RF channels, or `None` if they have none."""
    return self._get_changepoint_info(self._input_data.rf_channel)

  @property
  def _changepoint_prior_broadcast(
      self,
  ) -> prior_distribution._PriorDistributionWithChangepoints:  # pylint: disable=protected-access
    """`prior_broadcast` of a model with changepoints, with the `zeta` priors.

    Raises:
      ValueError: If the model has no changepoints.
    """
    prior = self.prior_broadcast
    if not isinstance(
        prior,
        prior_distribution._PriorDistributionWithChangepoints,  # pylint: disable=protected-access
    ):
      raise ValueError("The model has no changepoints.")
    return prior

  def _changepoint_inference_data_coords(self) -> dict[str, np.ndarray]:
    """The inference data coordinates of the time-varying parameters."""
    media_info = self._media_changepoint_info
    rf_info = self._rf_changepoint_info
    return {
        constants.CHANGEPOINT_MEDIA_CHANNEL: np.array(
            media_info.channel_names if media_info else [], dtype=str
        ),
        constants.CHANGEPOINT_RF_CHANNEL: np.array(
            rf_info.channel_names if rf_info else [], dtype=str
        ),
        constants.CHANGEPOINT_INTERVAL: np.arange(
            self._changepoint_max_intervals
        ),
    }

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    total_spend = self._input_data.get_total_spend()
    # Total spend can have 1, 2 or 3 dimensions. Aggregate by channel.
    if len(total_spend.shape) == 1:
      # Already aggregated by channel.
      agg_total_spend = total_spend
    elif len(total_spend.shape) == 2:
      agg_total_spend = np.sum(total_spend, axis=(0,))
    else:
      agg_total_spend = np.sum(total_spend, axis=(0, 1))

    prior_with_changepoints_cls = (
        prior_distribution._PriorDistributionWithChangepoints  # pylint: disable=protected-access
    )
    broadcast: Callable[..., prior_distribution.PriorDistribution] = (
        self._model_spec.prior.broadcast
    )
    if self._changepoints:
      # Models with changepoints also need the time-varying priors.
      prior = prior_with_changepoints_cls.from_prior(self._model_spec.prior)
      media_info = self._media_changepoint_info
      rf_info = self._rf_changepoint_info
      broadcast = functools.partial(
          prior.broadcast,
          n_changepoint_media_channels=(
              media_info.n_channels if media_info else 0
          ),
          n_changepoint_rf_channels=rf_info.n_channels if rf_info else 0,
      )
    elif isinstance(self._model_spec.prior, prior_with_changepoints_cls):
      # Without changepoints, broadcast like a regular prior so that the model
      # is the same as one built from a `PriorDistribution`.
      warnings.warn(
          "The `zeta_m` and `zeta_rf` priors are ignored because the model has"
          " no `changepoints`."
      )
      broadcast = functools.partial(
          prior_distribution.PriorDistribution.broadcast,
          self._model_spec.prior,
      )

    return broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_organic_media_channels=self.n_organic_media_channels,
        n_organic_rf_channels=self.n_organic_rf_channels,
        n_controls=self.n_controls,
        n_non_media_channels=self.n_non_media_channels,
        unique_sigma_for_each_geo=self.unique_sigma_for_each_geo,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=self._set_total_media_contribution_prior,
        kpi=np.sum(self._input_data.kpi.values),
        total_spend=agg_total_spend,
    )

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    if isinstance(self._model_spec.adstock_decay_spec, str):
      return adstock_hill.AdstockDecaySpec.from_consistent_type(
          self._model_spec.adstock_decay_spec
      )

    try:
      return self._create_adstock_decay_functions_from_channel_map(
          self._model_spec.adstock_decay_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `adstock_decay_spec` keys"
          f" {tuple(self._model_spec.adstock_decay_spec.keys())}. Keys should"
          " contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())}."
      ) from e

  def _create_adstock_decay_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> adstock_hill.AdstockDecaySpec:
    """Create `AdstockDecaySpec` from mapping from channels to decay functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      media_adstock_function = media_channel_builder(**channel_function_map)
    else:
      media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      rf_adstock_function = rf_channel_builder(**channel_function_map)
    else:
      rf_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_media_adstock_function = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_rf_adstock_function = organic_rf_channel_builder(
          **channel_function_map
      )
    else:
      organic_rf_adstock_function = constants.GEOMETRIC_DECAY

    return adstock_hill.AdstockDecaySpec(
        media=media_adstock_function,
        rf=rf_adstock_function,
        organic_media=organic_media_adstock_function,
        organic_rf=organic_rf_adstock_function,
    )

  @functools.cached_property
  def saturation_spec(self) -> SaturationSpec:
    """The SaturationSpec object with correctly mapped channels."""
    if isinstance(self._model_spec.saturation_spec, str):
      return SaturationSpec(
          media=self._model_spec.saturation_spec,
          rf=self._model_spec.saturation_spec,
          organic_media=self._model_spec.saturation_spec,
          organic_rf=self._model_spec.saturation_spec,
      )

    try:
      return self._create_saturation_functions_from_channel_map(
          self._model_spec.saturation_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `saturation_spec` keys"
          f" {tuple(self._model_spec.saturation_spec.keys())}. Keys should"
          " contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())}."
      ) from e

  def _create_saturation_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> SaturationSpec:
    """Creates `SaturationSpec` from mapping from channels to saturation functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.HILL
      )
      media_saturation = media_channel_builder(**channel_function_map)
    else:
      media_saturation = constants.HILL

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.HILL
      )
      rf_saturation = rf_channel_builder(**channel_function_map)
    else:
      rf_saturation = constants.HILL

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.HILL
      )
      organic_media_saturation = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_saturation = constants.HILL

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.HILL
      )
      organic_rf_saturation = organic_rf_channel_builder(**channel_function_map)
    else:
      organic_rf_saturation = constants.HILL

    return SaturationSpec(
        media=media_saturation,
        rf=rf_saturation,
        organic_media=organic_media_saturation,
        organic_rf=organic_rf_saturation,
    )

  def create_inference_data_coords(
      self, n_chains: int, n_draws: int
  ) -> Mapping[str, np.ndarray | Sequence[str]]:
    """Creates data coordinates for inference data."""
    media_channel_names = (
        self.input_data.media_channel
        if self.input_data.media_channel is not None
        else np.array([])
    )
    rf_channel_names = (
        self.input_data.rf_channel
        if self.input_data.rf_channel is not None
        else np.array([])
    )
    organic_media_channel_names = (
        self.input_data.organic_media_channel
        if self.input_data.organic_media_channel is not None
        else np.array([])
    )
    organic_rf_channel_names = (
        self.input_data.organic_rf_channel
        if self.input_data.organic_rf_channel is not None
        else np.array([])
    )
    non_media_channel_names = (
        self.input_data.non_media_channel
        if self.input_data.non_media_channel is not None
        else np.array([])
    )
    control_variable_names = (
        self.input_data.control_variable
        if self.input_data.control_variable is not None
        else np.array([])
    )
    coords: dict[str, np.ndarray | Sequence[str]] = {
        constants.CHAIN: np.arange(n_chains),
        constants.DRAW: np.arange(n_draws),
        constants.GEO: self.input_data.geo,  # pyrefly: ignore[bad-assignment]
        constants.TIME: self.input_data.time,  # pyrefly: ignore[bad-assignment]
        constants.MEDIA_TIME: self.input_data.media_time,  # pyrefly: ignore[bad-assignment]
        constants.KNOTS: np.arange(self.knot_info.n_knots),
        constants.CONTROL_VARIABLE: control_variable_names,  # pyrefly: ignore[bad-assignment]
        constants.NON_MEDIA_CHANNEL: non_media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.MEDIA_CHANNEL: media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.RF_CHANNEL: rf_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.ORGANIC_MEDIA_CHANNEL: organic_media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.ORGANIC_RF_CHANNEL: organic_rf_channel_names,  # pyrefly: ignore[bad-assignment]
    }
    if self._changepoints:
      coords.update(self._changepoint_inference_data_coords())
    return coords

  def create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    """Creates data dimensions for inference data."""
    inference_dims = dict(constants.INFERENCE_DIMS)
    if self.unique_sigma_for_each_geo:
      inference_dims[constants.SIGMA] = [constants.GEO]  # pyrefly: ignore[unsupported-operation]
    else:
      inference_dims[constants.SIGMA] = []  # pyrefly: ignore[unsupported-operation]
    if self._changepoints:
      inference_dims.update(constants.CHANGEPOINT_INFERENCE_DIMS)

    return {
        param: [constants.CHAIN, constants.DRAW] + list(dims)
        for param, dims in inference_dims.items()
    }

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, and so this method freezes the
    object's states before the computation graph is created.
    """
    cls = self.__class__
    # "Freeze" all @cached_property attributes by simply accessing them (with
    # `getattr()`).
    cached_properties = [
        attr
        for attr in dir(self)
        if isinstance(getattr(cls, attr, cls), functools.cached_property)
    ]
    for attr in cached_properties:
      _ = getattr(self, attr)

  def expand_selected_time_dims(
      self,
      start_date: tc.Date = None,
      end_date: tc.Date = None,
  ) -> list[str] | None:
    """Validates and returns time dimension values based on the selected times.

    If both `start_date` and `end_date` are None, returns None. If specified,
    both `start_date` and `end_date` are inclusive, and must be present in the
    time coordinates of the input data.

    Args:
      start_date: Start date of the selected time period. If None, implies the
        earliest time dimension value in the input data.
      end_date: End date of the selected time period. If None, implies the
        latest time dimension value in the input data.

    Returns:
      A list of time dimension values (as Meridian-formatted strings) in the
      input data within the selected time period, or do nothing and pass through
      None if both arguments are Nones, or if `start_date` and `end_date`
      correspond to the entire time range in the input data.

    Raises:
      ValueError if `start_date` or `end_date` is not in the input data time
      dimensions.
    """
    expanded = self.input_data.time_coordinates.expand_selected_time_dims(
        start_date=start_date, end_date=end_date
    )
    if expanded is None:
      return None
    return [date.strftime(constants.DATE_FORMAT) for date in expanded]

  def get_media_scaling_factor(
      self, channel_name: str
  ) -> backend.Tensor:
    """Retrieves the population-scaled median used to scale a channel's volume.

    For Reach & Frequency (RF) channels, this returns the scaling factor applied
    to the 'reach' component, as 'frequency' is not transformed.

    Args:
      channel_name: The string name of the paid or organic channel.

    Returns:
      A tensor of shape (n_geos,) representing the scaling factor.

    Raises:
      ValueError: If the channel is not found, or the transformer is
        uninitialized.
    """
    input_data = self.input_data

    if (
        input_data.non_media_channel is not None
        and channel_name in input_data.non_media_channel.values
    ):
      raise ValueError(
          "Cannot return a scaling factor for non-media treatment"
          f" '{channel_name}'."
      )

    if (
        input_data.control_variable is not None
        and channel_name in input_data.control_variable.values
    ):
      raise ValueError(
          "Cannot return a scaling factor for control variable"
          f" '{channel_name}'."
      )

    configs = [
        (
            input_data.media_channel,
            self.media_tensors.media_transformer,
            "media",
        ),
        (
            input_data.rf_channel,
            self.rf_tensors.reach_transformer,
            "RF",
        ),
        (
            input_data.organic_media_channel,
            self.organic_media_tensors.organic_media_transformer,
            "organic media",
        ),
        (
            input_data.organic_rf_channel,
            self.organic_rf_tensors.organic_reach_transformer,
            "organic RF",
        ),
    ]

    for channel_data, transformer, channel_type_name in configs:
      if channel_data is None or channel_name not in channel_data.values:
        continue

      if transformer is None:
        raise ValueError(
            f"Transformer for {channel_type_name} channel '{channel_name}'"
            " is missing."
        )
      (indices,) = np.where(channel_data.values == channel_name)
      idx = indices[0]
      return transformer.scale_factors_gm[:, idx]

    raise ValueError(f"Channel '{channel_name}' not found in any model inputs.")

  def get_channel_parameters(
      self, channel_name: str
  ) -> ChannelParameters:
    """Maps channel names to index, prefix, decay spec, and is_rf.

    Args:
      channel_name: Name of the channel.

    Returns:
      A ChannelParameters object containing the channel's metadata.

    Raises:
      ValueError: If the channel is not found.
    """
    input_data = self.input_data
    decay_spec = self.adstock_decay_spec

    configs = [
        (input_data.media_channel, "m", decay_spec.media, False),
        (input_data.rf_channel, "rf", decay_spec.rf, True),
        (
            input_data.organic_media_channel,
            "om",
            decay_spec.organic_media,
            False,
        ),
        (input_data.organic_rf_channel, "orf", decay_spec.organic_rf, True),
    ]

    for channel_data, prefix, channel_decay_spec, is_rf in configs:
      if channel_data is not None and channel_name in channel_data.values:
        index = list(channel_data.values).index(channel_name)
        return ChannelParameters(
            index=index,
            prefix=prefix,
            decay_spec=_get_decay(channel_decay_spec, index),
            is_rf=is_rf,
        )

    raise ValueError(f"Channel '{channel_name}' not found in the model.")

  def get_channel_parameter_tensor(
      self,
      dist_tensors: Any,
      *,
      param_base_name: str,
      channel_name: str,
  ) -> backend.Tensor:
    """Safely extracts a channel's parameter tensor (e.g., 'alpha', 'beta_g').

    Args:
      dist_tensors: An object containing batched distribution tensors (e.g.,
        DistributionTensors).
      param_base_name: The base name of the parameter (e.g., 'alpha', 'ec',
        'beta_g').
      channel_name: The name of the channel.

    Returns:
      The sliced parameter tensor for the specific channel.

    Raises:
      ValueError: If the parameter or channel is not found.
    """
    params = self.get_channel_parameters(channel_name)
    if param_base_name == constants.BETA_G:
      full_param_name = f"{param_base_name}{params.prefix}"
    else:
      full_param_name = f"{param_base_name}_{params.prefix}"
    try:
      tensor_block = getattr(dist_tensors, full_param_name)
    except AttributeError:
      raise ValueError(
          f"Parameter '{full_param_name}' not found in the distribution"
          " tensors."
      ) from None
    return tensor_block[..., params.index]
