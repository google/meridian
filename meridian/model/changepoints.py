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

"""Math helpers for time-varying media effects.

A paid media or reach and frequency (RF) channel can have changepoints: time
periods where its effect may change. A channel's changepoints split the time
range into intervals. The channel's coefficient for geo `g` and time period `t`
is

  beta_gt = exp(beta + eta * beta_g_dev + zeta * phi_t)    (log-normal)
  beta_gt = beta + eta * beta_g_dev + zeta * phi_t         (normal)

where `phi_t = varphi_k` for the interval `k` that contains `t`, `varphi_k` is
a standard-normal interval adjustment, and `zeta` is the scale of the variation
across intervals.

Channels with changepoints are stored together, per channel type, in a padded
`(channel, interval)` layout, so channels with different numbers of intervals
share one tensor. Padded intervals have zero weight in every time period, so
their values never reach the model.

`ModelContext` uses this module to build each channel type's
`ChangepointInfo`. Sampling doesn't use it yet.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses

from meridian import backend
from meridian import constants
import numpy as np

__all__ = [
    'ChangepointInfo',
    'changepoint_knot_overlaps',
    'expand_intervals',
    'get_changepoint_info',
    'interval_weights',
    'merge_channels',
    'solve_beta',
    'time_varying_coefficients',
]


def interval_weights(
    n_times: int, interval_starts: Sequence[int]
) -> np.ndarray:
  """Returns the interval indicator matrix for one channel.

  Args:
    n_times: The number of time periods.
    interval_starts: The first time period of each interval, as indices into the
      time periods. Must start with 0 and be strictly increasing. Interval `k`
      ends just before `interval_starts[k + 1]`, or at `n_times` for the last
      interval.

  Returns:
    An array of shape `(n_intervals, n_times)`. Row `k` is 1 for the time
    periods in interval `k` and 0 elsewhere, so each column sums to 1.

  Raises:
    ValueError: If `n_times` is not positive, or `interval_starts` is empty,
      doesn't start with 0, isn't strictly increasing, or has a value that is
      not less than `n_times`.
  """
  if n_times < 1:
    raise ValueError(f'`n_times` must be positive, got {n_times}.')
  starts = np.asarray(interval_starts, dtype=int)
  if starts.ndim != 1 or starts.size == 0 or starts[0] != 0:
    raise ValueError(
        '`interval_starts` must be a non-empty sequence that starts with 0, got'
        f' {list(interval_starts)}.'
    )
  if np.any(np.diff(starts) <= 0) or starts[-1] >= n_times:
    raise ValueError(
        '`interval_starts` must be strictly increasing and less than `n_times`'
        f' ({n_times}), got {starts.tolist()}.'
    )
  ends = np.append(starts[1:], n_times)
  times = np.arange(n_times)
  return (
      (times >= starts[:, np.newaxis]) & (times < ends[:, np.newaxis])
  ).astype(backend.np_float_dtype)


@dataclasses.dataclass(frozen=True)
class ChangepointInfo:
  """The changepoints of one channel type (paid media or RF).

  Only channels with changepoints are included. They are kept in the order of
  the channel type's channels.

  Attributes:
    channel_names: The names of the channels with changepoints.
    channel_indices: The position of each of these channels among all channels
      of the type, shape `(n_channels,)`.
    n_channels_total: The number of channels of the type, with or without
      changepoints.
    interval_starts: For each channel, the first time period of each interval,
      starting with 0.
    weights: The interval indicators, shape `(n_channels, max_intervals,
      n_times)`. Rows of padded intervals are all zero.
    mask: Whether each interval is real rather than padding, shape `(n_channels,
      max_intervals)`.
  """

  channel_names: tuple[str, ...]
  channel_indices: np.ndarray
  n_channels_total: int
  interval_starts: tuple[tuple[int, ...], ...]
  weights: np.ndarray
  mask: np.ndarray

  @property
  def n_channels(self) -> int:
    """The number of channels with changepoints."""
    return len(self.channel_names)

  @property
  def max_intervals(self) -> int:
    """The length of the padded interval axis."""
    return int(self.weights.shape[1])

  @property
  def merge_indices(self) -> np.ndarray:
    """Indices that put channels with changepoints back in channel order.

    For `x` with all channels of the type on its last axis and `sub` with only
    the channels with changepoints, gathering `concat([x, sub], axis=-1)` with
    these indices returns `x` with those channels replaced by `sub`. See
    `merge_channels`.
    """
    indices = np.arange(self.n_channels_total)
    indices[self.channel_indices] = self.n_channels_total + np.arange(
        self.n_channels
    )
    return indices

  @property
  def last_interval_one_hot(self) -> np.ndarray:
    """One-hot `(n_channels, max_intervals)` of each channel's last interval."""
    one_hot = np.zeros(self.mask.shape, dtype=backend.np_float_dtype)
    one_hot[np.arange(self.n_channels), self.mask.sum(axis=1) - 1] = 1.0
    return one_hot


def get_changepoint_info(
    n_times: int,
    changepoints: Mapping[str, Collection[int]],
    channel_names: Sequence[str],
    max_intervals: int | None = None,
) -> ChangepointInfo | None:
  """Builds the `ChangepointInfo` of one channel type.

  Args:
    n_times: The number of time periods.
    changepoints: Maps a channel name to its changepoints, as indices into the
      time periods. Each changepoint starts a new interval, so it must be
      between 1 and `n_times - 1`. Channels of other types are ignored.
    channel_names: The names of all channels of the type, in order.
    max_intervals: The length of the padded interval axis, so that paid media
      and RF channels can share one interval axis. Defaults to the largest
      number of intervals of any channel of the type.

  Returns:
    The `ChangepointInfo`, or `None` if no channel of the type has
    changepoints.

  Raises:
    ValueError: If a channel has no changepoints, a duplicate changepoint, or a
      changepoint outside `[1, n_times - 1]`, or if `max_intervals` is less
      than the number of intervals of a channel.
  """
  channel_names = list(channel_names)
  names = [name for name in channel_names if name in changepoints]
  if not names:
    return None

  starts = []
  for name in names:
    points = sorted(int(point) for point in changepoints[name])
    if not points:
      raise ValueError(f'Channel {name!r} has no changepoints.')
    if len(set(points)) != len(points):
      raise ValueError(
          f'Channel {name!r} has duplicate changepoints: {points}.'
      )
    if points[0] < 1 or points[-1] > n_times - 1:
      raise ValueError(
          f'The changepoints of channel {name!r} must be between 1 and'
          f' {n_times - 1}, got {points}.'
      )
    starts.append(tuple([0] + points))

  n_intervals = max(len(s) for s in starts)
  if max_intervals is not None:
    if max_intervals < n_intervals:
      raise ValueError(
          f'`max_intervals` must be at least {n_intervals}, got'
          f' {max_intervals}.'
      )
    n_intervals = max_intervals

  weights = np.zeros(
      (len(names), n_intervals, n_times), dtype=backend.np_float_dtype
  )
  mask = np.zeros((len(names), n_intervals), dtype=bool)
  for i, channel_starts in enumerate(starts):
    weights[i, : len(channel_starts)] = interval_weights(
        n_times, channel_starts
    )
    mask[i, : len(channel_starts)] = True
  return ChangepointInfo(
      channel_names=tuple(names),
      channel_indices=np.array([channel_names.index(name) for name in names]),
      n_channels_total=len(channel_names),
      interval_starts=tuple(starts),
      weights=weights,
      mask=mask,
  )


def changepoint_knot_overlaps(
    knot_locations: Collection[int],
    changepoints: Mapping[str, Collection[int]],
) -> dict[str, tuple[int, ...]]:
  """Returns the changepoints that are exactly on a baseline knot.

  Takes resolved knot positions, whatever their source (a list, `knots=<int>`
  in `ModelSpec`, the defaults, or automatic knot selection). The first time
  period is ignored: it always starts the first interval and is usually a knot
  too. A single knot gives a flat baseline, so it never overlaps.

  Args:
    knot_locations: The knot positions, as indices into the time periods.
    changepoints: Maps a channel name to its changepoints, as indices into the
      time periods.

  Returns:
    Maps each channel with an overlap to its overlapping changepoints, sorted.
    Empty if nothing overlaps.
  """
  locations = {int(k) for k in np.asarray(list(knot_locations)).ravel()}
  if len(locations) <= 1:
    return {}
  overlaps = {}
  for name, points in changepoints.items():
    hits = tuple(sorted({int(p) for p in points if 0 < p and p in locations}))
    if hits:
      overlaps[name] = hits
  return overlaps


def expand_intervals(
    varphi: backend.Tensor, weights: backend.Tensor | np.ndarray
) -> backend.Tensor:
  """Maps per-interval adjustments to per-time-period values.

  Args:
    varphi: The interval adjustments, shape `(..., n_channels, max_intervals)`.
    weights: `ChangepointInfo.weights`, shape `(n_channels, max_intervals,
      n_times)`.

  Returns:
    The time effect `phi`, shape `(..., n_times, n_channels)`: each time
    period's value is the adjustment of the interval that contains it.
  """
  return backend.einsum(
      '...vk,vkt->...tv', varphi, backend.cast(weights, varphi.dtype)
  )


def time_varying_coefficients(
    *,
    beta_gx: backend.Tensor,
    zeta: backend.Tensor,
    phi: backend.Tensor,
    media_effects_dist: str,
) -> backend.Tensor:
  """Returns the geo and time coefficients of channels with changepoints.

  Args:
    beta_gx: The geo coefficients without the time effect, shape `(..., n_geos,
      n_channels)`. For log-normal effects this is `exp(beta + eta *
      beta_g_dev)`; for normal effects it is `beta + eta * beta_g_dev`.
    zeta: The scale of the time effect, shape `(..., n_channels)`.
    phi: The time effect from `expand_intervals`, shape `(..., n_times,
      n_channels)`.
    media_effects_dist: `constants.MEDIA_EFFECTS_LOG_NORMAL` or
      `constants.MEDIA_EFFECTS_NORMAL`.

  Returns:
    The coefficients, shape `(..., n_geos, n_times, n_channels)`.
  """
  time_effect = zeta[..., backend.newaxis, :] * phi
  if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
    return (
        beta_gx[..., :, backend.newaxis, :]
        + time_effect[..., backend.newaxis, :, :]
    )
  return (
      beta_gx[..., :, backend.newaxis, :]
      * backend.exp(time_effect)[..., backend.newaxis, :, :]
  )


# TODO: Share the steps that `solve_beta` and
# `ModelEquations.calculate_beta_x` have in common through feature-neutral
# helpers in `equations.py`, keeping models without changepoints bit-identical.
def solve_beta(
    *,
    incremental_outcome_x: backend.Tensor,
    linear_predictor_counterfactual_difference: backend.Tensor,
    eta_x: backend.Tensor,
    beta_gx_dev: backend.Tensor,
    zeta: backend.Tensor,
    phi: backend.Tensor,
    population: backend.Tensor,
    population_scaled_stdev: backend.Tensor | float,
    revenue_per_kpi: backend.Tensor | None,
    media_effects_dist: str,
) -> backend.Tensor:
  """Solves `beta` of channels with changepoints from their incremental outcome.

  This is the counterpart of `ModelEquations.calculate_beta_x` for channels
  with changepoints. `beta` is chosen so that the incremental outcome over all
  geos and time periods, including the time effect, equals
  `incremental_outcome_x`. This keeps ROI, mROI and contribution priors on the
  whole training period, as for channels without changepoints. All inputs
  only include the channels with changepoints.

  Args:
    incremental_outcome_x: The incremental outcome implied by the prior or
      posterior draw, shape `(..., n_channels)`.
    linear_predictor_counterfactual_difference: The difference between the
      treatment and its counterfactual on the linear predictor scale, shape
      `(..., n_geos, n_times, n_channels)`.
    eta_x: The geo random effect scale, shape `(..., n_channels)`.
    beta_gx_dev: The standard-normal geo deviations, shape `(..., n_geos,
      n_channels)`.
    zeta: The scale of the time effect, shape `(..., n_channels)`.
    phi: The time effect from `expand_intervals`, shape `(..., n_times,
      n_channels)`.
    population: The geo populations, shape `(n_geos,)`.
    population_scaled_stdev: The standard deviation used to scale the
      population-scaled KPI.
    revenue_per_kpi: The revenue per KPI, shape `(n_geos, n_times)`, or `None`
      to use 1.
    media_effects_dist: `constants.MEDIA_EFFECTS_LOG_NORMAL` or
      `constants.MEDIA_EFFECTS_NORMAL`.

  Returns:
    `beta` of the channels with changepoints, shape `(..., n_channels)`.
  """
  dtype = linear_predictor_counterfactual_difference.dtype
  if revenue_per_kpi is None:
    revenue_per_kpi = backend.ones(
        linear_predictor_counterfactual_difference.shape[-3:-1], dtype=dtype
    )
  # Incremental outcome per unit coefficient, by geo, time and channel.
  outcome_per_beta = backend.einsum(
      '...gtx,gt,g,->...gtx',
      linear_predictor_counterfactual_difference,
      backend.cast(revenue_per_kpi, dtype),
      backend.cast(population, dtype),
      backend.cast(population_scaled_stdev, dtype),
  )
  if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
    random_effects_term = backend.einsum(
        '...gtx,...gx,...x->...x', outcome_per_beta, beta_gx_dev, eta_x
    ) + backend.einsum('...gtx,...tx,...x->...x', outcome_per_beta, phi, zeta)
    return (incremental_outcome_x - random_effects_term) / backend.einsum(
        '...gtx->...x', outcome_per_beta
    )
  # For log-normal effects, beta_gt = exp(beta) * exp(random effects), so
  # exp(beta) is the incremental outcome over the summed outcome per unit
  # exp(beta).
  geo_effect = backend.expand_dims(
      beta_gx_dev * eta_x[..., backend.newaxis, :], axis=-2
  )
  time_effect = backend.expand_dims(
      phi * zeta[..., backend.newaxis, :], axis=-3
  )
  denominator = backend.einsum(
      '...gtx,...gtx->...x',
      outcome_per_beta,
      backend.exp(geo_effect + time_effect),
  )
  return backend.log(incremental_outcome_x) - backend.log(denominator)  # pyrefly: ignore[bad-argument-type]


def merge_channels(
    *, x: backend.Tensor, sub: backend.Tensor, info: ChangepointInfo
) -> backend.Tensor:
  """Replaces the columns of channels with changepoints.

  Args:
    x: A tensor with all channels of the type on its last axis.
    sub: The replacement values, with only the channels with changepoints on its
      last axis and the same other dimensions as `x`.
    info: The `ChangepointInfo` of the channel type.

  Returns:
    `x` with the columns of channels with changepoints replaced by `sub`.
  """
  return backend.gather(
      backend.concatenate([x, sub], axis=-1),  # pyrefly: ignore[bad-argument-type]
      backend.to_tensor(info.merge_indices),
      axis=-1,
  )
