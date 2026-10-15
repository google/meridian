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
`ChangepointInfo`, and the samplers use it to draw the interval adjustments and
build each time period's coefficient.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
from typing import TypeAlias

from meridian import backend
from meridian import constants
import numpy as np

__all__ = [
    'ChangepointIndices',
    'ChangepointInfo',
    'changepoint_knot_overlaps',
    'expand_intervals',
    'get_changepoint_info',
    'interval_adjustments_distribution',
    'interval_weights',
    'merge_channels',
    'time_varying_coefficients',
]

# Maps a channel name to its changepoints, as indices into the time periods.
ChangepointIndices: TypeAlias = Mapping[str, Sequence[int]]


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
    changepoints: ChangepointIndices,
    channel_names: Sequence[str],
    max_intervals: int | None = None,
) -> ChangepointInfo | None:
  """Builds the `ChangepointInfo` of one channel type.

  Args:
    n_times: The number of time periods.
    changepoints: The changepoints of each channel. Each changepoint starts a
      new interval, so it must be between 1 and `n_times - 1`. Channels of other
      types are ignored.
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
    changepoints: ChangepointIndices,
) -> ChangepointIndices:
  """Returns the changepoints that are exactly on a baseline knot.

  Takes resolved knot positions, whatever their source (a list, `knots=<int>`
  in `ModelSpec`, the defaults, or automatic knot selection). The first time
  period is ignored: it always starts the first interval and is usually a knot
  too. A single knot gives a flat baseline, so it never overlaps.

  Args:
    knot_locations: The knot positions, as indices into the time periods.
    changepoints: The changepoints of each channel.

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


def interval_adjustments_distribution(
    info: ChangepointInfo, name: str
) -> backend.tfd.Distribution:
  """Returns the distribution of the interval adjustments `varphi`.

  The adjustments are standard normal, so they have no prior distribution to
  set. Prior and posterior sampling both use this distribution.

  Args:
    info: The `ChangepointInfo` of the channel type.
    name: The parameter name, `constants.VARPHI_M` or `constants.VARPHI_RF`.

  Returns:
    A distribution with event shape `(n_channels, max_intervals)`.
  """
  return backend.tfd.Sample(
      backend.tfd.Normal(
          loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
          scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
      ),
      [info.n_channels, info.max_intervals],
      name=name,
  )


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
