# Copyright 2024 The Meridian Authors.
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

"""Auxiliary functions for knots calculations."""

import bisect
from collections.abc import Collection, Sequence
import dataclasses
import numpy as np


__all__ = [
    'KnotInfo',
    'get_knot_info',
    'l1_distance_weights',
]


# TODO: Reimplement with a more readable method.
def _find_neighboring_knots_indices(
    times: np.ndarray,
    knot_locations: np.ndarray,
) -> Sequence[Sequence[int] | None]:
  """Return indices of neighboring knot locations.

  Returns indices in `knot_locations` that correspond to the neighboring knot
  locations for each time period. If a time point is at or before the first
  knot, the first knot is the only neighboring knot. If a time point is after
  the last knot, the last knot is the only neighboring knot.

  Args:
    times: Times `0, 1, 2,..., (n_times-1)`.
    knot_locations: The location of knots within `0, 1, 2,..., (n_times-1)`.

  Returns:
    List of length `n_times`. Each element is the indices of the neighboring
    knot locations for the respective time period. If a time point is at or
    before the first knot, the first knot is the only neighboring knot. If a
    time point is after the last knot, the last knot is the only neighboring
    knot.
  """
  neighboring_knots_indices = [None] * len(times)
  for t in times:
    # knot_locations assumed to be sorted.
    if t <= knot_locations[0]:
      neighboring_knots_indices[t] = [0]
    elif t >= knot_locations[-1]:
      neighboring_knots_indices[t] = [len(knot_locations) - 1]
    else:
      bisect_index = bisect.bisect_left(knot_locations, t)
      neighboring_knots_indices[t] = [bisect_index - 1, bisect_index]
  return neighboring_knots_indices


def l1_distance_weights(
    n_times: int, knot_locations: np.ndarray[int, np.dtype[int]]
) -> np.ndarray:
  """Computes weights at knots for every time period.

  The two neighboring knots inform the weight estimate of a particular time
  period. The amount each of the two neighboring knots inform the weight at a
  time period depends on how close (L1 distance) they are to the time period. If
  a time point coincides with a knot location, then 100% weight is given to that
  knot. If a time point lies outside the range of knots, then 100% weight is
  given to the nearest endpoint knot.

  This function computes an `(n_knots, n_times)` array of weights that are used
  to model trend and seasonality. For a given time, the array contains two
  non-zero weights. The weights are inversely proportional to the L1 distance
  from the given time to the neighboring knots. The two weights are normalized
  such that they sum to 1.

  Args:
    n_times: The number of time points.
    knot_locations: The location of knots within `0, 1, 2,..., (n_times-1)`.

  Returns:
    A weight array with dimensions `(n_knots, n_times)` with values summing up
    to 1 for each time period when summing over knots.
  """
  if knot_locations.ndim != 1:
    raise ValueError('`knot_locations` must be one-dimensional.')
  if not np.all(knot_locations == np.sort(knot_locations)):
    raise ValueError('`knot_locations` must be sorted.')
  if len(knot_locations) <= 1:
    raise ValueError('Number of knots must be greater than 1.')
  if len(knot_locations) != len(np.unique(knot_locations)):
    raise ValueError('`knot_locations` must be unique.')
  if np.any(knot_locations < 0):
    raise ValueError('knot_locations must be positive.')
  if np.any(knot_locations >= n_times):
    raise ValueError('knot_locations must be less than `n_times`.')

  times = np.arange(n_times)
  time_minus_knot = abs(knot_locations[:, np.newaxis] - times[np.newaxis, :])

  w = np.zeros(time_minus_knot.shape, dtype=np.float32)
  neighboring_knots_indices = _find_neighboring_knots_indices(
      times, knot_locations
  )
  for t in times:
    idx = neighboring_knots_indices[t]
    if len(idx) == 1:
      w[idx, t] = 1
    else:
      # Weight is in proportion to how close the two neighboring knots are.
      w[idx, t] = 1 - (time_minus_knot[idx, t] / time_minus_knot[idx, t].sum())

  return w


def _get_equally_spaced_knot_locations(n_times, n_knots):
  """Equally spaced knot locations starting at the endpoints."""
  return np.linspace(0, n_times - 1, n_knots, dtype=int)


@dataclasses.dataclass(frozen=True)
class KnotInfo:
  """Contains the number of knots, knot locations, and weights.

  Attributes:
    n_knots: The number of knots
    knot_locations: The location of knots
    weights: The weights used to multiply with the knot values to get time-
      varying coefficients.
  """

  n_knots: int
  knot_locations: np.ndarray[int, np.dtype[int]]
  weights: np.ndarray[int, np.dtype[float]]


def get_knot_info(
    n_times: int,
    knots: int | Collection[int] | None,
    is_national: bool = False,
) -> KnotInfo:
  """Returns the number of knots, knot locations, and weights.

  Args:
    n_times: The number of time periods in the data.
    knots: An optional integer or a collection of integers indicating the knots
      used to estimate time effects. When `knots` is a collection of integers,
      the knot locations are provided by that collection. Zero corresponds to a
      knot at the first time period, one corresponds to a knot at the second
      time, ..., and `(n_times - 1)` corresponds to a knot at the last time
      period. When `knots` is an integer, then there are knots with locations
      equally spaced across the time periods (including knots at zero and
      `(n_times - 1)`. When `knots` is `1`, there is a single common regression
      coefficient used for all time periods. If `knots` is `None`, then the
      numbers of knots used is equal to the number of time periods. This is
      equivalent to each time period having its own regression coefficient.
    is_national: A boolean indicator whether to adapt the knot information for a
      national model.

  Returns:
    A KnotInfo that contains the number of knots, the location of knots, and the
    weights used to multiply with the knot values to get time-varying
    coefficients.
  """

  if isinstance(knots, int):
    if knots < 1:
      raise ValueError('If knots is an integer, it must be at least 1.')
    elif knots > n_times:
      raise ValueError(
          f'The number of knots ({knots}) cannot be greater than the number of'
          f' time periods in the kpi ({n_times}).'
      )
    elif is_national and knots == n_times:
      raise ValueError(
          f'Number of knots ({knots}) must be less than number of time periods'
          f' ({n_times}) in a nationally aggregated model.'
      )
    n_knots = knots
    knot_locations = _get_equally_spaced_knot_locations(n_times, n_knots)
  elif isinstance(knots, Collection) and knots:
    if any(k < 0 for k in knots):
      raise ValueError('Knots must be all non-negative.')
    if any(k >= n_times for k in knots):
      raise ValueError(
          'Knots must all be less than the number of time periods.'
      )
    n_knots = len(knots)
    # np.unique also sorts
    knot_locations = np.unique(knots)
  elif isinstance(knots, Collection):
    raise ValueError('Knots cannot be empty.')
  else:
    # knots is None
    n_knots = 1 if is_national else n_times
    knot_locations = _get_equally_spaced_knot_locations(n_times, n_knots)

  if n_knots == 1:
    weights = np.ones((1, n_times), dtype=np.float32)
  else:
    weights = l1_distance_weights(n_times, knot_locations)

  return KnotInfo(n_knots, knot_locations, weights)
