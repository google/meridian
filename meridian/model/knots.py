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

"""Auxiliary functions for knots calculations."""

import bisect
from collections.abc import Collection, Sequence
import copy
import dataclasses
import math
from typing import Any
from meridian import constants
from meridian.data import input_data
import numpy as np
# TODO: b/437393442 - migrate patsy
from patsy import highlevel
from statsmodels.regression import linear_model


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
    enable_aks: bool = False,
    data: input_data.InputData | None = None,
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
    enable_aks: A boolean indicating whether to use the Automatic Knot Selection
      algorithm to select optimal number of knots for running the model instead
      of the default 1 for national and n_times for non-national models.
    data: An Optional InputData object used by the Automatic Knot Selection
      algorithm to calculate optimal number of knots from the provided Input
      Data.
    is_national: A boolean indicator whether to adapt the knot information for a
      national model.

  Returns:
    A KnotInfo that contains the number of knots, the location of knots, and the
    weights used to multiply with the knot values to get time-varying
    coefficients.
  """
  if enable_aks:
    if data is None:
      raise ValueError(
          'If enable_aks is true then input data must be provided.'
      )
    else:
      aks = AKS(data)
      knots = aks.automatic_knot_selection().knots
      n_knots = len(knots)
      knot_locations = knots
  elif isinstance(knots, int):
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


@dataclasses.dataclass(frozen=True)
class AKSResult:
  knots: np.ndarray[int, np.dtype[int]]
  model: linear_model.OLS


class AKS:
  """Class for automatically selecting knots in Meridian Core Library."""

  _BASE_PENALTY = np.logspace(-1, 2, 100)
  _DEGREE = 1

  def __init__(self, data: input_data.InputData):
    self._data = data

  def automatic_knot_selection(
      self,
      base_penalty: np.ndarray | None = None,
      min_internal_knots: int = 1,
      max_internal_knots: int | None = None,
  ) -> AKSResult:
    """Calculates the optimal number of knots for Meridian model using Automatic knot selection with A-spline.

    Args:
      base_penalty: A vector of positive penalty values. The adaptive spline
        regression is performed for every value of penalty.
      min_internal_knots: The minimum number of internal knots. Defaults to 1.
      max_internal_knots: The maximum number of internal knots. If None, this
        value is calculated as the number of initial knots minus the total count
        of all treatment and control variables. Otherwise, the user-provided
        value will be used.

    Returns:
      Selected knots and the corresponding B-spline model.
    """
    if base_penalty is None:
      base_penalty = self._BASE_PENALTY
    n_times = len(self._data.time)
    n_geos = len(self._data.geo)

    y_tensor = self._data.scaled_centered_kpi
    y = np.reshape(y_tensor, (n_geos * n_times,))
    x = np.reshape(
        np.repeat([range(n_times)], n_geos, axis=0), (n_geos * n_times,)
    )

    knots = self._calculate_initial_knots(x)
    max_internal_knots = self._calculate_and_validate_max_internal_knots(
        knots, min_internal_knots, max_internal_knots
    )
    geo_scaling_factor = 1 / np.sqrt(len(self._data.geo))
    penalty = geo_scaling_factor * base_penalty

    aspline = self.aspline(x=x, y=y, knots=knots, penalty=penalty)
    n_knots = np.array([len(x) for x in aspline[constants.KNOTS_SELECTED]])
    feasible_idx = np.where(
        (n_knots >= min_internal_knots) & (n_knots <= max_internal_knots)
    )[0]
    information_criterion = aspline[constants.AIC][feasible_idx]
    knots_sel = [aspline[constants.KNOTS_SELECTED][i] for i in feasible_idx]
    model = [aspline[constants.MODEL][i] for i in feasible_idx]
    opt_idx = max(
        np.where(information_criterion == min(information_criterion))[0]
    )

    return AKSResult(knots_sel[opt_idx], model[opt_idx])

  def _calculate_initial_knots(
      self,
      x: np.ndarray,
  ) -> np.ndarray:
    """Calculates initial knots based on unique x values.

    Args:
      x: A flattened array of indexed time coordinates, repeated n_geos times.
        e.g. [0, 1, 2, 3, ..., 0, 1, 2, 3, ...].

    Returns:
      A tuple containing:
        - The calculated knots.
    """
    x_vals_unique = np.unique(x)
    min_x_data, max_x_data = x_vals_unique.min(), x_vals_unique.max()
    knots = x_vals_unique[
        (x_vals_unique > min_x_data) & (x_vals_unique < max_x_data)
    ]
    knots = np.sort(np.unique(knots))
    # Drop one knot from the set of all knots because the algorithm requires one
    # fewer degree of freedom than the total number of knots to function.
    # Dropping the final knot is a natural and practical choice because it
    # often has minimal impact on the overall model fit.
    return knots[:-1]

  def _calculate_and_validate_max_internal_knots(
      self,
      knots: np.ndarray,
      min_internal_knots: int = 1,
      max_internal_knots: int | None = None,
  ) -> int:
    """Calculates the max internal knots, and validates the range.

    Args:
      knots: Initial knots to calculate max internal knots from.
      min_internal_knots: The minimum number of internal knots.
      max_internal_knots: The maximum number of internal knots. If None, this
        value will be calculated, otherwise will use user provided value.

    Returns:
      The maximum number of internal knots.
    """
    n_features = sum(
        len(feature) if feature is not None else 0
        for feature in [
            self._data.media_channel,
            self._data.rf_channel,
            self._data.organic_media_channel,
            self._data.organic_rf_channel,
            self._data.non_media_channel,
            self._data.control_variable,
        ]
    )

    if max_internal_knots is None:
      max_internal_knots = len(knots) - n_features
    if min_internal_knots > len(knots):
      raise ValueError(
          'The minimum number of internal knots cannot be greater than the'
          ' total number of initial knots.'
      )
    if max_internal_knots < min_internal_knots:
      raise ValueError(
          'The maximum number of internal knots cannot be less than the minimum'
          ' number of internal knots.'
      )
    return max_internal_knots

  def aspline(
      self,
      x: np.ndarray,
      y: np.ndarray,
      knots: np.ndarray,
      penalty: np.ndarray,
      max_iterations: int = 1000,
      epsilon: float = 1e-5,
      tol: float = 1e-6,
  ) -> dict[str, Any]:
    """Fits B-splines with automatic knot selection.

    Args:
      x: A flattened array of indexed time coordinates, repeated n_geos times.
        e.g. [0, 1, 2, 3, ..., 0, 1, 2, 3, ...].
      y: The flattened array of KPI values that have been population-scaled and
        mean-centered by geo.
      knots: Internal knots used for spline regression.
      penalty: A vector of positive penalty values. The adaptive spline
        regression is performed for every value of penalty.
      max_iterations: Maximum number of iterations in the main loop.
      epsilon: Value of the constant in the adaptive ridge procedure (see
        Frommlet, F., Nuel, G. (2016) An Adaptive Ridge Procedure for L0
        Regularization.)
      tol: The tolerance chosen to diagnose convergence of the adaptive ridge
        procedure.

    Returns:
      A dictionary of the following items:
        selection_coefs: A list of selection coefficients for every value of
        penalty.
        knots_selected: A list of selected knots for every value of penalty.
        model: A list of fitted models for every value of penalty.
        regression_coefs: A list of estimated regression coefficients for every
        value of penalty.
        selected_matrix: A matrix of selected knots for every value of penalty.
        aic: A list of AIC values for every value of penalty.
        bic: A list of BIC values for every value of penalty.
        ebic: A list of EBIC values for every value of penalty.
    """
    if x.ndim != 1 or y.ndim != 1:
      raise ValueError(
          'Provided x and y args for aspline must both be 1 dimensional!'
      )

    bs_cmd = (
        'bs(x,knots=['
        + ','.join(map(str, knots))
        + '],degree='
        + str(self._DEGREE)
        + ',include_intercept=True)-1'
    )
    xmat = highlevel.dmatrix(bs_cmd, {'x': x})
    nrow = xmat.shape[0]
    ncol = xmat.shape[1]

    xx = xmat.T.dot(xmat)
    xy = xmat.T.dot(y)
    xx_rot = np.concat(
        [
            self._mat2rot(xx + (1e-20 * np.identity(ncol))),
            np.zeros(ncol)[:, np.newaxis],
        ],
        axis=1,
    )
    sigma0sq = linear_model.OLS(y, xmat).fit().mse_resid ** 2
    model, x_sel, knots_sel, sel_ls, par_ls, aic, bic, ebic, dim, loglik = (
        [None] * len(penalty) for _ in range(10)
    )
    old_sel, w = [np.ones(ncol - self._DEGREE - 1) for _ in range(2)]
    par = np.ones(ncol)
    index_penalty = 0
    for _ in range(max_iterations):
      par = self._wridge_solver(
          xx_rot, xy, self._DEGREE, penalty[index_penalty], w, old_par=par
      )
      par_diff = np.diff(par, n=self._DEGREE + 1)

      w = 1 / (par_diff**2 + epsilon**2)
      sel = w * par_diff**2
      converge = max(abs(old_sel - sel)) < tol
      if converge:
        sel_ls[index_penalty] = sel
        knots_sel[index_penalty] = knots[sel > 0.99]
        bs_cmd_iter = (
            f"bs(x,knots=[{','.join(map(str, knots_sel[index_penalty]))}],degree={self._DEGREE},include_intercept=True)-1"
        )
        design_mat = highlevel.dmatrix(bs_cmd_iter, {'x': x})
        x_sel[index_penalty] = design_mat
        bs_model = linear_model.OLS(y, x_sel[index_penalty]).fit()
        model[index_penalty] = bs_model
        coefs = np.zeros(ncol, dtype=np.float32)
        idx = np.concat([sel > 0.99, np.repeat(True, self._DEGREE + 1)])
        coefs[idx] = bs_model.params
        par_ls[index_penalty] = coefs

        loglik[index_penalty] = sum(bs_model.resid**2 / sigma0sq) / 2
        dim[index_penalty] = len(knots_sel[index_penalty]) + self._DEGREE + 1
        aic[index_penalty] = 2 * dim[index_penalty] + 2 * loglik[index_penalty]
        bic[index_penalty] = (
            np.log(nrow) * dim[index_penalty] + 2 * loglik[index_penalty]
        )
        ebic[index_penalty] = bic[index_penalty] + 2 * np.log(
            np.float32(math.comb(ncol, design_mat.shape[1]))
        )
        index_penalty = index_penalty + 1
      if index_penalty > len(penalty) - 1:
        break
      old_sel = sel

    sel_mat = np.round(np.stack(sel_ls, axis=-1), 1)
    return {
        constants.SELECTION_COEFS: sel_ls,
        constants.KNOTS_SELECTED: knots_sel,
        constants.MODEL: model,
        constants.REGRESSION_COEFS: par_ls,
        constants.SELECTED_MATRIX: sel_mat,
        constants.AIC: np.array(aic),
        constants.BIC: np.array(bic),
        constants.EBIC: np.array(ebic),
    }

  def _mat2rot(self, band_mat: np.ndarray) -> np.ndarray:
    """Rotates a symmetric band matrix to get the rotated matrix associated.

    Each column of the rotated matrix corresponds to a diagonal. The first
    column is the main diagonal, the second one is the upper-diagonal and so on.
    Artificial 0s are placed at the end of each column if necessary.

    Args:
      band_mat: The band square matrix to be rotated.

    Returns:
      The rotated matrix of band_mat.
    """
    p = band_mat.shape[1]
    l = 0
    for i in range(p):
      lprime = np.where(band_mat[i, :] != 0)[0]
      l = np.maximum(l, lprime[len(lprime) - 1] - i)

    rot_mat = np.zeros([p, l + 1])
    rot_mat[:, 0] = np.diag(band_mat)
    if l > 0:
      for j in range(l):
        rot_mat[:, j + 1] = np.concat([
            np.diag(band_mat[range(p - j - 1), :][:, range(j + 1, p)]),
            np.zeros(j + 1),
        ])
    return rot_mat

  def _band_weight(self, w: np.ndarray, diff: int) -> np.ndarray:
    """Creates the penalty matrix for A-Spline.

    Args:
      w: Vector of weights.
      diff: Order of the differences to be applied to the parameters. Must be a
        strictly positive integer.

    Returns:
      Weighted penalty matrix D'diag(w)D, where
      D = diff(diag(len(w) + diff), differences = diff)}. Only the non-null
      superdiagonals of the weight matrix are returned, each column
      corresponding to a diagonal.
    """
    ws = len(w)
    rows = ws + diff
    cols = diff + 1

    # Compute the entries of the difference matrix
    binom = np.zeros(cols, dtype=np.int32)
    for i in range(cols):
      binom[i] = math.comb(diff, i) * (-1) ** i

    # Compute the limit indices
    ind_mat = np.zeros([rows, 2], dtype=np.int32)
    for ind in range(rows):
      ind_mat[ind, 0] = 0 if ind - diff < 0 else ind - diff
      ind_mat[ind, 1] = ind if ind < ws - 1 else ws - 1

    # Main loop
    result = np.zeros([rows, cols])
    for j in range(cols):
      for i in range(rows - j):
        temp = 0.0
        for k in range(ind_mat[i + j, 0], ind_mat[i, 1] + 1):
          temp += binom[i - k] * binom[i + j - k] * w[k]
        result[i, j] = temp

    return result

  def _ldl(self, rot_mat: np.ndarray) -> np.ndarray:
    """Solves the Fast LDL decomposition of symmetric band matrix of length k.

    Args:
      rot_mat: Rotated row-wised matrix of dimensions n*k, with first column
        corresponding to the diagonal, the second to the first super-diagonal
        and so on.

    Returns:
      Solution of the LDL decomposition.
    """
    n = rot_mat.shape[0]
    m = rot_mat.shape[1] - 1
    rot_mat_new = copy.deepcopy(rot_mat)
    for i in range(1, n + 1):
      j0 = np.maximum(1, i - m)
      for j in range(j0, i + 1):
        for k in range(j0, j):
          rot_mat_new[j - 1, i - j] -= (
              rot_mat_new[k - 1, i - k]
              * rot_mat_new[k - 1, j - k]
              * rot_mat_new[k - 1, 0]
          )
        if i > j:
          rot_mat_new[j - 1, i - j] /= rot_mat_new[j - 1, 0]

    return rot_mat_new

  def _bandsolve_kernel(
      self, rot_mat: np.ndarray, rhs_mat: np.ndarray
  ) -> np.ndarray:
    """Solves the symmetric bandlinear system Ax = b.

    This is the kernel function that solves the system, where A is the rotated
    form of the band matrix and b is the right hand side.

    Args:
      rot_mat: Band square matrix in the rotated form. It's the visual rotation
        by 90 degrees of the matrix, where subdiagonal are discarded.
      rhs_mat: right hand side of the equation. Can be either a vector or a
        matrix. If not supplied, the function return the inverse of rot_mat.

    Returns:
      Solution of the linear problem.
    """
    rot_mat_ldl = self._ldl(rot_mat)
    x = copy.deepcopy(rhs_mat)
    n = rot_mat.shape[0]
    k = rot_mat_ldl.shape[1] - 1
    l = rhs_mat.shape[1]

    for l in range(l):
      # solve b=inv(L)b
      for i in range(2, n + 1):
        jmax = np.minimum(i - 1, k)
        for j in range(1, jmax + 1):
          x[i - 1, l] -= rot_mat_ldl[i - j - 1, j] * x[i - j - 1, l]

      # solve b=b/D
      for i in range(n):
        x[i, l] /= rot_mat_ldl[i, 0]

      # solve b=inv(t(L))b=inv(L*D*t(L))b
      for i in range(n - 1, 0, -1):
        jmax = np.minimum(n - i, k)
        for j in range(1, jmax + 1):
          x[i - 1, l] -= rot_mat_ldl[i - 1, j] * x[i + j - 1, l]

    return x

  def _bandsolve(self, rot_mat: np.ndarray, rhs_mat: np.ndarray) -> np.ndarray:
    """Solves the symmetric bandlinear system Ax = b.

    Here A is the rotated form of the band matrix and b is the right hand side.

    Args:
      rot_mat: Band square matrix in the rotated form. It's the visual rotation
        by 90 degrees of the matrix, where subdiagonal are discarded.
      rhs_mat: right hand side of the equation. Can be either a vector or a
        matrix. If not supplied, the function return the inverse of rot_mat.

    Returns:
      Solution of the linear problem.
    """

    nrow = rot_mat.shape[0]
    ncol = rot_mat.shape[1]
    if (nrow == ncol) & (rot_mat[nrow - 1, ncol - 1] != 0):
      raise ValueError('rot_mat should be a rotated matrix!')
    if rot_mat[nrow - 1, 1] != 0:
      raise ValueError('rot_mat should be a rotated matrix!')
    if len(rhs_mat) != nrow:
      raise ValueError('Dimension problem!')

    return self._bandsolve_kernel(rot_mat, rhs_mat[:, np.newaxis])

  def _wridge_solver(
      self,
      xx_rot: np.ndarray,
      xy: np.ndarray,
      degree: int,
      penalty: float,
      w: np.ndarray,
      old_par: np.ndarray,
      max_iterations: int = 1000,
      tol: float = 1e-8,
  ) -> np.ndarray | None:
    """Fits B-Splines with weighted penalization over differences of parameters.

    Args:
      xx_rot: The matrix X'X where X is the design matrix. This argument is
        given in the form of a band matrix, i.e., successive columns represent
        superdiagonals.
      xy: The vector of currently estimated points X'y, where y is the
        y-coordinate of the data.
      degree: The degree of the B-splines.
      penalty: Positive penalty constant.
      w: Vector of weights. The case w = np.ones(xx_rot.shape[0] - degree - 1)
        corresponds to fitting P-splines with difference order degree + 1. See
        Eilers, P., Marx, B. (1996) Flexible smoothing with B-splines and
        penalties.
      old_par: The previous parameter vector.
      max_iterations: Maximum number of Newton-Raphson iterations to be
        computed.
      tol: The tolerance chosen to diagnose convergence of the adaptive ridge
        procedure.

    Returns:
      The estimated parameter of the spline regression.
    """

    def _hessian_solver(par, xx_rot, xy, penalty, w, diff):
      """Inverts the hessian and multiplies it by the score.

      Args:
        par: The parameter vector.
        xx_rot: The matrix X'X where X is the design matrix. This argument is
          given in the form of a rotated band matrix, i.e., successive columns
          represent superdiagonals.
        xy: The vector of currently estimated points X'y, where y is the
          y-coordinate of the data.
        penalty: Positive penalty constant.
        w: Vector of weights.
        diff: The order of the differences of the parameter. Equals degree + 1
          in adaptive spline regression.

      Returns:
        The solution of the linear system: (X'X + penalty*D'WD)^{-1} X'y - par
      """
      if xx_rot.shape[1] != diff + 1:
        raise ValueError('Error: xx_rot must have diff + 1 columns')
      return (
          self._bandsolve(xx_rot + penalty * self._band_weight(w, diff), xy)[
              :, 0
          ]
          - par
      )

    par = None
    for _ in range(max_iterations):
      par = old_par + _hessian_solver(
          par=old_par,
          xx_rot=xx_rot,
          xy=xy,
          penalty=penalty,
          w=w,
          diff=degree + 1,
      )
      index = old_par != 0
      rel_error = max(abs(par - old_par)[index] / abs(old_par)[index])
      if rel_error < tol:
        break
      old_par = par

    return par
