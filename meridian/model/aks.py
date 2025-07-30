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

"""This module defines functions for calculating Adaptive Spline (A-Spline) for knot selection in Meridian."""

import copy
import math
from typing import Any
# from meridian import constants
from meridian.data import input_data
import numpy as np
# TODO: b/437393442 - migrate patsy
from patsy import highlevel
from statsmodels.regression import linear_model
import tensorflow as tf

__all__ = ["AKS"]


class AKS:
  """Class for automatically selecting knots in Meridian Core Library."""

  def __init__(self):
    self._base_pen = np.logspace(-1, 2, 100)
    self._degree = 1

  def automatic_knot_selection(
      self, data: input_data.InputData
  ) -> tuple[list[float], linear_model.OLS]:
    """Calculates the optimal number of knots for Meridian model using Automatic knot selection with A-spline.

    Args:
      data: InputData object to calculate the knots from.

    Returns:
      Selected knots and the corresponding B-spline model.
    """
    x, y = self._calculate_xy_from_input_data(data)
    knots, min_internal_knots, max_internal_knots = (
        self._calculate_initial_knots(x)
    )
    self._validate_knots(
        knots, min_internal_knots, max_internal_knots
    )  # , data)
    geo_scaling_factor = 1 / np.sqrt(len(data.geo))
    pen = geo_scaling_factor * self._base_pen

    aspl = self._aspline(x=x, y=y, knots=knots, pen=pen)
    n_knots = np.array([len(x) for x in aspl["knots_sel"]])
    feasible_idx = np.where(
        (n_knots >= min_internal_knots) & (n_knots <= max_internal_knots)
    )[0]
    information_criterion = aspl["ebic"][feasible_idx]
    knots_sel = [aspl["knots_sel"][i] for i in feasible_idx]
    model = [aspl["model"][i] for i in feasible_idx]
    opt_idx = max(
        np.where(information_criterion == min(information_criterion))[0]
    )

    return list(knots_sel[opt_idx]), model[opt_idx]

  def _calculate_xy_from_input_data(self, data: input_data.InputData):
    """Calculates x and y from input data.

    Args:
      data: Input data.

    Returns:
      x and y data.
    """
    n_geos = len(data.geo)
    n_times = len(data.time)
    kpi = data.kpi.values
    pop = data.population.values

    x = np.reshape(
        np.repeat([range(n_times)], n_geos, axis=0), shape=(n_geos * n_times,)
    )
    pop_scaled_kpi = tf.math.divide_no_nan(kpi, pop[:, tf.newaxis])
    pop_scaled_mean = tf.reduce_mean(pop_scaled_kpi)
    pop_scaled_stdev = tf.math.reduce_std(pop_scaled_kpi)
    kpi_scaled = tf.math.divide_no_nan(
        tf.math.divide_no_nan(kpi, pop[:, tf.newaxis]) - pop_scaled_mean,
        pop_scaled_stdev,
    )
    y_tensor = kpi_scaled - (tf.reduce_mean(kpi_scaled, axis=1))[:, tf.newaxis]
    y = tf.reshape(y_tensor, shape=(n_geos * n_times,)).numpy()

    return x, y

  def _calculate_initial_knots(
      self, x: np.ndarray
  ) -> tuple[np.ndarray, int, int]:
    """Calculates initial knots based on unique x values.

    Args:
      x: Input data.

    Returns:
      A tuple containing:
        - The calculated knots.
        - The minimum number of internal knots.
        - The maximum number of internal knots.
    """
    x_vals_unique = np.unique(x)
    min_x_data, max_x_data = x_vals_unique.min(), x_vals_unique.max()
    knots = x_vals_unique[
        (x_vals_unique > min_x_data) & (x_vals_unique < max_x_data)
    ]
    knots = np.sort(np.unique(knots))
    knots = knots[:-1]
    min_internal_knots = 1
    max_internal_knots = len(knots) - 10

    return knots, min_internal_knots, max_internal_knots

  def _validate_knots(
      self,
      knots: np.ndarray,
      min_internal_knots: int,
      max_internal_knots: int,
      # data: input_data.InputData,
  ):
    """Validates the knots against the input data.

    Args:
      knots: The selected knots.
      min_internal_knots: The minimum number of internal knots.
      max_internal_knots: The maximum number of internal knots.

    Raises:
      ValueError: If the minimum number of internal knots is greater than the
      total number of initial knots.
      ValueError: If the maximum number of internal knots is less than the
      minimum number of internal knots.
      ValueError: If the maximum number of internal knots is less than the
      total times minus num media variables minus num control variables.
    """
    if min_internal_knots > len(knots):
      raise ValueError(
          "The minimum number of internal knots cannot be greater than the"
          " totalnumber of initial knots."
      )
    if max_internal_knots < min_internal_knots:
      raise ValueError(
          "The maximum number of internal knots cannot be less than the minimum"
          " number of internal knots."
      )
    # n_medias = (
    #    len(data.media.coords[constants.MEDIA_CHANNEL].values)
    #    if data.media is not None
    #    else 0
    # )
    # n_controls = (
    #    len(data.controls.coords[constants.CONTROL_VARIABLE].values)
    #    if data.controls is not None
    #    else 0
    # )
    # if max_internal_knots < len(data.time) - n_medias - n_controls:
    #   raise ValueError(
    #     "The maximum number of internal knots cannot be less than the total"
    #     " times minus num media variables minus num control variables."
    #   )

  def _aspline(
      self,
      x: np.ndarray,
      y: np.ndarray,
      knots: np.ndarray,
      pen: np.ndarray,
      max_iter: int = 1000,
      epsilon: float = 1e-5,
      tol: float = 1e-6,
  ) -> dict[str, Any]:
    """Fit B-splines with automatic knot selection.

    Args:
      x: Input data
      y: Input data of length of x
      knots: Internal knots used for spline regression
      pen: A vector of positive penalty values. The adaptive spline regression
        is performed for every value of pen
      max_iter: Maximum number of iterations in the main loop.
      epsilon: Value of the constant in the adaptive ridge procedure (see
        Frommlet, F., Nuel, G. (2016) An Adaptive Ridge Procedure for L0
        Regularization.)
      tol: The tolerance chosen to diagnose convergence of the adaptive ridge
        procedure.

    Returns:
      A dictionary of the following items:
        sel: A list of selection coefficients for every value of pen
        knots_sel: A list of selected knots for every value of pen
        model: A list of fitted models for every value of pen
        par: A list of estimated regression coefficients for every value of pen
        sel_mat: A matrix of selected knots for every value of pen
        aic: A list of AIC values for every value of pen
        bic: A list of BIC values for every value of pen
        ebic: A list of EBIC values for every value of pen
    """
    bs_cmd = (
        "bs(x,knots=["
        + ",".join(map(str, knots))
        + "],degree="
        + str(self._degree)
        + ",include_intercept=True)-1"
    )
    xmat = highlevel.dmatrix(bs_cmd, {"x": x})
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
        [None] * len(pen) for _ in range(10)
    )
    old_sel, w = [np.ones(ncol - self._degree - 1) for _ in range(2)]
    par = np.ones(ncol)
    ind_pen = 0
    for _ in range(max_iter):
      par = self._wridge_solver(
          xx_rot, xy, self._degree, pen[ind_pen], w, old_par=par
      )
      par_diff = np.diff(par, n=self._degree + 1)

      w = 1 / (par_diff**2 + epsilon**2)
      sel = w * par_diff**2
      converge = max(abs(old_sel - sel)) < tol
      if converge:
        sel_ls[ind_pen] = sel
        knots_sel[ind_pen] = knots[sel > 0.99]
        bs_cmd_iter = (
            f"bs(x,knots=[{','.join(map(str, knots_sel[ind_pen]))}],degree={self._degree},include_intercept=True)-1"
        )
        design_mat = highlevel.dmatrix(bs_cmd_iter, {"x": x})
        x_sel[ind_pen] = design_mat
        bs_model = linear_model.OLS(y, x_sel[ind_pen]).fit()
        model[ind_pen] = bs_model
        coefs = np.zeros(ncol, dtype=np.float64)
        idx = np.concat([sel > 0.99, np.repeat(True, self._degree + 1)])
        coefs[idx] = bs_model.params
        par_ls[ind_pen] = coefs

        loglik[ind_pen] = sum(bs_model.resid**2 / sigma0sq) / 2
        dim[ind_pen] = len(knots_sel[ind_pen]) + self._degree + 1
        aic[ind_pen] = 2 * dim[ind_pen] + 2 * loglik[ind_pen]
        bic[ind_pen] = (
            np.log(np.float32(nrow)) * dim[ind_pen] + 2 * loglik[ind_pen]
        )
        ebic[ind_pen] = bic[ind_pen] + 2 * np.log(
            np.float32(math.comb(ncol, design_mat.shape[1]))
        )
        ind_pen = ind_pen + 1
      if ind_pen > len(pen) - 1:
        break
      old_sel = sel

    sel_mat = np.round(np.stack(sel_ls, axis=-1), 1)
    return {
        "sel": sel_ls,
        "knots_sel": knots_sel,
        "model": model,
        "par": par_ls,
        "sel_mat": sel_mat,
        "aic": np.array(aic),
        "bic": np.array(bic),
        "ebic": np.array(ebic),
    }

  def _mat2rot(self, band_mat: np.ndarray) -> np.ndarray:
    """Rotate a symmetric band matrix to get the rotated matrix associated.

    Each column of the rotated matrix correspond to a diagonal. The first column
    is the main diagonal, the second one is the upper-diagonal and so on.
    Artificial 0s are placed at the end of each column if necessary.

    Args:
      band_mat: Band square matrix

    Returns:
      The rotated matrix of band_mat
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
    """Create the penalty matrix for A-Spline.

    Args:
      w: Vector of weights
      diff: Order of the differences to be applied to the parameters. Must be a
        strictly positive integer

    Returns:
      Weighted penalty matrix D'diag(w)D, where
      D = diff(diag(len(w) + diff), differences = diff)}. Only the non-null
      superdiagonals of the weight matrix are returned, each column
      corresponding
      to a diagonal.
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
    """Fast LDL decomposition of symmetric band matrix of length k.

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

  def _bandsolve(
      self, rot_mat: np.ndarray, rhs_mat: np.ndarray | None = None
  ) -> np.ndarray:
    """Main function to solve a symmetric bandlinear system Ax = b.

    Here A is the rotated form of the band matrix and b is the right hand side.

    Args:
      rot_mat: Band square matrix in the rotated form. It's the visual rotation
        by 90 degrees of the matrix, where subdiagonal are discarded.
      rhs_mat: right hand side of the equation. Can be either a vector or a
        matrix. If not supplied, the function return the inverse of rot_mat.

    Returns:
      Solution of the linear problem.
    """

    def _bandsolve_kernel(rot_mat, rhs_mat):
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

    nrow = rot_mat.shape[0]
    ncol = rot_mat.shape[1]
    if (nrow == ncol) & (rot_mat[nrow - 1, ncol - 1] != 0):
      raise ValueError("rot_mat should be a rotated matrix!")
    if rot_mat[nrow - 1, 1] != 0:
      raise ValueError("rot_mat should be a rotated matrix!")

    if rhs_mat is None:
      rhs_mat = np.identity(nrow, dtype=np.float32)
      return _bandsolve_kernel(rot_mat, rhs_mat)
    elif rhs_mat.ndim == 1:
      if len(rhs_mat) != nrow:
        raise ValueError("Dimension problem!")
      else:
        return _bandsolve_kernel(rot_mat, rhs_mat[:, np.newaxis])
    elif rhs_mat.ndim == 2:
      if rhs_mat.shape[0] != nrow:
        raise ValueError("Dimension problem!")
      else:
        return _bandsolve_kernel(rot_mat, rhs_mat[:, np.newaxis])
    else:
      raise ValueError("rhs_mat must either be a vector or a matrix")

  def _wridge_solver(
      self,
      xx_rot: np.ndarray,
      xy: np.ndarray,
      degree: int,
      pen: float,
      w: np.ndarray,
      old_par: np.ndarray,
      max_iter: int = 1000,
      tol: float = 1e-8,
  ) -> np.ndarray | None:
    """Fit B-Splines with weighted penalization over differences of parameters.

    Args:
      xx_rot: The matrix X'X where X is the design matrix. This argument is
        given in the form of a band matrix, i.e., successive columns represent
        superdiagonals.
      xy: The vector of currently estimated points X'y, where y is the
        y-coordinate of the data.
      degree: The degree of the B-splines.
      pen: Positive penalty constant.
      w: Vector of weights. The case w = np.ones(xx_rot.shape[0] - degree - 1)
        corresponds to fitting P-splines with difference order degree + 1. See
        Eilers, P., Marx, B. (1996) Flexible smoothing with B-splines and
        penalties.
      old_par: The previous parameter vector.
      max_iter: Maximum number of Newton-Raphson iterations to be computed.
      tol: The tolerance chosen to diagnose convergence of the adaptive ridge
        procedure.

    Returns:
      The estimated parameter of the spline regression.
    """

    def _hessian_solver(par, xx_rot, xy, pen, w, diff):
      """Inverse the hessian and multiply it by the score.

      Args:
        par: The parameter vector
        xx_rot: The matrix X'X where X is the design matrix. This argument is
          given in the form of a rotated band matrix, i.e., successive columns
          represent superdiagonals.
        xy: The vector of currently estimated points X'y, where y is the
          y-coordinate of the data.
        pen: Positive penalty constant.
        w: Vector of weights.
        diff: The order of the differences of the parameter. Equals degree + 1
          in adaptive spline regression.

      Returns:
        The solution of the linear system: (X'X + pen*D'WD)^{-1} X'y - par
      """
      if xx_rot.shape[1] != diff + 1:
        raise ValueError("Error: xx_rot must have diff + 1 columns")
      return (
          self._bandsolve(xx_rot + pen * self._band_weight(w, diff), xy)[:, 0]
          - par
      )

    par = None
    for _ in range(max_iter):
      par = old_par + _hessian_solver(
          par=old_par, xx_rot=xx_rot, xy=xy, pen=pen, w=w, diff=degree + 1
      )
      ind = old_par != 0
      rel_error = max(abs(par - old_par)[ind] / abs(old_par)[ind])
      if rel_error < tol:
        break
      old_par = par

    return par
