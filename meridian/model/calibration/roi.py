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

"""ROI calibration logic for Meridian."""

from __future__ import annotations

from collections.abc import Sequence
import datetime

from meridian import backend
from meridian import constants as meridian_constants
from meridian.model import adstock_hill
from meridian.model.calibration import base
from meridian.model.calibration import constants as calibration_constants
import numpy as np
import scipy.optimize as opt

# Constants for grid evaluation and calibration.
_SCOUT_GRID_MULTIPLIER = 5.0
_DEFAULT_SCOUT_MIN = -20.0
_DEFAULT_SCOUT_MAX = 50.0
_SCOUT_POINTS = 2000
_LOWER_PERCENTILE = 0.001
_UPPER_PERCENTILE = 0.999  # 99.8% confidence level interval
_FINE_GRID_POINTS = 10000
_OPTIMIZER_MIN_BOUND = 1e-4
_ZERO_PROBABILITY_MASK = -1e10
_MIN_POSITIVE_MEAN_SAFE = 1e-4
_MIN_POSITIVE_VAR_SAFE = 1e-6

# The recency half-life value (in weeks) used in recency adjustments.
_HALF_LIFE_WEEKS = 52.0
_DAYS_IN_WEEK = 7.0


# The base of the exponential decay in half-life calculations.
_HALF_LIFE_BASE = 0.5

# The minimum proportion of decay weights captured by the experiment duration.
_MIN_CAPTURE_PROPORTION = 1e-6
_MAX_CAPTURE_PROPORTION = 1.0
_MIN_TAU_FACTOR = 0.0


def _get_spend_adjustment(
    *,
    channel_avg_daily_spend: float,
    experiment_avg_daily_spend: float,
    channel_name: str,
) -> float:
  """Computes spend adjustment factor tau_spend.

  Args:
    channel_avg_daily_spend: The average daily spend of the channel in the
      model.
    experiment_avg_daily_spend: The average daily spend of the experiment.
    channel_name: The name of the channel.

  Returns:
    The spend adjustment factor tau_spend.

  Raises:
    ValueError: If the average daily channel spend or average daily experiment
      spend is not positive.
  """
  if channel_avg_daily_spend <= 0.0:
    raise ValueError(
        "Average daily channel spend must be positive. "
        f"Got: channel_avg_daily_spend={channel_avg_daily_spend} "
        f"for channel {channel_name!r}."
    )
  if experiment_avg_daily_spend <= 0.0:
    raise ValueError(
        "Average daily experiment spend must be positive. "
        f"Got: experiment_avg_daily_spend={experiment_avg_daily_spend} "
        f"for channel {channel_name!r}."
    )

  max_rate = max(channel_avg_daily_spend, experiment_avg_daily_spend)
  min_rate = min(channel_avg_daily_spend, experiment_avg_daily_spend)
  return (max_rate / min_rate) - 1.0


def _get_recency_adjustment(
    *,
    experiment_end_date: datetime.date,
    last_modeled_date: datetime.date,
) -> float:
  """Computes the recency adjustment factor tau_recency.

  Args:
    experiment_end_date: The end date of the experiment.
    last_modeled_date: The last date of the modeled period.

  Returns:
    The recency adjustment factor tau_recency.
  """
  weeks_passed = (
      max((last_modeled_date - experiment_end_date).days, 0) / _DAYS_IN_WEEK
  )
  lam = _HALF_LIFE_BASE ** (weeks_passed / _HALF_LIFE_WEEKS)
  return (1.0 - lam) / lam if lam > 0.0 else 0.0


def _calculate_capture_proportion(
    *,
    duration: float,
    max_lag: int,
    adstock_decay_function: str = meridian_constants.GEOMETRIC_DECAY,
    alpha: float = calibration_constants.DEFAULT_ALPHA,
) -> float:
  """Calculates the fraction of the total adstock effect captured during the experiment's duration.

  Args:
    duration: The duration of the experiment in model intervals.
    max_lag: The maximum lag value in model intervals.
    adstock_decay_function: The type of adstock decay function. Default is
      `'geometric'`.
    alpha: The adstock decay rate parameter. Default is `0.5`.

  Returns:
    The capture proportion p.
  """
  rounded_duration = int(round(duration))
  if rounded_duration <= 0:
    return _MIN_CAPTURE_PROPORTION
  elif alpha == 0.0:
    # Zero decay rate means weights decay instantly: [1.0, 0.0, 0.0, ...].
    # So numerator (D >= 1) and denominator (L >= 1) both sum to 1.0.
    return _MAX_CAPTURE_PROPORTION
  else:
    window_size_model = max_lag + 1
    l_range_model = backend.arange(window_size_model, dtype=backend.float_dtype)
    alpha_tensor = backend.to_tensor(alpha, dtype=backend.float_dtype)
    weights_model = adstock_hill.compute_decay_weights(
        alpha=alpha_tensor,
        l_range=l_range_model,
        window_size=window_size_model,
        decay_functions=adstock_decay_function,
        normalize=False,
    )
    denominator = float(np.asarray(backend.reduce_sum(weights_model)))  # pyrefly: ignore[bad-argument-type]
    if rounded_duration <= window_size_model:
      numerator = float(
          np.asarray(backend.reduce_sum(weights_model[:rounded_duration]))
      )
    else:
      if adstock_decay_function == meridian_constants.BINOMIAL_DECAY:
        numerator = denominator
      elif adstock_decay_function == meridian_constants.GEOMETRIC_DECAY:
        l_range_tail = backend.arange(
            window_size_model, rounded_duration, dtype=backend.float_dtype
        )
        tail_sum = float(np.asarray(backend.reduce_sum(alpha**l_range_tail)))
        numerator = denominator + tail_sum
      else:
        raise ValueError(
            f"Unsupported adstock decay function: {adstock_decay_function!r}"
        )

    p = (
        numerator / denominator
        if denominator > 0.0
        else _MAX_CAPTURE_PROPORTION
    )
    p = max(p, _MIN_CAPTURE_PROPORTION)
    if adstock_decay_function == meridian_constants.BINOMIAL_DECAY:
      p = min(p, _MAX_CAPTURE_PROPORTION)
    return p


def _duration_adjustment(
    *,
    duration: float,
    max_lag: int,
    adstock_decay_function: str = meridian_constants.GEOMETRIC_DECAY,
    alpha: float = calibration_constants.DEFAULT_ALPHA,
) -> tuple[float, float]:
  """Computes the duration adjustment factors (gamma_duration, tau_duration).

  Args:
    duration: The duration of the experiment in model intervals.
    max_lag: The maximum lag value in model intervals.
    adstock_decay_function: The type of adstock decay function. Default is
      `'geometric'`.
    alpha: The adstock decay rate parameter. Default is `0.5`.

  Returns:
    A tuple (gamma_duration, tau_duration), where gamma_duration is the duration
    scaling factor and tau_duration is the duration adjustment factor.

  Raises:
    ValueError: If the adstock decay function is invalid.
  """
  if adstock_decay_function not in meridian_constants.ADSTOCK_DECAY_FUNCTIONS:
    raise ValueError(
        f"Invalid adstock_decay_function {adstock_decay_function!r}. Valid "
        f"options are {sorted(meridian_constants.ADSTOCK_DECAY_FUNCTIONS)}."
    )

  p = _calculate_capture_proportion(
      duration=duration,
      max_lag=max_lag,
      adstock_decay_function=adstock_decay_function,
      alpha=alpha,
  )

  gamma_duration = _MAX_CAPTURE_PROPORTION / p
  tau_duration = max((_MAX_CAPTURE_PROPORTION - p) / p, _MIN_TAU_FACTOR)

  return gamma_duration, tau_duration


def _get_duration_adjustment_and_scaling(
    *,
    experiment_start_date: datetime.date,
    experiment_end_date: datetime.date,
    max_lag: int,
    interval_days: int,
    adstock_decay_function: str = meridian_constants.GEOMETRIC_DECAY,
    alpha: float = calibration_constants.DEFAULT_ALPHA,
) -> tuple[float, float]:
  """Computes duration adjustment and scaling factors.

  Args:
    experiment_start_date: The start date of the experiment.
    experiment_end_date: The end date of the experiment.
    max_lag: The maximum lag value in model intervals.
    interval_days: The interval size of the Meridian model time coordinates in
      days.
    adstock_decay_function: The type of adstock decay function. Default is
      `'geometric'`.
    alpha: The adstock decay rate parameter. Default is `0.5`.

  Returns:
    A tuple (gamma_duration, tau_duration), where gamma_duration is the duration
    scaling factor and tau_duration is the duration adjustment factor.
  """
  duration = (experiment_end_date - experiment_start_date).days / interval_days
  return _duration_adjustment(
      duration=duration,
      max_lag=max_lag,
      adstock_decay_function=adstock_decay_function,
      alpha=alpha,
  )


def _get_adjusted_mean_and_std(
    cfg: base.CalibrationData,
    *,
    gamma: float,
    tau: float,
) -> tuple[float, float]:
  """Applies adjustments to compute the calibrated mean and standard deviation.

  Args:
    cfg: The `meridian.model.calibration.base.CalibrationData` config,
      containing experiment result and user-specified adjustment factors.
    gamma: The point estimate adjustment factor.
    tau: The total standard error adjustment factor.

  Returns:
    A tuple (mu_adj, std_adj), where mu_adj is the adjusted mean and std_adj is
    the adjusted standard deviation.

  Raises:
    ValueError: If the adjusted tau is less than or equal to -1.0.
  """
  adjusted_gamma = gamma + (cfg.point_estimate_adjustment or 0.0)
  adjusted_tau = tau + (cfg.standard_error_adjustment or 0.0)

  if adjusted_tau <= -1.0:
    raise ValueError(f"Tau must be greater than -1.0. Got: {adjusted_tau}")

  return (
      adjusted_gamma * cfg.experiment_result.point_estimate,
      cfg.experiment_result.standard_error * np.sqrt(1.0 + adjusted_tau),
  )


def _compute_grid_bounds(
    *,
    prior: backend.tfd.Distribution,
    likelihoods: Sequence[backend.tfd.Distribution],
) -> tuple[float, float]:
  """Computes adaptive grid bounds using a two-pass scouting step.

  Args:
    prior: The prior distribution.
    likelihoods: A sequence of likelihood distributions.

  Returns:
    A tuple (grid_min, grid_max), where grid_min is the lower bound of the
    adaptive grid and grid_max is the upper bound.
  """
  all_dists = [prior] + list(likelihoods)
  low_bounds, high_bounds = [], []
  for d in all_dists:
    try:
      m = float(d.mean().numpy())
      s = float(d.stddev().numpy())
      if np.isfinite(m) and np.isfinite(s):
        low_bounds.append(m - _SCOUT_GRID_MULTIPLIER * s)
        high_bounds.append(m + _SCOUT_GRID_MULTIPLIER * s)
    except (AttributeError, NotImplementedError, ValueError):
      continue

  broad_min = min(low_bounds) if low_bounds else _DEFAULT_SCOUT_MIN
  broad_max = max(high_bounds) if high_bounds else _DEFAULT_SCOUT_MAX

  grid_scout = backend.to_tensor(
      np.linspace(broad_min, broad_max, _SCOUT_POINTS),
      dtype=backend.to_tensor(1.0).dtype,
  )

  log_post_scout = prior.log_prob(grid_scout)
  log_post_scout = backend.where(
      backend.is_finite(log_post_scout), log_post_scout, -np.inf
  )
  for dist in likelihoods:
    log_prob_val = dist.log_prob(grid_scout)
    log_post_scout += backend.where(
        backend.is_finite(log_prob_val),
        log_prob_val,
        -np.inf,
    )

  max_log_scout = backend.reduce_max(log_post_scout)
  if not np.isfinite(float(max_log_scout)):
    raise ValueError(
        "Scouting pass resulted in zero probability mass everywhere (all"
        " log-probabilities are -inf)."
    )
  pdf_scout = backend.exp(log_post_scout - max_log_scout)
  pdf_scout_sum = float(backend.reduce_sum(pdf_scout))
  if not np.isfinite(pdf_scout_sum) or pdf_scout_sum <= 0.0:
    raise ValueError("Scouting pass resulted in zero probability mass.")

  if hasattr(pdf_scout, "numpy"):
    pdf_scout_np = pdf_scout.numpy()
  else:
    pdf_scout_np = np.asarray(pdf_scout)
  cdf_scout = np.cumsum(pdf_scout_np) / pdf_scout_sum

  if hasattr(grid_scout, "numpy"):
    grid_scout_np = grid_scout.numpy()
  else:
    grid_scout_np = np.asarray(grid_scout)

  return (
      float(grid_scout_np[np.searchsorted(cdf_scout, _LOWER_PERCENTILE)]),
      float(grid_scout_np[np.searchsorted(cdf_scout, _UPPER_PERCENTILE)]),
  )


def _evaluate_posterior(
    *,
    prior: backend.tfd.Distribution,
    likelihoods: Sequence[backend.tfd.Distribution],
    grid_min: float,
    grid_max: float,
    num_points: int = _FINE_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Evaluates and normalizes the posterior density on a grid.

  Args:
    prior: The prior distribution.
    likelihoods: A sequence of likelihood distributions.
    grid_min: The lower bound of the grid.
    grid_max: The upper bound of the grid.
    num_points: The number of points in the grid.

  Returns:
    A tuple (grid_np, pdf_np, dx), where grid_np is the parameter grid,
    pdf_np is the posterior probability density, and dx is the step size.
  """
  grid = backend.to_tensor(
      np.linspace(grid_min, grid_max, num_points),
      dtype=backend.to_tensor(1.0).dtype,
  )
  dx = (grid_max - grid_min) / (num_points - 1)

  log_posterior = prior.log_prob(grid)
  log_posterior = backend.where(
      backend.is_finite(log_posterior), log_posterior, -np.inf
  )
  for dist in likelihoods:
    log_prob_val = dist.log_prob(grid)
    log_posterior += backend.where(
        backend.is_finite(log_prob_val), log_prob_val, -np.inf
    )

  max_log = backend.reduce_max(log_posterior)
  if not np.isfinite(float(max_log)):
    raise ValueError(
        "Posterior evaluation resulted in zero probability mass everywhere (all"
        " log-probabilities are -inf)."
    )
  unnormalized_p = backend.exp(log_posterior - max_log)
  posterior_sum = float(backend.reduce_sum(unnormalized_p))
  if not np.isfinite(posterior_sum) or posterior_sum <= 0.0:
    raise ValueError(
        "Posterior evaluation resulted in zero or non-finite probability mass."
    )
  posterior_pdf = unnormalized_p / (posterior_sum * dx)

  if hasattr(grid, "numpy"):
    grid_np = grid.numpy()
  else:
    grid_np = np.asarray(grid)

  if hasattr(posterior_pdf, "numpy"):
    pdf_np = posterior_pdf.numpy()
  else:
    pdf_np = np.asarray(posterior_pdf)
  return grid_np, pdf_np, dx


class ImproperUniformPrior(backend.tfd.Distribution):
  """An improper flat uniform prior over (0, infinity)."""

  def __init__(
      self,
      dtype=None,
      validate_args=False,
      allow_nan_stats=True,
      name="improper_uniform_prior",
  ):
    if dtype is None:
      dtype = backend.float_dtype
    parameters = dict(locals())
    super().__init__(
        dtype=dtype,
        reparameterization_type=backend.tfd.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )
    self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return {}

  def _batch_shape_tensor(self):
    return backend.to_tensor(np.array([], dtype=np.int32))

  def _batch_shape(self):
    return ()

  def _event_shape_tensor(self):
    return backend.to_tensor(np.array([], dtype=np.int32))

  def _event_shape(self):
    return ()

  def _sample_n(self, n: int, seed=None) -> backend.Tensor:
    raise NotImplementedError(
        "ImproperUniformPrior has infinite mass and cannot be sampled."
    )

  def _log_prob(self, value):
    return backend.where(
        value > 0.0,
        backend.cast(0.0, self.dtype),
        backend.cast(-np.inf, self.dtype),
    )

  def _prob(self, value):
    return backend.where(
        value > 0.0,
        backend.cast(1.0, self.dtype),
        backend.cast(0.0, self.dtype),
    )


class GridDistribution(backend.tfd.Distribution):
  """A continuous distribution defined by a probability density function on a grid."""

  def __init__(
      self,
      grid: np.ndarray,
      pdf: np.ndarray,
      dx: float,
      dtype=backend.float_dtype,
      validate_args=False,
      allow_nan_stats=True,
      name="grid_distribution",
  ):
    if dtype is None:
      dtype = backend.float_dtype
    parameters = dict(locals())
    self._grid = np.array(grid, dtype=np.float32)
    self._pdf = np.array(pdf, dtype=np.float32)
    self._dx = float(dx)
    total_mass = np.sum(self._pdf) * self._dx
    if len(self._grid) > 0:
      if not (np.isfinite(total_mass) and total_mass > 0.0):
        raise ValueError(
            "Invalid PDF: total mass must be finite and positive. Got"
            f" {total_mass}"
        )
      self._pdf /= total_mass

    cdf = np.cumsum(self._pdf) * self._dx
    if len(cdf) > 0:
      cdf[-1] = 1.0

    self._grid_tensor = backend.to_tensor(self._grid, dtype=dtype)
    self._cdf_tensor = backend.to_tensor(cdf, dtype=dtype)
    self._pdf_tensor = backend.to_tensor(self._pdf, dtype=dtype)
    self._grid_min = float(self._grid[0]) if len(self._grid) > 0 else 0.0
    self._grid_max = float(self._grid[-1]) if len(self._grid) > 0 else 0.0
    self._inv_dx = 1.0 / self._dx if self._dx > 0.0 else 0.0

    super().__init__(
        dtype=dtype,
        reparameterization_type=backend.tfd.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )
    self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        grid=backend.util.ParameterProperties(),
        pdf=backend.util.ParameterProperties(),
        dx=backend.util.ParameterProperties(),
    )

  def _batch_shape_tensor(self):
    return backend.to_tensor([], dtype=backend.int32)

  def _batch_shape(self):
    return backend.TensorShape([])

  def _event_shape_tensor(self):
    return backend.to_tensor([], dtype=backend.int32)

  def _event_shape(self):
    return backend.TensorShape([])

  def _sample_n(self, n: int, seed=None) -> backend.Tensor:
    seed_state = backend.random.sanitize_seed(seed)
    grid_size = len(self._grid)
    if grid_size == 0:
      return backend.to_tensor([], dtype=self.dtype)

    u = backend.random.stateless_uniform(seed_state, (n,), dtype=self.dtype)
    idx = backend.searchsorted(self._cdf_tensor, u)  # pyrefly: ignore[bad-argument-type]
    idx = backend.minimum(idx, grid_size - 1)
    return backend.gather(self._grid_tensor, idx)

  def _log_prob(self, value):
    raise NotImplementedError(
        "Log probability evaluation is not supported for GridDistribution."
    )


def _normal_loss(
    p: Sequence[float],
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
) -> float:
  """Loss function for fitting Normal distribution."""
  if p[1] <= 0.0:
    return np.inf
  log_prob = np.asarray(
      backend.tfd.Normal(loc=float(p[0]), scale=float(p[1])).log_prob(grid_np)
  )
  return (
      -np.sum(
          pdf_np
          * np.where(np.isfinite(log_prob), log_prob, _ZERO_PROBABILITY_MASK)
      )
      * dx
  )


def _lognormal_loss(
    p: Sequence[float],
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
) -> float:
  """Loss function for fitting LogNormal distribution."""
  if p[1] <= 0.0:
    return np.inf
  mask = grid_np > 0.0
  log_prob = np.full_like(grid_np, _ZERO_PROBABILITY_MASK)
  if np.any(mask):
    log_prob[mask] = np.asarray(
        backend.tfd.LogNormal(loc=float(p[0]), scale=float(p[1])).log_prob(
            grid_np[mask]
        )
    )
  return -np.sum(pdf_np * log_prob) * dx


def _gamma_loss(
    p: Sequence[float],
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
) -> float:
  """Loss function for fitting Gamma distribution."""
  if p[0] <= 0.0 or p[1] <= 0.0:
    return np.inf
  mask = grid_np > 0.0
  log_prob = np.full_like(grid_np, _ZERO_PROBABILITY_MASK)
  if np.any(mask):
    log_prob[mask] = np.asarray(
        backend.tfd.Gamma(concentration=float(p[0]), rate=float(p[1])).log_prob(
            grid_np[mask]
        )
    )
  return -np.sum(pdf_np * log_prob) * dx


def _fit_normal(
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
    mean_emp: float,
    std_emp: float,
) -> tuple[backend.tfd.Distribution, float]:
  """Fits Normal distribution and returns the fitted distribution and loss."""
  res = opt.minimize(
      lambda p: _normal_loss(p, grid_np=grid_np, pdf_np=pdf_np, dx=dx),
      x0=[mean_emp, std_emp],
      bounds=[(None, None), (_OPTIMIZER_MIN_BOUND, None)],
      method="L-BFGS-B",
  )
  dist = backend.tfd.Normal(
      loc=backend.cast(float(res.x[0]), backend.float_dtype),
      scale=backend.cast(float(res.x[1]), backend.float_dtype),
  )
  return dist, float(res.fun)


def _fit_lognormal(
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
    mean_emp: float,
    var_emp: float,
) -> tuple[backend.tfd.Distribution, float]:
  """Fits LogNormal distribution and returns fitted distribution and loss."""
  mean_emp_safe = max(mean_emp, _MIN_POSITIVE_MEAN_SAFE)
  var_emp_safe = max(var_emp, _MIN_POSITIVE_VAR_SAFE)
  log_var_guess = np.log(1.0 + (var_emp_safe / (mean_emp_safe**2)))
  log_mean_guess = np.log(mean_emp_safe) - 0.5 * log_var_guess
  log_std_guess = np.sqrt(log_var_guess)

  res = opt.minimize(
      lambda p: _lognormal_loss(p, grid_np=grid_np, pdf_np=pdf_np, dx=dx),
      x0=[log_mean_guess, log_std_guess],
      bounds=[(None, None), (_OPTIMIZER_MIN_BOUND, None)],
      method="L-BFGS-B",
  )
  dist = backend.tfd.LogNormal(
      loc=backend.cast(float(res.x[0]), backend.float_dtype),
      scale=backend.cast(float(res.x[1]), backend.float_dtype),
  )
  return dist, float(res.fun)


def _fit_gamma(
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
    mean_emp: float,
    var_emp: float,
) -> tuple[backend.tfd.Distribution, float]:
  """Fits Gamma distribution and returns the fitted distribution and loss."""
  mean_emp_safe = max(mean_emp, _MIN_POSITIVE_MEAN_SAFE)
  var_emp_safe = max(var_emp, _MIN_POSITIVE_VAR_SAFE)
  alpha_guess = (mean_emp_safe**2) / var_emp_safe
  beta_guess = mean_emp_safe / var_emp_safe

  res = opt.minimize(
      lambda p: _gamma_loss(p, grid_np=grid_np, pdf_np=pdf_np, dx=dx),
      x0=[alpha_guess, beta_guess],
      bounds=[(_OPTIMIZER_MIN_BOUND, None), (_OPTIMIZER_MIN_BOUND, None)],
      method="L-BFGS-B",
  )
  dist = backend.tfd.Gamma(
      concentration=backend.cast(float(res.x[0]), backend.float_dtype),
      rate=backend.cast(float(res.x[1]), backend.float_dtype),
  )
  return dist, float(res.fun)


def _has_negative_support(prior: backend.tfd.Distribution | None) -> bool:
  """Checks whether the baseline prior permits negative support.

  Args:
    prior: The baseline prior distribution, or None.

  Returns:
    True if the prior assigns a finite density/probability in the negative
    region (specifically at -1e-5), False otherwise.
  """
  if prior is None:
    return False
  try:
    dtype = prior.dtype
    neg_val = backend.to_tensor(-1e-5, dtype=dtype)
    log_prob_neg = prior.log_prob(neg_val)
    is_finite = backend.is_finite(log_prob_neg)
    if hasattr(is_finite, "numpy"):
      return bool(np.all(is_finite.numpy()))
    else:
      return bool(np.all(np.asarray(is_finite)))
  except (
      AttributeError,
      NotImplementedError,
      ValueError,
      TypeError,
      RuntimeError,
  ):
    return False


def _fit_distribution(
    *,
    grid_np: np.ndarray,
    pdf_np: np.ndarray,
    dx: float,
    has_negative_support: bool = False,
    channel_name: str,
) -> backend.tfd.Distribution:
  """Fits Normal, LogNormal, and Gamma, returning the best fit.

  Args:
    grid_np: The parameter grid.
    pdf_np: The probability density function on the grid.
    dx: The step size of the grid.
    has_negative_support: Whether the prior permits negative support.
    channel_name: The name of the channel.

  Returns:
    A fitted `backend.tfd.Distribution` corresponding to the best-fitting
    shape.
  """
  mean_emp = np.sum(grid_np * pdf_np) * dx
  var_emp = np.sum(((grid_np - mean_emp) ** 2) * pdf_np) * dx
  std_emp = np.sqrt(var_emp)

  if has_negative_support:
    candidates = [
        (
            "Normal",
            lambda: _fit_normal(
                grid_np=grid_np,
                pdf_np=pdf_np,
                dx=dx,
                mean_emp=mean_emp,
                std_emp=std_emp,
            ),
        ),
    ]
  else:
    candidates = []

  candidates.extend([
      (
          "LogNormal",
          lambda: _fit_lognormal(
              grid_np=grid_np,
              pdf_np=pdf_np,
              dx=dx,
              mean_emp=mean_emp,
              var_emp=var_emp,
          ),
      ),
      (
          "Gamma",
          lambda: _fit_gamma(
              grid_np=grid_np,
              pdf_np=pdf_np,
              dx=dx,
              mean_emp=mean_emp,
              var_emp=var_emp,
          ),
      ),
  ])

  best_dist = None
  best_loss = np.inf

  for _, fit_func in candidates:
    try:
      dist, loss = fit_func()
      if np.isfinite(loss) and loss < best_loss:
        best_loss = loss
        best_dist = dist
    except ValueError:
      continue

  if best_dist is None:
    raise ValueError(
        "Failed to fit any candidate distribution shape for channel"
        f" {channel_name!r}."
    )

  return best_dist


def _merge_distributions(
    *,
    means: Sequence[float],
    stds: Sequence[float],
    baseline_prior: backend.tfd.Distribution | None,
    channel_name: str,
) -> tuple[backend.tfd.Distribution, backend.tfd.Distribution]:
  """Merges multiple distributions and baseline prior using a numerical grid update.

  Args:
    means: A sequence of adjusted means.
    stds: A sequence of adjusted standard deviations.
    baseline_prior: An optional baseline prior distribution.
    channel_name: The name of the channel.

  Returns:
    A tuple of (fitted_distribution, intermediary_prior), where:
      - fitted_distribution: The parameterized `backend.tfd.Distribution`
        corresponding to the best-fitting shape.
      - intermediary_prior: The unparameterized `GridDistribution` prior.
  """
  if np.any(np.array(stds) <= 0.0):
    raise ValueError("Standard errors must be positive for Bayesian merging.")

  prior = (
      baseline_prior if baseline_prior is not None else ImproperUniformPrior()
  )
  likelihoods = [
      backend.tfd.Normal(loc=float(m), scale=float(s))
      for m, s in zip(means, stds)
  ]

  grid_min, grid_max = _compute_grid_bounds(
      prior=prior, likelihoods=likelihoods
  )
  grid_np, pdf_np, dx = _evaluate_posterior(
      prior=prior, likelihoods=likelihoods, grid_min=grid_min, grid_max=grid_max
  )

  has_negative_support = _has_negative_support(baseline_prior)

  fitted_prior = _fit_distribution(
      grid_np=grid_np,
      pdf_np=pdf_np,
      dx=dx,
      has_negative_support=has_negative_support,
      channel_name=channel_name,
  )
  intermediary_prior = GridDistribution(
      grid=grid_np,
      pdf=pdf_np,
      dx=dx,
  )

  return fitted_prior, intermediary_prior


def get_calibrated_roi_prior(
    calibration_data: Sequence[base.CalibrationData],
    *,
    channel_name: str,
    total_channel_spend: float,
    last_modeled_date: datetime.date,
    adstock_decay_function: str,
    alpha: float,
    max_lag: int,
    interval_days: int,
    model_duration_days: int,
    baseline_prior: backend.tfd.Distribution | None = None,
) -> tuple[backend.tfd.Distribution, base.CalibrationOutput]:
  """Calculates the calibrated ROI prior from one or more experiment results.

  This function aggregates incrementality experiment results from one or more
  experiments, applies spend, recency, and duration adjustments, and merges
  them into a single joint prior distribution. If a baseline prior is provided,
  its mean and variance are used to regularize the experiment results.

  Args:
    calibration_data: A sequence of
      `meridian.model.calibration.base.CalibrationData` objects, each containing
      an experiment result and associated experiment information.
    channel_name: The name of the channel to calibrate.
    total_channel_spend: The total spend of the channel in the Meridian model.
    last_modeled_date: The last date of the modeled period in the Meridian
      model.
    adstock_decay_function: The adstock decay function used for duration
      adjustments (`'geometric'` or `'binomial'`).
    alpha: The decay rate parameter used for duration adjustments.
    max_lag: The maximum lag value in model intervals.
    interval_days: The interval size of the Meridian model time coordinates in
      days.
    model_duration_days: The number of days in the modeled period of the
      Meridian model.
    baseline_prior: An optional `tfd.Distribution` representing the baseline
      prior for the KPI. If provided, the baseline prior will regularize the
      incrementality experiment results.

  Returns:
    A tuple of (calibrated_prior, calibration_output), where:
      - calibrated_prior: A `tfd.Distribution` representing the calibrated ROI
        prior.
      - calibration_output: A `base.CalibrationOutput` object containing output
        diagnostics, baseline prior, and calibrated experiments.

  Raises:
    ValueError: If any of the following are true:
      - No calibration data is provided for the channel.
      - Total channel spend is less than or equal to 0.0.
      - Average daily channel spend or average daily experiment spend is not
        positive.
      - The adstock decay function is invalid.
      - The adjusted standard error factor (tau) is less than or equal to
        -1.0.
      - Posterior evaluation results in zero or non-finite probability mass.
  """
  if not calibration_data:
    raise ValueError(
        f"No calibration data provided for channel {channel_name!r}."
    )

  if model_duration_days <= 0:
    raise ValueError(
        f"Model duration in days must be positive. Got: {model_duration_days}."
    )

  means_and_stds = []
  calibrated_experiments = []
  for cfg in calibration_data:
    tau_spend = _get_spend_adjustment(
        channel_avg_daily_spend=total_channel_spend / model_duration_days,
        experiment_avg_daily_spend=cfg.experiment_info.avg_daily_spend,
        channel_name=channel_name,
    )
    tau_recency = _get_recency_adjustment(
        experiment_end_date=cfg.experiment_info.experiment_end_date,
        last_modeled_date=last_modeled_date,
    )
    gamma_duration, tau_duration = _get_duration_adjustment_and_scaling(
        experiment_start_date=cfg.experiment_info.experiment_start_date,
        experiment_end_date=cfg.experiment_info.experiment_end_date,
        max_lag=max_lag,
        interval_days=interval_days,
        adstock_decay_function=adstock_decay_function,
        alpha=alpha,
    )
    m_adj, s_adj = _get_adjusted_mean_and_std(
        cfg,
        gamma=gamma_duration,
        tau=tau_spend + tau_recency + tau_duration,
    )
    means_and_stds.append((m_adj, s_adj))

    calibrated_experiments.append(
        base.CalibratedExperiment(
            source_type=cfg.source_type,
            raw_experiment_result=cfg.experiment_result,
            adjusted_experiment_result=base.ExperimentResult(
                point_estimate=m_adj,
                standard_error=s_adj,
            ),
            tau_spend=tau_spend,
            tau_recency=tau_recency,
            tau_duration=tau_duration,
            gamma_duration=gamma_duration,
            user_point_estimate_adjustment=cfg.point_estimate_adjustment,
            user_standard_error_adjustment=cfg.standard_error_adjustment,
        )
    )

  means, stds = zip(*means_and_stds)

  calibrated_prior, intermediary_prior = _merge_distributions(
      means=means,
      stds=stds,
      baseline_prior=baseline_prior,
      channel_name=channel_name,
  )

  # Contains calibration diagnostics information.
  calibration_output = base.CalibrationOutput(
      channel_name=channel_name,
      experiments=calibrated_experiments,
      baseline_prior=baseline_prior,
      intermediary_prior=intermediary_prior,
      adstock_decay_spec=adstock_decay_function,
      max_lag=max_lag,
  )

  return calibrated_prior, calibration_output
