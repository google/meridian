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

"""Adjustment for Density Maximization (ADM) for Meridian Full MAP estimation.

Plain Maximum a Posteriori (MAP) estimation returns the *mode* of the
posterior in constrained parameter space. For skewed posteriors -- the norm for
the positively supported scale and ROI parameters of a Marketing Mix Model --
the mode is a biased estimate of the posterior *mean*, which is the summary
Meridian reports from MCMC. Worse, the mode of a standard deviation parameter
with a HalfNormal prior sits on the zero boundary (complete shrinkage), which
MCMC never reproduces.

Adjustment for Density Maximization (ADM), due to Morris (1988), fixes this by
maximizing an *adjusted* density

    p_adj(theta) = a(theta) * p(theta | y)

where the adjustment factor `a` is chosen so that

    argmax_theta [a(theta) p(theta | y)] ~= E[theta | y].

References:

*   Morris, C. N. (1988). "Approximating posterior distributions and posterior
    moments." Bayesian Statistics 3, Oxford University Press, 327-344.
*   Morris, C. N. and Tang, R. (2011). "Estimating Random Effects via
    Adjustment for Density Maximization." Statistical Science 26(2), 271-287.
*   Tak, H., Kelly, J. and Morris, C. N. (2017). "Rgbp: An R Package for
    Gaussian, Poisson, and Binomial Random Effects Models." JSS 78(5), 1-33.

## Approximating families and adjustment factors

Writing `l` and `u` for the lower and upper boundary of the support:

*   `Normal(mu, s^2)` on the real line: `a(theta) = 1`.
*   `HalfNormal(s)` on `(l, inf)`: `a(theta) = (theta - l) ** (2 / pi)`.
*   `Gamma(k, b)` on `(l, inf)`: `a(theta) = theta - l`.
*   `LogNormal(mu, s^2)` on `(l, inf)`: `a(theta) = (theta - l) ** 1.5`.
*   `InverseGamma(k, b)` on `(l, inf)`: `a(theta) = (theta - l) ** 2`.
*   `Beta(a, b)` on `(l, u)`: `a(theta) = (theta - l) * (u - theta)`.

Parameters bounded above only use `u - theta` in place of `theta - l`.

In Meridian's full joint model:
*   Unbounded location and latent deviation parameters (`knot_values`,
    `tau_g_excl_baseline`, `gamma_c`, `gamma_n`, `beta_g*_dev`, `gamma_g*_dev`)
    use `NORMAL` (`log a = 0`).
*   Standard deviation parameters (`sigma`, `eta_m`, `eta_rf`, `eta_om`,
    `eta_orf`, `xi_c`, `xi_n`) and half-line parameters carrying `HalfNormal`
    priors (such as `beta_m`, `beta_rf`, `beta_om`, `beta_orf` when
    `prior_type="coefficient"`) use `HALF_NORMAL` (`c = 2 / pi ~= 0.6366`),
    which is exact in the prior-dominated limit and matches the chi- or
    truncated-Gaussian data-dominated limit.
*   Half-saturation parameters (`ec_m`, `ec_om`) under their default
    `TruncatedNormal(0.8, 0.8, 0.1, 10.0)` prior are one-sided truncated normals
    above `low = 0.1` (the upper bound `10.0` is `11.5` SDs above the mode and
    carries negligible density), so they also use `HALF_NORMAL` (`c = 2 / pi`)
    measured from `low = 0.1`.
*   Bounded unit-interval parameters (`alpha_m`, `alpha_rf`, `alpha_om`,
    `alpha_orf`, `contribution_*`) use `BETA` (`log(theta - low) +
    log(high - theta)`).
*   Positively supported right-skewed treatment effect and media parameters
    (`roi_m`, `roi_rf`, `mroi_m`, `mroi_rf`, `ec_rf`, `ec_orf`, `slope_*`) use
    `GAMMA` (`c = 1.0`) by default in the full joint model, where the marginal
    posterior blends a right-skewed `LogNormal` prior with a Gaussian likelihood
    and joint `(alpha, ec)` coupling; callers can override any parameter (for
    example `{"roi_m": "lognormal"}` for `c = 1.5`).

## Hessian-based intervals and draws

The same approximating families provide uncertainty estimates. Let `H` be the
Hessian of the negative adjusted log posterior at the ADM optimum `m`. The
marginal curvature of parameter `k` is `kappa_k = 1 / (H^-1)_kk`, and
`fit_marginal` selects the member of the parameter's family whose adjusted
density has mode `m_k` and curvature `kappa_k` -- so the fitted member has mean
`m_k` and recovers the exact posterior whenever the posterior lies in the
family. `ApproximatePosterior` couples the marginals through a Gaussian copula
with the correlation matrix implied by `H^-1`, which yields joint draws that
preserve both the approximating marginals and the local posterior correlation
structure.
"""

from collections.abc import Mapping
import dataclasses
import enum
import math
from typing import Any
import warnings

from meridian import backend
from meridian import constants
import numpy as np
from scipy import special
from scipy import stats

__all__ = [
    "AdjustmentFamily",
    "ApproximatePosterior",
    "ApproximatingMarginal",
    "HessianDiagnostics",
    "ParameterAdjustment",
    "SUMMARY_METRICS",
    "Support",
    "covariance_from_hessian",
    "default_family",
    "default_family_for_parameter",
    "describe_adjustments",
    "fit_approximate_posterior",
    "fit_marginal",
    "infer_support",
    "log_adjustment",
    "resolve_adjustments",
    "unwrap_distribution",
    "validate_family",
]

# Clamps arguments of `log` away from the boundary of the support.
_EPSILON: float = 1e-12
# Unconstrained probe points used to infer the support of a bijector.
_PROBE_NEAR: float = 20.0
_PROBE_FAR: float = 40.0
# A boundary is treated as unbounded when moving the probe from `_PROBE_NEAR`
# to `_PROBE_FAR` moves the constrained value by more than this much.
_UNBOUNDED_GAP: float = 1.0
# Boundaries closer to zero than this are snapped to exactly zero.
_ZERO_SNAP: float = 1e-10


class AdjustmentFamily(enum.Enum):
  """Approximating family used to derive an ADM adjustment factor.

  Attributes:
    NONE: No adjustment; recovers plain MAP estimation for this parameter.
    NORMAL: Symmetric family on the real line. The mode already equals the mean,
      so the adjustment factor is one.
    HALF_NORMAL: Positively supported family for *standard deviation* (or
      one-sided lower-truncated normal) parameters, with adjustment factor
      `(theta - low) ** (2 / pi)`.
    GAMMA: Positively supported family with adjustment factor `theta - low`.
    LOGNORMAL: Positively supported family with adjustment factor `(theta - low)
      ** 1.5`.
    INVERSE_GAMMA: Positively supported family with adjustment factor `(theta -
      low) ** 2`.
    BETA: Family on a bounded interval with adjustment factor `(theta - low) *
      (high - theta)`.
  """

  NONE = "none"
  NORMAL = "normal"
  HALF_NORMAL = "half_normal"
  GAMMA = "gamma"
  LOGNORMAL = "lognormal"
  INVERSE_GAMMA = "inverse_gamma"
  BETA = "beta"


_POWER_EXPONENT: Mapping[AdjustmentFamily, float] = {
    AdjustmentFamily.HALF_NORMAL: 2.0 / math.pi,
    AdjustmentFamily.GAMMA: 1.0,
    AdjustmentFamily.LOGNORMAL: 1.5,
    AdjustmentFamily.INVERSE_GAMMA: 2.0,
}

_TRIVIAL_FAMILIES = frozenset({AdjustmentFamily.NONE, AdjustmentFamily.NORMAL})

# Meridian parameters that are standard deviations carrying HalfNormal priors.
_SCALE_PARAMETERS = frozenset({
    constants.SIGMA,
    constants.ETA_M,
    constants.ETA_RF,
    constants.ETA_OM,
    constants.ETA_ORF,
    constants.XI_C,
    constants.XI_N,
})

# Half-saturation parameters that default to one-sided TruncatedNormal priors.
_HALF_SATURATION_PARAMETERS = frozenset({
    constants.EC_M,
    constants.EC_OM,
})


def unwrap_distribution(
    dist: backend.tfd.Distribution,
) -> backend.tfd.Distribution:
  """Unwraps Sample, Independent, and BatchBroadcast distribution wrappers."""
  current = dist
  while not isinstance(
      current, backend.tfd.TransformedDistribution
  ) and hasattr(current, "distribution"):
    current = current.distribution
  return current


@dataclasses.dataclass(frozen=True)
class Support:
  """Support of a constrained model parameter.

  Boundaries may be scalars or arrays; array boundaries describe a batch of
  parameters with elementwise bounds. They are normalized to numpy arrays on
  construction.

  Attributes:
    low: Lower boundary, or `None` when the parameter is unbounded below.
    high: Upper boundary, or `None` when the parameter is unbounded above.
  """

  low: np.ndarray | float | None = None
  high: np.ndarray | float | None = None

  def __post_init__(self):
    for field in ("low", "high"):
      value = getattr(self, field)
      if value is not None:
        object.__setattr__(self, field, np.asarray(value, dtype=np.float64))

  @property
  def is_real(self) -> bool:
    """Whether the support is the whole real line."""
    return self.low is None and self.high is None

  @property
  def is_bounded(self) -> bool:
    """Whether the support is a bounded interval."""
    return self.low is not None and self.high is not None

  @property
  def is_half_line(self) -> bool:
    """Whether the support is bounded on exactly one side."""
    return (self.low is None) != (self.high is None)

  def describe(self) -> str:
    """Returns a short human readable description of the support."""

    def _fmt(value: np.ndarray | float | None, default: str) -> str:
      if value is None:
        return default
      flat = np.reshape(np.asarray(value, dtype=np.float64), [-1])
      if flat.size == 0:
        return default
      if np.allclose(flat, flat[0]):
        return f"{float(flat[0]):.4g}"
      return f"[{float(np.min(flat)):.4g}, {float(np.max(flat)):.4g}]"

    return f"({_fmt(self.low, '-inf')}, {_fmt(self.high, 'inf')})"


@dataclasses.dataclass(frozen=True)
class ParameterAdjustment:
  """Resolved ADM adjustment for a single optimized parameter.

  Attributes:
    name: Name of the Meridian parameter, for example `roi_m`.
    family: Approximating family used to derive the adjustment factor.
    support: Inferred support of the parameter.
  """

  name: str
  family: AdjustmentFamily
  support: Support

  @property
  def is_trivial(self) -> bool:
    """Whether the adjustment factor is identically one (`log a = 0`)."""
    return self.family in _TRIVIAL_FAMILIES

  def describe(self) -> str:
    """Returns a one line summary of this adjustment."""
    return (
        f"{self.name}: support={self.support.describe()} "
        f"family={self.family.value} log_a={self.log_factor_formula()}"
    )

  def log_factor_formula(self) -> str:
    """Returns the symbolic form of `log a(theta)` for this parameter."""
    if self.is_trivial:
      return "0"
    if self.family == AdjustmentFamily.BETA:
      return "log(theta - low) + log(high - theta)"
    exponent = _POWER_EXPONENT[self.family]
    boundary = "theta - low" if self.support.low is not None else "high - theta"
    return f"{exponent:g} * log({boundary})"


def infer_support(bijector: Any) -> Support:
  """Infers the support of a parameter from its constraining bijector.

  The support is probed numerically by pushing large positive and large
  negative unconstrained values through the bijector. A boundary is declared
  unbounded when the constrained value keeps moving as the probe moves further
  out, and bounded (at the limiting value) otherwise.

  Args:
    bijector: A TFP bijector mapping the unconstrained space to the support of
      the parameter, typically obtained from
      `distribution.experimental_default_event_space_bijector()`.

  Returns:
    The inferred `Support`.
  """

  def _forward(value: float) -> np.ndarray:
    tensor = backend.to_tensor(value, dtype=backend.float_dtype)
    return np.asarray(bijector.forward(tensor), dtype=np.float64)

  low_near = _forward(-_PROBE_NEAR)
  low_far = _forward(-_PROBE_FAR)
  high_near = _forward(_PROBE_NEAR)
  high_far = _forward(_PROBE_FAR)

  lower_unbounded = bool(np.all(low_near - low_far > _UNBOUNDED_GAP))
  upper_unbounded = bool(np.all(high_far - high_near > _UNBOUNDED_GAP))

  def _snap(value: np.ndarray) -> np.ndarray:
    return np.where(np.abs(value) < _ZERO_SNAP, 0.0, value)

  return Support(
      low=None if lower_unbounded else _snap(low_far),
      high=None if upper_unbounded else _snap(high_far),
  )


def _numpy_dtype_of(value: Any) -> Any:
  """Returns the numpy dtype of a backend tensor, numpy array or scalar."""
  dtype = getattr(value, "dtype", None)
  if dtype is None:
    return np.float32
  return getattr(dtype, "as_numpy_dtype", dtype)


def default_family(support: Support) -> AdjustmentFamily:
  """Returns the default approximating family for a given support.

  Args:
    support: The support of the parameter.

  Returns:
    `NORMAL` for the real line, `BETA` for a bounded interval and `GAMMA` for a
    half line.
  """
  if support.is_real:
    return AdjustmentFamily.NORMAL
  if support.is_bounded:
    return AdjustmentFamily.BETA
  return AdjustmentFamily.GAMMA


def validate_family(family: AdjustmentFamily, support: Support) -> None:
  """Raises `ValueError` if `family` is incompatible with `support`.

  Args:
    family: Approximating family to check.
    support: Support of the parameter.

  Raises:
    ValueError: If the family cannot be used with this support.
  """
  if family in _TRIVIAL_FAMILIES:
    return
  if family == AdjustmentFamily.BETA:
    if not support.is_bounded:
      raise ValueError(
          "The 'beta' adjustment family requires a bounded support, got"
          f" {support.describe()}."
      )
    return
  if support.is_real:
    raise ValueError(
        f"The '{family.value}' adjustment family requires a bounded boundary,"
        f" got {support.describe()}."
    )


def log_adjustment(
    value: Any, family: AdjustmentFamily, support: Support
) -> Any:
  """Computes the elementwise log adjustment factor `log a(theta)`.

  Args:
    value: Constrained parameter value `theta`.
    family: Approximating family used to derive the factor.
    support: Support of the parameter.

  Returns:
    A tensor of the same shape as `value` holding `log a(theta)`.

  Raises:
    ValueError: If the family is incompatible with the support.
  """
  validate_family(family, support)
  value_tensor = backend.to_tensor(value, dtype=backend.float_dtype)
  if family in _TRIVIAL_FAMILIES:
    return backend.zeros_like(value_tensor)

  np_dtype = _numpy_dtype_of(value_tensor)

  def _bound(array: np.ndarray | float) -> backend.Tensor:
    return backend.to_tensor(
        np.asarray(array, dtype=np_dtype), dtype=backend.float_dtype
    )

  eps: Any = backend.to_tensor(_EPSILON, dtype=backend.float_dtype)
  low, high = support.low, support.high

  if family == AdjustmentFamily.BETA:
    if low is None or high is None:
      raise ValueError(
          "The 'beta' adjustment family requires a bounded support."
      )
    return backend.log(
        backend.maximum(value_tensor - _bound(low), eps)
    ) + backend.log(backend.maximum(_bound(high) - value_tensor, eps))

  exponent = backend.to_tensor(
      _POWER_EXPONENT[family], dtype=backend.float_dtype
  )
  if low is not None:
    distance = value_tensor - _bound(low)
  elif high is not None:
    distance = _bound(high) - value_tensor
  else:
    raise ValueError(
        f"The '{family.value}' adjustment family requires a bounded boundary."
    )
  return exponent * backend.log(backend.maximum(distance, eps))


def default_family_for_parameter(
    name: str,
    support: Support,
    dist: backend.tfd.Distribution | None = None,
) -> AdjustmentFamily:
  """Returns the default ADM family for a named Meridian parameter.

  Args:
    name: Parameter name in the Meridian joint distribution.
    support: Inferred support of the parameter.
    dist: Optional prior distribution of the parameter.

  Returns:
    The default `AdjustmentFamily` for `name`.
  """
  base_dist = None if dist is None else unwrap_distribution(dist)
  if support.is_half_line and (
      name in _SCALE_PARAMETERS or isinstance(base_dist, backend.tfd.HalfNormal)
  ):
    # Standard deviation parameters and any half-line parameter carrying a
    # HalfNormal prior (such as `beta_*` when `prior_type="coefficient"`) take
    # the exact HalfNormal exponent `2 / pi` rather than `1.0` (`GAMMA`).
    return AdjustmentFamily.HALF_NORMAL

  if (
      name in _HALF_SATURATION_PARAMETERS
      and support.low is not None
      and isinstance(base_dist, backend.tfd.TruncatedNormal)
  ):
    # Meridian's default `ec_m` / `ec_om` prior `TruncatedNormal(0.8, 0.8,
    # 0.1, 10.0)` is truncated at `low = 0.1` (`0.875` SDs below the mode) while
    # `high = 10.0` is `11.5` SDs above the mode. Treating it as a one-sided
    # lower-truncated normal (`HALF_NORMAL` above `low`) avoids an artificial
    # upward pull toward the interval midpoint `(low + high) / 2 = 5.05`.
    return AdjustmentFamily.HALF_NORMAL

  return default_family(support)


def resolve_adjustments(
    param_specs: Mapping[str, tuple[Any, backend.tfd.Distribution | None]],
    family_overrides: Mapping[str, AdjustmentFamily | str] | None = None,
) -> dict[str, ParameterAdjustment]:
  """Resolves the ADM adjustment for every optimized parameter.

  Args:
    param_specs: Mapping from parameter name to `(bijector, prior_dist)` for
      each free parameter in the optimization problem.
    family_overrides: Optional mapping from parameter name to an
      `AdjustmentFamily` (or its string value) overriding the default choice.

  Returns:
    A mapping from parameter name to its resolved `ParameterAdjustment`.

  Raises:
    ValueError: If an override names a parameter that is not optimized, or if
      an override is incompatible with the inferred support.
  """
  overrides: dict[str, AdjustmentFamily] = {}
  for name, family in (family_overrides or {}).items():
    overrides[name] = (
        family
        if isinstance(family, AdjustmentFamily)
        else AdjustmentFamily(family)
    )

  unknown = sorted(set(overrides) - set(param_specs))
  if unknown:
    raise ValueError(
        f"Unknown ADM family override(s) {unknown}. Optimized parameters are"
        f" {sorted(param_specs)}."
    )

  adjustments: dict[str, ParameterAdjustment] = {}
  for name, (bijector, dist) in param_specs.items():
    support = infer_support(bijector)
    family = overrides.get(
        name,
        default_family_for_parameter(name, support, dist),
    )
    validate_family(family, support)
    adjustments[name] = ParameterAdjustment(
        name=name, family=family, support=support
    )
  return adjustments


def describe_adjustments(
    adjustments: Mapping[str, ParameterAdjustment],
) -> str:
  """Returns a multi-line summary of resolved parameter adjustments."""
  return "\n".join(adjustment.describe() for adjustment in adjustments.values())


# ------------------------------------------------------------------------------
# Hessian-based approximate posterior.
# ------------------------------------------------------------------------------

# Eigenvalues of the Hessian below this fraction of the largest eigenvalue are
# raised to the floor before inversion.
_EIGENVALUE_FLOOR_RATIO: float = 1e-8
# Standard normal CDF values are clipped to this range before being mapped
# through marginal quantile functions.
_UNIFORM_CLIP: float = 1e-12
# Smallest distance from the anchoring boundary used when fitting a family.
_MIN_DISTANCE: float = 1e-8
# Metrics reported by `ApproximatingMarginal.summary`, in order.
SUMMARY_METRICS: tuple[str, ...] = (
    constants.MEAN,
    constants.SD,
    constants.MEDIAN,
    constants.CI_LO,
    constants.CI_HI,
)


@dataclasses.dataclass(frozen=True)
class HessianDiagnostics:
  """Diagnostics of the Hessian underlying an `ApproximatePosterior`.

  Attributes:
    dimension: Number of scalar free parameters.
    min_eigenvalue: Smallest eigenvalue of the symmetrized Hessian.
    max_eigenvalue: Largest eigenvalue of the symmetrized Hessian.
    is_positive_definite: Whether every eigenvalue is strictly positive, i.e.
      whether the optimum is a strict local maximum of the adjusted posterior.
    n_floored_eigenvalues: Number of eigenvalues that were raised to the floor
      `_EIGENVALUE_FLOOR_RATIO * max_eigenvalue` before inversion.
  """

  dimension: int
  min_eigenvalue: float
  max_eigenvalue: float
  is_positive_definite: bool
  n_floored_eigenvalues: int

  def as_attrs(self) -> dict[str, Any]:
    """Returns the diagnostics as plain values suitable for xarray attrs."""
    return {
        "hessian_dimension": int(self.dimension),
        "hessian_min_eigenvalue": float(self.min_eigenvalue),
        "hessian_max_eigenvalue": float(self.max_eigenvalue),
        "hessian_is_positive_definite": bool(self.is_positive_definite),
        "hessian_n_floored_eigenvalues": int(self.n_floored_eigenvalues),
    }


def covariance_from_hessian(
    hessian: np.ndarray,
    eigenvalue_floor_ratio: float = _EIGENVALUE_FLOOR_RATIO,
) -> tuple[np.ndarray, HessianDiagnostics]:
  """Inverts the Hessian of a negative log density into a covariance matrix.

  The Hessian is symmetrized and decomposed with `eigh`. Eigenvalues below
  `eigenvalue_floor_ratio * max_eigenvalue` (including any non-positive
  eigenvalues, which indicate that the optimum is not a strict local maximum
  or that optimization has not converged) are raised to that floor so that the
  result is always a valid covariance matrix.

  Args:
    hessian: `(dim, dim)` Hessian of the negative (adjusted) log posterior
      evaluated at its minimizer.
    eigenvalue_floor_ratio: Floor for eigenvalues, relative to the largest
      eigenvalue.

  Returns:
    A tuple `(covariance, diagnostics)` with the `(dim, dim)` float64 covariance
    matrix and the `HessianDiagnostics` of the decomposition.

  Raises:
    ValueError: If `hessian` is not a finite square matrix.
  """
  hessian = np.asarray(hessian, dtype=np.float64)
  if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
    raise ValueError(
        f"`hessian` must be a square matrix, got shape {hessian.shape}."
    )
  if not np.all(np.isfinite(hessian)):
    raise ValueError("`hessian` contains non-finite entries.")

  symmetric = 0.5 * (hessian + hessian.T)
  eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
  max_eigenvalue = float(np.max(eigenvalues))
  min_eigenvalue = float(np.min(eigenvalues))
  floor = (
      max(max_eigenvalue, np.finfo(np.float64).tiny) * eigenvalue_floor_ratio
  )
  floored = eigenvalues < floor
  n_floored = int(np.sum(floored))
  is_positive_definite = min_eigenvalue > 0.0
  if not is_positive_definite:
    warnings.warn(
        "The Hessian of the adjusted negative log posterior is not positive"
        f" definite (smallest eigenvalue {min_eigenvalue:.3g}); {n_floored}"
        " eigenvalue(s) were floored before inversion. The optimum may not be"
        " a strict local maximum, or the optimizer may not have converged, so"
        " the resulting intervals should be treated with caution."
    )
  safe_eigenvalues = np.where(floored, floor, eigenvalues)
  covariance = (eigenvectors / safe_eigenvalues) @ eigenvectors.T
  covariance = 0.5 * (covariance + covariance.T)
  diagnostics = HessianDiagnostics(
      dimension=int(hessian.shape[0]),
      min_eigenvalue=min_eigenvalue,
      max_eigenvalue=max_eigenvalue,
      is_positive_definite=is_positive_definite,
      n_floored_eigenvalues=n_floored,
  )
  return covariance, diagnostics


class _SqrtGamma:
  """Distribution of `sqrt(G)` for `G ~ Gamma(concentration, rate)`.

  This is the Nakagami family; `concentration = 1 / 2` recovers the
  `HalfNormal(1 / sqrt(2 * rate))` distribution. It is implemented on top of
  the Gamma distribution so that any positive `concentration` is supported.
  """

  def __init__(self, concentration: np.ndarray, rate: np.ndarray):
    self._concentration = np.asarray(concentration, dtype=np.float64)
    self._rate = np.asarray(rate, dtype=np.float64)
    self._gamma = stats.gamma(a=self._concentration, scale=1.0 / self._rate)

  def cdf(self, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0.0, self._gamma.cdf(np.square(np.maximum(x, 0))), 0.0)

  def ppf(self, q: np.ndarray) -> np.ndarray:
    return np.sqrt(self._gamma.ppf(q))

  def mean(self) -> np.ndarray:
    k = self._concentration
    return np.exp(special.gammaln(k + 0.5) - special.gammaln(k)) / np.sqrt(
        self._rate
    )

  def std(self) -> np.ndarray:
    variance = self._concentration / self._rate - np.square(self.mean())
    return np.sqrt(np.maximum(variance, 0.0))


@dataclasses.dataclass(frozen=True)
class ApproximatingMarginal:
  """Curvature-matched approximating marginal of one Meridian parameter.

  Let `m` be the ADM estimate of a parameter (the maximizer of the adjusted
  marginal posterior `a(theta) p(theta | y)`) and let `kappa` be the negative
  second derivative of `log[a(theta) p(theta | y)]` at `m`. The member of the
  approximating family whose *adjusted* density `a(theta) f(theta)` has mode
  `m` and the same curvature `kappa` is selected. Because ADM chose `a` so that
  the adjusted mode of every family member equals its mean, the fitted member
  reproduces the ADM estimate as its mean (exactly for the Normal, Gamma,
  LogNormal, InverseGamma and Beta families), and the fit is exact whenever the
  marginal posterior belongs to the family.

  Family parameters, using TensorFlow Probability naming:

  *   `NORMAL` / `NONE`: `Normal(loc, scale)` on `theta`.
  *   `GAMMA`: `Gamma(concentration, rate)` on the distance `d` from the
      anchoring boundary.
  *   `LOGNORMAL`: `LogNormal(loc, scale)` on `d`.
  *   `INVERSE_GAMMA`: `InverseGamma(concentration, scale)` on `d`.
  *   `HALF_NORMAL`: `d ** 2 ~ Gamma(concentration, rate)`, the Nakagami
      family, which generalizes `HalfNormal` (`concentration = 1 / 2`) and
      tends to a Normal as `concentration` grows.
  *   `BETA`: `Beta(concentration1, concentration0)` on `(theta - low) / (high
      - low)`.

  The anchoring boundary of the half-line families is `low` when it exists and
  `high` otherwise. When the family's natural support exceeds the parameter
  support (for example, a half-line family on a bounded parameter), quantiles
  and draws are truncated to the parameter support.

  Attributes:
    family: Approximating family.
    support: Support of the parameter.
    estimate: ADM estimate `m`, one value per parameter element.
    curvature: Curvature `kappa` of the adjusted log posterior at `m`.
    parameters: Fitted family parameters, broadcast to the parameter shape.
  """

  family: AdjustmentFamily
  support: Support
  estimate: np.ndarray
  curvature: np.ndarray
  parameters: Mapping[str, np.ndarray]

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the parameter."""
    return tuple(self.estimate.shape)

  @property
  def mean(self) -> np.ndarray:
    """Mean of the approximating marginal, i.e. the ADM estimate."""
    return self.estimate

  @property
  def sd(self) -> np.ndarray:
    """Standard deviation of the (untruncated) approximating family."""
    _, scale, base, _, _ = self._anchored_base()
    return np.abs(scale) * np.asarray(base.std(), dtype=np.float64)

  def quantile(self, q: np.ndarray | float) -> np.ndarray:
    """Evaluates the quantile function at probabilities `q`.

    Args:
      q: Probabilities in `(0, 1)`, broadcastable against the parameter shape
        (for example `(n_draws, *shape)`).

    Returns:
      Quantiles of the approximating marginal, truncated to the parameter
      support.
    """
    offset, scale, base, cdf_lo, cdf_hi = self._anchored_base()
    q = np.asarray(q, dtype=np.float64)
    # Families anchored at the upper boundary decrease in the base variable.
    q = np.where(scale < 0, 1.0 - q, q)
    x = base.ppf(cdf_lo + q * (cdf_hi - cdf_lo))
    return offset + scale * x

  def cdf(self, theta: np.ndarray | float) -> np.ndarray:
    """Evaluates the cumulative distribution function at `theta`."""
    offset, scale, base, cdf_lo, cdf_hi = self._anchored_base()
    x = (np.asarray(theta, dtype=np.float64) - offset) / scale
    p = (base.cdf(x) - cdf_lo) / (cdf_hi - cdf_lo)
    p = np.clip(p, 0.0, 1.0)
    return np.where(scale < 0, 1.0 - p, p)

  def summary(
      self, confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL
  ) -> np.ndarray:
    """Summarizes the marginal as `SUMMARY_METRICS` stacked on a leading axis.

    Args:
      confidence_level: Confidence level of the central credible interval.

    Returns:
      Array of shape `(len(SUMMARY_METRICS), *shape)` holding the mean (the ADM
      estimate), the standard deviation, the median and the lower and upper
      bounds of the central `confidence_level` credible interval.
    """
    alpha = (1.0 - confidence_level) / 2.0
    return np.stack(
        [
            np.broadcast_to(self.mean, self.shape),
            np.broadcast_to(self.sd, self.shape),
            self.quantile(0.5),
            self.quantile(alpha),
            self.quantile(1.0 - alpha),
        ],
        axis=0,
    )

  def _bounds(self) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns the support boundaries broadcast to the parameter shape."""
    low = (
        None
        if self.support.low is None
        else np.broadcast_to(self.support.low, self.shape)
    )
    high = (
        None
        if self.support.high is None
        else np.broadcast_to(self.support.high, self.shape)
    )
    return low, high

  def _anchored_base(self) -> tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    """Returns `(offset, scale, base, cdf_lo, cdf_hi)`.

    `theta = offset + scale * x` where `x` follows the (untruncated) `base`
    distribution restricted to the probability range `[cdf_lo, cdf_hi]`.
    """
    low, high = self._bounds()
    p = self.parameters
    zeros = np.zeros(self.shape)
    ones = np.ones(self.shape)
    if self.family in _TRIVIAL_FAMILIES:
      base = stats.norm(loc=p["loc"], scale=p["scale"])
      offset, scale = zeros, ones
      x_lo = -np.inf * ones if low is None else low
      x_hi = np.inf * ones if high is None else high
    elif self.family == AdjustmentFamily.BETA:
      if low is None or high is None:
        raise ValueError("The 'beta' family requires a bounded support.")
      base = stats.beta(a=p["concentration1"], b=p["concentration0"])
      offset, scale = low, high - low
      x_lo, x_hi = zeros, ones
    else:
      if self.family == AdjustmentFamily.GAMMA:
        base = stats.gamma(a=p["concentration"], scale=1.0 / p["rate"])
      elif self.family == AdjustmentFamily.LOGNORMAL:
        base = stats.lognorm(s=p["scale"], scale=np.exp(p["loc"]))
      elif self.family == AdjustmentFamily.INVERSE_GAMMA:
        base = stats.invgamma(a=p["concentration"], scale=p["scale"])
      elif self.family == AdjustmentFamily.HALF_NORMAL:
        base = _SqrtGamma(p["concentration"], p["rate"])
      else:
        raise ValueError(f"Unsupported adjustment family {self.family}.")
      if low is not None:
        offset, scale = low, ones
        x_hi = np.inf * ones if high is None else high - low
      elif high is not None:
        offset, scale = high, -ones
        x_hi = np.inf * ones
      else:
        raise ValueError(
            f"The '{self.family.value}' family requires a bounded boundary."
        )
      x_lo = zeros
    cdf_lo = np.asarray(base.cdf(x_lo), dtype=np.float64)
    cdf_hi = np.asarray(base.cdf(x_hi), dtype=np.float64)
    return offset, scale, base, cdf_lo, cdf_hi


def fit_marginal(
    family: AdjustmentFamily,
    support: Support,
    estimate: np.ndarray | float,
    curvature: np.ndarray | float,
) -> ApproximatingMarginal:
  """Fits a member of `family` by curvature matching at the ADM estimate.

  Args:
    family: Approximating family of the parameter.
    support: Support of the parameter.
    estimate: ADM estimate `m` of each parameter element.
    curvature: Negative second derivative of the adjusted log posterior with
      respect to each element at `m`, typically the reciprocal of the
      corresponding diagonal element of the inverse Hessian. Must be positive.

  Returns:
    The fitted `ApproximatingMarginal`.

  Raises:
    ValueError: If `family` is incompatible with `support`, or `curvature` is
      not strictly positive.
  """
  validate_family(family, support)
  m = np.asarray(estimate, dtype=np.float64)
  kappa = np.broadcast_to(np.asarray(curvature, dtype=np.float64), m.shape)
  if not np.all(kappa > 0.0):
    raise ValueError("`curvature` must be strictly positive.")
  low = None if support.low is None else np.broadcast_to(support.low, m.shape)
  high = (
      None if support.high is None else np.broadcast_to(support.high, m.shape)
  )

  if family in _TRIVIAL_FAMILIES:
    parameters = {"loc": m, "scale": 1.0 / np.sqrt(kappa)}
  elif family == AdjustmentFamily.BETA:
    if low is None or high is None:
      raise ValueError(f"{family} requires a bounded support.")
    width = high - low
    x = np.clip((m - low) / width, _MIN_DISTANCE, 1.0 - _MIN_DISTANCE)
    total = kappa * width**2 * x * (1.0 - x)
    parameters = {
        "concentration1": x * total,
        "concentration0": (1 - x) * total,
    }
  else:
    # Power families: parameterized by the distance `d` from the estimate to
    # the finite boundary of the half-line support.
    if low is not None:
      d = m - low
    elif high is not None:
      d = high - m
    else:
      raise ValueError(f"{family} requires a support bounded on one side.")
    d = np.maximum(d, _MIN_DISTANCE)
    if family == AdjustmentFamily.GAMMA:
      parameters = {"concentration": kappa * d**2, "rate": kappa * d}
    elif family == AdjustmentFamily.LOGNORMAL:
      variance = 1.0 / (kappa * d**2)
      parameters = {
          "loc": np.log(d) - 0.5 * variance,
          "scale": np.sqrt(variance),
      }
    elif family == AdjustmentFamily.INVERSE_GAMMA:
      parameters = {
          "concentration": kappa * d**2 + 1.0,
          "scale": kappa * d**3,
      }
    elif family == AdjustmentFamily.HALF_NORMAL:
      exponent = _POWER_EXPONENT[AdjustmentFamily.HALF_NORMAL]
      parameters = {
          "concentration": (0.5 * kappa * d**2 + 1.0 - exponent) / 2.0,
          "rate": kappa / 4.0,
      }
    else:
      raise ValueError(f"Unsupported adjustment family {family}.")

  return ApproximatingMarginal(
      family=family,
      support=support,
      estimate=m,
      curvature=np.array(kappa),
      parameters={
          k: np.asarray(v, dtype=np.float64) for k, v in parameters.items()
      },
  )


def _correlation_cholesky(correlation: np.ndarray) -> np.ndarray:
  """Returns a lower-triangular `L` with `L @ L.T` close to `correlation`."""
  try:
    return np.linalg.cholesky(correlation)
  except np.linalg.LinAlgError:
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, _EIGENVALUE_FLOOR_RATIO)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = 1.0 / np.sqrt(np.diag(repaired))
    repaired = repaired * np.outer(scale, scale)
    return np.linalg.cholesky(repaired)


@dataclasses.dataclass(frozen=True)
class ApproximatePosterior:
  """Approximate joint posterior derived from the ADM optimum and its Hessian.

  Each parameter block has a curvature-matched `ApproximatingMarginal`, and
  the blocks are coupled through a Gaussian copula whose correlation matrix is
  implied by the inverse Hessian of the adjusted negative log posterior. Draws
  therefore reproduce the approximating marginals exactly while preserving
  the local correlation structure of the posterior.

  Attributes:
    marginals: Approximating marginals keyed by parameter name, in the order of
      the flattened parameter vector.
    correlation: `(dim, dim)` copula correlation matrix.
    diagnostics: Diagnostics of the Hessian decomposition.
  """

  marginals: Mapping[str, ApproximatingMarginal]
  correlation: np.ndarray
  diagnostics: HessianDiagnostics

  @property
  def dimension(self) -> int:
    """Total number of scalar parameters."""
    return int(self.correlation.shape[0])

  def summary(
      self, confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL
  ) -> dict[str, np.ndarray]:
    """Summarizes every marginal; see `ApproximatingMarginal.summary`."""
    return {
        name: marginal.summary(confidence_level)
        for name, marginal in self.marginals.items()
    }

  def sample(self, standard_normals: np.ndarray) -> dict[str, np.ndarray]:
    """Transforms standard normal variates into approximate posterior draws.

    Args:
      standard_normals: `(n_draws, dim)` array of independent standard normal
        variates.

    Returns:
      Mapping from parameter name to draws shaped `(n_draws, *shape)`.

    Raises:
      ValueError: If `standard_normals` does not have shape `(n_draws, dim)`.
    """
    normals = np.asarray(standard_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != self.dimension:
      raise ValueError(
          "`standard_normals` must have shape `(n_draws,"
          f" {self.dimension})`, got {normals.shape}."
      )
    correlated = normals @ _correlation_cholesky(self.correlation).T
    uniforms = np.clip(
        stats.norm.cdf(correlated), _UNIFORM_CLIP, 1.0 - _UNIFORM_CLIP
    )
    draws: dict[str, np.ndarray] = {}
    offset = 0
    for name, marginal in self.marginals.items():
      size = int(np.prod(marginal.shape)) if marginal.shape else 1
      block = uniforms[:, offset : offset + size].reshape(
          (normals.shape[0],) + marginal.shape
      )
      draws[name] = marginal.quantile(block)
      offset += size
    return draws


def fit_approximate_posterior(
    estimates: Mapping[str, np.ndarray],
    hessian: np.ndarray,
    adjustments: Mapping[str, ParameterAdjustment],
) -> ApproximatePosterior:
  """Builds the `ApproximatePosterior` from the ADM optimum and its Hessian.

  Args:
    estimates: ADM estimates keyed by parameter name. The flattened
      concatenation of the values, in iteration order, must match the layout of
      `hessian`.
    hessian: `(dim, dim)` Hessian of the adjusted negative log posterior with
      respect to the flattened constrained parameter vector, evaluated at the
      ADM optimum.
    adjustments: Resolved adjustment (family and support) of every parameter in
      `estimates`.

  Returns:
    The fitted `ApproximatePosterior`.

  Raises:
    ValueError: If the sizes of `estimates` do not add up to the dimension of
      `hessian`, or a parameter is missing from `adjustments`.
  """
  arrays = {
      name: np.asarray(value, dtype=np.float64)
      for name, value in estimates.items()
  }
  missing = sorted(set(arrays) - set(adjustments))
  if missing:
    raise ValueError(f"Missing ADM adjustment for parameter(s) {missing}.")
  covariance, diagnostics = covariance_from_hessian(hessian)
  total_size = sum(int(value.size) for value in arrays.values())
  if total_size != diagnostics.dimension:
    raise ValueError(
        f"`estimates` hold {total_size} scalar parameters but `hessian` has"
        f" dimension {diagnostics.dimension}."
    )

  variances = np.diag(covariance)
  sds = np.sqrt(variances)
  correlation = covariance / np.outer(sds, sds)
  np.fill_diagonal(correlation, 1.0)

  marginals: dict[str, ApproximatingMarginal] = {}
  offset = 0
  for name, estimate in arrays.items():
    size = int(estimate.size)
    curvature = 1.0 / variances[offset : offset + size].reshape(estimate.shape)
    adjustment = adjustments[name]
    marginals[name] = fit_marginal(
        family=adjustment.family,
        support=adjustment.support,
        estimate=estimate,
        curvature=curvature,
    )
    offset += size
  return ApproximatePosterior(
      marginals=marginals, correlation=correlation, diagnostics=diagnostics
  )
