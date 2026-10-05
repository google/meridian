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

"""Tests for Adjustment for Density Maximization (ADM) in Meridian Full MAP."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.model import adm
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import spec
import numpy as np
from scipy import optimize
from scipy import stats

_POSITIVE_SUPPORT = adm.Support(low=np.float64(0.0), high=None)
_UNIT_SUPPORT = adm.Support(low=np.float64(0.0), high=np.float64(1.0))
_REAL_SUPPORT = adm.Support(low=None, high=None)
# `log_adjustment` is evaluated in the backend float dtype (float32 under
# TensorFlow), which limits how precisely a scalar optimizer can localize the
# maximizer of the adjusted density (roughly `sqrt(float32 eps) ~= 3e-4`).
_RELATIVE_TOLERANCE = 1e-3


def _maximize_adjusted_density(
    log_pdf,
    family: adm.AdjustmentFamily,
    support: adm.Support,
    bounds: tuple[float, float],
) -> float:
  """Numerically maximizes `log f(theta) + log a(theta)` over `bounds`."""

  def negative_log_adjusted_density(theta: float) -> float:
    value = np.float64(theta)
    log_a = float(np.asarray(adm.log_adjustment(value, family, support)))
    return -(float(log_pdf(theta)) + log_a)

  result = optimize.minimize_scalar(
      negative_log_adjusted_density,
      bounds=bounds,
      method="bounded",
      options={"xatol": 1e-12},
  )
  return float(result.x)


def _assert_relative_close(
    test_case: parameterized.TestCase, actual: float, expected: float
) -> None:
  """Asserts `actual == expected` up to `_RELATIVE_TOLERANCE`."""
  test_case.assertAlmostEqual(
      actual / float(expected), 1.0, delta=_RELATIVE_TOLERANCE
  )


class AdjustmentFactorTest(parameterized.TestCase):
  """Verifies that the mode of the adjusted density is the unadjusted mean."""

  @parameterized.named_parameters(
      dict(testcase_name="shape_2", concentration=2.0, rate=1.0),
      dict(testcase_name="shape_half", concentration=0.5, rate=3.0),
      dict(testcase_name="shape_7", concentration=7.0, rate=0.25),
  )
  def test_gamma_adjusted_mode_equals_mean(self, concentration, rate):
    dist = stats.gamma(a=concentration, scale=1.0 / rate)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.GAMMA,
        _POSITIVE_SUPPORT,
        (1e-8, 500.0),
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  @parameterized.named_parameters(
      dict(testcase_name="a2_b5", a=2.0, b=5.0),
      dict(testcase_name="a_half_b_half", a=0.5, b=0.5),
      dict(testcase_name="a9_b1", a=9.0, b=1.0),
  )
  def test_beta_adjusted_mode_equals_mean(self, a, b):
    dist = stats.beta(a=a, b=b)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf, adm.AdjustmentFamily.BETA, _UNIT_SUPPORT, (1e-8, 1 - 1e-8)
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  @parameterized.named_parameters(
      dict(testcase_name="sigma_small", mu=0.2, sigma=0.3),
      dict(testcase_name="sigma_meridian_roi", mu=0.2, sigma=0.9),
      dict(testcase_name="sigma_large", mu=-1.0, sigma=1.2),
  )
  def test_lognormal_adjusted_mode_equals_mean(self, mu, sigma):
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.LOGNORMAL,
        _POSITIVE_SUPPORT,
        (1e-8, 5000.0),
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  @parameterized.named_parameters(
      dict(testcase_name="scale_half", scale=0.5),
      dict(testcase_name="scale_one", scale=1.0),
      dict(testcase_name="scale_five", scale=5.0),
  )
  def test_half_normal_adjusted_mode_equals_mean(self, scale):
    dist = stats.halfnorm(scale=scale)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.HALF_NORMAL,
        _POSITIVE_SUPPORT,
        (1e-8, 100.0 * scale),
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  @parameterized.named_parameters(
      dict(testcase_name="df_2", df=2),
      dict(testcase_name="df_5", df=5),
      dict(testcase_name="df_50", df=50),
  )
  def test_half_normal_beats_gamma_for_chi_scale_parameters(self, df):
    dist = stats.chi(df)
    half_normal_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.HALF_NORMAL,
        _POSITIVE_SUPPORT,
        (1e-8, 200.0),
    )
    gamma_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.GAMMA,
        _POSITIVE_SUPPORT,
        (1e-8, 200.0),
    )
    mean = float(dist.mean())
    self.assertGreater(gamma_mode, mean)
    self.assertLess(
        abs(half_normal_mode - mean),
        abs(gamma_mode - mean),
        msg=f"df={df}: half_normal={half_normal_mode} gamma={gamma_mode}",
    )

  def test_half_normal_exponent_is_two_over_pi(self):
    adjustment = adm.ParameterAdjustment(
        name="sigma",
        family=adm.AdjustmentFamily.HALF_NORMAL,
        support=_POSITIVE_SUPPORT,
    )
    self.assertAlmostEqual(
        float(
            np.asarray(
                adm.log_adjustment(
                    np.array([np.e]),
                    adm.AdjustmentFamily.HALF_NORMAL,
                    _POSITIVE_SUPPORT,
                )
            )[0]
        ),
        2.0 / np.pi,
        places=5,
    )
    self.assertIn("sigma", adjustment.describe())

  @parameterized.named_parameters(
      dict(testcase_name="shape_3", concentration=3.0, scale=2.0),
      dict(testcase_name="shape_5", concentration=5.0, scale=0.5),
  )
  def test_inverse_gamma_adjusted_mode_equals_mean(self, concentration, scale):
    dist = stats.invgamma(a=concentration, scale=scale)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.INVERSE_GAMMA,
        _POSITIVE_SUPPORT,
        (1e-6, 500.0),
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  def test_normal_adjustment_is_identity(self):
    dist = stats.norm(loc=1.3, scale=2.0)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf, adm.AdjustmentFamily.NORMAL, _REAL_SUPPORT, (-50.0, 50.0)
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  def test_shifted_positive_support_adjusted_mode_equals_mean(self):
    shift = 0.1
    dist = stats.gamma(a=2.0, loc=shift, scale=1.0 / 3.0)
    support = adm.Support(low=np.float64(shift), high=None)
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf, adm.AdjustmentFamily.GAMMA, support, (shift + 1e-9, 100.0)
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  def test_shifted_scaled_beta_adjusted_mode_equals_mean(self):
    low, high = 0.1, 10.0
    dist = stats.beta(a=2.0, b=5.0, loc=low, scale=high - low)
    support = adm.Support(low=np.float64(low), high=np.float64(high))
    adjusted_mode = _maximize_adjusted_density(
        dist.logpdf,
        adm.AdjustmentFamily.BETA,
        support,
        (low + 1e-9, high - 1e-9),
    )
    _assert_relative_close(self, adjusted_mode, dist.mean())

  def test_none_family_returns_zero_adjustment(self):
    value = np.array([0.5, 2.0], dtype=np.float32)
    adjustment = adm.log_adjustment(
        value, adm.AdjustmentFamily.NONE, _POSITIVE_SUPPORT
    )
    np.testing.assert_allclose(np.asarray(adjustment), np.zeros(2))

  def test_power_families_match_documented_exponents(self):
    value = np.array([0.5, 2.0, 7.0], dtype=np.float32)
    for family, exponent in (
        (adm.AdjustmentFamily.GAMMA, 1.0),
        (adm.AdjustmentFamily.LOGNORMAL, 1.5),
        (adm.AdjustmentFamily.INVERSE_GAMMA, 2.0),
    ):
      adjustment = np.asarray(
          adm.log_adjustment(value, family, _POSITIVE_SUPPORT)
      )
      np.testing.assert_allclose(
          adjustment, exponent * np.log(value), rtol=1e-5
      )

  def test_beta_family_rejects_unbounded_support(self):
    with self.assertRaisesRegex(ValueError, "bounded support"):
      adm.log_adjustment(
          np.float32(1.0), adm.AdjustmentFamily.BETA, _POSITIVE_SUPPORT
      )

  def test_power_family_rejects_real_support(self):
    with self.assertRaisesRegex(ValueError, "bounded boundary"):
      adm.log_adjustment(
          np.float32(1.0), adm.AdjustmentFamily.GAMMA, _REAL_SUPPORT
      )


class SupportInferenceTest(parameterized.TestCase):
  """Verifies that parameter supports are recovered from their bijectors."""

  def _support_of(self, distribution) -> adm.Support:
    bijector = distribution.experimental_default_event_space_bijector()
    return adm.infer_support(bijector)

  def test_normal_support_is_real(self):
    support = self._support_of(backend.tfd.Normal(0.0, 5.0))
    self.assertTrue(support.is_real)
    self.assertEqual(adm.default_family(support), adm.AdjustmentFamily.NORMAL)

  def test_half_normal_support_is_positive_half_line(self):
    support = self._support_of(backend.tfd.HalfNormal(5.0))
    self.assertTrue(support.is_half_line)
    self.assertIsNotNone(support.low)
    self.assertIsNone(support.high)
    np.testing.assert_allclose(np.asarray(support.low), 0.0, atol=1e-9)
    self.assertEqual(adm.default_family(support), adm.AdjustmentFamily.GAMMA)

  def test_lognormal_support_is_positive_half_line(self):
    support = self._support_of(backend.tfd.LogNormal(0.2, 0.9))
    self.assertTrue(support.is_half_line)
    np.testing.assert_allclose(np.asarray(support.low), 0.0, atol=1e-9)

  def test_uniform_support_is_unit_interval(self):
    support = self._support_of(backend.tfd.Uniform(0.0, 1.0))
    self.assertTrue(support.is_bounded)
    np.testing.assert_allclose(np.asarray(support.low), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(support.high), 1.0, atol=1e-6)
    self.assertEqual(adm.default_family(support), adm.AdjustmentFamily.BETA)

  def test_truncated_normal_support_matches_truncation_bounds(self):
    support = self._support_of(backend.tfd.TruncatedNormal(0.8, 0.8, 0.1, 10.0))
    self.assertTrue(support.is_bounded)
    np.testing.assert_allclose(np.asarray(support.low), 0.1, atol=1e-5)
    np.testing.assert_allclose(np.asarray(support.high), 10.0, atol=1e-4)

  def test_describe_is_human_readable(self):
    self.assertEqual(_REAL_SUPPORT.describe(), "(-inf, inf)")
    self.assertEqual(_UNIT_SUPPORT.describe(), "(0, 1)")


class ResolveAdjustmentsTest(
    test_utils.MeridianTestCase,
    model_test_data.WithInputDataSamples,
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_resolves_default_families_for_full_joint_model(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(knots=5),
    )
    adjustments = meridian.posterior_map_callable.resolve_adm_adjustments()

    expected = {
        constants.KNOT_VALUES: adm.AdjustmentFamily.NORMAL,
        constants.TAU_G_EXCL_BASELINE: adm.AdjustmentFamily.NORMAL,
        constants.GAMMA_C: adm.AdjustmentFamily.NORMAL,
        constants.BETA_GM_DEV: adm.AdjustmentFamily.NORMAL,
        constants.BETA_GRF_DEV: adm.AdjustmentFamily.NORMAL,
        constants.GAMMA_GC_DEV: adm.AdjustmentFamily.NORMAL,
        constants.SIGMA: adm.AdjustmentFamily.HALF_NORMAL,
        constants.ETA_M: adm.AdjustmentFamily.HALF_NORMAL,
        constants.ETA_RF: adm.AdjustmentFamily.HALF_NORMAL,
        constants.XI_C: adm.AdjustmentFamily.HALF_NORMAL,
        constants.EC_M: adm.AdjustmentFamily.HALF_NORMAL,
        constants.ALPHA_M: adm.AdjustmentFamily.BETA,
        constants.ALPHA_RF: adm.AdjustmentFamily.BETA,
        constants.ROI_M: adm.AdjustmentFamily.GAMMA,
        constants.ROI_RF: adm.AdjustmentFamily.GAMMA,
        constants.EC_RF: adm.AdjustmentFamily.GAMMA,
        constants.SLOPE_RF: adm.AdjustmentFamily.GAMMA,
    }
    for name, family in expected.items():
      self.assertIn(name, adjustments)
      self.assertEqual(adjustments[name].family, family, msg=name)

    description = adm.describe_adjustments(adjustments)
    for name in expected:
      self.assertIn(name, description)

  def test_string_and_enum_overrides_are_accepted(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=spec.ModelSpec(knots=5),
    )
    adjustments = meridian.posterior_map_callable.resolve_adm_adjustments(
        family_overrides={
            constants.ROI_M: "lognormal",
            constants.SIGMA: adm.AdjustmentFamily.INVERSE_GAMMA,
        }
    )
    self.assertEqual(
        adjustments[constants.ROI_M].family, adm.AdjustmentFamily.LOGNORMAL
    )
    self.assertEqual(
        adjustments[constants.SIGMA].family, adm.AdjustmentFamily.INVERSE_GAMMA
    )

  def test_resolves_half_normal_for_coefficient_prior_and_custom_half_normal(
      self,
  ):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(
            knots=5,
            media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
            rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
        ),
    )
    adjustments = meridian.posterior_map_callable.resolve_adm_adjustments()
    self.assertEqual(
        adjustments[constants.BETA_M].family, adm.AdjustmentFamily.HALF_NORMAL
    )
    self.assertEqual(
        adjustments[constants.BETA_RF].family, adm.AdjustmentFamily.HALF_NORMAL
    )

    # Any half-line parameter carrying a custom HalfNormal prior also gets
    # HALF_NORMAL (c = 2/pi), while a scale parameter given a bounded prior
    # falls through to BETA.
    self.assertEqual(
        adm.default_family_for_parameter(
            constants.ROI_M,
            _POSITIVE_SUPPORT,
            backend.tfd.HalfNormal(2.0),
        ),
        adm.AdjustmentFamily.HALF_NORMAL,
    )
    self.assertEqual(
        adm.default_family_for_parameter(
            constants.ETA_M,
            _UNIT_SUPPORT,
            backend.tfd.Uniform(0.0, 1.0),
        ),
        adm.AdjustmentFamily.BETA,
    )

  def test_unknown_and_incompatible_overrides_raise(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=spec.ModelSpec(knots=5),
    )
    with self.assertRaisesRegex(ValueError, "Unknown ADM family override"):
      meridian.posterior_map_callable.resolve_adm_adjustments(
          family_overrides={"nonexistent_param": "gamma"}
      )
    with self.assertRaisesRegex(ValueError, "bounded boundary"):
      meridian.posterior_map_callable.resolve_adm_adjustments(
          family_overrides={constants.KNOT_VALUES: "gamma"}
      )


def _log_adjustment_float64(
    theta: float, family: adm.AdjustmentFamily, support: adm.Support
) -> float:
  """Float64 reference implementation of `log a(theta)` for the tests."""
  if family in (adm.AdjustmentFamily.NONE, adm.AdjustmentFamily.NORMAL):
    return 0.0
  low = float("-inf") if support.low is None else float(support.low)
  high = float("inf") if support.high is None else float(support.high)
  if family == adm.AdjustmentFamily.BETA:
    return np.log(theta - low) + np.log(high - theta)
  exponent = {
      adm.AdjustmentFamily.HALF_NORMAL: 2.0 / np.pi,
      adm.AdjustmentFamily.GAMMA: 1.0,
      adm.AdjustmentFamily.LOGNORMAL: 1.5,
      adm.AdjustmentFamily.INVERSE_GAMMA: 2.0,
  }[family]
  distance = theta - low if np.isfinite(low) else high - theta
  return exponent * np.log(distance)


def _adjusted_curvature(
    log_pdf, family: adm.AdjustmentFamily, support: adm.Support, m: float
) -> float:
  """Negative second derivative of `log f + log a` at `m` (central differences)."""

  def adjusted(theta: float) -> float:
    return float(log_pdf(theta)) + _log_adjustment_float64(
        theta, family, support
    )

  h = 1e-4 * max(1.0, abs(m))
  return -(adjusted(m + h) - 2.0 * adjusted(m) + adjusted(m - h)) / h**2


def _adjusted_mode(
    log_pdf,
    family: adm.AdjustmentFamily,
    support: adm.Support,
    bracket: tuple[float, float],
) -> float:
  """Maximizer of `log f + log a` within `bracket`, i.e. the ADM estimate."""
  result = optimize.minimize_scalar(
      lambda theta: -(
          float(log_pdf(theta))
          + _log_adjustment_float64(theta, family, support)
      ),
      bounds=bracket,
      method="bounded",
      options={"xatol": 1e-10},
  )
  return float(result.x)


class ApproximatingMarginalTest(parameterized.TestCase):
  """Verifies curvature matching recovers distributions within each family."""

  _QUANTILES = np.array([0.025, 0.1, 0.5, 0.9, 0.975])

  def _assert_recovers(self, dist, family, support, mean=None):
    mean = float(dist.mean()) if mean is None else mean
    curvature = _adjusted_curvature(dist.logpdf, family, support, mean)
    marginal = adm.fit_marginal(
        family, support, np.array(mean), np.array(curvature)
    )
    np.testing.assert_allclose(marginal.mean, mean)
    np.testing.assert_allclose(
        marginal.quantile(self._QUANTILES), dist.ppf(self._QUANTILES), rtol=1e-3
    )
    np.testing.assert_allclose(marginal.sd, dist.std(), rtol=1e-3)
    np.testing.assert_allclose(
        marginal.cdf(dist.ppf(self._QUANTILES)), self._QUANTILES, atol=1e-4
    )
    return marginal

  @parameterized.named_parameters(
      dict(testcase_name="shape_2", concentration=2.0, rate=1.0),
      dict(testcase_name="shape_half", concentration=0.5, rate=3.0),
      dict(testcase_name="shape_7", concentration=7.0, rate=0.25),
  )
  def test_gamma_family_is_recovered(self, concentration, rate):
    marginal = self._assert_recovers(
        stats.gamma(a=concentration, scale=1.0 / rate),
        adm.AdjustmentFamily.GAMMA,
        _POSITIVE_SUPPORT,
    )
    np.testing.assert_allclose(
        marginal.parameters["concentration"], concentration, rtol=1e-3
    )
    np.testing.assert_allclose(marginal.parameters["rate"], rate, rtol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name="sigma_small", mu=0.2, sigma=0.3),
      dict(testcase_name="sigma_meridian_roi", mu=0.2, sigma=0.9),
  )
  def test_lognormal_family_is_recovered(self, mu, sigma):
    marginal = self._assert_recovers(
        stats.lognorm(s=sigma, scale=np.exp(mu)),
        adm.AdjustmentFamily.LOGNORMAL,
        _POSITIVE_SUPPORT,
    )
    np.testing.assert_allclose(marginal.parameters["loc"], mu, atol=1e-3)
    np.testing.assert_allclose(marginal.parameters["scale"], sigma, rtol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name="shape_3", concentration=3.0, scale=2.0),
      dict(testcase_name="shape_5", concentration=5.0, scale=0.5),
  )
  def test_inverse_gamma_family_is_recovered(self, concentration, scale):
    self._assert_recovers(
        stats.invgamma(a=concentration, scale=scale),
        adm.AdjustmentFamily.INVERSE_GAMMA,
        _POSITIVE_SUPPORT,
    )

  @parameterized.named_parameters(
      dict(testcase_name="scale_half", scale=0.5),
      dict(testcase_name="scale_five", scale=5.0),
  )
  def test_half_normal_family_is_recovered(self, scale):
    marginal = self._assert_recovers(
        stats.halfnorm(scale=scale),
        adm.AdjustmentFamily.HALF_NORMAL,
        _POSITIVE_SUPPORT,
    )
    # HalfNormal(s) is the Nakagami member with concentration 1 / 2.
    np.testing.assert_allclose(
        marginal.parameters["concentration"], 0.5, atol=1e-3
    )

  def test_half_normal_family_generalizes_to_chi_distributions(self):
    # chi(k) is the Nakagami member with concentration `k / 2` and rate `1 / 2`.
    # The `HALF_NORMAL` adjustment only recovers the mean of the HalfNormal
    # member (`k = 1`); for other members the ADM estimate is the adjusted mode
    # `d* = sqrt(k - 1 + 2 / pi)`, from which curvature matching recovers the
    # distribution exactly.
    dist = stats.chi(df=6)
    adjusted_mode = _adjusted_mode(
        dist.logpdf,
        adm.AdjustmentFamily.HALF_NORMAL,
        _POSITIVE_SUPPORT,
        bracket=(1e-6, 10.0),
    )
    np.testing.assert_allclose(adjusted_mode, np.sqrt(5.0 + 2.0 / np.pi))
    marginal = self._assert_recovers(
        dist,
        adm.AdjustmentFamily.HALF_NORMAL,
        _POSITIVE_SUPPORT,
        mean=adjusted_mode,
    )
    np.testing.assert_allclose(
        marginal.parameters["concentration"], 3.0, rtol=1e-3
    )
    np.testing.assert_allclose(marginal.parameters["rate"], 0.5, rtol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name="a2_b5", a=2.0, b=5.0),
      dict(testcase_name="a9_b1", a=9.0, b=1.0),
  )
  def test_beta_family_is_recovered(self, a, b):
    marginal = self._assert_recovers(
        stats.beta(a=a, b=b), adm.AdjustmentFamily.BETA, _UNIT_SUPPORT
    )
    np.testing.assert_allclose(
        marginal.parameters["concentration1"], a, rtol=1e-3
    )
    np.testing.assert_allclose(
        marginal.parameters["concentration0"], b, rtol=1e-3
    )

  def test_normal_family_is_recovered(self):
    self._assert_recovers(
        stats.norm(loc=1.3, scale=2.0),
        adm.AdjustmentFamily.NORMAL,
        _REAL_SUPPORT,
    )

  def test_shifted_and_scaled_supports_are_recovered(self):
    self._assert_recovers(
        stats.gamma(a=2.0, loc=0.1, scale=1.0 / 3.0),
        adm.AdjustmentFamily.GAMMA,
        adm.Support(low=0.1, high=None),
    )
    self._assert_recovers(
        stats.beta(a=2.0, b=5.0, loc=0.1, scale=9.9),
        adm.AdjustmentFamily.BETA,
        adm.Support(low=0.1, high=10.0),
    )

  def test_truncated_normal_is_approximated_within_its_support(self):
    # Meridian's default `ec_m` prior, treated as HALF_NORMAL above `low`. The
    # truncated normal is not a Nakagami member, so the fit is approximate.
    low, high = 0.1, 10.0
    dist = stats.truncnorm((low - 0.8) / 0.8, (high - 0.8) / 0.8, 0.8, 0.8)
    support = adm.Support(low=low, high=high)
    family = adm.AdjustmentFamily.HALF_NORMAL
    estimate = _adjusted_mode(
        dist.logpdf, family, support, bracket=(low + 1e-6, 5.0)
    )
    curvature = _adjusted_curvature(dist.logpdf, family, support, estimate)
    marginal = adm.fit_marginal(family, support, estimate, curvature)

    np.testing.assert_allclose(marginal.mean, estimate)
    quantiles = marginal.quantile(self._QUANTILES)
    self.assertTrue(np.all(quantiles > low))
    self.assertTrue(np.all(quantiles < high))
    np.testing.assert_allclose(marginal.cdf(high), 1.0)
    np.testing.assert_allclose(marginal.cdf(low), 0.0)
    # Coverage of the true quantiles is within 5 percentage points of nominal
    # and the dispersion is matched.
    np.testing.assert_allclose(
        marginal.cdf(dist.ppf(self._QUANTILES)), self._QUANTILES, atol=0.05
    )
    np.testing.assert_allclose(marginal.sd, dist.std(), rtol=0.05)

  def test_upper_anchored_family_decreases_towards_the_boundary(self):
    support = adm.Support(low=None, high=1.0)
    marginal = adm.fit_marginal(
        adm.AdjustmentFamily.GAMMA, support, np.array(0.0), np.array(4.0)
    )
    quantiles = marginal.quantile(self._QUANTILES)
    self.assertEqual(quantiles.shape, self._QUANTILES.shape)
    self.assertTrue(np.all(np.diff(quantiles) > 0.0))
    self.assertTrue(np.all(quantiles < 1.0))
    # Mirror image of Gamma on the distance `1 - theta`.
    reference = stats.gamma(a=4.0, scale=0.25)
    np.testing.assert_allclose(
        quantiles, 1.0 - reference.ppf(1.0 - self._QUANTILES), rtol=1e-6
    )
    np.testing.assert_allclose(marginal.sd, reference.std())
    np.testing.assert_allclose(marginal.cdf(quantiles), self._QUANTILES)

  def test_summary_layout_and_ordering(self):
    marginal = adm.fit_marginal(
        adm.AdjustmentFamily.GAMMA,
        _POSITIVE_SUPPORT,
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[4.0, 1.0, 0.5]]),
    )
    summary = marginal.summary(confidence_level=0.9)
    self.assertEqual(summary.shape, (len(adm.SUMMARY_METRICS), 1, 3))
    metric_index = {name: i for i, name in enumerate(adm.SUMMARY_METRICS)}
    np.testing.assert_allclose(
        summary[metric_index[constants.MEAN]], [[1, 2, 3]]
    )
    self.assertTrue(np.all(summary[metric_index[constants.SD]] > 0.0))
    self.assertTrue(
        np.all(
            summary[metric_index[constants.CI_LO]]
            < summary[metric_index[constants.MEDIAN]]
        )
    )
    self.assertTrue(
        np.all(
            summary[metric_index[constants.MEDIAN]]
            < summary[metric_index[constants.CI_HI]]
        )
    )
    np.testing.assert_allclose(
        summary[metric_index[constants.CI_LO]],
        marginal.quantile(0.05),
    )

  def test_fit_marginal_rejects_invalid_inputs(self):
    with self.assertRaisesRegex(ValueError, "strictly positive"):
      adm.fit_marginal(adm.AdjustmentFamily.GAMMA, _POSITIVE_SUPPORT, 1.0, -1.0)
    with self.assertRaisesRegex(ValueError, "bounded support"):
      adm.fit_marginal(adm.AdjustmentFamily.BETA, _POSITIVE_SUPPORT, 1.0, 1.0)


class ApproximatePosteriorTest(parameterized.TestCase):
  """Verifies Hessian inversion and Gaussian copula sampling."""

  def test_covariance_from_hessian_inverts_positive_definite_matrices(self):
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    hessian = np.linalg.inv(covariance)
    recovered, diagnostics = adm.covariance_from_hessian(hessian)
    np.testing.assert_allclose(recovered, covariance, rtol=1e-10)
    self.assertTrue(diagnostics.is_positive_definite)
    self.assertEqual(diagnostics.n_floored_eigenvalues, 0)
    self.assertEqual(diagnostics.dimension, 2)
    attrs = diagnostics.as_attrs()
    self.assertEqual(attrs["hessian_dimension"], 2)
    self.assertTrue(attrs["hessian_is_positive_definite"])

  def test_covariance_from_hessian_floors_non_positive_eigenvalues(self):
    hessian = np.diag([4.0, -1.0])
    with self.assertWarnsRegex(UserWarning, "not positive definite"):
      covariance, diagnostics = adm.covariance_from_hessian(hessian)
    self.assertFalse(diagnostics.is_positive_definite)
    self.assertEqual(diagnostics.n_floored_eigenvalues, 1)
    self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))
    np.testing.assert_allclose(covariance[0, 0], 0.25)

  def test_covariance_from_hessian_rejects_invalid_matrices(self):
    with self.assertRaisesRegex(ValueError, "square matrix"):
      adm.covariance_from_hessian(np.ones((2, 3)))
    with self.assertRaisesRegex(ValueError, "non-finite"):
      adm.covariance_from_hessian(np.array([[np.nan]]))

  def _fit_two_parameter_posterior(self):
    estimates = {"rate": np.array([2.0, 4.0]), "location": np.array(-1.0)}
    adjustments = {
        "rate": adm.ParameterAdjustment(
            "rate", adm.AdjustmentFamily.GAMMA, _POSITIVE_SUPPORT
        ),
        "location": adm.ParameterAdjustment(
            "location", adm.AdjustmentFamily.NORMAL, _REAL_SUPPORT
        ),
    }
    sds = np.array([0.5, 1.0, 2.0])
    correlation = np.array([
        [1.0, 0.6, -0.3],
        [0.6, 1.0, 0.2],
        [-0.3, 0.2, 1.0],
    ])
    covariance = correlation * np.outer(sds, sds)
    posterior = adm.fit_approximate_posterior(
        estimates, np.linalg.inv(covariance), adjustments
    )
    return posterior, estimates, correlation, sds

  def test_fit_approximate_posterior_matches_marginals_and_correlation(self):
    posterior, estimates, correlation, sds = self._fit_two_parameter_posterior()
    self.assertEqual(posterior.dimension, 3)
    self.assertEqual(list(posterior.marginals), ["rate", "location"])
    np.testing.assert_allclose(posterior.correlation, correlation, rtol=1e-8)
    np.testing.assert_allclose(
        posterior.marginals["rate"].mean, estimates["rate"]
    )
    np.testing.assert_allclose(
        posterior.marginals["rate"].curvature, 1.0 / sds[:2] ** 2
    )
    # The Normal marginal reproduces the Gaussian standard deviation exactly.
    np.testing.assert_allclose(posterior.marginals["location"].sd, sds[2])
    summary = posterior.summary(confidence_level=0.5)
    self.assertEqual(summary["rate"].shape, (len(adm.SUMMARY_METRICS), 2))
    self.assertEqual(summary["location"].shape, (len(adm.SUMMARY_METRICS),))

  def test_sample_preserves_marginals_and_correlation(self):
    posterior, estimates, correlation, sds = self._fit_two_parameter_posterior()
    n_draws = 200_000
    normals = np.random.default_rng(0).standard_normal((n_draws, 3))
    draws = posterior.sample(normals)
    self.assertEqual(draws["rate"].shape, (n_draws, 2))
    self.assertEqual(draws["location"].shape, (n_draws,))
    self.assertTrue(np.all(draws["rate"] > 0.0))
    np.testing.assert_allclose(
        draws["rate"].mean(axis=0), estimates["rate"], rtol=0.02
    )
    np.testing.assert_allclose(draws["rate"].std(axis=0), sds[:2], rtol=0.02)
    np.testing.assert_allclose(draws["location"].mean(), -1.0, atol=0.02)
    np.testing.assert_allclose(draws["location"].std(), sds[2], rtol=0.02)
    # Gamma marginals are reproduced exactly (not just their moments).
    gamma_marginal = posterior.marginals["rate"]
    empirical_median = np.median(draws["rate"], axis=0)
    np.testing.assert_allclose(
        empirical_median, gamma_marginal.quantile(0.5), rtol=0.02
    )
    # The Gaussian copula preserves the sign and rough size of correlations.
    stacked = np.column_stack([draws["rate"], draws["location"]])
    empirical = np.corrcoef(stacked, rowvar=False)
    np.testing.assert_allclose(empirical, correlation, atol=0.05)

  def test_sample_rejects_wrong_shapes(self):
    posterior, _, _, _ = self._fit_two_parameter_posterior()
    with self.assertRaisesRegex(ValueError, "shape"):
      posterior.sample(np.zeros((10, 2)))

  def test_fit_approximate_posterior_validates_inputs(self):
    adjustments = {
        "rate": adm.ParameterAdjustment(
            "rate", adm.AdjustmentFamily.GAMMA, _POSITIVE_SUPPORT
        )
    }
    with self.assertRaisesRegex(ValueError, "dimension"):
      adm.fit_approximate_posterior(
          {"rate": np.array([1.0, 2.0])}, np.eye(3), adjustments
      )
    with self.assertRaisesRegex(ValueError, "Missing ADM adjustment"):
      adm.fit_approximate_posterior(
          {"rate": np.array([1.0]), "other": np.array([1.0])},
          np.eye(2),
          adjustments,
      )


if __name__ == "__main__":
  absltest.main()
