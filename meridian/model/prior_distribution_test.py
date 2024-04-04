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

from collections.abc import MutableMapping
import copy
from typing import Any
import warnings
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.model import prior_distribution
import tensorflow_probability as tfp

_N_GEOS = 10
_N_GEOS_NATIONAL = 1
_N_MEDIA_CHANNELS = 6
_N_RF_CHANNELS = 4
_N_CONTROLS = 3
_N_KNOTS = 5


class PriorDistributionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.sample_distributions = {
        c.KNOT_VALUES: tfp.distributions.Normal(0.0, 5.0, name=c.KNOT_VALUES),
        c.TAU_G_EXCL_BASELINE: tfp.distributions.Normal(
            0.0, 5.0, name=c.TAU_G_EXCL_BASELINE
        ),
        c.BETA_M: tfp.distributions.HalfNormal(5.0, name=c.BETA_M),
        c.BETA_RF: tfp.distributions.HalfNormal(5.0, name=c.BETA_RF),
        c.ETA_M: tfp.distributions.HalfNormal(1.0, name=c.ETA_M),
        c.ETA_RF: tfp.distributions.HalfNormal(1.0, name=c.ETA_RF),
        c.GAMMA_C: tfp.distributions.Normal(0.0, 5.0, name=c.GAMMA_C),
        c.XI_C: tfp.distributions.HalfNormal(5.0, name=c.XI_C),
        c.ALPHA_M: tfp.distributions.Uniform(0.0, 1.0, name=c.ALPHA_M),
        c.ALPHA_RF: tfp.distributions.Uniform(0.0, 1.0, name=c.ALPHA_RF),
        c.EC_M: tfp.distributions.TruncatedNormal(
            0.8, 0.8, 0.1, 10, name=c.EC_M
        ),
        c.EC_RF: tfp.distributions.TransformedDistribution(
            tfp.distributions.LogNormal(0.7, 0.4),
            tfp.bijectors.Shift(0.1),
            name=c.EC_RF,
        ),
        c.SLOPE_M: tfp.distributions.Deterministic(1.0, name=c.SLOPE_M),
        c.SLOPE_RF: tfp.distributions.LogNormal(0.7, 0.4, name=c.SLOPE_RF),
        c.SIGMA: tfp.distributions.HalfNormal(5.0, name=c.SIGMA),
        c.ROI_M: tfp.distributions.LogNormal(0.2, 0.9, name=c.ROI_M),
        c.ROI_RF: tfp.distributions.LogNormal(0.2, 0.9, name=c.ROI_RF),
    }

    self.sample_broadcast = prior_distribution.PriorDistribution().broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        sigma_shape=_N_GEOS,
        n_knots=_N_KNOTS,
        is_national=False,
    )

  def assert_distribution_params_are_equal(
      self,
      a: tfp.distributions.Distribution | MutableMapping[str, Any],
      b: tfp.distributions.Distribution | MutableMapping[str, Any],
  ):
    """Assert that two distributions are equal."""
    a_params = (
        a.parameters.copy()
        if isinstance(a, tfp.distributions.Distribution)
        else copy.deepcopy(a)
    )
    b_params = (
        b.parameters.copy()
        if isinstance(b, tfp.distributions.Distribution)
        else copy.deepcopy(b)
    )

    if c.DISTRIBUTION in a_params and c.DISTRIBUTION in b_params:
      self.assert_distribution_params_are_equal(
          a_params[c.DISTRIBUTION], b_params[c.DISTRIBUTION]
      )
      del a_params[c.DISTRIBUTION]
      del b_params[c.DISTRIBUTION]

    # Both should NOT have 'distribution' in parameters.
    self.assertNotIn(c.DISTRIBUTION, a_params)
    self.assertNotIn(c.DISTRIBUTION, b_params)
    self.assertEqual(a_params, b_params)

  def test_sample_distribution(self):
    distribution = prior_distribution.PriorDistribution()

    self.assert_distribution_params_are_equal(
        distribution.knot_values,
        self.sample_distributions[c.KNOT_VALUES],
    )
    self.assert_distribution_params_are_equal(
        distribution.tau_g_excl_baseline,
        self.sample_distributions[c.TAU_G_EXCL_BASELINE],
    )
    self.assert_distribution_params_are_equal(
        distribution.alpha_m, self.sample_distributions[c.ALPHA_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.alpha_rf, self.sample_distributions[c.ALPHA_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_m, self.sample_distributions[c.EC_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_m, self.sample_distributions[c.BETA_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_rf, self.sample_distributions[c.BETA_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_m, self.sample_distributions[c.ETA_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_rf, self.sample_distributions[c.ETA_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.gamma_c, self.sample_distributions[c.GAMMA_C]
    )
    self.assert_distribution_params_are_equal(
        distribution.xi_c, self.sample_distributions[c.XI_C]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_rf, self.sample_distributions[c.EC_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_m, self.sample_distributions[c.SLOPE_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_rf, self.sample_distributions[c.SLOPE_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.sigma, self.sample_distributions[c.SIGMA]
    )
    self.assert_distribution_params_are_equal(
        distribution.roi_m, self.sample_distributions[c.ROI_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.roi_rf, self.sample_distributions[c.ROI_RF]
    )

  def test_has_deterministic_param_broadcasted_distribution_correct(self):
    for d in self.sample_distributions:
      is_determenistic_param = d == c.SLOPE_M
      self.assertEqual(
          self.sample_broadcast.has_deterministic_param(d),
          is_determenistic_param,
      )

  def test_broadcast_preserves_distribution(self):
    distribution = prior_distribution.PriorDistribution()
    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        sigma_shape=_N_GEOS,
        n_knots=_N_KNOTS,
        is_national=False,
    )

    scalar_distributions_list = [
        distribution.knot_values,
        distribution.tau_g_excl_baseline,
        distribution.alpha_m,
        distribution.alpha_rf,
        distribution.ec_m,
        distribution.ec_rf,
        distribution.beta_m,
        distribution.beta_rf,
        distribution.eta_m,
        distribution.eta_rf,
        distribution.gamma_c,
        distribution.xi_c,
        distribution.slope_m,
        distribution.slope_rf,
        distribution.sigma,
        distribution.roi_m,
        distribution.roi_rf,
    ]

    broadcast_distributions_list = [
        broadcast_distribution.knot_values.parameters[c.DISTRIBUTION],
        broadcast_distribution.tau_g_excl_baseline.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.gamma_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.sigma.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_rf.parameters[c.DISTRIBUTION],
    ]

    # Compare Distributions.
    for scal, broad in zip(
        scalar_distributions_list, broadcast_distributions_list
    ):
      self.assert_distribution_params_are_equal(scal, broad)

  @parameterized.named_parameters(('one_sigma', False), ('unique_sigmas', True))
  def test_broadcast_preserves_shape(self, unique_sigma_for_each_geo: bool):
    sigma_shape = _N_GEOS if unique_sigma_for_each_geo else 1

    # Create prior distribution with beta_m broadcasted to n_media_channels and
    # other parameters as scalars.
    distribution = prior_distribution.PriorDistribution(
        beta_m=tfp.distributions.BatchBroadcast(
            tfp.distributions.HalfNormal(5.0), _N_MEDIA_CHANNELS, name=c.BETA_M
        )
    )

    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        sigma_shape=sigma_shape,
        n_knots=_N_KNOTS,
        is_national=False,
    )

    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        broadcast_distribution.tau_g_excl_baseline.batch_shape, (_N_GEOS - 1,)
    )

    # Validate `knot_values` distributions.
    self.assertEqual(
        broadcast_distribution.knot_values.batch_shape, (_N_KNOTS,)
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        broadcast_distribution.beta_m,
        broadcast_distribution.eta_m,
        broadcast_distribution.alpha_m,
        broadcast_distribution.slope_m,
        broadcast_distribution.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_MEDIA_CHANNELS,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        broadcast_distribution.beta_rf,
        broadcast_distribution.alpha_rf,
        broadcast_distribution.ec_rf,
        broadcast_distribution.eta_rf,
        broadcast_distribution.slope_rf,
        broadcast_distribution.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_RF_CHANNELS,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        broadcast_distribution.gamma_c,
        broadcast_distribution.xi_c,
    ]

    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_CONTROLS,))

    # Validate sigma.
    self.assertEqual(broadcast_distribution.sigma.batch_shape, (sigma_shape,))

  def test_broadcast_custom_slope_m_raises_warning(self):
    distribution = prior_distribution.PriorDistribution(
        slope_m=tfp.distributions.LogNormal(0.7, 0.4, name=c.SLOPE_M)
    )
    with warnings.catch_warnings(record=True) as warns:
      # Cause all warnings to always be triggered.
      warnings.simplefilter('always')
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          sigma_shape=_N_GEOS_NATIONAL,
          n_knots=_N_KNOTS,
          is_national=False,
      )
      self.assertLen(warns, 1)
      for w in warns:
        self.assertTrue(issubclass(w.category, UserWarning))
        self.assertIn(
            'Changing the prior for `slope_m` may lead to convex Hill curves.'
            ' This may lead to poor MCMC convergence and budget optimization'
            ' may no longer produce a global optimum.',
            str(w.message),
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_deteremenistic(0)',
          tau_g_excl_baseline=tfp.distributions.Deterministic(
              0, name='tau_g_excl_baseline'
          ),
          eta_m=tfp.distributions.Deterministic(0, name=c.ETA_M),
          eta_rf=tfp.distributions.Deterministic(0, name=c.ETA_RF),
          xi_c=tfp.distributions.Deterministic(0, name=c.XI_C),
          number_of_warnings=0,
      ),
      dict(
          testcase_name='with_deteremenistic(1)',
          tau_g_excl_baseline=tfp.distributions.Deterministic(
              1, name='tau_g_excl_baseline'
          ),
          eta_m=tfp.distributions.Deterministic(1, name=c.ETA_M),
          eta_rf=tfp.distributions.Deterministic(1, name=c.ETA_RF),
          xi_c=tfp.distributions.Deterministic(1, name=c.XI_C),
          number_of_warnings=4,
      ),
      dict(
          testcase_name='with_non-deteremenistic_defaults',
          tau_g_excl_baseline=None,
          eta_m=None,
          eta_rf=None,
          xi_c=None,
          number_of_warnings=4,
      ),
  )
  def test_broadcast_national_distribution(
      self,
      tau_g_excl_baseline: tfp.distributions.Distribution,
      eta_m: tfp.distributions.Distribution,
      eta_rf: tfp.distributions.Distribution,
      xi_c: tfp.distributions.Distribution,
      number_of_warnings: int,
  ):
    tau_g_excl_baseline = (
        self.sample_distributions['tau_g_excl_baseline']
        if not tau_g_excl_baseline
        else tau_g_excl_baseline
    )
    eta_m = self.sample_distributions[c.ETA_M] if not eta_m else eta_m
    eta_rf = self.sample_distributions[c.ETA_RF] if not eta_rf else eta_rf
    xi_c = self.sample_distributions[c.XI_C] if not xi_c else xi_c

    # Create prior distribution with given parameters.
    distribution = prior_distribution.PriorDistribution(
        tau_g_excl_baseline=tau_g_excl_baseline,
        eta_m=eta_m,
        eta_rf=eta_rf,
        xi_c=xi_c,
    )

    with warnings.catch_warnings(record=True) as warns:
      # Cause all warnings to always be triggered.
      warnings.simplefilter('always')
      broadcast_distribution = distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          sigma_shape=_N_GEOS_NATIONAL,
          n_knots=_N_KNOTS,
          is_national=True,
      )
      self.assertLen(warns, number_of_warnings)
      for w in warns:
        self.assertTrue(issubclass(w.category, UserWarning))
        self.assertIn(
            'Hierarchical distribution parameters must be deterministically'
            ' zero for national models.',
            str(w.message),
        )

    # Validate Determenistic(0) distributions.
    self.assertIsInstance(
        broadcast_distribution.tau_g_excl_baseline.parameters[c.DISTRIBUTION],
        tfp.distributions.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.tau_g_excl_baseline.parameters[
            c.DISTRIBUTION
        ].loc,
        0,
    )
    self.assertIsInstance(
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION],
        tfp.distributions.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION],
        tfp.distributions.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION],
        tfp.distributions.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION].loc, 0
    )

    # Validate `knot_values` distributions.
    self.assertEqual(
        broadcast_distribution.knot_values.batch_shape, (_N_KNOTS,)
    )

    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        broadcast_distribution.tau_g_excl_baseline.batch_shape,
        (_N_GEOS_NATIONAL - 1,),
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        broadcast_distribution.beta_m,
        broadcast_distribution.alpha_m,
        broadcast_distribution.ec_m,
        broadcast_distribution.eta_m,
        broadcast_distribution.slope_m,
        broadcast_distribution.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_MEDIA_CHANNELS,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        broadcast_distribution.beta_rf,
        broadcast_distribution.alpha_rf,
        broadcast_distribution.ec_rf,
        broadcast_distribution.eta_rf,
        broadcast_distribution.slope_rf,
        broadcast_distribution.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_RF_CHANNELS,))

    # Validate `n_controls` shape distributions.
    self.assertEqual(broadcast_distribution.gamma_c.batch_shape, (_N_CONTROLS,))
    self.assertEqual(broadcast_distribution.xi_c.batch_shape, (_N_CONTROLS,))

    # Validate sigma.
    self.assertEqual(
        broadcast_distribution.sigma.batch_shape, (_N_GEOS_NATIONAL,)
    )

  @parameterized.named_parameters(
      (c.KNOT_VALUES, c.KNOT_VALUES),
      (c.TAU_G_EXCL_BASELINE, c.TAU_G_EXCL_BASELINE),
      (c.BETA_M, c.BETA_M),
      (c.BETA_RF, c.BETA_RF),
      (c.ETA_M, c.ETA_M),
      (c.ETA_RF, c.ETA_RF),
      (c.GAMMA_C, c.GAMMA_C),
      (c.XI_C, c.XI_C),
      (c.ALPHA_M, c.ALPHA_M),
      (c.ALPHA_RF, c.ALPHA_RF),
      (c.EC_M, c.EC_M),
      (c.EC_RF, c.EC_RF),
      (c.SLOPE_M, c.SLOPE_M),
      (c.SLOPE_RF, c.SLOPE_RF),
      (c.SIGMA, c.SIGMA),
      (c.ROI_M, c.ROI_M),
      (c.ROI_RF, c.ROI_RF),
  )
  def test_getstate_correct(self, attribute):
    def _distribution_info(
        distribution: tfp.distributions.Distribution,
    ) -> MutableMapping[str, Any]:
      info = distribution.parameters | {c.DISTRIBUTION_TYPE: type(distribution)}
      if c.DISTRIBUTION in info:
        info[c.DISTRIBUTION] = _distribution_info(distribution.distribution)
      return info

    expected_distribution_state = {
        c.KNOT_VALUES: _distribution_info(self.sample_broadcast.knot_values),
        c.TAU_G_EXCL_BASELINE: _distribution_info(
            self.sample_broadcast.tau_g_excl_baseline
        ),
        c.BETA_M: _distribution_info(self.sample_broadcast.beta_m),
        c.BETA_RF: _distribution_info(self.sample_broadcast.beta_rf),
        c.ETA_M: _distribution_info(self.sample_broadcast.eta_m),
        c.ETA_RF: _distribution_info(self.sample_broadcast.eta_rf),
        c.GAMMA_C: _distribution_info(self.sample_broadcast.gamma_c),
        c.XI_C: _distribution_info(self.sample_broadcast.xi_c),
        c.ALPHA_M: _distribution_info(self.sample_broadcast.alpha_m),
        c.ALPHA_RF: _distribution_info(self.sample_broadcast.alpha_rf),
        c.EC_M: _distribution_info(self.sample_broadcast.ec_m),
        c.EC_RF: _distribution_info(self.sample_broadcast.ec_rf),
        c.SLOPE_M: _distribution_info(self.sample_broadcast.slope_m),
        c.SLOPE_RF: _distribution_info(self.sample_broadcast.slope_rf),
        c.SIGMA: _distribution_info(self.sample_broadcast.sigma),
        c.ROI_M: _distribution_info(self.sample_broadcast.roi_m),
        c.ROI_RF: _distribution_info(self.sample_broadcast.roi_rf),
    }

    self.assert_distribution_params_are_equal(
        self.sample_broadcast.__getstate__().get(attribute),
        expected_distribution_state.get(attribute),
    )

  def test_setstate_correct(self):
    distribution = prior_distribution.PriorDistribution().broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        sigma_shape=_N_GEOS,
        n_knots=_N_KNOTS,
        is_national=False,
    )
    distribution.__setstate__(distribution.__getstate__())

    for k, v in distribution.__dict__.items():
      self.assert_distribution_params_are_equal(
          v, self.sample_broadcast.__dict__[k]
      )


if __name__ == '__main__':
  absltest.main()
