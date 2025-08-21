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

from collections.abc import MutableMapping
import copy
from typing import Any
import warnings
from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants as c
from meridian.model import prior_distribution
import numpy as np
import tensorflow_probability as tfp

_N_GEOS = 10
_N_GEOS_NATIONAL = 1
_N_MEDIA_CHANNELS = 6
_N_RF_CHANNELS = 4
_N_ORGANIC_MEDIA_CHANNELS = 4
_N_ORGANIC_RF_CHANNELS = 1
_N_NON_MEDIA_CHANNELS = 2
_N_CONTROLS = 3
_N_KNOTS = 5


_INDEPENDENT_TEST_CASES = dict(
    name=[
        'shifted_uniform',
        'ragged_shifted_uniform',
        'differing',
        'ragged_differing',
    ],
    distributions=[
        (
            backend.tfd.Uniform(0.0, 1.0),
            backend.tfd.Uniform(1.0, 2.0),
            backend.tfd.Uniform(2.0, 3.0),
        ),
        (
            backend.tfd.Uniform([0.0, 1.0], [1.0, 2.0]),
            backend.tfd.Uniform(2.0, 3.0),
        ),
        (
            backend.tfd.HalfNormal(1),
            backend.tfd.Gamma(2, 2),
            backend.tfd.TruncatedNormal(0, 1, 1, 2),
        ),
        (
            backend.tfd.HalfNormal(1),
            backend.tfd.TruncatedNormal([0, 0], [1, 1], [1, 2], [2, 3]),
        ),
    ],
    expected_distribution_batch_shapes=[[1, 1, 1], [2, 1], [1, 1, 1], [1, 2]],
    expected_quantile_0=[
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 2.0),
    ],
    expected_quantile_1=[
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        (np.inf, np.inf, 2.0),
        (np.inf, 2.0, 3.0),
    ],
    expected_log_prob=[
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (0.0, 0.0, 0.0),
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
        ),
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (0.0, 0.0, 0.0),
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
        ),
        (
            (-np.inf, np.nan, -np.inf),
            (-np.inf, -0.306852818, -0.048140287),
            (-0.350791335, -1.208240509, -np.inf),
            (-1.350791335, -2.697414875, -np.inf),
            (-50.225791931, -16.31111908, -np.inf),
        ),
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (-0.350791335, -0.048140287, -0.199585199),
            (-1.350791335, -np.inf, -np.inf),
            (-50.225791931, -np.inf, -np.inf),
        ),
    ],
    expected_log_cdf=[
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (-np.log(2.0), -np.log(2.0), -np.log(2.0)),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (-np.log(2.0), -np.log(2.0), -np.log(2.0)),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -1.330893278, -0.391821384),
            (-0.959916353, -0.222079486, 0.0),
            (-0.143425226, -0.0412676, 0.0),
            (0.0, 0.0, 0.0),
        ),
        (
            (-np.inf, -np.inf, -np.inf),
            (-np.inf, -np.inf, -np.inf),
            (-0.959916353, -0.391821623, -0.257591963),
            (-0.143425226, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    ],
    expected_mean=[
        (0.5, 1.5, 2.5),
        (0.5, 1.5, 2.5),
        (np.sqrt(2 / np.pi), 1.0, 1.383169293),
        (np.sqrt(2 / np.pi), 1.383169293, 2.315821171),
    ],
    expected_variance=[
        (1 / 12, 1 / 12, 1 / 12),
        (1 / 12, 1 / 12, 1 / 12),
        (1 - 2 / np.pi, 1 / 2, 0.072742462),
        (1 - 2 / np.pi, 0.072742462, 0.061521053),
    ],
)


class PriorDistributionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.sample_distributions = {
        c.KNOT_VALUES: backend.tfd.Normal(0.0, 5.0, name=c.KNOT_VALUES),
        c.TAU_G_EXCL_BASELINE: backend.tfd.Normal(
            0.0, 5.0, name=c.TAU_G_EXCL_BASELINE
        ),
        c.BETA_M: backend.tfd.HalfNormal(5.0, name=c.BETA_M),
        c.BETA_RF: backend.tfd.HalfNormal(5.0, name=c.BETA_RF),
        c.BETA_OM: backend.tfd.HalfNormal(5.0, name=c.BETA_OM),
        c.BETA_ORF: backend.tfd.HalfNormal(5.0, name=c.BETA_ORF),
        c.ETA_M: backend.tfd.HalfNormal(1.0, name=c.ETA_M),
        c.ETA_RF: backend.tfd.HalfNormal(1.0, name=c.ETA_RF),
        c.ETA_OM: backend.tfd.HalfNormal(1.0, name=c.ETA_OM),
        c.ETA_ORF: backend.tfd.HalfNormal(1.0, name=c.ETA_ORF),
        c.GAMMA_C: backend.tfd.Normal(0.0, 5.0, name=c.GAMMA_C),
        c.GAMMA_N: backend.tfd.Normal(0.0, 5.0, name=c.GAMMA_N),
        c.XI_C: backend.tfd.HalfNormal(5.0, name=c.XI_C),
        c.XI_N: backend.tfd.HalfNormal(5.0, name=c.XI_N),
        c.ALPHA_M: backend.tfd.Uniform(0.0, 1.0, name=c.ALPHA_M),
        c.ALPHA_RF: backend.tfd.Uniform(0.0, 1.0, name=c.ALPHA_RF),
        c.ALPHA_OM: backend.tfd.Uniform(0.0, 1.0, name=c.ALPHA_OM),
        c.ALPHA_ORF: backend.tfd.Uniform(0.0, 1.0, name=c.ALPHA_ORF),
        c.EC_M: backend.tfd.TruncatedNormal(0.8, 0.8, 0.1, 10, name=c.EC_M),
        c.EC_RF: backend.tfd.TransformedDistribution(
            backend.tfd.LogNormal(0.7, 0.4),
            backend.bijectors.Shift(0.1),
            name=c.EC_RF,
        ),
        c.EC_OM: backend.tfd.TruncatedNormal(0.8, 0.8, 0.1, 10, name=c.EC_OM),
        c.EC_ORF: backend.tfd.TransformedDistribution(
            backend.tfd.LogNormal(0.7, 0.4),
            backend.bijectors.Shift(0.1),
            name=c.EC_ORF,
        ),
        c.SLOPE_M: backend.tfd.Deterministic(1.0, name=c.SLOPE_M),
        c.SLOPE_RF: backend.tfd.LogNormal(0.7, 0.4, name=c.SLOPE_RF),
        c.SLOPE_OM: backend.tfd.Deterministic(1.0, name=c.SLOPE_OM),
        c.SLOPE_ORF: backend.tfd.LogNormal(0.7, 0.4, name=c.SLOPE_ORF),
        c.SIGMA: backend.tfd.HalfNormal(5.0, name=c.SIGMA),
        c.ROI_M: backend.tfd.LogNormal(0.2, 0.9, name=c.ROI_M),
        c.ROI_RF: backend.tfd.LogNormal(0.2, 0.9, name=c.ROI_RF),
        c.MROI_M: backend.tfd.LogNormal(0.0, 0.5, name=c.MROI_M),
        c.MROI_RF: backend.tfd.LogNormal(0.0, 0.5, name=c.MROI_RF),
    }
    self.sample_broadcast = prior_distribution.PriorDistribution().broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        unique_sigma_for_each_geo=True,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
    )

  def assert_distribution_params_are_equal(
      self,
      a: backend.tfd.Distribution | MutableMapping[str, Any],
      b: backend.tfd.Distribution | MutableMapping[str, Any],
  ):
    """Assert that two distributions are equal."""
    a_params = (
        a.parameters.copy()
        if isinstance(a, backend.tfd.Distribution)
        else copy.deepcopy(a)
    )
    b_params = (
        b.parameters.copy()
        if isinstance(b, backend.tfd.Distribution)
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
        distribution.alpha_om, self.sample_distributions[c.ALPHA_OM]
    )
    self.assert_distribution_params_are_equal(
        distribution.alpha_orf, self.sample_distributions[c.ALPHA_ORF]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_m, self.sample_distributions[c.EC_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_rf, self.sample_distributions[c.EC_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_om, self.sample_distributions[c.EC_OM]
    )
    self.assert_distribution_params_are_equal(
        distribution.ec_orf, self.sample_distributions[c.EC_ORF]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_m, self.sample_distributions[c.BETA_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_rf, self.sample_distributions[c.BETA_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_om, self.sample_distributions[c.BETA_OM]
    )
    self.assert_distribution_params_are_equal(
        distribution.beta_orf, self.sample_distributions[c.BETA_ORF]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_m, self.sample_distributions[c.ETA_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_rf, self.sample_distributions[c.ETA_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_om, self.sample_distributions[c.ETA_OM]
    )
    self.assert_distribution_params_are_equal(
        distribution.eta_orf, self.sample_distributions[c.ETA_ORF]
    )
    self.assert_distribution_params_are_equal(
        distribution.gamma_c, self.sample_distributions[c.GAMMA_C]
    )
    self.assert_distribution_params_are_equal(
        distribution.gamma_n, self.sample_distributions[c.GAMMA_N]
    )
    self.assert_distribution_params_are_equal(
        distribution.xi_c, self.sample_distributions[c.XI_C]
    )
    self.assert_distribution_params_are_equal(
        distribution.xi_n, self.sample_distributions[c.XI_N]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_m, self.sample_distributions[c.SLOPE_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_rf, self.sample_distributions[c.SLOPE_RF]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_om, self.sample_distributions[c.SLOPE_OM]
    )
    self.assert_distribution_params_are_equal(
        distribution.slope_orf, self.sample_distributions[c.SLOPE_ORF]
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
    self.assert_distribution_params_are_equal(
        distribution.mroi_m, self.sample_distributions[c.MROI_M]
    )
    self.assert_distribution_params_are_equal(
        distribution.mroi_rf, self.sample_distributions[c.MROI_RF]
    )

  def test_has_deterministic_param_broadcasted_distribution_correct(self):
    for d in self.sample_distributions:
      is_deterministic_param = d in (c.SLOPE_M, c.SLOPE_OM)
      self.assertEqual(
          self.sample_broadcast.has_deterministic_param(d),
          is_deterministic_param,
      )

  def test_broadcast_preserves_distribution(self):
    distribution = prior_distribution.PriorDistribution()
    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        unique_sigma_for_each_geo=True,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
    )

    scalar_distributions_list = [
        distribution.knot_values,
        distribution.tau_g_excl_baseline,
        distribution.alpha_m,
        distribution.alpha_rf,
        distribution.alpha_om,
        distribution.alpha_orf,
        distribution.ec_m,
        distribution.ec_rf,
        distribution.ec_om,
        distribution.ec_orf,
        distribution.beta_m,
        distribution.beta_rf,
        distribution.beta_om,
        distribution.beta_orf,
        distribution.eta_m,
        distribution.eta_rf,
        distribution.eta_om,
        distribution.eta_orf,
        distribution.gamma_c,
        distribution.gamma_n,
        distribution.xi_c,
        distribution.xi_n,
        distribution.slope_m,
        distribution.slope_rf,
        distribution.slope_om,
        distribution.slope_orf,
        distribution.sigma,
        distribution.roi_m,
        distribution.roi_rf,
        distribution.mroi_m,
        distribution.mroi_rf,
    ]

    broadcast_distributions_list = [
        broadcast_distribution.knot_values.parameters[c.DISTRIBUTION],
        broadcast_distribution.tau_g_excl_baseline.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.gamma_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.gamma_n.parameters[c.DISTRIBUTION],
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.xi_n.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.sigma.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.mroi_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.mroi_rf.parameters[c.DISTRIBUTION],
    ]

    # Compare Distributions.
    for scal, broad in zip(
        scalar_distributions_list, broadcast_distributions_list
    ):
      self.assert_distribution_params_are_equal(scal, broad)

  def test_broadcast_no_controls_preserves_distribution(self):
    distribution = prior_distribution.PriorDistribution()
    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        n_controls=0,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        unique_sigma_for_each_geo=True,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
    )

    scalar_distributions_list = [
        distribution.knot_values,
        distribution.tau_g_excl_baseline,
        distribution.alpha_m,
        distribution.alpha_rf,
        distribution.alpha_om,
        distribution.alpha_orf,
        distribution.ec_m,
        distribution.ec_rf,
        distribution.ec_om,
        distribution.ec_orf,
        distribution.beta_m,
        distribution.beta_rf,
        distribution.beta_om,
        distribution.beta_orf,
        distribution.eta_m,
        distribution.eta_rf,
        distribution.eta_om,
        distribution.eta_orf,
        distribution.gamma_c,
        distribution.gamma_n,
        distribution.xi_c,
        distribution.xi_n,
        distribution.slope_m,
        distribution.slope_rf,
        distribution.slope_om,
        distribution.slope_orf,
        distribution.sigma,
        distribution.roi_m,
        distribution.roi_rf,
        distribution.mroi_m,
        distribution.mroi_rf,
    ]

    broadcast_distributions_list = [
        broadcast_distribution.knot_values.parameters[c.DISTRIBUTION],
        broadcast_distribution.tau_g_excl_baseline.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.alpha_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.ec_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.beta_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.eta_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.gamma_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.gamma_n.parameters[c.DISTRIBUTION],
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION],
        broadcast_distribution.xi_n.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_om.parameters[c.DISTRIBUTION],
        broadcast_distribution.slope_orf.parameters[c.DISTRIBUTION],
        broadcast_distribution.sigma.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.roi_rf.parameters[c.DISTRIBUTION],
        broadcast_distribution.mroi_m.parameters[c.DISTRIBUTION],
        broadcast_distribution.mroi_rf.parameters[c.DISTRIBUTION],
    ]

    # Compare Distributions.
    for scal, broad in zip(
        scalar_distributions_list, broadcast_distributions_list
    ):
      self.assert_distribution_params_are_equal(scal, broad)

  @parameterized.named_parameters(('one_sigma', False), ('unique_sigmas', True))
  def test_broadcast_preserves_shape(self, unique_sigma_for_each_geo: bool):
    # Create prior distribution with beta_m broadcasted to n_media_channels and
    # other parameters as scalars.
    distribution = prior_distribution.PriorDistribution(
        beta_m=backend.tfd.BatchBroadcast(
            backend.tfd.HalfNormal(5.0), _N_MEDIA_CHANNELS, name=c.BETA_M
        )
    )

    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        unique_sigma_for_each_geo=unique_sigma_for_each_geo,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
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
        broadcast_distribution.ec_m,
        broadcast_distribution.alpha_m,
        broadcast_distribution.slope_m,
        broadcast_distribution.roi_m,
        broadcast_distribution.mroi_m,
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
        broadcast_distribution.mroi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_RF_CHANNELS,))

    # Validate `n_organic_media_channels` shape distributions.
    n_organic_media_channels_distribution_list = [
        broadcast_distribution.beta_om,
        broadcast_distribution.eta_om,
        broadcast_distribution.alpha_om,
        broadcast_distribution.ec_om,
        broadcast_distribution.slope_om,
    ]
    for broad in n_organic_media_channels_distribution_list:
      self.assertEqual(broad.batch_shape, (_N_ORGANIC_MEDIA_CHANNELS,))

    # Validate `n_organic_rf_channels` shape distributions.
    n_organic_rf_channels_distribution_list = [
        broadcast_distribution.beta_orf,
        broadcast_distribution.eta_orf,
        broadcast_distribution.alpha_orf,
        broadcast_distribution.ec_orf,
        broadcast_distribution.slope_orf,
    ]
    for broad in n_organic_rf_channels_distribution_list:
      self.assertEqual(broad.batch_shape, (_N_ORGANIC_RF_CHANNELS,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        broadcast_distribution.gamma_c,
        broadcast_distribution.xi_c,
    ]

    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_CONTROLS,))

    # Validate `n_non_media_channels` shape distributions.
    n_non_media_distributions_list = [
        broadcast_distribution.gamma_n,
        broadcast_distribution.xi_n,
    ]

    for broad in n_non_media_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_NON_MEDIA_CHANNELS,))

    # Validate sigma.
    sigma_shape = (_N_GEOS,) if unique_sigma_for_each_geo else ()
    self.assertEqual(broadcast_distribution.sigma.batch_shape, sigma_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='scalar_deterministic',
          slope_m=backend.tfd.Deterministic(0.7, 0.4, name=c.SLOPE_M),
      ),
      dict(
          testcase_name='scalar_non_deterministic',
          slope_m=backend.tfd.LogNormal(1.0, 0.4, name=c.SLOPE_M),
      ),
      dict(
          testcase_name='list_deterministic',
          slope_m=backend.tfd.Deterministic(
              [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 0.9, name=c.SLOPE_M
          ),
      ),
      dict(
          testcase_name='list_non_deterministic',
          slope_m=backend.tfd.LogNormal(
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.9, name=c.SLOPE_M
          ),
      ),
  )
  def test_broadcast_custom_slope_m_raises_warning(
      self, slope_m: prior_distribution.PriorDistribution
  ):
    distribution = prior_distribution.PriorDistribution(slope_m=slope_m)
    with warnings.catch_warnings(record=True) as warns:
      # Cause all warnings to always be triggered.
      warnings.simplefilter('always')
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=0,
          n_organic_rf_channels=0,
          n_controls=_N_CONTROLS,
          n_non_media_channels=0,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
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
          testcase_name='roi_m',
          distribution=prior_distribution.PriorDistribution(
              roi_m=backend.tfd.LogNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ROI_M
              )
          ),
      ),
      dict(
          testcase_name='alpha_m',
          distribution=prior_distribution.PriorDistribution(
              alpha_m=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.ALPHA_M
              )
          ),
      ),
      dict(
          testcase_name='ec_m',
          distribution=prior_distribution.PriorDistribution(
              ec_m=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.EC_M
              )
          ),
      ),
      dict(
          testcase_name='slope_m',
          distribution=prior_distribution.PriorDistribution(
              slope_m=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.SLOPE_M
              )
          ),
      ),
      dict(
          testcase_name='eta_m',
          distribution=prior_distribution.PriorDistribution(
              eta_m=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ETA_M
              )
          ),
      ),
      dict(
          testcase_name='beta_m',
          distribution=prior_distribution.PriorDistribution(
              beta_m=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.BETA_M
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_media_channels(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the  number of media channels '
        "(6), representing a a custom prior for each channel. If you can't "
        'determine a custom prior, consider using the default prior for that '
        'channel.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=0,
          n_organic_rf_channels=0,
          n_controls=_N_CONTROLS,
          n_non_media_channels=0,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='roi_rf',
          distribution=prior_distribution.PriorDistribution(
              roi_rf=backend.tfd.LogNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ROI_RF
              )
          ),
      ),
      dict(
          testcase_name='alpha_rf',
          distribution=prior_distribution.PriorDistribution(
              alpha_rf=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.ALPHA_RF
              )
          ),
      ),
      dict(
          testcase_name='ec_rf',
          distribution=prior_distribution.PriorDistribution(
              ec_rf=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.EC_RF
              )
          ),
      ),
      dict(
          testcase_name='slope_rf',
          distribution=prior_distribution.PriorDistribution(
              slope_rf=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.SLOPE_RF
              )
          ),
      ),
      dict(
          testcase_name='eta_rf',
          distribution=prior_distribution.PriorDistribution(
              eta_rf=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ETA_RF
              )
          ),
      ),
      dict(
          testcase_name='beta_rf',
          distribution=prior_distribution.PriorDistribution(
              beta_rf=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.BETA_RF
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_rf_channels(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the number of RF channels (4), '
        "representing a custom prior for each channel. If you can't determine "
        'a custom prior, consider using the default prior for that channel.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=0,
          n_organic_rf_channels=0,
          n_controls=_N_CONTROLS,
          n_non_media_channels=0,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='alpha_om',
          distribution=prior_distribution.PriorDistribution(
              alpha_om=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.ALPHA_OM
              )
          ),
      ),
      dict(
          testcase_name='ec_om',
          distribution=prior_distribution.PriorDistribution(
              ec_om=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.EC_OM
              )
          ),
      ),
      dict(
          testcase_name='slope_om',
          distribution=prior_distribution.PriorDistribution(
              slope_om=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.SLOPE_OM
              )
          ),
      ),
      dict(
          testcase_name='eta_om',
          distribution=prior_distribution.PriorDistribution(
              eta_om=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ETA_OM
              )
          ),
      ),
      dict(
          testcase_name='beta_om',
          distribution=prior_distribution.PriorDistribution(
              beta_om=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.BETA_OM
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_organic_media_channels(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the  number of organic media '
        'channels (4), representing a custom prior for each channel. If you '
        "can't determine a custom prior, consider using the default prior for "
        'that channel.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='alpha_orf',
          distribution=prior_distribution.PriorDistribution(
              alpha_orf=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.ALPHA_ORF
              )
          ),
      ),
      dict(
          testcase_name='ec_orf',
          distribution=prior_distribution.PriorDistribution(
              ec_orf=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.EC_ORF
              )
          ),
      ),
      dict(
          testcase_name='slope_orf',
          distribution=prior_distribution.PriorDistribution(
              slope_orf=backend.tfd.Deterministic(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.SLOPE_ORF
              )
          ),
      ),
      dict(
          testcase_name='eta_orf',
          distribution=prior_distribution.PriorDistribution(
              eta_orf=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.ETA_ORF
              )
          ),
      ),
      dict(
          testcase_name='beta_orf',
          distribution=prior_distribution.PriorDistribution(
              beta_orf=backend.tfd.HalfNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.BETA_ORF
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_organic_rf_channels(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the number of organic RF channels '
        "(1), representing a custom prior for each channel. If you can't "
        'determine a custom prior, consider using the default prior for that '
        'channel.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='gamma_c',
          distribution=prior_distribution.PriorDistribution(
              gamma_c=backend.tfd.LogNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.GAMMA_C
              )
          ),
      ),
      dict(
          testcase_name='xi_c',
          distribution=prior_distribution.PriorDistribution(
              xi_c=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.XI_C
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_controls(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the number of control variables '
        '(3), representing a custom prior for each control variable. If you '
        "can't determine a custom prior, consider using the default prior for "
        'that variable.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=0,
          n_organic_rf_channels=0,
          n_controls=_N_CONTROLS,
          n_non_media_channels=0,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='gamma_n',
          distribution=prior_distribution.PriorDistribution(
              gamma_n=backend.tfd.LogNormal(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 0.9, name=c.GAMMA_N
              )
          ),
      ),
      dict(
          testcase_name='xi_n',
          distribution=prior_distribution.PriorDistribution(
              xi_n=backend.tfd.Uniform(
                  [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, name=c.XI_N
              )
          ),
      ),
  )
  def test_custom_priors_dont_match_non_media(
      self, distribution: prior_distribution.PriorDistribution
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors length (5) must match the number of non-media channels '
        "(2), representing a custom prior for each channel. If you can't "
        'determine a custom prior, consider using the default prior for that '
        'channel.',
    ):
      distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=False,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_deteremenistic_0',
          tau_g_excl_baseline=backend.tfd.Deterministic(
              0, name='tau_g_excl_baseline'
          ),
          eta_m=backend.tfd.Deterministic(0, name=c.ETA_M),
          eta_rf=backend.tfd.Deterministic(0, name=c.ETA_RF),
          eta_om=backend.tfd.Deterministic(0, name=c.ETA_OM),
          eta_orf=backend.tfd.Deterministic(0, name=c.ETA_ORF),
          xi_c=backend.tfd.Deterministic(0, name=c.XI_C),
          xi_n=backend.tfd.Deterministic(0, name=c.XI_N),
          number_of_warnings=0,
      ),
      dict(
          testcase_name='with_deteremenistic_1',
          tau_g_excl_baseline=backend.tfd.Deterministic(
              1, name='tau_g_excl_baseline'
          ),
          eta_m=backend.tfd.Deterministic(1, name=c.ETA_M),
          eta_rf=backend.tfd.Deterministic(1, name=c.ETA_RF),
          eta_om=backend.tfd.Deterministic(1, name=c.ETA_OM),
          eta_orf=backend.tfd.Deterministic(1, name=c.ETA_ORF),
          xi_c=backend.tfd.Deterministic(1, name=c.XI_C),
          xi_n=backend.tfd.Deterministic(1, name=c.XI_N),
          number_of_warnings=7,
      ),
      dict(
          testcase_name='with_non_deteremenistic_defaults',
          tau_g_excl_baseline=None,
          eta_m=None,
          eta_rf=None,
          eta_om=None,
          eta_orf=None,
          xi_c=None,
          xi_n=None,
          number_of_warnings=7,
      ),
  )
  def test_broadcast_national_distribution(
      self,
      tau_g_excl_baseline: backend.tfd.Distribution,
      eta_m: backend.tfd.Distribution,
      eta_rf: backend.tfd.Distribution,
      eta_om: backend.tfd.Distribution,
      eta_orf: backend.tfd.Distribution,
      xi_c: backend.tfd.Distribution,
      xi_n: backend.tfd.Distribution,
      number_of_warnings: int,
  ):
    tau_g_excl_baseline = (
        self.sample_distributions['tau_g_excl_baseline']
        if not tau_g_excl_baseline
        else tau_g_excl_baseline
    )
    eta_m = self.sample_distributions[c.ETA_M] if not eta_m else eta_m
    eta_rf = self.sample_distributions[c.ETA_RF] if not eta_rf else eta_rf
    eta_om = self.sample_distributions[c.ETA_OM] if not eta_om else eta_om
    eta_orf = self.sample_distributions[c.ETA_ORF] if not eta_orf else eta_orf
    xi_c = self.sample_distributions[c.XI_C] if not xi_c else xi_c
    xi_n = self.sample_distributions[c.XI_N] if not xi_n else xi_n

    # Create prior distribution with given parameters.
    distribution = prior_distribution.PriorDistribution(
        tau_g_excl_baseline=tau_g_excl_baseline,
        eta_m=eta_m,
        eta_rf=eta_rf,
        eta_om=eta_om,
        eta_orf=eta_orf,
        xi_c=xi_c,
        xi_n=xi_n,
    )

    with warnings.catch_warnings(record=True) as warns:
      # Cause all warnings to always be triggered.
      warnings.simplefilter('always')
      broadcast_distribution = distribution.broadcast(
          n_geos=_N_GEOS_NATIONAL,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          unique_sigma_for_each_geo=True,
          n_knots=_N_KNOTS,
          is_national=True,
          set_total_media_contribution_prior=False,
          kpi=1.0,
          total_spend=np.array([]),
      )
      self.assertLen(warns, number_of_warnings)
      for w in warns:
        self.assertTrue(issubclass(w.category, UserWarning))
        self.assertIn(
            'Hierarchical distribution parameters must be deterministically'
            ' zero for national models.',
            str(w.message),
        )

    # Validate Deterministic(0) distributions.
    self.assertIsInstance(
        broadcast_distribution.tau_g_excl_baseline.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.tau_g_excl_baseline.parameters[
            c.DISTRIBUTION
        ].loc,
        0,
    )
    self.assertIsInstance(
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_m.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_rf.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.eta_om.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_om.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.eta_orf.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.eta_orf.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.xi_c.parameters[c.DISTRIBUTION].loc, 0
    )
    self.assertIsInstance(
        broadcast_distribution.xi_n.parameters[c.DISTRIBUTION],
        backend.tfd.Deterministic,
    )
    self.assertEqual(
        broadcast_distribution.xi_n.parameters[c.DISTRIBUTION].loc, 0
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
        broadcast_distribution.mroi_m,
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
        broadcast_distribution.mroi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_RF_CHANNELS,))

    # Validate `n_organic_media_channels` shape distributions.
    n_organic_media_channels_distributions_list = [
        broadcast_distribution.beta_om,
        broadcast_distribution.alpha_om,
        broadcast_distribution.ec_om,
        broadcast_distribution.eta_om,
        broadcast_distribution.slope_om,
    ]
    for broad in n_organic_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_ORGANIC_MEDIA_CHANNELS,))

    # Validate `n_organic_rf_channels` shape distributions.
    n_organic_rf_channels_distributions_list = [
        broadcast_distribution.beta_orf,
        broadcast_distribution.alpha_orf,
        broadcast_distribution.ec_orf,
        broadcast_distribution.eta_orf,
        broadcast_distribution.slope_orf,
    ]
    for broad in n_organic_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (_N_ORGANIC_RF_CHANNELS,))

    # Validate `n_controls` shape distributions.
    self.assertEqual(broadcast_distribution.gamma_c.batch_shape, (_N_CONTROLS,))
    self.assertEqual(broadcast_distribution.xi_c.batch_shape, (_N_CONTROLS,))

    # Validate `n_non_media_channels` shape distributions.
    self.assertEqual(
        broadcast_distribution.gamma_n.batch_shape, (_N_NON_MEDIA_CHANNELS,)
    )
    self.assertEqual(
        broadcast_distribution.xi_n.batch_shape, (_N_NON_MEDIA_CHANNELS,)
    )

    # Validate sigma -- should be a scalar batch since n_geos is 1 for national.
    self.assertEqual(broadcast_distribution.sigma.batch_shape, ())

  def test_correct_total_media_contribution_roi_prior_distribution(
      self,
  ):
    roi_m_dist = self.sample_distributions[c.ROI_M]
    roi_rf_dist = self.sample_distributions[c.ROI_RF]

    # Create prior distribution with given parameters.
    distribution = prior_distribution.PriorDistribution(
        roi_m=roi_m_dist,
        roi_rf=roi_rf_dist,
    )

    kpi = 1.0
    total_spend = np.ones(_N_MEDIA_CHANNELS + _N_RF_CHANNELS)
    broadcast_distribution = distribution.broadcast(
        n_geos=_N_GEOS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_media_channels=0,
        n_organic_rf_channels=0,
        n_controls=_N_CONTROLS,
        n_non_media_channels=0,
        unique_sigma_for_each_geo=True,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=True,
        kpi=kpi,
        total_spend=total_spend,
    )

    self.assertEqual(
        broadcast_distribution.roi_m.batch_shape, (_N_MEDIA_CHANNELS,)
    )
    self.assertIsInstance(
        broadcast_distribution.roi_m.parameters[c.DISTRIBUTION],
        backend.tfd.LogNormal,
    )
    expected_roi_m = prior_distribution._get_total_media_contribution_prior(
        kpi=kpi,
        total_spend=total_spend,
        name=c.ROI_M,
    )
    self.assert_distribution_params_are_equal(
        broadcast_distribution.roi_m.distribution, expected_roi_m
    )
    self.assertEqual(
        broadcast_distribution.roi_rf.batch_shape, (_N_RF_CHANNELS,)
    )
    self.assertIsInstance(
        broadcast_distribution.roi_m.parameters[c.DISTRIBUTION],
        backend.tfd.LogNormal,
    )
    expected_roi_rf = prior_distribution._get_total_media_contribution_prior(
        kpi=kpi,
        total_spend=total_spend,
        name=c.ROI_RF,
    )
    self.assert_distribution_params_are_equal(
        broadcast_distribution.roi_rf.distribution, expected_roi_rf
    )

  @parameterized.named_parameters(
      (c.KNOT_VALUES, c.KNOT_VALUES),
      (c.TAU_G_EXCL_BASELINE, c.TAU_G_EXCL_BASELINE),
      (c.BETA_M, c.BETA_M),
      (c.BETA_RF, c.BETA_RF),
      (c.BETA_OM, c.BETA_OM),
      (c.BETA_ORF, c.BETA_ORF),
      (c.ETA_M, c.ETA_M),
      (c.ETA_RF, c.ETA_RF),
      (c.ETA_OM, c.ETA_OM),
      (c.ETA_ORF, c.ETA_ORF),
      (c.GAMMA_C, c.GAMMA_C),
      (c.GAMMA_N, c.GAMMA_N),
      (c.XI_C, c.XI_C),
      (c.XI_N, c.XI_N),
      (c.ALPHA_M, c.ALPHA_M),
      (c.ALPHA_RF, c.ALPHA_RF),
      (c.ALPHA_OM, c.ALPHA_OM),
      (c.ALPHA_ORF, c.ALPHA_ORF),
      (c.EC_M, c.EC_M),
      (c.EC_RF, c.EC_RF),
      (c.EC_OM, c.EC_OM),
      (c.EC_ORF, c.EC_ORF),
      (c.SLOPE_M, c.SLOPE_M),
      (c.SLOPE_RF, c.SLOPE_RF),
      (c.SLOPE_OM, c.SLOPE_OM),
      (c.SLOPE_ORF, c.SLOPE_ORF),
      (c.SIGMA, c.SIGMA),
      (c.ROI_M, c.ROI_M),
      (c.ROI_RF, c.ROI_RF),
      (c.MROI_M, c.MROI_M),
      (c.MROI_RF, c.MROI_RF),
  )
  def test_getstate_correct(self, attribute):
    def _distribution_info(
        distribution: backend.tfd.Distribution,
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
        c.BETA_OM: _distribution_info(self.sample_broadcast.beta_om),
        c.BETA_ORF: _distribution_info(self.sample_broadcast.beta_orf),
        c.ETA_M: _distribution_info(self.sample_broadcast.eta_m),
        c.ETA_RF: _distribution_info(self.sample_broadcast.eta_rf),
        c.ETA_OM: _distribution_info(self.sample_broadcast.eta_om),
        c.ETA_ORF: _distribution_info(self.sample_broadcast.eta_orf),
        c.GAMMA_C: _distribution_info(self.sample_broadcast.gamma_c),
        c.GAMMA_N: _distribution_info(self.sample_broadcast.gamma_n),
        c.XI_C: _distribution_info(self.sample_broadcast.xi_c),
        c.XI_N: _distribution_info(self.sample_broadcast.xi_n),
        c.ALPHA_M: _distribution_info(self.sample_broadcast.alpha_m),
        c.ALPHA_RF: _distribution_info(self.sample_broadcast.alpha_rf),
        c.ALPHA_OM: _distribution_info(self.sample_broadcast.alpha_om),
        c.ALPHA_ORF: _distribution_info(self.sample_broadcast.alpha_orf),
        c.EC_M: _distribution_info(self.sample_broadcast.ec_m),
        c.EC_RF: _distribution_info(self.sample_broadcast.ec_rf),
        c.EC_OM: _distribution_info(self.sample_broadcast.ec_om),
        c.EC_ORF: _distribution_info(self.sample_broadcast.ec_orf),
        c.SLOPE_M: _distribution_info(self.sample_broadcast.slope_m),
        c.SLOPE_RF: _distribution_info(self.sample_broadcast.slope_rf),
        c.SLOPE_OM: _distribution_info(self.sample_broadcast.slope_om),
        c.SLOPE_ORF: _distribution_info(self.sample_broadcast.slope_orf),
        c.SIGMA: _distribution_info(self.sample_broadcast.sigma),
        c.ROI_M: _distribution_info(self.sample_broadcast.roi_m),
        c.ROI_RF: _distribution_info(self.sample_broadcast.roi_rf),
        c.MROI_M: _distribution_info(self.sample_broadcast.mroi_m),
        c.MROI_RF: _distribution_info(self.sample_broadcast.mroi_rf),
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
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        n_controls=_N_CONTROLS,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        unique_sigma_for_each_geo=True,
        n_knots=_N_KNOTS,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
    )
    distribution.__setstate__(distribution.__getstate__())

    for k, v in distribution.__dict__.items():
      self.assert_distribution_params_are_equal(
          v, self.sample_broadcast.__dict__[k]
      )

  def test_get_total_media_contribution_prior(self):
    distribution = prior_distribution._get_total_media_contribution_prior(
        kpi=1.0,
        total_spend=np.array([1, 2, 3]),
        name='name',
    )
    expected_distribution = backend.tfd.LogNormal(
        -2.956268548965454, 0.7045827507972717, name='name'
    )

    self.assert_distribution_params_are_equal(
        distribution.parameters, expected_distribution.parameters
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='same',
          a=backend.tfd.Deterministic(0, name='name_1'),
          b=backend.tfd.Deterministic(0, name='name_1'),
          expected_result=True,
      ),
      dict(
          testcase_name='same_type_different_name',
          a=backend.tfd.Deterministic(1, name='name_1'),
          b=backend.tfd.Deterministic(1, name='name_2'),
          expected_result=False,
      ),
      dict(
          testcase_name='same_type_different_params',
          a=backend.tfd.LogNormal(0.7, 0.4, name='name_1'),
          b=backend.tfd.LogNormal(0.7, 0.6, name='name_1'),
          expected_result=False,
      ),
      dict(
          testcase_name='same_complex_distributions',
          a=backend.tfd.TransformedDistribution(
              backend.tfd.LogNormal(0.7, 0.4),
              backend.bijectors.Shift(0.1),
              name='name_1',
          ),
          b=backend.tfd.TransformedDistribution(
              backend.tfd.LogNormal(0.7, 0.4),
              backend.bijectors.Shift(0.1),
              name='name_1',
          ),
          expected_result=True,
      ),
      dict(
          testcase_name='different_outer_complex_distributions',
          a=backend.tfd.BatchBroadcast(backend.tfd.HalfNormal(5.0), 3),
          b=backend.tfd.BatchBroadcast(backend.tfd.HalfNormal(5.0), 7),
          expected_result=False,
      ),
      dict(
          testcase_name='different_inner_complex_distributions',
          a=backend.tfd.BatchBroadcast(backend.tfd.HalfNormal(5.0), 3),
          b=backend.tfd.BatchBroadcast(backend.tfd.Uniform(0.0, 1.0), 3),
          expected_result=False,
      ),
      dict(
          testcase_name='different_simple_and_complex_distributions',
          a=backend.tfd.HalfNormal(5.0),
          b=backend.tfd.BatchBroadcast(backend.tfd.HalfNormal(5.0), 3),
          expected_result=False,
      ),
      dict(
          testcase_name='different_complex_and_simple_distributions',
          a=backend.tfd.BatchBroadcast(backend.tfd.HalfNormal(5.0), 7),
          b=backend.tfd.HalfNormal(5.0),
          expected_result=False,
      ),
  )
  def test_distributions_are_equal(
      self,
      a: backend.tfd.Distribution,
      b: backend.tfd.Distribution,
      expected_result: bool,
  ):
    self.assertEqual(
        prior_distribution.distributions_are_equal(a, b), expected_result
    )

  @parameterized.named_parameters(
      (
          'alpha_m_can_be_negative',
          {'alpha_m': tfp.distributions.Uniform(-1, 1)},
      ),
      (
          'alpha_m_can_exceed_one',
          {'alpha_m': tfp.distributions.Uniform(0, 2)},
      ),
      (
          'alpha_m_deterministic_negative_one',
          {'alpha_m': tfp.distributions.Deterministic(-1)},
      ),
      (
          'alpha_m_deterministic_two',
          {'alpha_m': tfp.distributions.Deterministic(2)},
      ),
      (
          'eta_m_can_be_negative',
          {'eta_m': tfp.distributions.Normal(0, 1)},
      )
  )
  def test_validate_support_raises_value_error(
      self,
      prior_distribution_kwargs: dict[str, tfp.distributions.Distribution],
  ):
    with self.assertRaises(ValueError):
      prior_distribution.PriorDistribution(**prior_distribution_kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name='alpha_m_deterministic_prior_at_one',
          param_name='alpha_m',
          deterministic_value=1,
          expect_error=True,
      ),
      dict(
          testcase_name='ec_m_deterministic_prior_at_zero',
          param_name='ec_m',
          deterministic_value=0,
          expect_error=True,
      ),
      dict(
          testcase_name='slope_m_deterministic_prior_at_zero',
          param_name='slope_m',
          deterministic_value=0,
          expect_error=True,
      ),
      dict(
          testcase_name='alpha_m_deterministic_prior_at_zero',
          param_name='alpha_m',
          deterministic_value=0,
          expect_error=False,
      ),
      dict(
          testcase_name='eta_m_deterministic_prior_at_zero',
          param_name='eta_m',
          deterministic_value=0,
          expect_error=False,
      ),
  )
  def test_deterministic_prior_at_bound(
      self,
      param_name: str,
      deterministic_value: float,
      expect_error: bool,
  ):
    if expect_error:
      with self.assertRaisesRegex(
          ValueError,
          f'{param_name} was assigned a point mass',
      ):
        prior_distribution.PriorDistribution(
            **{param_name: tfp.distributions.Deterministic(deterministic_value)}
        )
    else:
      try:
        prior_distribution.PriorDistribution(
            **{param_name: tfp.distributions.Deterministic(deterministic_value)}
        )
      except ValueError:
        self.fail(
            f'Assigning Deterministic({deterministic_value}) prior to'
            f' {param_name} raises an unexpected ValueError.'
        )

  def test_quantile_not_implemented_raises_warning(self):
    with self.assertWarnsRegex(
        UserWarning,
        'The prior distribution for alpha_m does not have a `quantile` method'
    ):
      # Note that `tfp.distributions.Categorical.quantile` raises a
      # `NotImplementedError`.
      dist = tfp.distributions.Categorical(probs=[0.5, 0.5])
      prior_distribution.PriorDistribution(alpha_m=dist)


class TestIndependentMultivariateDistribution(parameterized.TestCase):

  def test_contains_deterministic_fail(self):
    distributions = [
        backend.tfd.Normal(0, 1),
        backend.tfd.Deterministic(3.0),
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'IndependentMultivariateDistribution cannot contain `Deterministic` '
        'distributions. To implement a nearly deterministic element of this '
        'distribution, we recommend using `backend.tfd.Uniform` with a '
        'small range. For example to define a distribution that is nearly '
        '`Deterministic(1.0)`, use '
        '`tfp.distribution.Uniform(1.0 - 1e-9, 1.0 + 1e-9)`',
    ):
      _ = prior_distribution.IndependentMultivariateDistribution(distributions)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
      )
  )
  def test_batch_shape_tensor(self, distributions):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )

    np.testing.assert_array_equal(
        distribution.batch_shape_tensor(), backend.to_tensor([3])
    )

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_distribution_batch_shapes'],
      )
  )
  def test_distribution_batch_shapes(
      self, distributions, expected_distribution_batch_shapes
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    self.assertEqual(
        distribution._distribution_batch_shapes,
        expected_distribution_batch_shapes,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='scalar_sample',
          sample_kwargs=dict(),
          expected_shape=(2,),
      ),
      dict(
          testcase_name='three_sample',
          sample_kwargs=dict(sample_shape=3),
          expected_shape=(3, 2),
      ),
      dict(
          testcase_name='batch_sample_2d',
          sample_kwargs=dict(sample_shape=[2, 3]),
          expected_shape=(2, 3, 2),
      ),
  )
  def test_independent_bivariate_distribution_sample_shape(
      self,
      sample_kwargs: dict[str, Any],
      expected_shape: tuple[int, ...],
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        [backend.tfd.Normal(-10, 1), backend.tfd.Normal(10, 1)]
    )
    sample = distribution.sample(**sample_kwargs)
    self.assertEqual(sample.shape, expected_shape)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_quantile_0'],
      )
  )
  def test_independent_distribution_support_lower_bound(
      self, distributions, expected_quantile_0
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )

    with self.subTest('0d'):
      lower_bound = distribution.quantile(0.0)
      np.testing.assert_allclose(lower_bound, expected_quantile_0, rtol=1e-5)

    with self.subTest('2d'):
      expected_2d = (expected_quantile_0, expected_quantile_0)
      lower_bound = distribution.quantile([[0.0], [0.0]])
      np.testing.assert_allclose(lower_bound, expected_2d, rtol=1e-5)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_quantile_1'],
      )
  )
  def test_independent_distribution_support_upper_bound(
      self, distributions, expected_quantile_1
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    with self.subTest('0d'):
      upper_bound = distribution.quantile(1.0)
      np.testing.assert_allclose(upper_bound, expected_quantile_1, rtol=1e-5)

    with self.subTest('2d'):
      expected_2d = (expected_quantile_1, expected_quantile_1)
      upper_bound = distribution.quantile([[1.0], [1.0]])
      np.testing.assert_allclose(upper_bound, expected_2d, rtol=1e-5)

  @parameterized.product(
      (
          dict(
              value=1.0,
              expected_broadcasted_value=backend.to_tensor([1.0, 1.0, 1.0]),
          ),
          dict(
              value=[1.0],
              expected_broadcasted_value=backend.to_tensor([1.0, 1.0, 1.0]),
          ),
          dict(
              value=[0.0, 0.5, 1.0],
              expected_broadcasted_value=backend.to_tensor([0.0, 0.5, 1.0]),
          ),
          dict(
              value=[[0.0], [1.0]],
              expected_broadcasted_value=backend.to_tensor([
                  [0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0],
              ]),
          ),
      ),
      distributions=_INDEPENDENT_TEST_CASES['distributions'],
  )
  def test_independent_distribution_broadcast_value(
      self,
      value,
      expected_broadcasted_value,
      distributions,
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    broadcasted_value = distribution._broadcast_value(value)
    np.testing.assert_array_equal(
        broadcasted_value.shape, expected_broadcasted_value.shape
    )

    np.testing.assert_array_equal(broadcasted_value, expected_broadcasted_value)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_log_prob'],
      )
  )
  def test_independent_distributions_log_prob(
      self, distributions, expected_log_prob
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )

    value = [
        [-1.0, -1.0, -1.0],
        [-0.5, 0.5, 1.5],
        [0.5, 1.5, 2.5],
        [1.5, 2.5, 3.5],
        [10, 10, 10],
    ]

    with self.subTest('2d'):
      log_prob = distribution.log_prob(value)
      np.testing.assert_allclose(log_prob, expected_log_prob, rtol=1e-5)

    with self.subTest('3d'):
      log_prob = distribution.log_prob([value])
      np.testing.assert_allclose(log_prob, (expected_log_prob,), rtol=1e-5)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_log_cdf'],
      )
  )
  def test_independent_distributions_log_cdf(
      self, distributions, expected_log_cdf
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    value = [
        [-1.0, -1.0, -1.0],
        [-0.5, 0.5, 1.5],
        [0.5, 1.5, 2.5],
        [1.5, 2.5, 3.5],
        [100.0, 100.0, 100.0],
    ]

    with self.subTest('2d'):
      log_cdf = distribution.log_cdf(value)
      np.testing.assert_allclose(log_cdf, expected_log_cdf, rtol=1e-5)

    with self.subTest('3d'):
      log_cdf = distribution.log_cdf([value])
      np.testing.assert_allclose(log_cdf, (expected_log_cdf,), rtol=1e-5)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_mean'],
      )
  )
  def test_independent_distributions_mean(self, distributions, expected_mean):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    mean = distribution.mean()
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5)

  @parameterized.named_parameters(
      *zip(
          _INDEPENDENT_TEST_CASES['name'],
          _INDEPENDENT_TEST_CASES['distributions'],
          _INDEPENDENT_TEST_CASES['expected_variance'],
      )
  )
  def test_independent_distributions_variance(
      self, distributions, expected_variance
  ):
    distribution = prior_distribution.IndependentMultivariateDistribution(
        distributions
    )
    variance = distribution.variance()
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
