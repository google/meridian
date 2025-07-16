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

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import posterior_sampler
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PosteriorMCMCSamplerTest(
    tf.test.TestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  def setUp(self):
    super().setUp()
    model_test_data.WithInputDataSamples.setup(self)

  def test_get_joint_dist_zeros(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Deterministic(0),
            tau_g_excl_baseline=tfp.distributions.Deterministic(0),
            beta_m=tfp.distributions.Deterministic(0),
            beta_rf=tfp.distributions.Deterministic(0),
            eta_m=tfp.distributions.Deterministic(0),
            eta_rf=tfp.distributions.Deterministic(0),
            gamma_c=tfp.distributions.Deterministic(0),
            xi_c=tfp.distributions.Deterministic(0),
            alpha_m=tfp.distributions.Deterministic(0),
            alpha_rf=tfp.distributions.Deterministic(0),
            ec_m=tfp.distributions.Deterministic(0),
            ec_rf=tfp.distributions.Deterministic(0),
            slope_m=tfp.distributions.Deterministic(0),
            slope_rf=tfp.distributions.Deterministic(0),
            sigma=tfp.distributions.Deterministic(0),
            roi_m=tfp.distributions.Deterministic(0),
            roi_rf=tfp.distributions.Deterministic(0),
        ),
        media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    sample = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(
            self._N_DRAWS
        )
    )
    self.assertAllEqual(
        sample.y,
        tf.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

  def test_get_joint_dist_zeros_no_controls_data(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Deterministic(0),
            tau_g_excl_baseline=tfp.distributions.Deterministic(0),
            beta_m=tfp.distributions.Deterministic(0),
            beta_rf=tfp.distributions.Deterministic(0),
            eta_m=tfp.distributions.Deterministic(0),
            eta_rf=tfp.distributions.Deterministic(0),
            gamma_c=tfp.distributions.Deterministic(0),
            xi_c=tfp.distributions.Deterministic(0),
            alpha_m=tfp.distributions.Deterministic(0),
            alpha_rf=tfp.distributions.Deterministic(0),
            ec_m=tfp.distributions.Deterministic(0),
            ec_rf=tfp.distributions.Deterministic(0),
            slope_m=tfp.distributions.Deterministic(0),
            slope_rf=tfp.distributions.Deterministic(0),
            sigma=tfp.distributions.Deterministic(0),
            roi_m=tfp.distributions.Deterministic(0),
            roi_rf=tfp.distributions.Deterministic(0),
        ),
        media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only_no_controls,
        model_spec=model_spec,
    )
    sample = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(
            self._N_DRAWS
        )
    )
    self.assertAllEqual(
        sample.y,
        tf.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

    # Without controls data, controls-related distributions should be absent.
    with self.assertRaises(AttributeError):
      _ = sample.gamma_gc
    with self.assertRaises(AttributeError):
      _ = sample.xi_c
    with self.assertRaises(AttributeError):
      _ = sample.gamma_gc_dev
    with self.assertRaises(AttributeError):
      _ = sample.gamma_gc

  @parameterized.product(
      paid_media_prior_type=[
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      ],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_only(
      self,
      paid_media_prior_type: str,
      media_effects_dist: str,
  ):
    model_spec = spec.ModelSpec(
        media_prior_type=paid_media_prior_type,
        rf_prior_type=paid_media_prior_type,
        media_effects_dist=media_effects_dist,
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_only,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(1)
    )
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) outcome
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) outcome data is "pinned" as "y".
    log_prob_parts_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist().log_prob_parts(
            par
        )
    )
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.GAMMA_GC,
        constants.MU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.EC_M,
        constants.SLOPE_M,
        constants.SIGMA,
    ]

    if paid_media_prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.ROI_M)
    elif paid_media_prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.MROI_M)
    else:
      prior_distribution_params.append(constants.BETA_M)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    beta_m = par[constants.BETA_GM][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.MU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_media, beta_m)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian.posterior_sampler_callable._get_joint_dist().log_prob(par)[0],
    )

  @parameterized.product(
      paid_media_prior_type=[
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      ],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_rf_only(
      self,
      paid_media_prior_type: str,
      media_effects_dist: str,
  ):
    model_spec = spec.ModelSpec(
        rf_prior_type=paid_media_prior_type,
        media_effects_dist=media_effects_dist,
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_rf_only,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(1)
    )
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) outcome
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) outcome data is "pinned" as "y".
    log_prob_parts_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist().log_prob_parts(
            par
        )
    )
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.MU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_RF,
        constants.EC_RF,
        constants.SLOPE_RF,
        constants.SIGMA,
    ]

    if paid_media_prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.ROI_RF)
    elif paid_media_prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.MROI_RF)
    else:
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.rf_tensors.reach_scaled,
        frequency=meridian.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    beta_rf = par[constants.BETA_GRF][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.MU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_reach, beta_rf)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian.posterior_sampler_callable._get_joint_dist().log_prob(par)[0],
    )

  # TODO: Add test for holdout_id.
  @parameterized.product(
      paid_media_prior_type=[
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      ],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_and_rf(
      self,
      paid_media_prior_type: str,
      media_effects_dist: str,
  ):
    model_spec = spec.ModelSpec(
        media_prior_type=paid_media_prior_type,
        rf_prior_type=paid_media_prior_type,
        media_effects_dist=media_effects_dist,
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_and_rf,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(1)
    )
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) outcome
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) outcome data is "pinned" as "y".
    log_prob_parts_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist().log_prob_parts(
            par
        )
    )
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.MU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.ALPHA_RF,
        constants.EC_M,
        constants.EC_RF,
        constants.SLOPE_M,
        constants.SLOPE_RF,
        constants.SIGMA,
    ]

    if paid_media_prior_type in constants.TREATMENT_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.ROI_M)
      prior_distribution_params.append(constants.ROI_RF)
    elif paid_media_prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.MROI_M)
      prior_distribution_params.append(constants.MROI_RF)
    else:
      prior_distribution_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.rf_tensors.reach_scaled,
        frequency=meridian.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    combined_transformed_media = tf.concat(
        [transformed_media, transformed_reach], axis=-1
    )

    combined_beta = tf.concat(
        [par[constants.BETA_GM][0, :, :], par[constants.BETA_GRF][0, :, :]],
        axis=-1,
    )
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.MU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", combined_transformed_media, combined_beta)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian.posterior_sampler_callable._get_joint_dist().log_prob(par)[0],
    )

  def test_sample_posterior_media_and_rf_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)
    rf_parameters = list(constants.RF_PARAMETER_NAMES)
    rf_parameters.remove(constants.BETA_GRF)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in media_parameters
        ],
        rf_channel_shape: [getattr(posterior, attr) for attr in rf_parameters],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in media_parameters
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_only_no_controls_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_only_no_controls,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only_no_controls,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )

    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )

    # Control parameters should not exist in the inference data posteriors.
    for param in (
        constants.CONTROL_PARAMETERS + constants.GEO_CONTROL_PARAMETERS
    ):
      with self.assertRaises(AttributeError):
        getattr(meridian.inference_data.posterior, param)

  def test_sample_posterior_rf_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_rf_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_rf_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    rf_parameters = list(constants.RF_PARAMETER_NAMES)
    rf_parameters.remove(constants.BETA_GRF)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        rf_channel_shape: [getattr(posterior, attr) for attr in rf_parameters],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_and_rf_sequential_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=[self._N_CHAINS, self._N_CHAINS],
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    n_total_chains = self._N_CHAINS * 2
    knots_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (n_total_chains, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (n_total_chains, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (n_total_chains, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (n_total_chains, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (n_total_chains, self._N_KEEP, 1)
    )
    geo_shape = (n_total_chains, self._N_KEEP, self._N_GEOS)
    time_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)
    rf_parameters = list(constants.RF_PARAMETER_NAMES)
    rf_parameters.remove(constants.BETA_GRF)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in media_parameters
        ],
        rf_channel_shape: [getattr(posterior, attr) for attr in rf_parameters],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_raises_oom_error_when_limits_exceeded(self):
    self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            side_effect=tf.errors.ResourceExhaustedError(
                None, None, "Resource exhausted"
            ),
        )
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )

    with self.assertRaises(model.MCMCOOMError):
      meridian.sample_posterior(
          n_chains=self._N_CHAINS,
          n_adapt=self._N_ADAPT,
          n_burnin=self._N_BURNIN,
          n_keep=self._N_KEEP,
      )

  def test_injected_sample_posterior_media_and_rf_returns_correct_shape(self):
    """Checks validation passes with correct shapes."""
    self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    inference_data = meridian.inference_data
    meridian_with_inference_data = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  def test_injected_sample_posterior_media_only_returns_correct_shape(self):
    """Checks validation passes with correct shapes."""
    self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    inference_data = meridian.inference_data
    meridian_with_inference_data = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="control_variables",
          coord=constants.CONTROL_VARIABLE,
          mismatched_posteriors={
              constants.GAMMA_C: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_CONTROLS + 1,
              ),
              constants.GAMMA_GC: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_CONTROLS + 1,
              ),
              constants.XI_C: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_CONTROLS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_CONTROLS + 1,
          expected_coord_size=input_data_samples._N_CONTROLS,
      ),
      dict(
          testcase_name="geos",
          coord=constants.GEO,
          mismatched_posteriors={
              constants.BETA_GM: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_MEDIA_CHANNELS,
              ),
              constants.BETA_GRF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_RF_CHANNELS,
              ),
              constants.GAMMA_GC: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_GEOS + 1,
          expected_coord_size=input_data_samples._N_GEOS,
      ),
      dict(
          testcase_name="knots",
          coord=constants.KNOTS,
          mismatched_posteriors={
              constants.KNOT_VALUES: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_TIMES_SHORT + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_TIMES_SHORT + 1,
          expected_coord_size=input_data_samples._N_TIMES_SHORT,
      ),
      dict(
          testcase_name="times",
          coord=constants.TIME,
          mismatched_posteriors={
              constants.MU_T: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_TIMES_SHORT + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_TIMES_SHORT + 1,
          expected_coord_size=input_data_samples._N_TIMES_SHORT,
      ),
      dict(
          testcase_name="media_channels",
          coord=constants.MEDIA_CHANNEL,
          mismatched_posteriors={
              constants.ALPHA_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.BETA_GM: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.BETA_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.EC_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.ETA_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.ROI_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.SLOPE_M: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_MEDIA_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_MEDIA_CHANNELS,
      ),
      dict(
          testcase_name="rf_channels",
          coord=constants.RF_CHANNEL,
          mismatched_posteriors={
              constants.ALPHA_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.BETA_GRF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.BETA_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.EC_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.ETA_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.ROI_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.SLOPE_RF: (
                  input_data_samples._N_CHAINS,
                  input_data_samples._N_KEEP,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_RF_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_RF_CHANNELS,
      ),
  )
  def test_validate_injected_inference_data_posterior_incorrect_coordinates(
      self,
      coord,
      mismatched_posteriors,
      mismatched_coord_size,
      expected_coord_size,
  ):
    """Checks posterior validation fails with incorrect coordinates."""
    self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )

    posterior_coords = meridian.create_inference_data_coords(
        self._N_CHAINS, self._N_KEEP
    )
    posterior_dims = meridian.create_inference_data_dims()
    posterior_samples = dict(meridian.inference_data.posterior)
    for posterior in mismatched_posteriors:
      posterior_samples[posterior] = tf.zeros(mismatched_posteriors[posterior])

    posterior_coords = dict(posterior_coords)
    posterior_coords[coord] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        posterior_samples,
        coords=posterior_coords,
        dims=posterior_dims,
        group=constants.POSTERIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        f"Injected inference data {constants.POSTERIOR} has incorrect"
        f" coordinate '{coord}': expected"
        f" {expected_coord_size}, got {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=self.short_input_data_with_media_and_rf,
          model_spec=model_spec,
          inference_data=inference_data,
      )

  @parameterized.named_parameters(
      dict(testcase_name="seed_is_none", seed=None),
      dict(testcase_name="seed_is_int", seed=42),
      dict(testcase_name="seed_is_pair", seed=[42, 123]),
  )
  def test_sample_posterior_with_seed(self, seed):
    if seed is not None:
      seed = tfp.random.sanitize_seed(seed)
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
        seed=seed,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=seed,
    )

  @parameterized.named_parameters(
      dict(testcase_name="seed_is_invalid_sequence", seed=[1, 2, 3]),
  )
  def test_sample_posterior_with_invalid_seed_sequence(self, seed):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid seed: Must be either a single integer (stateful seed) or a"
        " pair of two integers (stateless seed). See"
        " [tfp.random.sanitize_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed)"
        " for details.",
    ):
      model_spec = spec.ModelSpec()
      meridian = model.Meridian(
          input_data=self.short_input_data_with_media_and_rf,
          model_spec=model_spec,
      )
      meridian.sample_posterior(
          n_chains=self._N_CHAINS,
          n_adapt=self._N_ADAPT,
          n_burnin=self._N_BURNIN,
          n_keep=self._N_KEEP,
          seed=seed,
      )

  @parameterized.named_parameters(
      dict(testcase_name="seed_is_none", initial_seed=None),
      dict(
          testcase_name="seed_is_int",
          initial_seed=123,
      ),
      dict(
          testcase_name="seed_is_pair",
          initial_seed=[123, 456],
      ),
  )
  def test_sample_posterior_seed_increment(self, initial_seed):
    n_chains_list = [self._N_CHAINS, self._N_CHAINS]
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=n_chains_list,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
        seed=initial_seed,
    )

    calls = mock_sample_posterior.call_args_list
    self.assertLen(calls, len(n_chains_list))

    _, kwargs0 = calls[0]
    _, kwargs1 = calls[1]

    sanitized_seeds = []
    if initial_seed is None:
      sanitized_seeds.append(None)
      sanitized_seeds.append(None)
      self.assertIsNone(kwargs0["seed"])
      self.assertIsNone(kwargs1["seed"])
    else:
      sanitized_seed0 = kwargs0["seed"]
      sanitized_seed1 = kwargs1["seed"]
      self.assertAllEqual(sanitized_seed1, [x + 1 for x in sanitized_seed0])

  def test_sample_posterior_seed_int(self):
    n_chains_list = [self._N_CHAINS, self._N_CHAINS]
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            posterior_sampler,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=n_chains_list,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
        seed=123,
    )
    calls = mock_sample_posterior.call_args_list
    self.assertLen(calls, len(n_chains_list))

    _, kwargs0 = calls[0]
    _, kwargs1 = calls[1]

    self.assertAllEqual(kwargs0["seed"], [123, 123])
    self.assertAllEqual(kwargs1["seed"], [124, 124])


if __name__ == "__main__":
  absltest.main()
