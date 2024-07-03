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

"""Meridian module for the geo-level Bayesian hierarchical media mix model."""

from collections.abc import Mapping, Sequence
import os
import arviz as az
import joblib
from meridian import constants
from meridian.data import input_data as data
from meridian.model import adstock_hill
from meridian.model import model_data
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


__all__ = [
    "MCMCSamplingError",
    "MCMCOOMError",
    "Meridian",
    "NotFittedModelError",
]


class NotFittedModelError(Exception):
  """Model has not been fitted."""


class MCMCSamplingError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling failed."""


class MCMCOOMError(Exception):
  """The Markov Chain Monte Carlo (MCMC) exceeds memory limits."""


def _get_tau_g(
    tau_g_excl_baseline: tf.Tensor, baseline_geo_idx: int
) -> tfp.distributions.Distribution:
  """Computes `tau_g` from `tau_g_excl_baseline`.

  This function computes `tau_g` by inserting a column of zeros at the
  `baseline_geo` position in `tau_g_excl_baseline`.

  Args:
    tau_g_excl_baseline: A tensor of shape `[..., n_geos - 1]` for the
      user-defined dimensions of the `tau_g` parameter distribution.
    baseline_geo_idx: The index of the baseline geo to be set to zero.

  Returns:
    A tensor of shape `[..., n_geos]` with the final distribution of the `tau_g`
    parameter with zero at position `baseline_geo_idx` and matching
    `tau_g_excl_baseline` elsewhere.
  """
  rank = len(tau_g_excl_baseline.shape)
  shape = tau_g_excl_baseline.shape[:-1] + [1] if rank != 1 else 1
  tau_g = tf.concat(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          tf.zeros(shape, dtype=tau_g_excl_baseline.dtype),
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=rank - 1,
  )
  return tfp.distributions.Deterministic(tau_g, name="tau_g")


@tf.function(autograph=False, jit_compile=True)
def _xla_windowed_adaptive_nuts(**kwargs):
  """XLA wrapper for windowed_adaptive_nuts."""
  return tfp.experimental.mcmc.windowed_adaptive_nuts(**kwargs)


class Meridian:
  """Contains the main functionality for fitting the Meridian MMM model.

  Attributes:
    input_data: A `InputData` object that contains the input data for the model.
      This attribute is immutable and is the same `InputData` that was injected.
    model_spec: A `ModelSpec` object that contains the model specification. This
      attribute is immutable and is the same `ModelSpec` that was injected.
    model_data: A `ModelData` object that contains injected `InputData` and
      `ModelSpec` objects, as well as tensors and attributes derived from them.
      This attribute is immutable.
    inference_data: A *mutable* `arviz.InferenceData` object containing the
      resulting data from fitting the model.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec | None = None,
  ):
    # _model_data is immutable, derived strictly from input data and model spec.
    self._model_data = model_data.ModelData(
        input_data, model_spec if model_spec is not None else spec.ModelSpec()
    )
    # _inference_data is a mutable state that gets updated by sampling methods.
    self._inference_data = az.InferenceData()

  @property
  def model_data(self) -> model_data.ModelData:
    return self._model_data

  @property
  def input_data(self) -> data.InputData:
    return self.model_data.input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self.model_data.model_spec

  @property
  def inference_data(self) -> az.InferenceData:
    return self._inference_data

  def adstock_hill_media(
      self,
      media: tf.Tensor,  # pylint: disable=redefined-outer-name
      alpha: tf.Tensor,
      ec: tf.Tensor,
      slope: tf.Tensor,
  ) -> tf.Tensor:
    """Transforms media using Adstock and Hill functions in the desired order.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_media_channels)`
        containing non-negative media execution values. Typically this is
        impressions, but it can be any metric, such as `media_spend`. Clicks are
        often used for paid search ads.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_media_channels]`
      representing Adstock and Hill-transformed media.
    """
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=self.model_data.n_times,
    )
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    transformers_list = (
        [hill_transformer, adstock_transformer]
        if self.model_spec.hill_before_adstock
        else [adstock_transformer, hill_transformer]
    )

    media_out = media
    for transformer in transformers_list:
      media_out = transformer.forward(media_out)
    return media_out

  def adstock_hill_rf(
      self,
      reach: tf.Tensor,
      frequency: tf.Tensor,
      alpha: tf.Tensor,
      ec: tf.Tensor,
      slope: tf.Tensor,
  ) -> tf.Tensor:
    """Transforms reach and frequency (RF) using Hill and Adstock functions.

    Args:
      reach: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for reach.
      frequency: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for frequency.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_rf_channels]`
      representing Hill and Adstock-transformed RF.
    """
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=self.model_data.n_times,
    )
    adj_frequency = hill_transformer.forward(frequency)
    rf_out = adstock_transformer.forward(reach * adj_frequency)

    return rf_out

  def _get_roi_prior_beta_m_value(
      self,
      alpha_m: tf.Tensor,
      beta_gm_dev: tf.Tensor,
      ec_m: tf.Tensor,
      eta_m: tf.Tensor,
      roi_m: tf.Tensor,
      slope_m: tf.Tensor,
      media_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_m`."""
    mdata = self.model_data
    mtensors = mdata.media_tensors
    media_spend = mtensors.media_spend
    media_spend_counterfactual = mtensors.media_spend_counterfactual
    media_counterfactual_scaled = mtensors.media_counterfactual_scaled
    # If we got here, then we should already have media tensors derived from
    # non-None InputData.media data.
    assert media_spend is not None
    assert media_spend_counterfactual is not None
    assert media_counterfactual_scaled is not None

    inc_revenue_m = roi_m * tf.reduce_sum(
        media_spend - media_spend_counterfactual,
        range(media_spend.ndim - 1),
    )
    if self.model_spec.roi_calibration_period is not None:
      media_counterfactual_transformed = self.adstock_hill_media(
          media=media_counterfactual_scaled,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
      )
    else:
      media_counterfactual_transformed = tf.zeros_like(media_transformed)
    revenue_per_kpi = mdata.revenue_per_kpi
    if self.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([mdata.n_geos, mdata.n_times], dtype=tf.float32)
    media_contrib_gm = tf.einsum(
        "...gtm,g,,gt->...gm",
        media_transformed - media_counterfactual_transformed,
        mdata.population,
        mdata.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )

    if mdata.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
      media_contrib_m = tf.einsum("...gm->...m", media_contrib_gm)
      random_effect_m = tf.einsum(
          "...m,...gm,...gm->...m", eta_m, beta_gm_dev, media_contrib_gm
      )
      return (inc_revenue_m - random_effect_m) / media_contrib_m
    else:
      # For log_normal, beta_m and eta_m are not mean & std.
      # The parameterization is beta_gm ~ exp(beta_m + eta_m * N(0, 1)).
      random_effect_m = tf.einsum(
          "...gm,...gm->...m",
          tf.math.exp(beta_gm_dev * eta_m[..., tf.newaxis, :]),
          media_contrib_gm,
      )
    return tf.math.log(inc_revenue_m) - tf.math.log(random_effect_m)

  def _get_roi_prior_beta_rf_value(
      self,
      alpha_rf: tf.Tensor,
      beta_grf_dev: tf.Tensor,
      ec_rf: tf.Tensor,
      eta_rf: tf.Tensor,
      roi_rf: tf.Tensor,
      slope_rf: tf.Tensor,
      rf_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_rf`."""
    mdata = self.model_data
    rftensors = mdata.rf_tensors
    rf_spend = rftensors.rf_spend
    rf_spend_counterfactual = rftensors.rf_spend_counterfactual
    reach_counterfactual_scaled = rftensors.reach_counterfactual_scaled
    frequency = rftensors.frequency
    # If we got here, then we should already have RF media tensors derived from
    # non-None InputData.reach data.
    assert rf_spend is not None
    assert rf_spend_counterfactual is not None
    assert reach_counterfactual_scaled is not None
    assert frequency is not None

    inc_revenue_rf = roi_rf * tf.reduce_sum(
        rf_spend - rf_spend_counterfactual,
        range(rf_spend.ndim - 1),
    )
    if self.model_spec.rf_roi_calibration_period is not None:
      rf_counterfactual_transformed = self.adstock_hill_rf(
          reach=reach_counterfactual_scaled,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
      )
    else:
      rf_counterfactual_transformed = tf.zeros_like(rf_transformed)
    revenue_per_kpi = mdata.revenue_per_kpi
    if self.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([mdata.n_geos, mdata.n_times], dtype=tf.float32)

    media_contrib_grf = tf.einsum(
        "...gtm,g,,gt->...gm",
        rf_transformed - rf_counterfactual_transformed,
        mdata.population,
        mdata.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )
    if mdata.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
      media_contrib_rf = tf.einsum("...gm->...m", media_contrib_grf)
      random_effect_rf = tf.einsum(
          "...m,...gm,...gm->...m", eta_rf, beta_grf_dev, media_contrib_grf
      )
      return (inc_revenue_rf - random_effect_rf) / media_contrib_rf
    else:
      # For log_normal, beta_rf and eta_rf are not mean & std.
      # The parameterization is beta_grf ~ exp(beta_rf + eta_rf * N(0, 1)).
      random_effect_rf = tf.einsum(
          "...gm,...gm->...m",
          tf.math.exp(beta_grf_dev * eta_rf[..., tf.newaxis, :]),
          media_contrib_grf,
      )
      return tf.math.log(inc_revenue_rf) - tf.math.log(random_effect_rf)

  def _get_joint_dist_unpinned(self) -> tfp.distributions.Distribution:
    """Returns JointDistributionCoroutineAutoBatched function for MCMC."""

    mdata = self.model_data
    mdata.populate_cached_properties()

    # TODO(b/349416835): Extract this coroutine to be unittestable on its own.
    # This MCMC sampling technique is complex enough to have its own abstraction
    # and testable API, rather than being embedded as a private method in the
    # Meridian class.
    @tfp.distributions.JointDistributionCoroutineAutoBatched
    def joint_dist_unpinned():
      # Sample directly from prior.
      knot_values = yield mdata.prior_broadcast.knot_values
      gamma_c = yield mdata.prior_broadcast.gamma_c
      xi_c = yield mdata.prior_broadcast.xi_c
      sigma = yield mdata.prior_broadcast.sigma

      tau_g_excl_baseline = yield tfp.distributions.Sample(
          mdata.prior_broadcast.tau_g_excl_baseline,
          name=constants.TAU_G_EXCL_BASELINE,
      )
      tau_g = yield _get_tau_g(
          tau_g_excl_baseline=tau_g_excl_baseline,
          baseline_geo_idx=mdata.baseline_geo_idx,
      )
      tau_t = yield tfp.distributions.Deterministic(
          tf.einsum(
              "k,kt->t",
              knot_values,
              tf.convert_to_tensor(mdata.knot_info.weights),
          ),
          name=constants.TAU_T,
      )

      tau_gt = tau_g[:, tf.newaxis] + tau_t
      combined_media_transformed = tf.zeros(
          shape=(mdata.n_geos, mdata.n_times, 0), dtype=tf.float32
      )
      combined_beta = tf.zeros(shape=(mdata.n_geos, 0), dtype=tf.float32)
      if mdata.media_tensors.media is not None:
        alpha_m = yield mdata.prior_broadcast.alpha_m
        ec_m = yield mdata.prior_broadcast.ec_m
        eta_m = yield mdata.prior_broadcast.eta_m
        roi_m = yield mdata.prior_broadcast.roi_m
        slope_m = yield mdata.prior_broadcast.slope_m
        beta_gm_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [mdata.n_geos, mdata.n_media_channels],
            name=constants.BETA_GM_DEV,
        )
        media_transformed = self.adstock_hill_media(
            media=mdata.media_tensors.media_scaled,
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
        )
        if mdata.model_spec.use_roi_prior:
          beta_m_value = self._get_roi_prior_beta_m_value(
              alpha_m,
              beta_gm_dev,
              ec_m,
              eta_m,
              roi_m,
              slope_m,
              media_transformed,
          )
          beta_m = yield tfp.distributions.Deterministic(
              beta_m_value, name=constants.BETA_M
          )
        else:
          beta_m = yield mdata.prior_broadcast.beta_m

        if mdata.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
          beta_gm_value = beta_m + eta_m * beta_gm_dev
        else:
          # MEDIA_EFFECTS_LOG_NORMAL
          beta_gm_value = tf.math.exp(beta_m + eta_m * beta_gm_dev)

        beta_gm = yield tfp.distributions.Deterministic(
            beta_gm_value, name=constants.BETA_GM
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, media_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gm], axis=-1)

      if mdata.rf_tensors.reach is not None:
        alpha_rf = yield mdata.prior_broadcast.alpha_rf
        ec_rf = yield mdata.prior_broadcast.ec_rf
        eta_rf = yield mdata.prior_broadcast.eta_rf
        roi_rf = yield mdata.prior_broadcast.roi_rf
        slope_rf = yield mdata.prior_broadcast.slope_rf
        beta_grf_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [mdata.n_geos, mdata.n_rf_channels],
            name=constants.BETA_GRF_DEV,
        )
        rf_transformed = self.adstock_hill_rf(
            reach=mdata.rf_tensors.reach_scaled,
            frequency=mdata.rf_tensors.frequency,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
        )

        if mdata.model_spec.use_roi_prior:
          beta_rf_value = self._get_roi_prior_beta_rf_value(
              alpha_rf,
              beta_grf_dev,
              ec_rf,
              eta_rf,
              roi_rf,
              slope_rf,
              rf_transformed,
          )
          beta_rf = yield tfp.distributions.Deterministic(
              beta_rf_value,
              name=constants.BETA_RF,
          )
        else:
          beta_rf = yield mdata.prior_broadcast.beta_rf

        if mdata.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
          beta_grf_value = beta_rf + eta_rf * beta_grf_dev
        else:
          # MEDIA_EFFECTS_LOG_NORMAL
          beta_grf_value = tf.math.exp(beta_rf + eta_rf * beta_grf_dev)

        beta_grf = yield tfp.distributions.Deterministic(
            beta_grf_value, name=constants.BETA_GRF
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, rf_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_grf], axis=-1)

      sigma_gt = tf.transpose(
          tf.broadcast_to(sigma, [mdata.n_times, mdata.n_geos])
      )
      gamma_gc_dev = yield tfp.distributions.Sample(
          tfp.distributions.Normal(0, 1),
          [mdata.n_geos, mdata.n_controls],
          name=constants.GAMMA_GC_DEV,
      )
      gamma_gc = yield tfp.distributions.Deterministic(
          gamma_c + xi_c * gamma_gc_dev, name=constants.GAMMA_GC
      )
      y_pred = (
          tau_gt
          + tf.einsum("gtm,gm->gt", combined_media_transformed, combined_beta)
          + tf.einsum("gtc,gc->gt", mdata.controls_scaled, gamma_gc)
      )
      # If there are any holdout observations, the holdout KPI values will
      # be replaced with zeros using `experimental_pin`. For these
      # observations, we set the posterior mean equal to zero and standard
      # deviation to `1/sqrt(2pi)`, so the log-density is 0 regardless of the
      # sampled posterior parameter values.
      if mdata.holdout_id is not None:
        y_pred_holdout = tf.where(mdata.holdout_id, 0.0, y_pred)
        test_sd = tf.cast(1.0 / np.sqrt(2.0 * np.pi), tf.float32)
        sigma_gt_holdout = tf.where(mdata.holdout_id, test_sd, sigma_gt)
        yield tfp.distributions.Normal(
            y_pred_holdout, sigma_gt_holdout, name="y"
        )
      else:
        yield tfp.distributions.Normal(y_pred, sigma_gt, name="y")

    return joint_dist_unpinned

  def _get_joint_dist(self) -> tfp.distributions.Distribution:
    mdata = self.model_data
    y = (
        tf.where(mdata.holdout_id, 0.0, mdata.kpi_scaled)
        if mdata.holdout_id is not None
        else mdata.kpi_scaled
    )
    return self._get_joint_dist_unpinned().experimental_pin(y=y)

  def _create_inference_data_coords(
      self, n_chains: int, n_draws: int
  ) -> Mapping[str, np.ndarray | Sequence[str]]:
    """Creates data coordinates for inference data."""
    media_channel_values = (
        self.input_data.media_channel
        if self.input_data.media_channel is not None
        else np.array([])
    )
    rf_channel_values = (
        self.input_data.rf_channel
        if self.input_data.rf_channel is not None
        else np.array([])
    )
    return {
        constants.CHAIN: np.arange(n_chains),
        constants.DRAW: np.arange(n_draws),
        constants.GEO: self.input_data.geo,
        constants.TIME: self.input_data.time,
        constants.MEDIA_TIME: self.input_data.media_time,
        constants.KNOTS: np.arange(self.model_data.knot_info.n_knots),
        constants.CONTROL_VARIABLE: self.input_data.control_variable,
        constants.MEDIA_CHANNEL: media_channel_values,
        constants.RF_CHANNEL: rf_channel_values,
    }

  def _create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    inference_dims = dict(constants.INFERENCE_DIMS)
    if self.model_data.unique_sigma_for_each_geo:
      inference_dims[constants.SIGMA] = [constants.GEO]
    else:
      inference_dims[constants.SIGMA] = [constants.SIGMA_DIM]

    return {
        param: [constants.CHAIN, constants.DRAW] + list(dims)
        for param, dims in inference_dims.items()
    }

  def _sample_media_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of the media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of media parameter names to a tensor of shape [n_draws, n_geos,
      n_media_channels] or [n_draws, n_media_channels] containing the
      samples.
    """
    mdata = self.model_data
    prior = mdata.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    media_vars = {
        constants.ALPHA_M: prior.alpha_m.sample(**sample_kwargs),
        constants.EC_M: prior.ec_m.sample(**sample_kwargs),
        constants.ETA_M: prior.eta_m.sample(**sample_kwargs),
        constants.ROI_M: prior.roi_m.sample(**sample_kwargs),
        constants.SLOPE_M: prior.slope_m.sample(**sample_kwargs),
    }
    beta_gm_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mdata.n_geos, mdata.n_media_channels],
        name=constants.BETA_GM_DEV,
    ).sample(**sample_kwargs)
    media_transformed = self.adstock_hill_media(
        media=mdata.media_tensors.media_scaled,
        alpha=media_vars[constants.ALPHA_M],
        ec=media_vars[constants.EC_M],
        slope=media_vars[constants.SLOPE_M],
    )

    if self.model_spec.use_roi_prior:
      beta_m_value = self._get_roi_prior_beta_m_value(
          beta_gm_dev=beta_gm_dev,
          media_transformed=media_transformed,
          **media_vars,
      )
      media_vars[constants.BETA_M] = tfp.distributions.Deterministic(
          beta_m_value, name=constants.BETA_M
      ).sample()
    else:
      media_vars[constants.BETA_M] = prior.beta_m.sample(**sample_kwargs)

    beta_gm_value = (
        media_vars[constants.BETA_M][..., tf.newaxis, :]
        + media_vars[constants.ETA_M][..., tf.newaxis, :] * beta_gm_dev
    )
    if mdata.media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL:
      beta_gm_value = tf.math.exp(beta_gm_value)
    media_vars[constants.BETA_GM] = tfp.distributions.Deterministic(
        beta_gm_value, name=constants.BETA_GM
    ).sample()

    return media_vars

  def _sample_rf_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of the RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of RF parameter names to a tensor of shape [n_draws, n_geos,
      n_rf_channels] or [n_draws, n_rf_channels] containing the samples.
    """
    mdata = self.model_data
    prior = mdata.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    rf_vars = {
        constants.ALPHA_RF: prior.alpha_rf.sample(**sample_kwargs),
        constants.EC_RF: prior.ec_rf.sample(**sample_kwargs),
        constants.ETA_RF: prior.eta_rf.sample(**sample_kwargs),
        constants.ROI_RF: prior.roi_rf.sample(**sample_kwargs),
        constants.SLOPE_RF: prior.slope_rf.sample(**sample_kwargs),
    }
    beta_grf_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mdata.n_geos, mdata.n_rf_channels],
        name=constants.BETA_GRF_DEV,
    ).sample(**sample_kwargs)
    rf_transformed = self.adstock_hill_rf(
        reach=mdata.rf_tensors.reach_scaled,
        frequency=mdata.rf_tensors.frequency,
        alpha=rf_vars[constants.ALPHA_RF],
        ec=rf_vars[constants.EC_RF],
        slope=rf_vars[constants.SLOPE_RF],
    )

    if self.model_spec.use_roi_prior:
      beta_rf_value = self._get_roi_prior_beta_rf_value(
          beta_grf_dev=beta_grf_dev,
          rf_transformed=rf_transformed,
          **rf_vars,
      )
      rf_vars[constants.BETA_RF] = tfp.distributions.Deterministic(
          beta_rf_value,
          name=constants.BETA_RF,
      ).sample()
    else:
      rf_vars[constants.BETA_RF] = prior.beta_rf.sample(**sample_kwargs)

    beta_grf_value = (
        rf_vars[constants.BETA_RF][..., tf.newaxis, :]
        + rf_vars[constants.ETA_RF][..., tf.newaxis, :] * beta_grf_dev
    )
    if mdata.media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL:
      beta_grf_value = tf.math.exp(beta_grf_value)
    rf_vars[constants.BETA_GRF] = tfp.distributions.Deterministic(
        beta_grf_value, name=constants.BETA_GRF
    ).sample()

    return rf_vars

  def _sample_prior_fn(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Returns a mapping of prior parameters to tensors of the samples."""
    # For stateful sampling, the random seed must be set to ensure that any
    # random numbers that are generated are deterministic.
    if seed is not None:
      tf.keras.utils.set_random_seed(1)
    mdata = self.model_data
    prior = mdata.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}

    tau_g_excl_baseline = prior.tau_g_excl_baseline.sample(**sample_kwargs)
    base_vars = {
        constants.KNOT_VALUES: prior.knot_values.sample(**sample_kwargs),
        constants.GAMMA_C: prior.gamma_c.sample(**sample_kwargs),
        constants.XI_C: prior.xi_c.sample(**sample_kwargs),
        constants.SIGMA: prior.sigma.sample(**sample_kwargs),
        constants.TAU_G: _get_tau_g(
            tau_g_excl_baseline=tau_g_excl_baseline,
            baseline_geo_idx=mdata.baseline_geo_idx,
        ).sample(),
    }
    base_vars[constants.TAU_T] = tfp.distributions.Deterministic(
        tf.einsum(
            "...k,kt->...t",
            base_vars[constants.KNOT_VALUES],
            tf.convert_to_tensor(mdata.knot_info.weights),
        ),
        name=constants.TAU_T,
    ).sample()

    gamma_gc_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mdata.n_geos, mdata.n_controls],
        name=constants.GAMMA_GC_DEV,
    ).sample(**sample_kwargs)
    base_vars[constants.GAMMA_GC] = tfp.distributions.Deterministic(
        base_vars[constants.GAMMA_C][..., tf.newaxis, :]
        + base_vars[constants.XI_C][..., tf.newaxis, :] * gamma_gc_dev,
        name=constants.GAMMA_GC,
    ).sample()

    media_vars = (
        self._sample_media_priors(n_draws, seed)
        if mdata.media_tensors.media is not None
        else {}
    )
    rf_vars = (
        self._sample_rf_priors(n_draws, seed)
        if mdata.rf_tensors.reach is not None
        else {}
    )

    return base_vars | media_vars | rf_vars

  def sample_prior(self, n_draws: int, seed: int | None = None):
    """Draws samples from the prior distributions.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_draws = self._sample_prior_fn(n_draws, seed=seed)
    # Create Arviz InferenceData for prior draws.
    prior_coords = self._create_inference_data_coords(1, n_draws)
    prior_dims = self._create_inference_data_dims()
    prior_inference_data = az.convert_to_inference_data(
        prior_draws, coords=prior_coords, dims=prior_dims, group=constants.PRIOR
    )
    self.inference_data.extend(prior_inference_data, join="right")

  def sample_posterior(
      self,
      n_chains: Sequence[int] | int,
      n_adapt: int,
      n_burnin: int,
      n_keep: int,
      current_state: Mapping[str, tf.Tensor] | None = None,
      init_step_size: int | None = None,
      dual_averaging_kwargs: Mapping[str, int] | None = None,
      max_tree_depth: int = 10,
      max_energy_diff: float = 500.0,
      unrolled_leapfrog_steps: int = 1,
      parallel_iterations: int = 10,
      seed: Sequence[int] | None = None,
      **pins,
  ):
    """Runs Markov Chain Monte Carlo (MCMC) sampling of posterior distributions.

    For more details about the arguments, see [`windowed_adaptive_nuts`]
    (https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/windowed_adaptive_nuts).

    Args:
      n_chains: Number of MCMC chains. Given a sequence of integers,
        `windowed_adaptive_nuts` will be called once for each element. The
        `n_chains` argument of each `windowed_adaptive_nuts` call will be equal
        to the respective integer element. Using a list of integers, one can
        split the chains of a `windowed_adaptive_nuts` call into multiple calls
        with fewer chains per call. This can reduce memory usage. This might
        require an increased number of adaptation steps for convergence, as the
        optimization is occurring across fewer chains per sampling call.
      n_adapt: Number of adaptation draws per chain.
      n_burnin: Number of burn-in draws per chain. Burn-in draws occur after
        adaptation draws and before the kept draws.
      n_keep: Integer number of draws per chain to keep for inference.
      current_state: Optional structure of tensors at which to initialize
        sampling. Use the same shape and structure as
        `model.experimental_pin(**pins).sample(n_chains)`.
      init_step_size: Optional integer determining where to initialize the step
        size for the leapfrog integrator. The structure must broadcast with
        `current_state`. For example, if the initial state is:  ``` { 'a':
        tf.zeros(n_chains), 'b': tf.zeros([n_chains, n_features]), } ```  then
        any of `1.`, `{'a': 1., 'b': 1.}`, or `{'a': tf.ones(n_chains), 'b':
        tf.ones([n_chains, n_features])}` will work. Defaults to the dimension
        of the log density to the ¼ power.
      dual_averaging_kwargs: Optional dict keyword arguments to pass to
        `tfp.mcmc.DualAveragingStepSizeAdaptation`. By default, a
        `target_accept_prob` of `0.85` is set, acceptance probabilities across
        chains are reduced using a harmonic mean, and the class defaults are
        used otherwise.
      max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth`, for
        example, the number of nodes in a binary tree `max_tree_depth` nodes
        deep. The default setting of `10` takes up to 1024 leapfrog steps.
      max_energy_diff: Scalar threshold of energy differences at each leapfrog,
        divergence samples are defined as leapfrog steps that exceed this
        threshold. Default is `1000`.
      unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
        expansion step. Applies a direct linear multiplier to the maximum
        trajectory length implied by `max_tree_depth`. Defaults is `1`.
      parallel_iterations: Number of iterations allowed to run in parallel. Must
        be a positive integer. For more information, see `tf.while_loop`.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially (see
        https://developers.google.com/meridian/docs/advanced-modeling/additional-considerations#gpu-oom-error).
    """
    seed = tfp.random.sanitize_seed(seed) if seed else None
    n_chains_list = [n_chains] if isinstance(n_chains, int) else n_chains
    total_chains = np.sum(n_chains_list)

    states = []
    traces = []
    for n_chains_batch in n_chains_list:
      try:
        mcmc = _xla_windowed_adaptive_nuts(
            n_draws=n_burnin + n_keep,
            joint_dist=self._get_joint_dist(),
            n_chains=n_chains_batch,
            num_adaptation_steps=n_adapt,
            current_state=current_state,
            init_step_size=init_step_size,
            dual_averaging_kwargs=dual_averaging_kwargs,
            max_tree_depth=max_tree_depth,
            max_energy_diff=max_energy_diff,
            unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            parallel_iterations=parallel_iterations,
            seed=seed,
            **pins,
        )
      except tf.errors.ResourceExhaustedError as error:
        raise MCMCOOMError(
            "ERROR: Out of memory. Try reducing `n_keep` or pass a list of"
            " integers as `n_chains` to sample chains serially (see"
            " https://developers.google.com/meridian/docs/advanced-modeling/additional-considerations#gpu-oom-error)"
        ) from error
      states.append(mcmc.all_states._asdict())
      traces.append(mcmc.trace)

    mcmc_states = {
        k: tf.einsum(
            "ij...->ji...",
            tf.concat([state[k] for state in states], axis=1)[n_burnin:, ...],
        )
        for k in states[0].keys()
        if k not in constants.IGNORED_PRIOR_PARAMETERS
    }
    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self._create_inference_data_coords(total_chains, n_keep)
    posterior_dims = self._create_inference_data_dims()
    infdata_posterior = az.convert_to_inference_data(
        mcmc_states, coords=posterior_coords, dims=posterior_dims
    )

    # Save trace metrics in InferenceData.
    mcmc_trace = {}
    for k in traces[0].keys():
      if k not in constants.IGNORED_TRACE_METRICS:
        mcmc_trace[k] = tf.concat(
            [
                tf.broadcast_to(
                    tf.transpose(trace[k][n_burnin:, ...]),
                    [n_chains_list[i], n_keep],
                )
                for i, trace in enumerate(traces)
            ],
            axis=0,
        )

    trace_coords = {
        constants.CHAIN: np.arange(total_chains),
        constants.DRAW: np.arange(n_keep),
    }
    trace_dims = {
        k: [constants.CHAIN, constants.DRAW] for k in mcmc_trace.keys()
    }
    infdata_trace = az.convert_to_inference_data(
        mcmc_trace, coords=trace_coords, dims=trace_dims, group="trace"
    )

    # Create Arviz InferenceData for divergent transitions and other sampling
    # statistics. Note that InferenceData has a different naming convention
    # than Tensorflow, and only certain variables are recongnized.
    # https://arviz-devs.github.io/arviz/schema/schema.html#sample-stats
    # The list of values returned by windowed_adaptive_nuts() is the following:
    # 'step_size', 'tune', 'target_log_prob', 'diverging', 'accept_ratio',
    # 'variance_scaling', 'n_steps', 'is_accepted'.

    sample_stats = {
        constants.SAMPLE_STATS_METRICS[k]: v
        for k, v in mcmc_trace.items()
        if k in constants.SAMPLE_STATS_METRICS
    }
    sample_stats_dims = {
        constants.SAMPLE_STATS_METRICS[k]: v
        for k, v in trace_dims.items()
        if k in constants.SAMPLE_STATS_METRICS
    }
    # Tensorflow does not include a "draw" dimension on step size metric if same
    # step size is used for all chains. Step size must be broadcast to the
    # correct shape.
    sample_stats[constants.STEP_SIZE] = tf.broadcast_to(
        sample_stats[constants.STEP_SIZE], [total_chains, n_keep]
    )
    sample_stats_dims[constants.STEP_SIZE] = [constants.CHAIN, constants.DRAW]
    infdata_sample_stats = az.convert_to_inference_data(
        sample_stats,
        coords=trace_coords,
        dims=sample_stats_dims,
        group="sample_stats",
    )
    posterior_inference_data = az.concat(
        infdata_posterior, infdata_trace, infdata_sample_stats
    )
    self.inference_data.extend(posterior_inference_data, join="right")


def save_mmm(mmm: Meridian, file_path: str):
  """Save the model object to a `pickle` file path.

  Args:
    mmm: Model object to save.
    file_path: File path to save a pickled model object.
  """
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))

  with open(file_path, "wb") as f:
    joblib.dump(mmm, f)


def load_mmm(file_path: str) -> Meridian:
  """Load the model object from a `pickle` file path.

  Args:
    file_path: File path to load a pickled model object from.

  Returns:
    mmm: Model object loaded from the file path.

  Raises:
      FileNotFoundError: If `file_path` does not exist.
  """
  try:
    with open(file_path, "rb") as f:
      mmm = joblib.load(f)
    return mmm
  except FileNotFoundError:
    raise FileNotFoundError(f"No such file or directory: {file_path}") from None
