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

"""Module for sampling prior distributions in a Meridian model."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import arviz as az
from meridian import constants
import tensorflow as tf
import tensorflow_probability as tfp

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "PriorDistributionSampler",
]


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


class PriorDistributionSampler:
  """A callable that samples from a model spec's prior distributions."""

  def __init__(self, meridian: "model.Meridian"):
    self._meridian = meridian

  def get_roi_prior_beta_m_value(
      self,
      alpha_m: tf.Tensor,
      beta_gm_dev: tf.Tensor,
      ec_m: tf.Tensor,
      eta_m: tf.Tensor,
      roi_or_mroi_m: tf.Tensor,
      slope_m: tf.Tensor,
      media_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_m`."""
    mmm = self._meridian

    # The `roi_or_mroi_m` parameter represents either ROI or mROI. For reach &
    # frequency channels, marginal ROI priors are defined as "mROI by reach",
    # which is equivalent to ROI.
    media_spend = mmm.media_tensors.media_spend
    media_spend_counterfactual = mmm.media_tensors.media_spend_counterfactual
    media_counterfactual_scaled = mmm.media_tensors.media_counterfactual_scaled
    # If we got here, then we should already have media tensors derived from
    # non-None InputData.media data.
    assert media_spend is not None
    assert media_spend_counterfactual is not None
    assert media_counterfactual_scaled is not None

    # Use absolute value here because this difference will be negative for
    # marginal ROI priors.
    inc_revenue_m = roi_or_mroi_m * tf.reduce_sum(
        tf.abs(media_spend - media_spend_counterfactual),
        range(media_spend.ndim - 1),
    )

    if mmm.model_spec.roi_calibration_period is None and (
        mmm.model_spec.media_prior_type == constants.TREATMENT_PRIOR_TYPE_ROI
    ):
      # We can skip the adstock/hill computation step in this case.
      media_counterfactual_transformed = tf.zeros_like(media_transformed)
    else:
      media_counterfactual_transformed = mmm.adstock_hill_media(
          media=media_counterfactual_scaled,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
      )

    revenue_per_kpi = mmm.revenue_per_kpi
    if mmm.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([mmm.n_geos, mmm.n_times], dtype=tf.float32)
    # Note: use absolute value here because this difference will be negative for
    # marginal ROI priors.
    media_contrib_gm = tf.einsum(
        "...gtm,g,,gt->...gm",
        tf.abs(media_transformed - media_counterfactual_transformed),
        mmm.population,
        mmm.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )

    if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
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

  def get_roi_prior_beta_rf_value(
      self,
      alpha_rf: tf.Tensor,
      beta_grf_dev: tf.Tensor,
      ec_rf: tf.Tensor,
      eta_rf: tf.Tensor,
      roi_or_mroi_rf: tf.Tensor,
      slope_rf: tf.Tensor,
      rf_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_rf`."""
    mmm = self._meridian

    rf_spend = mmm.rf_tensors.rf_spend
    rf_spend_counterfactual = mmm.rf_tensors.rf_spend_counterfactual
    reach_counterfactual_scaled = mmm.rf_tensors.reach_counterfactual_scaled
    frequency = mmm.rf_tensors.frequency
    # If we got here, then we should already have RF media tensors derived from
    # non-None InputData.reach data.
    assert rf_spend is not None
    assert rf_spend_counterfactual is not None
    assert reach_counterfactual_scaled is not None
    assert frequency is not None

    inc_revenue_rf = roi_or_mroi_rf * tf.reduce_sum(
        rf_spend - rf_spend_counterfactual,
        range(rf_spend.ndim - 1),
    )
    if mmm.model_spec.rf_roi_calibration_period is not None:
      rf_counterfactual_transformed = mmm.adstock_hill_rf(
          reach=reach_counterfactual_scaled,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
      )
    else:
      rf_counterfactual_transformed = tf.zeros_like(rf_transformed)
    revenue_per_kpi = mmm.revenue_per_kpi
    if mmm.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([mmm.n_geos, mmm.n_times], dtype=tf.float32)

    media_contrib_grf = tf.einsum(
        "...gtm,g,,gt->...gm",
        rf_transformed - rf_counterfactual_transformed,
        mmm.population,
        mmm.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )
    if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
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
      A mapping of media parameter names to a tensor of shape `[n_draws, n_geos,
      n_media_channels]` or `[n_draws, n_media_channels]` containing the
      samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    media_vars = {
        constants.ALPHA_M: prior.alpha_m.sample(**sample_kwargs),
        constants.EC_M: prior.ec_m.sample(**sample_kwargs),
        constants.ETA_M: prior.eta_m.sample(**sample_kwargs),
        constants.SLOPE_M: prior.slope_m.sample(**sample_kwargs),
    }
    beta_gm_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_media_channels],
        name=constants.BETA_GM_DEV,
    ).sample(**sample_kwargs)
    media_transformed = mmm.adstock_hill_media(
        media=mmm.media_tensors.media_scaled,
        alpha=media_vars[constants.ALPHA_M],
        ec=media_vars[constants.EC_M],
        slope=media_vars[constants.SLOPE_M],
    )

    prior_type = mmm.model_spec.media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      roi_m = prior.roi_m.sample(**sample_kwargs)
      beta_m_value = self.get_roi_prior_beta_m_value(
          beta_gm_dev=beta_gm_dev,
          media_transformed=media_transformed,
          roi_or_mroi_m=roi_m,
          **media_vars,
      )
      media_vars[constants.ROI_M] = roi_m
      media_vars[constants.BETA_M] = tfp.distributions.Deterministic(
          beta_m_value, name=constants.BETA_M
      ).sample()
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      mroi_m = prior.mroi_m.sample(**sample_kwargs)
      beta_m_value = self.get_roi_prior_beta_m_value(
          beta_gm_dev=beta_gm_dev,
          media_transformed=media_transformed,
          roi_or_mroi_m=mroi_m,
          **media_vars,
      )
      media_vars[constants.MROI_M] = mroi_m
      media_vars[constants.BETA_M] = tfp.distributions.Deterministic(
          beta_m_value, name=constants.BETA_M
      ).sample()
    else:
      media_vars[constants.BETA_M] = prior.beta_m.sample(**sample_kwargs)

    beta_eta_combined = (
        media_vars[constants.BETA_M][..., tf.newaxis, :]
        + media_vars[constants.ETA_M][..., tf.newaxis, :] * beta_gm_dev
    )
    beta_gm_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else tf.math.exp(beta_eta_combined)
    )
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
      A mapping of RF parameter names to a tensor of shape `[n_draws, n_geos,
      n_rf_channels]` or `[n_draws, n_rf_channels]` containing the samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    rf_vars = {
        constants.ALPHA_RF: prior.alpha_rf.sample(**sample_kwargs),
        constants.EC_RF: prior.ec_rf.sample(**sample_kwargs),
        constants.ETA_RF: prior.eta_rf.sample(**sample_kwargs),
        constants.SLOPE_RF: prior.slope_rf.sample(**sample_kwargs),
    }
    beta_grf_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_rf_channels],
        name=constants.BETA_GRF_DEV,
    ).sample(**sample_kwargs)
    rf_transformed = mmm.adstock_hill_rf(
        reach=mmm.rf_tensors.reach_scaled,
        frequency=mmm.rf_tensors.frequency,
        alpha=rf_vars[constants.ALPHA_RF],
        ec=rf_vars[constants.EC_RF],
        slope=rf_vars[constants.SLOPE_RF],
    )

    prior_type = mmm.model_spec.rf_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      roi_rf = prior.roi_rf.sample(**sample_kwargs)
      beta_rf_value = self.get_roi_prior_beta_rf_value(
          beta_grf_dev=beta_grf_dev,
          rf_transformed=rf_transformed,
          roi_or_mroi_rf=roi_rf,
          **rf_vars,
      )
      rf_vars[constants.ROI_RF] = roi_rf
      rf_vars[constants.BETA_RF] = tfp.distributions.Deterministic(
          beta_rf_value,
          name=constants.BETA_RF,
      ).sample()
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      mroi_rf = prior.mroi_rf.sample(**sample_kwargs)
      beta_rf_value = self.get_roi_prior_beta_rf_value(
          beta_grf_dev=beta_grf_dev,
          rf_transformed=rf_transformed,
          roi_or_mroi_rf=mroi_rf,
          **rf_vars,
      )
      rf_vars[constants.MROI_RF] = mroi_rf
      rf_vars[constants.BETA_RF] = tfp.distributions.Deterministic(
          beta_rf_value,
          name=constants.BETA_RF,
      ).sample()
    else:
      rf_vars[constants.BETA_RF] = prior.beta_rf.sample(**sample_kwargs)

    beta_eta_combined = (
        rf_vars[constants.BETA_RF][..., tf.newaxis, :]
        + rf_vars[constants.ETA_RF][..., tf.newaxis, :] * beta_grf_dev
    )
    beta_grf_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else tf.math.exp(beta_eta_combined)
    )
    rf_vars[constants.BETA_GRF] = tfp.distributions.Deterministic(
        beta_grf_value, name=constants.BETA_GRF
    ).sample()

    return rf_vars

  def _sample_organic_media_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of organic media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of organic media parameter names to a tensor of shape [n_draws,
      n_geos, n_organic_media_channels] or [n_draws, n_organic_media_channels]
      containing the samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    organic_media_vars = {
        constants.ALPHA_OM: prior.alpha_om.sample(**sample_kwargs),
        constants.EC_OM: prior.ec_om.sample(**sample_kwargs),
        constants.ETA_OM: prior.eta_om.sample(**sample_kwargs),
        constants.SLOPE_OM: prior.slope_om.sample(**sample_kwargs),
    }
    beta_gom_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_organic_media_channels],
        name=constants.BETA_GOM_DEV,
    ).sample(**sample_kwargs)

    organic_media_vars[constants.BETA_OM] = prior.beta_om.sample(
        **sample_kwargs
    )

    beta_eta_combined = (
        organic_media_vars[constants.BETA_OM][..., tf.newaxis, :]
        + organic_media_vars[constants.ETA_OM][..., tf.newaxis, :]
        * beta_gom_dev
    )
    beta_gom_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else tf.math.exp(beta_eta_combined)
    )
    organic_media_vars[constants.BETA_GOM] = tfp.distributions.Deterministic(
        beta_gom_value, name=constants.BETA_GOM
    ).sample()

    return organic_media_vars

  def _sample_organic_rf_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of the organic RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of organic RF parameter names to a tensor of shape [n_draws,
      n_geos, n_organic_rf_channels] or [n_draws, n_organic_rf_channels]
      containing the samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    organic_rf_vars = {
        constants.ALPHA_ORF: prior.alpha_orf.sample(**sample_kwargs),
        constants.EC_ORF: prior.ec_orf.sample(**sample_kwargs),
        constants.ETA_ORF: prior.eta_orf.sample(**sample_kwargs),
        constants.SLOPE_ORF: prior.slope_orf.sample(**sample_kwargs),
    }
    beta_gorf_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_organic_rf_channels],
        name=constants.BETA_GORF_DEV,
    ).sample(**sample_kwargs)

    organic_rf_vars[constants.BETA_ORF] = prior.beta_orf.sample(**sample_kwargs)

    beta_eta_combined = (
        organic_rf_vars[constants.BETA_ORF][..., tf.newaxis, :]
        + organic_rf_vars[constants.ETA_ORF][..., tf.newaxis, :] * beta_gorf_dev
    )
    beta_gorf_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else tf.math.exp(beta_eta_combined)
    )
    organic_rf_vars[constants.BETA_GORF] = tfp.distributions.Deterministic(
        beta_gorf_value, name=constants.BETA_GORF
    ).sample()

    return organic_rf_vars

  def _sample_non_media_treatments_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws from the prior distributions of the non-media treatment variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of non-media treatment parameter names to a tensor of shape
      [n_draws,
      n_geos, n_non_media_channels] or [n_draws, n_non_media_channels]
      containing the samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    non_media_treatments_vars = {
        constants.GAMMA_N: prior.gamma_n.sample(**sample_kwargs),
        constants.XI_N: prior.xi_n.sample(**sample_kwargs),
    }
    gamma_gn_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_non_media_channels],
        name=constants.GAMMA_GN_DEV,
    ).sample(**sample_kwargs)
    non_media_treatments_vars[constants.GAMMA_GN] = (
        tfp.distributions.Deterministic(
            non_media_treatments_vars[constants.GAMMA_N][..., tf.newaxis, :]
            + non_media_treatments_vars[constants.XI_N][..., tf.newaxis, :]
            * gamma_gn_dev,
            name=constants.GAMMA_GN,
        ).sample()
    )
    return non_media_treatments_vars

  def _sample_prior(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Returns a mapping of prior parameters to tensors of the samples."""
    mmm = self._meridian

    # For stateful sampling, the random seed must be set to ensure that any
    # random numbers that are generated are deterministic.
    if seed is not None:
      tf.keras.utils.set_random_seed(1)

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}

    tau_g_excl_baseline = prior.tau_g_excl_baseline.sample(**sample_kwargs)
    base_vars = {
        constants.KNOT_VALUES: prior.knot_values.sample(**sample_kwargs),
        constants.GAMMA_C: prior.gamma_c.sample(**sample_kwargs),
        constants.XI_C: prior.xi_c.sample(**sample_kwargs),
        constants.SIGMA: prior.sigma.sample(**sample_kwargs),
        constants.TAU_G: (
            _get_tau_g(
                tau_g_excl_baseline=tau_g_excl_baseline,
                baseline_geo_idx=mmm.baseline_geo_idx,
            ).sample()
        ),
    }
    base_vars[constants.MU_T] = tfp.distributions.Deterministic(
        tf.einsum(
            "...k,kt->...t",
            base_vars[constants.KNOT_VALUES],
            tf.convert_to_tensor(mmm.knot_info.weights),
        ),
        name=constants.MU_T,
    ).sample()

    gamma_gc_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [mmm.n_geos, mmm.n_controls],
        name=constants.GAMMA_GC_DEV,
    ).sample(**sample_kwargs)
    base_vars[constants.GAMMA_GC] = tfp.distributions.Deterministic(
        base_vars[constants.GAMMA_C][..., tf.newaxis, :]
        + base_vars[constants.XI_C][..., tf.newaxis, :] * gamma_gc_dev,
        name=constants.GAMMA_GC,
    ).sample()

    media_vars = (
        self._sample_media_priors(n_draws, seed)
        if mmm.media_tensors.media is not None
        else {}
    )
    rf_vars = (
        self._sample_rf_priors(n_draws, seed)
        if mmm.rf_tensors.reach is not None
        else {}
    )
    organic_media_vars = (
        self._sample_organic_media_priors(n_draws, seed)
        if mmm.organic_media_tensors.organic_media is not None
        else {}
    )
    organic_rf_vars = (
        self._sample_organic_rf_priors(n_draws, seed)
        if mmm.organic_rf_tensors.organic_reach is not None
        else {}
    )
    non_media_treatments_vars = (
        self._sample_non_media_treatments_priors(n_draws, seed)
        if mmm.non_media_treatments_scaled is not None
        else {}
    )

    return (
        base_vars
        | media_vars
        | rf_vars
        | organic_media_vars
        | organic_rf_vars
        | non_media_treatments_vars
    )

  def __call__(self, n_draws: int, seed: int | None = None) -> az.InferenceData:
    """Draws samples from prior distributions.

    Returns:
      An Arviz `InferenceData` object containing prior samples only.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_draws = self._sample_prior(n_draws, seed=seed)
    # Create Arviz InferenceData for prior draws.
    prior_coords = self._meridian.create_inference_data_coords(1, n_draws)
    prior_dims = self._meridian.create_inference_data_dims()
    return az.convert_to_inference_data(
        prior_draws, coords=prior_coords, dims=prior_dims, group=constants.PRIOR
    )
