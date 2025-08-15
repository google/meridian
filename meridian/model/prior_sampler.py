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

"""Module for sampling prior distributions in a Meridian model."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import arviz as az
from meridian import backend
from meridian import constants

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "PriorDistributionSampler",
]


def _get_tau_g(
    tau_g_excl_baseline: backend.Tensor, baseline_geo_idx: int
) -> backend.tfd.Distribution:
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
  tau_g = backend.concatenate(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          backend.zeros(shape, dtype=tau_g_excl_baseline.dtype),
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=rank - 1,
  )
  return backend.tfd.Deterministic(tau_g, name="tau_g")


class PriorDistributionSampler:
  """A callable that samples from a model spec's prior distributions."""

  def __init__(self, meridian: "model.Meridian"):
    self._meridian = meridian

  def _sample_media_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
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
    beta_gm_dev = backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [mmm.n_geos, mmm.n_media_channels],
        name=constants.BETA_GM_DEV,
    ).sample(**sample_kwargs)

    prior_type = mmm.model_spec.effective_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      media_vars[constants.BETA_M] = prior.beta_m.sample(**sample_kwargs)
    else:
      if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
        treatment_parameter_m = prior.roi_m.sample(**sample_kwargs)
        media_vars[constants.ROI_M] = treatment_parameter_m
      elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
        treatment_parameter_m = prior.mroi_m.sample(**sample_kwargs)
        media_vars[constants.MROI_M] = treatment_parameter_m
      elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
        treatment_parameter_m = prior.contribution_m.sample(**sample_kwargs)
        media_vars[constants.CONTRIBUTION_M] = treatment_parameter_m
      else:
        raise ValueError(f"Unsupported prior type: {prior_type}")
      incremental_outcome_m = (
          treatment_parameter_m * mmm.media_tensors.prior_denominator
      )
      media_transformed = mmm.adstock_hill_media(
          media=mmm.media_tensors.media_scaled,
          alpha=media_vars[constants.ALPHA_M],
          ec=media_vars[constants.EC_M],
          slope=media_vars[constants.SLOPE_M],
      )
      linear_predictor_counterfactual_difference = (
          mmm.linear_predictor_counterfactual_difference_media(
              media_transformed=media_transformed,
              alpha_m=media_vars[constants.ALPHA_M],
              ec_m=media_vars[constants.EC_M],
              slope_m=media_vars[constants.SLOPE_M],
          )
      )
      beta_m_value = mmm.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_m,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=media_vars[constants.ETA_M],
          beta_gx_dev=beta_gm_dev,
      )
      media_vars[constants.BETA_M] = backend.tfd.Deterministic(
          beta_m_value, name=constants.BETA_M
      ).sample()

    beta_eta_combined = (
        media_vars[constants.BETA_M][..., backend.newaxis, :]
        + media_vars[constants.ETA_M][..., backend.newaxis, :] * beta_gm_dev
    )
    beta_gm_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    media_vars[constants.BETA_GM] = backend.tfd.Deterministic(
        beta_gm_value, name=constants.BETA_GM
    ).sample()

    return media_vars

  def _sample_rf_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of RF parameter names to a tensor of shape
      `[n_draws, n_geos, n_rf_channels]` or `[n_draws, n_rf_channels]`
      containing the samples.
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
    beta_grf_dev = backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [mmm.n_geos, mmm.n_rf_channels],
        name=constants.BETA_GRF_DEV,
    ).sample(**sample_kwargs)

    prior_type = mmm.model_spec.effective_rf_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      rf_vars[constants.BETA_RF] = prior.beta_rf.sample(**sample_kwargs)
    else:
      if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
        treatment_parameter_rf = prior.roi_rf.sample(**sample_kwargs)
        rf_vars[constants.ROI_RF] = treatment_parameter_rf
      elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
        treatment_parameter_rf = prior.mroi_rf.sample(**sample_kwargs)
        rf_vars[constants.MROI_RF] = treatment_parameter_rf
      elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
        treatment_parameter_rf = prior.contribution_rf.sample(**sample_kwargs)
        rf_vars[constants.CONTRIBUTION_RF] = treatment_parameter_rf
      else:
        raise ValueError(f"Unsupported prior type: {prior_type}")
      incremental_outcome_rf = (
          treatment_parameter_rf * mmm.rf_tensors.prior_denominator
      )
      rf_transformed = mmm.adstock_hill_rf(
          reach=mmm.rf_tensors.reach_scaled,
          frequency=mmm.rf_tensors.frequency,
          alpha=rf_vars[constants.ALPHA_RF],
          ec=rf_vars[constants.EC_RF],
          slope=rf_vars[constants.SLOPE_RF],
      )
      linear_predictor_counterfactual_difference = (
          mmm.linear_predictor_counterfactual_difference_rf(
              rf_transformed=rf_transformed,
              alpha_rf=rf_vars[constants.ALPHA_RF],
              ec_rf=rf_vars[constants.EC_RF],
              slope_rf=rf_vars[constants.SLOPE_RF],
          )
      )
      beta_rf_value = mmm.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_rf,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=rf_vars[constants.ETA_RF],
          beta_gx_dev=beta_grf_dev,
      )
      rf_vars[constants.BETA_RF] = backend.tfd.Deterministic(
          beta_rf_value,
          name=constants.BETA_RF,
      ).sample()

    beta_eta_combined = (
        rf_vars[constants.BETA_RF][..., backend.newaxis, :]
        + rf_vars[constants.ETA_RF][..., backend.newaxis, :] * beta_grf_dev
    )
    beta_grf_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    rf_vars[constants.BETA_GRF] = backend.tfd.Deterministic(
        beta_grf_value, name=constants.BETA_GRF
    ).sample()

    return rf_vars

  def _sample_organic_media_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of organic media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of organic media parameter names to a tensor of shape
      `[n_draws, n_geos, n_organic_media_channels]` or
      `[n_draws, n_organic_media_channels]` containing the samples.
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
    beta_gom_dev = backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [mmm.n_geos, mmm.n_organic_media_channels],
        name=constants.BETA_GOM_DEV,
    ).sample(**sample_kwargs)

    prior_type = mmm.model_spec.organic_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      organic_media_vars[constants.BETA_OM] = prior.beta_om.sample(
          **sample_kwargs
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      organic_media_vars[constants.CONTRIBUTION_OM] = (
          prior.contribution_om.sample(**sample_kwargs)
      )
      incremental_outcome_om = (
          organic_media_vars[constants.CONTRIBUTION_OM] * mmm.total_outcome
      )
      organic_media_transformed = mmm.adstock_hill_media(
          media=mmm.organic_media_tensors.organic_media_scaled,
          alpha=organic_media_vars[constants.ALPHA_OM],
          ec=organic_media_vars[constants.EC_OM],
          slope=organic_media_vars[constants.SLOPE_OM],
      )
      beta_om_value = mmm.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_om,
          linear_predictor_counterfactual_difference=organic_media_transformed,
          eta_x=organic_media_vars[constants.ETA_OM],
          beta_gx_dev=beta_gom_dev,
      )
      organic_media_vars[constants.BETA_OM] = backend.tfd.Deterministic(
          beta_om_value,
          name=constants.BETA_OM,
      ).sample()
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = (
        organic_media_vars[constants.BETA_OM][..., backend.newaxis, :]
        + organic_media_vars[constants.ETA_OM][..., backend.newaxis, :]
        * beta_gom_dev
    )
    beta_gom_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    organic_media_vars[constants.BETA_GOM] = backend.tfd.Deterministic(
        beta_gom_value, name=constants.BETA_GOM
    ).sample()

    return organic_media_vars

  def _sample_organic_rf_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the organic RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of organic RF parameter names to a tensor of shape
      `[n_draws, n_geos, n_organic_rf_channels]` or
      `[n_draws, n_organic_rf_channels]` containing the samples.
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
    beta_gorf_dev = backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [mmm.n_geos, mmm.n_organic_rf_channels],
        name=constants.BETA_GORF_DEV,
    ).sample(**sample_kwargs)

    prior_type = mmm.model_spec.organic_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      organic_rf_vars[constants.BETA_ORF] = prior.beta_orf.sample(
          **sample_kwargs
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      organic_rf_vars[constants.CONTRIBUTION_ORF] = (
          prior.contribution_orf.sample(**sample_kwargs)
      )
      incremental_outcome_orf = (
          organic_rf_vars[constants.CONTRIBUTION_ORF] * mmm.total_outcome
      )
      organic_rf_transformed = mmm.adstock_hill_rf(
          reach=mmm.organic_rf_tensors.organic_reach_scaled,
          frequency=mmm.organic_rf_tensors.organic_frequency,
          alpha=organic_rf_vars[constants.ALPHA_ORF],
          ec=organic_rf_vars[constants.EC_ORF],
          slope=organic_rf_vars[constants.SLOPE_ORF],
      )
      beta_orf_value = mmm.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_orf,
          linear_predictor_counterfactual_difference=organic_rf_transformed,
          eta_x=organic_rf_vars[constants.ETA_ORF],
          beta_gx_dev=beta_gorf_dev,
      )
      organic_rf_vars[constants.BETA_ORF] = backend.tfd.Deterministic(
          beta_orf_value,
          name=constants.BETA_ORF,
      ).sample()
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = (
        organic_rf_vars[constants.BETA_ORF][..., backend.newaxis, :]
        + organic_rf_vars[constants.ETA_ORF][..., backend.newaxis, :]
        * beta_gorf_dev
    )
    beta_gorf_value = (
        beta_eta_combined
        if mmm.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    organic_rf_vars[constants.BETA_GORF] = backend.tfd.Deterministic(
        beta_gorf_value, name=constants.BETA_GORF
    ).sample()

    return organic_rf_vars

  def _sample_non_media_treatments_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
    """Draws from the prior distributions of the non-media treatment variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of non-media treatment parameter names to a tensor of shape
      `[n_draws, n_geos, n_non_media_channels]` or
      `[n_draws, n_non_media_channels]` containing the samples.
    """
    mmm = self._meridian

    prior = mmm.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    non_media_treatments_vars = {
        constants.XI_N: prior.xi_n.sample(**sample_kwargs),
    }
    gamma_gn_dev = backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [mmm.n_geos, mmm.n_non_media_channels],
        name=constants.GAMMA_GN_DEV,
    ).sample(**sample_kwargs)
    prior_type = mmm.model_spec.non_media_treatments_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      non_media_treatments_vars[constants.GAMMA_N] = prior.gamma_n.sample(
          **sample_kwargs
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      non_media_treatments_vars[constants.CONTRIBUTION_N] = (
          prior.contribution_n.sample(**sample_kwargs)
      )
      incremental_outcome_n = (
          non_media_treatments_vars[constants.CONTRIBUTION_N]
          * mmm.total_outcome
      )
      baseline_scaled = mmm.non_media_transformer.forward(  # pytype: disable=attribute-error
          mmm.compute_non_media_treatments_baseline()
      )
      linear_predictor_counterfactual_difference = (
          mmm.non_media_treatments_normalized - baseline_scaled
      )
      gamma_n_value = mmm.calculate_beta_x(
          is_non_media=True,
          incremental_outcome_x=incremental_outcome_n,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=non_media_treatments_vars[constants.XI_N],
          beta_gx_dev=gamma_gn_dev,
      )
      non_media_treatments_vars[constants.GAMMA_N] = backend.tfd.Deterministic(
          gamma_n_value, name=constants.GAMMA_N
      ).sample()
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")
    non_media_treatments_vars[constants.GAMMA_GN] = backend.tfd.Deterministic(
        non_media_treatments_vars[constants.GAMMA_N][..., backend.newaxis, :]
        + non_media_treatments_vars[constants.XI_N][..., backend.newaxis, :]
        * gamma_gn_dev,
        name=constants.GAMMA_GN,
    ).sample()
    return non_media_treatments_vars

  def _sample_prior(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, backend.Tensor]:
    """Returns a mapping of prior parameters to tensors of the samples."""
    mmm = self._meridian

    # For stateful sampling, the random seed must be set to ensure that any
    # random numbers that are generated are deterministic.
    if seed is not None:
      backend.set_random_seed(1)

    prior = mmm.prior_broadcast
    # `sample_shape` is prepended to the shape of each BatchBroadcast in `prior`
    # when it is sampled.
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}

    tau_g_excl_baseline = prior.tau_g_excl_baseline.sample(**sample_kwargs)
    base_vars = {
        constants.KNOT_VALUES: prior.knot_values.sample(**sample_kwargs),
        constants.SIGMA: prior.sigma.sample(**sample_kwargs),
        constants.TAU_G: (
            _get_tau_g(
                tau_g_excl_baseline=tau_g_excl_baseline,
                baseline_geo_idx=mmm.baseline_geo_idx,
            ).sample()
        ),
    }

    base_vars[constants.MU_T] = backend.tfd.Deterministic(
        backend.einsum(
            "...k,kt->...t",
            base_vars[constants.KNOT_VALUES],
            backend.to_tensor(mmm.knot_info.weights),
        ),
        name=constants.MU_T,
    ).sample()

    # Omit gamma_c, xi_c, and gamma_gc parameters from sampled distributions if
    # there are no control variables in the model.
    if mmm.n_controls:
      base_vars |= {
          constants.GAMMA_C: prior.gamma_c.sample(**sample_kwargs),
          constants.XI_C: prior.xi_c.sample(**sample_kwargs),
      }

      gamma_gc_dev = backend.tfd.Sample(
          backend.tfd.Normal(0, 1),
          [mmm.n_geos, mmm.n_controls],
          name=constants.GAMMA_GC_DEV,
      ).sample(**sample_kwargs)
      base_vars[constants.GAMMA_GC] = backend.tfd.Deterministic(
          base_vars[constants.GAMMA_C][..., backend.newaxis, :]
          + base_vars[constants.XI_C][..., backend.newaxis, :] * gamma_gc_dev,
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
        if mmm.non_media_treatments_normalized is not None
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

  def __call__(self, n_draws: int, seed: int | None = None) -> None:
    """Draws samples from prior distributions.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_draws = self._sample_prior(n_draws=n_draws, seed=seed)
    # Create Arviz InferenceData for prior draws.
    prior_coords = self._meridian.create_inference_data_coords(1, n_draws)
    prior_dims = self._meridian.create_inference_data_dims()
    prior_inference_data = az.convert_to_inference_data(
        prior_draws, coords=prior_coords, dims=prior_dims, group=constants.PRIOR
    )
    self._meridian.inference_data.extend(prior_inference_data, join="right")
