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

"""Module for MCMC sampling of posterior distributions in a Meridian model."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import arviz as az
from meridian import constants
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "MCMCSamplingError",
    "MCMCOOMError",
    "PosteriorMCMCSampler",
]


class MCMCSamplingError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling failed."""


class MCMCOOMError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling exceeds memory limits."""


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


class PosteriorMCMCSampler:
  """A callable that samples from posterior distributions using MCMC."""

  def __init__(self, meridian: "model.Meridian"):
    self._meridian = meridian

  @property
  def model(self) -> "model.Meridian":
    return self._meridian

  def _get_joint_dist_unpinned(self) -> tfp.distributions.Distribution:
    """Returns a `JointDistributionCoroutineAutoBatched` function for MCMC."""
    mmm = self.model
    mmm.populate_cached_properties()

    # This lists all the derived properties and states of this Meridian object
    # that are referenced by the joint distribution coroutine.
    # That is, these are the list of captured parameters.
    prior_broadcast = mmm.prior_broadcast
    baseline_geo_idx = mmm.baseline_geo_idx
    knot_info = mmm.knot_info
    n_geos = mmm.n_geos
    n_times = mmm.n_times
    n_media_channels = mmm.n_media_channels
    n_rf_channels = mmm.n_rf_channels
    n_organic_media_channels = mmm.n_organic_media_channels
    n_organic_rf_channels = mmm.n_organic_rf_channels
    n_controls = mmm.n_controls
    n_non_media_channels = mmm.n_non_media_channels
    holdout_id = mmm.holdout_id
    media_tensors = mmm.media_tensors
    rf_tensors = mmm.rf_tensors
    organic_media_tensors = mmm.organic_media_tensors
    organic_rf_tensors = mmm.organic_rf_tensors
    controls_scaled = mmm.controls_scaled
    non_media_treatments_normalized = mmm.non_media_treatments_normalized
    media_effects_dist = mmm.media_effects_dist
    adstock_hill_media_fn = mmm.adstock_hill_media
    adstock_hill_rf_fn = mmm.adstock_hill_rf
    total_outcome = mmm.total_outcome

    @tfp.distributions.JointDistributionCoroutineAutoBatched
    def joint_dist_unpinned():
      # Sample directly from prior.
      knot_values = yield prior_broadcast.knot_values
      sigma = yield prior_broadcast.sigma

      tau_g_excl_baseline = yield tfp.distributions.Sample(
          prior_broadcast.tau_g_excl_baseline,
          name=constants.TAU_G_EXCL_BASELINE,
      )
      tau_g = yield _get_tau_g(
          tau_g_excl_baseline=tau_g_excl_baseline,
          baseline_geo_idx=baseline_geo_idx,
      )
      mu_t = yield tfp.distributions.Deterministic(
          tf.einsum(
              "k,kt->t",
              knot_values,
              tf.convert_to_tensor(knot_info.weights),
          ),
          name=constants.MU_T,
      )

      tau_gt = tau_g[:, tf.newaxis] + mu_t
      combined_media_transformed = tf.zeros(
          shape=(n_geos, n_times, 0), dtype=tf.float32
      )
      combined_beta = tf.zeros(shape=(n_geos, 0), dtype=tf.float32)
      if media_tensors.media is not None:
        alpha_m = yield prior_broadcast.alpha_m
        ec_m = yield prior_broadcast.ec_m
        eta_m = yield prior_broadcast.eta_m
        slope_m = yield prior_broadcast.slope_m
        beta_gm_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_media_channels],
            name=constants.BETA_GM_DEV,
        )
        media_transformed = adstock_hill_media_fn(
            media=media_tensors.media_scaled,
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
        )
        prior_type = mmm.model_spec.effective_media_prior_type
        if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          beta_m = yield prior_broadcast.beta_m
        else:
          if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
            treatment_parameter_m = yield prior_broadcast.roi_m
          elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
            treatment_parameter_m = yield prior_broadcast.mroi_m
          elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
            treatment_parameter_m = yield prior_broadcast.contribution_m
          else:
            raise ValueError(f"Unsupported prior type: {prior_type}")
          incremental_outcome_m = (
              treatment_parameter_m * media_tensors.prior_denominator
          )
          linear_predictor_counterfactual_difference = (
              mmm.linear_predictor_counterfactual_difference_media(
                  media_transformed=media_transformed,
                  alpha_m=alpha_m,
                  ec_m=ec_m,
                  slope_m=slope_m,
              )
          )
          beta_m_value = mmm.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome_m,
              linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
              eta_x=eta_m,
              beta_gx_dev=beta_gm_dev,
          )
          beta_m = yield tfp.distributions.Deterministic(
              beta_m_value, name=constants.BETA_M
          )

        beta_eta_combined = beta_m + eta_m * beta_gm_dev
        beta_gm_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_gm = yield tfp.distributions.Deterministic(
            beta_gm_value, name=constants.BETA_GM
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, media_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gm], axis=-1)

      if rf_tensors.reach is not None:
        alpha_rf = yield prior_broadcast.alpha_rf
        ec_rf = yield prior_broadcast.ec_rf
        eta_rf = yield prior_broadcast.eta_rf
        slope_rf = yield prior_broadcast.slope_rf
        beta_grf_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_rf_channels],
            name=constants.BETA_GRF_DEV,
        )
        rf_transformed = adstock_hill_rf_fn(
            reach=rf_tensors.reach_scaled,
            frequency=rf_tensors.frequency,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
        )

        prior_type = mmm.model_spec.effective_rf_prior_type
        if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          beta_rf = yield prior_broadcast.beta_rf
        else:
          if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
            treatment_parameter_rf = yield prior_broadcast.roi_rf
          elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
            treatment_parameter_rf = yield prior_broadcast.mroi_rf
          elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
            treatment_parameter_rf = yield prior_broadcast.contribution_rf
          else:
            raise ValueError(f"Unsupported prior type: {prior_type}")
          incremental_outcome_rf = (
              treatment_parameter_rf * rf_tensors.prior_denominator
          )
          linear_predictor_counterfactual_difference = (
              mmm.linear_predictor_counterfactual_difference_rf(
                  rf_transformed=rf_transformed,
                  alpha_rf=alpha_rf,
                  ec_rf=ec_rf,
                  slope_rf=slope_rf,
              )
          )
          beta_rf_value = mmm.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome_rf,
              linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
              eta_x=eta_rf,
              beta_gx_dev=beta_grf_dev,
          )
          beta_rf = yield tfp.distributions.Deterministic(
              beta_rf_value, name=constants.BETA_RF
          )

        beta_eta_combined = beta_rf + eta_rf * beta_grf_dev
        beta_grf_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_grf = yield tfp.distributions.Deterministic(
            beta_grf_value, name=constants.BETA_GRF
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, rf_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_grf], axis=-1)

      if organic_media_tensors.organic_media is not None:
        alpha_om = yield prior_broadcast.alpha_om
        ec_om = yield prior_broadcast.ec_om
        eta_om = yield prior_broadcast.eta_om
        slope_om = yield prior_broadcast.slope_om
        beta_gom_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_organic_media_channels],
            name=constants.BETA_GOM_DEV,
        )
        organic_media_transformed = adstock_hill_media_fn(
            media=organic_media_tensors.organic_media_scaled,
            alpha=alpha_om,
            ec=ec_om,
            slope=slope_om,
        )
        prior_type = mmm.model_spec.organic_media_prior_type
        if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          beta_om = yield prior_broadcast.beta_om
        elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
          contribution_om = yield prior_broadcast.contribution_om
          incremental_outcome_om = contribution_om * total_outcome
          beta_om_value = mmm.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome_om,
              linear_predictor_counterfactual_difference=organic_media_transformed,
              eta_x=eta_om,
              beta_gx_dev=beta_gom_dev,
          )
          beta_om = yield tfp.distributions.Deterministic(
              beta_om_value, name=constants.BETA_OM
          )
        else:
          raise ValueError(f"Unsupported prior type: {prior_type}")

        beta_eta_combined = beta_om + eta_om * beta_gom_dev
        beta_gom_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_gom = yield tfp.distributions.Deterministic(
            beta_gom_value, name=constants.BETA_GOM
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, organic_media_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gom], axis=-1)

      if organic_rf_tensors.organic_reach is not None:
        alpha_orf = yield prior_broadcast.alpha_orf
        ec_orf = yield prior_broadcast.ec_orf
        eta_orf = yield prior_broadcast.eta_orf
        slope_orf = yield prior_broadcast.slope_orf
        beta_gorf_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_organic_rf_channels],
            name=constants.BETA_GORF_DEV,
        )
        organic_rf_transformed = adstock_hill_rf_fn(
            reach=organic_rf_tensors.organic_reach_scaled,
            frequency=organic_rf_tensors.organic_frequency,
            alpha=alpha_orf,
            ec=ec_orf,
            slope=slope_orf,
        )

        prior_type = mmm.model_spec.organic_rf_prior_type
        if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          beta_orf = yield prior_broadcast.beta_orf
        elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
          contribution_orf = yield prior_broadcast.contribution_orf
          incremental_outcome_orf = contribution_orf * total_outcome
          beta_orf_value = mmm.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome_orf,
              linear_predictor_counterfactual_difference=organic_rf_transformed,
              eta_x=eta_orf,
              beta_gx_dev=beta_gorf_dev,
          )
          beta_orf = yield tfp.distributions.Deterministic(
              beta_orf_value, name=constants.BETA_ORF
          )
        else:
          raise ValueError(f"Unsupported prior type: {prior_type}")

        beta_eta_combined = beta_orf + eta_orf * beta_gorf_dev
        beta_gorf_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else tf.math.exp(beta_eta_combined)
        )
        beta_gorf = yield tfp.distributions.Deterministic(
            beta_gorf_value, name=constants.BETA_GORF
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, organic_rf_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_gorf], axis=-1)

      sigma_gt = tf.transpose(tf.broadcast_to(sigma, [n_times, n_geos]))
      y_pred_combined_media = tau_gt + tf.einsum(
          "gtm,gm->gt", combined_media_transformed, combined_beta
      )
      # Omit gamma_c, xi_c, and gamma_gc from joint distribution output if
      # there are no control variables in the model.
      if n_controls:
        gamma_c = yield prior_broadcast.gamma_c
        xi_c = yield prior_broadcast.xi_c
        gamma_gc_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_controls],
            name=constants.GAMMA_GC_DEV,
        )
        gamma_gc = yield tfp.distributions.Deterministic(
            gamma_c + xi_c * gamma_gc_dev, name=constants.GAMMA_GC
        )
        y_pred_combined_media += tf.einsum(
            "gtc,gc->gt", controls_scaled, gamma_gc
        )

      if mmm.non_media_treatments is not None:
        xi_n = yield prior_broadcast.xi_n
        gamma_gn_dev = yield tfp.distributions.Sample(
            tfp.distributions.Normal(0, 1),
            [n_geos, n_non_media_channels],
            name=constants.GAMMA_GN_DEV,
        )
        prior_type = mmm.model_spec.non_media_treatments_prior_type
        if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          gamma_n = yield prior_broadcast.gamma_n
        elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
          contribution_n = yield prior_broadcast.contribution_n
          incremental_outcome_n = contribution_n * total_outcome
          baseline_scaled = mmm.non_media_transformer.forward(  # pytype: disable=attribute-error
              mmm.compute_non_media_treatments_baseline()
          )
          linear_predictor_counterfactual_difference = (
              non_media_treatments_normalized - baseline_scaled
          )
          gamma_n_value = mmm.calculate_beta_x(
              is_non_media=True,
              incremental_outcome_x=incremental_outcome_n,
              linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
              eta_x=xi_n,
              beta_gx_dev=gamma_gn_dev,
          )
          gamma_n = yield tfp.distributions.Deterministic(
              gamma_n_value, name=constants.GAMMA_N
          )
        else:
          raise ValueError(f"Unsupported prior type: {prior_type}")
        gamma_gn = yield tfp.distributions.Deterministic(
            gamma_n + xi_n * gamma_gn_dev, name=constants.GAMMA_GN
        )
        y_pred = y_pred_combined_media + tf.einsum(
            "gtn,gn->gt", non_media_treatments_normalized, gamma_gn
        )
      else:
        y_pred = y_pred_combined_media

      # If there are any holdout observations, the holdout KPI values will
      # be replaced with zeros using `experimental_pin`. For these
      # observations, we set the posterior mean equal to zero and standard
      # deviation to `1/sqrt(2pi)`, so the log-density is 0 regardless of the
      # sampled posterior parameter values.
      if holdout_id is not None:
        y_pred_holdout = tf.where(holdout_id, 0.0, y_pred)
        test_sd = tf.cast(1.0 / np.sqrt(2.0 * np.pi), tf.float32)
        sigma_gt_holdout = tf.where(holdout_id, test_sd, sigma_gt)
        yield tfp.distributions.Normal(
            y_pred_holdout, sigma_gt_holdout, name="y"
        )
      else:
        yield tfp.distributions.Normal(y_pred, sigma_gt, name="y")

    return joint_dist_unpinned

  def _get_joint_dist(self) -> tfp.distributions.Distribution:
    mmm = self.model
    y = (
        tf.where(mmm.holdout_id, 0.0, mmm.kpi_scaled)
        if mmm.holdout_id is not None
        else mmm.kpi_scaled
    )
    return self._get_joint_dist_unpinned().experimental_pin(y=y)

  def __call__(
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
      seed: Sequence[int] | int | None = None,
      **pins,
  ) -> None:
    """Runs Markov Chain Monte Carlo (MCMC) sampling of posterior distributions.

    For more information about the arguments, see [`windowed_adaptive_nuts`]
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
      seed: An `int32[2]` Tensor or a Python list or tuple of 2 `int`s, which
        will be treated as stateless seeds; or a Python `int` or `None`, which
        will be converted into a stateless seed. See [tfp.random.sanitize_seed]
        (https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed).
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially. For more
        information, see
        [ResourceExhaustedError when running Meridian.sample_posterior]
        (https://developers.google.com/meridian/docs/advanced-modeling/model-debugging#gpu-oom-error).
    """
    if seed is not None and isinstance(seed, Sequence) and len(seed) != 2:
      raise ValueError(
          "Invalid seed: Must be either a single integer (stateful seed) or a"
          " pair of two integers (stateless seed). See"
          " [tfp.random.sanitize_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed)"
          " for details."
      )
    if seed is not None and isinstance(seed, int):
      seed = (seed, seed)
    seed = tfp.random.sanitize_seed(seed) if seed is not None else None
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
            " https://developers.google.com/meridian/docs/advanced-modeling/model-debugging#gpu-oom-error)"
        ) from error
      if seed is not None:
        seed += 1
      states.append(mcmc.all_states._asdict())
      traces.append(mcmc.trace)

    mcmc_states = {
        k: tf.einsum(
            "ij...->ji...",
            tf.concat([state[k] for state in states], axis=1)[n_burnin:, ...],
        )
        for k in states[0].keys()
        if k not in constants.UNSAVED_PARAMETERS
    }
    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self.model.create_inference_data_coords(
        total_chains, n_keep
    )
    posterior_dims = self.model.create_inference_data_dims()
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
    self.model.inference_data.extend(posterior_inference_data, join="right")
