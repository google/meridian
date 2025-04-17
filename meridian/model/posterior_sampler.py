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

"""Module for MCMC sampling of posterior distributions in a Meridian model."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import arviz as az
from meridian import constants
import numpy as np
from meridian import backend

if TYPE_CHECKING:
  # Keep this type check import separate
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
    tau_g_excl_baseline: backend.Tensor, baseline_geo_idx: int
) -> backend.tfp_distributions.Distribution:
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
  shape = backend.shape(tau_g_excl_baseline)[:-1] + [1] if rank != 1 else 1 # Use backend.shape
  tau_g = backend.concat(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          backend.zeros(shape, dtype=tau_g_excl_baseline.dtype), # Use backend.zeros
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=rank - 1,
  )
  return backend.tfp_distributions.Deterministic(tau_g, name="tau_g")


# This function is now defined in the backend, no need to redefine here.
# @backend.function(autograph=False, jit_compile=True) # Assuming tf.function maps to backend.function
# def _xla_windowed_adaptive_nuts(**kwargs):
#   """XLA wrapper for windowed_adaptive_nuts."""
#   return backend.tfp_experimental_mcmc.windowed_adaptive_nuts(**kwargs) # Use backend.tfp_experimental_mcmc


class PosteriorMCMCSampler:
  """A callable that samples from posterior distributions using MCMC."""

  def __init__(self, meridian: "model.Meridian"):
    self._meridian = meridian

  def _get_joint_dist_unpinned(self) -> backend.tfp_distributions.Distribution:
    """Returns a `JointDistributionCoroutineAutoBatched` function for MCMC."""
    mmm = self._meridian
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
    non_media_treatments_scaled = mmm.non_media_treatments_scaled
    media_effects_dist = mmm.media_effects_dist
    adstock_hill_media_fn = mmm.adstock_hill_media
    adstock_hill_rf_fn = mmm.adstock_hill_rf
    get_roi_prior_beta_m_value_fn = (
        mmm.prior_sampler_callable.get_roi_prior_beta_m_value
    )
    get_roi_prior_beta_rf_value_fn = (
        mmm.prior_sampler_callable.get_roi_prior_beta_rf_value
    )

    @backend.tfp_distributions.JointDistributionCoroutineAutoBatched # Use backend.tfp_distributions
    def joint_dist_unpinned():
      # Sample directly from prior.
      knot_values = yield prior_broadcast.knot_values
      gamma_c = yield prior_broadcast.gamma_c
      xi_c = yield prior_broadcast.xi_c
      sigma = yield prior_broadcast.sigma

      tau_g_excl_baseline = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
          prior_broadcast.tau_g_excl_baseline,
          name=constants.TAU_G_EXCL_BASELINE,
      )
      tau_g = yield _get_tau_g(
          tau_g_excl_baseline=tau_g_excl_baseline,
          baseline_geo_idx=baseline_geo_idx,
      )
      mu_t = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
          backend.einsum( # Use backend.einsum
              "k,kt->t",
              knot_values,
              backend.convert_to_tensor(knot_info.weights), # Use backend.convert_to_tensor
          ),
          name=constants.MU_T,
      )

      tau_gt = tau_g[:, backend.newaxis] + mu_t # Use backend.newaxis
      combined_media_transformed = backend.zeros( # Use backend.zeros
          shape=(n_geos, n_times, 0), dtype=backend.float32 # Use backend.float32
      )
      combined_beta = backend.zeros(shape=(n_geos, 0), dtype=backend.float32) # Use backend.zeros, backend.float32
      if media_tensors.media is not None:
        alpha_m = yield prior_broadcast.alpha_m
        ec_m = yield prior_broadcast.ec_m
        eta_m = yield prior_broadcast.eta_m
        slope_m = yield prior_broadcast.slope_m
        beta_gm_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
            backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
            [n_geos, n_media_channels],
            name=constants.BETA_GM_DEV,
        )
        media_transformed = adstock_hill_media_fn(
            media=media_tensors.media_scaled,
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
        )
        prior_type = mmm.model_spec.paid_media_prior_type
        if prior_type in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
          if prior_type == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
            roi_or_mroi_m = yield prior_broadcast.roi_m
          else:
            roi_or_mroi_m = yield prior_broadcast.mroi_m
          beta_m_value = get_roi_prior_beta_m_value_fn(
              alpha_m,
              beta_gm_dev,
              ec_m,
              eta_m,
              roi_or_mroi_m,
              slope_m,
              media_transformed,
          )
          beta_m = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
              beta_m_value, name=constants.BETA_M
          )
        else:
          beta_m = yield prior_broadcast.beta_m

        beta_eta_combined = beta_m + eta_m * beta_gm_dev
        beta_gm_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else backend.exp(beta_eta_combined) # Use backend.exp
        )
        beta_gm = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
            beta_gm_value, name=constants.BETA_GM
        )
        combined_media_transformed = backend.concat( # Use backend.concat
            [combined_media_transformed, media_transformed], axis=-1
        )
        combined_beta = backend.concat([combined_beta, beta_gm], axis=-1) # Use backend.concat

      if rf_tensors.reach is not None:
        alpha_rf = yield prior_broadcast.alpha_rf
        ec_rf = yield prior_broadcast.ec_rf
        eta_rf = yield prior_broadcast.eta_rf
        slope_rf = yield prior_broadcast.slope_rf
        beta_grf_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
            backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
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

        prior_type = mmm.model_spec.paid_media_prior_type
        if prior_type in constants.PAID_MEDIA_ROI_PRIOR_TYPES:
          if prior_type == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
            roi_or_mroi_rf = yield prior_broadcast.roi_rf
          else:
            roi_or_mroi_rf = yield prior_broadcast.mroi_rf
          beta_rf_value = get_roi_prior_beta_rf_value_fn(
              alpha_rf,
              beta_grf_dev,
              ec_rf,
              eta_rf,
              roi_or_mroi_rf,
              slope_rf,
              rf_transformed,
          )
          beta_rf = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
              beta_rf_value,
              name=constants.BETA_RF,
          )
        else:
          beta_rf = yield prior_broadcast.beta_rf

        beta_eta_combined = beta_rf + eta_rf * beta_grf_dev
        beta_grf_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else backend.exp(beta_eta_combined) # Use backend.exp
        )
        beta_grf = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
            beta_grf_value, name=constants.BETA_GRF
        )
        combined_media_transformed = backend.concat( # Use backend.concat
            [combined_media_transformed, rf_transformed], axis=-1
        )
        combined_beta = backend.concat([combined_beta, beta_grf], axis=-1) # Use backend.concat

      if organic_media_tensors.organic_media is not None:
        alpha_om = yield prior_broadcast.alpha_om
        ec_om = yield prior_broadcast.ec_om
        eta_om = yield prior_broadcast.eta_om
        slope_om = yield prior_broadcast.slope_om
        beta_gom_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
            backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
            [n_geos, n_organic_media_channels],
            name=constants.BETA_GOM_DEV,
        )
        organic_media_transformed = adstock_hill_media_fn(
            media=organic_media_tensors.organic_media_scaled,
            alpha=alpha_om,
            ec=ec_om,
            slope=slope_om,
        )
        beta_om = yield prior_broadcast.beta_om

        beta_eta_combined = beta_om + eta_om * beta_gom_dev
        beta_gom_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else backend.exp(beta_eta_combined) # Use backend.exp
        )
        beta_gom = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
            beta_gom_value, name=constants.BETA_GOM
        )
        combined_media_transformed = backend.concat( # Use backend.concat
            [combined_media_transformed, organic_media_transformed], axis=-1
        )
        combined_beta = backend.concat([combined_beta, beta_gom], axis=-1) # Use backend.concat

      if organic_rf_tensors.organic_reach is not None:
        alpha_orf = yield prior_broadcast.alpha_orf
        ec_orf = yield prior_broadcast.ec_orf
        eta_orf = yield prior_broadcast.eta_orf
        slope_orf = yield prior_broadcast.slope_orf
        beta_gorf_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
            backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
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

        beta_orf = yield prior_broadcast.beta_orf

        beta_eta_combined = beta_orf + eta_orf * beta_gorf_dev
        beta_gorf_value = (
            beta_eta_combined
            if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
            else backend.exp(beta_eta_combined) # Use backend.exp
        )
        beta_gorf = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
            beta_gorf_value, name=constants.BETA_GORF
        )
        combined_media_transformed = backend.concat( # Use backend.concat
            [combined_media_transformed, organic_rf_transformed], axis=-1
        )
        combined_beta = backend.concat([combined_beta, beta_gorf], axis=-1) # Use backend.concat

      sigma_gt = backend.transpose(backend.broadcast_to(sigma, [n_times, n_geos])) # Use backend.transpose, backend.broadcast_to
      gamma_gc_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
          backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
          [n_geos, n_controls],
          name=constants.GAMMA_GC_DEV,
      )
      gamma_gc = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
          gamma_c + xi_c * gamma_gc_dev, name=constants.GAMMA_GC
      )
      y_pred_combined_media = (
          tau_gt
          + backend.einsum("gtm,gm->gt", combined_media_transformed, combined_beta) # Use backend.einsum
          + backend.einsum("gtc,gc->gt", controls_scaled, gamma_gc) # Use backend.einsum
      )

      if mmm.non_media_treatments is not None:
        gamma_n = yield prior_broadcast.gamma_n
        xi_n = yield prior_broadcast.xi_n
        gamma_gn_dev = yield backend.tfp_distributions.Sample( # Use backend.tfp_distributions
            backend.tfp_distributions.Normal(0, 1), # Use backend.tfp_distributions
            [n_geos, n_non_media_channels],
            name=constants.GAMMA_GN_DEV,
        )
        gamma_gn = yield backend.tfp_distributions.Deterministic( # Use backend.tfp_distributions
            gamma_n + xi_n * gamma_gn_dev, name=constants.GAMMA_GN
        )
        y_pred = y_pred_combined_media + backend.einsum( # Use backend.einsum
            "gtn,gn->gt", non_media_treatments_scaled, gamma_gn
        )
      else:
        y_pred = y_pred_combined_media

      # If there are any holdout observations, the holdout KPI values will
      # be replaced with zeros using `experimental_pin`. For these
      # observations, we set the posterior mean equal to zero and standard
      # deviation to `1/sqrt(2pi)`, so the log-density is 0 regardless of the
      # sampled posterior parameter values.
      if holdout_id is not None:
        y_pred_holdout = backend.where(holdout_id, 0.0, y_pred) # Use backend.where
        test_sd = backend.cast(1.0 / np.sqrt(2.0 * np.pi), backend.float32) # Use backend.cast, backend.float32
        sigma_gt_holdout = backend.where(holdout_id, test_sd, sigma_gt) # Use backend.where
        yield backend.tfp_distributions.Normal( # Use backend.tfp_distributions
            y_pred_holdout, sigma_gt_holdout, name="y"
        )
      else:
        yield backend.tfp_distributions.Normal(y_pred, sigma_gt, name="y") # Use backend.tfp_distributions

    return joint_dist_unpinned

  def _get_joint_dist(self) -> backend.tfp_distributions.Distribution:
  def _get_joint_dist(self) -> backend.tfp_distributions.Distribution:
    mmm = self._meridian
    y = (
        backend.where(mmm.holdout_id, 0.0, mmm.kpi_scaled) # Use backend.where
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
      current_state: Mapping[str, backend.Tensor] | None = None, # Use backend.Tensor
      init_step_size: int | None = None,
      dual_averaging_kwargs: Mapping[str, int] | None = None,
      max_tree_depth: int = 10,
      max_energy_diff: float = 500.0,
      unrolled_leapfrog_steps: int = 1,
      parallel_iterations: int = 10,
      seed: Sequence[int] | int | None = None,
      **pins,
  ) -> az.InferenceData:
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
        will be treated as stateful seeds. See [tfp.random.sanitize_seed]
        https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed). # Needs backend.tfp_random
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Returns:
      An Arviz `InferenceData` object containing posterior samples only.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially. For more
        information, see
        [backend.ResourceExhaustedError when running Meridian.sample_posterior] # Use backend.ResourceExhaustedError
        (https://developers.google.com/meridian/docs/advanced-modeling/model-debugging#gpu-oom-error).
    """
    if seed is not None and isinstance(seed, Sequence) and len(seed) != 2:
      raise ValueError(
          "Invalid seed: Must be either a single integer (stateful seed) or a"
          " pair of two integers (stateless seed). See"
          " [backend.tfp_random.sanitize_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed)" # Use backend.tfp_random
          " for details."
      )
    seed = backend.tfp_random.sanitize_seed(seed) if seed is not None else None # Use backend.tfp_random
    n_chains_list = [n_chains] if isinstance(n_chains, int) else n_chains
    total_chains = np.sum(n_chains_list)

    states = []
    traces = []
    for n_chains_batch in n_chains_list:
      try:
        # Use backend.xla_windowed_adaptive_nuts directly
        mcmc = backend.xla_windowed_adaptive_nuts(
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
      except backend.ResourceExhaustedError as error: # Use backend.ResourceExhaustedError
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
        k: backend.einsum( # Use backend.einsum
            "ij...->ji...",
            backend.concat([state[k] for state in states], axis=1)[n_burnin:, ...], # Use backend.concat
        )
        for k in states[0].keys()
        if k not in constants.UNSAVED_PARAMETERS
    }
    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self._meridian.create_inference_data_coords(
        total_chains, n_keep
    )
    posterior_dims = self._meridian.create_inference_data_dims()
    infdata_posterior = az.convert_to_inference_data(
        mcmc_states, coords=posterior_coords, dims=posterior_dims
    )

    # Save trace metrics in InferenceData.
    mcmc_trace = {}
    for k in traces[0].keys():
      if k not in constants.IGNORED_TRACE_METRICS:
        mcmc_trace[k] = backend.concat( # Use backend.concat
            [
                backend.broadcast_to( # Use backend.broadcast_to
                    backend.transpose(trace[k][n_burnin:, ...]), # Use backend.transpose
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
    sample_stats[constants.STEP_SIZE] = backend.broadcast_to( # Use backend.broadcast_to
        sample_stats[constants.STEP_SIZE], [total_chains, n_keep]
    )
    sample_stats_dims[constants.STEP_SIZE] = [constants.CHAIN, constants.DRAW]
    infdata_sample_stats = az.convert_to_inference_data(
        sample_stats,
        coords=trace_coords,
        dims=sample_stats_dims,
        group="sample_stats",
    )
    return az.concat(infdata_posterior, infdata_trace, infdata_sample_stats)
