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

"""Module for MCMC sampling of posterior distributions in a Meridian model."""

from collections.abc import Generator, Mapping, Sequence
import functools
from typing import Any, Optional, TYPE_CHECKING
import warnings

import arviz as az
from meridian import backend
from meridian import constants
from meridian.model import context
from meridian.model import equations
import numpy as np

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


def _compute_tau_g(
    tau_g_excl_baseline: backend.Tensor, baseline_geo_idx: int
) -> backend.Tensor:
  """Computes `tau_g` from `tau_g_excl_baseline`.

  This function computes `tau_g` by inserting zeros at the `baseline_geo_idx`
  position along the last dimension of `tau_g_excl_baseline`.

  Args:
    tau_g_excl_baseline: A tensor of shape `[..., n_geos - 1]` representing the
      `tau_g` parameters excluding the baseline geo.
    baseline_geo_idx: The index of the baseline geo to be set to zero.

  Returns:
    A tensor of shape `[..., n_geos]` containing the `tau_g` values, with
    zero at `baseline_geo_idx` and values from `tau_g_excl_baseline`
    elsewhere.
  """
  rank = len(tau_g_excl_baseline.shape)
  shape = tau_g_excl_baseline.shape[:-1] + [1] if rank != 1 else [1]
  return backend.concatenate(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          backend.zeros(shape, dtype=tau_g_excl_baseline.dtype),
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=rank - 1,
  )


def _get_tau_g(
    tau_g_excl_baseline: backend.Tensor, baseline_geo_idx: int
) -> backend.tfd.Distribution:
  """Creates a deterministic distribution for `tau_g`.

  This wraps the computation of `tau_g` (inserting zeros for the baseline geo)
  in a `tfd.Deterministic` distribution, allowing it to be tracked as a
  named node in the probabilistic model graph.

  Args:
    tau_g_excl_baseline: A tensor of shape `[..., n_geos - 1]` representing the
      `tau_g` parameters excluding the baseline geo.
    baseline_geo_idx: The index of the baseline geo to be set to zero.

  Returns:
    A `tfd.Deterministic` distribution yielding the full `tau_g` tensor
    of shape `[..., n_geos]`.
  """
  tau_g = _compute_tau_g(tau_g_excl_baseline, baseline_geo_idx)
  return backend.tfd.Deterministic(tau_g, name="tau_g")


def _joint_dist_base_logic(
    model_context: context.ModelContext,
    model_equations: equations.ModelEquations,
    yield_deterministics: bool,
) -> Generator[backend.Tensor | backend.tfd.Distribution, Any, None]:
  """Shared logic for joint distribution definition.

  This generator defines the probabilistic model. It conditionally yields
  Deterministic nodes based on `yield_deterministics`. When False, intermediate
  values are computed as raw tensors (Sampling Graph) to minimize Autograd tape
  overhead. When True, they are yielded as distributions (Full Graph) to allow
  state reconstruction.

  Args:
    model_context: An instance of `context.ModelContext` containing model data
      and settings.
    model_equations: An instance of `equations.ModelEquations` providing the
      model's mathematical equations.
    yield_deterministics: If True, yields `backend.tfd.Deterministic`
      distributions for deterministic variables. If False, yields raw
      `backend.Tensor` values.

  Yields:
    A `backend.Tensor` when `yield_deterministics` is False, representing
    intermediate values in the sampling graph.
    A `backend.tfd.Distribution` when `yield_deterministics` is True,
    representing nodes in the full probabilistic model graph.
  """
  # This lists all the derived properties and states of this Meridian object
  # that are referenced by the joint distribution coroutine.
  # That is, these are the list of captured parameters.
  prior_broadcast = model_context.prior_broadcast
  baseline_geo_idx = model_context.baseline_geo_idx
  knot_info = model_context.knot_info
  n_geos = model_context.n_geos
  n_times = model_context.n_times
  n_media_channels = model_context.n_media_channels
  n_rf_channels = model_context.n_rf_channels
  n_organic_media_channels = model_context.n_organic_media_channels
  n_organic_rf_channels = model_context.n_organic_rf_channels
  n_controls = model_context.n_controls
  n_non_media_channels = model_context.n_non_media_channels
  holdout_id = model_context.holdout_id
  media_tensors = model_context.media_tensors
  rf_tensors = model_context.rf_tensors
  organic_media_tensors = model_context.organic_media_tensors
  organic_rf_tensors = model_context.organic_rf_tensors
  controls_scaled = model_context.controls_scaled
  non_media_treatments_normalized = (
      model_context.non_media_treatments_normalized
  )
  media_effects_dist = model_context.media_effects_dist
  adstock_hill_media_fn = model_equations.adstock_hill_media
  adstock_hill_rf_fn = model_equations.adstock_hill_rf
  total_outcome = model_context.total_outcome

  # Sample directly from prior.
  knot_values = yield prior_broadcast.knot_values
  sigma = yield prior_broadcast.sigma

  tau_g_excl_baseline = yield backend.tfd.Sample(
      prior_broadcast.tau_g_excl_baseline,
      name=constants.TAU_G_EXCL_BASELINE,
  )

  # Deterministic: tau_g
  tau_g = _compute_tau_g(tau_g_excl_baseline, baseline_geo_idx)
  if yield_deterministics:
    yield backend.tfd.Deterministic(tau_g, name="tau_g")

  # Deterministic: mu_t
  mu_t = backend.einsum(
      "k,kt->t",
      knot_values,
      backend.to_tensor(knot_info.weights),
  )
  if yield_deterministics:
    yield backend.tfd.Deterministic(mu_t, name=constants.MU_T)

  # Robust broadcasting:
  # tau_g (..., G) -> (..., G, 1)
  # mu_t  (..., T) -> (..., 1, T)
  # result (..., G, T)
  tau_gt = backend.expand_dims(tau_g, -1) + backend.expand_dims(mu_t, -2)

  # Accumulate media tensors in lists to avoid empty tensor shape/rank issues
  media_transformed_list = []
  beta_list = []

  if media_tensors.media is not None:
    alpha_m = yield prior_broadcast.alpha_m
    ec_m = yield prior_broadcast.ec_m
    eta_m = yield prior_broadcast.eta_m
    slope_m = yield prior_broadcast.slope_m
    beta_gm_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_media_channels],
        name=constants.BETA_GM_DEV,
    )
    media_transformed = adstock_hill_media_fn(
        media=media_tensors.media_scaled,
        alpha=alpha_m,
        ec=ec_m,
        slope=slope_m,
        decay_functions=model_context.adstock_decay_spec.media,
    )
    prior_type = model_context.model_spec.effective_media_prior_type
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
          model_equations.linear_predictor_counterfactual_difference_media(
              media_transformed=media_transformed,
              alpha_m=alpha_m,
              ec_m=ec_m,
              slope_m=slope_m,
          )
      )
      beta_m = model_equations.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_m,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=eta_m,
          beta_gx_dev=beta_gm_dev,
      )
      if yield_deterministics:
        yield backend.tfd.Deterministic(beta_m, name=constants.BETA_M)

    beta_eta_combined = beta_m + eta_m * beta_gm_dev
    beta_gm = (
        beta_eta_combined
        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    if yield_deterministics:
      yield backend.tfd.Deterministic(beta_gm, name=constants.BETA_GM)

    media_transformed_list.append(media_transformed)
    beta_list.append(beta_gm)

  if rf_tensors.reach is not None:
    alpha_rf = yield prior_broadcast.alpha_rf
    ec_rf = yield prior_broadcast.ec_rf
    eta_rf = yield prior_broadcast.eta_rf
    slope_rf = yield prior_broadcast.slope_rf
    beta_grf_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_rf_channels],
        name=constants.BETA_GRF_DEV,
    )
    rf_transformed = adstock_hill_rf_fn(
        reach=rf_tensors.reach_scaled,
        frequency=rf_tensors.frequency,
        alpha=alpha_rf,
        ec=ec_rf,
        slope=slope_rf,
        decay_functions=model_context.adstock_decay_spec.rf,
    )

    prior_type = model_context.model_spec.effective_rf_prior_type
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
          model_equations.linear_predictor_counterfactual_difference_rf(
              rf_transformed=rf_transformed,
              alpha_rf=alpha_rf,
              ec_rf=ec_rf,
              slope_rf=slope_rf,
          )
      )
      beta_rf = model_equations.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_rf,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=eta_rf,
          beta_gx_dev=beta_grf_dev,
      )
      if yield_deterministics:
        yield backend.tfd.Deterministic(beta_rf, name=constants.BETA_RF)

    beta_eta_combined = beta_rf + eta_rf * beta_grf_dev
    beta_grf = (
        beta_eta_combined
        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    if yield_deterministics:
      yield backend.tfd.Deterministic(beta_grf, name=constants.BETA_GRF)

    media_transformed_list.append(rf_transformed)
    beta_list.append(beta_grf)

  if organic_media_tensors.organic_media is not None:
    alpha_om = yield prior_broadcast.alpha_om
    ec_om = yield prior_broadcast.ec_om
    eta_om = yield prior_broadcast.eta_om
    slope_om = yield prior_broadcast.slope_om
    beta_gom_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_organic_media_channels],
        name=constants.BETA_GOM_DEV,
    )
    organic_media_transformed = adstock_hill_media_fn(
        media=organic_media_tensors.organic_media_scaled,
        alpha=alpha_om,
        ec=ec_om,
        slope=slope_om,
        decay_functions=model_context.adstock_decay_spec.organic_media,
    )
    prior_type = model_context.model_spec.organic_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      beta_om = yield prior_broadcast.beta_om
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      contribution_om = yield prior_broadcast.contribution_om
      incremental_outcome_om = contribution_om * total_outcome
      beta_om = model_equations.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_om,
          linear_predictor_counterfactual_difference=organic_media_transformed,
          eta_x=eta_om,
          beta_gx_dev=beta_gom_dev,
      )
      if yield_deterministics:
        yield backend.tfd.Deterministic(beta_om, name=constants.BETA_OM)
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = beta_om + eta_om * beta_gom_dev
    beta_gom = (
        beta_eta_combined
        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    if yield_deterministics:
      yield backend.tfd.Deterministic(beta_gom, name=constants.BETA_GOM)

    media_transformed_list.append(organic_media_transformed)
    beta_list.append(beta_gom)

  if organic_rf_tensors.organic_reach is not None:
    alpha_orf = yield prior_broadcast.alpha_orf
    ec_orf = yield prior_broadcast.ec_orf
    eta_orf = yield prior_broadcast.eta_orf
    slope_orf = yield prior_broadcast.slope_orf
    beta_gorf_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_organic_rf_channels],
        name=constants.BETA_GORF_DEV,
    )
    organic_rf_transformed = adstock_hill_rf_fn(
        reach=organic_rf_tensors.organic_reach_scaled,
        frequency=organic_rf_tensors.organic_frequency,
        alpha=alpha_orf,
        ec=ec_orf,
        slope=slope_orf,
        decay_functions=model_context.adstock_decay_spec.organic_rf,
    )

    prior_type = model_context.model_spec.organic_rf_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      beta_orf = yield prior_broadcast.beta_orf
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      contribution_orf = yield prior_broadcast.contribution_orf
      incremental_outcome_orf = contribution_orf * total_outcome
      beta_orf = model_equations.calculate_beta_x(
          is_non_media=False,
          incremental_outcome_x=incremental_outcome_orf,
          linear_predictor_counterfactual_difference=organic_rf_transformed,
          eta_x=eta_orf,
          beta_gx_dev=beta_gorf_dev,
      )
      if yield_deterministics:
        yield backend.tfd.Deterministic(beta_orf, name=constants.BETA_ORF)
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = beta_orf + eta_orf * beta_gorf_dev
    beta_gorf = (
        beta_eta_combined
        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    if yield_deterministics:
      yield backend.tfd.Deterministic(beta_gorf, name=constants.BETA_GORF)

    media_transformed_list.append(organic_rf_transformed)
    beta_list.append(beta_gorf)

  # Calculate y_pred_combined_media
  if media_transformed_list:
    combined_media_transformed = backend.concatenate(
        media_transformed_list, axis=-1
    )
    combined_beta = backend.concatenate(beta_list, axis=-1)

    # Use ellipses in einsum for robust broadcasting across batch dims
    y_pred_combined_media = tau_gt + backend.einsum(
        "...gtm,...gm->...gt", combined_media_transformed, combined_beta
    )
  else:
    y_pred_combined_media = tau_gt

  # Omit gamma_c, xi_c, and gamma_gc from joint distribution output if
  # there are no control variables in the model.
  if n_controls:
    gamma_c = yield prior_broadcast.gamma_c
    xi_c = yield prior_broadcast.xi_c
    gamma_gc_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_controls],
        name=constants.GAMMA_GC_DEV,
    )
    gamma_gc = gamma_c + xi_c * gamma_gc_dev
    if yield_deterministics:
      yield backend.tfd.Deterministic(gamma_gc, name=constants.GAMMA_GC)

    y_pred_combined_media += backend.einsum(
        "...gtc,...gc->...gt", controls_scaled, gamma_gc
    )

  if model_context.non_media_treatments is not None:
    xi_n = yield prior_broadcast.xi_n
    gamma_gn_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(0, 1),
        [n_geos, n_non_media_channels],
        name=constants.GAMMA_GN_DEV,
    )
    prior_type = model_context.model_spec.non_media_treatments_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      gamma_n = yield prior_broadcast.gamma_n
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      contribution_n = yield prior_broadcast.contribution_n
      incremental_outcome_n = contribution_n * total_outcome
      baseline_scaled = model_context.non_media_transformer.forward(  # pytype: disable=attribute-error
          model_equations.compute_non_media_treatments_baseline()
      )
      linear_predictor_counterfactual_difference = (
          non_media_treatments_normalized - baseline_scaled
      )
      gamma_n = model_equations.calculate_beta_x(
          is_non_media=True,
          incremental_outcome_x=incremental_outcome_n,
          linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
          eta_x=xi_n,
          beta_gx_dev=gamma_gn_dev,
      )
      if yield_deterministics:
        yield backend.tfd.Deterministic(gamma_n, name=constants.GAMMA_N)
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    gamma_gn = gamma_n + xi_n * gamma_gn_dev
    if yield_deterministics:
      yield backend.tfd.Deterministic(gamma_gn, name=constants.GAMMA_GN)

    y_pred = y_pred_combined_media + backend.einsum(
        "...gtn,...gn->...gt", non_media_treatments_normalized, gamma_gn
    )
  else:
    y_pred = y_pred_combined_media

  sigma_gt = backend.transpose(backend.broadcast_to(sigma, [n_times, n_geos]))

  # If there are any holdout observations, the holdout KPI values will
  # be replaced with zeros using `experimental_pin`. For these
  # observations, we set the posterior mean equal to zero and standard
  # deviation to `1/sqrt(2pi)`, so the log-density is 0 regardless of the
  # sampled posterior parameter values.
  if holdout_id is not None:
    y_pred_holdout = backend.where(holdout_id, 0.0, y_pred)
    test_sd = backend.cast(1.0 / np.sqrt(2.0 * np.pi), backend.float32)
    sigma_gt_holdout = backend.where(holdout_id, test_sd, sigma_gt)
    yield backend.tfd.Normal(y_pred_holdout, sigma_gt_holdout, name="y")
  else:
    yield backend.tfd.Normal(y_pred, sigma_gt, name="y")


def _joint_dist_unpinned(
    model_context: context.ModelContext,
    model_equations: equations.ModelEquations,
):
  """Returns unpinned joint distribution."""
  return _joint_dist_base_logic(
      model_context, model_equations, yield_deterministics=True
  )


def _joint_dist_sampling(
    model_context: context.ModelContext,
    model_equations: equations.ModelEquations,
):
  """Returns sampling-optimized unpinned joint distribution."""
  return _joint_dist_base_logic(
      model_context, model_equations, yield_deterministics=False
  )


class PosteriorMCMCSampler:
  """A callable that samples from posterior distributions using MCMC."""

  # TODO: Deprecate direct injection of `model.Meridian`.
  def __init__(
      self,
      meridian: Optional["model.Meridian"] = None,
      *,
      model_context: context.ModelContext | None = None,
  ):
    if meridian is not None:
      warnings.warn(
          "Initializing PosteriorMCMCSampler with a Meridian object is"
          " deprecated and will be removed in a future version. Please use"
          " `model_context` instead.",
          DeprecationWarning,
          stacklevel=2,
      )
      self._meridian = meridian
      self._model_context = meridian.model_context
    elif model_context is not None:
      self._meridian = None
      self._model_context = model_context
    else:
      raise ValueError("Either `meridian` or `model_context` must be provided.")
    self._joint_dist = None
    self._joint_dist_sampling = None

  @functools.cached_property
  def _model_equations(self) -> equations.ModelEquations:
    return equations.ModelEquations(self._model_context)

  def __getstate__(self):
    state = self.__dict__.copy()
    # Exclude unpickleable objects.
    if "_joint_dist" in state:
      del state["_joint_dist"]
    if "_joint_dist_sampling" in state:
      del state["_joint_dist_sampling"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._joint_dist = None
    self._joint_dist_sampling = None

  # TODO: Remove this property in favor of using `model_context`
  # and `model_equations` directly.
  @property
  def model(self) -> Optional["model.Meridian"]:
    return self._meridian

  def _joint_dist_unpinned_fn(self):
    return _joint_dist_unpinned(self._model_context, self._model_equations)

  def _joint_dist_sampling_fn(self) -> Generator[Any, Any, None]:
    return _joint_dist_sampling(self._model_context, self._model_equations)

  def _get_joint_dist_unpinned(self) -> backend.tfd.Distribution:
    """Builds a `JointDistributionCoroutineAutoBatched` function for MCMC."""
    self._model_context.populate_cached_properties()
    return backend.tfd.JointDistributionCoroutineAutoBatched(
        self._joint_dist_unpinned_fn
    )

  def _get_joint_dist_sampling_unpinned(self) -> backend.tfd.Distribution:
    """Builds a sampling-optimized `JointDistributionCoroutineAutoBatched`."""
    self._model_context.populate_cached_properties()
    return backend.tfd.JointDistributionCoroutineAutoBatched(
        self._joint_dist_sampling_fn
    )

  def _pin_dist(
      self, dist: backend.tfd.Distribution
  ) -> backend.tfd.Distribution:
    if self._model_context.holdout_id is not None:
      y = backend.where(
          self._model_context.holdout_id,
          0.0,
          self._model_context.kpi_scaled,
      )
    else:
      y = self._model_context.kpi_scaled
    return dist.experimental_pin(y=y)

  def _get_joint_dist(self) -> backend.tfd.Distribution:
    """Returns a joint distribution for MCMC sampling."""
    if self._joint_dist is None:
      self._joint_dist = self._pin_dist(self._get_joint_dist_unpinned())
    return self._joint_dist

  def _get_joint_dist_sampling(self) -> backend.tfd.Distribution:
    """Returns a sampling-optimized joint distribution."""
    if self._joint_dist_sampling is None:
      self._joint_dist_sampling = self._pin_dist(
          self._get_joint_dist_sampling_unpinned()
      )
    return self._joint_dist_sampling

  def _prepare_latents_for_reconstruction(
      self,
      latents: Mapping[str, backend.Tensor],
  ) -> dict[str, backend.Tensor]:
    """Prepares latent samples for injection into the reconstruction graph.

    Raw NUTS output shapes or mock test data may occasionally contain spurious
    dimensions (e.g., trailing `1`s on scalar latents) that differ from the
    expected tensor shapes in the reconstruction graph. This method applies
    necessary reshaping, squeezing, or dimension alignment to prevent
    broadcasting errors during the `sample()` call.

    Args:
      latents: All sampled latent tensors.

    Returns:
      A dictionary of latent tensors formatted for the reconstruction graph.
    """
    latents_for_reconstruction = dict(latents)
    if (
        constants.SIGMA in latents_for_reconstruction
        and not self._model_context.unique_sigma_for_each_geo
    ):
      sigma_val = latents_for_reconstruction[constants.SIGMA]
      if len(sigma_val.shape) == 3 and sigma_val.shape[-1] == 1:
        latents_for_reconstruction[constants.SIGMA] = backend.squeeze(
            sigma_val, -1
        )

    return latents_for_reconstruction

  def _reconstruct_posteriors(
      self,
      latents: Mapping[str, backend.Tensor],
      rng_handler: backend.RNGHandler,
  ) -> Mapping[str, backend.Tensor]:
    """Reconstructs deterministic state variables from latent samples.

    This runs the full model logic forward using the sampled latents to compute
    all deterministic variables (e.g. ROI, budgets) required for the final
    InferenceData.

    Args:
      latents: A dictionary mapping latent variable names to sampled tensors.
      rng_handler: Random number generator handler.

    Returns:
      A dictionary containing the full state (latents + deterministics).
    """
    # Create a mutable copy to avoid modifying the input
    latents_for_reconstruction = dict(latents)

    full_dist_unpinned = self._get_joint_dist_unpinned()

    # JointDistributionCoroutineAutoBatched typically yields a StructTuple.
    # To pass values to `sample(value=...)`, we must construct a matching
    # structure (tuple) corresponding to the distribution's yielded order.
    # We inspect the distribution's dtype fields to determine this order.
    if hasattr(full_dist_unpinned.dtype, "_fields"):
      fields = full_dist_unpinned.dtype._fields
      # Map known latents to the positional order; use None for deterministics
      # that need to be computed.
      ordered_state = [latents_for_reconstruction.get(f, None) for f in fields]
      values = tuple(ordered_state)
    else:
      # Fallback for distributions that support dictionary-based inputs.
      values = latents_for_reconstruction

    return full_dist_unpinned.sample(
        value=values, seed=rng_handler.get_next_seed()
    )._asdict()

  def _prepare_mcmc_states(
      self,
      latents: Mapping[str, Any],
      reconstructed: Mapping[str, Any],
  ) -> Mapping[str, Any]:
    """Merges latent and reconstructed variables for InferenceData.

    This method combines the sampled latent variables (passed as `latents`) with
    the deterministic variables computed from them (passed as `reconstructed`).
    The primary goal is to produce a single dictionary of all relevant posterior
    samples.

    Args:
      latents: Sampled latent tensors.
      reconstructed: Deterministic tensors generated via a forward pass using
        the latents.

    Returns:
      A dictionary of final posterior tensors ready for ArviZ conversion.
    """
    mcmc_states = {}

    for k, v in latents.items():
      if k not in constants.UNSAVED_PARAMETERS:
        mcmc_states[k] = v

    for k, v in reconstructed.items():
      if (
          k not in constants.UNSAVED_PARAMETERS
          and k != "y"
          and k not in mcmc_states
      ):
        mcmc_states[k] = v

    return mcmc_states

  def __call__(
      self,
      n_chains: Sequence[int] | int,
      n_adapt: int,
      n_burnin: int,
      n_keep: int,
      current_state: Mapping[str, backend.Tensor] | None = None,
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
        will be converted into a stateless seed. See [tfp.random.sanitize_seed]
        (https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed).
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Returns:
      An `arviz.InferenceData` object containing the posterior samples, trace
      metrics, and sampling statistics.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially. For more
        information, see
        [ResourceExhaustedError when running Meridian.sample_posterior]
        (https://developers.google.com/meridian/docs/post-modeling/model-debugging#gpu-oom-error).
    """
    # Initialize the backend-agnostic RNG handler. This handles differences
    # between JAX (explicit PRNG keys) and TF (stateless/stateful seeds)
    # internally, including auto-generating a seed if `None` is provided.
    rng_handler = backend.RNGHandler(seed)
    n_chains_list = [n_chains] if isinstance(n_chains, int) else n_chains
    total_chains = np.sum(n_chains_list)

    # Clear joint distribution cache prior to sampling to ensure fresh state.
    self._joint_dist = None
    self._joint_dist_sampling = None

    states = []
    traces = []
    for n_chains_batch in n_chains_list:
      kernel_seed = rng_handler.get_kernel_seed()

      try:
        # Use sampling-optimized joint distribution (latents only) for NUTS.
        mcmc = backend.xla_windowed_adaptive_nuts(
            n_draws=n_burnin + n_keep,
            joint_dist=self._get_joint_dist_sampling(),
            n_chains=n_chains_batch,
            num_adaptation_steps=n_adapt,
            current_state=current_state,
            init_step_size=init_step_size,
            dual_averaging_kwargs=dual_averaging_kwargs,
            max_tree_depth=max_tree_depth,
            max_energy_diff=max_energy_diff,
            unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            parallel_iterations=parallel_iterations,
            seed=kernel_seed,
            **pins,
        )
      except backend.errors.ResourceExhaustedError as error:
        raise MCMCOOMError(
            "ERROR: Out of memory. Try reducing `n_keep` or pass a list of"
            " integers as `n_chains` to sample chains serially (see"
            " https://developers.google.com/meridian/docs/post-modeling/model-debugging#gpu-oom-error)"
        ) from error
      rng_handler = rng_handler.advance_handler()
      states.append(mcmc.all_states._asdict())
      traces.append(mcmc.trace)

    # Combine latent samples from all chain batches.
    all_latents = {
        k: backend.einsum(
            "ij...->ji...",
            backend.concatenate([state[k] for state in states], axis=1)[
                n_burnin:, ...
            ],
        )
        for k in states[0].keys()
    }

    latents_for_reconstruction = self._prepare_latents_for_reconstruction(
        all_latents
    )
    # Reconstruct full state (including deterministics) using the Full Graph.
    reconstructed_items = self._reconstruct_posteriors(
        latents_for_reconstruction, rng_handler
    )
    mcmc_states = self._prepare_mcmc_states(all_latents, reconstructed_items)

    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self._model_context.create_inference_data_coords(
        total_chains, n_keep
    )
    posterior_dims = self._model_context.create_inference_data_dims()
    infdata_posterior = az.convert_to_inference_data(
        mcmc_states, coords=posterior_coords, dims=posterior_dims
    )

    # Save trace metrics in InferenceData.
    mcmc_trace = {}
    for k in traces[0].keys():
      if k not in constants.IGNORED_TRACE_METRICS:
        mcmc_trace[k] = backend.concatenate(
            [
                backend.broadcast_to(
                    backend.transpose(trace[k][n_burnin:, ...]),
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
    sample_stats[constants.STEP_SIZE] = backend.broadcast_to(
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
