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
import dataclasses
import functools
from typing import Any, Optional, TYPE_CHECKING
import warnings

import arviz as az
from meridian import backend
from meridian import constants
from meridian.model import adm
from meridian.model import context
from meridian.model import equations
import numpy as np
from scipy import optimize
import xarray as xr

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "MAPOptimizationError",
    "MCMCSamplingError",
    "MCMCOOMError",
    "PosteriorMAPOptimizer",
    "PosteriorMCMCSampler",
]


class MCMCSamplingError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling failed."""


class MCMCOOMError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling exceeds memory limits."""


class MAPOptimizationError(Exception):
  """The Maximum A Posteriori (MAP) optimization failed."""


# Name of the likelihood node yielded by the joint distribution coroutine.
_LIKELIHOOD_NODE_NAME = "y"


def _flatten_chain_and_draw_dims(
    latents: Mapping[str, backend.Tensor],
) -> tuple[dict[str, backend.Tensor], int, int]:
  """Merges the leading (chain, draw) axes of each latent into one axis.

  A single flat batch axis lets the reconstruction loop honor its batch size
  exactly, rather than only in multiples of the chain count.

  Args:
    latents: Latent samples keyed by parameter name, each shaped `[n_chains,
      n_draws, ...]`.

  Returns:
    A `(flattened, n_chains, n_draws)` tuple, where `flattened` maps each
    parameter name to a tensor shaped `[n_chains * n_draws, ...]`.

  Raises:
    ValueError: If `latents` is empty.
  """
  if not latents:
    raise ValueError("Cannot reconstruct posteriors from empty latents.")
  reference = next(iter(latents.values()))
  n_chains, n_draws = int(reference.shape[0]), int(reference.shape[1])
  flattened = {
      name: backend.reshape(
          value, (n_chains * n_draws,) + tuple(value.shape[2:])  # pyrefly: ignore[bad-argument-type]
      )
      for name, value in latents.items()
  }
  return flattened, n_chains, n_draws


def _build_reconstruction_values(
    joint_dist: backend.tfd.Distribution,
    latent_chunk: Mapping[str, backend.Tensor],
    pinned_likelihood: backend.Tensor,
) -> tuple[backend.Tensor | None, ...] | Mapping[str, backend.Tensor]:
  """Arranges one batch of latents into the structure `sample(value=)` expects.

  `JointDistributionCoroutineAutoBatched` models yield their nodes as a
  namedtuple, so values must be supplied positionally in node order. Each
  node receives:

    * its latent sample, if the node is a sampled latent variable;
    * the observed KPI tensor, for the likelihood node — pinning it skips an
      otherwise-discarded likelihood sample and makes reconstruction a
      deterministic function of the latents; or
    * `None`, for deterministic nodes, which instructs TFP to compute them
      from the values above.

  Args:
    joint_dist: The full (unpinned) joint distribution.
    latent_chunk: One batch of flattened latent samples, keyed by node name.
    pinned_likelihood: The observed KPI, broadcast to the batch shape.

  Returns:
    A tuple of values in node order, or the latent mapping unchanged for
    distributions with dict-valued (non-namedtuple) structure.
  """
  structure = joint_dist.dtype
  if not hasattr(structure, "_fields"):
    # Dict-structured distributions accept a partial mapping of values.
    return dict(latent_chunk)

  values = []
  for node_name in structure._fields:  # `_fields` is public namedtuple API.
    if node_name in latent_chunk:
      values.append(latent_chunk[node_name])
    elif node_name == _LIKELIHOOD_NODE_NAME:
      values.append(pinned_likelihood)
    else:
      values.append(None)
  return tuple(values)


class _HostResultAccumulator:
  """Accumulates per-batch device results into preallocated host arrays.

  Output shapes and dtypes are only known once the first batch has been
  computed, so each variable's buffer is allocated lazily on first write.
  Host memory is capped at the final output size plus one in-flight batch,
  and device memory never holds more than one batch of results.
  """

  def __init__(self, total_size: int):
    self._total_size = total_size
    self._buffers: dict[str, np.ndarray] = {}

  def write(self, name: str, chunk: np.ndarray, start: int) -> None:
    """Writes `chunk` into rows `[start, start + len(chunk))` of `name`."""
    if name not in self._buffers:
      self._buffers[name] = np.empty(
          (self._total_size,) + chunk.shape[1:], dtype=chunk.dtype
      )
    self._buffers[name][start : start + chunk.shape[0]] = chunk

  def to_arrays(self, leading_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Returns all buffers with the flat axis reshaped to `leading_shape`."""
    return {
        name: buffer.reshape(leading_shape + buffer.shape[1:])
        for name, buffer in self._buffers.items()
    }


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
  shape = tau_g_excl_baseline.shape[:-1] + [1] if rank != 1 else [1]  # pyrefly: ignore[unsupported-operation]
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


def _sample_controls(
    model_context: context.ModelContext,
    yield_deterministics: bool,
) -> Generator[backend.tfd.Distribution, Any, backend.Tensor | None]:
  """Samples the control coefficients in the joint distribution.

  When delegated to via `yield from`, returns the geo-level control coefficients
  `gamma_gc`, or `None` if the model has no control variables.

  Args:
    model_context: An instance of `context.ModelContext` containing model data
      and settings.
    yield_deterministics: If True, also yields `gamma_gc` as a
      `backend.tfd.Deterministic` distribution.

  Yields:
    The distributions of the control parameters.
  """
  # Omit gamma_c, xi_c, and gamma_gc from joint distribution output if
  # there are no control variables in the model.
  if not model_context.n_controls:
    return None
  prior_broadcast = model_context.prior_broadcast
  gamma_c = yield prior_broadcast.gamma_c
  xi_c = yield prior_broadcast.xi_c
  gamma_gc_dev = yield backend.tfd.Sample(
      backend.tfd.Normal(
          loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
          scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
      ),
      [model_context.n_geos, model_context.n_controls],
      name=constants.GAMMA_GC_DEV,
  )
  gamma_gc = gamma_c + xi_c * gamma_gc_dev
  if yield_deterministics:
    yield backend.tfd.Deterministic(gamma_gc, name=constants.GAMMA_GC)
  return gamma_gc


def _sample_non_media_treatments(
    model_context: context.ModelContext,
    model_equations: equations.ModelEquations,
    yield_deterministics: bool,
) -> Generator[backend.tfd.Distribution, Any, backend.Tensor | None]:
  """Samples the non-media treatment coefficients in the joint distribution.

  When delegated to via `yield from`, returns the geo-level non-media treatment
  coefficients `gamma_gn`, or `None` if the model has no non-media treatments.

  Args:
    model_context: An instance of `context.ModelContext` containing model data
      and settings.
    model_equations: An instance of `equations.ModelEquations` providing the
      model's mathematical equations.
    yield_deterministics: If True, also yields the derived coefficients as
      `backend.tfd.Deterministic` distributions.

  Yields:
    The distributions of the non-media treatment parameters.
  """
  if model_context.non_media_treatments is None:
    return None
  prior_broadcast = model_context.prior_broadcast
  xi_n = yield prior_broadcast.xi_n
  gamma_gn_dev = yield backend.tfd.Sample(
      backend.tfd.Normal(
          loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
          scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
      ),
      [model_context.n_geos, model_context.n_non_media_channels],
      name=constants.GAMMA_GN_DEV,
  )
  prior_type = model_context.model_spec.non_media_treatments_prior_type
  if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
    gamma_n = yield prior_broadcast.gamma_n
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
    contribution_n = yield prior_broadcast.contribution_n
    incremental_outcome_n = contribution_n * model_context.total_outcome
    # The baseline values are already population-scaled.
    baseline_scaled = model_context.non_media_transformer.forward(  # pyrefly: ignore[missing-attribute]
        model_equations.compute_non_media_treatments_baseline(),
        apply_population_scaling=False,
    )
    linear_predictor_counterfactual_difference = (
        model_context.non_media_treatments_normalized - baseline_scaled
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
  return gamma_gn


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
  baseline_constraint = model_context.aggregate_baseline_constraint
  gamma_gc = None
  gamma_gn = None

  if baseline_constraint is not None:
    # 1. Sample sigma and tau_g
    sigma = yield prior_broadcast.sigma
    tau_g_excl_baseline = yield backend.tfd.Sample(
        prior_broadcast.tau_g_excl_baseline,
        name=constants.TAU_G_EXCL_BASELINE,
    )
    tau_g = _compute_tau_g(tau_g_excl_baseline, baseline_geo_idx)
    if yield_deterministics:
      yield backend.tfd.Deterministic(tau_g, name="tau_g")

    # 2. Controls
    gamma_gc = yield from _sample_controls(model_context, yield_deterministics)
    if gamma_gc is not None:
      controls_offset = backend.einsum(
          "gc,...gc->...",
          baseline_constraint.controls_weights,
          gamma_gc,
      )
    else:
      controls_offset = backend.to_tensor(0.0, dtype=backend.float_dtype)

    # 3. Non-media treatments
    gamma_gn = yield from _sample_non_media_treatments(
        model_context, model_equations, yield_deterministics
    )
    if (
        gamma_gn is not None
        and model_context.non_media_transformer is not None
    ):
      non_media_baseline = (
          model_equations.compute_non_media_treatments_baseline()
      )
      # The baseline values are already population-scaled.
      non_media_baseline_scaled = model_context.non_media_transformer.forward(
          non_media_baseline, apply_population_scaling=False
      )
      broadcast_non_media = backend.broadcast_to(
          non_media_baseline_scaled,
          [n_geos, n_times, n_non_media_channels],
      )
      non_media_offset = backend.einsum(
          "gt,gtn,...gn->...",
          baseline_constraint.outcome_weights,
          broadcast_non_media,
          gamma_gn,
      )
    else:
      non_media_offset = backend.to_tensor(0.0, dtype=backend.float_dtype)

    # 4. Offsets and knots rotation
    geo_offset = float(n_times) * backend.einsum(
        "g,...g->...", baseline_constraint.geo_weights, tau_g
    )
    offset = geo_offset + controls_offset + non_media_offset

    knot_dist = prior_broadcast.knot_values
    while isinstance(knot_dist, backend.tfd.BatchBroadcast):
      knot_dist = knot_dist.distribution
    assert isinstance(knot_dist, backend.tfd.Normal)
    knot_dtype = knot_dist.dtype
    knot_scale_val = float(np.reshape(np.asarray(knot_dist.scale), -1)[0])
    knot_loc_val = float(np.reshape(np.asarray(knot_dist.loc), -1)[0])
    knot_scale = backend.to_tensor(knot_scale_val, dtype=knot_dtype)
    knot_loc = backend.to_tensor(knot_loc_val, dtype=knot_dtype)

    low = backend.cast(
        backend.divide(
            baseline_constraint.threshold_l - offset,
            baseline_constraint.knot_norm_w,
        ),
        dtype=knot_dtype,
    )
    high = backend.cast(
        backend.maximum(low + 50.0 * knot_scale, 50.0 * knot_scale),
        dtype=knot_dtype,
    )
    rotated_knot_0 = yield backend.tfd.TruncatedNormal(
        loc=knot_loc,
        scale=knot_scale,
        low=low,
        high=high,
        name=constants.ROTATED_KNOT_0,
    )
    if knot_info.n_knots > 1:
      rotated_knot_rest = yield backend.tfd.Sample(
          backend.tfd.Normal(loc=knot_loc, scale=knot_scale),
          [knot_info.n_knots - 1],
          name=constants.ROTATED_KNOT_REST,
      )
      rotated_knots = backend.concatenate(
          [backend.expand_dims(rotated_knot_0, -1), rotated_knot_rest],
          axis=-1,
      )
      knot_values = backend.einsum(
          "...k,kl->...l",
          rotated_knots,
          backend.to_tensor(
              baseline_constraint.knot_rotation_matrix, dtype=knot_dtype
          ),
      )
    else:
      knot_values = backend.expand_dims(rotated_knot_0, -1)

    if yield_deterministics:
      yield backend.tfd.Deterministic(knot_values, name=constants.KNOT_VALUES)

    mu_t = backend.einsum(
        "...k,kt->...t",
        knot_values,
        backend.to_tensor(knot_info.weights, dtype=knot_dtype),
    )
    if yield_deterministics:
      yield backend.tfd.Deterministic(mu_t, name=constants.MU_T)

    tau_gt = backend.expand_dims(tau_g, -1) + backend.expand_dims(mu_t, -2)  # pyrefly: ignore[bad-argument-type]

    media_transformed_list = []
    beta_list = []

    if media_tensors.media is not None:
      alpha_m = yield prior_broadcast.alpha_m
      ec_m = yield prior_broadcast.ec_m
      eta_m = yield prior_broadcast.eta_m
      slope_m = yield prior_broadcast.slope_m
      beta_gm_dev = yield backend.tfd.Sample(
          backend.tfd.Normal(
              loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
              scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
          ),
          [n_geos, n_media_channels],
          name=constants.BETA_GM_DEV,
      )
      media_transformed = adstock_hill_media_fn(
          media=media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
          decay_functions=model_context.adstock_decay_spec.media,
          saturation_spec=model_context.saturation_spec.media,
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
            linear_predictor_counterfactual_difference=(
                linear_predictor_counterfactual_difference
            ),
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
          backend.tfd.Normal(
              loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
              scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
          ),
          [n_geos, n_rf_channels],
          name=constants.BETA_GRF_DEV,
      )
      rf_transformed = adstock_hill_rf_fn(
          reach=rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
          frequency=rf_tensors.frequency,  # pyrefly: ignore[bad-argument-type]
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
          decay_functions=model_context.adstock_decay_spec.rf,
          saturation_spec=model_context.saturation_spec.rf,
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
            linear_predictor_counterfactual_difference=(
                linear_predictor_counterfactual_difference
            ),
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
          backend.tfd.Normal(
              loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
              scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
          ),
          [n_geos, n_organic_media_channels],
          name=constants.BETA_GOM_DEV,
      )
      organic_media_transformed = adstock_hill_media_fn(
          media=organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
          alpha=alpha_om,
          ec=ec_om,
          slope=slope_om,
          decay_functions=model_context.adstock_decay_spec.organic_media,
          saturation_spec=model_context.saturation_spec.organic_media,
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
            linear_predictor_counterfactual_difference=(
                organic_media_transformed
            ),
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
          backend.tfd.Normal(
              loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
              scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
          ),
          [n_geos, n_organic_rf_channels],
          name=constants.BETA_GORF_DEV,
      )
      organic_rf_transformed = adstock_hill_rf_fn(
          reach=organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
          frequency=organic_rf_tensors.organic_frequency,  # pyrefly: ignore[bad-argument-type]
          alpha=alpha_orf,
          ec=ec_orf,
          slope=slope_orf,
          decay_functions=model_context.adstock_decay_spec.organic_rf,
          saturation_spec=model_context.saturation_spec.organic_rf,
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
      y_pred_combined_media = tau_gt + backend.einsum(
          "...gtm,...gm->...gt", combined_media_transformed, combined_beta
      )
    else:
      y_pred_combined_media = tau_gt

    if n_controls and gamma_gc is not None:
      y_pred_combined_media += backend.einsum(
          "...gtc,...gc->...gt", controls_scaled, gamma_gc
      )

    if (
        model_context.non_media_treatments is not None
        and gamma_gn is not None
    ):
      y_pred = y_pred_combined_media + backend.einsum(
          "...gtn,...gn->...gt", non_media_treatments_normalized, gamma_gn
      )
    else:
      y_pred = y_pred_combined_media

    sigma_gt = backend.transpose(backend.broadcast_to(sigma, [n_times, n_geos]))

    if holdout_id is not None:
      y_pred_holdout = backend.where(
          holdout_id, backend.to_tensor(0.0, dtype=backend.float_dtype), y_pred  # pyrefly: ignore[bad-argument-type]
      )
      test_sd = backend.cast(1.0 / np.sqrt(2.0 * np.pi), backend.float_dtype)
      sigma_gt_holdout = backend.where(holdout_id, test_sd, sigma_gt)  # pyrefly: ignore[bad-argument-type]
      yield backend.tfd.Normal(y_pred_holdout, sigma_gt_holdout, name="y")
    else:
      yield backend.tfd.Normal(y_pred, sigma_gt, name="y")
    return

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
  tau_gt = backend.expand_dims(tau_g, -1) + backend.expand_dims(mu_t, -2)  # pyrefly: ignore[bad-argument-type]

  # Accumulate media tensors in lists to avoid empty tensor shape/rank issues
  media_transformed_list = []
  beta_list = []

  if media_tensors.media is not None:
    alpha_m = yield prior_broadcast.alpha_m
    ec_m = yield prior_broadcast.ec_m
    eta_m = yield prior_broadcast.eta_m
    slope_m = yield prior_broadcast.slope_m
    beta_gm_dev = yield backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [n_geos, n_media_channels],
        name=constants.BETA_GM_DEV,
    )
    media_transformed = adstock_hill_media_fn(
        media=media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
        alpha=alpha_m,
        ec=ec_m,
        slope=slope_m,
        decay_functions=model_context.adstock_decay_spec.media,
        saturation_spec=model_context.saturation_spec.media,
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
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [n_geos, n_rf_channels],
        name=constants.BETA_GRF_DEV,
    )
    rf_transformed = adstock_hill_rf_fn(
        reach=rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
        frequency=rf_tensors.frequency,  # pyrefly: ignore[bad-argument-type]
        alpha=alpha_rf,
        ec=ec_rf,
        slope=slope_rf,
        decay_functions=model_context.adstock_decay_spec.rf,
        saturation_spec=model_context.saturation_spec.rf,
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
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [n_geos, n_organic_media_channels],
        name=constants.BETA_GOM_DEV,
    )
    organic_media_transformed = adstock_hill_media_fn(
        media=organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
        alpha=alpha_om,
        ec=ec_om,
        slope=slope_om,
        decay_functions=model_context.adstock_decay_spec.organic_media,
        saturation_spec=model_context.saturation_spec.organic_media,
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
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [n_geos, n_organic_rf_channels],
        name=constants.BETA_GORF_DEV,
    )
    organic_rf_transformed = adstock_hill_rf_fn(
        reach=organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
        frequency=organic_rf_tensors.organic_frequency,  # pyrefly: ignore[bad-argument-type]
        alpha=alpha_orf,
        ec=ec_orf,
        slope=slope_orf,
        decay_functions=model_context.adstock_decay_spec.organic_rf,
        saturation_spec=model_context.saturation_spec.organic_rf,
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

  gamma_gc = yield from _sample_controls(model_context, yield_deterministics)
  if gamma_gc is not None:
    y_pred_combined_media += backend.einsum(
        "...gtc,...gc->...gt", controls_scaled, gamma_gc
    )

  gamma_gn = yield from _sample_non_media_treatments(
      model_context, model_equations, yield_deterministics
  )
  if gamma_gn is not None:
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
    y_pred_holdout = backend.where(
        holdout_id, backend.to_tensor(0.0, dtype=backend.float_dtype), y_pred  # pyrefly: ignore[bad-argument-type]
    )
    test_sd = backend.cast(1.0 / np.sqrt(2.0 * np.pi), backend.float_dtype)
    sigma_gt_holdout = backend.where(holdout_id, test_sd, sigma_gt)  # pyrefly: ignore[bad-argument-type]
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
          self._model_context.holdout_id,  # pyrefly: ignore[bad-argument-type]
          0.0,
          self._model_context.kpi_scaled,  # pyrefly: ignore[bad-argument-type]
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
            sigma_val, -1  # pyrefly: ignore[bad-argument-type]
        )

    if constants.ROTATED_KNOT_0 in latents_for_reconstruction:
      rot_0_val = latents_for_reconstruction[constants.ROTATED_KNOT_0]
      if len(rot_0_val.shape) == 3 and rot_0_val.shape[-1] == 1:
        latents_for_reconstruction[constants.ROTATED_KNOT_0] = backend.squeeze(
            rot_0_val, -1  # pyrefly: ignore[bad-argument-type]
        )

    return latents_for_reconstruction

  def _reconstruct_posteriors(
      self,
      *,
      latents: Mapping[str, backend.Tensor],
      rng_handler: backend.RNGHandler,
      batch_size: int,
  ) -> Mapping[str, np.ndarray]:
    """Reconstructs deterministic model variables from latent samples.

    Deterministic variables (e.g. `beta_m`, `mu_t`, `gamma_gc`) are pure,
    draw-wise functions of the latent samples, so reconstruction is an
    embarrassingly parallel map over the (chain, draw) axes. To bound peak
    device memory, the map is evaluated in batches of at most `batch_size`
    draws, and each batch's results are copied to host memory before the
    next batch runs. Peak device memory is therefore O(`batch_size`) and
    independent of the total number of posterior draws.

    Args:
      latents: Latent samples keyed by parameter name, each shaped `[n_chains,
        n_draws, ...]`. Must already be normalized by
        `_prepare_latents_for_reconstruction`.
      rng_handler: Source of per-batch seeds. The TFP sampling API requires a
        seed, but it does not affect the output: every stochastic node is
        pinned, so reconstruction is deterministic given `latents`.
      batch_size: Maximum number of draws (flattened across chains) to evaluate
        per forward pass.

    Returns:
      Deterministic variables keyed by parameter name, as host (NumPy)
      arrays shaped `[n_chains, n_draws, ...]`. Latents and the likelihood
      node are excluded; `_prepare_mcmc_states` merges latents separately.

    Raises:
      ValueError: If `batch_size` is not a positive integer.
    """
    if batch_size < 1:
      raise ValueError(
          "`reconstruction_batch_size` must be a positive integer, got"
          f" {batch_size}."
      )

    joint_dist = self._get_joint_dist_unpinned()
    flat_latents, n_chains, n_draws = _flatten_chain_and_draw_dims(latents)
    n_total_draws = n_chains * n_draws
    observed_kpi = self._model_context.kpi_scaled

    results = _HostResultAccumulator(total_size=n_total_draws)
    for start in range(0, n_total_draws, batch_size):
      stop = min(start + batch_size, n_total_draws)
      latent_chunk = {
          name: value[start:stop] for name, value in flat_latents.items()
      }
      pinned_likelihood = backend.broadcast_to(
          observed_kpi, (stop - start,) + tuple(observed_kpi.shape)  # pyrefly: ignore[bad-argument-type]
      )
      values = _build_reconstruction_values(
          joint_dist, latent_chunk, pinned_likelihood
      )
      sampled = joint_dist.sample(
          value=values, seed=rng_handler.get_next_seed()
      )
      sampled_by_name = (
          sampled._asdict() if hasattr(sampled, "_asdict") else dict(sampled)
      )
      for name, tensor in sampled_by_name.items():
        if name in flat_latents or name == _LIKELIHOOD_NODE_NAME:
          continue  # Latents pass through unchanged; likelihood is unused.
        results.write(name, np.asarray(tensor), start=start)

    return results.to_arrays(leading_shape=(n_chains, n_draws))

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
          and k != _LIKELIHOOD_NODE_NAME
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
      reconstruction_batch_size: int = constants.DEFAULT_RECONSTRUCTION_BATCH_SIZE,
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
      reconstruction_batch_size: Maximum number of combined chain-and-draw
        samples to reconstruct in a single forward pass. Limits device OOMs
        caused by wide vectorized models.
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Returns:
      An `arviz.InferenceData` object containing the posterior samples, trace
      metrics, and sampling statistics.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep`, pass
        a list of integers as `n_chains` to sample chains serially, or lower
        `reconstruction_batch_size` to reduce forward pass memory footprint.
        For more information, see
        [ResourceExhaustedError when running Meridian.sample_posterior]
        (https://developers.google.com/meridian/docs/post-modeling/model-debugging#gpu-oom-error).
    """
    if reconstruction_batch_size < 1:
      raise ValueError(
          "`reconstruction_batch_size` must be a positive integer, got"
          f" {reconstruction_batch_size}."
      )
    # TODO: Sample time-varying media effects.
    if getattr(self._model_context.model_spec, "changepoints", None):
      raise NotImplementedError(
          "Sampling models with changepoints is not supported yet."
      )

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

    del mcmc, states  # Free NUTS state buffers before the first batch runs.
    latents_for_reconstruction = self._prepare_latents_for_reconstruction(
        all_latents
    )
    # Reconstruct full state (including deterministics) using the Full Graph.
    try:
      reconstructed_items = self._reconstruct_posteriors(
          latents=latents_for_reconstruction,
          rng_handler=rng_handler,
          batch_size=reconstruction_batch_size,
      )
    except backend.errors.ResourceExhaustedError as error:
      raise MCMCOOMError(
          "ERROR: Out of memory while reconstructing deterministic"
          " parameters. Reconstruction peak memory scales with"
          " `reconstruction_batch_size` (currently"
          f" {reconstruction_batch_size}, minimum 1). Reduce it and re-run;"
          " NUTS results are unaffected. See"
          " https://developers.google.com/meridian/docs/post-modeling/model-debugging#gpu-oom-error"
      ) from error
    mcmc_states = self._prepare_mcmc_states(all_latents, reconstructed_items)

    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self._model_context.create_inference_data_coords(
        total_chains, n_keep  # pyrefly: ignore[bad-argument-type]
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

    return az.concat(infdata_posterior, infdata_trace, infdata_sample_stats)  # pyrefly: ignore[no-matching-overload]


def _is_deterministic_distribution(dist: backend.tfd.Distribution) -> bool:
  """Returns True if `dist` (or its unwrapped base) is `tfd.Deterministic`."""
  return isinstance(adm.unwrap_distribution(dist), backend.tfd.Deterministic)


def _needs_jacobian_adjustment(dist: backend.tfd.Distribution) -> bool:
  """Returns True if unconstrained MAP requires the bijector log-det-Jacobian.

  Normal distributions have an identity bijector, and TruncatedNormal
  distributions already possess an interior Gaussian mode on their bounded
  support. Half-line (`HalfNormal`, `LogNormal`) and beta/uniform bounded
  distributions (`Uniform`, `Beta`) have boundary or skewed constrained-space
  modes (`0` for `HalfNormal`), so optimizing in unconstrained space (which
  adds `forward_log_det_jacobian`) prevents boundary collapse and places the
  prior mode in the interior.

  Args:
    dist: A TFP distribution node from the joint model coroutine.
  """
  base = adm.unwrap_distribution(dist)
  return not isinstance(
      base,
      (
          backend.tfd.Normal,
          backend.tfd.TruncatedNormal,
          backend.tfd.Deterministic,
      ),
  )


def _default_unconstrained_init_for_dist(
    dist: backend.tfd.Distribution,
    shape: tuple[int, ...],
    bijector: Any,
) -> backend.Tensor:
  """Computes a finite interior initial value in unconstrained space."""
  base = adm.unwrap_distribution(dist)
  try:
    if isinstance(base, backend.tfd.LogNormal):
      half = backend.to_tensor(0.5, dtype=backend.float_dtype)
      constrained_val = backend.broadcast_to(base.quantile(half), shape)  # pyrefly: ignore[bad-argument-type]
    else:
      constrained_val = backend.broadcast_to(dist.mean(), shape)  # pyrefly: ignore[bad-argument-type]
    z = backend.cast(bijector.inverse(constrained_val), backend.float_dtype)
    z_np = np.asarray(z)
    if np.all(np.isfinite(z_np)):
      return z
  except Exception:  # pylint: disable=broad-except
    pass
  return backend.zeros(shape, dtype=backend.float_dtype)


@dataclasses.dataclass(frozen=True)
class _FreeParameterSpec:
  """Describes one stochastic parameter optimized by `PosteriorMAPOptimizer`.

  Attributes:
    name: Parameter name, as yielded by the joint sampling coroutine.
    shape: Constrained parameter shape (batch shape followed by event shape).
    size: Number of scalar elements in the parameter (`prod(shape)`).
    bijector: Default event-space bijector of the parameter's base distribution,
      mapping the unconstrained optimization space onto the constrained
      parameter space.
    use_jac: Whether the bijector log-det-Jacobian is added to the objective
      when `include_jacobian=True`.
    dist: The distribution node yielded by the coroutine for this parameter.
  """

  name: str
  shape: tuple[int, ...]
  size: int
  bijector: Any
  use_jac: bool
  dist: backend.tfd.Distribution


def _split_free_vector(
    vector: backend.Tensor, free_specs: Sequence[_FreeParameterSpec]
) -> dict[str, backend.Tensor]:
  """Splits a flat parameter vector into per-parameter tensors.

  Args:
    vector: 1D tensor laid out in `free_specs` order, with length equal to the
      sum of `spec.size` over `free_specs`.
    free_specs: Free parameter specifications.

  Returns:
    Mapping from parameter name to the corresponding slice of `vector`,
    reshaped to `spec.shape`.
  """
  parts: dict[str, backend.Tensor] = {}
  offset = 0
  for spec in free_specs:
    parts[spec.name] = backend.reshape(
        vector[offset : offset + spec.size], spec.shape
    )
    offset += spec.size
  return parts


def _constrain_free_parameters(
    z_parts: Mapping[str, backend.Tensor],
    free_specs: Sequence[_FreeParameterSpec],
) -> dict[str, backend.Tensor]:
  """Maps per-parameter unconstrained tensors to constrained parameter space."""
  return {
      spec.name: backend.cast(
          spec.bijector.forward(z_parts[spec.name]), backend.float_dtype
      )
      for spec in free_specs
  }


# `(scale, deviation)` parameter pairs of the non-centered geo random-effect
# parameterization `delta_gx = scale_x * dev_gx`.
_GEO_RANDOM_EFFECT_PAIRS: tuple[tuple[str, str], ...] = (
    (constants.ETA_M, constants.BETA_GM_DEV),
    (constants.ETA_RF, constants.BETA_GRF_DEV),
    (constants.ETA_OM, constants.BETA_GOM_DEV),
    (constants.ETA_ORF, constants.BETA_GORF_DEV),
    (constants.XI_C, constants.GAMMA_GC_DEV),
    (constants.XI_N, constants.GAMMA_GN_DEV),
)

# Geo random-effect pairs whose empirical RMS does not exceed this threshold are
# left as-is by `_renormalize_geo_random_effects`.
_RMS_RENORMALIZATION_THRESHOLD = 1e-12
# Under ADM the scale parameters are held away from zero by the adjustment, so
# only random effects with non-negligible empirical spread are re-normalized.
_ADM_RMS_RENORMALIZATION_THRESHOLD = 1e-2


def _renormalize_geo_random_effects(
    latents: Mapping[str, backend.Tensor],
    threshold: float = _RMS_RENORMALIZATION_THRESHOLD,
) -> dict[str, backend.Tensor]:
  """Rescales geo random-effect factors so deviations have unit RMS over geos.

  In the non-centered parameterization `delta_gx = scale_x * dev_gx`
  (`dev_gx ~ Normal(0, 1)`), the likelihood and all deterministic model
  equations depend only on the product `scale_x * dev_gx`. Re-normalizing the
  factorization so `dev_gx` has unit RMS across geos preserves the product
  identically while ensuring `scale_x` equals the empirical geo-level standard
  deviation of the random effects.

  Args:
    latents: Mapping from parameter name to constrained value. Deviation
      parameters carry the geo axis second-to-last and scale parameters share
      the remaining axes, so any number of leading batch (e.g. draw) axes is
      supported.
    threshold: Pairs whose empirical RMS does not exceed this value are left
      unchanged, which avoids dividing by (near) zero when a random effect has
      collapsed.

  Returns:
    A copy of `latents` in which every `(scale, dev)` pair from
    `_GEO_RANDOM_EFFECT_PAIRS` that is present has been re-normalized.
  """
  result = dict(latents)
  for scale_name, dev_name in _GEO_RANDOM_EFFECT_PAIRS:
    if scale_name not in result or dev_name not in result:
      continue
    scale = np.asarray(result[scale_name], dtype=np.float64)
    dev = np.asarray(result[dev_name], dtype=np.float64)
    delta = np.expand_dims(scale, axis=-2) * dev
    rms = np.sqrt(np.mean(np.square(delta), axis=-2))
    valid = rms > threshold
    safe_rms = np.expand_dims(np.where(valid, rms, 1.0), axis=-2)
    result[scale_name] = backend.to_tensor(
        np.where(valid, rms, scale), dtype=backend.float_dtype
    )
    result[dev_name] = backend.to_tensor(
        np.where(np.expand_dims(valid, axis=-2), delta / safe_rms, dev),
        dtype=backend.float_dtype,
    )
  return result


def _resolve_adm_adjustments(
    free_specs: Sequence[_FreeParameterSpec],
    family_overrides: Mapping[str, adm.AdjustmentFamily | str] | None,
) -> dict[str, adm.ParameterAdjustment]:
  """Resolves the ADM adjustment of every free parameter."""
  return adm.resolve_adjustments(
      param_specs={
          spec.name: (spec.bijector, spec.dist) for spec in free_specs
      },
      family_overrides=family_overrides,
  )


class PosteriorMAPOptimizer(PosteriorMCMCSampler):
  """A callable that estimates full posterior modes or ADM means via L-BFGS-B.

  Unlike simplified MAP approximations that fix nonlinear media parameters
  (such as `alpha_m` or `ec_m`) or geo random effects, `PosteriorMAPOptimizer`
  jointly optimizes all non-deterministic stochastic parameters in the Meridian
  model simultaneously using L-BFGS-B with exact automatic differentiation
  gradients and multi-start initialization. When `use_adm=True`, Adjustment for
  Density Maximization (Morris, 1988; Morris & Tang, 2011) is applied in
  constrained parameter space so the maximizer targets the posterior mean rather
  than the posterior mode.
  """

  def _build_optimization_spec(
      self,
      rng_handler: backend.RNGHandler,
      **pins,
  ) -> tuple[
      list[_FreeParameterSpec],
      dict[str, backend.Tensor],
      set[str],
      backend.Tensor,
      backend.Tensor,
  ]:
    """Inspects the sampling coroutine and partitions free vs fixed parameters.

    Also warms all cached properties on `ModelContext` and `ModelEquations`
    prior to JIT compilation.

    Args:
      rng_handler: Random number generator handler for initializing any
        deterministic distributions.
      **pins: Optional pinned parameter values.

    Returns:
      A tuple `(free_specs, fixed_params, pinned_names, default_init_z,
      observed_y)` where:
        * `free_specs` is a list of `_FreeParameterSpec` for each stochastic
          parameter to be optimized.
        * `fixed_params` maps deterministic, zero-sized, or user-pinned
          parameter names to their constant tensors.
        * `pinned_names` is the set of user-supplied pin names (which are
          excluded from the returned posterior `InferenceData`, matching MCMC).
        * `default_init_z` is the 1D unconstrained initialization vector built
          from each prior distribution's interior central tendency.
        * `observed_y` is the scaled (and holdout-masked) observed KPI tensor.
    """
    # Warm cached properties eagerly before tracing inside `value_and_grad`.
    self._model_context.populate_cached_properties()
    _ = self._get_joint_dist_unpinned()
    _ = self._get_joint_dist_sampling_unpinned()

    if self._model_context.holdout_id is not None:
      observed_y = backend.where(
          self._model_context.holdout_id,  # pyrefly: ignore[bad-argument-type]
          0.0,
          self._model_context.kpi_scaled,  # pyrefly: ignore[bad-argument-type]
      )
    else:
      observed_y = self._model_context.kpi_scaled

    pinned_names = set(pins.keys())
    free_specs: list[_FreeParameterSpec] = []
    fixed_params: dict[str, backend.Tensor] = {}
    default_init_parts: list[backend.Tensor] = []

    gen = self._joint_dist_sampling_fn()
    dist = next(gen)
    while True:
      name = dist.name
      if name == _LIKELIHOOD_NODE_NAME:
        break

      shape = tuple(int(d) for d in dist.batch_shape) + tuple(
          int(d) for d in dist.event_shape
      )
      size = int(np.prod(shape)) if shape else 1

      if name in pins:
        val = backend.cast(
            backend.broadcast_to(backend.to_tensor(pins[name]), shape),  # pyrefly: ignore[bad-argument-type]
            backend.float_dtype,
        )
        fixed_params[name] = val
      elif _is_deterministic_distribution(dist) or size == 0:
        val = backend.cast(
            dist.sample(seed=rng_handler.get_next_seed()),
            backend.float_dtype,
        )
        fixed_params[name] = val
      else:
        base_dist = adm.unwrap_distribution(dist)
        bijector = base_dist.experimental_default_event_space_bijector()
        use_jac = _needs_jacobian_adjustment(dist)
        z_init = _default_unconstrained_init_for_dist(dist, shape, bijector)
        default_init_parts.append(
            backend.reshape(z_init, (-1,))  # pyrefly: ignore[bad-argument-type]
        )
        val = backend.cast(bijector.forward(z_init), backend.float_dtype)
        free_specs.append(
            _FreeParameterSpec(
                name=name,
                shape=shape,
                size=size,
                bijector=bijector,
                use_jac=use_jac,
                dist=dist,
            )
        )

      dist = gen.send(val)

    if default_init_parts:
      default_init_z = backend.concatenate(
          default_init_parts, axis=0  # pyrefly: ignore[bad-argument-type]
      )
    else:
      default_init_z = backend.zeros((0,), dtype=backend.float_dtype)

    return free_specs, fixed_params, pinned_names, default_init_z, observed_y

  def _log_joint_density(
      self,
      theta: Mapping[str, backend.Tensor],
      fixed_params: Mapping[str, backend.Tensor],
      observed_y: backend.Tensor,
  ) -> backend.Tensor:
    """Evaluates the unnormalized log posterior at constrained parameters.

    Walks the joint sampling coroutine, feeding every stochastic node its value
    from `theta` (free parameters) or `fixed_params` (deterministic, pinned, or
    zero-sized parameters), and accumulates the log prior density of the free
    parameters plus the log likelihood of `observed_y`.

    Args:
      theta: Constrained values of the free parameters, keyed by name.
      fixed_params: Constant values of all non-free parameters, keyed by name.
      observed_y: Scaled (and holdout-masked) observed KPI tensor.

    Returns:
      Scalar `log p(theta) + log p(y | theta)`, up to an additive constant.
    """
    gen = self._joint_dist_sampling_fn()
    dist = next(gen)
    log_prob_terms: list[backend.Tensor] = []
    while True:
      name = dist.name
      if name == _LIKELIHOOD_NODE_NAME:
        log_prob_terms.append(
            backend.reduce_sum(dist.log_prob(observed_y))  # pyrefly: ignore[bad-argument-type]
        )
        break
      if name in fixed_params:
        val = fixed_params[name]
      else:
        val = theta[name]
        log_prob_terms.append(
            backend.reduce_sum(dist.log_prob(val))  # pyrefly: ignore[bad-argument-type]
        )
      dist = gen.send(val)
    return sum(
        log_prob_terms, backend.to_tensor(0.0, dtype=backend.float_dtype)
    )

  def _adjusted_log_posterior(
      self,
      theta: Mapping[str, backend.Tensor],
      fixed_params: Mapping[str, backend.Tensor],
      observed_y: backend.Tensor,
      adjustments: Mapping[str, adm.ParameterAdjustment],
  ) -> backend.Tensor:
    """Evaluates the ADM-adjusted log posterior `log a(theta) + log p(theta, y)`.

    Args:
      theta: Constrained values of the free parameters, keyed by name.
      fixed_params: Constant values of all non-free parameters, keyed by name.
      observed_y: Scaled (and holdout-masked) observed KPI tensor.
      adjustments: Resolved ADM adjustment of every free parameter.

    Returns:
      Scalar adjusted log posterior, up to an additive constant.
    """
    log_density = self._log_joint_density(theta, fixed_params, observed_y)
    for name, value in theta.items():
      adjustment = adjustments[name]
      if adjustment.is_trivial:
        continue
      log_density += backend.reduce_sum(
          adm.log_adjustment(value, adjustment.family, adjustment.support)
      )
    return log_density

  def resolve_adm_adjustments(
      self,
      family_overrides: Mapping[str, adm.AdjustmentFamily | str] | None = None,
      **pins,
  ) -> dict[str, adm.ParameterAdjustment]:
    """Resolves the ADM adjustment family and support for each free parameter.

    Args:
      family_overrides: Optional mapping from parameter name to an
        `adm.AdjustmentFamily` (or its string value) overriding the default.
      **pins: Optional pinned parameter values (which are excluded from the free
        parameter set).

    Returns:
      Mapping from free parameter name to its resolved `ParameterAdjustment`.
    """
    rng_handler = backend.RNGHandler(0)
    free_specs, _, _, _, _ = self._build_optimization_spec(
        rng_handler=rng_handler, **pins
    )
    return _resolve_adm_adjustments(free_specs, family_overrides)

  def _sample_candidate_vectors(
      self,
      n_candidates: int,
      free_specs: Sequence[_FreeParameterSpec],
      fixed_params: Mapping[str, backend.Tensor],
      rng_handler: backend.RNGHandler,
  ) -> np.ndarray:
    """Samples `n_candidates` prior draws mapped to unconstrained vectors."""
    gen = self._joint_dist_sampling_fn()
    dist = next(gen)
    sample_by_name: dict[str, backend.Tensor] = {}

    while True:
      name = dist.name
      if name == _LIKELIHOOD_NODE_NAME:
        break
      if name in fixed_params:
        val = fixed_params[name]
      else:
        sampled = backend.cast(
            dist.sample(n_candidates, seed=rng_handler.get_next_seed()),
            backend.float_dtype,
        )
        sample_by_name[name] = sampled
        val = sampled[0]
      dist = gen.send(val)

    z_columns = []
    for spec in free_specs:
      z_part = backend.cast(
          spec.bijector.inverse(sample_by_name[spec.name]), backend.float_dtype
      )
      z_columns.append(
          np.asarray(
              backend.reshape(z_part, (n_candidates, spec.size)),
              dtype=np.float64,
          )
      )
    return np.concatenate(z_columns, axis=1)

  def _fit_approximate_posterior(
      self,
      theta_star: Mapping[str, backend.Tensor],
      free_specs: Sequence[_FreeParameterSpec],
      fixed_params: Mapping[str, backend.Tensor],
      observed_y: backend.Tensor,
      adjustments: Mapping[str, adm.ParameterAdjustment],
      batch_size: int,
  ) -> adm.ApproximatePosterior:
    """Fits the Hessian-based approximate posterior at the ADM optimum.

    The Hessian of the negative adjusted log posterior is evaluated in
    constrained parameter space at `theta_star` (the stationary point returned
    by the optimizer, before any geo random-effect re-normalization) and
    converted into curvature-matched approximating marginals coupled by a
    Gaussian copula; see `adm.fit_approximate_posterior`.

    Args:
      theta_star: Constrained ADM estimates keyed by free parameter name.
      free_specs: Free parameter specifications, in optimization order.
      fixed_params: Constant values of all non-free parameters.
      observed_y: Scaled (and holdout-masked) observed KPI tensor.
      adjustments: Resolved ADM adjustment of every free parameter.
      batch_size: Number of Hessian rows evaluated per autodiff pass.

    Returns:
      The fitted `adm.ApproximatePosterior`.
    """
    theta_vector = backend.concatenate(
        [
            backend.reshape(theta_star[spec.name], (-1,))  # pyrefly: ignore[bad-argument-type]
            for spec in free_specs
        ],
        axis=0,  # pyrefly: ignore[bad-argument-type]
    )

    def negative_adjusted_log_posterior(
        vector: backend.Tensor,
    ) -> backend.Tensor:
      return -self._adjusted_log_posterior(
          _split_free_vector(vector, free_specs),
          fixed_params,
          observed_y,
          adjustments,
      )

    hessian = backend.hessian(
        negative_adjusted_log_posterior, theta_vector, batch_size
    )
    return adm.fit_approximate_posterior(
        estimates={
            spec.name: np.asarray(theta_star[spec.name], dtype=np.float64)
            for spec in free_specs
        },
        hessian=np.asarray(hessian, dtype=np.float64),
        adjustments=adjustments,
    )

  def _evaluate_adjusted_log_posterior(
      self,
      draws: Mapping[str, np.ndarray],
      free_specs: Sequence[_FreeParameterSpec],
      fixed_params: Mapping[str, backend.Tensor],
      observed_y: backend.Tensor,
      adjustments: Mapping[str, adm.ParameterAdjustment],
      batch_size: int,
  ) -> np.ndarray:
    """Evaluates the adjusted log posterior at each of a set of draws.

    The model coroutine is written for a single parameter set (it is
    vectorized by `JointDistributionCoroutineAutoBatched` during sampling), so
    draws are evaluated through `backend.vectorized_map` in chunks.

    Args:
      draws: Constrained parameter draws keyed by free parameter name, each
        shaped `(n_draws, *shape)`.
      free_specs: Free parameter specifications.
      fixed_params: Constant values of all non-free parameters.
      observed_y: Scaled (and holdout-masked) observed KPI tensor.
      adjustments: Resolved ADM adjustment of every free parameter.
      batch_size: Maximum number of draws evaluated per vectorized pass.

    Returns:
      Array of shape `(n_draws,)` holding the adjusted log posterior of each
      draw, up to an additive constant.
    """

    def _single_draw(theta: Mapping[str, backend.Tensor]) -> backend.Tensor:
      return self._adjusted_log_posterior(
          theta, fixed_params, observed_y, adjustments
      )

    n_draws = int(next(iter(draws.values())).shape[0])
    log_probs = np.empty((n_draws,), dtype=np.float64)
    for start in range(0, n_draws, batch_size):
      stop = min(start + batch_size, n_draws)
      theta_batch = {
          spec.name: backend.to_tensor(
              draws[spec.name][start:stop], dtype=backend.float_dtype
          )
          for spec in free_specs
      }
      log_probs[start:stop] = np.asarray(
          backend.vectorized_map(_single_draw, theta_batch), dtype=np.float64
      )
    return log_probs

  def _free_latents_with_sample_axes(
      self,
      values: Mapping[str, Any],
      free_specs: Sequence[_FreeParameterSpec],
      n_draws: int,
      use_adm: bool,
  ) -> dict[str, backend.Tensor]:
    """Shapes free parameter values as `(chain=1, draw=n_draws, *shape)`.

    Geo random effects are re-normalized so that deviations have unit RMS
    across geos (see `_renormalize_geo_random_effects`), which is only
    meaningful when there is more than one geo.

    Args:
      values: Constrained values keyed by free parameter name, each holding
        `n_draws * spec.size` elements (a single estimate when `n_draws=1`, or
        draws stacked on a leading axis).
      free_specs: Free parameter specifications.
      n_draws: Number of draws held by `values`.
      use_adm: Whether the values come from ADM estimation, which selects the
        re-normalization threshold.

    Returns:
      Mapping from free parameter name to a `(1, n_draws, *shape)` tensor.
    """
    latents = {
        spec.name: backend.reshape(
            backend.to_tensor(  # pyrefly: ignore[bad-argument-type]
                values[spec.name], dtype=backend.float_dtype
            ),
            (1, n_draws) + spec.shape,
        )
        for spec in free_specs
    }
    if self._model_context.n_geos > 1:
      latents = _renormalize_geo_random_effects(
          latents,
          threshold=(
              _ADM_RMS_RENORMALIZATION_THRESHOLD
              if use_adm
              else _RMS_RENORMALIZATION_THRESHOLD
          ),
      )
    return latents

  def _reconstruct_states(
      self,
      free_latents: Mapping[str, backend.Tensor],
      fixed_params: Mapping[str, backend.Tensor],
      pinned_names: set[str],
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int,
  ) -> Mapping[str, Any]:
    """Reconstructs all posterior variables from `(1, n_draws, ...)` latents.

    Args:
      free_latents: Free parameter values shaped `(1, n_draws, *shape)`.
      fixed_params: Constant values of all non-free parameters; those that were
        not pinned by the caller are broadcast to the draw axes and included in
        the output, matching MCMC.
      pinned_names: Names of user-pinned parameters (excluded from the output).
      n_draws: Number of draws along the draw axis.
      rng_handler: Seed source for the (deterministic) reconstruction pass.
      batch_size: Maximum number of draws reconstructed per forward pass.

    Returns:
      Mapping from variable name to its `(1, n_draws, ...)` values, ready for
      conversion into an `InferenceData` posterior group.
    """
    all_latents: dict[str, backend.Tensor] = dict(free_latents)
    for name, val in fixed_params.items():
      if name not in pinned_names:
        shape = tuple(int(d) for d in val.shape)
        all_latents[name] = backend.broadcast_to(
            val, (1, n_draws) + shape  # pyrefly: ignore[bad-argument-type]
        )
    reconstructed = self._reconstruct_posteriors(
        latents=self._prepare_latents_for_reconstruction(all_latents),
        rng_handler=rng_handler,
        batch_size=batch_size,
    )
    return self._prepare_mcmc_states(all_latents, reconstructed)

  def _build_trace_groups(
      self,
      log_probs: np.ndarray,
      n_iterations: int,
      converged: bool,
  ) -> az.InferenceData:
    """Builds `trace` and `sample_stats` groups for `(1, n_draws)` estimates.

    Args:
      log_probs: Target (adjusted) log posterior of each draw, shape
        `(n_draws,)`.
      n_iterations: Number of L-BFGS-B iterations of the best restart.
      converged: Whether the best restart reported convergence.

    Returns:
      An `InferenceData` holding the `trace` and `sample_stats` groups.
    """
    n_draws = int(log_probs.shape[0])
    trace_coords = {
        constants.CHAIN: np.arange(1),
        constants.DRAW: np.arange(n_draws),
    }

    def _filled(value: Any, dtype: Any) -> backend.Tensor:
      return backend.to_tensor(np.full((1, n_draws), value), dtype=dtype)

    trace = {
        constants.STEP_SIZE: _filled(1.0, backend.float_dtype),
        constants.TUNE: _filled(False, backend.bool_),
        constants.TARGET_LOG_PROBABILITY_TF: backend.to_tensor(
            log_probs[np.newaxis, :], dtype=backend.float_dtype
        ),
        constants.DIVERGING: _filled(False, backend.bool_),
        constants.ACCEPT_RATIO: _filled(1.0, backend.float_dtype),
        constants.N_STEPS: _filled(n_iterations, backend.int32),
        "is_accepted": _filled(converged, backend.bool_),
    }
    trace_dims = {k: [constants.CHAIN, constants.DRAW] for k in trace}
    infdata_trace = az.convert_to_inference_data(
        trace, coords=trace_coords, dims=trace_dims, group="trace"
    )

    sample_stats = {
        constants.SAMPLE_STATS_METRICS[k]: v
        for k, v in trace.items()
        if k in constants.SAMPLE_STATS_METRICS
    }
    sample_stats_dims = {
        constants.SAMPLE_STATS_METRICS[k]: v
        for k, v in trace_dims.items()
        if k in constants.SAMPLE_STATS_METRICS
    }
    infdata_sample_stats = az.convert_to_inference_data(
        sample_stats,
        coords=trace_coords,
        dims=sample_stats_dims,
        group="sample_stats",
    )
    return az.concat([infdata_trace, infdata_sample_stats], inplace=False)

  def _build_adm_marginals_dataset(
      self,
      approximate_posterior: adm.ApproximatePosterior,
      confidence_level: float,
  ) -> xr.Dataset:
    """Summarizes the approximating marginals as an xarray dataset.

    Args:
      approximate_posterior: Fitted approximate posterior.
      confidence_level: Confidence level of the reported credible intervals.

    Returns:
      A dataset with one variable per saved free parameter, dimensioned
      `(metric, *parameter_dims)` where `metric` enumerates
      `adm.SUMMARY_METRICS`.
    """
    posterior_dims = self._model_context.create_inference_data_dims()
    coords = {
        name: values
        for name, values in self._model_context.create_inference_data_coords(
            1, 1
        ).items()
        if name not in (constants.CHAIN, constants.DRAW)
    }
    coords[constants.METRIC] = list(adm.SUMMARY_METRICS)

    data: dict[str, np.ndarray] = {}
    dims: dict[str, list[str]] = {}
    for name, summary in approximate_posterior.summary(
        confidence_level
    ).items():
      if name in constants.UNSAVED_PARAMETERS or name not in posterior_dims:
        continue
      data[name] = summary
      dims[name] = [constants.METRIC] + [
          dim
          for dim in posterior_dims[name]
          if dim not in (constants.CHAIN, constants.DRAW)
      ]
    dataset = az.dict_to_dataset(
        data, coords=coords, dims=dims, default_dims=[]
    )
    dataset.attrs.update({
        constants.CONFIDENCE_LEVEL: float(confidence_level),
        **approximate_posterior.diagnostics.as_attrs(),
    })
    return dataset

  def __call__(  # pyrefly: ignore[bad-override]
      self,
      n_restarts: int = constants.DEFAULT_MAP_N_RESTARTS,
      n_candidates: int = constants.DEFAULT_MAP_N_CANDIDATES,
      maxiter: int = constants.DEFAULT_MAP_MAXITER,
      ftol: float = constants.DEFAULT_MAP_FTOL,
      gtol: float = constants.DEFAULT_MAP_GTOL,
      init_state: Mapping[str, backend.Tensor] | None = None,
      use_adm: bool = False,
      adm_family_overrides: (
          Mapping[str, adm.AdjustmentFamily | str] | None
      ) = None,
      include_jacobian: bool = False,
      compute_hessian: bool = False,
      n_draws: int = 0,
      confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
      hessian_batch_size: int = constants.DEFAULT_HESSIAN_BATCH_SIZE,
      seed: Sequence[int] | int | None = None,
      reconstruction_batch_size: int = constants.DEFAULT_RECONSTRUCTION_BATCH_SIZE,
      **pins,
  ) -> az.InferenceData:
    """Runs Full MAP or ADM estimation for the Meridian model.

    All stochastic model parameters (including adstock decay `alpha_*`, Hill
    saturation `ec_*`, geo random-effect standard deviations `eta_*`/`xi_*`,
    and geo-level deviations `beta_g*_dev`/`gamma_g*_dev`) are jointly
    optimized using L-BFGS-B. All deterministic parameters are then
    reconstructed via the full model graph so that downstream analysis,
    summarization, visualization, and budget optimization work seamlessly.

    When `use_adm=True`, the Hessian of the adjusted negative log posterior at
    the optimum can additionally be used to quantify uncertainty
    (`compute_hessian=True` or `n_draws > 0`). Each free parameter receives a
    curvature-matched member of its ADM approximating family (see
    `adm.ApproximatingMarginal`), summarized in an `adm_marginals` group, and
    `n_draws` joint draws can be generated from these marginals through a
    Gaussian copula that preserves the local posterior correlations. The draws
    populate the `posterior` group so that downstream credible intervals work
    exactly as with MCMC, while the point estimate is preserved in a
    `point_estimate` group.

    Args:
      n_restarts: Number of L-BFGS-B optimization restarts from distinct
        starting points. Must be at least 1.
      n_candidates: Number of random prior draws evaluated to select the best
        starting points when `n_restarts > 1`. Must be non-negative.
      maxiter: Maximum number of L-BFGS-B iterations per restart.
      ftol: Relative function value tolerance (`ftol`) for L-BFGS-B.
      gtol: Projected gradient norm tolerance (`gtol`) for L-BFGS-B.
      init_state: Optional mapping of constrained parameter values to use for
        initializing the first optimization run. Any unspecified parameters fall
        back to their default interior prior initialization.
      use_adm: Whether to apply Adjustment for Density Maximization (ADM) in
        constrained parameter space so that optimization targets the posterior
        mean rather than the posterior mode. Defaults to `False` (plain MAP in
        constrained space).
      adm_family_overrides: Optional mapping from parameter name to an
        `adm.AdjustmentFamily` (or its string value) overriding the default
        approximating family when `use_adm=True`.
      include_jacobian: Whether to include the bijector log-det-Jacobian for
        half-line (`HalfNormal`, `LogNormal`) and bounded (`Uniform`, `Beta`)
        parameters when `use_adm=False`. Ignored when `use_adm=True`.
      compute_hessian: Whether to compute the Hessian at the ADM optimum and
        report the approximating marginals in an `adm_marginals` group. Requires
        `use_adm=True`. Implied by `n_draws > 0`.
      n_draws: Number of approximate posterior draws to generate from the
        Hessian-based approximation. When positive, the `posterior` group has
        `draw=n_draws` and the point estimate is stored in a `point_estimate`
        group. Requires `use_adm=True`. Defaults to `0` (point estimate only).
      confidence_level: Confidence level of the credible intervals reported in
        the `adm_marginals` group.
      hessian_batch_size: Number of Hessian rows evaluated per automatic
        differentiation pass. Lower values reduce peak memory.
      seed: Random seed for candidate generation, approximate posterior draws,
        and deterministic reconstruction.
      reconstruction_batch_size: Batch size for reconstructing deterministic
        parameters and for evaluating the log posterior of the draws.
      **pins: Optional parameter values to pin during optimization.

    Returns:
      An `arviz.InferenceData` object containing the `posterior`, `trace`, and
      `sample_stats` groups with dimensions `(chain=1, draw=max(n_draws, 1),
      ...)`, plus an `adm_marginals` group when the Hessian is computed and a
      `point_estimate` group when `n_draws > 0`.

    Raises:
      ValueError: If `reconstruction_batch_size < 1`, `n_restarts < 1`,
        `n_candidates < 0`, `maxiter < 1`, `n_draws < 0`,
        `hessian_batch_size < 1`, `confidence_level` is not in `(0, 1)`, or
        `adm_family_overrides`, `compute_hessian`, or `n_draws > 0` is used
        when `use_adm=False`.
      MAPOptimizationError: If all optimization restarts fail to produce a
        finite parameter vector.
    """
    if reconstruction_batch_size < 1:
      raise ValueError(
          "`reconstruction_batch_size` must be a positive integer, got"
          f" {reconstruction_batch_size}."
      )
    if n_restarts < 1:
      raise ValueError(
          f"`n_restarts` must be a positive integer, got {n_restarts}."
      )
    if n_candidates < 0:
      raise ValueError(
          f"`n_candidates` must be a non-negative integer, got {n_candidates}."
      )
    if maxiter < 1:
      raise ValueError(f"`maxiter` must be a positive integer, got {maxiter}.")
    if adm_family_overrides is not None and not use_adm:
      raise ValueError(
          "`adm_family_overrides` cannot be set when `use_adm` is False."
      )
    if n_draws < 0:
      raise ValueError(
          f"`n_draws` must be a non-negative integer, got {n_draws}."
      )
    if hessian_batch_size < 1:
      raise ValueError(
          "`hessian_batch_size` must be a positive integer, got"
          f" {hessian_batch_size}."
      )
    if not 0.0 < confidence_level < 1.0:
      raise ValueError(
          f"`confidence_level` must be in (0, 1), got {confidence_level}."
      )
    fit_approximation = compute_hessian or n_draws > 0
    if fit_approximation and not use_adm:
      raise ValueError(
          "Hessian-based approximating marginals and draws (`compute_hessian`,"
          " `n_draws > 0`) require `use_adm=True`."
      )

    rng_handler = backend.RNGHandler(seed)
    self._joint_dist = None
    self._joint_dist_sampling = None

    (
        free_specs,
        fixed_params,
        pinned_names,
        default_init_z,
        observed_y,
    ) = self._build_optimization_spec(rng_handler=rng_handler, **pins)

    adm_adjustments: dict[str, adm.ParameterAdjustment] = {}
    if use_adm:
      adm_adjustments = _resolve_adm_adjustments(
          free_specs, adm_family_overrides
      )

    total_dim = sum(spec.size for spec in free_specs)

    def neg_log_posterior(z: backend.Tensor) -> backend.Tensor:
      z_parts = _split_free_vector(z, free_specs)
      theta = _constrain_free_parameters(z_parts, free_specs)
      if use_adm:
        return -self._adjusted_log_posterior(
            theta, fixed_params, observed_y, adm_adjustments
        )
      log_density = self._log_joint_density(theta, fixed_params, observed_y)
      if include_jacobian:
        for spec in free_specs:
          if spec.use_jac:
            log_density += backend.reduce_sum(
                spec.bijector.forward_log_det_jacobian(
                    z_parts[spec.name], event_ndims=0
                )
            )
      return -log_density

    if total_dim > 0:
      val_and_grad_fn = backend.value_and_grad(neg_log_posterior)

      def scipy_objective(z_np: np.ndarray) -> tuple[float, np.ndarray]:
        z_tensor = backend.to_tensor(z_np, dtype=backend.float_dtype)
        loss, grad = val_and_grad_fn(z_tensor)
        loss_f = float(np.asarray(loss))
        grad_np = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(loss_f) or not np.all(np.isfinite(grad_np)):
          return 1e30, np.zeros_like(z_np, dtype=np.float64)
        return loss_f, grad_np

      # Build starting point 0 from `init_state` or `default_init_z`.
      if init_state is not None:
        init_parts = []
        offset = 0
        for spec in free_specs:
          if spec.name in init_state:
            constrained_val = backend.cast(
                backend.broadcast_to(
                    backend.to_tensor(init_state[spec.name]), spec.shape  # pyrefly: ignore[bad-argument-type]
                ),
                backend.float_dtype,
            )
            z_part = backend.reshape(
                backend.cast(
                    spec.bijector.inverse(constrained_val), backend.float_dtype
                ),
                (spec.size,),
            )
          else:
            z_part = default_init_z[offset : offset + spec.size]
          init_parts.append(np.asarray(z_part, dtype=np.float64))
          offset += spec.size
        z0_primary = np.concatenate(init_parts, axis=0)
      else:
        z0_primary = np.asarray(default_init_z, dtype=np.float64)

      if not use_adm and not include_jacobian:
        # When optimizing in constrained space without ADM or the HalfNormal
        # log-det Jacobian, (eta_x=0, beta_gx_dev=0) is a saddle point when dev
        # starts at identically zero. Add a small deterministic perturbation to
        # break symmetry on Restart 0.
        offset = 0
        z0_primary = z0_primary.copy()
        for spec in free_specs:
          if spec.name.endswith("_dev") and np.all(
              z0_primary[offset : offset + spec.size] == 0.0
          ):
            z0_primary[offset : offset + spec.size] = np.linspace(
                -0.05, 0.05, spec.size, dtype=np.float64
            )
          offset += spec.size

      start_vectors: list[np.ndarray] = [z0_primary]
      if n_restarts > 1 and n_candidates > 0:
        n_pool = max(n_candidates, n_restarts - 1)
        cand_matrix = self._sample_candidate_vectors(
            n_candidates=n_pool,
            free_specs=free_specs,
            fixed_params=fixed_params,
            rng_handler=rng_handler,
        )
        scored_candidates: list[tuple[float, np.ndarray]] = []
        for idx in range(n_pool):
          cand_vec = cand_matrix[idx]
          if not np.all(np.isfinite(cand_vec)):
            continue
          cand_loss, _ = scipy_objective(cand_vec)
          if cand_loss < 1e29:
            scored_candidates.append((cand_loss, cand_vec))
        scored_candidates.sort(key=lambda item: item[0])
        for _, cand_vec in scored_candidates[: n_restarts - 1]:
          start_vectors.append(cand_vec)

      best_loss = float("inf")
      best_z: np.ndarray | None = None
      best_nit = 0
      best_converged = False

      for z0 in start_vectors:
        res = optimize.minimize(
            scipy_objective,
            z0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
            },
        )
        res_loss = float(res.fun)
        if np.isfinite(res_loss) and res_loss < best_loss:
          best_loss = res_loss
          best_z = np.asarray(res.x, dtype=np.float64)
          best_nit = int(getattr(res, "nit", 0))
          best_converged = bool(res.success)

      if best_z is None or not np.all(np.isfinite(best_z)):
        raise MAPOptimizationError(
            "MAP optimization failed to produce finite parameter estimates"
            f" across {len(start_vectors)} start(s)."
        )
      best_z_tensor = backend.to_tensor(best_z, dtype=backend.float_dtype)
    else:
      best_z_tensor = backend.zeros((0,), dtype=backend.float_dtype)
      best_loss = float(np.asarray(neg_log_posterior(best_z_tensor)))
      best_nit = 0
      best_converged = True

    # Stationary point in constrained space, before geo random-effect
    # re-normalization (which preserves the likelihood but not the Hessian).
    theta_star = _constrain_free_parameters(
        _split_free_vector(best_z_tensor, free_specs), free_specs
    )
    approximate_posterior: adm.ApproximatePosterior | None = None
    if fit_approximation and total_dim > 0:
      approximate_posterior = self._fit_approximate_posterior(
          theta_star,
          free_specs,
          fixed_params,
          observed_y,
          adm_adjustments,
          batch_size=hessian_batch_size,
      )

    point_latents = self._free_latents_with_sample_axes(
        theta_star, free_specs, n_draws=1, use_adm=use_adm
    )
    point_states = self._reconstruct_states(
        point_latents,
        fixed_params,
        pinned_names,
        n_draws=1,
        rng_handler=rng_handler,
        batch_size=reconstruction_batch_size,
    )

    if approximate_posterior is not None and n_draws > 0:
      standard_normals = backend.tfd.Normal(0.0, 1.0).sample(
          (n_draws, total_dim), seed=rng_handler.get_next_seed()
      )
      draws = approximate_posterior.sample(
          np.asarray(standard_normals, dtype=np.float64)
      )
      target_log_probs = self._evaluate_adjusted_log_posterior(
          draws,
          free_specs,
          fixed_params,
          observed_y,
          adm_adjustments,
          batch_size=reconstruction_batch_size,
      )
      draw_latents = self._free_latents_with_sample_axes(
          draws, free_specs, n_draws=n_draws, use_adm=use_adm
      )
      posterior_states = self._reconstruct_states(
          draw_latents,
          fixed_params,
          pinned_names,
          n_draws=n_draws,
          rng_handler=rng_handler,
          batch_size=reconstruction_batch_size,
      )
      n_posterior_draws = n_draws
    else:
      posterior_states = point_states
      target_log_probs = np.array([-best_loss], dtype=np.float64)
      n_posterior_draws = 1

    posterior_dims = self._model_context.create_inference_data_dims()
    infdata_posterior = az.convert_to_inference_data(
        posterior_states,
        coords=self._model_context.create_inference_data_coords(
            1, n_posterior_draws
        ),
        dims=posterior_dims,
    )
    posterior_attrs = {
        "inference_method": (
            constants.METHOD_ADM if use_adm else constants.METHOD_MAP
        ),
        "use_adm": bool(use_adm),
        "adm_adjustments": (
            {name: adj.family.value for name, adj in adm_adjustments.items()}
            if use_adm
            else {}
        ),
        "map_converged": best_converged,
        "map_n_iterations": best_nit,
        "map_objective": float(-best_loss),
        "n_draws": int(n_draws if approximate_posterior is not None else 0),
    }
    if approximate_posterior is not None:
      posterior_attrs.update(approximate_posterior.diagnostics.as_attrs())
    infdata_posterior.posterior.attrs.update(posterior_attrs)

    inference_data = az.concat(
        [
            infdata_posterior,
            self._build_trace_groups(
                target_log_probs, best_nit, best_converged
            ),
        ],
        inplace=False,
    )
    if approximate_posterior is not None:
      inference_data.add_groups({
          constants.ADM_MARGINALS: self._build_adm_marginals_dataset(
              approximate_posterior, confidence_level
          )
      })
      if n_draws > 0:
        inference_data.add_groups({
            constants.POINT_ESTIMATE: az.dict_to_dataset(
                point_states,
                coords=self._model_context.create_inference_data_coords(1, 1),
                dims=posterior_dims,
            )
        })
    return inference_data
