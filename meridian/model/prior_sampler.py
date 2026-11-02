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

"""Module for sampling prior distributions in a Meridian model."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Optional, TYPE_CHECKING
import warnings

from meridian import backend
from meridian import constants
from meridian.model import changepoints
from meridian.model import context
from meridian.model import equations
import numpy as np

# TODO: Break this circular dependency.
if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "PriorDistributionSampler",
]


def _compute_in_draw_batches(
    fn: Callable[..., backend.Tensor],
    draw_args: Sequence[backend.Tensor],
    n_draws: int,
    batch_size: int,
) -> backend.Tensor:
  """Evaluates `fn(*draw_args)` in batches along the draw dimension (axis=1).

  `fn` runs through `backend.compile_temporary_function`, because a compiled
  function uses less memory than running operation by operation. Every batch
  has the same number of draws, so that `fn` is compiled only once. When the
  draws don't divide evenly, the last batch is shifted back to end at the final
  draw, and the draws it shares with the previous batch are dropped.

  Args:
    fn: Function that takes a batch of each tensor in `draw_args` and returns a
      tensor with draws on axis 1. Each draw must be computed independently.
    draw_args: Tensors with draws on axis 1.
    n_draws: Number of draws.
    batch_size: Maximum number of draws per batch.

  Returns:
    The results for all draws, concatenated along axis 1.
  """
  compiled_fn = backend.compile_temporary_function(fn)
  if n_draws <= batch_size:
    return compiled_fn(*draw_args)
  batches = []
  for start in range(0, n_draws, batch_size):
    batch_start = min(start, n_draws - batch_size)
    batch = compiled_fn(
        *[arg[:, batch_start : batch_start + batch_size] for arg in draw_args]
    )
    batches.append(batch[:, start - batch_start :])
  return backend.concatenate(batches, axis=1)  # pyrefly: ignore[bad-argument-type]


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
  tau_g = equations.compute_tau_g(tau_g_excl_baseline, baseline_geo_idx)
  return backend.tfd.Deterministic(tau_g, name="tau_g")


def _sample_time_varying_priors(
    *,
    changepoint_info: changepoints.ChangepointInfo,
    zeta_prior: backend.tfd.Distribution,
    zeta_name: str,
    varphi_name: str,
    sample_shape: Sequence[int],
    rng_handler: backend.RNGHandler,
) -> dict[str, backend.Tensor]:
  """Draws the time effect parameters of the channels with changepoints.

  With a coefficient prior, `beta` is drawn as usual and is the base coefficient
  that the time effect adjusts. With an ROI, mROI or contribution prior, the
  caller solves `beta` with
  `ModelEquations._calculate_beta_x_with_changepoints`, so the prior keeps its
  whole-history meaning.

  Args:
    changepoint_info: The `ChangepointInfo` of the channel type.
    zeta_prior: The broadcast prior on the time random effect standard
      deviation parameter.
    zeta_name: The name of the time random effect standard deviation
      parameter.
    varphi_name: The name of the latent standard normal parameter values of
      the intervals between changepoints.
    sample_shape: The sample shape, `[1, n_draws]`.
    rng_handler: The backend-agnostic RNG handler managing the seed state.

  Returns:
    The draws of `zeta_name`, shape `[1, n_draws, n_changepoint_channels]`, and
    `varphi_name`, shape `[1, n_draws, n_changepoint_channels, max_intervals]`.
  """
  zeta_draws = zeta_prior.sample(
      sample_shape=sample_shape, seed=rng_handler.get_next_seed()
  )
  varphi_draws = changepoints.interval_adjustments_distribution(
      changepoint_info, varphi_name
  ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
  return {zeta_name: zeta_draws, varphi_name: varphi_draws}


class PriorDistributionSampler:
  """A callable that samples from a model spec's prior distributions."""

  # TODO: Deprecate direct injection of `model.Meridian`.
  def __init__(
      self,
      meridian: Optional["model.Meridian"] = None,
      *,
      model_context: context.ModelContext | None = None,
  ):
    if meridian is not None:
      warnings.warn(
          "Initializing PriorDistributionSampler with a Meridian object is"
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

  @functools.cached_property
  def _model_equations(self) -> equations.ModelEquations:
    return equations.ModelEquations(self._model_context)

  def _sample_media_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
      batch_size: Maximum number of draws to process per batch when computing
        transformed media and `beta_m`.

    Returns:
      A mapping of media parameter names to a tensor of shape `[n_draws, n_geos,
      n_media_channels]` or `[n_draws, n_media_channels]` containing the
      samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    media_vars = {
        constants.ALPHA_M: prior.alpha_m.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.EC_M: prior.ec_m.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.ETA_M: prior.eta_m.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.SLOPE_M: prior.slope_m.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    beta_gm_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_media_channels],
        name=constants.BETA_GM_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
    changepoint_info = ctx._media_changepoint_info  # pylint: disable=protected-access
    if changepoint_info is not None:
      media_vars |= _sample_time_varying_priors(
          changepoint_info=changepoint_info,
          zeta_prior=ctx._changepoint_prior_broadcast.zeta_m,  # pylint: disable=protected-access
          zeta_name=constants.ZETA_M,
          varphi_name=constants.VARPHI_M,
          sample_shape=sample_shape,
          rng_handler=rng_handler,
      )

    def _solve_beta_m(
        incremental_outcome_m: backend.Tensor,
        time_varying_info: changepoints.ChangepointInfo | None,
    ) -> backend.Tensor:

      def _compute_beta_m(
          alpha_m: backend.Tensor,
          ec_m: backend.Tensor,
          slope_m: backend.Tensor,
          incremental_outcome: backend.Tensor,
          eta_m: backend.Tensor,
          beta_gx_dev: backend.Tensor,
          zeta_m: backend.Tensor | None = None,
          varphi_m: backend.Tensor | None = None,
      ) -> backend.Tensor:
        media_transformed = self._model_equations.adstock_hill_media(
            media=ctx.media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
            decay_functions=ctx.adstock_decay_spec.media,
            saturation_spec=ctx.saturation_spec.media,
        )
        linear_predictor_counterfactual_difference = self._model_equations.linear_predictor_counterfactual_difference_media(
            media_transformed=media_transformed,
            alpha_m=alpha_m,
            ec_m=ec_m,
            slope_m=slope_m,
        )
        if time_varying_info is None:
          return self._model_equations.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome,
              linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
              eta_x=eta_m,
              beta_gx_dev=beta_gx_dev,
          )
        assert zeta_m is not None and varphi_m is not None
        return self._model_equations._calculate_beta_x_with_changepoints(  # pylint: disable=protected-access
            changepoint_info=time_varying_info,
            incremental_outcome_x=incremental_outcome,
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=eta_m,
            beta_gx_dev=beta_gx_dev,
            zeta_x=zeta_m,
            varphi_x=varphi_m,
        )

      draw_args = [
          media_vars[constants.ALPHA_M],
          media_vars[constants.EC_M],
          media_vars[constants.SLOPE_M],
          incremental_outcome_m,
          media_vars[constants.ETA_M],
          beta_gm_dev,
      ]
      if time_varying_info is not None:
        draw_args += [
            media_vars[constants.ZETA_M],
            media_vars[constants.VARPHI_M],
        ]
      return _compute_in_draw_batches(
          _compute_beta_m,
          draw_args=draw_args,
          n_draws=n_draws,
          batch_size=batch_size,
      )

    self._sample_coefficient_means(
        xvars=media_vars,
        is_rf=False,
        solve=_solve_beta_m,
        sample_shape=sample_shape,
        rng_handler=rng_handler,
    )

    beta_eta_combined = (
        media_vars[constants.BETA_M][..., backend.newaxis, :]
        + media_vars[constants.ETA_M][..., backend.newaxis, :] * beta_gm_dev
    )
    beta_gm_value = self._model_equations._coefficients_from_random_effects(  # pylint: disable=protected-access
        beta_eta_combined, changepoint_info
    )
    media_vars[constants.BETA_GM] = backend.tfd.Deterministic(
        beta_gm_value, name=constants.BETA_GM
    ).sample(seed=rng_handler.get_next_seed())

    return media_vars

  def _sample_coefficient_means(
      self,
      *,
      xvars: dict[str, backend.Tensor],
      is_rf: bool,
      solve: Callable[
          [backend.Tensor, changepoints.ChangepointInfo | None], backend.Tensor
      ],
      sample_shape: Sequence[int],
      rng_handler: backend.RNGHandler,
  ) -> None:
    """Samples `beta_m` or `beta_rf` and the treatment parameters it uses.

    Channels with changepoints use `changepoint_media_prior_type` or
    `changepoint_rf_prior_type`, and the other channels `media_prior_type` or
    `rf_prior_type`. When the two differ, the draws of a treatment parameter
    are NaN for the channels that don't use it.

    Args:
      xvars: The draws of the channel type, updated in place.
      is_rf: Whether to sample the RF channels rather than the media channels.
      solve: Returns `beta_m` or `beta_rf` from the incremental outcome of each
        channel and the `ChangepointInfo` of the channels to solve with their
        time effects.
      sample_shape: The sample shape of the draws.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
    """
    # pylint: disable=protected-access
    ctx = self._model_context
    if is_rf:
      changepoint_info = ctx._rf_changepoint_info
      prior_denominator = ctx.rf_tensors.prior_denominator
      parameters = constants.RF_TREATMENT_PARAMETERS
    else:
      changepoint_info = ctx._media_changepoint_info
      prior_denominator = ctx.media_tensors.prior_denominator
      parameters = constants.MEDIA_TREATMENT_PARAMETERS
    # If every channel of the type has changepoints, both are the changepoint
    # prior type.
    prior_types = list(ctx._paid_prior_types(is_rf).values())
    prior_type, changepoint_prior_type = prior_types[0], prior_types[-1]
    # pylint: enable=protected-access
    coefficient = constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    beta = parameters[coefficient]
    draws = {}
    for draw_prior_type in dict.fromkeys((prior_type, changepoint_prior_type)):
      if draw_prior_type not in parameters:
        raise ValueError(f"Unsupported prior type: {draw_prior_type}")
      draws[draw_prior_type] = getattr(
          ctx.prior_broadcast, parameters[draw_prior_type]
      ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
    if list(draws) == [coefficient]:
      xvars[beta] = draws[coefficient]
      return

    uses_two_prior_types = len(draws) > 1
    # A coefficient prior only replaces the solution of its channels at the end.
    coefficient_draws = draws.pop(coefficient, None)
    for draw_prior_type, value in draws.items():
      if uses_two_prior_types:
        value = backend.where(
            ctx._channels_using_prior_type(is_rf, draw_prior_type),  # pylint: disable=protected-access
            value,
            np.nan,
        )
      xvars[parameters[draw_prior_type]] = value
    if len(draws) > 1:
      treatment_parameter = backend.where(
          ctx._channels_using_prior_type(is_rf, changepoint_prior_type),  # pylint: disable=protected-access
          draws[changepoint_prior_type],
          draws[prior_type],
      )
    else:
      treatment_parameter = next(iter(draws.values()))

    assert prior_denominator is not None
    beta_value = solve(
        treatment_parameter * prior_denominator,
        # Channels with changepoints and a coefficient prior don't need their
        # time effects solved.
        None if changepoint_prior_type == coefficient else changepoint_info,
    )
    if coefficient_draws is not None:
      beta_value = backend.where(
          ctx._channels_using_prior_type(is_rf, coefficient),  # pylint: disable=protected-access
          coefficient_draws,
          beta_value,  # pyrefly: ignore[bad-argument-type]
      )
    xvars[beta] = backend.tfd.Deterministic(beta_value, name=beta).sample(
        seed=rng_handler.get_next_seed()
    )

  def _sample_rf_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
      batch_size: Maximum number of draws to process per batch when computing
        transformed RF and `beta_rf`.

    Returns:
      A mapping of RF parameter names to a tensor of shape
      `[n_draws, n_geos, n_rf_channels]` or `[n_draws, n_rf_channels]`
      containing the samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    rf_vars = {
        constants.ALPHA_RF: prior.alpha_rf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.EC_RF: prior.ec_rf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.ETA_RF: prior.eta_rf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.SLOPE_RF: prior.slope_rf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    beta_grf_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_rf_channels],
        name=constants.BETA_GRF_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
    changepoint_info = ctx._rf_changepoint_info  # pylint: disable=protected-access
    if changepoint_info is not None:
      rf_vars |= _sample_time_varying_priors(
          changepoint_info=changepoint_info,
          zeta_prior=ctx._changepoint_prior_broadcast.zeta_rf,  # pylint: disable=protected-access
          zeta_name=constants.ZETA_RF,
          varphi_name=constants.VARPHI_RF,
          sample_shape=sample_shape,
          rng_handler=rng_handler,
      )

    def _solve_beta_rf(
        incremental_outcome_rf: backend.Tensor,
        time_varying_info: changepoints.ChangepointInfo | None,
    ) -> backend.Tensor:

      def _compute_beta_rf(
          alpha_rf: backend.Tensor,
          ec_rf: backend.Tensor,
          slope_rf: backend.Tensor,
          incremental_outcome: backend.Tensor,
          eta_rf: backend.Tensor,
          beta_gx_dev: backend.Tensor,
          zeta_rf: backend.Tensor | None = None,
          varphi_rf: backend.Tensor | None = None,
      ) -> backend.Tensor:
        rf_transformed = self._model_equations.adstock_hill_rf(
            reach=ctx.rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
            frequency=ctx.rf_tensors.frequency,  # pyrefly: ignore[bad-argument-type]
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
            decay_functions=ctx.adstock_decay_spec.rf,
            saturation_spec=ctx.saturation_spec.rf,
        )
        linear_predictor_counterfactual_difference = (
            self._model_equations.linear_predictor_counterfactual_difference_rf(
                rf_transformed=rf_transformed,
                alpha_rf=alpha_rf,
                ec_rf=ec_rf,
                slope_rf=slope_rf,
            )
        )
        if time_varying_info is None:
          return self._model_equations.calculate_beta_x(
              is_non_media=False,
              incremental_outcome_x=incremental_outcome,
              linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
              eta_x=eta_rf,
              beta_gx_dev=beta_gx_dev,
          )
        assert zeta_rf is not None and varphi_rf is not None
        return self._model_equations._calculate_beta_x_with_changepoints(  # pylint: disable=protected-access
            changepoint_info=time_varying_info,
            incremental_outcome_x=incremental_outcome,
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=eta_rf,
            beta_gx_dev=beta_gx_dev,
            zeta_x=zeta_rf,
            varphi_x=varphi_rf,
        )

      draw_args = [
          rf_vars[constants.ALPHA_RF],
          rf_vars[constants.EC_RF],
          rf_vars[constants.SLOPE_RF],
          incremental_outcome_rf,
          rf_vars[constants.ETA_RF],
          beta_grf_dev,
      ]
      if time_varying_info is not None:
        draw_args += [rf_vars[constants.ZETA_RF], rf_vars[constants.VARPHI_RF]]
      return _compute_in_draw_batches(
          _compute_beta_rf,
          draw_args=draw_args,
          n_draws=n_draws,
          batch_size=batch_size,
      )

    self._sample_coefficient_means(
        xvars=rf_vars,
        is_rf=True,
        solve=_solve_beta_rf,
        sample_shape=sample_shape,
        rng_handler=rng_handler,
    )

    beta_eta_combined = (
        rf_vars[constants.BETA_RF][..., backend.newaxis, :]
        + rf_vars[constants.ETA_RF][..., backend.newaxis, :] * beta_grf_dev
    )
    beta_grf_value = self._model_equations._coefficients_from_random_effects(  # pylint: disable=protected-access
        beta_eta_combined, changepoint_info
    )
    rf_vars[constants.BETA_GRF] = backend.tfd.Deterministic(
        beta_grf_value, name=constants.BETA_GRF
    ).sample(seed=rng_handler.get_next_seed())

    return rf_vars

  def _sample_organic_media_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of organic media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
      batch_size: Maximum number of draws to process per batch when computing
        transformed organic media and `beta_om`.

    Returns:
      A mapping of organic media parameter names to a tensor of shape
      `[n_draws, n_geos, n_organic_media_channels]` or
      `[n_draws, n_organic_media_channels]` containing the samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    organic_media_vars = {
        constants.ALPHA_OM: prior.alpha_om.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.EC_OM: prior.ec_om.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.ETA_OM: prior.eta_om.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.SLOPE_OM: prior.slope_om.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    beta_gom_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_organic_media_channels],
        name=constants.BETA_GOM_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())

    prior_type = ctx.model_spec.organic_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      organic_media_vars[constants.BETA_OM] = prior.beta_om.sample(
          sample_shape=sample_shape, seed=rng_handler.get_next_seed()
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      organic_media_vars[constants.CONTRIBUTION_OM] = (
          prior.contribution_om.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          )
      )
      incremental_outcome_om = (
          organic_media_vars[constants.CONTRIBUTION_OM] * ctx.total_outcome
      )

      def _compute_beta_om(
          alpha_om: backend.Tensor,
          ec_om: backend.Tensor,
          slope_om: backend.Tensor,
          incremental_outcome: backend.Tensor,
          eta_om: backend.Tensor,
          beta_gx_dev: backend.Tensor,
      ) -> backend.Tensor:
        organic_media_transformed = self._model_equations.adstock_hill_media(
            media=ctx.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
            alpha=alpha_om,
            ec=ec_om,
            slope=slope_om,
            decay_functions=ctx.adstock_decay_spec.organic_media,
            saturation_spec=ctx.saturation_spec.organic_media,
        )
        return self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome,
            linear_predictor_counterfactual_difference=organic_media_transformed,
            eta_x=eta_om,
            beta_gx_dev=beta_gx_dev,
        )

      beta_om_value = _compute_in_draw_batches(
          _compute_beta_om,
          draw_args=[
              organic_media_vars[constants.ALPHA_OM],
              organic_media_vars[constants.EC_OM],
              organic_media_vars[constants.SLOPE_OM],
              incremental_outcome_om,
              organic_media_vars[constants.ETA_OM],
              beta_gom_dev,
          ],
          n_draws=n_draws,
          batch_size=batch_size,
      )
      organic_media_vars[constants.BETA_OM] = backend.tfd.Deterministic(
          beta_om_value,
          name=constants.BETA_OM,
      ).sample(seed=rng_handler.get_next_seed())
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = (
        organic_media_vars[constants.BETA_OM][..., backend.newaxis, :]
        + organic_media_vars[constants.ETA_OM][..., backend.newaxis, :]
        * beta_gom_dev
    )
    beta_gom_value = (
        beta_eta_combined
        if ctx.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    organic_media_vars[constants.BETA_GOM] = backend.tfd.Deterministic(
        beta_gom_value, name=constants.BETA_GOM
    ).sample(seed=rng_handler.get_next_seed())

    return organic_media_vars

  def _sample_organic_rf_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the organic RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
      batch_size: Maximum number of draws to process per batch when computing
        transformed organic RF and `beta_orf`.

    Returns:
      A mapping of organic RF parameter names to a tensor of shape
      `[n_draws, n_geos, n_organic_rf_channels]` or
      `[n_draws, n_organic_rf_channels]` containing the samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    organic_rf_vars = {
        constants.ALPHA_ORF: prior.alpha_orf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.EC_ORF: prior.ec_orf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.ETA_ORF: prior.eta_orf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.SLOPE_ORF: prior.slope_orf.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    beta_gorf_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_organic_rf_channels],
        name=constants.BETA_GORF_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())

    prior_type = ctx.model_spec.organic_media_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      organic_rf_vars[constants.BETA_ORF] = prior.beta_orf.sample(
          sample_shape=sample_shape, seed=rng_handler.get_next_seed()
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      organic_rf_vars[constants.CONTRIBUTION_ORF] = (
          prior.contribution_orf.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          )
      )
      incremental_outcome_orf = (
          organic_rf_vars[constants.CONTRIBUTION_ORF] * ctx.total_outcome
      )

      def _compute_beta_orf(
          alpha_orf: backend.Tensor,
          ec_orf: backend.Tensor,
          slope_orf: backend.Tensor,
          incremental_outcome: backend.Tensor,
          eta_orf: backend.Tensor,
          beta_gx_dev: backend.Tensor,
      ) -> backend.Tensor:
        organic_rf_transformed = self._model_equations.adstock_hill_rf(
            reach=ctx.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
            frequency=ctx.organic_rf_tensors.organic_frequency,  # pyrefly: ignore[bad-argument-type]
            alpha=alpha_orf,
            ec=ec_orf,
            slope=slope_orf,
            decay_functions=ctx.adstock_decay_spec.organic_rf,
            saturation_spec=ctx.saturation_spec.organic_rf,
        )
        return self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome,
            linear_predictor_counterfactual_difference=organic_rf_transformed,
            eta_x=eta_orf,
            beta_gx_dev=beta_gx_dev,
        )

      beta_orf_value = _compute_in_draw_batches(
          _compute_beta_orf,
          draw_args=[
              organic_rf_vars[constants.ALPHA_ORF],
              organic_rf_vars[constants.EC_ORF],
              organic_rf_vars[constants.SLOPE_ORF],
              incremental_outcome_orf,
              organic_rf_vars[constants.ETA_ORF],
              beta_gorf_dev,
          ],
          n_draws=n_draws,
          batch_size=batch_size,
      )
      organic_rf_vars[constants.BETA_ORF] = backend.tfd.Deterministic(
          beta_orf_value,
          name=constants.BETA_ORF,
      ).sample(seed=rng_handler.get_next_seed())
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")

    beta_eta_combined = (
        organic_rf_vars[constants.BETA_ORF][..., backend.newaxis, :]
        + organic_rf_vars[constants.ETA_ORF][..., backend.newaxis, :]
        * beta_gorf_dev
    )
    beta_gorf_value = (
        beta_eta_combined
        if ctx.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else backend.exp(beta_eta_combined)
    )
    organic_rf_vars[constants.BETA_GORF] = backend.tfd.Deterministic(
        beta_gorf_value, name=constants.BETA_GORF
    ).sample(seed=rng_handler.get_next_seed())

    return organic_rf_vars

  def _sample_controls_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from the prior distributions of the control variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.

    Returns:
      A mapping of control parameter names to a tensor of shape `[n_draws,
      n_geos, n_controls]` or `[n_draws, n_controls]` containing the samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    controls_vars = {
        constants.GAMMA_C: prior.gamma_c.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
        constants.XI_C: prior.xi_c.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    gamma_gc_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_controls],
        name=constants.GAMMA_GC_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
    controls_vars[constants.GAMMA_GC] = backend.tfd.Deterministic(
        controls_vars[constants.GAMMA_C][..., backend.newaxis, :]
        + controls_vars[constants.XI_C][..., backend.newaxis, :] * gamma_gc_dev,
        name=constants.GAMMA_GC,
    ).sample(seed=rng_handler.get_next_seed())
    return controls_vars

  def _sample_non_media_treatments_priors(
      self,
      n_draws: int,
      rng_handler: backend.RNGHandler,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws from the prior distributions of the non-media treatment variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      rng_handler: The backend-agnostic RNG handler managing the seed state.
      batch_size: Maximum number of draws to process per batch when computing
        `gamma_n`.

    Returns:
      A mapping of non-media treatment parameter names to a tensor of shape
      `[n_draws, n_geos, n_non_media_channels]` or
      `[n_draws, n_non_media_channels]` containing the samples.
    """
    ctx = self._model_context

    prior = ctx.prior_broadcast
    sample_shape = [1, n_draws]

    non_media_treatments_vars = {
        constants.XI_N: prior.xi_n.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        ),
    }
    gamma_gn_dev = backend.tfd.Sample(
        backend.tfd.Normal(
            loc=backend.to_tensor(0.0, dtype=backend.float_dtype),
            scale=backend.to_tensor(1.0, dtype=backend.float_dtype),
        ),
        [ctx.n_geos, ctx.n_non_media_channels],
        name=constants.GAMMA_GN_DEV,
    ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
    prior_type = ctx.model_spec.non_media_treatments_prior_type
    if prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      non_media_treatments_vars[constants.GAMMA_N] = prior.gamma_n.sample(
          sample_shape=sample_shape, seed=rng_handler.get_next_seed()
      )
    elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      non_media_treatments_vars[constants.CONTRIBUTION_N] = (
          prior.contribution_n.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          )
      )
      incremental_outcome_n = (
          non_media_treatments_vars[constants.CONTRIBUTION_N]
          * ctx.total_outcome
      )
      # The baseline values are already population-scaled.
      baseline_scaled = ctx.non_media_transformer.forward(  # pyrefly: ignore[missing-attribute]
          self._model_equations.compute_non_media_treatments_baseline(),
          apply_population_scaling=False,
      )
      linear_predictor_counterfactual_difference = (
          ctx.non_media_treatments_normalized - baseline_scaled
      )

      def _compute_gamma_n(
          incremental_outcome: backend.Tensor,
          xi_n: backend.Tensor,
          gamma_gx_dev: backend.Tensor,
      ) -> backend.Tensor:
        return self._model_equations.calculate_beta_x(
            is_non_media=True,
            incremental_outcome_x=incremental_outcome,
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=xi_n,
            beta_gx_dev=gamma_gx_dev,
        )

      gamma_n_value = _compute_in_draw_batches(
          _compute_gamma_n,
          draw_args=[
              incremental_outcome_n,
              non_media_treatments_vars[constants.XI_N],
              gamma_gn_dev,
          ],
          n_draws=n_draws,
          batch_size=batch_size,
      )
      non_media_treatments_vars[constants.GAMMA_N] = backend.tfd.Deterministic(
          gamma_n_value, name=constants.GAMMA_N
      ).sample(seed=rng_handler.get_next_seed())
    else:
      raise ValueError(f"Unsupported prior type: {prior_type}")
    non_media_treatments_vars[constants.GAMMA_GN] = backend.tfd.Deterministic(
        non_media_treatments_vars[constants.GAMMA_N][..., backend.newaxis, :]
        + non_media_treatments_vars[constants.XI_N][..., backend.newaxis, :]
        * gamma_gn_dev,
        name=constants.GAMMA_GN,
    ).sample(seed=rng_handler.get_next_seed())
    return non_media_treatments_vars

  def __call__(
      self,
      n_draws: int,
      seed: int | None = None,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> Mapping[str, backend.Tensor]:
    """Draws samples from prior distributions.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
      batch_size: Maximum number of draws to process per batch when computing
        media/RF transformations and derived parameters. Must be at least 1.

    Returns:
      A mapping of prior parameter names to tensors of the samples.

    Raises:
      ValueError: If `batch_size < 1`.
    """
    if batch_size < 1:
      raise ValueError(f"`batch_size` must be at least 1, got {batch_size}.")
    ctx = self._model_context

    # For stateful sampling, the random seed must be set to ensure that any
    # random numbers that are generated are deterministic.
    if seed is not None:
      backend.set_random_seed(seed)

    rng_handler = backend.RNGHandler(seed)
    # Compute the context's cached tensors now. Otherwise a compiled function
    # could compute and cache one while it's being compiled, and that cached
    # value can't be used afterwards.
    ctx.populate_cached_properties()

    prior = ctx.prior_broadcast
    # `sample_shape` is prepended to the shape of each BatchBroadcast in `prior`
    # when it is sampled.
    sample_shape = [1, n_draws]
    baseline_constraint = ctx.aggregate_baseline_constraint
    tau_g_excl_baseline = prior.tau_g_excl_baseline.sample(
        sample_shape=sample_shape, seed=rng_handler.get_next_seed()
    )

    # TODO: Deduplicate sigma and tau_g sampling across branches.
    if (
        baseline_constraint is None
    ):  # Meaning, baseline is allowed to go negative.
      base_vars = {
          constants.KNOT_VALUES: prior.knot_values.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          ),
          constants.SIGMA: prior.sigma.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          ),
          constants.TAU_G: (
              _get_tau_g(
                  tau_g_excl_baseline=tau_g_excl_baseline,
                  baseline_geo_idx=ctx.baseline_geo_idx,
              ).sample(seed=rng_handler.get_next_seed())
          ),
      }

      base_vars[constants.MU_T] = backend.tfd.Deterministic(
          backend.einsum(
              "...k,kt->...t",
              base_vars[constants.KNOT_VALUES],
              backend.to_tensor(ctx.knot_info.weights),
          ),
          name=constants.MU_T,
      ).sample(seed=rng_handler.get_next_seed())

      # Omit gamma_c, xi_c, and gamma_gc parameters from sampled distributions
      # if there are no control variables in the model.
      if ctx.n_controls:
        base_vars |= self._sample_controls_priors(n_draws, rng_handler)

    else:
      base_vars = {
          constants.SIGMA: prior.sigma.sample(
              sample_shape=sample_shape, seed=rng_handler.get_next_seed()
          ),
          constants.TAU_G: (
              _get_tau_g(
                  tau_g_excl_baseline=tau_g_excl_baseline,
                  baseline_geo_idx=ctx.baseline_geo_idx,
              ).sample(seed=rng_handler.get_next_seed())
          ),
      }

      if ctx.n_controls:
        base_vars |= self._sample_controls_priors(n_draws, rng_handler)

      if ctx.non_media_treatments_normalized is not None:
        base_vars |= self._sample_non_media_treatments_priors(  # pyrefly: ignore[unsupported-operation]
            n_draws, rng_handler, batch_size=batch_size
        )

      if ctx.non_media_transformer is not None:
        # The baseline values are already population-scaled.
        non_media_baseline_normalized = ctx.non_media_transformer.forward(
            self._model_equations.compute_non_media_treatments_baseline(),
            apply_population_scaling=False,
        )
      else:
        non_media_baseline_normalized = None
      offset = baseline_constraint.offset(
          tau_g=base_vars[constants.TAU_G],
          gamma_gc=base_vars.get(constants.GAMMA_GC),
          gamma_gn=base_vars.get(constants.GAMMA_GN),
          non_media_baseline_normalized=non_media_baseline_normalized,
      )

      knot_dtype = baseline_constraint.knot_scale.dtype
      rotated_knot_0 = baseline_constraint.rotated_knot_0_prior(offset).sample(
          seed=rng_handler.get_next_seed()
      )
      rotated_knot_rest_prior = baseline_constraint.rotated_knot_rest_prior()
      if rotated_knot_rest_prior is not None:
        rotated_knot_rest = rotated_knot_rest_prior.sample(
            sample_shape=sample_shape, seed=rng_handler.get_next_seed()
        )
      else:
        rotated_knot_rest = None
      knot_values = baseline_constraint.knot_values(
          rotated_knot_0, rotated_knot_rest
      )

      base_vars[constants.KNOT_VALUES] = knot_values
      base_vars[constants.MU_T] = backend.tfd.Deterministic(
          backend.einsum(
              "...k,kt->...t",
              knot_values,
              backend.to_tensor(ctx.knot_info.weights, dtype=knot_dtype),
          ),
          name=constants.MU_T,
      ).sample(seed=rng_handler.get_next_seed())

    if ctx.media_tensors.media is not None:
      base_vars |= self._sample_media_priors(
          n_draws, rng_handler, batch_size=batch_size
      )
    if ctx.rf_tensors.reach is not None:
      base_vars |= self._sample_rf_priors(
          n_draws, rng_handler, batch_size=batch_size
      )
    if ctx.organic_media_tensors.organic_media is not None:
      base_vars |= self._sample_organic_media_priors(
          n_draws, rng_handler, batch_size=batch_size
      )
    if ctx.organic_rf_tensors.organic_reach is not None:
      base_vars |= self._sample_organic_rf_priors(
          n_draws, rng_handler, batch_size=batch_size
      )
    # The constrained path has already sampled the non-media treatments, before
    # the knots.
    if (
        baseline_constraint is None
        and ctx.non_media_treatments_normalized is not None
    ):
      base_vars |= self._sample_non_media_treatments_priors(
          n_draws, rng_handler, batch_size=batch_size
      )

    return base_vars
