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
    fn: Callable[[slice], backend.Tensor],
    n_draws: int,
    batch_size: int,
) -> backend.Tensor:
  """Evaluates `fn(draw_slice)` in batches along the draw dimension (axis=1)."""
  if n_draws <= batch_size:
    return fn(slice(None))
  batches = [
      fn(slice(start, min(start + batch_size, n_draws)))
      for start in range(0, n_draws, batch_size)
  ]
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
  shape = tau_g_excl_baseline.shape[:-1] + (1,)
  tau_g = backend.concatenate(
      [
          tau_g_excl_baseline[..., :baseline_geo_idx],
          backend.zeros(shape, dtype=tau_g_excl_baseline.dtype),
          tau_g_excl_baseline[..., baseline_geo_idx:],
      ],
      axis=-1,
  )
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
  caller re-solves `beta` with `ModelEquations._merge_time_varying_beta_x`, so
  the prior keeps its whole-history meaning.

  Args:
    changepoint_info: The `ChangepointInfo` of the channel type.
    zeta_prior: The broadcast prior on the scale of the time effect.
    zeta_name: The name of the scale parameter.
    varphi_name: The name of the interval adjustments.
    sample_shape: The sample shape, `[1, n_draws]`.
    rng_handler: The backend-agnostic RNG handler managing the seed state.

  Returns:
    The draws of `zeta_name`, shape `[1, n_draws, n_changepoint_channels]`, and
    `varphi_name`, shape `[1, n_draws, n_changepoint_channels, max_intervals]`,
    with zeros in padded intervals.
  """
  zeta_draws = zeta_prior.sample(
      sample_shape=sample_shape, seed=rng_handler.get_next_seed()
  )
  varphi_draws = changepoints.interval_adjustments_distribution(
      changepoint_info, varphi_name
  ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
  return {
      zeta_name: zeta_draws,
      varphi_name: changepoints.pad_intervals(varphi_draws, changepoint_info),
  }


_MEDIA_TREATMENT_PARAMETERS = {
    constants.TREATMENT_PRIOR_TYPE_ROI: constants.ROI_M,
    constants.TREATMENT_PRIOR_TYPE_MROI: constants.MROI_M,
    constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION: constants.CONTRIBUTION_M,
    constants.TREATMENT_PRIOR_TYPE_COEFFICIENT: constants.BETA_M,
}
_RF_TREATMENT_PARAMETERS = {
    constants.TREATMENT_PRIOR_TYPE_ROI: constants.ROI_RF,
    constants.TREATMENT_PRIOR_TYPE_MROI: constants.MROI_RF,
    constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION: constants.CONTRIBUTION_RF,
    constants.TREATMENT_PRIOR_TYPE_COEFFICIENT: constants.BETA_RF,
}


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

      def _compute_beta_m_batch(draw_slice: slice) -> backend.Tensor:
        alpha_m = media_vars[constants.ALPHA_M][:, draw_slice]
        ec_m = media_vars[constants.EC_M][:, draw_slice]
        slope_m = media_vars[constants.SLOPE_M][:, draw_slice]
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
        beta_m = self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome_m[:, draw_slice],
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=media_vars[constants.ETA_M][:, draw_slice],
            beta_gx_dev=beta_gm_dev[:, draw_slice],
        )
        if time_varying_info is None:
          return beta_m
        return self._model_equations._merge_time_varying_beta_x(  # pylint: disable=protected-access
            changepoint_info=time_varying_info,
            beta_x=beta_m,
            incremental_outcome_x=incremental_outcome_m[:, draw_slice],
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=media_vars[constants.ETA_M][:, draw_slice],
            beta_gx_dev=beta_gm_dev[:, draw_slice],
            zeta_x=media_vars[constants.ZETA_M][:, draw_slice],
            varphi_x=media_vars[constants.VARPHI_M][:, draw_slice],
        )

      return _compute_in_draw_batches(
          _compute_beta_m_batch, n_draws=n_draws, batch_size=batch_size
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

    if changepoint_info is not None:
      # Padded intervals belong to no time period, so they have no draws.
      media_vars[constants.VARPHI_M] = backend.where(
          changepoint_info.mask, media_vars[constants.VARPHI_M], np.nan
      )
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
      prior_type = ctx.model_spec.effective_rf_prior_type
      prior_denominator = ctx.rf_tensors.prior_denominator
      parameters = _RF_TREATMENT_PARAMETERS
    else:
      changepoint_info = ctx._media_changepoint_info
      prior_type = ctx.model_spec.effective_media_prior_type
      prior_denominator = ctx.media_tensors.prior_denominator
      parameters = _MEDIA_TREATMENT_PARAMETERS
    changepoint_prior_type = prior_type
    if changepoint_info is not None:
      changepoint_prior_type = ctx._changepoint_prior_type(is_rf)
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

    def _changepoint_columns(x: backend.Tensor) -> backend.Tensor:
      assert changepoint_info is not None
      return backend.gather(
          x, backend.to_tensor(changepoint_info.channel_indices), axis=-1
      )

    solve_changepoint_info = changepoint_info
    if changepoint_prior_type == coefficient:
      treatment_parameter = draws[prior_type]
      # The coefficient prior replaces the solution of these channels.
      solve_changepoint_info = None
    elif prior_type in (coefficient, changepoint_prior_type):
      treatment_parameter = draws[changepoint_prior_type]
    else:
      assert changepoint_info is not None
      treatment_parameter = changepoints.merge_channels(
          x=draws[prior_type],
          sub=_changepoint_columns(draws[changepoint_prior_type]),
          info=changepoint_info,
      )
    if len(draws) > 1:
      assert changepoint_info is not None
      is_changepoint_channel = np.isin(
          np.arange(changepoint_info.n_channels_total),
          changepoint_info.channel_indices,
      )
      for draw_prior_type, value in draws.items():
        if draw_prior_type != coefficient:
          xvars[parameters[draw_prior_type]] = backend.where(
              is_changepoint_channel
              == (draw_prior_type == changepoint_prior_type),
              value,
              np.nan,
          )
    else:
      xvars[parameters[prior_type]] = treatment_parameter

    assert prior_denominator is not None
    beta_value = solve(
        treatment_parameter * prior_denominator, solve_changepoint_info
    )
    if prior_type == coefficient:
      assert changepoint_info is not None
      beta_value = changepoints.merge_channels(
          x=draws[coefficient],
          sub=_changepoint_columns(beta_value),
          info=changepoint_info,
      )
    elif changepoint_prior_type == coefficient:
      assert changepoint_info is not None
      beta_value = changepoints.merge_channels(
          x=beta_value,
          sub=_changepoint_columns(draws[coefficient]),
          info=changepoint_info,
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

      def _compute_beta_rf_batch(draw_slice: slice) -> backend.Tensor:
        alpha_rf = rf_vars[constants.ALPHA_RF][:, draw_slice]
        ec_rf = rf_vars[constants.EC_RF][:, draw_slice]
        slope_rf = rf_vars[constants.SLOPE_RF][:, draw_slice]
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
        beta_rf = self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome_rf[:, draw_slice],
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=rf_vars[constants.ETA_RF][:, draw_slice],
            beta_gx_dev=beta_grf_dev[:, draw_slice],
        )
        if time_varying_info is None:
          return beta_rf
        return self._model_equations._merge_time_varying_beta_x(  # pylint: disable=protected-access
            changepoint_info=time_varying_info,
            beta_x=beta_rf,
            incremental_outcome_x=incremental_outcome_rf[:, draw_slice],
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=rf_vars[constants.ETA_RF][:, draw_slice],
            beta_gx_dev=beta_grf_dev[:, draw_slice],
            zeta_x=rf_vars[constants.ZETA_RF][:, draw_slice],
            varphi_x=rf_vars[constants.VARPHI_RF][:, draw_slice],
        )

      return _compute_in_draw_batches(
          _compute_beta_rf_batch, n_draws=n_draws, batch_size=batch_size
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

    if changepoint_info is not None:
      # Padded intervals belong to no time period, so they have no draws.
      rf_vars[constants.VARPHI_RF] = backend.where(
          changepoint_info.mask, rf_vars[constants.VARPHI_RF], np.nan
      )
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

      def _compute_beta_om_batch(draw_slice: slice) -> backend.Tensor:
        organic_media_transformed = self._model_equations.adstock_hill_media(
            media=ctx.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
            alpha=organic_media_vars[constants.ALPHA_OM][:, draw_slice],
            ec=organic_media_vars[constants.EC_OM][:, draw_slice],
            slope=organic_media_vars[constants.SLOPE_OM][:, draw_slice],
            decay_functions=ctx.adstock_decay_spec.organic_media,
            saturation_spec=ctx.saturation_spec.organic_media,
        )
        return self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome_om[:, draw_slice],
            linear_predictor_counterfactual_difference=organic_media_transformed,
            eta_x=organic_media_vars[constants.ETA_OM][:, draw_slice],
            beta_gx_dev=beta_gom_dev[:, draw_slice],
        )

      beta_om_value = _compute_in_draw_batches(
          _compute_beta_om_batch, n_draws=n_draws, batch_size=batch_size
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

      def _compute_beta_orf_batch(draw_slice: slice) -> backend.Tensor:
        organic_rf_transformed = self._model_equations.adstock_hill_rf(
            reach=ctx.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
            frequency=ctx.organic_rf_tensors.organic_frequency,  # pyrefly: ignore[bad-argument-type]
            alpha=organic_rf_vars[constants.ALPHA_ORF][:, draw_slice],
            ec=organic_rf_vars[constants.EC_ORF][:, draw_slice],
            slope=organic_rf_vars[constants.SLOPE_ORF][:, draw_slice],
            decay_functions=ctx.adstock_decay_spec.organic_rf,
            saturation_spec=ctx.saturation_spec.organic_rf,
        )
        return self._model_equations.calculate_beta_x(
            is_non_media=False,
            incremental_outcome_x=incremental_outcome_orf[:, draw_slice],
            linear_predictor_counterfactual_difference=organic_rf_transformed,
            eta_x=organic_rf_vars[constants.ETA_ORF][:, draw_slice],
            beta_gx_dev=beta_gorf_dev[:, draw_slice],
        )

      beta_orf_value = _compute_in_draw_batches(
          _compute_beta_orf_batch, n_draws=n_draws, batch_size=batch_size
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

      def _compute_gamma_n_batch(draw_slice: slice) -> backend.Tensor:
        return self._model_equations.calculate_beta_x(
            is_non_media=True,
            incremental_outcome_x=incremental_outcome_n[:, draw_slice],
            linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
            eta_x=non_media_treatments_vars[constants.XI_N][:, draw_slice],
            beta_gx_dev=gamma_gn_dev[:, draw_slice],
        )

      gamma_n_value = _compute_in_draw_batches(
          _compute_gamma_n_batch, n_draws=n_draws, batch_size=batch_size
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

      geo_offset = float(ctx.n_times) * backend.einsum(
          "g,...g->...",
          baseline_constraint.geo_weights,
          base_vars[constants.TAU_G],
      )
      offset = geo_offset
      if ctx.n_controls:
        controls_offset = backend.einsum(
            "gc,...gc->...",
            baseline_constraint.controls_weights,
            base_vars[constants.GAMMA_GC],
        )
        offset = offset + controls_offset
      if (
          ctx.non_media_treatments_normalized is not None
          and ctx.non_media_transformer is not None
      ):
        non_media_baseline = (
            self._model_equations.compute_non_media_treatments_baseline()
        )
        # The baseline values are already population-scaled.
        non_media_baseline_scaled = ctx.non_media_transformer.forward(
            non_media_baseline, apply_population_scaling=False
        )
        broadcast_non_media = backend.broadcast_to(
            non_media_baseline_scaled,
            [ctx.n_geos, ctx.n_times, ctx.n_non_media_channels],
        )
        non_media_offset = backend.einsum(
            "gt,gtn,...gn->...",
            baseline_constraint.outcome_weights,
            broadcast_non_media,
            base_vars[constants.GAMMA_GN],
        )
        offset = offset + non_media_offset

      knot_dist = prior.knot_values
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
      rotated_knot_0 = backend.tfd.TruncatedNormal(
          loc=knot_loc,
          scale=knot_scale,
          low=low,
          high=high,
          name=constants.ROTATED_KNOT_0,
      ).sample(seed=rng_handler.get_next_seed())

      if ctx.knot_info.n_knots > 1:
        rotated_knot_rest = backend.tfd.Sample(
            backend.tfd.Normal(loc=knot_loc, scale=knot_scale),
            [ctx.knot_info.n_knots - 1],
            name=constants.ROTATED_KNOT_REST,
        ).sample(sample_shape=sample_shape, seed=rng_handler.get_next_seed())
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
