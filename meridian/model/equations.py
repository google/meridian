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

"""Core mathematical equations for the Meridian model.

This module defines the `ModelEquations` class, which encapsulates the stateless
mathematical functions used in the Meridian MMM. This includes the core model
definitions, such as adstock, hill, and other transformations used
during model fitting. It requires a `ModelContext` instance for data access.
"""

from collections.abc import Mapping, Sequence
import numbers

from meridian import backend
from meridian import constants
from meridian.model import adstock_hill
from meridian.model import changepoints
from meridian.model import context

__all__ = [
    "ModelEquations",
]


def _solve_coefficient_mean(
    *,
    incremental_outcome_x: backend.Tensor,
    outcome_per_unit: backend.Tensor,
    eta_x: backend.Tensor,
    beta_gx_dev: backend.Tensor,
    random_effects_normal: bool,
    time_effect_tx: backend.Tensor | None = None,
) -> backend.Tensor:
  """Solves the coefficient mean from the incremental outcome it implies.

  The coefficient in geo `g` (and time period `t`, if there is a time effect) is
  `beta_x + eta_x * beta_gx_dev + time_effect_tx` for normal random effects, or
  `exp` of that for log-normal ones. `beta_x` is chosen so that the incremental
  outcome summed over geos (and time periods) equals `incremental_outcome_x`.

  Args:
    incremental_outcome_x: The incremental outcome, shape `(..., n_channels)`.
    outcome_per_unit: The incremental outcome per unit coefficient, shape `(...,
      n_geos, n_channels)`, or `(..., n_geos, n_times, n_channels)` with a time
      effect. This is the term `M_{g,t,m}` of the ROI parameterization, summed
      over time periods when there is no time effect:
      https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations#roi
    eta_x: The geo random effect standard deviation parameter values, shape
      `(..., n_channels)`.
    beta_gx_dev: The latent standard normal parameter values of the geo-level
      coefficients, shape `(..., n_geos, n_channels)`.
    random_effects_normal: Whether the random effects are normal rather than
      log-normal.
    time_effect_tx: The time effect, shape `(..., n_times, n_channels)`, or
      `None` for no time effect.

  Returns:
    The coefficient mean, shape `(..., n_channels)`.
  """
  units = "g" if time_effect_tx is None else "gt"
  if random_effects_normal:
    numerator_term_x = backend.einsum(
        f"...{units}x,...gx,...x->...x",
        outcome_per_unit,
        beta_gx_dev,
        eta_x,
    )
    if time_effect_tx is not None:
      numerator_term_x += backend.einsum(
          "...gtx,...tx->...x", outcome_per_unit, time_effect_tx
      )
    denominator_term_x = backend.einsum(f"...{units}x->...x", outcome_per_unit)
    return (incremental_outcome_x - numerator_term_x) / denominator_term_x
  # For log-normal random effects, beta_x and eta_x are not mean & std.
  # The parameterization is beta_gx ~ exp(beta_x + eta_x * N(0, 1)).
  random_effects = beta_gx_dev * eta_x[..., backend.newaxis, :]
  if time_effect_tx is not None:
    random_effects = (
        random_effects[..., :, backend.newaxis, :]
        + time_effect_tx[..., backend.newaxis, :, :]
    )
  denominator_term_x = backend.einsum(
      f"...{units}x,...{units}x->...x",
      outcome_per_unit,
      backend.exp(random_effects),
  )
  # pyrefly: ignore[bad-argument-type]
  log_incremental_outcome_x = backend.log(incremental_outcome_x)
  return log_incremental_outcome_x - backend.log(denominator_term_x)


class ModelEquations:
  """Provides core, stateless mathematical functions for Meridian MMM."""

  def __init__(self, model_context: context.ModelContext):
    self._context = model_context

  def _resolve_n_times_output(
      self,
      n_times: int,
      n_times_output: int | None = None,
      tensor_name: str = constants.MEDIA,
  ) -> int:
    """Resolves `n_times_output` against `n_media_times` and `n_times`."""
    if n_times_output is None and n_times == self._context.n_media_times:
      return self._context.n_times
    if n_times_output is None:
      raise ValueError(
          "n_times_output is required. This argument is only optional when "
          f"`{tensor_name}` has a number of time periods equal to"
          " `n_media_times`."
      )
    return n_times_output

  def get_saturation_mask(
      self,
      saturation_spec: str | Sequence[str],
      n_channels: int,
  ) -> backend.Tensor:
    """Returns a boolean mask indicating which channels use Hill saturation.

    Args:
      saturation_spec: String or sequence of strings denoting the saturation
        function(s) ('hill' or 'none') for each channel.
      n_channels: Number of channels when `saturation_spec` is a single string.

    Returns:
      A 1D boolean tensor of shape `(n_channels,)` where `True` indicates Hill
      saturation and `False` indicates linear ('none') saturation.

    Raises:
      ValueError: If `saturation_spec` contains an unrecognized saturation
        function name.
    """
    valid_saturation_functions = {constants.HILL, constants.NONE}
    if isinstance(saturation_spec, str):
      if saturation_spec not in valid_saturation_functions:
        raise ValueError(
            f"Invalid saturation_spec: {saturation_spec}. Must be"
            f" '{constants.HILL}' or '{constants.NONE}'."
        )
      if saturation_spec == constants.HILL:
        return backend.ones((n_channels,), dtype=backend.bool_)
      return backend.zeros((n_channels,), dtype=backend.bool_)
    for s in saturation_spec:
      if s not in valid_saturation_functions:
        raise ValueError(
            f"Invalid saturation function in saturation_spec: {s}. Must be"
            f" '{constants.HILL}' or '{constants.NONE}'."
        )
    return backend.to_tensor(
        [s == constants.HILL for s in saturation_spec], dtype=backend.bool_
    )

  def prepare_adstock_hill_media(
      self,
      *,
      media: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      n_times_output: int | None = None,
      time_indices: backend.Tensor | None = None,
  ) -> tuple[backend.Tensor, backend.Tensor, backend.Tensor]:
    """Precomputes base media and Hill powers `(media_base, base_t1, t2)`.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_media_channels)`
        containing non-negative media execution values.
      alpha: Tensor of dimensions `[..., n_media_channels]` for Adstock decay.
      ec: Tensor of dimensions `[..., n_media_channels]` for Hill half-max
        effective concentration.
      slope: Tensor of dimensions `[..., n_media_channels]` for Hill slope.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      n_times_output: Number of time periods to output. Optional when the number
        of time periods in `media` equals `n_media_times`.
      time_indices: Optional 1D index tensor along the output time axis to slice
        `media_base` before computing `media_base ** slope` when
        `hill_before_adstock=False`.

    Returns:
      A tuple `(media_base, base_t1, t2)` where `media_base` is either raw
      `media` (when `hill_before_adstock=True`) or Adstock-transformed `media`
      (when `hill_before_adstock=False`), `base_t1` is `media_base ** slope`,
      and `t2` is `(ec ** slope)[..., newaxis, newaxis, :]`.
    """
    n_times_output = self._resolve_n_times_output(
        media.shape[1], n_times_output, constants.MEDIA
    )
    if self._context.model_spec.hill_before_adstock:
      media_base = media
    else:
      media_base = adstock_hill.AdstockTransformer(
          alpha=alpha,
          max_lag=self._context.model_spec.max_lag,
          n_times_output=n_times_output,
          decay_functions=decay_functions,
      ).forward(media)
      if time_indices is not None:
        media_base = backend.gather(media_base, time_indices, axis=-2)
    base_t1, t2 = adstock_hill.compute_hill_powers(
        media=media_base, ec=ec, slope=slope
    )
    return media_base, base_t1, t2

  def apply_adstock_hill_media(
      self,
      *,
      multiplier: backend.Tensor,
      alpha: backend.Tensor,
      slope: backend.Tensor,
      media_base: backend.Tensor,
      base_t1: backend.Tensor,
      t2: backend.Tensor,
      saturation_mask: backend.Tensor,
      n_times_output: int,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      time_indices: backend.Tensor | None = None,
  ) -> backend.Tensor:
    """Applies a spend multiplier, Hill saturation, and (if Hill-first) Adstock.

    Args:
      multiplier: Tensor of dimensions `(n_media_channels,)` (or `[...,
        n_media_channels]`) containing non-negative per-channel media spend
        multipliers.
      alpha: Tensor of dimensions `[..., n_media_channels]` for Adstock decay.
      slope: Tensor of dimensions `[..., n_media_channels]` for Hill slope.
      media_base: Precomputed base media tensor from
        `prepare_adstock_hill_media`.
      base_t1: Precomputed `media_base ** slope` tensor from
        `prepare_adstock_hill_media`.
      t2: Precomputed `(ec ** slope)[..., newaxis, newaxis, :]` tensor from
        `prepare_adstock_hill_media`.
      saturation_mask: Boolean mask of shape `(n_media_channels,)` indicating
        which channels use Hill saturation.
      n_times_output: Number of output time periods before `time_indices`
        slicing.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      time_indices: Optional 1D index tensor along the output time axis when
        `hill_before_adstock=True`.

    Returns:
      Transformed media tensor with dimensions `[..., n_geos, n_times,
      n_media_channels]`.
    """
    multiplier_power = (multiplier**slope)[
        ..., backend.newaxis, backend.newaxis, :
    ]
    t1 = multiplier_power * base_t1
    hill_out = t1 / (t1 + t2)
    selected = backend.where(saturation_mask, hill_out, multiplier * media_base)  # pyrefly: ignore[bad-argument-type]
    if self._context.model_spec.hill_before_adstock:
      selected = adstock_hill.AdstockTransformer(
          alpha=alpha,
          max_lag=self._context.model_spec.max_lag,
          n_times_output=n_times_output,
          decay_functions=decay_functions,
      ).forward(selected)
      if time_indices is not None:
        selected = backend.gather(selected, time_indices, axis=-2)
    return selected

  def adstock_hill_media(
      self,
      *,
      media: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      saturation_spec: str | Sequence[str] = constants.HILL,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    """Transforms media using Adstock and Hill functions in the desired order.

    If `model_spec.hill_before_adstock` is `False` (the default), this computes
    `Hill(Adstock(media))`. Otherwise, it computes `Adstock(Hill(media))`.
    Adstock uses `alpha` and `model_spec.max_lag`, and Hill uses `ec` and
    `slope`. See `AdstockTransformer` and `HillTransformer` for the definitions
    of Adstock and Hill. Channels whose `saturation_spec` is 'none' skip the
    Hill step.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_channels)`
        containing non-negative media execution values, where `n_channels` is
        the number of paid or organic media channels (`n_media_channels` or
        `n_organic_media_channels`). Typically this is impressions, but it can
        be any metric, such as `media_spend`. Clicks are often used for paid
        search ads.
      alpha: Tensor of Adstock decay parameters with dimensions `[...,
        n_channels]`.
      ec: Tensor of Hill half-saturation parameters with dimensions `[...,
        n_channels]`.
      slope: Tensor of Hill slope parameters with dimensions `[...,
        n_channels]`.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      saturation_spec: String or sequence of strings denoting the saturation
        function(s) for each channel. Default: 'hill'.
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `media` equals
        `n_media_times`, in which case `n_times_output` defaults to `n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_channels]`
      representing Adstock and Hill-transformed media.
    """
    n_times_output = self._resolve_n_times_output(
        media.shape[1], n_times_output, constants.MEDIA
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self._context.model_spec.max_lag,
        n_times_output=n_times_output,
        decay_functions=decay_functions,
    )
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    mask = self.get_saturation_mask(saturation_spec, media.shape[-1])

    if self._context.model_spec.hill_before_adstock:
      hill_out = hill_transformer.forward(media)
      selected_media = backend.where(mask, hill_out, media)  # pyrefly: ignore[bad-argument-type]
      return adstock_transformer.forward(selected_media)

    adstock_out = adstock_transformer.forward(media)
    hill_out = hill_transformer.forward(adstock_out)
    return backend.where(mask, hill_out, adstock_out)  # pyrefly: ignore[bad-argument-type]

  def adstock_hill_rf(
      self,
      *,
      reach: backend.Tensor,
      frequency: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      saturation_spec: str | Sequence[str] = constants.HILL,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    """Transforms reach and frequency (RF) using Hill and Adstock functions.

    This computes `Adstock(reach * Hill(frequency))`. Hill is always applied to
    frequency before Adstock, regardless of `model_spec.hill_before_adstock`.
    Adstock uses `alpha` and `model_spec.max_lag`, and Hill uses `ec` and
    `slope`. See `AdstockTransformer` and `HillTransformer` for the definitions
    of Adstock and Hill. Channels whose `saturation_spec` is 'none' skip the
    Hill step.

    Args:
      reach: Tensor of dimensions `(n_geos, n_media_times, n_channels)`
        containing non-negative media for reach, where `n_channels` is the
        number of paid or organic RF channels (`n_rf_channels` or
        `n_organic_rf_channels`).
      frequency: Tensor of dimensions `(n_geos, n_media_times, n_channels)`
        containing non-negative media for frequency.
      alpha: Tensor of Adstock decay parameters with dimensions `[...,
        n_channels]`.
      ec: Tensor of Hill half-saturation parameters with dimensions `[...,
        n_channels]`.
      slope: Tensor of Hill slope parameters with dimensions `[...,
        n_channels]`.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      saturation_spec: String or sequence of strings denoting the saturation
        function(s) for each channel. Default: 'hill'.
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `reach` equals
        `n_media_times`, in which case `n_times_output` defaults to `n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_channels]`
      representing Hill and Adstock-transformed RF.
    """
    n_times_output = self._resolve_n_times_output(
        reach.shape[1], n_times_output, constants.REACH
    )
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self._context.model_spec.max_lag,
        n_times_output=n_times_output,
        decay_functions=decay_functions,
    )
    adj_frequency = hill_transformer.forward(frequency)
    mask = self.get_saturation_mask(saturation_spec, frequency.shape[-1])

    selected_frequency = backend.where(mask, adj_frequency, frequency)  # pyrefly: ignore[bad-argument-type]
    rf_out = adstock_transformer.forward(reach * selected_frequency)

    return rf_out

  def compute_non_media_treatments_baseline(
      self,
      non_media_baseline_values: (
          Mapping[str, str | float] | Sequence[str | float] | None
      ) = None,
  ) -> backend.Tensor:
    """Computes the baseline for each non-media treatment channel.

    Args:
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`, or mapping from non-media channel names to
        baseline values. Each element is either a float (which means that the
        fixed value will be used as baseline for the given channel) or one of
        the strings "min" or "max" (which mean that the global minimum or
        maximum value will be used as baseline for the values of the given
        non_media treatment channel). If float values are provided, it is
        expected that they are scaled by population for the channels where
        `model_spec.non_media_population_scaling_id` is `True`. If `None`, the
        `model_spec.non_media_baseline_values` is used, which defaults to the
        minimum value for each non_media treatment channel.

    Returns:
      A tensor of shape `(n_non_media_channels,)` containing the
      baseline values for each non-media treatment channel.
    """
    if non_media_baseline_values is None:
      non_media_baseline_values_filled = (
          self._context.compiled_non_media_baseline_values
      )
    else:
      non_media_baseline_values_filled = (
          self._context.resolve_non_media_baseline_values(
              non_media_baseline_values
          )
      )

    no_op_scaling_factor = backend.ones_like(self._context.population)[
        :, backend.newaxis, backend.newaxis
    ]
    if self._context.compiled_non_media_population_scaling_id is not None:
      scaling_factors = backend.where(
          self._context.compiled_non_media_population_scaling_id,
          self._context.population[:, backend.newaxis, backend.newaxis],
          no_op_scaling_factor,
      )
    else:
      scaling_factors = no_op_scaling_factor

    non_media_treatments_population_scaled = backend.divide_no_nan(
        self._context.non_media_treatments, scaling_factors
    )

    if non_media_baseline_values_filled is None:
      # No baseline values anywhere: use the minimum value for each non-media
      # treatment channel as the baseline.
      non_media_baseline_values_filled = [
          constants.NON_MEDIA_BASELINE_MIN
      ] * non_media_treatments_population_scaled.shape[-1]

    if non_media_treatments_population_scaled.shape[-1] != len(
        non_media_baseline_values_filled
    ):
      raise ValueError(
          "The number of non-media channels"
          f" ({non_media_treatments_population_scaled.shape[-1]}) does not"
          " match the number of baseline values"
          f" ({len(non_media_baseline_values_filled)})."
      )

    baseline_list = []
    for channel in range(non_media_treatments_population_scaled.shape[-1]):
      baseline_value = non_media_baseline_values_filled[channel]

      if baseline_value == constants.NON_MEDIA_BASELINE_MIN:
        baseline_for_channel = backend.reduce_min(
            non_media_treatments_population_scaled[..., channel], axis=[0, 1]
        )
      elif baseline_value == constants.NON_MEDIA_BASELINE_MAX:
        baseline_for_channel = backend.reduce_max(
            non_media_treatments_population_scaled[..., channel], axis=[0, 1]
        )
      elif isinstance(baseline_value, numbers.Number):
        baseline_for_channel = backend.to_tensor(
            baseline_value, dtype=backend.float_dtype
        )
      else:
        raise ValueError(
            f"Invalid non_media_baseline_values value: '{baseline_value}'. Only"
            " float numbers and strings 'min' and 'max' are supported."
        )

      baseline_list.append(baseline_for_channel)

    return backend.stack(baseline_list, axis=-1)

  def linear_predictor_counterfactual_difference_media(
      self,
      *,
      media_transformed: backend.Tensor,
      alpha_m: backend.Tensor,
      ec_m: backend.Tensor,
      slope_m: backend.Tensor,
  ) -> backend.Tensor:
    """Calculates linear predictor counterfactual difference for non-RF media.

    For non-RF media variables (paid or organic), this function calculates the
    linear predictor difference between the treatment variable and its
    counterfactual. "Linear predictor" refers to the output of the hill/adstock
    function, which is multiplied by the geo-level coefficient.

    This function does the calculation efficiently by only calling the
    hill/adstock function if the prior counterfactual is not all zeros.

    Args:
      media_transformed: The output of the hill/adstock function for actual
        historical media data.
      alpha_m: The adstock alpha parameter values.
      ec_m: The Hill ec parameter values.
      slope_m: The Hill slope parameter values.

    Returns:
      The linear predictor difference between the treatment variable and its
      counterfactual.
    """
    if self._context.media_tensors.prior_media_scaled_counterfactual is None:
      return media_transformed
    media_transformed_counterfactual = self.adstock_hill_media(
        media=self._context.media_tensors.prior_media_scaled_counterfactual,
        alpha=alpha_m,
        ec=ec_m,
        slope=slope_m,
        decay_functions=self._context.adstock_decay_spec.media,
    )
    # Absolute values is needed because the difference is negative for mROI
    # priors and positive for ROI and contribution priors.
    return backend.absolute(
        media_transformed - media_transformed_counterfactual
    )

  def linear_predictor_counterfactual_difference_rf(
      self,
      *,
      rf_transformed: backend.Tensor,
      alpha_rf: backend.Tensor,
      ec_rf: backend.Tensor,
      slope_rf: backend.Tensor,
  ) -> backend.Tensor:
    """Calculates linear predictor counterfactual difference for RF media.

    For RF media variables (paid or organic), this function calculates the
    linear predictor difference between the treatment variable and its
    counterfactual. "Linear predictor" refers to the output of the hill/adstock
    function, which is multiplied by the geo-level coefficient.

    This function does the calculation efficiently by only calling the
    hill/adstock function if the prior counterfactual is not all zeros.

    Args:
      rf_transformed: The output of the hill/adstock function for actual
        historical media data.
      alpha_rf: The adstock alpha parameter values.
      ec_rf: The Hill ec parameter values.
      slope_rf: The Hill slope parameter values.

    Returns:
      The linear predictor difference between the treatment variable and its
      counterfactual.
    """
    if self._context.rf_tensors.prior_reach_scaled_counterfactual is None:
      return rf_transformed
    rf_transformed_counterfactual = self.adstock_hill_rf(
        reach=self._context.rf_tensors.prior_reach_scaled_counterfactual,
        frequency=self._context.rf_tensors.frequency,  # pyrefly: ignore[bad-argument-type]
        alpha=alpha_rf,
        ec=ec_rf,
        slope=slope_rf,
        decay_functions=self._context.adstock_decay_spec.rf,
    )
    # Absolute values is needed because the difference is negative for mROI
    # priors and positive for ROI and contribution priors.
    return backend.absolute(rf_transformed - rf_transformed_counterfactual)

  def calculate_beta_x(
      self,
      *,
      is_non_media: bool,
      incremental_outcome_x: backend.Tensor,
      linear_predictor_counterfactual_difference: backend.Tensor,
      eta_x: backend.Tensor,
      beta_gx_dev: backend.Tensor,
  ) -> backend.Tensor:
    """Calculates coefficient mean parameter for any treatment variable type.

    The "beta_x" in the function name refers to the coefficient mean parameter
    of any treatment variable. The "x" can represent "m", "rf", "om", or "orf".
    This function can also be used to calculate "gamma_n" for any non-media
    treatments.

    Args:
      is_non_media: Boolean indicating whether the treatment variable is a
        non-media treatment. This argument is used to determine whether the
        coefficient random effects are normal or log-normal. If `True`, then
        random effects are assumed to be normal. Otherwise, the distribution is
        inferred from `self._context.media_effects_dist`.
      incremental_outcome_x: The incremental outcome of the treatment variable,
        which depends on the parameter values of a particular prior or posterior
        draw. The "_x" indicates that this is a tensor with length equal to the
        dimension of the treatment variable.
      linear_predictor_counterfactual_difference: The difference between the
        treatment variable and its counterfactual on the linear predictor scale.
        "Linear predictor" refers to the quantity that is multiplied by the
        geo-level coefficient. For media variables, this is the output of the
        hill/adstock transformation function. For non-media treatments, this is
        simply the treatment variable after centering/scaling transformations.
        This tensor has dimensions for geo, time, and channel.
      eta_x: The random effect standard deviation parameter values. For media
        variables, the "x" represents "m", "rf", "om", or "orf". For non-media
        treatments, this argument should be set to `xi_n`, which is analogous to
        "eta".
      beta_gx_dev: The latent standard normal parameter values of the geo-level
        coefficients. For media variables, the "x" represents "m", "rf", "om",
        or "orf". For non-media treatments, this argument should be set to
        `gamma_gn_dev`, which is analogous to "beta_gx_dev".

    Returns:
      The coefficient mean parameter of the treatment variable, which has
      dimension equal to the number of treatment channels.
    """
    if is_non_media:
      random_effects_normal = True
    else:
      random_effects_normal = (
          self._context.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
      )
    return _solve_coefficient_mean(
        incremental_outcome_x=incremental_outcome_x,
        outcome_per_unit=self._outcome_per_unit_coefficient(
            linear_predictor_counterfactual_difference, sum_over_time=True
        ),
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
        random_effects_normal=random_effects_normal,
    )

  def _calculate_time_varying_beta_x(
      self,
      *,
      incremental_outcome_x: backend.Tensor,
      linear_predictor_counterfactual_difference: backend.Tensor,
      eta_x: backend.Tensor,
      beta_gx_dev: backend.Tensor,
      zeta_x: backend.Tensor,
      phi_tx: backend.Tensor,
  ) -> backend.Tensor:
    """Calculates the coefficient mean of paid channels with changepoints.

    This is `calculate_beta_x` with the time effect of channels that have
    changepoints. Their coefficient in geo `g` and time period `t` is
    `exp(beta_x + eta_x * beta_gx_dev + zeta_x * phi_tx)`, or the same without
    `exp` for normal random effects. `beta_x` is chosen so that the incremental
    outcome over all geos and time periods, including the time effect, equals
    `incremental_outcome_x`. This keeps ROI, mROI and contribution priors on the
    whole training period, as for channels without changepoints. All inputs
    only include the channels with changepoints.

    Args:
      incremental_outcome_x: The incremental outcome implied by the prior or
        posterior draw, shape `(..., n_channels)`.
      linear_predictor_counterfactual_difference: The difference between the
        treatment and its counterfactual on the linear predictor scale, shape
        `(..., n_geos, n_times, n_channels)`.
      eta_x: The geo random effect standard deviation parameter values, shape
        `(..., n_channels)`. "x" represents "m" or "rf".
      beta_gx_dev: The latent standard normal parameter values of the geo-level
        coefficients, shape `(..., n_geos, n_channels)`. "x" represents "m" or
        "rf".
      zeta_x: The time random effect standard deviation parameter values, shape
        `(..., n_channels)`. "x" represents "m" or "rf".
      phi_tx: The time effect from `changepoints.expand_intervals`, shape `(...,
        n_times, n_channels)`.

    Returns:
      The coefficient mean of the channels with changepoints, shape `(...,
      n_channels)`.
    """
    return _solve_coefficient_mean(
        incremental_outcome_x=incremental_outcome_x,
        outcome_per_unit=self._outcome_per_unit_coefficient(
            linear_predictor_counterfactual_difference, sum_over_time=False
        ),
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
        random_effects_normal=(
            self._context._changepoint_media_effects_dist  # pylint: disable=protected-access
            == constants.MEDIA_EFFECTS_NORMAL
        ),
        time_effect_tx=zeta_x[..., backend.newaxis, :] * phi_tx,
    )

  def _calculate_beta_x_with_changepoints(
      self,
      *,
      changepoint_info: changepoints.ChangepointInfo,
      incremental_outcome_x: backend.Tensor,
      linear_predictor_counterfactual_difference: backend.Tensor,
      eta_x: backend.Tensor,
      beta_gx_dev: backend.Tensor,
      zeta_x: backend.Tensor,
      varphi_x: backend.Tensor,
  ) -> backend.Tensor:
    """Calculates `beta_x` of a paid channel type that has changepoints.

    `calculate_beta_x` ignores the time effect. This function uses
    `calculate_beta_x` for the channels without changepoints, and
    `_calculate_time_varying_beta_x` for the channels with changepoints. "x"
    represents "m" or "rf".

    Args:
      changepoint_info: The `ChangepointInfo` of the channel type.
      incremental_outcome_x: As in `calculate_beta_x`, for all channels of the
        type, shape `(..., n_channels)`. Here, `n_channels` is the number of all
        channels of the type, not just the number of channels with changepoints.
      linear_predictor_counterfactual_difference: As in `calculate_beta_x`, for
        all channels.
      eta_x: As in `calculate_beta_x`, for all channels.
      beta_gx_dev: As in `calculate_beta_x`, for all channels.
      zeta_x: The time random effect standard deviation parameter values of the
        channels with changepoints, shape `(..., n_changepoint_channels)`.
      varphi_x: The latent standard normal parameter values of the intervals
        between changepoints, shape `(..., n_changepoint_channels,
        max_intervals)`.

    Returns:
      `beta_x` of all channels of the type, shape `(..., n_channels)`.
    """
    indices = backend.to_tensor(changepoint_info.channel_indices)
    time_varying_beta_x = self._calculate_time_varying_beta_x(
        incremental_outcome_x=backend.gather(
            incremental_outcome_x, indices, axis=-1
        ),
        linear_predictor_counterfactual_difference=backend.gather(
            linear_predictor_counterfactual_difference, indices, axis=-1
        ),
        eta_x=backend.gather(eta_x, indices, axis=-1),
        beta_gx_dev=backend.gather(beta_gx_dev, indices, axis=-1),
        zeta_x=zeta_x,
        phi_tx=changepoints.expand_intervals(
            varphi_x, changepoint_info.weights
        ),
    )
    # This runs `calculate_beta_x` on all channels of the type, with
    # `media_effects_dist`. `merge_channels` then replaces the results of the
    # channels with changepoints with `time_varying_beta_x`. Their inputs get
    # placeholder values, because when `changepoint_media_effects_dist` differs,
    # their real inputs could give infinite results here, and infinite values
    # break posterior sampling even when they're replaced.
    regular_beta_x = self.calculate_beta_x(
        is_non_media=False,
        incremental_outcome_x=changepoints.fill_channels(
            incremental_outcome_x, value=1.0, info=changepoint_info
        ),
        linear_predictor_counterfactual_difference=(
            linear_predictor_counterfactual_difference
        ),
        eta_x=eta_x,
        beta_gx_dev=changepoints.fill_channels(
            beta_gx_dev, value=0.0, info=changepoint_info
        ),
    )
    return changepoints.merge_channels(
        x=regular_beta_x, sub=time_varying_beta_x, info=changepoint_info
    )

  def _split_time_varying_outcome(
      self,
      *,
      changepoint_info: changepoints.ChangepointInfo,
      media_transformed: backend.Tensor,
      beta_gx: backend.Tensor,
      zeta_x: backend.Tensor,
      varphi_x: backend.Tensor,
  ) -> tuple[backend.Tensor, backend.Tensor]:
    """Computes the outcome of the channels with changepoints separately.

    Args:
      changepoint_info: The `ChangepointInfo` of the channel type.
      media_transformed: The adstock and Hill output for all channels of the
        type, shape `(..., n_geos, n_times, n_channels)`.
      beta_gx: The geo coefficients of all channels of the type without the time
        effect, shape `(..., n_geos, n_channels)`.
      zeta_x: The time random effect standard deviation parameter values of the
        channels with changepoints, shape `(..., n_changepoint_channels)`.
      varphi_x: The latent standard normal parameter values of the intervals
        between changepoints, shape `(..., n_changepoint_channels,
        max_intervals)`.

    Returns:
      A tuple of the channels with changepoints' summed contribution to the
      linear predictor, using each time period's coefficient, shape `(...,
      n_geos, n_times)`; and `beta_gx` with zeros for those channels, so that
      the regular calculation leaves them out.
    """
    indices = backend.to_tensor(changepoint_info.channel_indices)
    outcome = backend.einsum(
        "...gtx,...gtx->...gt",
        backend.gather(media_transformed, indices, axis=-1),
        self._changepoint_coefficients(
            changepoint_info=changepoint_info,
            beta_gx=beta_gx,
            zeta_x=zeta_x,
            varphi_x=varphi_x,
        ),
    )
    beta_gx_without = changepoints.fill_channels(
        beta_gx, value=0.0, info=changepoint_info
    )
    return outcome, beta_gx_without

  def _changepoint_coefficients(
      self,
      *,
      changepoint_info: changepoints.ChangepointInfo,
      beta_gx: backend.Tensor,
      zeta_x: backend.Tensor,
      varphi_x: backend.Tensor,
  ) -> backend.Tensor:
    """Returns the coefficients of the channels with changepoints by time.

    Args:
      changepoint_info: The `ChangepointInfo` of the channel type.
      beta_gx: The geo coefficients of all channels of the type without the time
        effect, shape `(..., n_geos, n_channels)`.
      zeta_x: The time random effect standard deviation parameter values of the
        channels with changepoints, shape `(..., n_changepoint_channels)`.
      varphi_x: The latent standard normal parameter values of the intervals
        between changepoints, shape `(..., n_changepoint_channels,
        max_intervals)`.

    Returns:
      The coefficients of the channels with changepoints, shape `(..., n_geos,
      n_times, n_changepoint_channels)`.
    """
    indices = backend.to_tensor(changepoint_info.channel_indices)
    return changepoints.time_varying_coefficients(
        beta_gx=backend.gather(beta_gx, indices, axis=-1),
        zeta=zeta_x,
        phi=changepoints.expand_intervals(varphi_x, changepoint_info.weights),
        media_effects_dist=self._context._changepoint_media_effects_dist,  # pylint: disable=protected-access
    )

  def _coefficients_by_time(
      self,
      *,
      changepoint_info: changepoints.ChangepointInfo,
      beta_gx: backend.Tensor,
      zeta_x: backend.Tensor,
      varphi_x: backend.Tensor,
  ) -> backend.Tensor:
    """Returns the coefficient of each geo and time period of a channel type.

    The channels without changepoints have the same coefficient in every time
    period. "x" represents "m" or "rf".

    Args:
      changepoint_info: The `ChangepointInfo` of the channel type.
      beta_gx: The geo coefficients of all channels of the type without the time
        effect, shape `(..., n_geos, n_channels)`.
      zeta_x: As in `_changepoint_coefficients`.
      varphi_x: As in `_changepoint_coefficients`.

    Returns:
      The coefficients, shape `(..., n_geos, n_times, n_channels)`.
    """
    n_times = changepoint_info.weights.shape[-1]
    regular_beta_gtx = backend.broadcast_to(
        beta_gx[..., :, backend.newaxis, :],
        [*beta_gx.shape[:-1], n_times, beta_gx.shape[-1]],
    )
    return changepoints.merge_channels(
        x=regular_beta_gtx,
        sub=self._changepoint_coefficients(
            changepoint_info=changepoint_info,
            beta_gx=beta_gx,
            zeta_x=zeta_x,
            varphi_x=varphi_x,
        ),
        info=changepoint_info,
    )

  def _coefficients_from_random_effects(
      self,
      beta_eta_combined: backend.Tensor,
      changepoint_info: changepoints.ChangepointInfo | None,
  ) -> backend.Tensor:
    """Returns the geo-level coefficients of the paid channels of one type.

    Channels with changepoints use `changepoint_media_effects_dist` and the
    others use `media_effects_dist`. "x" represents "m" or "rf".

    Args:
      beta_eta_combined: `beta_x + eta_x * beta_gx_dev`, shape `(..., n_geos,
        n_channels)`.
      changepoint_info: The channels of the type with changepoints, or `None`
        if none has changepoints.

    Returns:
      Per channel, `beta_eta_combined` for normal random effects, or its `exp`
      for log-normal ones.
    """
    normal = self._context.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
    changepoint_normal = (
        self._context._changepoint_media_effects_dist  # pylint: disable=protected-access
        == constants.MEDIA_EFFECTS_NORMAL
    )
    if changepoint_info is None or changepoint_normal == normal:
      if normal:
        return beta_eta_combined
      # pyrefly: ignore[bad-argument-type]
      return backend.exp(beta_eta_combined)
    indices = backend.to_tensor(changepoint_info.channel_indices)
    time_varying = backend.gather(beta_eta_combined, indices, axis=-1)
    # Placeholder zeros for the channels with changepoints, so that `exp` of
    # their unused values can't be infinite.
    regular = changepoints.fill_channels(
        beta_eta_combined, value=0.0, info=changepoint_info
    )
    if not normal:
      # pyrefly: ignore[bad-argument-type]
      regular = backend.exp(regular)
    if not changepoint_normal:
      # pyrefly: ignore[bad-argument-type]
      time_varying = backend.exp(time_varying)
    return changepoints.merge_channels(
        x=regular, sub=time_varying, info=changepoint_info
    )

  def _outcome_per_unit_coefficient(
      self,
      linear_predictor_counterfactual_difference: backend.Tensor,
      *,
      sum_over_time: bool,
  ) -> backend.Tensor:
    """Returns the incremental outcome per unit coefficient.

    This is the term `M_{g,t,m}` of the ROI parameterization in
    https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations#roi.
    With `sum_over_time`, as for channels without changepoints, it is summed
    over time periods.

    Args:
      linear_predictor_counterfactual_difference: The difference between the
        treatment and its counterfactual on the linear predictor scale, shape
        `(..., n_geos, n_times, n_channels)`.
      sum_over_time: Whether to sum over the time periods.

    Returns:
      Shape `(..., n_geos, n_channels)` if `sum_over_time`, else `(..., n_geos,
      n_times, n_channels)`.
    """
    if self._context.revenue_per_kpi is None:
      revenue_per_kpi = backend.ones(
          [self._context.n_geos, self._context.n_times],
          dtype=backend.float_dtype,
      )
    else:
      revenue_per_kpi = self._context.revenue_per_kpi
    output = "...gx" if sum_over_time else "...gtx"
    return backend.einsum(
        f"...gtx,gt,g,->{output}",
        linear_predictor_counterfactual_difference,
        revenue_per_kpi,
        self._context.population,
        self._context.kpi_transformer.population_scaled_stdev,
    )
