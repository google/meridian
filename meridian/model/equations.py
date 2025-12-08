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

"""Core mathematical equations for the Meridian model.

This module defines the `ModelEquations` class, which encapsulates the stateless
mathematical functions used in the Meridian MMM. This includes the core model
definitions, such as adstock, hill, and other transformations used
during model fitting. It requires a `ModelContext` instance for data access.
"""

from collections.abc import Sequence

from meridian import backend
from meridian import constants
from meridian.model import adstock_hill
from meridian.model import context


__all__ = [
    "ModelEquations",
]


class ModelEquations:
  """Provides core, stateless mathematical functions for Meridian MMM."""

  def __init__(self, model_context: context.ModelContext):
    self._context = model_context

  def adstock_hill_media(
      self,
      media: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    """Transforms media or using Adstock and Hill functions in the desired order.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_media_channels)`
        containing non-negative media execution values. Typically this is
        impressions, but it can be any metric, such as `media_spend`. Clicks are
        often used for paid search ads.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `media` equals
        `n_media_times`, in which case `n_times_output` defaults to `n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_media_channels]`
      representing Adstock and Hill-transformed media.
    """
    if n_times_output is None and (
        media.shape[1] == self._context.n_media_times
    ):
      n_times_output = self._context.n_times
    elif n_times_output is None:
      raise ValueError(
          "n_times_output is required. This argument is only optional when "
          "`media` has a number of time periods equal to `n_media_times`."
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
    transformers_list = (
        [hill_transformer, adstock_transformer]
        if self._context.model_spec.hill_before_adstock
        else [adstock_transformer, hill_transformer]
    )

    media_out = media
    for transformer in transformers_list:
      media_out = transformer.forward(media_out)
    return media_out

  def adstock_hill_rf(
      self,
      reach: backend.Tensor,
      frequency: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    """Transforms reach and frequency (RF) using Hill and Adstock functions.

    Args:
      reach: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for reach.
      frequency: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for frequency.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.
      decay_functions: String or sequence of strings denoting the adstock decay
        function(s) for each channel. Default: 'geometric'.
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `reach` equals
        `n_media_times`, in which case `n_times_output` defaults to `n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_rf_channels]`
      representing Hill and Adstock-transformed RF.
    """
    if n_times_output is None and (
        reach.shape[1] == self._context.n_media_times
    ):
      n_times_output = self._context.n_times
    elif n_times_output is None:
      raise ValueError(
          "n_times_output is required. This argument is only optional when "
          "`reach` has a number of time periods equal to `n_media_times`."
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
    rf_out = adstock_transformer.forward(reach * adj_frequency)

    return rf_out

  def compute_non_media_treatments_baseline(
      self,
      non_media_baseline_values: Sequence[str | float] | None = None,
  ) -> backend.Tensor:
    raise NotImplementedError

  def linear_predictor_counterfactual_difference_media(
      self,
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

    This function does the calculation efficiently by only calculating calling
    the hill/adstock function if the prior counterfactual is not all zeros.

    Args:
      media_transformed: The output of the hill/adstock function for actual
        historical media data.
      alpha_m: The adstock alpha parameter values.
      ec_m: The adstock ec parameter values.
      slope_m: The adstock hill slope parameter values.

    Returns:
      The linear predictor difference between the treatment variable and its
      counterfactual.
    """
    if self._context.media_tensors.prior_media_scaled_counterfactual is None:
      return media_transformed
    media_transformed_counterfactual = self.adstock_hill_media(
        self._context.media_tensors.prior_media_scaled_counterfactual,
        alpha_m,
        ec_m,
        slope_m,
        decay_functions=self._context.adstock_decay_spec.media,
    )
    # Absolute values is needed because the difference is negative for mROI
    # priors and positive for ROI and contribution priors.
    return backend.absolute(
        media_transformed - media_transformed_counterfactual
    )

  def linear_predictor_counterfactual_difference_rf(
      self,
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

    This function does the calculation efficiently by only calculating calling
    the hill/adstock function if the prior counterfactual is not all zeros.

    Args:
      rf_transformed: The output of the hill/adstock function for actual
        historical media data.
      alpha_rf: The adstock alpha parameter values.
      ec_rf: The adstock ec parameter values.
      slope_rf: The adstock hill slope parameter values.

    Returns:
      The linear predictor difference between the treatment variable and its
      counterfactual.
    """
    if self._context.rf_tensors.prior_reach_scaled_counterfactual is None:
      return rf_transformed
    rf_transformed_counterfactual = self.adstock_hill_rf(
        reach=self._context.rf_tensors.prior_reach_scaled_counterfactual,
        frequency=self._context.rf_tensors.frequency,
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
      dimension equal to the number of treatment channels..
    """
    if is_non_media:
      random_effects_normal = True
    else:
      random_effects_normal = (
          self._context.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
      )
    if self._context.revenue_per_kpi is None:
      revenue_per_kpi = backend.ones(
          [self._context.n_geos, self._context.n_times], dtype=backend.float32
      )
    else:
      revenue_per_kpi = self._context.revenue_per_kpi
    incremental_outcome_gx_over_beta_gx = backend.einsum(
        "...gtx,gt,g,->...gx",
        linear_predictor_counterfactual_difference,
        revenue_per_kpi,
        self._context.population,
        self._context.kpi_transformer.population_scaled_stdev,
    )
    if random_effects_normal:
      numerator_term_x = backend.einsum(
          "...gx,...gx,...x->...x",
          incremental_outcome_gx_over_beta_gx,
          beta_gx_dev,
          eta_x,
      )
      denominator_term_x = backend.einsum(
          "...gx->...x", incremental_outcome_gx_over_beta_gx
      )
      return (incremental_outcome_x - numerator_term_x) / denominator_term_x
    # For log-normal random effects, beta_x and eta_x are not mean & std.
    # The parameterization is beta_gx ~ exp(beta_x + eta_x * N(0, 1)).
    denominator_term_x = backend.einsum(
        "...gx,...gx->...x",
        incremental_outcome_gx_over_beta_gx,
        backend.exp(beta_gx_dev * eta_x[..., backend.newaxis, :]),
    )
    return backend.log(incremental_outcome_x) - backend.log(denominator_term_x)
