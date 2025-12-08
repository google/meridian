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
import numbers

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
    """Computes the baseline for each non-media treatment channel.

    Args:
      non_media_baseline_values: Optional list of shape
        `(n_non_media_channels,)`. Each element is either a float (which means
        that the fixed value will be used as baseline for the given channel) or
        one of the strings "min" or "max" (which mean that the global minimum or
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
      non_media_baseline_values = (
          self._context.model_spec.non_media_baseline_values
      )

    no_op_scaling_factor = backend.ones_like(self._context.population)[
        :, backend.newaxis, backend.newaxis
    ]
    if self._context.model_spec.non_media_population_scaling_id is not None:
      scaling_factors = backend.where(
          self._context.model_spec.non_media_population_scaling_id,
          self._context.population[:, backend.newaxis, backend.newaxis],
          no_op_scaling_factor,
      )
    else:
      scaling_factors = no_op_scaling_factor

    non_media_treatments_population_scaled = backend.divide_no_nan(
        self._context.non_media_treatments, scaling_factors
    )

    if non_media_baseline_values is None:
      # If non_media_baseline_values is not provided, use the minimum
      # value for each non_media treatment channel as the baseline.
      non_media_baseline_values_filled = [
          constants.NON_MEDIA_BASELINE_MIN
      ] * non_media_treatments_population_scaled.shape[-1]
    else:
      non_media_baseline_values_filled = non_media_baseline_values

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
            baseline_value, dtype=backend.float32
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
      media_transformed: backend.Tensor,
      alpha_m: backend.Tensor,
      ec_m: backend.Tensor,
      slope_m: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError

  def linear_predictor_counterfactual_difference_rf(
      self,
      rf_transformed: backend.Tensor,
      alpha_rf: backend.Tensor,
      ec_rf: backend.Tensor,
      slope_rf: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError

  def calculate_beta_x(
      self,
      is_non_media: bool,
      incremental_outcome_x: backend.Tensor,
      linear_predictor_counterfactual_difference: backend.Tensor,
      eta_x: backend.Tensor,
      beta_gx_dev: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError
