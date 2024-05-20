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

"""Methods to compute analysis metrics of the model and the data."""

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import warnings

from meridian import constants
from meridian.model import adstock_hill
from meridian.model import model
from meridian.model import transformers
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


__all__ = [
    "Analyzer",
]


def _calc_rsquared(expected, actual):
  """Calculates r-squared between actual and expected impact."""
  return 1 - np.nanmean((expected - actual) ** 2) / np.nanvar(actual)


def _calc_mape(expected, actual):
  """Calculates MAPE between actual and expected impact."""
  return np.nanmean(np.abs((actual - expected) / actual))


def _calc_weighted_mape(expected, actual):
  """Calculates wMAPE between actual and expected impact."""
  return np.nansum(np.abs(actual - expected)) / np.nansum(actual)


def _warn_if_geo_arg_in_kwargs(**kwargs):
  """Raise warning if a geo-level argument is used with national model."""
  for kwarg, value in kwargs.items():
    if (
        kwarg in constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS
        and value != constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]
    ):
      warnings.warn(
          f"The `{kwarg}` argument is ignored in the national model. It will be"
          " reset to"
          f" `{constants.NATIONAL_ANALYZER_PARAMETERS_DEFAULTS[kwarg]}`."
      )


def _check_shape_matches(
    t1: tf.Tensor | None = None,
    t1_name: str = "",
    t2: tf.Tensor | None = None,
    t2_name: str = "",
):
  """Raise an error if dimensions of 2 tensors don't match."""
  if t1 is not None and t2 is not None and t1.shape != t2.shape:
    raise ValueError(f"{t1_name}.shape must match {t2_name}.shape.")


def _check_spend_shape_matches(
    spend: tf.Tensor,
    spend_name: str,
    shapes: Sequence[tf.TensorShape],
):
  """Raises an error if dimensions of spend don't match expected shape."""
  if spend is not None and spend.shape not in shapes:
    raise ValueError(
        f"{spend_name}.shape: {spend.shape} must match either {shapes[0]} or"
        + f" {shapes[1]}."
    )


def _scale_tensors_by_multiplier(
    media: tf.Tensor | None,
    reach: tf.Tensor | None,
    frequency: tf.Tensor | None,
    multiplier: float,
    by_reach: bool,
) -> Mapping[str, tf.Tensor | None]:
  """Get scaled tensors for incremental impact calculation.

  Args:
    media: Optional tensor with dimensions matching media.
    reach: Optional tensor with dimensions matching reach.
    frequency: Optional tensor with dimensions matching frequency.
    multiplier: Float indicating the factor to scale tensors by.
    by_reach: Boolean indicating whether to scale reach or frequency when rf
      data is available.

  Returns:
    Dictionary containing scaled tensor parameters.
  """
  scaled_tensors = {}
  if media is not None:
    scaled_tensors["new_media"] = media * multiplier
  if reach is not None and frequency is not None:
    if by_reach:
      scaled_tensors["new_frequency"] = frequency
      scaled_tensors["new_reach"] = reach * multiplier
    else:
      scaled_tensors["new_frequency"] = frequency * multiplier
      scaled_tensors["new_reach"] = reach
  return scaled_tensors


def _mean_and_ci(
    prior: tf.Tensor,
    posterior: tf.Tensor,
    metric_name: str,
    xr_dims: Sequence[str],
    xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    confidence_level: float,
) -> xr.Dataset:
  """Computes mean and confidence intervals for the given metric.

  Args:
    prior: A tensor with the prior data for the metric.
    posterior: A tensor with the posterior data for the metric.
    metric_name: The name of the input metric for the compuations.
    xr_dims: A list of dimensions for the output dataset.
    xr_coords: A dictionary with the coordinates for the output dataset.
    confidence_level: The threshold for computing the confidence intervals.

  Returns:
    An xarray Dataset containing mean and confidence intervals for prior and
    posterior data for the metric.
  """
  prior_mean = np.mean(prior, (0, 1))
  prior_ci_lo = np.quantile(prior, (1 - confidence_level) / 2, (0, 1))
  prior_ci_hi = np.quantile(prior, (1 + confidence_level) / 2, (0, 1))
  posterior_mean = np.mean(posterior, (0, 1))
  posterior_ci_lo = np.quantile(posterior, (1 - confidence_level) / 2, (0, 1))
  posterior_ci_hi = np.quantile(posterior, (1 + confidence_level) / 2, (0, 1))
  prior_metrics = np.stack([prior_mean, prior_ci_lo, prior_ci_hi], axis=-1)
  posterior_metrics = np.stack(
      [posterior_mean, posterior_ci_lo, posterior_ci_hi], axis=-1
  )
  metrics = np.stack([prior_metrics, posterior_metrics], axis=-1)
  xr_data = {metric_name: (xr_dims, metrics)}
  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


class Analyzer:
  """Runs calculations to analyze the raw data after fitting the model."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian

  @tf.function(jit_compile=True)
  def _get_kpi_means(
      self,
      tau_t: tf.Tensor,
      tau_g: tf.Tensor,
      gamma_gc: tf.Tensor | None,
      controls_scaled: tf.Tensor,
      media_scaled: tf.Tensor | None,
      reach_scaled: tf.Tensor | None,
      frequency: tf.Tensor | None,
      alpha_m: tf.Tensor | None = None,
      alpha_rf: tf.Tensor | None = None,
      ec_m: tf.Tensor | None = None,
      ec_rf: tf.Tensor | None = None,
      slope_m: tf.Tensor | None = None,
      slope_rf: tf.Tensor | None = None,
      beta_gm: tf.Tensor | None = None,
      beta_grf: tf.Tensor | None = None,
  ) -> tf.Tensor:
    """Computes batched KPI means on the unit scale.

    Args:
      tau_t: tau_t distribution from inference data.
      tau_g: tau_g distribution from inference data.
      gamma_gc: gamma_gc distribution from inference data.
      controls_scaled: ControlTransformer scaled controls tensor.
      media_scaled: MediaTransformer scaled media tensor.
      reach_scaled: MediaTransformer scaled reach tensor.
      frequency: Non scaled frequency tensor.
      alpha_m: Optional parameter for adstock calculations. Used in conjunction
        with `media`.
      alpha_rf: Optional parameter for adstock calculations. Used in conjunction
        with `reach` and `frequency`.
      ec_m: Optional parameter for hill calculations. Used in conjunction with
        `media`.
      ec_rf: Optional parameter for hill calculations. Used in conjunction with
        `reach` and `frequency`.
      slope_m: Optional parameter for hill calculations. Used in conjunction
        with `media`.
      slope_rf: Optional parameter for hill calculations. Used in conjunction
        with `reach` and `frequency`.
      beta_gm: Optional parameter from inference data. Used in conjunction with
        `media`.
      beta_grf: Optional parameter from inference data. Used in conjunction with
        `reach` and `frequency`.

    Returns:
      Tensor representing adstock/hill-transformed media.
    """
    tau_gt = tf.expand_dims(tau_g, -1) + tf.expand_dims(tau_t, -2)
    combined_media_transformed, combined_beta = (
        self._get_transformed_media_and_beta(
            media=media_scaled,
            reach=reach_scaled,
            frequency=frequency,
            alpha_m=alpha_m,
            alpha_rf=alpha_rf,
            ec_m=ec_m,
            ec_rf=ec_rf,
            slope_m=slope_m,
            slope_rf=slope_rf,
            beta_gm=beta_gm,
            beta_grf=beta_grf,
        )
    )

    return (
        tau_gt
        + tf.einsum(
            "...gtm,...gm->...gt", combined_media_transformed, combined_beta
        )
        + tf.einsum("...gtc,...gc->...gt", controls_scaled, gamma_gc)
    )

  def _check_revenue_data_exists(self, use_kpi: bool = False) -> None:
    """Raise an error if `use_kpi` is False but revenue data does not exist."""
    if not use_kpi and self._meridian.revenue_per_kpi is None:
      raise ValueError(
          "`use_kpi` must be True when `revenue_per_kpi` is not defined."
      )

  def _validate_roi_functionality(self) -> None:
    """Validates whether ROI metrics can be computed."""
    if self._meridian.revenue_per_kpi is None:
      raise ValueError(
          "ROI-related metrics can't be computed when `revenue_per_kpi` is not"
          " defined."
      )

  def _get_adstock_dataframe(
      self,
      channel_type: str,
      l_range: np.ndarray,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float = 0.9,
  ) -> pd.DataFrame:
    """Computes decayed effect means and CIs for media or RF channels.

    Args:
      channel_type: Specifies `media` or `reach` for computing prior and
        posterior decayed effects.
      l_range: The range of time across which the adstock effect is computed.
      xr_dims: A list of dimensions for the output dataset.
      xr_coords: A dictionary with the coordinates for the output dataset.
      confidence_level: The threshold for computing the confidence intervals.

    Returns:
      Pandas DataFrame containing the channel, time_units, distribution, ci_hi,
      ci_lo, and mean decayed effects for either media or RF channel types.
    """
    if channel_type is constants.MEDIA:
      prior = self._meridian.inference_data.prior.alpha_m.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_m.values,
          (-1, self._meridian.n_media_channels),
      )
    else:
      prior = self._meridian.inference_data.prior.alpha_rf.values[0]
      posterior = np.reshape(
          self._meridian.inference_data.posterior.alpha_rf.values,
          (-1, self._meridian.n_rf_channels),
      )

    decayed_effect_prior = (
        prior[np.newaxis, ...] ** l_range[:, np.newaxis, np.newaxis, np.newaxis]
    )
    decayed_effect_posterior = (
        posterior[np.newaxis, ...]
        ** l_range[:, np.newaxis, np.newaxis, np.newaxis]
    )

    decayed_effect_prior_transpose = tf.transpose(
        decayed_effect_prior, perm=[1, 2, 0, 3]
    )
    decayed_effect_posterior_transpose = tf.transpose(
        decayed_effect_posterior, perm=[1, 2, 0, 3]
    )
    adstock_dataset = _mean_and_ci(
        decayed_effect_prior_transpose,
        decayed_effect_posterior_transpose,
        constants.EFFECT,
        xr_dims,
        xr_coords,
        confidence_level,
    )
    return (
        adstock_dataset[constants.EFFECT]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.TIME_UNITS,
                constants.DISTRIBUTION,
            ],
            columns=constants.METRIC,
            values=constants.EFFECT,
        )
        .reset_index()
    )

  def _get_adstock_hill_tensors(
      self,
      new_media: tf.Tensor | None,
      new_reach: tf.Tensor | None,
      new_frequency: tf.Tensor | None,
  ) -> dict[str, tf.Tensor | None]:
    """Get adstock_hill parameter tensors based on data availability.

    Args:
      new_media: Optional tensor with dimensions matching media.
      new_reach: Optional tensor with dimensions matching reach.
      new_frequency: Optional tensor with dimensions matching frequency.

    Returns:
      dictionary containing optional media, reach, and frequency data tensors.
    """
    adstock_tensors = {}
    adstock_tensors["media_scaled"] = (
        self._meridian.media_scaled
        if new_media is None or self._meridian.media_transformer is None
        else self._meridian.media_transformer.forward(new_media)
    )
    adstock_tensors["reach_scaled"] = (
        self._meridian.reach_scaled
        if new_reach is None or self._meridian.reach_transformer is None
        else self._meridian.reach_transformer.forward(new_reach)
    )
    adstock_tensors["frequency"] = (
        new_frequency if new_frequency is not None else self._meridian.frequency
    )
    return adstock_tensors

  def _get_adstock_hill_param_names(self) -> list[str]:
    """Gets adstock_hill distributions.

    Returns:
      A list containing available media and rf parameters names in inference
      data.
    """
    params = []
    if self._meridian.media is not None:
      params.extend([
          constants.EC_M,
          constants.SLOPE_M,
          constants.ALPHA_M,
          constants.BETA_GM,
      ])
    if self._meridian.reach is not None:
      params.extend([
          constants.EC_RF,
          constants.SLOPE_RF,
          constants.ALPHA_RF,
          constants.BETA_GRF,
      ])
    return params

  def _get_transformed_media_and_beta(
      self,
      media: tf.Tensor | None = None,
      reach: tf.Tensor | None = None,
      frequency: tf.Tensor | None = None,
      alpha_m: tf.Tensor | None = None,
      alpha_rf: tf.Tensor | None = None,
      ec_m: tf.Tensor | None = None,
      ec_rf: tf.Tensor | None = None,
      slope_m: tf.Tensor | None = None,
      slope_rf: tf.Tensor | None = None,
      beta_gm: tf.Tensor | None = None,
      beta_grf: tf.Tensor | None = None,
  ) -> tuple[tf.Tensor | None, tf.Tensor | None]:
    """Function for transforming media using adstock and hill functions.

    This transforms the media tensor using the adstock and hill functions, in
    the desired order.

    Args:
      media: Optional media tensor.
      reach: Optional reach tensor.
      frequency: Optional frequency tensor.
      alpha_m: Optional parameter for adstock calculations. Used in conjunction
        with `media`.
      alpha_rf: Optional parameter for adstock calculations. Used in conjunction
        with `reach` and `frequency`.
      ec_m: Optional parameter for hill calculations. Used in conjunction with
        `media`.
      ec_rf: Optional parameter for hill calculations. Used in conjunction with
        `reach` and `frequency`.
      slope_m: Optional parameter for hill calculations. Used in conjunction
        with `media`.
      slope_rf: Optional parameter for hill calculations. Used in conjunction
        with `reach` and `frequency`.
      beta_gm: Optional parameter from inference data. Used in conjunction with
        `media`.
      beta_grf: Optional parameter from inference data. Used in conjunction with
        `reach` and `frequency`.

    Returns:
      A tuple `(combined_media_transformed, combined_beta)`.
    """
    if media is not None:
      media_transformed = self._meridian.adstock_hill_media(
          media=media,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
      )
    else:
      media_transformed = None
    if reach is not None:
      rf_transformed = self._meridian.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
      )
    else:
      rf_transformed = None

    if media_transformed is not None and rf_transformed is not None:
      combined_media_transformed = tf.concat(
          [media_transformed, rf_transformed], axis=-1
      )
      combined_beta = tf.concat([beta_gm, beta_grf], axis=-1)
    elif media_transformed is not None:
      combined_media_transformed = media_transformed
      combined_beta = beta_gm
    else:
      combined_media_transformed = rf_transformed
      combined_beta = beta_grf
    return combined_media_transformed, combined_beta

  def filter_and_aggregate_geos_and_times(
      self,
      tensor: tf.Tensor,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
  ) -> tf.Tensor:
    """Filters and/or aggregates geo and time dimensions of a tensor.

    Args:
      tensor: Tensor with dimensions `[..., n_geos, n_times]` or `[..., n_geos,
        n_times, n_channels]`.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included. The selected geos should match those in
        `InputData.geo`.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included. The selected times should match
        those in `InputData.time`.
      aggregate_geos: Boolean. If `True`, the tensor is summed over all geos.
      aggregate_times: Boolean. If `True`, the tensor is summed over all time
        periods.

    Returns:
      A tensor with filtered and/or aggregated geo and time dimensions.
    """
    mmm = self._meridian
    if tensor.shape[-3:] in (
        tf.TensorShape([mmm.n_geos, mmm.n_times, mmm.n_media_channels]),
        tf.TensorShape([mmm.n_geos, mmm.n_times, mmm.n_rf_channels]),
        tf.TensorShape(
            [mmm.n_geos, mmm.n_times, mmm.n_media_channels + mmm.n_rf_channels]
        ),
    ):
      has_media_dim = True
    elif tensor.shape[-2:] == tf.TensorShape([mmm.n_geos, mmm.n_times]):
      has_media_dim = False
    else:
      raise ValueError(
          "The tensor must have shape [..., n_geos, n_times, n_channels] or"
          " [..., n_geos, n_times]."
      )

    if selected_geos and any(
        geo not in mmm.input_data.geo for geo in selected_geos
    ):
      raise ValueError(
          "`selected_geos` must match the geo dimension names from "
          "meridian.InputData."
      )

    if selected_times and any(
        time not in mmm.input_data.time for time in selected_times
    ):
      raise ValueError(
          "`selected_times` must match the time dimension names from "
          "meridian.InputData."
      )

    if selected_geos:
      geo_mask = [x in selected_geos for x in mmm.input_data.geo]
      tensor = tf.boolean_mask(
          tensor, geo_mask, axis=tensor.ndim - 2 - has_media_dim
      )
    if selected_times:
      time_mask = [x in selected_times for x in mmm.input_data.time]
      tensor = tf.boolean_mask(
          tensor, time_mask, axis=tensor.ndim - 1 - has_media_dim
      )
    tensor_dims = "...gt" + "m" * has_media_dim
    output_dims = (
        "g" * (not aggregate_geos)
        + "t" * (not aggregate_times)
        + "m" * has_media_dim
    )
    return tf.einsum(f"{tensor_dims}->...{output_dims}", tensor)

  def expected_impact(
      self,
      use_posterior: bool = True,
      new_media: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      new_controls: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_impact: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> tf.Tensor:
    """Calculates either the expected impact posterior or prior.

    This calculates `E(Impact|Media, Controls)` for each posterior (or prior)
    parameter draw, where `Impact` refers to either `revenue` if
    `use_kpi=False`, or `kpi` if `use_kpi=True`. When `revenue_per_kpi` is not
    defined, `use_kpi` cannot be `False`.

    By default, this calculates expected impact conditional on the media and
    control values that the Meridian object was initialized with. The user can
    also pass other media values as long as the dimensions match, and similarly
    for controls. In principle, the expected impact could be calculated with
    other time dimensions (for example, future predictions), but this is not
    allowed with this method because of the additional complexites this
    introduces:

    1.  Corresponding price (revenue per kpi) data would also be needed.
    2.  If the model contains weekly effect parameters, then some method is
        needed to estimate or predict these effects for time periods outside of
        the training data window.

    Args:
      use_posterior: Boolean. If `True`, then the expected impact posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      new_media: Optional tensor with dimensions matching media.
      new_reach: Optional tensor with dimensions matching reach.
      new_frequency: Optional tensor with dimensions matching frequency.
      new_controls: Optional tensor with dimensions matching controls.
      selected_geos: Optional list of containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list of containing a subset of dates to include.
        By default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected impact is summed over all
        regions.
      aggregate_times: Boolean. If `True`, the expected impact is summed over
        all time periods.
      inverse_transform_impact: Boolean. If `True`, returns the expected impact
        in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        impact after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: Boolean. If `True`, the expected KPI is calculated. If `False`,
        the expected revenue `(kpi * revenue_per_kpi)` is calculated. Only used
        if `inverse_transform_impact=True`. `use_kpi` must be `True` when
        `revenue_per_kpi` is not defined.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of expected impact (either KPI or revenue, depending on the
      `use_kpi` argument) with dimensions `(n_chains, n_draws, n_geos,
      n_times)`. The `n_geos` and `n_times` dimensions is dropped if
      `aggregate_geos=True` or `aggregate_time=True`, respectively.
    Raises:
      NotFittedModelError: if `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
    """
    self._check_revenue_data_exists(use_kpi)
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling"
          " `expected_impact()`."
      )
    _check_shape_matches(
        new_controls, "new_controls", self._meridian.controls, "controls"
    )
    _check_shape_matches(new_media, "new_media", self._meridian.media, "media")
    _check_shape_matches(new_reach, "new_reach", self._meridian.reach, "reach")
    _check_shape_matches(
        new_frequency, "new_frequency", self._meridian.frequency, "frequency"
    )

    params = (
        self._meridian.inference_data.posterior
        if use_posterior
        else self._meridian.inference_data.prior
    )
    tensor_kwargs = self._get_adstock_hill_tensors(
        new_media, new_reach, new_frequency
    )
    tensor_kwargs["controls_scaled"] = (
        self._meridian.controls_scaled
        if new_controls is None
        else self._meridian.controls_transformer.forward(new_controls)
    )
    n_draws = params.draw.size
    n_chains = params.chain.size
    impact_means = tf.zeros(
        (n_chains, 0, self._meridian.n_geos, self._meridian.n_times)
    )
    batch_starting_indices = np.arange(n_draws, step=batch_size)
    param_list = [
        constants.TAU_T,
        constants.TAU_G,
        constants.GAMMA_GC,
    ] + self._get_adstock_hill_param_names()
    impact_means_temps = []
    for start_index in batch_starting_indices:
      stop_index = np.min([n_draws, start_index + batch_size])
      batch_dists = {
          k: tf.convert_to_tensor(params[k][:, start_index:stop_index, ...])
          for k in param_list
      }
      impact_means_temps.append(
          self._get_kpi_means(
              **tensor_kwargs,
              **batch_dists,
          )
      )
    impact_means = tf.concat([impact_means, *impact_means_temps], axis=1)
    if inverse_transform_impact:
      impact_means = self._meridian.kpi_transformer.inverse(impact_means)
      if not use_kpi:
        impact_means *= self._meridian.revenue_per_kpi

    return self.filter_and_aggregate_geos_and_times(
        impact_means,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
    )

  def _get_modeled_incremental_kpi(
      self,
      media_scaled: tf.Tensor | None,
      reach_scaled: tf.Tensor | None,
      frequency: tf.Tensor | None,
      alpha_m: tf.Tensor | None = None,
      alpha_rf: tf.Tensor | None = None,
      ec_m: tf.Tensor | None = None,
      ec_rf: tf.Tensor | None = None,
      slope_m: tf.Tensor | None = None,
      slope_rf: tf.Tensor | None = None,
      beta_gm: tf.Tensor | None = None,
      beta_grf: tf.Tensor | None = None,
  ) -> tf.Tensor:
    """Function to compute modeled incremental KPI on the unit scale.

    Args:
      media_scaled: Optional scaled media tensor.
      reach_scaled: Optional scaled reach tensor.
      frequency: Optional non scaled frequency tensor.
      alpha_m: Optional parameter for adstock calculations. Used in conjunction
        with `media`.
      alpha_rf: Optional parameter for adstock calculations. Used in conjunction
        with `reach` and `frequency`.
      ec_m: Optional parameter for hill calculations. Used in conjunction with
        `media`.
      ec_rf: Optional parameter for hill calculations. Used in conjunction with
        `reach` and `frequency`.
      slope_m: Optional parameter for hill calculations. Used in conjunction
        with `media`.
      slope_rf: Optional parameter for hill calculations. Used in conjunction
        with `reach` and `frequency`.
      beta_gm: Optional parameter from inference data. Used in conjunction with
        `media`.
      beta_grf: Optional parameter from inference data. Used in conjunction with
        `reach` and `frequency`.

    Returns:
      Tensor of incremental impact modeled from parameter distributions.
    """
    combined_media_transformed, combined_beta = (
        self._get_transformed_media_and_beta(
            media=media_scaled,
            reach=reach_scaled,
            frequency=frequency,
            alpha_m=alpha_m,
            alpha_rf=alpha_rf,
            ec_m=ec_m,
            ec_rf=ec_rf,
            slope_m=slope_m,
            slope_rf=slope_rf,
            beta_gm=beta_gm,
            beta_grf=beta_grf,
        )
    )
    return tf.einsum(
        "...gtm,...gm->...gtm",
        combined_media_transformed,
        combined_beta,
    )

  def _inverse_impact(
      self, modeled_incremental_impact: tf.Tensor, use_kpi: bool
  ) -> tf.Tensor:
    """Function to inverse incremental impact (revenue or KPI).

    This method assumes that additive changes on the model kpi scale
    correspond to additive changes on the original kpi scale. In other
    words, the intercept and control effects do not influence the media effects.

    Args:
      modeled_incremental_impact: Tensor of incremenal impact modeled from
        parameter distributions.
      use_kpi: Boolean. If True, the incremental KPI is calculated. If False,
        incremental revenue `(KPI * revenue_per_kpi)` is calculated. Only used
        if `inverse_transform_impact=True`. `use_kpi` must be True when
        `revenue_per_kpi` is not defined.

    Returns:
       Tensor of incremental impact returned in terms of revenue or KPI.
    """
    self._check_revenue_data_exists(use_kpi)
    t1 = self._meridian.kpi_transformer.inverse(
        tf.einsum("...m->m...", modeled_incremental_impact)
    )
    t2 = self._meridian.kpi_transformer.inverse(tf.zeros_like(t1))
    kpi = tf.einsum("m...->...m", t1 - t2)

    if use_kpi:
      return kpi
    return tf.einsum(
        "gt,...gtm->...gtm",
        self._meridian.revenue_per_kpi,
        kpi,
    )

  @tf.function(jit_compile=True)
  def _incremental_impact_impl(
      self,
      media_scaled: tf.Tensor | None,
      reach_scaled: tf.Tensor | None,
      frequency: tf.Tensor | None,
      alpha_m: tf.Tensor | None = None,
      alpha_rf: tf.Tensor | None = None,
      ec_m: tf.Tensor | None = None,
      ec_rf: tf.Tensor | None = None,
      slope_m: tf.Tensor | None = None,
      slope_rf: tf.Tensor | None = None,
      beta_gm: tf.Tensor | None = None,
      beta_grf: tf.Tensor | None = None,
      inverse_transform_impact: bool | None = None,
      use_kpi: bool | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
  ) -> tf.Tensor:
    """Computes incremental impact (revenue or KPI) on a batch of data.

    Args:
      media_scaled: `media` data scaled by the per-geo median, normalized by the
        geo population. Shape (n_geos x n_media_times x n_media_channels).
      reach_scaled: `reach` data scaled by the per-geo median, normalized by the
        geo population. Shape (n_geos x n_media_times x n_rf_channels).
      frequency: Contains frequency data with shape(n_geos x n_media_times x
        n_rf_channels).
      alpha_m: media_channel specific alpha parameter for adstock calculations.
        Used in conjunction with `media`.
      alpha_rf: rf_channel specific alpha parameter for adstock calculations.
        Used in conjunction with `reach` and `frequency`.
      ec_m: media_channel specific ec parameter for hill calculations. Used in
        conjunction with `media`.
      ec_rf: rf_channel specific ec parameter for hill calculations. Used in
        conjunction with `reach` and `frequency`.
      slope_m: media_channel specific slope parameter for hill calculations.
        Used in conjunction with `media`.
      slope_rf: rf_channel specific slope parameter for hill calculations. Used
        in conjunction with `reach` and `frequency`.
      beta_gm: media_channel specific parameter from inference data. Used in
        conjunction with `media`.
      beta_grf: rf_channel specific beta_g parameter from inference data. Used
        in conjunction with `reach` and `frequency`.
      inverse_transform_impact: Boolean. If `True`, returns the expected impact
        in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        impact after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: If True, the incremental KPI is calculated. If False, incremental
        revenue `(KPI * revenue_per_kpi)` is calculated. Only used if
        `inverse_transform_impact=True`. `use_kpi` must be True when
        `revenue_per_kpi` is not defined.
      selected_geos: Contains a subset of geos to include. By default, all geos
        are included.
      selected_times: Contains a subset of dates to include. By default, all
        time periods are included.
      aggregate_geos: If True, then incremental impact is summed over all
        regions.
      aggregate_times: If True, then incremental impact is summed over all time
        periods.

    Returns:
      Tensor of incremental impact modeled from parameter distributions.
    """
    self._check_revenue_data_exists(use_kpi)
    transformed_impact = self._get_modeled_incremental_kpi(
        media_scaled=media_scaled,
        reach_scaled=reach_scaled,
        frequency=frequency,
        alpha_m=alpha_m,
        alpha_rf=alpha_rf,
        ec_m=ec_m,
        ec_rf=ec_rf,
        slope_m=slope_m,
        slope_rf=slope_rf,
        beta_gm=beta_gm,
        beta_grf=beta_grf,
    )
    incremental_impact = (
        self._inverse_impact(transformed_impact, use_kpi=use_kpi)
        if inverse_transform_impact
        else transformed_impact
    )
    return self.filter_and_aggregate_geos_and_times(
        tensor=incremental_impact,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
    )

  def incremental_impact(
      self,
      use_posterior: bool = True,
      new_media: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      inverse_transform_impact: bool = True,
      use_kpi: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> tf.Tensor:
    """Calculates either the incremental impact posterior or prior.

    This calculates the incremental impact for each posterior or prior parameter
    draw. Incremental impact is defined as `E(Impact|Media, Controls)` minus
    `E(Impact|Media_0, Controls)`, where `Media_0` means that media execution
    for a given channel is set to zero and all other media are set to the values
    that the Meridian object was initialized with. This is the case for all geos
    and time periods, including lag periods. Impact refers to either
    `revenue` if `use_kpi=False`, or `kpi` if `use_kpi=True`. When
    `revenue_per_kpi` is not defined, `use_kpi` cannot be False.

    By default, this calculates incremental impact conditional on the media
    and control values that the Meridian object was initialized with.

    The calculation in this method depends on two key assumptions made in the
    current Meridian implementation, that could potentially be dropped in the
    future:

    1.  additivity of media effects (no interactions).
    2.  additive changes on the model KPI scale correspond to additive
        changes on the original KPI scale. In other words, the intercept and
        control effects do not influence the media effects. This assumption
        currently holds because the impact transformation only involves
        centering and scaling, for example, no log transformations.

    In principle, the incremental impact can be calculated
    with other time dimensions (such as future predictions), but this is not
    allowed with this method because of the additional complexites
    this introduces:

    1.  corresponding price (revenue per KPI) data is also needed
    2.  if the model contains weekly effect parameters, then some method is
        needed to estimate or predict these effects for time periods outside of
        the training data window.

    Args:
      use_posterior: Boolean. If `True`, then the incremental impact posterior
        distribution is calculated. Otherwise, the prior distribution is
        calculated.
      new_media: Optional tensor with dimensions matching media.
      new_reach: Optional tensor with dimensions matching reach.
      new_frequency: Optional tensor with dimensions matching frequency.
      selected_geos: Optional list of containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list of containing a subset of dates to include.
        By default, all time periods are included.
      aggregate_geos: Boolean. If `True`, then incremental impact is summed over
        all regions.
      aggregate_times: Boolean. If `True`, then incremental impact is summed
        over all time periods.
      inverse_transform_impact: Boolean. If `True`, returns the expected impact
        in the original KPI or revenue (depending on what is passed to
        `use_kpi`), as it was passed to `InputData`. If False, returns the
        impact after transformation by `KpiTransformer`, reflecting how its
        represented within the model.
      use_kpi: Boolean. If `True`, the incremental KPI is calculated. If
        `False`, incremental revenue (`KPI * revenue_per_kpi`) is calculated.
        Only used if `inverse_transform_impact=True`. `use_kpi` must be `True`
        when `revenue_per_kpi` is not defined.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of incremental impact (either KPI or revenue, depending on
      `use_kpi` argument) with dimensions `(n_chains, n_draws, n_geos,
      n_times, n_channels)` where `n_channels` is the total number of media and
      RF channels. The `n_geos` and `n_times` dimensions are dropped if
      `aggregate_geos=True` or `aggregate_times=True`, respectively.
    Raises:
      NotFittedModelError: If `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
      ValueError: If `new_media` arguments does not have the same tensor shape
        as media.
    """
    self._check_revenue_data_exists(use_kpi)
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR

    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )

    _check_shape_matches(new_media, "new_media", self._meridian.media, "media")
    _check_shape_matches(new_reach, "new_reach", self._meridian.reach, "reach")
    _check_shape_matches(
        new_frequency, "new_frequency", self._meridian.frequency, "frequency"
    )

    params = (
        self._meridian.inference_data.posterior
        if use_posterior
        else self._meridian.inference_data.prior
    )

    tensor_kwargs = self._get_adstock_hill_tensors(
        new_media, new_reach, new_frequency
    )
    n_draws = params.draw.size
    batch_starting_indices = np.arange(n_draws, step=batch_size)
    param_list = self._get_adstock_hill_param_names()
    incremental_impact_temps = [None] * len(batch_starting_indices)
    for i, start_index in enumerate(batch_starting_indices):
      stop_index = np.min([n_draws, start_index + batch_size])
      batch_dists = {
          k: tf.convert_to_tensor(params[k][:, start_index:stop_index, ...])
          for k in param_list
      }
      incremental_impact_temps[i] = self._incremental_impact_impl(
          **tensor_kwargs,
          **batch_dists,
          selected_geos=selected_geos,
          selected_times=selected_times,
          aggregate_geos=aggregate_geos,
          aggregate_times=aggregate_times,
          inverse_transform_impact=inverse_transform_impact,
          use_kpi=use_kpi,
      )
    return tf.concat(incremental_impact_temps, axis=1)

  @dataclasses.dataclass(frozen=True)
  class PerformanceData:
    """Dataclass for data required in profitability calculations."""

    media: tf.Tensor | None
    media_spend: tf.Tensor | None
    reach: tf.Tensor | None
    frequency: tf.Tensor | None
    rf_spend: tf.Tensor | None

    def total_spend(self) -> tf.Tensor | None:
      if self.media_spend is not None and self.rf_spend is not None:
        total_spend = tf.concat([self.media_spend, self.rf_spend], axis=-1)
      elif self.media_spend is not None:
        total_spend = self.media_spend
      else:
        total_spend = self.rf_spend
      return total_spend

  def _get_performance_tensors(
      self,
      new_media: tf.Tensor | None = None,
      new_media_spend: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      new_rf_spend: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
  ) -> PerformanceData:
    """Get tensors required for profitability calculations (ROI, mROI, CPIK).

    Verify dimensionality requirements and return a dictionary with data tensors
    required for profitability calculations.

    Args:
      new_media: Optional. Media data, with the same shape as
        `meridian.input_data.media`, to be used to compute ROI for alternative
        media data. Default uses `meridian.input_data.media`.
      new_media_spend: Optional. Media spend data, with the same shape as
        `meridian.input_data.media_spend`, to be used to compute ROI for
        alternative `media_spend` data. Default uses
        `meridian.input_data.media_spend`.
      new_reach: Optional. Reach data with the same shape as
        `meridian.input_data.reach`, to be used to compute ROI for alternative
        reach data. Default uses `meridian.input_data.reach`.
      new_frequency: Optional. Frequency data with the same shape as
        `meridian.input_data.frequency`, to be used to compute ROI for
        alternative frequency data. Defaults to `meridian.input_data.frequency`.
      new_rf_spend: Optional. RF Spend data with the same shape as
        `meridian.input_data.rf_spend`, to be used to compute ROI for
        alternative `rf_spend` data. Defaults to `meridian.input_data.rf_spend`.
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional. Contains a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: If `True`, then expected revenue is summed over all
        regions.
      aggregate_times: If `True`, then expected revenue is summed over all time
        periods.

    Returns:
      PerformanceData object containing the media, rf, and spend data for
        profitability calculations.
    """

    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          aggregate_geos=aggregate_geos,
          selected_geos=selected_geos,
      )
    if selected_geos is not None or not aggregate_geos:
      if (
          self._meridian.media_spend is not None
          and not self._meridian.input_data.media_spend_has_geo_dimension
      ):
        raise ValueError(
            "aggregate_geos=False not allowed because Meridian media_spend data"
            " does not have geo dimension."
        )
      if (
          self._meridian.rf_spend is not None
          and not self._meridian.input_data.rf_spend_has_geo_dimension
      ):
        raise ValueError(
            "aggregate_geos=False not allowed because Meridian rf_spend data"
            " does not have geo dimension."
        )

    if selected_times is not None or not aggregate_times:
      if (
          self._meridian.media_spend is not None
          and not self._meridian.input_data.media_spend_has_time_dimension
      ):
        raise ValueError(
            "aggregate_times=False not allowed because Meridian media_spend"
            " data does not have time dimension."
        )
      if (
          self._meridian.rf_spend is not None
          and not self._meridian.input_data.rf_spend_has_time_dimension
      ):
        raise ValueError(
            "aggregate_geos=False not allowed because Meridian rf_spend data"
            " does not have time dimension."
        )

    _check_shape_matches(
        new_media, constants.NEW_MEDIA, self._meridian.media, constants.MEDIA
    )
    _check_spend_shape_matches(
        new_media_spend,
        constants.NEW_MEDIA_SPEND,
        (
            tf.TensorShape((self._meridian.n_media_channels)),
            tf.TensorShape((
                self._meridian.n_geos,
                self._meridian.n_times,
                self._meridian.n_media_channels,
            )),
        ),
    )
    _check_shape_matches(
        new_reach, constants.NEW_REACH, self._meridian.reach, constants.REACH
    )
    _check_shape_matches(
        new_frequency,
        constants.NEW_FREQUENCY,
        self._meridian.frequency,
        constants.FREQUENCY,
    )
    _check_spend_shape_matches(
        new_rf_spend,
        constants.NEW_RF_SPEND,
        (
            tf.TensorShape((self._meridian.n_rf_channels)),
            tf.TensorShape((
                self._meridian.n_geos,
                self._meridian.n_times,
                self._meridian.n_rf_channels,
            )),
        ),
    )

    media = self._meridian.media if new_media is None else new_media
    reach = self._meridian.reach if new_reach is None else new_reach
    frequency = (
        self._meridian.frequency if new_frequency is None else new_frequency
    )

    media_spend = (
        self._meridian.media_spend
        if new_media_spend is None
        else new_media_spend
    )
    rf_spend = self._meridian.rf_spend if new_rf_spend is None else new_rf_spend

    return self.PerformanceData(
        media=media,
        media_spend=media_spend,
        reach=reach,
        frequency=frequency,
        rf_spend=rf_spend,
    )

  def marginal_roi(
      self,
      incremental_increase: float = 0.01,
      use_posterior: bool = True,
      new_media: tf.Tensor | None = None,
      new_media_spend: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      new_rf_spend: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      by_reach: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> tf.Tensor | None:
    """Calculates the marginal ROI prior or posterior distribution.

    The marginal ROI (mROI) numerator is the change in expected revenue (`kpi *
    revenue_per_kpi`) when one channel's spend is increased by a small fraction.
    The mROI denominator is the corresponding small fraction of the
    channel's total spend.

    Args:
      incremental_increase: Small fraction by which each channel's spend is
        increased when calculating its mROI numerator. The mROI denominator is
        this fraction of the channel's total spend. Only used if marginal is
        `True`.
      use_posterior: If `True` then the posterior distribution is calculated.
        Otherwise, the prior distribution is calculated.
      new_media: Optional. Media data with the same shape as
        `meridian.input_data.media`. Used to compute ROI for alternative media
        data. Default uses `meridian.input_data.media`.
      new_media_spend: Optional. Media spend data with the same shape as
        `meridian.input_data.spend`. Used to compute ROI for alternative
        `media_spend` data. Default uses `meridian.input_data.media_spend`.
      new_reach: Optional. Reach data with the same shape as
        `meridian.input_data.reach`. Used to compute ROI for alternative reach
        data. Default uses `meridian.input_data.reach`.
      new_frequency: Optional. Frequency data with the same shape as
        `meridian.input_data.frequency`. Used to compute ROI for alternative
        frequency data. Default uses `meridian.input_data.frequency`.
      new_rf_spend: Optional. RF Spend data with the same shape as
        `meridian.input_data.rf_spend`. Used to compute ROI for alternative
        `rf_spend` data. Default uses `meridian.input_data.rf_spend`.
      selected_geos: Optional. Contains a subset of geos to include. By default,
        all geos are included.
      selected_times: Optional. Contains a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: If `True`, the expected revenue is summed over all of the
        regions.
      aggregate_times: If `True`, the expected revenue is summed over all of
        time periods.
      by_reach: Used for a channel with reach and frequency. If `True`, returns
        the mROI by reach for a given fixed frequency. If `False`, returns the
        mROI by frequency for a given fixed reach.
      batch_size: Maximum draws per chain in each batch. The calculation is run
        in batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      Tensor of mROI values with dimensions `(n_chains, n_draws, n_geos,
      n_times, (n_media_channels + n_rf_channels))`. The `n_geos` and `n_times`
      dimensions are dropped if `aggregate_geos=True` or
      `aggregate_times=True`, respectively.
    """
    self._validate_roi_functionality()
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    incremental_revenue_kwargs = {
        "inverse_transform_impact": True,
        "use_kpi": False,
        "use_posterior": use_posterior,
        "batch_size": batch_size,
    }
    roi_tensors = self._get_performance_tensors(
        new_media,
        new_media_spend,
        new_reach,
        new_frequency,
        new_rf_spend,
        **dim_kwargs,
    )
    incremental_revenue = self.incremental_impact(
        new_media=roi_tensors.media,
        new_reach=roi_tensors.reach,
        new_frequency=roi_tensors.frequency,
        **incremental_revenue_kwargs,
        **dim_kwargs,
    )
    incremented_tensors = _scale_tensors_by_multiplier(
        roi_tensors.media,
        roi_tensors.reach,
        roi_tensors.frequency,
        incremental_increase + 1,
        by_reach,
    )
    incremental_revenue_kwargs.update(incremented_tensors)
    incremental_impact_with_multiplier = self.incremental_impact(
        **dim_kwargs, **incremental_revenue_kwargs
    )
    numerator = incremental_impact_with_multiplier - incremental_revenue
    roi_spend = roi_tensors.total_spend() * incremental_increase
    if roi_spend is not None and roi_spend.ndim == 3:
      denominator = self.filter_and_aggregate_geos_and_times(
          roi_spend, **dim_kwargs
      )
    else:
      denominator = roi_spend
    return tf.math.divide_no_nan(numerator, denominator)

  def roi(
      self,
      use_posterior: bool = True,
      new_media: tf.Tensor | None = None,
      new_media_spend: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      new_rf_spend: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> tf.Tensor:
    """Calculates ROI prior or posterior distribution for each media channel.

    The ROI numerator is the change in expected revenue (`kpi *
    revenue_per_kpi`) when one channel's spend is set to zero, leaving all other
    channels' spend unchanged. The ROI denominator is the total spend of the
    channel.

    Args:
      use_posterior: Boolean. If `True` then the posterior distribution is
        calculated. Otherwise, the prior distribution is calculated.
      new_media: Optional tensor with media. Used to compute ROI.
      new_media_spend: Optional tensor with `media_spend` to be used to compute
        ROI.
      new_reach: Optional tensor with reach. Used to compute ROI.
      new_frequency: Optional tensor with frequency. Used to compute ROI.
      new_rf_spend: Optional tensor with rf_spend to be used to compute ROI.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected revenue is summed over
        all of the regions.
      aggregate_times: Boolean. If `True`, the expected revenue is summed over
        all of the time periods.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of ROI values with dimensions `(n_chains, n_draws, n_geos, n_times,
      n_media_channels, n_rf_channels)`. The `n_geos` and `n_times`
      dimensions are dropped if `aggregate_geos=True` or
      `aggregate_times=True`, respectively.
    """
    self._validate_roi_functionality()
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    incremental_impact_kwargs = {
        "inverse_transform_impact": True,
        "use_kpi": False,
        "use_posterior": use_posterior,
        "batch_size": batch_size,
    }
    roi_tensors = self._get_performance_tensors(
        new_media,
        new_media_spend,
        new_reach,
        new_frequency,
        new_rf_spend,
        **dim_kwargs,
    )
    incremental_revenue = self.incremental_impact(
        new_media=roi_tensors.media,
        new_reach=roi_tensors.reach,
        new_frequency=roi_tensors.frequency,
        **incremental_impact_kwargs,
        **dim_kwargs,
    )

    roi_spend = roi_tensors.total_spend()
    if roi_spend is not None and roi_spend.ndim == 3:
      denominator = self.filter_and_aggregate_geos_and_times(
          roi_spend, **dim_kwargs
      )
    else:
      denominator = roi_spend
    return tf.math.divide_no_nan(incremental_revenue, denominator)

  def cpik(
      self,
      use_posterior: bool = True,
      new_media: tf.Tensor | None = None,
      new_media_spend: tf.Tensor | None = None,
      new_reach: tf.Tensor | None = None,
      new_frequency: tf.Tensor | None = None,
      new_rf_spend: tf.Tensor | None = None,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> tf.Tensor:
    """Calculates the cost per incremental KPI distribution for each channel.

    The CPIK numerator is the total spend on the channel. The CPIK denominator
    is the change in expected KPI when one channel's spend is set to zero,
    leaving all other channels' spend unchanged.

    Args:
      use_posterior: Boolean. If `True` then the posterior distribution is
        calculated. Otherwise, the prior distribution is calculated.
      new_media: Optional tensor with media. Used to compute CPIK.
      new_media_spend: Optional tensor with `media_spend` to be used to compute
        CPIK.
      new_reach: Optional tensor with reach. Used to compute CPIK.
      new_frequency: Optional tensor with frequency. Used to compute CPIK.
      new_rf_spend: Optional tensor with rf_spend to be used to compute CPIK.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected KPI is summed over all of
        the regions.
      aggregate_times: Boolean. If `True`, the expected KPI is summed over all
        of the time periods.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      Tensor of CPIK values with dimensions `(n_chains, n_draws, n_geos,
      n_times, n_media_channels, n_rf_channels)`. The `n_geos` and `n_times`
      dimensions are dropped if `aggregate_geos=True` or
      `aggregate_times=True`, respectively.
    """
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    incremental_impact_kwargs = {
        "inverse_transform_impact": True,
        "use_kpi": True,
        "use_posterior": use_posterior,
        "batch_size": batch_size,
    }
    tensors = self._get_performance_tensors(
        new_media,
        new_media_spend,
        new_reach,
        new_frequency,
        new_rf_spend,
        **dim_kwargs,
    )
    incremental_kpi = self.incremental_impact(
        new_media=tensors.media,
        new_reach=tensors.reach,
        new_frequency=tensors.frequency,
        **incremental_impact_kwargs,
        **dim_kwargs,
    )

    cpik_spend = tensors.total_spend()
    if cpik_spend is not None and cpik_spend.ndim == 3:
      numerator = self.filter_and_aggregate_geos_and_times(
          cpik_spend, **dim_kwargs
      )
    else:
      numerator = cpik_spend
    return tf.math.divide_no_nan(numerator, incremental_kpi)

  def expected_vs_actual_data(
      self, confidence_level: float = 0.9
  ) -> xr.Dataset:
    """Calculates the data for the expected versus actual impact over time.

    Args:
      confidence_level: Confidence level for expected impact credible intervals,
        represented as a value between zero and one. Default: `0.9`.

    Returns:
      A dataset with the expected, baseline, and actual impact metrics.
    """
    mmm = self._meridian
    use_kpi = self._meridian.input_data.revenue_per_kpi is None
    expected_tensor = self.expected_impact(
        aggregate_geos=False, aggregate_times=False, use_kpi=use_kpi
    )
    baseline_tensor = self.expected_impact(
        new_media=tf.zeros_like(mmm.media) if mmm.media is not None else None,
        new_reach=tf.zeros_like(mmm.reach) if mmm.reach is not None else None,
        new_frequency=tf.zeros_like(mmm.frequency)
        if mmm.frequency is not None
        else None,
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=use_kpi,
    )
    expected = np.stack(
        [
            np.mean(expected_tensor, (0, 1)),
            np.quantile(expected_tensor, (1 - confidence_level) / 2, (0, 1)),
            np.quantile(expected_tensor, (1 + confidence_level) / 2, (0, 1)),
        ],
        axis=-1,
    )
    baseline = np.stack(
        [
            np.mean(baseline_tensor, (0, 1)),
            np.quantile(baseline_tensor, (1 - confidence_level) / 2, (0, 1)),
            np.quantile(baseline_tensor, (1 + confidence_level) / 2, (0, 1)),
        ],
        axis=-1,
    )
    if use_kpi:
      actual = mmm.kpi
    else:
      actual = mmm.kpi * mmm.revenue_per_kpi

    coords = {
        constants.GEO: ([constants.GEO], mmm.input_data.geo.data),
        constants.TIME: ([constants.TIME], mmm.input_data.time.data),
        constants.METRIC: (
            [constants.METRIC],
            [constants.MEAN, constants.CI_LO, constants.CI_HI],
        ),
    }
    data_vars = {
        constants.EXPECTED: (
            (constants.GEO, constants.TIME, constants.METRIC),
            expected,
        ),
        constants.BASELINE: (
            (constants.GEO, constants.TIME, constants.METRIC),
            baseline,
        ),
        constants.ACTUAL: ((constants.GEO, constants.TIME), actual),
    }
    attrs = {constants.CONFIDENCE_LEVEL: confidence_level}

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

  def _compute_incremental_impact_aggregate(
      self, use_posterior: bool, use_kpi: bool | None = None, **roi_kwargs
  ):
    """Aggregates the incremental impact for MediaSummary metrics."""
    use_kpi = use_kpi or self._meridian.input_data.revenue_per_kpi is None
    expected_impact = self.expected_impact(
        use_posterior=use_posterior, use_kpi=use_kpi, **roi_kwargs
    )
    incremental_impact_m = self.incremental_impact(
        use_posterior=use_posterior, use_kpi=use_kpi, **roi_kwargs
    )
    new_media = (
        tf.zeros_like(self._meridian.media)
        if self._meridian.media is not None
        else None
    )
    new_reach = (
        tf.zeros_like(self._meridian.reach)
        if self._meridian.reach is not None
        else None
    )
    new_frequency = (
        tf.zeros_like(self._meridian.frequency)
        if self._meridian.frequency is not None
        else None
    )
    incremental_impact_total = expected_impact - self.expected_impact(
        use_posterior=use_posterior,
        new_media=new_media,
        new_reach=new_reach,
        new_frequency=new_frequency,
        use_kpi=use_kpi,
        **roi_kwargs,
    )
    return tf.concat(
        [incremental_impact_m, incremental_impact_total[..., None]],
        axis=-1,
    )

  def media_summary_metrics(
      self,
      confidence_level: float,
      marginal_roi_by_reach: bool = True,
      marginal_roi_incremental_increase: float = 0.01,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      aggregate_geos: bool = True,
      aggregate_times: bool = True,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Returns media summary metrics.

    Args:
      confidence_level: Confidence level for media summary metrics credible
        intervals, represented as a value between zero and one.
      marginal_roi_by_reach: Boolean. Marginal ROI (mROI) is defined as the
        return on the next dollar spent. If this argument is `True`, the
        assumption is that the next dollar spent only impacts reach, holding
        frequency constant. If this argument is `False`, the assumption is that
        the next dollar spent only impacts frequency, holding reach constant.
      marginal_roi_incremental_increase: Small fraction by which each channel's
        spend is increased when calculating its mROI numerator. The mROI
        denominator is this fraction of the channel's total spend.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      aggregate_geos: Boolean. If `True`, the expected impact is summed over all
        of the regions.
      aggregate_times: Boolean. If `True`, the expected impact is summed over
        all of the time periods.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
      An `xr.Dataset` with coordinates: `channel`, `metric` (`mean`, `ci_high`,
      `ci_low`), `distribution` (prior, posterior) and contains the following
      data variables: `impressions`, `pct_of_impressions`, `spend`,
      `pct_of_spend`, `CPM`, `incremental_impact`, `pct_of_contribution`, `roi`,
      `effectiveness`, `mroi`.
    """
    use_kpi = self._meridian.input_data.revenue_per_kpi is None
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": aggregate_geos,
        "aggregate_times": aggregate_times,
    }
    roi_kwargs = {"batch_size": batch_size, **dim_kwargs}
    spend_list = []
    if self._meridian.n_media_channels > 0:
      spend_list.append(self._meridian.media_spend)
    if self._meridian.n_rf_channels > 0:
      spend_list.append(self._meridian.rf_spend)
    # TODO(b/309655751) Add support for 1-dimensional spend.
    aggregated_spend = self.filter_and_aggregate_geos_and_times(
        tensor=tf.concat(spend_list, axis=-1), **dim_kwargs
    )
    spend_with_total = tf.concat(
        [aggregated_spend, tf.reduce_sum(aggregated_spend, -1, keepdims=True)],
        axis=-1,
    )

    impressions_list = []

    if self._meridian.n_media_channels > 0:
      impressions_list.append(
          self._meridian.media[:, -self._meridian.n_times :, :]
      )

    if self._meridian.n_rf_channels > 0:
      impressions_list.append(
          self._meridian.reach[:, -self._meridian.n_times :, :]
          * self._meridian.frequency[:, -self._meridian.n_times :, :]
      )
    aggregated_impressions = self.filter_and_aggregate_geos_and_times(
        tensor=tf.concat(impressions_list, axis=-1), **dim_kwargs
    )
    impressions_with_total = tf.concat(
        [
            aggregated_impressions,
            tf.reduce_sum(aggregated_impressions, -1, keepdims=True),
        ],
        axis=-1,
    )

    incremental_impact_prior = self._compute_incremental_impact_aggregate(
        use_posterior=False, **roi_kwargs
    )
    incremental_impact_posterior = self._compute_incremental_impact_aggregate(
        use_posterior=True, **roi_kwargs
    )
    expected_impact_prior = self.expected_impact(
        use_posterior=False, use_kpi=use_kpi, **roi_kwargs
    )
    expected_impact_posterior = self.expected_impact(
        use_posterior=True, use_kpi=use_kpi, **roi_kwargs
    )

    xr_dims = (
        ((constants.GEO,) if not aggregate_geos else ())
        + ((constants.TIME,) if not aggregate_times else ())
        + (constants.CHANNEL,)
    )
    xr_coords = {
        constants.CHANNEL: (
            [constants.CHANNEL],
            list(self._meridian.input_data.get_all_channels())
            + [constants.ALL_CHANNELS],
        ),
    }
    if not aggregate_geos:
      geo_dims = (
          self._meridian.input_data.geo.data
          if selected_geos is None
          else selected_geos
      )
      xr_coords[constants.GEO] = ([constants.GEO], geo_dims)
    if not aggregate_times:
      time_dims = (
          self._meridian.input_data.time.data
          if selected_times is None
          else selected_times
      )
      xr_coords[constants.TIME] = ([constants.TIME], time_dims)
    xr_dims_with_ci_and_distribution = xr_dims + (
        constants.METRIC,
        constants.DISTRIBUTION,
    )
    xr_coords_with_ci_and_distribution = {
        constants.METRIC: (
            [constants.METRIC],
            [constants.MEAN, constants.CI_LO, constants.CI_HI],
        ),
        constants.DISTRIBUTION: (
            [constants.DISTRIBUTION],
            [constants.PRIOR, constants.POSTERIOR],
        ),
        **xr_coords,
    }
    spend_data = self._compute_spend_data_aggregate(
        spend_with_total=spend_with_total,
        impressions_with_total=impressions_with_total,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
    )
    incremental_impact = _mean_and_ci(
        prior=incremental_impact_prior,
        posterior=incremental_impact_posterior,
        metric_name=constants.INCREMENTAL_IMPACT,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
    )
    pct_of_contribution = self._compute_pct_of_contribution(
        incremental_impact_prior=incremental_impact_prior,
        incremental_impact_posterior=incremental_impact_posterior,
        expected_impact_prior=expected_impact_prior,
        expected_impact_posterior=expected_impact_posterior,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
    )
    effectiveness = self._compute_effectiveness_aggregate(
        incremental_impact_prior=incremental_impact_prior,
        incremental_impact_posterior=incremental_impact_posterior,
        impressions_with_total=impressions_with_total,
        xr_dims=xr_dims_with_ci_and_distribution,
        xr_coords=xr_coords_with_ci_and_distribution,
        confidence_level=confidence_level,
    )
    if use_kpi:
      roi = xr.Dataset()
      mroi = xr.Dataset()
      cpik = self._compute_cpik_aggregate(
          incremental_kpi_prior=self._compute_incremental_impact_aggregate(
              use_posterior=False, use_kpi=True, **roi_kwargs
          ),
          incremental_kpi_posterior=self._compute_incremental_impact_aggregate(
              use_posterior=True, use_kpi=True, **roi_kwargs
          ),
          spend_with_total=spend_with_total,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
      )
    else:
      cpik = xr.Dataset()
      roi = self._compute_roi_aggregate(
          incremental_revenue_prior=incremental_impact_prior,
          incremental_revenue_posterior=incremental_impact_posterior,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
          spend_with_total=spend_with_total,
      )
      mroi = self._compute_marginal_roi_aggregate(
          marginal_roi_by_reach=marginal_roi_by_reach,
          marginal_roi_incremental_increase=marginal_roi_incremental_increase,
          expected_revenue_prior=expected_impact_prior,
          expected_revenue_posterior=expected_impact_posterior,
          xr_dims=xr_dims_with_ci_and_distribution,
          xr_coords=xr_coords_with_ci_and_distribution,
          confidence_level=confidence_level,
          spend_with_total=spend_with_total,
          **roi_kwargs,
      )
    if not aggregate_times:
      # Impact metrics should not be normalized by weekly media metrics, which
      # do not have a clear interpretation due to lagged effects. Therefore,
      # the NA values are returned for certain metrics if
      # aggregate_times=False.
      if use_kpi:
        warning = (
            "Effectiveness and CPIK are not reported because they do not have a"
            " clear interpretation by time period."
        )
        cpik *= np.nan
      else:
        warning = (
            "ROI, mROI, and Effectiveness are not reported because they do not"
            " have a clear interpretation by time period."
        )
        roi *= np.nan
        mroi *= np.nan
      effectiveness *= np.nan
      warnings.warn(warning)
    return xr.merge([
        spend_data,
        incremental_impact,
        pct_of_contribution,
        roi,
        effectiveness,
        mroi,
        cpik,
    ])

  def optimal_freq(
      self,
      freq_grid: Sequence[float] | None = None,
      confidence_level: float = 0.9,
      use_posterior: bool = True,
      selected_geos: Sequence[str | int] | None = None,
      selected_times: Sequence[str | int] | None = None,
  ) -> xr.Dataset:
    """Calculates the optimal frequency that maximizes posterior mean ROI/CPIK.

    In the case that `revenue_per_kpi` is not known and ROI is not available,
    the optimal frequency is calculated using cost per incremental KPI instead.

    For this optimization, frequency is restricted to be constant across all
    geographic regions and time periods. Reach is calculated for each
    geographic area and time period such that the number of impressions
    remains unchanged as frequency varies. Meridian solves for the frequency at
    which posterior mean ROI or CPIK is maximized.

    Args:
      freq_grid: List of frequency values. The ROI/CPIK of each channel is
        calculated for each frequency value in the list. By default, the list
        includes numbers from `1.0` to the maximum frequency in increments of
        `0.1`.
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.
      use_posterior: Boolean. If `True`, posterior optimal frequencies are
        generated. If `False`, prior optimal frequencies are generated.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.

    Returns:
      An xarray Dataset containing two variables: `optimal_frequency` and
        `roi_by_frequency` or `cpik_by_frequency`. `optimal_frequency` is the
        frequency that optimizes the posterior mean of ROI or CPIK.
        `roi_by_frequency` is the ROI for each frequency value while
        `cpik_by_frequency` is the CPIK fro each frequency value.

    Raises:
      NotFittedModelError: If `sample_posterior()` (for `use_posterior=True`)
        or `sample_prior()` (for `use_posterior=False`) has not been called
        prior to calling this method.
      ValueError: If there are no channels with reach and frequency data.
    """
    dist_type = constants.POSTERIOR if use_posterior else constants.PRIOR
    if self._meridian.n_rf_channels == 0:
      raise ValueError(
          "Must have at least one channel with reach and frequency data."
      )
    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          f"sample_{dist_type}() must be called prior to calling this method."
      )
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": True,
        "aggregate_times": True,
    }
    use_roi = self._meridian.input_data.revenue_per_kpi is not None

    max_freq = np.max(np.array(self._meridian.frequency))
    if freq_grid is None:
      freq_grid = np.arange(1, max_freq, 0.1)
    metric = np.zeros(
        (len(freq_grid), self._meridian.n_rf_channels, 3)
    )  #  Last argument is 3 for the mean, lower and upper confidence intervals.

    for i, freq in enumerate(freq_grid):
      new_frequency = tf.ones_like(self._meridian.frequency) * freq
      new_reach = (
          self._meridian.frequency * self._meridian.reach / new_frequency
      )
      if use_roi:
        metric_temp = self.roi(
            new_reach=new_reach,
            new_frequency=new_frequency,
            use_posterior=use_posterior,
            **dim_kwargs,
        )[..., -self._meridian.n_rf_channels :]
      else:
        metric_temp = self.cpik(
            new_reach=new_reach,
            new_frequency=new_frequency,
            use_posterior=use_posterior,
            **dim_kwargs,
        )[..., -self._meridian.n_rf_channels :]
      metric[i, :, 0] = np.mean(metric_temp, (0, 1))
      metric[i, :, 1] = np.quantile(
          metric_temp, (1 - confidence_level) / 2, (0, 1)
      )
      metric[i, :, 2] = np.quantile(
          metric_temp, (1 + confidence_level) / 2, (0, 1)
      )

    optimal_freq_idx = (
        np.nanargmax(metric[:, :, 0], axis=0)
        if use_roi
        else np.nanargmin(metric[:, :, 0], axis=0)
    )
    rf_channel_values = (
        self._meridian.input_data.rf_channel.values
        if self._meridian.input_data.rf_channel is not None
        else []
    )
    metric_name = constants.ROI if use_roi else constants.CPIK
    return xr.Dataset(
        data_vars={
            metric_name: (
                [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
                metric,
            ),
            constants.OPTIMAL_FREQUENCY: (
                [constants.RF_CHANNEL],
                [freq_grid[i] for i in optimal_freq_idx],
            ),
        },
        coords={
            constants.FREQUENCY: ([constants.FREQUENCY], freq_grid),
            constants.RF_CHANNEL: ([constants.RF_CHANNEL], rf_channel_values),
            constants.METRIC: (
                [constants.METRIC],
                [constants.MEAN, constants.CI_LO, constants.CI_HI],
            ),
        },
        attrs={
            constants.CONFIDENCE_LEVEL: confidence_level,
            "use_posterior": use_posterior,
        },
    )

  def predictive_accuracy(
      self,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Calculate `R-Squared`, `MAPE`, and `wMAPE` goodness of fit metrics.

    `R-Squared`, `MAPE`, and `wMAPE` are calculated on the KPI scale when
    `revenue_per_kpi = None`, or the revenue scale (`KPI * revenue_per_kpi`)
    when `revenue_per_kpi` is specified, which is the same scale as the ROI
    numerator (incremental revenue).

    `R-Squared`, `MAPE` and `wMAPE` are calculated both at the model-level (one
    observation per geo and time period) and at the national-level (aggregating
    KPI or revenue impact) across geos so there is one observation per time
    period).

    `R-Squared`, `MAPE`, and `wMAPE` are calculated for the full sample. If the
    model object has any holdout observations, then `R-squared`, `MAPE`, and
    `wMAPE` are also calculated for the `Train` and `Test` subsets.

    Args:
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list containing a subset of dates to include. By
        default, all time periods are included.
      batch_size: Integer representing the maximum draws per chain in each
        batch. By default, `batch_size` is `100`. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      An xarray Dataset containing the computed `R_Squared`, `MAPE`, and `wMAPE`
      values, with coordinates `metric`, `geo_granularity`, `evaluation_set`,
      and accompanying data variable `value`. If `holdout_id` exists, the data
      is split into `'Train'`, `'Test'`, and `'All Data'` subsections, and the
      three metrics are computed for each.
    """
    use_kpi = self._meridian.input_data.revenue_per_kpi is None
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          selected_geos=selected_geos,
      )
    dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": False,
        "aggregate_times": False,
    }

    xr_dims = [constants.METRIC, constants.GEO_GRANULARITY]
    xr_coords = {
        constants.METRIC: (
            [constants.METRIC],
            [constants.R_SQUARED, constants.MAPE, constants.WMAPE],
        ),
        constants.GEO_GRANULARITY: (
            [constants.GEO_GRANULARITY],
            [constants.GEO, constants.NATIONAL],
        ),
    }
    if self._meridian.revenue_per_kpi is not None:
      input_tensor = self._meridian.kpi * self._meridian.revenue_per_kpi
    else:
      input_tensor = self._meridian.kpi
    actual = self.filter_and_aggregate_geos_and_times(
        tensor=input_tensor,
        **dims_kwargs,
    ).numpy()
    expected = np.mean(
        self.expected_impact(
            batch_size=batch_size, use_kpi=use_kpi, **dims_kwargs
        ),
        (0, 1),
    )
    rsquared, mape, wmape = self._predictive_accuracy_helper(actual, expected)
    rsquared_national, mape_national, wmape_national = (
        self._predictive_accuracy_helper(np.sum(actual, 0), np.sum(expected, 0))
    )
    if self._meridian.model_spec.holdout_id is None:
      rsquared_arr = [rsquared, rsquared_national]
      mape_arr = [mape, mape_national]
      wmape_arr = [wmape, wmape_national]

      stacked_metric_values = np.stack([rsquared_arr, mape_arr, wmape_arr])

      xr_data = {constants.VALUE: (xr_dims, stacked_metric_values)}
      dataset = xr.Dataset(data_vars=xr_data, coords=xr_coords)
    else:
      xr_dims.append(constants.EVALUATION_SET_VAR)
      xr_coords[constants.EVALUATION_SET_VAR] = (
          [constants.EVALUATION_SET_VAR],
          list(constants.EVALUATION_SET),
      )
      nansum = lambda x: np.where(
          np.all(np.isnan(x), 0), np.nan, np.nansum(x, 0)
      )
      actual_train = np.where(
          self._meridian.model_spec.holdout_id, np.nan, actual
      )
      actual_test = np.where(
          self._meridian.model_spec.holdout_id, actual, np.nan
      )
      expected_train = np.where(
          self._meridian.model_spec.holdout_id, np.nan, expected
      )
      expected_test = np.where(
          self._meridian.model_spec.holdout_id, expected, np.nan
      )

      geo_train = self._predictive_accuracy_helper(actual_train, expected_train)
      national_train = self._predictive_accuracy_helper(
          nansum(actual_train), nansum(expected_train)
      )
      geo_test = self._predictive_accuracy_helper(actual_test, expected_test)
      national_test = self._predictive_accuracy_helper(
          nansum(actual_test), nansum(expected_test)
      )
      geo_all_data = [rsquared, mape, wmape]
      national_all_data = [rsquared_national, mape_national, wmape_national]

      stacked_train = np.stack([geo_train, national_train], axis=-1)
      stacked_test = np.stack([geo_test, national_test], axis=-1)
      stacked_all_data = np.stack([geo_all_data, national_all_data], axis=-1)
      stacked_total = np.stack(
          [stacked_train, stacked_test, stacked_all_data], axis=-1
      )
      xr_data = {constants.VALUE: (xr_dims, stacked_total)}
      dataset = xr.Dataset(data_vars=xr_data, coords=xr_coords)
    if self._meridian.is_national:
      # Remove the geo-level coordinate.
      dataset = dataset.sel(geo_granularity=[constants.NATIONAL])
    return dataset

  def _predictive_accuracy_helper(
      self,
      actual_eval_set: np.ndarray,
      expected_eval_set: np.ndarray,
  ) -> list[np.floating]:
    """Calculates the predictive accuracy metrics when `holdout_id` exists.

    Args:
      actual_eval_set: An array with filtered and/or aggregated geo and time
        dimensions for the `meridian.kpi * meridian.revenue_per_kpi` calculation
        for either the `'Train'`, `'Test'`, or `'All Data'` evaluation sets.
      expected_eval_set: An array of expected impact with dimensions `(n_chains,
        n_draws, n_geos, n_times)` for either the `'Train'`, `'Test'`, or
        `'All Data'` evaluation sets.

    Returns:
      A list containing the `geo` or `national` level data for the `R_Squared`,
      `MAPE`, and `wMAPE` metrics computed for either a `'Train'`, `'Test'`, or
      `'All Data'` evaluation set.
    """
    rsquared = _calc_rsquared(expected_eval_set, actual_eval_set)
    mape = _calc_mape(expected_eval_set, actual_eval_set)
    wmape = _calc_weighted_mape(expected_eval_set, actual_eval_set)
    return [rsquared, mape, wmape]

  def _get_r_hat(self) -> Mapping[str, tf.Tensor]:
    """Computes the R-hat values for each parameter in the model.

    Returns:
      A dictionary of r-hat values where each parameter is a key and values are
      r-hats corresponding to the parameter.

    Raises:
      NotFittedModelError: If self.sample_posterior() is not called before
        calling this method.
    """
    if constants.POSTERIOR not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          "sample_posterior() must be called prior to calling this method."
      )

    def _transpose_first_two_dims(x: tf.Tensor) -> tf.Tensor:
      n_dim = len(x.shape)
      perm = [1, 0] + list(range(2, n_dim))
      return tf.transpose(x, perm)

    r_hat = tfp.mcmc.potential_scale_reduction({
        k: _transpose_first_two_dims(v)
        for k, v in self._meridian.inference_data.posterior.data_vars.items()
    })
    return r_hat

  def r_hat_summary(self, bad_r_hat_threshold: float = 1.2) -> pd.DataFrame:
    """Computes a summary of the R-hat values for each parameter in the model.

    Calculates the Gelman & Rubin (1992) potential scale reduction for chain
    convergence, commonly referred to as R-hat. It is a convergence diagnostic
    measure that measures the degree to which variance (of the means) between
    chains exceeds what you would expect if the chains were identically
    distributed. Values close to 1.0 indicate convergence. R-hat < 1.2 indicates
    approximate convergence and is a reasonable threshold for many problems
    (Brooks & Gelman, 1998).

    References:
      Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
        Using Multiple Sequences. Statistical Science, 7(4):457-472, 1992.
      Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
        Convergence of Iterative Simulations. Journal of Computational and
        Graphical Statistics, 7(4), 1998.

    Args:
      bad_r_hat_threshold: The threshold for determining which R-hat values are
        considered bad.

    Returns:
      A DataFrame with the following columns:

      *   `n_params`: The number of respective parameters in the model.
      *   `avg_rhat`: The average R-hat value for the respective parameter.
      *   `n_params`: The number of respective parameters in the model.
      *   `avg_rhat`: The average R-hat value for the respective parameter.
      *   `max_rhat`: The maximum R-hat value for the respective parameter.
      *   `percent_bad_rhat`: The percentage of R-hat values for the respective
          parameter that are greater than `bad_r_hat_threshold`.
      *   `row_idx_bad_rhat`: The row indices of the R-hat values that are
          greater than `bad_r_hat_threshold`.
      *   `col_idx_bad_rhat`: The column indices of the R-hat values that are
          greater than `bad_r_hat_threshold`.

    Raises:
      NotFittedModelError: If `self.sample_posterior()` is not called before
        calling this method.
      ValueError: If the number of dimensions of the R-hat array for a parameter
        is not `1` or `2`.
    """
    r_hat = self._get_r_hat()

    r_hat_summary = []
    for param in r_hat:
      # Skip if parameter is deterministic according to the prior.
      if self._meridian.prior_broadcast.has_deterministic_param(param):
        continue

      bad_idx = np.where(r_hat[param] > bad_r_hat_threshold)
      if len(bad_idx) == 2:
        row_idx, col_idx = bad_idx
      elif len(bad_idx) == 1:
        row_idx = bad_idx[0]
        col_idx = []
      else:
        raise ValueError(f"Unexpected dimension for parameter {param}.")

      r_hat_summary.append(
          pd.Series({
              constants.PARAM: param,
              constants.N_PARAMS: np.prod(r_hat[param].shape),
              constants.AVG_RHAT: np.nanmean(r_hat[param]),
              constants.MAX_RHAT: np.nanmax(r_hat[param]),
              constants.PERCENT_BAD_RHAT: np.nanmean(
                  r_hat[param] > bad_r_hat_threshold
              ),
              constants.ROW_IDX_BAD_RHAT: row_idx,
              constants.COL_IDX_BAD_RHAT: col_idx,
          })
      )
    return pd.DataFrame(r_hat_summary)

  def response_curves(
      self,
      spend_multipliers: list[float] | None = None,
      confidence_level: float = 0.9,
      use_posterior: bool = True,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      by_reach: bool = True,
      use_optimal_frequency: bool = False,
      batch_size: int = constants.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Method to generate a response curves XArray Dataset.

    Response curves are calculated at the national-level, assuming the
    historical flighting pattern across geos and time periods for each media
    channel. A list of multipliers is applied to each media channel's total
    historical spend to obtain the `x-values` at which the channel's response
    curve is calculated.

    Args:
      spend_multipliers: List of multipliers. Each channel's total spend is
        multiplied by these factors to obtain the values at which the curve is
        calculated for that channel.
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.
      use_posterior: Boolean. If `True`, posterior response curves are
        generated. If `False`, prior response curves are generated.
      selected_geos: Optional list containing a subset of geos to include. By
        default, all geos are included.
      selected_times: Optional list of containing a subset of time dimensions to
        include. By default, all time periods are included. Time dimension
        strings and integers must align with the `Meridian.n_times`.
      by_reach: Boolean. For channels with reach and frequency. If `True`, plots
        the response curve by reach. If `False`, plots the response curve by
        frequency.
      use_optimal_frequency: If `True`, uses the optimal frequency to plot the
        response curves. Defaults to `False`.
      batch_size: Integer representing the maximum draws per chain in each
        batch. The calculation is run in batches to avoid memory exhaustion. If
        a memory error occurs, try reducing `batch_size`. The calculation will
        generally be faster with larger `batch_size` values.

    Returns:
        An xarray Dataset containing the data needed to visualize response
        curves.
    """
    use_kpi = self._meridian.input_data.revenue_per_kpi is None
    if self._meridian.is_national:
      _warn_if_geo_arg_in_kwargs(
          selected_geos=selected_geos,
      )
    dim_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
        "aggregate_geos": True,
        "aggregate_times": True,
    }
    if self._meridian.n_rf_channels > 0 and use_optimal_frequency:
      frequency = tf.ones_like(self._meridian.frequency) * tf.convert_to_tensor(
          self.optimal_freq(
              selected_geos=selected_geos, selected_times=selected_times
          ).optimal_frequency,
          dtype=tf.float32,
      )
      reach = tf.math.divide_no_nan(
          self._meridian.reach * self._meridian.frequency, frequency
      )
    else:
      frequency = self._meridian.frequency
      reach = self._meridian.reach
    if spend_multipliers is None:
      spend_multipliers = list(np.arange(0, 2.2, 0.2))
    incremental_impact = np.zeros((
        len(spend_multipliers),
        len(self._meridian.input_data.get_all_channels()),
        3,
    ))
    for i, multiplier in enumerate(spend_multipliers):
      if multiplier == 0:
        incremental_impact[i, :, :] = tf.zeros(
            (len(self._meridian.input_data.get_all_channels()), 3)
        )  # Last dimension = 3 for the mean, ci_lo and ci_hi.
        continue
      tensor_kwargs = _scale_tensors_by_multiplier(
          self._meridian.media,
          reach,
          frequency,
          multiplier=multiplier,
          by_reach=by_reach,
      )
      incimpact_temp = self.incremental_impact(
          use_posterior=use_posterior,
          inverse_transform_impact=True,
          batch_size=batch_size,
          use_kpi=use_kpi,
          **tensor_kwargs,
          **dim_kwargs,
      )
      incremental_impact[i, :, 0] = np.mean(incimpact_temp, (0, 1))
      incremental_impact[i, :, 1] = np.quantile(
          incimpact_temp, (1 - confidence_level) / 2, (0, 1)
      )
      incremental_impact[i, :, 2] = np.quantile(
          incimpact_temp, (1 + confidence_level) / 2, (0, 1)
      )
    if self._meridian.n_media_channels > 0 and self._meridian.n_rf_channels > 0:
      spend = tf.concat(
          [self._meridian.media_spend, self._meridian.rf_spend], axis=-1
      )
    elif self._meridian.n_media_channels > 0:
      spend = self._meridian.media_spend
    else:
      spend = self._meridian.rf_spend

    if tf.rank(spend) == 3:
      spend = self.filter_and_aggregate_geos_and_times(
          tensor=spend,
          **dim_kwargs,
      )
    spend_einsum = tf.einsum("k,m->km", np.array(spend_multipliers), spend)
    xr_coords = {
        constants.CHANNEL: (
            [constants.CHANNEL],
            self._meridian.input_data.get_all_channels(),
        ),
        constants.METRIC: (
            [constants.METRIC],
            [constants.MEAN, constants.CI_LO, constants.CI_HI],
        ),
        constants.SPEND_MULTIPLIER: (
            [constants.SPEND_MULTIPLIER],
            spend_multipliers,
        ),
    }
    xr_data_vars = {
        constants.SPEND: (
            [constants.SPEND_MULTIPLIER, constants.CHANNEL],
            spend_einsum,
        ),
        constants.INCREMENTAL_IMPACT: (
            [constants.SPEND_MULTIPLIER, constants.CHANNEL, constants.METRIC],
            incremental_impact,
        ),
    }
    attrs = {constants.CONFIDENCE_LEVEL: confidence_level}
    return xr.Dataset(data_vars=xr_data_vars, coords=xr_coords, attrs=attrs)

  def adstock_decay(self, confidence_level: float = 0.9) -> pd.DataFrame:
    """Calculates adstock decay for media and reach and frequency channels.

    Args:
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one.

    Returns:
      Pandas DataFrame containing the channel, `time_units`, distribution,
      `ci_hi`, `ci_lo`, and `mean` for the Adstock function.
    """
    if (
        constants.PRIOR not in self._meridian.inference_data.groups()
        or constants.POSTERIOR not in self._meridian.inference_data.groups()
    ):
      raise model.NotFittedModelError(
          "sample_prior() and sample_posterior() must be called prior to"
          " calling this method."
      )

    # Choose a step_size such that time_unit has consecutive integers defined
    # throughout.
    max_lag = max(self._meridian.model_spec.max_lag, 1)
    steps_per_time_period_max_lag = (
        constants.ADSTOCK_DECAY_MAX_TOTAL_STEPS // max_lag
    )
    steps_per_time_period = min(
        constants.ADSTOCK_DECAY_DEFAULT_STEPS_PER_TIME_PERIOD,
        steps_per_time_period_max_lag,
    )
    step_size = 1 / steps_per_time_period
    l_range = np.arange(0, max_lag, step_size)

    rf_channel_values = (
        self._meridian.input_data.rf_channel.values
        if self._meridian.input_data.rf_channel is not None
        else []
    )

    media_channel_values = (
        self._meridian.input_data.media_channel.values
        if self._meridian.input_data.media_channel is not None
        else []
    )

    xr_dims = [
        constants.TIME_UNITS,
        constants.CHANNEL,
        constants.METRIC,
        constants.DISTRIBUTION,
    ]
    xr_coords = {
        constants.TIME_UNITS: ([constants.TIME_UNITS], l_range),
        constants.CHANNEL: (
            [constants.CHANNEL],
            rf_channel_values,
        ),
        constants.DISTRIBUTION: (
            [constants.DISTRIBUTION],
            [constants.PRIOR, constants.POSTERIOR],
        ),
        constants.METRIC: (
            [constants.METRIC],
            [constants.MEAN, constants.CI_LO, constants.CI_HI],
        ),
    }
    final_df = pd.DataFrame()

    if self._meridian.n_rf_channels > 0:
      adstock_df_rf = self._get_adstock_dataframe(
          constants.REACH,
          l_range,
          xr_dims,
          xr_coords,
          confidence_level,
      )
      final_df = pd.concat([final_df, adstock_df_rf], axis=0)
    if self._meridian.n_media_channels > 0:
      xr_coords[constants.CHANNEL] = ([constants.CHANNEL], media_channel_values)
      adstock_df_m = self._get_adstock_dataframe(
          constants.MEDIA,
          l_range,
          xr_dims,
          xr_coords,
          confidence_level,
      )
      final_df = pd.concat([final_df, adstock_df_m], axis=0).reset_index(
          drop=True
      )

    # Adding an extra column that indicates whether time_units is an integer
    # for marking the discrete points on the plot.
    final_df[constants.IS_INT_TIME_UNIT] = final_df[constants.TIME_UNITS].apply(
        lambda x: x.is_integer()
    )
    return final_df

  def _get_hill_curves_dataframe(
      self, channel_type: str, confidence_level: float = 0.9
  ) -> pd.DataFrame:
    """Computes the point-wise mean and credible intervals for the Hill curves.

    Args:
      channel_type: Type of channel, either `media` or `rf`.
      confidence_level: Confidence level for `posterior` and `prior` credible
        intervals, represented as a value between zero and one.

    Returns:
      A DataFrame with data needed to plot the Hill curves, with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `media_units`: Media (for `media` channels) or average frequency (for
          `rf` channels) units.
      *   `distribution`: Indication of `posterior` or `prior` draw.
      *   `ci_hi`: Upper bound of the credible interval of the value of the Hill
          function.
      *   `ci_lo`: Lower bound of the credible interval of the value of the Hill
          function.
      *   `mean`: Point-wise mean of the value of the Hill function per draw.
      *   channel_type: Indication of a `media` or `rf` channel.
    """
    if (
        channel_type == constants.MEDIA
        and self._meridian.input_data.media_channel is not None
    ):
      ec = constants.EC_M
      slope = constants.SLOPE_M
      linspace = np.linspace(
          0,
          np.max(np.array(self._meridian.media_scaled), axis=(0, 1)),
          constants.HILL_NUM_STEPS,
      )
      channels = self._meridian.input_data.media_channel.values
    elif (
        channel_type == constants.RF
        and self._meridian.input_data.rf_channel is not None
    ):
      ec = constants.EC_RF
      slope = constants.SLOPE_RF
      linspace = np.linspace(
          0,
          np.max(np.array(self._meridian.frequency), axis=(0, 1)),
          constants.HILL_NUM_STEPS,
      )
      channels = self._meridian.input_data.rf_channel.values
    else:
      raise ValueError(
          f"Unsupported channel type: {channel_type} or the"
          " requested type of channels (`media` or `rf`) are not present."
      )
    linspace_filler = np.linspace(0, 1, constants.HILL_NUM_STEPS)
    xr_dims = [
        constants.MEDIA_UNITS,
        constants.CHANNEL,
        constants.METRIC,
        constants.DISTRIBUTION,
    ]
    xr_coords = {
        constants.MEDIA_UNITS: ([constants.MEDIA_UNITS], linspace_filler),
        constants.CHANNEL: (
            [constants.CHANNEL],
            list(channels),
        ),
        constants.DISTRIBUTION: (
            [constants.DISTRIBUTION],
            [constants.PRIOR, constants.POSTERIOR],
        ),
        constants.METRIC: (
            [constants.METRIC],
            [constants.MEAN, constants.CI_LO, constants.CI_HI],
        ),
    }
    # Expanding the linspace by one dimension since the HillTransformer requires
    # 3-dimensional input as (geo, time, channel).
    expanded_linspace = tf.expand_dims(linspace, axis=0)
    # Including [:, :, 0, :, :] in the output of the Hill Function to reduce the
    # tensors by the geo dimension. Original Hill dimension shape is (n_chains,
    # n_draws, n_geos, n_times, n_channels), and we want to plot the
    # dependency on time only.
    hill_vals_prior = adstock_hill.HillTransformer(
        self._meridian.inference_data.prior[ec].values,
        self._meridian.inference_data.prior[slope].values,
    ).forward(expanded_linspace)[:, :, 0, :, :]
    hill_vals_posterior = adstock_hill.HillTransformer(
        self._meridian.inference_data.posterior[ec].values,
        self._meridian.inference_data.posterior[slope].values,
    ).forward(expanded_linspace)[:, :, 0, :, :]

    hill_dataset = _mean_and_ci(
        hill_vals_prior,
        hill_vals_posterior,
        constants.HILL_SATURATION_LEVEL,
        xr_dims,
        xr_coords,
        confidence_level,
    )
    df = (
        hill_dataset[constants.HILL_SATURATION_LEVEL]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.MEDIA_UNITS,
                constants.DISTRIBUTION,
            ],
            columns=constants.METRIC,
            values=constants.HILL_SATURATION_LEVEL,
        )
        .reset_index()
    )

    # Fill media_units or frequency x-axis with the correct range.
    media_units_arr = []
    if channel_type == constants.MEDIA:
      media_transformers = transformers.MediaTransformer(
          self._meridian.media, self._meridian.population
      )
      population_scaled_median_m = media_transformers.population_scaled_median_m
      x_range_full_shape = (
          linspace * population_scaled_median_m[:, np.newaxis].T
      )
    else:
      x_range_full_shape = linspace

    # Flatten this into a list.
    x_range_list = x_range_full_shape.flatten("F").tolist()
    # Doubles each value in the list to account for alternating prior
    # and posterior.
    x_range_doubled = list(
        itertools.chain.from_iterable(zip(x_range_list, x_range_list))
    )
    media_units_arr.extend(x_range_doubled)

    df[constants.CHANNEL_TYPE] = channel_type
    df[constants.MEDIA_UNITS] = media_units_arr
    return df

  def _get_hill_histogram_dataframe(self, n_bins: int) -> pd.DataFrame:
    """Returns the bucketed media_units counts per each `media` or `rf` channel.

    Args:
      n_bins: Number of equal-width bins to include in the histogram for the
        plotting.

    Returns:
      Pandas DataFrame with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `channel_type`: `media` or `rf` channel type.
      *   `scaled_count_histogram`: Scaled count of media units or average
          frequencies within the bin.
      *   `count_histogram`: True count value of media units or average
          frequencies within the bin.
      *   `start_interval_histogram`: Media unit or average frequency starting
          point for a histogram bin.
      *   `end_interval_histogram`: Media unit or average frequency ending point
          for a histogram bin.

      This DataFrame will be used to plot the histograms showing the relative
      distribution of media units per capita for media channels or average
      frequency for RF channels over weeks and geos for the Hill plots.
    """
    n_geos = self._meridian.n_geos
    n_media_times = self._meridian.n_media_times
    n_rf_channels = self._meridian.n_rf_channels
    n_media_channels = self._meridian.n_media_channels

    (
        channels,
        scaled_count,
        channel_type_arr,
        start_interval_histogram,
        end_interval_histogram,
        count,
    ) = ([], [], [], [], [], [])

    # RF.
    if self._meridian.input_data.rf_channel is not None:
      frequency = (
          self._meridian.frequency
      )  # Shape: (n_geos, n_media_times, n_channels).
      reshaped_frequency = tf.reshape(
          frequency, (n_geos * n_media_times, n_rf_channels)
      )
      for i, channel in enumerate(self._meridian.input_data.rf_channel.values):
        # Bucketize the histogram data for RF channels.
        counts_per_bucket, buckets = np.histogram(
            reshaped_frequency[:, i], bins=n_bins, density=True
        )
        channels.extend([channel] * len(counts_per_bucket))
        channel_type_arr.extend([constants.RF] * len(counts_per_bucket))
        scaled_count.extend(counts_per_bucket / max(counts_per_bucket))
        count.extend(counts_per_bucket)
        start_interval_histogram.extend(buckets[:-1])
        end_interval_histogram.extend(buckets[1:])

    # Media.
    if self._meridian.input_data.media_channel is not None:
      transformer = transformers.MediaTransformer(
          self._meridian.media, self._meridian.population
      )
      scaled = (
          self._meridian.media_scaled
      )  # Shape: (n_geos, n_media_times, n_channels)
      population_scaled_median = transformer.population_scaled_median_m
      scaled_media_units = scaled * population_scaled_median
      reshaped_scaled_media_units = tf.reshape(
          scaled_media_units, (n_geos * n_media_times, n_media_channels)
      )
      for i, channel in enumerate(
          self._meridian.input_data.media_channel.values
      ):
        # Bucketize the histogram data for media channels.
        counts_per_bucket, buckets = np.histogram(
            reshaped_scaled_media_units[:, i], bins=n_bins, density=True
        )
        channel_type_arr.extend([constants.MEDIA] * len(counts_per_bucket))
        channels.extend([channel] * (len(counts_per_bucket)))
        scaled_count.extend(counts_per_bucket / max(counts_per_bucket))
        count.extend(counts_per_bucket)
        start_interval_histogram.extend(buckets[:-1])
        end_interval_histogram.extend(buckets[1:])

    return pd.DataFrame({
        constants.CHANNEL: channels,
        constants.CHANNEL_TYPE: channel_type_arr,
        constants.SCALED_COUNT_HISTOGRAM: scaled_count,
        constants.COUNT_HISTOGRAM: count,
        constants.START_INTERVAL_HISTOGRAM: start_interval_histogram,
        constants.END_INTERVAL_HISTOGRAM: end_interval_histogram,
    })

  def hill_curves(
      self, confidence_level: float = 0.9, n_bins: int = 25
  ) -> pd.DataFrame:
    """Estimates Hill curve tables used for plotting each channel's curves.

    Args:
      confidence_level: Confidence level for prior and posterior credible
        intervals, represented as a value between zero and one. Default is
        `0.9`.
      n_bins: Number of equal-width bins to include in the histogram for the
        plotting. Default is `25`.

    Returns:
      Hill Curves pd.DataFrame with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `media_units`: Media (for `media` channels) or average frequency (for
          `rf` channels) units.
      *   `distribution`: Indication of `posterior` or `prior` draw.
      *   `ci_hi`: Upper bound of the credible interval of the value of the Hill
          function.
      *   `ci_lo`: Lower bound of the credible interval of the value of the Hill
          function.
      *   `mean`: Point-wise mean of the value of the Hill function per draw.
      *   `channel_type`: Indication of a `media` or `rf` channel.
      *   `scaled_count_histogram`: Scaled count of media units or average
          frequencies within the bin.
      *   `count_histogram`: True count value of media units or average
          frequencies within the bin.
      *   `start_interval_histogram`: Media unit or average frequency starting
          point for a histogram bin.
      *   `end_interval_histogram`: Media unit or average frequency ending point
          for a histogram bin.
    """
    if (
        constants.PRIOR not in self._meridian.inference_data.groups()
        or constants.POSTERIOR not in self._meridian.inference_data.groups()
    ):
      raise model.NotFittedModelError(
          "sample_prior() and sample_posterior() must be called prior to"
          " calling this method."
      )

    final_dfs = [pd.DataFrame()]
    if self._meridian.n_media_channels > 0:
      hill_df_media = self._get_hill_curves_dataframe(
          constants.MEDIA, confidence_level
      )
      final_dfs.append(hill_df_media)

    if self._meridian.n_rf_channels > 0:
      hill_df_rf = self._get_hill_curves_dataframe(
          constants.RF, confidence_level
      )
      final_dfs.append(hill_df_rf)

    final_dfs.append(self._get_hill_histogram_dataframe(n_bins=n_bins))
    return pd.concat(final_dfs)

  def _compute_roi_aggregate(
      self,
      incremental_revenue_prior: tf.Tensor,
      incremental_revenue_posterior: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float,
      spend_with_total: tf.Tensor,
  ) -> xr.Dataset:
    # TODO(b/304834270): Support calibration_period_bool.
    return _mean_and_ci(
        prior=incremental_revenue_prior / spend_with_total,
        posterior=incremental_revenue_posterior / spend_with_total,
        metric_name=constants.ROI,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
    )

  def _compute_marginal_roi_aggregate(
      self,
      marginal_roi_by_reach: bool,
      marginal_roi_incremental_increase: float,
      expected_revenue_prior: tf.Tensor,
      expected_revenue_posterior: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float,
      spend_with_total: tf.Tensor,
      **roi_kwargs,
  ) -> xr.Dataset:
    self._validate_roi_functionality()
    mroi_prior = self.marginal_roi(
        use_posterior=False,
        by_reach=marginal_roi_by_reach,
        incremental_increase=marginal_roi_incremental_increase,
        **roi_kwargs,
    )
    mroi_posterior = self.marginal_roi(
        use_posterior=True,
        by_reach=marginal_roi_by_reach,
        incremental_increase=marginal_roi_incremental_increase,
        **roi_kwargs,
    )
    incremented_tensors = _scale_tensors_by_multiplier(
        media=self._meridian.media,
        reach=self._meridian.reach,
        frequency=self._meridian.frequency,
        multiplier=(1 + marginal_roi_incremental_increase),
        by_reach=marginal_roi_by_reach,
    )

    mroi_prior_total = (
        self.expected_impact(
            use_posterior=False,
            use_kpi=False,
            **incremented_tensors,
            **roi_kwargs,
        )
        - expected_revenue_prior
    ) / (marginal_roi_incremental_increase * spend_with_total[..., -1])
    mroi_posterior_total = (
        self.expected_impact(
            use_posterior=True,
            use_kpi=False,
            **incremented_tensors,
            **roi_kwargs,
        )
        - expected_revenue_posterior
    ) / (marginal_roi_incremental_increase * spend_with_total[..., -1])
    mroi_prior_concat = tf.concat(
        [mroi_prior, mroi_prior_total[..., None]], axis=-1
    )
    mroi_posterior_concat = tf.concat(
        [mroi_posterior, mroi_posterior_total[..., None]], axis=-1
    )
    return _mean_and_ci(
        prior=mroi_prior_concat,
        posterior=mroi_posterior_concat,
        metric_name=constants.MROI,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
    )

  def _compute_spend_data_aggregate(
      self,
      spend_with_total: tf.Tensor,
      impressions_with_total: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
  ) -> xr.Dataset:
    """Computes the MediaSummary metrics involving the input data.

    Returns:
      An xarray Dataset consisting of the following arrays:

      * `impressions`
      * `pct_of_impressions`
      * `spend`
      * `pct_of_spend`
      * `cpm` (spend for every 1,000 impressions)
    """
    pct_of_impressions = (
        impressions_with_total / impressions_with_total[..., -1:] * 100
    )
    pct_of_spend = spend_with_total / spend_with_total[..., -1:] * 100

    return xr.Dataset(
        data_vars={
            constants.IMPRESSIONS: (xr_dims, impressions_with_total),
            constants.PCT_OF_IMPRESSIONS: (xr_dims, pct_of_impressions),
            constants.SPEND: (xr_dims, spend_with_total),
            constants.PCT_OF_SPEND: (xr_dims, pct_of_spend),
            constants.CPM: (
                xr_dims,
                spend_with_total / impressions_with_total * 1000,
            ),
        },
        coords=xr_coords,
    )

  def _compute_effectiveness_aggregate(
      self,
      incremental_impact_prior: tf.Tensor,
      incremental_impact_posterior: tf.Tensor,
      impressions_with_total: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float,
  ) -> xr.Dataset:
    return _mean_and_ci(
        prior=incremental_impact_prior / impressions_with_total,
        posterior=incremental_impact_posterior / impressions_with_total,
        metric_name=constants.EFFECTIVENESS,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
    )

  def _compute_cpik_aggregate(
      self,
      incremental_kpi_prior: tf.Tensor,
      incremental_kpi_posterior: tf.Tensor,
      spend_with_total: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float,
  ) -> xr.Dataset:
    return _mean_and_ci(
        prior=spend_with_total / incremental_kpi_prior,
        posterior=spend_with_total / incremental_kpi_posterior,
        metric_name=constants.CPIK,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
    )

  def _compute_pct_of_contribution(
      self,
      incremental_impact_prior: tf.Tensor,
      incremental_impact_posterior: tf.Tensor,
      expected_impact_prior: tf.Tensor,
      expected_impact_posterior: tf.Tensor,
      xr_dims: Sequence[str],
      xr_coords: Mapping[str, tuple[Sequence[str], Sequence[str]]],
      confidence_level: float,
  ) -> xr.Dataset:
    """Computes the parts of `MediaSummary` related to mean expected impact."""
    mean_expected_impact_prior = tf.reduce_mean(expected_impact_prior, (0, 1))
    mean_expected_impact_posterior = tf.reduce_mean(
        expected_impact_posterior, (0, 1)
    )
    return _mean_and_ci(
        prior=incremental_impact_prior
        / mean_expected_impact_prior[..., None]
        * 100,
        posterior=incremental_impact_posterior
        / mean_expected_impact_posterior[..., None]
        * 100,
        metric_name=constants.PCT_OF_CONTRIBUTION,
        xr_dims=xr_dims,
        xr_coords=xr_coords,
        confidence_level=confidence_level,
    )
