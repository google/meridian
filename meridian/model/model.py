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

"""Meridian module for the geo-level Bayesian hierarchical media mix model."""

from collections.abc import Mapping, Sequence
import functools
import os
import warnings
import arviz as az
import joblib
from meridian import constants
from meridian.data import input_data as data
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


__all__ = [
    "MCMCSamplingError",
    "MCMCOOMError",
    "Meridian",
    "NotFittedModelError",
    "save_mmm",
    "load_mmm",
]


class NotFittedModelError(Exception):
  """Model has not been fitted."""


class MCMCSamplingError(Exception):
  """The Markov Chain Monte Carlo (MCMC) sampling failed."""


class MCMCOOMError(Exception):
  """The Markov Chain Monte Carlo (MCMC) exceeds memory limits."""


def _warn_setting_national_args(**kwargs):
  """Raises a warning if a geo argument is found in kwargs."""
  for kwarg, value in kwargs.items():
    if (
        kwarg in constants.NATIONAL_MODEL_SPEC_ARGS
        and value is not constants.NATIONAL_MODEL_SPEC_ARGS[kwarg]
    ):
      warnings.warn(
          f"In a nationally aggregated model, the `{kwarg}` will be reset to"
          f" `{constants.NATIONAL_MODEL_SPEC_ARGS[kwarg]}`."
      )


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


class Meridian:
  """Contains the main functionality for fitting the Meridian MMM model.

  Attributes:
    input_data: An `InputData` object containing the input data for the model.
    model_spec: A `ModelSpec` object containing the model specification.
    inference_data: A _mutable_ `arviz.InferenceData` object containing the
      resulting data from fitting the model.
    n_geos: Number of geos in the data.
    n_media_channels: Number of media channels in the data.
    n_rf_channels: Number of reach and frequency (RF) channels in the data.
    n_controls: Number of control variables in the data.
    n_times: Number of time periods in the KPI or spend data.
    n_media_times: Number of time periods in the media data.
    is_national: A boolean indicating whether the data is national (single geo)
      or not (multiple geos).
    knot_info: A `KnotInfo` derived from input data and model spec.
    kpi: A tensor constructed from `input_data.kpi`.
    revenue_per_kpi: A tensor constructed from `input_data.revenue_per_kpi`. If
      `input_data.revenue_per_kpi` is None, then this is also None.
    controls: A tensor constructed from `input_data.controls`.
    population: A tensor constructed from `input_data.population`.
    media_tensors: A collection of media tensors derived from `input_data`.
    rf_tensors: A collection of Reach & Frequency (RF) media tensors.
    total_spend: A tensor containing total spend, including
      `media_tensors.media_spend` and `rf_tensors.rf_spend`.
    controls_transformer: A `ControlsTransformer` to scale controls tensors
      using the model's controls data.
    kpi_transformer: A `KpiTransformer` to scale KPI tensors using the model's
      KPI data.
    controls_scaled: The controls tensor normalized by population and by the
      median value.
    kpi_scaled: The KPI tensor normalized by population and by the median value.
    media_effects_dist: A string to specify the distribution of media random
      effects across geos.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo.
    prior_broadcast: A `PriorDistribution` object containing broadcasted
      distributions.
    baseline_geo_idx: The index of the baseline geo.
    holdout_id: A tensor containing the holdout id, if present.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec | None = None,
  ):
    self._input_data = input_data
    self._model_spec = model_spec if model_spec else spec.ModelSpec()
    self._inference_data = az.InferenceData()

    self._validate_data_dependent_model_spec()

    if self.is_national:
      _warn_setting_national_args(
          media_effects_dist=self.model_spec.media_effects_dist,
          unique_sigma_for_each_geo=self.model_spec.unique_sigma_for_each_geo,
      )

    self._validate_custom_priors()
    self._validate_geo_invariants()

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @property
  def inference_data(self) -> az.InferenceData:
    return self._inference_data

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.build_media_tensors(self.input_data, self.model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.build_rf_tensors(self.input_data, self.model_spec)

  @functools.cached_property
  def kpi(self) -> tf.Tensor:
    return tf.convert_to_tensor(self.input_data.kpi, dtype=tf.float32)

  @functools.cached_property
  def revenue_per_kpi(self) -> tf.Tensor | None:
    if self.input_data.revenue_per_kpi is None:
      return None
    return tf.convert_to_tensor(
        self.input_data.revenue_per_kpi, dtype=tf.float32
    )

  @functools.cached_property
  def controls(self) -> tf.Tensor:
    return tf.convert_to_tensor(self.input_data.controls, dtype=tf.float32)

  @functools.cached_property
  def population(self) -> tf.Tensor:
    return tf.convert_to_tensor(self.input_data.population, dtype=tf.float32)

  @functools.cached_property
  def total_spend(self) -> tf.Tensor:
    return tf.convert_to_tensor(
        self.input_data.get_total_spend(), dtype=tf.float32
    )

  @property
  def n_geos(self) -> int:
    return len(self.input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self.input_data.media_channel is None:
      return 0
    return len(self.input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self.input_data.rf_channel is None:
      return 0
    return len(self.input_data.rf_channel)

  @property
  def n_controls(self) -> int:
    return len(self.input_data.control_variable)

  @property
  def n_times(self) -> int:
    return len(self.input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self.input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self.model_spec.knots,
        is_national=self.is_national,
    )

  @functools.cached_property
  def controls_transformer(self) -> transformers.ControlsTransformer:
    if self.model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = tf.convert_to_tensor(
          self.model_spec.control_population_scaling_id, dtype=bool
      )
    else:
      controls_population_scaling_id = None

    return transformers.ControlsTransformer(
        controls=self.controls,
        population=self.population,
        population_scaling_id=controls_population_scaling_id,
    )

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return transformers.KpiTransformer(self.kpi, self.population)

  @functools.cached_property
  def controls_scaled(self) -> tf.Tensor:
    return self.controls_transformer.forward(self.controls)

  @functools.cached_property
  def kpi_scaled(self) -> tf.Tensor:
    return self.kpi_transformer.forward(self.kpi)

  @functools.cached_property
  def media_effects_dist(self) -> str:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[constants.MEDIA_EFFECTS_DIST]
    else:
      return self.model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self.model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self.model_spec.baseline_geo, int):
      if (
          self.model_spec.baseline_geo < 0
          or self.model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self.model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self.model_spec.baseline_geo
    elif isinstance(self.model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self.input_data.geo == self.model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self.model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return tf.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> tf.Tensor | None:
    if self.model_spec.holdout_id is None:
      return None
    tensor = tf.convert_to_tensor(self.model_spec.holdout_id, dtype=bool)
    return tensor[tf.newaxis, ...] if self.is_national else tensor

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    sigma_shape = (
        len(self.input_data.geo) if self.unique_sigma_for_each_geo else 1
    )

    return self.model_spec.prior.broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_controls=self.n_controls,
        sigma_shape=sigma_shape,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
    )

  def _validate_data_dependent_model_spec(self):
    """Validates that the data dependent model specs have correct shapes."""

    if (
        self.model_spec.roi_calibration_period is not None
        and self.model_spec.roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_media_channels,
        )
    ):
      raise ValueError(
          "The shape of `roi_calibration_period`"
          f" {self.model_spec.roi_calibration_period.shape} is different from"
          f" `(n_media_times, n_media_channels) = ({self.n_media_times},"
          f" {self.n_media_channels})`."
      )

    if (
        self.model_spec.rf_roi_calibration_period is not None
        and self.model_spec.rf_roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_rf_channels,
        )
    ):
      raise ValueError(
          "The shape of `rf_roi_calibration_period`"
          f" {self.model_spec.rf_roi_calibration_period.shape} is different"
          f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
          f" {self.n_rf_channels})`."
      )

    if self.model_spec.holdout_id is not None:
      if self.is_national and (
          self.model_spec.holdout_id.shape != (self.n_times,)
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self.model_spec.holdout_id.shape} is"
            f" different from `(n_times,) = ({self.n_times},)`."
        )
      elif not self.is_national and (
          self.model_spec.holdout_id.shape
          != (
              self.n_geos,
              self.n_times,
          )
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self.model_spec.holdout_id.shape} is"
            f" different from `(n_geos, n_times) = ({self.n_geos},"
            f" {self.n_times})`."
        )

    if self.model_spec.control_population_scaling_id is not None and (
        self.model_spec.control_population_scaling_id.shape
        != (self.n_controls,)
    ):
      raise ValueError(
          "The shape of `control_population_scaling_id`"
          f" {self.model_spec.control_population_scaling_id.shape} is different"
          f" from `(n_controls,) = ({self.n_controls},)`."
      )

  def _validate_custom_priors(self):
    """Validates custom priors invariants."""
    # Check that custom priors were not set, by confirming that model_spec's
    # `roi_m` and `roi_rf` properties equal the default `roi_m` and `roi_rf`
    # properties. If the properties are equal then custom priors were not set.
    default_model_spec = spec.ModelSpec()
    default_roi_m = default_model_spec.prior.roi_m
    default_roi_rf = default_model_spec.prior.roi_rf

    # Check `roi_m` properties match default `roi_m` properties (that the
    # distributions (e.g. Normal, LogNormal) and parameters (loc and scale) are
    # equal).
    roi_m_properties_equal = (
        isinstance(self.model_spec.prior.roi_m, type(default_roi_m))
        and self.model_spec.prior.roi_m.parameters == default_roi_m.parameters
    )
    # Check `roi_rf` properties match default `roi_rf` properties.
    roi_rf_properties_equal = (
        isinstance(self.model_spec.prior.roi_rf, type(default_roi_rf))
        and self.model_spec.prior.roi_rf.parameters == default_roi_rf.parameters
    )

    if (
        self.input_data.revenue_per_kpi is None
        and self.input_data.kpi_type == constants.NON_REVENUE
        and roi_m_properties_equal
        and roi_rf_properties_equal
    ):
      raise ValueError(
          "Custom priors should be set during model creation since"
          " `kpi_type` = `non_revenue` and `revenue_per_kpi` was not passed in."
          " Further documentation is available at"
          " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi"
      )

  def _validate_geo_invariants(self):
    """Validates non-national model invariants."""
    if self.is_national:
      return

    self._check_if_no_geo_variation(
        self.controls_scaled,
        "controls",
        self.input_data.controls.coords[constants.CONTROL_VARIABLE].values,
    )
    if self.input_data.media is not None:
      self._check_if_no_geo_variation(
          self.media_tensors.media_scaled,
          "media",
          self.input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self.input_data.reach is not None:
      self._check_if_no_geo_variation(
          self.rf_tensors.reach_scaled,
          "reach",
          self.input_data.reach.coords[constants.RF_CHANNEL].values,
      )

  def _check_if_no_geo_variation(
      self,
      scaled_data: tf.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if `n_knots == n_time` and data lacks geo variation."""

    _, col_idx_full = np.where(np.std(scaled_data, axis=0) < epsilon)
    col_idx_unique, counts = np.unique(col_idx_full, return_counts=True)
    # We use the shape of scaled_data (instead of `n_time`) because the data may
    # be padded to account for lagged effects.
    data_n_time = scaled_data.shape[1]
    col_idx_bad = col_idx_unique[np.where(counts == data_n_time)[0]]
    dims_bad = [data_dims[i] for i in col_idx_bad]

    if col_idx_bad.shape[0] and self.knot_info.n_knots == self.n_times:
      raise ValueError(
          f"The following {data_name} variables do not vary across geos, making"
          f" a model with n_knots=n_time unidentifiable: {dims_bad}. This can"
          " lead to poor model convergence. Since these variables only vary"
          " across time and not across geo, they are collinear with time and"
          " redundant in a model with a parameter for each time period.  To"
          " address this, you can either: (1) decrease the number of knots"
          " (n_knots < n_time), or (2) drop the listed variables that do not"
          " vary across geos."
      )

  def adstock_hill_media(
      self,
      media: tf.Tensor,  # pylint: disable=redefined-outer-name
      alpha: tf.Tensor,
      ec: tf.Tensor,
      slope: tf.Tensor,
  ) -> tf.Tensor:
    """Transforms media using Adstock and Hill functions in the desired order.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_media_channels)`
        containing non-negative media execution values. Typically this is
        impressions, but it can be any metric, such as `media_spend`. Clicks are
        often used for paid search ads.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_media_channels]`
      representing Adstock and Hill-transformed media.
    """
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=self.n_times,
    )
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    transformers_list = (
        [hill_transformer, adstock_transformer]
        if self.model_spec.hill_before_adstock
        else [adstock_transformer, hill_transformer]
    )

    media_out = media
    for transformer in transformers_list:
      media_out = transformer.forward(media_out)
    return media_out

  def adstock_hill_rf(
      self,
      reach: tf.Tensor,
      frequency: tf.Tensor,
      alpha: tf.Tensor,
      ec: tf.Tensor,
      slope: tf.Tensor,
  ) -> tf.Tensor:
    """Transforms reach and frequency (RF) using Hill and Adstock functions.

    Args:
      reach: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for reach.
      frequency: Tensor of dimensions `(n_geos, n_media_times, n_rf_channels)`
        containing non-negative media for frequency.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_rf_channels]`
      representing Hill and Adstock-transformed RF.
    """
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=self.n_times,
    )
    adj_frequency = hill_transformer.forward(frequency)
    rf_out = adstock_transformer.forward(reach * adj_frequency)

    return rf_out

  def _get_roi_prior_beta_m_value(
      self,
      alpha_m: tf.Tensor,
      beta_gm_dev: tf.Tensor,
      ec_m: tf.Tensor,
      eta_m: tf.Tensor,
      roi_m: tf.Tensor,
      slope_m: tf.Tensor,
      media_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_m`."""
    media_spend = self.media_tensors.media_spend
    media_spend_counterfactual = self.media_tensors.media_spend_counterfactual
    media_counterfactual_scaled = self.media_tensors.media_counterfactual_scaled
    # If we got here, then we should already have media tensors derived from
    # non-None InputData.media data.
    assert media_spend is not None
    assert media_spend_counterfactual is not None
    assert media_counterfactual_scaled is not None

    inc_revenue_m = roi_m * tf.reduce_sum(
        media_spend - media_spend_counterfactual,
        range(media_spend.ndim - 1),
    )
    if self.model_spec.roi_calibration_period is not None:
      media_counterfactual_transformed = self.adstock_hill_media(
          media=media_counterfactual_scaled,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
      )
    else:
      media_counterfactual_transformed = tf.zeros_like(media_transformed)
    revenue_per_kpi = self.revenue_per_kpi
    if self.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([self.n_geos, self.n_times], dtype=tf.float32)
    media_contrib_gm = tf.einsum(
        "...gtm,g,,gt->...gm",
        media_transformed - media_counterfactual_transformed,
        self.population,
        self.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )

    if self.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
      media_contrib_m = tf.einsum("...gm->...m", media_contrib_gm)
      random_effect_m = tf.einsum(
          "...m,...gm,...gm->...m", eta_m, beta_gm_dev, media_contrib_gm
      )
      return (inc_revenue_m - random_effect_m) / media_contrib_m
    else:
      # For log_normal, beta_m and eta_m are not mean & std.
      # The parameterization is beta_gm ~ exp(beta_m + eta_m * N(0, 1)).
      random_effect_m = tf.einsum(
          "...gm,...gm->...m",
          tf.math.exp(beta_gm_dev * eta_m[..., tf.newaxis, :]),
          media_contrib_gm,
      )
    return tf.math.log(inc_revenue_m) - tf.math.log(random_effect_m)

  def _get_roi_prior_beta_rf_value(
      self,
      alpha_rf: tf.Tensor,
      beta_grf_dev: tf.Tensor,
      ec_rf: tf.Tensor,
      eta_rf: tf.Tensor,
      roi_rf: tf.Tensor,
      slope_rf: tf.Tensor,
      rf_transformed: tf.Tensor,
  ) -> tf.Tensor:
    """Returns a tensor to be used in `beta_rf`."""
    rf_spend = self.rf_tensors.rf_spend
    rf_spend_counterfactual = self.rf_tensors.rf_spend_counterfactual
    reach_counterfactual_scaled = self.rf_tensors.reach_counterfactual_scaled
    frequency = self.rf_tensors.frequency
    # If we got here, then we should already have RF media tensors derived from
    # non-None InputData.reach data.
    assert rf_spend is not None
    assert rf_spend_counterfactual is not None
    assert reach_counterfactual_scaled is not None
    assert frequency is not None

    inc_revenue_rf = roi_rf * tf.reduce_sum(
        rf_spend - rf_spend_counterfactual,
        range(rf_spend.ndim - 1),
    )
    if self.model_spec.rf_roi_calibration_period is not None:
      rf_counterfactual_transformed = self.adstock_hill_rf(
          reach=reach_counterfactual_scaled,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
      )
    else:
      rf_counterfactual_transformed = tf.zeros_like(rf_transformed)
    revenue_per_kpi = self.revenue_per_kpi
    if self.input_data.revenue_per_kpi is None:
      revenue_per_kpi = tf.ones([self.n_geos, self.n_times], dtype=tf.float32)

    media_contrib_grf = tf.einsum(
        "...gtm,g,,gt->...gm",
        rf_transformed - rf_counterfactual_transformed,
        self.population,
        self.kpi_transformer.population_scaled_stdev,
        revenue_per_kpi,
    )
    if self.media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
      media_contrib_rf = tf.einsum("...gm->...m", media_contrib_grf)
      random_effect_rf = tf.einsum(
          "...m,...gm,...gm->...m", eta_rf, beta_grf_dev, media_contrib_grf
      )
      return (inc_revenue_rf - random_effect_rf) / media_contrib_rf
    else:
      # For log_normal, beta_rf and eta_rf are not mean & std.
      # The parameterization is beta_grf ~ exp(beta_rf + eta_rf * N(0, 1)).
      random_effect_rf = tf.einsum(
          "...gm,...gm->...m",
          tf.math.exp(beta_grf_dev * eta_rf[..., tf.newaxis, :]),
          media_contrib_grf,
      )
      return tf.math.log(inc_revenue_rf) - tf.math.log(random_effect_rf)

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, so we want to freeze the object's
    states before the computation graph is created.
    """
    cls = self.__class__
    # "Freeze" all @cached_property attributes by simply accessing them (with
    # `getattr()`).
    cached_properties = [
        attr
        for attr in dir(self)
        if isinstance(getattr(cls, attr, cls), functools.cached_property)
    ]
    for attr in cached_properties:
      _ = getattr(self, attr)

  def _get_joint_dist_unpinned(self) -> tfp.distributions.Distribution:
    """Returns JointDistributionCoroutineAutoBatched function for MCMC."""

    self.populate_cached_properties()

    # This lists all the derived properties and states of this Meridian object
    # that are referenced by the joint distribution coroutine.
    # That is, these are the list of captured parameters.
    prior_broadcast = self.prior_broadcast
    baseline_geo_idx = self.baseline_geo_idx
    knot_info = self.knot_info
    n_geos = self.n_geos
    n_times = self.n_times
    n_media_channels = self.n_media_channels
    n_rf_channels = self.n_rf_channels
    n_controls = self.n_controls
    holdout_id = self.holdout_id
    media_tensors = self.media_tensors
    rf_tensors = self.rf_tensors
    controls_scaled = self.controls_scaled
    media_effects_dist = self.media_effects_dist
    model_spec = self.model_spec
    adstock_hill_media_fn = self.adstock_hill_media
    adstock_hill_rf_fn = self.adstock_hill_rf
    get_roi_prior_beta_m_value_fn = self._get_roi_prior_beta_m_value
    get_roi_prior_beta_rf_value_fn = self._get_roi_prior_beta_rf_value

    # TODO(b/349416835): Extract this coroutine to be unittestable on its own.
    # This MCMC sampling technique is complex enough to have its own abstraction
    # and testable API, rather than being embedded as a private method in the
    # Meridian class.
    @tfp.distributions.JointDistributionCoroutineAutoBatched
    def joint_dist_unpinned():
      # Sample directly from prior.
      knot_values = yield prior_broadcast.knot_values
      gamma_c = yield prior_broadcast.gamma_c
      xi_c = yield prior_broadcast.xi_c
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
        roi_m = yield prior_broadcast.roi_m
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
        if model_spec.use_roi_prior:
          beta_m_value = get_roi_prior_beta_m_value_fn(
              alpha_m,
              beta_gm_dev,
              ec_m,
              eta_m,
              roi_m,
              slope_m,
              media_transformed,
          )
          beta_m = yield tfp.distributions.Deterministic(
              beta_m_value, name=constants.BETA_M
          )
        else:
          beta_m = yield prior_broadcast.beta_m

        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
          beta_gm_value = beta_m + eta_m * beta_gm_dev
        else:
          # MEDIA_EFFECTS_LOG_NORMAL
          beta_gm_value = tf.math.exp(beta_m + eta_m * beta_gm_dev)

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
        roi_rf = yield prior_broadcast.roi_rf
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

        if model_spec.use_roi_prior:
          beta_rf_value = get_roi_prior_beta_rf_value_fn(
              alpha_rf,
              beta_grf_dev,
              ec_rf,
              eta_rf,
              roi_rf,
              slope_rf,
              rf_transformed,
          )
          beta_rf = yield tfp.distributions.Deterministic(
              beta_rf_value,
              name=constants.BETA_RF,
          )
        else:
          beta_rf = yield prior_broadcast.beta_rf

        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
          beta_grf_value = beta_rf + eta_rf * beta_grf_dev
        else:
          # MEDIA_EFFECTS_LOG_NORMAL
          beta_grf_value = tf.math.exp(beta_rf + eta_rf * beta_grf_dev)

        beta_grf = yield tfp.distributions.Deterministic(
            beta_grf_value, name=constants.BETA_GRF
        )
        combined_media_transformed = tf.concat(
            [combined_media_transformed, rf_transformed], axis=-1
        )
        combined_beta = tf.concat([combined_beta, beta_grf], axis=-1)

      sigma_gt = tf.transpose(tf.broadcast_to(sigma, [n_times, n_geos]))
      gamma_gc_dev = yield tfp.distributions.Sample(
          tfp.distributions.Normal(0, 1),
          [n_geos, n_controls],
          name=constants.GAMMA_GC_DEV,
      )
      gamma_gc = yield tfp.distributions.Deterministic(
          gamma_c + xi_c * gamma_gc_dev, name=constants.GAMMA_GC
      )
      y_pred = (
          tau_gt
          + tf.einsum("gtm,gm->gt", combined_media_transformed, combined_beta)
          + tf.einsum("gtc,gc->gt", controls_scaled, gamma_gc)
      )
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
    y = (
        tf.where(self.holdout_id, 0.0, self.kpi_scaled)
        if self.holdout_id is not None
        else self.kpi_scaled
    )
    return self._get_joint_dist_unpinned().experimental_pin(y=y)

  def _create_inference_data_coords(
      self, n_chains: int, n_draws: int
  ) -> Mapping[str, np.ndarray | Sequence[str]]:
    """Creates data coordinates for inference data."""
    media_channel_values = (
        self.input_data.media_channel
        if self.input_data.media_channel is not None
        else np.array([])
    )
    rf_channel_values = (
        self.input_data.rf_channel
        if self.input_data.rf_channel is not None
        else np.array([])
    )
    return {
        constants.CHAIN: np.arange(n_chains),
        constants.DRAW: np.arange(n_draws),
        constants.GEO: self.input_data.geo,
        constants.TIME: self.input_data.time,
        constants.MEDIA_TIME: self.input_data.media_time,
        constants.KNOTS: np.arange(self.knot_info.n_knots),
        constants.CONTROL_VARIABLE: self.input_data.control_variable,
        constants.MEDIA_CHANNEL: media_channel_values,
        constants.RF_CHANNEL: rf_channel_values,
    }

  def _create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    inference_dims = dict(constants.INFERENCE_DIMS)
    if self.unique_sigma_for_each_geo:
      inference_dims[constants.SIGMA] = [constants.GEO]
    else:
      inference_dims[constants.SIGMA] = [constants.SIGMA_DIM]

    return {
        param: [constants.CHAIN, constants.DRAW] + list(dims)
        for param, dims in inference_dims.items()
    }

  def _sample_media_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of the media variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of media parameter names to a tensor of shape [n_draws, n_geos,
      n_media_channels] or [n_draws, n_media_channels] containing the
      samples.
    """
    prior = self.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    media_vars = {
        constants.ALPHA_M: prior.alpha_m.sample(**sample_kwargs),
        constants.EC_M: prior.ec_m.sample(**sample_kwargs),
        constants.ETA_M: prior.eta_m.sample(**sample_kwargs),
        constants.ROI_M: prior.roi_m.sample(**sample_kwargs),
        constants.SLOPE_M: prior.slope_m.sample(**sample_kwargs),
    }
    beta_gm_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [self.n_geos, self.n_media_channels],
        name=constants.BETA_GM_DEV,
    ).sample(**sample_kwargs)
    media_transformed = self.adstock_hill_media(
        media=self.media_tensors.media_scaled,
        alpha=media_vars[constants.ALPHA_M],
        ec=media_vars[constants.EC_M],
        slope=media_vars[constants.SLOPE_M],
    )

    if self.model_spec.use_roi_prior:
      beta_m_value = self._get_roi_prior_beta_m_value(
          beta_gm_dev=beta_gm_dev,
          media_transformed=media_transformed,
          **media_vars,
      )
      media_vars[constants.BETA_M] = tfp.distributions.Deterministic(
          beta_m_value, name=constants.BETA_M
      ).sample()
    else:
      media_vars[constants.BETA_M] = prior.beta_m.sample(**sample_kwargs)

    beta_gm_value = (
        media_vars[constants.BETA_M][..., tf.newaxis, :]
        + media_vars[constants.ETA_M][..., tf.newaxis, :] * beta_gm_dev
    )
    if self.media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL:
      beta_gm_value = tf.math.exp(beta_gm_value)
    media_vars[constants.BETA_GM] = tfp.distributions.Deterministic(
        beta_gm_value, name=constants.BETA_GM
    ).sample()

    return media_vars

  def _sample_rf_priors(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Draws samples from the prior distributions of the RF variables.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).

    Returns:
      A mapping of RF parameter names to a tensor of shape [n_draws, n_geos,
      n_rf_channels] or [n_draws, n_rf_channels] containing the samples.
    """
    prior = self.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}
    rf_vars = {
        constants.ALPHA_RF: prior.alpha_rf.sample(**sample_kwargs),
        constants.EC_RF: prior.ec_rf.sample(**sample_kwargs),
        constants.ETA_RF: prior.eta_rf.sample(**sample_kwargs),
        constants.ROI_RF: prior.roi_rf.sample(**sample_kwargs),
        constants.SLOPE_RF: prior.slope_rf.sample(**sample_kwargs),
    }
    beta_grf_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [self.n_geos, self.n_rf_channels],
        name=constants.BETA_GRF_DEV,
    ).sample(**sample_kwargs)
    rf_transformed = self.adstock_hill_rf(
        reach=self.rf_tensors.reach_scaled,
        frequency=self.rf_tensors.frequency,
        alpha=rf_vars[constants.ALPHA_RF],
        ec=rf_vars[constants.EC_RF],
        slope=rf_vars[constants.SLOPE_RF],
    )

    if self.model_spec.use_roi_prior:
      beta_rf_value = self._get_roi_prior_beta_rf_value(
          beta_grf_dev=beta_grf_dev,
          rf_transformed=rf_transformed,
          **rf_vars,
      )
      rf_vars[constants.BETA_RF] = tfp.distributions.Deterministic(
          beta_rf_value,
          name=constants.BETA_RF,
      ).sample()
    else:
      rf_vars[constants.BETA_RF] = prior.beta_rf.sample(**sample_kwargs)

    beta_grf_value = (
        rf_vars[constants.BETA_RF][..., tf.newaxis, :]
        + rf_vars[constants.ETA_RF][..., tf.newaxis, :] * beta_grf_dev
    )
    if self.media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL:
      beta_grf_value = tf.math.exp(beta_grf_value)
    rf_vars[constants.BETA_GRF] = tfp.distributions.Deterministic(
        beta_grf_value, name=constants.BETA_GRF
    ).sample()

    return rf_vars

  def _sample_prior_fn(
      self,
      n_draws: int,
      seed: int | None = None,
  ) -> Mapping[str, tf.Tensor]:
    """Returns a mapping of prior parameters to tensors of the samples."""
    # For stateful sampling, the random seed must be set to ensure that any
    # random numbers that are generated are deterministic.
    if seed is not None:
      tf.keras.utils.set_random_seed(1)
    prior = self.prior_broadcast
    sample_shape = [1, n_draws]
    sample_kwargs = {constants.SAMPLE_SHAPE: sample_shape, constants.SEED: seed}

    tau_g_excl_baseline = prior.tau_g_excl_baseline.sample(**sample_kwargs)
    base_vars = {
        constants.KNOT_VALUES: prior.knot_values.sample(**sample_kwargs),
        constants.GAMMA_C: prior.gamma_c.sample(**sample_kwargs),
        constants.XI_C: prior.xi_c.sample(**sample_kwargs),
        constants.SIGMA: prior.sigma.sample(**sample_kwargs),
        constants.TAU_G: _get_tau_g(
            tau_g_excl_baseline=tau_g_excl_baseline,
            baseline_geo_idx=self.baseline_geo_idx,
        ).sample(),
    }
    base_vars[constants.MU_T] = tfp.distributions.Deterministic(
        tf.einsum(
            "...k,kt->...t",
            base_vars[constants.KNOT_VALUES],
            tf.convert_to_tensor(self.knot_info.weights),
        ),
        name=constants.MU_T,
    ).sample()

    gamma_gc_dev = tfp.distributions.Sample(
        tfp.distributions.Normal(0, 1),
        [self.n_geos, self.n_controls],
        name=constants.GAMMA_GC_DEV,
    ).sample(**sample_kwargs)
    base_vars[constants.GAMMA_GC] = tfp.distributions.Deterministic(
        base_vars[constants.GAMMA_C][..., tf.newaxis, :]
        + base_vars[constants.XI_C][..., tf.newaxis, :] * gamma_gc_dev,
        name=constants.GAMMA_GC,
    ).sample()

    media_vars = (
        self._sample_media_priors(n_draws, seed)
        if self.media_tensors.media is not None
        else {}
    )
    rf_vars = (
        self._sample_rf_priors(n_draws, seed)
        if self.rf_tensors.reach is not None
        else {}
    )

    return base_vars | media_vars | rf_vars

  def sample_prior(self, n_draws: int, seed: int | None = None):
    """Draws samples from the prior distributions.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_draws = self._sample_prior_fn(n_draws, seed=seed)
    # Create Arviz InferenceData for prior draws.
    prior_coords = self._create_inference_data_coords(1, n_draws)
    prior_dims = self._create_inference_data_dims()
    prior_inference_data = az.convert_to_inference_data(
        prior_draws, coords=prior_coords, dims=prior_dims, group=constants.PRIOR
    )
    self.inference_data.extend(prior_inference_data, join="right")

  def sample_posterior(
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
      seed: Sequence[int] | None = None,
      **pins,
  ):
    """Runs Markov Chain Monte Carlo (MCMC) sampling of posterior distributions.

    For more details about the arguments, see [`windowed_adaptive_nuts`]
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
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially (see
        https://developers.google.com/meridian/docs/advanced-modeling/additional-considerations#gpu-oom-error).
    """
    seed = tfp.random.sanitize_seed(seed) if seed else None
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
            " https://developers.google.com/meridian/docs/advanced-modeling/additional-considerations#gpu-oom-error)"
        ) from error
      states.append(mcmc.all_states._asdict())
      traces.append(mcmc.trace)

    mcmc_states = {
        k: tf.einsum(
            "ij...->ji...",
            tf.concat([state[k] for state in states], axis=1)[n_burnin:, ...],
        )
        for k in states[0].keys()
        if k not in constants.IGNORED_PRIOR_PARAMETERS
    }
    # Create Arviz InferenceData for posterior draws.
    posterior_coords = self._create_inference_data_coords(total_chains, n_keep)
    posterior_dims = self._create_inference_data_dims()
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
    self.inference_data.extend(posterior_inference_data, join="right")


def save_mmm(mmm: Meridian, file_path: str):
  """Save the model object to a `pickle` file path.

  WARNING: There is no guarantee for future compaibility of the binary file
  output of this function. It is recommended to use `load_mmm()` with the same
  version of the library that was used to save the model.

  Args:
    mmm: Model object to save.
    file_path: File path to save a pickled model object.
  """
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))

  with open(file_path, "wb") as f:
    joblib.dump(mmm, f)


def load_mmm(file_path: str) -> Meridian:
  """Load the model object from a `pickle` file path.

  WARNING: There is no guarantee for backward compatibility of the binary file
  input of this function. It is recommended to use `load_mmm()` with the same
  version of the library that was used to save the model's pickleed file.

  Args:
    file_path: File path to load a pickled model object from.

  Returns:
    mmm: Model object loaded from the file path.

  Raises:
      FileNotFoundError: If `file_path` does not exist.
  """
  try:
    with open(file_path, "rb") as f:
      mmm = joblib.load(f)
    return mmm
  except FileNotFoundError:
    raise FileNotFoundError(f"No such file or directory: {file_path}") from None
