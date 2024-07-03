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

"""Defines types and containers for Meridian model data internals."""

from collections.abc import Sequence
import functools
import warnings
from meridian import constants
from meridian.data import input_data as data
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers
import numpy as np
import tensorflow as tf


__all__ = [
    "ModelData",
]


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


class ModelData:
  """An immutable set of attributes derived from input data and model spec.

  Attributes:
    input_data: An `InputData` object containing the input data for the model.
    model_spec: A `ModelSpec` object containing the model specification.
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
      model_spec: spec.ModelSpec,
  ):
    self._input_data = input_data
    self._model_spec = model_spec

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

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    data object as part of a captured closure. Within the computation graph,
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
