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
from meridian.data import time_coordinates as tc
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import posterior_sampler
from meridian.model import prior_distribution
from meridian.model import prior_sampler
from meridian.model import spec
from meridian.model import transformers
import numpy as np
from meridian import backend


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


MCMCSamplingError = posterior_sampler.MCMCSamplingError
MCMCOOMError = posterior_sampler.MCMCOOMError


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
    n_organic_media_channels: Number of organic media channels in the data.
    n_organic_rf_channels: Number of organic reach and frequency (RF) channels
      in the data.
    n_controls: Number of control variables in the data.
    n_non_media_channels: Number of non-media treatment channels in the data.
    n_times: Number of time periods in the KPI or spend data.
    n_media_times: Number of time periods in the media data.
    is_national: A boolean indicating whether the data is national (single geo)
      or not (multiple geos).
    knot_info: A `KnotInfo` derived from input data and model spec.
    kpi: A tensor constructed from `input_data.kpi`.
    revenue_per_kpi: A tensor constructed from `input_data.revenue_per_kpi`. If
      `input_data.revenue_per_kpi` is None, then this is also None.
    controls: A tensor constructed from `input_data.controls`.
    non_media_treatments: A tensor constructed from
      `input_data.non_media_treatments`.
    population: A tensor constructed from `input_data.population`.
    media_tensors: A collection of media tensors derived from `input_data`.
    rf_tensors: A collection of Reach & Frequency (RF) media tensors.
    organic_media_tensors: A collection of organic media tensors.
    organic_rf_tensors: A collection of organic Reach & Frequency (RF) media
      tensors.
    total_spend: A tensor containing total spend, including
      `media_tensors.media_spend` and `rf_tensors.rf_spend`.
    controls_transformer: A `ControlsTransformer` to scale controls tensors
      using the model's controls data.
    non_media_transformer: A `CenteringAndScalingTransformer` to scale non-media
      treatmenttensors using the model's non-media treatment data.
    kpi_transformer: A `KpiTransformer` to scale KPI tensors using the model's
      KPI data.
    controls_scaled: The controls tensor normalized by population and by the
      median value.
    non_media_treatments_scaled: The non-media treatment tensor normalized by
      population and by the median value.
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
      inference_data: (
          az.InferenceData | None
      ) = None,  # for deserializer use only
  ):
    self._input_data = input_data
    self._model_spec = model_spec if model_spec else spec.ModelSpec()
    self._inference_data = (
        inference_data if inference_data else az.InferenceData()
    )

    self._validate_data_dependent_model_spec()
    self._validate_injected_inference_data()

    if self.is_national:
      _warn_setting_national_args(
          media_effects_dist=self.model_spec.media_effects_dist,
          unique_sigma_for_each_geo=self.model_spec.unique_sigma_for_each_geo,
      )
    self._warn_setting_ignored_priors()
    self._validate_paid_media_prior_type()
    self._validate_geo_invariants()
    self._validate_time_invariants()
    self._validate_kpi_transformer()

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
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self.input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self.input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.convert_to_tensor(self.input_data.kpi, dtype=backend.float32)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self.input_data.revenue_per_kpi is None:
      return None
    return backend.convert_to_tensor(
        self.input_data.revenue_per_kpi, dtype=backend.float32
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor:
    return backend.convert_to_tensor(self.input_data.controls, dtype=backend.float32)

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self.input_data.non_media_treatments is None:
      return None
    return backend.convert_to_tensor(
        self.input_data.non_media_treatments, dtype=backend.float32
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.convert_to_tensor(self.input_data.population, dtype=backend.float32)

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.convert_to_tensor(
        self.input_data.get_total_spend(), dtype=backend.float32
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
  def n_organic_media_channels(self) -> int:
    if self.input_data.organic_media_channel is None:
      return 0
    return len(self.input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self.input_data.organic_rf_channel is None:
      return 0
    return len(self.input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    return len(self.input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self.input_data.non_media_channel is None:
      return 0
    return len(self.input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self.input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self.input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @property
  def _sigma_shape(self) -> int:
    return len(self.input_data.geo) if self.unique_sigma_for_each_geo else 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self.model_spec.knots,
        is_national=self.is_national,
    )

  @functools.cached_property
  def controls_transformer(self) -> transformers.CenteringAndScalingTransformer:
    if self.model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = backend.convert_to_tensor(
          self.model_spec.control_population_scaling_id, dtype=backend.bool
      )
    else:
      controls_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.controls,
        population=self.population,
        population_scaling_id=controls_population_scaling_id,
    )

  @functools.cached_property
  def non_media_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for non-media treatments."""
    if self.non_media_treatments is None:
      return None
    if self.model_spec.non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.convert_to_tensor(
          self.model_spec.non_media_population_scaling_id, dtype=backend.bool
      )
    else:
      non_media_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.non_media_treatments,
        population=self.population,
        population_scaling_id=non_media_population_scaling_id,
    )

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return transformers.KpiTransformer(self.kpi, self.population)

  @functools.cached_property
  def controls_scaled(self) -> backend.Tensor:
    return self.controls_transformer.forward(self.controls)

  @functools.cached_property
  def non_media_treatments_scaled(self) -> backend.Tensor | None:
    if self.non_media_transformer is not None:
      return self.non_media_transformer.forward(self.non_media_treatments)  # pytype: disable=attribute-error
      return None

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
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
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    if self.model_spec.holdout_id is None:
      return None
    tensor = backend.convert_to_tensor(self.model_spec.holdout_id, dtype=backend.bool)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    set_total_media_contribution_prior = (
        self.input_data.revenue_per_kpi is None
        and self.input_data.kpi_type == constants.NON_REVENUE
        and self.model_spec.paid_media_prior_type
        == constants.PAID_MEDIA_PRIOR_TYPE_ROI
    )
    total_spend = self.input_data.get_total_spend()
    # Total spend can have 1, 2 or 3 dimensions. Aggregate by channel.
    if len(total_spend.shape) == 1:
      # Already aggregated by channel.
      agg_total_spend = total_spend
    elif len(total_spend.shape) == 2:
      agg_total_spend = np.sum(total_spend, axis=(0,))
    else:
      agg_total_spend = np.sum(total_spend, axis=(0, 1))

    return self.model_spec.prior.broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_organic_media_channels=self.n_organic_media_channels,
        n_organic_rf_channels=self.n_organic_rf_channels,
        n_controls=self.n_controls,
        n_non_media_channels=self.n_non_media_channels,
        sigma_shape=self._sigma_shape,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=set_total_media_contribution_prior,
        kpi=np.sum(self.input_data.kpi.values),
        total_spend=agg_total_spend,
        media_effects_dist=self.media_effects_dist,
    )

  @functools.cached_property
  def prior_sampler_callable(self) -> prior_sampler.PriorDistributionSampler:
    """A `PriorDistributionSampler` callable bound to this model."""
    return prior_sampler.PriorDistributionSampler(self)

  @functools.cached_property
  def posterior_sampler_callable(
      self,
  ) -> posterior_sampler.PosteriorMCMCSampler:
    """A `PosteriorMCMCSampler` callable bound to this model."""
    return posterior_sampler.PosteriorMCMCSampler(self)

  def expand_selected_time_dims(
      self,
      start_date: tc.Date | None = None,
      end_date: tc.Date | None = None,
  ) -> list[str] | None:
    """Validates and returns time dimension values based on the selected times.

    If both `start_date` and `end_date` are None, returns None. If specified,
    both `start_date` and `end_date` are inclusive, and must be present in the
    time coordinates of the input data.

    Args:
      start_date: Start date of the selected time period. If None, implies the
        earliest time dimension value in the input data.
      end_date: End date of the selected time period. If None, implies the
        latest time dimension value in the input data.

    Returns:
      A list of time dimension values (as Meridian-formatted strings) in the
      input data within the selected time period, or do nothing and pass through
      None if both arguments are Nones, or if `start_date` and `end_date`
      correspond to the entire time range in the input data.

    Raises:
      ValueError if `start_date` or `end_date` is not in the input data time
      dimensions.
    """
    expanded = self.input_data.time_coordinates.expand_selected_time_dims(
        start_date=start_date, end_date=end_date
    )
    if expanded is None:
      return None
    return [date.strftime(constants.DATE_FORMAT) for date in expanded]

  def _validate_injected_inference_data(self):
    """Validates that the injected inference data has correct shapes.

    Raises:
      ValueError: If the injected `InferenceData` has incorrect shapes.
    """
    if hasattr(self.inference_data, constants.PRIOR):
      self._validate_injected_inference_data_group(
          self.inference_data, constants.PRIOR
      )
    if hasattr(self.inference_data, constants.POSTERIOR):
      self._validate_injected_inference_data_group(
          self.inference_data, constants.POSTERIOR
      )

  def _validate_injected_inference_data_group_coord(
      self,
      inference_data: az.InferenceData,
      group: str,
      coord: str,
      expected_size: int,
  ):
    """Validates that the injected inference data group coordinate has the expected size.

    Args:
      inference_data: The injected `InferenceData` to be validated.
      group: The group of the coordinate to be validated.
      coord: The coordinate to be validated.
      expected_size: The expected size of the coordinate.

    Raises:
      ValueError: If the injected `InferenceData` has incorrect size for the
      coordinate.
    """

    injected_size = (
        inference_data[group].coords[coord].size
        if coord in inference_data[group].coords
        else 0
    )
    if injected_size != expected_size:
      raise ValueError(
          f"Injected inference data {group} has incorrect coordinate '{coord}':"
          f" expected {expected_size}, got {injected_size}"
      )

  def _validate_injected_inference_data_group(
      self,
      inference_data: az.InferenceData,
      group: str,
  ):
    """Validates that the injected inference data group has correct shapes.

    Args:
      inference_data: The injected `InferenceData` to be validated.
      group: The group of the coordinate to be validated.

    Raises:
      ValueError: If the injected `InferenceData` has incorrect shapes.
    """

    self._validate_injected_inference_data_group_coord(
        inference_data, group, constants.GEO, self.n_geos
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.CONTROL_VARIABLE,
        self.n_controls,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.KNOTS,
        self.knot_info.n_knots,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data, group, constants.TIME, self.n_times
    )
    if not self.model_spec.unique_sigma_for_each_geo:
      self._validate_injected_inference_data_group_coord(
          inference_data, group, constants.SIGMA_DIM, self._sigma_shape
      )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.MEDIA_CHANNEL,
        self.n_media_channels,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.RF_CHANNEL,
        self.n_rf_channels,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.ORGANIC_MEDIA_CHANNEL,
        self.n_organic_media_channels,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.ORGANIC_RF_CHANNEL,
        self.n_organic_rf_channels,
    )
    self._validate_injected_inference_data_group_coord(
        inference_data,
        group,
        constants.NON_MEDIA_CHANNEL,
        self.n_non_media_channels,
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

    if self.model_spec.non_media_population_scaling_id is not None and (
        self.model_spec.non_media_population_scaling_id.shape
        != (self.n_non_media_channels,)
    ):
      raise ValueError(
          "The shape of `non_media_population_scaling_id`"
          f" {self.model_spec.non_media_population_scaling_id.shape} is"
          " different from `(n_non_media_channels,) ="
          f" ({self.n_non_media_channels},)`."
      )

  def _warn_setting_ignored_priors(self):
    """Raises a warning if ignored priors are set."""
    default_distribution = prior_distribution.PriorDistribution()
    prior_type = self.model_spec.paid_media_prior_type

    ignored_custom_priors = []
    for prior in constants.IGNORED_PRIORS.get(prior_type, []):
      self_prior = getattr(self.model_spec.prior, prior)
      default_prior = getattr(default_distribution, prior)
      if not prior_distribution.distributions_are_equal(
          self_prior, default_prior
      ):
        ignored_custom_priors.append(prior)
    if ignored_custom_priors:
      ignored_priors_str = ", ".join(ignored_custom_priors)
      warnings.warn(
          f"Custom prior(s) `{ignored_priors_str}` are ignored when"
          " `paid_media_prior_type` is set to"
          f' "{prior_type}".'
      )

  def _validate_paid_media_prior_type(self):
    """Validates the media prior type."""
    default_distribution = prior_distribution.PriorDistribution()
    mroi_m_not_set = (
        self.n_media_channels > 0
        and prior_distribution.distributions_are_equal(
            self.model_spec.prior.mroi_m, default_distribution.mroi_m
        )
    )
    mroi_rf_not_set = (
        self.n_rf_channels > 0
        and prior_distribution.distributions_are_equal(
            self.model_spec.prior.mroi_rf, default_distribution.mroi_rf
        )
    )
    if (
        self.input_data.revenue_per_kpi is None
        and self.input_data.kpi_type == constants.NON_REVENUE
        and self.model_spec.paid_media_prior_type
        == constants.PAID_MEDIA_PRIOR_TYPE_MROI
        and (mroi_m_not_set or mroi_rf_not_set)
    ):
      raise ValueError(
          f"Custom priors should be set on `{constants.MROI_M}` and"
          f" `{constants.MROI_RF}` when KPI is non-revenue and revenue per kpi"
          " data is missing."
      )

  def _validate_geo_invariants(self):
    """Validates non-national model invariants."""
    if self.is_national:
      return

    self._check_if_no_geo_variation(
        self.controls_scaled,
        constants.CONTROLS,
        self.input_data.controls.coords[constants.CONTROL_VARIABLE].values,
    )
    if self.input_data.non_media_treatments is not None:
      self._check_if_no_geo_variation(
          self.non_media_treatments_scaled,
          constants.NON_MEDIA_TREATMENTS,
          self.input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self.input_data.media is not None:
      self._check_if_no_geo_variation(
          self.media_tensors.media_scaled,
          constants.MEDIA,
          self.input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self.input_data.reach is not None:
      self._check_if_no_geo_variation(
          self.rf_tensors.reach_scaled,
          constants.REACH,
          self.input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self.input_data.organic_media is not None:
      self._check_if_no_geo_variation(
          self.organic_media_tensors.organic_media_scaled,
          "organic_media",
          self.input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self.input_data.organic_reach is not None:
      self._check_if_no_geo_variation(
          self.organic_rf_tensors.organic_reach_scaled,
          "organic_reach",
          self.input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
      )

  def _check_if_no_geo_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if `n_knots == n_time` and data lacks geo variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.where(backend.reduce_std(scaled_data, axis=0) < epsilon)[
        :, 1
    ]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    # We use the shape of scaled_data (instead of `n_time`) because the data may
    # be padded to account for lagged effects.
    data_n_time = scaled_data.shape[1]
    mask = backend.equal(counts, data_n_time)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)

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

  def _validate_time_invariants(self):
    """Validates model time invariants."""

    self._check_if_no_time_variation(
        self.controls_scaled,
        constants.CONTROLS,
        self.input_data.controls.coords[constants.CONTROL_VARIABLE].values,
    )
    if self.input_data.media is not None:
      self._check_if_no_time_variation(
          self.media_tensors.media_scaled,
          constants.MEDIA,
          self.input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self.input_data.reach is not None:
      self._check_if_no_time_variation(
          self.rf_tensors.reach_scaled,
          constants.REACH,
          self.input_data.reach.coords[constants.RF_CHANNEL].values,
      )

  def _check_if_no_time_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if data lacks time variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.where(backend.reduce_std(scaled_data, axis=1) < epsilon)[
        :, 1
    ]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    mask = backend.equal(counts, self.n_geos)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)

    if col_idx_bad.shape[0] and not self.is_national:
      raise ValueError(
          f"The following {data_name} variables do not vary across time, making"
          f" a model with geo main effects unidentifiable: {dims_bad}. This can"
          " lead to poor model convergence. Since these variables only vary"
          " across geo and not across time, they are collinear with geo and"
          " redundant in a model with geo main effects. To address this, drop"
          " the listed variables that do not vary across time."
      )

  def _validate_kpi_transformer(self):
    """Validates the KPI transformer."""
    if (
        self.kpi_transformer.population_scaled_stdev == 0
        and self.model_spec.paid_media_prior_type
        in constants.PAID_MEDIA_ROI_PRIOR_TYPES
    ):
      kpi = "kpi" if self.is_national else "population_scaled_kpi"
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          f" {self.model_spec.paid_media_prior_type} prior type."
      )

  def adstock_hill_media(
      self,
      media: backend.Tensor,  # pylint: disable=redefined-outer-name
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    """Transforms media using Adstock and Hill functions in the desired order.

    Args:
      media: Tensor of dimensions `(n_geos, n_media_times, n_media_channels)`
        containing non-negative media execution values. Typically this is
        impressions, but it can be any metric, such as `media_spend`. Clicks are
        often used for paid search ads.
      alpha: Uniform distribution for Adstock and Hill calculations.
      ec: Shifted half-normal distribution for Adstock and Hill calculations.
      slope: Deterministic distribution for Adstock and Hill calculations.
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `media` equals
        `self.n_media_times`, in which case `n_times_output` defaults to
        `self.n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_media_channels]`
      representing Adstock and Hill-transformed media.
    """
    if n_times_output is None and (backend.shape(media)[1] == self.n_media_times):
      n_times_output = self.n_times
    elif n_times_output is None:
      raise ValueError(
          "n_times_output is required. This argument is only optional when "
          "`media` has a number of time periods equal to `self.n_media_times`."
      )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=n_times_output,
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
      reach: backend.Tensor,
      frequency: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
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
      n_times_output: Number of time periods to output. This argument is
        optional when the number of time periods in `reach` equals
        `self.n_media_times`, in which case `n_times_output` defaults to
        `self.n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_rf_channels]`
      representing Hill and Adstock-transformed RF.
    """
    if n_times_output is None and (backend.shape(reach)[1] == self.n_media_times):
      n_times_output = self.n_times
    elif n_times_output is None:
      raise ValueError(
          "n_times_output is required. This argument is only optional when "
          "`reach` has a number of time periods equal to `self.n_media_times`."
      )
    hill_transformer = adstock_hill.HillTransformer(
        ec=ec,
        slope=slope,
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=self.model_spec.max_lag,
        n_times_output=n_times_output,
    )
    adj_frequency = hill_transformer.forward(frequency)
    rf_out = adstock_transformer.forward(reach * adj_frequency)

    return rf_out

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `backend.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, and so this method freezes the
    object's states before the computation graph is created.
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

  def create_inference_data_coords(
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
    organic_media_channel_values = (
        self.input_data.organic_media_channel
        if self.input_data.organic_media_channel is not None
        else np.array([])
    )
    organic_rf_channel_values = (
        self.input_data.organic_rf_channel
        if self.input_data.organic_rf_channel is not None
        else np.array([])
    )
    non_media_channel_values = (
        self.input_data.non_media_channel
        if self.input_data.non_media_channel is not None
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
        constants.NON_MEDIA_CHANNEL: non_media_channel_values,
        constants.MEDIA_CHANNEL: media_channel_values,
        constants.RF_CHANNEL: rf_channel_values,
        constants.ORGANIC_MEDIA_CHANNEL: organic_media_channel_values,
        constants.ORGANIC_RF_CHANNEL: organic_rf_channel_values,
    }

  def create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    inference_dims = dict(constants.INFERENCE_DIMS)
    if self.unique_sigma_for_each_geo:
      inference_dims[constants.SIGMA] = [constants.GEO]
    else:
      inference_dims[constants.SIGMA] = [constants.SIGMA_DIM]

    return {
        param: [constants.CHAIN, constants.DRAW] + list(dims)
        for param, dims in inference_dims.items()
    }

  def sample_prior(self, n_draws: int, seed: int | None = None):
    """Draws samples from the prior distributions.

    Drawn samples are merged into this model's Arviz `inference_data` property.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_inference_data = self.prior_sampler_callable(n_draws, seed)
    self.inference_data.extend(prior_inference_data, join="right")

  def sample_posterior(
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
  ):
    """Runs Markov Chain Monte Carlo (MCMC) sampling of posterior distributions.

    For more information about the arguments, see [`windowed_adaptive_nuts`]
    (https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/windowed_adaptive_nuts).

    Drawn samples are merged into this model's Arviz `inference_data` property.

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
        backend.zeros(n_chains), 'b': backend.zeros([n_chains, n_features]), } ```  then
        any of `1.`, `{'a': 1., 'b': 1.}`, or `{'a': backend.ones(n_chains), 'b':
        backend.ones([n_chains, n_features])}` will work. Defaults to the dimension
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
        be a positive integer. For more information, see `backend.while_loop`. # Assuming while_loop is exposed
      seed: An `int32[2]` Tensor or a Python list or tuple of 2 `int`s, which
        will be treated as stateless seeds; or a Python `int` or `None`, which
        will be treated as stateful seeds. See [tfp.random.sanitize_seed]
        (https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed). # Note: tfp.random needs backend handling later
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially. For more
        information, see
        [backend.ResourceExhaustedError when running Meridian.sample_posterior] # Replaced tf.errors
        (https://developers.google.com/meridian/docs/advanced-modeling/model-debugging#gpu-oom-error).
    """
    posterior_inference_data = self.posterior_sampler_callable(
        n_chains,
        n_adapt,
        n_burnin,
        n_keep,
        current_state,
        init_step_size,
        dual_averaging_kwargs,
        max_tree_depth,
        max_energy_diff,
        unrolled_leapfrog_steps,
        parallel_iterations,
        seed,
        **pins,
    )
    self.inference_data.extend(posterior_inference_data, join="right")


def save_mmm(mmm: Meridian, file_path: str):
  """Save the model object to a `pickle` file path.

  WARNING: There is no guarantee for future compatibility of the binary file
  output of this function. We recommend using `load_mmm()` with the same
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
  input of this function. We recommend using `load_mmm()` with the same
  version of the library that was used to save the model's pickled file.

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
