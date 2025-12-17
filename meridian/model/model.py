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

"""Meridian module for the geo-level Bayesian hierarchical media mix model."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import os
import warnings

import arviz as az
import joblib
from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.data import time_coordinates as tc
from meridian.model import adstock_hill
from meridian.model import context
from meridian.model import equations
from meridian.model import knots
from meridian.model import media
from meridian.model import posterior_sampler
from meridian.model import prior_distribution
from meridian.model import prior_sampler
from meridian.model import spec
from meridian.model import transformers
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
import numpy as np

__all__ = [
    "MCMCSamplingError",
    "MCMCOOMError",
    "Meridian",
    "ModelFittingError",
    "NotFittedModelError",
    "save_mmm",
    "load_mmm",
]


class ModelFittingError(Exception):
  """Model has critical issues preventing fitting."""


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
    model_context: A `ModelContext` object containing the model context.
    model_equations: A `ModelEquations` object containing stateless mathematical
      functions and utilities for Meridian MMM.
    inference_data: A _mutable_ `arviz.InferenceData` object containing the
      resulting data from fitting the model.
    eda_engine: An `EDAEngine` object containing the EDA engine.
    eda_spec: An `EDASpec` object containing the EDA specification.
    eda_outcomes: A list of `EDAOutcome` objects containing the outcomes from
      running critical EDA checks.
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
    total_outcome: A tensor containing the total outcome, aggregated over geos
      and times.
    controls_transformer: A `ControlsTransformer` to scale controls tensors
      using the model's controls data.
    non_media_transformer: A `CenteringAndScalingTransformer` to scale non-media
      treatmenttensors using the model's non-media treatment data.
    kpi_transformer: A `KpiTransformer` to scale KPI tensors using the model's
      KPI data.
    controls_scaled: The controls tensor after pre-modeling transformations
      including population scaling (for variables with
      `ModelSpec.control_population_scaling_id` set to `True`), centering by the
      mean, and scaling by the standard deviation.
    non_media_treatments_scaled: The non-media treatment tensor after
      pre-modeling transformations including population scaling (for variables
      with `ModelSpec.non_media_population_scaling_id` set to `True`), centering
      by the mean, and scaling by the standard deviation.
    kpi_scaled: The KPI tensor after pre-modeling transformations including
      population scaling, centering by the mean, and scaling by the standard
      deviation.
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
      eda_spec: eda_spec_module.EDASpec = eda_spec_module.EDASpec(),
  ):
    self._inference_data = (
        inference_data if inference_data else az.InferenceData()
    )
    self._model_context = context.ModelContext(
        input_data=input_data,
        model_spec=model_spec if model_spec else spec.ModelSpec(),
    )
    self._model_equations = equations.ModelEquations(self._model_context)

    self._eda_spec = eda_spec

    self._validate_injected_inference_data()

    if self.is_national:
      _warn_setting_national_args(
          media_effects_dist=self.model_spec.media_effects_dist,
          unique_sigma_for_each_geo=self.model_spec.unique_sigma_for_each_geo,
      )
    self._validate_kpi_variability()

  @property
  def input_data(self) -> data.InputData:
    return self._model_context.input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_context.model_spec

  @property
  def model_context(self) -> context.ModelContext:
    return self._model_context

  @property
  def model_equations(self) -> equations.ModelEquations:
    return self._model_equations

  @property
  def inference_data(self) -> az.InferenceData:
    return self._inference_data

  @functools.cached_property
  def eda_engine(self) -> eda_engine.EDAEngine:
    return eda_engine.EDAEngine(self, spec=self._eda_spec)

  @property
  def eda_spec(self) -> eda_spec_module.EDASpec:
    return self._eda_spec

  @property
  def eda_outcomes(self) -> eda_outcome.CriticalCheckEDAOutcomes:
    return self.eda_engine.run_all_critical_checks()

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return self._model_context.media_tensors

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return self._model_context.rf_tensors

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return self._model_context.organic_media_tensors

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return self._model_context.organic_rf_tensors

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return self._model_context.kpi

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    return self._model_context.revenue_per_kpi

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    return self._model_context.controls

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    return self._model_context.non_media_treatments

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return self._model_context.population

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return self._model_context.total_spend

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return self._model_context.total_outcome

  @property
  def n_geos(self) -> int:
    return self._model_context.n_geos

  @property
  def n_media_channels(self) -> int:
    return self._model_context.n_media_channels

  @property
  def n_rf_channels(self) -> int:
    return self._model_context.n_rf_channels

  @property
  def n_organic_media_channels(self) -> int:
    return self._model_context.n_organic_media_channels

  @property
  def n_organic_rf_channels(self) -> int:
    return self._model_context.n_organic_rf_channels

  @property
  def n_controls(self) -> int:
    return self._model_context.n_controls

  @property
  def n_non_media_channels(self) -> int:
    return self._model_context.n_non_media_channels

  @property
  def n_times(self) -> int:
    return self._model_context.n_times

  @property
  def n_media_times(self) -> int:
    return self._model_context.n_media_times

  @property
  def is_national(self) -> bool:
    return self._model_context.is_national

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return self._model_context.knot_info

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    return self._model_context.controls_transformer

  @functools.cached_property
  def non_media_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    return self._model_context.non_media_transformer

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return self._model_context.kpi_transformer

  @functools.cached_property
  def controls_scaled(self) -> backend.Tensor | None:
    return self._model_context.controls_scaled

  @functools.cached_property
  def non_media_treatments_normalized(self) -> backend.Tensor | None:
    """Normalized non-media treatments.

    The non-media treatments values are scaled by population (for channels where
    `non_media_population_scaling_id` is `True`) and normalized by centering and
    scaling with means and standard deviations.
    """
    return self._model_context.non_media_treatments_normalized

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
    return self._model_context.kpi_scaled

  @functools.cached_property
  def media_effects_dist(self) -> str:
    return self._model_context.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    return self._model_context.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    return self._model_context.baseline_geo_idx

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    return self._model_context.holdout_id

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    return self._model_context.adstock_decay_spec

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    return self._model_context.prior_broadcast

  @functools.cached_property
  def prior_sampler_callable(self) -> prior_sampler.PriorDistributionSampler:
    """A `PriorDistributionSampler` callable bound to this model."""
    return prior_sampler.PriorDistributionSampler(
        model_context=self.model_context,
    )

  @functools.cached_property
  def posterior_sampler_callable(
      self,
  ) -> posterior_sampler.PosteriorMCMCSampler:
    """A `PosteriorMCMCSampler` callable bound to this model."""
    return posterior_sampler.PosteriorMCMCSampler(
        model_context=self.model_context,
    )

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
    warnings.warn(
        "Meridian.compute_non_media_treatments_baseline() is deprecated and"
        " will be removed in a future version. Use"
        " `ModelEquations.compute_non_media_treatments_baseline()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.model_equations.compute_non_media_treatments_baseline(
        non_media_baseline_values=non_media_baseline_values
    )

  def expand_selected_time_dims(
      self,
      start_date: tc.Date = None,
      end_date: tc.Date = None,
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

  def _validate_kpi_variability(self):
    """Validates the KPI variability."""
    if self.eda_engine.kpi_has_variability:
      return
    kpi = self.eda_engine.kpi_scaled_da.name

    if (
        self.n_media_channels > 0
        and self.model_spec.effective_media_prior_type
        in constants.PAID_MEDIA_ROI_PRIOR_TYPES
    ):
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          " `media_prior_type` ="
          f' "{self.model_spec.effective_media_prior_type}".'
      )
    if (
        self.n_rf_channels > 0
        and self.model_spec.effective_rf_prior_type
        in constants.PAID_MEDIA_ROI_PRIOR_TYPES
    ):
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          f' `rf_prior_type` = "{self.model_spec.effective_rf_prior_type}".'
      )
    if (
        self.n_organic_media_channels > 0
        and self.model_spec.organic_media_prior_type
        in [constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION]
    ):
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          " `organic_media_prior_type` ="
          f' "{self.model_spec.organic_media_prior_type}".'
      )
    if (
        self.n_organic_rf_channels > 0
        and self.model_spec.organic_rf_prior_type
        in [constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION]
    ):
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          " `organic_rf_prior_type` ="
          f' "{self.model_spec.organic_rf_prior_type}".'
      )
    if (
        self.n_non_media_channels > 0
        and self.model_spec.non_media_treatments_prior_type
        in [constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION]
    ):
      raise ValueError(
          f"`{kpi}` cannot be constant with"
          " `non_media_treatments_prior_type` ="
          f' "{self.model_spec.non_media_treatments_prior_type}".'
      )

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
    warnings.warn(
        "Meridian.linear_predictor_counterfactual_difference_media() is"
        " deprecated and will be removed in a future version. Use "
        "`ModelEquations.linear_predictor_counterfactual_difference_media()`"
        " instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return (
        self.model_equations.linear_predictor_counterfactual_difference_media(
            media_transformed=media_transformed,
            alpha_m=alpha_m,
            ec_m=ec_m,
            slope_m=slope_m,
        )
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
    warnings.warn(
        "Meridian.linear_predictor_counterfactual_difference_rf() is deprecated"
        " and will be removed in a future version. Use `ModelEquations."
        "linear_predictor_counterfactual_difference_rf()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.model_equations.linear_predictor_counterfactual_difference_rf(
        rf_transformed=rf_transformed,
        alpha_rf=alpha_rf,
        ec_rf=ec_rf,
        slope_rf=slope_rf,
    )

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
        inferred from `self.media_effects_dist`.
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
    warnings.warn(
        "Meridian.calculate_beta_x() is deprecated and will be removed in a"
        " future version. Use `ModelEquations.calculate_beta_x()`"
        " instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.model_equations.calculate_beta_x(
        is_non_media=is_non_media,
        incremental_outcome_x=incremental_outcome_x,
        linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
    )

  def adstock_hill_media(
      self,
      media: backend.Tensor,  # pylint: disable=redefined-outer-name
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
        `self.n_media_times`, in which case `n_times_output` defaults to
        `self.n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_media_channels]`
      representing Adstock and Hill-transformed media.
    """
    warnings.warn(
        "Meridian.adstock_hill_media() is deprecated and will be removed in a"
        " future version. Use `ModelEquations.adstock_hill_media()`"
        " instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.model_equations.adstock_hill_media(
        media=media,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

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
        `self.n_media_times`, in which case `n_times_output` defaults to
        `self.n_times`.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times, n_rf_channels]`
      representing Hill and Adstock-transformed RF.
    """
    warnings.warn(
        "Meridian.adstock_hill_rf() is deprecated and will be removed in a"
        " future version. Use `ModelEquations.adstock_hill_rf()`"
        " instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.model_equations.adstock_hill_rf(
        reach=reach,
        frequency=frequency,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, and so this method freezes the
    object's states before the computation graph is created.
    """
    self._model_context.populate_cached_properties()
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
    warnings.warn(
        "Meridian.create_inference_data_coords() is deprecated and will be"
        " removed in a future version. Use"
        " `ModelContext.create_inference_data_coords()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self._model_context.create_inference_data_coords(n_chains, n_draws)

  def create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    warnings.warn(
        "Meridian.create_inference_data_dims() is deprecated and will be"
        " removed in a future version. Use"
        " `ModelContext.create_inference_data_dims()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self._model_context.create_inference_data_dims()

  def sample_prior(self, n_draws: int, seed: int | None = None):
    """Draws samples from the prior distributions.

    Drawn samples are merged into this model's Arviz `inference_data` property.

    Args:
      n_draws: Number of samples drawn from the prior distribution.
      seed: Used to set the seed for reproducible results. For more information,
        see [PRNGS and seeds]
        (https://github.com/tensorflow/probability/blob/main/PRNGS.md).
    """
    prior_draws = self.prior_sampler_callable(n_draws=n_draws, seed=seed)
    # Create Arviz InferenceData for prior draws.
    prior_coords = self._model_context.create_inference_data_coords(1, n_draws)
    prior_dims = self._model_context.create_inference_data_dims()
    prior_inference_data = az.convert_to_inference_data(
        prior_draws,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )
    self.inference_data.extend(prior_inference_data, join="right")

  def _run_model_fitting_guardrail(self):
    """Raises an error if the model has critical EDA issues."""
    error_findings_by_type: dict[eda_outcome.EDACheckType, list[str]] = (
        collections.defaultdict(list)
    )
    for field in dataclasses.fields(self.eda_outcomes):
      outcome = getattr(self.eda_outcomes, field.name)
      error_findings = [
          finding
          for finding in outcome.findings
          if finding.severity == eda_outcome.EDASeverity.ERROR
      ]
      if error_findings:
        error_findings_by_type[outcome.check_type].extend(
            [finding.explanation for finding in error_findings]
        )

    if error_findings_by_type:
      error_message_lines = [
          "Model has critical EDA issues. Please fix before running"
          " `sample_posterior`.\n"
      ]
      for check_type, explanations in error_findings_by_type.items():
        error_message_lines.append(f"Check type: {check_type.name}")
        for explanation in explanations:
          error_message_lines.append(f"- {explanation}")
      error_message_lines.append(
          "For further details, please refer to `Meridian.eda_outcomes`."
      )
      raise ModelFittingError("\n".join(error_message_lines))

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
        will be treated as stateful seeds. See [tfp.random.sanitize_seed]
        (https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed).
      **pins: These are used to condition the provided joint distribution, and
        are passed directly to `joint_dist.experimental_pin(**pins)`.

    Throws:
      MCMCOOMError: If the model is out of memory. Try reducing `n_keep` or pass
        a list of integers as `n_chains` to sample chains serially. For more
        information, see
        [ResourceExhaustedError when running Meridian.sample_posterior]
        (https://developers.google.com/meridian/docs/post-modeling/model-debugging#gpu-oom-error).
    """
    self._run_model_fitting_guardrail()

    posterior_inference_data = self.posterior_sampler_callable(
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
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
  warnings.warn(
      "save_mmm is deprecated and will be removed in a future release. Please"
      " use `schema.serde.meridian_serde.save_meridian` instead. See"
      " https://developers.google.com/meridian/docs/user-guide/saving-model-object"
      " for details.",
      DeprecationWarning,
      stacklevel=2,
  )

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
  warnings.warn(
      "load_mmm is deprecated and will be removed in a future release. Please"
      " use `meridian.schema.serde.meridian_serde.load_meridian` instead. See"
      " https://developers.google.com/meridian/docs/user-guide/saving-model-object"
      " for details.",
      DeprecationWarning,
      stacklevel=2,
  )

  try:
    with open(file_path, "rb") as f:
      mmm = joblib.load(f)
    return mmm
  except FileNotFoundError:
    raise FileNotFoundError(f"No such file or directory: {file_path}") from None
