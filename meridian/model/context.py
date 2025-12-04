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

"""Defines ModelContext class for Meridian."""

from collections.abc import Mapping, Sequence
import functools
import warnings

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers
import numpy as np

__all__ = [
    "ModelContext",
]


class ModelContext:
  """Model context for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec,
  ):
    self._input_data = input_data
    self._model_spec = model_spec

    self._validate_data_dependent_model_spec()
    self._validate_model_spec_shapes()

    self._set_total_media_contribution_prior = False
    self._warn_setting_ignored_priors()
    self._validate_mroi_priors_non_revenue()
    self._validate_roi_priors_non_revenue()
    self._check_media_prior_support()
    self._validate_geo_invariants()
    self._validate_time_invariants()

  def _validate_data_dependent_model_spec(self):
    """Validates that the data dependent model specs have correct shapes."""

    if self._model_spec.roi_calibration_period is not None and (
        self._model_spec.roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_media_channels,
        )
    ):
      raise ValueError(
          "The shape of `roi_calibration_period`"
          f" {self._model_spec.roi_calibration_period.shape} is different from"
          f" `(n_media_times, n_media_channels) = ({self.n_media_times},"
          f" {self.n_media_channels})`."
      )

    if self._model_spec.rf_roi_calibration_period is not None and (
        self._model_spec.rf_roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_rf_channels,
        )
    ):
      raise ValueError(
          "The shape of `rf_roi_calibration_period`"
          f" {self._model_spec.rf_roi_calibration_period.shape} is different"
          f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
          f" {self.n_rf_channels})`."
      )

    if self._model_spec.holdout_id is not None:
      if self.is_national and (
          self._model_spec.holdout_id.shape != (self.n_times,)
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_times,) = ({self.n_times},)`."
        )
      elif not self.is_national and (
          self._model_spec.holdout_id.shape
          != (
              self.n_geos,
              self.n_times,
          )
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_geos, n_times) = ({self.n_geos},"
            f" {self.n_times})`."
        )

    if self._model_spec.control_population_scaling_id is not None and (
        self._model_spec.control_population_scaling_id.shape
        != (self.n_controls,)
    ):
      raise ValueError(
          "The shape of `control_population_scaling_id`"
          f" {self._model_spec.control_population_scaling_id.shape} is"
          f" different from `(n_controls,) = ({self.n_controls},)`."
      )

    if self._model_spec.non_media_population_scaling_id is not None and (
        self._model_spec.non_media_population_scaling_id.shape
        != (self.n_non_media_channels,)
    ):
      raise ValueError(
          "The shape of `non_media_population_scaling_id`"
          f" {self._model_spec.non_media_population_scaling_id.shape} is"
          " different from `(n_non_media_channels,) ="
          f" ({self.n_non_media_channels},)`."
      )

  def _validate_model_spec_shapes(self):
    """Validate shapes of model_spec attributes."""
    if self._model_spec.roi_calibration_period is not None:
      if self._model_spec.roi_calibration_period.shape != (
          self.n_media_times,
          self.n_media_channels,
      ):
        raise ValueError(
            "The shape of `roi_calibration_period`"
            f" {self._model_spec.roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_media_channels) = ({self.n_media_times},"
            f" {self.n_media_channels})`."
        )

    if self._model_spec.rf_roi_calibration_period is not None:
      if self._model_spec.rf_roi_calibration_period.shape != (
          self.n_media_times,
          self.n_rf_channels,
      ):
        raise ValueError(
            "The shape of `rf_roi_calibration_period`"
            f" {self._model_spec.rf_roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
            f" {self.n_rf_channels})`."
        )

    if self._model_spec.holdout_id is not None:
      expected_shape = (
          (self.n_times,) if self.is_national else (self.n_geos, self.n_times)
      )
      if self._model_spec.holdout_id.shape != expected_shape:
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            " different from"
            f" {'`(n_times,)`' if self.is_national else '`(n_geos, n_times)`'} ="
            f" {expected_shape}."
        )

    if self._model_spec.control_population_scaling_id is not None:
      if self._model_spec.control_population_scaling_id.shape != (
          self.n_controls,
      ):
        raise ValueError(
            "The shape of `control_population_scaling_id`"
            f" {self._model_spec.control_population_scaling_id.shape} is"
            f" different from `(n_controls,) = ({self.n_controls},)`."
        )

  def _validate_geo_invariants(self):
    """Validates non-national model invariants."""
    if self.is_national:
      return

    if self._input_data.controls is not None:
      self._check_if_no_geo_variation(
          self.controls_scaled,
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_geo_variation(
          self.non_media_treatments_normalized,
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_geo_variation(
          self.media_tensors.media_scaled,
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_geo_variation(
          self.rf_tensors.reach_scaled,
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_geo_variation(
          self.organic_media_tensors.organic_media_scaled,
          "organic_media",
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_geo_variation(
          self.organic_rf_tensors.organic_reach_scaled,
          "organic_reach",
          self._input_data.organic_reach.coords[
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
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=0) < epsilon
    )[:, 1]
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
    if self._input_data.controls is not None:
      self._check_if_no_time_variation(
          self.controls_scaled,
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_time_variation(
          self.non_media_treatments_normalized,
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_time_variation(
          self.media_tensors.media_scaled,
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_time_variation(
          self.rf_tensors.reach_scaled,
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_time_variation(
          self.organic_media_tensors.organic_media_scaled,
          constants.ORGANIC_MEDIA,
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_time_variation(
          self.organic_rf_tensors.organic_reach_scaled,
          constants.ORGANIC_REACH,
          self._input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
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
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=1) < epsilon
    )[:, 1]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    mask = backend.equal(counts, self.n_geos)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)
    if col_idx_bad.shape[0]:
      if self.is_national:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            " which is equivalent to no signal at all in a national model:"
            f" {dims_bad}.  This can lead to poor model convergence. To address"
            " this, drop the listed variables that do not vary across time."
        )
      else:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            f" making a model with geo main effects unidentifiable: {dims_bad}."
            " This can lead to poor model convergence. Since these variables"
            " only vary across geo and not across time, they are collinear"
            " with geo and redundant in a model with geo main effects. To"
            " address this, drop the listed variables that do not vary across"
            " time."
        )

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.build_media_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.build_rf_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self._input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self._input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.kpi, dtype=backend.float32)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self._input_data.revenue_per_kpi is None:
      return None
    return backend.to_tensor(
        self._input_data.revenue_per_kpi, dtype=backend.float32
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    if self._input_data.controls is None:
      return None
    return backend.to_tensor(self._input_data.controls, dtype=backend.float32)

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self._input_data.non_media_treatments is None:
      return None
    return backend.to_tensor(
        self._input_data.non_media_treatments, dtype=backend.float32
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.population, dtype=backend.float32)

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_spend(), dtype=backend.float32
    )

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_outcome(), dtype=backend.float32
    )

  @property
  def n_geos(self) -> int:
    return len(self._input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self._input_data.media_channel is None:
      return 0
    return len(self._input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self._input_data.rf_channel is None:
      return 0
    return len(self._input_data.rf_channel)

  @property
  def n_organic_media_channels(self) -> int:
    if self._input_data.organic_media_channel is None:
      return 0
    return len(self._input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self._input_data.organic_rf_channel is None:
      return 0
    return len(self._input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    if self._input_data.control_variable is None:
      return 0
    return len(self._input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self._input_data.non_media_channel is None:
      return 0
    return len(self._input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self._input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self._input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self._model_spec.knots,
        enable_aks=self._model_spec.enable_aks,
        data=self._input_data,
        is_national=self.is_national,
    )

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for controls, if it exists."""
    if self.controls is None:
      return None

    if self._model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = backend.to_tensor(
          self._model_spec.control_population_scaling_id, dtype=backend.bool_
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
    if self._model_spec.non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.to_tensor(
          self._model_spec.non_media_population_scaling_id, dtype=backend.bool_
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
  def controls_scaled(self) -> backend.Tensor | None:
    if self.controls is not None:
      # If `controls` is defined, then `controls_transformer` is also defined.
      return self.controls_transformer.forward(self.controls)  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def non_media_treatments_normalized(self) -> backend.Tensor | None:
    """Normalized non-media treatments.

    The non-media treatments values are scaled by population (for channels where
    `non_media_population_scaling_id` is `True`) and normalized by centering and
    scaling with means and standard deviations.
    """
    if self.non_media_transformer is not None:
      return self.non_media_transformer.forward(
          self.non_media_treatments
      )  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
    return self.kpi_transformer.forward(self.kpi)

  @functools.cached_property
  def media_effects_dist(self) -> str:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[constants.MEDIA_EFFECTS_DIST]
    else:
      return self._model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      # Should evaluate to False.
      return constants.NATIONAL_MODEL_SPEC_ARGS[
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self._model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self._model_spec.baseline_geo, int):
      if (
          self._model_spec.baseline_geo < 0
          or self._model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self._model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self._model_spec.baseline_geo
    elif isinstance(self._model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self._input_data.geo == self._model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self._model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    if self._model_spec.holdout_id is None:
      return None
    tensor = backend.to_tensor(self._model_spec.holdout_id, dtype=backend.bool_)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  def _warn_setting_ignored_priors(self):
    """Raises a warning if ignored priors are set."""
    default_distribution = prior_distribution.PriorDistribution()
    for ignored_priors_dict, prior_type, prior_type_name in (
        (
            constants.IGNORED_PRIORS_MEDIA,
            self._model_spec.effective_media_prior_type,
            "media_prior_type",
        ),
        (
            constants.IGNORED_PRIORS_RF,
            self._model_spec.effective_rf_prior_type,
            "rf_prior_type",
        ),
    ):
      ignored_custom_priors = []
      for prior in ignored_priors_dict.get(prior_type, []):
        self_prior = getattr(self._model_spec.prior, prior)
        default_prior = getattr(default_distribution, prior)
        if not prior_distribution.distributions_are_equal(
            self_prior, default_prior
        ):
          ignored_custom_priors.append(prior)
      if ignored_custom_priors:
        ignored_priors_str = ", ".join(ignored_custom_priors)
        warnings.warn(
            f"Custom prior(s) `{ignored_priors_str}` are ignored when"
            f' `{prior_type_name}` is set to "{prior_type}".'
        )

  def _validate_mroi_priors_non_revenue(self):
    """Validates mroi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      if (
          self.n_media_channels > 0
          and (
              self._model_spec.effective_media_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.mroi_m, default_distribution.mroi_m
          )
      ):
        raise ValueError(
            f"Custom priors should be set on `{constants.MROI_M}` when"
            ' `media_prior_type` is "mroi", KPI is non-revenue and revenue per'
            " kpi data is missing."
        )
      if (
          self.n_rf_channels > 0
          and (
              self._model_spec.effective_rf_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.mroi_rf, default_distribution.mroi_rf
          )
      ):
        raise ValueError(
            f"Custom priors should be set on `{constants.MROI_RF}` when"
            ' `rf_prior_type` is "mroi", KPI is non-revenue and revenue per kpi'
            " data is missing."
        )

  def _validate_roi_priors_non_revenue(self):
    """Validates roi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      default_roi_m_used = (
          self._model_spec.effective_media_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.roi_m, default_distribution.roi_m
          )
      )
      default_roi_rf_used = (
          self._model_spec.effective_rf_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.roi_rf, default_distribution.roi_rf
          )
      )
      # If ROI priors are used with the default prior distribution for all paid
      # channels (media and RF), then use the "total paid media contribution
      # prior" procedure.
      if (
          (default_roi_m_used and default_roi_rf_used)
          or (self.n_media_channels == 0 and default_roi_rf_used)
          or (self.n_rf_channels == 0 and default_roi_m_used)
      ):
        self._set_total_media_contribution_prior = True
        warnings.warn(
            "Consider setting custom ROI priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with"
            f" `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}`. Further"
            " documentation available at "
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior",
        )
      elif self.n_media_channels > 0 and default_roi_m_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_M}` when"
            ' `media_prior_type` is "roi", custom priors are assigned on'
            ' `{constants.ROI_RF}` or `rf_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )
      elif self.n_rf_channels > 0 and default_roi_rf_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_RF}` when"
            ' `rf_prior_type` is "roi", custom priors are assigned on'
            ' `{constants.ROI_M}` or `media_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )

  def _check_media_prior_support(self) -> None:
    """Checks ROI, mROI, and Contribution prior support when random effects are log-normal.

    Priors for ROI, mROI, and Contribution can only have negative support if the
    random effects follow a normal distribution. This check enforces that priors
    have non-negative support when random effects follow a log-normal
    distribution. This check only applies to geo-level models with log-normal
    random effects since national models do not have random effects.
    """
    prior = self._model_spec.prior
    if self.n_media_channels > 0:
      self._check_for_negative_support(
          prior.roi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
      )
      self._check_for_negative_support(
          prior.mroi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
      )
      self._check_for_negative_support(
          prior.contribution_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_rf_channels > 0:
      self._check_for_negative_support(
          prior.roi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
      )
      self._check_for_negative_support(
          prior.mroi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
      )
      self._check_for_negative_support(
          prior.contribution_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_organic_media_channels > 0:
      self._check_for_negative_support(
          prior.contribution_om,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_organic_rf_channels > 0:
      self._check_for_negative_support(
          prior.contribution_orf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )

  def _check_for_negative_support(
      self,
      dist: backend.tfd.Distribution,
      media_effects_dist: str,
      prior_type: str,
  ) -> None:
    """Checks for negative support in prior distributions.

    When `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`, prior distributions
    for media effects must be non-negative. This function raises a ValueError if
    any part of the distribution's CDF is greater than 0 at 0, indicating some
    probability mass below zero.

    Args:
      dist: The distribution to check.
      media_effects_dist: The type of media effects distribution.
      prior_type: The prior type that corresponds with current prior under test.

    Raises:
      ValueError: If the prior distribution has negative support when
      `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`.
    """
    if (
        prior_type == self._model_spec.media_prior_type
        and media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL
        and np.any(dist.cdf(0) > 0)
    ):
      raise ValueError(
          "Media priors must have non-negative support when"
          f' `media_effects_dist`="{media_effects_dist}". Found negative prior'
          f" distribution support for {dist.name}."
      )

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    total_spend = self._input_data.get_total_spend()
    # Total spend can have 1, 2 or 3 dimensions. Aggregate by channel.
    if len(total_spend.shape) == 1:
      # Already aggregated by channel.
      agg_total_spend = total_spend
    elif len(total_spend.shape) == 2:
      agg_total_spend = np.sum(total_spend, axis=(0,))
    else:
      agg_total_spend = np.sum(total_spend, axis=(0, 1))

    return self._model_spec.prior.broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_organic_media_channels=self.n_organic_media_channels,
        n_organic_rf_channels=self.n_organic_rf_channels,
        n_controls=self.n_controls,
        n_non_media_channels=self.n_non_media_channels,
        unique_sigma_for_each_geo=self.unique_sigma_for_each_geo,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=self._set_total_media_contribution_prior,
        kpi=np.sum(self._input_data.kpi.values),
        total_spend=agg_total_spend,
    )

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    if isinstance(self._model_spec.adstock_decay_spec, str):
      return adstock_hill.AdstockDecaySpec.from_consistent_type(
          self._model_spec.adstock_decay_spec
      )

    try:
      return self._create_adstock_decay_functions_from_channel_map(
          self._model_spec.adstock_decay_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `adstock_decay_spec` keys"
          f" {tuple(self._model_spec.adstock_decay_spec.keys())}. Keys should"
          " either contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())} or"
          " be one or more of {'media', 'rf', 'organic_media',"
          " 'organic_rf'}."
      ) from e

  def _create_adstock_decay_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> adstock_hill.AdstockDecaySpec:
    """Create `AdstockDecaySpec` from mapping from channels to decay functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      media_adstock_function = media_channel_builder(**channel_function_map)
    else:
      media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      rf_adstock_function = rf_channel_builder(**channel_function_map)
    else:
      rf_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_media_adstock_function = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_rf_adstock_function = organic_rf_channel_builder(
          **channel_function_map
      )
    else:
      organic_rf_adstock_function = constants.GEOMETRIC_DECAY

    return adstock_hill.AdstockDecaySpec(
        media=media_adstock_function,
        rf=rf_adstock_function,
        organic_media=organic_media_adstock_function,
        organic_rf=organic_rf_adstock_function,
    )

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
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
