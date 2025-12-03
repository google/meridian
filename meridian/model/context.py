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

import functools
import warnings
import numpy as np

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.model import media
from meridian.model import knots
from meridian.model import prior_distribution
from meridian.model import spec

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
    self._set_total_media_contribution_prior = False

  @property
  def n_media_channels(self) -> int:
    return (
        self._input_data.media.shape[-1]
        if self._input_data.media is not None
        else 0
    )

  @property
  def n_rf_channels(self) -> int:
    return (
        self._input_data.reach.shape[-1]
        if self._input_data.reach is not None
        else 0
    )

  @property
  def n_organic_media_channels(self) -> int:
    return (
        self._input_data.organic_media.shape[-1]
        if self._input_data.organic_media is not None
        else 0
    )

  @property
  def n_organic_rf_channels(self) -> int:
    return (
        self._input_data.organic_reach.shape[-1]
        if self._input_data.organic_reach is not None
        else 0
    )

  @property
  def n_controls(self) -> int:
    return (
        self._input_data.controls.shape[-1]
        if self._input_data.controls is not None
        else 0
    )

  @property
  def n_non_media_channels(self) -> int:
    return (
        self._input_data.non_media_treatments.shape[-1]
        if self._input_data.non_media_treatments is not None
        else 0
    )

  @property
  def unique_sigma_for_each_geo(self) -> bool:
    return (
        self._model_spec.unique_sigma_for_each_geo
        if not self.is_national
        else False
    )

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.KnotInfo(self._input_data, self._model_spec)

  @property
  def n_geos(self) -> int:
    return self._input_data.kpi.shape[0]

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.MediaTensors(self._input_data, self._model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.RfTensors(self._input_data, self._model_spec)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.kpi.values, backend.float32)

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return self.media_tensors.media_spend + self.rf_tensors.rf_spend

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.sum(self.kpi)

  @property
  def media_effects_dist(self) -> str:
    return (
        self._model_spec.media_effects_dist
        if not self.is_national
        else constants.MEDIA_EFFECTS_NORMAL
    )

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  def _warn_setting_ignored_priors(self):
    """Raises a warning if ignored priors are set."""
    default_distribution = prior_distribution.PriorDistribution()
    for ignored_priors_dict, prior_type, prior_type_name in (
        (
            constants.IGNORED_PRIORS_MEDIA,
            self.model_spec.effective_media_prior_type,
            "media_prior_type",
        ),
        (
            constants.IGNORED_PRIORS_RF,
            self.model_spec.effective_rf_prior_type,
            "rf_prior_type",
        ),
    ):
      ignored_custom_priors = []
      for prior in ignored_priors_dict.get(prior_type, []):
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
            f' `{prior_type_name}` is set to "{prior_type}".'
        )

  def _validate_mroi_priors_non_revenue(self):
    """Validates mroi priors in the non-revenue outcome case."""
    if (
        self.input_data.kpi_type == constants.NON_REVENUE
        and self.input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      if (
          self.n_media_channels > 0
          and (
              self.model_spec.effective_media_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self.model_spec.prior.mroi_m, default_distribution.mroi_m
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
              self.model_spec.effective_rf_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self.model_spec.prior.mroi_rf, default_distribution.mroi_rf
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
        self.input_data.kpi_type == constants.NON_REVENUE
        and self.input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      default_roi_m_used = (
          self.model_spec.effective_media_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self.model_spec.prior.roi_m, default_distribution.roi_m
          )
      )
      default_roi_rf_used = (
          self.model_spec.effective_rf_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self.model_spec.prior.roi_rf, default_distribution.roi_rf
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
    prior = self.model_spec.prior
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
        prior_type == self.model_spec.media_prior_type
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
        unique_sigma_for_each_geo=self.unique_sigma_for_each_geo,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=self._set_total_media_contribution_prior,
        kpi=np.sum(self.input_data.kpi.values),
        total_spend=agg_total_spend,
    )
