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

"""Defines model specification parameters for Meridian."""

from collections.abc import Mapping
import dataclasses
from typing import Sequence
import warnings

from meridian import constants
from meridian.model import prior_distribution
import numpy as np

__all__ = [
    "ModelSpec",
]


def _validate_roi_calibration_period(
    array: np.ndarray | None,
    array_name: str,
    channel_dim_name: str,
    prior_type: str,
    prior_type_name: str,
):
  """Validates the ROI calibration period array."""
  if array is None:
    return
  if prior_type != constants.TREATMENT_PRIOR_TYPE_ROI:
    raise ValueError(
        f"The `{array_name}` should be `None` unless `{prior_type_name}` is"
        f" '{constants.TREATMENT_PRIOR_TYPE_ROI}'."
    )
  if len(array.shape) != 2:
    raise ValueError(
        f"The shape of the `{array_name}` array {array.shape} should be"
        f" 2-dimensional (`n_media_times` x `{channel_dim_name}`)."
    )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Model specification parameters for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.

  Attributes:
    prior: A `PriorDistribution` object specifying the prior distribution of
      each set of model parameters. The distribution for a vector of parameters
      (for example, `alpha_m`) can be passed as either a scalar distribution or
      a vector distribution. If a scalar distribution is passed, it is broadcast
      to the actual shape of the parameter vector. See `paid_media_prior_type`
      for related details.
    media_effects_dist: A string to specify the distribution of media random
      effects across geos. This attribute is not used with a national-level
      model. Allowed values: `'normal'` or `'log_normal'`. Default:
      `'log_normal'`.
    hill_before_adstock: A boolean indicating whether to apply the Hill function
      before the Adstock function, instead of the default order of Adstock
      before Hill. This argument does not apply to RF channels. Default:
      `False`.
    max_lag: An integer indicating the maximum number of lag periods (≥ `0`) to
      include in the Adstock calculation. Default: `8`.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo. If `False`, then a single residual
      variance is used for all geos. Default: `False`.
    media_prior_type: A string to specify the prior type for the media
      coefficients. Allowed values: `'roi'`, `'mroi'`, `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `roi_m`, `mroi_m`,
      `contribution_m`, and `beta_m`, but only one of these is used depending on
      the `media_prior_type`. When `media_prior_type` is `'roi'`, the
      `PriorDistribution.roi_m` parameter is used to specify a prior on the ROI.
      When `media_prior_type` is `'mroi'`, the `PriorDistribution.mroi_m`
      parameter is used to specify a prior on the mROI. When `media_prior_type`
      is `'contribution'`, the `PriorDistribution.contribution_m` parameter is
      used to specify a prior on the contribution. When `media_prior_type` is
      `'coefficient'`, the `PriorDistribution.beta_m` parameter is used to
      specify a prior on the coefficient mean parameters. Default: `'roi'`.
    rf_prior_type: A string to specify the prior type for the RF coefficients.
      Allowed values: `'roi'`, `'mroi'`, `'contribution'`, `'coefficient'`. The
      `PriorDistribution` contains distributions `roi_rf`, `mroi_rf`,
      `contribution_rf`, and`beta_rf`, but only one of these is used depending
      on the `rf_prior_type`. When `rf_prior_type` is `'roi'`, the
      `PriorDistribution.roi_rf` parameter is used to specify a prior on the
      ROI. When `rf_media_prior_type` is `'mroi'`, the
      `PriorDistribution.mroi_rf` parameter is used to specify a prior on the
      mROI. When `rf_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_rf` parameter is used to specify a prior
      on the contribution. When `rf_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_rf` parameter is used to specify a prior on the
      coefficient mean parameters. Default: `'roi'`.
    paid_media_prior_type: Deprecated. Use `media_prior_type` and
      `rf_prior_type` instead. A string to specify the prior type for media and
      RF treatments at the same time. Ignored when `media_prior_type` or
      `rf_prior_type` are set. Default: `'roi'`.
    roi_calibration_period: An optional boolean array of shape `(n_media_times,
      n_media_channels)` indicating the subset of `time` that the ROI value of
      the `roi_m` prior applies to. The ROI numerator is the incremental outcome
      generated by media executed during the calibration period. More precisely,
      it is the difference in expected outcome between the counterfactual where
      media is set to historical values versus the counterfactual where media is
      set to zero during the calibration period and set to historical values for
      all other time periods. The denominator is the channel spend during
      calibration period (excluding any calibration time periods prior to the
      first KPI time period). Spend data by time period is required. If `None`,
      all times are used. Only used if `media_prior_type` is `'roi'`.
      Default: `None`.
    rf_roi_calibration_period: An optional boolean array of shape
      `(n_media_times, n_rf_channels)` indicating the subset of `time` that the
      ROI value of the `roi_rf` prior applies to. The ROI numerator is the
      incremental outcome generated by media executed during the calibration
      period. More precisely, it is the difference in expected outcome between
      the counterfactual where reach and frequency is set to historical values
      versus the counterfactual where reach is set to zero during the
      calibration period and set to historical values for all other time
      periods. The denominator is the channel spend during calibration period
      (excluding any calibration time periods prior to the first KPI time
      period). Spend data by time period is required. If `None`, all times are
      used. Only used if `rf_prior_type` is `'roi'`. Default: `None`.
    organic_media_prior_type: A string to specify the prior type for the organic
      media coefficients. Allowed values: `'contribution'`, `'coefficient'`.
      `PriorDistribution` contains `contribution_om` and `beta_om`, but only one
      of these is used depending on the `organic_media_prior_type`. When
      `organic_media_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_om` parameter is used to specify a prior
      on the contribution. When `organic_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.beta_om` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    organic_rf_prior_type: A string to specify the prior type for the organic
      reach and frequency coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains distributions
      `contribution_orf`, and `beta_orf`, but only one of these is used
      depending on the `organic_rf_prior_type`. When `organic_rf_prior_type` is
      `'contribution'`, the `PriorDistribution.contribution_orf` parameter is
      used to specify a prior on the contribution. When `organic_rf_prior_type`
      is `'coefficient'`, the `PriorDistribution.beta_orf` parameter is used to
      specify a prior on the coefficient mean parameters. Default:
      `'contribution'`.
    non_media_treatments_prior_type: A string to specify the prior type for the
      non-media treatment coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. `PriorDistribution` contains `contribution_n` and
      `gamma_n`, but only one of these is used depending on the
      `non_media_prior_type`. When `non_media_prior_type` is `'contribution'`,
      the `PriorDistribution.contribution_n` parameter is used to specify a
      prior on the contribution. When `non_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.gamma_n` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    non_media_baseline_values: Optional list with the shape
      `(n_non_media_channels,)`. Each element is either a float (which means
      that the fixed value will be used as baseline for the given channel) or
      one of the strings `"min"` or `"max"` (which mean that the global minimum
      or maximum value will be used as baseline for the scaled values of the
      given non_media treatments channel). If `None`, the minimum value is used
      as baseline for each non-media treatments channel. This attribute is used
      as the default value for the corresponding argument to `Analyzer` methods.
    knots: An optional integer or list of integers indicating the knots used to
      estimate time effects. When `knots` is a list of integers, the knot
      locations are provided by that list. Zero corresponds to a knot at the
      first time period, one corresponds to a knot at the second time period,
      ..., and `(n_times - 1)` corresponds to a knot at the last time period).
      Typically, we recommend including knots at `0` and `(n_times - 1)`, but
      this is not required. When `knots` is an integer, then there are knots
      with locations equally spaced across the time periods, (including knots at
      zero and `(n_times - 1)`. When `knots` is` 1`, there is a single common
      regression coefficient used for all time periods. If `knots` is set to
      `None`, then the numbers of knots used is equal to the number of time
      periods in the case of a geo model. This is equivalent to each time period
      having its own regression coefficient. If `knots` is set to `None` in the
      case of a national model, then the number of knots used is `1`. Default:
      `None`.
    baseline_geo: An optional integer or a string for the baseline geo. The
      baseline geo is treated as the reference geo in the dummy encoding of
      geos. Non-baseline geos have a corresponding `tau_g` indicator variable,
      meaning that they have a higher prior variance than the baseline geo. When
      set to `None`, the geo with the biggest population is used as the
      baseline. Default: `None`.
    holdout_id: Optional boolean tensor of dimensions `(n_geos, n_times)` for a
      geo-level model or `(n_times,)` for a national model, indicating which
      observations are part of the holdout sample, which are excluded from the
      training sample. Only KPI data is excluded from the training sample. Media
      data is still included as it can affect Adstock for subsequent weeks. If
      "ROI priors" are used, then the `roi_m` parameters correspond to the ROI
      of all geos and times, even those in the holdout sample.
    control_population_scaling_id: An optional boolean tensor of dimension
      `(n_controls,)` indicating the control variables for which the control
      value will be scaled by population. If `None`, no control variables are
      scaled by population. Default: `None`.
    non_media_population_scaling_id: An optional boolean tensor of dimension
      `(n_non_media_channels,)` indicating the non-media variables for which the
      non-media value will be scaled by population. If `None`, then no non-media
      variables are scaled by population. Default: `None`.
    adstock_decay_spec: A string or mapping specifying the adstock decay
      function for each media, RF, organic media and organic RF channel. If a
      string, must be either `'geometric'` or `'binomial'`, specifying that
      decay function for all channels. If a mapping, keys should be channel
      names and values should be `'geometric'` or `'binomial'`, with each
      key-value pair denoting the adstock decay function to use for that
      channel. Channels that are not specified in the mapping default to using
      'geometric'. Default: `'geometric'`.
    enable_aks: A boolean indicating whether to use the Automatic Knot Selection
      algorithm to select an optimal number of knots for running the model
      instead of the default 1 for national models and n_times for geo models.
      If this is set to `True` and the `knots` arg is provided, then an error
      will be raised. Default: `False`.
  """

  prior: prior_distribution.PriorDistribution = dataclasses.field(
      default_factory=prior_distribution.PriorDistribution,
  )
  media_effects_dist: str = constants.MEDIA_EFFECTS_LOG_NORMAL
  hill_before_adstock: bool = False
  max_lag: int = 8
  unique_sigma_for_each_geo: bool = False
  media_prior_type: str | None = None
  rf_prior_type: str | None = None
  paid_media_prior_type: str | None = None
  roi_calibration_period: np.ndarray | None = None
  rf_roi_calibration_period: np.ndarray | None = None
  organic_media_prior_type: str = constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  organic_rf_prior_type: str = constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  non_media_treatments_prior_type: str = (
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  )
  non_media_baseline_values: Sequence[float | str] | None = None
  knots: int | list[int] | None = None
  baseline_geo: int | str | None = None
  holdout_id: np.ndarray | None = None
  control_population_scaling_id: np.ndarray | None = None
  non_media_population_scaling_id: np.ndarray | None = None
  adstock_decay_spec: str | Mapping[str, str] = constants.GEOMETRIC_DECAY
  enable_aks: bool = False

  def __post_init__(self):
    # Validate media_effects_dist.
    if self.media_effects_dist not in constants.MEDIA_EFFECTS_DISTRIBUTIONS:
      raise ValueError(
          f"The `media_effects_dist` parameter '{self.media_effects_dist}' must"
          f" be one of {sorted(constants.MEDIA_EFFECTS_DISTRIBUTIONS)}."
      )
    # Support paid_media_prior_type for backwards compatibility.
    if self.paid_media_prior_type is not None:
      if self.media_prior_type is not None or self.rf_prior_type is not None:
        raise ValueError(
            "The deprecated `paid_media_prior_type` parameter cannot be used"
            " with `media_prior_type` or `rf_prior_type`. Use"
            " `media_prior_type` and `rf_prior_type` instead."
        )
      else:
        warnings.warn(
            "Using `paid_media_prior_type` parameter will set prior types for"
            " media and RF at the same time. This is deprecated and will be"
            " removed in a future version of Meridian. Use `media_prior_type`"
            " and `rf_prior_type` instead."
        )
    # Validate prior_type.
    if (
        self.effective_media_prior_type
        not in constants.PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `media_prior_type` parameter"
          f" '{self.effective_media_prior_type}' must be one of"
          f" {sorted(constants.PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.effective_rf_prior_type not in constants.PAID_TREATMENT_PRIOR_TYPES:
      raise ValueError(
          "The `rf_prior_type` parameter"
          f" '{self.effective_rf_prior_type}' must be one of"
          f" {sorted(constants.PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.organic_media_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_media_prior_type` parameter"
          f" '{self.organic_media_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.organic_rf_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_rf_prior_type` parameter"
          f" '{self.organic_rf_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )
    if self.non_media_treatments_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `non_media_treatments_prior_type` parameter"
          f" '{self.non_media_treatments_prior_type}' must be one of"
          f" {sorted(constants.NON_PAID_TREATMENT_PRIOR_TYPES)}."
      )

    _validate_roi_calibration_period(
        array=self.roi_calibration_period,
        array_name="roi_calibration_period",
        channel_dim_name="n_media_channels",
        prior_type=self.effective_media_prior_type,
        prior_type_name="media_prior_type",
    )
    _validate_roi_calibration_period(
        array=self.rf_roi_calibration_period,
        array_name="rf_roi_calibration_period",
        channel_dim_name="n_rf_channels",
        prior_type=self.effective_rf_prior_type,
        prior_type_name="rf_prior_type",
    )

    # Validate knots.
    if isinstance(self.knots, list) and not self.knots:
      raise ValueError("The `knots` parameter cannot be an empty list.")
    if isinstance(self.knots, int) and self.knots == 0:
      raise ValueError("The `knots` parameter cannot be zero.")
    if self.knots is not None and self.enable_aks:
      raise ValueError(
          "The `knots` parameter cannot be set when `enable_aks` is True."
      )

  @property
  def effective_media_prior_type(self) -> str:
    """Returns the effective media prior type.

    The recommended way to set prior types is to use `media_prior_type` and
    `rf_prior_type` directly. If both `media_prior_type` and `rf_prior_type`
    are not set, the deprecated `paid_media_prior_type` is used for both media
    and RF channels. If none of them are set, the default is `roi`.
    """
    if self.media_prior_type is not None:
      return self.media_prior_type
    elif self.paid_media_prior_type is not None:
      return self.paid_media_prior_type
    else:
      return constants.TREATMENT_PRIOR_TYPE_ROI

  @property
  def effective_rf_prior_type(self) -> str:
    """Returns the effective rf prior type.

    The recommended way to set prior types is to use `media_prior_type` and
    `rf_prior_type` directly. If both `media_prior_type` and `rf_prior_type`
    are not set, the deprecated `paid_media_prior_type` is used for both media
    and RF channels. If none of them are set, the default is `roi`.
    """
    if self.rf_prior_type is not None:
      return self.rf_prior_type
    elif self.paid_media_prior_type is not None:
      return self.paid_media_prior_type
    else:
      return constants.TREATMENT_PRIOR_TYPE_ROI
