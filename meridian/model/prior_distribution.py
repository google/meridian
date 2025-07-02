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

"""This file contains an object to store prior distributions.

The `PriorDistribution` object contains distributions for various parameters
used by the Meridian model object.
"""

from __future__ import annotations
from collections.abc import MutableMapping
import dataclasses
from typing import Any
import warnings
from meridian import constants
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


__all__ = [
    'PriorDistribution',
]


@dataclasses.dataclass(kw_only=True)
class PriorDistribution:
  """Contains prior distributions for each model parameter.

  PriorDistribution is a utility class for Meridian. The required shapes of the
  arguments to `PriorDistribution` depend on the modeling options and data
  shapes passed to Meridian. For example, `ec_m` is a parameter that represents
  the half-saturation for each media channel. The `ec_m` argument must have
  either `batch_shape=[]` or `batch_shape` equal to the number of media
  channels. In the case of the former, each media channel gets the same prior.

  An error is raised upon Meridian construction if any prior distribution
  has a shape that cannot be broadcast to the shape designated by the model
  specification.

  The parameter batch shapes are as follows:

  | Parameter             | Batch shape                |
  |-----------------------|----------------------------|
  | `knot_values`         | `n_knots`                  |
  | `tau_g_excl_baseline` | `n_geos - 1`               |
  | `beta_m`              | `n_media_channels`         |
  | `beta_rf`             | `n_rf_channels`            |
  | `beta_om`             | `n_organic_media_channels` |
  | `beta_orf`            | `n_organic_rf_channels`    |
  | `eta_m`               | `n_media_channels`         |
  | `eta_rf`              | `n_rf_channels`            |
  | `eta_om`              | `n_organic_media_channels` |
  | `eta_orf`             | `n_organic_rf_channels`    |
  | `gamma_c`             | `n_controls`               |
  | `gamma_n`             | `n_non_media_channels`     |
  | `xi_c`                | `n_controls`               |
  | `xi_n`                | `n_non_media_channels`     |
  | `alpha_m`             | `n_media_channels`         |
  | `alpha_rf`            | `n_rf_channels`            |
  | `alpha_om`            | `n_organic_media_channels` |
  | `alpha_orf`           | `n_organic_rf_channels`    |
  | `ec_m`                | `n_media_channels`         |
  | `ec_rf`               | `n_rf_channels`            |
  | `ec_om`               | `n_organic_media_channels` |
  | `ec_orf`              | `n_organic_rf_channels`    |
  | `slope_m`             | `n_media_channels`         |
  | `slope_rf`            | `n_rf_channels`            |
  | `slope_om`            | `n_organic_media_channels` |
  | `slope_orf`           | `n_organic_rf_channels`    |
  | `sigma`               | (σ)                        |
  | `roi_m`               | `n_media_channels`         |
  | `roi_rf`              | `n_rf_channels`            |
  | `mroi_m`              | `n_media_channels`         |
  | `mroi_rf`             | `n_rf_channels`            |
  | `contribution_m`      | `n_media_channels`         |
  | `contribution_rf`     | `n_rf_channels`            |
  | `contribution_om`     | `n_organic_media_channels` |
  | `contribution_orf`    | `n_organic_f_channels`     |
  | `contribution_n`      | `n_non_media_channels`     |

  (σ) `n_geos` if `unique_sigma_for_each_geo`, otherwise this is `1`

  Attributes:
    knot_values: Prior distribution on knots for time effects. Default
      distribution is `Normal(0.0, 5.0)`.
    tau_g_excl_baseline: Prior distribution on geo effects, which represent the
      average KPI of each geo relative to the baseline geo. This parameter is
      broadcast to a vector of length `n_geos - 1`, preserving the geo order and
      excluding the `baseline_geo`. After sampling, `Meridian.inference_data`
      includes a modified version of this parameter called `tau_g`, which has
      length `n_geos` and contains a zero in the position corresponding to
      `baseline_geo`. Meridian ignores this distribution if `n_geos = 1`.
      Default distribution is `Normal(0.0, 5.0)`.
    beta_m: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (`beta_gm`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical mean.
      When `media_effects_dist` is set to `'log_normal'`, it is the hierarchical
      parameter for the mean of the underlying, log-transformed, `Normal`
      distribution. Meridian ignores this distribution if
      `paid_media_prior_type` is `'roi'` or `'mroi'`, and uses the `roi_m` or
      `mroi_m` prior instead. Default distribution is `HalfNormal(5.0)`.
    beta_rf: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for reach and frequency media channels
      (`beta_grf`). When `media_effects_dist` is set to `'normal'`, it is the
      hierarchical mean. When `media_effects_dist` is set to `'log_normal'`, it
      is the hierarchical parameter for the mean of the underlying,
      log-transformed, `Normal` distribution. Meridian ignores this distribution
      if `paid_media_prior_type` is `'roi'` or `'mroi'`, and uses the `roi_m` or
      `mroi_rf` prior instead. Default distribution is `HalfNormal(5.0)`.
    beta_om: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for organic media channels (`beta_gom`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical mean.
      When `media_effects_dist` is set to `'log_normal'`, it is the hierarchical
      parameter for the mean of the underlying, log-transformed, `Normal`
      distribution. Default distribution is `HalfNormal(5.0)`.
    beta_orf: Prior distribution on a parameter for the hierarchical
      distribution of geo-level media effects for organic reach and frequency
      media channels (`beta_gorf`). When `media_effects_dist` is set to
      `'normal'`, it is the hierarchical mean. When `media_effects_dist` is set
      to `'log_normal'`, it is the hierarchical parameter for the mean of the
      underlying, log-transformed, `Normal` distribution. Default distribution
      is `HalfNormal(5.0)`.
    eta_m: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for impression media channels (`beta_gm`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical standard
      deviation. When `media_effects_dist` is set to `'log_normal'` it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, `Normal` distribution. Default distribution is
      `HalfNormal(1.0)`.
    eta_rf: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for RF media channels (`beta_grf`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical standard
      deviation. When `media_effects_dist` is set to `'log_normal'` it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, `Normal` distribution. Default distribution is
      `HalfNormal(1.0)`.
    eta_om: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for organic media channels (`beta_gom`). When
      `media_effects_dist` is set to `'normal'`, it is the hierarchical standard
      deviation. When `media_effects_dist` is set to `'log_normal'` it is the
      hierarchical parameter for the standard deviation of the underlying,
      log-transformed, `Normal` distribution. Default distribution is
      `HalfNormal(1.0)`.
    eta_orf: Prior distribution on a parameter for the hierarchical distribution
      of geo-level media effects for organic RF media channels (`beta_gorf`).
      When `media_effects_dist` is set to `'normal'`, it is the hierarchical
      standard deviation. When `media_effects_dist` is set to `'log_normal'` it
      is the hierarchical parameter for the standard deviation of the
      underlying, log-transformed, `Normal` distribution. Default distribution
      is `HalfNormal(1.0)`.
    gamma_c: Prior distribution on the hierarchical mean of `gamma_gc` which is
      the coefficient on control `c` for geo `g`. Hierarchy is defined over
      geos. Default distribution is `Normal(0.0, 5.0)`.
    gamma_n: Prior distribution on the hierarchical mean of `gamma_gn` which is
      the coefficient on non-media channel `n` for geo `g`. Hierarchy is defined
      over geos. Default distribution is `Normal(0.0, 5.0)`.
    xi_c: Prior distribution on the hierarchical standard deviation of
      `gamma_gc` which is the coefficient on control `c` for geo `g`. Hierarchy
      is defined over geos. Default distribution is `HalfNormal(5.0)`.
    xi_n: Prior distribution on the hierarchical standard deviation of
      `gamma_gn` which is the coefficient on non-media channel `n` for geo `g`.
      Hierarchy is defined over geos. Default distribution is `HalfNormal(5.0)`.
    alpha_m: Prior distribution on the `geometric decay` Adstock parameter for
      media input. Default distribution is `Uniform(0.0, 1.0)`.
    alpha_rf: Prior distribution on the `geometric decay` Adstock parameter for
      RF input. Default distribution is `Uniform(0.0, 1.0)`.
    alpha_om: Prior distribution on the `geometric decay` Adstock parameter for
      organic media input. Default distribution is `Uniform(0.0, 1.0)`.
    alpha_orf: Prior distribution on the `geometric decay` Adstock parameter for
      organic RF input. Default distribution is `Uniform(0.0, 1.0)`.
    ec_m: Prior distribution on the `half-saturation` Hill parameter for media
      input. Default distribution is `TruncatedNormal(0.8, 0.8, 0.1, 10)`.
    ec_rf: Prior distribution on the `half-saturation` Hill parameter for RF
      input. Default distribution is `TransformedDistribution(LogNormal(0.7,
      0.4), Shift(0.1))`.
    ec_om: Prior distribution on the `half-saturation` Hill parameter for
      organic media input. Default distribution is `TruncatedNormal(0.8, 0.8,
      0.1, 10)`.
    ec_orf: Prior distribution on the `half-saturation` Hill parameter for
      organic RF input. Default distribution is `TransformedDistribution(
      LogNormal(0.7, 0.4), Shift(0.1))`.
    slope_m: Prior distribution on the `slope` Hill parameter for media input.
      Default distribution is `Deterministic(1.0)`.
    slope_rf: Prior distribution on the `slope` Hill parameter for RF input.
      Default distribution is `LogNormal(0.7, 0.4)`.
    slope_om: Prior distribution on the `slope` Hill parameter for organic media
      input. Default distribution is `Deterministic(1.0)`.
    slope_orf: Prior distribution on the `slope` Hill parameter for organic RF
      input. Default distribution is `LogNormal(0.7, 0.4)`.
    sigma: Prior distribution on the standard deviation of noise. Default
      distribution is `HalfNormal(5.0)`.
    roi_m: Prior distribution on the ROI of each media channel. This parameter
      is only used when `paid_media_prior_type` is `'roi'`, in which case
      `beta_m` is calculated as a deterministic function of `roi_rf`,
      `alpha_rf`, `ec_rf`, `slope_rf`, and the spend associated with each media
      channel. Default distribution is `LogNormal(0.2, 0.9)`. When `kpi_type` is
      `'non_revenue'` and `revenue_per_kpi` is not provided, ROI is interpreted
      as incremental KPI units per monetary unit spent. In this case, the
      default value for `roi_m` and `roi_rf` will be ignored and a common ROI
      prior will be assigned to all channels to achieve a target mean and
      standard deviation on the total media contribution.
    roi_rf: Prior distribution on the ROI of each Reach & Frequency channel.
      This parameter is only used when `paid_media_prior_type` is `'roi'`, in
      which case `beta_rf` is calculated as a deterministic function of
      `roi_rf`, `alpha_rf`, `ec_rf`, `slope_rf`, and the spend associated with
      each RF channel. Default distribution is `LogNormal(0.2, 0.9)`. When
      `kpi_type` is `'non_revenue'` and `revenue_per_kpi` is not provided, ROI
      is interpreted as incremental KPI units per monetary unit spent. In this
      case, the default value for `roi_m` and `roi_rf` will be ignored and a
      common ROI prior will be assigned to all channels to achieve a target mean
      and standard deviation on the total media contribution.
    mroi_m: Prior distribution on the mROI of each media channel. This parameter
      is only used when `paid_media_prior_type` is `'mroi'`, in which case
      `beta_m` is calculated as a deterministic function of `mroi_m`, `alpha_m`,
      `ec_m`, `slope_m`, and the spend associated with each media channel.
      Default distribution is `LogNormal(0.0, 0.5)`. When `kpi_type` is
      `'non_revenue'` and `revenue_per_kpi` is not provided, mROI is interpreted
      as the marginal incremental KPI units per monetary unit spent. In this
      case, a default distribution is not provided, so the user must specify it.
    mroi_rf: Prior distribution on the mROI of each Reach & Frequency channel.
      This parameter is only used when `paid_media_prior_type` is `'mroi'`, in
      which case `beta_rf` is calculated as a deterministic function of
      `mroi_rf`, `alpha_rf`, `ec_rf`, `slope_rf`, and the spend associated with
      each media channel. Default distribution is `LogNormal(0.0, 0.5)`. When
      `kpi_type` is `'non_revenue'` and `revenue_per_kpi` is not provided, mROI
      is interpreted as the marginal incremental KPI units per monetary unit
      spent. In this case, a default distribution is not provided, so the user
      must specify it.
    contribution_m: Prior distribution on the contribution of each media channel
      as a percentage of total outcome. This parameter is only used when
      `paid_media_prior_type` is `'contribution'`, in which case `beta_m` is
      calculated as a deterministic function of `contribution_m`, `alpha_m`,
      `ec_m`, `slope_m`, and the total outcome. Default distribution is
      `Beta(1.0, 99.0)`.
    contribution_rf: Prior distribution on the contribution of each Reach &
      Frequency channel as a percentage of total outcome. This parameter is only
      used when `paid_media_prior_type` is `'contribution'`, in which case
      `beta_rf` is calculated as a deterministic function of `contribution_rf`,
      `alpha_rf`, `ec_rf`, `slope_rf`, and the total outcome. Default
      distribution is `Beta(1.0, 99.0)`.
    contribution_om: Prior distribution on the contribution of each organic
      media channel as a percentage of total outcome. This parameter is only
      used when `organic_media_prior_type` is `'contribution'`, in which case
      `beta_om` is calculated as a deterministic function of `contribution_om`,
      `alpha_om`, `ec_om`, `slope_om`, and the total outcome. Default
      distribution is `Beta(1.0, 99.0)`.
    contribution_orf: Prior distribution on the contribution of each organic
      Reach & Frequency channel as a percentage of total outcome. This parameter
      is only used when `organic_media_prior_type` is `'contribution'`, in which
      case `beta_orf` is calculated as a deterministic function of
      `contribution_orf`, `alpha_orf`, `ec_orf`, `slope_orf`, and the total
      outcome. Default distribution is `Beta(1.0, 99.0)`.
    contribution_n: Prior distribution on the contribution of each non-media
      treatment channel as a percentage of total outcome. This parameter is only
      used when `non_media_treatment_prior_type` is `'contribution'`, in which
      case `gamma_n` is calculated as a deterministic function of
      `contribution_n` and the total outcome. Default distribution is
      `TruncatedNormal(0.0, 0.1, -1.0, 1.0)`.
  """

  knot_values: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.KNOT_VALUES
      ),
  )
  tau_g_excl_baseline: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.TAU_G_EXCL_BASELINE
      ),
  )
  beta_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_M
      ),
  )
  beta_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_RF
      ),
  )
  beta_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_OM
      ),
  )
  beta_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.BETA_ORF
      ),
  )
  eta_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_M
      ),
  )
  eta_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_RF
      ),
  )
  eta_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_OM
      ),
  )
  eta_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          1.0, name=constants.ETA_ORF
      ),
  )
  gamma_c: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.GAMMA_C
      ),
  )
  gamma_n: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Normal(
          0.0, 5.0, name=constants.GAMMA_N
      ),
  )
  xi_c: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.XI_C
      ),
  )
  xi_n: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.XI_N
      ),
  )
  alpha_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_M
      ),
  )
  alpha_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_RF
      ),
  )
  alpha_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_OM
      ),
  )
  alpha_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Uniform(
          0.0, 1.0, name=constants.ALPHA_ORF
      ),
  )
  ec_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TruncatedNormal(
          0.8, 0.8, 0.1, 10, name=constants.EC_M
      ),
  )
  ec_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TransformedDistribution(
          tfp.distributions.LogNormal(0.7, 0.4),
          tfp.bijectors.Shift(0.1),
          name=constants.EC_RF,
      ),
  )
  ec_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TruncatedNormal(
          0.8, 0.8, 0.1, 10, name=constants.EC_OM
      ),
  )
  ec_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TransformedDistribution(
          tfp.distributions.LogNormal(0.7, 0.4),
          tfp.bijectors.Shift(0.1),
          name=constants.EC_ORF,
      ),
  )
  slope_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Deterministic(
          1.0, name=constants.SLOPE_M
      ),
  )
  slope_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.7, 0.4, name=constants.SLOPE_RF
      ),
  )
  slope_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Deterministic(
          1.0, name=constants.SLOPE_OM
      ),
  )
  slope_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.7, 0.4, name=constants.SLOPE_ORF
      ),
  )
  sigma: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.HalfNormal(
          5.0, name=constants.SIGMA
      ),
  )
  roi_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.2, 0.9, name=constants.ROI_M
      ),
  )
  roi_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.2, 0.9, name=constants.ROI_RF
      ),
  )
  mroi_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.0, 0.5, name=constants.MROI_M
      ),
  )
  mroi_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.LogNormal(
          0.0, 0.5, name=constants.MROI_RF
      ),
  )
  contribution_m: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_M
      ),
  )
  contribution_rf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_RF
      ),
  )
  contribution_om: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_OM
      ),
  )
  contribution_orf: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_ORF
      ),
  )
  contribution_n: tfp.distributions.Distribution = dataclasses.field(
      default_factory=lambda: tfp.distributions.TruncatedNormal(
          loc=0.0, scale=0.1, low=-1.0, high=1.0, name=constants.CONTRIBUTION_N
      ),
  )

  def __setstate__(self, state):
    # Override to support pickling.
    def _unpack_distribution_params(
        params: MutableMapping[str, Any],
    ) -> tfp.distributions.Distribution:
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _unpack_distribution_params(
            params[constants.DISTRIBUTION]
        )
      dist_type = params.pop(constants.DISTRIBUTION_TYPE)
      return dist_type(**params)

    new_state = {}
    for attribute, value in state.items():
      new_state[attribute] = _unpack_distribution_params(value)

    self.__dict__.update(new_state)

  def __getstate__(self):
    # Override to support pickling.
    state = self.__dict__.copy()

    def _pack_distribution_params(
        dist: tfp.distributions.Distribution,
    ) -> MutableMapping[str, Any]:
      params = dist.parameters
      params[constants.DISTRIBUTION_TYPE] = type(dist)
      if constants.DISTRIBUTION in params:
        params[constants.DISTRIBUTION] = _pack_distribution_params(
            dist.distribution
        )
      return params

    for attribute, value in state.items():
      state[attribute] = _pack_distribution_params(value)

    return state

  def has_deterministic_param(
      self, param: tfp.distributions.Distribution
  ) -> bool:
    return hasattr(self, param) and isinstance(
        getattr(self, param).distribution, tfp.distributions.Deterministic
    )

  def broadcast(
      self,
      n_geos: int,
      n_media_channels: int,
      n_rf_channels: int,
      n_organic_media_channels: int,
      n_organic_rf_channels: int,
      n_controls: int,
      n_non_media_channels: int,
      unique_sigma_for_each_geo: bool,
      n_knots: int,
      is_national: bool,
      set_total_media_contribution_prior: bool,
      kpi: float,
      total_spend: np.ndarray,
  ) -> PriorDistribution:
    """Returns a new `PriorDistribution` with broadcast distribution attributes.

    Args:
      n_geos: Number of geos.
      n_media_channels: Number of media channels used.
      n_rf_channels: Number of reach and frequency channels used.
      n_organic_media_channels: Number of organic media channels used.
      n_organic_rf_channels: Number of organic reach and frequency channels
        used.
      n_controls: Number of controls used.
      n_non_media_channels: Number of non-media channels used.
      unique_sigma_for_each_geo: A boolean indicator whether to use the same
        sigma parameter for all geos. Only used if `n_geos > 1`. For more
        information, see `ModelSpec`.
      n_knots: Number of knots used.
      is_national: A boolean indicator whether the prior distribution will be
        adapted for a national model.
      set_total_media_contribution_prior: A boolean indicator whether the ROI
        priors should be set to achieve a total media constribution prior with
        target mean and variance.
      kpi: Sum of the entire KPI across geos and time. Required if
        `set_total_media_contribution_prior=True`.
      total_spend: Spend per media channel summed across geos and time. Required
        if `set_total_media_contribution_prior=True`.

    Returns:
      A new `PriorDistribution` broadcast from this prior distribution,
      according to the given data dimensionality.

    Raises:
      ValueError: If custom priors are not set for all channels.
    """

    def _validate_media_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_media_channels != param.batch_shape[0]
      ):
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f' number of media channels ({n_media_channels}), representing a '
            "a custom prior for each channel. If you can't determine a custom "
            'prior, consider using the default prior for that channel.'
        )

    _validate_media_custom_priors(self.roi_m)
    _validate_media_custom_priors(self.mroi_m)
    _validate_media_custom_priors(self.contribution_m)
    _validate_media_custom_priors(self.alpha_m)
    _validate_media_custom_priors(self.ec_m)
    _validate_media_custom_priors(self.slope_m)
    _validate_media_custom_priors(self.eta_m)
    _validate_media_custom_priors(self.beta_m)

    def _validate_organic_media_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_organic_media_channels != param.batch_shape[0]
      ):
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f' number of organic media channels ({n_organic_media_channels}), '
            "representing a custom prior for each channel. If you can't "
            'determine a custom prior, consider using the default prior for '
            'that channel.'
        )

    _validate_organic_media_custom_priors(self.contribution_om)
    _validate_organic_media_custom_priors(self.alpha_om)
    _validate_organic_media_custom_priors(self.ec_om)
    _validate_organic_media_custom_priors(self.slope_om)
    _validate_organic_media_custom_priors(self.eta_om)
    _validate_organic_media_custom_priors(self.beta_om)

    def _validate_organic_rf_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_organic_rf_channels != param.batch_shape[0]
      ):
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f'number of organic RF channels ({n_organic_rf_channels}), '
            "representing a custom prior for each channel. If you can't "
            'determine a custom prior, consider using the default prior '
            'for that channel.'
        )

    _validate_organic_rf_custom_priors(self.contribution_orf)
    _validate_organic_rf_custom_priors(self.alpha_orf)
    _validate_organic_rf_custom_priors(self.ec_orf)
    _validate_organic_rf_custom_priors(self.slope_orf)
    _validate_organic_rf_custom_priors(self.eta_orf)
    _validate_organic_rf_custom_priors(self.beta_orf)

    def _validate_rf_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if param.batch_shape.as_list() and n_rf_channels != param.batch_shape[0]:
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f'number of RF channels ({n_rf_channels}), representing a custom '
            "prior for each channel. If you can't determine a custom prior, "
            'consider using the default prior for that channel.'
        )

    _validate_rf_custom_priors(self.roi_rf)
    _validate_rf_custom_priors(self.mroi_rf)
    _validate_rf_custom_priors(self.contribution_rf)
    _validate_rf_custom_priors(self.alpha_rf)
    _validate_rf_custom_priors(self.ec_rf)
    _validate_rf_custom_priors(self.slope_rf)
    _validate_rf_custom_priors(self.eta_rf)
    _validate_rf_custom_priors(self.beta_rf)

    def _validate_control_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if param.batch_shape.as_list() and n_controls != param.batch_shape[0]:
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f'number of control variables ({n_controls}), representing a '
            "custom prior for each control variable. If you can't determine a "
            'custom prior, consider using the default prior for that variable.'
        )

    _validate_control_custom_priors(self.gamma_c)
    _validate_control_custom_priors(self.xi_c)

    def _validate_non_media_custom_priors(
        param: tfp.distributions.Distribution,
    ) -> None:
      if (
          param.batch_shape.as_list()
          and n_non_media_channels != param.batch_shape[0]
      ):
        raise ValueError(
            f'Custom priors length ({param.batch_shape[0]}) must match the '
            f'number of non-media channels ({n_non_media_channels}), '
            "representing a custom prior for each channel. If you can't "
            'determine a custom prior, consider using the default prior for '
            'that channel.'
        )

    _validate_non_media_custom_priors(self.contribution_n)
    _validate_non_media_custom_priors(self.gamma_n)
    _validate_non_media_custom_priors(self.xi_n)

    knot_values = tfp.distributions.BatchBroadcast(
        self.knot_values,
        n_knots,
        name=constants.KNOT_VALUES,
    )
    if is_national:
      tau_g_converted = _convert_to_deterministic_0_distribution(
          self.tau_g_excl_baseline
      )
    else:
      tau_g_converted = self.tau_g_excl_baseline
    tau_g_excl_baseline = tfp.distributions.BatchBroadcast(
        tau_g_converted, n_geos - 1, name=constants.TAU_G_EXCL_BASELINE
    )
    beta_m = tfp.distributions.BatchBroadcast(
        self.beta_m, n_media_channels, name=constants.BETA_M
    )
    beta_rf = tfp.distributions.BatchBroadcast(
        self.beta_rf, n_rf_channels, name=constants.BETA_RF
    )
    beta_om = tfp.distributions.BatchBroadcast(
        self.beta_om, n_organic_media_channels, name=constants.BETA_OM
    )
    beta_orf = tfp.distributions.BatchBroadcast(
        self.beta_orf, n_organic_rf_channels, name=constants.BETA_ORF
    )
    if is_national:
      eta_m_converted = _convert_to_deterministic_0_distribution(self.eta_m)
      eta_rf_converted = _convert_to_deterministic_0_distribution(self.eta_rf)
      eta_om_converted = _convert_to_deterministic_0_distribution(self.eta_om)
      eta_orf_converted = _convert_to_deterministic_0_distribution(self.eta_orf)
    else:
      eta_m_converted = self.eta_m
      eta_rf_converted = self.eta_rf
      eta_om_converted = self.eta_om
      eta_orf_converted = self.eta_orf
    eta_m = tfp.distributions.BatchBroadcast(
        eta_m_converted, n_media_channels, name=constants.ETA_M
    )
    eta_rf = tfp.distributions.BatchBroadcast(
        eta_rf_converted, n_rf_channels, name=constants.ETA_RF
    )
    eta_om = tfp.distributions.BatchBroadcast(
        eta_om_converted,
        n_organic_media_channels,
        name=constants.ETA_OM,
    )
    eta_orf = tfp.distributions.BatchBroadcast(
        eta_orf_converted, n_organic_rf_channels, name=constants.ETA_ORF
    )
    gamma_c = tfp.distributions.BatchBroadcast(
        self.gamma_c, n_controls, name=constants.GAMMA_C
    )
    if is_national:
      xi_c_converted = _convert_to_deterministic_0_distribution(self.xi_c)
    else:
      xi_c_converted = self.xi_c
    xi_c = tfp.distributions.BatchBroadcast(
        xi_c_converted, n_controls, name=constants.XI_C
    )
    gamma_n = tfp.distributions.BatchBroadcast(
        self.gamma_n, n_non_media_channels, name=constants.GAMMA_N
    )
    if is_national:
      xi_n_converted = _convert_to_deterministic_0_distribution(self.xi_n)
    else:
      xi_n_converted = self.xi_n
    xi_n = tfp.distributions.BatchBroadcast(
        xi_n_converted, n_non_media_channels, name=constants.XI_N
    )
    alpha_m = tfp.distributions.BatchBroadcast(
        self.alpha_m, n_media_channels, name=constants.ALPHA_M
    )
    alpha_rf = tfp.distributions.BatchBroadcast(
        self.alpha_rf, n_rf_channels, name=constants.ALPHA_RF
    )
    alpha_om = tfp.distributions.BatchBroadcast(
        self.alpha_om, n_organic_media_channels, name=constants.ALPHA_OM
    )
    alpha_orf = tfp.distributions.BatchBroadcast(
        self.alpha_orf, n_organic_rf_channels, name=constants.ALPHA_ORF
    )
    ec_m = tfp.distributions.BatchBroadcast(
        self.ec_m, n_media_channels, name=constants.EC_M
    )
    ec_rf = tfp.distributions.BatchBroadcast(
        self.ec_rf, n_rf_channels, name=constants.EC_RF
    )
    ec_om = tfp.distributions.BatchBroadcast(
        self.ec_om, n_organic_media_channels, name=constants.EC_OM
    )
    ec_orf = tfp.distributions.BatchBroadcast(
        self.ec_orf, n_organic_rf_channels, name=constants.EC_ORF
    )
    if (
        not isinstance(self.slope_m, tfp.distributions.Deterministic)
        or (np.isscalar(self.slope_m.loc.numpy()) and self.slope_m.loc != 1.0)
        or (
            self.slope_m.batch_shape.as_list()
            and any(x != 1.0 for x in self.slope_m.loc)
        )
    ):
      warnings.warn(
          'Changing the prior for `slope_m` may lead to convex Hill curves.'
          ' This may lead to poor MCMC convergence and budget optimization'
          ' may no longer produce a global optimum.'
      )
    slope_m = tfp.distributions.BatchBroadcast(
        self.slope_m, n_media_channels, name=constants.SLOPE_M
    )
    slope_rf = tfp.distributions.BatchBroadcast(
        self.slope_rf, n_rf_channels, name=constants.SLOPE_RF
    )
    if (
        not isinstance(self.slope_om, tfp.distributions.Deterministic)
        or (np.isscalar(self.slope_om.loc.numpy()) and self.slope_om.loc != 1.0)
        or (
            self.slope_om.batch_shape.as_list()
            and any(x != 1.0 for x in self.slope_om.loc)
        )
    ):
      warnings.warn(
          'Changing the prior for `slope_om` may lead to convex Hill curves.'
          ' This may lead to poor MCMC convergence and budget optimization'
          ' may no longer produce a global optimum.'
      )
    slope_om = tfp.distributions.BatchBroadcast(
        self.slope_om, n_organic_media_channels, name=constants.SLOPE_OM
    )
    slope_orf = tfp.distributions.BatchBroadcast(
        self.slope_orf, n_organic_rf_channels, name=constants.SLOPE_ORF
    )

    # If `unique_sigma_for_each_geo == False`, then make a scalar batch.
    sigma_shape = n_geos if (n_geos > 1 and unique_sigma_for_each_geo) else []
    sigma = tfp.distributions.BatchBroadcast(
        self.sigma, sigma_shape, name=constants.SIGMA
    )

    if set_total_media_contribution_prior:
      roi_m_converted = _get_total_media_contribution_prior(
          kpi, total_spend, constants.ROI_M
      )
      roi_rf_converted = _get_total_media_contribution_prior(
          kpi, total_spend, constants.ROI_RF
      )
    else:
      roi_m_converted = self.roi_m
      roi_rf_converted = self.roi_rf
    roi_m = tfp.distributions.BatchBroadcast(
        roi_m_converted, n_media_channels, name=constants.ROI_M
    )
    roi_rf = tfp.distributions.BatchBroadcast(
        roi_rf_converted, n_rf_channels, name=constants.ROI_RF
    )

    mroi_m = tfp.distributions.BatchBroadcast(
        self.mroi_m, n_media_channels, name=constants.MROI_M
    )
    mroi_rf = tfp.distributions.BatchBroadcast(
        self.mroi_rf, n_rf_channels, name=constants.MROI_RF
    )

    contribution_m = tfp.distributions.BatchBroadcast(
        self.contribution_m, n_media_channels, name=constants.CONTRIBUTION_M
    )
    contribution_rf = tfp.distributions.BatchBroadcast(
        self.contribution_rf, n_rf_channels, name=constants.CONTRIBUTION_RF
    )
    contribution_om = tfp.distributions.BatchBroadcast(
        self.contribution_om,
        n_organic_media_channels,
        name=constants.CONTRIBUTION_OM,
    )
    contribution_orf = tfp.distributions.BatchBroadcast(
        self.contribution_orf,
        n_organic_rf_channels,
        name=constants.CONTRIBUTION_ORF,
    )
    contribution_n = tfp.distributions.BatchBroadcast(
        self.contribution_n, n_non_media_channels, name=constants.CONTRIBUTION_N
    )

    return PriorDistribution(
        knot_values=knot_values,
        tau_g_excl_baseline=tau_g_excl_baseline,
        beta_m=beta_m,
        beta_rf=beta_rf,
        beta_om=beta_om,
        beta_orf=beta_orf,
        eta_m=eta_m,
        eta_rf=eta_rf,
        eta_om=eta_om,
        eta_orf=eta_orf,
        gamma_c=gamma_c,
        gamma_n=gamma_n,
        xi_c=xi_c,
        xi_n=xi_n,
        alpha_m=alpha_m,
        alpha_rf=alpha_rf,
        alpha_om=alpha_om,
        alpha_orf=alpha_orf,
        ec_m=ec_m,
        ec_rf=ec_rf,
        ec_om=ec_om,
        ec_orf=ec_orf,
        slope_m=slope_m,
        slope_rf=slope_rf,
        slope_om=slope_om,
        slope_orf=slope_orf,
        sigma=sigma,
        roi_m=roi_m,
        roi_rf=roi_rf,
        mroi_m=mroi_m,
        mroi_rf=mroi_rf,
        contribution_m=contribution_m,
        contribution_rf=contribution_rf,
        contribution_om=contribution_om,
        contribution_orf=contribution_orf,
        contribution_n=contribution_n,
    )


def _convert_to_deterministic_0_distribution(
    distribution: tfp.distributions.Distribution,
) -> tfp.distributions.Distribution:
  """Converts the given distribution to a `Deterministic(0)` one.

  Args:
    distribution: `tfp.distributions.Distribution` object to be converted to
      `Deterministic(0)` distribution.

  Returns:
    `tfp.distribution.Deterministic(0, distribution.name)`

  Raises:
    Warning: If the argument distribution is not a `Deterministic(0)`
    distribution.
  """
  if (
      not isinstance(distribution, tfp.distributions.Deterministic)
      or distribution.loc != 0
  ):
    warnings.warn(
        'Hierarchical distribution parameters must be deterministically zero'
        f' for national models. {distribution.name} has been automatically set'
        ' to Deterministic(0).'
    )
    return tfp.distributions.Deterministic(loc=0, name=distribution.name)
  else:
    return distribution


def _get_total_media_contribution_prior(
    kpi: float,
    total_spend: np.ndarray,
    name: str,
    p_mean: float = constants.P_MEAN,
    p_sd: float = constants.P_SD,
) -> tfp.distributions.Distribution:
  """Determines ROI priors based on total media contribution.

  Args:
    kpi: Sum of the entire KPI across geos and time.
    total_spend: Spend per media channel summed across geos and time.
    name: Name of the distribution.
    p_mean: Prior mean proportion of KPI incremental due to all media. Default
      value is `0.4`.
    p_sd: Prior standard deviation proportion of KPI incremental to all media.
      Default value is `0.2`.

  Returns:
    A new `Distribution` based on total media contribution.
  """
  roi_mean = p_mean * kpi / np.sum(total_spend)
  roi_sd = p_sd * kpi / np.sqrt(np.sum(np.power(total_spend, 2)))
  lognormal_sigma = tf.cast(
      np.sqrt(np.log(roi_sd**2 / roi_mean**2 + 1)), dtype=tf.float32
  )
  lognormal_mu = tf.cast(
      np.log(roi_mean * np.exp(-(lognormal_sigma**2) / 2)), dtype=tf.float32
  )
  return tfp.distributions.LogNormal(lognormal_mu, lognormal_sigma, name=name)


def distributions_are_equal(
    a: tfp.distributions.Distribution, b: tfp.distributions.Distribution
) -> bool:
  """Determine if two distributions are equal."""
  if type(a) != type(b):  # pylint: disable=unidiomatic-typecheck
    return False

  a_params = a.parameters.copy()
  b_params = b.parameters.copy()

  if constants.DISTRIBUTION in a_params and constants.DISTRIBUTION in b_params:
    if not distributions_are_equal(
        a_params[constants.DISTRIBUTION], b_params[constants.DISTRIBUTION]
    ):
      return False
    del a_params[constants.DISTRIBUTION]
    del b_params[constants.DISTRIBUTION]

  if constants.DISTRIBUTION in a_params or constants.DISTRIBUTION in b_params:
    return False

  return a_params == b_params
