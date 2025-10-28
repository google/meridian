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

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any
import warnings

from meridian import backend
from meridian import constants
import numpy as np


__all__ = [
    'IndependentMultivariateDistribution',
    'PriorDistribution',
    'distributions_are_equal',
    'lognormal_dist_from_mean_std',
    'lognormal_dist_from_range',
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
    alpha_m: Prior distribution on the Adstock decay parameter for media input.
      Default distribution is `Uniform(0.0, 1.0)`.
    alpha_rf: Prior distribution on the Adstock decay parameter for RF input.
      Default distribution is `Uniform(0.0, 1.0)`.
    alpha_om: Prior distribution on the Adstock decay parameter for organic
      media input. Default distribution is `Uniform(0.0, 1.0)`.
    alpha_orf: Prior distribution on the Adstock decay parameter for organic RF
      input. Default distribution is `Uniform(0.0, 1.0)`.
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
      `beta_m` is calculated as a deterministic function of `roi_m`, `alpha_m`,
      `ec_m`, `slope_m`, and the spend associated with each media channel.
      Default distribution is `LogNormal(0.2, 0.9)`. When `kpi_type` is
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

  knot_values: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Normal(
          0.0, 5.0, name=constants.KNOT_VALUES
      ),
  )
  tau_g_excl_baseline: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Normal(
          0.0, 5.0, name=constants.TAU_G_EXCL_BASELINE
      ),
  )
  beta_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          5.0, name=constants.BETA_M
      ),
  )
  beta_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          5.0, name=constants.BETA_RF
      ),
  )
  beta_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          5.0, name=constants.BETA_OM
      ),
  )
  beta_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          5.0, name=constants.BETA_ORF
      ),
  )
  eta_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(1.0, name=constants.ETA_M),
  )
  eta_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          1.0, name=constants.ETA_RF
      ),
  )
  eta_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          1.0, name=constants.ETA_OM
      ),
  )
  eta_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(
          1.0, name=constants.ETA_ORF
      ),
  )
  gamma_c: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Normal(
          0.0, 5.0, name=constants.GAMMA_C
      ),
  )
  gamma_n: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Normal(
          0.0, 5.0, name=constants.GAMMA_N
      ),
  )
  xi_c: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(5.0, name=constants.XI_C),
  )
  xi_n: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(5.0, name=constants.XI_N),
  )
  alpha_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Uniform(
          0.0, 1.0, name=constants.ALPHA_M
      ),
  )
  alpha_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Uniform(
          0.0, 1.0, name=constants.ALPHA_RF
      ),
  )
  alpha_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Uniform(
          0.0, 1.0, name=constants.ALPHA_OM
      ),
  )
  alpha_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Uniform(
          0.0, 1.0, name=constants.ALPHA_ORF
      ),
  )
  ec_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.TruncatedNormal(
          0.8, 0.8, 0.1, 10, name=constants.EC_M
      ),
  )
  ec_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.TransformedDistribution(
          backend.tfd.LogNormal(0.7, 0.4),
          backend.bijectors.Shift(0.1),
          name=constants.EC_RF,
      ),
  )
  ec_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.TruncatedNormal(
          0.8, 0.8, 0.1, 10, name=constants.EC_OM
      ),
  )
  ec_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.TransformedDistribution(
          backend.tfd.LogNormal(0.7, 0.4),
          backend.bijectors.Shift(0.1),
          name=constants.EC_ORF,
      ),
  )
  slope_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Deterministic(
          1.0, name=constants.SLOPE_M
      ),
  )
  slope_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.7, 0.4, name=constants.SLOPE_RF
      ),
  )
  slope_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Deterministic(
          1.0, name=constants.SLOPE_OM
      ),
  )
  slope_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.7, 0.4, name=constants.SLOPE_ORF
      ),
  )
  sigma: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.HalfNormal(5.0, name=constants.SIGMA),
  )
  roi_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.2, 0.9, name=constants.ROI_M
      ),
  )
  roi_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.2, 0.9, name=constants.ROI_RF
      ),
  )
  mroi_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.0, 0.5, name=constants.MROI_M
      ),
  )
  mroi_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.LogNormal(
          0.0, 0.5, name=constants.MROI_RF
      ),
  )
  contribution_m: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_M
      ),
  )
  contribution_rf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_RF
      ),
  )
  contribution_om: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_OM
      ),
  )
  contribution_orf: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.Beta(
          1.0, 99.0, name=constants.CONTRIBUTION_ORF
      ),
  )
  contribution_n: backend.tfd.Distribution = dataclasses.field(
      default_factory=lambda: backend.tfd.TruncatedNormal(
          loc=0.0, scale=0.1, low=-1.0, high=1.0, name=constants.CONTRIBUTION_N
      ),
  )

  def __post_init__(self):
    for param, bounds in _parameter_space_bounds.items():
      prevent_deterministic_prior_at_bounds = (
          _prevent_deterministic_prior_at_bounds[param]
          if param in _prevent_deterministic_prior_at_bounds.keys()
          else (False, False)
      )
      _validate_support(
          param,
          getattr(self, param),
          bounds,
          prevent_deterministic_prior_at_bounds,
      )

  def __setstate__(self, state):
    # Override to support pickling.
    def _unpack_distribution_params(
        params: MutableMapping[str, Any],
    ) -> backend.tfd.Distribution:
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
        dist: backend.tfd.Distribution,
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

  def has_deterministic_param(self, param: backend.tfd.Distribution) -> bool:
    return hasattr(self, param) and isinstance(
        getattr(self, param).distribution, backend.tfd.Deterministic
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
        param: backend.tfd.Distribution,
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
        param: backend.tfd.Distribution,
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
        param: backend.tfd.Distribution,
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
        param: backend.tfd.Distribution,
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
        param: backend.tfd.Distribution,
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
        param: backend.tfd.Distribution,
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

    knot_values = backend.tfd.BatchBroadcast(
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
    tau_g_excl_baseline = backend.tfd.BatchBroadcast(
        tau_g_converted, n_geos - 1, name=constants.TAU_G_EXCL_BASELINE
    )
    beta_m = backend.tfd.BatchBroadcast(
        self.beta_m, n_media_channels, name=constants.BETA_M
    )
    beta_rf = backend.tfd.BatchBroadcast(
        self.beta_rf, n_rf_channels, name=constants.BETA_RF
    )
    beta_om = backend.tfd.BatchBroadcast(
        self.beta_om, n_organic_media_channels, name=constants.BETA_OM
    )
    beta_orf = backend.tfd.BatchBroadcast(
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
    eta_m = backend.tfd.BatchBroadcast(
        eta_m_converted, n_media_channels, name=constants.ETA_M
    )
    eta_rf = backend.tfd.BatchBroadcast(
        eta_rf_converted, n_rf_channels, name=constants.ETA_RF
    )
    eta_om = backend.tfd.BatchBroadcast(
        eta_om_converted,
        n_organic_media_channels,
        name=constants.ETA_OM,
    )
    eta_orf = backend.tfd.BatchBroadcast(
        eta_orf_converted, n_organic_rf_channels, name=constants.ETA_ORF
    )
    gamma_c = backend.tfd.BatchBroadcast(
        self.gamma_c, n_controls, name=constants.GAMMA_C
    )
    if is_national:
      xi_c_converted = _convert_to_deterministic_0_distribution(self.xi_c)
    else:
      xi_c_converted = self.xi_c
    xi_c = backend.tfd.BatchBroadcast(
        xi_c_converted, n_controls, name=constants.XI_C
    )
    gamma_n = backend.tfd.BatchBroadcast(
        self.gamma_n, n_non_media_channels, name=constants.GAMMA_N
    )
    if is_national:
      xi_n_converted = _convert_to_deterministic_0_distribution(self.xi_n)
    else:
      xi_n_converted = self.xi_n
    xi_n = backend.tfd.BatchBroadcast(
        xi_n_converted, n_non_media_channels, name=constants.XI_N
    )
    alpha_m = backend.tfd.BatchBroadcast(
        self.alpha_m, n_media_channels, name=constants.ALPHA_M
    )
    alpha_rf = backend.tfd.BatchBroadcast(
        self.alpha_rf, n_rf_channels, name=constants.ALPHA_RF
    )
    alpha_om = backend.tfd.BatchBroadcast(
        self.alpha_om, n_organic_media_channels, name=constants.ALPHA_OM
    )
    alpha_orf = backend.tfd.BatchBroadcast(
        self.alpha_orf, n_organic_rf_channels, name=constants.ALPHA_ORF
    )
    ec_m = backend.tfd.BatchBroadcast(
        self.ec_m, n_media_channels, name=constants.EC_M
    )
    ec_rf = backend.tfd.BatchBroadcast(
        self.ec_rf, n_rf_channels, name=constants.EC_RF
    )
    ec_om = backend.tfd.BatchBroadcast(
        self.ec_om, n_organic_media_channels, name=constants.EC_OM
    )
    ec_orf = backend.tfd.BatchBroadcast(
        self.ec_orf, n_organic_rf_channels, name=constants.EC_ORF
    )
    if (
        not isinstance(self.slope_m, backend.tfd.Deterministic)
        or (backend.rank(self.slope_m.loc) == 0 and self.slope_m.loc != 1.0)
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
    slope_m = backend.tfd.BatchBroadcast(
        self.slope_m, n_media_channels, name=constants.SLOPE_M
    )
    slope_rf = backend.tfd.BatchBroadcast(
        self.slope_rf, n_rf_channels, name=constants.SLOPE_RF
    )
    if (
        not isinstance(self.slope_om, backend.tfd.Deterministic)
        or (backend.rank(self.slope_om.loc) == 0 and self.slope_om.loc != 1.0)
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
    slope_om = backend.tfd.BatchBroadcast(
        self.slope_om, n_organic_media_channels, name=constants.SLOPE_OM
    )
    slope_orf = backend.tfd.BatchBroadcast(
        self.slope_orf, n_organic_rf_channels, name=constants.SLOPE_ORF
    )

    # If `unique_sigma_for_each_geo == False`, then make a scalar batch.
    sigma_shape = n_geos if (n_geos > 1 and unique_sigma_for_each_geo) else []
    sigma = backend.tfd.BatchBroadcast(
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
    roi_m = backend.tfd.BatchBroadcast(
        roi_m_converted, n_media_channels, name=constants.ROI_M
    )
    roi_rf = backend.tfd.BatchBroadcast(
        roi_rf_converted, n_rf_channels, name=constants.ROI_RF
    )

    mroi_m = backend.tfd.BatchBroadcast(
        self.mroi_m, n_media_channels, name=constants.MROI_M
    )
    mroi_rf = backend.tfd.BatchBroadcast(
        self.mroi_rf, n_rf_channels, name=constants.MROI_RF
    )

    contribution_m = backend.tfd.BatchBroadcast(
        self.contribution_m, n_media_channels, name=constants.CONTRIBUTION_M
    )
    contribution_rf = backend.tfd.BatchBroadcast(
        self.contribution_rf, n_rf_channels, name=constants.CONTRIBUTION_RF
    )
    contribution_om = backend.tfd.BatchBroadcast(
        self.contribution_om,
        n_organic_media_channels,
        name=constants.CONTRIBUTION_OM,
    )
    contribution_orf = backend.tfd.BatchBroadcast(
        self.contribution_orf,
        n_organic_rf_channels,
        name=constants.CONTRIBUTION_ORF,
    )
    contribution_n = backend.tfd.BatchBroadcast(
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


class IndependentMultivariateDistribution(backend.tfd.Distribution):
  """Container for a joint distribution created from independent distributions.

  This class is useful when one wants to define a joint distribution for a
  Meridian prior, where the elements are not necessarily from the same
  distribution family. For example, to define a distribution where
  one element is Uniform and the second is triangular:

  ```python
  distributions = [
      tfp.distributions.Uniform(0.0, 1.0),
      tfp.distributions.Triangular(0.0, 1.0, 0.5)
      ]
  distribution = IndependentMultivariateDistribution(distributions)
  ```

  It is also possible to define a distribution where multiple elements come
  from the same distribution family. For example, to define a distribution where
  the three elements are LogNormal(0.2, 0.9), LogNormal(0, 0.5) and
  Gamma(2, 2):

  ```python
  distributions = [
      tfp.distributions.LogNormal([0.2, 0.0], [0.9, 0.5]),
      tfp.distributions.Gamma(2.0, 2.0)
      ]
  distribution = IndependentMultivariateDistribution(distributions)
  ```

  This class cannot contain instances of `tfd.Deterministic`.
  """

  def __init__(
      self,
      distributions: Sequence[backend.tfd.Distribution],
      validate_args: bool = False,
      allow_nan_stats: bool = True,
      name: str | None = None,
  ):
    """Initializes a batch of independent distributions from different families.

    Args:
      distributions: List of `tfd.Distribution` from which to construct a
        multivariate distribution. The distributions must have scalar or one
        dimensional batch shapes; the resulting batch shape will be the sum of
        the underlying batch shapes.
      validate_args: Python `bool`. When `True` distribution parameters are
        checked for validity despite possibly degrading runtime performance.
        When `False` invalid inputs may silently render incorrect outputs.
        Default value is `False`.
      allow_nan_stats: Python `bool`. When `True`, statistics (e.g., mean, mode,
        variance) use the value "`NaN`" to indicate the result is undefined.
        When `False`, an exception is raised if one or more of the statistic's
        batch members are undefined. Default value is `True`.
      name: Python `str` name prefixed to Ops created by this class. Default
        value is 'IndependentMultivariate' followed by the names of the
        underlying distributions.

    Raises:
        ValueError: If one or more distributions are instances of
        `tfd.Deterministic` or dtypes differ between the
        distributions.
    """
    parameters = dict(locals())

    self._verify_distributions(distributions)

    self._distributions = [
        dist
        if not dist.is_scalar_batch()
        else backend.tfd.BatchBroadcast(dist, (1,))
        for dist in distributions
    ]

    self._distribution_batch_shapes = self._get_distribution_batch_shapes()
    self._distribution_batch_shape_tensors = backend.concatenate(
        [dist.batch_shape_tensor() for dist in self._distributions],
        axis=0,
    )

    dtype = self._verify_dtypes()

    name = name or '-'.join(
        [constants.INDEPENDENT_MULTIVARIATE] + [d.name for d in distributions]
    )

    super().__init__(
        dtype=dtype,
        reparameterization_type=backend.tfd.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )

  def _verify_distributions(
      self, distributions: Sequence[backend.tfd.Distribution]
  ):
    """Check for deterministic distributions and raise an error if found."""

    if any(
        isinstance(dist, backend.tfd.Deterministic) for dist in distributions
    ):
      raise ValueError(
          f'{self.__class__.__name__} cannot contain `Deterministic` '
          'distributions. To implement a nearly deterministic element of this '
          'distribution, we recommend using `backend.tfd.Uniform` with a '
          'small range. For example to define a distribution that is nearly '
          '`Deterministic(1.0)`, use '
          '`tfp.distribution.Uniform(1.0 - 1e-9, 1.0 + 1e-9)`'
      )

  def _verify_dtypes(self) -> str:
    dtypes = [dist.dtype for dist in self._distributions]
    if len(set(dtypes)) != 1:
      raise ValueError(
          f'All distributions must have the same dtype. Found: {dtypes}.'
      )

    return backend.result_type(*dtypes)

  def _event_shape(self):
    return backend.TensorShape([])

  def _batch_shape_tensor(self):
    distribution_batch_shape_tensors = backend.concatenate(
        [dist.batch_shape_tensor() for dist in self._distributions],
        axis=0,
    )
    return backend.reduce_sum(distribution_batch_shape_tensors, keepdims=True)

  def _batch_shape(self):
    return backend.TensorShape(sum(self._distribution_batch_shapes))

  def _sample_n(self, n, seed=None):
    return backend.concatenate(
        [dist.sample(n, seed) for dist in self._distributions], axis=-1
    )

  def _quantile(self, value):
    value = self._broadcast_value(value)
    split_value = backend.split(value, self._distribution_batch_shapes, axis=-1)
    quantiles = [
        dist.quantile(sv) for dist, sv in zip(self._distributions, split_value)
    ]

    return backend.concatenate(quantiles, axis=-1)

  def _log_prob(self, value):
    value = self._broadcast_value(value)
    split_value = backend.split(value, self._distribution_batch_shapes, axis=-1)
    log_probs = [
        dist.log_prob(sv) for dist, sv in zip(self._distributions, split_value)
    ]

    return backend.concatenate(log_probs, axis=-1)

  def _log_cdf(self, value):
    value = self._broadcast_value(value)
    split_value = backend.split(value, self._distribution_batch_shapes, axis=-1)

    log_cdfs = [
        dist.log_cdf(sv) for dist, sv in zip(self._distributions, split_value)
    ]

    return backend.concatenate(log_cdfs, axis=-1)

  def _mean(self):
    return backend.concatenate(
        [dist.mean() for dist in self._distributions], axis=0
    )

  def _variance(self):
    return backend.concatenate(
        [dist.variance() for dist in self._distributions], axis=0
    )

  def _default_event_space_bijector(self):
    """Mapping from R^n to the event space of the wrapped distributions.

    This is the blockwise concatenation of the underlying bijectors.

    Returns:
      A `tfp.bijectors.Blockwise` object that concatenates the underlying
      bijectors.
    """
    bijectors = [
        d.experimental_default_event_space_bijector()
        for d in self._distributions
    ]

    return backend.bijectors.Blockwise(
        bijectors,
        block_sizes=self._distribution_batch_shapes,
    )

  def _broadcast_value(self, value: backend.Tensor) -> backend.Tensor:
    value = backend.to_tensor(value)
    broadcast_shape = backend.broadcast_dynamic_shape(
        value.shape, self.batch_shape_tensor()
    )
    return backend.broadcast_to(value, broadcast_shape)

  def _get_distribution_batch_shapes(self) -> Sequence[int]:
    """Sequence of batch shapes of underlying distributions."""

    batch_shapes = []

    for dist in self._distributions:
      try:
        (dist_batch_shape,) = dist.batch_shape
      except ValueError as exc:
        raise ValueError(
            'All distributions must be 0- or 1-dimensional.'
            f' Found {len(dist.batch_shape)}-dimensional distribution:'
            f' {dist.batch_shape}.'
        ) from exc
      else:
        batch_shapes.append(dist_batch_shape)

    return batch_shapes


def distributions_are_equal(
    a: backend.tfd.Distribution, b: backend.tfd.Distribution
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

  if a_params.keys() != b_params.keys():
    return False

  for key in a_params.keys():
    if isinstance(
        a_params[key], (backend.Tensor, np.ndarray, float, int)
    ) and isinstance(b_params[key], (backend.Tensor, np.ndarray, float, int)):
      if not backend.allclose(a_params[key], b_params[key]):
        return False
    else:
      if a_params[key] != b_params[key]:
        return False

  return True


def lognormal_dist_from_mean_std(
    mean: float | Sequence[float], std: float | Sequence[float]
) -> backend.tfd.LogNormal:
  """Define a lognormal distribution from its mean and standard deviation.

  This function parameterizes lognormal distributions by their mean and
  standard deviation.

  Args:
    mean: A float or array-like object defining the distribution mean. Must be
      positive.
    std: A float or array-like object defining the distribution standard
      deviation. Must be non-negative.

  Returns:
    A `backend.tfd.LogNormal` object with the input mean and standard deviation.
  """

  mean = np.asarray(mean)
  std = np.asarray(std)

  mu = np.log(mean) - 0.5 * np.log((std / mean) ** 2 + 1)
  sigma = np.sqrt(np.log((std / mean) ** 2 + 1))

  return backend.tfd.LogNormal(mu, sigma)


def lognormal_dist_from_range(
    low: float | Sequence[float],
    high: float | Sequence[float],
    mass_percent: float | Sequence[float] = 0.95,
) -> backend.tfd.LogNormal:
  """Define a LogNormal distribution from a specified range.

  This function parameterizes lognormal distributions by the bounds of a range,
  so that the specified probability mass falls within the bounds defined by
  `low` and `high`. The probability mass is symmetric about the median. For
  example, to define a lognormal distribution with a 95% probability mass of
  (1, 10), use:

  ```python
  lognormal = lognormal_dist_from_range(1.0, 10.0, mass_percent=0.95)
  ```

  Args:
    low: Float or array-like denoting the lower bound of the range. Values must
      be non-negative.
    high: Float or array-like denoting the upper bound of range. Values must be
      non-negative.
    mass_percent: Float or array-like denoting the probability mass. Values must
      be between 0 and 1 (exclusive). Default: 0.95.

  Returns:
    A `backend.tfd.LogNormal` object with the input percentage mass falling
      within the given range.
  """
  low = np.asarray(low)
  high = np.asarray(high)
  mass_percent = np.asarray(mass_percent)

  if not ((0.0 < low).all() and (low < high).all()):  # pytype: disable=attribute-error
    raise ValueError("'low' and 'high' values must be non-negative and satisfy "
                     "high > low.")

  if not ((0.0 < mass_percent).all() and (mass_percent < 1.0).all()):  # pytype: disable=attribute-error
    raise ValueError(
        "'mass_percent' values must be between 0 and 1, exclusive."
        )

  normal = backend.tfd.Normal(0, 1)
  mass_lower = 0.5 - (mass_percent / 2)
  mass_upper = 0.5 + (mass_percent / 2)

  sigma = np.log(high / low) / (
      normal.quantile(mass_upper) - normal.quantile(mass_lower)
  )
  mu = np.log(high) - normal.quantile(mass_upper) * sigma

  return backend.tfd.LogNormal(mu, sigma)


def _convert_to_deterministic_0_distribution(
    distribution: backend.tfd.Distribution,
) -> backend.tfd.Distribution:
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
      not isinstance(distribution, backend.tfd.Deterministic)
      or distribution.loc != 0
  ):
    warnings.warn(
        'Hierarchical distribution parameters must be deterministically zero'
        f' for national models. {distribution.name} has been automatically set'
        ' to Deterministic(0).'
    )
    return backend.tfd.Deterministic(loc=0, name=distribution.name)
  else:
    return distribution


def _get_total_media_contribution_prior(
    kpi: float,
    total_spend: np.ndarray,
    name: str,
    p_mean: float = constants.P_MEAN,
    p_sd: float = constants.P_SD,
) -> backend.tfd.Distribution:
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
  lognormal_sigma = backend.cast(
      np.sqrt(np.log(roi_sd**2 / roi_mean**2 + 1)), dtype=backend.float32
  )
  lognormal_mu = backend.cast(
      np.log(roi_mean * np.exp(-(lognormal_sigma**2) / 2)),
      dtype=backend.float32,
  )
  return backend.tfd.LogNormal(lognormal_mu, lognormal_sigma, name=name)


def _validate_support(
    parameter_name: str,
    tfp_dist: backend.tfp.distributions.Distribution,
    bounds: tuple[float, float],
    prevent_deterministic_prior_at_bounds: tuple[bool, bool],
) -> None:
  """Validates that distribution support is within the parameter bounds.

  Args:
    parameter_name: Name of the parameter.
    tfp_dist: The TFP distribution to validate.
    bounds: Tuple containing the min and max values of the parameteter space.
    prevent_deterministic_prior_at_bounds: Tuple of two booleans indicating
      whether a deterministic prior is allowed at the lower and upper bounds,
      respectively.

  Raises:
    ValueError: If the distribution support is not within the parameter bounds.
  """
  # Note that `tfp.distributions.BatchBroadcast` objects have a `distribution`
  # attribute that points to a `tfp.distributions.Distribution` object.
  if isinstance(tfp_dist, backend.tfd.BatchBroadcast):
    tfp_dist = tfp_dist.distribution
  # Note that `tfp.distributions.Deterministic` does not have a `quantile`
  # method implemented, so the min and max values must be extracted from the
  # `loc` attribute instead.
  if isinstance(tfp_dist, backend.tfd.Deterministic):
    support_min_vals = tfp_dist.loc
    support_max_vals = tfp_dist.loc
    for i in (0, 1):
      if prevent_deterministic_prior_at_bounds[i] and np.any(
          tfp_dist.loc == bounds[i]
      ):
        raise ValueError(
            f'{parameter_name} was assigned a point mass (deterministic) prior'
            f' at {bounds[i]}, which is not allowed.'
        )
  elif isinstance(tfp_dist, backend.tfd.TruncatedNormal):
    # TruncatedNormal quantile method is not reliable, particularly when the
    # `low` or `high` value falls into extreme percentile of the untruncated
    # distribution. Note that
    # `TruncatedNormal.experimental_default_event_space_bijector()([-inf, inf])`
    # returns the correct support range, so this method could be used if the
    # `quantile` method is found to be unreliable for other distributions.
    support_min_vals = tfp_dist.low
    support_max_vals = tfp_dist.high
  else:
    try:
      support_min_vals = tfp_dist.quantile(0)
      support_max_vals = tfp_dist.quantile(1)
    except (AttributeError, NotImplementedError):
      warnings.warn(
          f'The prior distribution for {parameter_name} does not have a'
          ' `quantile` method implemented, so the support range validation'
          f' was skipped. Confirm that your prior for {parameter_name} is'
          ' appropriate.'
      )
      return
  if np.any(support_min_vals < bounds[0]):
    raise ValueError(
        f'{parameter_name} was assigned a prior distribution that allows values'
        f' less than the parameter minimum {bounds[0]}.'
    )
  if np.any(support_max_vals > bounds[1]):
    raise ValueError(
        f'{parameter_name} was assigned a prior distribution that allows values'
        f' greater than the parameter maximum {bounds[1]}.'
    )


# Dictionary of parameters that have a limited parameters space. The tuple
# contains the lower and upper bounds, respectively.
_parameter_space_bounds = {
    'eta_m': (0, np.inf),
    'eta_rf': (0, np.inf),
    'eta_om': (0, np.inf),
    'eta_orf': (0, np.inf),
    'xi_c': (0, np.inf),
    'xi_n': (0, np.inf),
    'alpha_m': (0, 1),
    'alpha_rf': (0, 1),
    'alpha_om': (0, 1),
    'alpha_orf': (0, 1),
    'ec_m': (0, np.inf),
    'ec_rf': (0, np.inf),
    'ec_om': (0, np.inf),
    'ec_orf': (0, np.inf),
    'slope_m': (0, np.inf),
    'slope_rf': (0, np.inf),
    'slope_om': (0, np.inf),
    'slope_orf': (0, np.inf),
    'sigma': (0, np.inf),
}

# Dictionary of parameters that do not allow a deterministic prior at one or
# more of the parameter space bounds. The boolean tuple indicates whether a
# deterministic prior is allowed at the lower bound or upper bound,
# respectively, where `True` means "not allowed". This check is specifically for
# point mass at finite paramteter space bounds, since point mass at infinity is
# generally problematic for all parameters. Note that `sigma` should generally
# not have point mass at zero, but this is not checked here because unit tests
# require the ability to simulate data with `sigma` set to zero.
_prevent_deterministic_prior_at_bounds = {
    'alpha_m': (False, True),
    'alpha_rf': (False, True),
    'alpha_om': (False, True),
    'alpha_orf': (False, True),
    'ec_m': (True, False),
    'ec_rf': (True, False),
    'ec_om': (True, False),
    'ec_orf': (True, False),
    'slope_m': (True, False),
    'slope_rf': (True, False),
    'slope_om': (True, False),
    'slope_orf': (True, False),
}
