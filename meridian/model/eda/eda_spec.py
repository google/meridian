# Copyright 2026 The Meridian Authors.
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

"""Meridian EDA Spec."""

import dataclasses
from typing import Any, Callable, Dict, TypeAlias

from meridian.model.eda import constants as eda_constants

__all__ = [
    "AggregationConfig",
    "KpiInvariabilitySpec",
    "PairwiseCorrSpec",
    "StandardDeviationSpec",
    "VIFSpec",
    "EDASpec",
]

AggregationFn: TypeAlias = Callable[..., Any]
AggregationMap: TypeAlias = Dict[str, AggregationFn]
_DEFAULT_VIF_THRESHOLD = 1000


@dataclasses.dataclass(frozen=True, kw_only=True)
class AggregationConfig:
  """A configuration for customizing variable aggregation functions.

  The aggregation function can be called in the form `f(x, axis=axis, **kwargs)`
  to return the result of reducing an `np.ndarray` over an integer valued axis.
  It's recommended to explicitly define the aggregation functions instead of
  using lambdas.

  Attributes:
    control_variables: A dictionary mapping control variable names to
      aggregation functions. Defaults to `np.sum` if a variable is not
      specified.
    non_media_treatments: A dictionary mapping non-media variable names to
      aggregation functions. Defaults to `np.sum` if a variable is not
      specified.
  """

  control_variables: AggregationMap = dataclasses.field(default_factory=dict)
  non_media_treatments: AggregationMap = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class KpiInvariabilitySpec:
  """A spec for the EDA KPI invariability check.

  Attributes:
    std_threshold: The threshold for KPI standard deviation. Exceeding this
      threshold triggers an ERROR.
  """

  std_threshold: float = eda_constants.STD_THRESHOLD


@dataclasses.dataclass(frozen=True, kw_only=True)
class PairwiseCorrSpec:
  """A spec for the EDA pairwise correlation check.

  Attributes:
    overall_threshold: The threshold for overall pairwise correlation. Exceeding
      this threshold triggers an ERROR.
    geo_threshold: The threshold for geo-level pairwise correlation. Exceeding
      this threshold triggers an ATTENTION.
    national_threshold: The threshold for national pairwise correlation.
      Exceeding this threshold triggers an ERROR.
  """

  overall_threshold: float = eda_constants.OVERALL_PAIRWISE_CORR_THRESHOLD
  geo_threshold: float = eda_constants.GEO_PAIRWISE_CORR_THRESHOLD
  national_threshold: float = eda_constants.NATIONAL_PAIRWISE_CORR_THRESHOLD


@dataclasses.dataclass(frozen=True, kw_only=True)
class StandardDeviationSpec:
  """A spec for the EDA standard deviation check.

  Attributes:
    geo_std_threshold: The threshold for geo-level standard deviation. Falling
      below this threshold triggers an ATTENTION.
    national_std_threshold: The threshold for national standard deviation.
      Falling below this threshold triggers an ATTENTION.
  """

  geo_std_threshold: float = eda_constants.STD_THRESHOLD
  national_std_threshold: float = eda_constants.STD_THRESHOLD


@dataclasses.dataclass(frozen=True, kw_only=True)
class VIFSpec:
  """A spec for the EDA VIF check.

  Attributes:
    geo_threshold: The threshold for geo-level VIF. Exceeding this threshold
      triggers an ATTENTION.
    overall_threshold: The threshold for overall VIF. Exceeding this threshold
      triggers an ERROR.
    national_threshold: The threshold for national VIF. Exceeding this threshold
      triggers an ERROR.
    std_threshold: The threshold for standard deviation. Used to determine if a
      variable is a constant.
  """

  geo_threshold: float = _DEFAULT_VIF_THRESHOLD
  overall_threshold: float = _DEFAULT_VIF_THRESHOLD
  national_threshold: float = _DEFAULT_VIF_THRESHOLD
  std_threshold: float = eda_constants.STD_THRESHOLD


@dataclasses.dataclass(frozen=True, kw_only=True)
class EDASpec:
  """A container for all user-configurable EDA check specs.

  This object allows users to customize the behavior of the EDA checks
  by passing a single configuration object into the EDAEngine constructor,
  avoiding a large number of arguments.

  Attributes:
    aggregation_config: A configuration object for custom aggregation functions.
    kpi_invariability_spec: A configuration object for the EDA KPI invariability
      check.
    pairwise_corr_spec: A configuration object for the EDA pairwise correlation
      check.
    std_spec: A configuration object for the EDA standard deviation check.
    vif_spec: A configuration object for the EDA VIF check.
  """

  aggregation_config: AggregationConfig = dataclasses.field(
      default_factory=AggregationConfig
  )
  kpi_invariability_spec: KpiInvariabilitySpec = dataclasses.field(
      default_factory=KpiInvariabilitySpec
  )
  pairwise_corr_spec: PairwiseCorrSpec = dataclasses.field(
      default_factory=PairwiseCorrSpec
  )
  std_spec: StandardDeviationSpec = dataclasses.field(
      default_factory=StandardDeviationSpec
  )
  vif_spec: VIFSpec = dataclasses.field(default_factory=VIFSpec)
