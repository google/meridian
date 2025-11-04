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

"""Meridian EDA Spec."""

import dataclasses
from typing import Any, Callable, Dict, TypeAlias

__all__ = [
    "AggregationConfig",
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
class VIFSpec:
  """A spec for the EDA VIF check.

  Attributes:
    geo_threshold: The threshold for geo-level VIF.
    overall_threshold: The threshold for overall VIF.
    national_threshold: The threshold for national VIF.
  """

  geo_threshold: float = _DEFAULT_VIF_THRESHOLD
  overall_threshold: float = _DEFAULT_VIF_THRESHOLD
  national_threshold: float = _DEFAULT_VIF_THRESHOLD


@dataclasses.dataclass(frozen=True, kw_only=True)
class EDASpec:
  """A container for all user-configurable EDA check specs.

  This object allows users to customize the behavior of the EDA checks
  by passing a single configuration object into the EDAEngine constructor,
  avoiding a large number of arguments.

  Attributes:
    aggregation_config: A configuration object for custom aggregation functions.
    vif_spec: A configuration object for the EDA VIF check.
  """

  aggregation_config: AggregationConfig = dataclasses.field(
      default_factory=AggregationConfig
  )
  vif_spec: VIFSpec = dataclasses.field(default_factory=VIFSpec)
