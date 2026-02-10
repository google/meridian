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

"""Classes and functions common to modules in this directory."""

import enum

from meridian import constants
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_pb
import xarray as xr


__all__ = [
    "TargetMetric",
    "KpiType",
    "to_estimate",
]


class TargetMetric(enum.Enum):
  KPI = target_pb.TargetMetric.KPI
  ROI = target_pb.TargetMetric.ROI
  MARGINAL_ROI = target_pb.TargetMetric.MARGINAL_ROI


@enum.unique
class KpiType(enum.Enum):
  """Enum for KPI type used in analysis and optimization."""

  REVENUE = kpi_type_pb.KpiType.REVENUE
  NON_REVENUE = kpi_type_pb.KpiType.NON_REVENUE


def to_estimate(
    dataarray: xr.DataArray,
    confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL,
    metric: str = constants.MEAN,
) -> estimate_pb.Estimate:
  """Converts a DataArray with (mean [or median for CPIK], ci_lo, ci_hi) `metric` data vars."""
  value = dataarray.sel(metric=metric).item()
  estimate = estimate_pb.Estimate(
      value=value
  )
  uncertainty = estimate_pb.Estimate.Uncertainty(
      probability=confidence_level,
      lowerbound=dataarray.sel(metric=constants.CI_LO).item(),
      upperbound=dataarray.sel(metric=constants.CI_HI).item(),
  )
  estimate.uncertainties.append(uncertainty)

  return estimate
