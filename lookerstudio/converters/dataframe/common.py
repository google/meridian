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

"""For common utility functions in this package."""
import re

from lookerstudio.converters.dataframe import constants as dc
from mmm.v1.common import target_metric_pb2 as target_metric_pb


def map_target_metric_str(metric: target_metric_pb.TargetMetric) -> str:
  match metric:
    case target_metric_pb.TargetMetric.KPI:
      return dc.OPTIMIZATION_SPEC_TARGET_METRIC_KPI
    case target_metric_pb.TargetMetric.ROI:
      return dc.OPTIMIZATION_SPEC_TARGET_METRIC_ROI
    case target_metric_pb.TargetMetric.MARGINAL_ROI:
      return dc.OPTIMIZATION_SPEC_TARGET_METRIC_MARGINAL_ROI
    case target_metric_pb.TargetMetric.COST_PER_INCREMENTAL_KPI:
      return dc.OPTIMIZATION_SPEC_TARGET_METRIC_CPIK
    case _:
      raise ValueError(f"Unsupported target metric: {metric}")


def _to_sheet_name_format(s: str) -> str:
  """Converts a string to a sheet name format.

  Replace consecutive spaces with a single underscore using regex.

  Args:
    s: The string to convert.

  Returns:
    The converted sheet name.
  """
  return re.sub(r"\s+", dc.SHEET_NAME_DELIMITER, s)


def create_grid_sheet_name(prefix: str, grid_name: str) -> str:
  """Creates a grid sheet name with the given prefix and grid name.

  Args:
    prefix: The prefix of the sheet name.
    grid_name: The name of the grid.

  Returns:
    The grid sheet name.
  """
  grid_sheet_name = _to_sheet_name_format(grid_name)
  sheet_prefix = _to_sheet_name_format(prefix)
  return f"{sheet_prefix}_{grid_sheet_name}"
