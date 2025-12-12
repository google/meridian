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

"""This module contains common validation functions for Meridian data."""

import datetime as dt
from meridian import constants
import xarray as xr


def validate_time_coord_format(array: xr.DataArray | None):
  """Validates the `time` dimensions format of the selected DataArray.

  The `time` dimension of the selected array must have labels that are
  formatted in the Meridian conventional `"yyyy-mm-dd"` format.

  Args:
    array: An optional DataArray to validate.
  """
  if array is None:
    return

  # The component data arrays from the input data builders that call this helper
  # method should only have one of either `media_time` or `time` as its time
  # dimension.
  target_coords = [constants.TIME, constants.MEDIA_TIME]

  for coord_name in target_coords:
    if (values := array.coords.get(coord_name)) is not None:
      for time in values:
        try:
          dt.datetime.strptime(time.item(), constants.DATE_FORMAT)
        except (TypeError, ValueError) as exc:
          raise ValueError(
              f"Invalid {coord_name} label: {time.item()!r}. "
              f"Expected format: '{constants.DATE_FORMAT}'"
          ) from exc
