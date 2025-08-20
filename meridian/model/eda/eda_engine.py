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

"""Meridian EDA Engine."""

import functools
from meridian import constants
from meridian.model import model
import xarray as xr

class EDAEngine:
  """Meridian EDA Engine."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian

  @functools.cached_property
  def controls_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.controls is None:
      return None
    controls_scaled_da = self._meridian.input_data.controls.copy()
    controls_scaled_da.values = self._meridian.controls_scaled
    return controls_scaled_da

  @functools.cached_property
  def media_raw_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    return self._truncate_media_time(self._meridian.input_data.media)

  @functools.cached_property
  def media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    media_scaled_da = self._meridian.input_data.media.copy()
    media_scaled_da.values = self._meridian.media_tensors.media_scaled
    return self._truncate_media_time(media_scaled_da)

  @functools.cached_property
  def media_spend_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media_spend is None:
      return None
    media_spend_da = self._meridian.input_data.media_spend.copy()
    media_spend_da.values = self._meridian.media_tensors.media_spend
    return media_spend_da

  def _truncate_media_time(self, da: xr.DataArray) -> xr.DataArray:
    """Truncates the first `start` elements of the media time of a variable."""
    if constants.MEDIA_TIME not in da.coords:
      return da.copy()
    start = self._meridian.n_media_times - self._meridian.n_times
    da = da.isel({constants.MEDIA_TIME: slice(start, None)})
    da = da.rename({constants.MEDIA_TIME: constants.TIME})
    return da
