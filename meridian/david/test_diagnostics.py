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

"""Tests for diagnostics helpers."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from meridian.david import diagnostics


def _expected_dw(y, yhat):
  r = np.asarray(y) - np.asarray(yhat)
  return np.sum(np.diff(r) ** 2) / np.sum(r ** 2)


def _base_observed():
  return xr.DataArray([1.0, 2.0, 3.0], dims=["time"])


def test_durbin_watson_from_idata_posterior_predictive():
  yobs = _base_observed()
  yhat = xr.DataArray(
      np.ones((2, 2, 3)), dims=("chain", "draw", "time")
  )
  idata = az.InferenceData(
      observed_data=xr.Dataset({"kpi": yobs}),
      posterior_predictive=xr.Dataset({"kpi": yhat}),
  )
  dw = diagnostics.durbin_watson_from_idata(idata)
  assert np.isclose(dw, _expected_dw(yobs, np.ones(3)))


def test_durbin_watson_from_idata_predictions_fallback():
  yobs = _base_observed()
  yhat = xr.DataArray(np.ones((2, 3)), dims=("draw", "time"))
  idata = az.InferenceData(observed_data=xr.Dataset({"kpi": yobs}))
  idata.add_groups(predictions=xr.Dataset({"kpi": yhat}))
  dw = diagnostics.durbin_watson_from_idata(idata)
  assert np.isclose(dw, _expected_dw(yobs, np.ones(3)))


def test_durbin_watson_from_idata_posterior_fallback():
  yobs = _base_observed()
  yhat = xr.DataArray(
      np.ones((2, 2, 3)), dims=("chain", "draw", "time")
  )
  post = xr.Dataset({"kpi_hat": yhat})
  idata = az.InferenceData(
      observed_data=xr.Dataset({"kpi": yobs}), posterior=post
  )
  dw = diagnostics.durbin_watson_from_idata(idata)
  assert np.isclose(dw, _expected_dw(yobs, np.ones(3)))


def test_durbin_watson_from_idata_missing_predictions():
  yobs = _base_observed()
  idata = az.InferenceData(observed_data=xr.Dataset({"kpi": yobs}))
  with pytest.raises(AttributeError):
    diagnostics.durbin_watson_from_idata(idata)

