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

"""Utility functions for testing EDA modules."""

from meridian import constants
import numpy as np
import pandas as pd
import xarray as xr


def construct_coords(
    dims: list[str],
    n_geos: int,
    n_times: int,
    n_vars: int,
    var_name: str,
) -> dict[str, list[str]]:
  """Helper to construct the coordinates of a DataArray."""
  coords = {}
  for dim in dims:
    if dim == constants.TIME:
      coords[dim] = pd.date_range(start="2023-01-01", periods=n_times, freq="W")
    elif dim == constants.GEO:
      coords[dim] = [f"{constants.GEO}{i}" for i in range(n_geos)]
    else:
      coords[dim] = [f"{var_name}_{i+1}" for i in range(n_vars)]
  return coords


def construct_dims_and_shapes(
    data_shape: tuple[int, ...], var_name: str | None = None
):
  """Helper to construct the dimensions of a DataArray."""
  ndim = len(data_shape)
  if var_name is None:
    n_vars = 0
    if ndim == 2:
      dims = [constants.GEO, constants.TIME]
      n_geos, n_times = data_shape
    elif ndim == 1:
      dims = [constants.TIME]
      n_geos, n_times = 0, data_shape[0]
    else:
      raise ValueError(f"Unsupported data shape: {data_shape}")
  else:
    var_dim_name = f"{var_name}_dim"
    if ndim == 3:
      dims = [constants.GEO, constants.TIME, var_dim_name]
      n_geos, n_times, n_vars = data_shape
    elif ndim == 2:
      dims = [constants.TIME, var_dim_name]
      n_times, n_vars = data_shape
      n_geos = 0
    else:
      raise ValueError(f"Unsupported data shape: {data_shape}")

  return dims, n_geos, n_times, n_vars


def create_dataset_with_var_dim(
    data: np.ndarray, var_name: str = "media"
) -> xr.Dataset:
  """Helper to create a dataset with a single variable dimension."""
  dims, n_geos, n_times, n_vars = construct_dims_and_shapes(
      data.shape, var_name
  )
  coords = construct_coords(dims, n_geos, n_times, n_vars, var_name)
  xarray_data_vars = {var_name: (dims, data)}

  return xr.Dataset(data_vars=xarray_data_vars, coords=coords)


def create_data_array_with_var_dim(
    data: np.ndarray, name: str, var_name: str | None = None
) -> xr.DataArray:
  """Helper to create a data array with a single variable dimension."""
  dims, n_geos, n_times, n_vars = construct_dims_and_shapes(
      data.shape, var_name
  )
  if var_name is None:
    var_name = name
  coords = construct_coords(dims, n_geos, n_times, n_vars, var_name)

  return xr.DataArray(data, name=name, dims=dims, coords=coords)
