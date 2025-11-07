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

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Optional, Protocol, Sequence

from meridian import backend
from meridian import constants
from meridian.model import transformers
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import outliers_influence
import xarray as xr


if typing.TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top

__all__ = ['EDAEngine', 'GeoLevelCheckOnNationalModelError']

_DEFAULT_DA_VAR_AGG_FUNCTION = np.sum
_CORRELATION_COL_NAME = eda_constants.CORRELATION
_STACK_VAR_COORD_NAME = eda_constants.VARIABLE
_CORR_VAR1 = eda_constants.VARIABLE_1
_CORR_VAR2 = eda_constants.VARIABLE_2
_CORRELATION_MATRIX_NAME = 'correlation_matrix'
_OVERALL_PAIRWISE_CORR_THRESHOLD = 0.999
_GEO_PAIRWISE_CORR_THRESHOLD = 0.999
_NATIONAL_PAIRWISE_CORR_THRESHOLD = 0.999
_EMPTY_DF_FOR_EXTREME_CORR_PAIRS = pd.DataFrame(
    columns=[_CORR_VAR1, _CORR_VAR2, _CORRELATION_COL_NAME]
)
_Q1_THRESHOLD = 0.25
_Q3_THRESHOLD = 0.75
_IQR_MULTIPLIER = 1.5
_STD_WITH_OUTLIERS_VAR_NAME = 'std_with_outliers'
_STD_WITHOUT_OUTLIERS_VAR_NAME = 'std_without_outliers'
_STD_THRESHOLD = 1e-4
_OUTLIERS_COL_NAME = 'outliers'
_ABS_OUTLIERS_COL_NAME = 'abs_outliers'
_VIF_COL_NAME = 'VIF'


class _NamedEDACheckCallable(Protocol):
  """A callable that returns an EDAOutcome and has a __name__ attribute."""

  __name__: str

  def __call__(self) -> eda_outcome.EDAOutcome:
    ...


class GeoLevelCheckOnNationalModelError(Exception):
  """Raised when a geo-level check is called on a national model."""

  pass


@dataclasses.dataclass(frozen=True)
class _RFNames:
  """Holds constant names for reach and frequency data arrays."""

  reach: str
  reach_scaled: str
  frequency: str
  impressions: str
  impressions_scaled: str
  national_reach: str
  national_reach_scaled: str
  national_frequency: str
  national_impressions: str
  national_impressions_scaled: str


_ORGANIC_RF_NAMES = _RFNames(
    reach=constants.ORGANIC_REACH,
    reach_scaled=constants.ORGANIC_REACH_SCALED,
    frequency=constants.ORGANIC_FREQUENCY,
    impressions=constants.ORGANIC_RF_IMPRESSIONS,
    impressions_scaled=constants.ORGANIC_RF_IMPRESSIONS_SCALED,
    national_reach=constants.NATIONAL_ORGANIC_REACH,
    national_reach_scaled=constants.NATIONAL_ORGANIC_REACH_SCALED,
    national_frequency=constants.NATIONAL_ORGANIC_FREQUENCY,
    national_impressions=constants.NATIONAL_ORGANIC_RF_IMPRESSIONS,
    national_impressions_scaled=constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED,
)


_RF_NAMES = _RFNames(
    reach=constants.REACH,
    reach_scaled=constants.REACH_SCALED,
    frequency=constants.FREQUENCY,
    impressions=constants.RF_IMPRESSIONS,
    impressions_scaled=constants.RF_IMPRESSIONS_SCALED,
    national_reach=constants.NATIONAL_REACH,
    national_reach_scaled=constants.NATIONAL_REACH_SCALED,
    national_frequency=constants.NATIONAL_FREQUENCY,
    national_impressions=constants.NATIONAL_RF_IMPRESSIONS,
    national_impressions_scaled=constants.NATIONAL_RF_IMPRESSIONS_SCALED,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReachFrequencyData:
  """Holds reach and frequency data arrays.

  Attributes:
    reach_raw_da: Raw reach data.
    reach_scaled_da: Scaled reach data.
    national_reach_raw_da: National raw reach data.
    national_reach_scaled_da: National scaled reach data.
    frequency_da: Frequency data.
    national_frequency_da: National frequency data.
    rf_impressions_scaled_da: Scaled reach * frequency impressions data.
    national_rf_impressions_scaled_da: National scaled reach * frequency
      impressions data.
    rf_impressions_raw_da: Raw reach * frequency impressions data.
    national_rf_impressions_raw_da: National raw reach * frequency impressions
      data.
  """

  reach_raw_da: xr.DataArray
  reach_scaled_da: xr.DataArray
  national_reach_raw_da: xr.DataArray
  national_reach_scaled_da: xr.DataArray
  frequency_da: xr.DataArray
  national_frequency_da: xr.DataArray
  rf_impressions_scaled_da: xr.DataArray
  national_rf_impressions_scaled_da: xr.DataArray
  rf_impressions_raw_da: xr.DataArray
  national_rf_impressions_raw_da: xr.DataArray


def _data_array_like(
    *, da: xr.DataArray, values: np.ndarray | backend.Tensor
) -> xr.DataArray:
  """Returns a DataArray from `values` with the same structure as `da`.

  Args:
    da: The DataArray whose structure (dimensions, coordinates, name, and attrs)
      will be used for the new DataArray.
    values: The numpy array or backend tensor to use as the values for the new
      DataArray.

  Returns:
    A new DataArray with the provided `values` and the same structure as `da`.
  """
  return xr.DataArray(
      values,
      coords=da.coords,
      dims=da.dims,
      name=da.name,
      attrs=da.attrs,
  )


def _stack_variables(
    ds: xr.Dataset, coord_name: str = _STACK_VAR_COORD_NAME
) -> xr.DataArray:
  """Stacks data variables other than time and geo into a single variable."""
  dims = []
  coords = []
  sample_dims = []
  # Dimensions have the same names as the coordinates.
  for dim in ds.dims:
    if dim in [constants.TIME, constants.GEO]:
      sample_dims.append(dim)
      continue
    dims.append(dim)
    coords.extend(ds.coords[dim].values.tolist())

  da = ds.to_stacked_array(coord_name, sample_dims=sample_dims)
  da = da.reset_index(dims, drop=True).assign_coords({coord_name: coords})
  return da


def _compute_correlation_matrix(
    input_da: xr.DataArray, dims: str | Sequence[str]
) -> xr.DataArray:
  """Computes the correlation matrix for variables in a DataArray.

  Args:
    input_da: An xr.DataArray containing variables for which to compute
      correlations.
    dims: Dimensions along which to compute correlations. Can only be TIME or
      GEO.

  Returns:
    An xr.DataArray containing the correlation matrix.
  """
  # Create two versions for correlation
  da1 = input_da.rename({_STACK_VAR_COORD_NAME: _CORR_VAR1})
  da2 = input_da.rename({_STACK_VAR_COORD_NAME: _CORR_VAR2})

  # Compute pairwise correlation across dims. Other dims are broadcasted.
  corr_mat_da = xr.corr(da1, da2, dim=dims)
  corr_mat_da.name = _CORRELATION_MATRIX_NAME
  return corr_mat_da


def _get_upper_triangle_corr_mat(corr_mat_da: xr.DataArray) -> xr.DataArray:
  """Gets the upper triangle of a correlation matrix.

  Args:
    corr_mat_da: An xr.DataArray containing the correlation matrix.

  Returns:
    An xr.DataArray containing only the elements in the upper triangle of the
    correlation matrix, with other elements masked as NaN.
  """
  n_vars = corr_mat_da.sizes[_CORR_VAR1]
  mask_np = np.triu(np.ones((n_vars, n_vars), dtype=bool), k=1)
  mask = xr.DataArray(
      mask_np,
      dims=[_CORR_VAR1, _CORR_VAR2],
      coords={
          _CORR_VAR1: corr_mat_da[_CORR_VAR1],
          _CORR_VAR2: corr_mat_da[_CORR_VAR2],
      },
  )
  return corr_mat_da.where(mask)


def _find_extreme_corr_pairs(
    extreme_corr_da: xr.DataArray, extreme_corr_threshold: float
) -> pd.DataFrame:
  """Finds extreme correlation pairs in a correlation matrix."""
  corr_tri = _get_upper_triangle_corr_mat(extreme_corr_da)
  extreme_corr_da = corr_tri.where(abs(corr_tri) > extreme_corr_threshold)

  df = extreme_corr_da.to_dataframe(name=_CORRELATION_COL_NAME).dropna()
  if df.empty:
    return _EMPTY_DF_FOR_EXTREME_CORR_PAIRS.copy()
  return df.sort_values(
      by=_CORRELATION_COL_NAME, ascending=False, inplace=False
  )


def _calculate_std(
    input_da: xr.DataArray,
) -> tuple[xr.Dataset, pd.DataFrame]:
  """Helper function to compute std with and without outliers.

  Args:
    input_da: A DataArray for which to calculate the std.

  Returns:
    A tuple where the first element is a Dataset with two data variables:
    'std_incl_outliers' and 'std_excl_outliers'. The second element is a
    DataFrame with columns for variables, geo (if applicable), time, and
    outlier values.
  """
  std_with_outliers = input_da.std(dim=constants.TIME, ddof=1)

  # TODO: Allow users to specify custom outlier definitions.
  q1 = input_da.quantile(_Q1_THRESHOLD, dim=constants.TIME)
  q3 = input_da.quantile(_Q3_THRESHOLD, dim=constants.TIME)
  iqr = q3 - q1
  lower_bound = q1 - _IQR_MULTIPLIER * iqr
  upper_bound = q3 + _IQR_MULTIPLIER * iqr

  da_no_outlier = input_da.where(
      (input_da >= lower_bound) & (input_da <= upper_bound)
  )
  std_without_outliers = da_no_outlier.std(dim=constants.TIME, ddof=1)

  std_ds = xr.Dataset({
      _STD_WITH_OUTLIERS_VAR_NAME: std_with_outliers,
      _STD_WITHOUT_OUTLIERS_VAR_NAME: std_without_outliers,
  })

  outlier_da = input_da.where(
      (input_da < lower_bound) | (input_da > upper_bound)
  )

  outlier_df = outlier_da.to_dataframe(name=_OUTLIERS_COL_NAME).dropna()
  outlier_df = outlier_df.assign(
      **{_ABS_OUTLIERS_COL_NAME: np.abs(outlier_df[_OUTLIERS_COL_NAME])}
  ).sort_values(by=_ABS_OUTLIERS_COL_NAME, ascending=False, inplace=False)

  return std_ds, outlier_df


def _calculate_vif(input_da: xr.DataArray, var_dim: str) -> xr.DataArray:
  """Helper function to compute variance inflation factor.

  Args:
    input_da: A DataArray for which to calculate the VIF over sample dimensions
      (e.g. time and geo if applicable).
    var_dim: The dimension name of the variable to compute VIF for.

  Returns:
    A DataArray containing the VIF for each variable in the variable dimension.
  """
  num_vars = input_da.sizes[var_dim]
  np_data = input_da.values.reshape(-1, num_vars)
  np_data_with_const = sm.add_constant(np_data, prepend=True)

  # Compute VIF for each variable excluding const which is the first one in the
  # 'variable' dimension.
  vifs = [
      outliers_influence.variance_inflation_factor(np_data_with_const, i)
      for i in range(1, num_vars + 1)
  ]

  vif_da = xr.DataArray(
      vifs,
      coords={var_dim: input_da[var_dim].values},
      dims=[var_dim],
  )
  return vif_da


class EDAEngine:
  """Meridian EDA Engine."""

  def __init__(
      self,
      meridian: model.Meridian,
      spec: eda_spec.EDASpec = eda_spec.EDASpec(),
  ):
    self._meridian = meridian
    self._spec = spec
    self._agg_config = self._spec.aggregation_config

  @property
  def spec(self) -> eda_spec.EDASpec:
    return self._spec

  @property
  def _is_national_data(self) -> bool:
    return self._meridian.is_national

  @functools.cached_property
  def controls_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.controls is None:
      return None
    controls_scaled_da = _data_array_like(
        da=self._meridian.input_data.controls,
        values=self._meridian.controls_scaled,
    )
    controls_scaled_da.name = constants.CONTROLS_SCALED
    return controls_scaled_da

  @functools.cached_property
  def national_controls_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled controls data array."""
    if self._meridian.input_data.controls is None:
      return None
    if self._is_national_data:
      if self.controls_scaled_da is None:
        # This case should be impossible given the check above.
        raise RuntimeError(
            'controls_scaled_da is None when controls is not None.'
        )
      national_da = self.controls_scaled_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_CONTROLS_SCALED
    else:
      national_da = self._aggregate_and_scale_geo_da(
          self._meridian.input_data.controls,
          constants.NATIONAL_CONTROLS_SCALED,
          transformers.CenteringAndScalingTransformer,
          constants.CONTROL_VARIABLE,
          self._agg_config.control_variables,
      )
    return national_da

  @functools.cached_property
  def media_raw_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    raw_media_da = self._truncate_media_time(self._meridian.input_data.media)
    raw_media_da.name = constants.MEDIA
    return raw_media_da

  @functools.cached_property
  def media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.media is None:
      return None
    media_scaled_da = _data_array_like(
        da=self._meridian.input_data.media,
        values=self._meridian.media_tensors.media_scaled,
    )
    media_scaled_da.name = constants.MEDIA_SCALED
    return self._truncate_media_time(media_scaled_da)

  @functools.cached_property
  def media_spend_da(self) -> xr.DataArray | None:
    """Returns media spend.

    If the input spend is aggregated, it is allocated across geo and time
    proportionally to media units.
    """
    # No need to truncate the media time for media spend.
    da = self._meridian.input_data.allocated_media_spend
    if da is None:
      return None
    da = da.copy()
    da.name = constants.MEDIA_SPEND
    return da

  @functools.cached_property
  def national_media_spend_da(self) -> xr.DataArray | None:
    """Returns the national media spend data array."""
    if self.media_spend_da is None:
      return None
    if self._is_national_data:
      national_da = self.media_spend_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_MEDIA_SPEND
    else:
      national_da = self._aggregate_and_scale_geo_da(
          self._meridian.input_data.allocated_media_spend,
          constants.NATIONAL_MEDIA_SPEND,
          None,
      )
    return national_da

  @functools.cached_property
  def national_media_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw media data array."""
    if self.media_raw_da is None:
      return None
    if self._is_national_data:
      national_da = self.media_raw_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_MEDIA
    else:
      # Note that media is summable by assumption.
      national_da = self._aggregate_and_scale_geo_da(
          self.media_raw_da,
          constants.NATIONAL_MEDIA,
          None,
      )
    return national_da

  @functools.cached_property
  def national_media_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled media data array."""
    if self.media_scaled_da is None:
      return None
    if self._is_national_data:
      national_da = self.media_scaled_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_MEDIA_SCALED
    else:
      # Note that media is summable by assumption.
      national_da = self._aggregate_and_scale_geo_da(
          self.media_raw_da,
          constants.NATIONAL_MEDIA_SCALED,
          transformers.MediaTransformer,
      )
    return national_da

  @functools.cached_property
  def organic_media_raw_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.organic_media is None:
      return None
    raw_organic_media_da = self._truncate_media_time(
        self._meridian.input_data.organic_media
    )
    raw_organic_media_da.name = constants.ORGANIC_MEDIA
    return raw_organic_media_da

  @functools.cached_property
  def organic_media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.organic_media is None:
      return None
    organic_media_scaled_da = _data_array_like(
        da=self._meridian.input_data.organic_media,
        values=self._meridian.organic_media_tensors.organic_media_scaled,
    )
    organic_media_scaled_da.name = constants.ORGANIC_MEDIA_SCALED
    return self._truncate_media_time(organic_media_scaled_da)

  @functools.cached_property
  def national_organic_media_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw organic media data array."""
    if self.organic_media_raw_da is None:
      return None
    if self._is_national_data:
      national_da = self.organic_media_raw_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_ORGANIC_MEDIA
    else:
      # Note that organic media is summable by assumption.
      national_da = self._aggregate_and_scale_geo_da(
          self.organic_media_raw_da, constants.NATIONAL_ORGANIC_MEDIA, None
      )
    return national_da

  @functools.cached_property
  def national_organic_media_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled organic media data array."""
    if self.organic_media_scaled_da is None:
      return None
    if self._is_national_data:
      national_da = self.organic_media_scaled_da.squeeze(
          constants.GEO, drop=True
      )
      national_da.name = constants.NATIONAL_ORGANIC_MEDIA_SCALED
    else:
      # Note that organic media is summable by assumption.
      national_da = self._aggregate_and_scale_geo_da(
          self.organic_media_raw_da,
          constants.NATIONAL_ORGANIC_MEDIA_SCALED,
          transformers.MediaTransformer,
      )
    return national_da

  @functools.cached_property
  def non_media_scaled_da(self) -> xr.DataArray | None:
    if self._meridian.input_data.non_media_treatments is None:
      return None
    non_media_scaled_da = _data_array_like(
        da=self._meridian.input_data.non_media_treatments,
        values=self._meridian.non_media_treatments_normalized,
    )
    non_media_scaled_da.name = constants.NON_MEDIA_TREATMENTS_SCALED
    return non_media_scaled_da

  @functools.cached_property
  def national_non_media_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled non-media treatment data array."""
    if self._meridian.input_data.non_media_treatments is None:
      return None
    if self._is_national_data:
      if self.non_media_scaled_da is None:
        # This case should be impossible given the check above.
        raise RuntimeError(
            'non_media_scaled_da is None when non_media_treatments is not None.'
        )
      national_da = self.non_media_scaled_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED
    else:
      national_da = self._aggregate_and_scale_geo_da(
          self._meridian.input_data.non_media_treatments,
          constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
          transformers.CenteringAndScalingTransformer,
          constants.NON_MEDIA_CHANNEL,
          self._agg_config.non_media_treatments,
      )
    return national_da

  @functools.cached_property
  def rf_spend_da(self) -> xr.DataArray | None:
    """Returns RF spend.

    If the input spend is aggregated, it is allocated across geo and time
    proportionally to RF impressions (reach * frequency).
    """
    da = self._meridian.input_data.allocated_rf_spend
    if da is None:
      return None
    da = da.copy()
    da.name = constants.RF_SPEND
    return da

  @functools.cached_property
  def national_rf_spend_da(self) -> xr.DataArray | None:
    """Returns the national RF spend data array."""
    if self.rf_spend_da is None:
      return None
    if self._is_national_data:
      national_da = self.rf_spend_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_RF_SPEND
    else:
      national_da = self._aggregate_and_scale_geo_da(
          self._meridian.input_data.allocated_rf_spend,
          constants.NATIONAL_RF_SPEND,
          None,
      )
    return national_da

  @functools.cached_property
  def _rf_data(self) -> ReachFrequencyData | None:
    if self._meridian.input_data.reach is None:
      return None
    return self._get_rf_data(
        self._meridian.input_data.reach,
        self._meridian.input_data.frequency,
        is_organic=False,
    )

  @property
  def reach_raw_da(self) -> xr.DataArray | None:
    if self._rf_data is None:
      return None
    return self._rf_data.reach_raw_da

  @property
  def reach_scaled_da(self) -> xr.DataArray | None:
    if self._rf_data is None:
      return None
    return self._rf_data.reach_scaled_da  # pytype: disable=attribute-error

  @property
  def national_reach_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw reach data array."""
    if self._rf_data is None:
      return None
    return self._rf_data.national_reach_raw_da

  @property
  def national_reach_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled reach data array."""
    if self._rf_data is None:
      return None
    return self._rf_data.national_reach_scaled_da  # pytype: disable=attribute-error

  @property
  def frequency_da(self) -> xr.DataArray | None:
    if self._rf_data is None:
      return None
    return self._rf_data.frequency_da  # pytype: disable=attribute-error

  @property
  def national_frequency_da(self) -> xr.DataArray | None:
    """Returns the national frequency data array."""
    if self._rf_data is None:
      return None
    return self._rf_data.national_frequency_da  # pytype: disable=attribute-error

  @property
  def rf_impressions_raw_da(self) -> xr.DataArray | None:
    if self._rf_data is None:
      return None
    return self._rf_data.rf_impressions_raw_da

  @property
  def national_rf_impressions_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw RF impressions data array."""
    if self._rf_data is None:
      return None
    return self._rf_data.national_rf_impressions_raw_da

  @property
  def rf_impressions_scaled_da(self) -> xr.DataArray | None:
    if self._rf_data is None:
      return None
    return self._rf_data.rf_impressions_scaled_da

  @property
  def national_rf_impressions_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled RF impressions data array."""
    if self._rf_data is None:
      return None
    return self._rf_data.national_rf_impressions_scaled_da

  @functools.cached_property
  def _organic_rf_data(self) -> ReachFrequencyData | None:
    if self._meridian.input_data.organic_reach is None:
      return None
    return self._get_rf_data(
        self._meridian.input_data.organic_reach,
        self._meridian.input_data.organic_frequency,
        is_organic=True,
    )

  @property
  def organic_reach_raw_da(self) -> xr.DataArray | None:
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.reach_raw_da

  @property
  def organic_reach_scaled_da(self) -> xr.DataArray | None:
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.reach_scaled_da  # pytype: disable=attribute-error

  @property
  def national_organic_reach_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw organic reach data array."""
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.national_reach_raw_da

  @property
  def national_organic_reach_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled organic reach data array."""
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.national_reach_scaled_da  # pytype: disable=attribute-error

  @property
  def organic_rf_impressions_scaled_da(self) -> xr.DataArray | None:
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.rf_impressions_scaled_da

  @property
  def national_organic_rf_impressions_scaled_da(self) -> xr.DataArray | None:
    """Returns the national scaled organic RF impressions data array."""
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.national_rf_impressions_scaled_da

  @property
  def organic_frequency_da(self) -> xr.DataArray | None:
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.frequency_da  # pytype: disable=attribute-error

  @property
  def national_organic_frequency_da(self) -> xr.DataArray | None:
    """Returns the national organic frequency data array."""
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.national_frequency_da  # pytype: disable=attribute-error

  @property
  def organic_rf_impressions_raw_da(self) -> xr.DataArray | None:
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.rf_impressions_raw_da

  @property
  def national_organic_rf_impressions_raw_da(self) -> xr.DataArray | None:
    """Returns the national raw organic RF impressions data array."""
    if self._organic_rf_data is None:
      return None
    return self._organic_rf_data.national_rf_impressions_raw_da

  @functools.cached_property
  def geo_population_da(self) -> xr.DataArray | None:
    if self._is_national_data:
      return None
    return xr.DataArray(
        self._meridian.population,
        coords={constants.GEO: self._meridian.input_data.geo.values},
        dims=[constants.GEO],
        name=constants.POPULATION,
    )

  @functools.cached_property
  def kpi_scaled_da(self) -> xr.DataArray:
    scaled_kpi_da = _data_array_like(
        da=self._meridian.input_data.kpi,
        values=self._meridian.kpi_scaled,
    )
    scaled_kpi_da.name = constants.KPI_SCALED
    return scaled_kpi_da

  @functools.cached_property
  def _overall_scaled_kpi_invariability_artifact(
      self,
  ) -> eda_outcome.KpiInvariabilityArtifact:
    """Returns an artifact of overall scaled KPI invariability."""
    return eda_outcome.KpiInvariabilityArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        kpi_da=self.kpi_scaled_da,
        kpi_stdev=self.kpi_scaled_da.std(ddof=1),
    )

  @functools.cached_property
  def national_kpi_scaled_da(self) -> xr.DataArray:
    """Returns the national scaled KPI data array."""
    if self._is_national_data:
      national_da = self.kpi_scaled_da.squeeze(constants.GEO, drop=True)
      national_da.name = constants.NATIONAL_KPI_SCALED
    else:
      # Note that kpi is summable by assumption.
      national_da = self._aggregate_and_scale_geo_da(
          self._meridian.input_data.kpi,
          constants.NATIONAL_KPI_SCALED,
          transformers.CenteringAndScalingTransformer,
      )
    return national_da

  @functools.cached_property
  def treatment_control_scaled_ds(self) -> xr.Dataset:
    """Returns a Dataset containing all scaled treatments and controls.

    This includes media, RF impressions, organic media, organic RF impressions,
    non-media treatments, and control variables, all at the geo level.
    """
    to_merge = [
        da
        for da in [
            self.media_scaled_da,
            self.rf_impressions_scaled_da,
            self.organic_media_scaled_da,
            self.organic_rf_impressions_scaled_da,
            self.controls_scaled_da,
            self.non_media_scaled_da,
        ]
        if da is not None
    ]
    return xr.merge(to_merge, join='inner')

  @functools.cached_property
  def _stacked_treatment_control_scaled_da(self) -> xr.DataArray:
    """Returns a stacked DataArray of treatment_control_scaled_ds."""
    da = _stack_variables(self.treatment_control_scaled_ds)
    da.name = constants.TREATMENT_CONTROL_SCALED
    return da

  @functools.cached_property
  def national_treatment_control_scaled_ds(self) -> xr.Dataset:
    """Returns a Dataset containing all scaled treatments and controls.

    This includes media, RF impressions, organic media, organic RF impressions,
    non-media treatments, and control variables, all at the national level.
    """
    to_merge_national = [
        da
        for da in [
            self.national_media_scaled_da,
            self.national_rf_impressions_scaled_da,
            self.national_organic_media_scaled_da,
            self.national_organic_rf_impressions_scaled_da,
            self.national_controls_scaled_da,
            self.national_non_media_scaled_da,
        ]
        if da is not None
    ]
    return xr.merge(to_merge_national, join='inner')

  @functools.cached_property
  def _stacked_national_treatment_control_scaled_da(self) -> xr.DataArray:
    """Returns a stacked DataArray of national_treatment_control_scaled_ds."""
    da = _stack_variables(self.national_treatment_control_scaled_ds)
    da.name = constants.NATIONAL_TREATMENT_CONTROL_SCALED
    return da

  @functools.cached_property
  def all_reach_scaled_da(self) -> xr.DataArray | None:
    """Returns a DataArray containing all scaled reach data.

    This includes both paid and organic reach, concatenated along the RF_CHANNEL
    dimension.

    Returns:
      A DataArray containing all scaled reach data, or None if no RF or organic
      RF channels are present.
    """
    reach_das = []
    if self.reach_scaled_da is not None:
      reach_das.append(self.reach_scaled_da)
    if self.organic_reach_scaled_da is not None:
      reach_das.append(
          self.organic_reach_scaled_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          )
      )
    if not reach_das:
      return None
    da = xr.concat(reach_das, dim=constants.RF_CHANNEL)
    da.name = constants.ALL_REACH_SCALED
    return da

  @functools.cached_property
  def all_freq_da(self) -> xr.DataArray | None:
    """Returns a DataArray containing all frequency data.

    This includes both paid and organic frequency, concatenated along the
    RF_CHANNEL dimension.

    Returns:
      A DataArray containing all frequency data, or None if no RF or organic
      RF channels are present.
    """
    freq_das = []
    if self.frequency_da is not None:
      freq_das.append(self.frequency_da)
    if self.organic_frequency_da is not None:
      freq_das.append(
          self.organic_frequency_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          )
      )
    if not freq_das:
      return None
    da = xr.concat(freq_das, dim=constants.RF_CHANNEL)
    da.name = constants.ALL_FREQUENCY
    return da

  @functools.cached_property
  def national_all_reach_scaled_da(self) -> xr.DataArray | None:
    """Returns a DataArray containing all national-level scaled reach data.

    This includes both paid and organic reach, concatenated along the
    RF_CHANNEL dimension.

    Returns:
      A DataArray containing all national-level scaled reach data, or None if
      no RF or organic RF channels are present.
    """
    national_reach_das = []
    if self.national_reach_scaled_da is not None:
      national_reach_das.append(self.national_reach_scaled_da)
    national_organic_reach_scaled_da = self.national_organic_reach_scaled_da
    if national_organic_reach_scaled_da is not None:
      national_reach_das.append(
          national_organic_reach_scaled_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          )
      )
    if not national_reach_das:
      return None
    da = xr.concat(national_reach_das, dim=constants.RF_CHANNEL)
    da.name = constants.NATIONAL_ALL_REACH_SCALED
    return da

  @functools.cached_property
  def national_all_freq_da(self) -> xr.DataArray | None:
    """Returns a DataArray containing all national-level frequency data.

    This includes both paid and organic frequency, concatenated along the
    RF_CHANNEL dimension.

    Returns:
      A DataArray containing all national-level frequency data, or None if no
      RF or organic RF channels are present.
    """
    national_freq_das = []
    if self.national_frequency_da is not None:
      national_freq_das.append(self.national_frequency_da)
    national_organic_frequency_da = self.national_organic_frequency_da
    if national_organic_frequency_da is not None:
      national_freq_das.append(
          national_organic_frequency_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          )
      )
    if not national_freq_das:
      return None
    da = xr.concat(national_freq_das, dim=constants.RF_CHANNEL)
    da.name = constants.NATIONAL_ALL_FREQUENCY
    return da

  @property
  def _critical_checks(
      self,
  ) -> list[tuple[_NamedEDACheckCallable, eda_outcome.EDACheckType]]:
    """Returns a list of critical checks to be performed."""
    checks = [
        (
            self.check_overall_kpi_invariability,
            eda_outcome.EDACheckType.KPI_INVARIABILITY,
        ),
        (self.check_vif, eda_outcome.EDACheckType.MULTICOLLINEARITY),
        (
            self.check_pairwise_corr,
            eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        ),
    ]
    return checks

  def _truncate_media_time(self, da: xr.DataArray) -> xr.DataArray:
    """Truncates the first `start` elements of the media time of a variable."""
    # This should not happen. If it does, it means this function is mis-used.
    if constants.MEDIA_TIME not in da.coords:
      raise ValueError(
          f'Variable does not have a media time coordinate: {da.name}.'
      )

    start = self._meridian.n_media_times - self._meridian.n_times
    da = da.copy().isel({constants.MEDIA_TIME: slice(start, None)})
    da = da.rename({constants.MEDIA_TIME: constants.TIME})
    return da

  def _scale_xarray(
      self,
      xarray: xr.DataArray,
      transformer_class: Optional[type[transformers.TensorTransformer]],
      population: Optional[backend.Tensor] = None,
  ) -> xr.DataArray:
    """Scales xarray values with a TensorTransformer."""
    da = xarray.copy()

    if transformer_class is None:
      return da
    if population is None:
      population = backend.ones([1], dtype=backend.float32)
    if transformer_class is transformers.CenteringAndScalingTransformer:
      xarray_transformer = transformers.CenteringAndScalingTransformer(
          tensor=da.values, population=population
      )
    elif transformer_class is transformers.MediaTransformer:
      xarray_transformer = transformers.MediaTransformer(
          media=da.values, population=population
      )
    else:
      raise ValueError(
          'Unknown transformer class: '
          + str(transformer_class)
          + '.\nMust be one of: CenteringAndScalingTransformer or'
          ' MediaTransformer.'
      )
    da.values = xarray_transformer.forward(da.values)
    return da

  def _aggregate_variables(
      self,
      geo_da: xr.DataArray,
      channel_dim: str,
      da_var_agg_map: eda_spec.AggregationMap,
      keepdims: bool = True,
  ) -> xr.DataArray:
    """Aggregates variables within a DataArray based on user-defined functions.

    Args:
      geo_da: The geo-level DataArray containing multiple variables along
        channel_dim.
      channel_dim: The name of the dimension coordinate to aggregate over (e.g.,
        constants.CONTROL_VARIABLE).
      da_var_agg_map: A dictionary mapping dataArray variable names to
        aggregation functions.
      keepdims: Whether to keep the dimensions of the aggregated DataArray.

    Returns:
      An xr.DataArray aggregated to the national level, with each variable
      aggregated according to the da_var_agg_map.
    """
    agg_results = []
    for var_name in geo_da[channel_dim].values:
      var_data = geo_da.sel({channel_dim: var_name})
      agg_func = da_var_agg_map.get(var_name, _DEFAULT_DA_VAR_AGG_FUNCTION)
      # Apply the aggregation function over the GEO dimension
      aggregated_data = var_data.reduce(
          agg_func, dim=constants.GEO, keepdims=keepdims
      )
      agg_results.append(aggregated_data)

    # Combine the aggregated variables back into a single DataArray
    return xr.concat(agg_results, dim=channel_dim).transpose(..., channel_dim)

  def _aggregate_and_scale_geo_da(
      self,
      geo_da: xr.DataArray,
      national_da_name: str,
      transformer_class: Optional[type[transformers.TensorTransformer]],
      channel_dim: Optional[str] = None,
      da_var_agg_map: Optional[eda_spec.AggregationMap] = None,
  ) -> xr.DataArray:
    """Aggregate geo-level xr.DataArray to national level and then scale values.

    Args:
      geo_da: The geo-level DataArray to convert.
      national_da_name: The name for the returned national DataArray.
      transformer_class: The TensorTransformer class to apply after summing to
        national level. Must be None, CenteringAndScalingTransformer, or
        MediaTransformer.
      channel_dim: The name of the dimension coordinate to aggregate over (e.g.,
        constants.CONTROL_VARIABLE). If None, standard sum aggregation is used.
      da_var_agg_map: A dictionary mapping dataArray variable names to
        aggregation functions. Used only if channel_dim is not None.

    Returns:
      An xr.DataArray representing the aggregated and scaled national-level
        data.
    """
    temp_geo_dim = constants.NATIONAL_MODEL_DEFAULT_GEO_NAME

    if da_var_agg_map is None:
      da_var_agg_map = {}

    if channel_dim is not None:
      national_da = self._aggregate_variables(
          geo_da, channel_dim, da_var_agg_map
      )
    else:
      national_da = geo_da.sum(
          dim=constants.GEO, keepdims=True, skipna=False, keep_attrs=True
      )

    national_da = national_da.assign_coords({constants.GEO: [temp_geo_dim]})
    national_da.values = backend.cast(national_da.values, dtype=backend.float32)
    national_da = self._scale_xarray(national_da, transformer_class)

    national_da = national_da.sel({constants.GEO: temp_geo_dim}, drop=True)
    national_da.name = national_da_name
    return national_da

  def _get_rf_data(
      self,
      reach_raw_da: xr.DataArray,
      freq_raw_da: xr.DataArray,
      is_organic: bool,
  ) -> ReachFrequencyData:
    """Get impressions and frequencies data arrays for RF channels."""
    if is_organic:
      scaled_reach_values = (
          self._meridian.organic_rf_tensors.organic_reach_scaled
      )
      names = _ORGANIC_RF_NAMES
    else:
      scaled_reach_values = self._meridian.rf_tensors.reach_scaled
      names = _RF_NAMES

    reach_scaled_da = _data_array_like(
        da=reach_raw_da, values=scaled_reach_values
    )
    reach_scaled_da.name = names.reach_scaled
    # Truncate the media time for reach and scaled reach.
    reach_raw_da = self._truncate_media_time(reach_raw_da)
    reach_raw_da.name = names.reach
    reach_scaled_da = self._truncate_media_time(reach_scaled_da)

    # The geo level frequency
    frequency_da = self._truncate_media_time(freq_raw_da)
    frequency_da.name = names.frequency

    # The raw geo level impression
    # It's equal to reach * frequency.
    impressions_raw_da = reach_raw_da * frequency_da
    impressions_raw_da.name = names.impressions
    impressions_raw_da.values = backend.cast(
        impressions_raw_da.values, dtype=backend.float32
    )

    if self._is_national_data:
      national_reach_raw_da = reach_raw_da.squeeze(constants.GEO, drop=True)
      national_reach_raw_da.name = names.national_reach
      national_reach_scaled_da = reach_scaled_da.squeeze(
          constants.GEO, drop=True
      )
      national_reach_scaled_da.name = names.national_reach_scaled
      national_impressions_raw_da = impressions_raw_da.squeeze(
          constants.GEO, drop=True
      )
      national_impressions_raw_da.name = names.national_impressions
      national_frequency_da = frequency_da.squeeze(constants.GEO, drop=True)
      national_frequency_da.name = names.national_frequency

      # Scaled impressions
      impressions_scaled_da = self._scale_xarray(
          impressions_raw_da, transformers.MediaTransformer
      )
      impressions_scaled_da.name = names.impressions_scaled
      national_impressions_scaled_da = impressions_scaled_da.squeeze(
          constants.GEO, drop=True
      )
      national_impressions_scaled_da.name = names.national_impressions_scaled
    else:
      national_reach_raw_da = self._aggregate_and_scale_geo_da(
          reach_raw_da, names.national_reach, None
      )
      national_reach_scaled_da = self._aggregate_and_scale_geo_da(
          reach_raw_da,
          names.national_reach_scaled,
          transformers.MediaTransformer,
      )
      national_impressions_raw_da = self._aggregate_and_scale_geo_da(
          impressions_raw_da,
          names.national_impressions,
          None,
      )

      # National frequency is a weighted average of geo frequencies,
      # weighted by reach.
      national_frequency_da = xr.where(
          national_reach_raw_da == 0.0,
          0.0,
          national_impressions_raw_da / national_reach_raw_da,
      )
      national_frequency_da.name = names.national_frequency
      national_frequency_da.values = backend.cast(
          national_frequency_da.values, dtype=backend.float32
      )

      # Scale the impressions by population
      impressions_scaled_da = self._scale_xarray(
          impressions_raw_da,
          transformers.MediaTransformer,
          population=self._meridian.population,
      )
      impressions_scaled_da.name = names.impressions_scaled

      # Scale the national impressions
      national_impressions_scaled_da = self._aggregate_and_scale_geo_da(
          impressions_raw_da,
          names.national_impressions_scaled,
          transformers.MediaTransformer,
      )

    return ReachFrequencyData(
        reach_raw_da=reach_raw_da,
        reach_scaled_da=reach_scaled_da,
        national_reach_raw_da=national_reach_raw_da,
        national_reach_scaled_da=national_reach_scaled_da,
        frequency_da=frequency_da,
        national_frequency_da=national_frequency_da,
        rf_impressions_scaled_da=impressions_scaled_da,
        national_rf_impressions_scaled_da=national_impressions_scaled_da,
        rf_impressions_raw_da=impressions_raw_da,
        national_rf_impressions_raw_da=national_impressions_raw_da,
    )

  def _pairwise_corr_for_geo_data(
      self, dims: str | Sequence[str], extreme_corr_threshold: float
  ) -> tuple[xr.DataArray, pd.DataFrame]:
    """Get pairwise correlation among treatments and controls for geo data."""
    corr_mat = _compute_correlation_matrix(
        self._stacked_treatment_control_scaled_da, dims=dims
    )
    extreme_corr_var_pairs_df = _find_extreme_corr_pairs(
        corr_mat, extreme_corr_threshold
    )
    return corr_mat, extreme_corr_var_pairs_df

  def check_geo_pairwise_corr(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    """Checks pairwise correlation among treatments and controls for geo data.

    Returns:
      An EDAOutcome object with findings and result values.

    Raises:
      GeoLevelCheckOnNationalModelError: If the model is national.
    """
    # If the model is national, raise an error.
    if self._is_national_data:
      raise GeoLevelCheckOnNationalModelError(
          'check_geo_pairwise_corr is not supported for national models.'
      )

    findings = []

    overall_corr_mat, overall_extreme_corr_var_pairs_df = (
        self._pairwise_corr_for_geo_data(
            dims=[constants.GEO, constants.TIME],
            extreme_corr_threshold=_OVERALL_PAIRWISE_CORR_THRESHOLD,
        )
    )
    if not overall_extreme_corr_var_pairs_df.empty:
      var_pairs = overall_extreme_corr_var_pairs_df.index.to_list()
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ERROR,
              explanation=(
                  'Some variables have perfect pairwise correlation across all'
                  ' times and geos. For each pair of perfectly-correlated'
                  ' variables, please remove one of the variables from the'
                  f' model.\nPairs with perfect correlation: {var_pairs}'
              ),
          )
      )

    geo_corr_mat, geo_extreme_corr_var_pairs_df = (
        self._pairwise_corr_for_geo_data(
            dims=constants.TIME,
            extreme_corr_threshold=_GEO_PAIRWISE_CORR_THRESHOLD,
        )
    )
    # Overall correlation and per-geo correlation findings are mutually
    # exclusive, and overall correlation finding takes precedence.
    if (
        overall_extreme_corr_var_pairs_df.empty
        and not geo_extreme_corr_var_pairs_df.empty
    ):
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ATTENTION,
              explanation=(
                  'Some variables have perfect pairwise correlation in certain'
                  ' geo(s). Consider checking your data, and/or combining these'
                  ' variables if they also have high pairwise correlations in'
                  ' other geos.'
              ),
          )
      )

    # If there are no findings, add a INFO level finding indicating that no
    # severe correlations were found and what it means for user's data.
    if not findings:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review the computed pairwise correlations. Note that'
                  ' high pairwise correlation may cause model identifiability'
                  ' and convergence issues. Consider combining the variables if'
                  ' high correlation exists.'
              ),
          )
      )

    pairwise_corr_artifacts = [
        eda_outcome.PairwiseCorrArtifact(
            level=eda_outcome.AnalysisLevel.OVERALL,
            corr_matrix=overall_corr_mat,
            extreme_corr_var_pairs=overall_extreme_corr_var_pairs_df,
            extreme_corr_threshold=_OVERALL_PAIRWISE_CORR_THRESHOLD,
        ),
        eda_outcome.PairwiseCorrArtifact(
            level=eda_outcome.AnalysisLevel.GEO,
            corr_matrix=geo_corr_mat,
            extreme_corr_var_pairs=geo_extreme_corr_var_pairs_df,
            extreme_corr_threshold=_GEO_PAIRWISE_CORR_THRESHOLD,
        ),
    ]

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        findings=findings,
        analysis_artifacts=pairwise_corr_artifacts,
    )

  def check_national_pairwise_corr(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    """Checks pairwise correlation among treatments and controls for national data.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    findings = []

    corr_mat = _compute_correlation_matrix(
        self._stacked_national_treatment_control_scaled_da, dims=constants.TIME
    )
    extreme_corr_var_pairs_df = _find_extreme_corr_pairs(
        corr_mat, _NATIONAL_PAIRWISE_CORR_THRESHOLD
    )

    if not extreme_corr_var_pairs_df.empty:
      var_pairs = extreme_corr_var_pairs_df.index.to_list()
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ERROR,
              explanation=(
                  'Some variables have perfect pairwise correlation across all'
                  ' times. For each pair of perfectly-correlated'
                  ' variables, please remove one of the variables from the'
                  f' model.\nPairs with perfect correlation: {var_pairs}'
              ),
          )
      )
    else:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review the computed pairwise correlations. Note that'
                  ' high pairwise correlation may cause model identifiability'
                  ' and convergence issues. Consider combining the variables if'
                  ' high correlation exists.'
              ),
          )
      )

    pairwise_corr_artifacts = [
        eda_outcome.PairwiseCorrArtifact(
            level=eda_outcome.AnalysisLevel.NATIONAL,
            corr_matrix=corr_mat,
            extreme_corr_var_pairs=extreme_corr_var_pairs_df,
            extreme_corr_threshold=_NATIONAL_PAIRWISE_CORR_THRESHOLD,
        )
    ]
    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        findings=findings,
        analysis_artifacts=pairwise_corr_artifacts,
    )

  def check_pairwise_corr(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    """Checks pairwise correlation among treatments and controls.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    if self._is_national_data:
      return self.check_national_pairwise_corr()
    else:
      return self.check_geo_pairwise_corr()

  def _check_std(
      self,
      data: xr.DataArray,
      level: eda_outcome.AnalysisLevel,
      zero_std_message: str,
  ) -> tuple[
      Optional[eda_outcome.EDAFinding], eda_outcome.StandardDeviationArtifact
  ]:
    """Helper to check standard deviation."""
    std_ds, outlier_df = _calculate_std(data)

    finding = None
    if (std_ds[_STD_WITHOUT_OUTLIERS_VAR_NAME] < _STD_THRESHOLD).any():
      finding = eda_outcome.EDAFinding(
          severity=eda_outcome.EDASeverity.ATTENTION,
          explanation=zero_std_message,
      )

    artifact = eda_outcome.StandardDeviationArtifact(
        variable=str(data.name),
        level=level,
        std_ds=std_ds,
        outlier_df=outlier_df,
    )

    return finding, artifact

  def check_geo_std(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    """Checks std for geo-level KPI, treatments, R&F, and controls."""
    if self._is_national_data:
      raise ValueError('check_geo_std is not applicable for national models.')

    findings = []
    artifacts = []

    checks = [
        (
            self.kpi_scaled_da,
            (
                'KPI has zero standard deviation after removing outliers'
                ' in certain geos, indicating weak or no signal in the response'
                ' variable for these geos.  Please review the input data,'
                ' and/or consider grouping these geos together.'
            ),
        ),
        (
            self._stacked_treatment_control_scaled_da,
            (
                'Some treatment or control variables have zero standard'
                ' deviation after removing outliers in certain geo(s). Please'
                ' review the input data. If these variables are sparse,'
                ' consider combining them to mitigate potential model'
                ' identifiability and convergence issues.'
            ),
        ),
        (
            self.all_reach_scaled_da,
            (
                'There are RF or Organic RF channels with zero variation of'
                ' reach across time at a geo after outliers are removed. If'
                ' these channels also have low variation of reach in other'
                ' geos, consider modeling them as impression-based channels'
                ' instead by taking reach * frequency.'
            ),
        ),
        (
            self.all_freq_da,
            (
                'There are RF or Organic RF channels with zero variation of'
                ' frequency across time at a geo after outliers are removed. If'
                ' these channels also have low variation of frequency in other'
                ' geos, consider modeling them as impression-based channels'
                ' instead by taking reach * frequency.'
            ),
        ),
    ]

    for data_da, message in checks:
      if data_da is None:
        continue
      finding, artifact = self._check_std(
          level=eda_outcome.AnalysisLevel.GEO,
          data=data_da,
          zero_std_message=message,
      )
      artifacts.append(artifact)
      if finding:
        findings.append(finding)

    # Add an INFO finding if no findings were added.
    if not findings:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review any identified outliers and the standard'
                  ' deviation.'
              ),
          )
      )

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        findings=findings,
        analysis_artifacts=artifacts,
    )

  def check_national_std(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    """Checks std for national-level KPI, treatments, R&F, and controls."""
    findings = []
    artifacts = []

    checks = [
        (
            self.national_kpi_scaled_da,
            (
                'The standard deviation of the scaled KPI drops from positive'
                ' to zero after removing outliers, indicating sparsity of KPI'
                ' i.e. lack of signal in the response variable. Please review'
                ' the input data, and/or reconsider the feasibility of model'
                ' fitting with this dataset.'
            ),
        ),
        (
            self._stacked_national_treatment_control_scaled_da,
            (
                'The standard deviation of these scaled treatment or control'
                ' variables drops from positive to zero after removing'
                ' outliers. This indicates sparsity of these variables, which'
                ' may cause model identifiability and convergence issues.'
                ' Please review the input data, and/or consider combining these'
                ' variables to mitigate sparsity.'
            ),
        ),
        (
            self.national_all_reach_scaled_da,
            (
                'There are RF channels with totally zero variation of reach'
                ' across time at the national level after outliers are removed.'
                ' Consider modeling these RF channels as impression-based'
                ' channels instead.'
            ),
        ),
        (
            self.national_all_freq_da,
            (
                'There are RF channels with totally zero variation of frequency'
                ' across time at the national level after outliers are removed.'
                ' Consider modeling these RF channels as impression-based'
                ' channels instead.'
            ),
        ),
    ]

    for data_da, message in checks:
      if data_da is None:
        continue
      finding, artifact = self._check_std(
          data=data_da,
          level=eda_outcome.AnalysisLevel.NATIONAL,
          zero_std_message=message,
      )
      artifacts.append(artifact)
      if finding:
        findings.append(finding)

    # Add an INFO finding if no findings were added.
    if not findings:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review any identified outliers and the standard'
                  ' deviation.'
              ),
          )
      )

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        findings=findings,
        analysis_artifacts=artifacts,
    )

  def check_std(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    """Checks standard deviation for treatments and controls.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    if self._is_national_data:
      return self.check_national_std()
    else:
      return self.check_geo_std()

  def check_geo_vif(self) -> eda_outcome.EDAOutcome[eda_outcome.VIFArtifact]:
    """Computes geo-level variance inflation factor among treatments and controls."""
    if self._is_national_data:
      raise ValueError(
          'Geo-level VIF checks are not applicable for national models.'
      )

    # Overall level VIF check for geo data.
    tc_da = self._stacked_treatment_control_scaled_da
    overall_threshold = self._spec.vif_spec.overall_threshold

    overall_vif_da = _calculate_vif(tc_da, _STACK_VAR_COORD_NAME)
    extreme_overall_vif_da = overall_vif_da.where(
        overall_vif_da > overall_threshold
    )
    extreme_overall_vif_df = extreme_overall_vif_da.to_dataframe(
        name=_VIF_COL_NAME
    ).dropna()

    overall_vif_artifact = eda_outcome.VIFArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        vif_da=overall_vif_da,
        outlier_df=extreme_overall_vif_df,
    )

    # Geo level VIF check.
    geo_threshold = self._spec.vif_spec.geo_threshold
    geo_vif_da = tc_da.groupby(constants.GEO).map(
        lambda x: _calculate_vif(x, _STACK_VAR_COORD_NAME)
    )
    extreme_geo_vif_da = geo_vif_da.where(geo_vif_da > geo_threshold)
    extreme_geo_vif_df = extreme_geo_vif_da.to_dataframe(
        name=_VIF_COL_NAME
    ).dropna()

    geo_vif_artifact = eda_outcome.VIFArtifact(
        level=eda_outcome.AnalysisLevel.GEO,
        vif_da=geo_vif_da,
        outlier_df=extreme_geo_vif_df,
    )

    findings = []
    if not extreme_overall_vif_df.empty:
      high_vif_vars = extreme_overall_vif_df.index.to_list()
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ERROR,
              explanation=(
                  'Some variables have extreme multicollinearity (VIF'
                  f' >{overall_threshold}) across all times and geos. To'
                  ' address multicollinearity, please drop any variable that'
                  ' is a linear combination of other variables. Otherwise,'
                  ' consider combining variables.\n'
                  f'Variables with extreme VIF: {high_vif_vars}'
              ),
          )
      )
    elif not extreme_geo_vif_df.empty:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ATTENTION,
              explanation=(
                  'Some variables have extreme multicollinearity (with VIF >'
                  f' {geo_threshold}) in certain geo(s). Consider checking your'
                  ' data, and/or combining these variables if they also have'
                  ' high VIF in other geos.'
              ),
          )
      )
    else:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review the computed VIFs. Note that high VIF suggests'
                  ' multicollinearity issues in the dataset, which may'
                  ' jeopardize model identifiability and model convergence.'
                  ' Consider combining the variables if high VIF occurs.'
              ),
          )
      )

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.MULTICOLLINEARITY,
        findings=findings,
        analysis_artifacts=[overall_vif_artifact, geo_vif_artifact],
    )

  def check_national_vif(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.VIFArtifact]:
    """Computes national-level variance inflation factor among treatments and controls."""
    national_tc_da = self._stacked_national_treatment_control_scaled_da
    national_threshold = self._spec.vif_spec.national_threshold
    national_vif_da = _calculate_vif(national_tc_da, _STACK_VAR_COORD_NAME)

    extreme_national_vif_df = (
        national_vif_da.where(national_vif_da > national_threshold)
        .to_dataframe(name=_VIF_COL_NAME)
        .dropna()
    )
    national_vif_artifact = eda_outcome.VIFArtifact(
        level=eda_outcome.AnalysisLevel.NATIONAL,
        vif_da=national_vif_da,
        outlier_df=extreme_national_vif_df,
    )

    findings = []
    if not extreme_national_vif_df.empty:
      high_vif_vars = extreme_national_vif_df.index.to_list()
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.ERROR,
              explanation=(
                  'Some variables have extreme multicollinearity (with VIF >'
                  f' {national_threshold}) across all times. To address'
                  ' multicollinearity, please drop any variable that is a'
                  ' linear combination of other variables. Otherwise, consider'
                  ' combining variables.\n'
                  f'Variables with extreme VIF: {high_vif_vars}'
              ),
          )
      )
    else:
      findings.append(
          eda_outcome.EDAFinding(
              severity=eda_outcome.EDASeverity.INFO,
              explanation=(
                  'Please review the computed VIFs. Note that high VIF suggests'
                  ' multicollinearity issues in the dataset, which may'
                  ' jeopardize model identifiability and model convergence.'
                  ' Consider combining the variables if high VIF occurs.'
              ),
          )
      )
    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.MULTICOLLINEARITY,
        findings=findings,
        analysis_artifacts=[national_vif_artifact],
    )

  def check_vif(self) -> eda_outcome.EDAOutcome[eda_outcome.VIFArtifact]:
    """Computes variance inflation factor among treatments and controls.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    if self._is_national_data:
      return self.check_national_vif()
    else:
      return self.check_geo_vif()

  @property
  def kpi_has_variability(self) -> bool:
    """Returns True if the KPI has variability across geos and times."""
    return (
        self._overall_scaled_kpi_invariability_artifact.kpi_stdev.item()
        >= _STD_THRESHOLD
    )

  def check_overall_kpi_invariability(self) -> eda_outcome.EDAOutcome:
    """Checks if the KPI is constant across all geos and times."""
    kpi = self._overall_scaled_kpi_invariability_artifact.kpi_da.name
    geo_text = '' if self._is_national_data else 'geos and '

    if not self.kpi_has_variability:
      eda_finding = eda_outcome.EDAFinding(
          severity=eda_outcome.EDASeverity.ERROR,
          explanation=(
              f'`{kpi}` is constant across all {geo_text}times, indicating no'
              ' signal in the data. Please fix this data error.'
          ),
      )
    else:
      eda_finding = eda_outcome.EDAFinding(
          severity=eda_outcome.EDASeverity.INFO,
          explanation=(
              f'The {kpi} has variability across {geo_text}times in the data.'
          ),
      )

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.KPI_INVARIABILITY,
        findings=[eda_finding],
        analysis_artifacts=[self._overall_scaled_kpi_invariability_artifact],
    )

  def run_all_critical_checks(self) -> list[eda_outcome.EDAOutcome]:
    """Runs all critical EDA checks.

    Critical checks are those that can result in EDASeverity.ERROR findings.

    Returns:
      A list of EDA outcomes, one for each check.
    """
    outcomes = []
    for check, check_type in self._critical_checks:
      try:
        outcomes.append(check())
      except Exception as e:  # pylint: disable=broad-except
        error_finding = eda_outcome.EDAFinding(
            severity=eda_outcome.EDASeverity.ERROR,
            explanation=f'An error occurred during check {check.__name__}: {e}',
        )
        outcomes.append(
            eda_outcome.EDAOutcome(
                check_type=check_type,
                findings=[error_finding],
                analysis_artifacts=[],
            )
        )
    return outcomes
