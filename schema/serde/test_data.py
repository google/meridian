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

"""Test data for serde module."""

import inspect
import types
from typing import Any, Sequence
from unittest import mock

from meridian import backend
from meridian import constants as c
from meridian.model import prior_distribution
from meridian.model import spec
from mmm.v1.marketing import marketing_data_pb2 as marketing_pb
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
import numpy as np
import xarray as xr

from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import tensor_shape_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import types_pb2  # pylint: disable=g-direct-tensorflow-import

_MediaEffectsDist = meridian_pb.MediaEffectsDistribution
_PaidMediaPriorType = meridian_pb.PaidMediaPriorType
_NonPaidTreatmentsPriorType = meridian_pb.NonPaidTreatmentsPriorType
_NonMediaBaselineFunction = (
    meridian_pb.NonMediaBaselineValue.NonMediaBaselineFunction
)

# Shared constants
_TIME_STRS = ['2021-02-01', '2021-02-08']
_MEDIA_TIME_STRS = ['2021-01-25', '2021-02-01', '2021-02-08']
_GEO_IDS = ['geo_0', 'geo_1']
_MEDIA_CHANNEL_PAID = ['ch_paid_0', 'ch_paid_1']
_MEDIA_CHANNEL_ORGANIC = ['ch_organic_0', 'ch_organic_1']
_RF_CHANNEL_PAID = ['rf_ch_paid_0', 'rf_ch_paid_1']
_RF_CHANNEL_ORGANIC = ['rf_ch_organic_0', 'rf_ch_organic_1']
_CONTROL_VARIABLES = ['control_0', 'control_1']
_NON_MEDIA_TREATMENT_VARIABLES = [
    'non_media_treatment_0',
    'non_media_treatment_1',
]


def make_tensor_shape_proto(
    dims: Sequence[int],
) -> tensor_shape_pb2.TensorShapeProto:
  tensor_shape = tensor_shape_pb2.TensorShapeProto()
  for dim in dims:
    tensor_shape.dim.append(tensor_shape_pb2.TensorShapeProto.Dim(size=dim))
  return tensor_shape


def make_tensor_proto(
    dims: Sequence[int],
    dtype: types_pb2.DataType = types_pb2.DT_FLOAT,
    bool_vals: Sequence[bool] = (),
    string_vals: Sequence[str] = (),
    tensor_content: bytes = b'',
) -> tensor_pb2.TensorProto:
  return tensor_pb2.TensorProto(
      dtype=dtype,
      tensor_shape=make_tensor_shape_proto(dims),
      bool_val=bool_vals,
      string_val=[x.encode() for x in string_vals],
      tensor_content=tensor_content,
  )


def make_sample_dataset(
    n_chains: int,
    n_draws: int,
    n_geos: int = 5,
    n_controls: int = 2,
    n_knots: int = 0,
    n_times: int = 0,
    n_media_channels: int = 0,
    n_rf_channels: int = 0,
    n_organic_media_channels: int = 0,
    n_organic_rf_channels: int = 0,
    n_non_media_channels: int = 0,
) -> xr.Dataset:
  """Creates a sample dataset with all relevant Meridian dimensions.

  Args:
    n_chains: The number of chains.
    n_draws: The number of draws per chain.
    n_geos: The number of geos.
    n_controls: The number of control variables.
    n_knots: The number of knots.
    n_times: The number of time periods.
    n_media_channels: The number of media channels.
    n_rf_channels: The number of reach and frequency channels.
    n_organic_media_channels: The number of organic media channels.
    n_organic_rf_channels: The number of organic reach and frequency channels.
    n_non_media_channels: The number of non-media channels.

  Returns:
    An xarray Dataset with sample data.
  """
  data_vars = {
      c.STEP_SIZE: (
          [c.CHAIN, c.DRAW],
          np.random.normal(size=(n_chains, n_draws)),
      ),
      c.TUNE: (
          [c.CHAIN, c.DRAW],
          np.full((n_chains, n_draws), False),
      ),
      c.TARGET_LOG_PROBABILITY_TF: (
          [c.CHAIN, c.DRAW],
          np.random.normal(size=(n_chains, n_draws)),
      ),
      c.DIVERGING: (
          [c.CHAIN, c.DRAW],
          np.full((n_chains, n_draws), False),
      ),
      c.ACCEPT_RATIO: (
          [c.CHAIN, c.DRAW],
          np.random.normal(size=(n_chains, n_draws)),
      ),
      c.N_STEPS: (
          [c.CHAIN, c.DRAW],
          np.random.normal(size=(n_chains, n_draws)),
      ),
      'is_accepted': (
          [c.CHAIN, c.DRAW],
          np.full((n_chains, n_draws), True),
      ),
  }
  coords = {
      c.CHAIN: ([c.CHAIN], np.arange(n_chains)),
      c.DRAW: ([c.DRAW], np.arange(n_draws)),
      c.GEO: ([c.GEO], np.arange(n_geos)),
      c.CONTROL_VARIABLE: (
          [c.CONTROL_VARIABLE],
          np.arange(n_controls),
      ),
  }

  if n_knots > 0:
    coords[c.KNOTS] = ([c.KNOTS], np.arange(n_knots))

  if n_times > 0:
    coords[c.TIME] = ([c.TIME], np.arange(n_times))

  if n_media_channels > 0:
    coords[c.MEDIA_CHANNEL] = (
        [c.MEDIA_CHANNEL],
        np.arange(n_media_channels),
    )

  if n_rf_channels > 0:
    coords[c.RF_CHANNEL] = (
        [c.RF_CHANNEL],
        np.arange(n_rf_channels),
    )

  if n_organic_media_channels > 0:
    coords[c.ORGANIC_MEDIA_CHANNEL] = (
        [c.ORGANIC_MEDIA_CHANNEL],
        np.arange(n_organic_media_channels),
    )

  if n_organic_rf_channels > 0:
    coords[c.ORGANIC_RF_CHANNEL] = (
        [c.ORGANIC_RF_CHANNEL],
        np.arange(n_organic_rf_channels),
    )

  if n_non_media_channels > 0:
    coords[c.NON_MEDIA_CHANNEL] = (
        [c.NON_MEDIA_CHANNEL],
        np.arange(n_non_media_channels),
    )

  return xr.Dataset(data_vars, coords=coords)


# Marketing data test data
MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE = mock.MagicMock(
    kpi_type=c.NON_REVENUE,
    geo=xr.DataArray(np.array(['national_geo'])),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: ['national_geo']},
        data=np.array([1.0]),
        name=c.POPULATION,
    ),
    media=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[41, 42], [43, 44]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[141, 142], [143, 144]]]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=True,
    media_spend_has_time_dimension=True,
    reach=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[51.0, 52.0], [53.0, 54.0]]]),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[1.1, 1.2], [2, 3]]]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={c.RF_CHANNEL: _RF_CHANNEL_PAID},
        data=np.array([502, 504]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=False,
    rf_spend_has_time_dimension=False,
    kpi=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
        },
        data=np.array([[1, 2]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
        },
        data=np.array([[11, 12]]),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: ['control_0', 'control_1'],
        },
        data=np.array([[[31, 32], [33, 34]]]),
        name=c.CONTROLS,
    ),
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "national_geo"
        population: 1
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
        media_spend: 141.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
        media_spend: 142.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 51
        average_frequency: 1.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 52
        average_frequency: 1.2
      }
      kpi {
        name: "non_revenue"
        non_revenue {
          value: 1.0
          revenue_per_kpi: 11.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "national_geo"
        population: 1
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
        media_spend: 143.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
        media_spend: 144.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 53
        average_frequency: 2.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 54
        average_frequency: 3.0
      }
      kpi {
        name: "non_revenue"
        non_revenue {
          value: 2.0
          revenue_per_kpi: 12.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        spend: 502.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        spend: 504.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      channel_dimensions {
        name: "reach"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "frequency"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "non_revenue"
    }
        """,
    marketing_pb.MarketingData(),
)

# Same as above, but with no controls.
MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS = mock.MagicMock(
    kpi_type=c.NON_REVENUE,
    geo=xr.DataArray(np.array(['national_geo'])),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: ['national_geo']},
        data=np.array([1.0]),
        name=c.POPULATION,
    ),
    media=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[41, 42], [43, 44]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[141, 142], [143, 144]]]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=True,
    media_spend_has_time_dimension=True,
    reach=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[51.0, 52.0], [53.0, 54.0]]]),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[1.1, 1.2], [2, 3]]]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={c.RF_CHANNEL: _RF_CHANNEL_PAID},
        data=np.array([502, 504]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=False,
    rf_spend_has_time_dimension=False,
    kpi=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
        },
        data=np.array([[1, 2]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={
            c.GEO: ['national_geo'],
            c.TIME: _TIME_STRS,
        },
        data=np.array([[11, 12]]),
        name=c.REVENUE_PER_KPI,
    ),
    controls=None,
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "national_geo"
        population: 1
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
        media_spend: 141.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
        media_spend: 142.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 51
        average_frequency: 1.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 52
        average_frequency: 1.2
      }
      kpi {
        name: "non_revenue"
        non_revenue {
          value: 1.0
          revenue_per_kpi: 11.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "national_geo"
        population: 1
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
        media_spend: 143.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
        media_spend: 144.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 53
        average_frequency: 2.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 54
        average_frequency: 3.0
      }
      kpi {
        name: "non_revenue"
        non_revenue {
          value: 2.0
          revenue_per_kpi: 12.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        spend: 502.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        spend: 504.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      channel_dimensions {
        name: "reach"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "frequency"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      kpi_type: "non_revenue"
    }
        """,
    marketing_pb.MarketingData(),
)

# Media, Paid, Expanded, Lagged
MOCK_INPUT_DATA_MEDIA_PAID_EXPANDED_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_MEDIA_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS},
        data=np.array([11.1, 12.2]),
        name=c.POPULATION,
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array(
            [[[39, 40], [41, 42], [43, 44]], [[45, 46], [47, 48], [49, 50]]]
        ),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID},
        data=np.array([492, 496]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=False,
    media_spend_has_time_dimension=False,
    reach=None,
    frequency=None,
    rf_spend=None,
    kpi=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
        },
        data=np.array([[2, 3], [4, 5]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
        },
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[32, 33], [34, 35]], [[36, 37], [38, 39]]]),
        name=c.CONTROLS,
    ),
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_MEDIA_PAID_EXPANDED_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 39.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 40.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 32.0
      }
      control_variables {
        name: "control_1"
        value: 33.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 34.0
      }
      control_variables {
        name: "control_1"
        value: 35.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 45.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 46.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 36.0
      }
      control_variables {
        name: "control_1"
        value: 37.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 47.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 48.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 4.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 38.0
      }
      control_variables {
        name: "control_1"
        value: 39.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 49.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 50.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 5.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        media_spend: 492.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        media_spend: 496.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 1
          day: 25
        }
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

# Media, Paid, Granular, Not Lagged
MOCK_INPUT_DATA_MEDIA_PAID_GRANULAR_NOT_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[39, 40], [41, 42]], [[43, 44], [45, 46]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[123, 124], [125, 126]], [[127, 128], [129, 130]]]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=True,
    media_spend_has_time_dimension=True,
    reach=None,
    frequency=None,
    rf_spend=None,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[1, 2], [3, 4]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED_STRING = """
   marketing_data_points {
  geo_info {
    geo_id: "geo_0"
    population: 11
  }
  date_interval {
    start_date {
      year: 2021
      month: 2
      day: 1
    }
    end_date {
      year: 2021
      month: 2
      day: 8
    }
  }
  control_variables {
    name: "control_0"
    value: 31.0
  }
  control_variables {
    name: "control_1"
    value: 32.0
  }
  media_variables {
    channel_name: "ch_paid_0"
    scalar_metric {
      name: "impressions"
      value: 39.0
    }
    media_spend: 123.0
  }
  media_variables {
    channel_name: "ch_paid_1"
    scalar_metric {
      name: "impressions"
      value: 40.0
    }
    media_spend: 124.0
  }
  kpi {
    name: "revenue"
    revenue {
      value: 1.0
    }
  }
}
marketing_data_points {
  geo_info {
    geo_id: "geo_0"
    population: 11
  }
  date_interval {
    start_date {
      year: 2021
      month: 2
      day: 8
    }
    end_date {
      year: 2021
      month: 2
      day: 15
    }
  }
  control_variables {
    name: "control_0"
    value: 33.0
  }
  control_variables {
    name: "control_1"
    value: 34.0
  }
  media_variables {
    channel_name: "ch_paid_0"
    scalar_metric {
      name: "impressions"
      value: 41.0
    }
    media_spend: 125.0
  }
  media_variables {
    channel_name: "ch_paid_1"
    scalar_metric {
      name: "impressions"
      value: 42.0
    }
    media_spend: 126.0
  }
  kpi {
    name: "revenue"
    revenue {
      value: 2.0
    }
  }
}
marketing_data_points {
  geo_info {
    geo_id: "geo_1"
    population: 12
  }
  date_interval {
    start_date {
      year: 2021
      month: 2
      day: 1
    }
    end_date {
      year: 2021
      month: 2
      day: 8
    }
  }
  control_variables {
    name: "control_0"
    value: 35.0
  }
  control_variables {
    name: "control_1"
    value: 36.0
  }
  media_variables {
    channel_name: "ch_paid_0"
    scalar_metric {
      name: "impressions"
      value: 43.0
    }
    media_spend: 127.0
  }
  media_variables {
    channel_name: "ch_paid_1"
    scalar_metric {
      name: "impressions"
      value: 44.0
    }
    media_spend: 128.0
  }
  kpi {
    name: "revenue"
    revenue {
      value: 3.0
    }
  }
}
marketing_data_points {
  geo_info {
    geo_id: "geo_1"
    population: 12
  }
  date_interval {
    start_date {
      year: 2021
      month: 2
      day: 8
    }
    end_date {
      year: 2021
      month: 2
      day: 15
    }
  }
  control_variables {
    name: "control_0"
    value: 37.0
  }
  control_variables {
    name: "control_1"
    value: 38.0
  }
  media_variables {
    channel_name: "ch_paid_0"
    scalar_metric {
      name: "impressions"
      value: 45.0
    }
    media_spend: 129.0
  }
  media_variables {
    channel_name: "ch_paid_1"
    scalar_metric {
      name: "impressions"
      value: 46.0
    }
    media_spend: 130.0
  }
  kpi {
    name: "revenue"
    revenue {
      value: 4.0
    }
  }
}
metadata {
  time_dimensions {
    name: "time"
    dates {
      year: 2021
      month: 2
      day: 1
    }
    dates {
      year: 2021
      month: 2
      day: 8
    }
  }
  time_dimensions {
    name: "media_time"
    dates {
      year: 2021
      month: 2
      day: 1
    }
    dates {
      year: 2021
      month: 2
      day: 8
    }
  }
  channel_dimensions {
    name: "media"
    channels: "ch_paid_0"
    channels: "ch_paid_1"
  }
  control_names: "control_0"
  control_names: "control_1"
  kpi_type: "revenue"
}
"""

MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED = text_format.Parse(
    MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED_STRING,
    marketing_pb.MarketingData(),
)

# Media, Organic, Expanded, Lagged
MOCK_INPUT_DATA_MEDIA_ORGANIC_EXPANDED_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_MEDIA_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID},
        data=np.array([492, 496]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=False,
    media_spend_has_time_dimension=False,
    reach=None,
    frequency=None,
    rf_spend=None,
    organic_media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.ORGANIC_MEDIA_CHANNEL: _MEDIA_CHANNEL_ORGANIC,
        },
        data=np.array(
            [[[39, 40], [41, 42], [43, 44]], [[45, 46], [47, 48], [49, 50]]]
        ),
        name=c.ORGANIC_MEDIA,
    ),
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[2, 2], [3, 3]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
)

MOCK_PROTO_MEDIA_ORGANIC_EXPANDED_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 1.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 2.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 39.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 40.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 3.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 4.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 5.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 6.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 7.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 8.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 45.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 46.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 35.0
      }
      control_variables {
        name: "control_1"
        value: 36.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 9.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 10.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 47.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 48.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 11.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 12.0
        }
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 49.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 50.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        media_spend: 492.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        media_spend: 496.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 1
          day: 25
        }
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      channel_dimensions {
        name: "organic_media"
        channels: "ch_organic_0"
        channels: "ch_organic_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

# Media, Organic, Granular, Not Lagged
MOCK_INPUT_DATA_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[123, 124], [125, 126]], [[127, 128], [129, 130]]]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=True,
    media_spend_has_time_dimension=True,
    reach=None,
    frequency=None,
    rf_spend=None,
    organic_media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.ORGANIC_MEDIA_CHANNEL: _MEDIA_CHANNEL_ORGANIC,
        },
        data=np.array([[[39, 40], [41, 42]], [[43, 44], [45, 46]]]),
        name=c.ORGANIC_MEDIA,
    ),
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[2, 2], [3, 3]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
)

MOCK_PROTO_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 1.0
        }
        media_spend: 123.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 2.0
        }
        media_spend: 124.0
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 39.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 40.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 3.0
        }
        media_spend: 125.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 4.0
        }
        media_spend: 126.0
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 35.0
      }
      control_variables {
        name: "control_1"
        value: 36.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 5.0
        }
        media_spend: 127.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 6.0
        }
        media_spend: 128.0
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 7.0
        }
        media_spend: 129.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 8.0
        }
        media_spend: 130.0
      }
      media_variables {
        channel_name: "ch_organic_0"
        scalar_metric {
          name: "impressions"
          value: 45.0
        }
      }
      media_variables {
        channel_name: "ch_organic_1"
        scalar_metric {
          name: "impressions"
          value: 46.0
        }
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      channel_dimensions {
        name: "organic_media"
        channels: "ch_organic_0"
        channels: "ch_organic_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

# Reach and Frequency, Paid, Expanded, Lagged
MOCK_INPUT_DATA_RF_PAID_EXPANDED_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_MEDIA_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS},
        data=np.array([11.1, 12.2]),
        name=c.POPULATION,
    ),
    media=None,
    media_spend=None,
    reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array(
            [[[51, 52], [53, 54], [55, 56]], [[57, 58], [59, 60], [61, 62]]]
        ),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([
            [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
            [[1.7, 1.8], [1.9, 2.0], [2.1, 2.2]],
        ]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={c.RF_CHANNEL: _RF_CHANNEL_PAID},
        data=np.array([1004, 1008]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=False,
    rf_spend_has_time_dimension=False,
    kpi=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
        },
        data=np.array([[2, 3], [4, 5]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
        },
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[32, 33], [34, 35]], [[36, 37], [38, 39]]]),
        name=c.CONTROLS,
    ),
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_RF_PAID_EXPANDED_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 51
        average_frequency: 1.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 52
        average_frequency: 1.2
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 32.0
      }
      control_variables {
        name: "control_1"
        value: 33.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 53
        average_frequency: 1.3
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 54
        average_frequency: 1.4
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 34.0
      }
      control_variables {
        name: "control_1"
        value: 35.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 55
        average_frequency: 1.5
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 56
        average_frequency: 1.6
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 57
        average_frequency: 1.7
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 58
        average_frequency: 1.8
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 36.0
      }
      control_variables {
        name: "control_1"
        value: 37.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 59
        average_frequency: 1.9
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 60
        average_frequency: 2.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 4.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 38.0
      }
      control_variables {
        name: "control_1"
        value: 39.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 61
        average_frequency: 2.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 62
        average_frequency: 2.2
      }
      kpi {
        name: "revenue"
        revenue {
          value: 5.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        spend: 1004.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        spend: 1008.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 1
          day: 25
        }
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "reach"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "frequency"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

# Reach and Frequency, Paid, Granular, Not Lagged
MOCK_INPUT_DATA_RF_PAID_GRANULAR_NOT_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=None,
    media_spend=None,
    reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array(
            [[[51.0, 52.0], [53.0, 54.0]], [[55.0, 56.0], [57.0, 58.0]]]
        ),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[1.1, 1.2], [2, 3]], [[4, 5], [6, 7]]]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[252, 253], [254, 255]], [[256, 257], [258, 259]]]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=True,
    rf_spend_has_time_dimension=True,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[1, 2], [3, 4]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

MOCK_PROTO_RF_PAID_GRANULAR_NOT_LAGGED = text_format.Parse(
    """
   marketing_data_points {
    geo_info {
      geo_id: "geo_0"
      population: 11
    }
    date_interval {
      start_date {
        year: 2021
        month: 2
        day: 1
      }
      end_date {
        year: 2021
        month: 2
        day: 8
      }
    }
    control_variables {
      name: "control_0"
      value: 31.0
    }
    control_variables {
      name: "control_1"
      value: 32.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_0"
      reach: 51
      average_frequency: 1.1
      spend: 252.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_1"
      reach: 52
      average_frequency: 1.2
      spend: 253.0
    }
    kpi {
      name: "revenue"
      revenue {
        value: 1.0
      }
    }
  }
  marketing_data_points {
    geo_info {
      geo_id: "geo_0"
      population: 11
    }
    date_interval {
      start_date {
        year: 2021
        month: 2
        day: 8
      }
      end_date {
        year: 2021
        month: 2
        day: 15
      }
    }
    control_variables {
      name: "control_0"
      value: 33.0
    }
    control_variables {
      name: "control_1"
      value: 34.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_0"
      reach: 53
      average_frequency: 2.0
      spend: 254.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_1"
      reach: 54
      average_frequency: 3.0
      spend: 255.0
    }
    kpi {
      name: "revenue"
      revenue {
        value: 2.0
      }
    }
  }
  marketing_data_points {
    geo_info {
      geo_id: "geo_1"
      population: 12
    }
    date_interval {
      start_date {
        year: 2021
        month: 2
        day: 1
      }
      end_date {
        year: 2021
        month: 2
        day: 8
      }
    }
    control_variables {
      name: "control_0"
      value: 35.0
    }
    control_variables {
      name: "control_1"
      value: 36.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_0"
      reach: 55
      average_frequency: 4.0
      spend: 256.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_1"
      reach: 56
      average_frequency: 5.0
      spend: 257.0
    }
    kpi {
      name: "revenue"
      revenue {
        value: 3.0
      }
    }
  }
  marketing_data_points {
    geo_info {
      geo_id: "geo_1"
      population: 12
    }
    date_interval {
      start_date {
        year: 2021
        month: 2
        day: 8
      }
      end_date {
        year: 2021
        month: 2
        day: 15
      }
    }
    control_variables {
      name: "control_0"
      value: 37.0
    }
    control_variables {
      name: "control_1"
      value: 38.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_0"
      reach: 57
      average_frequency: 6.0
      spend: 258.0
    }
    reach_frequency_variables {
      channel_name: "rf_ch_paid_1"
      reach: 58
      average_frequency: 7.0
      spend: 259.0
    }
    kpi {
      name: "revenue"
      revenue {
        value: 4.0
      }
    }
  }
  metadata {
    time_dimensions {
      name: "time"
      dates {
        year: 2021
        month: 2
        day: 1
      }
      dates {
        year: 2021
        month: 2
        day: 8
      }
    }
    time_dimensions {
      name: "media_time"
      dates {
        year: 2021
        month: 2
        day: 1
      }
      dates {
        year: 2021
        month: 2
        day: 8
      }
    }
    channel_dimensions {
      name: "reach"
      channels: "rf_ch_paid_0"
      channels: "rf_ch_paid_1"
    }
    channel_dimensions {
      name: "frequency"
      channels: "rf_ch_paid_0"
      channels: "rf_ch_paid_1"
    }
    control_names: "control_0"
    control_names: "control_1"
    kpi_type: "revenue"
  }
    """,
    marketing_pb.MarketingData(),
)

# Reach and Frequency, Organic, Expanded, Lagged
MOCK_INPUT_DATA_RF_ORGANIC_EXPANDED_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_MEDIA_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=None,
    media_spend=None,
    reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([
            [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
            [[2.7, 2.8], [2.9, 3.0], [3.1, 3.2]],
        ]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={c.RF_CHANNEL: _RF_CHANNEL_PAID},
        data=np.array([1004, 1008]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=False,
    rf_spend_has_time_dimension=False,
    organic_media=None,
    organic_reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.ORGANIC_RF_CHANNEL: _RF_CHANNEL_ORGANIC,
        },
        data=np.array(
            [[[51, 52], [53, 54], [55, 56]], [[57, 58], [59, 60], [61, 62]]]
        ),
        name=c.ORGANIC_REACH,
    ),
    organic_frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.ORGANIC_RF_CHANNEL: _RF_CHANNEL_ORGANIC,
        },
        data=np.array([
            [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
            [[1.7, 1.8], [1.9, 2.0], [2.1, 2.2]],
        ]),
        name=c.ORGANIC_FREQUENCY,
    ),
    non_media_treatments=None,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[2, 2], [3, 3]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
)

MOCK_PROTO_RF_ORGANIC_EXPANDED_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 1
        average_frequency: 2.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 2
        average_frequency: 2.2
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 51
        average_frequency: 1.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 52
        average_frequency: 1.2
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 3
        average_frequency: 2.3
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 4
        average_frequency: 2.4
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 53
        average_frequency: 1.3
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 54
        average_frequency: 1.4
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 5
        average_frequency: 2.5
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 6
        average_frequency: 2.6
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 55
        average_frequency: 1.5
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 56
        average_frequency: 1.6
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 7
        average_frequency: 2.7
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 8
        average_frequency: 2.8
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 57
        average_frequency: 1.7
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 58
        average_frequency: 1.8
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 35.0
      }
      control_variables {
        name: "control_1"
        value: 36.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 9
        average_frequency: 2.9
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 10
        average_frequency: 3.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 59
        average_frequency: 1.9
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 60
        average_frequency: 2.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 11
        average_frequency: 3.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 12
        average_frequency: 3.2
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 61
        average_frequency: 2.1
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 62
        average_frequency: 2.2
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        spend: 1004.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        spend: 1008.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 1
          day: 25
        }
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "reach"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "frequency"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "organic_reach"
        channels: "rf_ch_organic_0"
        channels: "rf_ch_organic_1"
      }
      channel_dimensions {
        name: "organic_frequency"
        channels: "rf_ch_organic_0"
        channels: "rf_ch_organic_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

# Reach and Frequency, Organic, Granular, Not Lagged
MOCK_INPUT_DATA_RF_ORGANIC_GRANULAR_NOT_LAGGED = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    media=None,
    media_spend=None,
    reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        name=c.REACH,
    ),
    frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[2.1, 2.2], [3, 4]], [[5, 6], [7, 8]]]),
        name=c.FREQUENCY,
    ),
    rf_spend=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.RF_CHANNEL: _RF_CHANNEL_PAID,
        },
        data=np.array([[[252, 253], [254, 255]], [[256, 257], [258, 259]]]),
        name=c.RF_SPEND,
    ),
    rf_spend_has_geo_dimension=True,
    rf_spend_has_time_dimension=True,
    organic_media=None,
    organic_reach=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.ORGANIC_RF_CHANNEL: _RF_CHANNEL_ORGANIC,
        },
        data=np.array(
            [[[51.0, 52.0], [53.0, 54.0]], [[55.0, 56.0], [57.0, 58.0]]]
        ),
        name=c.ORGANIC_REACH,
    ),
    organic_frequency=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.ORGANIC_RF_CHANNEL: _RF_CHANNEL_ORGANIC,
        },
        data=np.array(
            [[[51.0, 52.0], [53.0, 54.0]], [[55.0, 56.0], [57.0, 58.0]]]
        ),
        name=c.ORGANIC_FREQUENCY,
    ),
    non_media_treatments=None,
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[2, 2], [3, 3]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
)

MOCK_PROTO_RF_ORGANIC_GRANULAR_NOT_LAGGED = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 1
        average_frequency: 2.1
        spend: 252.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 2
        average_frequency: 2.2
        spend: 253.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 51
        average_frequency: 51.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 52
        average_frequency: 52.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 3
        average_frequency: 3.0
        spend: 254.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 4
        average_frequency: 4.0
        spend: 255.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 53
        average_frequency: 53.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 54
        average_frequency: 54.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 35.0
      }
      control_variables {
        name: "control_1"
        value: 36.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 5
        average_frequency: 5.0
        spend: 256.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 6
        average_frequency: 6.0
        spend: 257.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 55
        average_frequency: 55.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 56
        average_frequency: 56.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_0"
        reach: 7
        average_frequency: 7.0
        spend: 258.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_paid_1"
        reach: 8
        average_frequency: 8.0
        spend: 259.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_0"
        reach: 57
        average_frequency: 57.0
      }
      reach_frequency_variables {
        channel_name: "rf_ch_organic_1"
        reach: 58
        average_frequency: 58.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "reach"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "frequency"
        channels: "rf_ch_paid_0"
        channels: "rf_ch_paid_1"
      }
      channel_dimensions {
        name: "organic_reach"
        channels: "rf_ch_organic_0"
        channels: "rf_ch_organic_1"
      }
      channel_dimensions {
        name: "organic_frequency"
        channels: "rf_ch_organic_0"
        channels: "rf_ch_organic_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

MOCK_INPUT_DATA_NON_MEDIA_TREATMENTS = mock.MagicMock(
    kpi_type=c.REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_MEDIA_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS}, data=np.array([11.1, 12.2]), name=c.POPULATION
    ),
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[1, 2], [3, 4]]),
        name=c.KPI,
    ),
    revenue_per_kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.ones((2, 2)),
        name=c.REVENUE_PER_KPI,
    ),
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
    non_media_treatments=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.NON_MEDIA_CHANNEL: _NON_MEDIA_TREATMENT_VARIABLES,
        },
        data=np.array([[[61, 62], [63, 64]], [[65, 66], [67, 68]]]),
        name=c.NON_MEDIA_TREATMENTS,
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _MEDIA_TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array(
            [[[39, 40], [41, 42], [43, 44]], [[45, 46], [47, 48], [49, 50]]]
        ),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID},
        data=np.array([492, 496]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=False,
    media_spend_has_time_dimension=False,
    reach=None,
    frequency=None,
    rf_spend=None,
    rf_spend_has_geo_dimension=False,
    rf_spend_has_time_dimension=False,
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
)

MOCK_PROTO_NON_MEDIA_TREATMENTS = text_format.Parse(
    """
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 39.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 40.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 31.0
      }
      control_variables {
        name: "control_1"
        value: 32.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
      }
      non_media_treatment_variables {
        name: "non_media_treatment_0"
        value: 61.0
      }
      non_media_treatment_variables {
        name: "non_media_treatment_1"
        value: 62.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 1.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_0"
        population: 11
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 33.0
      }
      control_variables {
        name: "control_1"
        value: 34.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
      }
      non_media_treatment_variables {
        name: "non_media_treatment_0"
        value: 63.0
      }
      non_media_treatment_variables {
        name: "non_media_treatment_1"
        value: 64.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 2.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 1
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 45.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 46.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 8
        }
      }
      control_variables {
        name: "control_0"
        value: 35.0
      }
      control_variables {
        name: "control_1"
        value: 36.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 47.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 48.0
        }
      }
      non_media_treatment_variables {
        name: "non_media_treatment_0"
        value: 65.0
      }
      non_media_treatment_variables {
        name: "non_media_treatment_1"
        value: 66.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 3.0
        }
      }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 12
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
      media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 49.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 50.0
        }
      }
      non_media_treatment_variables {
        name: "non_media_treatment_0"
        value: 67.0
      }
      non_media_treatment_variables {
        name: "non_media_treatment_1"
        value: 68.0
      }
      kpi {
        name: "revenue"
        revenue {
          value: 4.0
        }
      }
    }
    marketing_data_points {
      date_interval {
        start_date {
          year: 2021
          month: 1
          day: 25
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        media_spend: 492.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        media_spend: 496.0
      }
    }
    metadata {
      time_dimensions {
        name: "time"
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      time_dimensions {
        name: "media_time"
        dates {
          year: 2021
          month: 1
          day: 25
        }
        dates {
          year: 2021
          month: 2
          day: 1
        }
        dates {
          year: 2021
          month: 2
          day: 8
        }
      }
      channel_dimensions {
        name: "media"
        channels: "ch_paid_0"
        channels: "ch_paid_1"
      }
      control_names: "control_0"
      control_names: "control_1"
      non_media_treatment_names: "non_media_treatment_0"
      non_media_treatment_names: "non_media_treatment_1"
      kpi_type: "revenue"
    }
    """,
    marketing_pb.MarketingData(),
)

MOCK_INPUT_DATA_NO_REVENUE_PER_KPI = mock.MagicMock(
    kpi_type=c.NON_REVENUE,
    geo=xr.DataArray(np.array(_GEO_IDS)),
    time=xr.DataArray(np.array(_TIME_STRS)),
    media_time=xr.DataArray(np.array(_TIME_STRS)),
    population=xr.DataArray(
        coords={c.GEO: _GEO_IDS},
        data=np.array([1000.0, 1200.0]),
        name=c.POPULATION,
    ),
    kpi=xr.DataArray(
        coords={c.GEO: _GEO_IDS, c.TIME: _TIME_STRS},
        data=np.array([[50, 60], [70, 80]]),
        name=c.KPI,
    ),
    revenue_per_kpi=None,
    controls=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.TIME: _TIME_STRS,
            c.CONTROL_VARIABLE: _CONTROL_VARIABLES,
        },
        data=np.array([[[31, 32], [33, 34]], [[35, 36], [37, 38]]]),
        name=c.CONTROLS,
    ),
    media=xr.DataArray(
        coords={
            c.GEO: _GEO_IDS,
            c.MEDIA_TIME: _TIME_STRS,
            c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID,
        },
        data=np.array([[[39, 40], [41, 42]], [[43, 44], [45, 46]]]),
        name=c.MEDIA,
    ),
    media_spend=xr.DataArray(
        coords={c.MEDIA_CHANNEL: _MEDIA_CHANNEL_PAID},
        data=np.array([492, 496]),
        name=c.MEDIA_SPEND,
    ),
    media_spend_has_geo_dimension=False,
    media_spend_has_time_dimension=False,
    reach=None,
    frequency=None,
    rf_spend=None,
    organic_media=None,
    organic_reach=None,
    organic_frequency=None,
    non_media_treatments=None,
)

# Expected Protobuf (Textproto format)
MOCK_PROTO_NO_REVENUE_PER_KPI = text_format.Parse(
    """
    marketing_data_points {
      geo_info { geo_id: "geo_0" population: 1000 }
      date_interval {
        start_date { year: 2021 month: 2 day: 1 }
        end_date { year: 2021 month: 2 day: 8 }
      }
      control_variables { name: "control_0" value: 31.0 }
      control_variables { name: "control_1" value: 32.0 }
       media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 39.0
        }
       }
        media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 40.0
        }
       }
      kpi { name: "non_revenue" non_revenue { value: 50.0 } }
    }
    marketing_data_points {
      geo_info { geo_id: "geo_0" population: 1000 }
      date_interval {
        start_date { year: 2021 month: 2 day: 8 }
        end_date { year: 2021 month: 2 day: 15 }
      }
      control_variables { name: "control_0" value: 33.0 }
      control_variables { name: "control_1" value: 34.0 }
     media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 41.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 42.0
        }
      }
      kpi { name: "non_revenue" non_revenue { value: 60.0 } }
    }
    marketing_data_points {
      geo_info { geo_id: "geo_1" population: 1200 }
      date_interval {
        start_date { year: 2021 month: 2 day: 1 }
        end_date { year: 2021 month: 2 day: 8 }
      }
      control_variables { name: "control_0" value: 35.0 }
      control_variables { name: "control_1" value: 36.0 }
            media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 43.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 44.0
        }
      }
      kpi { name: "non_revenue" non_revenue { value: 70.0 } }
    }
    marketing_data_points {
      geo_info {
        geo_id: "geo_1"
        population: 1200
      }
      date_interval {
        start_date {
          year: 2021
          month: 2
          day: 8
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      control_variables {
        name: "control_0"
        value: 37.0
      }
      control_variables {
        name: "control_1"
        value: 38.0
      }
          media_variables {
        channel_name: "ch_paid_0"
        scalar_metric {
          name: "impressions"
          value: 45.0
        }
      }
      media_variables {
        channel_name: "ch_paid_1"
        scalar_metric {
          name: "impressions"
          value: 46.0
        }
      }
      kpi { name: "non_revenue" non_revenue { value: 80.0 } }
    }
    marketing_data_points {
        date_interval {
        start_date {
          year: 2021
          month: 2
          day: 1
        }
        end_date {
          year: 2021
          month: 2
          day: 15
        }
      }
      media_variables {
        channel_name: "ch_paid_0"
        media_spend: 492.0
      }
      media_variables {
        channel_name: "ch_paid_1"
        media_spend: 496.0
      }
    }
    metadata {
      time_dimensions { name: "time" dates { year: 2021 month: 2 day: 1 } dates { year: 2021 month: 2 day: 8} }
      time_dimensions { name: "media_time" dates { year: 2021 month: 2 day: 1 } dates { year: 2021 month: 2 day: 8 } }
      channel_dimensions { name: "media" channels: "ch_paid_0" channels: "ch_paid_1" }
      control_names: "control_0"
      control_names: "control_1"
      kpi_type: "non_revenue"
    }
    """,
    marketing_pb.MarketingData(),
)


# Hyperparameters test data
def get_default_model_spec() -> spec.ModelSpec:
  return spec.ModelSpec()


DEFAULT_HYPERPARAMETERS_PROTO = meridian_pb.Hyperparameters(
    media_effects_dist=_MediaEffectsDist.LOG_NORMAL,
    hill_before_adstock=False,
    max_lag=8,
    unique_sigma_for_each_geo=False,
    media_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    rf_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    paid_media_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    organic_media_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    organic_rf_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    non_media_treatments_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    enable_aks=False,
    global_adstock_decay='geometric',
)


def get_custom_model_spec_1() -> spec.ModelSpec:
  return spec.ModelSpec(
      prior=prior_distribution.PriorDistribution(),
      media_effects_dist=c.MEDIA_EFFECTS_NORMAL,
      hill_before_adstock=True,
      max_lag=777,
      unique_sigma_for_each_geo=True,
      media_prior_type=c.TREATMENT_PRIOR_TYPE_MROI,
      rf_prior_type=c.TREATMENT_PRIOR_TYPE_MROI,
      knots=2,
      baseline_geo='baseline_geo',
      roi_calibration_period=None,
      rf_roi_calibration_period=None,
      holdout_id=None,
      control_population_scaling_id=None,
      adstock_decay_spec='binomial',
  )


CUSTOM_HYPERPARAMETERS_PROTO_1 = meridian_pb.Hyperparameters(
    media_effects_dist=_MediaEffectsDist.NORMAL,
    hill_before_adstock=True,
    max_lag=777,
    unique_sigma_for_each_geo=True,
    media_prior_type=_PaidMediaPriorType.MROI,
    rf_prior_type=_PaidMediaPriorType.MROI,
    paid_media_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    organic_media_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    organic_rf_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    non_media_treatments_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    knots=[2],
    baseline_geo_string='baseline_geo',
    enable_aks=False,
    global_adstock_decay='binomial',
)


def get_custom_model_spec_2() -> spec.ModelSpec:
  return spec.ModelSpec(
      prior=prior_distribution.PriorDistribution(),
      media_effects_dist='log_normal',
      hill_before_adstock=True,
      max_lag=777,
      unique_sigma_for_each_geo=True,
      media_prior_type=c.TREATMENT_PRIOR_TYPE_ROI,
      rf_prior_type=c.TREATMENT_PRIOR_TYPE_ROI,
      organic_media_prior_type=c.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      organic_rf_prior_type=c.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      non_media_treatments_prior_type=c.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      non_media_baseline_values=['min', 0.5, 'max'],
      knots=[1, 5, 8],
      baseline_geo=3,
      roi_calibration_period=np.full((2, 3), True),
      rf_roi_calibration_period=np.full((4, 5), False),
      holdout_id=np.full((6,), True),
      control_population_scaling_id=np.full((7, 8), False),
      non_media_population_scaling_id=np.full((9, 10), False),
      adstock_decay_spec={'ch_paid_0': 'binomial', 'rf_ch_paid_1': 'geometric'},
  )

CUSTOM_HYPERPARAMETERS_PROTO_2 = meridian_pb.Hyperparameters(
    media_effects_dist=_MediaEffectsDist.LOG_NORMAL,
    hill_before_adstock=True,
    max_lag=777,
    unique_sigma_for_each_geo=True,
    media_prior_type=_PaidMediaPriorType.ROI,
    rf_prior_type=_PaidMediaPriorType.ROI,
    paid_media_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    organic_media_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    organic_rf_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_COEFFICIENT,
    non_media_treatments_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_COEFFICIENT,
    knots=[1, 5, 8],
    baseline_geo_int=3,
    roi_calibration_period=make_tensor_proto(
        dims=[2, 3],
        dtype=types_pb2.DT_BOOL,
        bool_vals=[True] * (2 * 3),
    ),
    rf_roi_calibration_period=make_tensor_proto(
        dims=[4, 5],
        dtype=types_pb2.DT_BOOL,
        bool_vals=[False] * (4 * 5),
    ),
    holdout_id=make_tensor_proto(
        dims=[6],
        dtype=types_pb2.DT_BOOL,
        bool_vals=[True] * 6,
    ),
    control_population_scaling_id=make_tensor_proto(
        dims=[7, 8],
        dtype=types_pb2.DT_BOOL,
        bool_vals=[False] * (7 * 8),
    ),
    non_media_population_scaling_id=make_tensor_proto(
        dims=[9, 10],
        dtype=types_pb2.DT_BOOL,
        bool_vals=[False] * (9 * 10),
    ),
    non_media_baseline_values=[
        meridian_pb.NonMediaBaselineValue(
            function_value=_NonMediaBaselineFunction.MIN
        ),
        meridian_pb.NonMediaBaselineValue(value=0.5),
        meridian_pb.NonMediaBaselineValue(
            function_value=_NonMediaBaselineFunction.MAX
        ),
    ],
    enable_aks=False,
    adstock_decay_by_channel=meridian_pb.AdstockDecayByChannel(
        channel_decays={'ch_paid_0': 'binomial', 'rf_ch_paid_1': 'geometric'}
    ),
)


def get_custom_model_spec_3() -> spec.ModelSpec:
  return spec.ModelSpec(
      prior=prior_distribution.PriorDistribution(),
      media_effects_dist=c.MEDIA_EFFECTS_NORMAL,
      hill_before_adstock=True,
      max_lag=777,
      unique_sigma_for_each_geo=True,
      media_prior_type=c.TREATMENT_PRIOR_TYPE_MROI,
      rf_prior_type=c.TREATMENT_PRIOR_TYPE_MROI,
      organic_media_prior_type=c.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      organic_rf_prior_type=c.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      non_media_treatments_prior_type=c.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      baseline_geo='baseline_geo',
      roi_calibration_period=None,
      rf_roi_calibration_period=None,
      holdout_id=None,
      control_population_scaling_id=None,
      enable_aks=True,
  )


CUSTOM_HYPERPARAMETERS_PROTO_3 = meridian_pb.Hyperparameters(
    media_effects_dist=_MediaEffectsDist.NORMAL,
    hill_before_adstock=True,
    max_lag=777,
    unique_sigma_for_each_geo=True,
    media_prior_type=_PaidMediaPriorType.MROI,
    rf_prior_type=_PaidMediaPriorType.MROI,
    paid_media_prior_type=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    organic_media_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    organic_rf_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    non_media_treatments_prior_type=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION,
    baseline_geo_string='baseline_geo',
    enable_aks=True,
    global_adstock_decay='geometric',
)


def _create_tfp_params_from_dict(
    param_dict: dict[str, Any],
    distribution: backend.tfd.Distribution | backend.bijectors.Bijector,
) -> dict[str, meridian_pb.TfpParameterValue]:
  param_dict.update({
      'validate_args': False,
  })
  return {
      key: _create_tfp_param(key, value, distribution)
      for key, value in param_dict.items()
  }


def create_distribution_proto(
    distribution_type: str, **kwargs
) -> meridian_pb.TfpDistribution:
  distribution = getattr(backend.tfd, distribution_type)
  return meridian_pb.TfpDistribution(
      distribution_type=distribution_type,
      parameters=_create_tfp_params_from_dict(kwargs, distribution),
  )


def create_bijector_proto(
    bijector_type: str, **kwargs
) -> meridian_pb.TfpBijector:
  bijector = getattr(backend.bijectors, bijector_type)
  return meridian_pb.TfpBijector(
      bijector_type=bijector_type,
      parameters=_create_tfp_params_from_dict(kwargs, bijector),
  )


def _create_tfp_param(param_name, param_value, distribution):
  """Creates a TfpParameterValue object based on the input value's type."""
  match param_value:
    case float():
      return meridian_pb.TfpParameterValue(scalar_value=param_value)
    case int():
      return meridian_pb.TfpParameterValue(int_value=param_value)
    case bool():
      return meridian_pb.TfpParameterValue(bool_value=param_value)
    case str():
      return meridian_pb.TfpParameterValue(string_value=param_value)
    case None:
      return meridian_pb.TfpParameterValue(none_value=True)
    case list():
      value_generator = (
          _create_tfp_param(param_name, v, distribution) for v in param_value
      )
      tfp_list_value = meridian_pb.TfpParameterValue.List(
          values=value_generator
      )
      return meridian_pb.TfpParameterValue(list_value=tfp_list_value)
    case dict():
      dict_value = {
          key: _create_tfp_param(key, v, distribution)
          for key, v in param_value.items()
      }
      return meridian_pb.TfpParameterValue(dict_value=dict_value)
    case _ if isinstance(param_value, (np.ndarray, backend.Tensor)):
      return meridian_pb.TfpParameterValue(
          tensor_value=backend.make_tensor_proto(param_value)
      )
    case meridian_pb.TfpDistribution():
      return meridian_pb.TfpParameterValue(distribution_value=param_value)
    case meridian_pb.TfpBijector():
      return meridian_pb.TfpParameterValue(bijector_value=param_value)
    case backend.tfd.ReparameterizationType():
      fully_reparameterized = param_value == backend.tfd.FULLY_REPARAMETERIZED
      return meridian_pb.TfpParameterValue(
          fully_reparameterized=fully_reparameterized
      )
    case types.FunctionType():
      # Add custom functions used for tests.
      test_registry = {'distribution_fn': distribution_fn}

      for function_key, func in test_registry.items():
        if func == param_value:  # pylint: disable=comparison-with-callable
          return meridian_pb.TfpParameterValue(
              function_param=meridian_pb.TfpParameterValue.FunctionParam(
                  function_key=function_key
              )
          )
      # Function has default value.
      signature = inspect.signature(distribution.__init__)
      param = signature.parameters[param_name]
      if param.default:
        return meridian_pb.TfpParameterValue(
            function_param=meridian_pb.TfpParameterValue.FunctionParam(
                uses_default=True
            )
        )
      raise TypeError(
          f'No function found in registry for "{param_value.__name__}"'
      )
    case _:
      # Handle unsupported types.
      raise TypeError(f'Unsupported type: {type(param_value)}')


# Arbitrary function used for testing `tfd.Autoregressive`.
# https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/autoregressive_test.py#L89
def distribution_fn(sample0):
  num_frames = sample0.shape[-1]
  mask = backend.one_hot(0, num_frames)[:, backend.newaxis]
  probs = backend.roll(backend.one_hot(sample0, 3), shift=1, axis=-2)
  probs = probs * (1.0 - mask) + backend.to_tensor([0.5, 0.5, 0]) * mask
  return backend.tfd.Independent(
      backend.tfd.Categorical(probs=probs), reinterpreted_batch_ndims=1
  )


def get_default_kwargs_split_fn():
  """Returns the default `kwargs_split_fn` used for tfd Distributions."""
  # `dist` can be any Distribution that has kwargs_split_fn in its signature.
  dist = backend.tfd.TransformedDistribution
  signature = inspect.signature(dist.__init__)
  kwargs_split_fn_param = signature.parameters['kwargs_split_fn']
  return kwargs_split_fn_param.default
