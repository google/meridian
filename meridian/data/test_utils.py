# Copyright 2024 The Meridian Authors.
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

"""Helper functions for Data module unit tests."""

from collections.abc import Sequence
import dataclasses
import datetime
import immutabledict
from meridian import constants as c
from meridian.data import input_data
from meridian.data import load
import numpy as np
import pandas as pd
import xarray as xr

_SAMPLE_START_DATE = datetime.date(2021, 1, 25)


def _sample_names(prefix: str, n_names: int | None) -> list[str] | None:
  """Generates a list of sample names.

  It concatenates the same prefix with consecutive numbers to generate a list
  of strings that can be used as sample names of columns/arrays/etc.
  """
  return [prefix + str(n) for n in range(n_names)] if n_names else None


def _sample_times(
    n_times: int,
    start_date: datetime.date = _SAMPLE_START_DATE,
    date_format: str = c.DATE_FORMAT,
) -> list[str]:
  """Generates sample `time`s."""
  return [
      (start_date + datetime.timedelta(weeks=w)).strftime(date_format)
      for w in range(n_times)
  ]


_REQUIRED_COORDS = immutabledict.immutabledict({
    c.GEO: ['geo_0', 'geo_1'],
    c.TIME: _sample_times(n_times=3),
    c.MEDIA_TIME: _sample_times(n_times=3),
    c.CONTROL_VARIABLE: ['control_0', 'control_1'],
})
_MEDIA_COORDS = immutabledict.immutabledict(
    {c.MEDIA_CHANNEL: ['media_channel_0', 'media_channel_1', 'media_channel_2']}
)
_RF_COORDS = immutabledict.immutabledict(
    {c.RF_CHANNEL: ['rf_channel_0', 'rf_channel_1']}
)

_REQUIRED_DATA_VARS = immutabledict.immutabledict({
    c.KPI: (['geo', 'time'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    c.CONTROLS: (
        ['geo', 'time', 'control_variable'],
        [
            [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
            [[2.11, 2.21], [2.31, 2.41], [2.51, 2.61]],
        ],
    ),
    c.POPULATION: (['geo'], [3.1, 3.1]),
})
_OPTIONAL_DATA_VARS = immutabledict.immutabledict({
    c.REVENUE_PER_KPI: (['geo', 'time'], [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]),
})
_MEDIA_DATA_VARS = immutabledict.immutabledict({
    c.MEDIA: (
        ['geo', 'media_time', 'media_channel'],
        [
            [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]],
            [[4.11, 4.21, 4.31], [4.41, 4.51, 4.61], [4.71, 4.81, 4.91]],
        ],
    ),
    c.MEDIA_SPEND: (
        ['geo', 'time', 'media_channel'],
        [
            [[5.1, 5.2, 5.3], [5.4, 5.5, 5.6], [5.7, 5.8, 5.9]],
            [[5.11, 5.21, 5.31], [5.41, 5.51, 5.61], [5.71, 5.81, 5.91]],
        ],
    ),
})
_RF_DATA_VARS = immutabledict.immutabledict({
    c.REACH: (
        ['geo', 'media_time', 'rf_channel'],
        [
            [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
            [[6.11, 6.21], [6.31, 6.41], [6.51, 6.61]],
        ],
    ),
    c.FREQUENCY: (
        ['geo', 'media_time', 'rf_channel'],
        [
            [[7.1, 7.2], [7.3, 7.4], [7.5, 7.6]],
            [[7.11, 7.21], [7.31, 7.41], [7.51, 7.61]],
        ],
    ),
    c.RF_SPEND: (
        ['geo', 'time', 'rf_channel'],
        [
            [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
            [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
        ],
    ),
})

WRONG_DATASET_WO_MEDIA_WO_RF = xr.Dataset(
    coords=_REQUIRED_COORDS,
    data_vars=_REQUIRED_DATA_VARS,
)
WRONG_DATASET_PARTIAL_MEDIA_WO_RF = xr.Dataset(
    coords=_REQUIRED_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | {
        c.MEDIA: (
            ['geo', 'media_time', 'media_channel'],
            [
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]],
                [[4.11, 4.21, 4.31], [4.41, 4.51, 4.61], [4.71, 4.81, 4.91]],
            ],
        ),
    },
)
WRONG_DATASET_WO_MEDIA_PARTIAL_RF = xr.Dataset(
    coords=_REQUIRED_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | {
        c.REACH: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
                [[6.11, 6.21], [6.31, 6.41], [6.51, 6.61]],
            ],
        ),
        c.RF_SPEND: (
            ['geo', 'time', 'rf_channel'],
            [
                [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
                [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
            ],
        ),
    },
)
WRONG_DATASET_PARTIAL_MEDIA_PARTIAL_RF = xr.Dataset(
    coords=_REQUIRED_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | {
        c.MEDIA_SPEND: (
            ['geo', 'time', 'media_channel'],
            [
                [[5.1, 5.2, 5.3], [5.4, 5.5, 5.6], [5.7, 5.8, 5.9]],
                [[5.11, 5.21, 5.31], [5.41, 5.51, 5.61], [5.71, 5.81, 5.91]],
            ],
        ),
    }
    | {
        c.REACH: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
                [[6.11, 6.21], [6.31, 6.41], [6.51, 6.61]],
            ],
        ),
        c.RF_SPEND: (
            ['geo', 'time', 'rf_channel'],
            [
                [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
                [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
            ],
        ),
    },
)
WRONG_DATASET_W_MEDIA_PARTIAL_RF = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _MEDIA_DATA_VARS
    | {
        c.REACH: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
                [[6.11, 6.21], [6.31, 6.41], [6.51, 6.61]],
            ],
        ),
        c.RF_SPEND: (
            ['geo', 'time', 'rf_channel'],
            [
                [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
                [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
            ],
        ),
    },
)
WRONG_DATASET_PARTIAL_MEDIA_W_RF = xr.Dataset(
    coords=_REQUIRED_COORDS | _RF_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _RF_DATA_VARS
    | {
        c.MEDIA: (
            ['geo', 'media_time', 'media_channel'],
            [
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]],
                [[4.11, 4.21, 4.31], [4.41, 4.51, 4.61], [4.71, 4.81, 4.91]],
            ],
        ),
    },
)

DATASET_WITHOUT_GEO_VARIATION_IN_CONTROLS = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_MEDIA_DATA_VARS
    | _RF_DATA_VARS
    | {
        c.KPI: (['geo', 'time'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        c.REVENUE_PER_KPI: (
            ['geo', 'time'],
            [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
        ),
        c.CONTROLS: (
            ['geo', 'time', 'control_variable'],
            [
                [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
                [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
            ],
        ),
        c.POPULATION: (['geo'], [3.1, 3.1]),
    },
)

DATASET_WITHOUT_GEO_VARIATION_IN_MEDIA = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _OPTIONAL_DATA_VARS
    | _RF_DATA_VARS
    | {
        c.MEDIA: (
            ['geo', 'media_time', 'media_channel'],
            [
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]],
                [[4.11, 4.2, 4.3], [4.41, 4.5, 4.6], [4.71, 4.8, 4.9]],
            ],
        ),
        c.MEDIA_SPEND: (
            ['geo', 'time', 'media_channel'],
            [
                [[5.1, 5.2, 5.3], [5.4, 5.5, 5.6], [5.7, 5.8, 5.9]],
                [[5.11, 5.21, 5.31], [5.41, 5.51, 5.61], [5.71, 5.81, 5.91]],
            ],
        ),
    },
)

DATASET_WITHOUT_GEO_VARIATION_IN_REACH = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _OPTIONAL_DATA_VARS
    | _MEDIA_DATA_VARS
    | {
        c.REACH: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
                [[6.1, 6.2], [6.3, 6.4], [6.5, 6.6]],
            ],
        ),
        c.FREQUENCY: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[7.1, 7.2], [7.3, 7.4], [7.5, 7.6]],
                [[7.11, 7.21], [7.31, 7.41], [7.51, 7.61]],
            ],
        ),
        c.RF_SPEND: (
            ['geo', 'time', 'rf_channel'],
            [
                [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
                [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
            ],
        ),
    },
)

DATASET_WITHOUT_TIME_VARIATION_IN_CONTROLS = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_MEDIA_DATA_VARS
    | _RF_DATA_VARS
    | {
        c.KPI: (['geo', 'time'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        c.REVENUE_PER_KPI: (
            ['geo', 'time'],
            [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
        ),
        c.CONTROLS: (
            ['geo', 'time', 'control_variable'],
            [
                [[2.1, 2.2], [2.1, 2.2], [2.1, 2.2]],
                [[2.7, 2.8], [2.7, 2.8], [2.7, 2.8]],
            ],
        ),
        c.POPULATION: (['geo'], [3.1, 3.1]),
    },
)

DATASET_WITHOUT_TIME_VARIATION_IN_MEDIA = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _OPTIONAL_DATA_VARS
    | _RF_DATA_VARS
    | {
        c.MEDIA: (
            ['geo', 'media_time', 'media_channel'],
            [
                [[4.1, 4.2, 4.3], [4.4, 4.2, 4.3], [4.7, 4.2, 4.3]],
                [[4.11, 4.21, 4.31], [4.41, 4.21, 4.31], [4.71, 4.21, 4.31]],
            ],
        ),
        c.MEDIA_SPEND: (
            ['geo', 'time', 'media_channel'],
            [
                [[5.1, 5.2, 5.3], [5.4, 5.5, 5.6], [5.7, 5.8, 5.9]],
                [[5.11, 5.21, 5.31], [5.41, 5.51, 5.61], [5.71, 5.81, 5.91]],
            ],
        ),
    },
)

DATASET_WITHOUT_TIME_VARIATION_IN_REACH = xr.Dataset(
    coords=_REQUIRED_COORDS | _MEDIA_COORDS | _RF_COORDS,
    data_vars=_REQUIRED_DATA_VARS
    | _OPTIONAL_DATA_VARS
    | _MEDIA_DATA_VARS
    | {
        c.REACH: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[6.1, 6.2], [6.1, 6.2], [6.1, 6.2]],
                [[6.7, 6.8], [6.7, 6.8], [6.7, 6.8]],
            ],
        ),
        c.FREQUENCY: (
            ['geo', 'media_time', 'rf_channel'],
            [
                [[7.1, 7.2], [7.3, 7.4], [7.5, 7.6]],
                [[7.11, 7.21], [7.31, 7.41], [7.51, 7.61]],
            ],
        ),
        c.RF_SPEND: (
            ['geo', 'time', 'rf_channel'],
            [
                [[8.1, 8.2], [8.3, 8.4], [8.5, 8.8]],
                [[8.11, 8.21], [8.31, 8.41], [8.51, 8.61]],
            ],
        ),
    },
)

_NATIONAL_COORDS = immutabledict.immutabledict({
    c.TIME: [
        _SAMPLE_START_DATE.strftime(c.DATE_FORMAT),
        (_SAMPLE_START_DATE + datetime.timedelta(weeks=1)).strftime(
            c.DATE_FORMAT
        ),
    ],
    c.CONTROL_VARIABLE: ['control_0', 'control_1'],
    c.MEDIA_TIME: [
        _SAMPLE_START_DATE.strftime(c.DATE_FORMAT),
        (_SAMPLE_START_DATE + datetime.timedelta(weeks=1)).strftime(
            c.DATE_FORMAT
        ),
    ],
    c.MEDIA_CHANNEL: ['media_channel_0'],
})

_NATIONAL_DATA_VARS_W_GEO = immutabledict.immutabledict({
    c.CONTROLS: (
        ['geo', 'time', 'control_variable'],
        [[[0.1, 0.2], [0.3, 0.4]]],
    ),
    c.MEDIA: (['geo', 'media_time', 'media_channel'], [[[1.1], [1.2]]]),
    c.KPI: (['geo', 'time'], [[2.1, 2.2]]),
    c.MEDIA_SPEND: (['geo', 'time', 'media_channel'], [[[3.1], [3.2]]]),
    c.REVENUE_PER_KPI: (['geo', 'time'], [[4.1, 4.2]]),
})
_NATIONAL_DATA_VARS_WO_GEO = immutabledict.immutabledict({
    c.CONTROLS: (['time', 'control_variable'], [[0.1, 0.2], [0.3, 0.4]]),
    c.MEDIA: (['media_time', 'media_channel'], [[1.1], [1.2]]),
    c.KPI: (['time'], [2.1, 2.2]),
    c.MEDIA_SPEND: (['time', 'media_channel'], [[3.1], [3.2]]),
    c.REVENUE_PER_KPI: (['time'], [4.1, 4.2]),
})

NATIONAL_DATASET_W_POPULATION_W_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS | {c.GEO: ['geo_0']},
    data_vars=_NATIONAL_DATA_VARS_W_GEO | {c.POPULATION: (['geo'], [5.1])},
)
NATIONAL_DATASET_W_SCALAR_POPULATION_WO_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS,
    data_vars=_NATIONAL_DATA_VARS_WO_GEO | {c.POPULATION: 5.1},
)
NATIONAL_DATASET_W_SINGLE_POPULATION_WO_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS,
    data_vars=_NATIONAL_DATA_VARS_WO_GEO | {c.POPULATION: 5.1},
)
NATIONAL_DATASET_W_NONE_POPULATION_WO_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS,
    data_vars=_NATIONAL_DATA_VARS_WO_GEO | {c.POPULATION: None},
)
NATIONAL_DATASET_WO_POPULATION_W_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS | {c.GEO: ['geo_0']},
    data_vars=_NATIONAL_DATA_VARS_W_GEO,
)
NATIONAL_DATASET_WO_POPULATION_WO_GEO = xr.Dataset(
    coords=_NATIONAL_COORDS,
    data_vars=_NATIONAL_DATA_VARS_WO_GEO,
)
EXPECTED_NATIONAL_DATASET = xr.Dataset(
    coords=_NATIONAL_COORDS | {c.GEO: [c.NATIONAL_MODEL_DEFAULT_GEO_NAME]},
    data_vars=_NATIONAL_DATA_VARS_W_GEO | {c.POPULATION: (['geo'], [1.0])},
)


NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO = immutabledict.immutabledict({
    'time': [
        _SAMPLE_START_DATE.strftime(c.DATE_FORMAT),
        (_SAMPLE_START_DATE + datetime.timedelta(weeks=1)).strftime(
            c.DATE_FORMAT
        ),
    ],
    'kpi': [0.1, 0.2],
    'revenue_per_kpi': [1.1, 1.2],
    'media_0': [3.1, 3.2],
    'media_1': [3.3, 3.4],
    'media_2': [3.5, 3.6],
    'media_spend_0': [4.1, 4.2],
    'media_spend_1': [4.3, 4.4],
    'media_spend_2': [4.5, 4.6],
    'control_0': [5.1, 5.2],
    'control_1': [5.3, 5.4],
})
NATIONAL_DATA_DICT_W_POPULATION_W_GEO = (
    NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO
    | {
        'geo': 'geo_0',
        'population': 2.1,
    }
)
NATIONAL_DATA_DICT_W_POPULATION_W_GEO_RENAMED = (
    NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO
    | {
        'City': 'geo_0',
        'Population': 2.1,
    }
)
NATIONAL_DATA_DICT_W_POPULATION_WO_GEO = (
    NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO | {'population': 2.1}
)
NATIONAL_DATA_DICT_WO_POPULATION_W_GEO = (
    NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO | {'geo': 'geo_0'}
)
EXPECTED_NATIONAL_DATA_DICT = NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO | {
    'geo': c.NATIONAL_MODEL_DEFAULT_GEO_NAME,
    'population': c.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE,
}

NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO = load.CoordToColumns(
    time='time',
    kpi='kpi',
    revenue_per_kpi='revenue_per_kpi',
    media=['media_0', 'media_1', 'media_2'],
    media_spend=['media_spend_0', 'media_spend_1', 'media_spend_2'],
    controls=['control_0', 'control_1'],
)
NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO_RENAMED = dataclasses.replace(
    NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
    population='Population',
    geo='City',
)
NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO = dataclasses.replace(
    NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
    population='population',
    geo='geo',
)
NATIONAL_COORD_TO_COLUMNS_W_POPULATION_WO_GEO = dataclasses.replace(
    NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
    population='population',
)
NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_W_GEO = dataclasses.replace(
    NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
    geo='geo',
)


def random_media_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_media_channels: int,
    seed: int = 0,
    date_format: str = c.DATE_FORMAT,
    explicit_time_index: Sequence[str] | None = None,
    explicit_media_channel_names: Sequence[str] | None = None,
    array_name: str = 'media',
    channel_variable_name: str = 'media_channel',
    channel_prefix: str = 'ch_',
) -> xr.DataArray:
  """Generates a sample `media` DataArray.

  Args:
    n_geos: Number of geos
    n_times: Number of time periods
    n_media_times: Number of media time periods
    n_media_channels: Number of media channels
    seed: Random seed used by `np.random.seed()`
    date_format: The date format to use for time coordinate labels
    explicit_time_index: If given, ignore `date_format` and use this as is
    explicit_media_channel_names: If given, ignore `n_media_channels` and use
      this as is
    array_name: The name of the array to be created
    channel_variable_name: The name of the channel variable
    channel_prefix: The prefix of the channel names

  Returns:
    A DataArray containing random data.
  """

  np.random.seed(seed)

  start_date = _SAMPLE_START_DATE
  if n_times < n_media_times:
    start_date -= datetime.timedelta(weeks=(n_media_times - n_times))

  media = np.round(
      abs(
          np.random.normal(5, 5, size=(n_geos, n_media_times, n_media_channels))
      )
  )
  if explicit_time_index is None:
    media_time = _sample_times(
        n_times=n_media_times,
        start_date=start_date,
        date_format=date_format,
    )
  else:
    media_time = explicit_time_index

  media_channels = (
      explicit_media_channel_names
      if explicit_media_channel_names is not None
      else _sample_names(prefix=channel_prefix, n_names=n_media_channels)
  )
  return xr.DataArray(
      media,
      dims=['geo', 'media_time', channel_variable_name],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'media_time': media_time,
          channel_variable_name: media_channels,
      },
      name=array_name,
  )


def random_organic_media_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_organic_media_channels: int,
    seed: int = 0,
    date_format: str = c.DATE_FORMAT,
    explicit_time_index: Sequence[str] | None = None,
    explicit_media_channel_names: Sequence[str] | None = None,
) -> xr.DataArray:
  """Generates a sample `organic_media` DataArray."""
  return random_media_da(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_media_channels=n_organic_media_channels,
      seed=seed,
      date_format=date_format,
      explicit_time_index=explicit_time_index,
      explicit_media_channel_names=explicit_media_channel_names,
      array_name='organic_media',
      channel_variable_name='organic_media_channel',
      channel_prefix='organic_media_',
  )


def random_media_spend_nd_da(
    n_geos: int | None = None,
    n_times: int | None = None,
    n_media_channels: int | None = None,
    seed=0,
) -> xr.DataArray:
  """Generates a sample N-dimensional `media_spend` DataArray.

  This function generates a 1-D, 2-D or 3-D version of the `media_spend`
  DataArray depending on the `n_geos`, `n_times` and `n_media_channels`
  arguments. There are only 2 shapes accepted by the `InputData` class:
  `(geo, time, media_channel)` and `(media_channel)` but we use also the 2-D
  version of this function to test if `InputData` fails to initialize with a 2-D
  media_spend data.

  Args:
    n_geos: Number of geos in the created `media_spend` array or `None` if the
      created `media_spend` should not contain the `geo` dimension.
    n_times: Number of time periods in the created `media_spend` array or `None`
      if the created array should not contain the `time` dimension.
    n_media_channels: Number of channels in the created `media_spend` array.
    seed: Random seed used by `np.random.seed()`.

  Returns:
    A DataArray containing the generated `media_spend` data with the given
    dimensions.
  """
  np.random.seed(seed)

  dims = []
  coords = {}
  if n_geos is not None:
    dims.append('geo')
    coords['geo'] = _sample_names(prefix='geo_', n_names=n_geos)
  if n_times is not None:
    dims.append('time')
    coords['time'] = _sample_times(n_times=n_times)
  if n_media_channels is not None:
    dims.append('media_channel')
    coords['media_channel'] = _sample_names(
        prefix='ch_', n_names=n_media_channels
    )

  if dims == ['geo', 'time', 'media_channel']:
    shape = (n_geos, n_times, n_media_channels)
  elif dims == ['media_channel']:
    shape = (n_media_channels,)
  elif dims == ['geo', 'media_channel']:
    shape = (n_geos, n_media_channels)
  elif dims == ['time', 'media_channel']:
    shape = (n_times, n_media_channels)
  else:
    raise ValueError(
        f'Shape {dims} not supported by the random_media_spend_nd_da function.'
    )

  media_spend = abs(np.random.normal(1, 1, size=shape))

  return xr.DataArray(
      media_spend,
      dims=dims,
      coords=coords,
      name='media_spend',
  )


def random_controls_da(
    media: xr.DataArray,
    n_geos: int,
    n_times: int,
    n_controls: int,
    seed: int = 0,
    date_format: str = c.DATE_FORMAT,
    explicit_time_index: Sequence[str] | None = None,
) -> xr.DataArray:
  """Generates a sample `controls` DataArray.

  Args:
    media: The media data array
    n_geos: Number of geos
    n_times: Number of time periods
    n_controls: Number of controls
    seed: Random seed used by `np.random.seed()`
    date_format: The date format to use for time coordinate labels
    explicit_time_index: If given, ignore `date_format` and use this as is

  Returns:
    A DataArray containing random controls variable.
  """

  np.random.seed(seed)

  controls_var = abs(np.random.normal(2, 1, size=(n_geos, n_times, 1)))
  media_common = media[:, (len(media.coords['media_time']) - n_times) :, :]
  controls = np.random.normal(
      0.8 * media_common.mean(axis=2, keepdims=True),
      controls_var,
      size=(n_geos, n_times, n_controls),
  )
  return xr.DataArray(
      controls,
      dims=['geo', 'time', 'control_variable'],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'time': (
              _sample_times(n_times=n_times, date_format=date_format)
              if explicit_time_index is None
              else explicit_time_index
          ),
          'control_variable': _sample_names(
              prefix='control_', n_names=n_controls
          ),
      },
      name='controls',
  )


def random_kpi_da(
    media: xr.DataArray,
    controls: xr.DataArray,
    n_geos: int,
    n_times: int,
    n_media_channels: int,
    n_controls: int,
    seed: int = 0,
) -> xr.DataArray:
  """Generates a sample `kpi` DataArray."""

  np.random.seed(seed)

  # Adds geo variance to media and geos to help model convergence.
  media_common = media[:, (len(media.coords['media_time']) - n_times) :, :]
  media_geo_sd = abs(np.random.normal(0, 5, size=n_geos))
  media_geo_sd = np.repeat(
      np.repeat(media_geo_sd[:, np.newaxis], n_times, axis=1)[..., np.newaxis],
      n_media_channels,
      axis=2,
  )
  control_geo_sd = abs(np.random.normal(0, 5, size=n_geos))
  control_geo_sd = np.repeat(
      np.repeat(control_geo_sd[:, np.newaxis], n_times, axis=1)[
          ..., np.newaxis
      ],
      n_controls,
      axis=2,
  )

  # Simulates outcome which is the dependent variable. Typically this is the
  # number of units sold, but it can be any metric (e.g. revenue).
  media_portion = np.random.normal(media_common, media_geo_sd).sum(axis=2)
  control_portion = np.random.normal(controls, control_geo_sd).sum(axis=2)
  error = np.random.normal(0, 2, size=(n_geos, n_times))
  kpi = abs(media_portion + control_portion + error)

  return xr.DataArray(
      kpi,
      dims=['geo', 'time'],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'time': _sample_times(n_times=n_times),
      },
      name=c.KPI,
  )


def constant_revenue_per_kpi(
    n_geos: int, n_times: int, value: float
) -> xr.DataArray:
  """Generates a constant `revenue_per_kpi` DataArray."""

  revenue_per_kpi = value * np.ones((n_geos, n_times))

  return xr.DataArray(
      revenue_per_kpi,
      dims=['geo', 'time'],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'time': _sample_times(n_times=n_times),
      },
      name='revenue_per_kpi',
  )


def random_population(n_geos: int, seed: int = 0) -> xr.DataArray:
  """Generates a sample `population` DataArray."""

  np.random.seed(seed)

  population = np.round(10 + abs(np.random.normal(3000, 100, size=n_geos)))

  return xr.DataArray(
      population,
      dims=['geo'],
      coords={'geo': _sample_names(prefix='geo_', n_names=n_geos)},
      name='population',
  )


def random_reach_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_rf_channels: int,
    seed: int = 0,
    explicit_rf_channel_names: Sequence[str] | None = None,
    array_name: str = 'reach',
    channel_variable_name: str = 'rf_channel',
    channel_prefix: str = 'rf_ch_',
) -> xr.DataArray:
  """Generates a sample `reach` DataArray."""

  np.random.seed(seed)

  start_date = _SAMPLE_START_DATE
  if n_times < n_media_times:
    start_date -= datetime.timedelta(weeks=(n_media_times - n_times))

  reach = np.round(
      abs(
          np.random.normal(
              3000, 100, size=(n_geos, n_media_times, n_rf_channels)
          )
      )
  )

  channels = (
      explicit_rf_channel_names
      if explicit_rf_channel_names is not None
      else _sample_names(prefix=channel_prefix, n_names=n_rf_channels)
  )
  return xr.DataArray(
      reach,
      dims=['geo', 'media_time', channel_variable_name],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'media_time': _sample_times(
              n_times=n_media_times, start_date=start_date
          ),
          channel_variable_name: channels,
      },
      name=array_name,
  )


def random_organic_reach_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_organic_rf_channels: int,
    seed: int = 0,
    explicit_organic_rf_channel_names: Sequence[str] | None = None,
) -> xr.DataArray:
  """Generates a sample `organic_reach` DataArray."""
  return random_reach_da(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_rf_channels=n_organic_rf_channels,
      seed=seed,
      explicit_rf_channel_names=explicit_organic_rf_channel_names,
      array_name='organic_reach',
      channel_variable_name='organic_rf_channel',
      channel_prefix='organic_rf_ch_',
  )


def random_frequency_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_rf_channels: int,
    seed: int = 0,
    explicit_rf_channel_names: Sequence[str] | None = None,
    array_name: str = 'frequency',
    channel_variable_name: str = 'rf_channel',
    channel_prefix: str = 'rf_ch_',
) -> xr.DataArray:
  """Generates a sample `frequency` DataArray."""

  np.random.seed(seed)

  start_date = _SAMPLE_START_DATE
  if n_times < n_media_times:
    start_date -= datetime.timedelta(weeks=(n_media_times - n_times))

  frequency = abs(
      np.random.normal(3, 5, size=(n_geos, n_media_times, n_rf_channels))
  )

  channels = (
      explicit_rf_channel_names
      if explicit_rf_channel_names is not None
      else _sample_names(prefix=channel_prefix, n_names=n_rf_channels)
  )

  return xr.DataArray(
      frequency,
      dims=['geo', 'media_time', channel_variable_name],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'media_time': _sample_times(
              n_times=n_media_times, start_date=start_date
          ),
          channel_variable_name: channels,
      },
      name=array_name,
  )


def random_organic_frequency_da(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_organic_rf_channels: int,
    seed: int = 0,
    explicit_organic_rf_channel_names: Sequence[str] | None = None,
) -> xr.DataArray:
  """Generates a sample `organic_frequency` DataArray."""
  return random_frequency_da(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_rf_channels=n_organic_rf_channels,
      seed=seed,
      explicit_rf_channel_names=explicit_organic_rf_channel_names,
      array_name='organic_frequency',
      channel_variable_name='organic_rf_channel',
      channel_prefix='organic_rf_ch_',
  )


def random_rf_spend_nd_da(
    n_geos: int | None = None,
    n_times: int | None = None,
    n_rf_channels: int | None = None,
    seed=0,
) -> xr.DataArray:
  """Generates a sample N-dimensional `rf_spend` DataArray.

  This function generates a 1-D, 2-D or 3-D version of the `rf_spend` DataArray
  depending on the `n_geos`, `n_times` and `n_rf_channels` arguments.
  There are 3 accepted shapes accepted by the `InputData` class:
  `(rf_channel)`, `(geo, time, rf_channel)` and `(geo, rf_channel)`.

  Args:
    n_geos: Number of geos in the created `rf_spend` array or `None` if the
      created `rf_spend` should not contain the `geo` dimension.
    n_times: Number of time periods in the created `rf_spend` array or `None` if
      the created array should not contain the `time` dimension.
    n_rf_channels: Number of channels in the created `rf_spend` array.
    seed: Random seed used by `np.random.seed()`.

  Returns:
    A DataArray containing the generated `rf_spend` data with the given
    dimensions.
  """
  np.random.seed(seed)

  dims = []
  coords = {}
  if n_geos is not None:
    dims.append('geo')
    coords['geo'] = _sample_names(prefix='geo_', n_names=n_geos)
  if n_times is not None:
    dims.append('time')
    coords['time'] = _sample_times(n_times=n_times)
  if n_rf_channels is not None:
    dims.append('rf_channel')
    coords['rf_channel'] = _sample_names(prefix='rf_ch_', n_names=n_rf_channels)

  if dims == ['geo', 'time', 'rf_channel']:
    shape = (n_geos, n_times, n_rf_channels)
  elif dims == ['rf_channel']:
    shape = (n_rf_channels,)
  elif dims == ['geo', 'rf_channel']:
    shape = (n_geos, n_rf_channels)
  else:
    raise ValueError(
        f'Shape {dims} not supported by the random_rf_spend_nd_da function.'
    )

  rf_spend = abs(np.random.normal(1, 1, size=shape))

  return xr.DataArray(
      rf_spend,
      dims=dims,
      coords=coords,
      name='rf_spend',
  )


def random_non_media_treatments_da(
    media: xr.DataArray,
    n_geos: int,
    n_times: int,
    n_non_media_channels: int,
    seed: int = 0,
    date_format: str = c.DATE_FORMAT,
    explicit_time_index: Sequence[str] | None = None,
) -> xr.DataArray:
  """Generates a sample `non_media_treatments` DataArray.

  Args:
    media: The media data array
    n_geos: Number of geos
    n_times: Number of time periods
    n_non_media_channels: Number of non media channels
    seed: Random seed used by `np.random.seed()`
    date_format: The date format to use for time coordinate labels
    explicit_time_index: If given, ignore `date_format` and use this as is

  Returns:
    A DataArray containing random non-media variable.
  """

  np.random.seed(seed)

  non_media_channel = abs(np.random.normal(2, 1, size=(n_geos, n_times, 1)))
  media_common = media[:, (len(media.coords['media_time']) - n_times) :, :]
  non_media_treatments = np.random.normal(
      0.8 * media_common.mean(axis=2, keepdims=True),
      non_media_channel,
      size=(n_geos, n_times, n_non_media_channels),
  )
  return xr.DataArray(
      non_media_treatments,
      dims=['geo', 'time', 'non_media_channel'],
      coords={
          'geo': _sample_names(prefix='geo_', n_names=n_geos),
          'time': (
              _sample_times(n_times=n_times, date_format=date_format)
              if explicit_time_index is None
              else explicit_time_index
          ),
          'non_media_channel': _sample_names(
              prefix='non_media_', n_names=n_non_media_channels
          ),
      },
      name='non_media_treatments',
  )


def random_dataset(
    n_geos: int,
    n_times: int,
    n_media_times: int,
    n_controls: int,
    n_non_media_channels: int | None = None,
    n_organic_media_channels: int | None = None,
    n_organic_rf_channels: int | None = None,
    n_media_channels: int | None = None,
    n_rf_channels: int | None = None,
    revenue_per_kpi_value: float | None = 3.14,
    seed: int = 0,
    remove_media_time: bool = False,
):
  """Generates a random dataset."""
  if n_media_channels:
    media = random_media_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        seed=seed,
    )
    media_spend = random_media_spend_nd_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_channels=n_media_channels,
        seed=seed,
    )
  else:
    media = None
    media_spend = None

  if n_rf_channels:
    reach = random_reach_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_rf_channels=n_rf_channels,
        seed=seed,
    )
    frequency = random_frequency_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_rf_channels=n_rf_channels,
        seed=seed,
    )
    rf_spend = random_rf_spend_nd_da(
        n_geos=n_geos,
        n_times=n_times,
        n_rf_channels=n_rf_channels,
        seed=seed,
    )
  else:
    reach = None
    frequency = None
    rf_spend = None

  if revenue_per_kpi_value is not None:
    revenue_per_kpi = constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=revenue_per_kpi_value
    )
  else:
    revenue_per_kpi = None

  controls = random_controls_da(
      media=media if n_media_channels else reach,
      n_geos=n_geos,
      n_times=n_times,
      n_controls=n_controls,
      seed=seed,
  )
  if n_non_media_channels:
    non_media_treatments = random_non_media_treatments_da(
        media=media if n_media_channels else reach,
        n_geos=n_geos,
        n_times=n_times,
        n_non_media_channels=n_non_media_channels,
        seed=seed,
    )
  else:
    non_media_treatments = None

  if n_organic_media_channels:
    organic_media = random_organic_media_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_organic_media_channels=n_organic_media_channels,
        seed=seed,
    )
  else:
    organic_media = None

  if n_organic_rf_channels:
    organic_reach = random_organic_reach_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_organic_rf_channels=n_organic_rf_channels,
        seed=seed,
    )
    organic_frequency = random_organic_frequency_da(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_organic_rf_channels=n_organic_rf_channels,
        seed=seed,
    )
  else:
    organic_reach = None
    organic_frequency = None

  kpi = random_kpi_da(
      media=media if n_media_channels else reach,
      controls=controls,
      n_geos=n_geos,
      n_times=n_times,
      n_media_channels=n_media_channels or n_rf_channels or 0,
      n_controls=n_controls,
  )
  population = random_population(n_geos=n_geos, seed=seed)

  dataset = xr.combine_by_coords([kpi, population, controls])
  if revenue_per_kpi is not None:
    dataset = xr.combine_by_coords([dataset, revenue_per_kpi])
  if media is not None:
    media_renamed = (
        media.rename({'media_time': 'time'}) if remove_media_time else media
    )

    dataset = xr.combine_by_coords([dataset, media_renamed, media_spend])
  if reach is not None:
    reach_renamed = (
        reach.rename({'media_time': 'time'}) if remove_media_time else reach
    )
    frequency_renamed = (
        frequency.rename({'media_time': 'time'})
        if remove_media_time
        else frequency
    )
    dataset = xr.combine_by_coords(
        [dataset, reach_renamed, frequency_renamed, rf_spend]
    )
  if non_media_treatments is not None:
    dataset = xr.combine_by_coords([dataset, non_media_treatments])
  if organic_media is not None:
    organic_media_renamed = (
        organic_media.rename({'media_time': 'time'})
        if remove_media_time
        else organic_media
    )
    dataset = xr.combine_by_coords([dataset, organic_media_renamed])
  if organic_reach is not None:
    organic_reach_renamed = (
        organic_reach.rename({'media_time': 'time'})
        if remove_media_time
        else organic_reach
    )
    organic_frequency_renamed = (
        organic_frequency.rename({'media_time': 'time'})
        if remove_media_time
        else organic_frequency
    )
    dataset = xr.combine_by_coords(
        [dataset, organic_reach_renamed, organic_frequency_renamed]
    )
  return dataset


def dataset_to_dataframe(
    dataset: xr.Dataset,
    controls_column_names: list[str],
    media_column_names: list[str] | None = None,
    media_spend_column_names: list[str] | None = None,
    reach_column_names: list[str] | None = None,
    frequency_column_names: list[str] | None = None,
    rf_spend_column_names: list[str] | None = None,
    non_media_column_names: list[str] | None = None,
    organic_media_column_names: list[str] | None = None,
    organic_reach_column_names: list[str] | None = None,
    organic_frequency_column_names: list[str] | None = None,
) -> pd.DataFrame:
  """Converts a dataset into the `pd.DataFrame` format.

  Args:
    dataset: An `xarray.Dataset` consisting of the following DataArrays: `kpi`,
      `revenue_per_kpi`, `population`, `media`, `media_spend`, and `controls`.
    controls_column_names: A list of desired column names for controls data in
      the output DataFrame.
    media_column_names: A list of desired column names for media data in the
      output DataFrame.
    media_spend_column_names: A list of desired column names for media_spend
      data in the output DataFrame.
    reach_column_names: A list of desired column names for reach data in the
      output DataFrame.
    frequency_column_names: A list of desired column names for frequency data in
      the output DataFrame.
    rf_spend_column_names: A list of desired column names for `rf_spend` data in
      the output DataFrame.
    non_media_column_names: A list of desired column names for
      non_media_treatments data in the output DataFrame.
    organic_media_column_names: A list of desired column names for organic_media
      data in the output DataFrame.
    organic_reach_column_names: A list of desired column names for organic_reach
      data in the output DataFrame.
    organic_frequency_column_names: A list of desired column names for
      organic_frequency data in the output DataFrame.

  Returns:
    A Pandas DataFrame with columns `geo` and `time` followed by `kpi`,
    `revenue_per_kpi`, `population`, `controls`, `media`, `media_spend`,
    `reach`, `frequency`, `rf_spend`, `non_media_treatments`, `organic media`,
    `organic_reach`, and `organic_frequency` data.
  """
  kpi = dataset[c.KPI].to_dataframe(name=c.KPI)
  revenue_per_kpi = dataset[c.REVENUE_PER_KPI].to_dataframe(
      name=c.REVENUE_PER_KPI
  )
  population = dataset[c.POPULATION].to_dataframe(name=c.POPULATION)

  controls = dataset[c.CONTROLS].to_dataframe(name=c.CONTROLS).unstack()
  controls.columns = controls_column_names

  result = kpi.join(revenue_per_kpi).join(population).join(controls)

  if non_media_column_names is not None:
    non_media_treatments = (
        dataset[c.NON_MEDIA_TREATMENTS]
        .to_dataframe(name=c.NON_MEDIA_TREATMENTS)
        .unstack()
    )
    non_media_treatments.columns = non_media_column_names
    result = result.join(non_media_treatments, how='right')

  if media_column_names is not None:
    media = dataset[c.MEDIA].to_dataframe(name=c.MEDIA).unstack()
    media.columns = media_column_names
    media.index.names = [c.GEO, c.TIME]

    media_spend = (
        dataset[c.MEDIA_SPEND].to_dataframe(name=c.MEDIA_SPEND).unstack()
    )
    media_spend.columns = media_spend_column_names

    result = result.join(media, how='right').join(media_spend)

  if reach_column_names is not None:
    reach = dataset[c.REACH].to_dataframe(name=c.REACH).unstack()
    reach.columns = reach_column_names
    reach.index.names = [c.GEO, c.TIME]

    frequency = dataset[c.FREQUENCY].to_dataframe(name=c.FREQUENCY).unstack()
    frequency.columns = frequency_column_names
    frequency.index.names = [c.GEO, c.TIME]

    rf_spend = dataset[c.RF_SPEND].to_dataframe(name=c.RF_SPEND).unstack()
    rf_spend.columns = rf_spend_column_names

    result = (
        result.join(reach, how='right')
        .join(frequency, how='right')
        .join(rf_spend)
    )

  if organic_media_column_names is not None:
    organic_media = (
        dataset[c.ORGANIC_MEDIA].to_dataframe(name=c.ORGANIC_MEDIA).unstack()
    )
    organic_media.columns = organic_media_column_names
    organic_media.index.names = [c.GEO, c.TIME]
    result = result.join(organic_media, how='right')

  if organic_reach_column_names is not None:
    organic_reach = (
        dataset[c.ORGANIC_REACH].to_dataframe(name=c.ORGANIC_REACH).unstack()
    )
    organic_reach.columns = organic_reach_column_names
    organic_reach.index.names = [c.GEO, c.TIME]

    organic_frequency = (
        dataset[c.ORGANIC_FREQUENCY]
        .to_dataframe(name=c.ORGANIC_FREQUENCY)
        .unstack()
    )
    organic_frequency.columns = organic_frequency_column_names
    organic_frequency.index.names = [c.GEO, c.TIME]
    result = result.join(organic_reach, how='right').join(organic_frequency)

  return result.reset_index()


def random_dataframe(
    n_geos,
    n_times,
    n_media_times,
    n_controls,
    n_media_channels=None,
    n_rf_channels=None,
    seed=0,
):
  """Generates a DataFrame for a random dataset."""
  dataset = random_dataset(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_controls=n_controls,
      n_media_channels=n_media_channels,
      n_rf_channels=n_rf_channels,
      seed=seed,
  )

  return dataset_to_dataframe(
      dataset,
      controls_column_names=_sample_names('control_', n_controls),
      media_column_names=_sample_names('media_', n_media_channels),
      media_spend_column_names=_sample_names('media_spend_', n_media_channels),
      reach_column_names=_sample_names('reach_', n_rf_channels),
      frequency_column_names=_sample_names('frequency_', n_rf_channels),
      rf_spend_column_names=_sample_names('rf_spend_', n_rf_channels),
  )


def sample_coord_to_columns(
    n_controls: int,
    n_media_channels: int | None = None,
    n_rf_channels: int | None = None,
    n_non_media_channels: int | None = None,
    n_organic_media_channels: int | None = None,
    n_organic_rf_channels: int | None = None,
    include_revenue_per_kpi: bool = True,
) -> load.CoordToColumns:
  """Returns a sample `coord_to_columns` mapping for testing."""

  if n_media_channels is not None:
    media = _sample_names('media_', n_media_channels)
    media_spend = _sample_names('media_spend_', n_media_channels)
  else:
    media = None
    media_spend = None

  if n_rf_channels is not None:
    reach = _sample_names('reach_', n_rf_channels)
    frequency = _sample_names('frequency_', n_rf_channels)
    rf_spend = _sample_names('rf_spend_', n_rf_channels)
  else:
    reach = None
    frequency = None
    rf_spend = None

  if n_organic_media_channels is not None:
    organic_media = _sample_names('organic_media_', n_organic_media_channels)
  else:
    organic_media = None

  if n_organic_rf_channels is not None:
    organic_reach = _sample_names('organic_reach_', n_organic_rf_channels)
    organic_frequency = _sample_names(
        'organic_frequency_', n_organic_rf_channels
    )
  else:
    organic_reach = None
    organic_frequency = None

  if n_non_media_channels is not None:
    non_media_treatments = _sample_names('non_media_', n_non_media_channels)
  else:
    non_media_treatments = None

  return load.CoordToColumns(
      geo=c.GEO,
      time=c.TIME,
      kpi=c.KPI,
      revenue_per_kpi=c.REVENUE_PER_KPI if include_revenue_per_kpi else None,
      population=c.POPULATION,
      controls=_sample_names('control_', n_controls),
      media=media,
      media_spend=media_spend,
      reach=reach,
      frequency=frequency,
      rf_spend=rf_spend,
      non_media_treatments=non_media_treatments,
      organic_media=organic_media,
      organic_reach=organic_reach,
      organic_frequency=organic_frequency,
  )


def sample_input_data_from_dataset(dataset: xr.Dataset, kpi_type: str):
  """Generates a sample `InputData` from a full xarray Dataset."""
  return input_data.InputData(
      kpi=dataset.kpi,
      kpi_type=kpi_type,
      revenue_per_kpi=dataset.revenue_per_kpi,
      population=dataset.population,
      controls=dataset.controls,
      media=dataset.media,
      media_spend=dataset.media_spend,
      reach=dataset.reach,
      frequency=dataset.frequency,
      rf_spend=dataset.rf_spend,
  )


def sample_input_data_revenue(
    n_geos: int = 10,
    n_times: int = 50,
    n_media_times: int = 53,
    n_controls: int = 2,
    n_non_media_channels: int | None = None,
    n_media_channels: int | None = None,
    n_rf_channels: int | None = None,
    n_organic_media_channels: int | None = None,
    n_organic_rf_channels: int | None = None,
    seed: int = 0,
):
  """Generates sample InputData for `kpi_type='revenue'`."""
  dataset = random_dataset(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_controls=n_controls,
      n_non_media_channels=n_non_media_channels,
      n_media_channels=n_media_channels,
      n_rf_channels=n_rf_channels,
      n_organic_media_channels=n_organic_media_channels,
      n_organic_rf_channels=n_organic_rf_channels,
      revenue_per_kpi_value=1.0,
      seed=seed,
  )
  return input_data.InputData(
      kpi=dataset.kpi,
      kpi_type=c.REVENUE,
      revenue_per_kpi=None,
      population=dataset.population,
      controls=dataset.controls,
      non_media_treatments=dataset.non_media_treatments
      if n_non_media_channels
      else None,
      media=dataset.media if n_media_channels else None,
      media_spend=dataset.media_spend if n_media_channels else None,
      reach=dataset.reach if n_rf_channels else None,
      frequency=dataset.frequency if n_rf_channels else None,
      rf_spend=dataset.rf_spend if n_rf_channels else None,
      organic_media=dataset.organic_media if n_organic_media_channels else None,
      organic_reach=dataset.organic_reach if n_organic_rf_channels else None,
      organic_frequency=dataset.organic_frequency
      if n_organic_rf_channels
      else None,
  )


def sample_input_data_non_revenue_revenue_per_kpi(
    n_geos: int = 10,
    n_times: int = 50,
    n_media_times: int = 53,
    n_controls: int = 2,
    n_non_media_channels: int | None = None,
    n_media_channels: int | None = None,
    n_rf_channels: int | None = None,
    n_organic_media_channels: int | None = None,
    n_organic_rf_channels: int | None = None,
    seed: int = 0,
):
  """Generates sample InputData for `non_revenue` KPI w/ revenue_per_kpi."""
  dataset = random_dataset(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_controls=n_controls,
      n_non_media_channels=n_non_media_channels,
      n_media_channels=n_media_channels,
      n_rf_channels=n_rf_channels,
      n_organic_media_channels=n_organic_media_channels,
      n_organic_rf_channels=n_organic_rf_channels,
      seed=seed,
  )
  return input_data.InputData(
      kpi=dataset.kpi,
      kpi_type=c.NON_REVENUE,
      revenue_per_kpi=dataset.revenue_per_kpi,
      population=dataset.population,
      controls=dataset.controls,
      non_media_treatments=dataset.non_media_treatments
      if n_non_media_channels
      else None,
      media=dataset.media if n_media_channels else None,
      media_spend=dataset.media_spend if n_media_channels else None,
      reach=dataset.reach if n_rf_channels else None,
      frequency=dataset.frequency if n_rf_channels else None,
      rf_spend=dataset.rf_spend if n_rf_channels else None,
      organic_media=dataset.organic_media if n_organic_media_channels else None,
      organic_reach=dataset.organic_reach if n_organic_rf_channels else None,
      organic_frequency=dataset.organic_frequency
      if n_organic_rf_channels
      else None,
  )


def sample_input_data_non_revenue_no_revenue_per_kpi(
    n_geos: int = 10,
    n_times: int = 50,
    n_media_times: int = 53,
    n_controls: int = 2,
    n_non_media_channels: int | None = None,
    n_media_channels: int | None = None,
    n_rf_channels: int | None = None,
    n_organic_media_channels: int | None = None,
    n_organic_rf_channels: int | None = None,
    seed: int = 0,
):
  """Generates sample InputData for `non_revenue` KPI w/o revenue_per_kpi."""
  dataset = random_dataset(
      n_geos=n_geos,
      n_times=n_times,
      n_media_times=n_media_times,
      n_controls=n_controls,
      n_non_media_channels=n_non_media_channels,
      n_media_channels=n_media_channels,
      n_rf_channels=n_rf_channels,
      n_organic_media_channels=n_organic_media_channels,
      n_organic_rf_channels=n_organic_rf_channels,
      seed=seed,
  )
  return input_data.InputData(
      kpi=dataset.kpi,
      kpi_type=c.NON_REVENUE,
      revenue_per_kpi=None,
      population=dataset.population,
      controls=dataset.controls,
      non_media_treatments=dataset.non_media_treatments
      if n_non_media_channels
      else None,
      media=dataset.media if n_media_channels else None,
      media_spend=dataset.media_spend if n_media_channels else None,
      reach=dataset.reach if n_rf_channels else None,
      frequency=dataset.frequency if n_rf_channels else None,
      rf_spend=dataset.rf_spend if n_rf_channels else None,
      organic_media=dataset.organic_media if n_organic_media_channels else None,
      organic_reach=dataset.organic_reach if n_organic_rf_channels else None,
      organic_frequency=dataset.organic_frequency
      if n_organic_rf_channels
      else None,
  )
