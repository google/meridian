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

import datetime
from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import input_data_builder
import xarray as xr


class InputDataBuilderTest(parameterized.TestCase):
  BASIC_POPULATION_DA = xr.DataArray(
      [1000, 2000, 3000],
      coords={constants.GEO: ['geo_1', 'geo_2', 'geo_10']},
      dims=[constants.GEO],
      name=constants.POPULATION,
  )
  NA_POPULATION_DA = xr.DataArray(
      [None, 2000, 3000],
      coords={constants.GEO: ['geo_1', 'geo_2', 'geo_10']},
      dims=[constants.GEO],
      name=constants.POPULATION,
  )
  UNSORTED_POPULATION_DA = xr.DataArray(
      [3000, 2000, 1000],
      coords={constants.GEO: ['geo_10', 'geo_2', 'geo_1']},
      dims=[constants.GEO],
      name=constants.POPULATION,
  )
  NA_KPI_DA = xr.DataArray(
      [[1, None, 1], [None, 2, 2], [3, 3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
      },
      dims=[constants.GEO, constants.TIME],
  )
  NA_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, None, 1], [2, 2, 2], [3, None, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.REVENUE_PER_KPI,
  )
  WRONG_FORMAT_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 1], [2, 2], [3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.REVENUE_PER_KPI,
  )
  DATETIME_KPI_DA = xr.DataArray(
      [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.KPI,
  )
  DATETIME_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.REVENUE_PER_KPI,
  )
  WRONG_FORMAT_KPI_DA = xr.DataArray(
      [[1, 1], [2, 2], [3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.KPI,
  )
  UNSORTED_KPI_DA = xr.DataArray(
      [[3, 3, 3], [2, 2, 2], [1, 1, 1]],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.KPI,
  )
  BASIC_KPI_DA = xr.DataArray(
      [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.KPI,
  )
  UNSORTED_REVENUE_PER_KPI_DA = xr.DataArray(
      [[3, 3, 3], [2, 2, 2], [1, 1, 1]],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.REVENUE_PER_KPI,
  )
  BASIC_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
      },
      dims=[constants.GEO, constants.TIME],
      name=constants.REVENUE_PER_KPI,
  )
  BASIC_CONTROLS_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      name=constants.CONTROLS,
  )
  NA_CONTROLS_DA = xr.DataArray(
      [
          [[1, None], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      name=constants.CONTROLS,
  )
  UNSORTED_CONTROLS_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      name=constants.CONTROLS,
  )
  DATETIME_CONTROLS_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      name=constants.CONTROLS,
  )
  WRONG_FORMAT_CONTROLS_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      name=constants.CONTROLS,
  )
  WRONG_FORMAT_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.NON_MEDIA_CHANNEL: ['non_media_ch_1', 'non_media_ch_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      name=constants.NON_MEDIA_TREATMENTS,
  )
  BASIC_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.NON_MEDIA_CHANNEL: ['non_media_ch_1', 'non_media_ch_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      name=constants.NON_MEDIA_TREATMENTS,
  )
  NA_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [None, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.NON_MEDIA_CHANNEL: ['non_media_ch_1', 'non_media_ch_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      name=constants.NON_MEDIA_TREATMENTS,
  )
  UNSORTED_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.NON_MEDIA_CHANNEL: ['non_media_ch_1', 'non_media_ch_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      name=constants.NON_MEDIA_TREATMENTS,
  )
  DATETIME_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.NON_MEDIA_CHANNEL: ['non_media_ch_1', 'non_media_ch_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      name=constants.NON_MEDIA_TREATMENTS,
  )
  BASIC_ORGANIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_1',
              'organic_media_2',
          ],
      },
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      name=constants.ORGANIC_MEDIA,
  )
  NA_ORGANIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[None, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_1',
              'organic_media_2',
          ],
      },
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      name=constants.ORGANIC_MEDIA,
  )
  UNSORTED_ORGANIC_MEDIA_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_1',
              'organic_media_2',
          ],
      },
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      name=constants.ORGANIC_MEDIA,
  )
  DATETIME_ORGANIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_1',
              'organic_media_2',
          ],
      },
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      name=constants.ORGANIC_MEDIA,
  )
  WRONG_FORMAT_ORGANIC_MEDIA_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
      },
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      name=constants.ORGANIC_MEDIA,
  )
  BASIC_ORGANIC_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  NA_ORGANIC_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [None, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  BASIC_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  NA_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[None, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  UNSORTED_ORGANIC_REACH_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  UNSORTED_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  DATETIME_ORGANIC_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  DATETIME_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  WRONG_FORMAT_ORGANIC_REACH_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  WRONG_FORMAT_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: ['organic_rf_ch_1', 'organic_rf_ch_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  WRONG_ORGANIC_RF_CHANNELS_ORGANIC_REACH_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: ['organic_reach_1', 'organic_reach_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_REACH,
  )
  WRONG_ORGANIC_RF_CHANNELS_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
          constants.ORGANIC_RF_CHANNEL: [
              'organic_frequency_1',
              'organic_frequency_2',
          ],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      name=constants.ORGANIC_FREQUENCY,
  )
  BASIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  NA_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, None], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  BASIC_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  NA_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, None], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  UNSORTED_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  UNSORTED_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  DATETIME_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  DATETIME_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  WRONG_FORMAT_MEDIA_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  WRONG_FORMAT_MEDIA_SPEND_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_channel_1', 'media_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  WRONG_MEDIA_CHANNELS_MEDIA_DA = xr.DataArray(
      [
          [[1, 1], [1, 1]],
          [[2, 2], [2, 2]],
          [[3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_1', 'media_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA,
  )
  WRONG_MEDIA_CHANNELS_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1]],
          [[2, 2], [2, 2]],
          [[3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02'],
          constants.MEDIA_CHANNEL: ['media_spend_1', 'media_spend_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      name=constants.MEDIA_SPEND,
  )
  BASIC_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  NA_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, None], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  BASIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  NA_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, None], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  BASIC_RF_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )
  NA_RF_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, None], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02', '2024-01-03'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )
  UNSORTED_REACH_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  UNSORTED_FREQUENCY_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.MEDIA_TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  UNSORTED_RF_SPEND_DA = xr.DataArray(
      [
          [[3, 3], [3, 3], [3, 3]],
          [[2, 2], [2, 2], [2, 2]],
          [[1, 1], [1, 1], [1, 1]],
      ],
      coords={
          constants.GEO: ['geo_10', 'geo_2', 'geo_1'],
          constants.TIME: ['2024-01-03', '2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )
  DATETIME_REACH_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  DATETIME_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  DATETIME_RF_SPEND_DA = xr.DataArray(
      [
          [[1, 1], [1, 1], [1, 1]],
          [[2, 2], [2, 2], [2, 2]],
          [[3, 3], [3, 3], [3, 3]],
      ],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: [
              datetime.datetime(2024, 1, 1),
              datetime.datetime(2024, 1, 2),
              datetime.datetime(2024, 1, 3),
          ],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )
  WRONG_FORMAT_REACH_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  WRONG_FORMAT_FREQUENCY_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  WRONG_FORMAT_RF_SPEND_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01T00:00:00', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_channel_1', 'rf_channel_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )
  WRONG_RF_CHANNELS_REACH_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['reach_1', 'reach_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.REACH,
  )
  WRONG_RF_CHANNELS_FREQUENCY_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['frequency_1', 'frequency_2'],
      },
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      name=constants.FREQUENCY,
  )
  WRONG_RF_CHANNELS_RF_SPEND_DA = xr.DataArray(
      [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
      coords={
          constants.GEO: ['geo_1', 'geo_2', 'geo_10'],
          constants.TIME: ['2024-01-01', '2024-01-02'],
          constants.RF_CHANNEL: ['rf_spend_1', 'rf_spend_2'],
      },
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      name=constants.RF_SPEND,
  )

  @parameterized.named_parameters(
      dict(
          testcase_name='population',
          da=BASIC_POPULATION_DA,
          setter=lambda builder, da: setattr(builder, 'population', da),
          getter=lambda builder: builder.population,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=None,
      ),
      dict(
          testcase_name='kpi',
          da=BASIC_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'kpi', da),
          getter=lambda builder: builder.kpi,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='controls',
          da=BASIC_CONTROLS_DA,
          setter=lambda builder, da: setattr(builder, 'controls', da),
          getter=lambda builder: builder.controls,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='revenue_per_kpi',
          da=BASIC_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'revenue_per_kpi', da),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='non_media_treatments',
          da=BASIC_NON_MEDIA_TREATMENTS_DA,
          setter=lambda builder, da: setattr(
              builder, 'non_media_treatments', da
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='organic_media',
          da=BASIC_ORGANIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'organic_media', da),
          getter=lambda builder: builder.organic_media,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_reach',
          da=BASIC_ORGANIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'organic_reach', da),
          getter=lambda builder: builder.organic_reach,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_frequency',
          da=BASIC_ORGANIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'organic_frequency', da),
          getter=lambda builder: builder.organic_frequency,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media',
          da=BASIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'media', da),
          getter=lambda builder: builder.media,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media_spend',
          da=BASIC_MEDIA_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'media_spend', da),
          getter=lambda builder: builder.media_spend,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='reach',
          da=BASIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'reach', da),
          getter=lambda builder: builder.reach,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='frequency',
          da=BASIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'frequency', da),
          getter=lambda builder: builder.frequency,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='rf_spend',
          da=BASIC_RF_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'rf_spend', da),
          getter=lambda builder: builder.rf_spend,
          expected_geos=['geo_1', 'geo_2', 'geo_10'],
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
  )
  def test_basic_setter_getter(
      self,
      da: xr.DataArray,
      setter: Callable[
          [input_data_builder.InputDataBuilder, xr.DataArray], None
      ],
      getter: Callable[[input_data_builder.InputDataBuilder], xr.DataArray],
      expected_geos: list[str],
      expected_time: list[str] | None,
      is_media_time: bool = False,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    original_da = da.copy()
    setter(builder, da)
    xr.testing.assert_equal(getter(builder), original_da)
    self.assertEqual(builder.geos, expected_geos)
    if is_media_time:
      self.assertEqual(builder.media_time_coords, expected_time)
    else:
      self.assertEqual(builder.time_coords, expected_time)

  @parameterized.named_parameters(
      dict(
          testcase_name='population',
          component='Population',
          da=BASIC_POPULATION_DA,
          setter=lambda builder, da: setattr(builder, 'population', da),
      ),
      dict(
          testcase_name='kpi',
          component='KPI',
          da=BASIC_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'kpi', da),
      ),
      dict(
          testcase_name='controls',
          component='Controls',
          da=BASIC_CONTROLS_DA,
          setter=lambda builder, da: setattr(builder, 'controls', da),
      ),
      dict(
          testcase_name='revenue_per_kpi',
          component='Revenue per KPI',
          da=BASIC_REVENUE_PER_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'revenue_per_kpi', da),
      ),
      dict(
          testcase_name='non_media_treatments',
          component='Non-media treatments',
          da=BASIC_NON_MEDIA_TREATMENTS_DA,
          setter=lambda builder, da: setattr(
              builder, 'non_media_treatments', da
          ),
      ),
      dict(
          testcase_name='organic_media',
          component='Organic media',
          da=BASIC_ORGANIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'organic_media', da),
      ),
      dict(
          testcase_name='organic_reach',
          component='Organic reach',
          da=BASIC_ORGANIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'organic_reach', da),
      ),
      dict(
          testcase_name='organic_frequency',
          component='Organic frequency',
          da=BASIC_ORGANIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'organic_frequency', da),
      ),
      dict(
          testcase_name='media',
          component='Media',
          da=BASIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'media', da),
      ),
      dict(
          testcase_name='media_spend',
          component='Media spend',
          da=BASIC_MEDIA_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'media_spend', da),
      ),
      dict(
          testcase_name='reach',
          component='Reach',
          da=BASIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'reach', da),
      ),
      dict(
          testcase_name='frequency',
          component='Frequency',
          da=BASIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'frequency', da),
      ),
      dict(
          testcase_name='rf_spend',
          component='RF spend',
          da=BASIC_RF_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'rf_spend', da),
      ),
  )
  def test_setter_with_existing_component(
      self,
      component: str,
      da: xr.DataArray,
      setter: Callable[
          [input_data_builder.InputDataBuilder, xr.DataArray], None
      ],
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    setter(builder, da)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'{component} was already set to {da}.',
    ):
      setter(builder, xr.DataArray([10, 200, 3000], dims=[constants.GEO]))

  @parameterized.named_parameters(
      dict(
          testcase_name='population',
          da=UNSORTED_POPULATION_DA,
          setter=lambda builder, da: setattr(builder, 'population', da),
          getter=lambda builder: builder.population,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=None,
      ),
      dict(
          testcase_name='kpi',
          da=UNSORTED_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'kpi', da),
          getter=lambda builder: builder.kpi,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
      dict(
          testcase_name='controls',
          da=UNSORTED_CONTROLS_DA,
          setter=lambda builder, da: setattr(builder, 'controls', da),
          getter=lambda builder: builder.controls,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
      dict(
          testcase_name='revenue_per_kpi',
          da=UNSORTED_REVENUE_PER_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'revenue_per_kpi', da),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
      dict(
          testcase_name='non_media_treatments',
          da=UNSORTED_NON_MEDIA_TREATMENTS_DA,
          setter=lambda builder, da: setattr(
              builder, 'non_media_treatments', da
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
      dict(
          testcase_name='organic_media',
          da=UNSORTED_ORGANIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'organic_media', da),
          getter=lambda builder: builder.organic_media,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_reach',
          da=UNSORTED_ORGANIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'organic_reach', da),
          getter=lambda builder: builder.organic_reach,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_frequency',
          da=UNSORTED_ORGANIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'organic_frequency', da),
          getter=lambda builder: builder.organic_frequency,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media',
          da=UNSORTED_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'media', da),
          getter=lambda builder: builder.media,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media_spend',
          da=UNSORTED_MEDIA_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'media_spend', da),
          getter=lambda builder: builder.media_spend,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
      dict(
          testcase_name='reach',
          da=UNSORTED_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'reach', da),
          getter=lambda builder: builder.reach,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='frequency',
          da=UNSORTED_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'frequency', da),
          getter=lambda builder: builder.frequency,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
          is_media_time=True,
      ),
      dict(
          testcase_name='rf_spend',
          da=UNSORTED_RF_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'rf_spend', da),
          getter=lambda builder: builder.rf_spend,
          expected_geos=['geo_10', 'geo_2', 'geo_1'],
          expected_time=['2024-01-03', '2024-01-01', '2024-01-02'],
      ),
  )
  def test_component_setter_also_sets_geos_and_time_but_does_not_sort(
      self,
      da: xr.DataArray,
      setter: Callable[
          [input_data_builder.InputDataBuilder, xr.DataArray], None
      ],
      getter: Callable[[input_data_builder.InputDataBuilder], xr.DataArray],
      expected_geos: list[str],
      expected_time: list[str] | None,
      is_media_time: bool = False,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    original_da = da.copy()
    setter(builder, da)
    xr.testing.assert_equal(getter(builder), original_da)
    self.assertEqual(builder.geos, expected_geos)
    if is_media_time:
      self.assertEqual(builder.media_time_coords, expected_time)
    else:
      self.assertEqual(builder.time_coords, expected_time)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          da=DATETIME_KPI_DA,
          expected_da=BASIC_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'kpi', da),
          getter=lambda builder: builder.kpi,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='controls',
          da=DATETIME_CONTROLS_DA,
          expected_da=BASIC_CONTROLS_DA,
          setter=lambda builder, da: setattr(builder, 'controls', da),
          getter=lambda builder: builder.controls,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='revenue_per_kpi',
          da=DATETIME_REVENUE_PER_KPI_DA,
          expected_da=BASIC_REVENUE_PER_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'revenue_per_kpi', da),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='non_media_treatments',
          da=DATETIME_NON_MEDIA_TREATMENTS_DA,
          expected_da=BASIC_NON_MEDIA_TREATMENTS_DA,
          setter=lambda builder, da: setattr(
              builder, 'non_media_treatments', da
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='organic_media',
          da=DATETIME_ORGANIC_MEDIA_DA,
          expected_da=BASIC_ORGANIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'organic_media', da),
          getter=lambda builder: builder.organic_media,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_frequency',
          da=DATETIME_ORGANIC_FREQUENCY_DA,
          expected_da=BASIC_ORGANIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'organic_frequency', da),
          getter=lambda builder: builder.organic_frequency,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media',
          da=DATETIME_MEDIA_DA,
          expected_da=BASIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'media', da),
          getter=lambda builder: builder.media,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='media_spend',
          da=DATETIME_MEDIA_SPEND_DA,
          expected_da=BASIC_MEDIA_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'media_spend', da),
          getter=lambda builder: builder.media_spend,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
      dict(
          testcase_name='reach',
          da=DATETIME_REACH_DA,
          expected_da=BASIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'reach', da),
          getter=lambda builder: builder.reach,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='frequency',
          da=DATETIME_FREQUENCY_DA,
          expected_da=BASIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'frequency', da),
          getter=lambda builder: builder.frequency,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
          is_media_time=True,
      ),
      dict(
          testcase_name='rf_spend',
          da=DATETIME_RF_SPEND_DA,
          expected_da=BASIC_RF_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'rf_spend', da),
          getter=lambda builder: builder.rf_spend,
          expected_time=['2024-01-01', '2024-01-02', '2024-01-03'],
      ),
  )
  def test_setter_with_datetime_time_coordinates_normalized(
      self,
      da: xr.DataArray,
      expected_da: xr.DataArray,
      setter: Callable[
          [input_data_builder.InputDataBuilder, xr.DataArray], None
      ],
      getter: Callable[[input_data_builder.InputDataBuilder], xr.DataArray],
      expected_time: list[str],
      is_media_time: bool = False,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    setter(builder, da)
    xr.testing.assert_equal(getter(builder), expected_da)
    if is_media_time:
      self.assertEqual(builder.media_time_coords, expected_time)
    else:
      self.assertEqual(builder.time_coords, expected_time)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          da=WRONG_FORMAT_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'kpi', da),
      ),
      dict(
          testcase_name='controls',
          da=WRONG_FORMAT_CONTROLS_DA,
          setter=lambda builder, da: setattr(builder, 'controls', da),
      ),
      dict(
          testcase_name='revenue_per_kpi',
          da=WRONG_FORMAT_REVENUE_PER_KPI_DA,
          setter=lambda builder, da: setattr(builder, 'revenue_per_kpi', da),
      ),
      dict(
          testcase_name='non_media_treatments',
          da=WRONG_FORMAT_NON_MEDIA_TREATMENTS_DA,
          setter=lambda builder, da: setattr(
              builder, 'non_media_treatments', da
          ),
      ),
      dict(
          testcase_name='organic_media',
          da=WRONG_FORMAT_ORGANIC_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'organic_media', da),
      ),
      dict(
          testcase_name='organic_reach',
          da=WRONG_FORMAT_ORGANIC_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'organic_reach', da),
      ),
      dict(
          testcase_name='organic_frequency',
          da=WRONG_FORMAT_ORGANIC_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'organic_frequency', da),
      ),
      dict(
          testcase_name='media',
          da=WRONG_FORMAT_MEDIA_DA,
          setter=lambda builder, da: setattr(builder, 'media', da),
      ),
      dict(
          testcase_name='media_spend',
          da=WRONG_FORMAT_MEDIA_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'media_spend', da),
      ),
      dict(
          testcase_name='reach',
          da=WRONG_FORMAT_REACH_DA,
          setter=lambda builder, da: setattr(builder, 'reach', da),
      ),
      dict(
          testcase_name='frequency',
          da=WRONG_FORMAT_FREQUENCY_DA,
          setter=lambda builder, da: setattr(builder, 'frequency', da),
      ),
      dict(
          testcase_name='rf_spend',
          da=WRONG_FORMAT_RF_SPEND_DA,
          setter=lambda builder, da: setattr(builder, 'rf_spend', da),
      ),
  )
  def test_setter_with_incorrect_time_format_raises_error(
      self,
      da: xr.DataArray,
      setter: Callable[
          [input_data_builder.InputDataBuilder, xr.DataArray], None
      ],
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid time label: '2024-01-01T00:00:00'. Expected format:"
        " '%Y-%m-%d'",
    ):
      setter(builder, da)

  @parameterized.named_parameters(
      dict(
          testcase_name='organic_rf_channel_organic_frequency',
          channel_name=constants.ORGANIC_RF_CHANNEL,
          da1=WRONG_ORGANIC_RF_CHANNELS_ORGANIC_REACH_DA,
          setter1=lambda builder, da: setattr(builder, 'organic_reach', da),
          da2=WRONG_ORGANIC_RF_CHANNELS_ORGANIC_FREQUENCY_DA,
          setter2=lambda builder, da: setattr(builder, 'organic_frequency', da),
          components=['organic_frequency', 'organic_reach'],
      ),
      dict(
          testcase_name='organic_rf_channel_organic_reach',
          channel_name=constants.ORGANIC_RF_CHANNEL,
          da1=WRONG_ORGANIC_RF_CHANNELS_ORGANIC_FREQUENCY_DA,
          setter1=lambda builder, da: setattr(builder, 'organic_frequency', da),
          da2=WRONG_ORGANIC_RF_CHANNELS_ORGANIC_REACH_DA,
          setter2=lambda builder, da: setattr(builder, 'organic_reach', da),
          components=['organic_reach', 'organic_frequency'],
      ),
      dict(
          testcase_name='media_channel_media',
          channel_name=constants.MEDIA_CHANNEL,
          da1=WRONG_MEDIA_CHANNELS_MEDIA_SPEND_DA,
          setter1=lambda builder, da: setattr(builder, 'media_spend', da),
          da2=WRONG_MEDIA_CHANNELS_MEDIA_DA,
          setter2=lambda builder, da: setattr(builder, 'media', da),
          components=['media', 'media_spend'],
      ),
      dict(
          testcase_name='media_channel_media_spend',
          channel_name=constants.MEDIA_CHANNEL,
          da1=WRONG_MEDIA_CHANNELS_MEDIA_DA,
          setter1=lambda builder, da: setattr(builder, 'media', da),
          da2=WRONG_MEDIA_CHANNELS_MEDIA_SPEND_DA,
          setter2=lambda builder, da: setattr(builder, 'media_spend', da),
          components=['media_spend', 'media'],
      ),
      dict(
          testcase_name='rf_channel_reach',
          channel_name=constants.RF_CHANNEL,
          da1=WRONG_RF_CHANNELS_RF_SPEND_DA,
          setter1=lambda builder, da: setattr(builder, 'rf_spend', da),
          da2=WRONG_RF_CHANNELS_REACH_DA,
          setter2=lambda builder, da: setattr(builder, 'reach', da),
          components=['reach', 'rf_spend'],
      ),
      dict(
          testcase_name='rf_channel_frequency',
          channel_name=constants.RF_CHANNEL,
          da1=WRONG_RF_CHANNELS_RF_SPEND_DA,
          setter1=lambda builder, da: setattr(builder, 'rf_spend', da),
          da2=WRONG_RF_CHANNELS_FREQUENCY_DA,
          setter2=lambda builder, da: setattr(builder, 'frequency', da),
          components=['frequency', 'rf_spend'],
      ),
      dict(
          testcase_name='rf_channel_rf_spend',
          channel_name=constants.RF_CHANNEL,
          da1=WRONG_RF_CHANNELS_REACH_DA,
          setter1=lambda builder, da: setattr(builder, 'reach', da),
          da2=WRONG_RF_CHANNELS_RF_SPEND_DA,
          setter2=lambda builder, da: setattr(builder, 'rf_spend', da),
          components=['rf_spend', 'reach'],
      ),
  )
  def test_inconsistent_channels_raises_exception(
      self, channel_name, da1, setter1, da2, setter2, components
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'{channel_name} coordinates must be the same between {components}.',
    ):
      builder = input_data_builder.InputDataBuilder(
          kpi_type=constants.NON_REVENUE
      )
      setter1(builder, da1)
      setter2(builder, da2)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_kpi',
          setter=lambda builder: None,
          error_msg='KPI is required.',
      ),
      dict(
          testcase_name='no_population',
          setter=lambda builder: setattr(
              builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA
          ),
          error_msg='Population is required for non national models.',
      ),
      dict(
          testcase_name='no_media',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
          ],
          error_msg='Media and media spend must be provided together.',
      ),
      dict(
          testcase_name='no_reach',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
          ],
          error_msg='Reach, frequency, and rf_spend must be provided together.',
      ),
      dict(
          testcase_name='no_organic_reach',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(
                  builder,
                  'organic_reach',
                  InputDataBuilderTest.BASIC_ORGANIC_REACH_DA,
              ),
          ],
          error_msg=(
              'Organic reach and organic frequency must be provided together.'
          ),
      ),
      dict(
          testcase_name='no_media_no_reach',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
          ],
          error_msg=(
              'It is required to have at least one of media or reach +'
              ' frequency.'
          ),
      ),
  )
  def test_build_not_all_required_components_raises_error(
      self, setter, error_msg
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      setter(builder)
      builder.build()

  @parameterized.named_parameters(
      dict(
          testcase_name='na_kpi',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.NA_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the kpi data.',
      ),
      dict(
          testcase_name='na_population',
          setter=lambda builder: [
              setattr(
                  builder, 'population', InputDataBuilderTest.NA_POPULATION_DA
              ),
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the population data.',
      ),
      dict(
          testcase_name='na_controls',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(builder, 'controls', InputDataBuilderTest.NA_CONTROLS_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the controls data.',
      ),
      dict(
          testcase_name='na_revenue_per_kpi',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'revenue_per_kpi',
                  InputDataBuilderTest.NA_REVENUE_PER_KPI_DA,
              ),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the revenue per kpi data.',
      ),
      dict(
          testcase_name='na_media_spend',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder, 'media_spend', InputDataBuilderTest.NA_MEDIA_SPEND_DA
              ),
          ],
          error_msg='NA values found in the media spend data.',
      ),
      dict(
          testcase_name='na_rf_spend',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.BASIC_FREQUENCY_DA
              ),
              setattr(builder, 'rf_spend', InputDataBuilderTest.NA_RF_SPEND_DA),
          ],
          error_msg='NA values found in the rf spend data.',
      ),
      dict(
          testcase_name='na_non_media_treatments',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'non_media_treatments',
                  InputDataBuilderTest.NA_NON_MEDIA_TREATMENTS_DA,
              ),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.BASIC_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the non media treatments data.',
      ),
      dict(
          testcase_name='na_media',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'media', InputDataBuilderTest.NA_MEDIA_DA),
              setattr(
                  builder,
                  'media_spend',
                  InputDataBuilderTest.BASIC_MEDIA_SPEND_DA,
              ),
          ],
          error_msg='NA values found in the media data.',
      ),
      dict(
          testcase_name='na_reach',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.NA_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.BASIC_FREQUENCY_DA
              ),
              setattr(
                  builder, 'rf_spend', InputDataBuilderTest.BASIC_RF_SPEND_DA
              ),
          ],
          error_msg='NA values found in the reach data.',
      ),
      dict(
          testcase_name='na_frequency',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.NA_FREQUENCY_DA
              ),
              setattr(
                  builder, 'rf_spend', InputDataBuilderTest.BASIC_RF_SPEND_DA
              ),
          ],
          error_msg='NA values found in the frequency data.',
      ),
      dict(
          testcase_name='na_organic_media',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.BASIC_FREQUENCY_DA
              ),
              setattr(
                  builder, 'rf_spend', InputDataBuilderTest.BASIC_RF_SPEND_DA
              ),
              setattr(
                  builder,
                  'organic_media',
                  InputDataBuilderTest.NA_ORGANIC_MEDIA_DA,
              ),
          ],
          error_msg='NA values found in the organic media data.',
      ),
      dict(
          testcase_name='na_organic_reach',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.BASIC_FREQUENCY_DA
              ),
              setattr(
                  builder, 'rf_spend', InputDataBuilderTest.BASIC_RF_SPEND_DA
              ),
              setattr(
                  builder,
                  'organic_frequency',
                  InputDataBuilderTest.BASIC_ORGANIC_FREQUENCY_DA,
              ),
              setattr(
                  builder,
                  'organic_reach',
                  InputDataBuilderTest.NA_ORGANIC_REACH_DA,
              ),
          ],
          error_msg='NA values found in the organic reach data.',
      ),
      dict(
          testcase_name='na_organic_frequency',
          setter=lambda builder: [
              setattr(builder, 'kpi', InputDataBuilderTest.BASIC_KPI_DA),
              setattr(
                  builder,
                  'population',
                  InputDataBuilderTest.BASIC_POPULATION_DA,
              ),
              setattr(builder, 'reach', InputDataBuilderTest.BASIC_REACH_DA),
              setattr(
                  builder, 'frequency', InputDataBuilderTest.BASIC_FREQUENCY_DA
              ),
              setattr(
                  builder, 'rf_spend', InputDataBuilderTest.BASIC_RF_SPEND_DA
              ),
              setattr(
                  builder,
                  'organic_frequency',
                  InputDataBuilderTest.NA_ORGANIC_FREQUENCY_DA,
              ),
              setattr(
                  builder,
                  'organic_reach',
                  InputDataBuilderTest.BASIC_ORGANIC_REACH_DA,
              ),
          ],
          error_msg='NA values found in the organic frequency data.',
      ),
  )
  def test_build_with_nas_raises_error(self, setter, error_msg):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      setter(builder)
      builder.build()

  def test_build_non_national_population(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.kpi = self.BASIC_KPI_DA
    builder.population = self.BASIC_POPULATION_DA
    builder.media = self.BASIC_MEDIA_DA
    builder.media_spend = self.BASIC_MEDIA_SPEND_DA
    input_data = builder.build()
    xr.testing.assert_equal(input_data.kpi, self.BASIC_KPI_DA)
    xr.testing.assert_equal(input_data.population, self.BASIC_POPULATION_DA)
    xr.testing.assert_equal(input_data.media, self.BASIC_MEDIA_DA)
    xr.testing.assert_equal(input_data.media_spend, self.BASIC_MEDIA_SPEND_DA)

  @parameterized.named_parameters(
      dict(
          testcase_name='revenue_per_kpi',
          setter=lambda builder: setattr(
              builder,
              'revenue_per_kpi',
              InputDataBuilderTest.UNSORTED_REVENUE_PER_KPI_DA.copy(),
          ),
          getter=lambda data: data.revenue_per_kpi,
          expected_da=BASIC_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name='controls',
          setter=lambda builder: setattr(
              builder,
              'controls',
              InputDataBuilderTest.UNSORTED_CONTROLS_DA.copy(),
          ),
          getter=lambda data: data.controls,
          expected_da=BASIC_CONTROLS_DA,
      ),
      dict(
          testcase_name='reach',
          setter=lambda builder: [
              setattr(
                  builder,
                  'reach',
                  InputDataBuilderTest.UNSORTED_REACH_DA.copy(),
              ),
              setattr(
                  builder,
                  'frequency',
                  InputDataBuilderTest.UNSORTED_FREQUENCY_DA.copy(),
              ),
              setattr(
                  builder,
                  'rf_spend',
                  InputDataBuilderTest.UNSORTED_RF_SPEND_DA.copy(),
              ),
          ],
          getter=lambda data: [data.reach, data.frequency, data.rf_spend],
          expected_da=[BASIC_REACH_DA, BASIC_FREQUENCY_DA, BASIC_RF_SPEND_DA],
      ),
      dict(
          testcase_name='organic_media',
          setter=lambda builder: setattr(
              builder,
              'organic_media',
              InputDataBuilderTest.UNSORTED_ORGANIC_MEDIA_DA.copy(),
          ),
          getter=lambda data: data.organic_media,
          expected_da=BASIC_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name='organic_rf',
          setter=lambda builder: [
              setattr(
                  builder,
                  'organic_reach',
                  InputDataBuilderTest.UNSORTED_ORGANIC_REACH_DA.copy(),
              ),
              setattr(
                  builder,
                  'organic_frequency',
                  InputDataBuilderTest.UNSORTED_ORGANIC_FREQUENCY_DA.copy(),
              ),
          ],
          getter=lambda data: [data.organic_reach, data.organic_frequency],
          expected_da=[BASIC_ORGANIC_REACH_DA, BASIC_ORGANIC_FREQUENCY_DA],
      ),
      dict(
          testcase_name='non_media_treatments',
          setter=lambda builder: setattr(
              builder,
              'non_media_treatments',
              InputDataBuilderTest.UNSORTED_NON_MEDIA_TREATMENTS_DA.copy(),
          ),
          getter=lambda data: data.non_media_treatments,
          expected_da=BASIC_NON_MEDIA_TREATMENTS_DA,
      ),
  )
  def test_build_unsorted_sorts_input_data(self, setter, getter, expected_da):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )

    builder.kpi = self.UNSORTED_KPI_DA.copy()
    builder.population = self.UNSORTED_POPULATION_DA.copy()
    builder.media = self.UNSORTED_MEDIA_DA.copy()
    builder.media_spend = self.UNSORTED_MEDIA_SPEND_DA.copy()
    setter(builder)
    input_data = builder.build()
    xr.testing.assert_equal(input_data.kpi, self.BASIC_KPI_DA)
    xr.testing.assert_equal(input_data.population, self.BASIC_POPULATION_DA)

    if isinstance(expected_da, list):
      actual_das = getter(input_data)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(input_data), expected_da)

  def test_build_national_population_not_set(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.kpi = xr.DataArray(
        [[1, 1]],
        coords={
            constants.GEO: ['A'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
        },
        dims=[constants.GEO, constants.TIME],
        name=constants.KPI,
    )
    builder.media = xr.DataArray(
        [[[1, 1], [1, 1]]],
        coords={
            constants.GEO: ['A'],
            constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
            constants.MEDIA_CHANNEL: [
                'media_channel_1',
                'media_channel_2',
            ],
        },
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.MEDIA_CHANNEL,
        ],
        name=constants.MEDIA,
    )
    builder.media_spend = xr.DataArray(
        [
            [[1, 1], [1, 1]],
        ],
        coords={
            constants.GEO: ['A'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
            constants.MEDIA_CHANNEL: [
                'media_channel_1',
                'media_channel_2',
            ],
        },
        dims=[
            constants.GEO,
            constants.TIME,
            constants.MEDIA_CHANNEL,
        ],
        name=constants.MEDIA_SPEND,
    )
    input_data = builder.build()
    xr.testing.assert_equal(
        input_data.population,
        xr.DataArray(
            [1],
            coords={
                constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            },
            dims=[constants.GEO],
        ),
    )

  def test_build_national_population_set(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.kpi = xr.DataArray(
        [[1, 1]],
        coords={
            constants.GEO: ['A'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
        },
        dims=[constants.GEO, constants.TIME],
        name=constants.KPI,
    )
    builder.media = xr.DataArray(
        [[[1, 1], [1, 1]]],
        coords={
            constants.GEO: ['A'],
            constants.MEDIA_TIME: ['2024-01-01', '2024-01-02'],
            constants.MEDIA_CHANNEL: [
                'media_channel_1',
                'media_channel_2',
            ],
        },
        dims=[
            constants.GEO,
            constants.MEDIA_TIME,
            constants.MEDIA_CHANNEL,
        ],
        name=constants.MEDIA,
    )
    builder.media_spend = xr.DataArray(
        [
            [[1, 1], [1, 1]],
        ],
        coords={
            constants.GEO: ['A'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
            constants.MEDIA_CHANNEL: [
                'media_channel_1',
                'media_channel_2',
            ],
        },
        dims=[
            constants.GEO,
            constants.TIME,
            constants.MEDIA_CHANNEL,
        ],
        name=constants.MEDIA_SPEND,
    )
    builder.population = xr.DataArray(
        [1000],
        coords={
            constants.GEO: ['A'],
        },
        dims=[constants.GEO],
        name=constants.POPULATION,
    )
    input_data = builder.build()
    xr.testing.assert_equal(
        input_data.population,
        xr.DataArray(
            [1],
            coords={
                constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            },
            dims=[constants.GEO],
            name=constants.POPULATION,
        ),
    )

  def test_set_dataarrays_with_same_geos(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    kpi_da = xr.DataArray(
        [[1, 1], [2, 2], [3, 3]],
        coords={
            constants.GEO: ['A', 'B', 'C'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
        },
    )
    population_da = xr.DataArray(
        [1000, 2000, 3000],
        coords={constants.GEO: ['A', 'B', 'C']},
        dims=[constants.GEO],
    )
    controls_da = xr.DataArray(
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
        ],
        coords={
            constants.GEO: ['B', 'A', 'C'],
            constants.TIME: ['2024-01-02', '2024-01-01'],
            constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
        },
        dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
    )
    builder.kpi = kpi_da
    builder.population = population_da
    self.assertEqual(builder.geos, ['A', 'B', 'C'])
    self.assertEqual(builder.time_coords, ['2024-01-01', '2024-01-02'])
    builder.controls = controls_da
    self.assertEqual(builder.geos, ['B', 'A', 'C'])
    self.assertEqual(builder.time_coords, ['2024-01-02', '2024-01-01'])

  def test_set_dataarrays_with_different_coords(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    kpi_da = xr.DataArray(
        [[1, 1], [2, 2], [3, 3]],
        coords={
            constants.GEO: ['A', 'B', 'C'],
            constants.TIME: ['2024-01-01', '2024-01-02'],
        },
    )
    population_da = xr.DataArray(
        [1000, 2000],
        coords={constants.GEO: ['A', 'B']},
        dims=[constants.GEO],
    )
    controls_da = xr.DataArray(
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
        ],
        coords={
            constants.GEO: ['B', 'A', 'C'],
            constants.TIME: ['2024-01-03', '2024-01-01'],
            constants.CONTROL_VARIABLE: ['control_1', 'control_2'],
        },
        dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
    )
    builder.kpi = kpi_da
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "geos already set to ['A', 'B', 'C'].",
    ):
      builder.population = population_da
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`times` coords already set to ['2024-01-01', '2024-01-02'].",
    ):
      builder.controls = controls_da

  def test_geo_setter_non_unique_raises_error(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Geos must be unique.',
    ):
      builder.geos = ['A', 'B', 'C', 'A']

  def test_geo_setter_with_existing_geos_different_raises_error(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.geos = ['A', 'B', 'C']
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "geos already set to ['A', 'B', 'C'].",
    ):
      builder.geos = ['A', 'B', 'D']

  def test_geo_setter_with_existing_geos_same_set_uses_latest_value(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.geos = ['A', 'B', 'C']
    self.assertEqual(builder.geos, ['A', 'B', 'C'])
    builder.geos = ['B', 'A', 'C']
    self.assertEqual(builder.geos, ['B', 'A', 'C'])

  def test_time_coords_setter_non_unique_raises_error(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '`times` coords must be unique.',
    ):
      builder.time_coords = ['2024-01-01', '2024-01-01', '2024-01-02']

  def test_time_coords_setter_with_existing_time_coords_different_raises_error(
      self,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.time_coords = ['2024-01-01', '2024-01-02']
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`times` coords already set to ['2024-01-01', '2024-01-02'].",
    ):
      builder.time_coords = ['2024-01-01', '2024-01-03']

  def test_time_coords_setter_with_existing_time_coords_same_set_uses_latest_value(
      self,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.time_coords = ['2024-01-01', '2024-01-02']
    self.assertEqual(builder.time_coords, ['2024-01-01', '2024-01-02'])
    builder.time_coords = ['2024-01-02', '2024-01-01']
    self.assertEqual(builder.time_coords, ['2024-01-02', '2024-01-01'])

  def test_time_coords_setter_subset_of_media_time_coords(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.media_time_coords = [
        '2024-01-01',
        '2024-01-02',
        '2024-01-03',
        '2024-01-04',
    ]
    builder.time_coords = ['2024-01-02', '2024-01-03', '2024-01-04']
    self.assertEqual(
        builder.time_coords, ['2024-01-02', '2024-01-03', '2024-01-04']
    )
    self.assertEqual(
        builder.media_time_coords,
        ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    )

  def test_time_coords_setter_not_subset_of_media_time_coords(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.media_time_coords = [
        '2024-01-01',
        '2024-01-02',
        '2024-01-03',
        '2024-01-04',
    ]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '`times` coords must be subset of previously set `media_times` coords.',
    ):
      builder.time_coords = ['2024-01-01', '2024-01-02', '2024-01-05']

  def test_media_time_coords_setter_non_unique_raises_error(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '`media_times` coords must be unique.',
    ):
      builder.media_time_coords = ['2024-01-01', '2024-01-01', '2024-01-02']

  def test_media_times_setter_with_existing_media_times_different_raises_error(
      self,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.media_time_coords = ['2024-01-01', '2024-01-02']
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`media_times` coords already set to ['2024-01-01', '2024-01-02'].",
    ):
      builder.media_time_coords = ['2024-01-01', '2024-01-03']

  def test_media_times_setter_with_existing_media_times_same_set_uses_latest_value(
      self,
  ):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.media_time_coords = ['2024-01-01', '2024-01-02']
    self.assertEqual(builder.media_time_coords, ['2024-01-01', '2024-01-02'])
    builder.media_time_coords = ['2024-01-02', '2024-01-01']
    self.assertEqual(builder.media_time_coords, ['2024-01-02', '2024-01-01'])

  def test_media_times_setter_superset_of_time_coords(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.time_coords = ['2024-01-02', '2024-01-03', '2024-01-04']
    builder.media_time_coords = [
        '2024-01-01',
        '2024-01-02',
        '2024-01-03',
        '2024-01-04',
    ]
    self.assertEqual(
        builder.time_coords, ['2024-01-02', '2024-01-03', '2024-01-04']
    )
    self.assertEqual(
        builder.media_time_coords,
        ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    )

  def test_media_times_setter_not_superset_of_time_coords(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    builder.time_coords = ['2024-01-01', '2024-01-02']
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '`media_times` coords must be superset of previously set `times`'
        ' coords.',
    ):
      builder.media_time_coords = ['2024-01-01', '2024-01-03', '2024-01-04']

  def test_invalid_time_coordinates_object_flows_error(self):
    builder = input_data_builder.InputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'There must be more than one date index in the time coordinates.',
    ):
      builder.media_time_coords = ['2024-01-01']

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Time coordinates are not regularly spaced!',
    ):
      builder.time_coords = ['2024-01-01', '2024-01-02', '2025-03-07']


if __name__ == '__main__':
  absltest.main()
