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

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import nd_array_input_data_builder
import numpy as np
import xarray as xr


class NdArrayInputDataBuilderTest(parameterized.TestCase):
  TIME_COORDS = ['2024-01-02', '2024-01-03', '2024-01-01']
  GEOS = ['B', 'A', 'C']
  BASIC_KPI_ND = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  NATIONAL_KPI_ND = np.array([[1, 2, 3]])
  BASIC_KPI_DA = xr.DataArray(
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
      },
      name=constants.KPI,
  )
  NATIONAL_KPI_DA = xr.DataArray(
      [[1, 2, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
      },
      name=constants.KPI,
  )
  BASIC_CONTROLS_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_CONTROLS_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_CONTROLS_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
          constants.CONTROL_VARIABLE: ['control_2', 'control_1'],
      },
      name=constants.CONTROLS,
  )
  NATIONAL_CONTROLS_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
          constants.CONTROL_VARIABLE: ['control_2', 'control_1'],
      },
      name=constants.CONTROLS,
  )
  BASIC_POPULATION_ND = np.array([1, 2, 3])
  NATIONAL_POPULATION_ND = np.array([1])
  BASIC_POPULATION_DA = xr.DataArray(
      [1, 2, 3],
      dims=[constants.GEO],
      coords={constants.GEO: GEOS},
      name=constants.POPULATION,
  )
  NATIONAL_POPULATION_DA = xr.DataArray(
      [1],
      dims=[constants.GEO],
      coords={constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]},
      name=constants.POPULATION,
  )
  BASIC_REVENUE_PER_KPI_ND = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  NATIONAL_REVENUE_PER_KPI_ND = np.array([[1, 2, 3]])
  BASIC_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
      },
      name=constants.REVENUE_PER_KPI,
  )
  NATIONAL_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 2, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
      },
      name=constants.REVENUE_PER_KPI,
  )
  BASIC_NON_MEDIA_TREATMENTS_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_NON_MEDIA_TREATMENTS_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
          constants.NON_MEDIA_CHANNEL: [
              'non_media_channel_2',
              'non_media_channel_1',
          ],
      },
      name=constants.NON_MEDIA_TREATMENTS,
  )
  NATIONAL_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
          constants.NON_MEDIA_CHANNEL: [
              'non_media_channel_2',
              'non_media_channel_1',
          ],
      },
      name=constants.NON_MEDIA_TREATMENTS,
  )
  BASIC_ORGANIC_MEDIA_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_ORGANIC_MEDIA_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_ORGANIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_channel_2',
              'organic_media_channel_1',
          ],
      },
      name=constants.ORGANIC_MEDIA,
  )
  NATIONAL_ORGANIC_MEDIA_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_MEDIA_CHANNEL: [
              'organic_media_channel_2',
              'organic_media_channel_1',
          ],
      },
      name=constants.ORGANIC_MEDIA,
  )
  BASIC_ORGANIC_REACH_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_ORGANIC_REACH_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_ORGANIC_REACH_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_RF_CHANNEL: [
              'organic_rf_channel_2',
              'organic_rf_channel_1',
          ],
      },
      name=constants.ORGANIC_REACH,
  )
  NATIONAL_ORGANIC_REACH_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_RF_CHANNEL: [
              'organic_rf_channel_2',
              'organic_rf_channel_1',
          ],
      },
      name=constants.ORGANIC_REACH,
  )
  BASIC_ORGANIC_FREQUENCY_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_ORGANIC_FREQUENCY_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_RF_CHANNEL: [
              'organic_rf_channel_2',
              'organic_rf_channel_1',
          ],
      },
      name=constants.ORGANIC_FREQUENCY,
  )
  NATIONAL_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.ORGANIC_RF_CHANNEL: [
              'organic_rf_channel_2',
              'organic_rf_channel_1',
          ],
      },
      name=constants.ORGANIC_FREQUENCY,
  )
  BASIC_MEDIA_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_MEDIA_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_MEDIA_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.MEDIA_CHANNEL: [
              'media_channel_2',
              'media_channel_1',
          ],
      },
      name=constants.MEDIA,
  )
  NATIONAL_MEDIA_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.MEDIA_CHANNEL: [
              'media_channel_2',
              'media_channel_1',
          ],
      },
      name=constants.MEDIA,
  )
  BASIC_MEDIA_SPEND_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_MEDIA_SPEND_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_MEDIA_SPEND_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
          constants.MEDIA_CHANNEL: [
              'media_channel_2',
              'media_channel_1',
          ],
      },
      name=constants.MEDIA_SPEND,
  )
  NATIONAL_MEDIA_SPEND_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
          constants.MEDIA_CHANNEL: [
              'media_channel_2',
              'media_channel_1',
          ],
      },
      name=constants.MEDIA_SPEND,
  )
  BASIC_REACH_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_REACH_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_REACH_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.REACH,
  )
  NATIONAL_REACH_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.REACH,
  )
  BASIC_FREQUENCY_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_FREQUENCY_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_FREQUENCY_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.MEDIA_TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.FREQUENCY,
  )
  NATIONAL_FREQUENCY_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.FREQUENCY,
  )
  BASIC_RF_SPEND_ND = np.array([
      [[1, 5], [2, 6], [3, 4]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
  ])
  NATIONAL_RF_SPEND_ND = np.array([
      [[1, 2], [3, 4], [5, 6]],
  ])
  BASIC_RF_SPEND_DA = xr.DataArray(
      [
          [[1, 5], [2, 6], [3, 4]],
          [[7, 8], [9, 10], [11, 12]],
          [[13, 14], [15, 16], [17, 18]],
      ],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: GEOS,
          constants.TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.RF_SPEND,
  )
  NATIONAL_RF_SPEND_DA = xr.DataArray(
      [[[1, 2], [3, 4], [5, 6]]],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: TIME_COORDS,
          constants.RF_CHANNEL: [
              'rf_channel_2',
              'rf_channel_1',
          ],
      },
      name=constants.RF_SPEND,
  )

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          nd=BASIC_KPI_ND,
          times=TIME_COORDS,
          media_times=None,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_kpi(nd),
          getter=lambda builder: builder.kpi,
          expected_da=BASIC_KPI_DA,
      ),
      dict(
          testcase_name='controls',
          nd=BASIC_CONTROLS_ND,
          times=TIME_COORDS,
          media_times=None,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_controls(
              nd, ['control_2', 'control_1']
          ),
          getter=lambda builder: builder.controls,
          expected_da=BASIC_CONTROLS_DA,
      ),
      dict(
          testcase_name='population',
          nd=BASIC_POPULATION_ND,
          times=None,
          media_times=None,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_population(nd),
          getter=lambda builder: builder.population,
          expected_da=BASIC_POPULATION_DA,
      ),
      dict(
          testcase_name='revenue_per_kpi',
          nd=BASIC_REVENUE_PER_KPI_ND,
          times=TIME_COORDS,
          media_times=None,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_revenue_per_kpi(nd),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_da=BASIC_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=BASIC_NON_MEDIA_TREATMENTS_ND,
          times=TIME_COORDS,
          media_times=None,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel_2', 'non_media_channel_1']
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_da=BASIC_NON_MEDIA_TREATMENTS_DA,
      ),
      dict(
          testcase_name='organic_media',
          nd=BASIC_ORGANIC_MEDIA_ND,
          times=None,
          media_times=TIME_COORDS,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel_2', 'organic_media_channel_1']
          ),
          getter=lambda builder: builder.organic_media,
          expected_da=BASIC_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name='organic_rf',
          nd=[BASIC_ORGANIC_REACH_ND, BASIC_ORGANIC_FREQUENCY_ND],
          times=None,
          media_times=TIME_COORDS,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_2',
                  'organic_rf_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.organic_reach,
              builder.organic_frequency,
          ],
          expected_da=[BASIC_ORGANIC_REACH_DA, BASIC_ORGANIC_FREQUENCY_DA],
      ),
      dict(
          testcase_name='media',
          nd=[BASIC_MEDIA_ND, BASIC_MEDIA_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_2',
                  'media_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.media,
              builder.media_spend,
          ],
          expected_da=[BASIC_MEDIA_DA, BASIC_MEDIA_SPEND_DA],
      ),
      dict(
          testcase_name='reach',
          nd=[BASIC_REACH_ND, BASIC_FREQUENCY_ND, BASIC_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          geos=GEOS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_2',
                  'rf_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.reach,
              builder.frequency,
              builder.rf_spend,
          ],
          expected_da=[BASIC_REACH_DA, BASIC_FREQUENCY_DA, BASIC_RF_SPEND_DA],
      ),
  )
  def test_basic_with_component(
      self, nd, times, media_times, geos, setter, getter, expected_da
  ):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    if times is not None:
      builder.time_coords = times
    if media_times is not None:
      builder.media_time_coords = media_times
    builder.geos = geos
    setter(builder, nd)
    if isinstance(expected_da, list):
      actual_das = getter(builder)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(builder), expected_da)
    self.assertEqual(builder.geos, geos)
    if times is not None:
      self.assertEqual(builder.time_coords, times)
    if media_times is not None:
      self.assertEqual(builder.media_time_coords, media_times)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          nd=NATIONAL_KPI_ND,
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_kpi(nd),
          getter=lambda builder: builder.kpi,
          expected_da=NATIONAL_KPI_DA,
      ),
      dict(
          testcase_name='controls',
          nd=NATIONAL_CONTROLS_ND,
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_controls(
              nd, ['control_2', 'control_1']
          ),
          getter=lambda builder: builder.controls,
          expected_da=NATIONAL_CONTROLS_DA,
      ),
      dict(
          testcase_name='population',
          nd=NATIONAL_POPULATION_ND,
          times=None,
          media_times=None,
          setter=lambda builder, nd: builder.with_population(nd),
          getter=lambda builder: builder.population,
          expected_da=NATIONAL_POPULATION_DA,
      ),
      dict(
          testcase_name='revenue_per_kpi',
          nd=NATIONAL_REVENUE_PER_KPI_ND,
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_revenue_per_kpi(
              nd,
          ),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_da=NATIONAL_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=NATIONAL_NON_MEDIA_TREATMENTS_ND,
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel_2', 'non_media_channel_1']
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_da=NATIONAL_NON_MEDIA_TREATMENTS_DA,
      ),
      dict(
          testcase_name='organic_media',
          nd=NATIONAL_ORGANIC_MEDIA_ND,
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel_2', 'organic_media_channel_1']
          ),
          getter=lambda builder: builder.organic_media,
          expected_da=NATIONAL_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name='organic_rf',
          nd=[NATIONAL_ORGANIC_REACH_ND, NATIONAL_ORGANIC_FREQUENCY_ND],
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_2',
                  'organic_rf_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.organic_reach,
              builder.organic_frequency,
          ],
          expected_da=[
              NATIONAL_ORGANIC_REACH_DA,
              NATIONAL_ORGANIC_FREQUENCY_DA,
          ],
      ),
      dict(
          testcase_name='media',
          nd=[NATIONAL_MEDIA_ND, NATIONAL_MEDIA_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_2',
                  'media_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.media,
              builder.media_spend,
          ],
          expected_da=[
              NATIONAL_MEDIA_DA,
              NATIONAL_MEDIA_SPEND_DA,
          ],
      ),
      dict(
          testcase_name='reach',
          nd=[NATIONAL_REACH_ND, NATIONAL_FREQUENCY_ND, NATIONAL_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_2',
                  'rf_channel_1',
              ],
          ),
          getter=lambda builder: [
              builder.reach,
              builder.frequency,
              builder.rf_spend,
          ],
          expected_da=[
              NATIONAL_REACH_DA,
              NATIONAL_FREQUENCY_DA,
              NATIONAL_RF_SPEND_DA,
          ],
      ),
  )
  def test_with_no_geos_sets_to_national(
      self, nd, times, media_times, setter, getter, expected_da
  ):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    if times is not None:
      builder.time_coords = times
    if media_times is not None:
      builder.media_time_coords = media_times
    setter(builder, nd)
    if isinstance(expected_da, list):
      actual_das = getter(builder)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(builder), expected_da)
    self.assertEqual(builder.geos, [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME])
    if times is not None:
      self.assertEqual(builder.time_coords, times)
    if media_times is not None:
      self.assertEqual(builder.media_time_coords, media_times)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, kpi_nd: builder.with_kpi(kpi_nd),
          error_msg='Expected: 1 geos x 3 times. Got: (2,).',
      ),
      dict(
          testcase_name='controls',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, controls_nd: builder.with_controls(
              controls_nd, ['control_2', 'control_1']
          ),
          error_msg='Expected: 1 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='population',
          nd=np.array([[1]]),
          times=None,
          media_times=None,
          setter=lambda builder, population_nd: builder.with_population(
              population_nd
          ),
          error_msg='Expected: 1 geos. Got: (1, 1).',
      ),
      dict(
          testcase_name='revenue_per_kpi',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, kpi_nd: builder.with_revenue_per_kpi(kpi_nd),
          error_msg='Expected: 1 geos x 3 times. Got: (2,).',
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel_2', 'non_media_channel_1']
          ),
          error_msg='Expected: 1 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_media',
          nd=np.array([1, 2]),
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel_1', 'organic_media_channel_2']
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_reach',
          nd=[np.array([1, 2]), NATIONAL_ORGANIC_FREQUENCY_ND],
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_1',
                  'organic_rf_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_frequency',
          nd=[NATIONAL_ORGANIC_REACH_ND, np.array([1, 2])],
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_1',
                  'organic_rf_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='media_spend',
          nd=[NATIONAL_MEDIA_ND, np.array([1, 2])],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='media',
          nd=[np.array([1, 2]), NATIONAL_MEDIA_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='rf_spend',
          nd=[NATIONAL_REACH_ND, NATIONAL_FREQUENCY_ND, np.array([1, 2])],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='reach',
          nd=[np.array([1, 2]), NATIONAL_FREQUENCY_ND, NATIONAL_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='frequency',
          nd=[NATIONAL_REACH_ND, np.array([1, 2]), NATIONAL_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 1 geos x 3 media times x 2 dims. Got: (2,).',
      ),
  )
  def test_with_wrong_shape_national_geo(
      self, nd, times, media_times, setter, error_msg
  ):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    if times is not None:
      builder.time_coords = times
    if media_times is not None:
      builder.media_time_coords = media_times
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      setter(builder, nd)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_kpi(nd),
          error_msg='Expected: 3 geos x 3 times. Got: (2,).',
      ),
      dict(
          testcase_name='controls',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_controls(nd, ['control_2']),
          error_msg='Expected: 3 geos x 3 times x 1 dims. Got: (2,).',
      ),
      dict(
          testcase_name='population',
          nd=np.array([[1], [2], [3]]),
          times=None,
          media_times=None,
          setter=lambda builder, nd: builder.with_population(nd),
          error_msg='Expected: 3 geos. Got: (3, 1).',
      ),
      dict(
          testcase_name='revenue_per_kpi',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_revenue_per_kpi(nd),
          error_msg='Expected: 3 geos x 3 times. Got: (2,).',
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=np.array([1, 2]),
          times=TIME_COORDS,
          media_times=None,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel_2']
          ),
          error_msg='Expected: 3 geos x 3 times x 1 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_media',
          nd=np.array([1, 2]),
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel_1']
          ),
          error_msg='Expected: 3 geos x 3 media times x 1 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_reach',
          nd=[np.array([1, 2]), BASIC_ORGANIC_FREQUENCY_ND],
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_1',
                  'organic_rf_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='organic_frequency',
          nd=[BASIC_ORGANIC_REACH_ND, np.array([1, 2])],
          times=None,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_1',
                  'organic_rf_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='media_spend',
          nd=[BASIC_MEDIA_ND, np.array([1, 2])],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='media',
          nd=[np.array([1, 2]), BASIC_MEDIA_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='rf_spend',
          nd=[BASIC_REACH_ND, BASIC_FREQUENCY_ND, np.array([1, 2])],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='reach',
          nd=[np.array([1, 2]), BASIC_FREQUENCY_ND, BASIC_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 media times x 2 dims. Got: (2,).',
      ),
      dict(
          testcase_name='frequency',
          nd=[BASIC_REACH_ND, np.array([1, 2]), BASIC_RF_SPEND_ND],
          times=TIME_COORDS,
          media_times=TIME_COORDS,
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          error_msg='Expected: 3 geos x 3 media times x 2 dims. Got: (2,).',
      ),
  )
  def test_with_wrong_shape_non_national_geo(
      self, nd, times, media_times, setter, error_msg
  ):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    if times is not None:
      builder.time_coords = times
    if media_times is not None:
      builder.media_time_coords = media_times
    builder.geos = self.GEOS
    with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
      setter(builder, nd)

  @parameterized.named_parameters(
      dict(
          testcase_name='kpi',
          nd=BASIC_KPI_ND,
          setter=lambda builder, nd: builder.with_kpi(nd),
          is_media_time=False,
      ),
      dict(
          testcase_name='controls',
          nd=BASIC_CONTROLS_ND,
          setter=lambda builder, nd: builder.with_controls(
              nd, ['control_2', 'control_1']
          ),
          is_media_time=False,
      ),
      dict(
          testcase_name='revenue_per_kpi',
          nd=BASIC_REVENUE_PER_KPI_ND,
          setter=lambda builder, nd: builder.with_revenue_per_kpi(nd),
          is_media_time=False,
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=BASIC_NON_MEDIA_TREATMENTS_ND,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel_2', 'non_media_channel_1']
          ),
          is_media_time=False,
      ),
      dict(
          testcase_name='organic_media',
          nd=BASIC_ORGANIC_MEDIA_ND,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel_1', 'organic_media_channel_2']
          ),
          is_media_time=True,
      ),
      dict(
          testcase_name='organic_rf',
          nd=[BASIC_ORGANIC_REACH_ND, BASIC_ORGANIC_FREQUENCY_ND],
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel_1',
                  'organic_rf_channel_2',
              ],
          ),
          is_media_time=True,
      ),
      dict(
          testcase_name='media',
          nd=[BASIC_MEDIA_ND, BASIC_MEDIA_SPEND_ND],
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          is_media_time=True,
      ),
      dict(
          testcase_name='media_spend',
          nd=[BASIC_MEDIA_ND, BASIC_MEDIA_SPEND_ND],
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel_1',
                  'media_channel_2',
              ],
          ),
          is_media_time=False,
      ),
      dict(
          testcase_name='reach_frequency',
          nd=[BASIC_REACH_ND, BASIC_FREQUENCY_ND, BASIC_RF_SPEND_ND],
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          is_media_time=True,
      ),
      dict(
          testcase_name='rf_spend',
          nd=[BASIC_REACH_ND, BASIC_FREQUENCY_ND, BASIC_RF_SPEND_ND],
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel_1',
                  'rf_channel_2',
              ],
          ),
          is_media_time=False,
      ),
  )
  def test_with_no_time_coords_raises_error(self, nd, setter, is_media_time):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    error_msg = 'Time coordinates are required first. Set using .time_coords()'
    if is_media_time:
      error_msg = (
          'Media times are required first. Set using .media_time_coords()'
      )
    else:
      builder.media_time_coords = self.TIME_COORDS

    with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
      setter(builder, nd)

  @parameterized.named_parameters(
      dict(
          testcase_name='controls',
          nd=BASIC_CONTROLS_ND,
          setter=lambda builder, nd: builder.with_controls(
              nd, ['control_1', 'control_1']
          ),
          times=TIME_COORDS,
          media_times=None,
      ),
      dict(
          testcase_name='non_media_treatments',
          nd=BASIC_NON_MEDIA_TREATMENTS_ND,
          setter=lambda builder, nd: builder.with_non_media_treatments(
              nd, ['non_media_channel0', 'non_media_channel0']
          ),
          times=TIME_COORDS,
          media_times=None,
      ),
      dict(
          testcase_name='organic_media',
          nd=BASIC_ORGANIC_MEDIA_ND,
          setter=lambda builder, nd: builder.with_organic_media(
              nd, ['organic_media_channel0', 'organic_media_channel0']
          ),
          media_times=TIME_COORDS,
          times=None,
      ),
      dict(
          testcase_name='organic_rf',
          nd=[BASIC_ORGANIC_REACH_ND, BASIC_ORGANIC_FREQUENCY_ND],
          setter=lambda builder, nd: builder.with_organic_reach(
              or_nd=nd[0],
              of_nd=nd[1],
              organic_rf_channels=[
                  'organic_rf_channel0',
                  'organic_rf_channel0',
              ],
          ),
          media_times=TIME_COORDS,
          times=None,
      ),
      dict(
          testcase_name='media',
          nd=[BASIC_MEDIA_ND, BASIC_MEDIA_SPEND_ND],
          setter=lambda builder, nd: builder.with_media(
              m_nd=nd[0],
              ms_nd=nd[1],
              media_channels=[
                  'media_channel0',
                  'media_channel0',
              ],
          ),
          times=TIME_COORDS,
          media_times=TIME_COORDS,
      ),
      dict(
          testcase_name='reach',
          nd=[BASIC_REACH_ND, BASIC_FREQUENCY_ND, BASIC_RF_SPEND_ND],
          setter=lambda builder, nd: builder.with_reach(
              r_nd=nd[0],
              f_nd=nd[1],
              rfs_nd=nd[2],
              rf_channels=[
                  'rf_channel0',
                  'rf_channel0',
              ],
          ),
          times=TIME_COORDS,
          media_times=TIME_COORDS,
      ),
  )
  def test_with_duplicate_third_dim_raises_error(
      self, nd, setter, times, media_times
  ):
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    if media_times is not None:
      builder.media_time_coords = media_times
    if times is not None:
      builder.time_coords = times
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'given dimensions must be unique.'
    ):
      setter(builder, nd)

  def test_kpi_type_revenue_and_call_revenue_per_kpi(self):
    original_nd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).copy()
    builder = nd_array_input_data_builder.NDArrayInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    builder.geos = self.GEOS
    builder.time_coords = self.TIME_COORDS
    builder.with_revenue_per_kpi(original_nd)

    xr.testing.assert_equal(
        builder.revenue_per_kpi,
        xr.DataArray(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            dims=[constants.GEO, constants.TIME],
            coords={
                constants.GEO: self.GEOS,
                constants.TIME: self.TIME_COORDS,
            },
            name=constants.REVENUE_PER_KPI,
        ),
    )
    np.testing.assert_equal(
        original_nd, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


if __name__ == '__main__':
  absltest.main()
