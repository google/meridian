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

import os
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import data_frame_input_data_builder
import pandas as pd
import xarray as xr


class DataFrameInputDataBuilderTest(parameterized.TestCase):
  UNSORTED_TIMES = [
      "2024-01-02",
      "2024-01-02",
      "2024-01-01",
      "2024-01-02",
      "2024-01-01",
      "2024-01-01",
  ]
  BASIC_POPULATION_DF = pd.DataFrame({
      "geo": ["A", "A", "B", "B", "C", "C"],
      "population": [1000, 1000, 2000, 2000, 3000, 3000],
  })
  UNSORTED_POPULATION_DF = pd.DataFrame({
      "geo": ["B", "A", "C"],
      "population": [2000, 1000, 3000],
  })
  NATIONAL_POPULATION_DF = pd.DataFrame({
      "population": [1000, 1000, 2000, 2000, 3000, 3000],
  })
  BASIC_POPULATION_DA = xr.DataArray(
      [1000, 2000, 3000],
      dims=[constants.GEO],
      coords={constants.GEO: ["A", "B", "C"]},
      name=constants.POPULATION,
  )
  NATIONAL_POPULATION_DA = xr.DataArray(
      [2000],
      dims=[constants.GEO],
      coords={constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]},
      name=constants.POPULATION,
  )
  BASIC_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "kpi": [1, 1, 2, 2, 3, 3],
  })
  UNSORTED_KPI_DF = pd.DataFrame({
      "time": ["2024-01-02", "2024-01-01"] * 3,
      "geo": ["B", "A", "A", "C", "C", "B"],
      "kpi": [2, 1, 1, 3, 3, 2],
  })
  NATIONAL_KPI_DF = pd.DataFrame({
      "time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "kpi": [1, 1, 2, 2, 3, 3],
  })
  DUPE_TIME_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01"] * 6,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "kpi": [1, 1, 2, 2, 3, 3],
  })
  INCONSISTENT_TIME_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"],
      "geo": ["A", "A", "B", "B", "C", "C"],
      "kpi": [1, 1, 2, 2, 3, 3],
  })
  BASIC_KPI_DA = xr.DataArray(
      [[1, 1], [2, 2], [3, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
      },
      name=constants.KPI,
  )
  NATIONAL_KPI_DA = xr.DataArray(
      [[1, 1, 2, 2, 3, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      },
      name=constants.KPI,
  )
  BASIC_CONTROLS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "control_1": [1, 1, 2, 2, 3, 3],
      "control_2": [10, 10, 20, 20, 30, 30],
  })
  UNSORTED_CONTROLS_DF = pd.DataFrame({
      "time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "control_2": [20, 10, 10, 30, 20, 30],
      "control_1": [2, 1, 1, 3, 2, 3],
  })
  NATIONAL_CONTROLS_DF = pd.DataFrame({
      "time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "control_1": [0, 2, 1, 3, 2, 4],
      "control_2": [10, 20, 10, 30, 20, 40],
  })
  DUPE_TIME_CONTROLS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "control_1": [0, 2, 1, 3, 2, 4],
      "control_2": [10, 20, 10, 30, 20, 40],
  })
  INCONSISTENT_TIME_CONTROLS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"],
      "geo": ["A", "A", "B", "B", "C", "C"],
      "control_1": [1, 1, 2, 2, 3, 3],
      "control_2": [10, 10, 20, 20, 30, 30],
  })
  BASIC_CONTROLS_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
          constants.CONTROL_VARIABLE: ["control_1", "control_2"],
      },
      name=constants.CONTROLS,
  )
  NATIONAL_CONTROLS_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
          constants.CONTROL_VARIABLE: ["control_1", "control_2"],
      },
      name=constants.CONTROLS,
  )
  BASIC_REVENUE_PER_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "revenue_per_kpi": [1, 1, 2, 2, 3, 3],
  })
  UNSORTED_REVENUE_PER_KPI_DF = pd.DataFrame({
      "time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "revenue_per_kpi": [2, 1, 1, 3, 2, 3],
  })
  NATIONAL_REVENUE_PER_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "revenue_per_kpi": [1, 2, 3],
  })
  DUPE_TIME_REVENUE_PER_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01"] * 6,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "revenue_per_kpi": [1, 1, 2, 2, 3, 3],
  })
  INCONSISTENT_TIME_REVENUE_PER_KPI_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"],
      "geo": ["A", "A", "B", "B", "C", "C"],
      "revenue_per_kpi": [1, 1, 2, 2, 3, 3],
  })
  BASIC_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 1], [2, 2], [3, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
      },
      name=constants.REVENUE_PER_KPI,
  )
  NATIONAL_REVENUE_PER_KPI_DA = xr.DataArray(
      [[1, 2, 3]],
      dims=[constants.GEO, constants.TIME],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: ["2024-01-01", "2024-01-02", "2024-01-03"],
      },
      name=constants.REVENUE_PER_KPI,
  )
  BASIC_NON_MEDIA_TREATMENTS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "non_media_channel_1": [1, 1, 2, 2, 3, 3],
      "non_media_channel_2": [10, 10, 20, 20, 30, 30],
  })
  UNSORTED_NON_MEDIA_TREATMENTS_DF = pd.DataFrame({
      "time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "non_media_channel_2": [20, 10, 10, 30, 20, 30],
      "non_media_channel_1": [2, 1, 1, 3, 2, 3],
  })
  NATIONAL_NON_MEDIA_TREATMENTS_DF = pd.DataFrame({
      "time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "non_media_channel_1": [0, 2, 1, 3, 2, 4],
      "non_media_channel_2": [10, 20, 10, 30, 20, 40],
  })
  DUPE_TIME_NON_MEDIA_TREATMENTS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 3,
      "non_media_channel_1": [0, 2, 1, 3, 2, 4],
      "non_media_channel_2": [10, 20, 10, 30, 20, 40],
  })
  INCONSISTENT_TIME_NON_MEDIA_TREATMENTS_DF = pd.DataFrame({
      "time": ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"],
      "geo": ["A", "A", "B", "B", "C", "C"],
      "non_media_channel_1": [1, 1, 2, 2, 3, 3],
      "non_media_channel_2": [10, 10, 20, 20, 30, 30],
  })
  BASIC_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
          constants.NON_MEDIA_CHANNEL: [
              "non_media_channel_1",
              "non_media_channel_2",
          ],
      },
      name=constants.NON_MEDIA_TREATMENTS,
  )
  NATIONAL_NON_MEDIA_TREATMENTS_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
          constants.NON_MEDIA_CHANNEL: [
              "non_media_channel_1",
              "non_media_channel_2",
          ],
      },
      name=constants.NON_MEDIA_TREATMENTS,
  )
  BASIC_ORGANIC_MEDIA_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "organic_media_1": [1, 1, 2, 2, 3, 3],
      "organic_media_2": [10, 10, 20, 20, 30, 30],
  })
  UNSORTED_ORGANIC_MEDIA_DF = pd.DataFrame({
      "media_time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "organic_media_2": [20, 10, 10, 30, 20, 30],
      "organic_media_1": [2, 1, 1, 3, 2, 3],
  })
  NATIONAL_ORGANIC_MEDIA_DF = pd.DataFrame({
      "media_time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "organic_media_1": [0, 2, 1, 3, 2, 4],
      "organic_media_2": [10, 20, 10, 30, 20, 40],
  })
  DUPE_MEDIA_TIME_ORGANIC_MEDIA_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "organic_media_1": [0, 2, 1, 3, 2, 4],
      "organic_media_2": [10, 20, 10, 30, 20, 40],
  })
  INCONSISTENT_MEDIA_TIME_ORGANIC_MEDIA_DF = pd.DataFrame({
      "media_time": (
          ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"]
      ),
      "geo": ["A", "A", "B", "B", "C", "C"],
      "organic_media_1": [1, 1, 2, 2, 3, 3],
      "organic_media_2": [10, 10, 20, 20, 30, 30],
  })
  BASIC_ORGANIC_MEDIA_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: [
              "2024-01-01",
              "2024-01-02",
          ],
          constants.ORGANIC_MEDIA_CHANNEL: [
              "organic_media_1",
              "organic_media_2",
          ],
      },
      name=constants.ORGANIC_MEDIA,
  )
  NATIONAL_ORGANIC_MEDIA_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.ORGANIC_MEDIA_CHANNEL: [
              "organic_media_1",
              "organic_media_2",
          ],
      },
      name=constants.ORGANIC_MEDIA,
  )
  BASIC_ORGANIC_REACH_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "organic_reach_1": [1, 1, 2, 2, 3, 3],
      "organic_reach_2": [10, 10, 20, 20, 30, 30],
      "organic_frequency_1": [11, 11, 22, 22, 33, 33],
      "organic_frequency_2": [110, 110, 220, 220, 330, 330],
  })
  UNSORTED_ORGANIC_REACH_DF = pd.DataFrame({
      "media_time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "organic_reach_2": [20, 10, 10, 30, 20, 30],
      "organic_reach_1": [2, 1, 1, 3, 2, 3],
      "organic_frequency_2": [220, 110, 110, 330, 220, 330],
      "organic_frequency_1": [22, 11, 11, 33, 22, 33],
  })
  NATIONAL_ORGANIC_REACH_DF = pd.DataFrame({
      "media_time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "organic_reach_1": [0, 2, 1, 3, 2, 4],
      "organic_reach_2": [10, 20, 10, 30, 20, 40],
      "organic_frequency_1": [10, 22, 11, 33, 22, 44],
      "organic_frequency_2": [110, 220, 110, 330, 220, 440],
  })
  DUPE_MEDIA_TIME_ORGANIC_REACH_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "organic_reach_1": [0, 2, 1, 3, 2, 4],
      "organic_frequency_1": [10, 20, 10, 30, 20, 40],
  })
  INCONSISTENT_MEDIA_TIME_ORGANIC_REACH_DF = pd.DataFrame({
      "media_time": (
          ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"]
      ),
      "geo": ["A", "A", "B", "B", "C", "C"],
      "organic_reach_1": [1, 1, 2, 2, 3, 3],
      "organic_frequency_1": [10, 10, 20, 20, 30, 30],
  })
  BASIC_ORGANIC_REACH_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: ["2024-01-01", "2024-01-02"],
          constants.ORGANIC_RF_CHANNEL: [
              "organic_rf_channel_1",
              "organic_rf_channel_2",
          ],
      },
      name=constants.ORGANIC_REACH,
  )
  BASIC_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [[[11, 110], [11, 110]], [[22, 220], [22, 220]], [[33, 330], [33, 330]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.ORGANIC_RF_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: ["2024-01-01", "2024-01-02"],
          constants.ORGANIC_RF_CHANNEL: [
              "organic_rf_channel_1",
              "organic_rf_channel_2",
          ],
      },
      name=constants.ORGANIC_FREQUENCY,
  )
  NATIONAL_ORGANIC_REACH_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.ORGANIC_RF_CHANNEL: [
              "organic_rf_channel_1",
              "organic_rf_channel_2",
          ],
      },
      name=constants.ORGANIC_REACH,
  )
  NATIONAL_ORGANIC_FREQUENCY_DA = xr.DataArray(
      [[[10, 110], [22, 220], [11, 110], [33, 330], [22, 220], [44, 440]]],
      dims=[constants.GEO, constants.MEDIA_TIME, constants.ORGANIC_RF_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.ORGANIC_RF_CHANNEL: [
              "organic_rf_channel_1",
              "organic_rf_channel_2",
          ],
      },
      name=constants.ORGANIC_FREQUENCY,
  )
  BASIC_MEDIA_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "media_1": [1, 1, 2, 2, 3, 3],
      "media_2": [10, 10, 20, 20, 30, 30],
      "media_spend_1": [11, 11, 22, 22, 33, 33],
      "media_spend_2": [110, 110, 220, 220, 330, 330],
  })
  UNSORTED_MEDIA_DF = pd.DataFrame({
      "media_time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "media_2": [20, 10, 10, 30, 20, 30],
      "media_1": [2, 1, 1, 3, 2, 3],
      "media_spend_2": [220, 110, 110, 330, 220, 330],
      "media_spend_1": [22, 11, 11, 33, 22, 33],
  })
  NATIONAL_MEDIA_DF = pd.DataFrame({
      "media_time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "media_1": [0, 2, 1, 3, 2, 4],
      "media_2": [10, 20, 10, 30, 20, 40],
      "media_spend_1": [10, 22, 11, 33, 22, 44],
      "media_spend_2": [110, 220, 110, 330, 220, 440],
  })
  DUPE_MEDIA_TIME_MEDIA_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "media_1": [0, 2, 1, 3, 2, 4],
      "media_spend_1": [10, 20, 10, 30, 20, 40],
  })
  INCONSISTENT_MEDIA_TIME_MEDIA_DF = pd.DataFrame({
      "media_time": (
          ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"]
      ),
      "geo": ["A", "A", "B", "B", "C", "C"],
      "media_1": [1, 1, 2, 2, 3, 3],
      "media_spend_1": [10, 10, 20, 20, 30, 30],
  })
  BASIC_MEDIA_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: ["2024-01-01", "2024-01-02"],
          constants.MEDIA_CHANNEL: [
              "media_channel_1",
              "media_channel_2",
          ],
      },
      name=constants.MEDIA,
  )
  BASIC_MEDIA_SPEND_DA = xr.DataArray(
      [[[11, 110], [11, 110]], [[22, 220], [22, 220]], [[33, 330], [33, 330]]],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.MEDIA_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
          constants.MEDIA_CHANNEL: [
              "media_channel_1",
              "media_channel_2",
          ],
      },
      name=constants.MEDIA_SPEND,
  )
  NATIONAL_MEDIA_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.MEDIA_CHANNEL: [
              "media_channel_1",
              "media_channel_2",
          ],
      },
      name=constants.MEDIA,
  )
  NATIONAL_MEDIA_SPEND_DA = xr.DataArray(
      [[[10, 110], [22, 220], [11, 110], [33, 330], [22, 220], [44, 440]]],
      dims=[constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
          constants.MEDIA_CHANNEL: [
              "media_channel_1",
              "media_channel_2",
          ],
      },
      name=constants.MEDIA_SPEND,
  )
  BASIC_REACH_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "geo": ["A", "A", "B", "B", "C", "C"],
      "reach_1": [1, 1, 2, 2, 3, 3],
      "reach_2": [10, 10, 20, 20, 30, 30],
      "frequency_1": [11, 11, 22, 22, 33, 33],
      "frequency_2": [110, 110, 220, 220, 330, 330],
      "rf_spend_1": [111, 111, 222, 222, 333, 333],
      "rf_spend_2": [1110, 1110, 2220, 2220, 3330, 3330],
  })
  UNSORTED_REACH_DF = pd.DataFrame({
      "media_time": UNSORTED_TIMES,
      "geo": ["B", "A", "A", "C", "B", "C"],
      "reach_2": [20, 10, 10, 30, 20, 30],
      "reach_1": [2, 1, 1, 3, 2, 3],
      "frequency_2": [220, 110, 110, 330, 220, 330],
      "frequency_1": [22, 11, 11, 33, 22, 33],
      "rf_spend_2": [2220, 1110, 1110, 3330, 2220, 3330],
      "rf_spend_1": [222, 111, 111, 333, 222, 333],
  })
  NATIONAL_REACH_DF = pd.DataFrame({
      "media_time": [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
      "reach_1": [0, 2, 1, 3, 2, 4],
      "reach_2": [10, 20, 10, 30, 20, 40],
      "frequency_1": [10, 22, 11, 33, 22, 44],
      "frequency_2": [110, 220, 110, 330, 220, 440],
      "rf_spend_1": [110, 220, 110, 330, 220, 440],
      "rf_spend_2": [1110, 2220, 1110, 3330, 2220, 4440],
  })
  DUPE_MEDIA_TIME_REACH_DF = pd.DataFrame({
      "media_time": ["2024-01-01", "2024-01-02"] * 3,
      "reach_1": [0, 2, 1, 3, 2, 4],
      "frequency_1": [10, 20, 10, 30, 20, 40],
      "rf_spend_1": [111, 111, 222, 222, 333, 333],
  })
  INCONSISTENT_MEDIA_TIME_REACH_DF = pd.DataFrame({
      "media_time": (
          ["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-03"]
      ),
      "geo": ["A", "A", "B", "B", "C", "C"],
      "reach_1": [1, 1, 2, 2, 3, 3],
      "frequency_1": [10, 10, 20, 20, 30, 30],
      "rf_spend_1": [111, 111, 222, 222, 333, 333],
  })
  BASIC_REACH_DA = xr.DataArray(
      [[[1, 10], [1, 10]], [[2, 20], [2, 20]], [[3, 30], [3, 30]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: ["2024-01-01", "2024-01-02"],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.REACH,
  )
  BASIC_FREQUENCY_DA = xr.DataArray(
      [[[11, 110], [11, 110]], [[22, 220], [22, 220]], [[33, 330], [33, 330]]],
      dims=[
          constants.GEO,
          constants.MEDIA_TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.MEDIA_TIME: ["2024-01-01", "2024-01-02"],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.FREQUENCY,
  )
  BASIC_RF_SPEND_DA = xr.DataArray(
      [
          [[111, 1110], [111, 1110]],
          [[222, 2220], [222, 2220]],
          [[333, 3330], [333, 3330]],
      ],
      dims=[
          constants.GEO,
          constants.TIME,
          constants.RF_CHANNEL,
      ],
      coords={
          constants.GEO: ["A", "B", "C"],
          constants.TIME: ["2024-01-01", "2024-01-02"],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.RF_SPEND,
  )
  NATIONAL_REACH_DA = xr.DataArray(
      [[[0, 10], [2, 20], [1, 10], [3, 30], [2, 20], [4, 40]]],
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.REACH,
  )
  NATIONAL_FREQUENCY_DA = xr.DataArray(
      [[[10, 110], [22, 220], [11, 110], [33, 330], [22, 220], [44, 440]]],
      dims=[constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.MEDIA_TIME: [
              f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)
          ],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.FREQUENCY,
  )
  NATIONAL_RF_SPEND_DA = xr.DataArray(
      [[
          [110, 1110],
          [220, 2220],
          [110, 1110],
          [330, 3330],
          [220, 2220],
          [440, 4440],
      ]],
      dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
      coords={
          constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
          constants.TIME: [f"2024-01-{str(i).zfill(2)}" for i in range(1, 7)],
          constants.RF_CHANNEL: [
              "rf_channel_1",
              "rf_channel_2",
          ],
      },
      name=constants.RF_SPEND,
  )
  GEO_ORGANIC_DF = pd.read_csv(
      os.path.join(
          os.path.dirname(__file__),
          "unit_testing_data",
          "sample_data_with_organic_and_non_media.csv",
      )
  )
  LAGGED_MEDIA_DF = pd.read_csv(
      os.path.join(
          os.path.dirname(__file__),
          "unit_testing_data",
          "lagged_sample_data_media_only.csv",
      )
  )
  LAGGED_RF_DF = pd.read_csv(
      os.path.join(
          os.path.dirname(__file__),
          "unit_testing_data",
          "lagged_sample_data_rf_only.csv",
      )
  )

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          df=BASIC_POPULATION_DF,
          setter=lambda builder, df: builder.with_population(df),
          getter=lambda builder: builder.population,
          expected_da=BASIC_POPULATION_DA,
      ),
      dict(
          testcase_name="kpi",
          df=BASIC_KPI_DF,
          setter=lambda builder, df: builder.with_kpi(df),
          getter=lambda builder: builder.kpi,
          expected_da=BASIC_KPI_DA,
      ),
      dict(
          testcase_name="controls",
          df=BASIC_CONTROLS_DF,
          setter=lambda builder, df: builder.with_controls(
              df, ["control_1", "control_2"]
          ),
          getter=lambda builder: builder.controls,
          expected_da=BASIC_CONTROLS_DA,
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=BASIC_REVENUE_PER_KPI_DF,
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_da=BASIC_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name="non_media_treatments",
          df=BASIC_NON_MEDIA_TREATMENTS_DF,
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1", "non_media_channel_2"]
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_da=BASIC_NON_MEDIA_TREATMENTS_DA,
      ),
      dict(
          testcase_name="organic_media",
          df=BASIC_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1", "organic_media_2"]
          ),
          getter=lambda builder: builder.organic_media,
          expected_da=BASIC_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=BASIC_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1", "organic_reach_2"],
              ["organic_frequency_1", "organic_frequency_2"],
              ["organic_rf_channel_1", "organic_rf_channel_2"],
          ),
          getter=lambda builder: [
              builder.organic_reach,
              builder.organic_frequency,
          ],
          expected_da=[BASIC_ORGANIC_REACH_DA, BASIC_ORGANIC_FREQUENCY_DA],
      ),
      dict(
          testcase_name="media",
          df=BASIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_2"],
              "media_time",
          ),
          getter=lambda builder: [builder.media, builder.media_spend],
          expected_da=[BASIC_MEDIA_DA, BASIC_MEDIA_SPEND_DA],
      ),
      dict(
          testcase_name="reach",
          df=BASIC_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_2"],
              ["frequency_1", "frequency_2"],
              ["rf_spend_1", "rf_spend_2"],
              ["rf_channel_1", "rf_channel_2"],
              "media_time",
          ),
          getter=lambda builder: [
              builder.reach,
              builder.frequency,
              builder.rf_spend,
          ],
          expected_da=[BASIC_REACH_DA, BASIC_FREQUENCY_DA, BASIC_RF_SPEND_DA],
      ),
  )
  def test_with_component_basic(
      self,
      df,
      setter,
      getter,
      expected_da,
  ):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE,
        default_media_time_column="media_time",
    )
    setter(builder, df)
    if isinstance(expected_da, list):
      actual_das = getter(builder)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(builder), expected_da)

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          df=UNSORTED_POPULATION_DF,
          setter=lambda builder, df: builder.with_population(df),
          getter=lambda builder: builder.population,
          expected_da=BASIC_POPULATION_DA,
      ),
      dict(
          testcase_name="kpi",
          df=UNSORTED_KPI_DF,
          setter=lambda builder, df: builder.with_kpi(df),
          getter=lambda builder: builder.kpi,
          expected_da=BASIC_KPI_DA,
      ),
      dict(
          testcase_name="controls",
          df=UNSORTED_CONTROLS_DF,
          setter=lambda builder, df: builder.with_controls(
              df, ["control_1", "control_2"]
          ),
          getter=lambda builder: builder.controls,
          expected_da=BASIC_CONTROLS_DA,
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=UNSORTED_REVENUE_PER_KPI_DF,
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_da=BASIC_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name="non_media_treatments",
          df=UNSORTED_NON_MEDIA_TREATMENTS_DF,
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1", "non_media_channel_2"]
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_da=BASIC_NON_MEDIA_TREATMENTS_DA,
      ),
      dict(
          testcase_name="organic_media",
          df=UNSORTED_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1", "organic_media_2"]
          ),
          getter=lambda builder: builder.organic_media,
          expected_da=BASIC_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=UNSORTED_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1", "organic_reach_2"],
              ["organic_frequency_1", "organic_frequency_2"],
              [
                  "organic_rf_channel_1",
                  "organic_rf_channel_2",
              ],
          ),
          getter=lambda builder: [
              builder.organic_reach,
              builder.organic_frequency,
          ],
          expected_da=[BASIC_ORGANIC_REACH_DA, BASIC_ORGANIC_FREQUENCY_DA],
      ),
      dict(
          testcase_name="media",
          df=UNSORTED_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_2"],
              "media_time",
          ),
          getter=lambda builder: [builder.media, builder.media_spend],
          expected_da=[BASIC_MEDIA_DA, BASIC_MEDIA_SPEND_DA],
      ),
      dict(
          testcase_name="reach",
          df=UNSORTED_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_2"],
              ["frequency_1", "frequency_2"],
              ["rf_spend_1", "rf_spend_2"],
              ["rf_channel_1", "rf_channel_2"],
              "media_time",
          ),
          getter=lambda builder: [
              builder.reach,
              builder.frequency,
              builder.rf_spend,
          ],
          expected_da=[BASIC_REACH_DA, BASIC_FREQUENCY_DA, BASIC_RF_SPEND_DA],
      ),
  )
  def test_with_unsorted_df_returns_sorted_da_and_leaves_original_df_unmodified(
      self,
      df,
      setter,
      getter,
      expected_da,
  ):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE,
        default_media_time_column="media_time",
    )
    original_df = df.copy()
    setter(builder, df)
    if isinstance(expected_da, list):
      actual_das = getter(builder)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(builder), expected_da)
    pd.testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          setter=lambda builder, df: builder.with_population(df),
          getter=lambda builder: builder.population,
          is_population=True,
      ),
      dict(
          testcase_name="kpi",
          setter=lambda builder, df: builder.with_kpi(df),
          getter=lambda builder: builder.kpi,
      ),
      dict(
          testcase_name="controls",
          setter=lambda builder, df: builder.with_controls(
              df,
              control_cols=["control_0", "control_1"],
          ),
          getter=lambda builder: builder.controls,
      ),
      dict(
          testcase_name="revenue_per_kpi",
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          getter=lambda builder: builder.revenue_per_kpi,
      ),
      dict(
          testcase_name="non_media_treatments",
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_0", "non_media_1"]
          ),
          getter=lambda builder: builder.non_media_treatments,
      ),
      dict(
          testcase_name="organic_media",
          setter=lambda builder, df: builder.with_organic_media(
              df,
              media_time_col="time",
              organic_media_cols=[
                  "organic_media_0",
                  "organic_media_1",
                  "organic_media_2",
                  "organic_media_3",
              ],
          ),
          getter=lambda builder: builder.organic_media,
          is_media_time=True,
          is_time=False,
      ),
      dict(
          testcase_name="organic_reach_frequency",
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              media_time_col="time",
              organic_reach_cols=["organic_reach_0"],
              organic_frequency_cols=["organic_frequency_0"],
              organic_rf_channels=["organic_rf_channel_0"],
          ),
          getter=lambda builder: [
              builder.organic_reach,
              builder.organic_frequency,
          ],
          is_media_time=True,
          is_time=False,
      ),
      dict(
          testcase_name="media",
          setter=lambda builder, df: builder.with_media(
              df,
              time_col="time",
              media_cols=["media_0", "media_1", "media_2"],
              media_spend_cols=[
                  "media_spend_0",
                  "media_spend_1",
                  "media_spend_2",
              ],
              media_channels=[
                  "Channel0",
                  "Channel1",
                  "Channel2",
              ],
          ),
          getter=lambda builder: [builder.media, builder.media_spend],
          is_media_time=True,
          is_time=True,
      ),
      dict(
          testcase_name="reach",
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_0", "reach_1"],
              ["frequency_0", "frequency_1"],
              ["rf_spend_0", "rf_spend_1"],
              ["rf_channel_0", "rf_channel_1"],
          ),
          getter=lambda builder: [
              builder.reach,
              builder.frequency,
              builder.rf_spend,
          ],
          is_media_time=True,
          is_time=True,
      ),
  )
  def test_with_master_df(
      self,
      setter,
      getter,
      is_population=False,
      is_media_time=False,
      is_time=True,
  ):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )
    setter(builder, self.GEO_ORGANIC_DF)
    self.assertIsNotNone(getter(builder))
    self.assertIsNotNone(builder.geos)
    if is_population:
      self.assertIsNone(builder.time_coords)
    else:
      if is_media_time:
        self.assertIsNotNone(builder.media_time_coords)
      if is_time:
        self.assertIsNotNone(builder.time_coords)

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          df=BASIC_POPULATION_DF[["geo"]],
          setter=lambda builder, df: builder.with_population(df),
          missing_cols="['population']",
      ),
      dict(
          testcase_name="kpi",
          df=BASIC_KPI_DF[["geo", "time"]],
          setter=lambda builder, df: builder.with_kpi(df),
          missing_cols="['kpi', 'time']",
      ),
      dict(
          testcase_name="controls",
          df=BASIC_CONTROLS_DF[["geo", "time"]],
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["control_1", "control_2"]
          ),
          missing_cols="['control_1', 'control_2', 'time']",
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=BASIC_REVENUE_PER_KPI_DF[["geo", "time"]],
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          missing_cols="['revenue_per_kpi', 'time']",
      ),
      dict(
          testcase_name="non_media_treatments",
          df=BASIC_NON_MEDIA_TREATMENTS_DF[["geo", "time"]],
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1", "non_media_channel_2"]
          ),
          missing_cols="['non_media_channel_1', 'non_media_channel_2', 'time']",
      ),
      dict(
          testcase_name="organic_media",
          df=BASIC_ORGANIC_MEDIA_DF[["geo", "media_time"]],
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1", "organic_media_2"]
          ),
          missing_cols="['organic_media_1', 'organic_media_2', 'media_time']",
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=BASIC_ORGANIC_REACH_DF[["geo", "media_time"]],
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1", "organic_reach_2"],
              ["organic_frequency_1", "organic_frequency_2"],
              [
                  "organic_rf_channel_1",
                  "organic_rf_channel_2",
              ],
          ),
          missing_cols=(
              "['organic_reach_1', 'organic_reach_2', 'organic_frequency_1',"
              " 'organic_frequency_2', 'media_time']"
          ),
      ),
      dict(
          testcase_name="media",
          df=BASIC_MEDIA_DF[
              ["geo", "media_time", "media_spend_1", "media_spend_2"]
          ],
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_2"],
              "media_time",
          ),
          missing_cols="['media_1', 'media_2', 'media_time']",
      ),
      dict(
          testcase_name="media_spend",
          df=BASIC_MEDIA_DF[["geo", "media_time", "media_1", "media_2"]],
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_2"],
              "media_time",
          ),
          missing_cols="['media_spend_1', 'media_spend_2', 'media_time']",
      ),
      dict(
          testcase_name="reach",
          df=BASIC_REACH_DF[["geo", "media_time"]],
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_2"],
              ["frequency_1", "frequency_2"],
              ["rf_spend_1", "rf_spend_2"],
              ["rf_channel_1", "rf_channel_2"],
              "media_time",
          ),
          missing_cols=(
              "['reach_1', 'reach_2', 'frequency_1',"
              " 'frequency_2', 'rf_spend_1', 'rf_spend_2', 'media_time']"
          ),
      ),
  )
  def test_with_missing_data_column(
      self,
      df,
      setter,
      missing_cols,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"DataFrame is missing one or more columns from {missing_cols}",
    ):
      setter(
          data_frame_input_data_builder.DataFrameInputDataBuilder(
              kpi_type=constants.NON_REVENUE,
              default_media_time_column="media_time",
          ),
          df,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          df=NATIONAL_POPULATION_DF,
          setter=lambda builder, df: builder.with_population(df),
          getter=lambda builder: builder.population,
          expected_da=NATIONAL_POPULATION_DA,
      ),
      dict(
          testcase_name="kpi",
          df=NATIONAL_KPI_DF,
          setter=lambda builder, df: builder.with_kpi(df),
          getter=lambda builder: builder.kpi,
          expected_da=NATIONAL_KPI_DA,
      ),
      dict(
          testcase_name="controls",
          df=NATIONAL_CONTROLS_DF,
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["control_1", "control_2"]
          ),
          getter=lambda builder: builder.controls,
          expected_da=NATIONAL_CONTROLS_DA,
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=NATIONAL_REVENUE_PER_KPI_DF,
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          getter=lambda builder: builder.revenue_per_kpi,
          expected_da=NATIONAL_REVENUE_PER_KPI_DA,
      ),
      dict(
          testcase_name="non_media_treatments",
          df=NATIONAL_NON_MEDIA_TREATMENTS_DF,
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1", "non_media_channel_2"]
          ),
          getter=lambda builder: builder.non_media_treatments,
          expected_da=NATIONAL_NON_MEDIA_TREATMENTS_DA,
      ),
      dict(
          testcase_name="organic_media",
          df=NATIONAL_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1", "organic_media_2"]
          ),
          getter=lambda builder: builder.organic_media,
          expected_da=NATIONAL_ORGANIC_MEDIA_DA,
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=NATIONAL_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1", "organic_reach_2"],
              ["organic_frequency_1", "organic_frequency_2"],
              [
                  "organic_rf_channel_1",
                  "organic_rf_channel_2",
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
          testcase_name="media",
          df=NATIONAL_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              [
                  "media_channel_1",
                  "media_channel_2",
              ],
              "media_time",
          ),
          getter=lambda builder: [builder.media, builder.media_spend],
          expected_da=[NATIONAL_MEDIA_DA, NATIONAL_MEDIA_SPEND_DA],
      ),
      dict(
          testcase_name="reach",
          df=NATIONAL_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_2"],
              ["frequency_1", "frequency_2"],
              ["rf_spend_1", "rf_spend_2"],
              ["rf_channel_1", "rf_channel_2"],
              "media_time",
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
  def test_with_missing_geo_column(self, df, setter, getter, expected_da):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE,
        default_media_time_column="media_time",
    )
    setter(builder, df)
    if isinstance(expected_da, list):
      actual_das = getter(builder)
      for i, da in enumerate(expected_da):
        xr.testing.assert_equal(actual_das[i], da)
    else:
      xr.testing.assert_equal(getter(builder), expected_da)

  @parameterized.named_parameters(
      dict(
          testcase_name="population",
          setter=lambda builder, df: builder.with_population(
              df, population_col="geo"
          ),
          dupes="['geo', 'geo']",
      ),
      dict(
          testcase_name="kpi",
          setter=lambda builder, df: builder.with_kpi(df, kpi_col="time"),
          dupes="['time', 'time', 'geo']",
      ),
      dict(
          testcase_name="controls",
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["geo", "control_1"]
          ),
          dupes="['geo', 'control_1', 'time', 'geo']",
      ),
      dict(
          testcase_name="revenue_per_kpi",
          setter=lambda builder, df: builder.with_revenue_per_kpi(
              df, revenue_per_kpi_col="time"
          ),
          dupes="['time', 'time', 'geo']",
      ),
      dict(
          testcase_name="non_media_treatments",
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_1", "non_media_1"]
          ),
          dupes="['non_media_1', 'non_media_1', 'time', 'geo']",
      ),
      dict(
          testcase_name="organic_media",
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1", "organic_media_1"], media_time_col="time"
          ),
          dupes="['organic_media_1', 'organic_media_1', 'time', 'geo']",
      ),
      dict(
          testcase_name="organic_reach_frequency",
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_0", "organic_reach_0"],
              ["organic_frequency_0", "organic_frequency_0"],
              ["organic_rf_channel_1", "organic_rf_channel_2"],
              media_time_col="time",
          ),
          dupes=(
              "['organic_reach_0', 'organic_reach_0', 'organic_frequency_0',"
              " 'organic_frequency_0', 'time', 'geo']"
          ),
      ),
      dict(
          testcase_name="media",
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_1"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_2"],
          ),
          dupes="['media_1', 'media_1', 'time', 'geo']",
      ),
      dict(
          testcase_name="media_spend",
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_1"],
              ["media_channel_1", "media_channel_2"],
          ),
          dupes="['media_spend_1', 'media_spend_1', 'time', 'geo']",
      ),
      dict(
          testcase_name="reach",
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_1"],
              ["frequency_1", "frequency_1"],
              ["rf_spend_1", "rf_spend_1"],
              ["rf_channel_1", "rf_channel_2"],
          ),
          dupes=(
              "['reach_1', 'reach_1', 'frequency_1', 'frequency_1',"
              " 'rf_spend_1', 'rf_spend_1', 'time', 'geo']"
          ),
      ),
  )
  def test_with_duplicate_columns(self, setter, dupes):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"DataFrame has duplicate columns from {dupes}",
    ):
      setter(
          data_frame_input_data_builder.DataFrameInputDataBuilder(
              kpi_type=constants.NON_REVENUE
          ),
          self.GEO_ORGANIC_DF,
      )

  def test_media_and_spend_same_columns_no_error(self):
    data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    ).with_media(
        self.GEO_ORGANIC_DF,
        ["media_1", "media_2"],
        ["media_1", "media_2"],
        ["media_1", "media_2"],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="kpi",
          df=BASIC_KPI_DF[["geo", "kpi"]],
          setter=lambda builder, df: builder.with_kpi(df),
          missing_cols="['kpi', 'time']",
      ),
      dict(
          testcase_name="controls",
          df=BASIC_CONTROLS_DF[["geo", "control_1"]],
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["control_1"]
          ),
          missing_cols="['control_1', 'time']",
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=BASIC_REVENUE_PER_KPI_DF[["geo", "revenue_per_kpi"]],
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
          missing_cols="['revenue_per_kpi', 'time']",
      ),
      dict(
          testcase_name="non_media_treatments",
          df=BASIC_NON_MEDIA_TREATMENTS_DF[["geo", "non_media_channel_1"]],
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1"]
          ),
          missing_cols="['non_media_channel_1', 'time']",
      ),
      dict(
          testcase_name="organic_media",
          df=BASIC_ORGANIC_MEDIA_DF[["geo", "organic_media_1"]],
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1"]
          ),
          missing_cols="['organic_media_1', 'media_time']",
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=BASIC_ORGANIC_REACH_DF[
              ["geo", "organic_reach_1", "organic_frequency_1"]
          ],
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1"],
              ["organic_frequency_1"],
              ["organic_rf_channel_1"],
          ),
          missing_cols=(
              "['organic_reach_1', 'organic_frequency_1', 'media_time']"
          ),
      ),
      dict(
          testcase_name="media",
          df=BASIC_MEDIA_DF[["geo", "media_1", "media_spend_1"]],
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1"],
              ["media_spend_1"],
              ["media_channel_1"],
              "media_time",
          ),
          missing_cols="['media_1', 'media_time']",
      ),
      dict(
          testcase_name="reach",
          df=BASIC_REACH_DF[["geo", "reach_1", "frequency_1", "rf_spend_1"]],
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1"],
              ["frequency_1"],
              ["rf_spend_1"],
              ["rf_channel_1"],
              "media_time",
          ),
          missing_cols="['reach_1', 'frequency_1', 'rf_spend_1', 'media_time']",
      ),
  )
  def test_with_missing_time_column(self, df, setter, missing_cols):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"DataFrame is missing one or more columns from {missing_cols}",
    ):
      setter(
          data_frame_input_data_builder.DataFrameInputDataBuilder(
              kpi_type=constants.NON_REVENUE,
              default_media_time_column="media_time",
          ),
          df,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="kpi",
          df=DUPE_TIME_KPI_DF,
          setter=lambda builder, df: builder.with_kpi(df),
      ),
      dict(
          testcase_name="controls",
          df=DUPE_TIME_CONTROLS_DF,
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["control_1"]
          ),
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=DUPE_TIME_REVENUE_PER_KPI_DF,
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
      ),
      dict(
          testcase_name="non_media_treatments",
          df=DUPE_TIME_NON_MEDIA_TREATMENTS_DF,
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1"]
          ),
      ),
      dict(
          testcase_name="organic_media",
          df=DUPE_MEDIA_TIME_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1"]
          ),
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=DUPE_MEDIA_TIME_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1"],
              ["organic_frequency_1"],
              ["organic_rf_channel_1"],
          ),
      ),
      dict(
          testcase_name="media",
          df=DUPE_MEDIA_TIME_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1"],
              ["media_spend_1"],
              ["media_channel_1"],
              "media_time",
          ),
      ),
      dict(
          testcase_name="reach",
          df=DUPE_MEDIA_TIME_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1"],
              ["frequency_1"],
              ["rf_spend_1"],
              ["rf_channel_1"],
              "media_time",
          ),
      ),
  )
  def test_with_duplicate_times(self, df, setter):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Duplicate entries found in the 'time' column.",
    ):
      setter(
          data_frame_input_data_builder.DataFrameInputDataBuilder(
              kpi_type=constants.NON_REVENUE,
              default_media_time_column="media_time",
          ),
          df,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="kpi",
          df=INCONSISTENT_TIME_KPI_DF,
          setter=lambda builder, df: builder.with_kpi(df),
      ),
      dict(
          testcase_name="controls",
          df=INCONSISTENT_TIME_CONTROLS_DF,
          setter=lambda builder, df: builder.with_controls(
              df, control_cols=["control_1"]
          ),
      ),
      dict(
          testcase_name="revenue_per_kpi",
          df=INCONSISTENT_TIME_REVENUE_PER_KPI_DF,
          setter=lambda builder, df: builder.with_revenue_per_kpi(df),
      ),
      dict(
          testcase_name="non_media_treatments",
          df=INCONSISTENT_TIME_NON_MEDIA_TREATMENTS_DF,
          setter=lambda builder, df: builder.with_non_media_treatments(
              df, ["non_media_channel_1"]
          ),
      ),
      dict(
          testcase_name="organic_media",
          df=INCONSISTENT_MEDIA_TIME_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df, ["organic_media_1"]
          ),
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=INCONSISTENT_MEDIA_TIME_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1"],
              ["organic_frequency_1"],
              ["organic_rf_channel_1"],
          ),
      ),
      dict(
          testcase_name="media",
          df=INCONSISTENT_MEDIA_TIME_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1"],
              ["media_spend_1"],
              ["media_channel_1"],
              "media_time",
          ),
      ),
      dict(
          testcase_name="reach",
          df=INCONSISTENT_MEDIA_TIME_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1"],
              ["frequency_1"],
              ["rf_spend_1"],
              ["rf_channel_1"],
              "media_time",
          ),
      ),
  )
  def test_with_inconsistent_times_across_geos(self, df, setter):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Values in the 'time' column not consistent across different geos.",
    ):
      setter(
          data_frame_input_data_builder.DataFrameInputDataBuilder(
              kpi_type=constants.NON_REVENUE,
              default_media_time_column="media_time",
          ),
          df,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_media",
          df=BASIC_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df,
              ["organic_media_1"],
              ["organic_media_channel_1", "organic_media_channel_2"],
          ),
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=BASIC_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1"],
              ["organic_frequency_1"],
              ["organic_rf_channel_1", "organic_rf_channel_2"],
          ),
      ),
      dict(
          testcase_name="media",
          df=BASIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1"],
              ["media_spend_1"],
              ["media_channel_1", "media_channel_2"],
              "media_time",
          ),
      ),
      dict(
          testcase_name="reach",
          df=BASIC_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1"],
              ["frequency_1"],
              ["rf_spend_1"],
              ["rf_channel_1", "rf_channel_2"],
              "media_time",
          ),
      ),
  )
  def test_with_shared_components_inconsistent_cols_length(self, df, setter):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Given channel columns must have same length as channel names.",
    ):
      builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
          kpi_type=constants.REVENUE,
          default_media_time_column="media_time",
      )
      setter(builder, df)

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_media",
          df=BASIC_ORGANIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_organic_media(
              df,
              ["organic_media_1", "organic_media_2"],
              ["organic_media_channel_1", "organic_media_channel_1"],
          ),
      ),
      dict(
          testcase_name="organic_reach_frequency",
          df=BASIC_ORGANIC_REACH_DF,
          setter=lambda builder, df: builder.with_organic_reach(
              df,
              ["organic_reach_1", "organic_reach_2"],
              ["organic_frequency_1", "organic_frequency_2"],
              ["organic_rf_channel_1", "organic_rf_channel_1"],
          ),
      ),
      dict(
          testcase_name="media",
          df=BASIC_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_1", "media_2"],
              ["media_spend_1", "media_spend_2"],
              ["media_channel_1", "media_channel_1"],
              "media_time",
          ),
      ),
      dict(
          testcase_name="reach",
          df=BASIC_REACH_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_1", "reach_2"],
              ["frequency_1", "frequency_2"],
              ["rf_spend_1", "rf_spend_2"],
              ["rf_channel_1", "rf_channel_1"],
              "media_time",
          ),
      ),
  )
  def test_with_shared_components_dupe_channels(self, df, setter):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Channel names must be unique.",
    ):
      builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
          kpi_type=constants.REVENUE,
          default_media_time_column="media_time",
      )
      setter(builder, df)

  @parameterized.named_parameters(
      dict(
          testcase_name="media",
          df=LAGGED_MEDIA_DF,
          setter=lambda builder, df: builder.with_media(
              df,
              ["media_0", "media_1", "media_2"],
              ["media_spend_0", "media_spend_1", "media_spend_2"],
              ["media_channel_0", "media_channel_1", "media_channel_2"],
          ),
          getter=lambda builder: [
              builder.media.coords[constants.MEDIA_TIME].values.tolist(),
              builder.media_spend.coords[constants.TIME].values.tolist(),
          ],
          expected_n_times=[203, 200],
      ),
      dict(
          testcase_name="reach",
          df=LAGGED_RF_DF,
          setter=lambda builder, df: builder.with_reach(
              df,
              ["reach_0", "reach_1"],
              ["frequency_0", "frequency_1"],
              ["rf_spend_0", "rf_spend_1"],
              ["rf_channel_0", "rf_channel_1"],
          ),
          getter=lambda builder: [
              builder.reach.coords[constants.MEDIA_TIME].values.tolist(),
              builder.frequency.coords[constants.MEDIA_TIME].values.tolist(),
              builder.rf_spend.coords[constants.TIME].values.tolist(),
          ],
          expected_n_times=[203, 203, 200],
      ),
  )
  def test_time_and_media_time_same_column(
      self, df, setter, getter, expected_n_times
  ):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    setter(builder, df)
    for i, times in enumerate(getter(builder)):
      self.assertLen(times, expected_n_times[i])

  def test_kpi_type_revenue_and_call_revenue_per_kpi(self):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    ).with_revenue_per_kpi(self.BASIC_REVENUE_PER_KPI_DF)

    xr.testing.assert_equal(
        builder.revenue_per_kpi,
        xr.DataArray(
            [[1, 1], [1, 1], [1, 1]],
            dims=[constants.GEO, constants.TIME],
            coords={
                constants.GEO: ["A", "B", "C"],
                constants.TIME: ["2024-01-01", "2024-01-02"],
            },
            name=constants.REVENUE_PER_KPI,
        ),
    )

  def test_dataframe_data_loader_not_continuous_na_period_fails(self):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE
    )

    df = pd.DataFrame({
        "geo": ["0", "0", "0", "0"],
        "time": ["2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14"],
        "media_0": [1, 1, 1, 1],
        "media_1": [1, 1, 1, 1],
        "media_2": [1, 1, 1, 1],
        "reach_0": [1, 1, 1, 1],
        "reach_1": [1, 1, 1, 1],
        "frequency_0": [1, 1, 1, 1],
        "frequency_1": [1, 1, 1, 1],
        "kpi": [None, 1, None, 1],
        "revenue_per_kpi": [None, 1, None, 1],
        "population": [None, 2, None, 1],
        "control_0": [None, 1, None, 1],
        "control_1": [None, 1, None, 1],
        "media_spend_0": [None, 1, None, 1],
        "media_spend_1": [None, 1, None, 1],
        "media_spend_2": [None, 1, None, 1],
        "rf_spend_0": [None, 2, None, 1],
        "rf_spend_1": [None, 7, None, 1],
    })

    builder = (
        builder.with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(df, control_cols=["control_0", "control_1"])
    )

    expected_error_message = (
        "The 'lagged media' period (period with 100% NA values in all"
        " non-media columns) ['2021-01-11' '2021-01-13'] is"
        " not a continuous window starting from the earliest time period."
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      builder.with_media(
          df,
          ["media_0", "media_1", "media_2"],
          ["media_spend_0", "media_spend_1", "media_spend_2"],
          ["media_channel_0", "media_channel_1", "media_channel_2"],
          "time",
      )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error_message,
    ):
      builder.with_reach(
          df,
          ["reach_0", "reach_1"],
          ["frequency_0", "frequency_1"],
          ["rf_spend_0", "rf_spend_1"],
          ["rf_channel_0", "rf_channel_1"],
      )

  def test_df_load_geos_normalized(self):
    df = pd.DataFrame({
        "geo": [111, 111, 222, 222],
        "time": ["2021-01-11", "2021-01-12", "2021-01-11", "2021-01-12"],
        "media_0": [1, 1, 1, 1],
        "media_1": [1, 1, 1, 1],
        "media_2": [1, 1, 1, 1],
        "reach_0": [1, 1, 1, 1],
        "reach_1": [1, 1, 1, 1],
        "frequency_0": [1, 1, 1, 1],
        "frequency_1": [1, 1, 1, 1],
        "revenue": [1, 1, 1, 1],
        "population": [2, 2, 1, 1],
        "media_spend_0": [1, 1, 1, 1],
        "media_spend_1": [1, 1, 1, 1],
        "media_spend_2": [1, 1, 1, 1],
        "rf_spend_0": [2, 2, 1, 1],
        "rf_spend_1": [1, 7, 7, 1],
    })

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )

    data = (
        builder.with_kpi(df, kpi_col="revenue")
        .with_population(df)
        .with_media(
            df,
            ["media_0", "media_1", "media_2"],
            ["media_spend_0", "media_spend_1", "media_spend_2"],
            ["media_channel_0", "media_channel_1", "media_channel_2"],
        )
        .with_reach(
            df,
            ["reach_0", "reach_1"],
            ["frequency_0", "frequency_1"],
            ["rf_spend_0", "rf_spend_1"],
            ["rf_channel_0", "rf_channel_1"],
        )
        .build()
    )

    self.assertEqual(data.geo.values.tolist(), ["111", "222"])

  def test_with_controls_empty_control_cols_list(self):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertWarnsRegex(UserWarning, "Not adding controls data"):
      builder.with_controls(self.BASIC_CONTROLS_DF, control_cols=[])
    self.assertIsNone(builder.controls)

  def test_with_non_media_treatments_empty_non_media_treatment_cols_list(self):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertWarnsRegex(
        UserWarning,
        "Not adding non-media treatments data.",
    ):
      builder.with_non_media_treatments(
          self.BASIC_NON_MEDIA_TREATMENTS_DF, non_media_treatment_cols=[]
      )
    self.assertIsNone(builder.non_media_treatments)

  def test_with_organic_media_empty_organic_media_cols_list(self):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertRaisesRegex(
        ValueError, "`organic_media_cols` must not be empty."
    ):
      builder.with_organic_media(
          self.BASIC_ORGANIC_MEDIA_DF, organic_media_cols=[]
      )
    self.assertIsNone(builder.organic_media)

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_reach_cols_empty",
          with_organic_reach_kwargs={
              "organic_reach_cols": [],
              "organic_frequency_cols": ["organic_frequency_1"],
              "organic_rf_channels": ["organic_rf_channel_1"],
          },
      ),
      dict(
          testcase_name="organic_frequency_cols_empty",
          with_organic_reach_kwargs={
              "organic_reach_cols": ["organic_reach_1"],
              "organic_frequency_cols": [],
              "organic_rf_channels": ["organic_rf_channel_1"],
          },
      ),
      dict(
          testcase_name="organic_rf_channels_empty",
          with_organic_reach_kwargs={
              "organic_reach_cols": ["organic_reach_1"],
              "organic_frequency_cols": ["organic_frequency_1"],
              "organic_rf_channels": [],
          },
      ),
      dict(
          testcase_name="all_empty",
          with_organic_reach_kwargs={
              "organic_reach_cols": [],
              "organic_frequency_cols": [],
              "organic_rf_channels": [],
          },
      ),
  )
  def test_with_organic_reach_empty_organic_reach_cols_and_frequency_cols(
      self, with_organic_reach_kwargs
  ):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertRaisesRegex(
        ValueError,
        "`organic_reach_cols`, `organic_frequency_cols`, and "
        "`organic_rf_channels` must not be empty.",
    ):
      builder.with_organic_reach(
          self.BASIC_ORGANIC_REACH_DF,
          **with_organic_reach_kwargs,
      )
    self.assertIsNone(builder.organic_reach)
    self.assertIsNone(builder.organic_frequency)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_cols_empty",
          with_media_kwargs={
              "media_cols": [],
              "media_spend_cols": ["media_spend_1"],
              "media_channels": ["media_channel_1"],
          },
      ),
      dict(
          testcase_name="media_spend_cols_empty",
          with_media_kwargs={
              "media_cols": ["media_1"],
              "media_spend_cols": [],
              "media_channels": ["media_channel_1"],
          },
      ),
      dict(
          testcase_name="media_channels_empty",
          with_media_kwargs={
              "media_cols": ["media_1"],
              "media_spend_cols": ["media_spend_1"],
              "media_channels": [],
          },
      ),
      dict(
          testcase_name="all_empty",
          with_media_kwargs={
              "media_cols": [],
              "media_spend_cols": [],
              "media_channels": [],
          },
      ),
  )
  def test_with_media_empty_media_cols_list(self, with_media_kwargs):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertRaisesRegex(
        ValueError,
        "`media_cols`, `media_spend_cols`, and `media_channels` must not be"
        " empty.",
    ):
      builder.with_media(
          self.BASIC_MEDIA_DF,
          **with_media_kwargs,
      )
    self.assertIsNone(builder.media)
    self.assertIsNone(builder.media_spend)

  @parameterized.named_parameters(
      dict(
          testcase_name="reach_cols_empty",
          with_reach_kwargs={
              "reach_cols": [],
              "frequency_cols": ["frequency_1"],
              "rf_spend_cols": ["rf_spend_1"],
              "rf_channels": ["rf_channel_1"],
          },
      ),
      dict(
          testcase_name="frequency_cols_empty",
          with_reach_kwargs={
              "reach_cols": ["reach_1"],
              "frequency_cols": [],
              "rf_spend_cols": ["rf_spend_1"],
              "rf_channels": ["rf_channel_1"],
          },
      ),
      dict(
          testcase_name="rf_spend_cols_empty",
          with_reach_kwargs={
              "reach_cols": ["reach_1"],
              "frequency_cols": ["frequency_1"],
              "rf_spend_cols": [],
              "rf_channels": ["rf_channel_1"],
          },
      ),
      dict(
          testcase_name="rf_channels_empty",
          with_reach_kwargs={
              "reach_cols": ["reach_1"],
              "frequency_cols": ["frequency_1"],
              "rf_spend_cols": ["rf_spend_1"],
              "rf_channels": [],
          },
      ),
      dict(
          testcase_name="all_empty",
          with_reach_kwargs={
              "reach_cols": [],
              "frequency_cols": [],
              "rf_spend_cols": [],
              "rf_channels": [],
          },
      ),
  )
  def test_with_reach_empty_reach_cols_list(self, with_reach_kwargs):
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE
    )
    with self.assertRaisesRegex(
        ValueError,
        "`reach_cols`, `frequency_cols`, `rf_spend_cols`, and `rf_channels`"
        " must not be empty.",
    ):
      builder.with_reach(
          self.BASIC_REACH_DF,
          **with_reach_kwargs,
      )
    self.assertIsNone(builder.reach)
    self.assertIsNone(builder.frequency)
    self.assertIsNone(builder.rf_spend)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_geo_column",
          property_name="default_geo_column",
          expected_value=constants.GEO,
      ),
      dict(
          testcase_name="default_time_column",
          property_name="default_time_column",
          expected_value=constants.TIME,
      ),
      dict(
          testcase_name="default_media_time_column",
          property_name="default_media_time_column",
          expected_value=constants.TIME,
      ),
      dict(
          testcase_name="default_population_column",
          property_name="default_population_column",
          expected_value=constants.POPULATION,
      ),
      dict(
          testcase_name="default_kpi_column",
          property_name="default_kpi_column",
          expected_value=constants.KPI,
      ),
      dict(
          testcase_name="default_revenue_per_kpi_column",
          property_name="default_revenue_per_kpi_column",
          expected_value=constants.REVENUE_PER_KPI,
      ),
      dict(
          testcase_name="override_default_geo_column",
          property_name="default_geo_column",
          expected_value="my_geo",
          constructor_kwargs={"default_geo_column": "my_geo"},
      ),
      dict(
          testcase_name="override_default_time_column",
          property_name="default_time_column",
          expected_value="my_time",
          constructor_kwargs={"default_time_column": "my_time"},
      ),
      dict(
          testcase_name="override_default_media_time_column",
          property_name="default_media_time_column",
          expected_value="my_media_time",
          constructor_kwargs={"default_media_time_column": "my_media_time"},
      ),
      dict(
          testcase_name="override_default_population_column",
          property_name="default_population_column",
          expected_value="my_population",
          constructor_kwargs={"default_population_column": "my_population"},
      ),
      dict(
          testcase_name="override_default_kpi_column",
          property_name="default_kpi_column",
          expected_value="my_kpi",
          constructor_kwargs={"default_kpi_column": "my_kpi"},
      ),
      dict(
          testcase_name="override_default_revenue_per_kpi_column",
          property_name="default_revenue_per_kpi_column",
          expected_value="my_revenue_per_kpi",
          constructor_kwargs={
              "default_revenue_per_kpi_column": "my_revenue_per_kpi"
          },
      ),
  )
  def test_default_column_names(
      self, property_name, expected_value, constructor_kwargs=None
  ):
    constructor_kwargs = constructor_kwargs or {}
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI, **constructor_kwargs
    )
    self.assertEqual(getattr(builder, property_name), expected_value)

  def test_with_kpi_custom_columns(self):
    data = self.BASIC_KPI_DF.rename(
        columns={"kpi": "my_kpi", "time": "my_time", "geo": "my_geo"}
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_kpi(data, "my_kpi", time_col="my_time", geo_col="my_geo")
    self.assertIsNotNone(builder.kpi)

  def test_with_controls_custom_columns(self):
    data = self.BASIC_CONTROLS_DF.rename(
        columns={"time": "my_time", "geo": "my_geo"}
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_controls(data, ["control_1"], time_col="my_time", geo_col="my_geo")
    self.assertIsNotNone(builder.controls)

  def test_with_population_custom_columns(self):
    data = self.BASIC_POPULATION_DF.rename(
        columns={"population": "my_population", "geo": "my_geo"}
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_population(data, population_col="my_population", geo_col="my_geo")
    self.assertIsNotNone(builder.population)

  def test_with_revenue_per_kpi_custom_columns(self):
    data = self.BASIC_REVENUE_PER_KPI_DF.rename(
        columns={
            "revenue_per_kpi": "my_revenue_per_kpi",
            "geo": "my_geo",
            "time": "my_time",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_revenue_per_kpi(
        data,
        revenue_per_kpi_col="my_revenue_per_kpi",
        geo_col="my_geo",
        time_col="my_time",
    )
    self.assertIsNotNone(builder.revenue_per_kpi)

  def test_with_media_custom_columns(self):
    data = self.BASIC_MEDIA_DF.rename(
        columns={
            "media_time": "my_time",
            "geo": "my_geo",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_media(
        data,
        ["media_1"],
        ["media_spend_1"],
        ["media_channel_1"],
        time_col="my_time",
        geo_col="my_geo",
    )
    self.assertIsNotNone(builder.media)
    self.assertIsNotNone(builder.media_spend)

  def test_with_reach_custom_columns(self):
    data = self.BASIC_REACH_DF.rename(
        columns={
            "media_time": "my_time",
            "geo": "my_geo",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_reach(
        data,
        ["reach_1"],
        ["frequency_1"],
        ["rf_spend_1"],
        ["rf_channel_1"],
        time_col="my_time",
        geo_col="my_geo",
    )
    self.assertIsNotNone(builder.reach)
    self.assertIsNotNone(builder.frequency)
    self.assertIsNotNone(builder.rf_spend)

  def test_with_organic_media_custom_columns(self):
    data = self.BASIC_ORGANIC_MEDIA_DF.rename(
        columns={
            "media_time": "my_time",
            "geo": "my_geo",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_organic_media(
        data,
        ["organic_media_1"],
        media_time_col="my_time",
        geo_col="my_geo",
    )
    self.assertIsNotNone(builder.organic_media)

  def test_with_organic_reach_custom_columns(self):
    data = self.BASIC_ORGANIC_REACH_DF.rename(
        columns={
            "media_time": "my_time",
            "geo": "my_geo",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_organic_reach(
        data,
        ["organic_reach_1"],
        ["organic_frequency_1"],
        ["organic_rf_channel_1"],
        media_time_col="my_time",
        geo_col="my_geo",
    )
    self.assertIsNotNone(builder.organic_reach)
    self.assertIsNotNone(builder.organic_frequency)

  def test_with_non_media_treatments_custom_columns(self):
    data = self.BASIC_NON_MEDIA_TREATMENTS_DF.rename(
        columns={
            "time": "my_time",
            "geo": "my_geo",
        }
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.KPI
    ).with_non_media_treatments(
        data,
        ["non_media_channel_1"],
        time_col="my_time",
        geo_col="my_geo",
    )
    self.assertIsNotNone(builder.non_media_treatments)


if __name__ == "__main__":
  absltest.main()
