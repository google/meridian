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

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import datetime
import os
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import load
from meridian.data import test_utils
import numpy as np
import pandas as pd
import xarray as xr


_DATETIME = datetime.datetime
_UNIT_TEST_DATA_DIR_NAME = 'unit_testing_data'


class InputDataLoaderTest(parameterized.TestCase):
  _N_GEOS = 10
  _N_TIMES = 50
  _N_MEDIA_TIMES = 53
  _N_CONTROLS = 2
  _N_MEDIA_CHANNELS = 3
  _N_RF_CHANNELS = 2

  SAMPLE_DATASET_NO_MEDIA_TIME = test_utils.random_dataset(
      n_geos=50,
      n_times=200,
      n_media_times=203,
      n_media_channels=10,
      n_rf_channels=2,
      n_controls=5,
      remove_media_time=True,
  )

  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_KPI = copy.deepcopy(
      SAMPLE_DATASET_NO_MEDIA_TIME
  )
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_KPI['kpi'].loc[
      {'geo': 'geo_1', 'time': '2024-11-11'}
  ] = None
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_CONTROLS = copy.deepcopy(
      SAMPLE_DATASET_NO_MEDIA_TIME
  )
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_CONTROLS['controls'].loc[
      {'geo': 'geo_1', 'time': '2024-11-11', 'control_variable': 'control_0'}
  ] = None
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_REVENUE_PER_KPI = copy.deepcopy(
      SAMPLE_DATASET_NO_MEDIA_TIME
  )
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_REVENUE_PER_KPI['revenue_per_kpi'].loc[
      {'geo': 'geo_1', 'time': '2024-11-11'}
  ] = None
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_MEDIA_SPEND = copy.deepcopy(
      SAMPLE_DATASET_NO_MEDIA_TIME
  )
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_MEDIA_SPEND['media_spend'].loc[
      {'geo': 'geo_1', 'time': '2024-11-11', 'media_channel': 'ch_2'}
  ] = None
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_RF_SPEND = copy.deepcopy(
      SAMPLE_DATASET_NO_MEDIA_TIME
  )
  SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_RF_SPEND['rf_spend'].loc[
      {'geo': 'geo_1', 'time': '2024-11-11', 'rf_channel': 'rf_ch_1'}
  ] = None

  def setUp(self):
    super().setUp()

    self._correct_coord_to_columns_media_only = (
        test_utils.sample_coord_to_columns(
            n_controls=self._N_CONTROLS, n_media_channels=self._N_MEDIA_CHANNELS
        )
    )

    self._correct_coord_to_columns_rf_only = test_utils.sample_coord_to_columns(
        n_controls=self._N_CONTROLS, n_rf_channels=self._N_RF_CHANNELS
    )

    self._correct_coord_to_columns_media_and_rf = (
        test_utils.sample_coord_to_columns(
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
        )
    )

    self._correct_media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(self._N_MEDIA_CHANNELS)
    }
    self._correct_media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(self._N_MEDIA_CHANNELS)
    }
    self._correct_reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }
    self._correct_frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }
    self._correct_rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }

    self._coord_to_columns_wrong_value = dataclasses.replace(
        self._correct_coord_to_columns_media_only, kpi='revenue'
    )
    self._coord_to_columns_extra_control = dataclasses.replace(
        self._correct_coord_to_columns_media_only,
        controls=test_utils._sample_names('control_', self._N_CONTROLS + 1),
    )
    self._coord_to_columns_missing_media = dataclasses.replace(
        self._correct_coord_to_columns_media_only,
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
    )

    self._sample_df_with_media_only = test_utils.random_dataframe(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
    )
    self._sample_df_with_media_and_rf = test_utils.random_dataframe(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )

    sample_df_duplicate_time = copy.deepcopy(self._sample_df_with_media_and_rf)
    sample_df_duplicate_time.loc[:, constants.TIME].replace(
        {'2021-02-08': '2021-02-22'}, inplace=True
    )

    sample_df_not_matching_times = copy.deepcopy(
        self._sample_df_with_media_and_rf
    )
    sample_df_not_matching_times.loc[3, constants.TIME] = '2021-02-16'

    sample_df_not_matching_times_with_int_geo = copy.deepcopy(
        sample_df_not_matching_times
    )
    # Use DMA numbers as geos.
    sample_df_not_matching_times_with_int_geo['geo'] = (
        sample_df_not_matching_times['geo'].str.replace('geo_', '').astype(int)
        + 500
    )

    sample_lagged_df_with_media_and_rf = test_utils.random_dataframe(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_MEDIA_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )

    sample_df_na_in_media = copy.deepcopy(sample_lagged_df_with_media_and_rf)
    sample_df_na_in_media.loc[1, 'media_2'] = None

    sample_df_na_in_reach = copy.deepcopy(sample_lagged_df_with_media_and_rf)
    sample_df_na_in_reach.loc[1, 'reach_1'] = None

    sample_df_na_in_frequency = copy.deepcopy(
        sample_lagged_df_with_media_and_rf
    )
    sample_df_na_in_frequency.loc[1, 'frequency_1'] = None

    sample_df_non_na_in_lagged = copy.deepcopy(
        sample_lagged_df_with_media_and_rf
    )
    sample_df_non_na_in_lagged.loc[2, 'media_spend_1'] = 3.14

    sample_df_na_outside_lagged = copy.deepcopy(
        sample_lagged_df_with_media_and_rf
    )
    sample_df_na_outside_lagged.loc[4, constants.KPI] = None

    self._sample_df_not_continuous_na_period = copy.deepcopy(
        sample_lagged_df_with_media_and_rf
    )
    self._sample_df_not_continuous_na_period.loc[:, constants.TIME].replace(
        {'2021-01-11': '2021-02-01', '2021-02-01': '2021-01-11'}, inplace=True
    )
    self._wrong_coord_test_parameters = {
        'wrong_value': self._coord_to_columns_wrong_value,
        'extra_control': self._coord_to_columns_extra_control,
    }
    self._geo_time_test_parameters = {
        'duplicate_time': sample_df_duplicate_time,
        'not_matching_times': sample_df_not_matching_times,
        'not_matching_times_with_int_geo': (
            sample_df_not_matching_times_with_int_geo
        ),
    }
    self.lagged_media_test_parameters = {
        'NA_in_media': sample_df_na_in_media,
        'NA_in_reach': sample_df_na_in_reach,
        'NA_in_frequency': sample_df_na_in_frequency,
        'non_NA_in_lagged_period': sample_df_non_na_in_lagged,
        'NA_outside_lagged_period': sample_df_na_outside_lagged,
    }

  def test_xr_dataset_data_loader_loads_random_dataset(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=200,
        n_media_channels=10,
        n_controls=5,
    )
    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])

  def test_xr_dataset_data_loader_loads_lagged_dataset_without_media_time(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=203,
        n_media_channels=10,
        n_rf_channels=2,
        n_controls=5,
        remove_media_time=True,
    )

    self.assertNotIn(constants.MEDIA_TIME, dataset.coords.keys())

    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)
    data = loader.load()

    xr.testing.assert_equal(
        data.kpi, dataset[constants.KPI].dropna(dim=constants.TIME)
    )
    xr.testing.assert_equal(
        data.revenue_per_kpi,
        dataset[constants.REVENUE_PER_KPI].dropna(dim=constants.TIME),
    )
    xr.testing.assert_equal(
        data.controls, dataset[constants.CONTROLS].dropna(dim=constants.TIME)
    )
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(
        data.media,
        dataset[constants.MEDIA].rename({constants.TIME: constants.MEDIA_TIME}),
    )
    xr.testing.assert_equal(
        data.media_spend,
        dataset[constants.MEDIA_SPEND].dropna(dim=constants.TIME),
    )
    xr.testing.assert_equal(
        data.reach,
        dataset[constants.REACH].rename({constants.TIME: constants.MEDIA_TIME}),
    )
    xr.testing.assert_equal(
        data.frequency,
        dataset[constants.FREQUENCY].rename(
            {constants.TIME: constants.MEDIA_TIME}
        ),
    )
    xr.testing.assert_equal(
        data.rf_spend, dataset[constants.RF_SPEND].dropna(dim=constants.TIME)
    )

  def test_xr_dataset_data_loader_loads_not_lagged_dataset_without_media_time(
      self,
  ):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=200,
        n_media_channels=10,
        n_rf_channels=2,
        n_controls=5,
        remove_media_time=True,
    )

    self.assertNotIn(constants.MEDIA_TIME, dataset.coords.keys())

    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)
    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi,
        dataset[constants.REVENUE_PER_KPI],
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(
        data.media,
        dataset[constants.MEDIA].rename({constants.TIME: constants.MEDIA_TIME}),
    )
    xr.testing.assert_equal(
        data.media_spend,
        dataset[constants.MEDIA_SPEND],
    )
    xr.testing.assert_equal(
        data.reach,
        dataset[constants.REACH].rename({constants.TIME: constants.MEDIA_TIME}),
    )
    xr.testing.assert_equal(
        data.frequency,
        dataset[constants.FREQUENCY].rename(
            {constants.TIME: constants.MEDIA_TIME}
        ),
    )
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])

  def test_xr_dataset_data_loader_dataset_with_datetime_coords(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=200,
        n_media_channels=10,
        n_controls=5,
    )

    datetime_values = [
        _DATETIME.strptime(time, constants.DATE_FORMAT)
        for time in dataset.coords[constants.TIME].values
    ]
    dataset_with_datetime_values = dataset.assign_coords({
        constants.TIME: datetime_values,
        constants.MEDIA_TIME: datetime_values,
    })

    loader = load.XrDatasetDataLoader(
        dataset_with_datetime_values, kpi_type=constants.NON_REVENUE
    )
    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])

  def test_xr_dataset_data_loader_dataset_with_invalid_time_value(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=200,
        n_media_channels=10,
        n_controls=5,
    )

    time_values = dataset.coords[constants.MEDIA_TIME].values
    time_values[-5] = '2023-W4'
    dataset = dataset.assign_coords({
        constants.MEDIA_TIME: time_values,
    })

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid time label: '2023-W4'. Expected format:"
        f" '{constants.DATE_FORMAT}'",
    ):
      _ = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_population_with_geo',
          dataset=test_utils.NATIONAL_DATASET_W_POPULATION_W_GEO,
          has_population=True,
      ),
      dict(
          testcase_name='with_single_population_without_geo',
          dataset=test_utils.NATIONAL_DATASET_W_SINGLE_POPULATION_WO_GEO,
          has_population=True,
      ),
      dict(
          testcase_name='with_scalar_population_without_geo',
          dataset=test_utils.NATIONAL_DATASET_W_SCALAR_POPULATION_WO_GEO,
          has_population=True,
      ),
      dict(
          testcase_name='with_none_population_without_geo',
          dataset=test_utils.NATIONAL_DATASET_W_NONE_POPULATION_WO_GEO,
          has_population=True,
      ),
      dict(
          testcase_name='without_population_with_geo',
          dataset=test_utils.NATIONAL_DATASET_WO_POPULATION_W_GEO,
          has_population=False,
      ),
      dict(
          testcase_name='without_population_without_geo',
          dataset=test_utils.NATIONAL_DATASET_WO_POPULATION_WO_GEO,
          has_population=False,
      ),
  )
  def test_xr_dataset_national_data_loader_loads_dataset(
      self, dataset: xr.Dataset, has_population: bool
  ):
    expected_dataset = test_utils.EXPECTED_NATIONAL_DATASET

    with warnings.catch_warnings(record=True) as warning_list:
      # Cause all warnings to always be triggered.
      warnings.simplefilter('always')
      loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)
      if has_population:
        self.assertTrue(
            any(
                issubclass(warning.category, UserWarning)
                and (
                    str(warning.message)
                    == 'The `population` argument is ignored in a nationally'
                    ' aggregated model. It will be reset to [1]'
                )
                for warning in warning_list
            )
        )
    data = loader.load()

    xr.testing.assert_equal(data.kpi, expected_dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, expected_dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(
        data.population, expected_dataset[constants.POPULATION]
    )
    xr.testing.assert_equal(data.controls, expected_dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.media, expected_dataset[constants.MEDIA])
    xr.testing.assert_equal(
        data.media_spend, expected_dataset[constants.MEDIA_SPEND]
    )

  def test_xr_dataset_data_loader_loads_random_dataset_lagged_media(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=203,
        n_media_channels=10,
        n_controls=5,
    )
    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    self.assertIsNone(data.reach)
    self.assertIsNone(data.frequency)
    self.assertIsNone(data.rf_spend)

  def test_xr_dataset_data_loader_loads_random_dataset_with_media_and_rf(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=203,
        n_media_channels=10,
        n_rf_channels=2,
        n_controls=5,
    )
    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])

  def test_xr_dataset_data_loader_loads_random_dataset_with_rf_only(self):
    dataset = test_utils.random_dataset(
        n_geos=50,
        n_times=200,
        n_media_times=203,
        n_rf_channels=2,
        n_controls=5,
    )
    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])
    self.assertIsNone(data.media)
    self.assertIsNone(data.media_spend)

  def test_xr_dataset_data_loader_wrong_names_no_mapping_fails(self):
    dataset = test_utils.random_dataset(
        n_geos=10,
        n_times=52,
        n_media_times=52,
        n_media_channels=3,
        n_controls=1,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Array 'kpi' not found in dataset's arrays. Please use the"
        " 'name_mapping' argument to rename the arrays.",
    ):
      _ = load.XrDatasetDataLoader(
          dataset.rename({constants.KPI: 'conversions'}),
          kpi_type=constants.NON_REVENUE,
      )

  def test_xr_dataset_data_loader_name_mapping_works(self):
    dataset = test_utils.random_dataset(
        n_geos=20,
        n_times=100,
        n_media_times=103,
        n_media_channels=4,
        n_controls=2,
    )

    loader = load.XrDatasetDataLoader(
        dataset.rename({constants.KPI: 'conversions', constants.GEO: 'group'}),
        kpi_type=constants.NON_REVENUE,
        name_mapping={'conversions': constants.KPI, 'group': constants.GEO},
    )

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])

  def test_xr_dataset_data_loader_no_revenue_per_kpi_name_mapping(self):
    dataset = test_utils.random_dataset(
        n_geos=20,
        n_times=100,
        n_media_times=103,
        n_media_channels=4,
        n_controls=2,
        revenue_per_kpi_value=None,
    )
    loader = load.XrDatasetDataLoader(
        dataset.rename({constants.KPI: 'conversions', constants.GEO: 'group'}),
        kpi_type=constants.NON_REVENUE,
        name_mapping={'conversions': constants.KPI, 'group': constants.GEO},
    )
    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    self.assertIsNone(data.revenue_per_kpi)

  @parameterized.named_parameters(
      (
          'wrong_dataset_no_media_no_rf',
          test_utils.WRONG_DATASET_WO_MEDIA_WO_RF,
          (
              "Some required data is missing. Please use the 'name_mapping'"
              ' argument to rename the coordinates/arrays. It is required to'
              ' have at least one of media or reach and frequency.'
          ),
      ),
      (
          'wrong_dataset_partial_media_no_rf',
          test_utils.WRONG_DATASET_PARTIAL_MEDIA_WO_RF,
          (
              "Some required data is missing. Please use the 'name_mapping'"
              ' argument to rename the coordinates/arrays. It is required to'
              ' have at least one of media or reach and frequency.'
          ),
      ),
      (
          'wrong_dataset_no_media_partial_rf',
          test_utils.WRONG_DATASET_WO_MEDIA_PARTIAL_RF,
          (
              "Some required data is missing. Please use the 'name_mapping'"
              ' argument to rename the coordinates/arrays. It is required to'
              ' have at least one of media or reach and frequency.'
          ),
      ),
      (
          'wrong_dataset_partial_media_partial_rf',
          test_utils.WRONG_DATASET_PARTIAL_MEDIA_PARTIAL_RF,
          (
              "Some required data is missing. Please use the 'name_mapping'"
              ' argument to rename the coordinates/arrays. It is required to'
              ' have at least one of media or reach and frequency.'
          ),
      ),
      (
          'wrong_dataset_with_media_partial_rf',
          test_utils.WRONG_DATASET_W_MEDIA_PARTIAL_RF,
          (
              "RF data is partially missing. '['rf_channel', 'frequency']' not"
              " found in dataset's coordinates/arrays. Please use the"
              " 'name_mapping' argument to rename the coordinates/arrays."
          ),
      ),
      (
          'wrong_dataset_partial_media_with_rf',
          test_utils.WRONG_DATASET_PARTIAL_MEDIA_W_RF,
          (
              "Media data is partially missing. '['media_channel',"
              " 'media_spend']' not found in dataset's coordinates/arrays."
              " Please use the 'name_mapping' argument to rename the"
              ' coordinates/arrays.'
          ),
      ),
  )
  def test_xr_dataset_data_loader_missing_data_fails(self, data, error_message):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      _ = load.XrDatasetDataLoader(data, kpi_type=constants.NON_REVENUE)

  @parameterized.named_parameters(
      (
          'kpi',
          SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_KPI,
      ),
      (
          'revenue_per_kpi',
          SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_REVENUE_PER_KPI,
      ),
      (
          'controls',
          SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_CONTROLS,
      ),
      (
          'media_spend',
          SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_MEDIA_SPEND,
      ),
      (
          'rf_spend',
          SAMPLE_DATASET_NO_MEDIA_TIME_NA_IN_RF_SPEND,
      ),
  )
  def test_xr_dataset_data_loader_nas_in_data_fails(self, dataset):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The 'lagged media' period (period with 100% NA values in all non-media"
        " columns) ['2021-01-04' '2021-01-11' '2021-01-18' '2024-11-11'] is not"
        ' a continuous window starting from the earliest time period.',
    ):
      load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

  def test_xr_dataset_data_loader_wrong_name_mapping_fails(self):
    dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_controls=self._N_CONTROLS,
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "Target name 'revenue' from the mapping is none of the target"
            " coordinate names ('geo', 'time', 'media_time',"
            " 'control_variable', 'organic_media_channel',"
            " 'organic_rf_channel', 'non_media_channel', 'media_channel',"
            " 'rf_channel') or array names ('kpi', 'population', 'controls',"
            " 'revenue_per_kpi', 'organic_media', 'organic_reach',"
            " 'organic_frequency', 'non_media_treatments', 'media',"
            " 'media_spend', 'reach', 'frequency', 'rf_spend')."
        ),
    ):
      load.XrDatasetDataLoader(
          dataset.rename({constants.KPI: 'revenue'}),
          kpi_type=constants.NON_REVENUE,
          name_mapping={constants.KPI: 'revenue'},
      )

  def test_dataframe_data_loader_name_mapping_works(self):
    df = self._sample_df_with_media_and_rf
    changed_mapping = {
        constants.KPI: 'Revenue',
        constants.GEO: 'City',
        constants.TIME: 'Date',
        constants.REVENUE_PER_KPI: 'unit_price',
        constants.POPULATION: 'Population',
    }
    df = df.rename(columns=changed_mapping)
    coord_to_columns = load.CoordToColumns(
        kpi='Revenue',
        geo='City',
        time='Date',
        revenue_per_kpi='unit_price',
        population='Population',
        controls=test_utils._sample_names('control_', self._N_CONTROLS),
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
        media_spend=test_utils._sample_names(
            'media_spend_', self._N_MEDIA_CHANNELS
        ),
        reach=test_utils._sample_names('reach_', self._N_RF_CHANNELS),
        frequency=test_utils._sample_names('frequency_', self._N_RF_CHANNELS),
        rf_spend=test_utils._sample_names('rf_spend_', self._N_RF_CHANNELS),
    )
    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
        reach_to_channel=self._correct_reach_to_channel,
        frequency_to_channel=self._correct_frequency_to_channel,
        rf_spend_to_channel=self._correct_rf_spend_to_channel,
    )
    data = loader.load()

    expected_dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )

    xr.testing.assert_equal(data.kpi, expected_dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, expected_dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, expected_dataset[constants.CONTROLS])
    xr.testing.assert_equal(
        data.population, expected_dataset[constants.POPULATION]
    )
    xr.testing.assert_equal(data.media, expected_dataset[constants.MEDIA])
    xr.testing.assert_equal(
        data.media_spend, expected_dataset[constants.MEDIA_SPEND]
    )
    xr.testing.assert_equal(data.reach, expected_dataset[constants.REACH])
    xr.testing.assert_equal(
        data.frequency, expected_dataset[constants.FREQUENCY]
    )
    xr.testing.assert_equal(data.rf_spend, expected_dataset[constants.RF_SPEND])

  def test_dataframe_data_loader_no_controls_works(self):
    df = test_utils.random_dataframe(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=None,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )
    changed_mapping = {
        constants.KPI: 'Revenue',
        constants.GEO: 'City',
        constants.TIME: 'Date',
        constants.REVENUE_PER_KPI: 'unit_price',
        constants.POPULATION: 'Population',
    }
    df = df.rename(columns=changed_mapping)

    coord_to_columns = load.CoordToColumns(
        kpi='Revenue',
        geo='City',
        time='Date',
        revenue_per_kpi='unit_price',
        population='Population',
        controls=None,
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
        media_spend=test_utils._sample_names(
            'media_spend_', self._N_MEDIA_CHANNELS
        ),
        reach=test_utils._sample_names('reach_', self._N_RF_CHANNELS),
        frequency=test_utils._sample_names('frequency_', self._N_RF_CHANNELS),
        rf_spend=test_utils._sample_names('rf_spend_', self._N_RF_CHANNELS),
    )
    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
        reach_to_channel=self._correct_reach_to_channel,
        frequency_to_channel=self._correct_frequency_to_channel,
        rf_spend_to_channel=self._correct_rf_spend_to_channel,
    )
    data = loader.load()

    expected_dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=None,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )

    xr.testing.assert_equal(data.kpi, expected_dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, expected_dataset[constants.REVENUE_PER_KPI]
    )
    self.assertIsNone(data.controls)
    xr.testing.assert_equal(
        data.population, expected_dataset[constants.POPULATION]
    )
    xr.testing.assert_equal(data.media, expected_dataset[constants.MEDIA])
    xr.testing.assert_equal(
        data.media_spend, expected_dataset[constants.MEDIA_SPEND]
    )
    xr.testing.assert_equal(data.reach, expected_dataset[constants.REACH])
    xr.testing.assert_equal(
        data.frequency, expected_dataset[constants.FREQUENCY]
    )
    xr.testing.assert_equal(data.rf_spend, expected_dataset[constants.RF_SPEND])

  def test_dataframe_no_revenue_per_kpi_data_loader_name_mapping_works(self):
    df = self._sample_df_with_media_and_rf.drop(
        columns=[constants.REVENUE_PER_KPI]
    )
    changed_mapping = {
        constants.KPI: 'Revenue',
        constants.GEO: 'City',
        constants.TIME: 'Date',
        constants.POPULATION: 'Population',
    }
    df = df.rename(columns=changed_mapping)
    coord_to_columns = load.CoordToColumns(
        kpi='Revenue',
        geo='City',
        time='Date',
        population='Population',
        controls=test_utils._sample_names('control_', self._N_CONTROLS),
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
        media_spend=test_utils._sample_names(
            'media_spend_', self._N_MEDIA_CHANNELS
        ),
        reach=test_utils._sample_names('reach_', self._N_RF_CHANNELS),
        frequency=test_utils._sample_names('frequency_', self._N_RF_CHANNELS),
        rf_spend=test_utils._sample_names('rf_spend_', self._N_RF_CHANNELS),
    )
    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
        reach_to_channel=self._correct_reach_to_channel,
        frequency_to_channel=self._correct_frequency_to_channel,
        rf_spend_to_channel=self._correct_rf_spend_to_channel,
    )
    data = loader.load()

    expected_dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
        revenue_per_kpi_value=None,
    )
    xr.testing.assert_equal(data.kpi, expected_dataset[constants.KPI])
    xr.testing.assert_equal(data.controls, expected_dataset[constants.CONTROLS])
    xr.testing.assert_equal(
        data.population, expected_dataset[constants.POPULATION]
    )
    xr.testing.assert_equal(data.media, expected_dataset[constants.MEDIA])
    xr.testing.assert_equal(
        data.media_spend, expected_dataset[constants.MEDIA_SPEND]
    )
    xr.testing.assert_equal(data.reach, expected_dataset[constants.REACH])
    xr.testing.assert_equal(
        data.frequency, expected_dataset[constants.FREQUENCY]
    )
    xr.testing.assert_equal(data.rf_spend, expected_dataset[constants.RF_SPEND])
    self.assertIsNone(data.revenue_per_kpi)

  def test_dataframe_data_loader_datetime_values(self):
    df = self._sample_df_with_media_and_rf
    df[constants.TIME] = df[constants.TIME].map(
        lambda time: _DATETIME.strptime(time, constants.DATE_FORMAT)
    )

    coord_to_columns = load.CoordToColumns(
        revenue_per_kpi=constants.REVENUE_PER_KPI,
        controls=test_utils._sample_names('control_', self._N_CONTROLS),
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
        media_spend=test_utils._sample_names(
            'media_spend_', self._N_MEDIA_CHANNELS
        ),
        reach=test_utils._sample_names('reach_', self._N_RF_CHANNELS),
        frequency=test_utils._sample_names('frequency_', self._N_RF_CHANNELS),
        rf_spend=test_utils._sample_names('rf_spend_', self._N_RF_CHANNELS),
    )

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
        reach_to_channel=self._correct_reach_to_channel,
        frequency_to_channel=self._correct_frequency_to_channel,
        rf_spend_to_channel=self._correct_rf_spend_to_channel,
    )
    data = loader.load()

    expected_dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_TIMES,
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
    )

    xr.testing.assert_equal(data.kpi, expected_dataset[constants.KPI])

  def test_dataframe_data_loader_invalid_time_format(self):
    df = self._sample_df_with_media_and_rf.replace('2021-01-25', '2023-W42')

    coord_to_columns = load.CoordToColumns(
        revenue_per_kpi=constants.REVENUE_PER_KPI,
        controls=test_utils._sample_names('control_', self._N_CONTROLS),
        media=test_utils._sample_names('media_', self._N_MEDIA_CHANNELS),
        media_spend=test_utils._sample_names(
            'media_spend_', self._N_MEDIA_CHANNELS
        ),
        reach=test_utils._sample_names('reach_', self._N_RF_CHANNELS),
        frequency=test_utils._sample_names('frequency_', self._N_RF_CHANNELS),
        rf_spend=test_utils._sample_names('rf_spend_', self._N_RF_CHANNELS),
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid time label: '2023-W42'. Expected format: '%Y-%m-%d'",
    ):
      load.DataFrameDataLoader(
          df=df,
          coord_to_columns=coord_to_columns,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
          reach_to_channel=self._correct_reach_to_channel,
          frequency_to_channel=self._correct_frequency_to_channel,
          rf_spend_to_channel=self._correct_rf_spend_to_channel,
      ).load()

  @parameterized.named_parameters(
      ('not_lagged', 50, 200, 200, 10, 5), ('lagged', 50, 200, 203, 10, 5)
  )
  def test_dataframe_data_loader_loads_random_dataset_media_only(
      self,
      n_geos,
      n_times,
      n_media_times,
      n_media_channels,
      n_controls,
  ):
    dataset = test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_controls=n_controls,
    )
    df = test_utils.dataset_to_dataframe(
        dataset,
        controls_column_names=test_utils._sample_names('control_', n_controls),
        media_column_names=test_utils._sample_names('media_', n_media_channels),
        media_spend_column_names=test_utils._sample_names(
            'media_spend_', n_media_channels
        ),
    )
    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=n_controls,
        n_media_channels=n_media_channels,
    )
    media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    self.assertIsNone(data.reach)
    self.assertIsNone(data.frequency)
    self.assertIsNone(data.rf_spend)

  @parameterized.named_parameters(
      ('not_lagged', 50, 200, 200, 2, 5), ('lagged', 50, 200, 203, 2, 5)
  )
  def test_dataframe_data_loader_loads_random_dataset_rf_only(
      self,
      n_geos,
      n_times,
      n_media_times,
      n_rf_channels,
      n_controls,
  ):
    dataset = test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_rf_channels=n_rf_channels,
        n_controls=n_controls,
    )
    df = test_utils.dataset_to_dataframe(
        dataset,
        controls_column_names=test_utils._sample_names('control_', n_controls),
        reach_column_names=test_utils._sample_names('reach_', n_rf_channels),
        frequency_column_names=test_utils._sample_names(
            'frequency_', n_rf_channels
        ),
        rf_spend_column_names=test_utils._sample_names(
            'rf_spend_', n_rf_channels
        ),
    )
    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=n_controls,
        n_rf_channels=n_rf_channels,
    )
    reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
    )

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])
    self.assertIsNone(data.media)
    self.assertIsNone(data.media_spend)

  @parameterized.named_parameters(
      ('not_lagged', 50, 200, 200, 10, 2, 5), ('lagged', 50, 200, 203, 10, 2, 5)
  )
  def test_dataframe_data_loader_loads_random_dataset_media_and_rf(
      self,
      n_geos,
      n_times,
      n_media_times,
      n_media_channels,
      n_rf_channels,
      n_controls,
  ):
    dataset = test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_controls=n_controls,
    )
    df = test_utils.dataset_to_dataframe(
        dataset,
        controls_column_names=test_utils._sample_names('control_', n_controls),
        media_column_names=test_utils._sample_names('media_', n_media_channels),
        media_spend_column_names=test_utils._sample_names(
            'media_spend_', n_media_channels
        ),
        reach_column_names=test_utils._sample_names('reach_', n_rf_channels),
        frequency_column_names=test_utils._sample_names(
            'frequency_', n_rf_channels
        ),
        rf_spend_column_names=test_utils._sample_names(
            'rf_spend_', n_rf_channels
        ),
    )

    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=n_controls,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
    )

    media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
    )

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])

  @parameterized.named_parameters(
      ('not_lagged', 1, 200, 200, 10, 2, 5),
      ('lagged', 1, 200, 203, 10, 2, 5),
      ('single_lag', 1, 200, 201, 10, 2, 5),
  )
  def test_dataframe_data_loader_loads_random_dataset_national_media_and_rf(
      self,
      n_geos,
      n_times,
      n_media_times,
      n_media_channels,
      n_rf_channels,
      n_controls,
  ):
    dataset = test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_controls=n_controls,
    )
    df = test_utils.dataset_to_dataframe(
        dataset,
        controls_column_names=test_utils._sample_names('control_', n_controls),
        media_column_names=test_utils._sample_names('media_', n_media_channels),
        media_spend_column_names=test_utils._sample_names(
            'media_spend_', n_media_channels
        ),
        reach_column_names=test_utils._sample_names('reach_', n_rf_channels),
        frequency_column_names=test_utils._sample_names(
            'frequency_', n_rf_channels
        ),
        rf_spend_column_names=test_utils._sample_names(
            'rf_spend_', n_rf_channels
        ),
    )

    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=n_controls,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
    )

    media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
    )

    data = loader.load()
    dataset = dataset.assign_coords(
        {constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]},
    )
    dataset.update({
        constants.POPULATION: (
            constants.GEO,
            [constants.NATIONAL_MODEL_DEFAULT_POPULATION_VALUE],
        )
    })

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])

  @parameterized.named_parameters(
      dict(
          testcase_name='with_population_with_geo_renamed',
          data=test_utils.NATIONAL_DATA_DICT_W_POPULATION_W_GEO_RENAMED,
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO_RENAMED,
          number_of_warnings=1,
      ),
      dict(
          testcase_name='with_population_with_geo',
          data=test_utils.NATIONAL_DATA_DICT_W_POPULATION_W_GEO,
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO,
          number_of_warnings=1,
      ),
      dict(
          testcase_name='with_population_without_geo',
          data=test_utils.NATIONAL_DATA_DICT_W_POPULATION_WO_GEO,
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_WO_GEO,
          number_of_warnings=1,
      ),
      dict(
          testcase_name='without_population_with_geo',
          data=test_utils.NATIONAL_DATA_DICT_WO_POPULATION_W_GEO,
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_W_GEO,
          number_of_warnings=0,
      ),
      dict(
          testcase_name='without_population_without_geo',
          data=test_utils.NATIONAL_DATA_DICT_WO_POPULATION_WO_GEO,
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
          number_of_warnings=0,
      ),
  )
  def test_dataframe_data_loader_loads_national_dataset(
      self,
      data: Mapping[str, Sequence[str | float]],
      coord_to_columns: load.CoordToColumns,
      number_of_warnings: int,
  ):
    with warnings.catch_warnings(record=True) as warning_list:
      warnings.simplefilter('always')
      loader = load.DataFrameDataLoader(
          df=pd.DataFrame(dict(data)),
          coord_to_columns=coord_to_columns,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
      )
      if number_of_warnings == 0:
        self.assertFalse(
            any(
                issubclass(warning.category, UserWarning)
                for warning in warning_list
            )
        )
      else:  # number_of_warnings == 1:
        self.assertTrue(
            any(
                issubclass(warning.category, UserWarning)
                and str(warning.message)
                == 'The `population` argument is ignored in a nationally'
                ' aggregated model. It will be reset to [1, 1, ..., 1]'
            )
            for warning in warning_list
        )

    data = loader.load()

    expected_df = pd.DataFrame(dict(test_utils.EXPECTED_NATIONAL_DATA_DICT))
    expected_kpi = expected_df.get(constants.KPI).tolist()
    expected_revenue_per_kpi = expected_df.get(
        constants.REVENUE_PER_KPI
    ).tolist()
    # Group population by geo match xarray format.
    df_indexed = expected_df.set_index([constants.GEO, constants.TIME])
    expected_population = (
        df_indexed[constants.POPULATION].groupby(constants.GEO).mean().tolist()
    )

    expected_media = [expected_df.get(m) for m in loader.coord_to_columns.media]
    expected_media_spend = [
        expected_df.get(s) for s in loader.coord_to_columns.media_spend
    ]
    expected_controls = [
        expected_df.get(c) for c in loader.coord_to_columns.controls
    ]

    self.assertTrue((data.kpi.values == expected_kpi).all())
    self.assertTrue((data.population.values == expected_population).all())

    self.assertIsNotNone(data.revenue_per_kpi)
    self.assertTrue(
        (data.revenue_per_kpi.values == expected_revenue_per_kpi).all()  # pytype: disable=attribute-error
    )

    self.assertIsNotNone(data.media)
    self.assertTrue(
        (
            np.sort(data.media.values, axis=None)  # pytype: disable=attribute-error
            == np.sort(expected_media, axis=None)
        ).all()
    )

    self.assertIsNotNone(data.media_spend)
    self.assertTrue(
        (
            np.sort(data.media_spend.values, axis=None)  # pytype: disable=attribute-error
            == np.sort(expected_media_spend, axis=None)
        ).all()
    )

    self.assertIsNotNone(data.controls)
    self.assertTrue(
        (
            np.sort(data.controls.values, axis=None)  # pytype: disable=attribute-error
            == np.sort(expected_controls, axis=None)
        ).all()
    )

  def test_coords_to_columns_works(self):
    coord_to_columns = load.CoordToColumns(
        revenue_per_kpi=constants.REVENUE_PER_KPI,
        controls=[constants.CONTROLS],
        media=['Impressions_Channel_1', 'Impressions_Channel_2'],
        media_spend=['Cost_Channel_1', 'Cost_Channel_2'],
    )
    expected_coord_to_columns = load.CoordToColumns(
        time=constants.TIME,
        geo=constants.GEO,
        controls=[constants.CONTROLS],
        population=constants.POPULATION,
        kpi=constants.KPI,
        revenue_per_kpi=constants.REVENUE_PER_KPI,
        media=['Impressions_Channel_1', 'Impressions_Channel_2'],
        media_spend=['Cost_Channel_1', 'Cost_Channel_2'],
    )
    self.assertEqual(coord_to_columns, expected_coord_to_columns)

  def test_coords_to_columns_fails_missing_media_and_rf(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '`coord_to_columns` should include media data (`media` and'
        ' `media_spend`) or RF data (`reach`, `frequency` and `rf_spend`), or'
        ' both.',
    ):
      load.CoordToColumns(
          time='time',
          geo='time',
          controls=['Weather'],
          population='population',
          kpi='Revenue',
          revenue_per_kpi='revenue_per_kpi',
          media=['Impressions_Channel_1', 'Impressions_Channel_2'],
      )

  @parameterized.named_parameters(
      (
          'wrong_value',
          'wrong_value',
          "DataFrame is missing one or more columns from ['revenue', 'time']",
      ),
      (
          'extra_control',
          'extra_control',
          (
              "DataFrame is missing one or more columns from ['control_0',"
              " 'control_1', 'control_2', 'time']"
          ),
      ),
  )
  def test_dataframe_data_loader_wrong_coords_fails(
      self, coord_test_nr, error_message
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_message,
    ):
      load.DataFrameDataLoader(
          df=self._sample_df_with_media_only,
          coord_to_columns=self._wrong_coord_test_parameters[coord_test_nr],
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
      ).load()

  @parameterized.named_parameters(
      (
          'media_to_channel',
          ['media_to_channel'],
          "When media data is provided, 'media_to_channel' is required.",
      ),
      (
          'media_spend_to_channel',
          ['media_spend_to_channel'],
          (
              "When media_spend data is provided, 'media_spend_to_channel' is"
              ' required.'
          ),
      ),
      (
          'reach_to_channel',
          ['reach_to_channel'],
          "When reach data is provided, 'reach_to_channel' is required.",
      ),
      (
          'frequency_to_channel',
          ['frequency_to_channel'],
          (
              "When frequency data is provided, 'frequency_to_channel' is"
              ' required.'
          ),
      ),
      (
          'rf_spend_to_channel',
          ['rf_spend_to_channel'],
          "When rf_spend data is provided, 'rf_spend_to_channel' is required.",
      ),
  )
  def test_dataframe_data_loader_missing_mapping_fails(
      self, missing_mapping, error_message
  ):
    media_to_channel = (
        None
        if 'media_to_channel' in missing_mapping
        else self._correct_media_to_channel
    )
    media_spend_to_channel = (
        None
        if 'media_spend_to_channel' in missing_mapping
        else self._correct_media_spend_to_channel
    )
    reach_to_channel = (
        None
        if 'reach_to_channel' in missing_mapping
        else self._correct_reach_to_channel
    )
    frequency_to_channel = (
        None
        if 'frequency_to_channel' in missing_mapping
        else self._correct_frequency_to_channel
    )
    rf_spend_to_channel = (
        None
        if 'rf_spend_to_channel' in missing_mapping
        else self._correct_rf_spend_to_channel
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_message,
    ):
      load.DataFrameDataLoader(
          df=self._sample_df_with_media_and_rf,
          coord_to_columns=self._correct_coord_to_columns_media_and_rf,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=media_to_channel,
          media_spend_to_channel=media_spend_to_channel,
          reach_to_channel=reach_to_channel,
          frequency_to_channel=frequency_to_channel,
          rf_spend_to_channel=rf_spend_to_channel,
      ).load()

  @parameterized.named_parameters(
      (
          'duplicate_time',
          'duplicate_time',
          "Duplicate entries found in the 'time' column.",
      ),
      (
          'not_matching_times',
          'not_matching_times',
          "Values in the 'time' column not consistent across different geos.",
      ),
      (
          'not_matching_times_with_int_geo',
          'not_matching_times_with_int_geo',
          "Values in the 'time' column not consistent across different geos.",
      ),
  )
  def test_dataframe_data_loader_geo_time_fails(self, test_name, error_message):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_message,
    ):
      load.DataFrameDataLoader(
          df=self._geo_time_test_parameters[test_name],
          coord_to_columns=self._correct_coord_to_columns_media_only,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
      ).load()

  def test_dataframe_data_loader_extra_columns_ok(self):
    n_geos = 10
    n_times = 50
    n_media_times = 50
    n_media_channels = 3
    n_controls = 2
    df = test_utils.random_dataframe(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_controls=n_controls,
        n_media_channels=n_media_channels,
    )

    load.DataFrameDataLoader(
        df=df,
        coord_to_columns=self._coord_to_columns_missing_media,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
    )

  @parameterized.named_parameters(
      (
          'NA_in_media',
          'NA_in_media',
          'NA values found in the media data.',
      ),
      (
          'NA_in_reach',
          'NA_in_reach',
          'NA values found in the reach data.',
      ),
      (
          'NA_in_frequency',
          'NA_in_frequency',
          'NA values found in the frequency data.',
      ),
      (
          'non_NA_in_lagged_period',
          'non_NA_in_lagged_period',
          (
              "`times` coords already set to ['2021-01-25', '2021-02-01',"
              " '2021-02-08', '2021-02-15', '2021-02-22', '2021-03-01',"
              " '2021-03-08', '2021-03-15', '2021-03-22', '2021-03-29',"
              " '2021-04-05', '2021-04-12', '2021-04-19', '2021-04-26',"
              " '2021-05-03', '2021-05-10', '2021-05-17', '2021-05-24',"
              " '2021-05-31', '2021-06-07', '2021-06-14', '2021-06-21',"
              " '2021-06-28', '2021-07-05', '2021-07-12', '2021-07-19',"
              " '2021-07-26', '2021-08-02', '2021-08-09', '2021-08-16',"
              " '2021-08-23', '2021-08-30', '2021-09-06', '2021-09-13',"
              " '2021-09-20', '2021-09-27', '2021-10-04', '2021-10-11',"
              " '2021-10-18', '2021-10-25', '2021-11-01', '2021-11-08',"
              " '2021-11-15', '2021-11-22', '2021-11-29', '2021-12-06',"
              " '2021-12-13', '2021-12-20', '2021-12-27', '2022-01-03']."
          ),
      ),
      (
          'NA_outside_lagged_period',
          'NA_outside_lagged_period',
          'NA values found in the kpi data.',
      ),
  )
  def test_dataframe_data_loader_wrong_lagged_media_fails(
      self, test_name, error_message
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_message,
    ):
      load.DataFrameDataLoader(
          df=self.lagged_media_test_parameters[test_name],
          coord_to_columns=self._correct_coord_to_columns_media_and_rf,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
          reach_to_channel=self._correct_reach_to_channel,
          frequency_to_channel=self._correct_frequency_to_channel,
          rf_spend_to_channel=self._correct_rf_spend_to_channel,
      ).load()

  def test_dataframe_data_loader_not_continuous_na_period_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Time coordinates are not regularly spaced!'
    ):
      load.DataFrameDataLoader(
          df=self._sample_df_not_continuous_na_period,
          coord_to_columns=self._correct_coord_to_columns_media_only,
          kpi_type=constants.NON_REVENUE,
          media_to_channel=self._correct_media_to_channel,
          media_spend_to_channel=self._correct_media_spend_to_channel,
      ).load()

  @parameterized.named_parameters(
      dict(
          testcase_name='not_lagged_media_only',
          file_name='sample_data_media_only.csv',
          n_media_times=200,
          n_media_channels=3,
          n_rf_channels=None,
      ),
      dict(
          testcase_name='lagged_media_only',
          file_name='lagged_sample_data_media_only.csv',
          n_media_times=203,
          n_media_channels=3,
          n_rf_channels=None,
      ),
      dict(
          testcase_name='not_lagged_rf_only',
          file_name='sample_data_rf_only.csv',
          n_media_times=200,
          n_media_channels=None,
          n_rf_channels=2,
      ),
      dict(
          testcase_name='lagged_rf_only',
          file_name='lagged_sample_data_rf_only.csv',
          n_media_times=203,
          n_media_channels=None,
          n_rf_channels=2,
      ),
      dict(
          testcase_name='not_lagged_media_and_rf',
          file_name='sample_data_media_and_rf.csv',
          n_media_times=200,
          n_media_channels=3,
          n_rf_channels=2,
      ),
      dict(
          testcase_name='lagged_media_and_rf',
          file_name='lagged_sample_data_media_and_rf.csv',
          n_media_times=203,
          n_media_channels=3,
          n_rf_channels=2,
      ),
  )
  def test_csv_data_loader_loads_all_arrays(
      self,
      file_name: str,
      n_media_times: int,
      n_media_channels: int,
      n_rf_channels: int,
  ):
    """Tests loading data from csv.

    CSV files were generated using `random_dataset()` function with `seed=0`. In
    this test, the same dataset is generated and compared to the data read from
    the CSV file.
    """
    csv_file = os.path.join(
        os.path.dirname(__file__), _UNIT_TEST_DATA_DIR_NAME, file_name
    )
    if n_media_channels and n_rf_channels:
      coord_to_columns = self._correct_coord_to_columns_media_and_rf
    elif n_media_channels:
      coord_to_columns = self._correct_coord_to_columns_media_only
    else:
      coord_to_columns = self._correct_coord_to_columns_rf_only
    media_to_channel = (
        self._correct_media_to_channel if n_media_channels else None
    )
    media_spend_to_channel = (
        self._correct_media_spend_to_channel if n_media_channels else None
    )
    reach_to_channel = self._correct_reach_to_channel if n_rf_channels else None
    frequency_to_channel = (
        self._correct_frequency_to_channel if n_rf_channels else None
    )
    rf_spend_to_channel = (
        self._correct_rf_spend_to_channel if n_rf_channels else None
    )
    loader = load.CsvDataLoader(
        csv_path=csv_file,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
    )

    data = loader.load()

    dataset = test_utils.random_dataset(
        n_geos=5,
        n_times=200,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_controls=2,
        seed=0,
    )

    xr.testing.assert_allclose(data.kpi, dataset[constants.KPI])
    xr.testing.assert_allclose(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_allclose(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_allclose(data.population, dataset[constants.POPULATION])
    if n_media_channels:
      xr.testing.assert_allclose(data.media, dataset[constants.MEDIA])
      xr.testing.assert_allclose(
          data.media_spend, dataset[constants.MEDIA_SPEND]
      )
    else:
      self.assertIsNone(data.media)
      self.assertIsNone(data.media_spend)
    if n_rf_channels:
      xr.testing.assert_allclose(data.reach, dataset[constants.REACH])
      xr.testing.assert_allclose(data.frequency, dataset[constants.FREQUENCY])
      xr.testing.assert_allclose(data.rf_spend, dataset[constants.RF_SPEND])
    else:
      self.assertIsNone(data.reach)
      self.assertIsNone(data.frequency)
      self.assertIsNone(data.rf_spend)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_population_with_geo',
          file_name='sample_national_data_w_population_w_geo.csv',
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO,
      ),
      dict(
          testcase_name='with_population_without_geo',
          file_name='sample_national_data_w_population_wo_geo.csv',
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_WO_GEO,
      ),
      dict(
          testcase_name='without_population_with_geo',
          file_name='sample_national_data_wo_population_w_geo.csv',
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_W_GEO,
      ),
      dict(
          testcase_name='without_population_without_geo',
          file_name='sample_national_data_wo_population_wo_geo.csv',
          coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_WO_POPULATION_WO_GEO,
      ),
  )
  def test_national_csv_data_loader_loads(
      self, file_name: str, coord_to_columns: load.CoordToColumns
  ):
    """Tests loading data from 'sample_national_data_...csv'.

    CSV files differ in `population` and `geo` columns. All these files are
    considered equivalent to the `CsvDataLoader`.
    """
    csv_file = os.path.join(
        os.path.dirname(__file__), _UNIT_TEST_DATA_DIR_NAME, file_name
    )
    loader = load.CsvDataLoader(
        csv_path=csv_file,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
    )
    data = loader.load()

    expected_loader = load.CsvDataLoader(
        csv_path=os.path.join(
            os.path.dirname(__file__),
            _UNIT_TEST_DATA_DIR_NAME,
            'expected_national_data.csv',
        ),
        coord_to_columns=test_utils.NATIONAL_COORD_TO_COLUMNS_W_POPULATION_W_GEO,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
    )
    expected_data = expected_loader.load()

    xr.testing.assert_equal(data.kpi, expected_data.kpi)
    xr.testing.assert_equal(data.revenue_per_kpi, expected_data.revenue_per_kpi)
    xr.testing.assert_equal(data.media, expected_data.media)
    xr.testing.assert_equal(data.media_spend, expected_data.media_spend)
    xr.testing.assert_equal(data.controls, expected_data.controls)
    xr.testing.assert_equal(data.population, expected_data.population)

  def test_no_revenue_per_kpi_csv_data_loader(self):
    """Tests loading data without `revenue_per_kpi`."""
    csv_file = os.path.join(
        os.path.dirname(__file__),
        _UNIT_TEST_DATA_DIR_NAME,
        'sample_data_no_revenue_per_kpi.csv',
    )
    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=self._N_CONTROLS,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_rf_channels=self._N_RF_CHANNELS,
        include_revenue_per_kpi=False,
    )
    loader = load.CsvDataLoader(
        csv_path=csv_file,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=self._correct_media_to_channel,
        media_spend_to_channel=self._correct_media_spend_to_channel,
        reach_to_channel=self._correct_reach_to_channel,
        frequency_to_channel=self._correct_frequency_to_channel,
        rf_spend_to_channel=self._correct_rf_spend_to_channel,
    )
    data = loader.load()

    dataset = test_utils.random_dataset(
        n_geos=5,
        n_times=200,
        n_media_times=200,
        n_media_channels=3,
        n_rf_channels=2,
        n_controls=2,
        seed=0,
        revenue_per_kpi_value=None,
    )

    xr.testing.assert_allclose(data.kpi, dataset[constants.KPI])
    xr.testing.assert_allclose(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_allclose(data.population, dataset[constants.POPULATION])
    xr.testing.assert_allclose(data.media, dataset[constants.MEDIA])
    xr.testing.assert_allclose(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_allclose(data.reach, dataset[constants.REACH])
    xr.testing.assert_allclose(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_allclose(data.rf_spend, dataset[constants.RF_SPEND])
    self.assertIsNone(data.revenue_per_kpi)


class NonPaidInputDataLoaderTest(parameterized.TestCase):
  _N_GEOS = 5
  _N_TIMES = 200
  _N_MEDIA_TIMES = 203
  _N_CONTROLS = 2
  _N_MEDIA_CHANNELS = 3
  _N_RF_CHANNELS = 2
  _N_NON_MEDIA_CHANNELS = 2
  _N_ORGANIC_MEDIA_CHANNELS = 4
  _N_ORGANIC_RF_CHANNELS = 1

  def setUp(self):
    super().setUp()

    self._correct_media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(self._N_MEDIA_CHANNELS)
    }
    self._correct_media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(self._N_MEDIA_CHANNELS)
    }
    self._correct_reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }
    self._correct_frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }
    self._correct_rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(self._N_RF_CHANNELS)
    }
    self._correct_organic_reach_to_channel = {
        f'organic_reach_{x}': f'organic_rf_ch_{x}'
        for x in range(self._N_ORGANIC_RF_CHANNELS)
    }
    self._correct_organic_frequency_to_channel = {
        f'organic_frequency_{x}': f'organic_rf_ch_{x}'
        for x in range(self._N_ORGANIC_RF_CHANNELS)
    }

    self._correct_coord_to_columns_with_non_media = (
        test_utils.sample_coord_to_columns(
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            n_organic_media_channels=self._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=self._N_ORGANIC_RF_CHANNELS,
        )
    )

  def test_xr_dataset_data_loader_loads_non_media(self):
    dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=self._N_MEDIA_TIMES,
        n_media_channels=self._N_MEDIA_CHANNELS,
        n_controls=self._N_CONTROLS,
        n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
        n_organic_media_channels=self._N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=self._N_ORGANIC_RF_CHANNELS,
    )
    loader = load.XrDatasetDataLoader(dataset, kpi_type=constants.NON_REVENUE)

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_equal(
        data.non_media_treatments, dataset[constants.NON_MEDIA_TREATMENTS]
    )
    xr.testing.assert_equal(
        data.organic_media, dataset[constants.ORGANIC_MEDIA]
    )
    xr.testing.assert_equal(
        data.organic_reach, dataset[constants.ORGANIC_REACH]
    )
    xr.testing.assert_equal(
        data.organic_frequency, dataset[constants.ORGANIC_FREQUENCY]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='not_lagged',
          n_geos=_N_GEOS,
          n_times=_N_TIMES,
          n_media_times=_N_TIMES,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
      ),
      dict(
          testcase_name='lagged',
          n_geos=_N_GEOS,
          n_times=_N_TIMES,
          n_media_times=_N_MEDIA_TIMES,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_controls=_N_CONTROLS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
      ),
  )
  def test_dataframe_data_loader_loads_random_dataset_with_non_media(
      self,
      n_geos,
      n_times,
      n_media_times,
      n_media_channels,
      n_rf_channels,
      n_controls,
      n_non_media_channels,
      n_organic_media_channels,
      n_organic_rf_channels,
  ):
    dataset = test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_controls=n_controls,
        n_non_media_channels=n_non_media_channels,
        n_organic_media_channels=n_organic_media_channels,
        n_organic_rf_channels=n_organic_rf_channels,
    )
    df = test_utils.dataset_to_dataframe(
        dataset,
        controls_column_names=test_utils._sample_names('control_', n_controls),
        media_column_names=test_utils._sample_names('media_', n_media_channels),
        media_spend_column_names=test_utils._sample_names(
            'media_spend_', n_media_channels
        ),
        reach_column_names=test_utils._sample_names('reach_', n_rf_channels),
        frequency_column_names=test_utils._sample_names(
            'frequency_', n_rf_channels
        ),
        rf_spend_column_names=test_utils._sample_names(
            'rf_spend_', n_rf_channels
        ),
        non_media_column_names=test_utils._sample_names(
            'non_media_', n_non_media_channels
        ),
        organic_media_column_names=test_utils._sample_names(
            'organic_media_', n_organic_media_channels
        ),
        organic_reach_column_names=test_utils._sample_names(
            'organic_reach_', n_organic_rf_channels
        ),
        organic_frequency_column_names=test_utils._sample_names(
            'organic_frequency_', n_organic_rf_channels
        ),
    )

    coord_to_columns = test_utils.sample_coord_to_columns(
        n_controls=n_controls,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_non_media_channels=n_non_media_channels,
        n_organic_media_channels=n_organic_media_channels,
        n_organic_rf_channels=n_organic_rf_channels,
    )

    media_to_channel = {
        f'media_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    media_spend_to_channel = {
        f'media_spend_{x}': f'ch_{x}' for x in range(n_media_channels)
    }
    reach_to_channel = {
        f'reach_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    frequency_to_channel = {
        f'frequency_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    rf_spend_to_channel = {
        f'rf_spend_{x}': f'rf_ch_{x}' for x in range(n_rf_channels)
    }
    organic_reach_to_channel = {
        f'organic_reach_{x}': f'organic_rf_ch_{x}'
        for x in range(n_organic_rf_channels)
    }
    organic_frequency_to_channel = {
        f'organic_frequency_{x}': f'organic_rf_ch_{x}'
        for x in range(n_organic_rf_channels)
    }

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
        organic_reach_to_channel=organic_reach_to_channel,
        organic_frequency_to_channel=organic_frequency_to_channel,
    )

    data = loader.load()

    xr.testing.assert_equal(data.kpi, dataset[constants.KPI])
    xr.testing.assert_equal(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_equal(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_equal(data.population, dataset[constants.POPULATION])
    xr.testing.assert_equal(data.media, dataset[constants.MEDIA])
    xr.testing.assert_equal(data.media_spend, dataset[constants.MEDIA_SPEND])
    xr.testing.assert_equal(data.reach, dataset[constants.REACH])
    xr.testing.assert_equal(data.frequency, dataset[constants.FREQUENCY])
    xr.testing.assert_equal(data.rf_spend, dataset[constants.RF_SPEND])
    xr.testing.assert_equal(
        data.non_media_treatments, dataset[constants.NON_MEDIA_TREATMENTS]
    )
    xr.testing.assert_equal(
        data.organic_media, dataset[constants.ORGANIC_MEDIA]
    )
    xr.testing.assert_equal(
        data.organic_reach, dataset[constants.ORGANIC_REACH]
    )
    xr.testing.assert_equal(
        data.organic_frequency, dataset[constants.ORGANIC_FREQUENCY]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='sample_data_with_organic_and_non_media',
          file_name='sample_data_with_organic_and_non_media.csv',
          n_media_times=_N_TIMES,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
      ),
      dict(
          testcase_name='lagged_sample_data_with_organic_and_non_media',
          file_name='lagged_sample_data_with_organic_and_non_media.csv',
          n_media_times=_N_MEDIA_TIMES,
          n_media_channels=_N_MEDIA_CHANNELS,
          n_rf_channels=_N_RF_CHANNELS,
          n_non_media_channels=_N_NON_MEDIA_CHANNELS,
          n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
          n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
      ),
  )
  def test_csv_data_loader_loads_all_arrays(
      self,
      file_name: str,
      n_media_times: int,
      n_media_channels: int,
      n_rf_channels: int,
      n_non_media_channels: int,
      n_organic_media_channels: int,
      n_organic_rf_channels: int,
  ):
    """Tests loading data from csv.

    CSV files were generated using `random_dataset()` function with `seed=0`. In
    this test, the same dataset is generated and compared to the data read from
    the CSV file.

    Args:
      file_name: Name of the CSV file.
      n_media_times: Number of media times.
      n_media_channels: Number of media channels.
      n_rf_channels: Number of RF channels.
      n_non_media_channels: Number of non-media channels.
      n_organic_media_channels: Number of organic media channels.
      n_organic_rf_channels: Number of organic RF channels.
    """
    csv_file = os.path.join(
        os.path.dirname(__file__), _UNIT_TEST_DATA_DIR_NAME, file_name
    )
    coord_to_columns = self._correct_coord_to_columns_with_non_media
    media_to_channel = (
        self._correct_media_to_channel if n_media_channels else None
    )
    media_spend_to_channel = (
        self._correct_media_spend_to_channel if n_media_channels else None
    )
    reach_to_channel = self._correct_reach_to_channel if n_rf_channels else None
    frequency_to_channel = (
        self._correct_frequency_to_channel if n_rf_channels else None
    )
    rf_spend_to_channel = (
        self._correct_rf_spend_to_channel if n_rf_channels else None
    )
    organic_reach_to_channel = (
        self._correct_organic_reach_to_channel
        if n_organic_rf_channels
        else None
    )
    organic_frequency_to_channel = (
        self._correct_organic_frequency_to_channel
        if n_organic_rf_channels
        else None
    )
    loader = load.CsvDataLoader(
        csv_path=csv_file,
        coord_to_columns=coord_to_columns,
        kpi_type=constants.NON_REVENUE,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
        reach_to_channel=reach_to_channel,
        frequency_to_channel=frequency_to_channel,
        rf_spend_to_channel=rf_spend_to_channel,
        organic_reach_to_channel=organic_reach_to_channel,
        organic_frequency_to_channel=organic_frequency_to_channel,
    )

    data = loader.load()

    dataset = test_utils.random_dataset(
        n_geos=self._N_GEOS,
        n_times=self._N_TIMES,
        n_media_times=n_media_times,
        n_media_channels=n_media_channels,
        n_rf_channels=n_rf_channels,
        n_controls=self._N_CONTROLS,
        n_non_media_channels=n_non_media_channels,
        n_organic_media_channels=n_organic_media_channels,
        n_organic_rf_channels=n_organic_rf_channels,
        seed=0,
    )

    xr.testing.assert_allclose(data.kpi, dataset[constants.KPI])
    xr.testing.assert_allclose(
        data.revenue_per_kpi, dataset[constants.REVENUE_PER_KPI]
    )
    xr.testing.assert_allclose(data.controls, dataset[constants.CONTROLS])
    xr.testing.assert_allclose(data.population, dataset[constants.POPULATION])
    if n_media_channels:
      xr.testing.assert_allclose(data.media, dataset[constants.MEDIA])
      xr.testing.assert_allclose(
          data.media_spend, dataset[constants.MEDIA_SPEND]
      )
    else:
      self.assertIsNone(data.media)
      self.assertIsNone(data.media_spend)
    if n_rf_channels:
      xr.testing.assert_allclose(data.reach, dataset[constants.REACH])
      xr.testing.assert_allclose(data.frequency, dataset[constants.FREQUENCY])
      xr.testing.assert_allclose(data.rf_spend, dataset[constants.RF_SPEND])
    else:
      self.assertIsNone(data.reach)
      self.assertIsNone(data.frequency)
      self.assertIsNone(data.rf_spend)

    if n_non_media_channels:
      xr.testing.assert_allclose(
          data.non_media_treatments, dataset[constants.NON_MEDIA_TREATMENTS]
      )
    else:
      self.assertIsNone(data.non_media_treatments)

    if n_organic_media_channels:
      xr.testing.assert_allclose(
          data.organic_media, dataset[constants.ORGANIC_MEDIA]
      )
    else:
      self.assertIsNone(data.organic_media)

    if n_organic_rf_channels:
      xr.testing.assert_allclose(
          data.organic_reach, dataset[constants.ORGANIC_REACH]
      )
    else:
      self.assertIsNone(data.organic_reach)

    if n_organic_rf_channels:
      xr.testing.assert_allclose(
          data.organic_frequency, dataset[constants.ORGANIC_FREQUENCY]
      )
    else:
      self.assertIsNone(data.organic_frequency)


if __name__ == '__main__':
  absltest.main()
