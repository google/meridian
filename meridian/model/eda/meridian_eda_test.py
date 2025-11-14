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

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import meridian_eda
import numpy as np
import xarray as xr

mock = absltest.mock


class MeridianEdaTest(parameterized.TestCase):
  _N_GEOS = 2
  _N_TIMES = 3
  _N_MEDIA_CHANNELS = 2
  _N_CONTROLS = 2
  _N_RF_CHANNELS = 1
  _N_NON_MEDIA_CHANNELS = 1
  _GEO_NAMES = ['geo_0', 'geo_1']
  _MEDIA_CHANNEL_NAMES = ['ch_0', 'ch_1']
  _RF_CHANNEL_NAMES = ['rf_ch_0']
  _CONTROL_NAMES = ['control_0', 'control_1']
  _NON_MEDIA_CHANNEL_NAMES = ['non_media_0']

  def setUp(self):
    super().setUp()
    self._meridian = mock.MagicMock()
    self._meridian.is_national = False
    self._meridian.n_geos = self._N_GEOS
    self._meridian.eda_engine = mock.MagicMock()

    self._meridian.input_data.geo = self._GEO_NAMES

    self._meridian.input_data.get_n_top_largest_geos.side_effect = (
        lambda n: self._GEO_NAMES[:n]
    )

    self._eda = meridian_eda.MeridianEDA(self._meridian)

  # ============================================================================
  # KPI Tests
  # ============================================================================

  def test_plot_kpi_boxplot_geos(self):
    self._meridian.eda_engine.kpi_scaled_da = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3])

  def test_plot_kpi_boxplot_nationalize(self):
    self._meridian.eda_engine.national_kpi_scaled_da = xr.DataArray(
        np.array([[10, 20, 30]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(self._N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [10, 20, 30])

  def test_plot_kpi_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_kpi_scaled_da = xr.DataArray(
        np.array([[100, 200, 300]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(self._N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot()

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [100, 200, 300])

  # ============================================================================
  # Frequency Tests
  # ============================================================================

  def test_plot_frequency_boxplot_geos(self):
    self._meridian.eda_engine.all_freq_da = xr.DataArray(
        np.array([[[1], [2], [3]], [[4], [5], [6]]]),
        dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_frequency_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3])

  def test_plot_frequency_boxplot_nationalize(self):
    self._meridian.eda_engine.national_all_freq_da = xr.DataArray(
        np.array([[100], [200], [300]]),
        dims=[constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_frequency_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [100, 200, 300])

  def test_plot_frequency_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_all_freq_da = xr.DataArray(
        np.array([[1000], [2000], [3000]]),
        dims=[constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_frequency_boxplot()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [1000, 2000, 3000])

  # ============================================================================
  # Reach Tests
  # ============================================================================

  def test_plot_reach_boxplot_geos(self):
    self._meridian.eda_engine.all_reach_scaled_da = xr.DataArray(
        np.array([[[10], [11], [12]], [[13], [14], [15]]]),
        dims=[constants.GEO, constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_reach_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [10, 11, 12])

  def test_plot_reach_boxplot_nationalize(self):
    self._meridian.eda_engine.national_all_reach_scaled_da = xr.DataArray(
        np.array([[1000], [2000], [3000]]),
        dims=[constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_reach_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [1000, 2000, 3000])

  def test_plot_reach_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_all_reach_scaled_da = xr.DataArray(
        np.array([[5000], [6000], [7000]]),
        dims=[constants.TIME, constants.RF_CHANNEL],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_reach_boxplot()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [5000, 6000, 7000])

  # ============================================================================
  # Non Media Tests
  # ============================================================================

  def test_plot_non_media_boxplot_geos(self):
    self._meridian.eda_engine.non_media_scaled_da = xr.DataArray(
        np.array([[[1], [2], [3]], [[4], [5], [6]]]),
        dims=[
            constants.GEO,
            constants.TIME,
            constants.NON_MEDIA_CHANNEL,
        ],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.NON_MEDIA_CHANNEL: self._NON_MEDIA_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_non_media_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3])

  def test_plot_non_media_boxplot_nationalize(self):
    self._meridian.eda_engine.national_non_media_scaled_da = xr.DataArray(
        np.array([[50], [60], [70]]),
        dims=[
            constants.TIME,
            constants.NON_MEDIA_CHANNEL,
        ],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.NON_MEDIA_CHANNEL: self._NON_MEDIA_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_non_media_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [50, 60, 70])

  def test_plot_non_media_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_non_media_scaled_da = xr.DataArray(
        np.array([[500], [600], [700]]),
        dims=[
            constants.TIME,
            constants.NON_MEDIA_CHANNEL,
        ],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.NON_MEDIA_CHANNEL: self._NON_MEDIA_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_non_media_boxplot()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [500, 600, 700])

  # ============================================================================
  # Spend Tests (Stacked - Using xr.Dataset)
  # ============================================================================

  def test_plot_spend_boxplot_geos(self):
    self._meridian.eda_engine.all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array(
                    [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]
                ),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[10], [11], [12]], [[13], [14], [15]]]),
            ),
        },
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.MEDIA_CHANNEL: self._MEDIA_CHANNEL_NAMES,
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_spend_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [0, 1, 2, 3, 4, 5, 10, 11, 12])

  def test_plot_spend_boxplot_nationalize(self):
    self._meridian.eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[100, 101], [102, 103], [104, 105]]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[200], [201], [202]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(self._N_TIMES),
            constants.MEDIA_CHANNEL: self._MEDIA_CHANNEL_NAMES,
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_spend_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(
        actual_values, [100, 101, 102, 103, 104, 105, 200, 201, 202]
    )

  def test_plot_spend_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[1000, 1001], [1002, 1003], [1004, 1005]]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[2000], [2001], [2002]]]),
            ),
        },
        coords={
            constants.GEO: ['national_geo'],
            constants.TIME: range(self._N_TIMES),
            constants.MEDIA_CHANNEL: self._MEDIA_CHANNEL_NAMES,
            constants.RF_CHANNEL: self._RF_CHANNEL_NAMES,
        },
    )

    plot = self._eda.plot_spend_boxplot()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(
        actual_values, [1000, 1001, 1002, 1003, 1004, 1005, 2000, 2001, 2002]
    )

  # ============================================================================
  # Controls Tests
  # ============================================================================

  def test_plot_controls_boxplot_geos(self):
    self._meridian.eda_engine.controls_scaled_da = xr.DataArray(
        np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        dims=[constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.CONTROL_VARIABLE: self._CONTROL_NAMES,
        },
    )

    plot = self._eda.plot_controls_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3, 4, 5, 6])

  def test_plot_controls_boxplot_nationalize(self):
    self._meridian.eda_engine.national_controls_scaled_da = xr.DataArray(
        np.array([[10, 20], [30, 40], [50, 60]]),
        dims=[constants.TIME, constants.CONTROL_VARIABLE],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.CONTROL_VARIABLE: self._CONTROL_NAMES,
        },
    )

    plot = self._eda.plot_controls_boxplot(geos='nationalize')

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [10, 20, 30, 40, 50, 60])

  def test_plot_controls_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._meridian.eda_engine.national_controls_scaled_da = xr.DataArray(
        np.array([[100, 200], [300, 400], [500, 600]]),
        dims=[constants.TIME, constants.CONTROL_VARIABLE],
        coords={
            constants.TIME: range(self._N_TIMES),
            constants.CONTROL_VARIABLE: self._CONTROL_NAMES,
        },
    )

    plot = self._eda.plot_controls_boxplot()

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [100, 200, 300, 400, 500, 600])

  # ============================================================================
  # Treatments Without Non Media Tests (Stacked - Using xr.Dataset)
  # ============================================================================

  def test_plot_treatments_without_non_media_boxplot_geos(self):
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array(
                    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
                ),
            ),
        },
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
            constants.CHANNEL: self._MEDIA_CHANNEL_NAMES,
        },
    )

    self._meridian.eda_engine.treatments_without_non_media_scaled_ds = ds

    plot = self._eda.plot_treatments_without_non_media_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3, 4, 5, 6])

  def test_plot_treatments_without_non_media_boxplot_nationalize(self):
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[[10, 20], [30, 40], [50, 60]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(self._N_TIMES),
            constants.CHANNEL: self._MEDIA_CHANNEL_NAMES,
        },
    )

    self._meridian.eda_engine.national_treatments_without_non_media_scaled_ds = (
        ds
    )

    plot = self._eda.plot_treatments_without_non_media_boxplot(
        geos='nationalize'
    )

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [10, 20, 30, 40, 50, 60])

  def test_plot_treatments_without_non_media_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[[100, 200], [300, 400], [500, 600]]]),
            ),
        },
        coords={
            constants.GEO: ['national_geo'],
            constants.TIME: range(self._N_TIMES),
            constants.CHANNEL: self._MEDIA_CHANNEL_NAMES,
        },
    )

    self._meridian.eda_engine.national_treatments_without_non_media_scaled_ds = (
        ds
    )

    plot = self._eda.plot_treatments_without_non_media_boxplot()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    self.assertEqual(actual_values, [100, 200, 300, 400, 500, 600])

  # ============================================================================
  # Pairwise Correlation Tests
  # ============================================================================

  def test_plot_pairwise_correlation_geos(self):
    mock_artifact = mock.MagicMock()
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[
            [1.0, 0.8],
            [0.8, 1.0],
        ]]),
        coords={
            constants.GEO: ['geo_0'],
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[
            constants.GEO,
            eda_constants.VARIABLE_1,
            eda_constants.VARIABLE_2,
        ],
        name=eda_constants.CORRELATION,
    )
    self._meridian.eda_engine.check_geo_pairwise_corr.return_value.get_geo_artifact = (
        mock_artifact
    )

    plot = self._eda.plot_pairwise_correlation(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [0.8])

  def test_plot_pairwise_correlation_nationalize(self):

    mock_artifact = mock.MagicMock()
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[1.0, -0.2], [-0.2, 1.0]]),
        coords={
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        name=eda_constants.CORRELATION,
    )
    self._meridian.eda_engine.check_national_pairwise_corr.return_value.get_national_artifact = (
        mock_artifact
    )

    plot = self._eda.plot_pairwise_correlation(geos='nationalize')

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [-0.2])

  def test_plot_pairwise_correlation_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    mock_artifact = mock.MagicMock()
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[1.0, 0.99], [0.99, 1.0]]),
        coords={
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        name=eda_constants.CORRELATION,
    )
    self._meridian.eda_engine.check_national_pairwise_corr.return_value.get_national_artifact = (
        mock_artifact
    )

    plot = self._eda.plot_pairwise_correlation()

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [0.99])

  # ============================================================================
  # Error Scenarios
  # ============================================================================
  def test_plot_error_invalid_geo(self):
    self._meridian.eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((self._N_GEOS, self._N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
        },
    )

    with self.assertRaisesRegex(ValueError, 'Geo fake_geo does not exist'):
      self._eda.plot_kpi_boxplot(geos=['fake_geo'])

  def test_plot_error_duplicate_geos(self):
    self._meridian.eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((self._N_GEOS, self._N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
        },
    )
    with self.assertRaisesRegex(ValueError, 'geos must not contain duplicate'):
      self._eda.plot_kpi_boxplot(geos=['geo_0', 'geo_0'])

  def test_plot_error_integer_out_of_bounds(self):
    self._meridian.eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((self._N_GEOS, self._N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: self._GEO_NAMES,
            constants.TIME: range(self._N_TIMES),
        },
    )
    with self.assertRaisesRegex(ValueError, 'positive integer less than'):
      self._eda.plot_kpi_boxplot(geos=self._N_GEOS + 1)

  def test_no_data_to_plot(self):
    self._meridian.eda_engine.kpi_scaled_da = None
    with self.assertRaisesRegex(
        ValueError,
        'There is no data to plot! Make sure your InputData contains the'
        ' component you are triyng to plot.',
    ):
      self._eda.plot_kpi_boxplot()


if __name__ == '__main__':
  absltest.main()
