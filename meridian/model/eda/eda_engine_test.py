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

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model import model
from meridian.model import model_test_data
from meridian.model.eda import eda_engine
import numpy as np
import tensorflow as tf
import xarray as xr


class EDAEngineTest(
    parameterized.TestCase,
    tf.test.TestCase,
    model_test_data.WithInputDataSamples,
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.mock_scale_factor = 2.0
    mock_media_transformer_cls = self.enter_context(
        mock.patch.object(
            eda_engine.transformers, "MediaTransformer", autospec=True
        )
    )
    mock_media_transformer = mock_media_transformer_cls.return_value
    mock_media_transformer.forward.side_effect = (
        lambda x: tf.cast(x, tf.float32) * self.mock_scale_factor
    )
    mock_scaling_transformer_cls = self.enter_context(
        mock.patch.object(
            eda_engine.transformers,
            "CenteringAndScalingTransformer",
            autospec=True,
        )
    )
    mock_scaling_transformer = mock_scaling_transformer_cls.return_value
    mock_scaling_transformer.forward.side_effect = (
        lambda x: tf.cast(x, tf.float32) * self.mock_scale_factor
    )

  # --- Test cases for controls_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_CONTROLS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_CONTROLS,
          ),
      ),
  )
  def test_controls_scaled_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    controls_scaled_da = engine.controls_scaled_da
    self.assertIsInstance(controls_scaled_da, xr.DataArray)
    self.assertEqual(controls_scaled_da.shape, expected_shape)
    self.assertCountEqual(
        controls_scaled_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
    )
    self.assertAllClose(controls_scaled_da.values, meridian.controls_scaled)

  # --- Test cases for controls_scaled_da_national ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_default_agg",
          agg_config=eda_engine.AggregationConfig(),
          expected_values_func=lambda da: da.sum(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mean",
          agg_config=eda_engine.AggregationConfig(
              control_variables={
                  "control_0": np.mean,
                  "control_1": np.mean,
              }
          ),
          expected_values_func=lambda da: da.mean(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mix",
          agg_config=eda_engine.AggregationConfig(
              control_variables={"control_0": np.mean}
          ),
          expected_values_func=lambda da: xr.concat(
              [
                  da.sel({constants.CONTROL_VARIABLE: "control_0"}).mean(
                      dim=constants.GEO
                  ),
                  da.sel({constants.CONTROL_VARIABLE: "control_1"}).sum(
                      dim=constants.GEO
                  ),
              ],
              dim=constants.CONTROL_VARIABLE,
          ).transpose(constants.TIME, constants.CONTROL_VARIABLE),
      ),
  )
  def test_controls_scaled_da_national_with_geo_data(
      self, agg_config, expected_values_func
  ):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian, agg_config=agg_config)

    controls_scaled_da_national = engine.controls_scaled_da_national
    self.assertIsInstance(controls_scaled_da_national, xr.DataArray)
    self.assertEqual(
        controls_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_CONTROLS,
        ),
    )
    self.assertCountEqual(
        controls_scaled_da_national.coords.keys(),
        [constants.TIME, constants.CONTROL_VARIABLE],
    )

    # Check values
    self.assertIsInstance(meridian.input_data.controls, xr.DataArray)
    expected_da = expected_values_func(meridian.input_data.controls)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        controls_scaled_da_national.values, scaled_expected_values
    )

  def test_controls_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    controls_scaled_da_national = engine.controls_scaled_da_national
    self.assertIsInstance(controls_scaled_da_national, xr.DataArray)
    self.assertEqual(
        controls_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_CONTROLS,
        ),
    )
    expected_controls_scaled_da = engine.controls_scaled_da
    self.assertIsInstance(expected_controls_scaled_da, xr.DataArray)
    self.assertAllClose(
        controls_scaled_da_national.values, expected_controls_scaled_da.values
    )

  # --- Test cases for media_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_media_raw_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    media_da = engine.media_raw_da
    self.assertIsInstance(media_da, xr.DataArray)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_media_da = meridian.input_data.media
    self.assertIsInstance(true_media_da, xr.DataArray)
    self.assertAllClose(media_da.values, true_media_da.values[:, start:, :])

  # --- Test cases for media_raw_da_national ---
  def test_media_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)
    media_raw_da_national = engine.media_raw_da_national
    self.assertIsInstance(media_raw_da_national, xr.DataArray)
    self.assertEqual(
        media_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        media_raw_da_national.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_media_raw_da = meridian.input_data.media
    self.assertIsNotNone(true_media_raw_da)
    expected_da = true_media_raw_da.sum(dim=constants.GEO)
    self.assertAllClose(media_raw_da_national.values, expected_da.values)

  def test_media_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    media_raw_da_national = engine.media_raw_da_national
    self.assertIsInstance(media_raw_da_national, xr.DataArray)
    self.assertEqual(
        media_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    expected_media_raw_da = engine.media_raw_da
    self.assertIsInstance(expected_media_raw_da, xr.DataArray)
    self.assertAllClose(
        media_raw_da_national.values, expected_media_raw_da.values
    )

  # --- Test cases for media_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_media_scaled_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    media_da = engine.media_scaled_da
    self.assertIsInstance(media_da, xr.DataArray)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        media_da.values, meridian.media_tensors.media_scaled[:, start:, :]
    )

  # --- Test cases for media_scaled_da_national ---
  def test_media_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)

    media_scaled_da_national = engine.media_scaled_da_national
    self.assertIsInstance(media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        media_scaled_da_national.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_media_raw_da = meridian.input_data.media
    self.assertIsNotNone(true_media_raw_da)
    expected_da = true_media_raw_da.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        media_scaled_da_national.values,
        scaled_expected_values,
    )

  def test_media_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    media_scaled_da_national = engine.media_scaled_da_national
    self.assertIsInstance(media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    expected_media_scaled_da = engine.media_scaled_da
    self.assertIsInstance(expected_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        media_scaled_da_national.values, expected_media_scaled_da.values
    )

  # --- Test cases for media_spend_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_media_spend_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    media_da = engine.media_spend_da
    self.assertIsInstance(media_da, xr.DataArray)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    self.assertAllClose(media_da.values, meridian.media_tensors.media_spend)

  # --- Test cases for media_spend_da_national ---
  def test_media_spend_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    media_spend_da_national = engine.media_spend_da_national
    self.assertIsInstance(media_spend_da_national, xr.DataArray)
    self.assertEqual(
        media_spend_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        media_spend_da_national.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_media_spend_da = meridian.input_data.media_spend
    self.assertIsInstance(true_media_spend_da, xr.DataArray)
    expected_da = true_media_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(media_spend_da_national.values, expected_da.values)

  def test_media_spend_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    media_spend_da_national = engine.media_spend_da_national
    self.assertIsInstance(media_spend_da_national, xr.DataArray)
    self.assertEqual(
        media_spend_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    expected_media_spend_da = engine.media_spend_da
    self.assertIsInstance(expected_media_spend_da, xr.DataArray)
    self.assertAllClose(
        media_spend_da_national.values, expected_media_spend_da.values
    )

  # --- Test cases for organic_media_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_organic_media_raw_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_media_da = engine.organic_media_raw_da
    self.assertIsInstance(organic_media_da, xr.DataArray)
    self.assertEqual(organic_media_da.shape, expected_shape)
    self.assertCountEqual(
        organic_media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_organic_media_da = meridian.input_data.organic_media
    self.assertIsInstance(true_organic_media_da, xr.DataArray)
    self.assertAllClose(
        organic_media_da.values, true_organic_media_da.values[:, start:, :]
    )

  # --- Test cases for organic_media_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_organic_media_scaled_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_media_da = engine.organic_media_scaled_da
    self.assertIsInstance(organic_media_da, xr.DataArray)
    self.assertEqual(organic_media_da.shape, expected_shape)
    self.assertCountEqual(
        organic_media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        organic_media_da.values,
        meridian.organic_media_tensors.organic_media_scaled[:, start:, :],
    )

  # --- Test cases for organic_media_raw_da_national ---
  def test_organic_media_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)
    organic_media_raw_da_national = engine.organic_media_raw_da_national
    self.assertIsInstance(organic_media_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_media_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_media_raw_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )

    # Check values
    true_organic_media_raw_da = meridian.input_data.organic_media
    self.assertIsNotNone(true_organic_media_raw_da)
    expected_da = true_organic_media_raw_da.sum(dim=constants.GEO)
    self.assertAllClose(
        organic_media_raw_da_national.values, expected_da.values
    )

  def test_organic_media_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_media_raw_da_national = engine.organic_media_raw_da_national
    self.assertIsInstance(organic_media_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_media_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    expected_organic_media_raw_da = engine.organic_media_raw_da
    self.assertIsInstance(expected_organic_media_raw_da, xr.DataArray)
    self.assertAllClose(
        organic_media_raw_da_national.values,
        expected_organic_media_raw_da.values,
    )

  # --- Test cases for organic_media_scaled_da_national ---
  def test_organic_media_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)

    organic_media_scaled_da_national = engine.organic_media_scaled_da_national
    self.assertIsInstance(organic_media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        organic_media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_media_scaled_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )

    # Check values
    true_organic_media_raw_da = meridian.input_data.organic_media
    self.assertIsNotNone(true_organic_media_raw_da)
    expected_da = true_organic_media_raw_da.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        organic_media_scaled_da_national.values, scaled_expected_values
    )

  def test_organic_media_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_media_scaled_da_national = engine.organic_media_scaled_da_national
    self.assertIsInstance(organic_media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        organic_media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    expected_organic_media_scaled_da = engine.organic_media_scaled_da
    self.assertIsInstance(expected_organic_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        organic_media_scaled_da_national.values,
        expected_organic_media_scaled_da.values,
    )

  # --- Test cases for non_media_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
          ),
      ),
  )
  def test_non_media_scaled_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    non_media_da = engine.non_media_scaled_da
    self.assertIsInstance(non_media_da, xr.DataArray)
    self.assertEqual(non_media_da.shape, expected_shape)
    self.assertCountEqual(
        non_media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
    )
    self.assertAllClose(
        non_media_da.values, meridian.non_media_treatments_normalized
    )

  # --- Test cases for non_media_scaled_da_national ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_default_agg",
          agg_config=eda_engine.AggregationConfig(),
          expected_values_func=lambda da: da.sum(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mean",
          agg_config=eda_engine.AggregationConfig(
              non_media_treatments={
                  "non_media_0": np.mean,
              }
          ),
          expected_values_func=lambda da: xr.concat(
              [
                  da.sel({constants.NON_MEDIA_CHANNEL: "non_media_0"}).mean(
                      dim=constants.GEO
                  ),
                  da.sel({constants.NON_MEDIA_CHANNEL: "non_media_1"}).sum(
                      dim=constants.GEO
                  ),
              ],
              dim=constants.NON_MEDIA_CHANNEL,
          ).transpose(constants.TIME, constants.NON_MEDIA_CHANNEL),
      ),
  )
  def test_non_media_scaled_da_national_with_geo_data(
      self, agg_config, expected_values_func
  ):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian, agg_config=agg_config)

    non_media_scaled_da_national = engine.non_media_scaled_da_national
    self.assertIsInstance(non_media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        non_media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        non_media_scaled_da_national.coords.keys(),
        [constants.TIME, constants.NON_MEDIA_CHANNEL],
    )

    # Check values
    self.assertIsInstance(
        meridian.input_data.non_media_treatments, xr.DataArray
    )
    expected_da = expected_values_func(meridian.input_data.non_media_treatments)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        non_media_scaled_da_national.values, scaled_expected_values
    )

  def test_non_media_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    non_media_scaled_da_national = engine.non_media_scaled_da_national
    self.assertIsInstance(non_media_scaled_da_national, xr.DataArray)
    self.assertEqual(
        non_media_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
        ),
    )
    expected_non_media_scaled_da = engine.non_media_scaled_da
    self.assertIsInstance(expected_non_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        non_media_scaled_da_national.values, expected_non_media_scaled_da.values
    )

  # --- Test cases for rf_spend_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_rf_spend_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    rf_spend_da = engine.rf_spend_da
    self.assertIsInstance(rf_spend_da, xr.DataArray)
    self.assertEqual(rf_spend_da.shape, expected_shape)
    self.assertCountEqual(
        rf_spend_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    self.assertAllClose(rf_spend_da.values, meridian.rf_tensors.rf_spend)

  # --- Test cases for rf_spend_da_national ---
  def test_rf_spend_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_spend_da_national = engine.rf_spend_da_national
    self.assertIsInstance(rf_spend_da_national, xr.DataArray)
    self.assertEqual(
        rf_spend_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        rf_spend_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    true_rf_spend_da = meridian.input_data.rf_spend
    self.assertIsNotNone(true_rf_spend_da)
    expected_da = true_rf_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(rf_spend_da_national.values, expected_da.values)

  def test_rf_spend_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_spend_da_national = engine.rf_spend_da_national
    self.assertIsInstance(rf_spend_da_national, xr.DataArray)
    self.assertEqual(
        rf_spend_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_rf_spend_da = engine.rf_spend_da
    self.assertIsInstance(expected_rf_spend_da, xr.DataArray)
    self.assertAllClose(
        rf_spend_da_national.values, expected_rf_spend_da.values
    )

  # --- Test cases for reach_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_reach_raw_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    reach_da = engine.reach_raw_da
    self.assertIsInstance(reach_da, xr.DataArray)
    self.assertEqual(reach_da.shape, expected_shape)
    self.assertCountEqual(
        reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_reach_da = meridian.input_data.reach
    self.assertIsInstance(true_reach_da, xr.DataArray)
    self.assertAllClose(reach_da.values, true_reach_da.values[:, start:, :])

  # --- Test cases for reach_raw_da_national ---
  def test_reach_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    reach_raw_da_national = engine.reach_raw_da_national
    self.assertIsInstance(reach_raw_da_national, xr.DataArray)
    self.assertEqual(
        reach_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        reach_raw_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    reach_raw_da = engine.reach_raw_da
    self.assertIsInstance(reach_raw_da, xr.DataArray)
    expected_values = reach_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(reach_raw_da_national.values, expected_values)

  def test_reach_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    reach_raw_da_national = engine.reach_raw_da_national
    self.assertIsInstance(reach_raw_da_national, xr.DataArray)
    self.assertEqual(
        reach_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_reach_raw_da = engine.reach_raw_da
    self.assertIsInstance(expected_reach_raw_da, xr.DataArray)
    self.assertAllClose(
        reach_raw_da_national.values, expected_reach_raw_da.values
    )

  # --- Test cases for reach_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_reach_scaled_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    reach_da = engine.reach_scaled_da
    self.assertIsInstance(reach_da, xr.DataArray)
    self.assertEqual(reach_da.shape, expected_shape)
    self.assertCountEqual(
        reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        reach_da.values, meridian.rf_tensors.reach_scaled[:, start:, :]
    )

  # --- Test cases for reach_scaled_da_national ---
  def test_reach_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    reach_scaled_da_national = engine.reach_scaled_da_national
    self.assertIsInstance(reach_scaled_da_national, xr.DataArray)
    self.assertEqual(
        reach_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        reach_scaled_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    reach_raw_da = engine.reach_raw_da
    self.assertIsInstance(reach_raw_da, xr.DataArray)
    reach_raw_da_national = reach_raw_da.sum(dim=constants.GEO)
    # Scale the raw values by the mock scale factor
    expected_values = reach_raw_da_national.values * self.mock_scale_factor
    self.assertAllClose(reach_scaled_da_national.values, expected_values)

  def test_reach_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    reach_scaled_da_national = engine.reach_scaled_da_national
    self.assertIsInstance(reach_scaled_da_national, xr.DataArray)
    self.assertEqual(
        reach_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_reach_scaled_da = engine.reach_scaled_da
    self.assertIsInstance(expected_reach_scaled_da, xr.DataArray)
    expected_values = expected_reach_scaled_da.values
    self.assertAllClose(reach_scaled_da_national.values, expected_values)

  # --- Test cases for frequency_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_frequency_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    frequency_da = engine.frequency_da
    self.assertIsInstance(frequency_da, xr.DataArray)
    self.assertEqual(frequency_da.shape, expected_shape)
    self.assertCountEqual(
        frequency_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        frequency_da.values, meridian.rf_tensors.frequency[:, start:, :]
    )

  # --- Test cases for frequency_da_national ---
  def test_frequency_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    frequency_da_national = engine.frequency_da_national
    self.assertIsInstance(frequency_da_national, xr.DataArray)
    self.assertEqual(
        frequency_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        frequency_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    reach_raw_da_national = engine.reach_raw_da_national
    self.assertIsInstance(reach_raw_da_national, xr.DataArray)
    expected_impressions_raw_da_national = engine.rf_impressions_raw_da_national
    self.assertIsInstance(expected_impressions_raw_da_national, xr.DataArray)

    actual_impressions_raw_da = reach_raw_da_national * frequency_da_national
    self.assertAllClose(
        actual_impressions_raw_da.values,
        expected_impressions_raw_da_national.values,
    )

  def test_frequency_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    frequency_da_national = engine.frequency_da_national
    self.assertIsInstance(frequency_da_national, xr.DataArray)
    self.assertEqual(
        frequency_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_frequency_da = engine.frequency_da
    self.assertIsInstance(expected_frequency_da, xr.DataArray)
    self.assertAllClose(
        frequency_da_national.values, expected_frequency_da.values
    )

  # --- Test cases for rf_impressions_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_rf_impressions_raw_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsInstance(rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(rf_impressions_raw_da.shape, expected_shape)
    self.assertCountEqual(
        rf_impressions_raw_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    # Check values: rf_impressions_raw_da = reach_raw_da * frequency_da
    reach_raw_da = engine.reach_raw_da
    frequency_da = engine.frequency_da
    self.assertIsNotNone(reach_raw_da)
    self.assertIsNotNone(frequency_da)
    expected_values = reach_raw_da.values * frequency_da.values
    self.assertAllClose(rf_impressions_raw_da.values, expected_values)

  # --- Test cases for rf_impressions_raw_da_national ---
  def test_rf_impressions_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_raw_da_national = engine.rf_impressions_raw_da_national
    self.assertIsInstance(rf_impressions_raw_da_national, xr.DataArray)
    self.assertEqual(
        rf_impressions_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        rf_impressions_raw_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    # Check values: sum of geo-level raw impressions
    rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsInstance(rf_impressions_raw_da, xr.DataArray)
    expected_values = rf_impressions_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(rf_impressions_raw_da_national.values, expected_values)

  def test_rf_impressions_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_raw_da_national = engine.rf_impressions_raw_da_national
    self.assertIsInstance(rf_impressions_raw_da_national, xr.DataArray)
    self.assertEqual(
        rf_impressions_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsInstance(expected_rf_impressions_raw_da, xr.DataArray)
    self.assertAllClose(
        rf_impressions_raw_da_national.values,
        expected_rf_impressions_raw_da.values,
    )

  # --- Test cases for rf_impressions_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_rf_impressions_scaled_da_present(
      self, input_data_fixture, expected_shape
  ):
    def mock_media_transformer_init(media, population):
      del media  # Unused.
      mock_instance = mock.MagicMock()
      # Simplified scaling: tensor * mean(population) * mock_scale_factor
      mean_population = tf.reduce_mean(population)
      scale_factor = mean_population * self.mock_scale_factor
      mock_instance.forward.side_effect = (
          lambda tensor: tf.cast(tensor, tf.float32) * scale_factor
      )
      return mock_instance

    self.enter_context(
        mock.patch.object(
            eda_engine.transformers,
            "MediaTransformer",
            side_effect=mock_media_transformer_init,
        )
    )

    # Re-initialize engine to use the mocked MediaTransformer.
    meridian = model.Meridian(getattr(self, input_data_fixture))

    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_scaled_da = engine.rf_impressions_scaled_da
    self.assertIsNotNone(rf_impressions_scaled_da)

    self.assertIsInstance(rf_impressions_scaled_da, xr.DataArray)
    self.assertEqual(rf_impressions_scaled_da.shape, expected_shape)
    self.assertCountEqual(
        rf_impressions_scaled_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )

    # Expected values calculation: raw values * mean(population) *
    # mock_scale_factor
    mean_population = (
        1 if meridian.is_national else tf.reduce_mean(meridian.population)
    )
    expected_scale = mean_population * self.mock_scale_factor
    rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsNotNone(rf_impressions_raw_da)
    expected_values = rf_impressions_raw_da.values * expected_scale
    self.assertAllClose(rf_impressions_scaled_da.values, expected_values)

  # --- Test cases for rf_impressions_scaled_da_national ---
  def test_rf_impressions_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_scaled_da_national = engine.rf_impressions_scaled_da_national
    self.assertIsInstance(rf_impressions_scaled_da_national, xr.DataArray)
    self.assertEqual(
        rf_impressions_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        rf_impressions_scaled_da_national.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    # Check values: scaled version of rf_impressions_raw_da_national
    rf_impressions_raw_da_national = engine.rf_impressions_raw_da_national
    self.assertIsInstance(rf_impressions_raw_da_national, xr.DataArray)
    expected_values = (
        rf_impressions_raw_da_national.values * self.mock_scale_factor
    )
    self.assertAllClose(
        rf_impressions_scaled_da_national.values, expected_values
    )

  def test_rf_impressions_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    rf_impressions_scaled_da_national = engine.rf_impressions_scaled_da_national
    self.assertIsInstance(rf_impressions_scaled_da_national, xr.DataArray)
    self.assertEqual(
        rf_impressions_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    expected_rf_impressions_scaled_da = engine.rf_impressions_scaled_da
    self.assertIsInstance(expected_rf_impressions_scaled_da, xr.DataArray)
    self.assertAllClose(
        rf_impressions_scaled_da_national.values,
        expected_rf_impressions_scaled_da.values,
    )

  # --- Test cases for organic_reach_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_organic_reach_raw_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_reach_da = engine.organic_reach_raw_da
    self.assertIsInstance(organic_reach_da, xr.DataArray)
    self.assertEqual(organic_reach_da.shape, expected_shape)
    self.assertCountEqual(
        organic_reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_organic_reach_da = meridian.input_data.organic_reach
    self.assertIsInstance(true_organic_reach_da, xr.DataArray)
    self.assertAllClose(
        organic_reach_da.values, true_organic_reach_da.values[:, start:, :]
    )

  # --- Test cases for organic_reach_raw_da_national ---
  def test_organic_reach_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_reach_raw_da_national = engine.organic_reach_raw_da_national
    self.assertIsInstance(organic_reach_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_reach_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_reach_raw_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    organic_reach_raw_da = engine.organic_reach_raw_da
    self.assertIsInstance(organic_reach_raw_da, xr.DataArray)
    expected_values = organic_reach_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(organic_reach_raw_da_national.values, expected_values)

  def test_organic_reach_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_reach_raw_da_national = engine.organic_reach_raw_da_national
    self.assertIsInstance(organic_reach_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_reach_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    expected_da = engine.organic_reach_raw_da
    self.assertIsInstance(expected_da, xr.DataArray)
    self.assertAllClose(
        organic_reach_raw_da_national.values, expected_da.values
    )

  # --- Test cases for organic_reach_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_organic_reach_scaled_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_reach_da = engine.organic_reach_scaled_da
    self.assertIsInstance(organic_reach_da, xr.DataArray)
    self.assertEqual(organic_reach_da.shape, expected_shape)
    self.assertCountEqual(
        organic_reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    expected_da = engine.organic_reach_raw_da
    self.assertIsInstance(expected_da, xr.DataArray)

    self.assertAllClose(
        organic_reach_da.values,
        expected_da.values * self.mock_scale_factor,
    )

  # --- Test cases for organic_reach_scaled_da_national ---
  def test_organic_reach_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)

    organic_reach_scaled_da_national = engine.organic_reach_scaled_da_national
    self.assertIsInstance(organic_reach_scaled_da_national, xr.DataArray)
    self.assertEqual(
        organic_reach_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_reach_scaled_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    organic_reach_raw_da = engine.organic_reach_raw_da
    self.assertIsInstance(organic_reach_raw_da, xr.DataArray)
    organic_reach_raw_da_national = organic_reach_raw_da.sum(dim=constants.GEO)
    # Scale the raw values by the mock scale factor
    expected_values = (
        organic_reach_raw_da_national.values * self.mock_scale_factor
    )
    self.assertAllClose(
        organic_reach_scaled_da_national.values, expected_values
    )

  def test_organic_reach_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_reach_scaled_da_national = engine.organic_reach_scaled_da_national
    self.assertIsInstance(organic_reach_scaled_da_national, xr.DataArray)
    self.assertEqual(
        organic_reach_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    expected_da = engine.organic_reach_scaled_da
    self.assertIsInstance(expected_da, xr.DataArray)

    self.assertAllClose(
        organic_reach_scaled_da_national.values,
        expected_da.values,
    )

  # --- Test cases for organic_frequency_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_organic_frequency_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_frequency_da = engine.organic_frequency_da
    self.assertIsInstance(organic_frequency_da, xr.DataArray)
    self.assertEqual(organic_frequency_da.shape, expected_shape)
    self.assertCountEqual(
        organic_frequency_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_organic_frequency_da = meridian.input_data.organic_frequency
    self.assertIsInstance(true_organic_frequency_da, xr.DataArray)
    self.assertAllClose(
        organic_frequency_da.values,
        true_organic_frequency_da.values[:, start:, :],
    )

  # --- Test cases for organic_frequency_da_national ---
  def test_organic_frequency_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_frequency_da_national = engine.organic_frequency_da_national
    self.assertIsInstance(organic_frequency_da_national, xr.DataArray)
    self.assertEqual(
        organic_frequency_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_frequency_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    organic_reach_raw_da_national = engine.organic_reach_raw_da_national
    self.assertIsInstance(organic_reach_raw_da_national, xr.DataArray)
    expected_impressions_raw_da_national = (
        engine.organic_rf_impressions_raw_da_national
    )
    self.assertIsInstance(expected_impressions_raw_da_national, xr.DataArray)

    actual_impressions_raw_da = (
        organic_reach_raw_da_national * organic_frequency_da_national
    )
    self.assertAllClose(
        actual_impressions_raw_da.values,
        expected_impressions_raw_da_national.values,
    )

  def test_organic_frequency_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_frequency_da_national = engine.organic_frequency_da_national
    self.assertIsInstance(organic_frequency_da_national, xr.DataArray)
    self.assertEqual(
        organic_frequency_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )

    expected_da = engine.organic_frequency_da
    self.assertIsInstance(expected_da, xr.DataArray)
    self.assertAllClose(
        organic_frequency_da_national.values, expected_da.values
    )

  # --- Test cases for organic_rf_impressions_raw_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_organic_rf_impressions_raw_da_present(
      self, input_data_fixture, expected_shape
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_raw_da = engine.organic_rf_impressions_raw_da
    self.assertIsInstance(organic_rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(organic_rf_impressions_raw_da.shape, expected_shape)
    self.assertCountEqual(
        organic_rf_impressions_raw_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    # Check values: organic_rf_impressions_raw_da = organic_reach_raw_da *
    # organic_frequency_da
    organic_reach_raw_da = engine.organic_reach_raw_da
    organic_frequency_da = engine.organic_frequency_da
    self.assertIsNotNone(organic_reach_raw_da)
    self.assertIsNotNone(organic_frequency_da)
    expected_values = organic_reach_raw_da.values * organic_frequency_da.values
    self.assertAllClose(organic_rf_impressions_raw_da.values, expected_values)

  # --- Test cases for organic_rf_impressions_raw_da_national ---
  def test_organic_rf_impressions_raw_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_raw_da_national = (
        engine.organic_rf_impressions_raw_da_national
    )
    self.assertIsInstance(organic_rf_impressions_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_rf_impressions_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_rf_impressions_raw_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    # Check values: sum of geo-level raw impressions
    organic_rf_impressions_raw_da = engine.organic_rf_impressions_raw_da
    self.assertIsInstance(organic_rf_impressions_raw_da, xr.DataArray)
    expected_values = organic_rf_impressions_raw_da.sum(
        dim=constants.GEO
    ).values
    self.assertAllClose(
        organic_rf_impressions_raw_da_national.values, expected_values
    )

  def test_organic_rf_impressions_raw_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_raw_da_national = (
        engine.organic_rf_impressions_raw_da_national
    )
    self.assertIsInstance(organic_rf_impressions_raw_da_national, xr.DataArray)
    self.assertEqual(
        organic_rf_impressions_raw_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    expected_organic_rf_impressions_raw_da = (
        engine.organic_rf_impressions_raw_da
    )
    self.assertIsInstance(expected_organic_rf_impressions_raw_da, xr.DataArray)
    self.assertAllClose(
        organic_rf_impressions_raw_da_national.values,
        expected_organic_rf_impressions_raw_da.values,
    )

  # --- Test cases for organic_rf_impressions_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_organic_rf_impressions_scaled_da_present(
      self, input_data_fixture, expected_shape
  ):
    def mock_media_transformer_init(media, population):
      del media  # Unused.
      mock_instance = mock.MagicMock()
      # Simplified scaling: tensor * mean(population) * mock_scale_factor
      mean_population = tf.reduce_mean(population)
      scale_factor = mean_population * self.mock_scale_factor
      mock_instance.forward.side_effect = (
          lambda tensor: tf.cast(tensor, tf.float32) * scale_factor
      )
      return mock_instance

    self.enter_context(
        mock.patch.object(
            eda_engine.transformers,
            "MediaTransformer",
            side_effect=mock_media_transformer_init,
        )
    )

    # Re-initialize engine to use the mocked MediaTransformer.
    meridian = model.Meridian(getattr(self, input_data_fixture))

    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_scaled_da = engine.organic_rf_impressions_scaled_da
    self.assertIsNotNone(organic_rf_impressions_scaled_da)

    self.assertIsInstance(organic_rf_impressions_scaled_da, xr.DataArray)
    self.assertEqual(organic_rf_impressions_scaled_da.shape, expected_shape)
    self.assertCountEqual(
        organic_rf_impressions_scaled_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Expected values calculation: raw values * mean(population) *
    # mock_scale_factor
    mean_population = (
        1 if meridian.is_national else tf.reduce_mean(meridian.population)
    )
    expected_scale = mean_population * self.mock_scale_factor
    organic_rf_impressions_raw_da = engine.organic_rf_impressions_raw_da
    self.assertIsNotNone(organic_rf_impressions_raw_da)
    expected_values = organic_rf_impressions_raw_da.values * expected_scale
    self.assertAllClose(
        organic_rf_impressions_scaled_da.values, expected_values
    )

  # --- Test cases for organic_rf_impressions_scaled_da_national ---
  def test_organic_rf_impressions_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_scaled_da_national = (
        engine.organic_rf_impressions_scaled_da_national
    )
    self.assertIsInstance(
        organic_rf_impressions_scaled_da_national, xr.DataArray
    )
    self.assertEqual(
        organic_rf_impressions_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        organic_rf_impressions_scaled_da_national.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    # Check values: scaled version of organic_rf_impressions_raw_da_national
    organic_rf_impressions_raw_da_national = (
        engine.organic_rf_impressions_raw_da_national
    )
    self.assertIsInstance(organic_rf_impressions_raw_da_national, xr.DataArray)
    expected_values = (
        organic_rf_impressions_raw_da_national.values * self.mock_scale_factor
    )
    self.assertAllClose(
        organic_rf_impressions_scaled_da_national.values, expected_values
    )

  def test_organic_rf_impressions_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    organic_rf_impressions_scaled_da_national = (
        engine.organic_rf_impressions_scaled_da_national
    )
    self.assertIsInstance(
        organic_rf_impressions_scaled_da_national, xr.DataArray
    )
    self.assertEqual(
        organic_rf_impressions_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    expected_organic_rf_impressions_scaled_da = (
        engine.organic_rf_impressions_scaled_da
    )
    self.assertIsInstance(
        expected_organic_rf_impressions_scaled_da, xr.DataArray
    )
    self.assertAllClose(
        organic_rf_impressions_scaled_da_national.values,
        expected_organic_rf_impressions_scaled_da.values,
    )

  # --- Test cases for geo_population_da ---
  def test_geo_population_da_present(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    population_da = engine.geo_population_da
    self.assertIsInstance(population_da, xr.DataArray)
    self.assertEqual(
        population_da.shape,
        (model_test_data.WithInputDataSamples._N_GEOS,),
    )
    self.assertCountEqual(population_da.coords.keys(), [constants.GEO])
    self.assertAllClose(population_da.values, meridian.population)

  # --- Test cases for kpi_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
          ),
      ),
      dict(
          testcase_name="national",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
          ),
      ),
  )
  def test_kpi_scaled_da_present(self, input_data_fixture, expected_shape):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    kpi_da = engine.kpi_scaled_da
    self.assertIsInstance(kpi_da, xr.DataArray)
    self.assertEqual(kpi_da.shape, expected_shape)
    self.assertCountEqual(kpi_da.coords.keys(), [constants.GEO, constants.TIME])
    self.assertAllClose(kpi_da.values, meridian.kpi_scaled)

  # --- Test cases for kpi_scaled_da_national ---
  def test_kpi_scaled_da_national_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    kpi_scaled_da_national = engine.kpi_scaled_da_national
    self.assertIsInstance(kpi_scaled_da_national, xr.DataArray)
    self.assertEqual(
        kpi_scaled_da_national.shape,
        (model_test_data.WithInputDataSamples._N_TIMES,),
    )
    self.assertCountEqual(
        kpi_scaled_da_national.coords.keys(), [constants.TIME]
    )

    # Check values
    expected_da = meridian.input_data.kpi.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(kpi_scaled_da_national.values, scaled_expected_values)

  def test_kpi_scaled_da_national_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    kpi_scaled_da_national = engine.kpi_scaled_da_national
    self.assertIsInstance(kpi_scaled_da_national, xr.DataArray)
    self.assertEqual(
        kpi_scaled_da_national.shape,
        (
            model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
            model_test_data.WithInputDataSamples._N_TIMES,
        ),
    )
    self.assertAllClose(
        kpi_scaled_da_national.values, engine.kpi_scaled_da.values
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="controls_scaled_da",
          input_data_fixture="input_data_with_media_and_rf_no_controls",
          property_name="controls_scaled_da",
      ),
      dict(
          testcase_name="controls_scaled_da_national",
          input_data_fixture="input_data_with_media_and_rf_no_controls",
          property_name="controls_scaled_da_national",
      ),
      dict(
          testcase_name="media_raw_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_raw_da",
      ),
      dict(
          testcase_name="media_scaled_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_scaled_da",
      ),
      dict(
          testcase_name="media_raw_da_national",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_raw_da_national",
      ),
      dict(
          testcase_name="media_scaled_da_national",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_scaled_da_national",
      ),
      dict(
          testcase_name="media_spend_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_spend_da",
      ),
      dict(
          testcase_name="media_spend_da_national",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_spend_da_national",
      ),
      dict(
          testcase_name="organic_media_raw_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="organic_media_raw_da",
      ),
      dict(
          testcase_name="organic_media_scaled_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="organic_media_scaled_da",
      ),
      dict(
          testcase_name="organic_media_raw_da_national",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="organic_media_raw_da_national",
      ),
      dict(
          testcase_name="organic_media_scaled_da_national",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="organic_media_scaled_da_national",
      ),
      dict(
          testcase_name="non_media_scaled_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="non_media_scaled_da",
      ),
      dict(
          testcase_name="non_media_scaled_da_national",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="non_media_scaled_da_national",
      ),
      dict(
          testcase_name="rf_spend_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_spend_da",
      ),
      dict(
          testcase_name="rf_spend_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_spend_da_national",
      ),
      dict(
          testcase_name="reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_raw_da",
      ),
      dict(
          testcase_name="reach_raw_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_raw_da_national",
      ),
      dict(
          testcase_name="reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_scaled_da",
      ),
      dict(
          testcase_name="reach_scaled_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_scaled_da_national",
      ),
      dict(
          testcase_name="frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="frequency_da",
      ),
      dict(
          testcase_name="frequency_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="frequency_da_national",
      ),
      dict(
          testcase_name="rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_raw_da",
      ),
      dict(
          testcase_name="rf_impressions_raw_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_raw_da_national",
      ),
      dict(
          testcase_name="rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="rf_impressions_scaled_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_scaled_da_national",
      ),
      dict(
          testcase_name="organic_reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_raw_da",
      ),
      dict(
          testcase_name="organic_reach_raw_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_raw_da_national",
      ),
      dict(
          testcase_name="organic_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_scaled_da",
      ),
      dict(
          testcase_name="organic_reach_scaled_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_scaled_da_national",
      ),
      dict(
          testcase_name="organic_frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_frequency_da",
      ),
      dict(
          testcase_name="organic_frequency_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_frequency_da_national",
      ),
      dict(
          testcase_name="organic_rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_raw_da",
      ),
      dict(
          testcase_name="organic_rf_impressions_raw_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_raw_da_national",
      ),
      dict(
          testcase_name="organic_rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="organic_rf_impressions_scaled_da_national",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_scaled_da_national",
      ),
      dict(
          testcase_name="geo_population_da",
          input_data_fixture="national_input_data_media_and_rf",
          property_name="geo_population_da",
      ),
  )
  def test_property_absent(self, input_data_fixture, property_name):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    self.assertIsNone(getattr(engine, property_name))

  @parameterized.named_parameters(
      dict(
          testcase_name="n_media_times_gt_n_times",
          input_data_fixture="input_data_non_media_and_organic",
      ),
      dict(
          testcase_name="n_media_times_eq_n_times",
          input_data_fixture="input_data_non_media_and_organic_same_time_dims",
      ),
  )
  def test_properties_are_truncated(self, input_data_fixture):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)

    properties_to_test = [
        engine.media_raw_da,
        engine.media_raw_da_national,
        engine.media_scaled_da,
        engine.media_scaled_da_national,
        engine.organic_media_raw_da,
        engine.organic_media_raw_da_national,
        engine.organic_media_scaled_da,
        engine.organic_media_scaled_da_national,
        engine.reach_raw_da,
        engine.reach_raw_da_national,
        engine.reach_scaled_da,
        engine.reach_scaled_da_national,
        engine.frequency_da,
        engine.frequency_da_national,
        engine.rf_impressions_raw_da,
        engine.rf_impressions_raw_da_national,
        engine.rf_impressions_scaled_da,
        engine.rf_impressions_scaled_da_national,
        engine.organic_reach_raw_da,
        engine.organic_reach_raw_da_national,
        engine.organic_reach_scaled_da,
        engine.organic_reach_scaled_da_national,
        engine.organic_frequency_da,
        engine.organic_frequency_da_national,
        engine.organic_rf_impressions_raw_da,
        engine.organic_rf_impressions_raw_da_national,
        engine.organic_rf_impressions_scaled_da,
        engine.organic_rf_impressions_scaled_da_national,
    ]

    for prop in properties_to_test:
      self.assertIsInstance(prop, xr.DataArray)
      self.assertEqual(
          prop.sizes[constants.TIME],
          model_test_data.WithInputDataSamples._N_TIMES,
      )
      self.assertNotIn(constants.MEDIA_TIME, prop.coords)
      self.assertIn(constants.TIME, prop.coords)


if __name__ == "__main__":
  absltest.main()
