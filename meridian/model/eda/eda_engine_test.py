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
from meridian.model.eda import eda_outcome
import numpy as np
import pandas as pd
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
    self.assertEqual(controls_scaled_da.name, constants.CONTROLS_SCALED)
    self.assertEqual(controls_scaled_da.shape, expected_shape)
    self.assertCountEqual(
        controls_scaled_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
    )
    self.assertAllClose(controls_scaled_da.values, meridian.controls_scaled)

  # --- Test cases for national_controls_scaled_da ---
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
  def test_national_controls_scaled_da_with_geo_data(
      self, agg_config, expected_values_func
  ):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian, agg_config=agg_config)

    national_controls_scaled_da = engine.national_controls_scaled_da
    self.assertIsInstance(national_controls_scaled_da, xr.DataArray)
    self.assertEqual(
        national_controls_scaled_da.name, constants.NATIONAL_CONTROLS_SCALED
    )
    self.assertEqual(
        national_controls_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_CONTROLS,
        ),
    )
    self.assertCountEqual(
        national_controls_scaled_da.coords.keys(),
        [constants.TIME, constants.CONTROL_VARIABLE],
    )

    # Check values
    self.assertIsInstance(meridian.input_data.controls, xr.DataArray)
    expected_da = expected_values_func(meridian.input_data.controls)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        national_controls_scaled_da.values, scaled_expected_values
    )

  def test_national_controls_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_controls_scaled_da = engine.national_controls_scaled_da
    self.assertIsInstance(national_controls_scaled_da, xr.DataArray)
    self.assertEqual(
        national_controls_scaled_da.name, constants.NATIONAL_CONTROLS_SCALED
    )
    self.assertEqual(
        national_controls_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_CONTROLS,
        ),
    )
    self.assertCountEqual(
        national_controls_scaled_da.coords.keys(),
        [constants.TIME, constants.CONTROL_VARIABLE],
    )
    expected_controls_scaled_da = engine.controls_scaled_da
    self.assertIsNotNone(expected_controls_scaled_da)
    expected_controls_scaled_da = expected_controls_scaled_da.squeeze(
        constants.GEO
    )
    self.assertIsInstance(expected_controls_scaled_da, xr.DataArray)
    self.assertAllClose(
        national_controls_scaled_da.values,
        expected_controls_scaled_da.values,
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
    self.assertEqual(media_da.name, constants.MEDIA)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_raw_media_da = meridian.input_data.media
    self.assertIsInstance(true_raw_media_da, xr.DataArray)
    self.assertAllClose(media_da.values, true_raw_media_da.values[:, start:, :])

  # --- Test cases for national_media_raw_da ---
  def test_national_media_raw_da_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)
    national_media_raw_da = engine.national_media_raw_da
    self.assertIsInstance(national_media_raw_da, xr.DataArray)
    self.assertEqual(national_media_raw_da.name, constants.NATIONAL_MEDIA)
    self.assertEqual(
        national_media_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_raw_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_raw_media_da = meridian.input_data.media
    self.assertIsNotNone(true_raw_media_da)
    expected_da = true_raw_media_da.sum(dim=constants.GEO)
    self.assertAllClose(national_media_raw_da.values, expected_da.values)

  def test_national_media_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_media_raw_da = engine.national_media_raw_da
    self.assertIsInstance(national_media_raw_da, xr.DataArray)
    self.assertEqual(national_media_raw_da.name, constants.NATIONAL_MEDIA)
    self.assertEqual(
        national_media_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_raw_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )
    expected_media_raw_da = engine.media_raw_da
    self.assertIsNotNone(expected_media_raw_da)
    expected_media_raw_da = expected_media_raw_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_media_raw_da, xr.DataArray)
    self.assertAllClose(
        national_media_raw_da.values, expected_media_raw_da.values
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
    self.assertEqual(media_da.name, constants.MEDIA_SCALED)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        media_da.values, meridian.media_tensors.media_scaled[:, start:, :]
    )

  # --- Test cases for national_media_scaled_da ---
  def test_national_media_scaled_da_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)

    national_media_scaled_da = engine.national_media_scaled_da
    self.assertIsInstance(national_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_media_scaled_da.name, constants.NATIONAL_MEDIA_SCALED
    )
    self.assertEqual(
        national_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_scaled_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_raw_media_da = meridian.input_data.media
    self.assertIsNotNone(true_raw_media_da)
    expected_da = true_raw_media_da.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        national_media_scaled_da.values,
        scaled_expected_values,
    )

  def test_national_media_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_media_scaled_da = engine.national_media_scaled_da
    self.assertIsInstance(national_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_media_scaled_da.name, constants.NATIONAL_MEDIA_SCALED
    )
    self.assertEqual(
        national_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_scaled_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )
    expected_media_scaled_da = engine.media_scaled_da
    self.assertIsNotNone(expected_media_scaled_da)
    expected_media_scaled_da = expected_media_scaled_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        national_media_scaled_da.values, expected_media_scaled_da.values
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
    self.assertEqual(media_da.name, constants.MEDIA_SPEND)
    self.assertEqual(media_da.shape, expected_shape)
    self.assertCountEqual(
        media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )
    self.assertAllClose(media_da.values, meridian.media_tensors.media_spend)

  # --- Test cases for national_media_spend_da ---
  def test_national_media_spend_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_media_spend_da = engine.national_media_spend_da
    self.assertIsInstance(national_media_spend_da, xr.DataArray)
    self.assertEqual(
        national_media_spend_da.name, constants.NATIONAL_MEDIA_SPEND
    )
    self.assertEqual(
        national_media_spend_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_spend_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )

    # Check values
    true_media_spend_da = meridian.input_data.media_spend
    self.assertIsInstance(true_media_spend_da, xr.DataArray)
    expected_da = true_media_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(national_media_spend_da.values, expected_da.values)

  def test_national_media_spend_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_media_spend_da = engine.national_media_spend_da
    self.assertIsInstance(national_media_spend_da, xr.DataArray)
    self.assertEqual(
        national_media_spend_da.name, constants.NATIONAL_MEDIA_SPEND
    )
    self.assertEqual(
        national_media_spend_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_media_spend_da.coords.keys(),
        [constants.TIME, constants.MEDIA_CHANNEL],
    )
    expected_media_spend_da = engine.media_spend_da
    self.assertIsNotNone(expected_media_spend_da)
    expected_media_spend_da = expected_media_spend_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_media_spend_da, xr.DataArray)
    self.assertAllClose(
        national_media_spend_da.values, expected_media_spend_da.values
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
    self.assertEqual(organic_media_da.name, constants.ORGANIC_MEDIA)
    self.assertEqual(organic_media_da.shape, expected_shape)
    self.assertCountEqual(
        organic_media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_raw_organic_media_da = meridian.input_data.organic_media
    self.assertIsInstance(true_raw_organic_media_da, xr.DataArray)
    self.assertAllClose(
        organic_media_da.values, true_raw_organic_media_da.values[:, start:, :]
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
    self.assertEqual(organic_media_da.name, constants.ORGANIC_MEDIA_SCALED)
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

  # --- Test cases for national_organic_media_raw_da ---
  def test_national_organic_media_raw_da_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)
    national_organic_media_raw_da = engine.national_organic_media_raw_da
    self.assertIsInstance(national_organic_media_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_media_raw_da.name, constants.NATIONAL_ORGANIC_MEDIA
    )
    self.assertEqual(
        national_organic_media_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_media_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )

    # Check values
    true_organic_media_raw_da = meridian.input_data.organic_media
    self.assertIsNotNone(true_organic_media_raw_da)
    expected_da = true_organic_media_raw_da.sum(dim=constants.GEO)
    self.assertAllClose(
        national_organic_media_raw_da.values, expected_da.values
    )

  def test_national_organic_media_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_media_raw_da = engine.national_organic_media_raw_da
    self.assertIsInstance(national_organic_media_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_media_raw_da.name, constants.NATIONAL_ORGANIC_MEDIA
    )
    self.assertEqual(
        national_organic_media_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_media_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )
    expected_organic_media_raw_da = engine.organic_media_raw_da
    self.assertIsNotNone(expected_organic_media_raw_da)
    expected_organic_media_raw_da = expected_organic_media_raw_da.squeeze(
        constants.GEO
    )
    self.assertIsInstance(expected_organic_media_raw_da, xr.DataArray)
    self.assertAllClose(
        national_organic_media_raw_da.values,
        expected_organic_media_raw_da.values,
    )

  # --- Test cases for national_organic_media_scaled_da ---
  def test_national_organic_media_scaled_da_with_geo_data(self):
    meridian = model.Meridian(
        self.input_data_non_media_and_organic_same_time_dims
    )
    engine = eda_engine.EDAEngine(meridian)

    national_organic_media_scaled_da = engine.national_organic_media_scaled_da
    self.assertIsInstance(national_organic_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_organic_media_scaled_da.name,
        constants.NATIONAL_ORGANIC_MEDIA_SCALED,
    )
    self.assertEqual(
        national_organic_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_media_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )

    # Check values
    true_organic_media_raw_da = meridian.input_data.organic_media
    self.assertIsNotNone(true_organic_media_raw_da)
    expected_da = true_organic_media_raw_da.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        national_organic_media_scaled_da.values, scaled_expected_values
    )

  def test_national_organic_media_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_media_scaled_da = engine.national_organic_media_scaled_da
    self.assertIsInstance(national_organic_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_organic_media_scaled_da.name,
        constants.NATIONAL_ORGANIC_MEDIA_SCALED,
    )
    self.assertEqual(
        national_organic_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_media_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_MEDIA_CHANNEL],
    )
    expected_organic_media_scaled_da = engine.organic_media_scaled_da
    self.assertIsNotNone(expected_organic_media_scaled_da)
    expected_organic_media_scaled_da = expected_organic_media_scaled_da.squeeze(
        constants.GEO
    )
    self.assertIsInstance(expected_organic_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        national_organic_media_scaled_da.values,
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
    self.assertEqual(non_media_da.name, constants.NON_MEDIA_TREATMENTS_SCALED)
    self.assertEqual(non_media_da.shape, expected_shape)
    self.assertCountEqual(
        non_media_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
    )
    self.assertAllClose(
        non_media_da.values, meridian.non_media_treatments_normalized
    )

  # --- Test cases for national_non_media_scaled_da ---
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
  def test_national_non_media_scaled_da_with_geo_data(
      self, agg_config, expected_values_func
  ):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian, agg_config=agg_config)

    national_non_media_scaled_da = engine.national_non_media_scaled_da
    self.assertIsInstance(national_non_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_non_media_scaled_da.name,
        constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
    )
    self.assertEqual(
        national_non_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_non_media_scaled_da.coords.keys(),
        [constants.TIME, constants.NON_MEDIA_CHANNEL],
    )

    # Check values
    self.assertIsInstance(
        meridian.input_data.non_media_treatments, xr.DataArray
    )
    expected_da = expected_values_func(meridian.input_data.non_media_treatments)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(
        national_non_media_scaled_da.values, scaled_expected_values
    )

  def test_national_non_media_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_non_media_scaled_da = engine.national_non_media_scaled_da
    self.assertIsInstance(national_non_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_non_media_scaled_da.name,
        constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
    )
    self.assertEqual(
        national_non_media_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_NON_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_non_media_scaled_da.coords.keys(),
        [constants.TIME, constants.NON_MEDIA_CHANNEL],
    )
    expected_non_media_scaled_da = engine.non_media_scaled_da
    self.assertIsNotNone(expected_non_media_scaled_da)
    expected_non_media_scaled_da = expected_non_media_scaled_da.squeeze(
        constants.GEO
    )
    self.assertIsInstance(expected_non_media_scaled_da, xr.DataArray)
    self.assertAllClose(
        national_non_media_scaled_da.values,
        expected_non_media_scaled_da.values,
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
    self.assertEqual(rf_spend_da.name, constants.RF_SPEND)
    self.assertEqual(rf_spend_da.shape, expected_shape)
    self.assertCountEqual(
        rf_spend_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    self.assertAllClose(rf_spend_da.values, meridian.rf_tensors.rf_spend)

  # --- Test cases for national_rf_spend_da ---
  def test_national_rf_spend_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_spend_da = engine.national_rf_spend_da
    self.assertIsInstance(national_rf_spend_da, xr.DataArray)
    self.assertEqual(national_rf_spend_da.name, constants.NATIONAL_RF_SPEND)
    self.assertEqual(
        national_rf_spend_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_spend_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    true_rf_spend_da = meridian.input_data.rf_spend
    self.assertIsNotNone(true_rf_spend_da)
    expected_da = true_rf_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(national_rf_spend_da.values, expected_da.values)

  def test_national_rf_spend_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_spend_da = engine.national_rf_spend_da
    self.assertIsInstance(national_rf_spend_da, xr.DataArray)
    self.assertEqual(national_rf_spend_da.name, constants.NATIONAL_RF_SPEND)
    self.assertEqual(
        national_rf_spend_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_spend_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_rf_spend_da = engine.rf_spend_da
    self.assertIsNotNone(expected_rf_spend_da)
    expected_rf_spend_da = expected_rf_spend_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_rf_spend_da, xr.DataArray)
    self.assertAllClose(
        national_rf_spend_da.values, expected_rf_spend_da.values
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
    self.assertEqual(reach_da.name, constants.REACH)
    self.assertEqual(reach_da.shape, expected_shape)
    self.assertCountEqual(
        reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    true_reach_da = meridian.input_data.reach
    self.assertIsInstance(true_reach_da, xr.DataArray)
    self.assertAllClose(reach_da.values, true_reach_da.values[:, start:, :])

  # --- Test cases for national_reach_raw_da ---
  def test_national_reach_raw_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_reach_raw_da = engine.national_reach_raw_da
    self.assertIsInstance(national_reach_raw_da, xr.DataArray)
    self.assertEqual(national_reach_raw_da.name, constants.NATIONAL_REACH)
    self.assertEqual(
        national_reach_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_reach_raw_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    reach_raw_da = engine.reach_raw_da
    self.assertIsInstance(reach_raw_da, xr.DataArray)
    expected_values = reach_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(national_reach_raw_da.values, expected_values)

  def test_national_reach_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_reach_raw_da = engine.national_reach_raw_da
    self.assertIsInstance(national_reach_raw_da, xr.DataArray)
    self.assertEqual(national_reach_raw_da.name, constants.NATIONAL_REACH)
    self.assertEqual(
        national_reach_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_reach_raw_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_reach_raw_da = engine.reach_raw_da
    self.assertIsNotNone(expected_reach_raw_da)
    expected_reach_raw_da = expected_reach_raw_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_reach_raw_da, xr.DataArray)
    self.assertAllClose(
        national_reach_raw_da.values, expected_reach_raw_da.values
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
    self.assertEqual(reach_da.name, constants.REACH_SCALED)
    self.assertEqual(reach_da.shape, expected_shape)
    self.assertCountEqual(
        reach_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        reach_da.values, meridian.rf_tensors.reach_scaled[:, start:, :]
    )

  # --- Test cases for national_reach_scaled_da ---
  def test_national_reach_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    national_reach_scaled_da = engine.national_reach_scaled_da
    self.assertIsInstance(national_reach_scaled_da, xr.DataArray)
    self.assertEqual(
        national_reach_scaled_da.name, constants.NATIONAL_REACH_SCALED
    )
    self.assertEqual(
        national_reach_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_reach_scaled_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    reach_raw_da = engine.reach_raw_da
    self.assertIsInstance(reach_raw_da, xr.DataArray)
    national_reach_raw_da = reach_raw_da.sum(dim=constants.GEO)
    # Scale the raw values by the mock scale factor
    expected_values = national_reach_raw_da.values * self.mock_scale_factor
    self.assertAllClose(national_reach_scaled_da.values, expected_values)

  def test_national_reach_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_reach_scaled_da = engine.national_reach_scaled_da
    self.assertIsInstance(national_reach_scaled_da, xr.DataArray)
    self.assertEqual(
        national_reach_scaled_da.name, constants.NATIONAL_REACH_SCALED
    )
    self.assertEqual(
        national_reach_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_reach_scaled_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_reach_scaled_da = engine.reach_scaled_da
    self.assertIsNotNone(expected_reach_scaled_da)
    expected_reach_scaled_da = expected_reach_scaled_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_reach_scaled_da, xr.DataArray)
    expected_values = expected_reach_scaled_da.values
    self.assertAllClose(national_reach_scaled_da.values, expected_values)

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
    self.assertEqual(frequency_da.name, constants.FREQUENCY)
    self.assertEqual(frequency_da.shape, expected_shape)
    self.assertCountEqual(
        frequency_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )
    start = meridian.n_media_times - meridian.n_times
    self.assertAllClose(
        frequency_da.values, meridian.rf_tensors.frequency[:, start:, :]
    )

  # --- Test cases for national_frequency_da ---
  def test_national_frequency_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_frequency_da = engine.national_frequency_da
    self.assertIsInstance(national_frequency_da, xr.DataArray)
    self.assertEqual(national_frequency_da.name, constants.NATIONAL_FREQUENCY)
    self.assertEqual(
        national_frequency_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_frequency_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

    # Check values
    national_reach_raw_da = engine.national_reach_raw_da
    self.assertIsInstance(national_reach_raw_da, xr.DataArray)
    expected_national_rf_impressions_raw_da = (
        engine.national_rf_impressions_raw_da
    )
    self.assertIsInstance(expected_national_rf_impressions_raw_da, xr.DataArray)

    actual_rf_impressions_raw_da = national_reach_raw_da * national_frequency_da
    self.assertAllClose(
        actual_rf_impressions_raw_da.values,
        expected_national_rf_impressions_raw_da.values,
    )

  def test_national_frequency_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_frequency_da = engine.national_frequency_da
    self.assertIsInstance(national_frequency_da, xr.DataArray)
    self.assertEqual(national_frequency_da.name, constants.NATIONAL_FREQUENCY)
    self.assertEqual(
        national_frequency_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_frequency_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_frequency_da = engine.frequency_da
    self.assertIsNotNone(expected_frequency_da)
    expected_frequency_da = expected_frequency_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_frequency_da, xr.DataArray)
    self.assertAllClose(
        national_frequency_da.values, expected_frequency_da.values
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
    self.assertEqual(rf_impressions_raw_da.name, constants.RF_IMPRESSIONS)
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

  # --- Test cases for national_rf_impressions_raw_da ---
  def test_national_rf_impressions_raw_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_raw_da = engine.national_rf_impressions_raw_da
    self.assertIsInstance(national_rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(
        national_rf_impressions_raw_da.name, constants.NATIONAL_RF_IMPRESSIONS
    )
    self.assertEqual(
        national_rf_impressions_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_impressions_raw_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    # Check values: sum of geo-level raw impressions
    rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsInstance(rf_impressions_raw_da, xr.DataArray)
    expected_values = rf_impressions_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(
        national_rf_impressions_raw_da.values, expected_values
    )

  def test_national_rf_impressions_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_raw_da = engine.national_rf_impressions_raw_da
    self.assertIsInstance(national_rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(
        national_rf_impressions_raw_da.name, constants.NATIONAL_RF_IMPRESSIONS
    )
    self.assertEqual(
        national_rf_impressions_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_impressions_raw_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsNotNone(expected_rf_impressions_raw_da)
    expected_rf_impressions_raw_da = expected_rf_impressions_raw_da.squeeze(
        constants.GEO
    )
    self.assertIsInstance(expected_rf_impressions_raw_da, xr.DataArray)
    self.assertAllClose(
        national_rf_impressions_raw_da.values,
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
    self.assertEqual(
        rf_impressions_scaled_da.name, constants.RF_IMPRESSIONS_SCALED
    )
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

  # --- Test cases for national_rf_impressions_scaled_da ---
  def test_national_rf_impressions_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_scaled_da = (
        engine.national_rf_impressions_scaled_da
    )
    self.assertIsInstance(national_rf_impressions_scaled_da, xr.DataArray)
    self.assertEqual(
        national_rf_impressions_scaled_da.name,
        constants.NATIONAL_RF_IMPRESSIONS_SCALED,
    )
    self.assertEqual(
        national_rf_impressions_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_impressions_scaled_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    # Check values: scaled version of national_rf_impressions_raw_da
    national_rf_impressions_raw_da = engine.national_rf_impressions_raw_da
    self.assertIsInstance(national_rf_impressions_raw_da, xr.DataArray)
    expected_values = (
        national_rf_impressions_raw_da.values * self.mock_scale_factor
    )
    self.assertAllClose(
        national_rf_impressions_scaled_da.values, expected_values
    )

  def test_national_rf_impressions_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_scaled_da = (
        engine.national_rf_impressions_scaled_da
    )
    self.assertIsInstance(national_rf_impressions_scaled_da, xr.DataArray)
    self.assertEqual(
        national_rf_impressions_scaled_da.name,
        constants.NATIONAL_RF_IMPRESSIONS_SCALED,
    )
    self.assertEqual(
        national_rf_impressions_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_rf_impressions_scaled_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )
    expected_rf_impressions_scaled_da = engine.rf_impressions_scaled_da
    self.assertIsNotNone(expected_rf_impressions_scaled_da)
    expected_rf_impressions_scaled_da = (
        expected_rf_impressions_scaled_da.squeeze(constants.GEO)
    )
    self.assertIsInstance(expected_rf_impressions_scaled_da, xr.DataArray)
    self.assertAllClose(
        national_rf_impressions_scaled_da.values,
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
    self.assertEqual(organic_reach_da.name, constants.ORGANIC_REACH)
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

  # --- Test cases for national_organic_reach_raw_da ---
  def test_national_organic_reach_raw_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_reach_raw_da = engine.national_organic_reach_raw_da
    self.assertIsInstance(national_organic_reach_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_reach_raw_da.name, constants.NATIONAL_ORGANIC_REACH
    )
    self.assertEqual(
        national_organic_reach_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_reach_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    organic_reach_raw_da = engine.organic_reach_raw_da
    self.assertIsInstance(organic_reach_raw_da, xr.DataArray)
    expected_values = organic_reach_raw_da.sum(dim=constants.GEO).values
    self.assertAllClose(national_organic_reach_raw_da.values, expected_values)

  def test_national_organic_reach_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_reach_raw_da = engine.national_organic_reach_raw_da
    self.assertIsInstance(national_organic_reach_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_reach_raw_da.name, constants.NATIONAL_ORGANIC_REACH
    )
    self.assertEqual(
        national_organic_reach_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_reach_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    expected_da = engine.organic_reach_raw_da
    self.assertIsNotNone(expected_da)
    expected_da = expected_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_da, xr.DataArray)
    self.assertAllClose(
        national_organic_reach_raw_da.values, expected_da.values
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
    self.assertEqual(organic_reach_da.name, constants.ORGANIC_REACH_SCALED)
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

  # --- Test cases for national_organic_reach_scaled_da ---
  def test_national_organic_reach_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)

    national_organic_reach_scaled_da = engine.national_organic_reach_scaled_da
    self.assertIsInstance(national_organic_reach_scaled_da, xr.DataArray)
    self.assertEqual(
        national_organic_reach_scaled_da.name,
        constants.NATIONAL_ORGANIC_REACH_SCALED,
    )
    self.assertEqual(
        national_organic_reach_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_reach_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    organic_reach_raw_da = engine.organic_reach_raw_da
    self.assertIsInstance(organic_reach_raw_da, xr.DataArray)
    national_organic_reach_raw_da = organic_reach_raw_da.sum(dim=constants.GEO)
    # Scale the raw values by the mock scale factor
    expected_values = (
        national_organic_reach_raw_da.values * self.mock_scale_factor
    )
    self.assertAllClose(
        national_organic_reach_scaled_da.values, expected_values
    )

  def test_national_organic_reach_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_reach_scaled_da = engine.national_organic_reach_scaled_da
    self.assertIsInstance(national_organic_reach_scaled_da, xr.DataArray)
    self.assertEqual(
        national_organic_reach_scaled_da.name,
        constants.NATIONAL_ORGANIC_REACH_SCALED,
    )
    self.assertEqual(
        national_organic_reach_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_reach_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    expected_da = engine.organic_reach_scaled_da
    self.assertIsNotNone(expected_da)
    expected_da = expected_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_da, xr.DataArray)

    self.assertAllClose(
        national_organic_reach_scaled_da.values,
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
    self.assertEqual(organic_frequency_da.name, constants.ORGANIC_FREQUENCY)
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

  # --- Test cases for national_organic_frequency_da ---
  def test_national_organic_frequency_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_frequency_da = engine.national_organic_frequency_da
    self.assertIsInstance(national_organic_frequency_da, xr.DataArray)
    self.assertEqual(
        national_organic_frequency_da.name,
        constants.NATIONAL_ORGANIC_FREQUENCY,
    )
    self.assertEqual(
        national_organic_frequency_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_frequency_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    # Check values
    national_organic_reach_raw_da = engine.national_organic_reach_raw_da
    self.assertIsInstance(national_organic_reach_raw_da, xr.DataArray)
    expected_national_organic_rf_impressions_raw_da = (
        engine.national_organic_rf_impressions_raw_da
    )
    self.assertIsInstance(
        expected_national_organic_rf_impressions_raw_da, xr.DataArray
    )

    actual_organic_rf_impressions_raw_da = (
        national_organic_reach_raw_da * national_organic_frequency_da
    )
    self.assertAllClose(
        actual_organic_rf_impressions_raw_da.values,
        expected_national_organic_rf_impressions_raw_da.values,
    )

  def test_national_organic_frequency_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_frequency_da = engine.national_organic_frequency_da
    self.assertIsInstance(national_organic_frequency_da, xr.DataArray)
    self.assertEqual(
        national_organic_frequency_da.name,
        constants.NATIONAL_ORGANIC_FREQUENCY,
    )
    self.assertEqual(
        national_organic_frequency_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_frequency_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )

    expected_da = engine.organic_frequency_da
    self.assertIsNotNone(expected_da)
    expected_da = expected_da.squeeze(constants.GEO)
    self.assertIsInstance(expected_da, xr.DataArray)
    self.assertAllClose(
        national_organic_frequency_da.values, expected_da.values
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
    self.assertEqual(
        organic_rf_impressions_raw_da.name, constants.ORGANIC_RF_IMPRESSIONS
    )
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

  # --- Test cases for national_organic_rf_impressions_raw_da ---
  def test_national_organic_rf_impressions_raw_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_rf_impressions_raw_da = (
        engine.national_organic_rf_impressions_raw_da
    )
    self.assertIsInstance(national_organic_rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_rf_impressions_raw_da.name,
        constants.NATIONAL_ORGANIC_RF_IMPRESSIONS,
    )
    self.assertEqual(
        national_organic_rf_impressions_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_rf_impressions_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    # Check values: sum of geo-level raw impressions
    organic_rf_impressions_raw_da = engine.organic_rf_impressions_raw_da
    self.assertIsInstance(organic_rf_impressions_raw_da, xr.DataArray)
    expected_values = organic_rf_impressions_raw_da.sum(
        dim=constants.GEO
    ).values
    self.assertAllClose(
        national_organic_rf_impressions_raw_da.values, expected_values
    )

  def test_national_organic_rf_impressions_raw_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_rf_impressions_raw_da = (
        engine.national_organic_rf_impressions_raw_da
    )
    self.assertIsInstance(national_organic_rf_impressions_raw_da, xr.DataArray)
    self.assertEqual(
        national_organic_rf_impressions_raw_da.name,
        constants.NATIONAL_ORGANIC_RF_IMPRESSIONS,
    )
    self.assertEqual(
        national_organic_rf_impressions_raw_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_rf_impressions_raw_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    expected_organic_rf_impressions_raw_da = (
        engine.organic_rf_impressions_raw_da
    )
    self.assertIsNotNone(expected_organic_rf_impressions_raw_da)
    expected_organic_rf_impressions_raw_da = (
        expected_organic_rf_impressions_raw_da.squeeze(constants.GEO)
    )
    self.assertIsInstance(
        expected_organic_rf_impressions_raw_da, xr.DataArray
    )
    self.assertAllClose(
        national_organic_rf_impressions_raw_da.values,
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
    self.assertEqual(
        organic_rf_impressions_scaled_da.name,
        constants.ORGANIC_RF_IMPRESSIONS_SCALED,
    )
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

  # --- Test cases for national_organic_rf_impressions_scaled_da ---
  def test_national_organic_rf_impressions_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_rf_impressions_scaled_da = (
        engine.national_organic_rf_impressions_scaled_da
    )
    self.assertIsInstance(
        national_organic_rf_impressions_scaled_da, xr.DataArray
    )
    self.assertEqual(
        national_organic_rf_impressions_scaled_da.name,
        constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED,
    )
    self.assertEqual(
        national_organic_rf_impressions_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_rf_impressions_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    # Check values: scaled version of national_organic_rf_impressions_raw_da
    national_organic_rf_impressions_raw_da = (
        engine.national_organic_rf_impressions_raw_da
    )
    self.assertIsInstance(national_organic_rf_impressions_raw_da, xr.DataArray)
    expected_values = (
        national_organic_rf_impressions_raw_da.values * self.mock_scale_factor
    )
    self.assertAllClose(
        national_organic_rf_impressions_scaled_da.values,
        expected_values,
    )

  def test_national_organic_rf_impressions_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_non_media_and_organic)
    engine = eda_engine.EDAEngine(meridian)
    national_organic_rf_impressions_scaled_da = (
        engine.national_organic_rf_impressions_scaled_da
    )
    self.assertIsInstance(
        national_organic_rf_impressions_scaled_da, xr.DataArray
    )
    self.assertEqual(
        national_organic_rf_impressions_scaled_da.name,
        constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED,
    )
    self.assertEqual(
        national_organic_rf_impressions_scaled_da.shape,
        (
            model_test_data.WithInputDataSamples._N_TIMES,
            model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        national_organic_rf_impressions_scaled_da.coords.keys(),
        [constants.TIME, constants.ORGANIC_RF_CHANNEL],
    )
    expected_organic_rf_impressions_scaled_da = (
        engine.organic_rf_impressions_scaled_da
    )
    self.assertIsNotNone(expected_organic_rf_impressions_scaled_da)
    expected_organic_rf_impressions_scaled_da = (
        expected_organic_rf_impressions_scaled_da.squeeze(constants.GEO)
    )
    self.assertIsInstance(
        expected_organic_rf_impressions_scaled_da, xr.DataArray
    )
    self.assertAllClose(
        national_organic_rf_impressions_scaled_da.values,
        expected_organic_rf_impressions_scaled_da.values,
    )

  # --- Test cases for geo_population_da ---
  def test_geo_population_da_present(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    population_da = engine.geo_population_da
    self.assertIsInstance(population_da, xr.DataArray)
    self.assertEqual(population_da.name, constants.POPULATION)
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
    self.assertEqual(kpi_da.name, constants.KPI_SCALED)
    self.assertEqual(kpi_da.shape, expected_shape)
    self.assertCountEqual(kpi_da.coords.keys(), [constants.GEO, constants.TIME])
    self.assertAllClose(kpi_da.values, meridian.kpi_scaled)

  # --- Test cases for national_kpi_scaled_da ---
  def test_national_kpi_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    national_kpi_scaled_da = engine.national_kpi_scaled_da
    self.assertIsInstance(national_kpi_scaled_da, xr.DataArray)
    self.assertEqual(national_kpi_scaled_da.name, constants.NATIONAL_KPI_SCALED)
    self.assertEqual(
        national_kpi_scaled_da.shape,
        (model_test_data.WithInputDataSamples._N_TIMES,),
    )
    self.assertCountEqual(
        national_kpi_scaled_da.coords.keys(), [constants.TIME]
    )

    # Check values
    expected_da = meridian.input_data.kpi.sum(dim=constants.GEO)
    scaled_expected_values = expected_da.values * self.mock_scale_factor
    self.assertAllClose(national_kpi_scaled_da.values, scaled_expected_values)

  def test_national_kpi_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_kpi_scaled_da = engine.national_kpi_scaled_da
    self.assertIsInstance(national_kpi_scaled_da, xr.DataArray)
    self.assertEqual(national_kpi_scaled_da.name, constants.NATIONAL_KPI_SCALED)
    self.assertEqual(
        national_kpi_scaled_da.shape,
        (model_test_data.WithInputDataSamples._N_TIMES,),
    )
    self.assertCountEqual(
        national_kpi_scaled_da.coords.keys(),
        [constants.TIME],
    )
    self.assertAllClose(
        national_kpi_scaled_da.values,
        engine.kpi_scaled_da.squeeze(constants.GEO).values,
    )

  # --- Test cases for treatment_control_scaled_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.MEDIA_SCALED, constants.CONTROLS_SCALED],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.CONTROLS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
              constants.CONTROLS_SCALED,
          ],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.CONTROLS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="all_channels",
          input_data_fixture="input_data_non_media_and_organic",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
              constants.ORGANIC_MEDIA_SCALED,
              constants.ORGANIC_RF_IMPRESSIONS_SCALED,
              constants.CONTROLS_SCALED,
              constants.NON_MEDIA_TREATMENTS_SCALED,
          ],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.ORGANIC_MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.ORGANIC_MEDIA_CHANNEL,
              ],
              constants.ORGANIC_RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.ORGANIC_RF_CHANNEL,
              ],
              constants.CONTROLS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
              constants.NON_MEDIA_TREATMENTS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.NON_MEDIA_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="when_national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
              constants.CONTROLS_SCALED,
          ],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.CONTROLS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="when_national_all_channels",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
              constants.ORGANIC_MEDIA_SCALED,
              constants.ORGANIC_RF_IMPRESSIONS_SCALED,
              constants.CONTROLS_SCALED,
              constants.NON_MEDIA_TREATMENTS_SCALED,
          ],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.ORGANIC_MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.ORGANIC_MEDIA_CHANNEL,
              ],
              constants.ORGANIC_RF_IMPRESSIONS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.ORGANIC_RF_CHANNEL,
              ],
              constants.CONTROLS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
              constants.NON_MEDIA_TREATMENTS_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.NON_MEDIA_CHANNEL,
              ],
          },
      ),
  )
  def test_treatment_control_scaled_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    tc_scaled_ds = engine.treatment_control_scaled_ds
    self.assertIsInstance(tc_scaled_ds, xr.Dataset)

    self.assertCountEqual(tc_scaled_ds.data_vars.keys(), expected_vars)

    for var, dims in expected_dims.items():
      self.assertCountEqual(list(tc_scaled_ds[var].dims), dims)

  # --- Test cases for national_treatment_control_scaled_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_CONTROLS_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_CONTROLS_SCALED: [
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_CONTROLS_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.NATIONAL_CONTROLS_SCALED: [
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="all_channels",
          input_data_fixture="input_data_non_media_and_organic",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_ORGANIC_MEDIA_SCALED,
              constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_CONTROLS_SCALED,
              constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.NATIONAL_ORGANIC_MEDIA_SCALED: [
                  constants.TIME,
                  constants.ORGANIC_MEDIA_CHANNEL,
              ],
              constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.ORGANIC_RF_CHANNEL,
              ],
              constants.NATIONAL_CONTROLS_SCALED: [
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
              constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED: [
                  constants.TIME,
                  constants.NON_MEDIA_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_CONTROLS_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.NATIONAL_CONTROLS_SCALED: [
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
          },
      ),
      dict(
          testcase_name="national_all_channels",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_ORGANIC_MEDIA_SCALED,
              constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED,
              constants.NATIONAL_CONTROLS_SCALED,
              constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
              constants.NATIONAL_ORGANIC_MEDIA_SCALED: [
                  constants.TIME,
                  constants.ORGANIC_MEDIA_CHANNEL,
              ],
              constants.NATIONAL_ORGANIC_RF_IMPRESSIONS_SCALED: [
                  constants.TIME,
                  constants.ORGANIC_RF_CHANNEL,
              ],
              constants.NATIONAL_CONTROLS_SCALED: [
                  constants.TIME,
                  constants.CONTROL_VARIABLE,
              ],
              constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED: [
                  constants.TIME,
                  constants.NON_MEDIA_CHANNEL,
              ],
          },
      ),
  )
  def test_national_treatment_control_scaled_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_tc_scaled_ds = engine.national_treatment_control_scaled_ds
    self.assertIsInstance(national_tc_scaled_ds, xr.Dataset)

    self.assertCountEqual(
        national_tc_scaled_ds.data_vars.keys(),
        expected_vars,
    )

    for var in expected_vars:
      self.assertCountEqual(
          list(national_tc_scaled_ds[var].dims),
          expected_dims[var],
      )

  # --- Test cases for all_reach_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_paid_and_organic",
          input_data_fixture="input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              (
                  model_test_data.WithInputDataSamples._N_RF_CHANNELS
                  + model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS
              ),
          ),
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.reach_scaled_da,
                  engine.organic_reach_scaled_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="national_paid_and_organic",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              (
                  model_test_data.WithInputDataSamples._N_RF_CHANNELS
                  + model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS
              ),
          ),
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.reach_scaled_da,
                  engine.organic_reach_scaled_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="geo_paid_only",
          input_data_fixture="input_data_with_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
          expected_da_func=lambda engine: engine.reach_scaled_da,
      ),
      dict(
          testcase_name="national_paid_only",
          input_data_fixture="national_input_data_media_and_rf",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_RF_CHANNELS,
          ),
          expected_da_func=lambda engine: engine.reach_scaled_da,
      ),
      dict(
          testcase_name="geo_organic_only",
          input_data_fixture="input_data_organic_only",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
          expected_da_func=lambda engine: engine.organic_reach_scaled_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          ),
      ),
      dict(
          testcase_name="national_organic_only",
          input_data_fixture="national_input_data_organic_only",
          expected_shape=(
              model_test_data.WithInputDataSamples._N_GEOS_NATIONAL,
              model_test_data.WithInputDataSamples._N_TIMES,
              model_test_data.WithInputDataSamples._N_ORGANIC_RF_CHANNELS,
          ),
          expected_da_func=lambda engine: engine.organic_reach_scaled_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          ),
      ),
  )
  def test_all_reach_scaled_da_present(
      self, input_data_fixture, expected_shape, expected_da_func
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    all_reach_scaled_da = engine.all_reach_scaled_da

    self.assertIsInstance(all_reach_scaled_da, xr.DataArray)
    self.assertEqual(all_reach_scaled_da.name, constants.ALL_REACH_SCALED)

    expected_da = expected_da_func(engine)
    self.assertIsNotNone(expected_da)
    self.assertAllClose(all_reach_scaled_da.values, expected_da.values)

    self.assertEqual(all_reach_scaled_da.shape, expected_shape)

    self.assertCountEqual(
        all_reach_scaled_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )

  # --- Test cases for all_freq_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_paid_and_organic",
          input_data_fixture="input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.frequency_da,
                  engine.organic_frequency_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="national_paid_and_organic",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.frequency_da,
                  engine.organic_frequency_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="geo_paid_only",
          input_data_fixture="input_data_with_media_and_rf",
          expected_da_func=lambda engine: engine.frequency_da,
      ),
      dict(
          testcase_name="national_paid_only",
          input_data_fixture="national_input_data_media_and_rf",
          expected_da_func=lambda engine: engine.frequency_da,
      ),
      dict(
          testcase_name="geo_organic_only",
          input_data_fixture="input_data_organic_only",
          expected_da_func=lambda engine: engine.organic_frequency_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          ),
      ),
      dict(
          testcase_name="national_organic_only",
          input_data_fixture="national_input_data_organic_only",
          expected_da_func=lambda engine: engine.organic_frequency_da.rename(
              {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
          ),
      ),
  )
  def test_all_freq_da_present(self, input_data_fixture, expected_da_func):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    all_freq_da = engine.all_freq_da

    self.assertIsInstance(all_freq_da, xr.DataArray)
    self.assertEqual(all_freq_da.name, constants.ALL_FREQUENCY)

    expected_da = expected_da_func(engine)
    self.assertIsNotNone(expected_da)
    self.assertAllClose(all_freq_da.values, expected_da.values)

    self.assertCountEqual(
        all_freq_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )

  # --- Test cases for national_all_reach_scaled_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_paid_and_organic",
          input_data_fixture="input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.national_reach_scaled_da,
                  engine.national_organic_reach_scaled_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="national_paid_and_organic",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.national_reach_scaled_da,
                  engine.national_organic_reach_scaled_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="geo_paid_only",
          input_data_fixture="input_data_with_media_and_rf",
          expected_da_func=lambda engine: engine.national_reach_scaled_da,
      ),
      dict(
          testcase_name="national_paid_only",
          input_data_fixture="national_input_data_media_and_rf",
          expected_da_func=lambda engine: engine.national_reach_scaled_da,
      ),
      dict(
          testcase_name="geo_organic_only",
          input_data_fixture="input_data_organic_only",
          expected_da_func=lambda engine: (
              engine.national_organic_reach_scaled_da.rename(
                  {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
              )
          ),
      ),
      dict(
          testcase_name="national_organic_only",
          input_data_fixture="national_input_data_organic_only",
          expected_da_func=lambda engine: (
              engine.national_organic_reach_scaled_da.rename(
                  {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
              )
          ),
      ),
  )
  def test_national_all_reach_scaled_da_present(
      self, input_data_fixture, expected_da_func
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_all_reach_scaled_da = engine.national_all_reach_scaled_da

    self.assertIsInstance(national_all_reach_scaled_da, xr.DataArray)
    self.assertEqual(
        national_all_reach_scaled_da.name, constants.NATIONAL_ALL_REACH_SCALED
    )

    expected_da = expected_da_func(engine)
    self.assertIsNotNone(expected_da)
    self.assertAllClose(national_all_reach_scaled_da.values, expected_da.values)

    self.assertCountEqual(
        national_all_reach_scaled_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

  # --- Test cases for national_all_freq_da ---
  @parameterized.named_parameters(
      dict(
          testcase_name="geo_paid_and_organic",
          input_data_fixture="input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.national_frequency_da,
                  engine.national_organic_frequency_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="national_paid_and_organic",
          input_data_fixture="national_input_data_non_media_and_organic",
          expected_da_func=lambda engine: xr.concat(
              [
                  engine.national_frequency_da,
                  engine.national_organic_frequency_da.rename(
                      {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
                  ),
              ],
              dim=constants.RF_CHANNEL,
          ),
      ),
      dict(
          testcase_name="geo_paid_only",
          input_data_fixture="input_data_with_media_and_rf",
          expected_da_func=lambda engine: engine.national_frequency_da,
      ),
      dict(
          testcase_name="national_paid_only",
          input_data_fixture="national_input_data_media_and_rf",
          expected_da_func=lambda engine: engine.national_frequency_da,
      ),
      dict(
          testcase_name="geo_organic_only",
          input_data_fixture="input_data_organic_only",
          expected_da_func=lambda engine: (
              engine.national_organic_frequency_da.rename(
                  {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
              )
          ),
      ),
      dict(
          testcase_name="national_organic_only",
          input_data_fixture="national_input_data_organic_only",
          expected_da_func=lambda engine: (
              engine.national_organic_frequency_da.rename(
                  {constants.ORGANIC_RF_CHANNEL: constants.RF_CHANNEL}
              )
          ),
      ),
  )
  def test_national_all_freq_da_present(
      self, input_data_fixture, expected_da_func
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_all_freq_da = engine.national_all_freq_da

    self.assertIsInstance(national_all_freq_da, xr.DataArray)
    self.assertEqual(
        national_all_freq_da.name, constants.NATIONAL_ALL_FREQUENCY
    )

    expected_da = expected_da_func(engine)
    self.assertIsNotNone(expected_da)
    self.assertAllClose(national_all_freq_da.values, expected_da.values)

    self.assertCountEqual(
        national_all_freq_da.coords.keys(),
        [constants.TIME, constants.RF_CHANNEL],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="controls_scaled_da",
          input_data_fixture="input_data_with_media_and_rf_no_controls",
          property_name="controls_scaled_da",
      ),
      dict(
          testcase_name="national_controls_scaled_da",
          input_data_fixture="input_data_with_media_and_rf_no_controls",
          property_name="national_controls_scaled_da",
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
          testcase_name="national_media_raw_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="national_media_raw_da",
      ),
      dict(
          testcase_name="national_media_scaled_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="national_media_scaled_da",
      ),
      dict(
          testcase_name="media_spend_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="media_spend_da",
      ),
      dict(
          testcase_name="national_media_spend_da",
          input_data_fixture="input_data_with_rf_only",
          property_name="national_media_spend_da",
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
          testcase_name="national_organic_media_raw_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="national_organic_media_raw_da",
      ),
      dict(
          testcase_name="national_organic_media_scaled_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="national_organic_media_scaled_da",
      ),
      dict(
          testcase_name="non_media_scaled_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="non_media_scaled_da",
      ),
      dict(
          testcase_name="national_non_media_scaled_da",
          input_data_fixture="input_data_with_media_and_rf",
          property_name="national_non_media_scaled_da",
      ),
      dict(
          testcase_name="rf_spend_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_spend_da",
      ),
      dict(
          testcase_name="national_rf_spend_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_rf_spend_da",
      ),
      dict(
          testcase_name="reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_raw_da",
      ),
      dict(
          testcase_name="national_reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_reach_raw_da",
      ),
      dict(
          testcase_name="reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="reach_scaled_da",
      ),
      dict(
          testcase_name="national_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_reach_scaled_da",
      ),
      dict(
          testcase_name="frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="frequency_da",
      ),
      dict(
          testcase_name="national_frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_frequency_da",
      ),
      dict(
          testcase_name="rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_raw_da",
      ),
      dict(
          testcase_name="national_rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_rf_impressions_raw_da",
      ),
      dict(
          testcase_name="rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="national_rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="organic_reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_raw_da",
      ),
      dict(
          testcase_name="national_organic_reach_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_organic_reach_raw_da",
      ),
      dict(
          testcase_name="organic_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_reach_scaled_da",
      ),
      dict(
          testcase_name="national_organic_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_organic_reach_scaled_da",
      ),
      dict(
          testcase_name="organic_frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_frequency_da",
      ),
      dict(
          testcase_name="national_organic_frequency_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_organic_frequency_da",
      ),
      dict(
          testcase_name="organic_rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_raw_da",
      ),
      dict(
          testcase_name="national_organic_rf_impressions_raw_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_organic_rf_impressions_raw_da",
      ),
      dict(
          testcase_name="organic_rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="organic_rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="national_organic_rf_impressions_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_organic_rf_impressions_scaled_da",
      ),
      dict(
          testcase_name="geo_population_da",
          input_data_fixture="national_input_data_media_and_rf",
          property_name="geo_population_da",
      ),
      dict(
          testcase_name="all_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="all_reach_scaled_da",
      ),
      dict(
          testcase_name="all_freq_da",
          input_data_fixture="input_data_with_media_only",
          property_name="all_freq_da",
      ),
      dict(
          testcase_name="national_all_reach_scaled_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_all_reach_scaled_da",
      ),
      dict(
          testcase_name="national_all_freq_da",
          input_data_fixture="input_data_with_media_only",
          property_name="national_all_freq_da",
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
        engine.national_media_raw_da,
        engine.media_scaled_da,
        engine.national_media_scaled_da,
        engine.organic_media_raw_da,
        engine.national_organic_media_raw_da,
        engine.organic_media_scaled_da,
        engine.national_organic_media_scaled_da,
        engine.reach_raw_da,
        engine.national_reach_raw_da,
        engine.reach_scaled_da,
        engine.national_reach_scaled_da,
        engine.frequency_da,
        engine.national_frequency_da,
        engine.rf_impressions_raw_da,
        engine.national_rf_impressions_raw_da,
        engine.rf_impressions_scaled_da,
        engine.national_rf_impressions_scaled_da,
        engine.organic_reach_raw_da,
        engine.national_organic_reach_raw_da,
        engine.organic_reach_scaled_da,
        engine.national_organic_reach_scaled_da,
        engine.organic_frequency_da,
        engine.national_organic_frequency_da,
        engine.organic_rf_impressions_raw_da,
        engine.national_organic_rf_impressions_raw_da,
        engine.organic_rf_impressions_scaled_da,
        engine.national_organic_rf_impressions_scaled_da,
    ]

    for prop in properties_to_test:
      self.assertIsInstance(prop, xr.DataArray)
      self.assertEqual(
          prop.sizes[constants.TIME],
          model_test_data.WithInputDataSamples._N_TIMES,
      )
      self.assertNotIn(constants.MEDIA_TIME, prop.coords)
      self.assertIn(constants.TIME, prop.coords)

  def _create_mock_single_var_dataset_for_corr_test(
      self, data: np.ndarray, var_name: str = "media"
  ):
    """Helper to create a dataset for correlation testing."""
    var_dim_name = f"{var_name}_dim"

    if data.ndim == 3:
      n_geos, n_times, n_vars = data.shape
      dims = ["geo", "time", var_dim_name]
      geos = [f"geo{i}" for i in range(n_geos)]
    elif data.ndim == 2:
      n_times, n_vars = data.shape
      dims = ["time", var_dim_name]
      geos = None
    else:
      raise ValueError(f"Unsupported data shape: {data.shape}")

    times = pd.date_range(start="2023-01-01", periods=n_times, freq="W")
    var_coords = [f"{var_name}_{i+1}" for i in range(n_vars)]

    coords = {"time": times, var_dim_name: var_coords}
    if geos:
      coords["geo"] = geos

    xarray_data_vars = {var_name: (dims, data)}

    return xr.Dataset(data_vars=xarray_data_vars, coords=coords)

  def test_check_pairwise_corr_geo_one_error(self):
    # Create data where media_1 and media_2 are perfectly correlated
    data = np.array([
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
    ])  # Shape (2, 3, 2)
    mock_ds = self._create_mock_single_var_dataset_for_corr_test(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times and geos",
        finding.explanation,
    )

    overall_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIn("media_1", overall_result.extreme_corr_var_pairs.to_string())
    self.assertIn("media_2", overall_result.extreme_corr_var_pairs.to_string())
    self.assertEqual(
        overall_result.extreme_corr_threshold,
        eda_engine._PAIRWISE_OVERALL_CORR_THRESHOLD,
    )

  def test_check_pairwise_corr_geo_one_attention(self):
    # Create data where media_1 and media_2 are perfectly correlated per geo but
    # not overall.
    data = np.array([
        [[1, 1], [2, 2], [3, 3]],
        [[4, 7], [5, 8], [6, 9]],
    ])  # Shape (2, 3, 2)
    mock_ds = self._create_mock_single_var_dataset_for_corr_test(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(
        "perfect pairwise correlation in certain geo(s)",
        finding.explanation,
    )
    geo_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIn("media_1", geo_result.extreme_corr_var_pairs.to_string())
    self.assertIn("media_2", geo_result.extreme_corr_var_pairs.to_string())
    self.assertEqual(
        geo_result.extreme_corr_threshold,
        eda_engine._PAIRWISE_GEO_CORR_THRESHOLD,
    )

  def test_check_pairwise_corr_geo_info_only(self):
    # No high correlations
    data = np.array([
        [[1, 10], [2, 2], [3, 13]],
        [[4, 4], [5, 15], [6, 6]],
    ])  # Shape (2, 3, 2)
    mock_ds = self._create_mock_single_var_dataset_for_corr_test(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review the computed pairwise correlations",
        finding.explanation,
    )

    overall_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.OVERALL
    )
    geo_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.GEO
    )

    pd.testing.assert_frame_equal(
        overall_result.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )
    pd.testing.assert_frame_equal(
        geo_result.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )

  def test_check_pairwise_corr_geo_high_overall_corr(self):
    # Create data where media_1 and control_1 are perfectly correlated across
    # all geos.
    media_data = np.array([
        [[1], [2], [3]],
        [[4], [5], [6]],
    ])  # Shape (2, 3, 1)
    control_data = np.array([
        [[2], [4], [6]],
        [[8], [10], [12]],
    ])  # Shape (2, 3, 1)
    mock_media_ds = self._create_mock_single_var_dataset_for_corr_test(
        media_data, "media"
    )
    mock_control_ds = self._create_mock_single_var_dataset_for_corr_test(
        control_data, "control"
    )
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times and geos",
        finding.explanation,
    )
    overall_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIn("media_1", overall_result.extreme_corr_var_pairs.to_string())
    self.assertIn(
        "control_1", overall_result.extreme_corr_var_pairs.to_string()
    )

  def test_check_pairwise_corr_geo_high_corr_in_one_geo(self):
    # Create data where media_1 and control_1 are perfectly correlated in geo1
    # but not geo2.
    media_data = np.array([
        [[1], [2], [3]],
        [[4], [5], [6]],
    ])  # Shape (2, 3, 1)
    control_data = np.array([
        [[2], [4], [6]],
        [[8], [11], [14]],
    ])  # Shape (2, 3, 1)
    mock_media_ds = self._create_mock_single_var_dataset_for_corr_test(
        media_data, "media"
    )
    mock_control_ds = self._create_mock_single_var_dataset_for_corr_test(
        control_data, "control"
    )
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(
        "perfect pairwise correlation in certain geo(s)",
        finding.explanation,
    )
    geo_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIn("media_1", geo_result.extreme_corr_var_pairs.to_string())
    self.assertIn("control_1", geo_result.extreme_corr_var_pairs.to_string())
    self.assertIn("geo0", geo_result.extreme_corr_var_pairs.to_string())

  def test_check_pairwise_corr_geo_correlation_values(self):
    # Create data to test correlation computations.
    # geo0: media_1 = [1, 2, 3], control_1 = [1, 2, 3] -> corr = 1.0
    # geo1: media_1 = [4, 5, 6], control_1 = [6, 5, 4] -> corr = -1.0
    media_data = np.array([
        [[1], [2], [3]],
        [[4], [5], [6]],
    ])  # Shape (2, 3, 1)
    control_data = np.array([
        [[1], [2], [3]],
        [[6], [5], [4]],
    ])  # Shape (2, 3, 1)
    mock_media_ds = self._create_mock_single_var_dataset_for_corr_test(
        media_data, "media"
    )
    mock_control_ds = self._create_mock_single_var_dataset_for_corr_test(
        control_data, "control"
    )
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_geo()

    expected_overall_corr = np.corrcoef(
        media_data.flatten(), control_data.flatten()
    )[0, 1]

    # Expected geo correlations:
    expected_geo_corr = np.array([
        # geo0: media_1 = [1, 2, 3], control_1 = [1, 2, 3] -> corr = 1.0
        np.corrcoef(media_data[0, :, 0], control_data[0, :, 0])[0, 1],
        # geo1: media_1 = [4, 5, 6], control_1 = [6, 5, 4] -> corr = -1.0
        np.corrcoef(media_data[1, :, 0], control_data[1, :, 0])[0, 1],
    ])

    overall_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.OVERALL
    )
    geo_result = next(
        res
        for res in outcome.pairwise_corr_results
        if res.level == eda_outcome.AnalysisLevel.GEO
    )
    overall_corr_mat = overall_result.corr_matrix
    self.assertEqual(overall_corr_mat.name, eda_engine._CORRELATION_MATRIX_NAME)
    geo_corr_mat = geo_result.corr_matrix
    self.assertEqual(geo_corr_mat.name, eda_engine._CORRELATION_MATRIX_NAME)

    # Check overall correlation
    self.assertAllClose(
        overall_corr_mat.sel(var1="media_1", var2="control_1").values,
        expected_overall_corr,
    )

    # Check geo correlations
    self.assertAllClose(
        geo_corr_mat.sel(var1="media_1", var2="control_1").values,
        expected_geo_corr,
    )

  def test_check_pairwise_corr_geo_raises_error_for_national_model(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      engine.check_pairwise_corr_geo()

  def test_check_pairwise_corr_national_one_error(self):
    # Create data where media_1 and media_2 are perfectly correlated
    data = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
    ])  # Shape (3, 2)
    mock_ds = self._create_mock_single_var_dataset_for_corr_test(data)
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "national_treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_national()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 1)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times",
        finding.explanation,
    )

    result = outcome.pairwise_corr_results[0]
    self.assertEqual(result.level, eda_outcome.AnalysisLevel.NATIONAL)
    self.assertIn("media_1", result.extreme_corr_var_pairs.to_string())
    self.assertIn("media_2", result.extreme_corr_var_pairs.to_string())
    self.assertEqual(
        result.extreme_corr_threshold,
        eda_engine._PAIRWISE_NATIONAL_CORR_THRESHOLD,
    )

  def test_check_pairwise_corr_national_info_only(self):
    # No high correlations
    data = np.array([
        [1, 10],
        [2, 2],
        [3, 13],
    ])  # Shape (3, 2)
    mock_ds = self._create_mock_single_var_dataset_for_corr_test(data)
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "national_treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_national()

    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.pairwise_corr_results, 1)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review the computed pairwise correlations",
        finding.explanation,
    )

    result = outcome.pairwise_corr_results[0]
    pd.testing.assert_frame_equal(
        result.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )

  def test_check_pairwise_corr_national_correlation_values(self):
    # Create data to test correlation computations.
    media_data = np.array([
        [1],
        [2],
        [3],
    ])  # Shape (3, 1)
    control_data = np.array([
        [1],
        [2],
        [4],
    ])  # Shape (3, 1)
    mock_media_ds = self._create_mock_single_var_dataset_for_corr_test(
        media_data, "media"
    )
    mock_control_ds = self._create_mock_single_var_dataset_for_corr_test(
        control_data, "control"
    )
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            "national_treatment_control_scaled_ds",
            new_callable=mock.PropertyMock,
            return_value=mock_ds,
        )
    )
    outcome = engine.check_pairwise_corr_national()

    expected_corr = np.corrcoef(media_data.flatten(), control_data.flatten())[
        0, 1
    ]

    result = outcome.pairwise_corr_results[0]
    corr_mat = result.corr_matrix
    self.assertEqual(corr_mat.name, eda_engine._CORRELATION_MATRIX_NAME)

    self.assertAllClose(
        corr_mat.sel(var1="media_1", var2="control_1").values,
        expected_corr,
    )


if __name__ == "__main__":
  absltest.main()
