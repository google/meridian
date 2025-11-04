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
from meridian import backend
from meridian import constants
from meridian.model import model
from meridian.model import model_test_data
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import outliers_influence
import tensorflow as tf
import xarray as xr


def _construct_coords(
    dims: list[str],
    n_geos: int,
    n_times: int,
    n_vars: int,
    var_name: str,
) -> dict[str, list[str]]:
  coords = {}
  for dim in dims:
    if dim == constants.TIME:
      coords[dim] = pd.date_range(start="2023-01-01", periods=n_times, freq="W")
    elif dim == constants.GEO:
      coords[dim] = [f"{constants.GEO}{i}" for i in range(n_geos)]
    else:
      coords[dim] = [f"{var_name}_{i+1}" for i in range(n_vars)]
  return coords


def _construct_dims_and_shapes(
    data_shape: tuple[int, ...], var_name: str | None = None
):
  """Helper to construct the dimensions of a DataArray."""
  ndim = len(data_shape)
  if var_name is None:
    n_vars = 0
    if ndim == 2:
      dims = [constants.GEO, constants.TIME]
      n_geos, n_times = data_shape
    elif ndim == 1:
      dims = [constants.TIME]
      n_geos, n_times = 0, data_shape[0]
    else:
      raise ValueError(f"Unsupported data shape: {data_shape}")
  else:
    var_dim_name = f"{var_name}_dim"
    if ndim == 3:
      dims = [constants.GEO, constants.TIME, var_dim_name]
      n_geos, n_times, n_vars = data_shape
    elif ndim == 2:
      dims = [constants.TIME, var_dim_name]
      n_times, n_vars = data_shape
      n_geos = 0
    else:
      raise ValueError(f"Unsupported data shape: {data_shape}")

  return dims, n_geos, n_times, n_vars


def _create_dataset_with_var_dim(
    data: np.ndarray, var_name: str = "media"
) -> xr.Dataset:
  """Helper to create a dataset with a single variable dimension."""
  dims, n_geos, n_times, n_vars = _construct_dims_and_shapes(
      data.shape, var_name
  )
  coords = _construct_coords(dims, n_geos, n_times, n_vars, var_name)
  xarray_data_vars = {var_name: (dims, data)}

  return xr.Dataset(data_vars=xarray_data_vars, coords=coords)


def _create_data_array_with_var_dim(
    data: np.ndarray, name: str, var_name: str | None = None
) -> xr.DataArray:
  """Helper to create a data array with a single variable dimension."""
  dims, n_geos, n_times, n_vars = _construct_dims_and_shapes(
      data.shape, var_name
  )
  if var_name is None:
    var_name = name
  coords = _construct_coords(dims, n_geos, n_times, n_vars, var_name)

  return xr.DataArray(data, name=name, dims=dims, coords=coords)


_N_GEOS_VIF = 2
_N_TIMES_VIF = 20
_N_VARS_VIF = 3
_RNG = np.random.default_rng(42)


def _get_low_vif_da(geo_level: bool = True):
  shape = (_N_TIMES_VIF, _N_VARS_VIF)
  if geo_level:
    shape = (_N_GEOS_VIF,) + shape

  data = _RNG.random(shape)
  da = _create_data_array_with_var_dim(data, "VIF", "var")
  return da.rename({"var_dim": eda_engine._STACK_VAR_COORD_NAME})


def _get_geo_high_vif_da():
  v1 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
  v2 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
  v3_geo0 = v1[0, :] * 2 + v2[0, :] * 0.5 + _RNG.random(_N_TIMES_VIF) * 0.01
  v3_geo1 = _RNG.random(_N_TIMES_VIF)
  v3 = np.stack([v3_geo0, v3_geo1], axis=0)
  data = np.stack([v1, v2, v3], axis=-1)
  da = _create_data_array_with_var_dim(data, "VIF", "var")
  return da.rename({"var_dim": eda_engine._STACK_VAR_COORD_NAME})


def _get_overall_high_vif_da(geo_level: bool = True):
  sample_shape = (_N_GEOS_VIF, _N_TIMES_VIF) if geo_level else (_N_TIMES_VIF,)
  v1 = _RNG.random(sample_shape)
  v2 = _RNG.random(sample_shape)
  # v3 is a linear combination of v1 and v2, which results in an inf VIF value.
  v3 = v1 * 2 + v2 * 0.5
  data = np.stack([v1, v2, v3], axis=-1)
  da = _create_data_array_with_var_dim(data, "VIF", "var")
  return da.rename({"var_dim": eda_engine._STACK_VAR_COORD_NAME})


def _create_ndarray_with_std_below_threshold(
    n_times: int, is_national: bool
) -> np.ndarray:
  """Creates an array with std without outliers equal to _STD_THRESHOLD / 2."""
  target_std = eda_engine._STD_THRESHOLD / 2
  # Create a base array with n_times - 1 elements.
  base_data = np.arange(n_times - 1).astype(float)
  # Calculate the standard deviation of the base data.
  std_base = np.std(base_data, ddof=1)
  # Determine the scaling factor to achieve the target_std.
  scale_factor = target_std / std_base
  # Scale the base data.
  non_outlier_data = base_data * scale_factor

  # Use a large fixed number as an outlier.
  outlier = 1000.0

  # Combine non-outlier data and the outlier.
  mock_array = np.append(non_outlier_data, outlier)

  # Reshape based on whether it's a national or geo model.
  if is_national:
    return mock_array
  else:
    return mock_array.reshape(1, n_times)


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

  def _mock_eda_engine_property(self, property_name, return_value):
    self.enter_context(
        mock.patch.object(
            eda_engine.EDAEngine,
            property_name,
            new_callable=mock.PropertyMock,
            return_value=return_value,
        )
    )

  def test_spec_property_default_spec(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)
    self.assertEqual(engine.spec, eda_spec.EDASpec())
    self.assertEqual(engine.spec.vif_spec, eda_spec.VIFSpec())
    self.assertEqual(
        engine.spec.aggregation_config, eda_spec.AggregationConfig()
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="vif_spec",
          kwargs_to_pass=dict(vif_spec=eda_spec.VIFSpec(geo_threshold=500)),
      ),
      dict(
          testcase_name="aggregation_config",
          kwargs_to_pass=dict(
              aggregation_config=eda_spec.AggregationConfig(
                  control_variables={"control_0": np.mean},
              )
          ),
      ),
  )
  def test_spec_property_custom_spec_fields(self, kwargs_to_pass):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(**kwargs_to_pass)
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self.assertEqual(engine.spec, spec)

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
          agg_config=eda_spec.AggregationConfig(),
          expected_values_func=lambda da: da.sum(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mean",
          agg_config=eda_spec.AggregationConfig(
              control_variables={
                  "control_0": np.mean,
                  "control_1": np.mean,
              }
          ),
          expected_values_func=lambda da: da.mean(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mix",
          agg_config=eda_spec.AggregationConfig(
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
    spec = eda_spec.EDASpec(aggregation_config=agg_config)
    engine = eda_engine.EDAEngine(meridian, spec=spec)

    national_controls_scaled_da = engine.national_controls_scaled_da
    self.assertIsInstance(national_controls_scaled_da, xr.DataArray)
    self.assertEqual(
        national_controls_scaled_da.name, constants.NATIONAL_CONTROLS_SCALED
    )
    self.assertEqual(
        national_controls_scaled_da.shape,
        (
            self._N_TIMES,
            self._N_CONTROLS,
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
            self._N_TIMES,
            self._N_CONTROLS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
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

  def test_media_spend_da_with_1d_spend(self):
    input_data = self.input_data_with_media_and_rf.copy(deep=True)

    # Create 1D media_spend
    one_d_spend = np.array(
        [i + 1 for i in range(self._N_MEDIA_CHANNELS)], dtype=np.float64
    )
    input_data.media_spend = xr.DataArray(
        one_d_spend,
        dims=[constants.MEDIA_CHANNEL],
        coords={
            constants.MEDIA_CHANNEL: (
                input_data.media.coords[constants.MEDIA_CHANNEL].values
            )
        },
        name=constants.MEDIA_SPEND,
    )

    meridian = model.Meridian(input_data)
    engine = eda_engine.EDAEngine(meridian)

    # test media_spend_da
    media_spend_da = engine.media_spend_da
    self.assertIsInstance(media_spend_da, xr.DataArray)
    self.assertEqual(media_spend_da.name, constants.MEDIA_SPEND)
    self.assertEqual(
        media_spend_da.shape,
        (
            self._N_GEOS,
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
        ),
    )
    self.assertCountEqual(
        media_spend_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
    )

    # check values
    expected_allocated_spend = input_data.allocated_media_spend
    self.assertIsInstance(expected_allocated_spend, xr.DataArray)
    self.assertAllClose(media_spend_da.values, expected_allocated_spend.values)

    # test national_media_spend_da
    national_media_spend_da = engine.national_media_spend_da
    self.assertIsInstance(national_media_spend_da, xr.DataArray)
    self.assertEqual(
        national_media_spend_da.name, constants.NATIONAL_MEDIA_SPEND
    )
    self.assertEqual(
        national_media_spend_da.shape,
        (
            self._N_TIMES,
            self._N_MEDIA_CHANNELS,
        ),
    )
    # check values
    expected_national_spend = media_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(
        national_media_spend_da.values, expected_national_spend.values
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
            self._N_TIMES,
            self._N_ORGANIC_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_MEDIA_CHANNELS,
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
          agg_config=eda_spec.AggregationConfig(),
          expected_values_func=lambda da: da.sum(dim=constants.GEO),
      ),
      dict(
          testcase_name="geo_custom_agg_mean",
          agg_config=eda_spec.AggregationConfig(
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
    spec = eda_spec.EDASpec(aggregation_config=agg_config)
    engine = eda_engine.EDAEngine(meridian, spec=spec)

    national_non_media_scaled_da = engine.national_non_media_scaled_da
    self.assertIsInstance(national_non_media_scaled_da, xr.DataArray)
    self.assertEqual(
        national_non_media_scaled_da.name,
        constants.NATIONAL_NON_MEDIA_TREATMENTS_SCALED,
    )
    self.assertEqual(
        national_non_media_scaled_da.shape,
        (
            self._N_TIMES,
            self._N_NON_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_NON_MEDIA_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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

  def test_rf_spend_da_with_1d_spend(self):
    input_data = self.input_data_with_media_and_rf.copy(deep=True)

    # Create 1D rf_spend
    one_d_spend = np.array(
        [i + 1 for i in range(self._N_RF_CHANNELS)], dtype=np.float64
    )
    input_data.rf_spend = xr.DataArray(
        one_d_spend,
        dims=[constants.RF_CHANNEL],
        coords={
            constants.RF_CHANNEL: (
                input_data.reach.coords[constants.RF_CHANNEL].values
            )
        },
        name=constants.RF_SPEND,
    )

    meridian = model.Meridian(input_data)
    engine = eda_engine.EDAEngine(meridian)

    # test rf_spend_da
    rf_spend_da = engine.rf_spend_da
    self.assertIsInstance(rf_spend_da, xr.DataArray)
    self.assertEqual(rf_spend_da.name, constants.RF_SPEND)
    self.assertEqual(
        rf_spend_da.shape,
        (
            self._N_GEOS,
            self._N_TIMES,
            self._N_RF_CHANNELS,
        ),
    )
    self.assertCountEqual(
        rf_spend_da.coords.keys(),
        [constants.GEO, constants.TIME, constants.RF_CHANNEL],
    )

    # check values
    expected_allocated_spend = input_data.allocated_rf_spend
    self.assertIsInstance(expected_allocated_spend, xr.DataArray)
    self.assertAllClose(rf_spend_da.values, expected_allocated_spend.values)

    # test national_rf_spend_da
    national_rf_spend_da = engine.national_rf_spend_da
    self.assertIsInstance(national_rf_spend_da, xr.DataArray)
    self.assertEqual(national_rf_spend_da.name, constants.NATIONAL_RF_SPEND)
    self.assertEqual(
        national_rf_spend_da.shape,
        (
            self._N_TIMES,
            self._N_RF_CHANNELS,
        ),
    )
    # check values
    expected_national_spend = rf_spend_da.sum(dim=constants.GEO)
    self.assertAllClose(
        national_rf_spend_da.values, expected_national_spend.values
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
            self._N_TIMES,
            self._N_ORGANIC_RF_CHANNELS,
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
        (self._N_GEOS,),
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
        (self._N_TIMES,),
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
        (self._N_TIMES,),
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
          self._N_TIMES,
      )
      self.assertNotIn(constants.MEDIA_TIME, prop.coords)
      self.assertIn(constants.TIME, prop.coords)

  def test_check_geo_pairwise_corr_one_error(self):
    # Create data where media_1 and media_2 are perfectly correlated
    data = np.array([
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
    ])  # Shape (2, 3, 2)
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times and geos",
        finding.explanation,
    )
    self.assertIn(
        "Pairs with perfect correlation: [('media_1', 'media_2')]",
        finding.explanation,
    )

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIn(
        "media_1", overall_artifact.extreme_corr_var_pairs.to_string()
    )
    self.assertIn(
        "media_2", overall_artifact.extreme_corr_var_pairs.to_string()
    )
    self.assertEqual(
        overall_artifact.extreme_corr_threshold,
        eda_engine._OVERALL_PAIRWISE_CORR_THRESHOLD,
    )

  def test_check_geo_pairwise_corr_one_attention(self):
    # Create data where media_1 and media_2 are perfectly correlated per geo but
    # not overall.
    data = np.array([
        [[1, 1], [2, 2], [3, 3]],
        [[4, 7], [5, 8], [6, 9]],
    ])  # Shape (2, 3, 2)
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(
        "perfect pairwise correlation in certain geo(s)",
        finding.explanation,
    )
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIn("media_1", geo_artifact.extreme_corr_var_pairs.to_string())
    self.assertIn("media_2", geo_artifact.extreme_corr_var_pairs.to_string())
    self.assertEqual(
        geo_artifact.extreme_corr_threshold,
        eda_engine._GEO_PAIRWISE_CORR_THRESHOLD,
    )

  def test_check_geo_pairwise_corr_info_only(self):
    # No high correlations
    data = np.array([
        [[1, 10], [2, 2], [3, 13]],
        [[4, 4], [5, 15], [6, 6]],
    ])  # Shape (2, 3, 2)
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review the computed pairwise correlations",
        finding.explanation,
    )

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIsInstance(overall_artifact, eda_outcome.PairwiseCorrArtifact)
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIsInstance(geo_artifact, eda_outcome.PairwiseCorrArtifact)

    pd.testing.assert_frame_equal(
        overall_artifact.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )
    pd.testing.assert_frame_equal(
        geo_artifact.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )

  def test_check_geo_pairwise_corr_high_overall_corr(self):
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
    mock_media_ds = _create_dataset_with_var_dim(media_data, "media")
    mock_control_ds = _create_dataset_with_var_dim(control_data, "control")
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times and geos",
        finding.explanation,
    )
    self.assertIn(
        "Pairs with perfect correlation: [('media_1', 'control_1')]",
        finding.explanation,
    )
    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIn(
        "media_1", overall_artifact.extreme_corr_var_pairs.to_string()
    )
    self.assertIn(
        "control_1", overall_artifact.extreme_corr_var_pairs.to_string()
    )

  def test_check_geo_pairwise_corr_high_corr_in_one_geo(self):
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
    mock_media_ds = _create_dataset_with_var_dim(media_data, "media")
    mock_control_ds = _create_dataset_with_var_dim(control_data, "control")
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(
        "perfect pairwise correlation in certain geo(s)",
        finding.explanation,
    )
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIn("media_1", geo_artifact.extreme_corr_var_pairs.to_string())
    self.assertIn("control_1", geo_artifact.extreme_corr_var_pairs.to_string())
    self.assertIn("geo0", geo_artifact.extreme_corr_var_pairs.to_string())

  def test_check_geo_pairwise_corr_corr_matrix_has_correct_coordinates(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.analysis_artifacts, 2)

    for artifact in outcome.analysis_artifacts:
      if artifact.level == eda_outcome.AnalysisLevel.OVERALL:
        self.assertCountEqual(
            artifact.corr_matrix.coords.keys(),
            [eda_engine._CORR_VAR1, eda_engine._CORR_VAR2],
        )
      elif artifact.level == eda_outcome.AnalysisLevel.GEO:
        self.assertCountEqual(
            artifact.corr_matrix.coords.keys(),
            [constants.GEO, eda_engine._CORR_VAR1, eda_engine._CORR_VAR2],
        )
      else:
        self.fail(f"Unexpected level: {artifact.level}")

  def test_check_geo_pairwise_corr_correlation_values(self):
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
    mock_media_ds = _create_dataset_with_var_dim(media_data, "media")
    mock_control_ds = _create_dataset_with_var_dim(control_data, "control")
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
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

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    overall_corr_mat = overall_artifact.corr_matrix
    self.assertEqual(overall_corr_mat.name, eda_engine._CORRELATION_MATRIX_NAME)
    geo_corr_mat = geo_artifact.corr_matrix
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

  def test_check_geo_pairwise_corr_raises_error_for_national_model(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      engine.check_geo_pairwise_corr()

  def test_check_national_pairwise_corr_one_error(self):
    # Create data where media_1 and media_2 are perfectly correlated
    data = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
    ])  # Shape (3, 2)
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_ds
    )
    outcome = engine.check_national_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 1)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times",
        finding.explanation,
    )
    self.assertIn(
        "Pairs with perfect correlation: [('media_1', 'media_2')]",
        finding.explanation,
    )

    artifact = outcome.analysis_artifacts[0]
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)
    self.assertIn("media_1", artifact.extreme_corr_var_pairs.to_string())
    self.assertIn("media_2", artifact.extreme_corr_var_pairs.to_string())
    self.assertEqual(
        artifact.extreme_corr_threshold,
        eda_engine._NATIONAL_PAIRWISE_CORR_THRESHOLD,
    )

  def test_check_national_pairwise_corr_info_only(self):
    # No high correlations
    data = np.array([
        [1, 10],
        [2, 2],
        [3, 13],
    ])  # Shape (3, 2)
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_ds
    )
    outcome = engine.check_national_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 1)

    finding = outcome.findings[0]
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review the computed pairwise correlations",
        finding.explanation,
    )

    artifact = outcome.analysis_artifacts[0]
    pd.testing.assert_frame_equal(
        artifact.extreme_corr_var_pairs,
        eda_engine._EMPTY_DF_FOR_EXTREME_CORR_PAIRS,
    )

  def test_check_national_pairwise_corr_corr_matrix_has_correct_coordinates(
      self,
  ):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_national_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    self.assertLen(outcome.analysis_artifacts, 1)
    artifact = outcome.analysis_artifacts[0]
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)
    self.assertCountEqual(
        artifact.corr_matrix.coords.keys(),
        [eda_engine._CORR_VAR1, eda_engine._CORR_VAR2],
    )

  def test_check_national_pairwise_corr_correlation_values(self):
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
    mock_media_ds = _create_dataset_with_var_dim(media_data, "media")
    mock_control_ds = _create_dataset_with_var_dim(control_data, "control")
    mock_ds = xr.merge([mock_media_ds, mock_control_ds])
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_ds
    )
    outcome = engine.check_national_pairwise_corr()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORR)
    expected_corr = np.corrcoef(media_data.flatten(), control_data.flatten())[
        0, 1
    ]

    artifact = outcome.analysis_artifacts[0]
    corr_mat = artifact.corr_matrix
    self.assertEqual(corr_mat.name, eda_engine._CORRELATION_MATRIX_NAME)

    self.assertAllClose(
        corr_mat.sel(var1="media_1", var2="control_1").values,
        expected_corr,
    )

  def test_check_geo_std_raises_error_for_national_model(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)

    with self.assertRaisesRegex(
        ValueError, "check_geo_std is not applicable for national models."
    ):
      engine.check_geo_std()

  def test_check_geo_std_std_artifacts_have_correct_coordinates(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_geo_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 4)

    for artifact in outcome.analysis_artifacts:
      if artifact.variable == constants.KPI_SCALED:
        self.assertCountEqual(artifact.std_ds.coords.keys(), [constants.GEO])
      elif artifact.variable == constants.TREATMENT_CONTROL_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.GEO, eda_engine._STACK_VAR_COORD_NAME],
        )
      elif artifact.variable == constants.ALL_REACH_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.GEO, constants.RF_CHANNEL],
        )
      elif artifact.variable == constants.ALL_FREQUENCY:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.GEO, constants.RF_CHANNEL],
        )
      else:
        self.fail(f"Unexpected variable: {artifact.variable}")

  def test_check_geo_std_calculates_std_value_correctly(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.array([[1, 2, 3, 4, 5, 100]], dtype=float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.KPI_SCALED,
    )

    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_da)
    outcome = engine.check_geo_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 2)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.KPI_SCALED
    )

    expected_kpi_std_value_with_outliers = np.std([1, 2, 3, 4, 5, 100], ddof=1)
    expected_kpi_std_value_without_outliers = np.std([1, 2, 3, 4, 5], ddof=1)
    self.assertAllClose(
        kpi_artifact.std_ds[eda_engine._STD_WITH_OUTLIERS_VAR_NAME].values[0],
        expected_kpi_std_value_with_outliers,
    )
    self.assertAllClose(
        kpi_artifact.std_ds[eda_engine._STD_WITHOUT_OUTLIERS_VAR_NAME].values[
            0
        ],
        expected_kpi_std_value_without_outliers,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="small_outlier",
          outlier_value=8.0,
      ),
      dict(
          testcase_name="large_outlier",
          outlier_value=14.0,
      ),
  )
  def test_check_geo_std_correctly_identifies_outliers(self, outlier_value):
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.array([[10, 11, 12, 11, 10, 11, outlier_value]], dtype=float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.KPI_SCALED,
    )

    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_da)
    outcome = engine.check_geo_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 2)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.KPI_SCALED
    )

    self.assertGreater(
        kpi_artifact.std_ds[eda_engine._STD_WITH_OUTLIERS_VAR_NAME].values[0],
        kpi_artifact.std_ds[eda_engine._STD_WITHOUT_OUTLIERS_VAR_NAME].values[
            0
        ],
    )
    self.assertFalse(kpi_artifact.outlier_df.empty)
    self.assertEqual(
        kpi_artifact.outlier_df[eda_engine._OUTLIERS_COL_NAME].iloc[0],
        outlier_value,
    )

  def test_check_geo_std_returns_info_finding_when_no_issues(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = False
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.arange(7).reshape(1, 7).astype(float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.KPI_SCALED,
    )

    tc_data = np.tile(np.arange(7), (1, 1, 1)).astype(float)
    mock_tc_ds = _create_dataset_with_var_dim(tc_data)

    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_da)
    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_tc_ds)
    self._mock_eda_engine_property("all_reach_scaled_da", None)
    self._mock_eda_engine_property("all_freq_da", None)
    outcome = engine.check_geo_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(outcome.findings[0].severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review any identified outliers",
        outcome.findings[0].explanation,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_std_kpi",
          mock_kpi_ndarray=np.ones((1, 7), dtype=float),
          mock_tc_ndarray=np.tile(np.arange(7), (1, 1, 1)).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr="KPI has zero standard deviation",
      ),
      dict(
          testcase_name="zero_std_kpi_without_outliers",
          mock_kpi_ndarray=np.array([[1, 1, 1, 1, 1, 1, 100]], dtype=float),
          mock_tc_ndarray=np.tile(np.arange(7), (1, 1, 1)).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr="KPI has zero standard deviation",
      ),
      dict(
          testcase_name="zero_std_treatment_control",
          mock_kpi_ndarray=np.arange(7).reshape(1, 7).astype(float),
          mock_tc_ndarray=np.ones((1, 7, 1), dtype=float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr=(
              "Some treatment or control variables have zero standard deviation"
          ),
      ),
      dict(
          testcase_name="zero_std_reach",
          mock_kpi_ndarray=np.arange(7).reshape(1, 7).astype(float),
          mock_tc_ndarray=np.tile(np.arange(7), (1, 1, 1)).astype(float),
          mock_reach_ndarray=np.ones((1, 7, 1), dtype=float),
          mock_freq_ndarray=None,
          expected_message_substr="zero variation of reach across time",
      ),
      dict(
          testcase_name="zero_std_freq",
          mock_kpi_ndarray=np.arange(7).reshape(1, 7).astype(float),
          mock_tc_ndarray=np.tile(np.arange(7), (1, 1, 1)).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=np.ones((1, 7, 1), dtype=float),
          expected_message_substr="zero variation of frequency across time",
      ),
      dict(
          testcase_name="std_below_threshold_kpi",
          mock_kpi_ndarray=_create_ndarray_with_std_below_threshold(
              n_times=7, is_national=False
          ),
          mock_tc_ndarray=np.tile(np.arange(7), (1, 1, 1)).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr="KPI has zero standard deviation",
      ),
  )
  def test_check_geo_std_attention_cases(
      self,
      mock_kpi_ndarray,
      mock_tc_ndarray,
      mock_reach_ndarray,
      mock_freq_ndarray,
      expected_message_substr,
  ):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = False
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property(
        "kpi_scaled_da",
        _create_data_array_with_var_dim(
            mock_kpi_ndarray,
            name=constants.KPI_SCALED,
        ),
    )
    self._mock_eda_engine_property(
        "treatment_control_scaled_ds",
        _create_dataset_with_var_dim(
            mock_tc_ndarray,
            var_name=constants.TREATMENT_CONTROL_SCALED,
        ),
    )

    # Override mocks for RF data if provided
    if mock_reach_ndarray is not None:
      self._mock_eda_engine_property(
          "all_reach_scaled_da",
          _create_data_array_with_var_dim(
              mock_reach_ndarray,
              name=constants.ALL_REACH_SCALED,
              var_name=constants.RF_CHANNEL,
          ),
      )
    else:
      self._mock_eda_engine_property("all_reach_scaled_da", None)

    if mock_freq_ndarray is not None:
      self._mock_eda_engine_property(
          "all_freq_da",
          _create_data_array_with_var_dim(
              mock_freq_ndarray,
              name=constants.ALL_FREQUENCY,
              var_name=constants.RF_CHANNEL,
          ),
      )
    else:
      self._mock_eda_engine_property("all_freq_da", None)

    outcome = engine.check_geo_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.ATTENTION
    )
    self.assertIn(expected_message_substr, outcome.findings[0].explanation)

  def test_check_geo_std_handles_missing_rf_data(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = False
    engine = eda_engine.EDAEngine(meridian)

    mock_kpi_da = _create_data_array_with_var_dim(
        np.arange(7).reshape(1, 7).astype(float),
        name=constants.KPI_SCALED,
    )

    mock_tc_ds = _create_dataset_with_var_dim(
        np.tile(np.arange(7), (1, 1, 1)).astype(float),
        var_name=constants.TREATMENT_CONTROL_SCALED,
    )

    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_da)
    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_tc_ds)
    self._mock_eda_engine_property("all_reach_scaled_da", None)
    self._mock_eda_engine_property("all_freq_da", None)
    outcome = engine.check_geo_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 2)
    variables = [artifact.variable for artifact in outcome.analysis_artifacts]
    self.assertCountEqual(
        variables,
        [constants.KPI_SCALED, constants.TREATMENT_CONTROL_SCALED],
    )

  def test_check_national_std_std_artifacts_have_correct_coordinates(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_national_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 4)

    for artifact in outcome.analysis_artifacts:
      if artifact.variable == constants.NATIONAL_KPI_SCALED:
        self.assertCountEqual(artifact.std_ds.coords.keys(), [])
      elif artifact.variable == constants.NATIONAL_TREATMENT_CONTROL_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [eda_engine._STACK_VAR_COORD_NAME],
        )
      elif artifact.variable == constants.NATIONAL_ALL_REACH_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.RF_CHANNEL],
        )
      elif artifact.variable == constants.NATIONAL_ALL_FREQUENCY:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.RF_CHANNEL],
        )
      else:
        self.fail(f"Unexpected variable: {artifact.variable}")

  def test_check_national_std_calculates_std_value_correctly(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.array([1, 2, 3, 4, 5, 100], dtype=float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.NATIONAL_KPI_SCALED,
    )

    self._mock_eda_engine_property("national_kpi_scaled_da", mock_kpi_da)
    outcome = engine.check_national_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 4)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.NATIONAL_KPI_SCALED
    )

    expected_kpi_std_value_with_outliers = np.std([1, 2, 3, 4, 5, 100], ddof=1)
    expected_kpi_std_value_without_outliers = np.std([1, 2, 3, 4, 5], ddof=1)
    self.assertAllClose(
        kpi_artifact.std_ds[eda_engine._STD_WITH_OUTLIERS_VAR_NAME].values,
        expected_kpi_std_value_with_outliers,
    )
    self.assertAllClose(
        kpi_artifact.std_ds[eda_engine._STD_WITHOUT_OUTLIERS_VAR_NAME].values,
        expected_kpi_std_value_without_outliers,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="small_outlier",
          outlier_value=8.0,
      ),
      dict(
          testcase_name="large_outlier",
          outlier_value=14.0,
      ),
  )
  def test_check_national_std_correctly_identifies_outliers(
      self, outlier_value
  ):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.array([10, 11, 12, 11, 10, 11, outlier_value], dtype=float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.NATIONAL_KPI_SCALED,
    )

    self._mock_eda_engine_property("national_kpi_scaled_da", mock_kpi_da)
    outcome = engine.check_national_std()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 4)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.NATIONAL_KPI_SCALED
    )
    self.assertGreater(
        kpi_artifact.std_ds[
            eda_engine._STD_WITH_OUTLIERS_VAR_NAME
        ].values.item(),
        kpi_artifact.std_ds[
            eda_engine._STD_WITHOUT_OUTLIERS_VAR_NAME
        ].values.item(),
    )
    self.assertFalse(kpi_artifact.outlier_df.empty)
    self.assertEqual(
        kpi_artifact.outlier_df[eda_engine._OUTLIERS_COL_NAME].iloc[0],
        outlier_value,
    )

  def test_check_national_std_returns_info_finding_when_no_issues(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)

    kpi_data = np.arange(7).astype(float)
    mock_kpi_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.NATIONAL_KPI_SCALED,
    )

    tc_data = np.arange(7).reshape(7, 1).astype(float)
    mock_tc_ds = _create_dataset_with_var_dim(tc_data)

    self._mock_eda_engine_property("national_kpi_scaled_da", mock_kpi_da)
    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_tc_ds
    )
    self._mock_eda_engine_property("national_all_reach_scaled_da", None)
    self._mock_eda_engine_property("national_all_freq_da", None)
    outcome = engine.check_national_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(outcome.findings[0].severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review any identified outliers",
        outcome.findings[0].explanation,
    )

  def test_check_national_std_finds_zero_std_kpi(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)

    mock_kpi_da = _create_data_array_with_var_dim(
        np.ones(7, dtype=float),
        name=constants.NATIONAL_KPI_SCALED,
    )

    mock_tc_ds = _create_dataset_with_var_dim(
        np.arange(7).reshape(7, 1).astype(float),
        var_name=constants.NATIONAL_TREATMENT_CONTROL_SCALED,
    )

    self._mock_eda_engine_property("national_kpi_scaled_da", mock_kpi_da)
    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_tc_ds
    )
    self._mock_eda_engine_property("national_all_reach_scaled_da", None)
    self._mock_eda_engine_property("national_all_freq_da", None)

    outcome = engine.check_national_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.ATTENTION
    )
    self.assertIn(
        "The standard deviation of the scaled KPI drops",
        outcome.findings[0].explanation,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_std_kpi",
          mock_kpi_ndarray=np.ones(7, dtype=float),
          mock_tc_ndarray=np.arange(7).reshape(7, 1).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr=(
              "The standard deviation of the scaled KPI drops"
          ),
      ),
      dict(
          testcase_name="zero_std_treatment_control",
          mock_kpi_ndarray=np.arange(7).astype(float),
          mock_tc_ndarray=np.ones((7, 1), dtype=float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr=(
              "The standard deviation of these scaled treatment or control"
              " variables drops from positive to zero"
          ),
      ),
      dict(
          testcase_name="zero_std_reach",
          mock_kpi_ndarray=np.arange(7).astype(float),
          mock_tc_ndarray=np.arange(7).reshape(7, 1).astype(float),
          mock_reach_ndarray=np.ones((7, 1), dtype=float),
          mock_freq_ndarray=None,
          expected_message_substr="zero variation of reach across time",
      ),
      dict(
          testcase_name="zero_std_freq",
          mock_kpi_ndarray=np.arange(7).astype(float),
          mock_tc_ndarray=np.arange(7).reshape(7, 1).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=np.ones((7, 1), dtype=float),
          expected_message_substr="zero variation of frequency across time",
      ),
      dict(
          testcase_name="std_below_threshold_kpi",
          mock_kpi_ndarray=_create_ndarray_with_std_below_threshold(
              n_times=7, is_national=True
          ),
          mock_tc_ndarray=np.arange(7).reshape(7, 1).astype(float),
          mock_reach_ndarray=None,
          mock_freq_ndarray=None,
          expected_message_substr=(
              "The standard deviation of the scaled KPI drops"
          ),
      ),
  )
  def test_check_national_std_attention_cases(
      self,
      mock_kpi_ndarray,
      mock_tc_ndarray,
      mock_reach_ndarray,
      mock_freq_ndarray,
      expected_message_substr,
  ):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property(
        "national_kpi_scaled_da",
        _create_data_array_with_var_dim(
            mock_kpi_ndarray,
            name=constants.NATIONAL_KPI_SCALED,
        ),
    )
    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds",
        _create_dataset_with_var_dim(
            mock_tc_ndarray,
            var_name=constants.NATIONAL_TREATMENT_CONTROL_SCALED,
        ),
    )

    # Override mocks for RF data if provided
    if mock_reach_ndarray is not None:
      self._mock_eda_engine_property(
          "national_all_reach_scaled_da",
          _create_data_array_with_var_dim(
              mock_reach_ndarray,
              name=constants.NATIONAL_ALL_REACH_SCALED,
              var_name=constants.RF_CHANNEL,
          ),
      )
    else:
      self._mock_eda_engine_property("national_all_reach_scaled_da", None)

    if mock_freq_ndarray is not None:
      self._mock_eda_engine_property(
          "national_all_freq_da",
          _create_data_array_with_var_dim(
              mock_freq_ndarray,
              name=constants.NATIONAL_ALL_FREQUENCY,
              var_name=constants.RF_CHANNEL,
          ),
      )
    else:
      self._mock_eda_engine_property("national_all_freq_da", None)

    outcome = engine.check_national_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.ATTENTION
    )
    self.assertIn(expected_message_substr, outcome.findings[0].explanation)

  def test_check_national_std_handles_missing_rf_data(self):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)

    mock_kpi_da = _create_data_array_with_var_dim(
        np.arange(7).astype(float),
        name=constants.NATIONAL_KPI_SCALED,
    )

    mock_tc_ds = _create_dataset_with_var_dim(
        np.arange(7).reshape(7, 1).astype(float),
        var_name=constants.NATIONAL_TREATMENT_CONTROL_SCALED,
    )

    self._mock_eda_engine_property("national_kpi_scaled_da", mock_kpi_da)
    self._mock_eda_engine_property(
        "national_treatment_control_scaled_ds", mock_tc_ds
    )
    self._mock_eda_engine_property("national_all_reach_scaled_da", None)
    self._mock_eda_engine_property("national_all_freq_da", None)
    outcome = engine.check_national_std()
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.STD)
    self.assertLen(outcome.analysis_artifacts, 2)
    variables = [artifact.variable for artifact in outcome.analysis_artifacts]
    self.assertCountEqual(
        variables,
        [
            constants.NATIONAL_KPI_SCALED,
            constants.NATIONAL_TREATMENT_CONTROL_SCALED,
        ],
    )

  def test_check_geo_vif_raises_error_for_national_model(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    with self.assertRaisesRegex(
        ValueError,
        "Geo-level VIF checks are not applicable for national models.",
    ):
      engine.check_geo_vif()

  @parameterized.named_parameters(
      dict(
          testcase_name="info",
          data=_get_low_vif_da(),
          expected_severity=eda_outcome.EDASeverity.INFO,
          expected_explanation="Please review the computed VIFs.",
      ),
      dict(
          testcase_name="attention",
          data=_get_geo_high_vif_da(),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_explanation=(
              "Some variables have extreme multicollinearity (with VIF > 5) in"
              " certain geo(s)."
          ),
      ),
      dict(
          testcase_name="error",
          data=_get_overall_high_vif_da(),
          expected_severity=eda_outcome.EDASeverity.ERROR,
          expected_explanation=(
              "Some variables have extreme multicollinearity (VIF >10) across"
              " all times and geos. To address multicollinearity, please drop"
              " any variable that is a linear combination of other variables."
              " Otherwise, consider combining variables.\nVariables with"
              " extreme VIF: ['var_1', 'var_2', 'var_3']"
          ),
      ),
  )
  def test_check_geo_vif_returns_correct_finding_severity(
      self, data, expected_severity, expected_explanation
  ):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=10, geo_threshold=5)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property("_stacked_treatment_control_scaled_da", data)

    outcome = engine.check_geo_vif()

    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.findings, 1)
    self.assertEqual(outcome.findings[0].severity, expected_severity)
    self.assertIn(expected_explanation, outcome.findings[0].explanation)

  def test_check_geo_vif_overall_artifact_is_correct(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=1e6, geo_threshold=1)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", _get_geo_high_vif_da()
    )

    outcome = engine.check_geo_vif()
    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 2)

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIsInstance(overall_artifact, eda_outcome.VIFArtifact)
    self.assertEqual(overall_artifact.level, eda_outcome.AnalysisLevel.OVERALL)
    self.assertCountEqual(
        overall_artifact.vif_da.coords.keys(),
        [eda_engine._STACK_VAR_COORD_NAME],
    )
    self.assertEqual(overall_artifact.vif_da.shape, (_N_VARS_VIF,))
    # With overall_threshold=1e6 and _get_geo_vif_da(), we expect no overall
    # outliers
    self.assertTrue(overall_artifact.outlier_df.empty)

  def test_check_geo_vif_geo_artifact_is_correct(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=1e6, geo_threshold=10)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", _get_geo_high_vif_da()
    )

    outcome = engine.check_geo_vif()
    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 2)

    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIsInstance(geo_artifact, eda_outcome.VIFArtifact)
    self.assertEqual(geo_artifact.level, eda_outcome.AnalysisLevel.GEO)
    self.assertCountEqual(
        geo_artifact.vif_da.coords.keys(),
        [constants.GEO, eda_engine._STACK_VAR_COORD_NAME],
    )
    self.assertEqual(geo_artifact.vif_da.shape, (_N_GEOS_VIF, _N_VARS_VIF))
    # With geo_threshold=10 and _get_geo_vif_da(), we expect outliers in geo0
    self.assertFalse(geo_artifact.outlier_df.empty)
    self.assertIn(
        "geo0", geo_artifact.outlier_df.index.get_level_values(constants.GEO)
    )
    self.assertNotIn(
        "geo1", geo_artifact.outlier_df.index.get_level_values(constants.GEO)
    )

  def test_check_geo_vif_has_correct_vif_value_when_vif_is_inf(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=10, geo_threshold=5)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", _get_overall_high_vif_da()
    )

    outcome = engine.check_geo_vif()

    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 2)

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIsInstance(overall_artifact, eda_outcome.VIFArtifact)
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIsInstance(geo_artifact, eda_outcome.VIFArtifact)

    # With perfect multicollinearity, VIF values should be inf.
    self.assertTrue(np.isinf(overall_artifact.vif_da.values).all())
    self.assertTrue(np.isinf(geo_artifact.vif_da.values).all())

  def test_check_geo_vif_has_correct_vif_value(self):
    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=10, geo_threshold=5)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    data = _get_low_vif_da()
    self._mock_eda_engine_property("_stacked_treatment_control_scaled_da", data)

    outcome = engine.check_geo_vif()

    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 2)

    overall_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.OVERALL
    )
    self.assertIsInstance(overall_artifact, eda_outcome.VIFArtifact)
    geo_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.level == eda_outcome.AnalysisLevel.GEO
    )
    self.assertIsInstance(geo_artifact, eda_outcome.VIFArtifact)

    # Check overall VIF
    overall_data = data.values.reshape(-1, _N_VARS_VIF)
    overall_data_with_const = sm.add_constant(overall_data, prepend=True)
    expected_overall_vif = [
        outliers_influence.variance_inflation_factor(overall_data_with_const, i)
        for i in range(1, _N_VARS_VIF + 1)
    ]
    self.assertAllClose(overall_artifact.vif_da.values, expected_overall_vif)

    # Check geo VIF
    geo0_data = data.values[0, :, :]
    geo1_data = data.values[1, :, :]
    geo0_data_with_const = sm.add_constant(geo0_data, prepend=True)
    geo1_data_with_const = sm.add_constant(geo1_data, prepend=True)
    expected_geo0_vif = [
        outliers_influence.variance_inflation_factor(geo0_data_with_const, i)
        for i in range(1, _N_VARS_VIF + 1)
    ]
    expected_geo1_vif = [
        outliers_influence.variance_inflation_factor(geo1_data_with_const, i)
        for i in range(1, _N_VARS_VIF + 1)
    ]
    expected_geo_vif = np.stack([expected_geo0_vif, expected_geo1_vif], axis=0)
    self.assertAllClose(geo_artifact.vif_da.values, expected_geo_vif)

  @parameterized.named_parameters(
      dict(
          testcase_name="info",
          data=_get_low_vif_da(geo_level=False),
          expected_severity=eda_outcome.EDASeverity.INFO,
          expected_explanation="Please review the computed VIFs.",
      ),
      dict(
          testcase_name="error",
          data=_get_overall_high_vif_da(geo_level=False),
          expected_severity=eda_outcome.EDASeverity.ERROR,
          expected_explanation=(
              "Some variables have extreme multicollinearity (with VIF > 10)"
              " across all times. To address multicollinearity, please drop any"
              " variable that is a linear combination of other variables."
              " Otherwise, consider combining variables.\nVariables with"
              " extreme VIF: ['var_1', 'var_2', 'var_3']"
          ),
      ),
  )
  def test_check_national_vif_returns_correct_finding_severity(
      self, data, expected_severity, expected_explanation
  ):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    spec = eda_spec.EDASpec(vif_spec=eda_spec.VIFSpec(national_threshold=10))
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_national_treatment_control_scaled_da", data
    )

    outcome = engine.check_national_vif()

    self.assertLen(outcome.findings, 1)
    self.assertEqual(outcome.findings[0].severity, expected_severity)
    self.assertIn(expected_explanation, outcome.findings[0].explanation)

  @parameterized.named_parameters(
      dict(
          testcase_name="low_vif",
          data=_get_low_vif_da(geo_level=False),
          national_threshold=10,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="high_vif",
          data=_get_overall_high_vif_da(geo_level=False),
          national_threshold=10,
          expected_outlier_df_empty=False,
      ),
  )
  def test_check_national_vif_artifact_is_correct(
      self,
      data,
      national_threshold,
      expected_outlier_df_empty,
  ):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(national_threshold=national_threshold)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_national_treatment_control_scaled_da", data
    )

    outcome = engine.check_national_vif()
    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 1)

    national_artifact = outcome.analysis_artifacts[0]
    self.assertIsInstance(national_artifact, eda_outcome.VIFArtifact)
    self.assertEqual(
        national_artifact.level, eda_outcome.AnalysisLevel.NATIONAL
    )
    self.assertCountEqual(
        national_artifact.vif_da.coords.keys(),
        [eda_engine._STACK_VAR_COORD_NAME],
    )
    self.assertEqual(national_artifact.vif_da.shape, (_N_VARS_VIF,))
    self.assertEqual(
        national_artifact.outlier_df.empty, expected_outlier_df_empty
    )

  def test_check_national_vif_has_correct_vif_value_when_vif_is_inf(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    spec = eda_spec.EDASpec(vif_spec=eda_spec.VIFSpec(national_threshold=10))
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_national_treatment_control_scaled_da",
        _get_overall_high_vif_da(geo_level=False),
    )

    outcome = engine.check_national_vif()

    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 1)

    national_artifact = outcome.analysis_artifacts[0]
    self.assertIsInstance(national_artifact, eda_outcome.VIFArtifact)

    # With perfect multicollinearity, VIF values should be inf.
    self.assertTrue(np.isinf(national_artifact.vif_da.values).all())

  def test_check_national_vif_has_correct_vif_value(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    spec = eda_spec.EDASpec(vif_spec=eda_spec.VIFSpec(national_threshold=10))
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    data = _get_low_vif_da(geo_level=False)
    self._mock_eda_engine_property(
        "_stacked_national_treatment_control_scaled_da", data
    )

    outcome = engine.check_national_vif()

    self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
    self.assertEqual(outcome.check_type, eda_outcome.EDACheckType.VIF)
    self.assertLen(outcome.analysis_artifacts, 1)

    national_artifact = outcome.analysis_artifacts[0]
    self.assertIsInstance(national_artifact, eda_outcome.VIFArtifact)

    # Check national VIF
    national_data = data.values.reshape(-1, _N_VARS_VIF)
    national_data_with_const = sm.add_constant(national_data, prepend=True)
    expected_national_vif = [
        outliers_influence.variance_inflation_factor(
            national_data_with_const, i
        )
        for i in range(1, _N_VARS_VIF + 1)
    ]
    self.assertAllClose(national_artifact.vif_da.values, expected_national_vif)

  @parameterized.named_parameters(
      dict(
          testcase_name="has_variability",
          population_scaled_stdev=1.0,
          expected_result=True,
      ),
      dict(
          testcase_name="no_variability",
          population_scaled_stdev=0.0,
          expected_result=False,
      ),
      dict(
          testcase_name="below_threshold",
          population_scaled_stdev=eda_engine._STD_THRESHOLD / 2,
          expected_result=False,
      ),
      dict(
          testcase_name="at_threshold",
          population_scaled_stdev=eda_engine._STD_THRESHOLD,
          expected_result=True,
      ),
  )
  def test_kpi_has_variability(self, population_scaled_stdev, expected_result):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.kpi_transformer.population_scaled_stdev = population_scaled_stdev
    engine = eda_engine.EDAEngine(meridian)
    self.assertEqual(engine.kpi_has_variability, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          is_national=False,
          expected_kpi_name="population_scaled_kpi",
      ),
      dict(
          testcase_name="national",
          is_national=True,
          expected_kpi_name="kpi",
      ),
  )
  def test_check_overall_kpi_invariability_no_variability(
      self, is_national, expected_kpi_name
  ):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = is_national
    meridian.input_data.kpi = self.input_data_with_media_only.kpi
    mock_kpi_transformer = mock.Mock()
    mock_kpi_transformer.population_scaled_stdev = 0.0
    mock_kpi_transformer.population_scaled_mean = 100.0
    mock_kpi_transformer.population_scaled_kpi = backend.zeros(
        (self._N_GEOS, self._N_TIMES), dtype=backend.float32
    )
    meridian.kpi_transformer = mock_kpi_transformer
    engine = eda_engine.EDAEngine(meridian)

    outcome = engine.check_overall_kpi_invariability()

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.KPI_INVARIABILITY
    )
    self.assertLen(outcome.findings, 1)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.ERROR
    )
    expected_geo_text = "geos and " if not is_national else ""
    self.assertIn(
        f"`{expected_kpi_name}` is constant across all"
        f" {expected_geo_text}times",
        outcome.findings[0].explanation,
    )
    self.assertLen(outcome.analysis_artifacts, 1)
    artifact = outcome.analysis_artifacts[0]
    self.assertIsInstance(artifact, eda_outcome.KpiInvariabilityArtifact)
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.OVERALL)
    self.assertEqual(artifact.population_scaled_stdev, 0.0)
    self.assertEqual(artifact.population_scaled_mean, 100.0)
    self.assertAllClose(
        artifact.population_scaled_kpi_da.values,
        backend.to_tensor(mock_kpi_transformer.population_scaled_kpi),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          is_national=False,
      ),
      dict(
          testcase_name="national",
          is_national=True,
      ),
  )
  def test_check_overall_kpi_invariability_has_variability(self, is_national):
    meridian = mock.Mock(spec=model.Meridian)
    meridian.is_national = is_national
    meridian.input_data.kpi = self.input_data_with_media_only.kpi
    mock_kpi_transformer = mock.Mock()
    mock_kpi_transformer.population_scaled_stdev = 1.0
    mock_kpi_transformer.population_scaled_mean = 100.0
    mock_kpi_transformer.population_scaled_kpi = backend.to_tensor(
        np.arange(self._N_GEOS * self._N_TIMES).reshape(
            self._N_GEOS, self._N_TIMES
        ),
        dtype=backend.float32,
    )
    meridian.kpi_transformer = mock_kpi_transformer
    engine = eda_engine.EDAEngine(meridian)

    outcome = engine.check_overall_kpi_invariability()

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.KPI_INVARIABILITY
    )
    self.assertLen(outcome.findings, 1)
    self.assertEqual(outcome.findings[0].severity, eda_outcome.EDASeverity.INFO)
    expected_geo_text = "geos and " if not is_national else ""
    expected_kpi_name = "kpi" if is_national else "population_scaled_kpi"
    self.assertIn(
        f"The {expected_kpi_name} has variability across"
        f" {expected_geo_text}times",
        outcome.findings[0].explanation,
    )
    self.assertLen(outcome.analysis_artifacts, 1)
    artifact = outcome.analysis_artifacts[0]
    self.assertIsInstance(artifact, eda_outcome.KpiInvariabilityArtifact)
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.OVERALL)
    self.assertEqual(artifact.population_scaled_stdev, 1.0)
    self.assertEqual(artifact.population_scaled_mean, 100.0)
    self.assertAllClose(
        artifact.population_scaled_kpi_da.values,
        backend.to_tensor(mock_kpi_transformer.population_scaled_kpi),
    )


if __name__ == "__main__":
  absltest.main()
