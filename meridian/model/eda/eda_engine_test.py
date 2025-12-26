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

import itertools
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.model import model
from meridian.model import model_test_data
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import outliers_influence
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
  return da.rename({"var_dim": eda_constants.VARIABLE})


def _get_geo_high_vif_da():
  v1 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
  v2 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
  v3_geo0 = v1[0, :] * 2 + v2[0, :] * 0.5 + _RNG.random(_N_TIMES_VIF) * 0.01
  v3_geo1 = _RNG.random(_N_TIMES_VIF)
  v3 = np.stack([v3_geo0, v3_geo1], axis=0)
  data = np.stack([v1, v2, v3], axis=-1)
  da = _create_data_array_with_var_dim(data, "VIF", "var")
  return da.rename({"var_dim": eda_constants.VARIABLE})


def _get_overall_high_vif_da(geo_level: bool = True):
  sample_shape = (_N_GEOS_VIF, _N_TIMES_VIF) if geo_level else (_N_TIMES_VIF,)
  v1 = _RNG.random(sample_shape)
  v2 = _RNG.random(sample_shape)
  # v3 is a linear combination of v1 and v2, which results in an inf VIF value.
  v3 = v1 * 2 + v2 * 0.5
  data = np.stack([v1, v2, v3], axis=-1)
  da = _create_data_array_with_var_dim(data, "VIF", "var")
  return da.rename({"var_dim": eda_constants.VARIABLE})


def _create_ndarray_with_std_below_threshold(
    n_times: int, is_national: bool
) -> np.ndarray:
  """Creates an array with std without outliers equal to eda_constants.STD_THRESHOLD / 2."""
  target_std = eda_constants.STD_THRESHOLD / 2
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


def _create_eda_outcome(
    check_type: eda_outcome.EDACheckType,
    severity: eda_outcome.EDASeverity,
    finding_cause: eda_outcome.FindingCause,
) -> eda_outcome.EDAOutcome:
  """Creates an EDAOutcome with a single finding."""
  explanation = f"{check_type.name}: {severity.name}"
  return eda_outcome.EDAOutcome(
      check_type=check_type,
      findings=[
          eda_outcome.EDAFinding(
              severity=severity,
              explanation=explanation,
              finding_cause=finding_cause,
          )
      ],
      analysis_artifacts=[],
  )


class EDAEngineTest(
    test_utils.MeridianTestCase,
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
        lambda x: backend.cast(x, dtype=backend.float32)
        * self.mock_scale_factor
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
        lambda x: backend.cast(x, dtype=backend.float32)
        * self.mock_scale_factor
    )

  def _mock_critical_checks(
      self, mock_results: dict[str, eda_outcome.EDAOutcome | Exception]
  ):
    """Mocks critical EDA checks with specified return values or exceptions."""
    for check_name, result in mock_results.items():
      patcher = mock.patch.object(
          eda_engine.EDAEngine, check_name, autospec=True
      )
      mock_check = self.enter_context(patcher)
      if isinstance(result, Exception):
        mock_check.side_effect = result
      else:
        mock_check.return_value = result

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
    test_utils.assert_allclose(
        controls_scaled_da.values, meridian.controls_scaled
    )

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
        media_da.values, true_raw_media_da.values[:, start:, :]
    )

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
    test_utils.assert_allclose(national_media_raw_da.values, expected_da.values)

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
        media_da.values, meridian.media_tensors.media_spend
    )

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
    test_utils.assert_allclose(
        national_media_spend_da.values, expected_da.values
    )

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
    test_utils.assert_allclose(
        national_media_spend_da.values, expected_media_spend_da.values
    )

  def test_media_spend_da_with_1d_spend(self):
    input_data = self.input_data_with_media_and_rf.copy(deep=True)

    # Create 1D media_spend
    one_d_spend = np.array(
        [i + 1 for i in range(self._N_MEDIA_CHANNELS)], dtype=np.float64
    )
    media = input_data.media
    self.assertIsNotNone(media)
    input_data.media_spend = xr.DataArray(
        one_d_spend,
        dims=[constants.MEDIA_CHANNEL],
        coords={
            constants.MEDIA_CHANNEL: (
                media.coords[constants.MEDIA_CHANNEL].values
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
    test_utils.assert_allclose(
        media_spend_da.values, expected_allocated_spend.values
    )

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(rf_spend_da.values, meridian.rf_tensors.rf_spend)

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
    test_utils.assert_allclose(national_rf_spend_da.values, expected_da.values)

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
    test_utils.assert_allclose(
        national_rf_spend_da.values, expected_rf_spend_da.values
    )

  def test_rf_spend_da_with_1d_spend(self):
    input_data = self.input_data_with_media_and_rf.copy(deep=True)

    # Create 1D rf_spend
    one_d_spend = np.array(
        [i + 1 for i in range(self._N_RF_CHANNELS)], dtype=np.float64
    )
    reach = input_data.reach
    self.assertIsNotNone(reach)
    input_data.rf_spend = xr.DataArray(
        one_d_spend,
        dims=[constants.RF_CHANNEL],
        coords={
            constants.RF_CHANNEL: reach.coords[constants.RF_CHANNEL].values
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
    test_utils.assert_allclose(
        rf_spend_da.values, expected_allocated_spend.values
    )

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
        reach_da.values, true_reach_da.values[:, start:, :]
    )

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
    test_utils.assert_allclose(national_reach_raw_da.values, expected_values)

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(national_reach_scaled_da.values, expected_values)

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
    test_utils.assert_allclose(national_reach_scaled_da.values, expected_values)

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(rf_impressions_raw_da.values, expected_values)

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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
      mean_population = backend.reduce_mean(population)
      scale_factor = mean_population * self.mock_scale_factor
      mock_instance.forward.side_effect = (
          lambda tensor: backend.cast(tensor, dtype=backend.float32)
          * scale_factor
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
        1 if meridian.is_national else backend.reduce_mean(meridian.population)
    )
    expected_scale = mean_population * self.mock_scale_factor
    rf_impressions_raw_da = engine.rf_impressions_raw_da
    self.assertIsNotNone(rf_impressions_raw_da)
    expected_values = rf_impressions_raw_da.values * expected_scale
    test_utils.assert_allclose(rf_impressions_scaled_da.values, expected_values)

  # --- Test cases for national_rf_impressions_scaled_da ---
  def test_national_rf_impressions_scaled_da_with_geo_data(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_scaled_da = engine.national_rf_impressions_scaled_da
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
    test_utils.assert_allclose(
        national_rf_impressions_scaled_da.values, expected_values
    )

  def test_national_rf_impressions_scaled_da_with_national_data(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    national_rf_impressions_scaled_da = engine.national_rf_impressions_scaled_da
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
        national_organic_reach_raw_da.values, expected_values
    )

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
    test_utils.assert_allclose(
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

    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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

    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
        organic_rf_impressions_raw_da.values, expected_values
    )

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
    test_utils.assert_allclose(
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
    self.assertIsInstance(expected_organic_rf_impressions_raw_da, xr.DataArray)
    test_utils.assert_allclose(
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
      mean_population = backend.reduce_mean(population)
      scale_factor = mean_population * self.mock_scale_factor
      mock_instance.forward.side_effect = (
          lambda tensor: backend.cast(tensor, dtype=backend.float32)
          * scale_factor
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
        1 if meridian.is_national else backend.reduce_mean(meridian.population)
    )
    expected_scale = mean_population * self.mock_scale_factor
    organic_rf_impressions_raw_da = engine.organic_rf_impressions_raw_da
    self.assertIsNotNone(organic_rf_impressions_raw_da)
    expected_values = organic_rf_impressions_raw_da.values * expected_scale
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(
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
    test_utils.assert_allclose(population_da.values, meridian.population)

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
    test_utils.assert_allclose(kpi_da.values, meridian.kpi_scaled)

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
    test_utils.assert_allclose(
        national_kpi_scaled_da.values, scaled_expected_values
    )

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
    test_utils.assert_allclose(
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

  # --- Test cases for all_spend_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.MEDIA_SPEND],
          expected_dims={
              constants.MEDIA_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="rf_only",
          input_data_fixture="input_data_with_rf_only",
          expected_vars=[constants.RF_SPEND],
          expected_dims={
              constants.RF_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.RF_SPEND,
              constants.MEDIA_SPEND,
          ],
          expected_dims={
              constants.MEDIA_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="national_media_only",
          input_data_fixture="national_input_data_media_only",
          expected_vars=[constants.MEDIA_SPEND],
          expected_dims={
              constants.MEDIA_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="national_rf_only",
          input_data_fixture="national_input_data_rf_only",
          expected_vars=[constants.RF_SPEND],
          expected_dims={
              constants.RF_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.RF_SPEND,
              constants.MEDIA_SPEND,
          ],
          expected_dims={
              constants.MEDIA_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_SPEND: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
  )
  def test_all_spend_ds(self, input_data_fixture, expected_vars, expected_dims):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    tc_scaled_ds = engine.all_spend_ds
    self.assertIsInstance(tc_scaled_ds, xr.Dataset)

    self.assertCountEqual(tc_scaled_ds.data_vars.keys(), expected_vars)

    for var, dims in expected_dims.items():
      self.assertCountEqual(list(tc_scaled_ds[var].dims), dims)

  # --- Test cases for national_all_spend_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="national_media_only",
          input_data_fixture="national_input_data_media_only",
          expected_vars=[constants.NATIONAL_MEDIA_SPEND],
          expected_dims={
              constants.NATIONAL_MEDIA_SPEND: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="national_rf_only",
          input_data_fixture="national_input_data_rf_only",
          expected_vars=[constants.NATIONAL_RF_SPEND],
          expected_dims={
              constants.NATIONAL_RF_SPEND: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.NATIONAL_RF_SPEND,
              constants.NATIONAL_MEDIA_SPEND,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SPEND: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_SPEND: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.NATIONAL_MEDIA_SPEND],
          expected_dims={
              constants.NATIONAL_MEDIA_SPEND: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="rf_only",
          input_data_fixture="input_data_with_rf_only",
          expected_vars=[constants.NATIONAL_RF_SPEND],
          expected_dims={
              constants.NATIONAL_RF_SPEND: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.NATIONAL_RF_SPEND,
              constants.NATIONAL_MEDIA_SPEND,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SPEND: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_SPEND: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
  )
  def test_national_all_spend_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_all_spend_ds = engine.national_all_spend_ds
    self.assertIsInstance(national_all_spend_ds, xr.Dataset)

    self.assertCountEqual(
        national_all_spend_ds.data_vars.keys(),
        expected_vars,
    )

    for var in expected_vars:
      self.assertCountEqual(
          list(national_all_spend_ds[var].dims),
          expected_dims[var],
      )

  # --- Test cases for treatments_without_non_media_scaled_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.MEDIA_SCALED],
          expected_dims={
              constants.MEDIA_SCALED: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
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
          },
      ),
      dict(
          testcase_name="when_national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.MEDIA_SCALED,
              constants.RF_IMPRESSIONS_SCALED,
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
          },
      ),
  )
  def test_treatments_without_non_media_scaled_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    treatments_scaled_ds = engine.treatments_without_non_media_scaled_ds
    self.assertIsInstance(treatments_scaled_ds, xr.Dataset)

    self.assertCountEqual(treatments_scaled_ds.data_vars.keys(), expected_vars)

    for var, dims in expected_dims.items():
      self.assertCountEqual(list(treatments_scaled_ds[var].dims), dims)

  # --- Test cases for national_treatments_without_non_media_scaled_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA_SCALED: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
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
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA_SCALED,
              constants.NATIONAL_RF_IMPRESSIONS_SCALED,
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
          },
      ),
  )
  def test_national_treatments_without_non_media_scaled_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_treatments_scaled_ds = (
        engine.national_treatments_without_non_media_scaled_ds
    )
    self.assertIsInstance(national_treatments_scaled_ds, xr.Dataset)

    self.assertCountEqual(
        national_treatments_scaled_ds.data_vars.keys(),
        expected_vars,
    )

    for var in expected_vars:
      self.assertCountEqual(
          list(national_treatments_scaled_ds[var].dims),
          expected_dims[var],
      )

  # --- Test cases for paid_raw_media_units_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.MEDIA],
          expected_dims={
              constants.MEDIA: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="rf_only",
          input_data_fixture="input_data_with_rf_only",
          expected_vars=[constants.RF_IMPRESSIONS],
          expected_dims={
              constants.RF_IMPRESSIONS: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.MEDIA,
              constants.RF_IMPRESSIONS,
          ],
          expected_dims={
              constants.MEDIA: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.MEDIA,
              constants.RF_IMPRESSIONS,
          ],
          expected_dims={
              constants.MEDIA: [
                  constants.GEO,
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.RF_IMPRESSIONS: [
                  constants.GEO,
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
  )
  def test_paid_raw_media_units_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    paid_raw_media_units_ds = engine.paid_raw_media_units_ds
    self.assertIsInstance(paid_raw_media_units_ds, xr.Dataset)

    self.assertCountEqual(
        paid_raw_media_units_ds.data_vars.keys(), expected_vars
    )

    for var, dims in expected_dims.items():
      self.assertCountEqual(list(paid_raw_media_units_ds[var].dims), dims)

  # --- Test cases for national_paid_raw_media_units_ds ---
  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data_fixture="input_data_with_media_only",
          expected_vars=[constants.NATIONAL_MEDIA],
          expected_dims={
              constants.NATIONAL_MEDIA: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="rf_only",
          input_data_fixture="input_data_with_rf_only",
          expected_vars=[constants.NATIONAL_RF_IMPRESSIONS],
          expected_dims={
              constants.NATIONAL_RF_IMPRESSIONS: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ]
          },
      ),
      dict(
          testcase_name="media_rf",
          input_data_fixture="input_data_with_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA,
              constants.NATIONAL_RF_IMPRESSIONS,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
      dict(
          testcase_name="national_media_rf",
          input_data_fixture="national_input_data_media_and_rf",
          expected_vars=[
              constants.NATIONAL_MEDIA,
              constants.NATIONAL_RF_IMPRESSIONS,
          ],
          expected_dims={
              constants.NATIONAL_MEDIA: [
                  constants.TIME,
                  constants.MEDIA_CHANNEL,
              ],
              constants.NATIONAL_RF_IMPRESSIONS: [
                  constants.TIME,
                  constants.RF_CHANNEL,
              ],
          },
      ),
  )
  def test_national_paid_raw_media_units_ds(
      self, input_data_fixture, expected_vars, expected_dims
  ):
    meridian = model.Meridian(getattr(self, input_data_fixture))
    engine = eda_engine.EDAEngine(meridian)
    national_paid_raw_media_units_ds = engine.national_paid_raw_media_units_ds
    self.assertIsInstance(national_paid_raw_media_units_ds, xr.Dataset)

    self.assertCountEqual(
        national_paid_raw_media_units_ds.data_vars.keys(),
        expected_vars,
    )

    for var, dims in expected_dims.items():
      self.assertCountEqual(
          list(national_paid_raw_media_units_ds[var].dims), dims
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
    test_utils.assert_allclose(all_reach_scaled_da.values, expected_da.values)

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
    test_utils.assert_allclose(all_freq_da.values, expected_da.values)

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
    test_utils.assert_allclose(
        national_all_reach_scaled_da.values, expected_da.values
    )

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
    test_utils.assert_allclose(national_all_freq_da.values, expected_da.values)

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

    with self.subTest("check_type"):
      self.assertEqual(
          outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
      )

    with self.subTest("findings_and_artifacts_count"):
      self.assertLen(outcome.findings, 1)
      self.assertLen(outcome.analysis_artifacts, 2)

    (finding,) = outcome.findings
    with self.subTest("finding_details"):
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
    expected_overall_extreme_corr_df = pd.DataFrame(
        data={
            eda_constants.CORRELATION: [1.0],
            eda_constants.ABS_CORRELATION_COL_NAME: [1.0],
        },
        index=pd.MultiIndex.from_tuples(
            [("media_1", "media_2")],
            names=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        ),
    )
    with self.subTest("overall_artifact_details"):
      self.assertEqual(
          overall_artifact.extreme_corr_threshold,
          eda_constants.OVERALL_PAIRWISE_CORR_THRESHOLD,
      )
      pd.testing.assert_frame_equal(
          overall_artifact.extreme_corr_var_pairs,
          expected_overall_extreme_corr_df,
          check_dtype=False,
          atol=1e-6,
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

    with self.subTest("check_type"):
      self.assertEqual(
          outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
      )

    with self.subTest("findings_and_artifacts_count"):
      self.assertLen(outcome.findings, 1)
      self.assertLen(outcome.analysis_artifacts, 2)

    (finding,) = outcome.findings
    with self.subTest("finding_details"):
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
    with self.subTest("geo_artifact_details"):
      all_vars = (
          geo_artifact.extreme_corr_var_pairs.index.to_frame().stack().unique()
      )
      self.assertIn("media_1", all_vars)
      self.assertIn("media_2", all_vars)
      self.assertEqual(
          geo_artifact.extreme_corr_threshold,
          eda_constants.GEO_PAIRWISE_CORR_THRESHOLD,
      )

  def test_check_geo_pairwise_corr_returns_error_and_attention(self):
    # data shape: (2, 3, 3) -> (n_geos, n_times, n_vars)
    # media_1 and media_2 are perfectly correlated overall -> ERROR
    # In geo0, media_1, media_2, and media_3 are identical, so all pairwise
    # correlations are 1.0; in geo1, media_1 and media_2 are perfectly
    # correlated, but the others are not. -> ATTENTION for geo-level.
    data = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],  # geo0
            [[4, 4, 3], [5, 5, 7], [6, 6, 5]],  # geo1
        ],
        dtype=float,
    )
    mock_ds = _create_dataset_with_var_dim(data)
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    self._mock_eda_engine_property("treatment_control_scaled_ds", mock_ds)
    outcome = engine.check_geo_pairwise_corr()

    with self.subTest("two_findings"):
      self.assertLen(outcome.findings, 2)

    findings_by_severity = {
        severity: list(group)
        for severity, group in itertools.groupby(
            outcome.findings, key=lambda f: f.severity
        )
    }

    error_findings = findings_by_severity[eda_outcome.EDASeverity.ERROR]
    with self.subTest("error_finding"):
      self.assertLen(error_findings, 1)
      (error_finding,) = error_findings
      self.assertIn("('media_1', 'media_2')", error_finding.explanation)

    attention_findings = findings_by_severity[eda_outcome.EDASeverity.ATTENTION]
    with self.subTest("attention_finding"):
      self.assertLen(attention_findings, 1)
      (attention_finding,) = attention_findings
      self.assertIn(
          "perfect pairwise correlation in certain geo(s)",
          attention_finding.explanation,
      )

    artifacts_by_level = {
        level: list(group)
        for level, group in itertools.groupby(
            outcome.analysis_artifacts, key=lambda art: art.level
        )
    }
    overall_artifacts = artifacts_by_level[eda_outcome.AnalysisLevel.OVERALL]
    with self.subTest("overall_artifact"):
      self.assertLen(overall_artifacts, 1)
      (overall_artifact,) = overall_artifacts
      self.assertCountEqual(
          overall_artifact.extreme_corr_var_pairs.index.to_list(),
          [("media_1", "media_2")],
      )

    geo_artifacts = artifacts_by_level[eda_outcome.AnalysisLevel.GEO]
    with self.subTest("geo_artifact"):
      self.assertLen(geo_artifacts, 1)
      (geo_artifact,) = geo_artifacts
      # In geo0, media_1, media_2, and media_3 are all identical, so all
      # pairwise correlations are 1.0.
      self.assertCountEqual(
          [
              ("media_1", "media_2"),
              ("media_1", "media_3"),
              ("media_2", "media_3"),
          ],
          geo_artifact.extreme_corr_var_pairs.loc["geo0"].index.to_list(),
      )
      # In geo1, media_1 and media_2 are perfectly correlated, but the others
      # are not.
      self.assertCountEqual(
          [
              ("media_1", "media_2"),
          ],
          geo_artifact.extreme_corr_var_pairs.loc["geo1"].index.to_list(),
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    (finding,) = outcome.findings
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

    self.assertEmpty(overall_artifact.extreme_corr_var_pairs)
    self.assertEmpty(geo_artifact.extreme_corr_var_pairs)

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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    (finding,) = outcome.findings
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 2)

    (finding,) = outcome.findings
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.analysis_artifacts, 2)

    for artifact in outcome.analysis_artifacts:
      if artifact.level == eda_outcome.AnalysisLevel.OVERALL:
        self.assertCountEqual(
            artifact.corr_matrix.coords.keys(),
            [eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        )
      elif artifact.level == eda_outcome.AnalysisLevel.GEO:
        self.assertCountEqual(
            artifact.corr_matrix.coords.keys(),
            [constants.GEO, eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )

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
    geo_corr_mat = geo_artifact.corr_matrix

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

    # With correlation values 1.0 and -1.0, and threshold 0.999, both pairs
    # should be in extreme_corr_var_pairs, sorted by abs_correlation desc.
    expected_geo_extreme_corr_df = pd.DataFrame(
        data={
            eda_constants.CORRELATION: [1.0, -1.0],
            eda_constants.ABS_CORRELATION_COL_NAME: [1.0, 1.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("geo0", "media_1", "control_1"),
                ("geo1", "media_1", "control_1"),
            ],
            names=[
                constants.GEO,
                eda_constants.VARIABLE_1,
                eda_constants.VARIABLE_2,
            ],
        ),
    )

    with self.subTest("overall_artifact"):
      self.assertEqual(
          overall_corr_mat.name, eda_constants.CORRELATION_MATRIX_NAME
      )
      # Check overall correlation
      test_utils.assert_allclose(
          overall_corr_mat.sel(var1="media_1", var2="control_1").values,
          expected_overall_corr,
      )
      self.assertEmpty(overall_artifact.extreme_corr_var_pairs)

    with self.subTest("geo_artifact"):
      self.assertEqual(geo_corr_mat.name, eda_constants.CORRELATION_MATRIX_NAME)
      # Check geo correlations
      test_utils.assert_allclose(
          geo_corr_mat.sel(var1="media_1", var2="control_1").values,
          expected_geo_corr,
      )
      pd.testing.assert_frame_equal(
          geo_artifact.extreme_corr_var_pairs.sort_index(),
          expected_geo_extreme_corr_df.sort_index(),
          check_dtype=False,
          atol=1e-6,
      )

  def test_national_extreme_corr_var_pairs_are_correctly_sorted(self):
    self.enter_context(
        mock.patch.object(
            eda_constants, "NATIONAL_PAIRWISE_CORR_THRESHOLD", 0.7
        )
    )
    meridian = model.Meridian(self.national_input_data_media_only)
    engine = eda_engine.EDAEngine(meridian)
    # data for forcing correlation order
    # m0=[1,2,3], c0=[1,2,4], c1=[-1,-2,-2]
    # c(m0,c0)=0.98198, c(m0,c1)=-0.866025, c(c0,c1)=-0.755928
    # Abs: 0.98198, 0.866025, 0.755928.
    data = np.array([
        [1, 1, -1],
        [2, 2, -2],
        [3, 4, -2],
    ]).astype(float)
    national_da = (
        _create_data_array_with_var_dim(data, name="data", var_name="variable")
        .rename({"variable_dim": eda_constants.VARIABLE})
        .assign_coords(
            {
                eda_constants.VARIABLE: [
                    "media_0",
                    "control_0",
                    "control_1",
                ]
            }
        )
    )
    self._mock_eda_engine_property(
        "_stacked_national_treatment_control_scaled_da", national_da
    )

    outcome = engine.check_national_pairwise_corr()
    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)

    self.assertListEqual(
        artifact.extreme_corr_var_pairs.index.to_list(),
        [
            ("media_0", "control_0"),
            ("media_0", "control_1"),
            ("control_0", "control_1"),
        ],
    )

  def test_geo_extreme_corr_var_pairs_are_correctly_sorted(self):
    self.enter_context(
        mock.patch.object(eda_constants, "GEO_PAIRWISE_CORR_THRESHOLD", 0.7)
    )
    self.enter_context(
        mock.patch.object(eda_constants, "OVERALL_PAIRWISE_CORR_THRESHOLD", 0.7)
    )
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)
    # data for forcing correlation order
    # m0=[1,2,3], c0=[1,2,4], c1=[-1,-2,-2]
    # c(m0,c0)=0.98198, c(m0,c1)=-0.866025, c(c0,c1)=-0.755928
    # Abs: 0.98198, 0.866025, 0.755928.
    data_1geo = np.array([
        [1, 1, -1],
        [2, 2, -2],
        [3, 4, -2],
    ]).astype(float)
    # Use 2 geos for test
    n_geos = 2
    data = np.stack([data_1geo] * n_geos, axis=0)
    geo_da = (
        _create_data_array_with_var_dim(data, name="data", var_name="variable")
        .rename({"variable_dim": eda_constants.VARIABLE})
        .assign_coords(
            {
                eda_constants.VARIABLE: [
                    "media_0",
                    "control_0",
                    "control_1",
                ]
            }
        )
    )
    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", geo_da
    )
    outcome = engine.check_geo_pairwise_corr()
    overall_artifact = next(
        art
        for art in outcome.analysis_artifacts
        if art.level == eda_outcome.AnalysisLevel.OVERALL
    )
    geo_artifact = next(
        art
        for art in outcome.analysis_artifacts
        if art.level == eda_outcome.AnalysisLevel.GEO
    )
    geo_df = geo_artifact.extreme_corr_var_pairs.reset_index()

    with self.subTest("overall_artifact"):
      # Check OVERALL artifact
      self.assertListEqual(
          overall_artifact.extreme_corr_var_pairs.index.to_list(),
          [
              ("media_0", "control_0"),
              ("media_0", "control_1"),
              ("control_0", "control_1"),
          ],
      )

    with self.subTest("geo_artifact"):
      # Check GEO artifact
      self.assertListEqual(
          list(
              zip(
                  geo_df[eda_constants.VARIABLE_1],
                  geo_df[eda_constants.VARIABLE_2],
              )
          ),
          [
              ("media_0", "control_0"),
              ("media_0", "control_0"),
              ("media_0", "control_1"),
              ("media_0", "control_1"),
              ("control_0", "control_1"),
              ("control_0", "control_1"),
          ],
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 1)

    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    self.assertIn(
        "perfect pairwise correlation across all times",
        finding.explanation,
    )
    self.assertIn(
        "Pairs with perfect correlation: [('media_1', 'media_2')]",
        finding.explanation,
    )

    (artifact,) = outcome.analysis_artifacts
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)
    self.assertEqual(
        artifact.extreme_corr_threshold,
        eda_constants.NATIONAL_PAIRWISE_CORR_THRESHOLD,
    )
    expected_national_extreme_corr_df = pd.DataFrame(
        data={
            eda_constants.CORRELATION: [1.0],
            eda_constants.ABS_CORRELATION_COL_NAME: [1.0],
        },
        index=pd.MultiIndex.from_tuples(
            [("media_1", "media_2")],
            names=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        ),
    )
    pd.testing.assert_frame_equal(
        artifact.extreme_corr_var_pairs,
        expected_national_extreme_corr_df,
        check_dtype=False,
        atol=1e-6,
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.findings, 1)
    self.assertLen(outcome.analysis_artifacts, 1)

    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review the computed pairwise correlations",
        finding.explanation,
    )

    (artifact,) = outcome.analysis_artifacts
    self.assertEmpty(artifact.extreme_corr_var_pairs)

  def test_check_national_pairwise_corr_corr_matrix_has_correct_coordinates(
      self,
  ):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_national_pairwise_corr()

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)
    self.assertCountEqual(
        artifact.corr_matrix.coords.keys(),
        [eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    )
    expected_corr = np.corrcoef(media_data.flatten(), control_data.flatten())[
        0, 1
    ]

    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    corr_mat = artifact.corr_matrix
    self.assertEqual(corr_mat.name, eda_constants.CORRELATION_MATRIX_NAME)

    test_utils.assert_allclose(
        corr_mat.sel(var1="media_1", var2="control_1").values,
        expected_corr,
    )

  def test_check_geo_std_raises_error_for_national_model(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 4)

    for artifact in outcome.analysis_artifacts:
      if artifact.variable == constants.KPI_SCALED:
        self.assertCountEqual(artifact.std_ds.coords.keys(), [constants.GEO])
      elif artifact.variable == constants.TREATMENT_CONTROL_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [constants.GEO, eda_constants.VARIABLE],
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 2)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.KPI_SCALED
    )

    expected_kpi_std_value_with_outliers = np.std([1, 2, 3, 4, 5, 100], ddof=1)
    expected_kpi_std_value_without_outliers = np.std([1, 2, 3, 4, 5], ddof=1)
    test_utils.assert_allclose(
        kpi_artifact.std_ds[eda_constants.STD_WITH_OUTLIERS_VAR_NAME].values[0],
        expected_kpi_std_value_with_outliers,
    )
    test_utils.assert_allclose(
        kpi_artifact.std_ds[eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME].values[
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 2)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.KPI_SCALED
    )

    self.assertGreater(
        kpi_artifact.std_ds[eda_constants.STD_WITH_OUTLIERS_VAR_NAME].values[0],
        kpi_artifact.std_ds[eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME].values[
            0
        ],
    )
    self.assertFalse(kpi_artifact.outlier_df.empty)
    self.assertEqual(
        kpi_artifact.outlier_df[eda_constants.OUTLIERS_COL_NAME].iloc[0],
        outlier_value,
    )

  def test_check_geo_std_returns_info_finding_when_no_issues(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review any identified outliers",
        finding.explanation,
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
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(expected_message_substr, finding.explanation)

  def test_check_geo_std_handles_missing_rf_data(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 4)

    for artifact in outcome.analysis_artifacts:
      if artifact.variable == constants.NATIONAL_KPI_SCALED:
        self.assertCountEqual(artifact.std_ds.coords.keys(), [])
      elif artifact.variable == constants.NATIONAL_TREATMENT_CONTROL_SCALED:
        self.assertCountEqual(
            artifact.std_ds.coords.keys(),
            [eda_constants.VARIABLE],
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 4)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.NATIONAL_KPI_SCALED
    )

    expected_kpi_std_value_with_outliers = np.std([1, 2, 3, 4, 5, 100], ddof=1)
    expected_kpi_std_value_without_outliers = np.std([1, 2, 3, 4, 5], ddof=1)
    test_utils.assert_allclose(
        kpi_artifact.std_ds[eda_constants.STD_WITH_OUTLIERS_VAR_NAME].values,
        expected_kpi_std_value_with_outliers,
    )
    test_utils.assert_allclose(
        kpi_artifact.std_ds[eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME].values,
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.analysis_artifacts, 4)
    kpi_artifact = next(
        artifact
        for artifact in outcome.analysis_artifacts
        if artifact.variable == constants.NATIONAL_KPI_SCALED
    )
    self.assertGreater(
        kpi_artifact.std_ds[
            eda_constants.STD_WITH_OUTLIERS_VAR_NAME
        ].values.item(),
        kpi_artifact.std_ds[
            eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME
        ].values.item(),
    )
    self.assertFalse(kpi_artifact.outlier_df.empty)
    self.assertEqual(
        kpi_artifact.outlier_df[eda_constants.OUTLIERS_COL_NAME].iloc[0],
        outlier_value,
    )

  def test_check_national_std_returns_info_finding_when_no_issues(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    self.assertIn(
        "Please review any identified outliers",
        finding.explanation,
    )

  def test_check_national_std_finds_zero_std_kpi(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(
        "The standard deviation of the scaled KPI drops",
        finding.explanation,
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
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ATTENTION)
    self.assertIn(expected_message_substr, finding.explanation)

  def test_check_national_std_handles_missing_rf_data(self):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.STANDARD_DEVIATION
    )
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

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, expected_severity)
    self.assertIn(expected_explanation, finding.explanation)

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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
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
        [eda_constants.VARIABLE],
    )
    self.assertEqual(overall_artifact.vif_da.shape, (_N_VARS_VIF,))
    # With overall_threshold=1e6 and _get_geo_vif_da(), we expect no overall
    # outliers
    self.assertEmpty(overall_artifact.outlier_df)

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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
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
        [constants.GEO, eda_constants.VARIABLE],
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

  def test_check_geo_vif_returns_error_and_attention(self):
    # var_1 and var_2 are perfectly collinear -> ERROR
    # var_3 and var_4 are perfectly collinear in geo0 only -> ATTENTION
    v1 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
    v2 = v1
    v3 = _RNG.random((_N_GEOS_VIF, _N_TIMES_VIF))
    v4_geo0 = v3[0, :]
    v4_geo1 = _RNG.random(_N_TIMES_VIF)
    v4 = np.stack([v4_geo0, v4_geo1], axis=0)
    data = np.stack([v1, v2, v3, v4], axis=-1)
    mock_da = _create_data_array_with_var_dim(data, "VIF", "var").rename(
        {"var_dim": eda_constants.VARIABLE}
    )

    meridian = model.Meridian(self.input_data_with_media_only)
    spec = eda_spec.EDASpec(
        vif_spec=eda_spec.VIFSpec(overall_threshold=10, geo_threshold=5)
    )
    engine = eda_engine.EDAEngine(meridian, spec=spec)
    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", mock_da
    )

    outcome = engine.check_geo_vif()

    with self.subTest("two_findings"):
      self.assertLen(outcome.findings, 2)

    findings_by_severity = {
        severity: list(group)
        for severity, group in itertools.groupby(
            outcome.findings, key=lambda f: f.severity
        )
    }

    error_findings = findings_by_severity[eda_outcome.EDASeverity.ERROR]
    with self.subTest("error_finding"):
      self.assertLen(error_findings, 1)
      (error_finding,) = error_findings
      self.assertIn("var_1", error_finding.explanation)
      self.assertIn("var_2", error_finding.explanation)

    attention_findings = findings_by_severity[eda_outcome.EDASeverity.ATTENTION]
    with self.subTest("attention_finding"):
      self.assertLen(attention_findings, 1)
      (attention_finding,) = attention_findings
      self.assertIn(
          "Some variables have extreme multicollinearity (with VIF > 5) in"
          " certain geo(s).",
          attention_finding.explanation,
      )

    artifacts_by_level = {
        level: list(group)
        for level, group in itertools.groupby(
            outcome.analysis_artifacts, key=lambda art: art.level
        )
    }
    overall_artifacts = artifacts_by_level[eda_outcome.AnalysisLevel.OVERALL]
    with self.subTest("overall_artifact"):
      self.assertLen(overall_artifacts, 1)
      (overall_artifact,) = overall_artifacts
      self.assertCountEqual(
          overall_artifact.outlier_df.index.to_list(), ["var_1", "var_2"]
      )

    geo_artifacts = artifacts_by_level[eda_outcome.AnalysisLevel.GEO]
    with self.subTest("geo_artifact"):
      self.assertLen(geo_artifacts, 1)
      (geo_artifact,) = geo_artifacts
      self.assertCountEqual(
          geo_artifact.outlier_df.index.to_list(),
          [
              ("geo0", "var_1"),
              ("geo0", "var_2"),
              ("geo0", "var_3"),
              ("geo0", "var_4"),
              ("geo1", "var_1"),
              ("geo1", "var_2"),
          ],
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
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
    test_utils.assert_allclose(
        overall_artifact.vif_da.values, expected_overall_vif
    )

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
    test_utils.assert_allclose(geo_artifact.vif_da.values, expected_geo_vif)

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
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, expected_severity)
    self.assertIn(expected_explanation, finding.explanation)

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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
    self.assertLen(outcome.analysis_artifacts, 1)

    (national_artifact,) = outcome.analysis_artifacts
    self.assertIsInstance(national_artifact, eda_outcome.VIFArtifact)
    self.assertEqual(
        national_artifact.level, eda_outcome.AnalysisLevel.NATIONAL
    )
    self.assertCountEqual(
        national_artifact.vif_da.coords.keys(),
        [eda_constants.VARIABLE],
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
    self.assertLen(outcome.analysis_artifacts, 1)

    (national_artifact,) = outcome.analysis_artifacts
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
    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
    )
    self.assertLen(outcome.analysis_artifacts, 1)

    (national_artifact,) = outcome.analysis_artifacts
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
    test_utils.assert_allclose(
        national_artifact.vif_da.values, expected_national_vif
    )

  # TODO: Re-enable this test once the bug is fixed.
  # def test_check_vif_with_constant_variable(self):
  #   meridian = model.Meridian(self.national_input_data_media_and_rf)
  #   engine = eda_engine.EDAEngine(meridian)
  #   shape = (_N_TIMES_VIF,)
  #   v1 = _RNG.random(shape)
  #   v2 = np.ones(shape)
  #   v3 = _RNG.random(shape)
  #   data_np = np.stack([v1, v2, v3], axis=-1)
  #   data = (
  #       _create_data_array_with_var_dim(data_np, "VIF", "var")
  #       .rename({"var_dim": eda_constants.VARIABLE})
  #       .assign_coords({eda_constants.VARIABLE: ["var_1", "var_2", "var_3"]})
  #   )
  #   self._mock_eda_engine_property(
  #       "_stacked_national_treatment_control_scaled_da", data
  #   )

  #   outcome = engine.check_national_vif()

  #   self.assertIsInstance(outcome, eda_outcome.EDAOutcome)
  #   self.assertEqual(
  #       outcome.check_type, eda_outcome.EDACheckType.MULTICOLLINEARITY
  #   )
  #   self.assertLen(outcome.analysis_artifacts, 1)

  #   national_artifact, = outcome.analysis_artifacts
  #   self.assertIsInstance(national_artifact, eda_outcome.VIFArtifact)
  #   self.assertEqual(national_artifact.vif_da.sel(var="var_2"), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="national_model",
          is_national=True,
          expected_call="check_national_std",
      ),
      dict(
          testcase_name="geo_model",
          is_national=False,
          expected_call="check_geo_std",
      ),
  )
  def test_check_std_calls_correct_level(self, is_national, expected_call):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    engine = eda_engine.EDAEngine(meridian)

    mock_outcome = _create_eda_outcome(
        eda_outcome.EDACheckType.STANDARD_DEVIATION,
        eda_outcome.EDASeverity.INFO,
        eda_outcome.FindingCause.NONE,
    )
    mock_check = self.enter_context(
        mock.patch.object(
            engine, expected_call, autospec=True, return_value=mock_outcome
        )
    )
    result = engine.check_std()
    mock_check.assert_called_once()
    self.assertEqual(result, mock_outcome)

  @parameterized.named_parameters(
      dict(
          testcase_name="national_model",
          is_national=True,
          expected_call="check_national_vif",
      ),
      dict(
          testcase_name="geo_model",
          is_national=False,
          expected_call="check_geo_vif",
      ),
  )
  def test_check_vif_calls_correct_level(self, is_national, expected_call):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    engine = eda_engine.EDAEngine(meridian)

    mock_outcome = _create_eda_outcome(
        eda_outcome.EDACheckType.MULTICOLLINEARITY,
        eda_outcome.EDASeverity.INFO,
        eda_outcome.FindingCause.NONE,
    )
    mock_check = self.enter_context(
        mock.patch.object(
            engine, expected_call, autospec=True, return_value=mock_outcome
        )
    )
    result = engine.check_vif()
    mock_check.assert_called_once()
    self.assertEqual(result, mock_outcome)

  @parameterized.named_parameters(
      dict(
          testcase_name="national_model",
          is_national=True,
          expected_call="check_national_pairwise_corr",
      ),
      dict(
          testcase_name="geo_model",
          is_national=False,
          expected_call="check_geo_pairwise_corr",
      ),
  )
  def test_check_pairwise_corr_calls_correct_level(
      self, is_national, expected_call
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    engine = eda_engine.EDAEngine(meridian)

    mock_outcome = _create_eda_outcome(
        eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        eda_outcome.EDASeverity.INFO,
        eda_outcome.FindingCause.NONE,
    )
    mock_check = self.enter_context(
        mock.patch.object(
            engine, expected_call, autospec=True, return_value=mock_outcome
        )
    )
    result = engine.check_pairwise_corr()
    mock_check.assert_called_once()
    self.assertEqual(result, mock_outcome)

  @parameterized.named_parameters(
      dict(
          testcase_name="has_variability",
          kpi_scaled_stdev=1.0,
          expected_result=True,
      ),
      dict(
          testcase_name="no_variability",
          kpi_scaled_stdev=0.0,
          expected_result=False,
      ),
      dict(
          testcase_name="below_threshold",
          kpi_scaled_stdev=eda_constants.STD_THRESHOLD / 2,
          expected_result=False,
      ),
      dict(
          testcase_name="at_threshold",
          kpi_scaled_stdev=eda_constants.STD_THRESHOLD,
          expected_result=True,
      ),
  )
  def test_kpi_has_variability(self, kpi_scaled_stdev, expected_result):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    engine = eda_engine.EDAEngine(meridian)
    mock_kpi_scaled_da = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=False
    )
    mock_kpi_scaled_da.std.return_value = xr.DataArray(kpi_scaled_stdev)
    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_scaled_da)
    self.assertEqual(engine.kpi_has_variability, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          is_national=False,
          kpi_data=np.ones((5, 200)),
      ),
      dict(
          testcase_name="national",
          is_national=True,
          kpi_data=np.ones((1, 200)),
      ),
  )
  def test_check_overall_kpi_invariability_no_variability(
      self, is_national, kpi_data
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    meridian.input_data.kpi = self.input_data_with_media_only.kpi
    engine = eda_engine.EDAEngine(meridian)

    mock_kpi_scaled_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.KPI_SCALED,
    )
    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_scaled_da)

    outcome = engine.check_overall_kpi_invariability()

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.KPI_INVARIABILITY
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.ERROR)
    expected_geo_text = "geos and " if not is_national else ""
    self.assertIn(
        f"`{constants.KPI_SCALED}` is constant across all"
        f" {expected_geo_text}times",
        finding.explanation,
    )
    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertIsInstance(artifact, eda_outcome.KpiInvariabilityArtifact)
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.OVERALL)
    self.assertAlmostEqual(artifact.kpi_stdev, 0.0)
    test_utils.assert_allclose(
        artifact.kpi_da.values,
        kpi_data,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          is_national=False,
          kpi_data=np.arange(5 * 200).reshape(5, 200),
      ),
      dict(
          testcase_name="national",
          is_national=True,
          kpi_data=np.arange(1 * 200).reshape(1, 200),
      ),
  )
  def test_check_overall_kpi_invariability_has_variability(
      self, is_national, kpi_data
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    meridian.input_data.kpi = self.input_data_with_media_only.kpi
    engine = eda_engine.EDAEngine(meridian)

    mock_kpi_scaled_da = _create_data_array_with_var_dim(
        kpi_data,
        name=constants.KPI_SCALED,
    )
    self._mock_eda_engine_property("kpi_scaled_da", mock_kpi_scaled_da)

    outcome = engine.check_overall_kpi_invariability()

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.KPI_INVARIABILITY
    )
    self.assertLen(outcome.findings, 1)
    (finding,) = outcome.findings
    self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
    expected_geo_text = "geos and " if not is_national else ""
    self.assertIn(
        f"The {constants.KPI_SCALED} has variability across"
        f" {expected_geo_text}times",
        finding.explanation,
    )
    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertIsInstance(artifact, eda_outcome.KpiInvariabilityArtifact)
    self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.OVERALL)
    self.assertGreater(artifact.kpi_stdev, eda_constants.STD_THRESHOLD)
    test_utils.assert_allclose(
        artifact.kpi_da.values,
        kpi_data,
    )

  def test_check_geo_cost_per_media_unit_raises_error_for_national_model(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      engine.check_geo_cost_per_media_unit()

  @parameterized.named_parameters([
      dict(
          testcase_name="no_issues",
          spend_data=np.full((1, 10, 1), 10.0),
          media_unit_data=np.full((1, 10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.INFO,
          expected_findings_count=1,
          expected_inconsistency_df_empty=True,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="inconsistent_zero_spend",
          spend_data=np.array(
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
          ).reshape((1, 10, 1)),
          media_unit_data=np.full((1, 10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="inconsistent_positive_spend",
          spend_data=np.full((1, 10, 1), 10.0),
          media_unit_data=np.array(
              [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
          ).reshape((1, 10, 1)),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="outliers",
          spend_data=np.array(
              [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0]
          ).reshape((1, 10, 1)),
          media_unit_data=np.full((1, 10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=True,
          expected_outlier_df_empty=False,
      ),
      dict(
          testcase_name="inconsistency_and_outliers",
          spend_data=np.array(
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
          ).reshape((1, 10, 1)),
          media_unit_data=np.full((1, 10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=2,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=False,
      ),
  ])
  def test_check_geo_cost_per_media_unit(
      self,
      spend_data,
      media_unit_data,
      expected_severity,
      expected_findings_count,
      expected_inconsistency_df_empty,
      expected_outlier_df_empty,
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = False
    engine = eda_engine.EDAEngine(meridian)
    spend_ds = _create_dataset_with_var_dim(
        spend_data, var_name="media"
    ).rename(
        {"media_dim": constants.MEDIA_CHANNEL, "media": constants.MEDIA_SPEND}
    )
    media_unit_ds = _create_dataset_with_var_dim(
        media_unit_data, var_name="media"
    ).rename({"media_dim": constants.MEDIA_CHANNEL})
    self._mock_eda_engine_property("all_spend_ds", spend_ds)
    self._mock_eda_engine_property("paid_raw_media_units_ds", media_unit_ds)

    outcome = engine.check_geo_cost_per_media_unit()

    with self.subTest("check_type"):
      self.assertEqual(
          outcome.check_type, eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT
      )

    with self.subTest("findings"):
      self.assertLen(outcome.findings, expected_findings_count)
      self.assertEqual(
          [finding.severity for finding in outcome.findings],
          [expected_severity] * expected_findings_count,
      )

    with self.subTest("analysis_artifacts"):
      self.assertLen(outcome.analysis_artifacts, 1)
      (artifact,) = outcome.analysis_artifacts
      self.assertIsInstance(artifact, eda_outcome.CostPerMediaUnitArtifact)
      self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.GEO)
      self.assertEqual(
          artifact.cost_media_unit_inconsistency_df.empty,
          expected_inconsistency_df_empty,
      )
      self.assertEqual(
          artifact.outlier_df.empty,
          expected_outlier_df_empty,
      )

  @parameterized.named_parameters([
      dict(
          testcase_name="no_issues",
          spend_data=np.full((10, 1), 10.0),
          media_unit_data=np.full((10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.INFO,
          expected_findings_count=1,
          expected_inconsistency_df_empty=True,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="inconsistent_zero_spend",
          spend_data=np.array(
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
          ).reshape((10, 1)),
          media_unit_data=np.full((10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="inconsistent_positive_spend",
          spend_data=np.full((10, 1), 10.0),
          media_unit_data=np.array(
              [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
          ).reshape((10, 1)),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=True,
      ),
      dict(
          testcase_name="outliers",
          spend_data=np.array(
              [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0]
          ).reshape((10, 1)),
          media_unit_data=np.full((10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=1,
          expected_inconsistency_df_empty=True,
          expected_outlier_df_empty=False,
      ),
      dict(
          testcase_name="inconsistency_and_outliers",
          spend_data=np.array(
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
          ).reshape((10, 1)),
          media_unit_data=np.full((10, 1), 10.0),
          expected_severity=eda_outcome.EDASeverity.ATTENTION,
          expected_findings_count=2,
          expected_inconsistency_df_empty=False,
          expected_outlier_df_empty=False,
      ),
  ])
  def test_check_national_cost_per_media_unit(
      self,
      spend_data,
      media_unit_data,
      expected_severity,
      expected_findings_count,
      expected_inconsistency_df_empty,
      expected_outlier_df_empty,
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = True
    engine = eda_engine.EDAEngine(meridian)
    spend_ds = _create_dataset_with_var_dim(
        spend_data, var_name="media"
    ).rename(
        {"media_dim": constants.MEDIA_CHANNEL, "media": constants.MEDIA_SPEND}
    )
    media_unit_ds = _create_dataset_with_var_dim(
        media_unit_data, var_name="media"
    ).rename({"media_dim": constants.MEDIA_CHANNEL})
    self._mock_eda_engine_property("national_all_spend_ds", spend_ds)
    self._mock_eda_engine_property(
        "national_paid_raw_media_units_ds", media_unit_ds
    )

    outcome = engine.check_national_cost_per_media_unit()

    with self.subTest("check_type"):
      self.assertEqual(
          outcome.check_type, eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT
      )

    with self.subTest("findings"):
      self.assertLen(outcome.findings, expected_findings_count)
      self.assertEqual(
          [finding.severity for finding in outcome.findings],
          [expected_severity] * expected_findings_count,
      )

    with self.subTest("analysis_artifacts"):
      self.assertLen(outcome.analysis_artifacts, 1)
      (artifact,) = outcome.analysis_artifacts
      self.assertIsInstance(artifact, eda_outcome.CostPerMediaUnitArtifact)
      self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.NATIONAL)
      self.assertEqual(
          artifact.cost_media_unit_inconsistency_df.empty,
          expected_inconsistency_df_empty,
      )
      self.assertEqual(
          artifact.outlier_df.empty,
          expected_outlier_df_empty,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="national_model",
          is_national=True,
          expected_call="check_national_cost_per_media_unit",
      ),
      dict(
          testcase_name="geo_model",
          is_national=False,
          expected_call="check_geo_cost_per_media_unit",
      ),
  )
  def test_check_cost_per_media_unit_calls_correct_level(
      self, is_national, expected_call
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    engine = eda_engine.EDAEngine(meridian)

    mock_outcome = _create_eda_outcome(
        eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT,
        eda_outcome.EDASeverity.INFO,
        eda_outcome.FindingCause.NONE,
    )
    mock_check = self.enter_context(
        mock.patch.object(
            engine, expected_call, autospec=True, return_value=mock_outcome
        )
    )
    result = engine.check_cost_per_media_unit()
    mock_check.assert_called_once()
    self.assertEqual(result, mock_outcome)

  def _test_cost_per_media_unit_artifact_values(
      self,
      is_national,
      shape,
      level,
      spend_ds_prop,
      media_unit_ds_prop,
      check_method_name,
  ):
    meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=False
    )
    meridian.is_national = is_national
    engine = eda_engine.EDAEngine(meridian)

    spend_arr = np.array(
        [0.0, 10.0, 20.0, 30.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
    )
    media_unit_arr = np.array(
        [5.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    expected_cpi = np.array(
        [0.0, np.nan, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
    )

    spend_data = spend_arr.reshape(shape)
    media_unit_data = media_unit_arr.reshape(shape)
    spend_ds = _create_dataset_with_var_dim(
        spend_data, var_name="media"
    ).rename({"media_dim": constants.MEDIA_CHANNEL, "media": constants.SPEND})
    media_unit_ds = _create_dataset_with_var_dim(
        media_unit_data, var_name="media"
    ).rename(
        {"media_dim": constants.MEDIA_CHANNEL, "media": constants.MEDIA_UNITS}
    )

    self._mock_eda_engine_property(spend_ds_prop, spend_ds)
    self._mock_eda_engine_property(media_unit_ds_prop, media_unit_ds)

    check_method = getattr(engine, check_method_name)
    outcome = check_method()

    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertIsInstance(artifact, eda_outcome.CostPerMediaUnitArtifact)
    with self.subTest("level"):
      self.assertEqual(artifact.level, level)

    with self.subTest("cost_per_media_unit_da"):
      # Check cost_per_media_unit_da
      stacked_spend_da_structure = eda_engine.stack_variables(
          spend_ds, constants.CHANNEL
      )
      expected_cpi_da = xr.DataArray(
          expected_cpi.reshape(stacked_spend_da_structure.shape),
          coords=stacked_spend_da_structure.coords,
          dims=stacked_spend_da_structure.dims,
          name=eda_constants.COST_PER_MEDIA_UNIT,
      )
      xr.testing.assert_allclose(
          artifact.cost_per_media_unit_da, expected_cpi_da
      )

    with self.subTest("cost_media_unit_inconsistency_df"):
      # Check cost_media_unit_inconsistency_df
      inconsistency_df = artifact.cost_media_unit_inconsistency_df
      self.assertEqual(inconsistency_df.shape[0], 2)
      self.assertIn(
          pd.Timestamp("2023-01-01"),
          inconsistency_df.index.get_level_values(constants.TIME),
      )
      self.assertIn(
          pd.Timestamp("2023-01-08"),
          inconsistency_df.index.get_level_values(constants.TIME),
      )

    with self.subTest("outlier_df"):
      # Check outlier_df
      outlier_df = artifact.outlier_df
      self.assertEqual(outlier_df.shape[0], 1)
      self.assertEqual(
          outlier_df.index.get_level_values(constants.TIME)[0],
          pd.Timestamp("2023-03-05"),
      )
      self.assertAlmostEqual(outlier_df.iloc[0]["outliers"], 100.0)

  def test_check_geo_cost_per_media_unit_artifact_values(self):
    self._test_cost_per_media_unit_artifact_values(
        is_national=False,
        shape=(1, 10, 1),
        level=eda_outcome.AnalysisLevel.GEO,
        spend_ds_prop="all_spend_ds",
        media_unit_ds_prop="paid_raw_media_units_ds",
        check_method_name="check_geo_cost_per_media_unit",
    )

  def test_check_national_cost_per_media_unit_artifact_values(self):
    self._test_cost_per_media_unit_artifact_values(
        is_national=True,
        shape=(10, 1),
        level=eda_outcome.AnalysisLevel.NATIONAL,
        spend_ds_prop="national_all_spend_ds",
        media_unit_ds_prop="national_paid_raw_media_units_ds",
        check_method_name="check_national_cost_per_media_unit",
    )

  def test_run_all_critical_checks_all_pass(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    mock_results = {
        "check_overall_kpi_invariability": _create_eda_outcome(
            eda_outcome.EDACheckType.KPI_INVARIABILITY,
            eda_outcome.EDASeverity.INFO,
            eda_outcome.FindingCause.NONE,
        ),
        "check_vif": _create_eda_outcome(
            eda_outcome.EDACheckType.MULTICOLLINEARITY,
            eda_outcome.EDASeverity.INFO,
            eda_outcome.FindingCause.NONE,
        ),
        "check_pairwise_corr": _create_eda_outcome(
            eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
            eda_outcome.EDASeverity.INFO,
            eda_outcome.FindingCause.NONE,
        ),
    }
    self._mock_critical_checks(mock_results)

    outcomes = engine.run_all_critical_checks()

    self.assertIsInstance(outcomes, eda_outcome.CriticalCheckEDAOutcomes)

    with self.subTest("kpi_invariability"):
      self.assertLen(outcomes.kpi_invariability.findings, 1)
      (finding,) = outcomes.kpi_invariability.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.INFO,
      )

    with self.subTest("multicollinearity"):
      self.assertLen(outcomes.multicollinearity.findings, 1)
      (finding,) = outcomes.multicollinearity.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.INFO,
      )

    with self.subTest("pairwise_correlation"):
      self.assertLen(outcomes.pairwise_correlation.findings, 1)
      (finding,) = outcomes.pairwise_correlation.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.INFO,
      )

  def test_run_all_critical_checks_with_non_info_findings(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    mock_results = {
        "check_overall_kpi_invariability": _create_eda_outcome(
            eda_outcome.EDACheckType.KPI_INVARIABILITY,
            eda_outcome.EDASeverity.ERROR,
            eda_outcome.FindingCause.VARIABILITY,
        ),
        "check_vif": _create_eda_outcome(
            eda_outcome.EDACheckType.MULTICOLLINEARITY,
            eda_outcome.EDASeverity.ATTENTION,
            eda_outcome.FindingCause.MULTICOLLINEARITY,
        ),
        "check_pairwise_corr": _create_eda_outcome(
            eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
            eda_outcome.EDASeverity.INFO,
            eda_outcome.FindingCause.NONE,
        ),
    }
    self._mock_critical_checks(mock_results)

    outcomes = engine.run_all_critical_checks()

    self.assertIsInstance(outcomes, eda_outcome.CriticalCheckEDAOutcomes)

    with self.subTest("kpi_invariability"):
      self.assertLen(outcomes.kpi_invariability.findings, 1)
      (finding,) = outcomes.kpi_invariability.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.ERROR,
      )
      self.assertEqual(
          finding.finding_cause,
          eda_outcome.FindingCause.VARIABILITY,
      )

    with self.subTest("multicollinearity"):
      self.assertLen(outcomes.multicollinearity.findings, 1)
      (finding,) = outcomes.multicollinearity.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.ATTENTION,
      )
      self.assertEqual(
          finding.finding_cause,
          eda_outcome.FindingCause.MULTICOLLINEARITY,
      )

    with self.subTest("pairwise_correlation"):
      self.assertLen(outcomes.pairwise_correlation.findings, 1)
      (finding,) = outcomes.pairwise_correlation.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.INFO,
      )
      self.assertEqual(
          finding.finding_cause,
          eda_outcome.FindingCause.NONE,
      )

  def test_run_all_critical_checks_with_exception(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)

    mock_results = {
        "check_overall_kpi_invariability": _create_eda_outcome(
            eda_outcome.EDACheckType.KPI_INVARIABILITY,
            eda_outcome.EDASeverity.INFO,
            eda_outcome.FindingCause.NONE,
        ),
        "check_vif": ValueError("Test Error"),
        "check_pairwise_corr": TypeError("Another Error"),
    }
    self._mock_critical_checks(mock_results)

    outcomes = engine.run_all_critical_checks()

    self.assertIsInstance(outcomes, eda_outcome.CriticalCheckEDAOutcomes)

    with self.subTest("kpi_invariability"):
      self.assertEqual(
          outcomes.kpi_invariability.check_type,
          eda_outcome.EDACheckType.KPI_INVARIABILITY,
      )
      self.assertLen(outcomes.kpi_invariability.findings, 1)
      (finding,) = outcomes.kpi_invariability.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.INFO,
      )
      self.assertEqual(finding.finding_cause, eda_outcome.FindingCause.NONE)
      self.assertIsNone(finding.associated_artifact)

    with self.subTest("multicollinearity"):
      self.assertEqual(
          outcomes.multicollinearity.check_type,
          eda_outcome.EDACheckType.MULTICOLLINEARITY,
      )
      self.assertLen(outcomes.multicollinearity.findings, 1)
      (finding,) = outcomes.multicollinearity.findings
      self.assertEqual(
          finding.finding_cause,
          eda_outcome.FindingCause.RUNTIME_ERROR,
      )
      self.assertIsNone(finding.associated_artifact)
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.ERROR,
      )
      self.assertIn(
          "An error occurred during running check_vif: ValueError('Test"
          " Error')",
          finding.explanation,
      )

    with self.subTest("pairwise_correlation"):
      self.assertEqual(
          outcomes.pairwise_correlation.check_type,
          eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
      )
      self.assertLen(outcomes.pairwise_correlation.findings, 1)
      (finding,) = outcomes.pairwise_correlation.findings
      self.assertEqual(
          finding.severity,
          eda_outcome.EDASeverity.ERROR,
      )
      self.assertEqual(
          finding.finding_cause, eda_outcome.FindingCause.RUNTIME_ERROR
      )
      self.assertIsNone(finding.associated_artifact)
      self.assertIn(
          "An error occurred during running check_pairwise_corr:"
          " TypeError('Another Error')",
          finding.explanation,
      )

  def test_stack_variables(self):
    media_data = np.array(
        [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
        dtype="float32",
    )
    rf_data = np.array(
        [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]], dtype="float32"
    )
    media_ds = _create_dataset_with_var_dim(
        media_data, var_name="national_media_spend"
    )
    rf_ds = _create_dataset_with_var_dim(rf_data, var_name="national_rf_spend")
    ds = xr.merge([media_ds, rf_ds])
    xr.testing.assert_equal(
        eda_engine.stack_variables(ds),
        xr.DataArray(
            data=np.concatenate([media_data, rf_data], axis=1),
            dims=[constants.TIME, eda_constants.VARIABLE],
            coords={
                constants.TIME: pd.to_datetime(
                    ["2023-01-01", "2023-01-08", "2023-01-15"]
                ),
                eda_constants.VARIABLE: [
                    "national_media_spend_1",
                    "national_media_spend_2",
                    "national_media_spend_3",
                    "national_rf_spend_1",
                    "national_rf_spend_2",
                ],
            },
            name=None,
        ),
    )

  def test_check_variable_geo_time_collinearity_raises_for_national_model(self):
    meridian = model.Meridian(self.national_input_data_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    with self.assertRaisesRegex(
        ValueError,
        "check_variable_geo_time_collinearity is not supported for national"
        " models.",
    ):
      engine.check_variable_geo_time_collinearity()

  def test_check_variable_geo_time_collinearity_geo_model_output_correct(self):
    meridian = model.Meridian(self.input_data_with_media_and_rf)
    engine = eda_engine.EDAEngine(meridian)
    outcome = engine.check_variable_geo_time_collinearity()

    with self.subTest("check_type"):
      self.assertEqual(
          outcome.check_type,
          eda_outcome.EDACheckType.VARIABLE_GEO_TIME_COLLINEARITY,
      )
    with self.subTest("findings"):
      self.assertLen(outcome.findings, 2)
      self.assertEqual(
          [finding.severity for finding in outcome.findings],
          [eda_outcome.EDASeverity.INFO] * 2,
      )
      self.assertIn(
          "reducing `knots` argument in `ModelSpec`.",
          outcome.findings[0].explanation,
      )
      self.assertIn(
          "regresses each variable against geo as a categorical variable.",
          outcome.findings[1].explanation,
      )
    with self.subTest("analysis_artifacts"):
      self.assertLen(outcome.analysis_artifacts, 1)
      (artifact,) = outcome.analysis_artifacts
      self.assertIsInstance(
          artifact, eda_outcome.VariableGeoTimeCollinearityArtifact
      )
      self.assertEqual(artifact.level, eda_outcome.AnalysisLevel.OVERALL)
      self.assertIn(eda_constants.RSQUARED_GEO, artifact.rsquared_ds.data_vars)
      self.assertIn(eda_constants.RSQUARED_TIME, artifact.rsquared_ds.data_vars)

  def test_check_variable_geo_time_collinearity_geo_model_r2_values_correct(
      self,
  ):
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    n_geos = 3
    n_times = 5
    geo_dependent_data = np.tile(np.arange(n_geos), (n_times, 1)).T.reshape(
        n_geos, n_times, 1
    )
    time_dependent_data = np.tile(np.arange(n_times), (n_geos, 1)).reshape(
        n_geos, n_times, 1
    )

    mock_data = np.concatenate(
        [geo_dependent_data, time_dependent_data], axis=-1
    )

    mock_da = _create_data_array_with_var_dim(
        mock_data,
        name=constants.TREATMENT_CONTROL_SCALED,
        var_name="var",
    )
    mock_da = mock_da.rename({"var_dim": eda_constants.VARIABLE})
    mock_da = mock_da.assign_coords(
        {
            eda_constants.VARIABLE: [
                "var_geo_dependent",
                "var_time_dependent",
            ]
        }
    )

    self._mock_eda_engine_property(
        "_stacked_treatment_control_scaled_da", mock_da
    )

    outcome = engine.check_variable_geo_time_collinearity()

    self.assertLen(outcome.analysis_artifacts, 1)
    (artifact,) = outcome.analysis_artifacts
    self.assertIsInstance(
        artifact, eda_outcome.VariableGeoTimeCollinearityArtifact
    )
    rsquared_ds = artifact.rsquared_ds

    with self.subTest("geo_dependent_variable"):
      # Check var_geo_dependent: r2_geo should be ~1, r2_time should be low
      self.assertAlmostEqual(
          rsquared_ds[eda_constants.RSQUARED_GEO]
          .sel({eda_constants.VARIABLE: "var_geo_dependent"})
          .item(),
          1.0,
      )
      self.assertLessEqual(
          rsquared_ds[eda_constants.RSQUARED_TIME]
          .sel({eda_constants.VARIABLE: "var_geo_dependent"})
          .item(),
          0.0,
      )

    with self.subTest("time_dependent_variable"):
      # Check var_time_dependent: r2_geo should be low, r2_time should be ~1
      self.assertLessEqual(
          rsquared_ds[eda_constants.RSQUARED_GEO]
          .sel({eda_constants.VARIABLE: "var_time_dependent"})
          .item(),
          0.0,
      )
      self.assertAlmostEqual(
          rsquared_ds[eda_constants.RSQUARED_TIME]
          .sel({eda_constants.VARIABLE: "var_time_dependent"})
          .item(),
          1.0,
      )

  @parameterized.named_parameters(
      (
          "kpi_invariability",
          "check_overall_kpi_invariability",
          lambda: {
              "kpi_scaled_da": _create_data_array_with_var_dim(
                  np.ones((1, 10)),
                  name=constants.KPI_SCALED,
              )
          },
          eda_outcome.FindingCause.VARIABILITY,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "kpi_invariability_info",
          "check_overall_kpi_invariability",
          lambda: {
              "kpi_scaled_da": _create_data_array_with_var_dim(
                  np.arange(10).reshape(1, 10),
                  name=constants.KPI_SCALED,
              )
          },
          eda_outcome.FindingCause.NONE,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "cost_per_media_unit_outlier",
          "check_cost_per_media_unit",
          lambda: {
              "all_spend_ds": (
                  _create_dataset_with_var_dim(
                      np.array([1.0] * 9 + [1000.0]).reshape(1, 10, 1),
                      var_name="media",
                  ).rename({
                      "media_dim": constants.MEDIA_CHANNEL,
                      "media": constants.MEDIA_SPEND,
                  })
              ),
              "paid_raw_media_units_ds": (
                  _create_dataset_with_var_dim(
                      np.ones((1, 10, 1)), var_name="media"
                  ).rename({"media_dim": constants.MEDIA_CHANNEL})
              ),
          },
          eda_outcome.FindingCause.VARIABILITY,
          1,
          lambda outcome: outcome.get_geo_artifacts(),
      ),
      (
          "cost_per_media_unit_inconsistent_data",
          "check_cost_per_media_unit",
          lambda: {
              "all_spend_ds": (
                  _create_dataset_with_var_dim(
                      np.array([100.0]).reshape(1, 1, 1),
                      var_name="media",
                  ).rename({
                      "media_dim": constants.MEDIA_CHANNEL,
                      "media": constants.MEDIA_SPEND,
                  })
              ),
              "paid_raw_media_units_ds": (
                  _create_dataset_with_var_dim(
                      np.array([0.0]).reshape(1, 1, 1), var_name="media"
                  ).rename({"media_dim": constants.MEDIA_CHANNEL})
              ),
          },
          eda_outcome.FindingCause.INCONSISTENT_DATA,
          1,
          lambda outcome: outcome.get_geo_artifacts(),
      ),
      (
          "cost_per_media_unit_info",
          "check_cost_per_media_unit",
          lambda: {
              "all_spend_ds": (
                  _create_dataset_with_var_dim(
                      np.full((1, 10, 1), 10.0),
                      var_name="media",
                  ).rename({
                      "media_dim": constants.MEDIA_CHANNEL,
                      "media": constants.MEDIA_SPEND,
                  })
              ),
              "paid_raw_media_units_ds": (
                  _create_dataset_with_var_dim(
                      np.full((1, 10, 1), 10.0), var_name="media"
                  ).rename({"media_dim": constants.MEDIA_CHANNEL})
              ),
          },
          eda_outcome.FindingCause.NONE,
          1,
          lambda outcome: outcome.get_geo_artifacts(),
      ),
      (
          "vif_multicollinearity",
          "check_vif",
          lambda: {
              "_stacked_treatment_control_scaled_da": _get_overall_high_vif_da()
          },
          eda_outcome.FindingCause.MULTICOLLINEARITY,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "vif_info",
          "check_vif",
          lambda: {
              "_stacked_treatment_control_scaled_da": _get_low_vif_da(),
          },
          eda_outcome.FindingCause.NONE,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "pairwise_corr_multicollinearity",
          "check_pairwise_corr",
          lambda: {
              "_stacked_treatment_control_scaled_da": (
                  _create_data_array_with_var_dim(
                      np.repeat(
                          np.linspace(0, 1, 100).reshape(1, 100, 1), 2, axis=-1
                      ),
                      name=constants.TREATMENT_CONTROL_SCALED,
                      var_name=eda_constants.VARIABLE,
                  ).rename({"var_dim": eda_constants.VARIABLE})
              ),
          },
          eda_outcome.FindingCause.MULTICOLLINEARITY,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "pairwise_corr_info",
          "check_pairwise_corr",
          lambda: {
              "_stacked_treatment_control_scaled_da": (
                  _create_data_array_with_var_dim(
                      np.array([
                          [[1, 10], [2, 2], [3, 13]],
                          [[4, 4], [5, 15], [6, 6]],
                      ]),
                      name=constants.TREATMENT_CONTROL_SCALED,
                      var_name=eda_constants.VARIABLE,
                  ).rename({"var_dim": eda_constants.VARIABLE})
              ),
          },
          eda_outcome.FindingCause.NONE,
          1,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
      (
          "std_invariability",
          "check_std",
          lambda: {
              "kpi_scaled_da": _create_data_array_with_var_dim(
                  np.ones((1, 7)),
                  name=constants.KPI_SCALED,
              ),
              "_stacked_treatment_control_scaled_da": (
                  _create_data_array_with_var_dim(
                      np.arange(7).reshape(1, 7, 1),
                      name=constants.TREATMENT_CONTROL_SCALED,
                      var_name=eda_constants.VARIABLE,
                  )
              ),
              "all_reach_scaled_da": None,
              "all_freq_da": None,
          },
          eda_outcome.FindingCause.VARIABILITY,
          1,
          lambda outcome: outcome.get_geo_artifacts(),
      ),
      (
          "std_info",
          "check_std",
          lambda: {
              "kpi_scaled_da": _create_data_array_with_var_dim(
                  np.arange(7).reshape(1, 7),
                  name=constants.KPI_SCALED,
              ),
              "_stacked_treatment_control_scaled_da": (
                  _create_data_array_with_var_dim(
                      np.arange(7).reshape(1, 7, 1),
                      name=constants.TREATMENT_CONTROL_SCALED,
                      var_name=eda_constants.VARIABLE,
                  )
              ),
              "all_reach_scaled_da": None,
              "all_freq_da": None,
          },
          eda_outcome.FindingCause.NONE,
          1,
          lambda outcome: outcome.get_geo_artifacts(),
      ),
      (
          "variable_geo_time_collinearity_info",
          "check_variable_geo_time_collinearity",
          lambda: {},
          eda_outcome.FindingCause.NONE,
          2,
          lambda outcome: outcome.get_overall_artifacts(),
      ),
  )
  def test_finding_mapping(
      self,
      method_name,
      mock_data_factory,
      expected_type,
      expected_findings_count,
      artifact_accessor,
  ):
    meridian = model.Meridian(self.input_data_with_media_only)
    engine = eda_engine.EDAEngine(meridian)

    for attr, val in mock_data_factory().items():
      self._mock_eda_engine_property(attr, val)
    outcome = getattr(engine, method_name)()
    artifacts = artifact_accessor(outcome)
    target_findings = [
        f for f in outcome.findings if f.finding_cause == expected_type
    ]
    self.assertLen(
        target_findings,
        expected_findings_count,
        f"Expected {expected_findings_count} finding(s) of type"
        f" {expected_type}, but got {len(target_findings)}.",
    )

    if expected_type == eda_outcome.FindingCause.NONE:
      for finding in target_findings:
        self.assertIsNone(finding.associated_artifact)
    else:
      for finding in target_findings:
        self.assertIn(finding.associated_artifact, artifacts)


if __name__ == "__main__":
  absltest.main()
