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

"""Unit tests for model.py."""

import collections
from collections.abc import Collection, Sequence
import os
from unittest import mock
import warnings
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import test_utils
from meridian.model import adstock_hill
from meridian.model import knots as knots_module
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


def _convert_with_swap(array: xr.DataArray, n_burnin: int) -> tf.Tensor:
  """Converts a DataArray to a tf.Tensor with the correct MCMC format.

  This function converts a DataArray to tf.Tensor, swaps first two dimensions
  and adds the burnin part. This is needed to properly mock the
  _xla_windowed_adaptive_nuts() function output in the sample_posterior
  tests.

  Args:
    array: The array to be converted.
    n_burnin: The number of extra draws to be padded with as the 'burnin' part.

  Returns:
    A tensor in the same format as returned by the _xla_windowed_adaptive_nuts()
    function.
  """
  tensor = tf.convert_to_tensor(array)
  perm = [1, 0] + [i for i in range(2, len(tensor.shape))]
  transposed_tensor = tf.transpose(tensor, perm=perm)

  # Add the "burnin" part to the mocked output of _xla_windowed_adaptive_nuts
  # to make sure sample_posterior returns the correct "keep" part.
  if array.dtype == bool:
    pad_value = False
  else:
    pad_value = 0.0 if array.dtype.kind == "f" else 0

  burnin = tf.fill([n_burnin] + transposed_tensor.shape[1:], pad_value)
  return tf.concat(
      [burnin, transposed_tensor],
      axis=0,
  )


class ModelTest(tf.test.TestCase, parameterized.TestCase):
  # TODO(b/302713435): Update the sample data to span over 1 or 2 year(s).
  _TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
  _TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH = os.path.join(
      _TEST_DIR,
      "sample_prior_media_and_rf.nc",
  )
  _TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH = os.path.join(
      _TEST_DIR,
      "sample_prior_media_only.nc",
  )
  _TEST_SAMPLE_PRIOR_RF_ONLY_PATH = os.path.join(
      _TEST_DIR,
      "sample_prior_rf_only.nc",
  )
  _TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH = os.path.join(
      _TEST_DIR,
      "sample_posterior_media_and_rf.nc",
  )
  _TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH = os.path.join(
      _TEST_DIR,
      "sample_posterior_media_only.nc",
  )
  _TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH = os.path.join(
      _TEST_DIR,
      "sample_posterior_rf_only.nc",
  )
  _TEST_SAMPLE_TRACE_PATH = os.path.join(
      _TEST_DIR,
      "sample_trace.nc",
  )

  # Data dimensions for sample input.
  _N_CHAINS = 2
  _N_ADAPT = 2
  _N_BURNIN = 5
  _N_KEEP = 10
  _N_DRAWS = 10
  _N_GEOS = 5
  _N_GEOS_NATIONAL = 1
  _N_TIMES = 200
  _N_TIMES_SHORT = 49
  _N_MEDIA_TIMES = 203
  _N_MEDIA_TIMES_SHORT = 52
  _N_MEDIA_CHANNELS = 3
  _N_RF_CHANNELS = 2
  _N_CONTROLS = 2
  _ROI_CALIBRATION_PERIOD = tf.cast(
      tf.ones((_N_MEDIA_TIMES_SHORT, _N_MEDIA_CHANNELS)),
      dtype=tf.bool,
  )
  _RF_ROI_CALIBRATION_PERIOD = tf.cast(
      tf.ones((_N_MEDIA_TIMES_SHORT, _N_RF_CHANNELS)),
      dtype=tf.bool,
  )

  def setUp(self):
    super().setUp()
    self.input_data_non_revenue_no_revenue_per_kpi = (
        test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.national_input_data_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_NATIONAL,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.national_input_data_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_NATIONAL,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )

    test_prior_media_and_rf = xr.open_dataset(
        self._TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH
    )
    test_prior_media_only = xr.open_dataset(
        self._TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH
    )
    test_prior_rf_only = xr.open_dataset(self._TEST_SAMPLE_PRIOR_RF_ONLY_PATH)
    self.test_dist_media_and_rf = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_media_and_rf[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })
    self.test_dist_media_only = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_media_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    })
    self.test_dist_rf_only = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_rf_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })

    test_posterior_media_and_rf = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH
    )
    test_posterior_media_only = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH
    )
    test_posterior_rf_only = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH
    )
    posterior_params_to_tensors_media_and_rf = {
        param: _convert_with_swap(
            test_posterior_media_and_rf[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    posterior_params_to_tensors_media_only = {
        param: _convert_with_swap(
            test_posterior_media_only[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    }
    posterior_params_to_tensors_rf_only = {
        param: _convert_with_swap(
            test_posterior_rf_only[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    self.test_posterior_states_media_and_rf = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_and_rf)
    self.test_posterior_states_media_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_only)
    self.test_posterior_states_rf_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_rf_only)

    test_trace = xr.open_dataset(self._TEST_SAMPLE_TRACE_PATH)
    self.test_trace = {
        param: _convert_with_swap(test_trace[param], n_burnin=self._N_BURNIN)
        for param in test_trace.data_vars
    }

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
      ),
      dict(
          testcase_name="geo",
          input_data_type="geo",
      ),
  )
  def test_init_with_wrong_roi_calibration_period_shape_fails(
      self,
      input_data_type: str,
  ):
    error_msg = (
        "The shape of `roi_calibration_period` (2, 3) is different"
        " from `(n_media_times, n_media_channels) = (203, 3)`."
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=np.ones((2, 3), dtype=bool)
    )
    input_data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      model.Meridian(input_data=input_data, model_spec=model_spec)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
      ),
      dict(
          testcase_name="geo",
          input_data_type="geo",
      ),
  )
  def test_init_with_wrong_rf_roi_calibration_period_shape_fails(
      self,
      input_data_type: str,
  ):
    error_msg = (
        "The shape of `rf_roi_calibration_period` (4, 5) is different"
        " from `(n_media_times, n_rf_channels) = (203, 2)`."
    )
    model_spec = spec.ModelSpec(
        rf_roi_calibration_period=np.ones((4, 5), dtype=bool)
    )
    input_data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
      model.Meridian(input_data=input_data, model_spec=model_spec)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
          error_msg=(
              "The shape of `holdout_id` (2, 8) is different"
              " from `(n_times,) = (200,)`."
          ),
      ),
      dict(
          testcase_name="geo",
          input_data_type="geo",
          error_msg=(
              "The shape of `holdout_id` (2, 8) is different"
              " from `(n_geos, n_times) = (5, 200)`."
          ),
      ),
  )
  def test_init_with_wrong_holdout_id_shape_fails(
      self, input_data_type: str, error_msg: str
  ):
    model_spec = spec.ModelSpec(holdout_id=np.ones((2, 8), dtype=bool))
    input_data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      model.Meridian(input_data=input_data, model_spec=model_spec)

  def test_init_with_wrong_control_population_scaling_id_shape_fails(self):
    model_spec = spec.ModelSpec(
        control_population_scaling_id=np.ones((7), dtype=bool)
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of `control_population_scaling_id` (7,) is different from"
        " `(n_controls,) = (2,)`.",
    ):
      model.Meridian(
          input_data=self.input_data_with_media_and_rf, model_spec=model_spec
      )

  @parameterized.named_parameters(
      ("none", None, 200), ("int", 3, 3), ("list", [0, 50, 100, 150], 4)
  )
  def test_n_knots(self, knots, expected_n_knots):
    # Create sample model spec with given knots
    model_spec = spec.ModelSpec(knots=knots)

    meridian = model.Meridian(
        input_data=self.input_data_with_media_only, model_spec=model_spec
    )

    self.assertEqual(meridian._n_knots, expected_n_knots)

  @parameterized.named_parameters(
      dict(
          testcase_name="too_many",
          knots=201,
          msg=(
              "The number of knots (201) cannot be greater than the number of"
              " time periods in the kpi (200)."
          ),
      ),
      dict(
          testcase_name="less_than_one",
          knots=-1,
          msg="If knots is an integer, it must be at least 1.",
      ),
      dict(
          testcase_name="negative",
          knots=[-2, 17],
          msg="Knots must be all non-negative.",
      ),
      dict(
          testcase_name="too_large",
          knots=[3, 202],
          msg="Knots must all be less than the number of time periods.",
      ),
  )
  def test_init_with_wrong_knots_fails(
      self, knots: int | Collection[int] | None, msg: str
  ):
    # Create sample model spec with given knots
    model_spec = spec.ModelSpec(knots=knots)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        msg,
    ):
      model.Meridian(
          input_data=self.input_data_with_media_only, model_spec=model_spec
      )

  @parameterized.named_parameters(
      dict(testcase_name="none", knots=None, is_national=False),
      dict(testcase_name="none_and_national", knots=None, is_national=True),
      dict(testcase_name="int", knots=3, is_national=False),
      dict(testcase_name="list", knots=[0, 50, 100, 150], is_national=False),
  )
  def test_get_knot_info_is_called(
      self, knots: int | Collection[int] | None, is_national: bool
  ):
    with mock.patch.object(
        knots_module,
        "get_knot_info",
        autospec=True,
        return_value=(3, [2, 5, 8], np.eye(3)),
    ) as mock_get_knot_info:
      input_data = (
          self.national_input_data_media_only
          if is_national
          else self.input_data_with_media_only
      )
      model.Meridian(
          input_data=input_data,
          model_spec=spec.ModelSpec(knots=knots),
      )
      mock_get_knot_info.assert_called_once_with(
          self._N_TIMES, knots, is_national
      )

  def test_custom_priors_not_passed_in(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Custom priors should be set during model creation since"
        " `kpi_type` = `non_revenue` and `revenue_per_kpi` was not passed in."
        " Further documentation is available at"
        " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi",
    ):
      model.Meridian(
          input_data=self.input_data_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(),
      )

  def test_get_knot_info_fails(self):
    error_msg = "Knots must be all non-negative."
    with mock.patch.object(
        knots_module,
        "get_knot_info",
        autospec=True,
        side_effect=ValueError(error_msg),
    ):
      with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
        model.Meridian(
            input_data=self.input_data_with_media_only,
            model_spec=spec.ModelSpec(knots=4),
        )

  def test_init_with_default_parameters_works(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_only)

    # Compare input data.
    self.assertEqual(meridian.input_data, self.input_data_with_media_only)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_init_with_default_national_parameters_works(self):
    meridian = model.Meridian(input_data=self.national_input_data_media_only)

    # Compare input data.
    self.assertEqual(meridian.input_data, self.national_input_data_media_only)

    # Create sample model spec for comparison
    expected_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(expected_spec))

  def test_init_geo_args_no_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      model.Meridian(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist="normal", unique_sigma_for_each_geo=True
          ),
      )
      self.assertEmpty(w)

  def test_init_national_args_with_broadcast_warnings(self):
    with warnings.catch_warnings(record=True) as warns:
      warnings.simplefilter("module")
      model.Meridian(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist=constants.MEDIA_EFFECTS_NORMAL
          ),
      )
      self.assertLen(warns, 4)
      for w in warns:
        self.assertTrue(issubclass(w.category, UserWarning))
        self.assertIn(
            "Hierarchical distribution parameters must be deterministically"
            " zero for national models.",
            str(w.message),
        )

  def test_init_national_args_with_model_spec_warnings(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      model.Meridian(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(unique_sigma_for_each_geo=True),
      )
      self.assertLen(w, 6)
      # 4 warnings from the broadcasting + 2 from model spec.
      self.assertTrue(
          any(
              "In a nationally aggregated model, the `media_effects_dist` will"
              " be reset to `normal`."
              in str(warning.message)
              for warning in w
          )
      )
      self.assertTrue(
          any(
              "In a nationally aggregated model, the"
              " `unique_sigma_for_each_geo` will be reset to `False`."
              in str(warning.message)
              for warning in w
          )
      )

  def test_base_geo_properties(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
    self.assertEqual(meridian.n_geos, self._N_GEOS)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertFalse(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  def test_base_national_properties(self):
    meridian = model.Meridian(input_data=self.national_input_data_media_only)
    self.assertEqual(meridian.n_geos, self._N_GEOS_NATIONAL)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertTrue(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=_N_MEDIA_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_only",
          input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=_N_RF_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_and_media",
          input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=_N_MEDIA_CHANNELS, n_rf_channels=_N_RF_CHANNELS
          ),
      ),
  )
  def test_input_data_tensor_properties(self, input_data):
    meridian = model.Meridian(input_data=input_data)
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.kpi, dtype=tf.float32),
        meridian.kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.revenue_per_kpi, dtype=tf.float32),
        meridian.revenue_per_kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.controls, dtype=tf.float32),
        meridian.controls,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.population, dtype=tf.float32),
        meridian.population,
    )
    if input_data.media is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.media, dtype=tf.float32),
          meridian.media,
      )
    if input_data.media_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32),
          meridian.media_spend,
      )
    if input_data.reach is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.reach, dtype=tf.float32),
          meridian.reach,
      )
    if input_data.frequency is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.frequency, dtype=tf.float32),
          meridian.frequency,
      )
    if input_data.rf_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32),
          meridian.rf_spend,
      )
    if input_data.media_spend is not None and input_data.rf_spend is not None:
      self.assertAllClose(
          tf.concat(
              [
                  tf.convert_to_tensor(
                      input_data.media_spend, dtype=tf.float32
                  ),
                  tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32),
              ],
              axis=-1,
          ),
          meridian.total_spend,
      )
    elif input_data.media_spend is not None:
      self.assertAllClose(
          tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32),
          meridian.total_spend,
      )
    else:
      self.assertAllClose(
          tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32),
          meridian.total_spend,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo_normal",
          n_geos=_N_GEOS,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
      dict(
          testcase_name="geo_log_normal",
          n_geos=_N_GEOS,
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
      ),
      dict(
          testcase_name="national_normal",
          n_geos=_N_GEOS_NATIONAL,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
      dict(
          testcase_name="national_log_normal",
          n_geos=_N_GEOS_NATIONAL,
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
  )
  def test_media_effects_dist_property(
      self, n_geos, media_effects_dist, expected_media_effects_dist
  ):
    meridian = model.Meridian(
        input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(media_effects_dist=media_effects_dist),
    )
    self.assertEqual(meridian.media_effects_dist, expected_media_effects_dist)

  @parameterized.named_parameters(
      dict(
          testcase_name="geo_unique_sigma_for_each_geo_true",
          n_geos=_N_GEOS,
          unique_sigma_for_each_geo=True,
          expected_unique_sigma_for_each_geo=True,
      ),
      dict(
          testcase_name="geo_unique_sigma_for_each_geo_false",
          n_geos=_N_GEOS,
          unique_sigma_for_each_geo=False,
          expected_unique_sigma_for_each_geo=False,
      ),
      dict(
          testcase_name="national_unique_sigma_for_each_geo_true",
          n_geos=_N_GEOS_NATIONAL,
          unique_sigma_for_each_geo=True,
          expected_unique_sigma_for_each_geo=False,
      ),
      dict(
          testcase_name="national_unique_sigma_for_each_geo_false",
          n_geos=_N_GEOS_NATIONAL,
          unique_sigma_for_each_geo=False,
          expected_unique_sigma_for_each_geo=False,
      ),
  )
  def test_unique_sigma_for_each_geo_property(
      self,
      n_geos,
      unique_sigma_for_each_geo,
      expected_unique_sigma_for_each_geo,
  ):
    meridian = model.Meridian(
        input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(
            unique_sigma_for_each_geo=unique_sigma_for_each_geo
        ),
    )
    self.assertEqual(
        meridian.unique_sigma_for_each_geo, expected_unique_sigma_for_each_geo
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_controls",
          dataset=test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_CONTROLS,
          data_name="controls",
          dims_bad=["control_0", "control_1"],
      ),
      dict(
          testcase_name="wrong_media",
          dataset=test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_MEDIA,
          data_name="media",
          dims_bad=["media_channel_1", "media_channel_2"],
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_REACH,
          data_name="reach",
          dims_bad=["rf_channel_0", "rf_channel_1"],
      ),
  )
  def test_init_without_geo_variation_fails(
      self, dataset: xr.Dataset, data_name: str, dims_bad: Sequence[str]
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"There are {data_name} variables that do not vary across geos, which"
        " makes a model with n_knots=n_time unidentifiable. Such"
        " identifiability issues can lead to poor model convergence. We"
        " recommend decreasing the number of knots or dropping the following"
        f" {data_name} variables, which do not vary across geos for all time"
        f" periods: {dims_bad}",
    ):
      model.Meridian(
          input_data=test_utils.sample_input_data_from_dataset(
              dataset, kpi_type=constants.NON_REVENUE
          )
      )

  def test_broadcast_prior_distribution_is_called_in_meridian_init(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        meridian.prior_broadcast.tau_g_excl_baseline.batch_shape,
        (meridian.n_geos - 1,),
    )

    # Validate `n_knots` shape distributions.
    self.assertEqual(
        meridian.prior_broadcast.knot_values.batch_shape, (meridian._n_knots,)
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        meridian.prior_broadcast.beta_m,
        meridian.prior_broadcast.eta_m,
        meridian.prior_broadcast.alpha_m,
        meridian.prior_broadcast.ec_m,
        meridian.prior_broadcast.slope_m,
        meridian.prior_broadcast.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_media_channels,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        meridian.prior_broadcast.beta_rf,
        meridian.prior_broadcast.eta_rf,
        meridian.prior_broadcast.alpha_rf,
        meridian.prior_broadcast.ec_rf,
        meridian.prior_broadcast.slope_rf,
        meridian.prior_broadcast.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_rf_channels,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        meridian.prior_broadcast.gamma_c,
        meridian.prior_broadcast.xi_c,
    ]
    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_controls,))

    # Validate sigma.
    self.assertEqual(meridian.prior_broadcast.sigma.batch_shape, (1,))

  def test_media_attributes_not_set(self):
    meridian = model.Meridian(input_data=self.input_data_with_rf_only)
    self.assertEqual(meridian.n_media_channels, 0)
    self.assertIsNone(meridian.media_transformer)
    self.assertIsNone(meridian.media_scaled)
    self.assertIsNone(meridian._media_counterfactual)
    self.assertIsNone(meridian._media_counterfactual_scaled)
    self.assertIsNone(meridian._media_spend_counterfactual)

  def test_rf_attributes_not_set(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_only)
    self.assertEqual(meridian.n_rf_channels, 0)
    self.assertIsNone(meridian.reach_transformer)
    self.assertIsNone(meridian.reach_scaled)
    self.assertIsNone(meridian._reach_counterfactual)
    self.assertIsNone(meridian._reach_counterfactual_scaled)
    self.assertIsNone(meridian._rf_spend_counterfactual)

  def test_scaled_data_shape(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
    if (
        self.input_data_with_media_and_rf.media is not None
        and meridian.media_scaled is not None
    ):
      self.assertAllEqual(
          meridian.media_scaled.shape,
          self.input_data_with_media_and_rf.media.shape,
          msg=(
              "Shape of `_media_scaled` does not match the shape of `media`"
              " from the input data."
          ),
      )
    if (
        self.input_data_with_media_and_rf.reach is not None
        and meridian.reach_scaled is not None
    ):
      self.assertAllEqual(
          meridian.reach_scaled.shape,
          self.input_data_with_media_and_rf.reach.shape,
          msg=(
              "Shape of `_reach_scaled` does not match the shape of `reach`"
              " from the input data."
          ),
      )
    self.assertAllEqual(
        meridian.controls_scaled.shape,
        self.input_data_with_media_and_rf.controls.shape,
        msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
    self.assertAllEqual(
        meridian.kpi_scaled.shape,
        self.input_data_with_media_and_rf.kpi.shape,
        msg=(
            "Shape of `_kpi_scaled` does not match the shape of"
            " `kpi` from the input data."
        ),
    )

  def test_population_scaled_conrols_transformer_set(self):
    model_spec = spec.ModelSpec(
        control_population_scaling_id=tf.convert_to_tensor(
            [True for _ in self.input_data_with_media_and_rf.control_variable]
        )
    )
    meridian = model.Meridian(
        input_data=self.input_data_with_media_and_rf, model_spec=model_spec
    )
    self.assertIsNotNone(
        meridian.controls_transformer._population_scaling_factors,
        msg=(
            "`_population_scaling_factors` not set for the controls"
            " transformer."
        ),
    )
    self.assertAllEqual(
        meridian.controls_transformer._population_scaling_factors.shape,
        [
            len(self.input_data_with_media_and_rf.geo),
            len(self.input_data_with_media_and_rf.control_variable),
        ],
        msg=(
            "Shape of `controls_transformer._population_scaling_factors` does"
            " not match (`n_geos`, `n_controls`)."
        ),
    )

  def test_scaled_data_inverse_is_identity(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)

    # With the default tolerance of eps * 10 the test fails due to rounding
    # errors.
    atol = np.finfo(np.float32).eps * 100
    if (
        meridian.media_scaled is not None
        and meridian.media_transformer is not None
        and self.input_data_with_media_and_rf.media is not None
    ):
      self.assertAllClose(
          meridian.media_transformer.inverse(meridian.media_scaled),
          self.input_data_with_media_and_rf.media,
          atol=atol,
      )
    if (
        meridian.reach_scaled is not None
        and meridian.reach_transformer is not None
        and self.input_data_with_media_and_rf.reach is not None
    ):
      self.assertAllClose(
          meridian.reach_transformer.inverse(meridian.reach_scaled),
          self.input_data_with_media_and_rf.reach,
          atol=atol,
      )
    self.assertAllClose(
        meridian.controls_transformer.inverse(meridian.controls_scaled),
        self.input_data_with_media_and_rf.controls,
        atol=atol,
    )
    self.assertAllClose(
        meridian.kpi_transformer.inverse(meridian.kpi_scaled),
        self.input_data_with_media_and_rf.kpi,
        atol=atol,
    )

  def test_counterfactual_data_no_roi_calibration(self):
    meridian = model.Meridian(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(
            roi_calibration_period=None, rf_roi_calibration_period=None
        ),
    )

    self.assertAllEqual(
        meridian._media_counterfactual,
        tf.zeros_like(self.input_data_with_media_and_rf.media),
    )
    self.assertAllEqual(
        meridian._media_counterfactual_scaled,
        tf.zeros_like(meridian.media_scaled),
    )
    self.assertAllEqual(
        meridian._media_spend_counterfactual,
        tf.zeros_like(self.input_data_with_media_and_rf.media_spend),
    )
    if meridian.reach_scaled is not None:
      self.assertAllEqual(
          meridian._reach_counterfactual,
          tf.zeros_like(self.input_data_with_media_and_rf.reach),
      )
      self.assertAllEqual(
          meridian._reach_counterfactual_scaled,
          tf.zeros_like(meridian.reach_scaled),
      )
      self.assertAllEqual(
          meridian._rf_spend_counterfactual,
          tf.zeros_like(self.input_data_with_media_and_rf.rf_spend),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
      ),
      dict(
          testcase_name="geo-level",
          input_data_type="geo-level",
      ),
  )
  def test_counterfactual_data_with_roi_calibration(self, input_data_type: str):
    roi_calibration_shape = (self._N_MEDIA_TIMES, self._N_MEDIA_CHANNELS)
    roi_calibration_period = np.random.choice(
        a=[False, True], size=roi_calibration_shape
    )
    input_data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=spec.ModelSpec(
            roi_calibration_period=roi_calibration_period
        ),
    )

    self.assertAllClose(
        meridian._media_counterfactual,
        tf.where(
            roi_calibration_period,
            0,
            meridian.media,
        ),
    )
    self.assertAllClose(
        meridian._media_counterfactual_scaled,
        tf.where(roi_calibration_period, 0, meridian.media_scaled),
    )

    self.assertAllClose(
        meridian._media_spend_counterfactual,
        tf.where(
            roi_calibration_period[..., -meridian.n_times :, :],
            0,
            meridian.media_spend,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
      ),
      dict(
          testcase_name="geo-level",
          input_data_type="geo-level",
      ),
  )
  def test_counterfactual_data_with_rf_roi_calibration(
      self, input_data_type: str
  ):
    rf_roi_calibration_shape = (self._N_MEDIA_TIMES, self._N_RF_CHANNELS)
    rf_roi_calibration_period = np.random.choice(
        a=[False, True], size=rf_roi_calibration_shape
    )
    input_data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=spec.ModelSpec(
            rf_roi_calibration_period=rf_roi_calibration_period
        ),
    )

    if meridian.input_data.reach is not None:
      self.assertAllClose(
          meridian._reach_counterfactual,
          tf.where(
              rf_roi_calibration_period,
              0,
              meridian.reach,
          ),
      )
      self.assertAllClose(
          meridian._reach_counterfactual_scaled,
          tf.where(rf_roi_calibration_period, 0, meridian.reach_scaled),
      )

      self.assertAllClose(
          meridian._rf_spend_counterfactual,
          tf.where(
              rf_roi_calibration_period[..., -meridian.n_times :, :],
              0,
              meridian.rf_spend,
          ),
      )

  @parameterized.named_parameters(
      dict(testcase_name="int", baseline_geo=4, expected_idx=4),
      dict(testcase_name="str", baseline_geo="geo_1", expected_idx=1),
      dict(testcase_name="none", baseline_geo=None, expected_idx=2),
  )
  def test_baseline_geo_idx(
      self, baseline_geo: int | str | None, expected_idx: int
  ):
    self.input_data_with_media_only.population.data = [
        2.0,
        5.0,
        20.0,
        7.0,
        10.0,
    ]
    meridian = model.Meridian(
        input_data=self.input_data_with_media_only,
        model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
    )
    self.assertEqual(meridian._baseline_geo_idx, expected_idx)

  @parameterized.named_parameters(
      dict(
          testcase_name="int",
          baseline_geo=7,
          msg="Baseline geo index 7 out of range [0, 4].",
      ),
      dict(
          testcase_name="str",
          baseline_geo="incorrect",
          msg="Baseline geo 'incorrect' not found.",
      ),
  )
  def test_wrong_baseline_geo_id_fails(
      self, baseline_geo: int | str | None, msg: str
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      model.Meridian(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
      )

  # TODO(b/295163156) Move this test to a higher-level public API unit test.
  @parameterized.named_parameters(
      dict(
          testcase_name="adstock_first",
          hill_before_adstock=False,
          expected_called_names=["mock_adstock", "mock_hill"],
      ),
      dict(
          testcase_name="hill_first",
          hill_before_adstock=True,
          expected_called_names=["mock_hill", "mock_adstock"],
      ),
  )
  def test_adstock_hill_media(
      self,
      hill_before_adstock,
      expected_called_names,
  ):
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_only.media,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_only.media,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    meridian = model.Meridian(
        input_data=self.input_data_with_media_only,
        model_spec=spec.ModelSpec(
            hill_before_adstock=hill_before_adstock,
        ),
    )
    meridian.adstock_hill_media(
        media=meridian.media,
        alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
    )

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  # TODO(b/295163156) Move this test to a higher-level public API unit test.
  def test_adstock_hill_rf(
      self,
  ):
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_and_rf.frequency,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_and_rf.reach
            * self.input_data_with_media_and_rf.frequency,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    meridian = model.Meridian(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    meridian.adstock_hill_rf(
        reach=meridian.reach,
        frequency=meridian.frequency,
        alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
        ec=np.ones(shape=(self._N_RF_CHANNELS,)),
        slope=np.ones(shape=(self._N_RF_CHANNELS,)),
    )

    expected_called_names = ["mock_hill", "mock_adstock"]

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_get_joint_dist_zeros(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Deterministic(0),
            tau_g_excl_baseline=tfp.distributions.Deterministic(0),
            beta_m=tfp.distributions.Deterministic(0),
            beta_rf=tfp.distributions.Deterministic(0),
            eta_m=tfp.distributions.Deterministic(0),
            eta_rf=tfp.distributions.Deterministic(0),
            gamma_c=tfp.distributions.Deterministic(0),
            xi_c=tfp.distributions.Deterministic(0),
            alpha_m=tfp.distributions.Deterministic(0),
            alpha_rf=tfp.distributions.Deterministic(0),
            ec_m=tfp.distributions.Deterministic(0),
            ec_rf=tfp.distributions.Deterministic(0),
            slope_m=tfp.distributions.Deterministic(0),
            slope_rf=tfp.distributions.Deterministic(0),
            sigma=tfp.distributions.Deterministic(0),
            roi_m=tfp.distributions.Deterministic(0),
            roi_rf=tfp.distributions.Deterministic(0),
        )
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    sample = meridian._get_joint_dist_unpinned().sample(self._N_DRAWS)
    self.assertAllEqual(
        sample.y,
        tf.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_only(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_only,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.EC_M,
        constants.SLOPE_M,
        constants.SIGMA,
        constants.ROI_M,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_M)
    else:
      prior_distribution_params.append(constants.BETA_M)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    beta_m = par[constants.BETA_GM][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_media, beta_m)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_rf_only(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_rf_only,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_RF,
        constants.EC_RF,
        constants.SLOPE_RF,
        constants.SIGMA,
        constants.ROI_RF,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_RF)
    else:
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.reach_scaled,
        frequency=meridian.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    beta_rf = par[constants.BETA_GRF][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_reach, beta_rf)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  # TODO(b/307543975): Add test for holdout_id.
  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_and_rf(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_and_rf,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.ALPHA_RF,
        constants.EC_M,
        constants.EC_RF,
        constants.SLOPE_M,
        constants.SLOPE_RF,
        constants.SIGMA,
        constants.ROI_M,
        constants.ROI_RF,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
    else:
      prior_distribution_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.reach_scaled,
        frequency=meridian.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    combined_transformed_media = tf.concat(
        [transformed_media, transformed_reach], axis=-1
    )

    combined_beta = tf.concat(
        [par[constants.BETA_GM][0, :, :], par[constants.BETA_GRF][0, :, :]],
        axis=-1,
    )
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", combined_transformed_media, combined_beta)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  def test_sample_prior_seed_same_seed(self):
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=1)
    self.assertEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_sample_prior_different_seed(self):
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=2)

    self.assertNotEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_sample_prior_media_and_rf_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_media_and_rf,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(prior, attr) for attr in constants.RF_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_media_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_media_only,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_rf_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_rf_only,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_rf_only,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(prior, attr) for attr in constants.RF_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_posterior_media_and_rf_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_rf_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_rf_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_rf_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_and_rf_sequential_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=[self._N_CHAINS, self._N_CHAINS],
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    n_total_chains = self._N_CHAINS * 2
    knots_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (n_total_chains, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (n_total_chains, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (n_total_chains, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (n_total_chains, self._N_KEEP, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (n_total_chains, self._N_KEEP, 1)
    )
    geo_shape = (n_total_chains, self._N_KEEP, self._N_GEOS)
    time_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )

  def test_save_and_load_works(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, "joblib")
    mmm = model.Meridian(input_data=self.input_data_with_media_and_rf)
    model.save_mmm(mmm, str(file_path))
    self.assertTrue(os.path.exists(file_path))
    new_mmm = model.load_mmm(file_path)
    for attr in dir(mmm):
      if isinstance(getattr(mmm, attr), (int, bool)):
        with self.subTest(name=attr):
          self.assertEqual(getattr(mmm, attr), getattr(new_mmm, attr))
      elif isinstance(getattr(mmm, attr), tf.Tensor):
        with self.subTest(name=attr):
          self.assertAllClose(getattr(mmm, attr), getattr(new_mmm, attr))

  def test_load_error(self):
    with self.assertRaisesWithLiteralMatch(
        FileNotFoundError, "No such file or directory: this/path/does/not/exist"
    ):
      model.load_mmm("this/path/does/not/exist")


if __name__ == "__main__":
  absltest.main()
