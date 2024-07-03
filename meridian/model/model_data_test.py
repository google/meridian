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

from collections.abc import Collection, Sequence
import os
from unittest import mock
import warnings
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import test_utils
from meridian.model import knots as knots_module
from meridian.model import model
from meridian.model import model_data
from meridian.model import spec
import numpy as np
import tensorflow as tf
import xarray as xr


class ModelDataTest(tf.test.TestCase, parameterized.TestCase):

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
      model_data.ModelData(input_data=input_data, model_spec=model_spec)

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
      model_data.ModelData(input_data=input_data, model_spec=model_spec)

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
      _ = model_data.ModelData(
          input_data=input_data, model_spec=model_spec
      ).holdout_id

  def test_init_with_wrong_control_population_scaling_id_shape_fails(self):
    model_spec = spec.ModelSpec(
        control_population_scaling_id=np.ones((7), dtype=bool)
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of `control_population_scaling_id` (7,) is different from"
        " `(n_controls,) = (2,)`.",
    ):
      _ = model_data.ModelData(
          input_data=self.input_data_with_media_and_rf, model_spec=model_spec
      ).controls_scaled

  @parameterized.named_parameters(
      ("none", None, 200), ("int", 3, 3), ("list", [0, 50, 100, 150], 4)
  )
  def test_n_knots(self, knots, expected_n_knots):
    # Create sample model spec with given knots
    model_spec = spec.ModelSpec(knots=knots)

    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_only, model_spec=model_spec
    )

    self.assertEqual(mdata.knot_info.n_knots, expected_n_knots)

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
      _ = model_data.ModelData(
          input_data=self.input_data_with_media_only, model_spec=model_spec
      ).knot_info

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
        return_value=knots_module.KnotInfo(3, np.array([2, 5, 8]), np.eye(3)),
    ) as mock_get_knot_info:
      input_data = (
          self.national_input_data_media_only
          if is_national
          else self.input_data_with_media_only
      )
      _ = model_data.ModelData(
          input_data=input_data,
          model_spec=spec.ModelSpec(knots=knots),
      ).knot_info

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
      model_data.ModelData(
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
        _ = model_data.ModelData(
            input_data=self.input_data_with_media_only,
            model_spec=spec.ModelSpec(knots=4),
        ).knot_info

  def test_init_geo_args_no_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      model_data.ModelData(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist="normal", unique_sigma_for_each_geo=True
          ),
      )
      self.assertEmpty(w)

  def test_init_national_args_with_broadcast_warnings(self):
    with warnings.catch_warnings(record=True) as warns:
      warnings.simplefilter("module")
      _ = model_data.ModelData(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist=constants.MEDIA_EFFECTS_NORMAL
          ),
      ).prior_broadcast
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
      _ = model_data.ModelData(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(unique_sigma_for_each_geo=True),
      ).prior_broadcast
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
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    self.assertEqual(mdata.n_geos, self._N_GEOS)
    self.assertEqual(mdata.n_controls, self._N_CONTROLS)
    self.assertEqual(mdata.n_times, self._N_TIMES)
    self.assertEqual(mdata.n_media_times, self._N_MEDIA_TIMES)
    self.assertFalse(mdata.is_national)
    self.assertIsNotNone(mdata.prior_broadcast)

  def test_base_national_properties(self):
    mdata = model_data.ModelData(
        input_data=self.national_input_data_media_only,
        model_spec=spec.ModelSpec(),
    )
    self.assertEqual(mdata.n_geos, self._N_GEOS_NATIONAL)
    self.assertEqual(mdata.n_controls, self._N_CONTROLS)
    self.assertEqual(mdata.n_times, self._N_TIMES)
    self.assertEqual(mdata.n_media_times, self._N_MEDIA_TIMES)
    self.assertTrue(mdata.is_national)
    self.assertIsNotNone(mdata.prior_broadcast)

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
    mdata = model_data.ModelData(
        input_data=input_data, model_spec=spec.ModelSpec()
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.kpi, dtype=tf.float32),
        mdata.kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.revenue_per_kpi, dtype=tf.float32),
        mdata.revenue_per_kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.controls, dtype=tf.float32),
        mdata.controls,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(input_data.population, dtype=tf.float32),
        mdata.population,
    )
    if input_data.media is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.media, dtype=tf.float32),
          mdata.media_tensors.media,
      )
    if input_data.media_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32),
          mdata.media_tensors.media_spend,
      )
    if input_data.reach is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.reach, dtype=tf.float32),
          mdata.rf_tensors.reach,
      )
    if input_data.frequency is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.frequency, dtype=tf.float32),
          mdata.rf_tensors.frequency,
      )
    if input_data.rf_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32),
          mdata.rf_tensors.rf_spend,
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
          mdata.total_spend,
      )
    elif input_data.media_spend is not None:
      self.assertAllClose(
          tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32),
          mdata.total_spend,
      )
    else:
      self.assertAllClose(
          tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32),
          mdata.total_spend,
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
    mdata = model_data.ModelData(
        input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(media_effects_dist=media_effects_dist),
    )
    self.assertEqual(mdata.media_effects_dist, expected_media_effects_dist)

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
    mdata = model_data.ModelData(
        input_data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(
            unique_sigma_for_each_geo=unique_sigma_for_each_geo
        ),
    )
    self.assertEqual(
        mdata.unique_sigma_for_each_geo, expected_unique_sigma_for_each_geo
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
        f"The following {data_name} variables do not vary across geos, making a"
        f" model with n_knots=n_time unidentifiable: {dims_bad}. This can lead"
        " to poor model convergence. Since these variables only vary across"
        " time and not across geo, they are collinear with time and redundant"
        " in a model with a parameter for each time period.  To address this,"
        " you can either: (1) decrease the number of knots (n_knots < n_time),"
        " or (2) drop the listed variables that do not vary across geos.",
    ):
      model_data.ModelData(
          input_data=test_utils.sample_input_data_from_dataset(
              dataset, kpi_type=constants.NON_REVENUE
          ),
          model_spec=spec.ModelSpec(),
      )

  def test_broadcast_prior_distribution(self):
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        mdata.prior_broadcast.tau_g_excl_baseline.batch_shape,
        (mdata.n_geos - 1,),
    )

    # Validate `n_knots` shape distributions.
    self.assertEqual(
        mdata.prior_broadcast.knot_values.batch_shape,
        (mdata.knot_info.n_knots,),
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        mdata.prior_broadcast.beta_m,
        mdata.prior_broadcast.eta_m,
        mdata.prior_broadcast.alpha_m,
        mdata.prior_broadcast.ec_m,
        mdata.prior_broadcast.slope_m,
        mdata.prior_broadcast.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (mdata.n_media_channels,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        mdata.prior_broadcast.beta_rf,
        mdata.prior_broadcast.eta_rf,
        mdata.prior_broadcast.alpha_rf,
        mdata.prior_broadcast.ec_rf,
        mdata.prior_broadcast.slope_rf,
        mdata.prior_broadcast.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (mdata.n_rf_channels,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        mdata.prior_broadcast.gamma_c,
        mdata.prior_broadcast.xi_c,
    ]
    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (mdata.n_controls,))

    # Validate sigma.
    self.assertEqual(mdata.prior_broadcast.sigma.batch_shape, (1,))

  def test_media_attributes_not_set(self):
    mdata = model_data.ModelData(
        input_data=self.input_data_with_rf_only, model_spec=spec.ModelSpec()
    )
    self.assertEqual(mdata.n_media_channels, 0)
    self.assertIsNone(mdata.media_tensors.media_transformer)
    self.assertIsNone(mdata.media_tensors.media_scaled)
    self.assertIsNone(mdata.media_tensors.media_counterfactual)
    self.assertIsNone(mdata.media_tensors.media_counterfactual_scaled)
    self.assertIsNone(mdata.media_tensors.media_spend_counterfactual)

  def test_rf_attributes_not_set(self):
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_only, model_spec=spec.ModelSpec()
    )
    self.assertEqual(mdata.n_rf_channels, 0)
    self.assertIsNone(mdata.rf_tensors.reach_transformer)
    self.assertIsNone(mdata.rf_tensors.reach_scaled)
    self.assertIsNone(mdata.rf_tensors.reach_counterfactual)
    self.assertIsNone(mdata.rf_tensors.reach_counterfactual_scaled)
    self.assertIsNone(mdata.rf_tensors.rf_spend_counterfactual)

  def test_scaled_data_shape(self):
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    if (
        self.input_data_with_media_and_rf.media is not None
        and mdata.media_tensors.media_scaled is not None
    ):
      self.assertAllEqual(
          mdata.media_tensors.media_scaled.shape,
          self.input_data_with_media_and_rf.media.shape,
          msg=(
              "Shape of `_media_scaled` does not match the shape of `media`"
              " from the input data."
          ),
      )
    if (
        self.input_data_with_media_and_rf.reach is not None
        and mdata.rf_tensors.reach_scaled is not None
    ):
      self.assertAllEqual(
          mdata.rf_tensors.reach_scaled.shape,
          self.input_data_with_media_and_rf.reach.shape,
          msg=(
              "Shape of `_reach_scaled` does not match the shape of `reach`"
              " from the input data."
          ),
      )
    self.assertAllEqual(
        mdata.controls_scaled.shape,
        self.input_data_with_media_and_rf.controls.shape,
        msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
    self.assertAllEqual(
        mdata.kpi_scaled.shape,
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
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf, model_spec=model_spec
    )
    self.assertIsNotNone(
        mdata.controls_transformer._population_scaling_factors,
        msg=(
            "`_population_scaling_factors` not set for the controls"
            " transformer."
        ),
    )
    self.assertAllEqual(
        mdata.controls_transformer._population_scaling_factors.shape,
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
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )

    # With the default tolerance of eps * 10 the test fails due to rounding
    # errors.
    atol = np.finfo(np.float32).eps * 100
    if (
        mdata.media_tensors.media_scaled is not None
        and mdata.media_tensors.media_transformer is not None
        and self.input_data_with_media_and_rf.media is not None
    ):
      self.assertAllClose(
          mdata.media_tensors.media_transformer.inverse(
              mdata.media_tensors.media_scaled
          ),
          self.input_data_with_media_and_rf.media,
          atol=atol,
      )
    if (
        mdata.rf_tensors.reach_scaled is not None
        and mdata.rf_tensors.reach_transformer is not None
        and self.input_data_with_media_and_rf.reach is not None
    ):
      self.assertAllClose(
          mdata.rf_tensors.reach_transformer.inverse(
              mdata.rf_tensors.reach_scaled
          ),
          self.input_data_with_media_and_rf.reach,
          atol=atol,
      )
    self.assertAllClose(
        mdata.controls_transformer.inverse(mdata.controls_scaled),
        self.input_data_with_media_and_rf.controls,
        atol=atol,
    )
    self.assertAllClose(
        mdata.kpi_transformer.inverse(mdata.kpi_scaled),
        self.input_data_with_media_and_rf.kpi,
        atol=atol,
    )

  def test_counterfactual_data_no_roi_calibration(self):
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(
            roi_calibration_period=None, rf_roi_calibration_period=None
        ),
    )

    self.assertAllEqual(
        mdata.media_tensors.media_counterfactual,
        tf.zeros_like(self.input_data_with_media_and_rf.media),
    )
    self.assertAllEqual(
        mdata.media_tensors.media_counterfactual_scaled,
        tf.zeros_like(mdata.media_tensors.media_scaled),
    )
    self.assertAllEqual(
        mdata.media_tensors.media_spend_counterfactual,
        tf.zeros_like(self.input_data_with_media_and_rf.media_spend),
    )
    if mdata.rf_tensors.reach_scaled is not None:
      self.assertAllEqual(
          mdata.rf_tensors.reach_counterfactual,
          tf.zeros_like(self.input_data_with_media_and_rf.reach),
      )
      self.assertAllEqual(
          mdata.rf_tensors.reach_counterfactual_scaled,
          tf.zeros_like(mdata.rf_tensors.reach_scaled),
      )
      self.assertAllEqual(
          mdata.rf_tensors.rf_spend_counterfactual,
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
    mdata = model_data.ModelData(
        input_data=input_data,
        model_spec=spec.ModelSpec(
            roi_calibration_period=roi_calibration_period
        ),
    )

    self.assertAllClose(
        mdata.media_tensors.media_counterfactual,
        tf.where(
            roi_calibration_period,
            0,
            mdata.media_tensors.media,
        ),
    )
    self.assertAllClose(
        mdata.media_tensors.media_counterfactual_scaled,
        tf.where(roi_calibration_period, 0, mdata.media_tensors.media_scaled),
    )

    self.assertAllClose(
        mdata.media_tensors.media_spend_counterfactual,
        tf.where(
            roi_calibration_period[..., -mdata.n_times :, :],
            0,
            mdata.media_tensors.media_spend,
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
    mdata = model_data.ModelData(
        input_data=input_data,
        model_spec=spec.ModelSpec(
            rf_roi_calibration_period=rf_roi_calibration_period
        ),
    )

    if mdata.input_data.reach is not None:
      self.assertAllClose(
          mdata.rf_tensors.reach_counterfactual,
          tf.where(
              rf_roi_calibration_period,
              0,
              mdata.rf_tensors.reach,
          ),
      )
      self.assertAllClose(
          mdata.rf_tensors.reach_counterfactual_scaled,
          tf.where(rf_roi_calibration_period, 0, mdata.rf_tensors.reach_scaled),
      )

      self.assertAllClose(
          mdata.rf_tensors.rf_spend_counterfactual,
          tf.where(
              rf_roi_calibration_period[..., -mdata.n_times :, :],
              0,
              mdata.rf_tensors.rf_spend,
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
    mdata = model_data.ModelData(
        input_data=self.input_data_with_media_only,
        model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
    )
    self.assertEqual(mdata.baseline_geo_idx, expected_idx)

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
      _ = model_data.ModelData(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
      ).baseline_geo_idx


if __name__ == "__main__":
  absltest.main()
