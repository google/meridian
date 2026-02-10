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

from collections.abc import Collection, Mapping, Sequence
from typing import Any
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.data import input_data
from meridian.data import test_utils as data_test_utils
from meridian.model import context
from meridian.model import knots as knots_module
from meridian.model import model_test_data
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import xarray as xr


# Data dimensions for sample input.
_MEDIA_CHANNEL_NAMES = ("ch_0", "ch_1", "ch_2")
_RF_CHANNEL_NAMES = ("rf_ch_0", "rf_ch_1", "rf_ch_2")


class ContextTest(
    test_utils.MeridianTestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

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
    data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      context.ModelContext(input_data=data, model_spec=model_spec)

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
    data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
      context.ModelContext(input_data=data, model_spec=model_spec)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          input_data_type="national",
          error_msg=(
              "The shape of `holdout_id` (2, 8) is different from `(n_times,) ="
              " (200,)`."
          ),
      ),
      dict(
          testcase_name="geo",
          input_data_type="geo",
          error_msg=(
              "The shape of `holdout_id` (2, 8) is different from `(n_geos,"
              " n_times) = (5, 200)`."
          ),
      ),
  )
  def test_init_with_wrong_holdout_id_shape_fails(
      self, input_data_type: str, error_msg: str
  ):
    model_spec = spec.ModelSpec(holdout_id=np.ones((2, 8), dtype=bool))
    data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      _ = context.ModelContext(
          input_data=data, model_spec=model_spec
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
      context.ModelContext(
          input_data=self.input_data_with_media_and_rf, model_spec=model_spec
      )

  def test_init_with_wrong_non_media_population_scaling_id_shape_fails(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(
        non_media_population_scaling_id=np.ones((7), dtype=bool)
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of `non_media_population_scaling_id` (7,) is different from"
        " `(n_non_media_channels,) = (2,)`.",
    ):
      context.ModelContext(
          input_data=data,
          model_spec=model_spec,
      )

  @parameterized.named_parameters(
      ("none", None, 200), ("int", 3, 3), ("list", [0, 50, 100, 150], 4)
  )
  def test_n_knots(self, knots, expected_n_knots):
    # Create sample model spec with given knots
    model_spec = spec.ModelSpec(knots=knots)

    model_context = context.ModelContext(
        input_data=self.input_data_with_media_only, model_spec=model_spec
    )

    self.assertEqual(model_context.knot_info.n_knots, expected_n_knots)

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
      _ = context.ModelContext(
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
      data = (
          self.national_input_data_media_only
          if is_national
          else self.input_data_with_media_only
      )
      _ = context.ModelContext(
          input_data=data,
          model_spec=spec.ModelSpec(knots=knots),
      ).knot_info
      mock_get_knot_info.assert_called_once_with(
          self._N_TIMES,
          knots,
          False,
          data,
          is_national,
      )

  def test_base_geo_properties(self):
    model_context = context.ModelContext(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    self.assertEqual(model_context.n_geos, self._N_GEOS)
    self.assertEqual(model_context.n_controls, self._N_CONTROLS)
    self.assertEqual(model_context.n_times, self._N_TIMES)
    self.assertEqual(model_context.n_media_times, self._N_MEDIA_TIMES)
    self.assertFalse(model_context.is_national)

  def test_base_national_properties(self):
    model_context = context.ModelContext(
        input_data=self.national_input_data_media_only,
        model_spec=spec.ModelSpec(),
    )
    self.assertEqual(model_context.n_geos, self._N_GEOS_NATIONAL)
    self.assertEqual(model_context.n_controls, self._N_CONTROLS)
    self.assertEqual(model_context.n_times, self._N_TIMES)
    self.assertEqual(model_context.n_media_times, self._N_MEDIA_TIMES)
    self.assertTrue(model_context.is_national)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_only",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_only",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=input_data_samples._N_RF_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_and_media",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="media_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="rf_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="media_rf_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_input_data_tensor_properties(self, data):
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.kpi, dtype=backend.float32),
        model_context.kpi,
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.revenue_per_kpi, dtype=backend.float32),
        model_context.revenue_per_kpi,
    )
    if data.controls is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.controls, dtype=backend.float32),
          model_context.controls,
      )
    if data.non_media_treatments is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.non_media_treatments, dtype=backend.float32),
          model_context.non_media_treatments,
      )
    test_utils.assert_allequal(
        backend.to_tensor(data.population, dtype=backend.float32),
        model_context.population,
    )
    if data.media is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.media, dtype=backend.float32),
          model_context.media_tensors.media,
      )
    if data.media_spend is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.media_spend, dtype=backend.float32),
          model_context.media_tensors.media_spend,
      )
    if data.reach is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.reach, dtype=backend.float32),
          model_context.rf_tensors.reach,
      )
    if data.frequency is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.frequency, dtype=backend.float32),
          model_context.rf_tensors.frequency,
      )
    if data.rf_spend is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.rf_spend, dtype=backend.float32),
          model_context.rf_tensors.rf_spend,
      )
    if data.organic_media is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_media, dtype=backend.float32),
          model_context.organic_media_tensors.organic_media,
      )
    if data.organic_reach is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_reach, dtype=backend.float32),
          model_context.organic_rf_tensors.organic_reach,
      )
    if data.organic_frequency is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_frequency, dtype=backend.float32),
          model_context.organic_rf_tensors.organic_frequency,
      )
    if data.media_spend is not None and data.rf_spend is not None:
      test_utils.assert_allclose(
          backend.concatenate(
              [
                  backend.to_tensor(data.media_spend, dtype=backend.float32),
                  backend.to_tensor(data.rf_spend, dtype=backend.float32),
              ],
              axis=-1,
          ),
          model_context.total_spend,
      )
    elif data.media_spend is not None:
      test_utils.assert_allclose(
          backend.to_tensor(data.media_spend, dtype=backend.float32),
          model_context.total_spend,
      )
    elif data.rf_spend is not None:
      test_utils.assert_allclose(
          backend.to_tensor(data.rf_spend, dtype=backend.float32),
          model_context.total_spend,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo_normal",
          n_geos=input_data_samples._N_GEOS,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
      dict(
          testcase_name="geo_log_normal",
          n_geos=input_data_samples._N_GEOS,
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
      ),
      dict(
          testcase_name="national_normal",
          n_geos=input_data_samples._N_GEOS_NATIONAL,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
      dict(
          testcase_name="national_log_normal",
          n_geos=input_data_samples._N_GEOS_NATIONAL,
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
          expected_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
  )
  def test_media_effects_dist_property(
      self, n_geos, media_effects_dist, expected_media_effects_dist
  ):
    model_context = context.ModelContext(
        input_data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(media_effects_dist=media_effects_dist),
    )
    self.assertEqual(
        model_context.media_effects_dist, expected_media_effects_dist
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo_unique_sigma_for_each_geo_true",
          n_geos=input_data_samples._N_GEOS,
          unique_sigma_for_each_geo=True,
          expected_unique_sigma_for_each_geo=True,
      ),
      dict(
          testcase_name="geo_unique_sigma_for_each_geo_false",
          n_geos=input_data_samples._N_GEOS,
          unique_sigma_for_each_geo=False,
          expected_unique_sigma_for_each_geo=False,
      ),
      dict(
          testcase_name="national_unique_sigma_for_each_geo_true",
          n_geos=input_data_samples._N_GEOS_NATIONAL,
          unique_sigma_for_each_geo=True,
          expected_unique_sigma_for_each_geo=False,
      ),
      dict(
          testcase_name="national_unique_sigma_for_each_geo_false",
          n_geos=input_data_samples._N_GEOS_NATIONAL,
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
    model_context = context.ModelContext(
        input_data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=n_geos, n_media_channels=self._N_MEDIA_CHANNELS
        ),
        model_spec=spec.ModelSpec(
            unique_sigma_for_each_geo=unique_sigma_for_each_geo
        ),
    )
    self.assertEqual(
        model_context.unique_sigma_for_each_geo,
        expected_unique_sigma_for_each_geo,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="tau_g_excl_baseline",
          dist_name="tau_g_excl_baseline",
          expected_shape_func=lambda mc: (mc.n_geos - 1,),
      ),
      dict(
          testcase_name="knot_values",
          dist_name="knot_values",
          expected_shape_func=lambda mc: (mc.knot_info.n_knots,),
      ),
      dict(
          testcase_name="beta_m",
          dist_name="beta_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="eta_m",
          dist_name="eta_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="alpha_m",
          dist_name="alpha_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="ec_m",
          dist_name="ec_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="slope_m",
          dist_name="slope_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="roi_m",
          dist_name="roi_m",
          expected_shape_func=lambda mc: (mc.n_media_channels,),
      ),
      dict(
          testcase_name="beta_rf",
          dist_name="beta_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="eta_rf",
          dist_name="eta_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="alpha_rf",
          dist_name="alpha_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="ec_rf",
          dist_name="ec_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="slope_rf",
          dist_name="slope_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="roi_rf",
          dist_name="roi_rf",
          expected_shape_func=lambda mc: (mc.n_rf_channels,),
      ),
      dict(
          testcase_name="beta_om",
          dist_name="beta_om",
          expected_shape_func=lambda mc: (mc.n_organic_media_channels,),
      ),
      dict(
          testcase_name="eta_om",
          dist_name="eta_om",
          expected_shape_func=lambda mc: (mc.n_organic_media_channels,),
      ),
      dict(
          testcase_name="alpha_om",
          dist_name="alpha_om",
          expected_shape_func=lambda mc: (mc.n_organic_media_channels,),
      ),
      dict(
          testcase_name="ec_om",
          dist_name="ec_om",
          expected_shape_func=lambda mc: (mc.n_organic_media_channels,),
      ),
      dict(
          testcase_name="slope_om",
          dist_name="slope_om",
          expected_shape_func=lambda mc: (mc.n_organic_media_channels,),
      ),
      dict(
          testcase_name="beta_orf",
          dist_name="beta_orf",
          expected_shape_func=lambda mc: (mc.n_organic_rf_channels,),
      ),
      dict(
          testcase_name="eta_orf",
          dist_name="eta_orf",
          expected_shape_func=lambda mc: (mc.n_organic_rf_channels,),
      ),
      dict(
          testcase_name="alpha_orf",
          dist_name="alpha_orf",
          expected_shape_func=lambda mc: (mc.n_organic_rf_channels,),
      ),
      dict(
          testcase_name="ec_orf",
          dist_name="ec_orf",
          expected_shape_func=lambda mc: (mc.n_organic_rf_channels,),
      ),
      dict(
          testcase_name="slope_orf",
          dist_name="slope_orf",
          expected_shape_func=lambda mc: (mc.n_organic_rf_channels,),
      ),
      dict(
          testcase_name="gamma_c",
          dist_name="gamma_c",
          expected_shape_func=lambda mc: (mc.n_controls,),
      ),
      dict(
          testcase_name="xi_c",
          dist_name="xi_c",
          expected_shape_func=lambda mc: (mc.n_controls,),
      ),
      dict(
          testcase_name="gamma_n",
          dist_name="gamma_n",
          expected_shape_func=lambda mc: (mc.n_non_media_channels,),
      ),
      dict(
          testcase_name="xi_n",
          dist_name="xi_n",
          expected_shape_func=lambda mc: (mc.n_non_media_channels,),
      ),
      dict(
          testcase_name="sigma",
          dist_name="sigma",
          expected_shape_func=lambda mc: (),
      ),
  )
  def test_broadcast_prior_distribution_shapes(
      self, dist_name, expected_shape_func
  ):
    data = self.input_data_non_media_and_organic
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )
    dist = getattr(model_context.prior_broadcast, dist_name)
    self.assertEqual(dist.batch_shape, expected_shape_func(model_context))

  def test_scaled_data_shape(self):
    data = self.input_data_non_media_and_organic
    controls = data.controls
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )
    self.assertIsNotNone(model_context.controls_scaled)
    self.assertIsNotNone(controls)
    test_utils.assert_allequal(
        model_context.controls_scaled.shape,  # pytype: disable=attribute-error
        controls.shape,
        err_msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
    self.assertIsNotNone(model_context.non_media_treatments_normalized)
    self.assertIsNotNone(data.non_media_treatments)
    # pytype: disable=attribute-error
    test_utils.assert_allequal(
        model_context.non_media_treatments_normalized.shape,
        data.non_media_treatments.shape,
        err_msg=(
            "Shape of `_non_media_treatments_scaled` does not match the shape"
            " of `non_media_treatments` from the input data."
        ),
    )
    # pytype: enable=attribute-error
    test_utils.assert_allequal(
        model_context.kpi_scaled.shape,
        data.kpi.shape,
        err_msg=(
            "Shape of `_kpi_scaled` does not match the shape of"
            " `kpi` from the input data."
        ),
    )

  def test_scaled_data_no_controls(self):
    data = self.input_data_with_media_and_rf_no_controls
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )

    self.assertEqual(model_context.n_controls, 0)
    self.assertIsNone(model_context.controls)
    self.assertIsNone(model_context.controls_transformer)
    self.assertIsNone(model_context.controls_scaled)
    test_utils.assert_allequal(
        model_context.kpi_scaled.shape,
        data.kpi.shape,
        err_msg=(
            "Shape of `_kpi_scaled` does not match the shape of"
            " `kpi` from the input data."
        ),
    )

  def test_population_scaled_controls_transformer_set(self):
    data = self.input_data_with_media_and_rf
    model_spec = spec.ModelSpec(
        control_population_scaling_id=backend.to_tensor(
            [True for _ in data.control_variable]
        )
    )
    model_context = context.ModelContext(input_data=data, model_spec=model_spec)
    self.assertIsNotNone(model_context.controls_transformer)
    self.assertIsNotNone(
        model_context.controls_transformer._population_scaling_factors,  # pytype: disable=attribute-error
        msg=(
            "`_population_scaling_factors` not set for the controls"
            " transformer."
        ),
    )
    test_utils.assert_allequal(
        model_context.controls_transformer._population_scaling_factors.shape,  # pytype: disable=attribute-error
        [len(data.geo), len(data.control_variable)],
        err_msg=(
            "Shape of `controls_transformer._population_scaling_factors` does"
            " not match (`n_geos`, `n_controls`)."
        ),
    )

  def test_population_scaled_non_media_transformer_set(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(
        non_media_population_scaling_id=backend.to_tensor(
            [True for _ in data.non_media_channel]
        )
    )
    model_context = context.ModelContext(input_data=data, model_spec=model_spec)
    self.assertIsNotNone(model_context.non_media_transformer)
    # pytype: disable=attribute-error
    self.assertIsNotNone(
        model_context.non_media_transformer._population_scaling_factors,
        msg=(
            "`_population_scaling_factors` not set for the non-media"
            " transformer."
        ),
    )
    test_utils.assert_allequal(
        model_context.non_media_transformer._population_scaling_factors.shape,
        [
            len(data.geo),
            len(data.non_media_channel),
        ],
        err_msg=(
            "Shape of"
            " `non_media_transformer._population_scaling_factors` does"
            " not match (`n_geos`, `n_non_media_channels`)."
        ),
    )
    # pytype: enable=attribute-error

  def test_scaled_data_inverse_is_identity(self):
    data = self.input_data_non_media_and_organic
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )

    # With the default tolerance of eps * 10 the test fails due to rounding
    # errors.
    atol = np.finfo(np.float32).eps * 100
    test_utils.assert_allclose(
        model_context.controls_transformer.inverse(model_context.controls_scaled),  # pytype: disable=attribute-error
        data.controls,
        atol=atol,
    )
    self.assertIsNotNone(model_context.non_media_transformer)
    # pytype: disable=attribute-error
    test_utils.assert_allclose(
        model_context.non_media_transformer.inverse(
            model_context.non_media_treatments_normalized
        ),
        data.non_media_treatments,
        atol=atol,
    )
    # pytype: enable=attribute-error
    test_utils.assert_allclose(
        model_context.kpi_transformer.inverse(model_context.kpi_scaled),
        data.kpi,
        atol=atol,
    )

  @parameterized.named_parameters(
      dict(testcase_name="int", baseline_geo=4, expected_idx=4),
      dict(testcase_name="str", baseline_geo="geo_1", expected_idx=1),
      dict(testcase_name="none", baseline_geo=None, expected_idx=2),
  )
  def test_baseline_geo_idx(
      self, baseline_geo: int | str | None, expected_idx: int
  ):
    data = self.input_data_with_media_only
    data.population.data = [
        2.0,
        5.0,
        20.0,
        7.0,
        10.0,
    ]
    model_context = context.ModelContext(
        input_data=data,
        model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
    )
    self.assertEqual(model_context.baseline_geo_idx, expected_idx)

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
      _ = context.ModelContext(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
      ).baseline_geo_idx

  def test_validate_media_prior_type_mroi(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Custom priors should be set on `mroi_m` when `media_prior_type` is"
        ' "mroi", KPI is non-revenue and revenue per kpi data is missing.',
    ):
      context.ModelContext(
          input_data=self.input_data_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(
              media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI
          ),
      )

  def test_validate_rf_prior_type_mroi(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Custom priors should be set on `mroi_rf` when `rf_prior_type` is"
        ' "mroi", KPI is non-revenue and revenue per kpi data is missing.',
    ):
      context.ModelContext(
          input_data=self.input_data_media_and_rf_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(
              rf_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI
          ),
      )

  def test_validate_media_prior_type_roi(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Custom priors should be set on `roi_m` when `media_prior_type` is"
        ' "roi", custom priors are assigned on `{constants.ROI_RF}` or'
        ' `rf_prior_type` is not "roi", KPI is non-revenue and revenue per kpi'
        " data is missing.",
    ):
      context.ModelContext(
          input_data=self.input_data_media_and_rf_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(
              media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
              rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          ),
      )

  def test_validate_rf_prior_type_roi(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Custom priors should be set on `roi_rf` when `rf_prior_type` is "roi",'
        " custom priors are assigned on `{constants.ROI_M}` or"
        ' `media_prior_type` is not "roi", KPI is non-revenue and revenue per'
        " kpi data is missing.",
    ):
      context.ModelContext(
          input_data=self.input_data_media_and_rf_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(
              rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
              media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="roi_m",
          dist_args=([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.9),
          dist_name=constants.ROI_M,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="roi_rf",
          dist_args=(0.0, 0.9),
          dist_name=constants.ROI_RF,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="mroi_m",
          dist_args=(0.5, 0.9),
          dist_name=constants.MROI_M,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
      ),
      dict(
          testcase_name="mroi_rf",
          dist_args=([0.0, 0.0, 0.0, 0.0], 0.9),
          dist_name=constants.MROI_RF,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
      ),
      dict(
          testcase_name="contribution_m",
          dist_args=(0.0, 0.9),
          dist_name=constants.CONTRIBUTION_M,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ),
      dict(
          testcase_name="contribution_rf",
          dist_args=(0.0, 0.9),
          dist_name=constants.CONTRIBUTION_RF,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ),
  )
  def test_check_for_negative_support_paid_media_raises_error(
      self,
      dist_args: tuple[list[float] | float, float],
      dist_name: str,
      media_prior_type: str,
  ):
    dist = backend.tfd.Normal(*dist_args, name=dist_name)
    prior_dist = prior_distribution.PriorDistribution(**{dist_name: dist})
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Media priors must have non-negative support when"
        ' `media_effects_dist`="log_normal". Found negative prior distribution'
        f" support for {dist_name}.",
    ):
      context.ModelContext(
          input_data=self.input_data_with_media_and_rf,
          model_spec=spec.ModelSpec(
              media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
              media_prior_type=media_prior_type,
              prior=prior_dist,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="contribution_om",
          dist_args=(0.0, 0.9),
          dist_name=constants.CONTRIBUTION_OM,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ),
      dict(
          testcase_name="contribution_orf",
          dist_args=(0.0, 0.9),
          dist_name=constants.CONTRIBUTION_ORF,
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ),
  )
  def test_check_for_negative_support_organic_media_raises_error(
      self,
      dist_args: tuple[list[float] | float, float],
      dist_name: str,
      media_prior_type: str,
  ):
    dist = backend.tfd.Normal(*dist_args, name=dist_name)
    prior_dist = prior_distribution.PriorDistribution(**{dist_name: dist})
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Media priors must have non-negative support when"
        ' `media_effects_dist`="log_normal". Found negative prior distribution'
        f" support for {dist_name}.",
    ):
      context.ModelContext(
          input_data=self.input_data_non_media_and_organic,
          model_spec=spec.ModelSpec(
              media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
              media_prior_type=media_prior_type,
              prior=prior_dist,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="custom_beta_m_prior_type_roi",
          custom_dist_kwargs={
              constants.BETA_M: {
                  "loc": 0.2,
                  "scale": 0.8,
                  "name": constants.BETA_M,
              }
          },
          ignored_priors="beta_m",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          wrong_prior_type_var_name="media_prior_type",
          wrong_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="custom_mroi_rf_prior_type_roi",
          custom_dist_kwargs={
              constants.MROI_M: {
                  "loc": 0.2,
                  "scale": 0.8,
                  "name": constants.MROI_M,
              },
              constants.MROI_RF: {
                  "loc": 0.2,
                  "scale": 0.8,
                  "name": constants.MROI_RF,
              },
          },
          ignored_priors="mroi_rf",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          wrong_prior_type_var_name="rf_prior_type",
          wrong_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="custom_beta_m_roi_m_prior_type_mroi",
          custom_dist_kwargs={
              constants.BETA_M: {
                  "loc": 0.7,
                  "scale": 0.9,
                  "name": constants.BETA_M,
              },
              constants.BETA_RF: {
                  "loc": 0.8,
                  "scale": 0.9,
                  "name": constants.BETA_RF,
              },
              constants.ROI_M: {
                  "loc": 0.2,
                  "scale": 0.1,
                  "name": constants.ROI_M,
              },
          },
          ignored_priors="beta_m, roi_m",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          wrong_prior_type_var_name="media_prior_type",
          wrong_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
      ),
      dict(
          testcase_name="custom_roi_rf_prior_type_coefficient",
          custom_dist_kwargs={
              constants.ROI_RF: {
                  "loc": 0.2,
                  "scale": 0.1,
                  "name": constants.ROI_RF,
              }
          },
          ignored_priors="roi_rf",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          wrong_prior_type_var_name="rf_prior_type",
          wrong_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      ),
  )
  def test_warn_setting_ignored_priors(
      self,
      custom_dist_kwargs: Mapping[str, Mapping[str, Any]],
      ignored_priors: str,
      media_prior_type: str,
      rf_prior_type: str,
      wrong_prior_type_var_name: str,
      wrong_prior_type: str,
  ):
    custom_distributions = {
        name: backend.tfd.LogNormal(**kwargs)
        for name, kwargs in custom_dist_kwargs.items()
    }
    # Create prior distribution with given parameters.
    distribution = prior_distribution.PriorDistribution(**custom_distributions)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      context.ModelContext(
          input_data=self.input_data_with_media_and_rf,
          model_spec=spec.ModelSpec(
              prior=distribution,
              media_prior_type=media_prior_type,
              rf_prior_type=rf_prior_type,
          ),
      )
      self.assertLen(w, 1)
      self.assertEqual(
          (
              f"Custom prior(s) `{ignored_priors}` are ignored when"
              f" `{wrong_prior_type_var_name}` is set to"
              f' "{wrong_prior_type}".'
          ),
          str(w[0].message),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_controls",
          dataset=data_test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_CONTROLS,
          data_name=constants.CONTROLS,
          dims_bad=np.array(["control_0", "control_1"]),
      ),
      dict(
          testcase_name="wrong_media",
          dataset=data_test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_MEDIA,
          data_name=constants.MEDIA,
          dims_bad=np.array(["media_channel_1", "media_channel_2"]),
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=data_test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_REACH,
          data_name=constants.REACH,
          dims_bad=np.array(["rf_channel_0", "rf_channel_1"]),
      ),
  )
  def test_init_without_geo_variation_fails(
      self, dataset: xr.Dataset, data_name: str, dims_bad: Sequence[str]
  ):
    with self.assertRaisesRegex(
        ValueError,
        f"The following {data_name} variables do not vary across geos.*"
        f"{'.*'.join(dims_bad)}",
    ):
      context.ModelContext(
          input_data=data_test_utils.sample_input_data_from_dataset(
              dataset, kpi_type=constants.NON_REVENUE
          ),
          model_spec=spec.ModelSpec(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_controls",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_CONTROLS,
          data_name=constants.CONTROLS,
          dims_bad=np.array(["control_0", "control_1"]),
      ),
      dict(
          testcase_name="wrong_non_media_treatments",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_NON_MEDIA_TREATMENTS,
          data_name=constants.NON_MEDIA_TREATMENTS,
          dims_bad=np.array(["non_media_channel_0", "non_media_channel_1"]),
      ),
      dict(
          testcase_name="wrong_media",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_MEDIA,
          data_name=constants.MEDIA,
          dims_bad=np.array(["media_channel_1", "media_channel_2"]),
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_REACH,
          data_name=constants.REACH,
          dims_bad=np.array(["rf_channel_0", "rf_channel_1"]),
      ),
      dict(
          testcase_name="wrong_organic_media",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_ORGANIC_MEDIA,
          data_name=constants.ORGANIC_MEDIA,
          dims_bad=np.array(["organic_media_channel_0"]),
      ),
      dict(
          testcase_name="wrong_organic_rf",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_ORGANIC_REACH,
          data_name=constants.ORGANIC_REACH,
          dims_bad=np.array(["organic_rf_channel_1"]),
      ),
  )
  def test_init_without_time_variation_fails(
      self, dataset: xr.Dataset, data_name: str, dims_bad: Sequence[str]
  ):
    with self.assertRaisesRegex(
        ValueError,
        f"The following {data_name} variables do not vary across time.*"
        f"{'.*'.join(dims_bad)}",
    ):
      context.ModelContext(
          input_data=data_test_utils.sample_input_data_from_dataset(
              dataset, kpi_type=constants.NON_REVENUE
          ),
          model_spec=spec.ModelSpec(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_controls",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_CONTROLS,
          data_name=constants.CONTROLS,
          dims_bad=np.array(["control_0", "control_1"]),
      ),
      dict(
          testcase_name="wrong_non_media_treatments",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_NON_MEDIA_TREATMENTS,
          data_name=constants.NON_MEDIA_TREATMENTS,
          dims_bad=np.array(["non_media_channel_0", "non_media_channel_1"]),
      ),
      dict(
          testcase_name="wrong_media",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_MEDIA,
          data_name=constants.MEDIA,
          dims_bad=np.array(["media_channel_1", "media_channel_2"]),
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_REACH,
          data_name=constants.REACH,
          dims_bad=np.array(["rf_channel_0", "rf_channel_1"]),
      ),
      dict(
          testcase_name="wrong_organic_media",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_ORGANIC_MEDIA,
          data_name=constants.ORGANIC_MEDIA,
          dims_bad=np.array(["organic_media_channel_0"]),
      ),
      dict(
          testcase_name="wrong_organic_rf",
          dataset=data_test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_ORGANIC_REACH,
          data_name=constants.ORGANIC_REACH,
          dims_bad=np.array(["organic_rf_channel_1"]),
      ),
  )
  def test_init_without_time_variation_national_model_fails(
      self, dataset: xr.Dataset, data_name: str, dims_bad: Sequence[str]
  ):
    national_dataset = dataset.sel(geo=["geo_0"])
    with self.assertRaisesRegex(
        ValueError,
        f"The following {data_name} variables do not vary across time.*"
        f"{'.*'.join(dims_bad)}",
    ):
      context.ModelContext(
          input_data=data_test_utils.sample_input_data_from_dataset(
              national_dataset, kpi_type=constants.NON_REVENUE
          ),
          model_spec=spec.ModelSpec(),
      )

  def test_custom_priors_not_passed_in_ok(self):
    data = self.input_data_non_revenue_no_revenue_per_kpi
    model_context = context.ModelContext(
        input_data=data,
        model_spec=spec.ModelSpec(
            media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
        ),
    )
    # Compare input data.
    self.assertEqual(model_context.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(
        media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    )

    # Compare model spec.
    self.assertEqual(repr(model_context.model_spec), repr(sample_spec))

  def test_custom_priors_okay_with_array_params(self):
    prior = prior_distribution.PriorDistribution(
        roi_m=backend.tfd.LogNormal([1, 1], [1, 1])
    )
    data = self.input_data_non_revenue_no_revenue_per_kpi
    model_context = context.ModelContext(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior),
    )
    # Compare input data.
    self.assertEqual(model_context.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(prior=prior)

    # Compare model spec.
    self.assertEqual(repr(model_context.model_spec), repr(sample_spec))

  def test_get_knot_info_fails(self):
    error_msg = "Knots must be all non-negative."
    with mock.patch.object(
        knots_module,
        "get_knot_info",
        autospec=True,
        side_effect=ValueError(error_msg),
    ):
      with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
        _ = context.ModelContext(
            input_data=self.input_data_with_media_only,
            model_spec=spec.ModelSpec(knots=4),
        ).knot_info

  def test_init_with_default_parameters_works(self):
    data = self.input_data_with_media_only
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )

    # Compare input data.
    self.assertEqual(model_context.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(model_context.model_spec), repr(sample_spec))

  def test_init_with_default_national_parameters_works(self):
    data = self.national_input_data_media_only
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )

    # Compare input data.
    self.assertEqual(model_context.input_data, data)

    # Create sample model spec for comparison
    expected_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(model_context.model_spec), repr(expected_spec))

  def test_init_geo_args_no_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      context.ModelContext(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist="normal", unique_sigma_for_each_geo=True
          ),
      )
      self.assertEmpty(w)

  def test_init_national_args_with_broadcast_warnings(self):
    with warnings.catch_warnings(record=True) as warns:
      warnings.simplefilter("module")
      _ = context.ModelContext(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(
              media_effects_dist=constants.MEDIA_EFFECTS_NORMAL
          ),
      ).prior_broadcast
      # 7 warnings from the broadcasting (tau_g_excl_baseline, eta_m, eta_rf,
      # xi_c, eta_om, eta_orf, xi_n)
      self.assertLen(warns, 7)
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
      _ = context.ModelContext(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(unique_sigma_for_each_geo=True),
      ).prior_broadcast
      # 7 warnings from the broadcasting (tau_g_excl_baseline, eta_m, eta_rf,
      # xi_c, eta_om, eta_orf, xi_n).
      self.assertLen(w, 7)

  @parameterized.named_parameters(
      dict(
          testcase_name="1d",
          get_total_spend=np.array([1.0, 2.0, 3.0, 4.0]),
          expected_total_spend=np.array([1.0, 2.0, 3.0, 4.0]),
      ),
      dict(
          testcase_name="2d",
          get_total_spend=np.array([[1.0, 2.0], [4.0, 5.0]]),
          expected_total_spend=np.array([5.0, 7.0]),
      ),
      dict(
          testcase_name="3d",
          get_total_spend=np.array([
              [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
              [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
          ]),
          expected_total_spend=np.array([55.0, 77.0, 99.0]),
      ),
  )
  def test_broadcast_is_called_non_revenue_no_revenue_per_kpi_total_spend(
      self, get_total_spend: np.ndarray, expected_total_spend: np.ndarray
  ):
    mock_get_total_spend = self.enter_context(
        mock.patch.object(
            input_data.InputData,
            "get_total_spend",
            autospec=True,
        )
    )
    mock_get_total_spend.return_value = get_total_spend
    mock_broadcast = self.enter_context(
        mock.patch.object(
            prior_distribution.PriorDistribution,
            "broadcast",
            autospec=True,
        )
    )
    model_context = context.ModelContext(
        input_data=self.input_data_non_revenue_no_revenue_per_kpi,
        model_spec=spec.ModelSpec(),
    )
    _ = model_context.prior_broadcast

    _, mock_kwargs = mock_broadcast.call_args
    self.assertEqual(mock_kwargs["set_total_media_contribution_prior"], True)
    self.assertEqual(mock_kwargs["kpi"], np.sum(model_context.input_data.kpi))
    np.testing.assert_allclose(mock_kwargs["total_spend"], expected_total_spend)

  def test_default_roi_prior_distribution_raises_warning(
      self,
  ):
    data = self.input_data_non_revenue_no_revenue_per_kpi
    with warnings.catch_warnings(record=True) as warns:
      # Cause all warnings to always be triggered.
      warnings.simplefilter("always")

      model_context = context.ModelContext(
          input_data=data, model_spec=spec.ModelSpec()
      )

      _ = model_context.prior_broadcast
      self.assertLen(warns, 1, f"warns: {[w.message for w in warns]}")
      for w in warns:
        self.assertTrue(issubclass(w.category, UserWarning))
        self.assertIn(
            "Consider setting custom ROI priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with `p_mean=0.4` and"
            " `p_sd=0.2`. Further documentation available at "
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior",
            str(w.message),
        )

  def test_aks_returns_correct_knot_info(self):
    data, expected_knot_info = (
        data_test_utils.sample_input_data_for_aks_with_expected_knot_info()
    )
    model_spec = spec.ModelSpec(enable_aks=True)
    actual_knot_info = context.ModelContext(
        input_data=data, model_spec=model_spec
    ).knot_info
    self.assertEqual(actual_knot_info.n_knots, expected_knot_info.n_knots)
    np.testing.assert_equal(
        actual_knot_info.knot_locations, expected_knot_info.knot_locations
    )
    np.testing.assert_equal(
        actual_knot_info.weights, expected_knot_info.weights
    )


class AdstockDecaySpecFromChannelMappingTest(
    test_utils.MeridianTestCase,
):

  @parameterized.product(**data_test_utils.ADSTOCK_DECAY_SPEC_CASES)
  def test_from_channel(
      self,
      media,
      rf,
      organic_media,
      organic_rf,
  ):
    """Test if adstock decay functions are explicitly passed for all channels."""

    if not (media or rf):
      self.skipTest("Invalid test case: Meridian requires paid media.")

    inp_data = data_test_utils.sample_input_data_revenue(
        n_media_channels=len(media),
        n_rf_channels=len(rf),
        n_organic_media_channels=len(organic_media),
        n_organic_rf_channels=len(organic_rf),
    )

    decay_spec = media | rf | organic_media | organic_rf
    model_spec = spec.ModelSpec(adstock_decay_spec=decay_spec)
    model_context = context.ModelContext(
        input_data=inp_data, model_spec=model_spec
    )

    expected_media = list(media.values()) or constants.GEOMETRIC_DECAY
    expected_rf = list(rf.values()) or constants.GEOMETRIC_DECAY
    expected_organic_media = (
        list(organic_media.values()) or constants.GEOMETRIC_DECAY
    )
    expected_organic_rf = list(organic_rf.values()) or constants.GEOMETRIC_DECAY

    self.assertSequenceEqual(
        model_context.adstock_decay_spec.media, expected_media
    )
    self.assertSequenceEqual(model_context.adstock_decay_spec.rf, expected_rf)
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_media, expected_organic_media
    )
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_rf, expected_organic_rf
    )

  @parameterized.product(
      **data_test_utils.ADSTOCK_DECAY_SPEC_CASES,
      has_undefined_media_channel=(True, False),
      has_undefined_rf_channel=(True, False),
      has_undefined_organic_media_channel=(True, False),
      has_undefined_organic_rf_channel=(True, False),
  )
  def test_from_channels_some_undefined(
      self,
      media,
      rf,
      organic_media,
      organic_rf,
      has_undefined_media_channel,
      has_undefined_rf_channel,
      has_undefined_organic_media_channel,
      has_undefined_organic_rf_channel,
  ):
    """Test if adstock decay functions are not explicitly passed for all channels."""
    if not (
        media or rf or has_undefined_media_channel or has_undefined_rf_channel
    ):
      self.skipTest("Invalid test case: Meridian requires paid media.")

    if not sum((
        has_undefined_media_channel,
        has_undefined_rf_channel,
        has_undefined_organic_media_channel,
        has_undefined_organic_rf_channel,
    )):
      self.skipTest("Redundant test case: no undefined channels.")

    inp_data = data_test_utils.sample_input_data_revenue(
        n_media_channels=len(media) + has_undefined_media_channel,
        n_rf_channels=len(rf) + has_undefined_rf_channel,
        n_organic_media_channels=len(organic_media)
        + has_undefined_organic_media_channel,
        n_organic_rf_channels=len(organic_rf)
        + has_undefined_organic_rf_channel,
    )

    decay_spec = media | rf | organic_media | organic_rf
    model_spec = spec.ModelSpec(adstock_decay_spec=decay_spec)
    model_context = context.ModelContext(
        input_data=inp_data, model_spec=model_spec
    )

    if media:
      expected_media = list(media.values())

      if has_undefined_media_channel:
        expected_media.append(constants.GEOMETRIC_DECAY)
    elif has_undefined_media_channel:
      expected_media = [constants.GEOMETRIC_DECAY]
    else:
      expected_media = constants.GEOMETRIC_DECAY

    if rf:
      expected_rf = list(rf.values())

      if has_undefined_rf_channel:
        expected_rf.append(constants.GEOMETRIC_DECAY)
    elif has_undefined_rf_channel:
      expected_rf = [constants.GEOMETRIC_DECAY]
    else:
      expected_rf = constants.GEOMETRIC_DECAY

    if organic_media:
      expected_organic_media = list(organic_media.values())

      if has_undefined_organic_media_channel:
        expected_organic_media.append(constants.GEOMETRIC_DECAY)
    elif has_undefined_organic_media_channel:
      expected_organic_media = [constants.GEOMETRIC_DECAY]
    else:
      expected_organic_media = constants.GEOMETRIC_DECAY

    if organic_rf:
      expected_organic_rf = list(organic_rf.values())

      if has_undefined_organic_rf_channel:
        expected_organic_rf.append(constants.GEOMETRIC_DECAY)
    elif has_undefined_organic_rf_channel:
      expected_organic_rf = [constants.GEOMETRIC_DECAY]
    else:
      expected_organic_rf = constants.GEOMETRIC_DECAY

    self.assertSequenceEqual(
        model_context.adstock_decay_spec.media, expected_media
    )
    self.assertSequenceEqual(model_context.adstock_decay_spec.rf, expected_rf)
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_media, expected_organic_media
    )
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_rf, expected_organic_rf
    )

  @parameterized.product(**data_test_utils.ADSTOCK_DECAY_SPEC_CASES)
  def test_from_channel_explicit_media_name(
      self,
      media,
      rf,
      organic_media,
      organic_rf,
  ):
    """Test if one media channel has the name "media"."""

    if not (media or rf):
      self.skipTest("Invalid test case: Meridian requires paid media.")

    media = media | {"media": constants.BINOMIAL_DECAY}

    inp_data = data_test_utils.sample_input_data_revenue(
        n_media_channels=len(media),
        n_rf_channels=len(rf),
        n_organic_media_channels=len(organic_media),
        n_organic_rf_channels=len(organic_rf),
        explicit_media_channel_names=list(media.keys()),
    )

    decay_spec = media | rf | organic_media | organic_rf
    model_spec = spec.ModelSpec(adstock_decay_spec=decay_spec)
    model_context = context.ModelContext(
        input_data=inp_data, model_spec=model_spec
    )

    expected_media = list(media.values()) or constants.GEOMETRIC_DECAY
    expected_rf = list(rf.values()) or constants.GEOMETRIC_DECAY
    expected_organic_media = (
        list(organic_media.values()) or constants.GEOMETRIC_DECAY
    )
    expected_organic_rf = list(organic_rf.values()) or constants.GEOMETRIC_DECAY

    self.assertSequenceEqual(
        model_context.adstock_decay_spec.media, expected_media
    )
    self.assertSequenceEqual(model_context.adstock_decay_spec.rf, expected_rf)
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_media, expected_organic_media
    )
    self.assertSequenceEqual(
        model_context.adstock_decay_spec.organic_rf, expected_organic_rf
    )

  @parameterized.product(
      **data_test_utils.ADSTOCK_DECAY_SPEC_CASES,
      bad_channel=({"nonexistent_channel": constants.GEOMETRIC_DECAY},),
  )
  def test_from_channels_misnamed_channel_raises_error(
      self, media, rf, organic_media, organic_rf, bad_channel
  ):
    """Test if an exception is raised with an unrecognized channel."""
    if not (media or rf):
      self.skipTest("Invalid test case: Meridian requires paid media.")

    inp_data = data_test_utils.sample_input_data_revenue(
        n_media_channels=len(media),
        n_rf_channels=len(rf),
        n_organic_media_channels=len(organic_media),
        n_organic_rf_channels=len(organic_rf),
    )

    decay_spec = media | rf | organic_media | organic_rf | bad_channel
    model_spec = spec.ModelSpec(adstock_decay_spec=decay_spec)

    valid_channel_names = tuple(
        (media | rf | organic_media | organic_rf).keys()
    )

    model_context = context.ModelContext(
        input_data=inp_data, model_spec=model_spec
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Unrecognized channel names found in `adstock_decay_spec` keys "
        f"{tuple(decay_spec.keys())}. Keys should either contain only "
        f"channel_names {valid_channel_names} or be "
        "one or more of {'media', 'rf', 'organic_media', 'organic_rf'}.",
    ):
      _ = model_context.adstock_decay_spec

  def test_validate_media_spend_for_paid_media_channels_raises_error(self):
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_media_channels=len(_MEDIA_CHANNEL_NAMES),
    )
    spend = data_test_utils.random_media_spend_nd_da(
        n_media_channels=len(_MEDIA_CHANNEL_NAMES),
        explicit_media_channel_names=list(_MEDIA_CHANNEL_NAMES),
    )

    # Change spend values to zero for a single channel.
    channel_to_zero = _MEDIA_CHANNEL_NAMES[0]
    spend.loc[{constants.MEDIA_CHANNEL: channel_to_zero}] = 0
    data.media_spend = spend

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"Zero total spend detected for paid channels: {channel_to_zero}."
        " If data is correct and this is expected, please consider modeling"
        " the data as organic media.",
    ):
      context.ModelContext(
          input_data=data,
          model_spec=spec.ModelSpec(
              media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI
          ),
      )

  def test_validate_rf_spend_for_paid_channels_raises_error(self):
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_rf_channels=len(_RF_CHANNEL_NAMES),
    )
    spend = data_test_utils.random_rf_spend_nd_da(
        n_rf_channels=len(_RF_CHANNEL_NAMES),
    )

    # Change spend values to zero for a single channel.
    channel_to_zero = _RF_CHANNEL_NAMES[0]
    spend.loc[{constants.RF_CHANNEL: channel_to_zero}] = 0
    data.rf_spend = spend

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"Zero total spend detected for paid channels: {channel_to_zero}."
        " If data is correct and this is expected, please consider modeling"
        " the data as organic media.",
    ):
      context.ModelContext(
          input_data=data,
          model_spec=spec.ModelSpec(
              media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          ),
      )


class InferenceDataTest(
    test_utils.MeridianTestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_inference_data_non_paid_correct_dims(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec()
    model_context = context.ModelContext(
        input_data=data,
        model_spec=model_spec,
    )
    n_chains = 1
    n_draws = 7
    coords = model_context.create_inference_data_coords(n_chains, n_draws)
    dims = model_context.create_inference_data_dims()

    expected_coords_len = {
        constants.CHAIN: n_chains,
        constants.DRAW: n_draws,
        constants.GEO: model_context.n_geos,
        constants.TIME: model_context.n_times,
        constants.MEDIA_TIME: model_context.n_media_times,
        constants.KNOTS: model_context.knot_info.n_knots,
        constants.CONTROL_VARIABLE: model_context.n_controls,
        constants.NON_MEDIA_CHANNEL: model_context.n_non_media_channels,
        constants.MEDIA_CHANNEL: model_context.n_media_channels,
        constants.RF_CHANNEL: model_context.n_rf_channels,
        constants.ORGANIC_MEDIA_CHANNEL: model_context.n_organic_media_channels,
        constants.ORGANIC_RF_CHANNEL: model_context.n_organic_rf_channels,
    }
    actual_coords_len = {k: len(v) for k, v in coords.items()}

    with self.subTest("coords"):
      self.assertDictEqual(actual_coords_len, expected_coords_len)
    with self.subTest("dims"):
      self.assertEqual(dims[constants.SIGMA], ["chain", "draw"])

  def test_inference_data_with_unique_sigma_geo_correct_dims(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(unique_sigma_for_each_geo=True)
    model_context = context.ModelContext(
        input_data=data,
        model_spec=model_spec,
    )
    n_chains = 1
    n_draws = 7
    coords = model_context.create_inference_data_coords(n_chains, n_draws)
    dims = model_context.create_inference_data_dims()

    expected_coords_len = {
        constants.CHAIN: n_chains,
        constants.DRAW: n_draws,
        constants.GEO: model_context.n_geos,
        constants.TIME: model_context.n_times,
        constants.MEDIA_TIME: model_context.n_media_times,
        constants.KNOTS: model_context.knot_info.n_knots,
        constants.CONTROL_VARIABLE: model_context.n_controls,
        constants.NON_MEDIA_CHANNEL: model_context.n_non_media_channels,
        constants.MEDIA_CHANNEL: model_context.n_media_channels,
        constants.RF_CHANNEL: model_context.n_rf_channels,
        constants.ORGANIC_MEDIA_CHANNEL: model_context.n_organic_media_channels,
        constants.ORGANIC_RF_CHANNEL: model_context.n_organic_rf_channels,
    }
    actual_coords_len = {k: len(v) for k, v in coords.items()}

    with self.subTest("coords"):
      self.assertDictEqual(actual_coords_len, expected_coords_len)
    with self.subTest("dims"):
      self.assertEqual(dims[constants.SIGMA], ["chain", "draw", constants.GEO])


if __name__ == "__main__":
  absltest.main()
