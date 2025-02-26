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

from collections.abc import Collection, Mapping, Sequence
import os
from unittest import mock
import warnings

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants
from meridian.data import input_data
from meridian.data import test_utils
from meridian.model import adstock_hill
from meridian.model import knots as knots_module
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


class ModelTest(
    tf.test.TestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  def setUp(self):
    super().setUp()
    model_test_data.WithInputDataSamples.setup(self)

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
      model.Meridian(input_data=data, model_spec=model_spec)

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
      model.Meridian(input_data=data, model_spec=model_spec)

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
    data = (
        self.national_input_data_media_and_rf
        if input_data_type == "national"
        else self.input_data_with_media_and_rf
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      _ = model.Meridian(input_data=data, model_spec=model_spec).holdout_id

  def test_init_with_wrong_control_population_scaling_id_shape_fails(self):
    model_spec = spec.ModelSpec(
        control_population_scaling_id=np.ones((7), dtype=bool)
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of `control_population_scaling_id` (7,) is different from"
        " `(n_controls,) = (2,)`.",
    ):
      _ = model.Meridian(
          input_data=self.input_data_with_media_and_rf, model_spec=model_spec
      ).controls_scaled

  @parameterized.named_parameters(
      ("none", None, 200), ("int", 3, 3), ("list", [0, 50, 100, 150], 4)
  )
  def test_n_knots(self, knots, expected_n_knots):
    # Create sample model spec with given knots
    model_spec = spec.ModelSpec(knots=knots)

    meridian = model.Meridian(
        input_data=self.input_data_with_media_only, model_spec=model_spec
    )

    self.assertEqual(meridian.knot_info.n_knots, expected_n_knots)

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
      _ = model.Meridian(
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
      _ = model.Meridian(
          input_data=data,
          model_spec=spec.ModelSpec(knots=knots),
      ).knot_info
      mock_get_knot_info.assert_called_once_with(
          self._N_TIMES, knots, is_national
      )

  def test_validate_paid_media_prior_type(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Custom priors should be set on `mroi_m` and `mroi_rf` when KPI is"
        " non-revenue and revenue per kpi data is missing.",
    ):
      model.Meridian(
          input_data=self.input_data_non_revenue_no_revenue_per_kpi,
          model_spec=spec.ModelSpec(
              paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_MROI
          ),
      )

  def test_custom_priors_not_passed_in_ok(self):
    meridian = model.Meridian(
        input_data=self.input_data_non_revenue_no_revenue_per_kpi,
        model_spec=spec.ModelSpec(
            paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_COEFFICIENT
        ),
    )
    # Compare input data.
    self.assertEqual(
        meridian.input_data, self.input_data_non_revenue_no_revenue_per_kpi
    )

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(
        paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_COEFFICIENT
    )

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_custom_priors_okay_with_array_params(self):
    my_prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal([1, 1], [1, 1])
    )
    meridian = model.Meridian(
        input_data=self.input_data_non_revenue_no_revenue_per_kpi,
        model_spec=spec.ModelSpec(prior=my_prior),
    )
    # Compare input data.
    self.assertEqual(
        meridian.input_data, self.input_data_non_revenue_no_revenue_per_kpi
    )

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(prior=my_prior)

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_get_knot_info_fails(self):
    error_msg = "Knots must be all non-negative."
    with mock.patch.object(
        knots_module,
        "get_knot_info",
        autospec=True,
        side_effect=ValueError(error_msg),
    ):
      with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
        _ = model.Meridian(
            input_data=self.input_data_with_media_only,
            model_spec=spec.ModelSpec(knots=4),
        ).knot_info

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
      _ = model.Meridian(
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
      _ = model.Meridian(
          input_data=self.national_input_data_media_only,
          model_spec=spec.ModelSpec(unique_sigma_for_each_geo=True),
      ).prior_broadcast
      # 7 warnings from the broadcasting (tau_g_excl_baseline, eta_m, eta_rf,
      # xi_c, eta_om, eta_orf, xi_n) + 2 from model spec.
      self.assertLen(w, 9)
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

  @parameterized.named_parameters(
      dict(
          testcase_name="custom_beta_m_prior_type_roi",
          custom_distributions={
              constants.BETA_M: tfp.distributions.LogNormal(
                  0.2, 0.8, name=constants.BETA_M
              )
          },
          ignored_priors="beta_m",
          paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="custom_mroi_m_mroi_rf_prior_type_roi",
          custom_distributions={
              constants.MROI_M: tfp.distributions.LogNormal(
                  0.2, 0.8, name=constants.MROI_M
              ),
              constants.MROI_RF: tfp.distributions.LogNormal(
                  0.2, 0.8, name=constants.MROI_RF
              ),
          },
          ignored_priors="mroi_m, mroi_rf",
          paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_ROI,
      ),
      dict(
          testcase_name="custom_beta_m_beta_rf_roi_m_prior_type_mroi",
          custom_distributions={
              constants.BETA_M: tfp.distributions.LogNormal(
                  0.7, 0.9, name=constants.ROI_M
              ),
              constants.BETA_RF: tfp.distributions.LogNormal(
                  0.8, 0.9, name=constants.ROI_RF
              ),
              constants.ROI_M: tfp.distributions.LogNormal(
                  0.2, 0.1, name=constants.ROI_M
              ),
          },
          ignored_priors="beta_m, beta_rf, roi_m",
          paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_MROI,
      ),
      dict(
          testcase_name="custom_roi_rf_prior_type_coefficient",
          custom_distributions={
              constants.ROI_RF: tfp.distributions.LogNormal(
                  0.2, 0.1, name=constants.ROI_RF
              )
          },
          ignored_priors="roi_rf",
          paid_media_prior_type=constants.PAID_MEDIA_PRIOR_TYPE_COEFFICIENT,
      ),
  )
  def test_warn_setting_ignored_priors(
      self,
      custom_distributions: Mapping[str, tfp.distributions.Distribution],
      ignored_priors: str,
      paid_media_prior_type: str,
  ):
    # Create prior distribution with given parameters.
    distribution = prior_distribution.PriorDistribution(**custom_distributions)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("module")
      model.Meridian(
          input_data=self.input_data_with_media_and_rf,
          model_spec=spec.ModelSpec(
              prior=distribution, paid_media_prior_type=paid_media_prior_type
          ),
      )
      self.assertTrue(
          any(
              f"Custom prior(s) `{ignored_priors}` are ignored when"
              " `paid_media_prior_type` is set to"
              f' "{paid_media_prior_type}".'
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
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_only",
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=input_data_samples._N_RF_CHANNELS
          ),
      ),
      dict(
          testcase_name="rf_and_media",
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
          ),
      ),
  )
  def test_input_data_tensor_properties(self, data):
    meridian = model.Meridian(input_data=data)
    self.assertAllEqual(
        tf.convert_to_tensor(data.kpi, dtype=tf.float32),
        meridian.kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.revenue_per_kpi, dtype=tf.float32),
        meridian.revenue_per_kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.controls, dtype=tf.float32),
        meridian.controls,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.population, dtype=tf.float32),
        meridian.population,
    )
    if data.media is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.media, dtype=tf.float32),
          meridian.media_tensors.media,
      )
    if data.media_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
          meridian.media_tensors.media_spend,
      )
    if data.reach is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.reach, dtype=tf.float32),
          meridian.rf_tensors.reach,
      )
    if data.frequency is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.frequency, dtype=tf.float32),
          meridian.rf_tensors.frequency,
      )
    if data.rf_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
          meridian.rf_tensors.rf_spend,
      )
    if data.media_spend is not None and data.rf_spend is not None:
      self.assertAllClose(
          tf.concat(
              [
                  tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
                  tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
              ],
              axis=-1,
          ),
          meridian.total_spend,
      )
    elif data.media_spend is not None:
      self.assertAllClose(
          tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
          meridian.total_spend,
      )
    else:
      self.assertAllClose(
          tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
          meridian.total_spend,
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
          data_name=constants.CONTROLS,
          dims_bad=np.array([b"control_0", b"control_1"]),
      ),
      dict(
          testcase_name="wrong_media",
          dataset=test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_MEDIA,
          data_name=constants.MEDIA,
          dims_bad=np.array([b"media_channel_1", b"media_channel_2"]),
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=test_utils.DATASET_WITHOUT_GEO_VARIATION_IN_REACH,
          data_name=constants.REACH,
          dims_bad=np.array([b"rf_channel_0", b"rf_channel_1"]),
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
      model.Meridian(
          input_data=test_utils.sample_input_data_from_dataset(
              dataset, kpi_type=constants.NON_REVENUE
          )
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_controls",
          dataset=test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_CONTROLS,
          data_name=constants.CONTROLS,
          dims_bad=np.array([b"control_0", b"control_1"]),
      ),
      dict(
          testcase_name="wrong_media",
          dataset=test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_MEDIA,
          data_name=constants.MEDIA,
          dims_bad=np.array([b"media_channel_1", b"media_channel_2"]),
      ),
      dict(
          testcase_name="wrong_rf",
          dataset=test_utils.DATASET_WITHOUT_TIME_VARIATION_IN_REACH,
          data_name=constants.REACH,
          dims_bad=np.array([b"rf_channel_0", b"rf_channel_1"]),
      ),
  )
  def test_init_without_time_variation_fails(
      self, dataset: xr.Dataset, data_name: str, dims_bad: Sequence[str]
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"The following {data_name} variables do not vary across time, making"
        f" a model with geo main effects unidentifiable: {dims_bad}. This can"
        " lead to poor model convergence. Since these variables only vary"
        " across geo and not across time, they are collinear with geo and"
        " redundant in a model with geo main effects. To address this, drop"
        " the listed variables that do not vary across time.",
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
        meridian.prior_broadcast.knot_values.batch_shape,
        (meridian.knot_info.n_knots,),
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
    meridian = model.Meridian(
        input_data=self.input_data_non_revenue_no_revenue_per_kpi
    )
    _ = meridian.prior_broadcast

    _, mock_kwargs = mock_broadcast.call_args
    self.assertEqual(mock_kwargs["set_total_media_contribution_prior"], True)
    self.assertEqual(mock_kwargs["kpi"], np.sum(meridian.input_data.kpi))
    np.testing.assert_allclose(mock_kwargs["total_spend"], expected_total_spend)

  def test_scaled_data_shape(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
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
    self.assertEqual(meridian.baseline_geo_idx, expected_idx)

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
      _ = model.Meridian(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(baseline_geo=baseline_geo),
      ).baseline_geo_idx

  def test_adstock_hill_media_missing_required_n_times_output(self):
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `media` has a number of time periods equal to `self.n_media_times`.",
    ):
      meridian = model.Meridian(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(),
      )
      meridian.adstock_hill_media(
          media=meridian.media_tensors.media[:, :-8, :],
          alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
      )

  def test_adstock_hill_media_n_times_output(self):
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autosepc=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = (
          self.input_data_with_media_only.media
      )
      meridian = model.Meridian(
          input_data=self.input_data_with_media_only,
          model_spec=spec.ModelSpec(),
      )
      meridian.adstock_hill_media(
          media=meridian.media_tensors.media,
          alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=np.ones(shape=(self._N_MEDIA_CHANNELS)),
          n_times_output=8,
      )

      calls = mock_adstock_cls.call_args_list
      _, mock_kwargs = calls[0]
      self.assertEqual(mock_kwargs["n_times_output"], 8)

  # TODO Move this test to a higher-level public API unit test.
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
        media=meridian.media_tensors.media,
        alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
    )

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_adstock_hill_rf_missing_required_n_times_output(self):
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `reach` has a number of time periods equal to `self.n_media_times`.",
    ):
      meridian = model.Meridian(
          input_data=self.input_data_with_media_and_rf,
          model_spec=spec.ModelSpec(),
      )
      meridian.adstock_hill_rf(
          reach=meridian.rf_tensors.reach[:, :-8, :],
          frequency=meridian.rf_tensors.frequency,
          alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
          ec=np.ones(shape=(self._N_RF_CHANNELS,)),
          slope=np.ones(shape=(self._N_RF_CHANNELS,)),
      )

  def test_adstock_hill_rf_n_times_output(self):
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autosepc=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = (
          self.input_data_with_media_and_rf.media
      )
      meridian = model.Meridian(
          input_data=self.input_data_with_media_and_rf,
          model_spec=spec.ModelSpec(),
      )
      meridian.adstock_hill_rf(
          reach=meridian.rf_tensors.reach,
          frequency=meridian.rf_tensors.frequency,
          alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
          ec=np.ones(shape=(self._N_RF_CHANNELS,)),
          slope=np.ones(shape=(self._N_RF_CHANNELS,)),
          n_times_output=8,
      )

      calls = mock_adstock_cls.call_args_list
      _, mock_kwargs = calls[0]
      self.assertEqual(mock_kwargs["n_times_output"], 8)

  # TODO Move this test to a higher-level public API unit test.
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
        reach=meridian.rf_tensors.reach,
        frequency=meridian.rf_tensors.frequency,
        alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
        ec=np.ones(shape=(self._N_RF_CHANNELS,)),
        slope=np.ones(shape=(self._N_RF_CHANNELS,)),
    )

    expected_called_names = ["mock_hill", "mock_adstock"]

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

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


class NonPaidModelTest(
    tf.test.TestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  def setUp(self):
    super().setUp()
    model_test_data.WithInputDataSamples.setup(self)

  def test_init_with_wrong_non_media_population_scaling_id_shape_fails(self):
    model_spec = spec.ModelSpec(
        non_media_population_scaling_id=np.ones((7), dtype=bool)
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of `non_media_population_scaling_id` (7,) is different from"
        " `(n_non_media_channels,) = (2,)`.",
    ):
      model.Meridian(
          input_data=self.input_data_non_media_and_organic,
          model_spec=model_spec,
      )

  def test_base_geo_properties(self):
    meridian = model.Meridian(input_data=self.input_data_non_media_and_organic)
    self.assertEqual(meridian.n_geos, self._N_GEOS)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_non_media_channels, self._N_NON_MEDIA_CHANNELS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertFalse(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  def test_base_national_properties(self):
    meridian = model.Meridian(
        input_data=self.national_input_data_non_media_and_organic
    )
    self.assertEqual(meridian.n_geos, self._N_GEOS_NATIONAL)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_non_media_channels, self._N_NON_MEDIA_CHANNELS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertTrue(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_non_media_and_organic",
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="rf_non_media_and_organic",
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="media_rf_non_media_and_organic",
          data=test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_input_data_tensor_properties(self, data):
    meridian = model.Meridian(input_data=data)
    self.assertAllEqual(
        tf.convert_to_tensor(data.kpi, dtype=tf.float32),
        meridian.kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.revenue_per_kpi, dtype=tf.float32),
        meridian.revenue_per_kpi,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.controls, dtype=tf.float32),
        meridian.controls,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.non_media_treatments, dtype=tf.float32),
        meridian.non_media_treatments,
    )
    self.assertAllEqual(
        tf.convert_to_tensor(data.population, dtype=tf.float32),
        meridian.population,
    )
    if data.media is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.media, dtype=tf.float32),
          meridian.media_tensors.media,
      )
    if data.media_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
          meridian.media_tensors.media_spend,
      )
    if data.reach is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.reach, dtype=tf.float32),
          meridian.rf_tensors.reach,
      )
    if data.frequency is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.frequency, dtype=tf.float32),
          meridian.rf_tensors.frequency,
      )
    if data.rf_spend is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
          meridian.rf_tensors.rf_spend,
      )
    if data.organic_media is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.organic_media, dtype=tf.float32),
          meridian.organic_media_tensors.organic_media,
      )
    if data.organic_reach is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.organic_reach, dtype=tf.float32),
          meridian.organic_rf_tensors.organic_reach,
      )
    if data.organic_frequency is not None:
      self.assertAllEqual(
          tf.convert_to_tensor(data.organic_frequency, dtype=tf.float32),
          meridian.organic_rf_tensors.organic_frequency,
      )
    if data.media_spend is not None and data.rf_spend is not None:
      self.assertAllClose(
          tf.concat(
              [
                  tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
                  tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
              ],
              axis=-1,
          ),
          meridian.total_spend,
      )
    elif data.media_spend is not None:
      self.assertAllClose(
          tf.convert_to_tensor(data.media_spend, dtype=tf.float32),
          meridian.total_spend,
      )
    else:
      self.assertAllClose(
          tf.convert_to_tensor(data.rf_spend, dtype=tf.float32),
          meridian.total_spend,
      )

  def test_broadcast_prior_distribution_is_called_in_meridian_init(self):
    meridian = model.Meridian(input_data=self.input_data_non_media_and_organic)
    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        meridian.prior_broadcast.tau_g_excl_baseline.batch_shape,
        (meridian.n_geos - 1,),
    )

    # Validate `n_knots` shape distributions.
    self.assertEqual(
        meridian.prior_broadcast.knot_values.batch_shape,
        (meridian.knot_info.n_knots,),
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

    # Validate `n_organic_media_channels` shape distributions.
    n_organic_media_channels_distributions_list = [
        meridian.prior_broadcast.beta_om,
        meridian.prior_broadcast.eta_om,
        meridian.prior_broadcast.alpha_om,
        meridian.prior_broadcast.ec_om,
        meridian.prior_broadcast.slope_om,
    ]
    for broad in n_organic_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_organic_media_channels,))

    # Validate `n_organic_rf_channels` shape distributions.
    n_organic_rf_channels_distributions_list = [
        meridian.prior_broadcast.beta_orf,
        meridian.prior_broadcast.eta_orf,
        meridian.prior_broadcast.alpha_orf,
        meridian.prior_broadcast.ec_orf,
        meridian.prior_broadcast.slope_orf,
    ]
    for broad in n_organic_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_organic_rf_channels,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        meridian.prior_broadcast.gamma_c,
        meridian.prior_broadcast.xi_c,
    ]
    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_controls,))

    # Validate `n_non_media_channels` shape distributions.
    n_non_media_distributions_list = [
        meridian.prior_broadcast.gamma_n,
        meridian.prior_broadcast.xi_n,
    ]
    for broad in n_non_media_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_non_media_channels,))

    # Validate sigma.
    self.assertEqual(meridian.prior_broadcast.sigma.batch_shape, (1,))

  def test_scaled_data_shape(self):
    meridian = model.Meridian(input_data=self.input_data_non_media_and_organic)
    self.assertAllEqual(
        meridian.controls_scaled.shape,
        self.input_data_non_media_and_organic.controls.shape,
        msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
    self.assertIsNotNone(meridian.non_media_treatments_scaled)
    self.assertIsNotNone(
        self.input_data_non_media_and_organic.non_media_treatments
    )
    # pytype: disable=attribute-error
    self.assertAllEqual(
        meridian.non_media_treatments_scaled.shape,
        self.input_data_non_media_and_organic.non_media_treatments.shape,
        msg=(
            "Shape of `_non_media_treatments_scaled` does not match the shape"
            " of `non_media_treatments` from the input data."
        ),
    )
    # pytype: enable=attribute-error
    self.assertAllEqual(
        meridian.kpi_scaled.shape,
        self.input_data_non_media_and_organic.kpi.shape,
        msg=(
            "Shape of `_kpi_scaled` does not match the shape of"
            " `kpi` from the input data."
        ),
    )

  def test_population_scaled_non_media_transformer_set(self):
    model_spec = spec.ModelSpec(
        non_media_population_scaling_id=tf.convert_to_tensor([
            True
            for _ in self.input_data_non_media_and_organic.non_media_channel
        ])
    )
    meridian = model.Meridian(
        input_data=self.input_data_non_media_and_organic, model_spec=model_spec
    )
    self.assertIsNotNone(meridian.non_media_transformer)
    # pytype: disable=attribute-error
    self.assertIsNotNone(
        meridian.non_media_transformer._population_scaling_factors,
        msg=(
            "`_population_scaling_factors` not set for the non-media"
            " transformer."
        ),
    )
    self.assertAllEqual(
        meridian.non_media_transformer._population_scaling_factors.shape,
        [
            len(self.input_data_non_media_and_organic.geo),
            len(self.input_data_non_media_and_organic.non_media_channel),
        ],
        msg=(
            "Shape of `non_media_transformer._population_scaling_factors` does"
            " not match (`n_geos`, `n_non_media_channels`)."
        ),
    )
    # pytype: enable=attribute-error

  def test_scaled_data_inverse_is_identity(self):
    meridian = model.Meridian(input_data=self.input_data_non_media_and_organic)

    # With the default tolerance of eps * 10 the test fails due to rounding
    # errors.
    atol = np.finfo(np.float32).eps * 100
    self.assertAllClose(
        meridian.controls_transformer.inverse(meridian.controls_scaled),
        self.input_data_non_media_and_organic.controls,
        atol=atol,
    )
    self.assertIsNotNone(meridian.non_media_transformer)
    # pytype: disable=attribute-error
    self.assertAllClose(
        meridian.non_media_transformer.inverse(
            meridian.non_media_treatments_scaled
        ),
        self.input_data_non_media_and_organic.non_media_treatments,
        atol=atol,
    )
    # pytype: enable=attribute-error
    self.assertAllClose(
        meridian.kpi_transformer.inverse(meridian.kpi_scaled),
        self.input_data_non_media_and_organic.kpi,
        atol=atol,
    )

  def test_get_joint_dist_zeros(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Deterministic(0),
            tau_g_excl_baseline=tfp.distributions.Deterministic(0),
            beta_m=tfp.distributions.Deterministic(0),
            beta_rf=tfp.distributions.Deterministic(0),
            beta_om=tfp.distributions.Deterministic(0),
            beta_orf=tfp.distributions.Deterministic(0),
            eta_m=tfp.distributions.Deterministic(0),
            eta_rf=tfp.distributions.Deterministic(0),
            eta_om=tfp.distributions.Deterministic(0),
            eta_orf=tfp.distributions.Deterministic(0),
            gamma_c=tfp.distributions.Deterministic(0),
            xi_c=tfp.distributions.Deterministic(0),
            gamma_n=tfp.distributions.Deterministic(0),
            xi_n=tfp.distributions.Deterministic(0),
            alpha_m=tfp.distributions.Deterministic(0),
            alpha_rf=tfp.distributions.Deterministic(0),
            alpha_om=tfp.distributions.Deterministic(0),
            alpha_orf=tfp.distributions.Deterministic(0),
            ec_m=tfp.distributions.Deterministic(0),
            ec_rf=tfp.distributions.Deterministic(0),
            ec_om=tfp.distributions.Deterministic(0),
            ec_orf=tfp.distributions.Deterministic(0),
            slope_m=tfp.distributions.Deterministic(0),
            slope_rf=tfp.distributions.Deterministic(0),
            slope_om=tfp.distributions.Deterministic(0),
            slope_orf=tfp.distributions.Deterministic(0),
            sigma=tfp.distributions.Deterministic(0),
            roi_m=tfp.distributions.Deterministic(0),
            roi_rf=tfp.distributions.Deterministic(0),
        ),
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_non_media,
        model_spec=model_spec,
    )
    sample = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(
            self._N_DRAWS
        )
    )
    self.assertAllEqual(
        sample.y,
        tf.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

  @parameterized.product(
      paid_media_prior_type=[
          constants.PAID_MEDIA_PRIOR_TYPE_ROI,
          constants.PAID_MEDIA_PRIOR_TYPE_MROI,
          constants.PAID_MEDIA_PRIOR_TYPE_COEFFICIENT,
      ],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_non_media(
      self, paid_media_prior_type: str, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        paid_media_prior_type=paid_media_prior_type,
        media_effects_dist=media_effects_dist,
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_non_media_and_organic,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(1)
    )
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) outcome
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) outcome data is "pinned" as "y".
    log_prob_parts_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist().log_prob_parts(
            par
        )
    )
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.BETA_GRF,
        constants.BETA_GOM,
        constants.BETA_GORF,
        constants.GAMMA_GC,
        constants.GAMMA_GN,
        constants.MU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.BETA_OM,
        constants.BETA_ORF,
        constants.ETA_M,
        constants.ETA_RF,
        constants.ETA_OM,
        constants.ETA_ORF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.GAMMA_N,
        constants.XI_N,
        constants.ALPHA_M,
        constants.ALPHA_RF,
        constants.ALPHA_OM,
        constants.ALPHA_ORF,
        constants.EC_M,
        constants.EC_RF,
        constants.EC_OM,
        constants.EC_ORF,
        constants.SLOPE_M,
        constants.SLOPE_RF,
        constants.SLOPE_OM,
        constants.SLOPE_ORF,
        constants.SIGMA,
    ]
    if paid_media_prior_type == constants.PAID_MEDIA_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.ROI_M)
      prior_distribution_params.append(constants.ROI_RF)
    elif paid_media_prior_type == constants.PAID_MEDIA_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.MROI_M)
      prior_distribution_params.append(constants.MROI_RF)
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
        constants.BETA_GOM_DEV,
        constants.BETA_GORF_DEV,
        constants.GAMMA_GC_DEV,
        constants.GAMMA_GN_DEV,
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
        media=meridian.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.rf_tensors.reach_scaled,
        frequency=meridian.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    transformed_organic_media = meridian.adstock_hill_media(
        media=meridian.organic_media_tensors.organic_media_scaled,
        alpha=par[constants.ALPHA_OM],
        ec=par[constants.EC_OM],
        slope=par[constants.SLOPE_OM],
    )[0, :, :, :]
    transformed_organic_reach = meridian.adstock_hill_rf(
        reach=meridian.organic_rf_tensors.organic_reach_scaled,
        frequency=meridian.organic_rf_tensors.organic_frequency,
        alpha=par[constants.ALPHA_ORF],
        ec=par[constants.EC_ORF],
        slope=par[constants.SLOPE_ORF],
    )[0, :, :, :]
    combined_transformed_media = tf.concat(
        [
            transformed_media,
            transformed_reach,
            transformed_organic_media,
            transformed_organic_reach,
        ],
        axis=-1,
    )

    combined_beta = tf.concat(
        [
            par[constants.BETA_GM][0, :, :],
            par[constants.BETA_GRF][0, :, :],
            par[constants.BETA_GOM][0, :, :],
            par[constants.BETA_GORF][0, :, :],
        ],
        axis=-1,
    )
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.MU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", combined_transformed_media, combined_beta)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
        + tf.einsum(
            "gtn,gn->gt",
            meridian.non_media_treatments_scaled,
            par[constants.GAMMA_GN][0, :, :],
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
        meridian.posterior_sampler_callable._get_joint_dist().log_prob(par)[0],
        rtol=1e-3,
    )

  def test_inference_data_non_paid_correct_dims(self):
    model_spec = spec.ModelSpec()
    mmm = model.Meridian(
        input_data=self.input_data_non_media_and_organic,
        model_spec=model_spec,
    )
    n_draws = 7
    prior_draws = mmm.prior_sampler_callable._sample_prior(n_draws, seed=1)
    # Create Arviz InferenceData for prior draws.
    prior_coords = mmm.create_inference_data_coords(1, n_draws)
    prior_dims = mmm.create_inference_data_dims()

    for param, tensor in prior_draws.items():
      self.assertIn(param, prior_dims)
      dims = prior_dims[param]
      self.assertEqual(len(dims), len(tensor.shape))
      for dim, shape_dim in zip(dims, tensor.shape):
        if dim != constants.SIGMA_DIM:
          self.assertIn(dim, prior_coords)
          self.assertLen(prior_coords[dim], shape_dim)

  def test_validate_injected_inference_data_correct_shapes(self):
    """Checks validation passes with correct shapes."""
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.input_data_non_media_and_organic,
        model_spec=model_spec,
    )
    n_chains = 1
    n_draws = 10
    prior_samples = meridian.prior_sampler_callable._sample_prior(n_draws)
    prior_coords = meridian.create_inference_data_coords(n_chains, n_draws)
    prior_dims = meridian.create_inference_data_dims()
    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    # This should not raise an error
    meridian_with_inference_data = model.Meridian(
        input_data=self.input_data_non_media_and_organic,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="non_media_channels",
          coord=constants.NON_MEDIA_CHANNEL,
          mismatched_priors={
              constants.GAMMA_GN: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
              constants.GAMMA_N: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
              constants.XI_N: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_NON_MEDIA_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_NON_MEDIA_CHANNELS,
      ),
      dict(
          testcase_name="organic_rf_channels",
          coord=constants.ORGANIC_RF_CHANNEL,
          mismatched_priors={
              constants.ALPHA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.BETA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.BETA_GORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.EC_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.ETA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.SLOPE_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_ORGANIC_RF_CHANNELS,
      ),
  )
  def test_validate_injected_inference_data_prior_incorrect_coordinates(
      self, coord, mismatched_priors, mismatched_coord_size, expected_coord_size
  ):
    """Checks validation fails with incorrect coordinates."""
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.input_data_non_media_and_organic,
        model_spec=model_spec,
    )
    prior_samples = meridian.prior_sampler_callable._sample_prior(self._N_DRAWS)
    prior_coords = meridian.create_inference_data_coords(1, self._N_DRAWS)
    prior_dims = meridian.create_inference_data_dims()

    prior_samples = dict(prior_samples)
    for param in mismatched_priors:
      prior_samples[param] = tf.zeros(mismatched_priors[param])
    prior_coords = dict(prior_coords)
    prior_coords[coord] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Injected inference data prior has incorrect coordinate"
        f" '{coord}': expected"
        f" {expected_coord_size}, got"
        f" {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=self.input_data_non_media_and_organic,
          model_spec=model_spec,
          inference_data=inference_data,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="sigma_dims_unique_sigma",
          coord=constants.GEO,
          mismatched_priors={
              constants.BETA_GOM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              ),
              constants.BETA_GORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_RF_CHANNELS,
              ),
              constants.GAMMA_GN: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_NON_MEDIA_CHANNELS,
              ),
              constants.GAMMA_GC: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G_EXCL_BASELINE: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
              ),
              constants.BETA_GM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_MEDIA_CHANNELS,
              ),
              constants.BETA_GRF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_RF_CHANNELS,
              ),
              constants.BETA_GOM_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              ),
              constants.BETA_GORF_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_RF_CHANNELS,
              ),
              constants.GAMMA_GN_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_NON_MEDIA_CHANNELS,
              ),
              constants.GAMMA_GC_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.SIGMA: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_GEOS + 1,
          expected_coord_size=input_data_samples._N_GEOS,
          unique_sigma=True,
      ),
      dict(
          testcase_name="sigma_dims_not_unique_sigma",
          coord=constants.SIGMA_DIM,
          mismatched_priors={
              constants.SIGMA: (
                  1,
                  input_data_samples._N_DRAWS,
                  2,
              ),
          },
          mismatched_coord_size=2,
          expected_coord_size=1,
          unique_sigma=False,
      ),
  )
  def test_validate_injected_inference_data_prior_incorrect_sigma_coordinates(
      self,
      coord,
      mismatched_priors,
      mismatched_coord_size,
      expected_coord_size,
      unique_sigma,
  ):
    """Checks validation fails with incorrect coordinates for sigma."""
    model_spec = spec.ModelSpec(unique_sigma_for_each_geo=unique_sigma)
    meridian = model.Meridian(
        input_data=self.input_data_non_media_and_organic,
        model_spec=model_spec,
    )
    prior_samples = meridian.prior_sampler_callable._sample_prior(self._N_DRAWS)
    prior_coords = meridian.create_inference_data_coords(1, self._N_DRAWS)
    prior_dims = meridian.create_inference_data_dims()

    prior_samples = dict(prior_samples)
    for param in mismatched_priors:
      prior_samples[param] = tf.zeros(mismatched_priors[param])
    prior_coords = dict(prior_coords)
    prior_coords[coord] = np.arange(mismatched_coord_size)
    if unique_sigma:
      prior_coords[constants.GEO] = np.arange(mismatched_coord_size)
    else:
      prior_coords[constants.SIGMA_DIM] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Injected inference data prior has incorrect coordinate"
        f" '{coord}': expected"
        f" {expected_coord_size}, got"
        f" {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=self.input_data_non_media_and_organic,
          model_spec=model_spec,
          inference_data=inference_data,
      )


if __name__ == "__main__":
  absltest.main()
