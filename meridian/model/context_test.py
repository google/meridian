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

from collections.abc import Collection
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import context
from meridian.model import knots as knots_module
from meridian.model import model_test_data
from meridian.model import spec
import numpy as np


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
              "The shape of `holdout_id` (2, 8) is different"
              " from `(n_times,)` = (200,)."
          ),
      ),
      dict(
          testcase_name="geo",
          input_data_type="geo",
          error_msg=(
              "The shape of `holdout_id` (2, 8) is different"
              " from `(n_geos, n_times)` = (5, 200)."
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
      _ = context.ModelContext(
          input_data=self.input_data_with_media_and_rf, model_spec=model_spec
      ).controls_scaled

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
    test_utils.assert_allequal(
        backend.to_tensor(data.controls, dtype=backend.float32),
        model_context.controls,
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
    else:
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

  def test_scaled_data_shape(self):
    controls = self.input_data_with_media_and_rf.controls
    data = self.input_data_with_media_and_rf
    model_context = context.ModelContext(
        input_data=data, model_spec=spec.ModelSpec()
    )
    self.assertIsNotNone(controls)
    test_utils.assert_allequal(
        model_context.controls_scaled.shape,  # pytype: disable=attribute-error
        controls.shape,
        err_msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
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

  def test_population_scaled_conrols_transformer_set(self):
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

  def test_scaled_data_inverse_is_identity(self):
    data = self.input_data_with_media_and_rf
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


if __name__ == "__main__":
  absltest.main()
