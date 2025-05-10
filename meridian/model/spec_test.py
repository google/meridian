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

from absl.testing import absltest
from absl.testing import parameterized
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np


class ModelSpecTest(parameterized.TestCase):

  def test_spec_inits_with_default_params(self):
    model_spec = spec.ModelSpec()
    default_priors = prior_distribution.PriorDistribution()

    self.assertEqual(repr(model_spec.prior), repr(default_priors))
    self.assertEqual(model_spec.media_effects_dist, "log_normal")
    self.assertFalse(model_spec.hill_before_adstock)
    self.assertEqual(model_spec.max_lag, 8)
    self.assertFalse(model_spec.unique_sigma_for_each_geo)
    self.assertEqual(model_spec.effective_media_prior_type, "roi")
    self.assertEqual(model_spec.effective_rf_prior_type, "roi")
    self.assertIsNone(model_spec.roi_calibration_period)
    self.assertIsNone(model_spec.rf_roi_calibration_period)
    self.assertIsNone(model_spec.knots)
    self.assertIsNone(model_spec.baseline_geo)
    self.assertIsNone(model_spec.holdout_id)
    self.assertIsNone(model_spec.control_population_scaling_id)
    self.assertIsNone(model_spec.non_media_population_scaling_id)

  @parameterized.named_parameters(
      ("log_normal", "log_normal"),
      ("normal", "normal"),
  )
  def test_spec_inits_valid_media_effects_works(self, dist):
    model_spec = spec.ModelSpec(media_effects_dist=dist)
    self.assertEqual(model_spec.media_effects_dist, dist)

  @parameterized.named_parameters(
      (
          "empty",
          "",
          (
              "The `media_effects_dist` parameter '' must be one of"
              " ['log_normal', 'normal']."
          ),
      ),
      (
          "invalid",
          "invalid",
          (
              "The `media_effects_dist` parameter 'invalid' must be one of"
              " ['log_normal', 'normal']."
          ),
      ),
  )
  def test_spec_inits_invalid_media_effects_fails(self, dist, error_message):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(media_effects_dist=dist)

  @parameterized.named_parameters(
      ("roi_roi", "roi", "roi"),
      ("mroi_coefficient", "mroi", "coefficient"),
      ("coefficient_roi", "coefficient", "roi"),
  )
  def test_spec_inits_valid_prior_type_works(
      self, media_prior_type, rf_prior_type
  ):
    model_spec = spec.ModelSpec(
        media_prior_type=media_prior_type, rf_prior_type=rf_prior_type
    )
    self.assertEqual(model_spec.effective_media_prior_type, media_prior_type)
    self.assertEqual(model_spec.effective_rf_prior_type, rf_prior_type)

  @parameterized.named_parameters(
      (
          "empty",
          "",
          "roi",
          (
              "The `media_prior_type` parameter '' must be one of"
              " ['coefficient', 'contribution', 'mroi', 'roi']."
          ),
      ),
      (
          "invalid",
          "coefficient",
          "invalid",
          (
              "The `rf_prior_type` parameter 'invalid' must be one"
              " of ['coefficient', 'contribution', 'mroi', 'roi']."
          ),
      ),
  )
  def test_spec_inits_invalid_prior_type_fails(
      self, media_prior_type, rf_prior_type, error_message
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(
          media_prior_type=media_prior_type,
          rf_prior_type=rf_prior_type,
      )

  def test_spec_inits_valid_roi_calibration_works(self):
    shape = (3, 7)
    model_spec = spec.ModelSpec(
        roi_calibration_period=np.random.normal(size=shape)
    )
    self.assertIsNotNone(model_spec.roi_calibration_period)
    if model_spec.roi_calibration_period is not None:
      self.assertTupleEqual(model_spec.roi_calibration_period.shape, shape)

  @parameterized.named_parameters(
      (
          "1d",
          (14,),
          (
              "The shape of the `roi_calibration_period` array (14,) should be"
              " 2-dimensional (`n_media_times` x `n_media_channels`)."
          ),
      ),
      (
          "3d",
          (5, 10, 15),
          (
              "The shape of the `roi_calibration_period` array (5, 10, 15)"
              " should be 2-dimensional (`n_media_times` x `n_media_channels`)."
          ),
      ),
      (
          "4d",
          (2, 4, 3, 5),
          (
              "The shape of the `roi_calibration_period` array (2, 4, 3, 5)"
              " should be 2-dimensional (`n_media_times` x `n_media_channels`)."
          ),
      ),
  )
  def test_spec_inits_invalid_roi_calibration_fails(self, shape, error_message):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(roi_calibration_period=np.random.normal(size=shape))

  @parameterized.named_parameters(
      (
          "1d",
          (14,),
          (
              "The shape of the `rf_roi_calibration_period` array (14,) should"
              " be 2-dimensional (`n_media_times` x `n_rf_channels`)."
          ),
      ),
      (
          "3d",
          (5, 10, 15),
          (
              "The shape of the `rf_roi_calibration_period` array (5, 10, 15)"
              " should be 2-dimensional (`n_media_times` x `n_rf_channels`)."
          ),
      ),
      (
          "4d",
          (2, 4, 3, 5),
          (
              "The shape of the `rf_roi_calibration_period` array (2, 4, 3, 5)"
              " should be 2-dimensional (`n_media_times` x `n_rf_channels`)."
          ),
      ),
  )
  def test_spec_inits_invalid_rf_roi_calibration_fails(
      self, shape, error_message
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(rf_roi_calibration_period=np.random.normal(size=shape))

  def test_spec_inits_disallowed_roi_calibration_fails(self):
    shape = (3, 7)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `roi_calibration_period` should be `None` unless"
        " `media_prior_type` is 'roi'.",
    ):
      spec.ModelSpec(
          media_prior_type="mroi",
          roi_calibration_period=np.random.normal(size=shape),
      )

  def test_spec_inits_disallowed_rf_roi_calibration_fails(self):
    shape = (3, 7)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `rf_roi_calibration_period` should be `None` unless"
        " `rf_prior_type` is 'roi'.",
    ):
      spec.ModelSpec(
          rf_prior_type="coefficient",
          rf_roi_calibration_period=np.random.normal(size=shape),
      )

  @parameterized.named_parameters(
      (
          "zero",
          0,
          "The `knots` parameter cannot be zero.",
      ),
      (
          "empty_list",
          [],
          "The `knots` parameter cannot be an empty list.",
      ),
  )
  def test_spec_inits_empty_knots_fails(self, knots, error_message):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(knots=knots)

  def test_effective_media_prior_type_with_media_prior_type_set(self):
    """Tests effective_media_prior_type when media_prior_type is set."""
    model_spec = spec.ModelSpec(media_prior_type="mroi")
    self.assertEqual(model_spec.effective_media_prior_type, "mroi")

  def test_effective_media_prior_type_with_paid_media_prior_type_set(self):
    """Tests effective_media_prior_type when paid_media_prior_type is set."""
    warning_regex = (
        "Using `paid_media_prior_type` parameter will set prior types for media"
        " and RF at the same time. This is deprecated and will be removed in a"
        " future version of Meridian. Use `media_prior_type` and"
        " `rf_prior_type` instead."
    )
    with self.assertWarnsRegex(UserWarning, warning_regex):
      model_spec = spec.ModelSpec(
          media_prior_type=None, paid_media_prior_type="coefficient"
      )
      self.assertEqual(model_spec.effective_media_prior_type, "coefficient")

  def test_effective_media_prior_type_with_both_none(self):
    """Tests effective_media_prior_type when both are None."""
    model_spec = spec.ModelSpec(
        media_prior_type=None, paid_media_prior_type=None
    )
    self.assertEqual(model_spec.effective_media_prior_type, "roi")  # Default

  def test_effective_rf_prior_type_with_rf_prior_type_set(self):
    """Tests effective_rf_prior_type when rf_prior_type is set."""
    model_spec = spec.ModelSpec(rf_prior_type="coefficient")
    self.assertEqual(model_spec.effective_rf_prior_type, "coefficient")

  def test_effective_rf_prior_type_with_paid_media_prior_type_set(self):
    """Tests effective_rf_prior_type when paid_media_prior_type is set."""
    warning_regex = (
        "Using `paid_media_prior_type` parameter will set prior types for media"
        " and RF at the same time. This is deprecated and will be removed in a"
        " future version of Meridian. Use `media_prior_type` and"
        " `rf_prior_type` instead."
    )
    with self.assertWarnsRegex(UserWarning, warning_regex):
      model_spec = spec.ModelSpec(
          rf_prior_type=None, paid_media_prior_type="mroi"
      )
      self.assertEqual(model_spec.effective_rf_prior_type, "mroi")

  def test_effective_rf_prior_type_with_both_none(self):
    """Tests effective_rf_prior_type when both are None."""
    model_spec = spec.ModelSpec(rf_prior_type=None, paid_media_prior_type=None)
    self.assertEqual(model_spec.effective_rf_prior_type, "roi")  # Default

  def test_init_fails_with_paid_media_and_media_prior_types(self):
    """Tests ValueError if paid_media_prior_type and media_prior_type are set."""
    error_message = (
        "The deprecated `paid_media_prior_type` parameter cannot be used with"
        " `media_prior_type` or `rf_prior_type`. Use `media_prior_type` and"
        " `rf_prior_type` instead."
    )
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(
          paid_media_prior_type="roi", media_prior_type="coefficient"
      )

  def test_init_fails_with_paid_media_and_rf_prior_types(self):
    """Tests ValueError if paid_media_prior_type and rf_prior_type are set."""
    error_message = (
        "The deprecated `paid_media_prior_type` parameter cannot be used with"
        " `media_prior_type` or `rf_prior_type`. Use `media_prior_type` and"
        " `rf_prior_type` instead."
    )
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(paid_media_prior_type="roi", rf_prior_type="mroi")

  def test_init_warns_with_only_paid_media_prior_type(self):
    """Tests UserWarning if only paid_media_prior_type is set."""
    warning_message = (
        "Using `paid_media_prior_type` parameter will set prior types for media"
        " and RF at the same time. This is deprecated and will be removed in a"
        " future version of Meridian. Use `media_prior_type` and"
        " `rf_prior_type` instead."
    )
    with self.assertWarnsRegex(UserWarning, warning_message):
      spec.ModelSpec(paid_media_prior_type="roi")


if __name__ == "__main__":
  absltest.main()
