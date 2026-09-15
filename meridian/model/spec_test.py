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

import datetime
import types
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model.calibration import base as calibration_base
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
    self.assertEqual(model_spec.organic_media_prior_type, "contribution")
    self.assertEqual(model_spec.organic_rf_prior_type, "contribution")
    self.assertEqual(model_spec.non_media_treatments_prior_type, "contribution")
    self.assertIsNone(model_spec.roi_calibration)
    self.assertIsNone(model_spec.roi_calibration_period)
    self.assertIsNone(model_spec.rf_roi_calibration)
    self.assertIsNone(model_spec.rf_roi_calibration_period)
    self.assertIsNone(model_spec.knots)
    self.assertIsNone(model_spec.baseline_geo)
    self.assertIsNone(model_spec.holdout)
    self.assertIsNone(model_spec.holdout_id)
    self.assertIsNone(model_spec.population_scaled_controls)
    self.assertIsNone(model_spec.control_population_scaling_id)
    self.assertIsNone(model_spec.population_scaled_non_media_channels)
    self.assertIsNone(model_spec.non_media_population_scaling_id)
    self.assertIsNone(model_spec.non_media_baseline_values)

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
      ("hill", constants.HILL),
      ("none", "none"),
  )
  def test_spec_inits_valid_saturation_spec_works(self, saturation):
    model_spec = spec.ModelSpec(saturation_spec=saturation)
    self.assertEqual(model_spec.saturation_spec, saturation)

  def test_spec_inits_valid_saturation_spec_mapping_works(self):
    saturation_mapping = {"ch1": constants.HILL, "ch2": "none"}
    model_spec = spec.ModelSpec(saturation_spec=saturation_mapping)
    self.assertEqual(model_spec.saturation_spec, saturation_mapping)

  def test_spec_inits_invalid_saturation_spec_str_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `saturation_spec` parameter 'invalid' must be one of ['hill',"
        " 'none'].",
    ):
      spec.ModelSpec(saturation_spec="invalid")

  def test_spec_inits_invalid_saturation_spec_mapping_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `saturation_spec` for channel 'ch1' must be one of ['hill',"
        " 'none'], but got 'invalid'.",
    ):
      spec.ModelSpec(saturation_spec={"ch1": "invalid"})

  def test_spec_inits_unsupported_saturation_spec_type_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Unsupported type for `saturation_spec` parameter: <class 'int'>",
    ):
      spec.ModelSpec(saturation_spec=123)  # pytype: disable=wrong-arg-types

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          media_prior_type="roi",
          rf_prior_type="roi",
          organic_media_prior_type="contribution",
          organic_rf_prior_type="contribution",
          non_media_treatments_prior_type="contribution",
      ),
      dict(
          testcase_name="mixed1",
          media_prior_type="mroi",
          rf_prior_type="coefficient",
          organic_media_prior_type="coefficient",
          organic_rf_prior_type="contribution",
          non_media_treatments_prior_type="coefficient",
      ),
      dict(
          testcase_name="mixed2",
          media_prior_type="coefficient",
          rf_prior_type="contribution",
          organic_media_prior_type="contribution",
          organic_rf_prior_type="coefficient",
          non_media_treatments_prior_type="contribution",
      ),
      dict(
          testcase_name="mixed3",
          media_prior_type="contribution",
          rf_prior_type="mroi",
          organic_media_prior_type="coefficient",
          organic_rf_prior_type="coefficient",
          non_media_treatments_prior_type="contribution",
      ),
  )
  def test_spec_inits_valid_prior_type_works(
      self,
      media_prior_type: str,
      rf_prior_type: str,
      organic_media_prior_type: str,
      organic_rf_prior_type: str,
      non_media_treatments_prior_type: str,
  ):
    model_spec = spec.ModelSpec(
        media_prior_type=media_prior_type,
        rf_prior_type=rf_prior_type,
        organic_media_prior_type=organic_media_prior_type,
        organic_rf_prior_type=organic_rf_prior_type,
        non_media_treatments_prior_type=non_media_treatments_prior_type,
    )
    self.assertEqual(model_spec.effective_media_prior_type, media_prior_type)
    self.assertEqual(model_spec.effective_rf_prior_type, rf_prior_type)
    self.assertEqual(
        model_spec.organic_media_prior_type, organic_media_prior_type
    )
    self.assertEqual(model_spec.organic_rf_prior_type, organic_rf_prior_type)
    self.assertEqual(
        model_spec.non_media_treatments_prior_type,
        non_media_treatments_prior_type,
    )

  @parameterized.named_parameters(
      (
          "empty",
          "",
          "roi",
          "coefficient",
          "contribution",
          "coefficient",
          (
              "The `media_prior_type` parameter '' must be one of"
              " ['coefficient', 'contribution', 'mroi', 'roi']."
          ),
      ),
      (
          "invalid",
          "coefficient",
          "invalid",
          "contribution",
          "coefficient",
          "contribution",
          (
              "The `rf_prior_type` parameter 'invalid' must be one"
              " of ['coefficient', 'contribution', 'mroi', 'roi']."
          ),
      ),
      (
          "roi_organic_media",
          "coefficient",
          "coefficient",
          "roi",
          "coefficient",
          "coefficient",
          (
              "The `organic_media_prior_type` parameter 'roi' must be one"
              " of ['coefficient', 'contribution']."
          ),
      ),
      (
          "mroi_organic_rf",
          "roi",
          "mroi",
          "coefficient",
          "mroi",
          "coefficient",
          (
              "The `organic_rf_prior_type` parameter 'mroi' must be one"
              " of ['coefficient', 'contribution']."
          ),
      ),
      (
          "contribution_non_media_treatments",
          "roi",
          "roi",
          "contribution",
          "coefficient",
          "roi",
          (
              "The `non_media_treatments_prior_type` parameter 'roi'"
              " must be one of ['coefficient', 'contribution']."
          ),
      ),
  )
  def test_spec_inits_invalid_prior_type_fails(
      self,
      media_prior_type: str,
      rf_prior_type: str,
      organic_media_prior_type: str,
      organic_rf_prior_type: str,
      non_media_treatments_prior_type: str,
      error_message,
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(
          media_prior_type=media_prior_type,
          rf_prior_type=rf_prior_type,
          organic_media_prior_type=organic_media_prior_type,
          organic_rf_prior_type=organic_rf_prior_type,
          non_media_treatments_prior_type=non_media_treatments_prior_type,
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

  def test_spec_inits_knots_and_aks_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `knots` parameter cannot be set when `enable_aks` is True.",
    ):
      spec.ModelSpec(knots=10, enable_aks=True)

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

  @parameterized.named_parameters(
      dict(testcase_name="media", media_prior_type="coefficient"),
      dict(testcase_name="rf", rf_prior_type="coefficient"),
      dict(
          testcase_name="organic_media", organic_media_prior_type="coefficient"
      ),
      dict(testcase_name="organic_rf", organic_rf_prior_type="coefficient"),
      dict(
          testcase_name="non_media",
          non_media_treatments_prior_type="coefficient",
      ),
  )
  def test_init_warns_with_coefficient_prior_type(self, **kwargs):
    """Tests UserWarning if coefficient prior type is used."""
    warning_message = (
        r"Using coefficient priors \(`coefficient`\) is not recommended\."
    )
    with self.assertWarnsRegex(UserWarning, warning_message):
      spec.ModelSpec(**kwargs)

  @parameterized.named_parameters(
      ("ndarray", np.array([2, 5, 8], dtype=int), [2, 5, 8]),
      ("tuple", (2, 5, 8), [2, 5, 8]),
      ("set", {2, 5, 8}, [2, 5, 8]),
      ("list", [2, 5, 8], [2, 5, 8]),
      ("dict_keys", {2: "a", 5: "b", 8: "c"}, [2, 5, 8]),
  )
  def test_spec_inits_knots_with_collection_converts_to_list(
      self, knots_input, expected
  ):
    """Tests that passing any collection for knots converts it to a list[int]."""
    model_spec = spec.ModelSpec(knots=knots_input)

    self.assertIsInstance(model_spec.knots, list)
    self.assertCountEqual(model_spec.knots, expected)

  @parameterized.named_parameters(
      ("strings_list", ["a", "b"]),
      ("strings_tuple", ("a", "b")),
      ("floats_list", [1.1, 2.2]),
      ("mixed_tuple", (1, "a")),
  )
  def test_spec_inits_knots_with_non_integers_fails(self, knots_input):
    """Tests that collections containing non-integers raise ValueError."""
    with self.assertRaisesRegex(
        ValueError, "`knots` must be a sequence of integers"
    ):
      spec.ModelSpec(knots=knots_input)

  def test_spec_inits_knots_with_unsupported_type_fails(self):
    """Tests that passing an unsupported type (e.g. dict) raises ValueError."""
    with self.assertRaisesRegex(
        ValueError, "Unsupported type for `knots` parameter"
    ):
      spec.ModelSpec(knots=3.5)  # pytype: disable=wrong-arg-types

  @parameterized.named_parameters(
      ("geometric", constants.GEOMETRIC_DECAY),
      ("binomial", constants.BINOMIAL_DECAY),
      (
          "mapping",
          {
              "ch1": constants.GEOMETRIC_DECAY,
              "ch2": constants.BINOMIAL_DECAY,
          },
      ),
  )
  def test_spec_inits_valid_adstock_decay_spec_works(self, decay_spec):
    model_spec = spec.ModelSpec(adstock_decay_spec=decay_spec)
    self.assertEqual(model_spec.adstock_decay_spec, decay_spec)

  @parameterized.named_parameters(
      (
          "string",
          "invalid",
          (
              "The `adstock_decay_spec` parameter 'invalid' must be one of"
              " ['binomial', 'geometric']."
          ),
      ),
      (
          "mapping",
          {"ch1": "invalid"},
          (
              "The `adstock_decay_spec` for channel 'ch1' must be one of"
              " ['binomial', 'geometric'], but got 'invalid'."
          ),
      ),
  )
  def test_spec_inits_invalid_adstock_decay_spec_fails(
      self, adstock_decay_spec, error_message
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(adstock_decay_spec=adstock_decay_spec)

  def test_spec_inits_unsupported_adstock_decay_spec_type_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Unsupported type for `adstock_decay_spec` parameter: <class 'int'>",
    ):
      spec.ModelSpec(adstock_decay_spec=123)  # pyrefly: ignore[bad-argument-type]

  @parameterized.named_parameters(
      ("negative", -1),
      ("boolean", True),
      ("float", 3.5),
  )
  def test_spec_inits_invalid_max_lag_fails(self, max_lag):
    with self.assertRaisesRegex(
        ValueError, r"'max_lag' must be a non-negative integer\."
    ):
      spec.ModelSpec(max_lag=max_lag)  # pyrefly: ignore[bad-argument-type]

  @parameterized.named_parameters(
      (
          "string_decay",
          constants.GEOMETRIC_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          8,
      ),
      (
          "mapping_decay",
          constants.BINOMIAL_DECAY,
          4,
          {"Search": constants.BINOMIAL_DECAY},
          4,
      ),
      (
          "unmentioned_channel_defaults_to_geometric",
          constants.GEOMETRIC_DECAY,
          8,
          {"Other": constants.BINOMIAL_DECAY},
          8,
      ),
      (
          "mapping_prior",
          constants.GEOMETRIC_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          8,
          "mapping",
      ),
      (
          "object_prior",
          constants.GEOMETRIC_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          8,
          "object",
      ),
      (
          "partially_calibrated",
          constants.GEOMETRIC_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          8,
          "dataclass",
          True,
      ),
  )
  def test_spec_inits_matching_calibrated_prior_works(
      self,
      cal_adstock_decay_spec,
      cal_max_lag,
      spec_adstock_decay_spec,
      spec_max_lag,
      prior_type: str = "dataclass",
      has_uncalibrated_channel: bool = False,
  ):
    cal_output = calibration_base.CalibrationOutput(
        channel_name="Search",
        intermediary_prior=backend.tfd.Normal(0.0, 1.0),
        adstock_decay_spec=cal_adstock_decay_spec,
        max_lag=cal_max_lag,
    )
    if has_uncalibrated_channel:
      distributions = [
          backend.tfd.Normal(0.1, 0.5),
          backend.tfd.Normal(0.2, 0.9),
      ]
      is_calibrated = [False, True]
      calibration_outputs = [None, cal_output]
    else:
      distributions = [backend.tfd.Normal(0.2, 0.9)]
      is_calibrated = [True]
      calibration_outputs = [cal_output]

    roi_dist = calibration_base.CalibratedDistribution(
        distributions=distributions,
        is_calibrated=is_calibrated,
        calibration_outputs=calibration_outputs,
    )
    prior: Any
    if prior_type == "mapping":
      prior = {constants.ROI_M: roi_dist}
    elif prior_type == "object":
      prior = types.SimpleNamespace(roi_m=roi_dist)
    else:
      prior = prior_distribution.PriorDistribution(roi_m=roi_dist)

    model_spec = spec.ModelSpec(
        prior=prior,  # pyrefly: ignore[bad-argument-type]
        max_lag=spec_max_lag,
        adstock_decay_spec=spec_adstock_decay_spec,
    )
    self.assertEqual(model_spec.max_lag, spec_max_lag)
    self.assertEqual(model_spec.adstock_decay_spec, spec_adstock_decay_spec)

  @parameterized.named_parameters(
      (
          "mismatched_max_lag",
          "Search",
          constants.ROI_M,
          constants.GEOMETRIC_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          4,
          (
              "The `max_lag` for calibrated channel 'Search' (8) does not"
              " match the ModelSpec `max_lag` (4). `max_lag` is used to"
              " calculate the duration adjustment during prior calibration. To"
              " fix this, set `ModelSpec(max_lag=...)` to match the value"
              " used during prior calibration, or recalibrate the prior using"
              " the desired `max_lag`."
          ),
      ),
      (
          "mismatched_adstock_decay_spec_str",
          "Search",
          constants.ROI_M,
          constants.GEOMETRIC_DECAY,
          8,
          constants.BINOMIAL_DECAY,
          8,
          (
              "The `adstock_decay_spec` for calibrated channel 'Search'"
              " ('geometric') does not match the ModelSpec `adstock_decay_spec`"
              " ('binomial'). `adstock_decay_spec` is used to calculate the"
              " duration adjustment during prior calibration. To fix this, set"
              " `ModelSpec(adstock_decay_spec=...)` to match the value used"
              " during prior calibration, or recalibrate the prior using the"
              " desired `adstock_decay_spec`."
          ),
      ),
      (
          "mismatched_adstock_decay_spec_mapping",
          "Search",
          constants.ROI_M,
          constants.GEOMETRIC_DECAY,
          8,
          {"Search": constants.BINOMIAL_DECAY},
          8,
          (
              "The `adstock_decay_spec` for calibrated channel 'Search'"
              " ('geometric') does not match the ModelSpec `adstock_decay_spec`"
              " ('binomial'). `adstock_decay_spec` is used to calculate the"
              " duration adjustment during prior calibration. To fix this, set"
              " `ModelSpec(adstock_decay_spec=...)` to match the value used"
              " during prior calibration, or recalibrate the prior using the"
              " desired `adstock_decay_spec`."
          ),
      ),
      (
          "mismatched_rf_channel",
          "YouTube_RF",
          constants.ROI_RF,
          constants.BINOMIAL_DECAY,
          8,
          constants.GEOMETRIC_DECAY,
          8,
          (
              "The `adstock_decay_spec` for calibrated channel 'YouTube_RF'"
              " ('binomial') does not match the ModelSpec `adstock_decay_spec`"
              " ('geometric'). `adstock_decay_spec` is used to calculate the"
              " duration adjustment during prior calibration. To fix this, set"
              " `ModelSpec(adstock_decay_spec=...)` to match the value used"
              " during prior calibration, or recalibrate the prior using the"
              " desired `adstock_decay_spec`."
          ),
      ),
      (
          "mismatched_adstock_decay_spec_unmentioned_in_mapping",
          "Search",
          constants.ROI_M,
          constants.BINOMIAL_DECAY,
          8,
          {"Other": constants.BINOMIAL_DECAY},
          8,
          (
              "The `adstock_decay_spec` for calibrated channel 'Search'"
              " ('binomial') does not match the ModelSpec `adstock_decay_spec`"
              " ('geometric'). `adstock_decay_spec` is used to calculate the"
              " duration adjustment during prior calibration. To fix this, set"
              " `ModelSpec(adstock_decay_spec=...)` to match the value used"
              " during prior calibration, or recalibrate the prior using the"
              " desired `adstock_decay_spec`."
          ),
      ),
  )
  def test_spec_inits_mismatched_calibrated_prior_fails(
      self,
      channel_name,
      prior_attr,
      cal_adstock_decay_spec,
      cal_max_lag,
      spec_adstock_decay_spec,
      spec_max_lag,
      error_message,
  ):
    cal_output = calibration_base.CalibrationOutput(
        channel_name=channel_name,
        intermediary_prior=backend.tfd.Normal(0.0, 1.0),
        adstock_decay_spec=cal_adstock_decay_spec,
        max_lag=cal_max_lag,
    )
    roi_dist = calibration_base.CalibratedDistribution(
        distributions=[backend.tfd.Normal(0.2, 0.9)],
        is_calibrated=[True],
        calibration_outputs=[cal_output],
    )
    prior = prior_distribution.PriorDistribution(**{prior_attr: roi_dist})
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      spec.ModelSpec(
          prior=prior,
          max_lag=spec_max_lag,
          adstock_decay_spec=spec_adstock_decay_spec,
      )

  def test_spec_inits_valid_roi_calibration_works(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    cal_spec = spec.CalibrationSpec([dr])
    model_spec = spec.ModelSpec(
        media_prior_type="roi",
        roi_calibration=cal_spec,
    )
    self.assertEqual(model_spec.roi_calibration, cal_spec)

  def test_spec_inits_disallowed_roi_calibration_fails(self):
    cal_spec = spec.CalibrationSpec(
        [spec.DateRange("2021-01-01", "2021-01-31")]
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `roi_calibration` should be `None` unless `media_prior_type` is"
        " 'roi'.",
    ):
      spec.ModelSpec(
          media_prior_type="mroi",
          roi_calibration=cal_spec,
      )

  def test_spec_inits_both_roi_calibration_and_period_warns(self):
    cal_spec = spec.CalibrationSpec(
        [spec.DateRange("2021-01-01", "2021-01-31")]
    )
    period = np.ones((2, 2))
    with self.assertWarnsRegex(
        UserWarning,
        "Both `roi_calibration` and the deprecated `roi_calibration_period`"
        " were specified. `roi_calibration_period` takes precedence for"
        " backward compatibility; a future version of Meridian will ignore"
        " it in favor of `roi_calibration`.",
    ):
      model_spec = spec.ModelSpec(
          media_prior_type="roi",
          roi_calibration=cal_spec,
          roi_calibration_period=period,
      )
    # Both are retained: the legacy field still drives behavior until the
    # compilation layer consumes `roi_calibration`.
    self.assertEqual(model_spec.roi_calibration, cal_spec)
    np.testing.assert_array_equal(model_spec.roi_calibration_period, period)

  def test_spec_inits_legacy_roi_calibration_period_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`roi_calibration_period` is deprecated and will be removed in a"
        " future version of Meridian. Use `roi_calibration` instead.",
    ):
      spec.ModelSpec(roi_calibration_period=np.ones((2, 2)))

  def test_spec_inits_valid_rf_roi_calibration_works(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    cal_spec = spec.CalibrationSpec([dr])
    model_spec = spec.ModelSpec(
        rf_prior_type="roi",
        rf_roi_calibration=cal_spec,
    )
    self.assertEqual(model_spec.rf_roi_calibration, cal_spec)

  def test_spec_inits_disallowed_rf_roi_calibration_fails(self):
    cal_spec = spec.CalibrationSpec(
        [spec.DateRange("2021-01-01", "2021-01-31")]
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The `rf_roi_calibration` should be `None` unless `rf_prior_type` is"
        " 'roi'.",
    ):
      spec.ModelSpec(
          rf_prior_type="mroi",
          rf_roi_calibration=cal_spec,
      )

  def test_spec_inits_both_rf_roi_calibration_and_period_warns(self):
    cal_spec = spec.CalibrationSpec(
        [spec.DateRange("2021-01-01", "2021-01-31")]
    )
    period = np.ones((2, 2))
    with self.assertWarnsRegex(
        UserWarning,
        "Both `rf_roi_calibration` and the deprecated"
        " `rf_roi_calibration_period` were specified."
        " `rf_roi_calibration_period` takes precedence for backward"
        " compatibility; a future version of Meridian will ignore it in"
        " favor of `rf_roi_calibration`.",
    ):
      model_spec = spec.ModelSpec(
          rf_prior_type="roi",
          rf_roi_calibration=cal_spec,
          rf_roi_calibration_period=period,
      )
    self.assertEqual(model_spec.rf_roi_calibration, cal_spec)
    np.testing.assert_array_equal(model_spec.rf_roi_calibration_period, period)

  def test_spec_inits_legacy_rf_roi_calibration_period_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`rf_roi_calibration_period` is deprecated and will be removed in a"
        " future version of Meridian. Use `rf_roi_calibration` instead.",
    ):
      spec.ModelSpec(rf_roi_calibration_period=np.ones((2, 2)))

  def test_spec_inits_valid_holdout_works(self):
    h_spec = spec.HoldoutSpec(spec.RandomHoldoutSpec(0.2, seed=42))
    model_spec = spec.ModelSpec(holdout=h_spec)
    self.assertEqual(model_spec.holdout, h_spec)

  def test_spec_inits_both_holdout_and_holdout_id_warns(self):
    h_spec = spec.HoldoutSpec(spec.RandomHoldoutSpec(0.2))
    holdout_id = np.ones((2, 2))
    with self.assertWarnsRegex(
        UserWarning,
        "Both `holdout` and the deprecated `holdout_id` were specified."
        " `holdout_id` takes precedence for backward compatibility; a future"
        " version of Meridian will ignore it in favor of `holdout`.",
    ):
      model_spec = spec.ModelSpec(holdout=h_spec, holdout_id=holdout_id)
    self.assertEqual(model_spec.holdout, h_spec)
    np.testing.assert_array_equal(model_spec.holdout_id, holdout_id)

  def test_spec_inits_legacy_holdout_id_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`holdout_id` is deprecated and will be removed in a future version"
        " of Meridian. Use `holdout` instead.",
    ):
      spec.ModelSpec(holdout_id=np.ones((2, 2)))

  def test_spec_inits_valid_population_scaled_controls_works(self):
    model_spec = spec.ModelSpec(population_scaled_controls=["c1", "c2"])
    self.assertEqual(model_spec.population_scaled_controls, ("c1", "c2"))

  def test_spec_inits_both_population_scaled_controls_and_legacy_warns(self):
    scaling_id = np.ones((1,))
    with self.assertWarnsRegex(
        UserWarning,
        "Both `population_scaled_controls` and the deprecated"
        " `control_population_scaling_id` were specified."
        " `control_population_scaling_id` takes precedence for backward"
        " compatibility; a future version of Meridian will ignore it in"
        " favor of `population_scaled_controls`.",
    ):
      model_spec = spec.ModelSpec(
          population_scaled_controls=["c1"],
          control_population_scaling_id=scaling_id,
      )
    self.assertEqual(model_spec.population_scaled_controls, ("c1",))
    np.testing.assert_array_equal(
        model_spec.control_population_scaling_id, scaling_id
    )

  def test_spec_inits_legacy_control_population_scaling_id_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`control_population_scaling_id` is deprecated and will be removed"
        " in a future version of Meridian. Use `population_scaled_controls`"
        " instead.",
    ):
      spec.ModelSpec(control_population_scaling_id=np.ones((1,)))

  def test_spec_inits_population_scaled_controls_single_string_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`population_scaled_controls` must be a sequence of strings, not a"
        " single string.",
    ):
      spec.ModelSpec(population_scaled_controls="c1")  # pyrefly: ignore[bad-argument-type]

  def test_spec_inits_valid_population_scaled_non_media_channels_works(self):
    model_spec = spec.ModelSpec(
        population_scaled_non_media_channels=["nm1", "nm2"]
    )
    self.assertEqual(
        model_spec.population_scaled_non_media_channels, ("nm1", "nm2")
    )

  def test_spec_inits_both_population_scaled_non_media_channels_and_legacy_warns(
      self,
  ):
    scaling_id = np.ones((1,))
    with self.assertWarnsRegex(
        UserWarning,
        "Both `population_scaled_non_media_channels` and the deprecated"
        " `non_media_population_scaling_id` were specified."
        " `non_media_population_scaling_id` takes precedence for backward"
        " compatibility; a future version of Meridian will ignore it in"
        " favor of `population_scaled_non_media_channels`.",
    ):
      model_spec = spec.ModelSpec(
          population_scaled_non_media_channels=["nm1"],
          non_media_population_scaling_id=scaling_id,
      )
    self.assertEqual(model_spec.population_scaled_non_media_channels, ("nm1",))
    np.testing.assert_array_equal(
        model_spec.non_media_population_scaling_id, scaling_id
    )

  def test_spec_inits_legacy_non_media_population_scaling_id_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`non_media_population_scaling_id` is deprecated and will be removed"
        " in a future version of Meridian. Use"
        " `population_scaled_non_media_channels` instead.",
    ):
      spec.ModelSpec(non_media_population_scaling_id=np.ones((1,)))

  def test_spec_inits_population_scaled_non_media_channels_single_string_fails(
      self,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`population_scaled_non_media_channels` must be a sequence of"
        " strings, not a single string.",
    ):
      spec.ModelSpec(
          population_scaled_non_media_channels="nm1"  # pyrefly: ignore[bad-argument-type]
      )

  def test_spec_inits_valid_non_media_baseline_values_mapping_works(self):
    baseline_map = {"promo": "min", "discount": 1.5, "event": "max"}
    model_spec = spec.ModelSpec(non_media_baseline_values=baseline_map)
    self.assertEqual(model_spec.non_media_baseline_values, baseline_map)

  def test_spec_inits_invalid_non_media_baseline_values_mapping_value_fails(
      self,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid value for non-media channel 'ch1' in"
        " `non_media_baseline_values`: 'invalid'. Must be a float or"
        " 'min'/'max'.",
    ):
      spec.ModelSpec(non_media_baseline_values={"ch1": "invalid"})

  def test_spec_inits_legacy_non_media_baseline_values_sequence_warns(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "Passing a sequence for `non_media_baseline_values` is deprecated",
    ):
      spec.ModelSpec(non_media_baseline_values=["min", 1.0])


class DateRangeTest(parameterized.TestCase):

  def test_init_with_iso_strings(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    self.assertEqual(dr.start_date, datetime.date(2021, 1, 1))
    self.assertEqual(dr.end_date, datetime.date(2021, 1, 31))

  def test_init_with_datetime_dates(self):
    d1 = datetime.date(2021, 1, 1)
    d2 = datetime.date(2021, 1, 31)
    dr = spec.DateRange(d1, d2)
    self.assertEqual(dr.start_date, d1)
    self.assertEqual(dr.end_date, d2)

  def test_init_with_datetime_objects(self):
    dt1 = datetime.datetime(2021, 1, 1, 12, 0)
    dt2 = datetime.datetime(2021, 1, 31, 18, 0)
    dr = spec.DateRange(dt1, dt2)
    self.assertEqual(dr.start_date, datetime.date(2021, 1, 1))
    self.assertEqual(dr.end_date, datetime.date(2021, 1, 31))

  def test_init_with_np_datetime64(self):
    np_dt1 = np.datetime64("2021-01-01")
    np_dt2 = np.datetime64("2021-01-31")
    dr = spec.DateRange(np_dt1, np_dt2)
    self.assertEqual(dr.start_date, datetime.date(2021, 1, 1))
    self.assertEqual(dr.end_date, datetime.date(2021, 1, 31))

  def test_init_start_equal_end_selects_single_date(self):
    """Both bounds are inclusive, so an equal pair is a valid single date."""
    dr = spec.DateRange("2021-01-01", "2021-01-01")
    self.assertEqual(dr.start_date, datetime.date(2021, 1, 1))
    self.assertEqual(dr.end_date, datetime.date(2021, 1, 1))

  def test_init_invalid_start_after_end_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`start_date` (2021-02-01) must be <= `end_date` (2021-01-01).",
    ):
      spec.DateRange("2021-02-01", "2021-01-01")

  def test_init_invalid_iso_string_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"time data 'invalid' does not match format '%Y-%m-%d'",
    ):
      spec.DateRange("invalid", "2021-01-01")

  def test_init_invalid_type_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Unsupported date value type: <class 'int'> for 123",
    ):
      spec.DateRange(123, "2021-01-01")  # pyrefly: ignore[bad-argument-type]

  def test_init_with_optional_dates(self):
    dr_start_only = spec.DateRange(start_date="2021-01-01")
    self.assertEqual(dr_start_only.start_date, datetime.date(2021, 1, 1))
    self.assertIsNone(dr_start_only.end_date)
    self.assertEqual(
        dr_start_only.date_interval, (datetime.date(2021, 1, 1), None)
    )

    dr_end_only = spec.DateRange(end_date="2021-01-31")
    self.assertIsNone(dr_end_only.start_date)
    self.assertEqual(dr_end_only.end_date, datetime.date(2021, 1, 31))
    self.assertEqual(
        dr_end_only.date_interval, (None, datetime.date(2021, 1, 31))
    )

    dr_unbounded = spec.DateRange()
    self.assertIsNone(dr_unbounded.start_date)
    self.assertIsNone(dr_unbounded.end_date)
    self.assertEqual(dr_unbounded.date_interval, (None, None))

  def test_date_interval_property(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    self.assertEqual(
        dr.date_interval,
        (datetime.date(2021, 1, 1), datetime.date(2021, 1, 31)),
    )

  def test_from_date_interval(self):
    dr = spec.DateRange.from_date_interval(("2021-01-01", "2021-01-31"))
    self.assertEqual(dr.start_date, datetime.date(2021, 1, 1))
    self.assertEqual(dr.end_date, datetime.date(2021, 1, 31))


class ChannelCalibrationSpecTest(parameterized.TestCase):

  def test_init_valid(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ccs = spec.ChannelCalibrationSpec(
        channels=["Search", "Display"],
        date_ranges=[dr],
    )
    self.assertEqual(ccs.channels, ("Search", "Display"))
    self.assertEqual(ccs.date_ranges, (dr,))

  def test_init_single_string_channel_fails(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`channels` must be a sequence of strings, not a single string.",
    ):
      spec.ChannelCalibrationSpec(
          channels="Search",  # pyrefly: ignore[bad-argument-type]
          date_ranges=[dr],
      )

  def test_init_empty_channels_fails(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`channels` cannot be empty.",
    ):
      spec.ChannelCalibrationSpec(channels=[], date_ranges=[dr])

  def test_init_empty_date_ranges_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`date_ranges` cannot be empty.",
    ):
      spec.ChannelCalibrationSpec(channels=["Search"], date_ranges=[])

class CalibrationSpecTest(parameterized.TestCase):

  def test_init_valid_global_date_ranges(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    cs = spec.CalibrationSpec([dr])
    self.assertEqual(cs.spec, (dr,))

  def test_init_valid_channel_calibration_specs(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ccs = spec.ChannelCalibrationSpec(["Search"], [dr])
    cs = spec.CalibrationSpec([ccs])
    self.assertEqual(cs.spec, (ccs,))

  def test_init_empty_spec_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`spec` cannot be empty.",
    ):
      spec.CalibrationSpec([])

  @parameterized.named_parameters(
      ("global_first", True),
      ("channel_first", False),
  )
  def test_init_mixed_spec_fails(self, global_first: bool):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ccs = spec.ChannelCalibrationSpec(["Search"], [dr])
    mixed = [dr, ccs] if global_first else [ccs, dr]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`spec` must be either a sequence of `DateRange` (applying globally"
        " to all channels) or a sequence of `ChannelCalibrationSpec`"
        " (per-channel); the two cannot be mixed.",
    ):
      spec.CalibrationSpec(mixed)  # pyrefly: ignore[bad-argument-type]


class GeoHoldoutSpecTest(parameterized.TestCase):

  def test_init_valid(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ghs = spec.GeoHoldoutSpec(geos=["US", "CA"], date_ranges=[dr])
    self.assertEqual(ghs.geos, ("US", "CA"))
    self.assertEqual(ghs.date_ranges, (dr,))

  def test_init_single_string_geo_fails(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`geos` must be a sequence of strings, not a single string.",
    ):
      spec.GeoHoldoutSpec(
          geos="US",  # pyrefly: ignore[bad-argument-type]
          date_ranges=[dr],
      )

  def test_init_empty_geos_fails(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`geos` cannot be empty.",
    ):
      spec.GeoHoldoutSpec(geos=[], date_ranges=[dr])

  def test_init_empty_date_ranges_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`date_ranges` cannot be empty.",
    ):
      spec.GeoHoldoutSpec(geos=["US"], date_ranges=[])

class RandomHoldoutSpecTest(parameterized.TestCase):

  def test_init_valid_without_seed(self):
    rhs = spec.RandomHoldoutSpec(0.2)
    self.assertEqual(rhs.ratio, 0.2)
    self.assertIsNone(rhs.seed)

  def test_init_valid_with_seed(self):
    rhs = spec.RandomHoldoutSpec(0.2, seed=42)
    self.assertEqual(rhs.ratio, 0.2)
    self.assertEqual(rhs.seed, 42)

  @parameterized.named_parameters(
      ("zero", 0.0),
      ("negative", -0.1),
      ("one", 1.0),
      ("greater_than_one", 1.5),
  )
  def test_init_ratio_out_of_bounds_fails(self, ratio):
    with self.assertRaisesRegex(
        ValueError,
        r"`ratio` must be strictly between 0\.0 and 1\.0",
    ):
      spec.RandomHoldoutSpec(ratio)


class HoldoutSpecTest(parameterized.TestCase):

  def test_init_valid_global_date_ranges(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    hs = spec.HoldoutSpec([dr])
    self.assertEqual(hs.spec, (dr,))

  def test_init_valid_geo_holdout_specs(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ghs = spec.GeoHoldoutSpec(["US"], [dr])
    hs = spec.HoldoutSpec([ghs])
    self.assertEqual(hs.spec, (ghs,))

  def test_init_valid_random_holdout_spec(self):
    rhs = spec.RandomHoldoutSpec(0.2, seed=42)
    hs = spec.HoldoutSpec(rhs)
    self.assertEqual(hs.spec, rhs)

  def test_init_empty_spec_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`spec` cannot be empty.",
    ):
      spec.HoldoutSpec([])

  @parameterized.named_parameters(
      ("global_first", True),
      ("geo_first", False),
  )
  def test_init_mixed_spec_fails(self, global_first: bool):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ghs = spec.GeoHoldoutSpec(["US"], [dr])
    mixed = [dr, ghs] if global_first else [ghs, dr]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`spec` must be either a sequence of `DateRange` (a global holdout"
        " applying to all geos) or a sequence of `GeoHoldoutSpec` (per-geo"
        " holdouts); the two cannot be mixed.",
    ):
      spec.HoldoutSpec(mixed)  # pyrefly: ignore[bad-argument-type]

  def test_init_resolved_with_random_spec(self):
    dr = spec.DateRange("2021-01-01", "2021-01-31")
    ghs = spec.GeoHoldoutSpec(["US"], [dr])
    rhs = spec.RandomHoldoutSpec(0.2, seed=42)
    hs = spec.HoldoutSpec(rhs, resolved=[ghs])
    self.assertEqual(hs.spec, rhs)
    self.assertEqual(hs.resolved, (ghs,))

  def test_init_resolved_defaults_to_none(self):
    rhs = spec.RandomHoldoutSpec(0.2, seed=42)
    hs = spec.HoldoutSpec(rhs)
    self.assertIsNone(hs.resolved)

  @parameterized.named_parameters(
      dict(
          testcase_name="global_date_ranges",
          holdout_spec=[spec.DateRange("2021-01-01", "2021-01-31")],
      ),
      dict(
          testcase_name="geo_holdout_specs",
          holdout_spec=[
              spec.GeoHoldoutSpec(
                  ["US"], [spec.DateRange("2021-01-01", "2021-01-31")]
              )
          ],
      ),
  )
  def test_init_resolved_with_deterministic_spec_fails(self, holdout_spec):
    resolved = [
        spec.GeoHoldoutSpec(
            ["US"], [spec.DateRange("2021-01-01", "2021-01-31")]
        )
    ]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`resolved` can only be set when `spec` is a `RandomHoldoutSpec`;"
        " deterministic holdout specifications reproduce their holdout from"
        " `spec` alone.",
    ):
      spec.HoldoutSpec(holdout_spec, resolved=resolved)

  def test_init_empty_resolved_fails(self):
    rhs = spec.RandomHoldoutSpec(0.2, seed=42)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`resolved` cannot be empty.",
    ):
      spec.HoldoutSpec(rhs, resolved=[])


if __name__ == "__main__":
  absltest.main()
