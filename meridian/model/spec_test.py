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


if __name__ == "__main__":
  absltest.main()
