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

"""Unit tests for media.py."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.model import media
from meridian.model import spec
from meridian.model import transformers
import numpy as np
import tensorflow as tf

# Dimensions: (n_geos=2, n_media_times=4, n_channels=3)
_MEDIA = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
    [[11, 12, 13], [14, 15, 16], [17, 18, 19], [110, 111, 112]],
])
# Dimensions: (n_geos=2, n_times=3, n_channels=3)
_SPEND = np.array([
    [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
    [[110, 120, 130], [140, 150, 160], [170, 180, 190]],
])
# Dimensions: (n_media_times=4, n_channels=3)
_ROI_CALIBRATION_PERIOD = np.array([
    [True, True, True],
    [True, True, False],
    [True, False, False],
    [False, False, False],
])
_MEDIA_COUNTERFACTUAL_ROI_CALIBRATION_PERIOD = np.array([
    [[0, 0, 0], [0, 0, 6], [0, 8, 9], [10, 11, 12]],
    [[0, 0, 0], [0, 0, 16], [0, 18, 19], [110, 111, 112]],
])
_SPEND_COUNTERFACTUAL_ROI_CALIBRATION_PERIOD = np.array([
    [[0, 0, 30], [0, 50, 60], [70, 80, 90]],
    [[0, 0, 130], [0, 150, 160], [170, 180, 190]],
])
_MEDIA_COUNTERFACTUAL_MROI_CALIBRATION_PERIOD = np.array([
    [
        [1 * c.MROI_FACTOR, 2 * c.MROI_FACTOR, 3 * c.MROI_FACTOR],
        [4 * c.MROI_FACTOR, 5 * c.MROI_FACTOR, 6],
        [7 * c.MROI_FACTOR, 8, 9],
        [10, 11, 12],
    ],
    [
        [11 * c.MROI_FACTOR, 12 * c.MROI_FACTOR, 13 * c.MROI_FACTOR],
        [14 * c.MROI_FACTOR, 15 * c.MROI_FACTOR, 16],
        [17 * c.MROI_FACTOR, 18, 19],
        [110, 111, 112],
    ],
])
_SPEND_COUNTERFACTUAL_MROI_CALIBRATION_PERIOD = np.array([
    [
        [10 * c.MROI_FACTOR, 20 * c.MROI_FACTOR, 30],
        [40 * c.MROI_FACTOR, 50, 60],
        [70, 80, 90],
    ],
    [
        [110 * c.MROI_FACTOR, 120 * c.MROI_FACTOR, 130],
        [140 * c.MROI_FACTOR, 150, 160],
        [170, 180, 190],
    ],
])
_INPUT_DATA_WITH_MEDIA_ONLY = mock.MagicMock(
    reach=None,
    frequency=None,
    rf_spend=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    media=_MEDIA,
    media_spend=_SPEND,
    allocated_media_spend=_SPEND,
    population=np.array([1, 2]),
)
_INPUT_DATA_WITH_RF_ONLY = mock.MagicMock(
    media=None,
    media_spend=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    reach=_MEDIA,
    frequency=_MEDIA,
    rf_spend=_SPEND,
    allocated_rf_spend=_SPEND,
    population=np.array([1, 2]),
)
_INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA = mock.MagicMock(
    reach=None,
    frequency=None,
    rf_spend=None,
    organic_reach=None,
    organic_frequency=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    media=_MEDIA,
    media_spend=_SPEND,
    allocated_media_spend=_SPEND,
    organic_media=_MEDIA,
    population=np.array([1, 2]),
)
_INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF = mock.MagicMock(
    reach=None,
    frequency=None,
    rf_spend=None,
    organic_media=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    media=_MEDIA,
    media_spend=_SPEND,
    allocated_media_spend=_SPEND,
    organic_reach=_MEDIA,
    organic_frequency=_MEDIA,
    population=np.array([1, 2]),
)


class MediaTensorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Return unchanged media values.
    self.enter_context(
        mock.patch.object(
            transformers.MediaTransformer,
            "forward",
            return_value=_INPUT_DATA_WITH_MEDIA_ONLY.media,
        )
    )

  def test_no_media_values(self):
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_RF_ONLY, spec.ModelSpec()
    )

    self.assertIsNone(media_tensors.media)
    self.assertIsNone(media_tensors.media_spend)
    self.assertIsNone(media_tensors.media_transformer)
    self.assertIsNone(media_tensors.media_scaled)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_calibration_period_paid_media_prior_type_roi",
          paid_media_prior_type=c.TREATMENT_PRIOR_TYPE_ROI,
          roi_calibration_period=None,
      ),
      dict(
          testcase_name="no_calibration_period_paid_media_prior_type_mroi",
          paid_media_prior_type=c.TREATMENT_PRIOR_TYPE_MROI,
          roi_calibration_period=None,
      ),
      dict(
          testcase_name="with_calibration_period_paid_media_prior_type_roi",
          paid_media_prior_type=c.TREATMENT_PRIOR_TYPE_ROI,
          roi_calibration_period=_ROI_CALIBRATION_PERIOD,
      ),
      dict(
          testcase_name=(
              "no_calibration_period_paid_media_prior_type_coefficient"
          ),
          paid_media_prior_type=c.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          roi_calibration_period=None,
      ),
  )
  def test_media_tensors(
      self,
      paid_media_prior_type: str,
      roi_calibration_period: np.ndarray | None,
  ):
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_MEDIA_ONLY,
        spec.ModelSpec(
            media_prior_type=paid_media_prior_type,
            rf_prior_type=paid_media_prior_type,
            roi_calibration_period=roi_calibration_period,
        ),
    )

    self.assertAllClose(media_tensors.media, _INPUT_DATA_WITH_MEDIA_ONLY.media)
    self.assertAllClose(
        media_tensors.media_spend, _INPUT_DATA_WITH_MEDIA_ONLY.media_spend
    )
    self.assertIsNotNone(media_tensors.media_transformer)
    self.assertAllClose(
        media_tensors.media_scaled, _INPUT_DATA_WITH_MEDIA_ONLY.media
    )


class OrganicMediaTensorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Return unchanged organic media values.
    self.enter_context(
        mock.patch.object(
            transformers.MediaTransformer,
            "forward",
            return_value=_INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA.organic_media,
        )
    )

  def test_no_organic_media_values(self):
    organic_media_tensors = media.build_organic_media_tensors(
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF
    )

    self.assertIsNone(organic_media_tensors.organic_media)
    self.assertIsNone(organic_media_tensors.organic_media_transformer)
    self.assertIsNone(organic_media_tensors.organic_media_scaled)

  def test_organic_media_tensors(self):
    organic_media_tensors = media.build_organic_media_tensors(
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA
    )

    self.assertAllClose(
        organic_media_tensors.organic_media,
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA.organic_media,
    )
    self.assertIsNotNone(organic_media_tensors.organic_media_transformer)
    self.assertAllClose(
        organic_media_tensors.organic_media_scaled,
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA.organic_media,
    )


class RfTensorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Return unchanged reach values.
    self.enter_context(
        mock.patch.object(
            transformers.MediaTransformer,
            "forward",
            return_value=_INPUT_DATA_WITH_RF_ONLY.reach,
        )
    )

  def test_no_rf_values(self):
    rf_tensors = media.build_rf_tensors(
        _INPUT_DATA_WITH_MEDIA_ONLY, spec.ModelSpec()
    )

    self.assertIsNone(rf_tensors.reach)
    self.assertIsNone(rf_tensors.frequency)
    self.assertIsNone(rf_tensors.rf_spend)
    self.assertIsNone(rf_tensors.reach_transformer)
    self.assertIsNone(rf_tensors.reach_scaled)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_calibration_period",
          rf_roi_calibration_period=None,
      ),
      dict(
          testcase_name="with_calibration_period",
          rf_roi_calibration_period=_ROI_CALIBRATION_PERIOD,
      ),
  )
  def test_rf_tensors(
      self,
      rf_roi_calibration_period: np.ndarray | None,
  ):
    rf_tensors = media.build_rf_tensors(
        _INPUT_DATA_WITH_RF_ONLY,
        spec.ModelSpec(rf_roi_calibration_period=rf_roi_calibration_period),
    )

    self.assertAllClose(rf_tensors.reach, _INPUT_DATA_WITH_RF_ONLY.reach)
    self.assertAllClose(
        rf_tensors.frequency, _INPUT_DATA_WITH_RF_ONLY.frequency
    )
    self.assertAllClose(rf_tensors.rf_spend, _INPUT_DATA_WITH_RF_ONLY.rf_spend)
    self.assertIsNotNone(rf_tensors.reach_transformer)
    self.assertAllClose(rf_tensors.reach_scaled, _INPUT_DATA_WITH_RF_ONLY.reach)


class OrganicRfTensorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Return unchanged reach values.
    self.enter_context(
        mock.patch.object(
            transformers.MediaTransformer,
            "forward",
            return_value=_INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF.organic_reach,
        )
    )

  def test_no_organic_rf_values(self):
    organic_rf_tensors = media.build_organic_rf_tensors(
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_MEDIA,
    )

    self.assertIsNone(organic_rf_tensors.organic_reach)
    self.assertIsNone(organic_rf_tensors.organic_frequency)
    self.assertIsNone(organic_rf_tensors.organic_reach_transformer)
    self.assertIsNone(organic_rf_tensors.organic_reach_scaled)

  def test_organic_rf_tensors(self):
    organic_rf_tensors = media.build_organic_rf_tensors(
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF,
    )

    self.assertAllClose(
        organic_rf_tensors.organic_reach,
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF.organic_reach,
    )
    self.assertAllClose(
        organic_rf_tensors.organic_frequency,
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF.organic_frequency,
    )
    self.assertIsNotNone(organic_rf_tensors.organic_reach_transformer)
    self.assertAllClose(
        organic_rf_tensors.organic_reach_scaled,
        _INPUT_DATA_WITH_MEDIA_AND_ORGANIC_RF.organic_reach,
    )


# pylint: disable=protected-access
class AggregateSpendByChannelTest(tf.test.TestCase):
  """Tests for the _aggregate_spend_by_channel function."""

  def test_basic_aggregation_no_calibration(self):
    """Tests aggregation without a calibration period."""
    # Shape: (n_geos=2, n_times=3, n_channels=2)
    spend_tensor = tf.constant(
        [
            [[10, 100], [20, 200], [30, 300]],  # Geo 0
            [[40, 400], [50, 500], [60, 600]],
        ],  # Geo 1
        dtype=tf.float32,
    )

    aggregated_spend = media._aggregate_spend_by_channel(spend_tensor)

    # Expected shape: (n_channels=2,)
    self.assertEqual(aggregated_spend.shape, (2,))

    # Expected values:
    # Channel 0: 10 + 20 + 30 + 40 + 50 + 60 = 210
    # Channel 1: 100 + 200 + 300 + 400 + 500 + 600 = 2100
    expected_result = tf.constant([210, 2100], dtype=tf.float32)
    self.assertAllClose(aggregated_spend, expected_result)

  def test_aggregation_with_calibration(self):
    """Tests aggregation with a calibration period."""
    # Shape: (n_geos=2, n_times=3, n_channels=2)
    spend_tensor = tf.constant(
        [
            [[10, 100], [20, 200], [30, 300]],  # Geo 0
            [[40, 400], [50, 500], [60, 600]],
        ],  # Geo 1
        dtype=tf.float32,
    )
    # Shape: (n_times=3, n_channels=2)
    # Use times: T0 for C0, T1 for C1, T2 for C0
    calibration_period = tf.constant(
        [
            [True, False],  # Time 0: Use Ch 0, Don't use Ch 1
            [False, True],  # Time 1: Don't use Ch 0, Use Ch 1
            [True, False],
        ],  # Time 2: Use Ch 0, Don't use Ch 1
        dtype=tf.bool,
    )

    aggregated_spend = media._aggregate_spend_by_channel(
        spend_tensor, calibration_period
    )

    # Expected shape: (n_channels=2,)
    self.assertEqual(aggregated_spend.shape, (2,))

    # Expected values:
    # Channel 0: Use T0 & T2 -> (10 + 40) + (30 + 60) = 50 + 90 = 140
    # Channel 1: Use T1      -> (200 + 500) = 700
    expected_result = tf.constant([140, 700], dtype=tf.float32)
    self.assertAllClose(aggregated_spend, expected_result)

  def test_calibration_period_shape_handling(self):
    """Tests aggregation when calibration period has more time steps."""
    # Shape: (n_geos=2, n_times=3, n_channels=2)
    spend_tensor = tf.constant(
        [
            [[10, 100], [20, 200], [30, 300]],  # Geo 0
            [[40, 400], [50, 500], [60, 600]],
        ],  # Geo 1
        dtype=tf.float32,
    )
    # Shape: (n_media_times=5, n_channels=2) -> n_media_times > n_times
    # Only the last 3 rows should be used for calculation
    calibration_period_long = tf.constant(
        [
            [False, False],  # Ignored
            [False, False],  # Ignored
            [True, False],  # Corresponds to spend T0: Use Ch 0
            [False, True],  # Corresponds to spend T1: Use Ch 1
            [True, False],
        ],  # Corresponds to spend T2: Use Ch 0
        dtype=tf.bool,
    )

    aggregated_spend = media._aggregate_spend_by_channel(
        spend_tensor, calibration_period_long
    )

    # Expected shape: (n_channels=2,)
    self.assertEqual(aggregated_spend.shape, (2,))

    # Expected values (same as test_aggregation_with_calibration):
    # Channel 0: Use T0 & T2 -> (10 + 40) + (30 + 60) = 50 + 90 = 140
    # Channel 1: Use T1      -> (200 + 500) = 700
    expected_result = tf.constant([140, 700], dtype=tf.float32)
    self.assertAllClose(aggregated_spend, expected_result)


if __name__ == "__main__":
  absltest.main()
