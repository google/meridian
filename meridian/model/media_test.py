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
_MEDIA_COUNTERFACTUAL = np.array([
    [[0, 0, 0], [0, 0, 6], [0, 8, 9], [10, 11, 12]],
    [[0, 0, 0], [0, 0, 16], [0, 18, 19], [110, 111, 112]],
])
_SPEND_COUNTERFACTUAL = np.array([
    [[0, 0, 30], [0, 50, 60], [70, 80, 90]],
    [[0, 0, 130], [0, 150, 160], [170, 180, 190]],
])
_INPUT_DATA_WITH_MEDIA_ONLY = mock.MagicMock(
    reach=None,
    frequency=None,
    rf_spend=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    media=_MEDIA,
    media_spend=_SPEND,
    population=np.array([1, 2]),
)
_INPUT_DATA_WITH_RF_ONLY = mock.MagicMock(
    media=None,
    media_spend=None,
    time=["2021-01-01", "2021-01-08", "2021-01-15"],
    reach=_MEDIA,
    frequency=_MEDIA,
    rf_spend=_SPEND,
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
    self.assertIsNone(media_tensors.media_counterfactual)
    self.assertIsNone(media_tensors.media_counterfactual_scaled)
    self.assertIsNone(media_tensors.media_spend_counterfactual)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_calibration_period",
          roi_calibration_period=None,
          expected_counterfactual=tf.zeros_like(_MEDIA_COUNTERFACTUAL),
          expected_spend_counterfactual=tf.zeros_like(_SPEND_COUNTERFACTUAL),
      ),
      dict(
          testcase_name="with_calibration_period",
          roi_calibration_period=_ROI_CALIBRATION_PERIOD,
          expected_counterfactual=_MEDIA_COUNTERFACTUAL,
          expected_spend_counterfactual=_SPEND_COUNTERFACTUAL,
      ),
  )
  def test_media_tensors(
      self,
      roi_calibration_period: np.ndarray | None,
      expected_counterfactual: tf.Tensor,
      expected_spend_counterfactual: tf.Tensor,
  ):
    media_tensors = media.build_media_tensors(
        _INPUT_DATA_WITH_MEDIA_ONLY,
        spec.ModelSpec(roi_calibration_period=roi_calibration_period),
    )

    self.assertAllClose(media_tensors.media, _INPUT_DATA_WITH_MEDIA_ONLY.media)
    self.assertAllClose(
        media_tensors.media_spend, _INPUT_DATA_WITH_MEDIA_ONLY.media_spend
    )
    self.assertIsNotNone(media_tensors.media_transformer)
    self.assertAllClose(
        media_tensors.media_scaled, _INPUT_DATA_WITH_MEDIA_ONLY.media
    )
    self.assertAllClose(
        media_tensors.media_counterfactual, expected_counterfactual
    )
    self.assertAllClose(
        media_tensors.media_counterfactual_scaled, expected_counterfactual
    )
    self.assertAllClose(
        media_tensors.media_spend_counterfactual, expected_spend_counterfactual
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
    self.assertIsNone(organic_media_tensors.organic_media_counterfactual)
    self.assertIsNone(organic_media_tensors.organic_media_counterfactual_scaled)

  def test_organic_media_tensors(self):
    expected_counterfactual = tf.zeros_like(_MEDIA_COUNTERFACTUAL)
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
    self.assertAllClose(
        organic_media_tensors.organic_media_counterfactual,
        expected_counterfactual,
    )
    self.assertAllClose(
        organic_media_tensors.organic_media_counterfactual_scaled,
        expected_counterfactual,
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
    self.assertIsNone(rf_tensors.reach_counterfactual)
    self.assertIsNone(rf_tensors.reach_counterfactual_scaled)
    self.assertIsNone(rf_tensors.rf_spend_counterfactual)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_calibration_period",
          rf_roi_calibration_period=None,
          expected_counterfactual=tf.zeros_like(_MEDIA_COUNTERFACTUAL),
          expected_spend_counterfactual=tf.zeros_like(_SPEND_COUNTERFACTUAL),
      ),
      dict(
          testcase_name="with_calibration_period",
          rf_roi_calibration_period=_ROI_CALIBRATION_PERIOD,
          expected_counterfactual=_MEDIA_COUNTERFACTUAL,
          expected_spend_counterfactual=_SPEND_COUNTERFACTUAL,
      ),
  )
  def test_rf_tensors(
      self,
      rf_roi_calibration_period: np.ndarray | None,
      expected_counterfactual: tf.Tensor,
      expected_spend_counterfactual: tf.Tensor,
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
    self.assertAllClose(
        rf_tensors.reach_counterfactual, expected_counterfactual
    )
    self.assertAllClose(
        rf_tensors.reach_counterfactual_scaled, expected_counterfactual
    )
    self.assertAllClose(
        rf_tensors.rf_spend_counterfactual, expected_spend_counterfactual
    )


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
    self.assertIsNone(organic_rf_tensors.organic_reach_counterfactual)
    self.assertIsNone(organic_rf_tensors.organic_reach_counterfactual_scaled)

  def test_organic_rf_tensors(self):
    expected_counterfactual = tf.zeros_like(_MEDIA_COUNTERFACTUAL)
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
    self.assertAllClose(
        organic_rf_tensors.organic_reach_counterfactual, expected_counterfactual
    )
    self.assertAllClose(
        organic_rf_tensors.organic_reach_counterfactual_scaled,
        expected_counterfactual,
    )


if __name__ == "__main__":
  absltest.main()
