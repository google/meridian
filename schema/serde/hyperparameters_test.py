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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian.model import media
from meridian.model import model
from meridian.model import spec
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import hyperparameters
from schema.serde import test_data
import numpy as np
import tensorflow as tf

from tensorflow.python.util.protobuf import compare

_MediaEffectsDist = meridian_pb.MediaEffectsDistribution
_PaidMediaPriorType = meridian_pb.PaidMediaPriorType


class HyperparametersSerdeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.serde = hyperparameters.HyperparametersSerde()

  def _mock_meridian(self) -> mock.MagicMock:
    """Creates a mock MMM object.

    Returns:
      A mock MMM object.
    """
    return mock.MagicMock(
        spec=model.Meridian,
        controls_scaled=tf.convert_to_tensor(
            np.full((2, 3), 5.0), dtype=tf.float32
        ),
        kpi_scaled=tf.convert_to_tensor(np.full((4,), 6.0), dtype=tf.float32),
        media_tensors=media.MediaTensors(),
        rf_tensors=media.RfTensors(),
        inference_data=az.InferenceData(),
        model_spec=spec.ModelSpec(),
    )

  def _setup_meridian(self):
    self._mock_meridian = self._mock_meridian()

  @parameterized.named_parameters(
      dict(
          testcase_name='default_model_spec',
          model_spec=test_data.DEFAULT_MODEL_SPEC,
          expected_proto=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
      ),
      dict(
          testcase_name='default_model_spec_with_infinite_max_lag',
          model_spec=test_data.DEFAULT_MODEL_SPEC_INFINITE_MAX_LAG,
          expected_proto=test_data.DEFAULT_HYPERPARAMETERS_PROTO_INFINITE_MAX_LAG,
      ),
      dict(
          testcase_name='custom_model_spec_1',
          model_spec=test_data.CUSTOM_MODEL_SPEC_1,
          expected_proto=test_data.CUSTOM_HYPERPARAMETERS_PROTO_1,
      ),
      dict(
          testcase_name='custom_model_spec_2',
          model_spec=test_data.CUSTOM_MODEL_SPEC_2,
          expected_proto=test_data.CUSTOM_HYPERPARAMETERS_PROTO_2,
      ),
  )
  def test_serialize(
      self,
      model_spec: mock.MagicMock,
      expected_proto: meridian_pb.Hyperparameters,
  ):
    self._setup_meridian()
    self._mock_meridian.model_spec = model_spec
    compare.assertProto2Equal(
        self,
        self.serde.serialize(model_spec),
        expected_proto,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='default_model_spec',
          hyperparameters_proto=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
          expected_model_spec=test_data.DEFAULT_MODEL_SPEC,
      ),
      dict(
          testcase_name='default_model_spec_infinite_max_lag',
          hyperparameters_proto=test_data.DEFAULT_HYPERPARAMETERS_PROTO_INFINITE_MAX_LAG,
          expected_model_spec=test_data.DEFAULT_MODEL_SPEC_INFINITE_MAX_LAG,
      ),
      dict(
          testcase_name='custom_model_spec_1',
          hyperparameters_proto=test_data.CUSTOM_HYPERPARAMETERS_PROTO_1,
          expected_model_spec=test_data.CUSTOM_MODEL_SPEC_1,
      ),
      dict(
          testcase_name='custom_model_spec_2',
          hyperparameters_proto=test_data.CUSTOM_HYPERPARAMETERS_PROTO_2,
          expected_model_spec=test_data.CUSTOM_MODEL_SPEC_2,
      ),
  )
  def test_deserialize_hyperparameters(
      self, hyperparameters_proto, expected_model_spec
  ):
    deserialized_model_spec = self.serde.deserialize(hyperparameters_proto)
    self.assertEqual(
        deserialized_model_spec.media_effects_dist,
        expected_model_spec.media_effects_dist,
    )
    self.assertEqual(
        deserialized_model_spec.hill_before_adstock,
        expected_model_spec.hill_before_adstock,
    )
    self.assertEqual(
        deserialized_model_spec.max_lag, expected_model_spec.max_lag
    )
    self.assertEqual(
        deserialized_model_spec.unique_sigma_for_each_geo,
        expected_model_spec.unique_sigma_for_each_geo,
    )
    self.assertEqual(
        deserialized_model_spec.media_prior_type,
        expected_model_spec.media_prior_type,
    )
    self.assertEqual(
        deserialized_model_spec.rf_prior_type,
        expected_model_spec.rf_prior_type,
    )
    self.assertEqual(deserialized_model_spec.knots, expected_model_spec.knots)
    self.assertEqual(
        deserialized_model_spec.baseline_geo, expected_model_spec.baseline_geo
    )
    if expected_model_spec.roi_calibration_period is None:
      self.assertIsNone(deserialized_model_spec.roi_calibration_period)
    else:
      np.testing.assert_equal(
          deserialized_model_spec.roi_calibration_period,
          expected_model_spec.roi_calibration_period,
      )

    if expected_model_spec.rf_roi_calibration_period is None:
      self.assertIsNone(deserialized_model_spec.rf_roi_calibration_period)
    else:
      np.testing.assert_equal(
          deserialized_model_spec.rf_roi_calibration_period,
          expected_model_spec.rf_roi_calibration_period,
      )

    if expected_model_spec.holdout_id is None:
      self.assertIsNone(deserialized_model_spec.holdout_id)
    else:
      np.testing.assert_equal(
          deserialized_model_spec.holdout_id, expected_model_spec.holdout_id
      )

    if expected_model_spec.control_population_scaling_id is None:
      self.assertIsNone(deserialized_model_spec.control_population_scaling_id)
    else:
      np.testing.assert_equal(
          deserialized_model_spec.control_population_scaling_id,
          expected_model_spec.control_population_scaling_id,
      )

    if expected_model_spec.non_media_population_scaling_id is None:
      self.assertIsNone(deserialized_model_spec.non_media_population_scaling_id)
    else:
      np.testing.assert_equal(
          deserialized_model_spec.non_media_population_scaling_id,
          expected_model_spec.non_media_population_scaling_id,
      )

  def test_deserialize_invalid_media_effects_dist(self):
    invalid_proto = meridian_pb.Hyperparameters()
    invalid_proto.CopyFrom(test_data.DEFAULT_HYPERPARAMETERS_PROTO)
    invalid_proto.media_effects_dist = (
        _MediaEffectsDist.MEDIA_EFFECTS_DISTRIBUTION_UNSPECIFIED
    )
    with self.assertRaisesRegex(
        ValueError, 'Unsupported MediaEffectsDistribution proto enum value'
    ):
      self.serde.deserialize(invalid_proto)

  def test_deserialize_invalid_paid_media_prior_type(self):
    invalid_proto = meridian_pb.Hyperparameters()
    invalid_proto.CopyFrom(test_data.DEFAULT_HYPERPARAMETERS_PROTO)
    invalid_proto.media_prior_type = -1
    with self.assertRaisesRegex(
        ValueError, 'Unsupported PaidMediaPriorType proto enum value'
    ):
      self.serde.deserialize(invalid_proto)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_model_spec',
          model_spec=test_data.DEFAULT_MODEL_SPEC,
      ),
      dict(
          testcase_name='default_model_spec_infinite_max_lag',
          model_spec=test_data.DEFAULT_MODEL_SPEC_INFINITE_MAX_LAG,
      ),
      dict(
          testcase_name='custom_model_spec_1',
          model_spec=test_data.CUSTOM_MODEL_SPEC_1,
      ),
      dict(
          testcase_name='custom_model_spec_2',
          model_spec=test_data.CUSTOM_MODEL_SPEC_2,
      ),
  )
  def test_serialize_deserialize_model_spec(self, model_spec: spec.ModelSpec):
    serialized = self.serde.serialize(model_spec)
    deserialized = self.serde.deserialize(serialized)
    serialized_again = self.serde.serialize(deserialized)

    compare.assertProto2Equal(
        self,
        serialized,
        serialized_again,
        precision=2,
    )


if __name__ == '__main__':
  absltest.main()
