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

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.data import input_data as meridian_input_data
from meridian.data import test_utils
from meridian.model import media
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
from mmm.v1.marketing import marketing_data_pb2 as marketing_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import distribution
from schema.serde import hyperparameters
from schema.serde import inference_data
from schema.serde import marketing_data
from schema.serde import meridian_serde as serde
from schema.serde import test_data
import numpy as np
import semver
import tensorflow as tf
import xarray.testing as xrt

from google.protobuf import any_pb2
from tensorflow.python.util.protobuf import compare
from tensorflow.core.framework import types_pb2  # pylint: disable=g-direct-tensorflow-import


_PRIOR_DATASET_CHAINS = 1
_PRIOR_DATASET_DRAWS = 2
_PRIOR_DATASET = test_data.make_sample_dataset(
    _PRIOR_DATASET_CHAINS,
    _PRIOR_DATASET_DRAWS,
    n_geos=5,
    n_controls=2,
    n_knots=49,
    n_times=49,
    n_media_channels=3,
)

_POSTERIOR_DATASET_CHAINS = 3
_POSTERIOR_DATASET_DRAWS = 7
_POSTERIOR_DATASET = test_data.make_sample_dataset(
    _POSTERIOR_DATASET_CHAINS,
    _POSTERIOR_DATASET_DRAWS,
    n_geos=5,
    n_controls=2,
    n_knots=49,
    n_times=49,
    n_media_channels=3,
)

_INPUT_DATA = test_utils.sample_input_data_non_revenue_revenue_per_kpi(
    n_geos=5,
    n_times=49,
    n_media_times=52,
    n_controls=2,
    n_media_channels=3,
    seed=1,
)
_INPUT_DATA_NO_CONTROLS = (
    test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=5,
        n_times=49,
        n_media_times=52,
        n_controls=0,
        n_media_channels=3,
        seed=1,
    )
)

_R_HATS = {
    constants.ALPHA_M: tf.convert_to_tensor(
        np.full((1, 2, 3), 22.0), dtype=tf.float32
    ),
    constants.BETA_GRF: tf.convert_to_tensor(
        np.full((4, 5), 33.0), dtype=tf.float32
    ),
    constants.TAU_G: tf.convert_to_tensor(
        np.full((6,), 44.0), dtype=tf.float32
    ),
}


class MeridianSerdeTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serde = serde.MeridianSerde()

    self._mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, 'Analyzer', autospec=True)
    )
    self._mock_visualizer = self.enter_context(
        mock.patch.object(visualizer, 'ModelDiagnostics', autospec=True)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          model_id='empty_model',
          meridian_version=semver.VersionInfo.parse('1.0.0'),
          input_data=_INPUT_DATA,
          model_spec=None,
          inf_data=None,
      ),
      dict(
          testcase_name='defaults',
          model_id='default_model',
          meridian_version=semver.VersionInfo.parse('1.0.1'),
          input_data=_INPUT_DATA,
          model_spec=test_data.DEFAULT_MODEL_SPEC,
          inf_data=az.InferenceData(),
      ),
      dict(
          testcase_name='priors_and_posteriors',
          model_id='priors_and_posteriors_model',
          meridian_version=semver.VersionInfo.parse('1.1.0'),
          input_data=_INPUT_DATA,
          model_spec=spec.ModelSpec(knots=49),
          inf_data=az.InferenceData(
              prior=_PRIOR_DATASET,
              posterior=_POSTERIOR_DATASET,
          ),
      ),
  )
  def test_serialize(
      self,
      model_id: str,
      meridian_version: semver.VersionInfo,
      input_data: meridian_input_data.InputData,
      model_spec: spec.ModelSpec,
      inf_data: az.InferenceData,
  ):
    meridian_model = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
        inference_data=inf_data,
    )

    serialized_model = self.serde.serialize(
        meridian_model, model_id, meridian_version
    )

    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)

    self.assertTrue(
        serialized_model.model.Is(meridian_pb.MeridianModel.DESCRIPTOR)
    )
    serialized_model.model.Unpack(unpacked_model)
    self.assertEqual(unpacked_model.model_version, str(meridian_version))

  def test_serialize_no_controls(self):
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA_NO_CONTROLS,
        model_spec=test_data.DEFAULT_MODEL_SPEC,
    )

    serialized_model = self.serde.serialize(
        meridian_model, 'test_model', semver.VersionInfo.parse('1.0.3')
    )

    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)

    self.assertTrue(
        serialized_model.model.Is(meridian_pb.MeridianModel.DESCRIPTOR)
    )
    serialized_model.model.Unpack(unpacked_model)
    self.assertEqual(unpacked_model.model_version, '1.0.3')

    self.assertFalse(unpacked_model.HasField('controls_scaled'))

  def test_serialize_tensors(self):
    with (
        mock.patch.object(
            marketing_data.MarketingDataSerde,
            'serialize',
            return_value=marketing_pb.MarketingData(),
        ),
        mock.patch.object(
            hyperparameters.HyperparametersSerde,
            'serialize',
            return_value=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        ),
        mock.patch.object(
            distribution.DistributionSerde,
            'serialize',
            return_value=meridian_pb.PriorDistributions(),
        ),
        mock.patch.object(
            inference_data.InferenceDataSerde,
            'serialize',
            return_value=meridian_pb.InferenceData(),
        ),
    ):
      controls_scaled = tf.convert_to_tensor(
          np.full((2, 3), 5.0), dtype=tf.float32
      )
      kpi_scaled = tf.convert_to_tensor(np.full((4,), 6.0), dtype=tf.float32)
      media_tensors = media.MediaTensors(
          media_scaled=tf.convert_to_tensor(
              np.full((5, 6, 7), 7.0), dtype=tf.float32
          )
      )
      rf_tensors = media.RfTensors(
          reach_scaled=tf.convert_to_tensor(
              np.full((8, 9), 8.0), dtype=tf.float32
          )
      )
      meridian_model = mock.MagicMock(
          controls_scaled=controls_scaled,
          kpi_scaled=kpi_scaled,
          media_tensors=media_tensors,
          rf_tensors=rf_tensors,
          inference_data=None,
          model_spec=test_data.DEFAULT_MODEL_SPEC,
      )
      mock_version = semver.VersionInfo.parse('1.0.3')
      serialized_model = self.serde.serialize(
          meridian_model, 'test_model', mock_version
      )

    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)

    expected_proto = meridian_pb.MeridianModel(
        model_id='test_model',
        model_version=str(mock_version),
        controls_scaled=test_data.make_tensor_proto(
            dims=[2, 3],
            dtype=types_pb2.DT_FLOAT,
            tensor_content=controls_scaled.numpy().tobytes(),
        ),
        kpi_scaled=test_data.make_tensor_proto(
            dims=[4],
            dtype=types_pb2.DT_FLOAT,
            tensor_content=kpi_scaled.numpy().tobytes(),
        ),
        media_scaled=test_data.make_tensor_proto(
            dims=[5, 6, 7],
            dtype=types_pb2.DT_FLOAT,
            tensor_content=media_tensors.media_scaled.numpy().tobytes(),
        ),
        reach_scaled=test_data.make_tensor_proto(
            dims=[8, 9],
            dtype=types_pb2.DT_FLOAT,
            tensor_content=rf_tensors.reach_scaled.numpy().tobytes(),
        ),
    )
    compare.assertProto2Equal(
        self,
        unpacked_model,
        expected_proto,
        # Ignored fields are tested separately
        ignored_fields=[
            'hyperparameters',
            'prior_distributions',
            'inference_data',
            'convergence_info',
        ],
    )

  def test_serialize_convergence_status_not_ok(self):
    with mock.patch.object(
        self._mock_visualizer.return_value,
        'plot_rhat_boxplot',
        side_effect=model.MCMCSamplingError(),
    ):
      serialized_model = self.serde.serialize(
          model.Meridian(input_data=_INPUT_DATA),
          'test_model',
          include_convergence_info=True,
      )
      unpacked_model = meridian_pb.MeridianModel()
      serialized_model.model.Unpack(unpacked_model)

      self.assertFalse(unpacked_model.convergence_info.convergence)

  def test_serialize_convergence_proto_plot_rhat_not_fitted(self):
    with mock.patch.object(
        self._mock_visualizer.return_value,
        'plot_rhat_boxplot',
        side_effect=model.NotFittedModelError(),
    ):
      serialized_model = self.serde.serialize(
          model.Meridian(input_data=_INPUT_DATA),
          'test_model',
          include_convergence_info=True,
      )
      unpacked_model = meridian_pb.MeridianModel()
      serialized_model.model.Unpack(unpacked_model)

      compare.assertProto2Equal(
          self,
          unpacked_model.convergence_info,
          meridian_pb.ModelConvergence(),
      )

  def test_serialize_convergence_proto_get_rhat_not_fitted(self):
    with mock.patch.object(
        self._mock_analyzer.return_value,
        'get_rhat',
        side_effect=model.NotFittedModelError(),
    ):
      serialized_model = self.serde.serialize(
          model.Meridian(input_data=_INPUT_DATA),
          'test_model',
          include_convergence_info=True,
      )
      unpacked_model = meridian_pb.MeridianModel()
      serialized_model.model.Unpack(unpacked_model)

      compare.assertProto2Equal(
          self,
          unpacked_model.convergence_info,
          meridian_pb.ModelConvergence(),
      )

  def test_serialize_include_convergence_info_false(self):
    self._mock_analyzer.return_value.get_rhat.return_value = _R_HATS

    serialized_model = self.serde.serialize(
        model.Meridian(input_data=_INPUT_DATA),
        'test_model',
        include_convergence_info=False,
    )
    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)

    compare.assertProto2Equal(
        self,
        meridian_pb.ModelConvergence(),
        unpacked_model.convergence_info,
    )

  def test_serialize_model_convergence_proto(self):
    self._mock_analyzer.return_value.get_rhat.return_value = _R_HATS

    expected_sampling_trace = meridian_pb.McmcSamplingTrace(
        num_chains=len(_POSTERIOR_DATASET.chain),
        num_draws=len(_POSTERIOR_DATASET.draw),
        step_size=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_DOUBLE,
            tensor_content=_POSTERIOR_DATASET.data_vars.get(
                constants.STEP_SIZE
            ).data.tobytes(),
        ),
        tune=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_BOOL,
            bool_vals=[False]
            * (_POSTERIOR_DATASET_CHAINS * _POSTERIOR_DATASET_DRAWS),
        ),
        target_log_prob=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_DOUBLE,
            tensor_content=_POSTERIOR_DATASET.data_vars.get(
                constants.TARGET_LOG_PROBABILITY_TF
            ).data.tobytes(),
        ),
        diverging=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_BOOL,
            bool_vals=[False]
            * (_POSTERIOR_DATASET_CHAINS * _POSTERIOR_DATASET_DRAWS),
        ),
        accept_ratio=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_DOUBLE,
            tensor_content=_POSTERIOR_DATASET.data_vars.get(
                constants.ACCEPT_RATIO
            ).data.tobytes(),
        ),
        n_steps=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_DOUBLE,
            tensor_content=_POSTERIOR_DATASET.data_vars.get(
                constants.N_STEPS
            ).data.tobytes(),
        ),
        is_accepted=test_data.make_tensor_proto(
            dims=[_POSTERIOR_DATASET_CHAINS, _POSTERIOR_DATASET_DRAWS],
            dtype=types_pb2.DT_BOOL,
            bool_vals=[True]
            * (_POSTERIOR_DATASET_CHAINS * _POSTERIOR_DATASET_DRAWS),
        ),
    )
    expected_proto = meridian_pb.ModelConvergence(
        convergence=True,
        mcmc_sampling_trace=expected_sampling_trace,
        r_hat_diagnostic=meridian_pb.RHatDiagnostic(
            parameter_r_hats=[
                meridian_pb.Parameter(
                    name=constants.ALPHA_M,
                    tensor=test_data.make_tensor_proto(
                        dims=[1, 2, 3],
                        dtype=types_pb2.DT_FLOAT,
                        tensor_content=_R_HATS[constants.ALPHA_M]
                        .numpy()
                        .tobytes(),
                    ),
                ),
                meridian_pb.Parameter(
                    name=constants.BETA_GRF,
                    tensor=test_data.make_tensor_proto(
                        dims=[4, 5],
                        dtype=types_pb2.DT_FLOAT,
                        tensor_content=_R_HATS[constants.BETA_GRF]
                        .numpy()
                        .tobytes(),
                    ),
                ),
                meridian_pb.Parameter(
                    name=constants.TAU_G,
                    tensor=test_data.make_tensor_proto(
                        dims=[6],
                        dtype=types_pb2.DT_FLOAT,
                        tensor_content=_R_HATS[constants.TAU_G]
                        .numpy()
                        .tobytes(),
                    ),
                ),
            ]
        ),
    )

    serialized_model = self.serde.serialize(
        model.Meridian(
            input_data=_INPUT_DATA,
            model_spec=spec.ModelSpec(knots=49),
            inference_data=az.InferenceData(
                prior=_PRIOR_DATASET,
                posterior=_POSTERIOR_DATASET,
                trace=_POSTERIOR_DATASET,
            ),
        ),
        'test_model',
        include_convergence_info=True,
    )
    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)

    compare.assertProto2Equal(
        self,
        expected_proto,
        unpacked_model.convergence_info,
    )

  def test_deserialize_not_meridian_model(self):
    not_meridian_model_proto = marketing_pb.MarketingData()
    any_model = any_pb2.Any()
    any_model.Pack(not_meridian_model_proto)

    with self.assertRaisesRegex(
        ValueError, '`serialized.model` is not a `MeridianModel`'
    ):
      self.serde.deserialize(kernel_pb.MmmKernel(model=any_model))

  def test_deserialize(self):
    meridian_model = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_distributions=meridian_pb.PriorDistributions(),
        inference_data=meridian_pb.InferenceData(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model)
    mmm_kernel = kernel_pb.MmmKernel(
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
        model=any_model,
    )

    deserialized_model = self.serde.deserialize(mmm_kernel)

    self.assertIsInstance(deserialized_model, model.Meridian)
    self.assertIsInstance(
        deserialized_model.input_data, meridian_input_data.InputData
    )
    self.assertIsInstance(deserialized_model.inference_data, az.InferenceData)
    self.assertIsInstance(deserialized_model.model_spec, spec.ModelSpec)
    self.assertIsInstance(
        deserialized_model.model_spec.prior,
        prior_distribution.PriorDistribution,
    )

  def test_deserialize_sets_model_spec_prior(self):
    mock_version = '1.2.3'

    meridian_model = meridian_pb.MeridianModel(
        model_version=mock_version,
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_distributions=meridian_pb.PriorDistributions(),
        inference_data=meridian_pb.InferenceData(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model)
    mmm_kernel = kernel_pb.MmmKernel(
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
        model=any_model,
    )

    mock_hyperparameters_model_spec = test_data.DEFAULT_MODEL_SPEC
    mock_prior_distributions = prior_distribution.PriorDistribution()
    with (
        mock.patch.object(
            hyperparameters.HyperparametersSerde,
            'deserialize',
            return_value=mock_hyperparameters_model_spec,
        ) as mock_hyperparameters_deserialize,
        mock.patch.object(
            distribution.DistributionSerde,
            'deserialize',
            return_value=mock_prior_distributions,
        ) as mock_prior_distributions_deserialize,
    ):

      deserialized_model = self.serde.deserialize(mmm_kernel)

      mock_hyperparameters_deserialize.assert_called_once_with(
          meridian_model.hyperparameters, mock_version
      )
      mock_prior_distributions_deserialize.assert_called_once_with(
          meridian_model.prior_distributions, mock_version
      )

      self.assertEqual(
          deserialized_model.model_spec.prior, mock_prior_distributions
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          input_data=_INPUT_DATA,
          model_spec=None,
          inf_data=None,
      ),
      dict(
          testcase_name='defaults',
          input_data=_INPUT_DATA,
          model_spec=test_data.DEFAULT_MODEL_SPEC,
          inf_data=az.InferenceData(),
      ),
      dict(
          testcase_name='priors_and_posteriors',
          input_data=_INPUT_DATA,
          model_spec=spec.ModelSpec(knots=49),
          inf_data=az.InferenceData(
              prior=_PRIOR_DATASET,
              posterior=_POSTERIOR_DATASET,
          ),
      ),
  )
  def test_serialize_deserialize_round_trip(
      self,
      input_data: meridian_input_data.InputData,
      model_spec: spec.ModelSpec,
      inf_data: az.InferenceData,
  ):
    with (
        mock.patch.object(
            model.Meridian, '_validate_geo_invariants', autospec=True
        ),
    ):
      original_model = model.Meridian(
          input_data=input_data,
          model_spec=model_spec,
          inference_data=inf_data,
      )
      serialized_model = self.serde.serialize(original_model, 'test_model')
      deserialized_model = self.serde.deserialize(serialized_model)

      # Assertions for key attributes.  A comprehensive deep comparison
      # is done in the serde subclasses' tests; here, we check a subset of
      # important attributes to ensure the overall flow is correct.
      self.assertIsInstance(deserialized_model, model.Meridian)
      xrt.assert_allclose(
          original_model.input_data.population,
          deserialized_model.input_data.population,
          atol=0.5,
          rtol=0,
      )
      xrt.assert_allclose(
          original_model.input_data.kpi,
          deserialized_model.input_data.kpi,
      )
      xrt.assert_allclose(
          original_model.input_data.controls,
          deserialized_model.input_data.controls,
      )

      self.assertEqual(
          original_model.model_spec.knots, deserialized_model.model_spec.knots
      )
      self.assertEqual(
          original_model.model_spec.max_lag,
          deserialized_model.model_spec.max_lag,
      )

      if not hasattr(inf_data, constants.PRIOR):
        self.assertFalse(
            hasattr(deserialized_model.inference_data, constants.PRIOR)
        )
      else:
        self.assertEqual(
            original_model.inference_data.prior,
            deserialized_model.inference_data.prior,
        )

      if not hasattr(inf_data, constants.POSTERIOR):
        self.assertFalse(
            hasattr(deserialized_model.inference_data, constants.POSTERIOR)
        )
      else:
        self.assertEqual(
            original_model.inference_data.posterior,
            deserialized_model.inference_data.posterior,
        )

  def test_save_load_meridian_binpb(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, 'serde.binpb')

    meridian_model = model.Meridian(
        input_data=_INPUT_DATA,
        model_spec=test_data.DEFAULT_MODEL_SPEC,
    )

    serde.save_meridian(meridian_model, file_path)
    self.assertTrue(os.path.exists(file_path))

    loaded_model = serde.load_meridian(file_path)

    for attr in dir(meridian_model):
      if isinstance(getattr(meridian_model, attr), (int, bool)):
        with self.subTest(name=attr):
          self.assertEqual(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )
      elif isinstance(getattr(meridian_model, attr), tf.Tensor):
        with self.subTest(name=attr):
          self.assertAllClose(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )

  def test_save_load_meridian_txtpb(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, 'serde.txtpb')

    meridian_model = model.Meridian(
        input_data=_INPUT_DATA,
        model_spec=test_data.DEFAULT_MODEL_SPEC,
    )

    serde.save_meridian(meridian_model, file_path)
    self.assertTrue(os.path.exists(file_path))

    loaded_model = serde.load_meridian(file_path)

    for attr in dir(meridian_model):
      if isinstance(getattr(meridian_model, attr), (int, bool)):
        with self.subTest(name=attr):
          self.assertEqual(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )
      elif isinstance(getattr(meridian_model, attr), tf.Tensor):
        with self.subTest(name=attr):
          self.assertAllClose(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )


if __name__ == '__main__':
  absltest.main()
