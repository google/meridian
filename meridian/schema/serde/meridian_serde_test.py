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

import os
from typing import Callable
from unittest import mock
import warnings

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.backend import config as backend_config
from meridian.backend import test_utils as backend_test_utils
from meridian.data import input_data as meridian_input_data
from meridian.data import test_utils
from meridian.model import context
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model.eda import eda_spec
from mmm.v1.marketing import marketing_data_pb2 as marketing_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from mmm.v1.model.meridian.eda import eda_spec_pb2
from meridian.schema.serde import distribution
from meridian.schema.serde import eda_spec as eda_spec_serde
from meridian.schema.serde import function_registry as function_registry_utils
from meridian.schema.serde import hyperparameters
from meridian.schema.serde import inference_data
from meridian.schema.serde import marketing_data
from meridian.schema.serde import meridian_serde as serde
from meridian.schema.serde import test_data
import numpy as np
import semver
import xarray.testing as xrt

from google.protobuf import any_pb2
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
_INPUT_DATA_NO_RPK = (
    test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
        n_geos=5,
        n_times=49,
        n_media_times=52,
        n_controls=2,
        n_media_channels=3,
        seed=1,
    )
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


class MeridianSerdeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serde = serde.MeridianSerde()

    self._mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, 'Analyzer', autospec=True)
    )
    self._mock_visualizer = self.enter_context(
        mock.patch.object(visualizer, 'ModelDiagnostics', autospec=True)
    )
    self._r_hats = {
        constants.ALPHA_M: backend.to_tensor(
            np.full((1, 2, 3), 22.0), dtype=backend.float32
        ),
        constants.BETA_GRF: backend.to_tensor(
            np.full((4, 5), 33.0), dtype=backend.float32
        ),
        constants.TAU_G: backend.to_tensor(
            np.full((6,), 44.0), dtype=backend.float32
        ),
    }

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          model_id='empty_model',
          meridian_version=semver.VersionInfo.parse('1.0.0'),
          input_data=_INPUT_DATA,
          model_spec_fn=lambda: None,
          inf_data_fn=lambda: None,
      ),
      dict(
          testcase_name='defaults',
          model_id='default_model',
          meridian_version=semver.VersionInfo.parse('1.0.1'),
          input_data=_INPUT_DATA,
          model_spec_fn=test_data.get_default_model_spec,
          inf_data_fn=az.InferenceData,
      ),
      dict(
          testcase_name='priors_and_posteriors',
          model_id='priors_and_posteriors_model',
          meridian_version=semver.VersionInfo.parse('1.1.0'),
          input_data=_INPUT_DATA,
          model_spec_fn=lambda: spec.ModelSpec(knots=49),
          inf_data_fn=lambda: az.InferenceData(
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
      model_spec_fn,
      inf_data_fn,
  ):
    meridian_model = model.Meridian(
        input_data=input_data,
        model_spec=model_spec_fn(),
        inference_data=inf_data_fn(),
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

  def test_serialize_sets_computation_backend(self):
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA, model_spec=test_data.get_default_model_spec()
    )
    with mock.patch.object(
        model.Meridian,
        'computation_backend',
        new_callable=mock.PropertyMock,
    ) as mock_backend:
      mock_backend.return_value = 'JAX'
      serialized_model = self.serde.serialize(meridian_model)

    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)
    self.assertEqual(
        unpacked_model.computation_backend, meridian_pb.ComputationBackend.JAX
    )

  def test_deserialize_warns_on_backend_mismatch(self):
    # Create a proto indicating it was trained with JAX.
    meridian_model = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        inference_data=meridian_pb.InferenceData(),
        computation_backend=meridian_pb.ComputationBackend.JAX,
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    # Force the current environment to be TENSORFLOW.
    with mock.patch.object(
        backend,
        'computation_backend',
        return_value=backend_config.ComputationBackend.TENSORFLOW,
    ):
      with self.assertWarnsRegex(
          UserWarning,
          'The model was trained using JAX, but the current backend is'
          ' TENSORFLOW',
      ):
        self.serde.deserialize(mmm_kernel)

  def test_deserialize_no_warning_on_backend_match(self):
    # Create a proto indicating it was trained with TENSORFLOW.
    meridian_model = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        inference_data=meridian_pb.InferenceData(),
        computation_backend=meridian_pb.ComputationBackend.TENSORFLOW,
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    # Force the current environment to be TENSORFLOW.
    with mock.patch.object(
        backend,
        'computation_backend',
        return_value=backend_config.ComputationBackend.TENSORFLOW,
    ):
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.serde.deserialize(mmm_kernel)
        # Ensure no backend-related warnings were raised.
        backend_warnings = [x for x in w if 'backend' in str(x.message).lower()]
        self.assertEmpty(backend_warnings)

  def test_serialize_no_controls(self):
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA_NO_CONTROLS,
        model_spec=test_data.get_default_model_spec(),
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

  def test_serialize_model_convergence_proto(self):
    self._mock_analyzer.return_value.get_rhat.return_value = self._r_hats

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
                        tensor_content=np.array(
                            self._r_hats[constants.ALPHA_M]
                        ).tobytes(),
                    ),
                ),
                meridian_pb.Parameter(
                    name=constants.BETA_GRF,
                    tensor=test_data.make_tensor_proto(
                        dims=[4, 5],
                        dtype=types_pb2.DT_FLOAT,
                        tensor_content=np.array(
                            self._r_hats[constants.BETA_GRF]
                        ).tobytes(),
                    ),
                ),
                meridian_pb.Parameter(
                    name=constants.TAU_G,
                    tensor=test_data.make_tensor_proto(
                        dims=[6],
                        dtype=types_pb2.DT_FLOAT,
                        tensor_content=np.array(
                            self._r_hats[constants.TAU_G]
                        ).tobytes(),
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

    backend_test_utils.assert_normalized_proto_equal(
        self,
        expected_proto,
        unpacked_model.convergence_info,
    )

  def test_serialize_model_without_eda_spec(self):
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA,
        model_spec=test_data.get_default_model_spec(),
        inference_data=az.InferenceData(),
    )
    delattr(meridian_model, '_eda_spec')

    serialized_model = self.serde.serialize(
        meridian_model, 'model_id', semver.VersionInfo.parse('1.0.0')
    )

    unpacked_model = meridian_pb.MeridianModel()
    serialized_model.model.Unpack(unpacked_model)
    self.assertFalse(unpacked_model.HasField('eda_spec'))

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
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        inference_data=meridian_pb.InferenceData(),
        eda_spec=eda_spec_pb2.EDASpec(),
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
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        inference_data=meridian_pb.InferenceData(),
        eda_spec=eda_spec_pb2.EDASpec(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model)
    mmm_kernel = kernel_pb.MmmKernel(
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
        model=any_model,
    )

    mock_hyperparameters_model_spec = test_data.get_default_model_spec()
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
        ) as mock_prior_tfp_distributions_deserialize,
    ):

      deserialized_model = self.serde.deserialize(mmm_kernel)

      mock_hyperparameters_deserialize.assert_called_once_with(
          meridian_model.hyperparameters, mock_version
      )
      mock_prior_tfp_distributions_deserialize.assert_called_once_with(
          meridian_model.prior_tfp_distributions,
          mock_version,
          force_deserialization=False,
      )

      self.assertEqual(
          deserialized_model.model_spec.prior, mock_prior_distributions
      )

  def test_serialize_with_default_distribution_registry(self):
    mock_distribution_serde = self.enter_context(
        mock.patch.object(distribution, 'DistributionSerde', autospec=True)
    )
    mock_distribution_serde.return_value.serialize.return_value = (
        meridian_pb.PriorTfpDistributions()
    )
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA, model_spec=test_data.get_default_model_spec()
    )
    self.serde.serialize(meridian_model)
    mock_distribution_serde.assert_called_once_with(
        function_registry_utils.FunctionRegistry()
    )

  def test_serialize_with_custom_distribution_registry(self):
    mock_distribution_serde = self.enter_context(
        mock.patch.object(distribution, 'DistributionSerde', autospec=True)
    )
    mock_distribution_serde.return_value.serialize.return_value = (
        meridian_pb.PriorTfpDistributions()
    )
    custom_registry = function_registry_utils.FunctionRegistry()
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA, model_spec=test_data.get_default_model_spec()
    )
    self.serde.serialize(
        meridian_model, distribution_function_registry=custom_registry
    )
    mock_distribution_serde.assert_called_once_with(custom_registry)

  def test_deserialize_with_default_distribution_registry(self):
    mock_distribution_serde = self.enter_context(
        mock.patch.object(distribution, 'DistributionSerde', autospec=True)
    )
    mock_distribution_serde.return_value.deserialize.return_value = (
        prior_distribution.PriorDistribution()
    )
    meridian_model_proto = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        eda_spec=eda_spec_pb2.EDASpec(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    self.serde.deserialize(mmm_kernel)

    mock_distribution_serde.assert_called_once_with(
        function_registry_utils.FunctionRegistry()
    )

  def test_deserialize_with_custom_distribution_registry(self):
    mock_distribution_serde = self.enter_context(
        mock.patch.object(distribution, 'DistributionSerde', autospec=True)
    )
    mock_distribution_serde.return_value.deserialize.return_value = (
        prior_distribution.PriorDistribution()
    )
    custom_registry = function_registry_utils.FunctionRegistry()
    meridian_model_proto = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        eda_spec=eda_spec_pb2.EDASpec(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    self.serde.deserialize(
        mmm_kernel, distribution_function_registry=custom_registry
    )

    mock_distribution_serde.assert_called_once_with(custom_registry)

  def test_serialize_with_default_eda_registry(self):
    mock_eda_spec_serde = self.enter_context(
        mock.patch.object(eda_spec_serde, 'EDASpecSerde', autospec=True)
    )
    mock_eda_spec_serde.return_value.serialize.return_value = (
        eda_spec_pb2.EDASpec()
    )
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA, model_spec=test_data.get_default_model_spec()
    )
    self.serde.serialize(meridian_model)
    mock_eda_spec_serde.assert_called_once_with(
        function_registry_utils.FunctionRegistry()
    )

  def test_serialize_with_custom_eda_registry(self):
    mock_eda_spec_serde = self.enter_context(
        mock.patch.object(eda_spec_serde, 'EDASpecSerde', autospec=True)
    )
    mock_eda_spec_serde.return_value.serialize.return_value = (
        eda_spec_pb2.EDASpec()
    )
    custom_registry = function_registry_utils.FunctionRegistry()
    meridian_model = model.Meridian(
        input_data=_INPUT_DATA, model_spec=test_data.get_default_model_spec()
    )
    self.serde.serialize(meridian_model, eda_function_registry=custom_registry)
    mock_eda_spec_serde.assert_called_once_with(custom_registry)

  def test_deserialize_with_default_eda_registry(self):
    mock_eda_spec_serde = self.enter_context(
        mock.patch.object(eda_spec_serde, 'EDASpecSerde', autospec=True)
    )
    meridian_model_proto = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        eda_spec=eda_spec_pb2.EDASpec(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    self.serde.deserialize(mmm_kernel)

    mock_eda_spec_serde.assert_called_once_with(
        function_registry_utils.FunctionRegistry()
    )

  def test_deserialize_with_custom_eda_registry(self):
    mock_eda_spec_serde = self.enter_context(
        mock.patch.object(eda_spec_serde, 'EDASpecSerde', autospec=True)
    )
    custom_registry = function_registry_utils.FunctionRegistry()
    meridian_model_proto = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        eda_spec=eda_spec_pb2.EDASpec(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    self.serde.deserialize(mmm_kernel, eda_function_registry=custom_registry)

    mock_eda_spec_serde.assert_called_once_with(custom_registry)

  def test_deserialize_without_eda_spec_warns(self):
    meridian_model_proto = meridian_pb.MeridianModel(
        model_version='1.2.3',
        hyperparameters=test_data.DEFAULT_HYPERPARAMETERS_PROTO,
        prior_tfp_distributions=meridian_pb.PriorTfpDistributions(),
        inference_data=meridian_pb.InferenceData(),
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    mmm_kernel = kernel_pb.MmmKernel(
        model=any_model,
        marketing_data=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
    )

    with self.assertWarnsRegex(
        UserWarning, 'MeridianModel does not contain an EDA spec.'
    ):
      deserialized_model = self.serde.deserialize(mmm_kernel)

    self.assertEqual(deserialized_model.eda_spec, eda_spec.EDASpec())

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          input_data=_INPUT_DATA,
          model_spec_fn=lambda: None,
          inf_data_fn=lambda: None,
      ),
      dict(
          testcase_name='defaults',
          input_data=_INPUT_DATA,
          model_spec_fn=test_data.get_default_model_spec,
          inf_data_fn=az.InferenceData,
      ),
      dict(
          testcase_name='priors_and_posteriors',
          input_data=_INPUT_DATA,
          model_spec_fn=lambda: spec.ModelSpec(knots=49),
          inf_data_fn=lambda: az.InferenceData(
              prior=_PRIOR_DATASET,
              posterior=_POSTERIOR_DATASET,
          ),
      ),
  )
  def test_serialize_deserialize_round_trip(
      self,
      input_data: meridian_input_data.InputData,
      model_spec_fn: Callable[[], spec.ModelSpec],
      inf_data_fn: Callable[[], az.InferenceData],
  ):
    model_spec = model_spec_fn()
    inf_data = inf_data_fn()
    with (
        mock.patch.object(
            context.ModelContext, '_validate_geo_invariants', autospec=True
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
        model_spec=test_data.get_default_model_spec(),
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
      elif isinstance(getattr(meridian_model, attr), backend.Tensor):
        with self.subTest(name=attr):
          backend_test_utils.assert_allclose(
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
        model_spec=test_data.get_default_model_spec(),
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
      elif isinstance(getattr(meridian_model, attr), backend.Tensor):
        with self.subTest(name=attr):
          backend_test_utils.assert_allclose(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )

  def test_save_load_total_media_prior_binpb(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, 'serde.binpb')

    meridian_model = model.Meridian(
        input_data=_INPUT_DATA_NO_RPK,
        model_spec=test_data.get_default_model_spec(),
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
      elif isinstance(getattr(meridian_model, attr), backend.Tensor):
        with self.subTest(name=attr):
          backend_test_utils.assert_allclose(
              getattr(meridian_model, attr), getattr(loaded_model, attr)
          )


if __name__ == '__main__':
  absltest.main()
