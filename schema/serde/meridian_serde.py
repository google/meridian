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

"""Serialization and deserialization of Meridian models into/from proto format.

The `meridian_serde.MeridianSerde` class provides an interface for serializing
and deserializing Meridian models into and from an `MmmKernel` proto message.

The Meridian model--when serialized into an `MmmKernel` proto--is internally
represented as the sum of the following components:

1. Marketing data: This includes the KPI, media, and control data present in
   the input data. They are structured into an MMM-agnostic `MarketingData`
   proto message.
2. Meridian model: A `MeridianModel` proto message encapsulates
   Meridian-specific model parameters, including hyperparameters, prior
   distributions, and sampled inference data.

Sample usage:

```python
from schema.serde import meridian_serde

serde = meridian_serde.MeridianSerde()
mmm = model.Meridian(...)
serialized_mmm = serde.serialize(mmm)  # An `MmmKernel` proto
deserialized_mmm = serde.deserialize(serialized_mmm)  # A `Meridian` object
```
"""

import dataclasses
import os
import warnings

from google.protobuf import text_format
import meridian
from meridian import backend
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.model import model
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import distribution
from schema.serde import eda_spec as eda_spec_serde
from schema.serde import function_registry as function_registry_utils
from schema.serde import hyperparameters
from schema.serde import inference_data
from schema.serde import marketing_data
from schema.serde import serde
import semver

from google.protobuf import any_pb2


_VERSION_INFO = semver.VersionInfo.parse(meridian.__version__)

FunctionRegistry = function_registry_utils.FunctionRegistry

_file_exists = os.path.exists
_make_dirs = os.makedirs
_file_open = open


class MeridianSerde(serde.Serde[kernel_pb.MmmKernel, model.Meridian]):
  """Serializes and deserializes a Meridian model into an `MmmKernel` proto."""

  def serialize(
      self,
      obj: model.Meridian,
      model_id: str = '',
      meridian_version: semver.VersionInfo = _VERSION_INFO,
      include_convergence_info: bool = False,
      distribution_function_registry: FunctionRegistry | None = None,
      eda_function_registry: FunctionRegistry | None = None,
  ) -> kernel_pb.MmmKernel:
    """Serializes the given Meridian model into an `MmmKernel` proto.

    Args:
      obj: The Meridian model to serialize.
      model_id: The ID of the model.
      meridian_version: The version of the Meridian model.
      include_convergence_info: Whether to include convergence information.
      distribution_function_registry: Optional. A lookup table that maps string
        keys to custom functions to be used as parameters in various
        `tfp.distributions`.
      eda_function_registry: A lookup table that maps string keys to custom
        functions to be used in `EDASpec`.

    Returns:
      An `MmmKernel` proto representing the serialized model.
    """
    distribution_registry = (
        distribution_function_registry
        if distribution_function_registry is not None
        else function_registry_utils.FunctionRegistry()
    )

    eda_function_registry = (
        eda_function_registry
        if eda_function_registry is not None
        else function_registry_utils.FunctionRegistry()
    )
    meridian_model_proto = self._make_meridian_model_proto(
        mmm=obj,
        model_id=model_id,
        meridian_version=meridian_version,
        distribution_function_registry=distribution_registry,
        eda_function_registry=eda_function_registry,
        include_convergence_info=include_convergence_info,
    )
    any_model = any_pb2.Any()
    any_model.Pack(meridian_model_proto)
    return kernel_pb.MmmKernel(
        marketing_data=marketing_data.MarketingDataSerde().serialize(
            obj.input_data
        ),
        model=any_model,
    )

  def _make_meridian_model_proto(
      self,
      mmm: model.Meridian,
      model_id: str,
      meridian_version: semver.VersionInfo,
      distribution_function_registry: FunctionRegistry,
      eda_function_registry: FunctionRegistry,
      include_convergence_info: bool = False,
  ) -> meridian_pb.MeridianModel:
    """Constructs a MeridianModel proto from the TrainedModel.

    Args:
      mmm: Meridian model.
      model_id: The ID of the model.
      meridian_version: The version of the Meridian model.
      distribution_function_registry: A lookup table that maps string keys to
        custom functions to be used as parameters in various
        `tfp.distributions`.
      eda_function_registry: A lookup table containing custom functions used by
        `EDASpec` objects.
      include_convergence_info: Whether to include convergence information.

    Returns:
      A MeridianModel proto.
    """
    model_proto = meridian_pb.MeridianModel(
        model_id=model_id,
        model_version=str(meridian_version),
        hyperparameters=hyperparameters.HyperparametersSerde().serialize(
            mmm.model_spec
        ),
        prior_tfp_distributions=distribution.DistributionSerde(
            distribution_function_registry
        ).serialize(mmm.model_spec.prior),
        inference_data=inference_data.InferenceDataSerde().serialize(
            mmm.inference_data
        ),
    )
    # For backwards compatibility, only serialize EDA spec if it exists.
    if hasattr(mmm, 'eda_spec'):
      model_proto.eda_spec.CopyFrom(
          eda_spec_serde.EDASpecSerde(eda_function_registry).serialize(
              mmm.eda_spec
          )
      )

    if include_convergence_info:
      convergence_proto = self._make_model_convergence_proto(mmm)
      if convergence_proto is not None:
        model_proto.convergence_info.CopyFrom(convergence_proto)

    return model_proto

  def _make_model_convergence_proto(
      self, mmm: model.Meridian
  ) -> meridian_pb.ModelConvergence | None:
    """Creates ModelConvergence proto."""
    model_convergence_proto = meridian_pb.ModelConvergence()
    try:
      # NotFittedModelError can be raised below. If raised,
      # return None. Otherwise, set convergence status based on
      # MCMCSamplingError (caught in the except block).
      rhats = analyzer.Analyzer(mmm).get_rhat()
      rhat_proto = meridian_pb.RHatDiagnostic()
      for name, tensor in rhats.items():
        rhat_proto.parameter_r_hats.add(
            name=name, tensor=backend.make_tensor_proto(tensor)
        )
      model_convergence_proto.r_hat_diagnostic.CopyFrom(rhat_proto)

      visualizer.ModelDiagnostics(mmm).plot_rhat_boxplot()
      model_convergence_proto.convergence = True
    except model.MCMCSamplingError:
      model_convergence_proto.convergence = False
    except model.NotFittedModelError:
      return None

    if hasattr(mmm.inference_data, 'trace'):
      trace = mmm.inference_data.trace
      mcmc_sampling_trace = meridian_pb.McmcSamplingTrace(
          num_chains=len(trace.chain),
          num_draws=len(trace.draw),
          step_size=backend.make_tensor_proto(trace.step_size),
          tune=backend.make_tensor_proto(trace.tune),
          target_log_prob=backend.make_tensor_proto(trace.target_log_prob),
          diverging=backend.make_tensor_proto(trace.diverging),
          accept_ratio=backend.make_tensor_proto(trace.accept_ratio),
          n_steps=backend.make_tensor_proto(trace.n_steps),
          is_accepted=backend.make_tensor_proto(trace.is_accepted),
      )
      model_convergence_proto.mcmc_sampling_trace.CopyFrom(mcmc_sampling_trace)

    return model_convergence_proto

  def deserialize(
      self,
      serialized: kernel_pb.MmmKernel,
      serialized_version: str = '',
      distribution_function_registry: FunctionRegistry | None = None,
      eda_function_registry: FunctionRegistry | None = None,
      force_deserialization=False,
  ) -> model.Meridian:
    """Deserializes the given `MmmKernel` proto into a Meridian model.

    Args:
      serialized: The serialized object in the form of an `MmmKernel` proto.
      serialized_version: The version of the serialized model. This is used to
        handle changes in deserialization logic across different versions.
      distribution_function_registry: Optional. A lookup table that maps string
        keys to custom functions to be used as parameters in various
        `tfp.distributions`.
      eda_function_registry: A lookup table containing custom functions used by
        `EDASpec` objects.
      force_deserialization: If True, bypasses the safety check that validates
        whether functions within a function registry have changed after
        serialization. Use with caution. This should only be used if you have
        intentionally modified a custom function and are confident that the
        changes will not affect the deserialized model. A safer alternative is
        to first deserialize the model with the original functions and then
        serialize it with the new ones.

    Returns:
      A Meridian model object.
    """
    if serialized.model.Is(meridian_pb.MeridianModel.DESCRIPTOR):
      ser_meridian = meridian_pb.MeridianModel()
    else:
      raise ValueError('`serialized.model` is not a `MeridianModel`.')
    serialized.model.Unpack(ser_meridian)
    serialized_version = semver.VersionInfo.parse(ser_meridian.model_version)

    deserialized_hyperparameters = (
        hyperparameters.HyperparametersSerde().deserialize(
            ser_meridian.hyperparameters, str(serialized_version)
        )
    )

    if ser_meridian.HasField('prior_distributions'):
      ser_meridian_priors = ser_meridian.prior_distributions
    elif ser_meridian.HasField('prior_tfp_distributions') and isinstance(
        ser_meridian, meridian_pb.MeridianModel
    ):
      ser_meridian_priors = ser_meridian.prior_tfp_distributions
    else:
      raise ValueError('MeridianModel does not contain any priors.')

    deserialized_prior_distributions = distribution.DistributionSerde(
        distribution_function_registry
        if distribution_function_registry is not None
        else function_registry_utils.FunctionRegistry()
    ).deserialize(
        ser_meridian_priors,
        str(serialized_version),
        force_deserialization=force_deserialization,
    )
    deserialized_marketing_data = (
        marketing_data.MarketingDataSerde().deserialize(
            serialized.marketing_data, str(serialized_version)
        )
    )
    deserialized_inference_data = (
        inference_data.InferenceDataSerde().deserialize(
            ser_meridian.inference_data, str(serialized_version)
        )
    )

    deserialized_model_spec = dataclasses.replace(
        deserialized_hyperparameters, prior=deserialized_prior_distributions
    )

    meridian_kwargs = dict(
        input_data=deserialized_marketing_data,
        model_spec=deserialized_model_spec,
        inference_data=deserialized_inference_data,
    )

    # For backwards compatibility, only deserialize EDA spec if it exists in the
    # serialized model. Otherwise, warn the user and create a model with default
    # EDA spec.
    if isinstance(
        ser_meridian, meridian_pb.MeridianModel
    ) and ser_meridian.HasField('eda_spec'):
      meridian_kwargs['eda_spec'] = eda_spec_serde.EDASpecSerde(
          eda_function_registry
          if eda_function_registry is not None
          else function_registry_utils.FunctionRegistry()
      ).deserialize(
          ser_meridian.eda_spec,
          str(serialized_version),
          force_deserialization=force_deserialization,
      )
    else:
      warnings.warn('MeridianModel does not contain an EDA spec.')

    return model.Meridian(**meridian_kwargs)


def save_meridian(
    mmm: model.Meridian,
    file_path: str,
    distribution_function_registry: FunctionRegistry | None = None,
    eda_function_registry: FunctionRegistry | None = None,
):
  """Save the model object as an `MmmKernel` proto in the given filepath.

  Supported file types:
    - `binpb` (wire-format proto)
    - `txtpb` (text-format proto)
    - `textproto` (text-format proto)

  Args:
    mmm: Model object to save.
    file_path: File path to save a serialized model object. If the file name
      ends with `.binpb`, it will be saved in the wire-format. If the filename
      ends with `.txtpb` or `.textproto`, it will be saved in the text-format.
    distribution_function_registry: Optional. A lookup table that maps string
      keys to custom functions to be used as parameters in various
      `tfp.distributions`.
    eda_function_registry: A lookup table that maps string keys to custom
      functions to be used in `EDASpec`.
  """
  if not _file_exists(os.path.dirname(file_path)):
    _make_dirs(os.path.dirname(file_path))

  with _file_open(file_path, 'wb') as f:
    # Creates an MmmKernel.
    serialized_kernel = MeridianSerde().serialize(
        mmm,
        distribution_function_registry=distribution_function_registry,
        eda_function_registry=eda_function_registry,
    )
    if file_path.endswith('.binpb'):
      f.write(serialized_kernel.SerializeToString())
    elif file_path.endswith('.textproto') or file_path.endswith('.txtpb'):
      f.write(text_format.MessageToString(serialized_kernel))
    else:
      raise ValueError(f'Unsupported file type: {file_path}')


def load_meridian(
    file_path: str,
    distribution_function_registry: FunctionRegistry | None = None,
    eda_function_registry: FunctionRegistry | None = None,
    force_deserialization=False,
) -> model.Meridian:
  """Load the model object from an `MmmKernel` proto file path.

  Supported file types:
    - `binpb` (wire-format proto)
    - `txtpb` (text-format proto)
    - `textproto` (text-format proto)

  Args:
    file_path: File path to load a serialized model object from.
    distribution_function_registry: A lookup table that maps string keys to
      custom functions to be used as parameters in various `tfp.distributions`.
    eda_function_registry: A lookup table that maps string keys to custom
      functions to be used in `EDASpec`.
    force_deserialization: If True, bypasses the safety check that validates
      whether functions within a function registry have changed after
      serialization. Use with caution. This should only be used if you have
      intentionally modified a custom function and are confident that the
      changes will not affect the deserialized model. A safer alternative is to
      first deserialize the model with the original functions and then serialize
      it with the new ones.

  Returns:
    Model object loaded from the file path.
  """
  with _file_open(file_path, 'rb') as f:
    if file_path.endswith('.binpb'):
      serialized_model = kernel_pb.MmmKernel.FromString(f.read())
    elif file_path.endswith('.textproto') or file_path.endswith('.txtpb'):
      serialized_model = kernel_pb.MmmKernel()
      text_format.Parse(f.read(), serialized_model)
    else:
      raise ValueError(f'Unsupported file type: {file_path}')
  return MeridianSerde().deserialize(
      serialized_model,
      distribution_function_registry=distribution_function_registry,
      eda_function_registry=eda_function_registry,
      force_deserialization=force_deserialization,
  )
