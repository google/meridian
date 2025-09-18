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

from google.protobuf import text_format
import meridian
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.model import model
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import distribution
from schema.serde import hyperparameters
from schema.serde import inference_data
from schema.serde import marketing_data
from schema.serde import serde
import semver
import tensorflow as tf

from google.protobuf import any_pb2


_VERSION_INFO = semver.VersionInfo.parse(meridian.__version__)


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
  ) -> kernel_pb.MmmKernel:
    """Serializes the given Meridian model into an `MmmKernel` proto.

    Args:
      obj: The Meridian model to serialize.
      model_id: The ID of the model.
      meridian_version: The version of the Meridian model.
      include_convergence_info: Whether to include convergence information.

    Returns:
      An `MmmKernel` proto representing the serialized model.
    """
    meridian_model_proto = self._make_meridian_model_proto(
        obj, model_id, meridian_version, include_convergence_info
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
      include_convergence_info: bool = False,
  ) -> meridian_pb.MeridianModel:
    """Constructs a MeridianModel proto from the TrainedModel.

    Args:
      mmm: Meridian model.
      model_id: The ID of the model.
      meridian_version: The version of the Meridian model.
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
        prior_distributions=distribution.DistributionSerde().serialize(
            mmm.model_spec.prior
        ),
        inference_data=inference_data.InferenceDataSerde().serialize(
            mmm.inference_data
        ),
        kpi_scaled=tf.make_tensor_proto(mmm.kpi_scaled),
    )

    if mmm.controls_scaled is not None:
      model_proto.controls_scaled.CopyFrom(
          tf.make_tensor_proto(mmm.controls_scaled)
      )

    media_tensors = mmm.media_tensors
    rf_tensors = mmm.rf_tensors
    if media_tensors.media_scaled is not None:
      model_proto.media_scaled.CopyFrom(
          tf.make_tensor_proto(media_tensors.media_scaled)
      )
    if rf_tensors.reach_scaled is not None:
      model_proto.reach_scaled.CopyFrom(
          tf.make_tensor_proto(rf_tensors.reach_scaled)
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
            name=name, tensor=tf.make_tensor_proto(tensor)
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
          step_size=tf.make_tensor_proto(trace.step_size),
          tune=tf.make_tensor_proto(trace.tune),
          target_log_prob=tf.make_tensor_proto(trace.target_log_prob),
          diverging=tf.make_tensor_proto(trace.diverging),
          accept_ratio=tf.make_tensor_proto(trace.accept_ratio),
          n_steps=tf.make_tensor_proto(trace.n_steps),
          is_accepted=tf.make_tensor_proto(trace.is_accepted),
      )
      model_convergence_proto.mcmc_sampling_trace.CopyFrom(mcmc_sampling_trace)

    return model_convergence_proto

  def deserialize(
      self, serialized: kernel_pb.MmmKernel, serialized_version: str = ''
  ) -> model.Meridian:
    """Deserializes the given `MmmKernel` proto into a Meridian model."""
    ser_meridian = meridian_pb.MeridianModel()
    if not serialized.model.Is(meridian_pb.MeridianModel.DESCRIPTOR):
      raise ValueError('`serialized.model` is not a `MeridianModel`.')
    serialized.model.Unpack(ser_meridian)
    serialized_version = semver.VersionInfo.parse(ser_meridian.model_version)

    deserialized_hyperparameters = (
        hyperparameters.HyperparametersSerde().deserialize(
            ser_meridian.hyperparameters, str(serialized_version)
        )
    )
    deserialized_prior_distributions = (
        distribution.DistributionSerde().deserialize(
            ser_meridian.prior_distributions, str(serialized_version)
        )
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

    return model.Meridian(
        input_data=deserialized_marketing_data,
        model_spec=deserialized_model_spec,
        inference_data=deserialized_inference_data,
    )


def save_meridian(mmm: model.Meridian, file_path: str):
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
  """
  if not _file_exists(os.path.dirname(file_path)):
    _make_dirs(os.path.dirname(file_path))

  with _file_open(file_path, 'wb') as f:
    serialized_kernel = MeridianSerde().serialize(mmm)  # Creates an MmmKernel.
    if file_path.endswith('.binpb'):
      f.write(serialized_kernel.SerializeToString())
    elif file_path.endswith('.textproto') or file_path.endswith('.txtpb'):
      f.write(text_format.MessageToString(serialized_kernel))
    else:
      raise ValueError(f'Unsupported file type: {file_path}')


def load_meridian(file_path: str) -> model.Meridian:
  """Load the model object from an `MmmKernel` proto file path.

  Supported file types:
    - `binpb` (wire-format proto)
    - `txtpb` (text-format proto)
    - `textproto` (text-format proto)

  Args:
    file_path: File path to load a serialized model object from.

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
  return MeridianSerde().deserialize(serialized_model)
