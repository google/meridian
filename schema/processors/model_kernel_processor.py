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

"""Module for transforming a Meridian model into a structured MMM schema.

This module provides the `ModelKernelProcessor`, which is responsible for
transforming the internal state of a trained Meridian model object into a
structured and portable format defined by the `MmmKernel` protobuf message.

The "kernel" includes essential information about the model, such as:

-   Model specifications and hyperparameters.
-   Inferred parameters distributions (as a serialized ArViz inference data).
-   MMM-agnostic marketing data (i.e. input data to the model).

This serialized representation allows the model to be saved, loaded, and
analyzed across different environments or by other tools that understand the
`MmmKernel` schema.

The serialization logic is primarily handled by the `MeridianSerde` class from
the `schema.serde` package.

Key Classes:

-   `ModelKernelProcessor`: The processor class that takes a Meridian model
    instance and populates an `MmmKernel` message.

Example Usage:

```python
import meridian
from meridian.model import model
from mmm.v1 import mmm_pb2
from schema.processors import model_kernel_processor
import semver

# Assuming 'mmm' is a `meridian.model.Meridian` object.
# Example:
# mmm = meridian.model.Meridian(...)
# mmm.sample_prior(...)
# mmm.sample_posterior(...)

processor = model_kernel_processor.ModelKernelProcessor(
    meridian_model=mmm,
    model_id="my_model_v1",
)

# Create an output Mmm proto message
output_proto = mmm_pb2.Mmm()

# Populate the mmm_kernel field
processor(output_proto)

# Now output_proto.mmm_kernel contains the serialized model.
# This can be saved to a file, sent over a network, etc.
print(f"Model Kernel ID: {output_proto.mmm_kernel.model_id}")
print(f"Meridian Version: {output_proto.mmm_kernel.meridian_version}")
# Access other fields within output_proto.mmm_kernel as needed.
```
"""

import abc

import meridian
from meridian.model import model
from mmm.v1 import mmm_pb2 as pb
from schema.serde import meridian_serde
import semver


class ModelKernelProcessor(abc.ABC):
  """Transcribes a model's stats into an `"MmmKernel` message."""

  def __init__(
      self,
      meridian_model: model.Meridian,
      model_id: str = '',
      meridian_version: semver.VersionInfo = semver.VersionInfo.parse(
          meridian.__version__
      ),
  ):
    """Initializes this `ModelKernelProcessor` with a Meridian model.

    Args:
      meridian_model: A Meridian model.
      model_id: An optional model identifier unique to the given model.
      meridian_version: The version of current Meridian framework.
    """
    self._meridian = meridian_model
    self._model_id = model_id
    self._meridian_version = meridian_version

  def __call__(self, output: pb.Mmm):
    """Sets `mmm_kernel` field in the given `Mmm` proto.

    Args:
      output: The output proto to modify.
    """
    output.mmm_kernel.CopyFrom(
        meridian_serde.MeridianSerde().serialize(
            self._meridian,
            self._model_id,
            self._meridian_version,
            include_convergence_info=True,
        )
    )
