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

"""Module for processing a trained Meridian model into an MMM schema."""

import abc

import meridian
from meridian.model import model
from proto.mmm.v1 import mmm_pb2 as pb
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
