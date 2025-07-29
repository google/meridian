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

"""Consumes a trained Meridian model and produces an `Mmm` proto.

The `Mmm` proto contains parts collected from the core model as well as
analysis results from trained model processors.
"""

from collections.abc import Mapping, Sequence
import functools
import inspect
from typing import Any, Generic, TypeAlias, TypeVar

from meridian.model import model
from mmm.v1 import mmm_pb2 as mmm_pb
from schema.processors import model_kernel_processor
from schema.processors import model_processor


SpecType: TypeAlias = type[model_processor.Spec]
ProcType = TypeVar("ProcType", bound=type[model_processor.ModelProcessor])


class ModelConsumer(Generic[ProcType]):
  """Consumes a trained Meridian model and produces an `Mmm` proto.

  Attributes:
    model_processors: A preset list of model processor types.
  """

  def __init__(
      self,
      model_processors_classes: Sequence[ProcType],
  ):
    self._model_processors_classes = model_processors_classes

  @functools.cached_property
  def specs_to_processors_classes(
      self,
  ) -> dict[SpecType, ProcType]:
    """Returns a mapping of spec types to their corresponding processor types.

    Raises:
      ValueError: If multiple model processors are found for the same spec type.
    """
    specs_to_processors_classes = {}
    for processor_class in self._model_processors_classes:
      if (
          specs_to_processors_classes.get(processor_class.spec_type())
          is not None
      ):
        raise ValueError(
            "Multiple model processors found for spec type:"
            f" {processor_class.spec_type()}"
        )
      specs_to_processors_classes[processor_class.spec_type()] = processor_class
    return specs_to_processors_classes

  def __call__(
      self,
      mmm: model.Meridian,
      specs: Sequence[model_processor.Spec],
      model_id: str = "",
  ) -> mmm_pb.Mmm:
    """Produces an `Mmm` schema for the model along with its analyses results.

    Args:
      mmm: A trained Meridian model. A trained model has its posterior
        distributions already sampled.
      specs: A sequence of specs that specify the analyses to run on the model.
        Specs of the same type will be grouped together and executed together by
        the corresponding model processor.
      model_id: An optional model identifier.

    Returns:
      A proto containing the model kernel at rest and its analysis results.
    """

    # Group specs by their type.
    specs_by_type = {}
    for spec in specs:
      specs_by_type.setdefault(spec.__class__, []).append(spec)

    tmodel = model_processor.TrainedModel(mmm)
    processor_params = {
        "trained_model": tmodel,
    }

    output = mmm_pb.Mmm()
    # Attach the model kernel to the Mmm proto.
    model_kernel_processor.ModelKernelProcessor(mmm, model_id)(output)

    # Perform analysis or optimization.
    for spec_type, specs in specs_by_type.items():
      processor_type = self.specs_to_processors_classes[spec_type]
      processor = _create_processor(processor_type, processor_params)
      # Attach the output of the processor to the output proto.
      processor(specs, output)

    return output


def _create_processor(
    processor_type: ProcType,
    processor_params: Mapping[str, Any],
) -> model_processor.ModelProcessor:
  """Creates a processor of the given type with a subset of the given params."""
  # Clone the given parameters dict first.
  params = dict(processor_params)
  # Remove any parameters that are not in the processor's constructor signature.
  sig = inspect.signature(processor_type.__init__)
  if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
    for missing in params.keys() - sig.parameters.keys():
      del params[missing]
  # Finally, construct the concrete processor.
  return processor_type(**params)
