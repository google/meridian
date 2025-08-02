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

"""Generates an `Mmm` (Marketing Mix Model) proto for Meridian.

The MMM proto schema contains parts collected from the core model as well as
analysis results from trained model processors.
"""

from collections.abc import Sequence
from typing import TypeVar

from meridian.model import model
from mmm.v1 import mmm_pb2 as mmm_pb
from schema import model_consumer
from schema.processors import budget_optimization_processor
from schema.processors import marketing_processor
from schema.processors import model_fit_processor
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor


__all__ = [
    "create_mmm_proto",
]


_TYPES = (
    model_fit_processor.ModelFitProcessor,
    marketing_processor.MarketingProcessor,
    budget_optimization_processor.BudgetOptimizationProcessor,
    reach_frequency_optimization_processor.ReachFrequencyOptimizationProcessor,
)

SpecType = TypeVar("SpecType", bound=model_processor.Spec)
DatedSpecType = TypeVar("DatedSpecType", bound=model_processor.DatedSpec)
OptimizationSpecType = TypeVar(
    "OptimizationSpecType", bound=model_processor.OptimizationSpec
)


def create_mmm_proto(
    mmm: model.Meridian,
    specs: Sequence[SpecType],
    model_id: str = "",
) -> mmm_pb.Mmm:
  """Creates a model schema and analyses for various time buckets.

  Args:
    mmm: A trained Meridian model. A trained model has its posterior
      distributions already sampled.
    specs: A sequence of specs that specify the analyses to run on the model.
    model_id: An optional model identifier.

  Returns:
    A proto containing the model kernel at rest and its analysis results given
    user specs.
  """
  consumer = model_consumer.ModelConsumer(_TYPES)
  return consumer(mmm, specs, model_id)
