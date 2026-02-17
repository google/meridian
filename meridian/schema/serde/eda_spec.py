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

"""Serialization and deserialization of `EDASpec` objects."""

from __future__ import annotations

import warnings

from meridian.model.eda import eda_spec
from mmm.v1.model.meridian.eda import eda_spec_pb2 as eda_spec_pb
from meridian.schema.serde import function_registry as function_registry_utils
from meridian.schema.serde import serde


FunctionRegistry = function_registry_utils.FunctionRegistry
_FUNCTION_REGISTRY_NAME = "function_registry"


class EDASpecSerde(serde.Serde[eda_spec_pb.EDASpec, eda_spec.EDASpec]):
  """Serializes and deserializes an `EDASpec` object into an `EDASpec` proto."""

  def __init__(self, function_registry: FunctionRegistry):
    """Initializes an `EDASpecSerde` instance.

    Args:
      function_registry: A lookup table containing custom functions used by
        `EDASpec` objects. It's recommended to explicitly define the custom
        functions instead of using lambdas, as lambda functions may not be
        serialized successfully.
    """
    self._function_registry = function_registry

  @property
  def function_registry(self) -> FunctionRegistry:
    return self._function_registry

  def serialize(self, obj: eda_spec.EDASpec) -> eda_spec_pb.EDASpec:
    """Serializes the given `EDASpec` object into an `EDASpec` proto."""
    proto = eda_spec_pb.EDASpec(
        aggregation_config=self._to_aggregation_config_proto(
            obj.aggregation_config
        ),
        vif_spec=self._to_vif_spec_proto(obj.vif_spec),
    )
    hashed_function_registry = self.function_registry.hashed_registry
    proto.function_registry.update(hashed_function_registry)
    return proto

  def deserialize(
      self,
      serialized: eda_spec_pb.EDASpec,
      serialized_version: str = "",
      force_deserialization: bool = False,
  ) -> eda_spec.EDASpec:
    """Deserializes the `EDASpec` proto.

    Args:
      serialized: A serialized `EDASpec` object.
      serialized_version: The version of the serialized Meridian model. This is
        used to handle changes in deserialization logic across different
        versions.
      force_deserialization: If True, bypasses the safety check that validates
        whether functions within `function_registry` have changed after
        serialization. Use with caution.

    Returns:
      A deserialized `EDASpec` object.
    """
    if force_deserialization:
      warnings.warn(
          "You're attempting to deserialize an EDASpec while ignoring changes"
          " to custom functions. This is a risky operation that can"
          " potentially lead to a deserialized EDASpec that behaves"
          " differently from the original, resulting in unexpected behavior."
          " Please proceed with caution."
      )
    else:
      hashed_function_registry = getattr(serialized, _FUNCTION_REGISTRY_NAME)
      try:
        self.function_registry.validate(hashed_function_registry)
      except Exception as e:
        raise ValueError(
            f"An issue found during deserializing EDASpec: {e}"
        ) from e

    aggregation_config = self._from_aggregation_config_proto(
        serialized.aggregation_config
    )
    vif_spec = self._from_vif_spec_proto(serialized.vif_spec)
    return eda_spec.EDASpec(
        aggregation_config=aggregation_config,
        vif_spec=vif_spec,
    )

  def _to_aggregation_config_proto(
      self, config: eda_spec.AggregationConfig
  ) -> eda_spec_pb.AggregationConfig:
    """Converts a Python `AggregationConfig` to a proto."""
    proto = eda_spec_pb.AggregationConfig()
    if config.control_variables:
      for key, func in config.control_variables.items():
        proto.control_variables[key].CopyFrom(
            self._to_aggregation_function_proto(func, key, "control_variables")
        )
    if config.non_media_treatments:
      for key, func in config.non_media_treatments.items():
        proto.non_media_treatments[key].CopyFrom(
            self._to_aggregation_function_proto(
                func, key, "non_media_treatments"
            )
        )
    return proto

  def _to_aggregation_function_proto(
      self, func: eda_spec.AggregationFn, key: str, field: str
  ) -> eda_spec_pb.AggregationFunction:
    """Converts a Python aggregation function to a proto."""
    function_key = self.function_registry.get_function_key(func)
    if function_key is not None:
      return eda_spec_pb.AggregationFunction(function_key=function_key)

    raise ValueError(
        f"Custom aggregation function `{key}` in `{field}` detected, but not"
        " found in registry. Please add custom functions to registry when"
        " saving models."
    )

  def _from_aggregation_config_proto(
      self, proto: eda_spec_pb.AggregationConfig
  ) -> eda_spec.AggregationConfig:
    """Converts a proto `AggregationConfig` to a Python object."""
    control_variables = {
        key: self._from_aggregation_function_proto(key, val)
        for key, val in proto.control_variables.items()
    }
    non_media_treatments = {
        key: self._from_aggregation_function_proto(key, val)
        for key, val in proto.non_media_treatments.items()
    }
    return eda_spec.AggregationConfig(
        control_variables=control_variables,
        non_media_treatments=non_media_treatments,
    )

  def _from_aggregation_function_proto(
      self, var_name: str, agg_func: eda_spec_pb.AggregationFunction
  ) -> eda_spec.AggregationFn:
    """Converts a proto `AggregationFunction` to a Python function."""
    if not agg_func.function_key:
      raise ValueError(
          "Function key is required in `AggregationFunction` proto message."
          f" The function key for {var_name} is empty."
      )

    if agg_func.function_key in self.function_registry:
      return self.function_registry[agg_func.function_key]

    raise ValueError(
        f"Function key `{agg_func.function_key}` not found in registry."
    )

  def _to_vif_spec_proto(
      self, vif_spec_obj: eda_spec.VIFSpec
  ) -> eda_spec_pb.VIFSpec:
    """Converts a Python `VIFSpec` to a proto."""
    return eda_spec_pb.VIFSpec(
        geo_threshold=vif_spec_obj.geo_threshold,
        overall_threshold=vif_spec_obj.overall_threshold,
        national_threshold=vif_spec_obj.national_threshold,
    )

  def _from_vif_spec_proto(
      self, vif_spec_proto: eda_spec_pb.VIFSpec
  ) -> eda_spec.VIFSpec:
    """Converts a proto `VIFSpec` to a Python object."""
    return eda_spec.VIFSpec(
        geo_threshold=vif_spec_proto.geo_threshold,
        overall_threshold=vif_spec_proto.overall_threshold,
        national_threshold=vif_spec_proto.national_threshold,
    )
