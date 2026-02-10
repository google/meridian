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

"""Serialization and deserialization of `Distribution` objects for priors."""

from __future__ import annotations

import inspect
import types
from typing import Any, Sequence, TypeVar
import warnings

from meridian import backend
from meridian import constants
from meridian.model import prior_distribution as pd
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import constants as sc
from schema.serde import function_registry as function_registry_utils
from schema.serde import serde

from tensorflow.core.framework import tensor_shape_pb2  # pylint: disable=g-direct-tensorflow-import

FunctionRegistry = function_registry_utils.FunctionRegistry


MeridianPriorDistributions = (
    meridian_pb.PriorTfpDistributions
)

_FUNCTION_REGISTRY_NAME = "function_registry"


# TODO: b/436637084 - Delete enumerated schema.
class DistributionSerde(
    serde.Serde[MeridianPriorDistributions, pd.PriorDistribution]
):
  """Serializes and deserializes a Meridian prior distributions container into a `Distribution` proto."""

  def __init__(self, function_registry: FunctionRegistry):
    """Initializes a `DistributionSerde` instance.

    Args:
      function_registry: A lookup table containing custom functions used by
        various `backend.tfd` classes. It's recommended to explicitly define the
        custom functions instead of using lambdas, as lambda functions may not
        be registered successfully.
    """
    self._function_registry = function_registry

  @property
  def function_registry(self) -> FunctionRegistry:
    return self._function_registry

  def serialize(
      self, obj: pd.PriorDistribution
  ) -> meridian_pb.PriorTfpDistributions:
    """Serializes the given Meridian priors container into a `MeridianPriorDistributions` proto."""
    proto = meridian_pb.PriorTfpDistributions()
    for param in constants.ALL_PRIOR_DISTRIBUTION_PARAMETERS:
      if not hasattr(obj, param):
        continue
      getattr(proto, param).CopyFrom(
          self._to_distribution_proto(getattr(obj, param))
      )
    proto.function_registry.update(self.function_registry.hashed_registry)
    return proto

  def deserialize(
      self,
      serialized: MeridianPriorDistributions,
      serialized_version: str = "",
      force_deserialization: bool = False,
  ) -> pd.PriorDistribution:
    """Deserializes the `PriorTfpDistributions` proto.

    WARNING: If any custom functions in the function registry are modified after
    serialization, the deserialized model can differ from the original model, as
    the original function's behavior is no longer guaranteed. This will result
    in an error during deserialization.

    For users who are intentionally changing functions and are confident that
    the changes will not affect the deserialized model, you can bypass safety
    mechanisms to force deserialization. See example:

    Args:
      serialized: A serialized `PriorDistributions` object.
      serialized_version: The version of the serialized Meridian model. This is
        used to handle changes in deserialization logic across different
        versions.
      force_deserialization: If True, bypasses the safety check that validates
        whether functions within `function_registry` have changed after
        serialization. Use with caution. This should only be used if you have
        intentionally modified a custom function and are confident that the
        changes will not affect the deserialized model. A safer alternative is
        to first deserialize the model with the original functions and then
        serialize it with the new ones.

    Returns:
      A deserialized `PriorDistribution` object.
    """
    kwargs = {}
    for param in constants.ALL_PRIOR_DISTRIBUTION_PARAMETERS:
      if not hasattr(serialized, param):
        continue
      # A parameter may be unspecified in a serialized proto message because:
      # (1) It is left unset for Meridian to set its default value.
      # (2) The message was created from a previous Meridian version after
      #     introducing a new parameter.
      if not serialized.HasField(param):
        continue
      param_name = getattr(serialized, param)
      if isinstance(serialized, meridian_pb.PriorTfpDistributions):
        if force_deserialization:
          warnings.warn(
              "You're attempting to deserialize a model while ignoring changes"
              " to custom functions. This is a risky operation that can"
              " potentially lead to a deserialized model that behaves"
              " differently from the original, resulting in unexpected behavior"
              " or model failure. We strongly recommend a safer two-step"
              " process: deserialize the model using the original function"
              " registry and reserialize the model using the updated registry."
              " Please proceed with caution."
          )
        else:
          stored_hashed_function_registry = getattr(
              serialized, _FUNCTION_REGISTRY_NAME
          )
          try:
            self.function_registry.validate(stored_hashed_function_registry)
          except ValueError as e:
            raise ValueError(
                f"An issue found during deserializing Distribution: {e}"
            ) from e

        kwargs[param] = self._from_distribution_proto(param_name)
      # copybara: strip_begin(legacy proto)
      elif isinstance(serialized, meridian_pb.PriorDistributions):
        kwargs[param] = _from_legacy_distribution_proto(param_name)
      # copybara: strip_end
    return pd.PriorDistribution(**kwargs)

  def _to_distribution_proto(
      self,
      dist: backend.tfd.Distribution,
  ) -> meridian_pb.TfpDistribution:
    """Converts a TensorFlow `Distribution` object to a `TfpDistribution` proto."""
    dist_name = type(dist).__name__
    dist_class = getattr(backend.tfd, dist_name)
    return meridian_pb.TfpDistribution(
        distribution_type=dist_name,
        parameters={
            name: self._to_parameter_value_proto(name, value, dist_class)
            for name, value in dist.parameters.items()
        },
    )

  def _to_bijector_proto(
      self,
      bijector: backend.bijectors.Bijector,
  ) -> meridian_pb.TfpBijector:
    """Converts a TensorFlow `Bijector` object to a `TfpBijector` proto."""
    bij_name = type(bijector).__name__
    bij_class = getattr(backend.bijectors, bij_name)
    return meridian_pb.TfpBijector(
        bijector_type=bij_name,
        parameters={
            name: self._to_parameter_value_proto(name, value, bij_class)
            for name, value in bijector.parameters.items()
        },
    )

  def _to_parameter_value_proto(
      self,
      param_name: str,
      value: Any,
      dist: backend.tfd.Distribution | backend.bijectors.Bijector,
  ) -> meridian_pb.TfpParameterValue:
    """Converts a TensorFlow `Distribution` parameter value to a `TfpParameterValue` proto."""
    # Handle built-in types.
    match value:
      case float():
        return meridian_pb.TfpParameterValue(scalar_value=value)
      case int():
        return meridian_pb.TfpParameterValue(int_value=value)
      # TODO: b/470407198 - case bool() has to be before int() because bool is a
      # subtype of int.
      case bool():
        return meridian_pb.TfpParameterValue(bool_value=value)
      case str():
        return meridian_pb.TfpParameterValue(string_value=value)
      case None:
        return meridian_pb.TfpParameterValue(none_value=True)
      case list():
        value_generator = (
            self._to_parameter_value_proto(param_name, v, dist) for v in value
        )
        return meridian_pb.TfpParameterValue(
            list_value=meridian_pb.TfpParameterValue.List(
                values=value_generator
            )
        )
      case dict():
        dict_value = {
            k: self._to_parameter_value_proto(param_name, v, dist)
            for k, v in value.items()
        }
        return meridian_pb.TfpParameterValue(
            dict_value=meridian_pb.TfpParameterValue.Dict(value_map=dict_value)
        )
      case backend.tfd.Distribution():
        return meridian_pb.TfpParameterValue(
            distribution_value=self._to_distribution_proto(value)
        )
      case backend.bijectors.Bijector():
        return meridian_pb.TfpParameterValue(
            bijector_value=self._to_bijector_proto(value)
        )
      case backend.tfd.ReparameterizationType():
        fully_reparameterized = value == backend.tfd.FULLY_REPARAMETERIZED
        return meridian_pb.TfpParameterValue(
            fully_reparameterized=fully_reparameterized
        )
      case types.FunctionType():
        # Check for default value
        signature = inspect.signature(dist.__init__)
        param = signature.parameters[param_name]
        if param.default and param.default is value:
          return meridian_pb.TfpParameterValue(
              function_param=meridian_pb.TfpParameterValue.FunctionParam(
                  uses_default=True
              )
          )

        # Check against registry.
        function_key = self.function_registry.get_function_key(value)
        if function_key is not None:
          return meridian_pb.TfpParameterValue(
              function_param=meridian_pb.TfpParameterValue.FunctionParam(
                  function_key=function_key
              )
          )
        raise ValueError(
            f"Custom function `{param_name}` detected for"
            f" {type(dist).__name__}, but not found in registry. Please"
            " add custom functions to registry when saving models."
        )
      case _:
        # Handle unsupported types by attempting to convert to a tensor proto.
        # This allows for more flexibility in handling types that are not
        # explicitly handled above, such as numpy arrays or backend tensors.
        try:
          return meridian_pb.TfpParameterValue(
              tensor_value=backend.make_tensor_proto(value)
          )
        except TypeError as e:
          raise TypeError(f"Unsupported type: {type(value)}, {value!r}") from e

  def _from_distribution_proto(
      self,
      dist_proto: meridian_pb.TfpDistribution,
  ) -> backend.tfd.Distribution:
    """Converts a `Distribution` proto to a TensorFlow `Distribution` object."""
    dist_class_name = dist_proto.distribution_type
    dist_class = getattr(backend.tfd, dist_class_name)
    dist_parameters = dist_proto.parameters
    input_parameters = {
        k: self._unpack_tfp_parameters(k, v, dist_class)
        for k, v in dist_parameters.items()
    }
    return dist_class(**input_parameters)

  def _from_bijector_proto(
      self,
      dist_proto: meridian_pb.TfpBijector,
  ) -> backend.bijectors.Bijector:
    """Converts a `Bijector` proto to a TensorFlow `Bijector` object."""
    dist_class_name = dist_proto.bijector_type
    dist_class = getattr(backend.bijectors, dist_class_name)
    dist_parameters = dist_proto.parameters
    input_parameters = {
        name: self._unpack_tfp_parameters(name, value, dist_class)
        for name, value in dist_parameters.items()
    }

    return dist_class(**input_parameters)

  def _unpack_tfp_parameters(
      self,
      param_name: str,
      param_value: meridian_pb.TfpParameterValue,
      dist_class: backend.tfd.Distribution,
  ) -> Any:
    """Unpacks a `TfpParameterValue` proto into a Python value."""
    match param_value.WhichOneof("value_type"):
      # Handle built-in types.
      case "scalar_value":
        return param_value.scalar_value
      case "int_value":
        return param_value.int_value
      case "bool_value":
        return param_value.bool_value
      case "string_value":
        return param_value.string_value
      case "none_value":
        return None
      case "list_value":
        return [
            self._unpack_tfp_parameters(param_name, v, dist_class)
            for v in param_value.list_value.values
        ]
      case "dict_value":
        items = param_value.dict_value.value_map.items()
        return {
            key: self._unpack_tfp_parameters(key, value, dist_class)
            for key, value in items
        }

      # Handle custom types.
      case "tensor_value":
        return backend.to_tensor(backend.make_ndarray(param_value.tensor_value))
      case "distribution_value":
        return self._from_distribution_proto(param_value.distribution_value)
      case "bijector_value":
        return self._from_bijector_proto(param_value.bijector_value)
      case "fully_reparameterized":
        if param_value.fully_reparameterized:
          return backend.tfd.FULLY_REPARAMETERIZED
        else:
          return backend.tfd.NOT_FULLY_REPARAMETERIZED

      # Handle functions.
      case "function_param":
        function_param = param_value.function_param
        # Check against registry.
        if function_param.HasField("function_key"):
          registry = self.function_registry
          if function_param.function_key in registry:
            return registry.get(function_param.function_key)
        # Check for default value.
        if (
            function_param.HasField("uses_default")
            and function_param.uses_default
        ):
          signature = inspect.signature(dist_class.__init__)
          return signature.parameters[param_name].default
        raise ValueError(f"No function found for {param_name}")

      # Handle unsupported types.
      case _:
        raise ValueError(
            f"Unsupported TFP distribution parameter type: {type(param_value)}"
        )

# copybara: strip_begin


def _from_legacy_bijector_proto(
    bijector_proto: meridian_pb.Distribution.Bijector,
) -> backend.bijectors.Bijector:
  """Converts a `Bijector` proto to a `Bijector` object."""
  bijector_type_field = bijector_proto.WhichOneof(sc.BIJECTOR_TYPE)
  match bijector_type_field:
    case sc.SHIFT_BIJECTOR:
      return backend.bijectors.Shift(
          shift=_deserialize_sequence(bijector_proto.shift.shifts)
      )
    case sc.SCALE_BIJECTOR:
      return backend.bijectors.Scale(
          scale=_deserialize_sequence(bijector_proto.scale.scales),
          log_scale=_deserialize_sequence(bijector_proto.scale.log_scales),
      )
    case sc.RECIPROCAL_BIJECTOR:
      return backend.bijectors.Reciprocal()
    case _:
      raise ValueError(
          f"Unsupported Bijector proto type: {bijector_type_field};"
          f" Bijector proto:\n{bijector_proto}"
      )


def _from_legacy_distribution_proto(
    dist_proto: meridian_pb.Distribution,
) -> backend.tfd.Distribution:
  """Converts a `Distribution` proto to a `Distribution` object."""
  dist_type_field = dist_proto.WhichOneof(sc.DISTRIBUTION_TYPE)
  match dist_type_field:
    case sc.BATCH_BROADCAST_DISTRIBUTION:
      return backend.tfd.BatchBroadcast(
          name=dist_proto.name,
          distribution=_from_legacy_distribution_proto(
              dist_proto.batch_broadcast.distribution
          ),
          with_shape=_from_shape_proto(dist_proto.batch_broadcast.batch_shape),
      )
    case sc.TRANSFORMED_DISTRIBUTION:
      return backend.tfd.TransformedDistribution(
          name=dist_proto.name,
          distribution=_from_legacy_distribution_proto(
              dist_proto.transformed.distribution
          ),
          bijector=_from_legacy_bijector_proto(dist_proto.transformed.bijector),
      )
    case sc.DETERMINISTIC_DISTRIBUTION:
      return backend.tfd.Deterministic(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.deterministic.locs),
      )
    case sc.HALF_NORMAL_DISTRIBUTION:
      return backend.tfd.HalfNormal(
          name=dist_proto.name,
          scale=_deserialize_sequence(dist_proto.half_normal.scales),
      )
    case sc.LOG_NORMAL_DISTRIBUTION:
      return backend.tfd.LogNormal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.log_normal.locs),
          scale=_deserialize_sequence(dist_proto.log_normal.scales),
      )
    case sc.NORMAL_DISTRIBUTION:
      return backend.tfd.Normal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.normal.locs),
          scale=_deserialize_sequence(dist_proto.normal.scales),
      )
    case sc.TRUNCATED_NORMAL_DISTRIBUTION:
      if (
          hasattr(dist_proto.truncated_normal, "lows")
          and dist_proto.truncated_normal.lows
      ):
        if dist_proto.truncated_normal.low:
          _show_warning("low", "TruncatedNormal")
        low = _deserialize_sequence(dist_proto.truncated_normal.lows)
      else:
        low = dist_proto.truncated_normal.low

      if (
          hasattr(dist_proto.truncated_normal, "highs")
          and dist_proto.truncated_normal.highs
      ):
        if dist_proto.truncated_normal.high:
          _show_warning("high", "TruncatedNormal")
        high = _deserialize_sequence(dist_proto.truncated_normal.highs)
      else:
        high = dist_proto.truncated_normal.high
      return backend.tfd.TruncatedNormal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.truncated_normal.locs),
          scale=_deserialize_sequence(dist_proto.truncated_normal.scales),
          low=low,
          high=high,
      )
    case sc.UNIFORM_DISTRIBUTION:
      if hasattr(dist_proto.uniform, "lows") and dist_proto.uniform.lows:
        if dist_proto.uniform.low:
          _show_warning("low", "Uniform")
        low = _deserialize_sequence(dist_proto.uniform.lows)
      else:
        low = dist_proto.uniform.low

      if hasattr(dist_proto.uniform, "highs") and dist_proto.uniform.highs:
        if dist_proto.uniform.high:
          _show_warning("high", "Uniform")
        high = _deserialize_sequence(dist_proto.uniform.highs)
      else:
        high = dist_proto.uniform.high

      return backend.tfd.Uniform(
          name=dist_proto.name,
          low=low,
          high=high,
      )
    case sc.BETA_DISTRIBUTION:
      return backend.tfd.Beta(
          name=dist_proto.name,
          concentration1=_deserialize_sequence(dist_proto.beta.alpha),
          concentration0=_deserialize_sequence(dist_proto.beta.beta),
      )
    case _:
      raise ValueError(
          f"Unsupported Distribution proto type: {dist_type_field};"
          f" Distribution proto:\n{dist_proto}"
      )


def _show_warning(field_name: str, dist_name: str) -> None:
  warnings.warn(
      f"Both `{field_name}s` and `{field_name}` are specified in"
      f" {dist_name} distribution proto. Prioritizing `{field_name}s` since"
      f" `{field_name}` is deprecated.",
      DeprecationWarning,
  )


def _from_shape_proto(
    shape_proto: tensor_shape_pb2.TensorShapeProto,
) -> backend.TensorShapeInstance:
  """Converts a `TensorShapeProto` to a `TensorShape`."""
  return backend.TensorShape([dim.size for dim in shape_proto.dim])


T = TypeVar("T")


def _deserialize_sequence(args: Sequence[T]) -> T | Sequence[T] | None:
  if not args:
    return None
  return args[0] if len(args) == 1 else list(args)

# copybara: strip_end
