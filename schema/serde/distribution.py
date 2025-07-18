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

"""Serialization and deserialization of `Distribution` objects for priors."""

from typing import Sequence, TypeVar

from meridian import constants
from meridian.model import prior_distribution as pd
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import constants as sc
from schema.serde import serde
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.core.framework import tensor_shape_pb2  # pylint: disable=g-direct-tensorflow-import


class DistributionSerde(
    serde.Serde[meridian_pb.PriorDistributions, pd.PriorDistribution]
):
  """Serializes and deserializes a Meridian prior distributions container into a `Distribution` proto."""

  def serialize(
      self, obj: pd.PriorDistribution
  ) -> meridian_pb.PriorDistributions:
    """Serializes the given Meridian priors container into a `PriorDistributions` proto."""
    proto = meridian_pb.PriorDistributions()
    for param in constants.ALL_PRIOR_DISTRIBUTION_PARAMETERS:
      if not hasattr(obj, param):
        continue
      getattr(proto, param).CopyFrom(
          _to_distribution_proto(getattr(obj, param))
      )
    return proto

  def deserialize(
      self,
      serialized: meridian_pb.PriorDistributions,
      serialized_version: str = "",
  ) -> pd.PriorDistribution:
    """Deserializes the given `PriorDistributions` proto into a Meridian prior distribution container."""
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
      kwargs[param] = _from_distribution_proto(getattr(serialized, param))
    return pd.PriorDistribution(**kwargs)


def _to_bijector_proto(
    bijector: tfp.bijectors.Bijector,
) -> meridian_pb.Distribution.Bijector:
  """Converts a TensorFlow `Bijector` object to a `Bijector` proto."""
  bijector_proto = meridian_pb.Distribution.Bijector(name=bijector.name)
  match type(bijector):
    case tfp.bijectors.Shift:
      bijector_proto.shift.CopyFrom(
          meridian_pb.Distribution.Bijector.Shift(
              shifts=_serialize_tensor_array(bijector.shift)
          )
      )
    case tfp.bijectors.Scale:
      bijector_proto.scale.CopyFrom(
          meridian_pb.Distribution.Bijector.Scale(
              scales=_serialize_tensor_array(bijector.scale),
              log_scales=_serialize_tensor_array(bijector.log_scale),
          )
      )
    case tfp.bijectors.Reciprocal:
      bijector_proto.reciprocal.CopyFrom(
          meridian_pb.Distribution.Bijector.Reciprocal()
      )
    case _:
      raise ValueError("Unsupported TFP bijector type: %s" % type(bijector))

  return bijector_proto


def _to_distribution_proto(
    dist: tfp.distributions.Distribution,
) -> meridian_pb.Distribution:
  """Converts a TensorFlow `Distribution` object to a `Distribution` proto."""
  dist_proto = meridian_pb.Distribution(name=dist.name)
  match type(dist):
    case tfp.distributions.BatchBroadcast:
      dist_proto.batch_broadcast.CopyFrom(
          meridian_pb.Distribution.BatchBroadcast(
              distribution=_to_distribution_proto(dist.distribution),
              batch_shape=dist.copy().batch_shape.as_proto(),
          )
      )
    case tfp.distributions.TransformedDistribution:
      dist_proto.transformed.CopyFrom(
          meridian_pb.Distribution.Transformed(
              distribution=_to_distribution_proto(dist.distribution),
              bijector=_to_bijector_proto(dist.bijector),
          )
      )
    case tfp.distributions.Deterministic:
      dist_proto.deterministic.CopyFrom(
          meridian_pb.Distribution.Deterministic(
              locs=_serialize_tensor_array(dist.loc),
          )
      )
    case tfp.distributions.HalfNormal:
      dist_proto.half_normal.CopyFrom(
          meridian_pb.Distribution.HalfNormal(
              scales=_serialize_tensor_array(dist.scale),
          )
      )
    case tfp.distributions.LogNormal:
      dist_proto.log_normal.CopyFrom(
          meridian_pb.Distribution.LogNormal(
              locs=_serialize_tensor_array(dist.loc),
              scales=_serialize_tensor_array(dist.scale),
          )
      )
    case tfp.distributions.Normal:
      dist_proto.normal.CopyFrom(
          meridian_pb.Distribution.Normal(
              locs=_serialize_tensor_array(dist.loc),
              scales=_serialize_tensor_array(dist.scale),
          )
      )
    case tfp.distributions.TruncatedNormal:
      dist_proto.truncated_normal.CopyFrom(
          meridian_pb.Distribution.TruncatedNormal(
              locs=_serialize_tensor_array(dist.loc),
              scales=_serialize_tensor_array(dist.scale),
              lows=_serialize_tensor_array(dist.low),
              highs=_serialize_tensor_array(dist.high),
          )
      )
    case tfp.distributions.Uniform:
      dist_proto.uniform.CopyFrom(
          meridian_pb.Distribution.Uniform(
              low=dist.low,
              high=dist.high,
          )
      )
    case tfp.distributions.Beta:
      dist_proto.beta.CopyFrom(
          meridian_pb.Distribution.Beta(
              alpha=_serialize_tensor_array(dist.concentration1),
              beta=_serialize_tensor_array(dist.concentration0),
          )
      )
    case _:
      raise ValueError("Unsupported TFP distribution type: %s" % type(dist))

  return dist_proto


def _serialize_tensor_array(
    tensor_array: tf.Tensor | None,
) -> list[float]:
  if tensor_array is None:
    return []
  # If the given tensor array has no shape, assume a scalar value and return
  # a list with a single element.
  if not tensor_array.shape.dims:
    return [tensor_array.numpy().item()]
  # Assume that the given tensor is a 1D tensor at most.
  return tensor_array.numpy().tolist()


def _from_bijector_proto(
    bijector_proto: meridian_pb.Distribution.Bijector,
) -> tfp.bijectors.Bijector:
  """Converts a `Bijector` proto to a TensorFlow `Bijector` object."""
  bijector_type_field = bijector_proto.WhichOneof(sc.BIJECTOR_TYPE)
  match bijector_type_field:
    case sc.SHIFT_BIJECTOR:
      return tfp.bijectors.Shift(
          shift=_deserialize_sequence(bijector_proto.shift.shifts)
      )
    case sc.SCALE_BIJECTOR:
      return tfp.bijectors.Scale(
          scale=_deserialize_sequence(bijector_proto.scale.scales),
          log_scale=_deserialize_sequence(bijector_proto.scale.log_scales),
      )
    case sc.RECIPROCAL_BIJECTOR:
      return tfp.bijectors.Reciprocal()
    case _:
      raise ValueError(
          f"Unsupported Bijector proto type: {bijector_type_field};"
          f" Bijector proto:\n{bijector_proto}"
      )


def _from_distribution_proto(
    dist_proto: meridian_pb.Distribution,
) -> tfp.distributions.Distribution:
  """Converts a `Distribution` proto to a TensorFlow `Distribution` object."""
  dist_type_field = dist_proto.WhichOneof(sc.DISTRIBUTION_TYPE)
  match dist_type_field:
    case sc.BATCH_BROADCAST_DISTRIBUTION:
      return tfp.distributions.BatchBroadcast(
          name=dist_proto.name,
          distribution=_from_distribution_proto(
              dist_proto.batch_broadcast.distribution
          ),
          with_shape=_from_shape_proto(dist_proto.batch_broadcast.batch_shape),
      )
    case sc.TRANSFORMED_DISTRIBUTION:
      return tfp.distributions.TransformedDistribution(
          name=dist_proto.name,
          distribution=_from_distribution_proto(
              dist_proto.transformed.distribution
          ),
          bijector=_from_bijector_proto(dist_proto.transformed.bijector),
      )
    case sc.DETERMINISTIC_DISTRIBUTION:
      return tfp.distributions.Deterministic(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.deterministic.locs),
      )
    case sc.HALF_NORMAL_DISTRIBUTION:
      return tfp.distributions.HalfNormal(
          name=dist_proto.name,
          scale=_deserialize_sequence(dist_proto.half_normal.scales),
      )
    case sc.LOG_NORMAL_DISTRIBUTION:
      return tfp.distributions.LogNormal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.log_normal.locs),
          scale=_deserialize_sequence(dist_proto.log_normal.scales),
      )
    case sc.NORMAL_DISTRIBUTION:
      return tfp.distributions.Normal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.normal.locs),
          scale=_deserialize_sequence(dist_proto.normal.scales),
      )
    case sc.TRUNCATED_NORMAL_DISTRIBUTION:
      if dist_proto.truncated_normal.HasField("low"):
        low = dist_proto.truncated_normal.low
      else:
        low = _deserialize_sequence(dist_proto.truncated_normal.lows)
      if dist_proto.truncated_normal.HasField("high"):
        high = dist_proto.truncated_normal.high
      else:
        high = _deserialize_sequence(dist_proto.truncated_normal.highs)
      return tfp.distributions.TruncatedNormal(
          name=dist_proto.name,
          loc=_deserialize_sequence(dist_proto.truncated_normal.locs),
          scale=_deserialize_sequence(dist_proto.truncated_normal.scales),
          low=low,
          high=high,
      )
    case sc.UNIFORM_DISTRIBUTION:
      return tfp.distributions.Uniform(
          name=dist_proto.name,
          low=dist_proto.uniform.low,
          high=dist_proto.uniform.high,
      )
    case sc.BETA_DISTRIBUTION:
      return tfp.distributions.Beta(
          name=dist_proto.name,
          concentration1=_deserialize_sequence(dist_proto.beta.alpha),
          concentration0=_deserialize_sequence(dist_proto.beta.beta),
      )
    case _:
      raise ValueError(
          f"Unsupported Distribution proto type: {dist_type_field};"
          f" Distribution proto:\n{dist_proto}"
      )


def _from_shape_proto(
    shape_proto: tensor_shape_pb2.TensorShapeProto,
) -> tf.TensorShape:
  """Converts a `TensorShapeProto` to a `TensorShape`."""
  return tf.TensorShape([dim.size for dim in shape_proto.dim])


T = TypeVar("T")


def _deserialize_sequence(args: Sequence[T]) -> T | Sequence[T] | None:
  if not args:
    return None
  return args[0] if len(args) == 1 else list(args)
