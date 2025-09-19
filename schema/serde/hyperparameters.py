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

"""Serde for Hyperparameters."""

from meridian import constants as c
from meridian.model import spec
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import constants as sc
from schema.serde import serde
import tensorflow as tf

_MediaEffectsDist = meridian_pb.MediaEffectsDistribution
_PaidMediaPriorType = meridian_pb.PaidMediaPriorType


def _media_effects_dist_to_proto_enum(
    media_effect_dict: str,
) -> _MediaEffectsDist:
  match media_effect_dict:
    case c.MEDIA_EFFECTS_LOG_NORMAL:
      return _MediaEffectsDist.LOG_NORMAL
    case c.MEDIA_EFFECTS_NORMAL:
      return _MediaEffectsDist.NORMAL
    case _:
      return _MediaEffectsDist.MEDIA_EFFECTS_DISTRIBUTION_UNSPECIFIED


def _paid_media_prior_type_to_proto_enum(
    paid_media_prior_type: str,
) -> _PaidMediaPriorType:
  match paid_media_prior_type:
    case c.TREATMENT_PRIOR_TYPE_ROI:
      return _PaidMediaPriorType.ROI
    case c.TREATMENT_PRIOR_TYPE_MROI:
      return _PaidMediaPriorType.MROI
    case c.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      return _PaidMediaPriorType.COEFFICIENT
    case _:
      return _PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED


def _proto_enum_to_media_effects_dist(
    proto_enum: _MediaEffectsDist,
) -> str:
  """Converts a `_MediaEffectsDist` enum to its string representation."""
  match proto_enum:
    case _MediaEffectsDist.LOG_NORMAL:
      return c.MEDIA_EFFECTS_LOG_NORMAL
    case _MediaEffectsDist.NORMAL:
      return c.MEDIA_EFFECTS_NORMAL
    case _:
      raise ValueError(
          "Unsupported MediaEffectsDistribution proto enum value:"
          f" {proto_enum}."
      )


def _proto_enum_to_paid_media_prior_type(
    proto_enum: _PaidMediaPriorType,
) -> str | None:
  """Converts a `_PaidMediaPriorType` enum to its string representation."""
  match proto_enum:
    case _PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED:
      return None
    case _PaidMediaPriorType.ROI:
      return c.TREATMENT_PRIOR_TYPE_ROI
    case _PaidMediaPriorType.MROI:
      return c.TREATMENT_PRIOR_TYPE_MROI
    case _PaidMediaPriorType.COEFFICIENT:
      return c.TREATMENT_PRIOR_TYPE_COEFFICIENT
    case _:
      raise ValueError(
          f"Unsupported PaidMediaPriorType proto enum value: {proto_enum}."
      )


class HyperparametersSerde(
    serde.Serde[meridian_pb.Hyperparameters, spec.ModelSpec]
):
  """Serializes and deserializes a ModelSpec into a `Hyperparameters` proto.

  Note that this Serde only handles the Hyperparameters part of ModelSpec.
  The 'prior' attribute of ModelSpec is serialized/deserialized separately
  using DistributionSerde.
  """

  def serialize(self, obj: spec.ModelSpec) -> meridian_pb.Hyperparameters:
    """Serializes the given ModelSpec into a `Hyperparameters` proto."""
    hyperparameters_proto = meridian_pb.Hyperparameters(
        media_effects_dist=_media_effects_dist_to_proto_enum(
            obj.media_effects_dist
        ),
        hill_before_adstock=obj.hill_before_adstock,
        unique_sigma_for_each_geo=obj.unique_sigma_for_each_geo,
        media_prior_type=_paid_media_prior_type_to_proto_enum(
            obj.media_prior_type
        ),
        rf_prior_type=_paid_media_prior_type_to_proto_enum(obj.rf_prior_type),
        paid_media_prior_type=_paid_media_prior_type_to_proto_enum(
            obj.paid_media_prior_type
        ),
    )
    if obj.max_lag is not None:
      hyperparameters_proto.max_lag = obj.max_lag

    if isinstance(obj.knots, int):
      hyperparameters_proto.knots.append(obj.knots)
    elif isinstance(obj.knots, list):
      hyperparameters_proto.knots.extend(obj.knots)

    if isinstance(obj.baseline_geo, str):
      hyperparameters_proto.baseline_geo_string = obj.baseline_geo
    elif isinstance(obj.baseline_geo, int):
      hyperparameters_proto.baseline_geo_int = obj.baseline_geo

    if obj.roi_calibration_period is not None:
      hyperparameters_proto.roi_calibration_period.CopyFrom(
          tf.make_tensor_proto(obj.roi_calibration_period)
      )
    if obj.rf_roi_calibration_period is not None:
      hyperparameters_proto.rf_roi_calibration_period.CopyFrom(
          tf.make_tensor_proto(obj.rf_roi_calibration_period)
      )
    if obj.holdout_id is not None:
      hyperparameters_proto.holdout_id.CopyFrom(
          tf.make_tensor_proto(obj.holdout_id)
      )
    if obj.control_population_scaling_id is not None:
      hyperparameters_proto.control_population_scaling_id.CopyFrom(
          tf.make_tensor_proto(obj.control_population_scaling_id)
      )
    if obj.non_media_population_scaling_id is not None:
      hyperparameters_proto.non_media_population_scaling_id.CopyFrom(
          tf.make_tensor_proto(obj.non_media_population_scaling_id)
      )
    return hyperparameters_proto

  def deserialize(
      self,
      serialized: meridian_pb.Hyperparameters,
      serialized_version: str = "",
  ) -> spec.ModelSpec:
    """Deserializes the given `Hyperparameters` proto into a ModelSpec.

    Note that this only deserializes the Hyperparameters part of ModelSpec.
    The 'prior' attribute of ModelSpec is deserialized separately
    using DistributionSerde and should be combined in the MeridianSerde.

    Args:
      serialized: The serialized `Hyperparameters` proto.
      serialized_version: The version of the serialized model.

    Returns:
      A Meridian model spec container.
    """
    baseline_geo = None
    baseline_geo_field = serialized.WhichOneof(sc.BASELINE_GEO_ONEOF)
    if baseline_geo_field == sc.BASELINE_GEO_INT:
      baseline_geo = serialized.baseline_geo_int
    elif baseline_geo_field == sc.BASELINE_GEO_STRING:
      baseline_geo = serialized.baseline_geo_string

    knots = None
    if serialized.knots:
      if len(serialized.knots) == 1:
        knots = serialized.knots[0]
      else:
        knots = list(serialized.knots)

    max_lag = serialized.max_lag if serialized.HasField(c.MAX_LAG) else None

    roi_calibration_period = (
        tf.make_ndarray(serialized.roi_calibration_period)
        if serialized.HasField(c.ROI_CALIBRATION_PERIOD)
        else None
    )
    rf_roi_calibration_period = (
        tf.make_ndarray(serialized.rf_roi_calibration_period)
        if serialized.HasField(c.RF_ROI_CALIBRATION_PERIOD)
        else None
    )

    holdout_id = (
        tf.make_ndarray(serialized.holdout_id)
        if serialized.HasField(sc.HOLDOUT_ID)
        else None
    )

    control_population_scaling_id = (
        tf.make_ndarray(serialized.control_population_scaling_id)
        if serialized.HasField(sc.CONTROL_POPULATION_SCALING_ID)
        else None
    )

    non_media_population_scaling_id = (
        tf.make_ndarray(serialized.non_media_population_scaling_id)
        if serialized.HasField(sc.NON_MEDIA_POPULATION_SCALING_ID)
        else None
    )

    return spec.ModelSpec(
        media_effects_dist=_proto_enum_to_media_effects_dist(
            serialized.media_effects_dist
        ),
        hill_before_adstock=serialized.hill_before_adstock,
        max_lag=max_lag,
        unique_sigma_for_each_geo=serialized.unique_sigma_for_each_geo,
        media_prior_type=_proto_enum_to_paid_media_prior_type(
            serialized.media_prior_type
        ),
        rf_prior_type=_proto_enum_to_paid_media_prior_type(
            serialized.rf_prior_type
        ),
        paid_media_prior_type=_proto_enum_to_paid_media_prior_type(
            serialized.paid_media_prior_type
        ),
        knots=knots,
        baseline_geo=baseline_geo,
        roi_calibration_period=roi_calibration_period,
        rf_roi_calibration_period=rf_roi_calibration_period,
        holdout_id=holdout_id,
        control_population_scaling_id=control_population_scaling_id,
        non_media_population_scaling_id=non_media_population_scaling_id,
    )
