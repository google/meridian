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

"""Serde for Hyperparameters."""

import warnings

import bidict
from meridian import backend
from meridian import constants as c
from meridian.model import spec
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from meridian.schema.serde import constants as sc
from meridian.schema.serde import serde
from meridian.schema.utils import proto_enum_converter
import numpy as np

_MediaEffectsDist = meridian_pb.MediaEffectsDistribution
_PaidMediaPriorType = meridian_pb.PaidMediaPriorType
_NonPaidTreatmentsPriorType = meridian_pb.NonPaidTreatmentsPriorType
_NonMediaBaselineFunction = (
    meridian_pb.NonMediaBaselineValue.NonMediaBaselineFunction
)

media_effects_converter = proto_enum_converter.ProtoEnumConverter(
    enum_display_name="Media effects distribution",
    enum_message=_MediaEffectsDist,
    mapping=bidict.bidict({
        c.MEDIA_EFFECTS_LOG_NORMAL: "LOG_NORMAL",
        c.MEDIA_EFFECTS_NORMAL: "NORMAL",
    }),
    enum_unspecified=_MediaEffectsDist.MEDIA_EFFECTS_DISTRIBUTION_UNSPECIFIED,
    default_when_unspecified=c.MEDIA_EFFECTS_LOG_NORMAL,
)

paid_media_prior_type_converter = proto_enum_converter.ProtoEnumConverter(
    enum_display_name="Paid media prior type",
    enum_message=_PaidMediaPriorType,
    mapping=bidict.bidict({
        c.TREATMENT_PRIOR_TYPE_ROI: "ROI",
        c.TREATMENT_PRIOR_TYPE_MROI: "MROI",
        c.TREATMENT_PRIOR_TYPE_COEFFICIENT: "COEFFICIENT",
        c.TREATMENT_PRIOR_TYPE_CONTRIBUTION: "CONTRIBUTION",
    }),
    enum_unspecified=_PaidMediaPriorType.PAID_MEDIA_PRIOR_TYPE_UNSPECIFIED,
    default_when_unspecified=None,
)

non_paid_treatments_prior_type_converter = proto_enum_converter.ProtoEnumConverter(
    enum_display_name="Non-paid treatments prior type",
    enum_message=_NonPaidTreatmentsPriorType,
    mapping=bidict.bidict({
        c.TREATMENT_PRIOR_TYPE_COEFFICIENT: (
            "NON_PAID_TREATMENTS_PRIOR_TYPE_COEFFICIENT"
        ),
        c.TREATMENT_PRIOR_TYPE_CONTRIBUTION: (
            "NON_PAID_TREATMENTS_PRIOR_TYPE_CONTRIBUTION"
        ),
    }),
    enum_unspecified=_NonPaidTreatmentsPriorType.NON_PAID_TREATMENTS_PRIOR_TYPE_UNSPECIFIED,
    default_when_unspecified=c.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
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
        media_effects_dist=media_effects_converter.to_proto(
            obj.media_effects_dist
        ),
        hill_before_adstock=obj.hill_before_adstock,
        unique_sigma_for_each_geo=obj.unique_sigma_for_each_geo,
        media_prior_type=paid_media_prior_type_converter.to_proto(
            obj.media_prior_type
        ),
        rf_prior_type=paid_media_prior_type_converter.to_proto(
            obj.rf_prior_type
        ),
        paid_media_prior_type=paid_media_prior_type_converter.to_proto(
            obj.paid_media_prior_type
        ),
        organic_media_prior_type=non_paid_treatments_prior_type_converter.to_proto(
            obj.organic_media_prior_type
        ),
        organic_rf_prior_type=non_paid_treatments_prior_type_converter.to_proto(
            obj.organic_rf_prior_type
        ),
        non_media_treatments_prior_type=non_paid_treatments_prior_type_converter.to_proto(
            obj.non_media_treatments_prior_type
        ),
        enable_aks=obj.enable_aks,
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
          backend.make_tensor_proto(np.array(obj.roi_calibration_period))
      )
    if obj.rf_roi_calibration_period is not None:
      hyperparameters_proto.rf_roi_calibration_period.CopyFrom(
          backend.make_tensor_proto(np.array(obj.rf_roi_calibration_period))
      )
    if obj.holdout_id is not None:
      hyperparameters_proto.holdout_id.CopyFrom(
          backend.make_tensor_proto(np.array(obj.holdout_id))
      )
    if obj.control_population_scaling_id is not None:
      hyperparameters_proto.control_population_scaling_id.CopyFrom(
          backend.make_tensor_proto(np.array(obj.control_population_scaling_id))
      )
    if obj.non_media_population_scaling_id is not None:
      hyperparameters_proto.non_media_population_scaling_id.CopyFrom(
          backend.make_tensor_proto(
              np.array(obj.non_media_population_scaling_id)
          )
      )

    if isinstance(obj.adstock_decay_spec, str):
      hyperparameters_proto.global_adstock_decay = obj.adstock_decay_spec
    elif isinstance(obj.adstock_decay_spec, dict):
      hyperparameters_proto.adstock_decay_by_channel.channel_decays.update(
          obj.adstock_decay_spec
      )

    if obj.non_media_baseline_values is not None:
      for value in obj.non_media_baseline_values:
        value_proto = hyperparameters_proto.non_media_baseline_values.add()
        if isinstance(value, str):
          if value.lower() == "min":
            value_proto.function_value = _NonMediaBaselineFunction.MIN
          elif value.lower() == "max":
            value_proto.function_value = _NonMediaBaselineFunction.MAX
        elif isinstance(value, (float, int)):
          value_proto.value = float(value)

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
      serialized_version: The version of the serialized model. This is used to
        handle changes in deserialization logic across different versions.

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
        backend.make_ndarray(serialized.roi_calibration_period)
        if serialized.HasField(c.ROI_CALIBRATION_PERIOD)
        else None
    )
    rf_roi_calibration_period = (
        backend.make_ndarray(serialized.rf_roi_calibration_period)
        if serialized.HasField(c.RF_ROI_CALIBRATION_PERIOD)
        else None
    )

    holdout_id = (
        backend.make_ndarray(serialized.holdout_id)
        if serialized.HasField(sc.HOLDOUT_ID)
        else None
    )

    control_population_scaling_id = (
        backend.make_ndarray(serialized.control_population_scaling_id)
        if serialized.HasField(sc.CONTROL_POPULATION_SCALING_ID)
        else None
    )

    non_media_population_scaling_id = (
        backend.make_ndarray(serialized.non_media_population_scaling_id)
        if serialized.HasField(sc.NON_MEDIA_POPULATION_SCALING_ID)
        else None
    )

    non_media_baseline_values = None
    if serialized.non_media_baseline_values:
      non_media_baseline_values = []
      for value_proto in serialized.non_media_baseline_values:
        field = value_proto.WhichOneof("non_media_baseline_value")
        if field == "value":
          non_media_baseline_values.append(value_proto.value)
        elif field == "function_value":
          if value_proto.function_value == _NonMediaBaselineFunction.MIN:
            non_media_baseline_values.append("min")
          elif value_proto.function_value == _NonMediaBaselineFunction.MAX:
            non_media_baseline_values.append("max")
          elif (
              value_proto.function_value
              == _NonMediaBaselineFunction.NON_MEDIA_BASELINE_FUNCTION_UNSPECIFIED
          ):
            warnings.warn(
                "Non-media baseline function value is unspecified. Resolving to"
                " 'min'."
            )
            non_media_baseline_values.append("min")
          else:
            raise ValueError(
                "Unsupported NonMediaBaselineFunction proto enum value:"
                f" {value_proto.function_value}."
            )
        else:
          raise ValueError(
              f"Unsupported NonMediaBaselineValue proto enum value: {field}."
          )

    adstock_decay_spec_field = serialized.WhichOneof(sc.ADSTOCK_DECAY_SPEC)
    if adstock_decay_spec_field == sc.GLOBAL_ADSTOCK_DECAY:
      adstock_decay_spec = serialized.global_adstock_decay
    elif adstock_decay_spec_field == sc.ADSTOCK_DECAY_BY_CHANNEL:
      adstock_decay_spec = dict(
          serialized.adstock_decay_by_channel.channel_decays
      )
    else:
      adstock_decay_spec = sc.DEFAULT_DECAY

    return spec.ModelSpec(
        media_effects_dist=media_effects_converter.from_proto(
            serialized.media_effects_dist
        ),
        hill_before_adstock=serialized.hill_before_adstock,
        max_lag=max_lag,
        unique_sigma_for_each_geo=serialized.unique_sigma_for_each_geo,
        media_prior_type=paid_media_prior_type_converter.from_proto(
            serialized.media_prior_type
        ),
        rf_prior_type=paid_media_prior_type_converter.from_proto(
            serialized.rf_prior_type
        ),
        paid_media_prior_type=paid_media_prior_type_converter.from_proto(
            serialized.paid_media_prior_type
        ),
        organic_media_prior_type=non_paid_treatments_prior_type_converter.from_proto(
            serialized.organic_media_prior_type
        ),
        organic_rf_prior_type=non_paid_treatments_prior_type_converter.from_proto(
            serialized.organic_rf_prior_type
        ),
        non_media_treatments_prior_type=non_paid_treatments_prior_type_converter.from_proto(
            serialized.non_media_treatments_prior_type
        ),
        non_media_baseline_values=non_media_baseline_values,
        knots=knots,
        enable_aks=serialized.enable_aks,
        baseline_geo=baseline_geo,
        roi_calibration_period=roi_calibration_period,
        rf_roi_calibration_period=rf_roi_calibration_period,
        holdout_id=holdout_id,
        control_population_scaling_id=control_population_scaling_id,
        non_media_population_scaling_id=non_media_population_scaling_id,
        adstock_decay_spec=adstock_decay_spec,
    )
