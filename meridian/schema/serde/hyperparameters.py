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

from collections.abc import Collection, Mapping, Sequence
import datetime
from typing import cast
import warnings

import bidict
from meridian import backend
from meridian import constants as c
from meridian.data import input_data as data
from meridian.data import time_coordinates as tc
from meridian.model import context as model_context_lib
from meridian.model import spec
from mmm.v1.common import date_interval_pb2
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from meridian.schema.serde import constants as sc
from meridian.schema.serde import serde
from meridian.schema.utils import proto_enum_converter
from meridian.schema.utils import time_record
import numpy as np


__all__ = [
    "HyperparametersSerde",
    "media_effects_converter",
    "non_paid_treatments_prior_type_converter",
    "paid_media_prior_type_converter",
]

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


# ------------------------------------------------------------------------------
# Declarative spec translation.
#
# `spec.DateRange` is the closed interval `[start_date, end_date]`, matching
# Meridian's other date selection APIs. `mmm.v1.common.DateInterval` is the
# half-open `[start_date, end_date)`. This module is the only place the two
# conventions meet, so every conversion between them lives here.
#
# The end bound is converted through `TimeCoordinates.get_period_bounds`, which
# steps forward by one *calendar* period.
# ------------------------------------------------------------------------------


def _serialize_date_range(
    date_range: spec.DateRange,
    period_ends: Mapping[datetime.date, datetime.date],
    *,
    spec_name: str,
) -> date_interval_pb2.DateInterval:
  """Converts a closed `DateRange` into a half-open `DateInterval` proto.

  An omitted bound is left unset rather than resolved against the data, so that
  an open-ended range survives a round trip as an open-ended range.

  Args:
    date_range: The closed date range to convert.
    period_ends: Exclusive period end for each time coordinate.
    spec_name: The `ModelSpec` attribute being serialized, used in error
      messages.

  Returns:
    The equivalent half-open `DateInterval`.

  Raises:
    ValueError: If `end_date` is not one of the time coordinates, and so has no
      period whose end could terminate the interval.
  """
  interval = date_interval_pb2.DateInterval()
  if date_range.start_date is not None:
    # The start bound needs no adjustment. It is also not checked against the
    # coordinates here because `ModelContext` already rejects an off-coordinate
    # bound when it compiles the spec.
    interval.start_date.CopyFrom(
        time_record.to_date_proto(tc.normalize_date(date_range.start_date))
    )
  if date_range.end_date is not None:
    end_date = tc.normalize_date(date_range.end_date)
    if end_date not in period_ends:
      raise ValueError(
          f"`{spec_name}` has a `DateRange` whose `end_date` ({end_date}) is"
          " not one of the input data's time coordinates, so the end of the"
          " period it selects is undefined. Date range bounds must name an"
          " exact time coordinate."
      )
    interval.end_date.CopyFrom(time_record.to_date_proto(period_ends[end_date]))
  return interval


def _deserialize_date_range(
    interval: date_interval_pb2.DateInterval,
    dates: Sequence[datetime.date],
    *,
    spec_name: str,
) -> spec.DateRange:
  """Converts a half-open `DateInterval` proto into a closed `DateRange`.

  Args:
    interval: The half-open date interval to convert.
    dates: The time coordinates the interval selects over, in order.
    spec_name: The `ModelSpec` attribute being deserialized, used in error
      messages.

  Returns:
    The equivalent closed `DateRange`.

  Raises:
    ValueError: If the exclusive `end_date` precedes every time coordinate, and
      so selects nothing.
  """
  start_date = (
      time_record.from_date_proto(interval.start_date)
      if interval.HasField(sc.START_DATE)
      else None
  )
  end_date = None
  if interval.HasField(sc.END_DATE):
    exclusive_end = time_record.from_date_proto(interval.end_date)
    # The last coordinate the half-open interval covers is the last one
    # strictly before its exclusive end.
    selected = [date for date in dates if date < exclusive_end]
    if not selected:
      raise ValueError(
          f"`{spec_name}` has a `DateInterval` whose exclusive `end_date`"
          f" ({exclusive_end}) is at or before every time coordinate in the"
          " input data, so it selects no dates."
      )
    end_date = selected[-1]
  return spec.DateRange(start_date=start_date, end_date=end_date)


def _serialize_calibration(
    calibration: spec.CalibrationSpec,
    period_ends: Mapping[datetime.date, datetime.date],
    *,
    spec_name: str,
) -> meridian_pb.CalibrationConfig:
  """Converts a `CalibrationSpec` into a `CalibrationConfig` proto."""
  config = meridian_pb.CalibrationConfig()
  entries = calibration.spec

  # `CalibrationSpec.__post_init__` guarantees the sequence is homogeneous and
  # non-empty, so the first element determines the scope of the whole spec.
  if isinstance(entries[0], spec.DateRange):
    config.global_date_ranges.date_intervals.extend(
        _serialize_date_range(entry, period_ends, spec_name=spec_name)
        for entry in cast(Sequence[spec.DateRange], entries)
    )
    return config

  for entry in cast(Sequence[spec.ChannelCalibrationSpec], entries):
    config.channel_date_ranges.channel_date_ranges.add(
        channels=entry.channels,
        date_intervals=[
            _serialize_date_range(date_range, period_ends, spec_name=spec_name)
            for date_range in entry.date_ranges
        ],
    )
  return config


def _deserialize_calibration(
    config: meridian_pb.CalibrationConfig,
    dates: Sequence[datetime.date],
    *,
    spec_name: str,
) -> spec.CalibrationSpec | None:
  """Converts a `CalibrationConfig` proto into a `CalibrationSpec`."""
  which = config.WhichOneof(sc.CONFIG_SPEC_ONEOF)
  if which == sc.GLOBAL_DATE_RANGES:
    return spec.CalibrationSpec(
        spec=[
            _deserialize_date_range(interval, dates, spec_name=spec_name)
            for interval in config.global_date_ranges.date_intervals
        ]
    )
  if which == sc.CHANNEL_DATE_RANGES:
    return spec.CalibrationSpec(
        spec=[
            spec.ChannelCalibrationSpec(
                channels=list(entry.channels),
                date_ranges=[
                    _deserialize_date_range(
                        interval, dates, spec_name=spec_name
                    )
                    for interval in entry.date_intervals
                ],
            )
            for entry in config.channel_date_ranges.channel_date_ranges
        ]
    )
  return None


def _serialize_geo_holdout(
    geo_spec: spec.GeoHoldoutSpec,
    period_ends: Mapping[datetime.date, datetime.date],
) -> meridian_pb.GeoDateRangeHoldout:
  """Converts a `GeoHoldoutSpec` into a `GeoDateRangeHoldout` proto."""
  return meridian_pb.GeoDateRangeHoldout(
      geos=geo_spec.geos,
      date_intervals=[
          _serialize_date_range(date_range, period_ends, spec_name="holdout")
          for date_range in geo_spec.date_ranges
      ],
  )


def _deserialize_geo_holdout(
    geo_proto: meridian_pb.GeoDateRangeHoldout,
    dates: Sequence[datetime.date],
) -> spec.GeoHoldoutSpec:
  """Converts a `GeoDateRangeHoldout` proto into a `GeoHoldoutSpec`."""
  return spec.GeoHoldoutSpec(
      geos=list(geo_proto.geos),
      date_ranges=[
          _deserialize_date_range(interval, dates, spec_name="holdout")
          for interval in geo_proto.date_intervals
      ],
  )


def _serialize_holdout(
    holdout: spec.HoldoutSpec,
    period_ends: Mapping[datetime.date, datetime.date],
    *,
    resolved: Sequence[spec.GeoHoldoutSpec] | None,
) -> meridian_pb.HoldoutConfig:
  """Converts a `HoldoutSpec` into a `HoldoutConfig` proto.

  Args:
    holdout: The declarative holdout specification.
    period_ends: Exclusive period end for each time coordinate.
    resolved: The draw to record, for a `RandomHoldoutSpec` whose draw this
      model actually used. `None` leaves `resolved` unset.

  Returns:
    The equivalent `HoldoutConfig`.
  """
  config = meridian_pb.HoldoutConfig()
  entries = holdout.spec

  if isinstance(entries, spec.RandomHoldoutSpec):
    config.random_holdout.ratio = entries.ratio
    if entries.seed is not None:
      config.random_holdout.seed = entries.seed
  elif isinstance(entries[0], spec.DateRange):
    config.global_date_ranges.date_intervals.extend(
        _serialize_date_range(entry, period_ends, spec_name="holdout")
        for entry in cast(Sequence[spec.DateRange], entries)
    )
  else:
    config.geo_date_ranges.geo_date_ranges.extend(
        _serialize_geo_holdout(entry, period_ends)
        for entry in cast(Sequence[spec.GeoHoldoutSpec], entries)
    )

  if resolved is not None:
    config.resolved.geo_date_ranges.extend(
        _serialize_geo_holdout(entry, period_ends) for entry in resolved
    )
  return config


def _deserialize_holdout(
    config: meridian_pb.HoldoutConfig,
    dates: Sequence[datetime.date],
) -> spec.HoldoutSpec | None:
  """Converts a `HoldoutConfig` proto into a `HoldoutSpec`.

  Args:
    config: The declarative holdout configuration.
    dates: The time coordinates the holdout selects over, in order.

  Returns:
    The equivalent `HoldoutSpec`, or `None` if `config` declares no holdout.

  Raises:
    ValueError: If `resolved` is present but holds no geos.
  """
  which = config.WhichOneof(sc.CONFIG_SPEC_ONEOF)
  if which is None:
    return None

  resolved = None
  holdout_spec: (
      Sequence[spec.DateRange]
      | Sequence[spec.GeoHoldoutSpec]
      | spec.RandomHoldoutSpec
  )
  if which == sc.RANDOM_HOLDOUT:
    holdout_spec = spec.RandomHoldoutSpec(
        ratio=config.random_holdout.ratio,
        seed=(
            config.random_holdout.seed
            if config.random_holdout.HasField(sc.SEED)
            else None
        ),
    )
    if config.HasField(sc.RESOLVED):
      resolved = [
          _deserialize_geo_holdout(geo_proto, dates)
          for geo_proto in config.resolved.geo_date_ranges
      ]
      if not resolved:
        raise ValueError(
            "`holdout_config.resolved` is present but holds no geos. A"
            " resolved random holdout must record at least one held-out geo."
        )
  elif which == sc.GLOBAL_DATE_RANGES:
    holdout_spec = [
        _deserialize_date_range(interval, dates, spec_name="holdout")
        for interval in config.global_date_ranges.date_intervals
    ]
  else:
    holdout_spec = [
        _deserialize_geo_holdout(geo_proto, dates)
        for geo_proto in config.geo_date_ranges.geo_date_ranges
    ]

  return spec.HoldoutSpec(spec=holdout_spec, resolved=resolved)


def _warn_if_random_holdout_is_unresolved(
    holdout: spec.HoldoutSpec | None,
    holdout_id: np.ndarray | None,
) -> None:
  """Warns if a random holdout will govern but its draw was not recorded.

  Args:
    holdout: The deserialized declarative holdout, if any.
    holdout_id: The deserialized deprecated holdout array, if any. When it is
      set it takes precedence, so the random specification is inert and there is
      nothing to warn about.
  """
  if holdout_id is not None:
    return
  if holdout is None or not isinstance(holdout.spec, spec.RandomHoldoutSpec):
    return
  if holdout.resolved is not None:
    return
  # The draw that the original fit used was not recorded, so it cannot be
  # restored; see `spec.RandomHoldoutSpec` for why a seed alone does not
  # reproduce one.
  warnings.warn(
      "The serialized model requests a random holdout but does not record the"
      " draw that was used. A new holdout sample will be drawn when the model"
      " is compiled, and it will not match the one used during the original"
      " fit. The model remains valid for inference, but its train/test"
      " predictive accuracy metrics are not meaningful.",
      UserWarning,
      stacklevel=2,
  )


def _fill_non_media_baseline_value(
    value_proto: meridian_pb.NonMediaBaselineValue, value: float | str
) -> None:
  """Populates a `NonMediaBaselineValue` proto from a baseline value."""
  if isinstance(value, str):
    if value.lower() == c.NON_MEDIA_BASELINE_MIN:
      value_proto.function_value = _NonMediaBaselineFunction.MIN
    elif value.lower() == c.NON_MEDIA_BASELINE_MAX:
      value_proto.function_value = _NonMediaBaselineFunction.MAX
  elif isinstance(value, (float, int)):
    value_proto.value = float(value)


def _read_non_media_baseline_value(
    value_proto: meridian_pb.NonMediaBaselineValue,
) -> float | str:
  """Reads a baseline value out of a `NonMediaBaselineValue` proto.

  Args:
    value_proto: The serialized baseline value.

  Returns:
    Either a fixed float value or a baseline function name.

  Raises:
    ValueError: If the proto holds an unrecognized value or function.
  """
  field = value_proto.WhichOneof("non_media_baseline_value")
  if field == "value":
    return value_proto.value
  if field == "function_value":
    if value_proto.function_value == _NonMediaBaselineFunction.MIN:
      return c.NON_MEDIA_BASELINE_MIN
    if value_proto.function_value == _NonMediaBaselineFunction.MAX:
      return c.NON_MEDIA_BASELINE_MAX
    if (
        value_proto.function_value
        == _NonMediaBaselineFunction.NON_MEDIA_BASELINE_FUNCTION_UNSPECIFIED
    ):
      warnings.warn(
          "Non-media baseline function value is unspecified. Resolving to"
          " 'min'."
      )
      return c.NON_MEDIA_BASELINE_MIN
    raise ValueError(
        "Unsupported NonMediaBaselineFunction proto enum value:"
        f" {value_proto.function_value}."
    )
  raise ValueError(
      f"Unsupported NonMediaBaselineValue proto enum value: {field}."
  )


def _require_model_context(
    model_context: model_context_lib.ModelContext | None, spec_name: str
) -> model_context_lib.ModelContext:
  """Returns `model_context`, or raises explaining why it is needed."""
  if model_context is None:
    raise ValueError(
        f"Serializing `{spec_name}` requires `model_context`, because its date"
        " ranges are expressed against the input data's time coordinates. Pass"
        " the `ModelContext` of the model being serialized."
    )
  return model_context


def _require_input_data(
    input_data: data.InputData | None, field_name: str
) -> data.InputData:
  """Returns `input_data`, or raises explaining why it is needed."""
  if input_data is None:
    raise ValueError(
        f"Deserializing `{field_name}` requires `input_data`, because its date"
        " intervals are interpreted against the input data's time coordinates."
        " Pass the `InputData` deserialized from the same payload."
    )
  return input_data


def _serialize_declarative_specs(
    obj: spec.ModelSpec,
    proto: meridian_pb.Hyperparameters,
    model_context: model_context_lib.ModelContext | None,
) -> None:
  """Writes `obj`'s declarative specifications into `proto`.

  Args:
    obj: The model spec being serialized.
    proto: The proto to write into.
    model_context: The context of the model being serialized, or `None`.

  Raises:
    ValueError: If a declarative date-range specification is set but
      `model_context` is `None`.
  """
  if obj.roi_calibration is not None:
    context = _require_model_context(model_context, "roi_calibration")
    proto.roi_calibration_config.CopyFrom(
        _serialize_calibration(
            obj.roi_calibration,
            context.input_data.media_time_coordinates.period_ends,
            spec_name="roi_calibration",
        )
    )
  if obj.rf_roi_calibration is not None:
    context = _require_model_context(model_context, "rf_roi_calibration")
    proto.rf_roi_calibration_config.CopyFrom(
        _serialize_calibration(
            obj.rf_roi_calibration,
            context.input_data.media_time_coordinates.period_ends,
            spec_name="rf_roi_calibration",
        )
    )
  if obj.holdout is not None:
    context = _require_model_context(model_context, "holdout")
    proto.holdout_config.CopyFrom(
        _serialize_holdout(
            obj.holdout,
            context.input_data.time_coordinates.period_ends,
            resolved=context.resolved_random_holdout,
        )
    )
  # A repeated field cannot distinguish an empty selection from an unset one,
  # so an empty sequence deserializes back as `None`. The two compile to the
  # same thing: nothing is scaled by population.
  if obj.population_scaled_controls is not None:
    proto.population_scaled_controls.extend(obj.population_scaled_controls)
  if obj.population_scaled_non_media_channels is not None:
    proto.population_scaled_non_media_channels.extend(
        obj.population_scaled_non_media_channels
    )


class HyperparametersSerde(
    serde.Serde[meridian_pb.Hyperparameters, spec.ModelSpec]
):
  """Serializes and deserializes a ModelSpec into a `Hyperparameters` proto.

  Note that this Serde only handles the Hyperparameters part of ModelSpec.
  The 'prior' attribute of ModelSpec is serialized/deserialized separately
  using DistributionSerde.

  Several `ModelSpec` attributes come in pairs: a declarative attribute and the
  deprecated array it supersedes. This Serde is a faithful mirror of whichever
  of them are set, and implements no precedence between them: it writes the
  declarative proto field if and only if the declarative attribute is set, and
  the deprecated proto field if and only if the deprecated attribute is set.
  Deserialization is the exact inverse, so every `ModelSpec` state round-trips
  unchanged -- including the state where both are set, which `ModelSpec` allows
  with a warning.

  Resolving which of a pair governs is `ModelContext`'s job, not this one. Were
  this Serde to make that decision too, the two could drift apart, and a model
  could silently fit differently after a save and reload.
  """

  def serialize(  # pyrefly: ignore[bad-override]
      self,
      obj: spec.ModelSpec,
      *,
      model_context: model_context_lib.ModelContext | None = None,
  ) -> meridian_pb.Hyperparameters:
    """Serializes the given ModelSpec into a `Hyperparameters` proto.

    Args:
      obj: The model spec to serialize.
      model_context: The context of the model being serialized. Required only if
        `obj` carries a declarative date-range specification, whose bounds are
        expressed against the input data's time coordinates, and which for a
        random holdout also supplies the draw to record.

    Returns:
      A `Hyperparameters` proto.

    Raises:
      ValueError: If `obj` carries a declarative date-range specification but
        `model_context` is `None`, or if a date range bound is not one of the
        input data's time coordinates.
    """

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

    if isinstance(obj.knots, int) and not isinstance(obj.knots, bool):
      hyperparameters_proto.n_knots = obj.knots
    elif isinstance(obj.knots, Collection):
      hyperparameters_proto.knot_locations.locations.extend(obj.knots)

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
    elif isinstance(obj.adstock_decay_spec, Mapping):
      hyperparameters_proto.adstock_decay_by_channel.channel_decays.update(
          obj.adstock_decay_spec
      )

    if isinstance(obj.saturation_spec, str):
      hyperparameters_proto.global_saturation = obj.saturation_spec
    elif isinstance(obj.saturation_spec, Mapping):
      hyperparameters_proto.saturation_by_channel.channel_saturations.update(
          obj.saturation_spec
      )

    # `non_media_baseline_values` holds either the declarative channel-name
    # mapping or the deprecated positional sequence, never both, so its type
    # selects which of the two proto fields is written.
    if isinstance(obj.non_media_baseline_values, Mapping):
      for channel, value in obj.non_media_baseline_values.items():
        _fill_non_media_baseline_value(
            hyperparameters_proto.non_media_baseline_values_map[channel], value
        )
    elif obj.non_media_baseline_values is not None:
      for value in obj.non_media_baseline_values:
        _fill_non_media_baseline_value(
            hyperparameters_proto.non_media_baseline_values.add(), value
        )

    _serialize_declarative_specs(obj, hyperparameters_proto, model_context)

    return hyperparameters_proto

  def deserialize(  # pyrefly: ignore[bad-override]
      self,
      serialized: meridian_pb.Hyperparameters,
      serialized_version: str = "",
      *,
      input_data: data.InputData | None = None,
  ) -> spec.ModelSpec:
    """Deserializes the given `Hyperparameters` proto into a ModelSpec.

    Note that this only deserializes the Hyperparameters part of ModelSpec.
    The 'prior' attribute of ModelSpec is deserialized separately
    using DistributionSerde and should be combined in the MeridianSerde.

    Args:
      serialized: The serialized `Hyperparameters` proto.
      serialized_version: The version of the serialized model. This is used to
        handle changes in deserialization logic across different versions.
      input_data: The input data deserialized from the same payload. Required
        only if `serialized` carries a declarative date-range configuration,
        whose half-open date intervals are interpreted against the input data's
        time coordinates.

    Returns:
      A Meridian model spec container.

    Raises:
      ValueError: If `serialized` carries a declarative date-range
        configuration but `input_data` is `None`.
    """

    baseline_geo = None
    baseline_geo_field = serialized.WhichOneof(sc.BASELINE_GEO_ONEOF)
    if baseline_geo_field == sc.BASELINE_GEO_INT:
      baseline_geo = serialized.baseline_geo_int
    elif baseline_geo_field == sc.BASELINE_GEO_STRING:
      baseline_geo = serialized.baseline_geo_string

    knots = None
    knots_field = serialized.WhichOneof(sc.KNOTS_SPEC)
    if knots_field == sc.N_KNOTS:
      knots = serialized.n_knots
    elif knots_field == sc.KNOT_LOCATIONS:
      knots = list(serialized.knot_locations.locations)
    # TODO: Remove fallback for legacy 'knots' repeated field once
    # downstream internal callers in ads/lift/mmm have migrated.
    elif serialized.knots:
      if len(serialized.knots) == 1:
        knots = serialized.knots[0]
      else:
        knots = list(serialized.knots)

    max_lag = (
        serialized.max_lag
        if serialized.HasField(c.MAX_LAG)
        else c.DEFAULT_MAX_LAG
    )

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

    # The declarative mapping supersedes the deprecated positional sequence.
    # `ModelSpec` holds them in a single attribute, so unlike the other
    # deprecated pairs they cannot both survive; the newer one wins, matching
    # how `knots` treats its own deprecated field above.
    non_media_baseline_values = None
    if serialized.non_media_baseline_values_map:
      non_media_baseline_values = {
          channel: _read_non_media_baseline_value(value_proto)
          for channel, value_proto in (
              serialized.non_media_baseline_values_map.items()
          )
      }
    elif serialized.non_media_baseline_values:
      non_media_baseline_values = [
          _read_non_media_baseline_value(value_proto)
          for value_proto in serialized.non_media_baseline_values
      ]

    roi_calibration = None
    if serialized.HasField(sc.ROI_CALIBRATION_CONFIG):
      roi_calibration = _deserialize_calibration(
          serialized.roi_calibration_config,
          _require_input_data(
              input_data, sc.ROI_CALIBRATION_CONFIG
          ).media_time_coordinates.all_dates,
          spec_name="roi_calibration",
      )
    rf_roi_calibration = None
    if serialized.HasField(sc.RF_ROI_CALIBRATION_CONFIG):
      rf_roi_calibration = _deserialize_calibration(
          serialized.rf_roi_calibration_config,
          _require_input_data(
              input_data, sc.RF_ROI_CALIBRATION_CONFIG
          ).media_time_coordinates.all_dates,
          spec_name="rf_roi_calibration",
      )
    holdout = None
    if serialized.HasField(sc.HOLDOUT_CONFIG):
      holdout = _deserialize_holdout(
          serialized.holdout_config,
          _require_input_data(
              input_data, sc.HOLDOUT_CONFIG
          ).time_coordinates.all_dates,
      )
    _warn_if_random_holdout_is_unresolved(holdout, holdout_id)

    population_scaled_controls = (
        list(serialized.population_scaled_controls)
        if serialized.population_scaled_controls
        else None
    )
    population_scaled_non_media_channels = (
        list(serialized.population_scaled_non_media_channels)
        if serialized.population_scaled_non_media_channels
        else None
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

    saturation_spec_field = serialized.WhichOneof(sc.SATURATION_SPEC)
    if saturation_spec_field == sc.GLOBAL_SATURATION:
      saturation_spec = serialized.global_saturation
    elif saturation_spec_field == sc.SATURATION_BY_CHANNEL:
      saturation_spec = dict(
          serialized.saturation_by_channel.channel_saturations
      )
    else:
      saturation_spec = sc.DEFAULT_SATURATION

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
        roi_calibration=roi_calibration,
        roi_calibration_period=roi_calibration_period,
        rf_roi_calibration=rf_roi_calibration,
        rf_roi_calibration_period=rf_roi_calibration_period,
        holdout=holdout,
        holdout_id=holdout_id,
        population_scaled_controls=population_scaled_controls,
        control_population_scaling_id=control_population_scaling_id,
        population_scaled_non_media_channels=(
            population_scaled_non_media_channels
        ),
        non_media_population_scaling_id=non_media_population_scaling_id,
        adstock_decay_spec=adstock_decay_spec,
        saturation_spec=saturation_spec,
    )
