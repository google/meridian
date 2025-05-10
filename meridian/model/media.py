# Copyright 2024 The Meridian Authors.
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

"""Structures and functions for manipulating media value data and tensors."""

import dataclasses
from meridian import constants
from meridian.data import input_data as data
from meridian.model import spec
from meridian.model import transformers
import tensorflow as tf


__all__ = [
    "MediaTensors",
    "RfTensors",
    "OrganicMediaTensors",
    "OrganicRfTensors",
    "build_media_tensors",
    "build_rf_tensors",
    "build_organic_media_tensors",
    "build_organic_rf_tensors",
]


def _aggregate_spend_by_channel(
    spend: tf.Tensor,
    calibration_period: tf.Tensor | None = None,
) -> tf.Tensor:
  """Calculate total spend by channel aggregated over geos and times.

  Args:
    spend: A tensor of spend values with shape `(n_geos, n_times, n_channels)`.
    calibration_period: A boolean tensor with the shape `(n_media_times,
      n_channels)` indicating which time periods are used for calculation for
      each channel. If `None`, all time periods are used for calculation.

  Returns:
    A tensor of spend values with shape `(n_channels,)`.
  """
  if len(spend.shape) != 3:
    raise ValueError(
        "The `spend` argument to `_aggregate_spend_by_channel` has shape"
        f" {spend.shape}. It should have shape"
        " `(n_geos, n_times, n_channels)`."
    )
  if calibration_period is None:
    return tf.reduce_sum(spend, axis=[0, 1])

  if len(calibration_period.shape) != 2:
    raise ValueError(
        "The `calibration_period` argument to `_aggregate_spend_by_channel`"
        f" has shape {calibration_period.shape}. It should have shape"
        " `(n_media_times, n_channels)`."
    )

  if spend.shape[2] != calibration_period.shape[1]:
    raise ValueError(
        "The number of channels of the `spend` argument to"
        f" `_aggregate_spend_by_channel` ({spend.shape[2]}) is different from"
        " the number of channels of the `calibration_period` argument"
        f" ({calibration_period.shape[1]})."
    )
  if spend.shape[1] > calibration_period.shape[0]:
    raise ValueError(
        "The number of time periods of the `spend` argument to"
        f" `_aggregate_spend_by_channel` ({spend.shape[1]}) cannot be greater"
        " than the number of time periods of the `calibration_period` argument"
        f" ({calibration_period.shape[0]})."
    )

  n_times = spend.shape[1]

  # The `calibration_period` has `n_media_times` time periods which may be more
  # than the number of time periods in the spend tensor which is `n_times`. If
  # that is the case, we only use the last `n_times` time periods of the
  # calibration period for spend calculation.
  factors = tf.where(calibration_period[-n_times:, :], 1.0, 0.0)

  return tf.einsum("gtm,tm->m", spend, factors)


def _roi_calibration_scaled_counterfactual(
    metric_scaled: tf.Tensor,
    calibration_period: tf.Tensor,
) -> tf.Tensor:
  """Calculate ROI calibration scaled counterfactual media or reach.

  Args:
    metric_scaled: A tensor of scaled metric values with shape `(n_geos,
      n_times, n_channels)`.
    calibration_period: A boolean tensor indicating which time periods are used
      for calculation.

  Returns:
    A tensor of scaled metric values with shape `(n_geos, n_times, n_channels)`
    where media values are set to zero during the calibration period.
  """
  factors = tf.where(calibration_period, 0.0, 1.0)
  return tf.einsum("gtm,tm->gtm", metric_scaled, factors)


@dataclasses.dataclass(frozen=True)
class MediaTensors:
  """Container for (paid) media tensors.

  Attributes:
    media: A tensor constructed from `InputData.media`.
    media_spend: A tensor constructed from `InputData.media_spend`.
    media_transformer: A `MediaTransformer` to scale media tensors using the
      model's media data.
    media_scaled: The media tensor normalized by population and by the median
      value.
    prior_media_scaled_counterfactual: A tensor containing `media_scaled` values
      corresponding to the counterfactual scenario required for the prior
      calculation. For ROI priors, the counterfactual scenario is where media is
      set to zero during the calibration period. For mROI priors, the
      counterfactual scenario is where media is increased by a small factor for
      all `n_media_times`. For contribution priors, the counterfactual scenario
      is where media is set to zero for all `n_media_times`. This attribute is
      set to `None` when it would otherwise be a tensor of zeros, i.e., when
      contribution contribution priors are used, or when ROI priors are used and
      `roi_calibration_period` is `None`.
    prior_denominator: If ROI, mROI, or contribution priors are used, this
      represents the denominator. It is a tensor with dimension equal to
      `n_media_channels`. For ROI priors, it is the spend during the overlapping
      time periods between the calibration period and the modeling time window.
      For mROI priors, it is the ROI prior denominator multiplied by a small
      factor. For contribution priors, it is the total observed outcome
      (repeated for each channel.)
  """

  media: tf.Tensor | None = None
  media_spend: tf.Tensor | None = None
  media_transformer: transformers.MediaTransformer | None = None
  media_scaled: tf.Tensor | None = None
  prior_media_scaled_counterfactual: tf.Tensor | None = None
  prior_denominator: tf.Tensor | None = None


def build_media_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> MediaTensors:
  """Derives a MediaTensors container from media values in given input data."""
  if input_data.media is None:
    return MediaTensors()

  # Derive and set media tensors from media values in the input data.
  media = tf.convert_to_tensor(input_data.media, dtype=tf.float32)
  media_spend = tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32)
  media_spend_allocated = tf.convert_to_tensor(
      input_data.allocated_media_spend, dtype=tf.float32
  )
  media_transformer = transformers.MediaTransformer(
      media, tf.convert_to_tensor(input_data.population, dtype=tf.float32)
  )
  media_scaled = media_transformer.forward(media)
  prior_type = model_spec.effective_media_prior_type
  calibration_period = model_spec.roi_calibration_period
  if calibration_period is not None:
    calibration_period = tf.convert_to_tensor(calibration_period, dtype=tf.bool)

  # Set `prior_media_scaled_counterfactual` and `prior_denominator` depending on
  # the prior type.
  if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
    prior_denominator = _aggregate_spend_by_channel(
        media_spend_allocated,
        calibration_period=calibration_period,
    )
    if calibration_period is None:
      prior_media_scaled_counterfactual = None
    else:
      prior_media_scaled_counterfactual = (
          _roi_calibration_scaled_counterfactual(
              media_scaled,
              calibration_period=calibration_period,
          )
      )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
    prior_media_scaled_counterfactual = media_scaled * constants.MROI_FACTOR
    prior_denominator = _aggregate_spend_by_channel(media_spend) * (
        constants.MROI_FACTOR - 1.0
    )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
    prior_media_scaled_counterfactual = None
    total_outcome = tf.cast(input_data.get_total_outcome(), tf.float32)
    prior_denominator = tf.repeat(total_outcome, len(input_data.media_channel))
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
    prior_media_scaled_counterfactual = None
    prior_denominator = None
  else:
    raise ValueError(f"Unsupported prior type: {prior_type}")

  return MediaTensors(
      media=media,
      media_spend=media_spend,
      media_transformer=media_transformer,
      media_scaled=media_scaled,
      prior_media_scaled_counterfactual=prior_media_scaled_counterfactual,
      prior_denominator=prior_denominator,
  )


@dataclasses.dataclass(frozen=True)
class OrganicMediaTensors:
  """Container for organic media tensors.

  Attributes:
    organic_media: A tensor constructed from `InputData.organic_media`.
    organic_media_transformer: A `MediaTransformer` to scale media tensors using
      the model's organic media data.
    organic_media_scaled: The organic media tensor normalized by population and
      by the median value.
  """

  organic_media: tf.Tensor | None = None
  organic_media_transformer: transformers.MediaTransformer | None = None
  organic_media_scaled: tf.Tensor | None = None


def build_organic_media_tensors(
    input_data: data.InputData,
) -> OrganicMediaTensors:
  """Derives a OrganicMediaTensors container from values in given input data."""
  if input_data.organic_media is None:
    return OrganicMediaTensors()

  # Derive and set media tensors from media values in the input data.
  organic_media = tf.convert_to_tensor(
      input_data.organic_media, dtype=tf.float32
  )
  organic_media_transformer = transformers.MediaTransformer(
      organic_media,
      tf.convert_to_tensor(input_data.population, dtype=tf.float32),
  )
  organic_media_scaled = organic_media_transformer.forward(organic_media)

  return OrganicMediaTensors(
      organic_media=organic_media,
      organic_media_transformer=organic_media_transformer,
      organic_media_scaled=organic_media_scaled,
  )


@dataclasses.dataclass(frozen=True)
class RfTensors:
  """Container for Reach and Frequency (RF) media tensors.

  Attributes:
    reach: A tensor constructed from `InputData.reach`.
    frequency: A tensor constructed from `InputData.frequency`.
    rf_spend: A tensor constructed from `InputData.rf_spend`.
    reach_transformer: A `MediaTransformer` to scale RF tensors using the
      model's RF data.
    reach_scaled: A reach tensor normalized by population and by the median
      value.
    prior_reach_scaled_counterfactual: A tensor containing `reach_scaled` values
      corresponding to the counterfactual scenario required for the prior
      calculation. For ROI priors, the counterfactual scenario is where reach is
      set to zero during the calibration period. For mROI priors, the
      counterfactual scenario is where reach is increased by a small factor for
      all `n_rf_times`. For contribution priors, the counterfactual scenario is
      where reach is set to zero for all `n_rf_times`. This attribute is set to
      `None` when it would otherwise be a tensor of zeros, i.e., when
      contribution contribution priors are used, or when ROI priors are used and
      `rf_roi_calibration_period` is `None`.
    prior_denominator: If ROI, mROI, or contribution priors are used, this
      represents the denominator. It is a tensor with dimension equal to
      `n_rf_channels`. For ROI priors, it is the spend during the overlapping
      time periods between the calibration period and the modeling time window.
      For mROI priors, it is the ROI prior denominator multiplied by a small
      factor. For contribution priors, it is the total observed outcome
      (repeated for each channel).
  """

  reach: tf.Tensor | None = None
  frequency: tf.Tensor | None = None
  rf_spend: tf.Tensor | None = None
  reach_transformer: transformers.MediaTransformer | None = None
  reach_scaled: tf.Tensor | None = None
  prior_reach_scaled_counterfactual: tf.Tensor | None = None
  prior_denominator: tf.Tensor | None = None


def build_rf_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> RfTensors:
  """Derives an RfTensors container from RF media values in given input."""
  if input_data.reach is None:
    return RfTensors()

  reach = tf.convert_to_tensor(input_data.reach, dtype=tf.float32)
  frequency = tf.convert_to_tensor(input_data.frequency, dtype=tf.float32)
  rf_spend = tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32)
  rf_spend_allocated = tf.convert_to_tensor(
      input_data.allocated_rf_spend, dtype=tf.float32
  )
  reach_transformer = transformers.MediaTransformer(
      reach, tf.convert_to_tensor(input_data.population, dtype=tf.float32)
  )
  reach_scaled = reach_transformer.forward(reach)
  prior_type = model_spec.effective_rf_prior_type
  calibration_period = model_spec.rf_roi_calibration_period
  if calibration_period is not None:
    calibration_period = tf.convert_to_tensor(calibration_period, dtype=tf.bool)

  # Set `prior_reach_scaled_counterfactual` and `prior_denominator` depending on
  # the prior type.
  if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
    prior_denominator = _aggregate_spend_by_channel(
        rf_spend_allocated,
        calibration_period=calibration_period,
    )
    if calibration_period is None:
      prior_reach_scaled_counterfactual = None
    else:
      prior_reach_scaled_counterfactual = (
          _roi_calibration_scaled_counterfactual(
              reach_scaled,
              calibration_period=calibration_period,
          )
      )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
    prior_reach_scaled_counterfactual = reach_scaled * constants.MROI_FACTOR
    prior_denominator = _aggregate_spend_by_channel(rf_spend) * (
        constants.MROI_FACTOR - 1.0
    )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
    prior_reach_scaled_counterfactual = None
    total_outcome = tf.cast(input_data.get_total_outcome(), tf.float32)
    prior_denominator = tf.repeat(total_outcome, len(input_data.rf_channel))
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
    prior_reach_scaled_counterfactual = None
    prior_denominator = None
  else:
    raise ValueError(f"Unsupported prior type: {prior_type}")

  return RfTensors(
      reach=reach,
      frequency=frequency,
      rf_spend=rf_spend,
      reach_transformer=reach_transformer,
      reach_scaled=reach_scaled,
      prior_reach_scaled_counterfactual=prior_reach_scaled_counterfactual,
      prior_denominator=prior_denominator,
  )


@dataclasses.dataclass(frozen=True)
class OrganicRfTensors:
  """Container for Reach and Frequency (RF) organic media tensors.

  Attributes:
    organic_reach: A tensor constructed from `InputData.organic_reach`.
    organic_frequency: A tensor constructed from `InputData.organic_frequency`.
    organic_reach_transformer: A `MediaTransformer` to scale organic RF tensors
      using the model's organic RF data.
    organic_reach_scaled: An organic reach tensor normalized by population and
      by the median value.
  """

  organic_reach: tf.Tensor | None = None
  organic_frequency: tf.Tensor | None = None
  organic_reach_transformer: transformers.MediaTransformer | None = None
  organic_reach_scaled: tf.Tensor | None = None


def build_organic_rf_tensors(
    input_data: data.InputData,
) -> OrganicRfTensors:
  """Derives an OrganicRfTensors container from values in given input."""
  if input_data.organic_reach is None:
    return OrganicRfTensors()

  organic_reach = tf.convert_to_tensor(
      input_data.organic_reach, dtype=tf.float32
  )
  organic_frequency = tf.convert_to_tensor(
      input_data.organic_frequency, dtype=tf.float32
  )
  organic_reach_transformer = transformers.MediaTransformer(
      organic_reach,
      tf.convert_to_tensor(input_data.population, dtype=tf.float32),
  )
  organic_reach_scaled = organic_reach_transformer.forward(organic_reach)

  return OrganicRfTensors(
      organic_reach=organic_reach,
      organic_frequency=organic_frequency,
      organic_reach_transformer=organic_reach_transformer,
      organic_reach_scaled=organic_reach_scaled,
  )
