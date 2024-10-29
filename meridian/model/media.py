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


@dataclasses.dataclass(frozen=True)
class MediaTensors:
  """Container for media tensors.

  Attributes:
    media: A tensor constructed from `InputData.media`.
    media_spend: A tensor constructed from `InputData.media_spend`.
    media_transformer: A `MediaTransformer` to scale media tensors using the
      model's media data.
    media_scaled: The media tensor normalized by population and by the median
      value.
    media_counterfactual: A tensor containing the media counterfactual values.
      If ROI priors are used, then the ROI of media channels is based on the
      difference in expected sales between the `media` tensor and this
      `media_counterfactual` tensor.
    media_counterfactual_scaled: A tensor containing the media counterfactual
      scaled values.
    media_spend_counterfactual: A tensor containing the media spend
      counterfactual values. If ROI priors are used, then the ROI of media
      channels is based on the spend difference between `media_spend` tensor and
      this `media_spend_counterfactual` tensor.
  """

  media: tf.Tensor | None = None
  media_spend: tf.Tensor | None = None
  media_transformer: transformers.MediaTransformer | None = None
  media_scaled: tf.Tensor | None = None
  media_counterfactual: tf.Tensor | None = None
  media_counterfactual_scaled: tf.Tensor | None = None
  media_spend_counterfactual: tf.Tensor | None = None


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
  media_transformer = transformers.MediaTransformer(
      media, tf.convert_to_tensor(input_data.population, dtype=tf.float32)
  )
  media_scaled = media_transformer.forward(media)
  if model_spec.roi_calibration_period is None:
    media_counterfactual = tf.zeros_like(media)
    media_counterfactual_scaled = tf.zeros_like(media_scaled)
    media_spend_counterfactual = tf.zeros_like(media_spend)
  else:
    media_counterfactual = tf.where(model_spec.roi_calibration_period, 0, media)
    media_counterfactual_scaled = tf.where(
        model_spec.roi_calibration_period, 0, media_scaled
    )
    n_times = len(input_data.time)
    media_spend_counterfactual = tf.where(
        model_spec.roi_calibration_period[..., -n_times:, :],
        0,
        media_spend,
    )

  return MediaTensors(
      media=media,
      media_spend=media_spend,
      media_transformer=media_transformer,
      media_scaled=media_scaled,
      media_counterfactual=media_counterfactual,
      media_counterfactual_scaled=media_counterfactual_scaled,
      media_spend_counterfactual=media_spend_counterfactual,
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
    organic_media_counterfactual: A tensor containing the organic media
      counterfactual values.
    organic_media_counterfactual_scaled: A tensor containing the organic media
      counterfactual scaled values.
  """

  organic_media: tf.Tensor | None = None
  organic_media_transformer: transformers.MediaTransformer | None = None
  organic_media_scaled: tf.Tensor | None = None
  organic_media_counterfactual: tf.Tensor | None = None
  organic_media_counterfactual_scaled: tf.Tensor | None = None


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
  organic_media_counterfactual = tf.zeros_like(organic_media)
  organic_media_counterfactual_scaled = tf.zeros_like(organic_media_scaled)

  return OrganicMediaTensors(
      organic_media=organic_media,
      organic_media_transformer=organic_media_transformer,
      organic_media_scaled=organic_media_scaled,
      organic_media_counterfactual=organic_media_counterfactual,
      organic_media_counterfactual_scaled=organic_media_counterfactual_scaled,
  )


@dataclasses.dataclass(frozen=True)
class RfTensors:
  """Container for Reach and Frequency (RF) media values tensors.

  Attributes:
    reach: A tensor constructed from `InputData.reach`.
    frequency: A tensor constructed from `InputData.frequency`.
    rf_spend: A tensor constructed from `InputData.rf_spend`.
    reach_transformer: A `MediaTransformer` to scale RF tensors using the
      model's RF data.
    reach_scaled: A reach tensor normalized by population and by the median
      value.
    reach_counterfactual: A reach tensor with media counterfactual values. If
      ROI priors are used, then the ROI of R&F channels is based on the
      difference in expected sales between the `reach` tensor and this
      `reach_counterfactual` tensor.
    reach_counterfactual_scaled: A reach tensor with media counterfactual scaled
      values.
    rf_spend_counterfactual: A reach tensor with media spend counterfactual
      values. If ROI priors are used, then the ROI of R&F channels is based on
      the spend difference between `rf_spend` tensor and this
      `rf_spend_counterfactual` tensor.
  """

  reach: tf.Tensor | None = None
  frequency: tf.Tensor | None = None
  rf_spend: tf.Tensor | None = None
  reach_transformer: transformers.MediaTransformer | None = None
  reach_scaled: tf.Tensor | None = None
  reach_counterfactual: tf.Tensor | None = None
  reach_counterfactual_scaled: tf.Tensor | None = None
  rf_spend_counterfactual: tf.Tensor | None = None


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
  reach_transformer = transformers.MediaTransformer(
      reach, tf.convert_to_tensor(input_data.population, dtype=tf.float32)
  )
  reach_scaled = reach_transformer.forward(reach)
  if model_spec.rf_roi_calibration_period is None:
    reach_counterfactual = tf.zeros_like(reach)
    reach_counterfactual_scaled = tf.zeros_like(reach_scaled)
    rf_spend_counterfactual = tf.zeros_like(rf_spend)
  else:
    reach_counterfactual = tf.where(
        model_spec.rf_roi_calibration_period, 0, reach
    )
    reach_counterfactual_scaled = tf.where(
        model_spec.rf_roi_calibration_period, 0, reach_scaled
    )
    n_times = len(input_data.time)
    rf_spend_counterfactual = tf.where(
        model_spec.rf_roi_calibration_period[..., -n_times:, :],
        0,
        rf_spend,
    )

  return RfTensors(
      reach=reach,
      frequency=frequency,
      rf_spend=rf_spend,
      reach_transformer=reach_transformer,
      reach_scaled=reach_scaled,
      reach_counterfactual=reach_counterfactual,
      reach_counterfactual_scaled=reach_counterfactual_scaled,
      rf_spend_counterfactual=rf_spend_counterfactual,
  )


@dataclasses.dataclass(frozen=True)
class OrganicRfTensors:
  """Container for Reach and Frequency (RF) organic media values tensors.

  Attributes:
    organic_reach: A tensor constructed from `InputData.organic_reach`.
    organic_frequency: A tensor constructed from `InputData.organic_frequency`.
    organic_reach_transformer: A `MediaTransformer` to scale organic RF tensors
      using the model's organic RF data.
    organic_reach_scaled: An organic reach tensor normalized by population and
      by the median value.
    organic_reach_counterfactual: An organic reach tensor with media
      counterfactual values.
    organic_reach_counterfactual_scaled: An organic reach tensor with media
      counterfactual scaled values.
  """

  organic_reach: tf.Tensor | None = None
  organic_frequency: tf.Tensor | None = None
  organic_reach_transformer: transformers.MediaTransformer | None = None
  organic_reach_scaled: tf.Tensor | None = None
  organic_reach_counterfactual: tf.Tensor | None = None
  organic_reach_counterfactual_scaled: tf.Tensor | None = None


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
  organic_reach_counterfactual = tf.zeros_like(organic_reach)
  organic_reach_counterfactual_scaled = tf.zeros_like(organic_reach_scaled)

  return OrganicRfTensors(
      organic_reach=organic_reach,
      organic_frequency=organic_frequency,
      organic_reach_transformer=organic_reach_transformer,
      organic_reach_scaled=organic_reach_scaled,
      organic_reach_counterfactual=organic_reach_counterfactual,
      organic_reach_counterfactual_scaled=organic_reach_counterfactual_scaled,
  )
