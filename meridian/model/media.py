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

"""Structures and functions for manipulating media value data and tensors."""

import dataclasses
from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.model import spec
from meridian.model import transformers


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


def _roi_calibration_scaled_counterfactual(
    metric_scaled: backend.Tensor,
    calibration_period: backend.Tensor,
) -> backend.Tensor:
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
  factors = backend.where(calibration_period, 0.0, 1.0)
  return backend.einsum("gtm,tm->gtm", metric_scaled, factors)


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

  media: backend.Tensor | None = None
  media_spend: backend.Tensor | None = None
  media_transformer: transformers.MediaTransformer | None = None
  media_scaled: backend.Tensor | None = None
  prior_media_scaled_counterfactual: backend.Tensor | None = None
  prior_denominator: backend.Tensor | None = None


def build_media_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> MediaTensors:
  """Derives a MediaTensors container from media values in given input data."""
  if input_data.media is None:
    return MediaTensors()

  # Derive and set media tensors from media values in the input data.
  media = backend.to_tensor(input_data.media, dtype=backend.float32)
  media_spend = backend.to_tensor(input_data.media_spend, dtype=backend.float32)
  media_transformer = transformers.MediaTransformer(
      media, backend.to_tensor(input_data.population, dtype=backend.float32)
  )
  media_scaled = media_transformer.forward(media)
  prior_type = model_spec.effective_media_prior_type
  calibration_period = model_spec.roi_calibration_period
  if calibration_period is not None:
    calibration_period_tensor = backend.to_tensor(
        calibration_period, dtype=backend.bool_
    )
  else:
    calibration_period_tensor = None

  aggregated_media_spend = backend.to_tensor(
      input_data.aggregate_media_spend(calibration_period=calibration_period),
      dtype=backend.float32,
  )
  # Set `prior_media_scaled_counterfactual` and `prior_denominator` depending on
  # the prior type.
  if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
    prior_denominator = aggregated_media_spend

    if calibration_period is None:
      prior_media_scaled_counterfactual = None
    else:
      prior_media_scaled_counterfactual = (
          _roi_calibration_scaled_counterfactual(
              media_scaled,
              calibration_period=calibration_period_tensor,
          )
      )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
    prior_media_scaled_counterfactual = media_scaled * constants.MROI_FACTOR
    prior_denominator = aggregated_media_spend * (constants.MROI_FACTOR - 1.0)
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
    prior_media_scaled_counterfactual = None
    total_outcome = backend.to_tensor(
        input_data.get_total_outcome(), dtype=backend.float32
    )
    prior_denominator = backend.repeat(
        total_outcome, len(input_data.media_channel)
    )
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

  organic_media: backend.Tensor | None = None
  organic_media_transformer: transformers.MediaTransformer | None = None
  organic_media_scaled: backend.Tensor | None = None


def build_organic_media_tensors(
    input_data: data.InputData,
) -> OrganicMediaTensors:
  """Derives a OrganicMediaTensors container from values in given input data."""
  if input_data.organic_media is None:
    return OrganicMediaTensors()

  # Derive and set media tensors from media values in the input data.
  organic_media = backend.to_tensor(
      input_data.organic_media, dtype=backend.float32
  )
  organic_media_transformer = transformers.MediaTransformer(
      organic_media,
      backend.to_tensor(input_data.population, dtype=backend.float32),
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
    rf_impressions: A tensor constructed from `InputData.reach` *
      `InputData.frequency`.
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

  reach: backend.Tensor | None = None
  frequency: backend.Tensor | None = None
  rf_impressions: backend.Tensor | None = None
  rf_spend: backend.Tensor | None = None
  reach_transformer: transformers.MediaTransformer | None = None
  reach_scaled: backend.Tensor | None = None
  prior_reach_scaled_counterfactual: backend.Tensor | None = None
  prior_denominator: backend.Tensor | None = None


def build_rf_tensors(
    input_data: data.InputData,
    model_spec: spec.ModelSpec,
) -> RfTensors:
  """Derives an RfTensors container from RF media values in given input."""
  if input_data.reach is None:
    return RfTensors()

  reach = backend.to_tensor(input_data.reach, dtype=backend.float32)
  frequency = backend.to_tensor(input_data.frequency, dtype=backend.float32)
  rf_impressions = (
      reach * frequency if reach is not None and frequency is not None else None
  )
  rf_spend = backend.to_tensor(input_data.rf_spend, dtype=backend.float32)
  reach_transformer = transformers.MediaTransformer(
      reach, backend.to_tensor(input_data.population, dtype=backend.float32)
  )
  reach_scaled = reach_transformer.forward(reach)
  prior_type = model_spec.effective_rf_prior_type
  calibration_period = model_spec.rf_roi_calibration_period
  if calibration_period is not None:
    calibration_period = backend.to_tensor(
        calibration_period, dtype=backend.bool_
    )
  aggregated_rf_spend = backend.to_tensor(
      input_data.aggregate_rf_spend(calibration_period=calibration_period),
      dtype=backend.float32,
  )
  # Set `prior_reach_scaled_counterfactual` and `prior_denominator` depending on
  # the prior type.
  if prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
    prior_denominator = aggregated_rf_spend
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
    prior_denominator = aggregated_rf_spend * (constants.MROI_FACTOR - 1.0)
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
    prior_reach_scaled_counterfactual = None
    total_outcome = backend.to_tensor(
        input_data.get_total_outcome(), dtype=backend.float32
    )
    prior_denominator = backend.repeat(
        total_outcome, len(input_data.rf_channel)
    )
  elif prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
    prior_reach_scaled_counterfactual = None
    prior_denominator = None
  else:
    raise ValueError(f"Unsupported prior type: {prior_type}")

  return RfTensors(
      reach=reach,
      frequency=frequency,
      rf_impressions=rf_impressions,
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

  organic_reach: backend.Tensor | None = None
  organic_frequency: backend.Tensor | None = None
  organic_reach_transformer: transformers.MediaTransformer | None = None
  organic_reach_scaled: backend.Tensor | None = None


def build_organic_rf_tensors(
    input_data: data.InputData,
) -> OrganicRfTensors:
  """Derives an OrganicRfTensors container from values in given input."""
  if input_data.organic_reach is None:
    return OrganicRfTensors()

  organic_reach = backend.to_tensor(
      input_data.organic_reach, dtype=backend.float32
  )
  organic_frequency = backend.to_tensor(
      input_data.organic_frequency, dtype=backend.float32
  )
  organic_reach_transformer = transformers.MediaTransformer(
      organic_reach,
      backend.to_tensor(input_data.population, dtype=backend.float32),
  )
  organic_reach_scaled = organic_reach_transformer.forward(organic_reach)

  return OrganicRfTensors(
      organic_reach=organic_reach,
      organic_frequency=organic_frequency,
      organic_reach_transformer=organic_reach_transformer,
      organic_reach_scaled=organic_reach_scaled,
  )
