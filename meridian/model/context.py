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

"""Defines ModelContext class for Meridian."""

from collections.abc import Mapping
import functools

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import spec
from meridian.model import transformers
import numpy as np

__all__ = [
    "ModelContext",
]


class ModelContext:
  """Model context for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec,
  ):
    self._input_data = input_data
    self._model_spec = model_spec

    self._validate_model_spec_shapes()

  def _validate_model_spec_shapes(self):
    """Validate shapes of model_spec attributes."""
    if self._model_spec.roi_calibration_period is not None:
      if self._model_spec.roi_calibration_period.shape != (
          self.n_media_times,
          self.n_media_channels,
      ):
        raise ValueError(
            "The shape of `roi_calibration_period`"
            f" {self._model_spec.roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_media_channels) = ({self.n_media_times},"
            f" {self.n_media_channels})`."
        )

    if self._model_spec.rf_roi_calibration_period is not None:
      if self._model_spec.rf_roi_calibration_period.shape != (
          self.n_media_times,
          self.n_rf_channels,
      ):
        raise ValueError(
            "The shape of `rf_roi_calibration_period`"
            f" {self._model_spec.rf_roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
            f" {self.n_rf_channels})`."
        )

    if self._model_spec.holdout_id is not None:
      expected_shape = (
          (self.n_times,) if self.is_national else (self.n_geos, self.n_times)
      )
      if self._model_spec.holdout_id.shape != expected_shape:
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            " different from"
            f" {'`(n_times,)`' if self.is_national else '`(n_geos, n_times)`'} ="
            f" {expected_shape}."
        )

    if self._model_spec.control_population_scaling_id is not None:
      if self._model_spec.control_population_scaling_id.shape != (
          self.n_controls,
      ):
        raise ValueError(
            "The shape of `control_population_scaling_id`"
            f" {self._model_spec.control_population_scaling_id.shape} is"
            f" different from `(n_controls,) = ({self.n_controls},)`."
        )

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.build_media_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.build_rf_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self._input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self._input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.kpi, dtype=backend.float32)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self._input_data.revenue_per_kpi is None:
      return None
    return backend.to_tensor(
        self._input_data.revenue_per_kpi, dtype=backend.float32
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    if self._input_data.controls is None:
      return None
    return backend.to_tensor(self._input_data.controls, dtype=backend.float32)

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self._input_data.non_media_treatments is None:
      return None
    return backend.to_tensor(
        self._input_data.non_media_treatments, dtype=backend.float32
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.population, dtype=backend.float32)

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_spend(), dtype=backend.float32
    )

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_outcome(), dtype=backend.float32
    )

  @property
  def n_geos(self) -> int:
    return len(self._input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self._input_data.media_channel is None:
      return 0
    return len(self._input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self._input_data.rf_channel is None:
      return 0
    return len(self._input_data.rf_channel)

  @property
  def n_organic_media_channels(self) -> int:
    if self._input_data.organic_media_channel is None:
      return 0
    return len(self._input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self._input_data.organic_rf_channel is None:
      return 0
    return len(self._input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    if self._input_data.control_variable is None:
      return 0
    return len(self._input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self._input_data.non_media_channel is None:
      return 0
    return len(self._input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self._input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self._input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self._model_spec.knots,
        enable_aks=self._model_spec.enable_aks,
        data=self._input_data,
        is_national=self.is_national,
    )

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for controls, if it exists."""
    if self.controls is None:
      return None

    if self._model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = backend.to_tensor(
          self._model_spec.control_population_scaling_id, dtype=backend.bool_
      )
    else:
      controls_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.controls,
        population=self.population,
        population_scaling_id=controls_population_scaling_id,
    )

  @functools.cached_property
  def non_media_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for non-media treatments."""
    if self.non_media_treatments is None:
      return None
    if self._model_spec.non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.to_tensor(
          self._model_spec.non_media_population_scaling_id, dtype=backend.bool_
      )
    else:
      non_media_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.non_media_treatments,
        population=self.population,
        population_scaling_id=non_media_population_scaling_id,
    )

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return transformers.KpiTransformer(self.kpi, self.population)

  @functools.cached_property
  def controls_scaled(self) -> backend.Tensor | None:
    if self.controls is not None:
      # If `controls` is defined, then `controls_transformer` is also defined.
      return self.controls_transformer.forward(self.controls)  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def non_media_treatments_normalized(self) -> backend.Tensor | None:
    """Normalized non-media treatments.

    The non-media treatments values are scaled by population (for channels where
    `non_media_population_scaling_id` is `True`) and normalized by centering and
    scaling with means and standard deviations.
    """
    if self.non_media_transformer is not None:
      return self.non_media_transformer.forward(
          self.non_media_treatments
      )  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
    return self.kpi_transformer.forward(self.kpi)

  @functools.cached_property
  def media_effects_dist(self) -> str:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[constants.MEDIA_EFFECTS_DIST]
    else:
      return self._model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      # Should evaluate to False.
      return constants.NATIONAL_MODEL_SPEC_ARGS[
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self._model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self._model_spec.baseline_geo, int):
      if (
          self._model_spec.baseline_geo < 0
          or self._model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self._model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self._model_spec.baseline_geo
    elif isinstance(self._model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self._input_data.geo == self._model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self._model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    if self._model_spec.holdout_id is None:
      return None
    tensor = backend.to_tensor(self._model_spec.holdout_id, dtype=backend.bool_)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    if isinstance(self._model_spec.adstock_decay_spec, str):
      return adstock_hill.AdstockDecaySpec.from_consistent_type(
          self._model_spec.adstock_decay_spec
      )

    try:
      return self._create_adstock_decay_functions_from_channel_map(
          self._model_spec.adstock_decay_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `adstock_decay_spec` keys"
          f" {tuple(self._model_spec.adstock_decay_spec.keys())}. Keys should"
          " either contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())} or"
          " be one or more of {'media', 'rf', 'organic_media',"
          " 'organic_rf'}."
      ) from e

  def _create_adstock_decay_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> adstock_hill.AdstockDecaySpec:
    """Create `AdstockDecaySpec` from mapping from channels to decay functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      media_adstock_function = media_channel_builder(**channel_function_map)
    else:
      media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      rf_adstock_function = rf_channel_builder(**channel_function_map)
    else:
      rf_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_media_adstock_function = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_rf_adstock_function = organic_rf_channel_builder(
          **channel_function_map
      )
    else:
      organic_rf_adstock_function = constants.GEOMETRIC_DECAY

    return adstock_hill.AdstockDecaySpec(
        media=media_adstock_function,
        rf=rf_adstock_function,
        organic_media=organic_media_adstock_function,
        organic_rf=organic_rf_adstock_function,
    )

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, and so this method freezes the
    object's states before the computation graph is created.
    """
    cls = self.__class__
    # "Freeze" all @cached_property attributes by simply accessing them (with
    # `getattr()`).
    cached_properties = [
        attr
        for attr in dir(self)
        if isinstance(getattr(cls, attr, cls), functools.cached_property)
    ]
    for attr in cached_properties:
      _ = getattr(self, attr)
