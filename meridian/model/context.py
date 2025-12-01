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

"""Meridian module for model context."""

from collections.abc import Mapping, Sequence
import functools

import numpy as np

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers


class ModelContext:
  """Contains model context derived from input data and model specification.

  Attributes:
    input_data: An `InputData` object containing the input data for the model.
    model_spec: A `ModelSpec` object containing the model specification.
    n_geos: Number of geos in the data.
    n_media_channels: Number of media channels in the data.
    n_rf_channels: Number of reach and frequency (RF) channels in the data.
    n_organic_media_channels: Number of organic media channels in the data.
    n_organic_rf_channels: Number of organic reach and frequency (RF) channels
      in the data.
    n_controls: Number of control variables in the data.
    n_non_media_channels: Number of non-media treatment channels in the data.
    n_times: Number of time periods in the KPI or spend data.
    n_media_times: Number of time periods in the media data.
    is_national: A boolean indicating whether the data is national (single geo)
      or not (multiple geos).
    knot_info: A `KnotInfo` derived from input data and model spec.
    kpi: A tensor constructed from `input_data.kpi`.
    revenue_per_kpi: A tensor constructed from `input_data.revenue_per_kpi`. If
      `input_data.revenue_per_kpi` is None, then this is also None.
    controls: A tensor constructed from `input_data.controls`.
    non_media_treatments: A tensor constructed from
      `input_data.non_media_treatments`.
    population: A tensor constructed from `input_data.population`.
    media_tensors: A collection of media tensors derived from `input_data`.
    rf_tensors: A collection of Reach & Frequency (RF) media tensors.
    organic_media_tensors: A collection of organic media tensors.
    organic_rf_tensors: A collection of organic Reach & Frequency (RF) media
      tensors.
    total_spend: A tensor containing total spend, including
      `media_tensors.media_spend` and `rf_tensors.rf_spend`.
    total_outcome: A tensor containing the total outcome, aggregated over geos
      and times.
    controls_transformer: A `ControlsTransformer` to scale controls tensors
      using the model's controls data.
    non_media_transformer: A `CenteringAndScalingTransformer` to scale non-media
      treatmenttensors using the model's non-media treatment data.
    kpi_transformer: A `KpiTransformer` to scale KPI tensors using the model's
      KPI data.
    controls_scaled: The controls tensor after pre-modeling transformations
      including population scaling (for variables with
      `ModelSpec.control_population_scaling_id` set to `True`), centering by the
      mean, and scaling by the standard deviation.
    non_media_treatments_scaled: The non-media treatment tensor after
      pre-modeling transformations including population scaling (for variables
      with `ModelSpec.non_media_population_scaling_id` set to `True`), centering
      by the mean, and scaling by the standard deviation.
    kpi_scaled: The KPI tensor after pre-modeling transformations including
      population scaling, centering by the mean, and scaling by the standard
      deviation.
    media_effects_dist: A string to specify the distribution of media random
      effects across geos.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo.
    prior_broadcast: A `PriorDistribution` object containing broadcasted
      distributions.
    baseline_geo_idx: The index of the baseline geo.
    holdout_id: A tensor containing the holdout id, if present.
    set_total_media_contribution_prior: A boolean indicating whether to use total
      media contribution prior.
  """

  def __init__(
      self, input_data: data.InputData, model_spec: spec.ModelSpec
  ):
    self._input_data = input_data
    self._model_spec = model_spec
    self.set_total_media_contribution_prior = False

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.build_media_tensors(self.input_data, self.model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.build_rf_tensors(self.input_data, self.model_spec)

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self.input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self.input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self.input_data.kpi, dtype=backend.float32)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self.input_data.revenue_per_kpi is None:
      return None
    return backend.to_tensor(
        self.input_data.revenue_per_kpi, dtype=backend.float32
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    if self.input_data.controls is None:
      return None
    return backend.to_tensor(self.input_data.controls, dtype=backend.float32)

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self.input_data.non_media_treatments is None:
      return None
    return backend.to_tensor(
        self.input_data.non_media_treatments, dtype=backend.float32
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.to_tensor(self.input_data.population, dtype=backend.float32)

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.to_tensor(
        self.input_data.get_total_spend(), dtype=backend.float32
    )

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.to_tensor(
        self.input_data.get_total_outcome(), dtype=backend.float32
    )

  @property
  def n_geos(self) -> int:
    return len(self.input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self.input_data.media_channel is None:
      return 0
    return len(self.input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self.input_data.rf_channel is None:
      return 0
    return len(self.input_data.rf_channel)

  @property
  def n_organic_media_channels(self) -> int:
    if self.input_data.organic_media_channel is None:
      return 0
    return len(self.input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self.input_data.organic_rf_channel is None:
      return 0
    return len(self.input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    if self.input_data.control_variable is None:
      return 0
    return len(self.input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self.input_data.non_media_channel is None:
      return 0
    return len(self.input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self.input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self.input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self.model_spec.knots,
        enable_aks=self.model_spec.enable_aks,
        data=self.input_data,
        is_national=self.is_national,
    )

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for controls, if it exists."""
    if self.controls is None:
      return None

    if self.model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = backend.to_tensor(
          self.model_spec.control_population_scaling_id, dtype=backend.bool_
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
    if self.model_spec.non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.to_tensor(
          self.model_spec.non_media_population_scaling_id, dtype=backend.bool_
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
      return self.model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      # Should evaluate to False.
      return constants.NATIONAL_MODEL_SPEC_ARGS[
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self.model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self.model_spec.baseline_geo, int):
      if (
          self.model_spec.baseline_geo < 0
          or self.model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self.model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self.model_spec.baseline_geo
    elif isinstance(self.model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self.input_data.geo == self.model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self.model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    if self.model_spec.holdout_id is None:
      return None
    tensor = backend.to_tensor(self.model_spec.holdout_id, dtype=backend.bool_)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    if isinstance(self.model_spec.adstock_decay_spec, str):
      return adstock_hill.AdstockDecaySpec.from_consistent_type(
          self.model_spec.adstock_decay_spec
      )

    try:
      return self._create_adstock_decay_functions_from_channel_map(
          self.model_spec.adstock_decay_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `adstock_decay_spec` keys"
          f" {tuple(self.model_spec.adstock_decay_spec.keys())}. Keys should"
          " either contain only channel_names"
          f" {tuple(self.input_data.get_all_adstock_hill_channels().tolist())} or"
          " be one or more of {'media', 'rf', 'organic_media',"
          " 'organic_rf'}."
      ) from e

  def _create_adstock_decay_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> adstock_hill.AdstockDecaySpec:
    """Create `AdstockDecaySpec` from mapping from channels to decay functions."""

    for channel in channel_function_map:
      if channel not in self.input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self.input_data.media_channel is not None:
      media_channel_builder = self.input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      media_adstock_function = media_channel_builder(**channel_function_map)
    else:
      media_adstock_function = constants.GEOMETRIC_DECAY

    if self.input_data.rf_channel is not None:
      rf_channel_builder = self.input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      rf_adstock_function = rf_channel_builder(**channel_function_map)
    else:
      rf_adstock_function = constants.GEOMETRIC_DECAY

    if self.input_data.organic_media_channel is not None:
      organic_media_channel_builder = self.input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_media_adstock_function = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_adstock_function = constants.GEOMETRIC_DECAY

    if self.input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self.input_data.get_organic_rf_channels_argument_builder().with_default_value(
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

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    total_spend = self.input_data.get_total_spend()
    # Total spend can have 1, 2 or 3 dimensions. Aggregate by channel.
    if len(total_spend.shape) == 1:
      # Already aggregated by channel.
      agg_total_spend = total_spend
    elif len(total_spend.shape) == 2:
      agg_total_spend = np.sum(total_spend, axis=(0,))
    else:
      agg_total_spend = np.sum(total_spend, axis=(0, 1))

    return self.model_spec.prior.broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_organic_media_channels=self.n_organic_media_channels,
        n_organic_rf_channels=self.n_organic_rf_channels,
        n_controls=self.n_controls,
        n_non_media_channels=self.n_non_media_channels,
        unique_sigma_for_each_geo=self.unique_sigma_for_each_geo,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=self.set_total_media_contribution_prior,
        kpi=np.sum(self.input_data.kpi.values),
        total_spend=agg_total_spend,
    )
