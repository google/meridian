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

"""Contains data transformers for various inputs of the Meridian model."""

import abc

from meridian import backend
import numpy as np


__all__ = [
    "TensorTransformer",
    "MediaTransformer",
    "CenteringAndScalingTransformer",
    "KpiTransformer",
]


class TensorTransformer(abc.ABC):
  """Abstract class for data transformers."""

  @abc.abstractmethod
  @backend.function(jit_compile=True)
  def forward(self, tensor: backend.Tensor) -> backend.Tensor:
    """Transforms a given tensor."""
    raise NotImplementedError("`forward` must be implemented.")

  @abc.abstractmethod
  @backend.function(jit_compile=True)
  def inverse(self, tensor: backend.Tensor) -> backend.Tensor:
    """Transforms back a given tensor."""
    raise NotImplementedError("`inverse` must be implemented.")


class MediaTransformer(TensorTransformer):
  """Contains forward and inverse media transformation methods.

  This class stores scale factors computed from per-geo medians of the `media`
  data, normalized by the geo population.
  """

  def __init__(
      self,
      media: backend.Tensor,
      population: backend.Tensor,
  ):
    """`MediaTransformer` constructor.

    Args:
      media: A tensor of dimension `(n_geos, n_media_times, n_media_channels)`
        containing the media data, used to compute the scale factors.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to compute the scale factors.
    """
    population_scaled_media = backend.divide_no_nan(
        media, population[:, backend.newaxis, backend.newaxis]
    )
    # Replace zeros with NaNs
    population_scaled_media_nan = backend.where(
        population_scaled_media == 0, np.nan, population_scaled_media
    )
    # Tensor of medians of the positive portion of `media`. Used as a component
    # for scaling.
    self._population_scaled_median_m = backend.numpy_function(
        func=lambda x: np.nanmedian(x, axis=[0, 1]),
        inp=[population_scaled_media_nan],
        Tout=backend.float32,
    )
    if backend.reduce_any(backend.is_nan(self._population_scaled_median_m)):
      raise ValueError(
          "MediaTransformer has a NaN population-scaled non-zero median due to"
          " a media channel with either all zeroes or all NaNs."
      )
    # Tensor of dimensions (`n_geos` x 1) of weights for scaling `metric`.
    self._scale_factors_gm = backend.einsum(
        "g,m->gm", population, self._population_scaled_median_m
    )

  @property
  def population_scaled_median_m(self):
    return self._population_scaled_median_m

  @backend.function(jit_compile=True)
  def forward(self, tensor: backend.Tensor) -> backend.Tensor:
    """Scales a given tensor using the stored scale factors."""
    return tensor / self._scale_factors_gm[:, backend.newaxis, :]

  @backend.function(jit_compile=True)
  def inverse(self, tensor: backend.Tensor) -> backend.Tensor:
    """Scales a given tensor using the inversed stored scale factors."""
    return tensor * self._scale_factors_gm[:, backend.newaxis, :]


class CenteringAndScalingTransformer(TensorTransformer):
  """Applies centering and scaling transformations to a tensor.

  This class transforms a tensor so each variable has mean zero and standard
  deviation one. Optionally, each variable can be scaled by population before
  the centering and scaling transformations are applied. The class stores the
  mean and standard deviation of each variable.
  """

  def __init__(
      self,
      tensor: backend.Tensor,
      population: backend.Tensor,
      population_scaling_id: backend.Tensor | None = None,
  ):
    """`CenteringAndScalingTransformer` constructor.

    Args:
      tensor: A tensor of dimension `(n_geos, n_times, n_channel)` used to
        compute the means and standard deviations.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to compute the scale factors.
      population_scaling_id: An optional boolean tensor of dimension
        `(n_channels,)` indicating the variables for which the value will be
        scaled by population.
    """
    if population_scaling_id is not None:
      self._population_scaling_factors = backend.where(
          population_scaling_id,
          population[:, None],
          backend.ones_like(population)[:, None],
      )
      population_scaled_tensor = (
          tensor / self._population_scaling_factors[:, None, :]
      )
      self._means = backend.reduce_mean(population_scaled_tensor, axis=(0, 1))
      self._stdevs = backend.reduce_std(population_scaled_tensor, axis=(0, 1))
    else:
      self._population_scaling_factors = None
      self._means = backend.reduce_mean(tensor, axis=(0, 1))
      self._stdevs = backend.reduce_std(tensor, axis=(0, 1))

  @backend.function(jit_compile=True)
  def forward(
      self, tensor: backend.Tensor, apply_population_scaling: bool = True
  ) -> backend.Tensor:
    """Scales a given tensor using the stored coefficients.

    Args:
      tensor: A tensor of dimension `(n_geos, n_times, n_channels)` to
        transform.
      apply_population_scaling: Whether to apply population scaling before the
        normalization by means and standard deviations.
    """
    if (
        apply_population_scaling
        and self._population_scaling_factors is not None
    ):
      tensor /= self._population_scaling_factors[:, None, :]
    return backend.divide_no_nan(tensor - self._means, self._stdevs)

  @backend.function(jit_compile=True)
  def inverse(self, tensor: backend.Tensor) -> backend.Tensor:
    """Scales back a given tensor using the stored coefficients."""
    scaled_tensor = tensor * self._stdevs + self._means
    return (
        scaled_tensor * self._population_scaling_factors[:, None, :]
        if self._population_scaling_factors is not None
        else scaled_tensor
    )


class KpiTransformer(TensorTransformer):
  """Contains forward and inverse KPI transformation methods.

  This class stores coefficients to scale KPI, first by geo and then
  by mean and standard deviation of KPI.
  """

  def __init__(
      self,
      kpi: backend.Tensor,
      population: backend.Tensor,
  ):
    """`KpiTransformer` constructor.

    Args:
      kpi: A tensor of dimension `(n_geos, n_times)` containing the KPI data,
        used to compute the mean and stddev.
      population: A tensor of dimension `(n_geos,)` containing the population of
        each geo, used to to compute the population scale factors.
    """
    self._population = population
    population_scaled_kpi = backend.divide_no_nan(
        kpi, self._population[:, backend.newaxis]
    )
    self._population_scaled_mean = backend.reduce_mean(population_scaled_kpi)
    self._population_scaled_stdev = backend.reduce_std(population_scaled_kpi)

  @property
  def population_scaled_mean(self):
    return self._population_scaled_mean

  @property
  def population_scaled_stdev(self):
    return self._population_scaled_stdev

  @backend.function(jit_compile=True)
  def forward(self, tensor: backend.Tensor) -> backend.Tensor:
    """Scales a given tensor using the stored coefficients."""
    return backend.divide_no_nan(
        backend.divide_no_nan(tensor, self._population[:, backend.newaxis])
        - self._population_scaled_mean,
        self._population_scaled_stdev,
    )

  @backend.function(jit_compile=True)
  def inverse(self, tensor: backend.Tensor) -> backend.Tensor:
    """Scales back a given tensor using the stored coefficients."""
    return (
        tensor * self._population_scaled_stdev + self._population_scaled_mean
    ) * self._population[:, backend.newaxis]
