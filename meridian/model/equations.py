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

"""Core mathematical equations for the Meridian model.

This module defines the `ModelEquations` class, which encapsulates the stateless
mathematical functions used in the Meridian MMM. This includes the core model
definitions, such as adstock, hill, and other transformations used
during model fitting. It requires a `ModelContext` instance for data access.
"""

from collections.abc import Sequence

from meridian import backend
from meridian import constants
from meridian.model import context


__all__ = [
    'ModelEquations',
]


class ModelEquations:
  """Provides core, stateless mathematical functions for Meridian MMM."""

  def __init__(self, model_context: context.ModelContext):
    self._context = model_context

  def adstock_hill_media(
      self,
      media: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    raise NotImplementedError

  def adstock_hill_rf(
      self,
      reach: backend.Tensor,
      frequency: backend.Tensor,
      alpha: backend.Tensor,
      ec: backend.Tensor,
      slope: backend.Tensor,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
      n_times_output: int | None = None,
  ) -> backend.Tensor:
    raise NotImplementedError

  def compute_non_media_treatments_baseline(
      self,
      non_media_baseline_values: Sequence[str | float] | None = None,
  ) -> backend.Tensor:
    raise NotImplementedError

  def linear_predictor_counterfactual_difference_media(
      self,
      media_transformed: backend.Tensor,
      alpha_m: backend.Tensor,
      ec_m: backend.Tensor,
      slope_m: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError

  def linear_predictor_counterfactual_difference_rf(
      self,
      rf_transformed: backend.Tensor,
      alpha_rf: backend.Tensor,
      ec_rf: backend.Tensor,
      slope_rf: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError

  def calculate_beta_x(
      self,
      is_non_media: bool,
      incremental_outcome_x: backend.Tensor,
      linear_predictor_counterfactual_difference: backend.Tensor,
      eta_x: backend.Tensor,
      beta_gx_dev: backend.Tensor,
  ) -> backend.Tensor:
    raise NotImplementedError
