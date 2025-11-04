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

"""Module containing Meridian related exploratory data analysis (EDA) functionalities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    "MeridianEDA",
]


class MeridianEDA:
  """Class for running pre-modeling exploratory data analysis for Meridian InputData."""

  def __init__(
      self,
      meridian: model.Meridian,
  ):
    self._meridian = meridian

  def generate_and_save_report(self, filename: str, filepath: str):
    """Generates and saves the 2 page HTML report containing findings in EDA about given InputData.

    Args:
      filename: The filename for the generated HTML output.
      filepath: The path to the directory where the file will be saved.
    """
    # TODO: Implement.
    raise NotImplementedError()

  def plot_pairwise_correlation(self) -> alt.Chart:
    """Plots the Pairwise Correlation data."""
    # TODO: Implement.
    raise NotImplementedError()

  def _generate_pairwise_correlation_report(self) -> str:
    """Creates the HTML snippet for Pairwise Correlation report section."""
    # TODO: Implement.
    raise NotImplementedError()
