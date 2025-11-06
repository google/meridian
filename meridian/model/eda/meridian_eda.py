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

from typing import Literal, TYPE_CHECKING, Union

import altair as alt
from meridian import constants
from meridian.model.eda import constants as eda_constants
import pandas as pd

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top

__all__ = [
    'MeridianEDA',
]


class MeridianEDA:
  """Class for running pre-modeling exploratory data analysis for Meridian InputData."""

  _PAIRWISE_CORR_COLOR_SCALE = alt.Scale(
      domain=[-1.0, 0.0, 1.0],
      range=['#1f78b4', '#f7f7f7', '#e34a33'],  # Blue-light grey-Orange
      type='linear',
  )

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

  def plot_pairwise_correlation(
      self, geos: Union[int, list[str], Literal['nationalize']] = 1
  ) -> alt.Chart:
    """Plots the Pairwise Correlation data.

    Args:
      geos: Defines which geos to plot. - int: The number of top geos to plot,
        ranked by population. - list[str]: A specific list of geo names to plot.
        - 'nationalize': Aggregates all geos into a single national view.
        Defaults to 1 (plotting the top geo). If the data is already at a
        national level, this parameter is ignored and a national plot is
        generated.

    Returns:
      Altair chart(s) of the Pairwise Correlation data.
    """
    geos_to_plot = self._validate_and_get_geos_to_plot(geos)
    is_national = self._meridian.is_national
    nationalize_geos = geos == 'nationalize'

    if is_national or nationalize_geos:
      pairwise_corr_artifact = (
          self._meridian.eda_engine.check_national_pairwise_corr().get_national_artifact
      )
      if pairwise_corr_artifact is None:
        raise ValueError('EDAOutcome does not have national artifact.')
    else:
      pairwise_corr_artifact = (
          self._meridian.eda_engine.check_geo_pairwise_corr().get_geo_artifact
      )
      if pairwise_corr_artifact is None:
        raise ValueError('EDAOutcome does not have geo artifact.')
    pairwise_corr_data = pairwise_corr_artifact.corr_matrix.to_dataframe()

    charts = []
    for geo_to_plot in geos_to_plot:
      title = (
          'Pairwise correlations among all treatments and controls for'
          f' {geo_to_plot}'
      )

      if not (is_national or nationalize_geos):
        plot_data = (
            pairwise_corr_data.xs(geo_to_plot, level=constants.GEO)
            .rename_axis(
                index=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2]
            )
            .reset_index()
        )
      else:
        plot_data = pairwise_corr_data.rename_axis(
            index=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2]
        ).reset_index()
      plot_data.columns = [
          eda_constants.VARIABLE_1,
          eda_constants.VARIABLE_2,
          eda_constants.CORRELATION,
      ]
      unique_variables = plot_data[eda_constants.VARIABLE_1].unique()
      variable_to_index = {name: i for i, name in enumerate(unique_variables)}

      plot_data['idx1'] = plot_data[eda_constants.VARIABLE_1].map(
          variable_to_index
      )
      plot_data['idx2'] = plot_data[eda_constants.VARIABLE_2].map(
          variable_to_index
      )
      lower_triangle_data = plot_data[plot_data['idx2'] > plot_data['idx1']]

      charts.append(
          self._plot_2d_heatmap(lower_triangle_data, title, unique_variables)
      )
    final_chart = (
        alt.vconcat(*charts)
        .resolve_legend(color='independent')
        .configure_axis(labelAngle=315)
        .configure_title(anchor='start')
        .configure_view(stroke=None)
    )
    return final_chart

  def _plot_2d_heatmap(
      self, data: pd.DataFrame, title: str, unique_variables: list[str]
  ) -> alt.Chart:
    """Plots a 2D heatmap."""
    # Base chart with position encodings
    base = (
        alt.Chart(data)
        .encode(
            x=alt.X(
                f'{eda_constants.VARIABLE_1}:N',
                title=None,
                sort=unique_variables,
                scale=alt.Scale(domain=unique_variables),
            ),
            y=alt.Y(
                f'{eda_constants.VARIABLE_2}:N',
                title=None,
                sort=unique_variables,
                scale=alt.Scale(domain=unique_variables),
            ),
        )
        .properties(title=title)
    )

    # Heatmap layer (rectangles)
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            f'{eda_constants.CORRELATION}:Q',
            scale=self._PAIRWISE_CORR_COLOR_SCALE,
            legend=alt.Legend(title=eda_constants.CORRELATION),
        ),
        tooltip=[
            eda_constants.VARIABLE_1,
            eda_constants.VARIABLE_2,
            alt.Tooltip(f'{eda_constants.CORRELATION}:Q', format='.3f'),
        ],
    )

    # Text annotation layer (values)
    text = base.mark_text().encode(
        text=alt.Text(f'{eda_constants.CORRELATION}:Q', format='.3f'),
        color=alt.value('black'),
    )

    # Combine layers and apply final configurations
    chart = (heatmap + text).properties(width=350, height=350)

    return chart

  def _generate_pairwise_correlation_report(self) -> str:
    """Creates the HTML snippet for Pairwise Correlation report section."""
    # TODO: Implement.
    raise NotImplementedError()

  def _validate_and_get_geos_to_plot(
      self, geos: Union[int, list[str], Literal['nationalize']]
  ) -> list[str]:
    """Validates and returns the geos to plot."""
    ## Validate
    is_national = self._meridian.is_national
    if is_national or geos == 'nationalize':
      geos_to_plot = [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]
    elif isinstance(geos, int):
      if geos > len(self._meridian.input_data.geo) or geos <= 0:
        raise ValueError(
            'geos must be a positive integer less than or equal to the number'
            ' of geos in the data.'
        )
      geos_to_plot = self._meridian.input_data.get_n_top_largest_geos(geos)
    else:
      geos_to_plot = geos

    if (
        not is_national and geos != 'nationalize'
    ):  # if national then geos_to_plot will be ignored
      for geo in geos_to_plot:
        if geo not in self._meridian.input_data.geo:
          raise ValueError(f'Geo {geo} does not exist in the data.')
      if len(geos_to_plot) != len(set(geos_to_plot)):
        raise ValueError('geos must not contain duplicate values.')

    return geos_to_plot
