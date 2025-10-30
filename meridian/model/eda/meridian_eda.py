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

import altair as alt
from meridian import constants
from meridian.model import model
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome


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
    # TODO: this needs to be changed after EDA Phase 0 refactor to
    # not depend directly on edaengine or meridian
    self._eda_engine = eda_engine.EDAEngine(meridian=meridian)

  def generate_and_save_report(self, filename: str, filepath: str):
    """Generates and saves the 2 page HTML report containing findings in EDA about given InputData.

    Args:
      filename: The filename for the generated HTML output.
      filepath: The path to the directory where the file will be saved.
    """
    # TODO: Implement.
    raise NotImplementedError()

  def plot_pairwise_correlation(
      self,
      n_geos_to_plot: int = 1,
      geos_to_plot: list[str] | None = None,
      nationalize_geos: bool = False,
  ) -> alt.Chart:
    """Plots the Pairwise Correlation data.

    Args:
      n_geos_to_plot: The number of desired geos to plot. Will choose the top n
        geos ranked by total spend then total KPI. This will be ignored if the
        data is national level.
      geos_to_plot: The specific geos desired to plot. Will raise an error if
        any of the given geos do not exist in the InputData. This will be
        ignored if the data is national level. If provided simulataneously with
        n_geos_to_plot, it will ignore n_geos_to_plot and geos_to_plot will take
        precedence.
      nationalize_geos: A boolean indicating whether to aggregate all geos into
        a single national level geo. If the data is already national, it will
        just run normally.

    Returns:
      Altair chart(s) of the Pairwise Correlation data.
    """
    geos_to_plot = self._validate_and_get_geos_to_plot(
        n_geos_to_plot, geos_to_plot, nationalize_geos
    )
    is_national = self._eda_engine._meridian.is_national

    if is_national or nationalize_geos:
      pairwise_corr_outcome = self._eda_engine.check_national_pairwise_corr()
      pairwise_corr_data = pairwise_corr_outcome.analysis_artifacts[
          0
      ].corr_matrix.to_dataframe()
    else:
      pairwise_corr_outcome = self._eda_engine.check_geo_pairwise_corr()
      pairwise_corr_data = pairwise_corr_outcome.analysis_artifacts[
          1
      ].corr_matrix.to_dataframe()

    for finding in pairwise_corr_outcome.findings:
      if finding.severity == eda_outcome.EDASeverity.ERROR:
        raise ValueError(finding.explanation)

    charts = []
    for geo_to_plot in geos_to_plot:
      title = (
          'Pairwise correlations among all treatments and controls for'
          f' {geo_to_plot}'
      )

      if not (is_national or nationalize_geos):
        plot_data = (
            pairwise_corr_data.xs(geo_to_plot, level='geo')
            .rename_axis(index=['variable1', 'variable2'])
            .reset_index()
        )
      else:
        plot_data = pairwise_corr_data.rename_axis(
            index=['variable1', 'variable2']
        ).reset_index()
      plot_data.columns = ['variable1', 'variable2', 'correlation']
      unique_variables = plot_data['variable1'].unique()
      variable_to_index = {name: i for i, name in enumerate(unique_variables)}

      plot_data['idx1'] = plot_data['variable1'].map(variable_to_index)
      plot_data['idx2'] = plot_data['variable2'].map(variable_to_index)
      lower_triangle_data = plot_data[plot_data['idx2'] > plot_data['idx1']]

      # Base chart with position encodings
      base = (
          alt.Chart(lower_triangle_data)
          .encode(
              x=alt.X(
                  'variable1:N',
                  title=None,
                  sort=unique_variables,
                  scale=alt.Scale(domain=unique_variables),
              ),
              y=alt.Y(
                  'variable2:N',
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
              'correlation:Q',
              scale=self._PAIRWISE_CORR_COLOR_SCALE,
              legend=alt.Legend(title='Correlation'),
          ),
          tooltip=[
              'variable1',
              'variable2',
              alt.Tooltip('correlation:Q', format='.3f'),
          ],
      )

      # Text annotation layer (values)
      text = base.mark_text().encode(
          text=alt.Text('correlation:Q', format='.3f'), color=alt.value('black')
      )

      # Combine layers and apply final configurations
      chart = (heatmap + text).properties(width=350, height=350)
      charts.append(chart)
    final_chart = (
        alt.vconcat(*charts)
        .resolve_legend(color='independent')
        .configure_axis(labelAngle=315)
        .configure_title(anchor='start')
        .configure_view(stroke=None)
    )
    return final_chart

  def _generate_pairwise_correlation_report(self) -> str:
    """Creates the HTML snippet for Pairwise Correlation report section."""
    # TODO: Implement.
    raise NotImplementedError()

  def _validate_and_get_geos_to_plot(
      self,
      n_geos_to_plot: int,
      geos_to_plot: list[str] | None,
      nationalize_geos: bool,
  ) -> list[str]:
    """Validates and returns the geos to plot."""
    ## Validate
    is_national = self._eda_engine._meridian.is_national
    n_geos = len(self._eda_engine._meridian.input_data.geo)
    if is_national or nationalize_geos:
      geos_to_plot = [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME]

    if geos_to_plot is None:
      if n_geos_to_plot > n_geos or n_geos_to_plot < 0:
        raise ValueError(
            'n_geos_to_plot must be less than or equal to the number of geos'
            ' in the data and greater than 0.'
        )
      geos_to_plot = self._eda_engine._meridian.input_data.ranked_geos[
          :n_geos_to_plot
      ]

    if (
        not is_national and not nationalize_geos
    ):  # if national then geos_to_plot will be ignored
      for geo in geos_to_plot:
        if geo not in self._eda_engine._meridian.input_data.geo:
          raise ValueError(f'Geo {geo} does not exist in the data.')
      if len(geos_to_plot) != len(set(geos_to_plot)):
        raise ValueError('geos_to_plot must not contain duplicate geos.')

    return geos_to_plot
