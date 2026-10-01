# Copyright 2026 The Meridian Authors.
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

"""Calibration overview chart builders for the Model Quality Checks."""

from typing import cast

import altair as alt
from meridian import backend
from meridian.analysis.review import constants
from meridian.analysis.review import results
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
import numpy as np
import pandas as pd

__all__ = [
    "build_calibration_overview_chart",
]


_create_roi_grid = calibration_plots.create_roi_grid


def build_calibration_overview_chart(
    ch_data: results.CalibrationOverviewChannelData | None,
) -> alt.HConcatChart | None:
  """Builds the 1x3 side-by-side calibration overview chart for a single channel."""
  if (
      ch_data is None
      or ch_data.calibrated_output is None
      or ch_data.calibrated_prior_dist is None
  ):
    return None

  indexed_experiments = calibration_plots.filter_and_sort_experiments(
      ch_data.calibrated_output.experiments,
      lambda exp: exp.adjusted_experiment_result.standard_error,
      limit_experiments=constants.MAX_EXPERIMENTS_FOR_OVERVIEW_CARD,
      sort_experiments=True,
  )
  plot_data = calibration_plots.prepare_calibration_data(
      calibrated_output=ch_data.calibrated_output,
      calibrated_prior_dist=ch_data.calibrated_prior_dist,
      indexed_experiments=indexed_experiments,
      rng_handler=backend.RNGHandler(eda_constants.DEFAULT_PRIOR_SEED),
  )

  # Prepare posterior DataFrame if available.
  posterior_df = None
  if (
      ch_data.posterior_samples is not None
      and len(ch_data.posterior_samples) > 0
  ):
    grid = _create_roi_grid(
        ch_data.calibrated_prior_dist, [exp for _, exp in indexed_experiments]
    )
    density, bins = np.histogram(
        ch_data.posterior_samples,
        bins=eda_constants.HISTOGRAM_BINS,
        range=(grid[0], grid[-1]),
        density=True,
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    posterior_df = calibration_plots.make_calibration_plot_df(
        bin_centers, density, constants.MERIDIAN_POSTERIOR
    )

  # Build unified color scale and domain.
  exp_labels = [
      df[eda_constants.LABEL].iloc[0]
      for df in plot_data.exp_dfs
      if not df.empty
  ]
  domain = []
  range_ = []
  if plot_data.baseline_df is not None and not plot_data.baseline_df.empty:
    domain.append(constants.BASELINE_PRIOR)
    range_.append(constants.BASELINE_PRIOR_COLOR)

  for i, label in enumerate(exp_labels):
    domain.append(label)
    range_.append(
        constants.CALIBRATION_EXPERIMENT_COLORS[
            i % len(constants.CALIBRATION_EXPERIMENT_COLORS)
        ]
    )

  domain.extend([
      constants.INTERMEDIARY_PRIOR,
      constants.CALIBRATED_MERIDIAN_PRIOR,
  ])
  range_.extend([
      constants.INTERMEDIARY_PRIOR_COLOR,
      constants.CALIBRATED_PRIOR_COLOR,
  ])

  if posterior_df is not None and not posterior_df.empty:
    domain.append(constants.MERIDIAN_POSTERIOR)
    range_.append(constants.POSTERIOR_HISTOGRAM_COLOR)

  unified_color_scale = alt.Scale(domain=domain, range=range_)
  legend_selection = alt.selection_point(
      fields=[eda_constants.LABEL], bind="legend"
  )
  tooltips = [
      alt.Tooltip(f"{eda_constants.LABEL}:N", title="Type"),
      alt.Tooltip(f"{constants.ROI}:Q", title="ROI", format=".2f"),
      alt.Tooltip(f"{eda_constants.DENSITY}:Q", title="Density", format=".4f"),
  ]

  def _make_bar_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{constants.ROI}:Q",
                title="ROI",
                scale=alt.Scale(domainMin=0, clamp=True),
            ),
            y=alt.Y(f"{eda_constants.DENSITY}:Q", title="Density"),
            color=alt.Color(
                f"{eda_constants.LABEL}:N",
                scale=unified_color_scale,
                legend=alt.Legend(title=None, symbolType="square"),
            ),
            opacity=alt.condition(
                legend_selection, alt.value(0.4), alt.value(0.1)
            ),
            tooltip=tooltips,
        )
    )

  intermediary_chart = _make_bar_chart(plot_data.intermediary_df)
  calibrated_line_chart = calibration_plots.make_density_line_chart(
      plot_data.calibrated_df,
      unified_color_scale,
      legend_selection,
      tooltips,
      stroke_width=2.5,
  )

  def _make_subplot(
      title: str, layers: list[alt.Chart], dfs: list[pd.DataFrame]
  ) -> alt.LayerChart:
    hover_layers = calibration_plots.create_interactive_hover_layers(
        cast(pd.DataFrame, pd.concat(dfs)), unified_color_scale, tooltips
    )
    return (
        alt.layer(*layers, *hover_layers)
        .properties(
            title=alt.TitleParams(text=title, anchor="start", fontSize=12),
            width=240,
            height=200,
        )
        .add_params(legend_selection)
    )

  # Subplot 1: Incrementality Experiments & Intermediary Prior (reused from EDA)
  left_layers = [intermediary_chart]
  left_dfs = [plot_data.intermediary_df]
  if plot_data.baseline_df is not None and not plot_data.baseline_df.empty:
    left_layers.append(
        calibration_plots.make_density_line_chart(
            plot_data.baseline_df,
            unified_color_scale,
            legend_selection,
            tooltips,
            stroke_dash=[5, 5],
        )
    )
    left_dfs.append(plot_data.baseline_df)

  if plot_data.exp_dfs:
    combined_exp_df = cast(pd.DataFrame, pd.concat(plot_data.exp_dfs))
    left_layers.append(
        calibration_plots.make_density_line_chart(
            combined_exp_df,
            unified_color_scale,
            legend_selection,
            tooltips,
        )
    )
    left_dfs.append(combined_exp_df)

  left_subplot = _make_subplot(
      constants.CALIBRATION_LEFT_PLOT_TITLE, left_layers, left_dfs
  )

  # Subplot 2: Intermediary & Calibrated Priors (reused from EDA)
  middle_subplot = _make_subplot(
      constants.CALIBRATION_MIDDLE_PLOT_TITLE,
      [intermediary_chart, calibrated_line_chart],
      [plot_data.intermediary_df, plot_data.calibrated_df],
  )

  # Subplot 3: Calibrated Prior & Meridian Posterior
  if posterior_df is not None and not posterior_df.empty:
    bar_chart = _make_bar_chart(posterior_df)
    bar_df = posterior_df
  else:
    bar_chart = intermediary_chart
    bar_df = plot_data.intermediary_df

  right_subplot = _make_subplot(
      constants.CALIBRATION_RIGHT_PLOT_TITLE,
      [bar_chart, calibrated_line_chart],
      [bar_df, plot_data.calibrated_df],
  )

  return alt.hconcat(left_subplot, middle_subplot, right_subplot).resolve_scale(
      y="shared"
  )
