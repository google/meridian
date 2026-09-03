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

"""Plotting and visualization functions for prior calibration in EDA."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import TypeVar

import altair as alt
from meridian import backend
from meridian import constants
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import constants as eda_constants
import numpy as np
import pandas as pd
import xarray as xr


def make_calibration_plot_df(
    x: np.ndarray, y: np.ndarray, label: str
) -> pd.DataFrame:
  """Helper to create a DataFrame for calibration plotting."""
  return pd.DataFrame({
      constants.ROI: x,
      eda_constants.DENSITY: y,
      eda_constants.LABEL: label,
  })


def create_roi_grid(
    calibrated_prior_dist: backend.tfd.Distribution,
    experiments: Sequence[calibration_base.CalibratedExperiment],
) -> np.ndarray:
  """Creates a grid of ROI values for plotting PDFs."""
  prior_upper_quantile = float(
      np.array(
          calibrated_prior_dist.quantile(eda_constants.PRIOR_QUANTILE_THRESHOLD)
      ).item()
  )
  prior_lower_quantile = float(
      np.array(
          calibrated_prior_dist.quantile(
              1 - eda_constants.PRIOR_QUANTILE_THRESHOLD
          )
      ).item()
  )

  exp_mins, exp_maxs = [], []
  for exp in experiments:
    res = exp.adjusted_experiment_result
    margin = eda_constants.EXPERIMENT_SD_MULTIPLIER * res.standard_error
    exp_mins.append(res.point_estimate - margin)
    exp_maxs.append(res.point_estimate + margin)

  min_val = min([prior_lower_quantile] + exp_mins)
  x_min = (
      eda_constants.ROI_GRID_BOUND_MULTIPLIER * min_val
      if min_val < 0
      else eda_constants.MIN_ROI
  )

  max_val = max([prior_upper_quantile] + exp_maxs)
  x_max = eda_constants.ROI_GRID_BOUND_MULTIPLIER * max_val
  return np.linspace(x_min, x_max, eda_constants.ROI_GRID_POINTS)


_T = TypeVar('_T')


def filter_and_sort_experiments(
    experiments: Sequence[_T],
    get_se_fn: Callable[[_T], float],
    limit_experiments: int | None,
    sort_experiments: bool,
) -> Sequence[tuple[int, _T]]:
  """Associates experiments with 1-based indices, filters by SE, and sorts by index."""
  indexed_experiments = list(enumerate(experiments, start=1))
  if sort_experiments:
    sorted_experiments = sorted(
        indexed_experiments, key=lambda item: get_se_fn(item[1])
    )
  else:
    sorted_experiments = list(indexed_experiments)

  if limit_experiments is not None:
    sorted_experiments = sorted_experiments[:limit_experiments]
  return sorted(sorted_experiments, key=lambda item: item[0])


@dataclasses.dataclass(frozen=True)
class CalibrationPlotData:
  """Container for calibration plotting DataFrames.

  Attributes:
    baseline_df: Optional DataFrame containing the baseline prior PDF curve
      data, or None if no baseline prior exists.
    exp_dfs: List of DataFrames, each containing a single experiment's
      likelihood PDF curve data.
    intermediary_df: DataFrame containing the intermediary prior density
      histogram data.
    calibrated_df: DataFrame containing the calibrated Meridian prior PDF curve
      data.
  """

  baseline_df: pd.DataFrame | None
  exp_dfs: list[pd.DataFrame]
  intermediary_df: pd.DataFrame
  calibrated_df: pd.DataFrame


def get_experiment_label_suffix(
    source_type: calibration_base.SourceType,
) -> str:
  """Returns the formatted label suffix based on experiment source type."""
  return eda_constants.EXPERIMENT_SOURCE_TYPE_TO_LABEL_SUFFIX.get(
      source_type,
      eda_constants.EXPERIMENT_SOURCE_TYPE_TO_LABEL_SUFFIX[
          calibration_base.SourceType.GENERIC
      ],
  )


def prepare_calibration_data(
    calibrated_output: calibration_base.CalibrationOutput,
    calibrated_prior_dist: backend.tfd.Distribution,
    indexed_experiments: Sequence[
        tuple[int, calibration_base.CalibratedExperiment]
    ],
    rng_handler: backend.RNGHandler,
) -> CalibrationPlotData:
  """Prepares the DataFrames for calibration plotting."""
  samples = np.array(
      calibrated_output.intermediary_prior.sample(
          sample_shape=(eda_constants.DEFAULT_PRIOR_N_DRAW,),
          seed=rng_handler.get_next_seed(),
      )
  ).flatten()

  experiments = [exp for _, exp in indexed_experiments]
  grid = create_roi_grid(calibrated_prior_dist, experiments)
  x_min, x_max = grid[0], grid[-1]

  density, bins = np.histogram(
      samples,
      bins=eda_constants.HISTOGRAM_BINS,
      range=(x_min, x_max),
      density=True,
  )
  bin_centers = (bins[:-1] + bins[1:]) / 2

  baseline_df = None
  if calibrated_output.baseline_prior is not None:
    baseline_pdf = np.array(calibrated_output.baseline_prior.prob(grid))
    baseline_df = make_calibration_plot_df(
        grid, baseline_pdf, eda_constants.BASELINE_PRIOR
    )

  exp_dfs = []
  for exp_idx, exp in indexed_experiments:
    mu = exp.adjusted_experiment_result.point_estimate
    sigma = exp.adjusted_experiment_result.standard_error
    exp_dist = backend.tfd.Normal(loc=mu, scale=sigma)
    exp_pdf = np.array(exp_dist.prob(grid))
    label_suffix = get_experiment_label_suffix(exp.source_type)
    label = f'{eda_constants.EXPERIMENT_LABEL_PREFIX} {exp_idx}{label_suffix}'
    exp_dfs.append(make_calibration_plot_df(grid, exp_pdf, label))

  intermediary_df = make_calibration_plot_df(
      bin_centers, density, eda_constants.INTERMEDIARY_PRIOR
  )

  calibrated_pdf = np.array(calibrated_prior_dist.prob(grid))
  calibrated_df = make_calibration_plot_df(
      grid, calibrated_pdf, eda_constants.CALIBRATED_PRIOR
  )

  return CalibrationPlotData(
      baseline_df=baseline_df,
      exp_dfs=exp_dfs,
      intermediary_df=intermediary_df,
      calibrated_df=calibrated_df,
  )


def make_density_line_chart(
    df: pd.DataFrame,
    color_scale: alt.Scale,
    legend_selection: alt.Parameter,
    tooltips: Sequence[alt.Tooltip],
    stroke_dash: Sequence[int] | None = None,
    stroke_width: float = 2.0,
) -> alt.Chart:
  """Creates a line chart with standardized encoding, tooltips, and legend selection opacity."""
  kwargs = {'strokeWidth': stroke_width}
  if stroke_dash:
    kwargs['strokeDash'] = stroke_dash  # pyrefly: ignore[bad-assignment]
  return (
      alt.Chart(df)
      .mark_line(**kwargs)
      .encode(
          x=f'{constants.ROI}:Q',
          y=f'{eda_constants.DENSITY}:Q',
          color=alt.Color(f'{eda_constants.LABEL}:N', scale=color_scale),
          opacity=alt.condition(  # pyrefly: ignore[no-matching-overload]
              legend_selection, alt.value(1.0), alt.value(0.15)
          ),
          tooltip=tooltips,
      )
  )


def create_interactive_hover_layers(
    combined_df: pd.DataFrame,
    color_scale: alt.Scale,
    tooltips: Sequence[alt.Tooltip],
) -> list[alt.Chart]:
  """Creates transparent trigger points, active circle highlights, and vertical rule lines."""
  hover_selection = alt.selection_point(
      fields=[constants.ROI],
      nearest=True,
      on='pointerover',
      empty=False,
  )
  points = (
      alt.Chart(combined_df)
      .mark_point(size=1)
      .encode(
          x=f'{constants.ROI}:Q',
          opacity=alt.value(0.0),
      )
      .add_params(hover_selection)
  )
  active_circles = (
      alt.Chart(combined_df)
      .mark_circle(size=50)
      .encode(
          x=f'{constants.ROI}:Q',
          y=f'{eda_constants.DENSITY}:Q',
          color=alt.Color(f'{eda_constants.LABEL}:N', scale=color_scale),
          opacity=alt.condition(
              hover_selection, alt.value(1.0), alt.value(0.0)
          ),
          tooltip=tooltips,
      )
  )
  rule = (
      alt.Chart(combined_df)
      .mark_rule(color='gray', strokeDash=[2, 2], opacity=0.7)
      .encode(x=f'{constants.ROI}:Q')
      .transform_filter(hover_selection)
  )
  return [points, active_circles, rule]


def build_calibration_chart(
    ch_name: str,
    plot_data: CalibrationPlotData,
    spend: float | None = None,
) -> alt.Chart:
  """Builds the Altair chart for a single channel.

  Args:
    ch_name: The name of the channel being plotted.
    plot_data: Container holding the empirical, baseline, calibrated, and
      experiment DataFrames.
    spend: Optional total media spend for the channel, used in plot titles.

  Returns:
    An Altair Chart containing the 1x2 side-by-side calibration subplots.
  """
  exp_labels = []
  for df in plot_data.exp_dfs:
    if not df.empty and eda_constants.LABEL in df.columns:
      exp_labels.append(df[eda_constants.LABEL].iloc[0])

  domain = []
  range_ = []
  if plot_data.baseline_df is not None:
    domain.append(eda_constants.BASELINE_PRIOR)
    range_.append(eda_constants.BASELINE_PRIOR_COLOR)

  domain.extend(exp_labels)
  range_.extend([
      eda_constants.EXPERIMENT_COLORS[i % len(eda_constants.EXPERIMENT_COLORS)]
      for i in range(len(exp_labels))
  ])

  domain.extend([
      eda_constants.INTERMEDIARY_PRIOR,
      eda_constants.CALIBRATED_PRIOR,
  ])
  range_.extend([
      eda_constants.INTERMEDIARY_PRIOR_COLOR,
      eda_constants.CALIBRATED_PRIOR_COLOR,
  ])

  unified_color_scale = alt.Scale(domain=domain, range=range_)

  x_title = constants.ROI.upper()
  y_title = eda_constants.DENSITY.capitalize()

  legend_selection = alt.selection_point(
      fields=[eda_constants.LABEL], bind='legend'
  )

  tooltips = [
      alt.Tooltip(f'{eda_constants.LABEL}:N', title='Type'),
      alt.Tooltip(f'{constants.ROI}:Q', title=x_title, format='.2f'),
      alt.Tooltip(f'{eda_constants.DENSITY}:Q', title=y_title, format='.4f'),
  ]

  intermediary_chart = (
      alt.Chart(plot_data.intermediary_df)
      .mark_bar()
      .encode(
          x=alt.X(f'{constants.ROI}:Q', title=x_title),
          y=alt.Y(f'{eda_constants.DENSITY}:Q', title=y_title),
          color=alt.Color(
              f'{eda_constants.LABEL}:N',
              scale=unified_color_scale,
              legend=alt.Legend(title=None),
          ),
          opacity=alt.condition(
              legend_selection, alt.value(0.4), alt.value(0.1)
          ),
          tooltip=tooltips,
      )
  )

  left_layers = [intermediary_chart]
  left_df_list = [plot_data.intermediary_df]

  if plot_data.baseline_df is not None:
    left_layers.append(
        make_density_line_chart(
            plot_data.baseline_df,
            unified_color_scale,
            legend_selection,  # pyrefly: ignore[bad-argument-type]
            tooltips,
            stroke_dash=[5, 5],
        )
    )
    left_df_list.append(plot_data.baseline_df)

  if plot_data.exp_dfs:
    combined_exp_df = pd.concat(plot_data.exp_dfs)
    left_layers.append(
        make_density_line_chart(
            combined_exp_df,
            unified_color_scale,
            legend_selection,  # pyrefly: ignore[bad-argument-type]
            tooltips,
        )
    )
    left_df_list.append(combined_exp_df)

  left_hover_layers = create_interactive_hover_layers(
      pd.concat(left_df_list), unified_color_scale, tooltips
  )
  left_layers.extend(left_hover_layers)

  left_subplot = (
      alt.layer(*left_layers)
      .properties(
          title=alt.TitleParams(
              text=eda_constants.CALIBRATION_LEFT_PLOT_TITLE,
              anchor='start',
              fontSize=12,
          ),
          width=280,
          height=200,
      )
      .add_params(legend_selection)
  )

  calibrated_line_chart = make_density_line_chart(
      plot_data.calibrated_df,
      unified_color_scale,
      legend_selection,  # pyrefly: ignore[bad-argument-type]
      tooltips,
      stroke_width=2.5,
  )

  right_combined_df = pd.concat(
      [plot_data.intermediary_df, plot_data.calibrated_df]
  )
  right_hover_layers = create_interactive_hover_layers(
      right_combined_df, unified_color_scale, tooltips
  )

  right_subplot = (
      alt.layer(
          intermediary_chart,
          calibrated_line_chart,
          *right_hover_layers,
      )
      .properties(
          title=alt.TitleParams(
              text=eda_constants.CALIBRATION_RIGHT_PLOT_TITLE,
              anchor='start',
              fontSize=12,
          ),
          width=280,
          height=200,
      )
      .add_params(legend_selection)
  )

  channel_title = f'Prior Calibration: {ch_name}'
  if spend is not None:
    channel_title += f' (Total Spend: ${spend:,.0f})'

  return (
      alt.hconcat(left_subplot, right_subplot)
      .resolve_scale(y='shared')
      .properties(
          title=alt.TitleParams(
              text=channel_title,
              anchor='start',
              fontSize=14,
              fontWeight='bold',
          )
      )
  )


def plot_single_channel_calibration(
    ch_name: str,
    calibrated_output: calibration_base.CalibrationOutput,
    calibrated_prior_dist: backend.tfd.Distribution,
    indexed_experiments: Sequence[
        tuple[int, calibration_base.CalibratedExperiment]
    ],
    rng_handler: backend.RNGHandler,
    total_spend_da: xr.DataArray | None,
    show_spend_in_title: bool,
) -> alt.Chart:
  """Coordinates plotting for a single channel."""
  plot_data = prepare_calibration_data(
      calibrated_output=calibrated_output,
      calibrated_prior_dist=calibrated_prior_dist,
      indexed_experiments=indexed_experiments,
      rng_handler=rng_handler,
  )

  if (
      show_spend_in_title
      and total_spend_da is not None
      and ch_name in total_spend_da[constants.CHANNEL].values
  ):
    spend = float(total_spend_da.sel({constants.CHANNEL: ch_name}).values)
  else:
    spend = None

  return build_calibration_chart(
      ch_name=ch_name,
      plot_data=plot_data,
      spend=spend,
  )
