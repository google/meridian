# Copyright 2024 The Meridian Authors.
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

"""Module to output budget optimization scenarios on the Meridian model."""

from collections.abc import Mapping, Sequence
import math
import os
from typing import Any, TypeAlias

import altair as alt
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.model import model
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr


_SpendConstraint: TypeAlias = float | Sequence[float]


class OptimizationNotRunError(Exception):
  """Optimization has not been run on the model."""


def _get_round_factor(budget: float, gtol: float) -> int:
  """Function for obtaining number of integer digits to round off of budget.

  Args:
    budget: float total advertising budget.
    gtol: float indicating the acceptable relative error for the udget used in
      the grid setup. The budget will be rounded by 10*n, where n is the
      smallest int such that (budget - rounded_budget) is less than or equal to
      (budget * gtol). gtol must be less than 1.

  Returns:
    int number of integer digits to round budget to.
  """
  tolerance = budget * gtol
  if gtol >= 1.0:
    raise ValueError('gtol must be less than one.')
  elif budget <= 0.0:
    raise ValueError('`budget` must be greater than zero.')
  elif tolerance < 1.0:
    return 0
  else:
    return -int(math.log10(tolerance)) - 1


def _exceeds_optimization_constraints(
    fixed_budget: bool,
    budget: float,
    spend: np.ndarray,
    incremental_impact: np.ndarray,
    roi_grid_point: float,
    target_mroi: float | None = None,
    target_roi: float | None = None,
) -> bool:
  """Checks optimization scenario constraints.

    Optimality is verified within the optimization constraints, such as budget
    flexibility, target_roi, and target_mroi.

  Args:
    fixed_budget: bool indicating whether it's a fixed budget optimization or
      flexible budget optimization.
    budget: integer indicating the total budget.
    spend: np.ndarray with dimensions (`n_total_channels`) containing spend per
      channel for all media and RF channels.
    incremental_impact: np.ndarray with dimensions (`n_total_channels`)
      containing incremental impact per channel for all media and RF channels.
    roi_grid_point: float roi for current optimation step.
    target_mroi: Optional float indicating the target marginal return on
      investment (mroi) constraint. This can be translated into "How much can I
      spend when I have flexible budget until the mroi of each channel hits the
      target mroi". It's still possible that the mroi of some channels will not
      be equal to the target mroi due to the feasible range of spend. However,
      the mroi will effectively shrink toward the target mroi.
    target_roi: Optional float indicating the target return on investment (roi)
      constraint. This can be translated into "How much can I spend when I have
      a flexible budget until the roi of total spend hits the target roi.

  Returns:
    bool indicating whether optimal spend and incremental impact have been
      found, given the optimization constraints.
  """
  target_mroi = target_mroi or 1.0
  if fixed_budget:
    return np.sum(spend) > budget
  elif target_roi is not None:
    return (np.sum(incremental_impact) / np.sum(spend)) < target_roi
  else:
    return roi_grid_point < target_mroi


class BudgetOptimizer:
  """Runs and outputs budget optimization scenarios on the Meridian model.

  Finds the optimal budget allocation that maximizes impact based on various
  scenarios where the budget, data, and constraints can be customized. The
  results can be viewed as plots and/or as an HTML summary output page.
  """

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(self._meridian)
    self._template_env = formatter.create_template_env()
    self._nonoptimized_data = None
    self._optimized_data = None
    self._spend_bounds = None

  @property
  def nonoptimized_data(self) -> xr.Dataset:
    """Dataset holding the non-optimized budget metrics.

    The dataset contains the following:
      Coordinates:
        channel
      Data variables:
        spend, pct_of_spend, roi, mroi, incremental_impact
      Attributes: budget, profit, total_incremental_impact, total_roi

    Raises:
      OptimizationNotRunError: if the optimization has not been run. The
      `nonoptimized_data` is only available after running the optimization
      since certain data points are dependent on the optimization parameters.
    """
    if self._nonoptimized_data is None:
      raise OptimizationNotRunError(
          'Non-optimized data is only available after running optimize().'
      )
    return self._nonoptimized_data

  @property
  def optimized_data(self) -> xr.Dataset:
    """Dataset holding the optimized budget metrics.

    The dataset contains the following:
      Coordinates:
        channel
      Data variables:
        spend, pct_of_spend, incremental_impact, roi, mroi
      Attributes: budget, profit, total_incremental_impact, total_roi

    Raises:
      OptimizationNotRunError: if the optimization has not been run. The
      `optimized_data` is only available after running the optimization.
    """
    if self._nonoptimized_data is None:
      raise OptimizationNotRunError(
          'Optimized data is only available after running optimize().'
      )
    return self._optimized_data

  def optimize(
      self,
      selected_times: tuple[str, str] | None = None,
      fixed_budget: bool = True,
      budget: float | None = None,
      pct_of_spend: Sequence[float] | None = None,
      spend_constraint_lower: _SpendConstraint | None = None,
      spend_constraint_upper: _SpendConstraint | None = None,
      target_roi: float | None = None,
      target_mroi: float | None = None,
      gtol: float = 0.0001,
      use_optimal_frequency: bool = True,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ):
    """Finds the optimal budget allocation that maximizes impact.

    Args:
      selected_times: Tuple containing the start and end time dimensions for the
        duration to run the optimization on. Time dimensions should align with
        the Meridian time dims. By default, all times periods are used.
      fixed_budget: Boolean indicating whether it's a fixed budget optimization
        or flexible budget optimization. Defaults to True. If False, must
        specify either `target_roi` or `target_mroi`.
      budget: Number indicating the total budget for the fixed budget scenario.
        Defaults to the historical budget.
      pct_of_spend: Numeric list of size `n_total_channels` containing the
        percentage allocation for spend for all media and RF channels. The order
        must match InputData.media with values between 0-1, summing to 1. By
        default, the historical allocation is used. Budget and allocation are
        used in conjunction to determine the non-optimized media-level spend,
        which is used to calculate the non-optimized performance metrics (ie.
        ROI) and construct the feasible range of media-level spend with the
        spend constraints.
      spend_constraint_lower: Numeric list of size `n_total_channels` or float
        (same constraint for all channels) indicating the lower bound of
        media-level spend. The lower bound of media-level spend is `(1 -
        spend_constraint_lower) * budget * allocation)`. The value must be
        between 0-1. Defaults to 0.3 for fixed budget and 1 for flexible.
      spend_constraint_upper: Numeric list of size `n_total_channels` or float
        (same constraint for all channels) indicating the upper bound of
        media-level spend. The upper bound of media-level spend is `(1 +
        spend_constraint_upper) * budget * allocation)`. Defaults to 0.3 for
        fixed budget and 1 for flexible.
      target_roi: Float indicating the target ROI constraint. Only used for
        flexible budget scenarios. The budget is constrained to when the ROI of
        the total spend hits target_roi.
      target_mroi: Float indicating the target marginal ROI constraint. Only
        used for flexible budget scenarios. The budget is constrained to when
        the marginal ROI of the total spend hits target_mroi.
      gtol: Float indicating the acceptable relative error for the budget used
        in the grid setup. The budget will be rounded by 10*n, where n is the
        smallest int such that (budget - rounded_budget) is less than or equal
        to (budget * gtol). gtol must be less than 1.
      use_optimal_frequency: If True use optimal_frequency calculated by trained
        meridian model for optimization, otherwise use historical frequency.
      batch_size: Max draws per chain in each batch. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.
    """
    if not hasattr(self._meridian.inference_data, c.POSTERIOR):
      raise model.NotFittedModelError(
          'Running budget optimization scenarios requires fitting the model.'
      )
    self._validate_budget(fixed_budget, budget, target_roi)
    selected_time_dims = self._get_selected_time_dims(selected_times)
    hist_spend = self._get_hist_spend(selected_time_dims)
    budget = budget or np.sum(hist_spend)
    pct_of_spend = self._validate_pct_of_spend(hist_spend, pct_of_spend)
    spend = budget * pct_of_spend
    round_factor = _get_round_factor(budget, gtol)
    step_size = 10 ** (-round_factor)
    rounded_spend = np.round(spend, round_factor).astype(int)
    if self._meridian.n_rf_channels > 0 and use_optimal_frequency:
      optimal_frequency = tf.convert_to_tensor(
          self._analyzer.optimal_freq().optimal_frequency, dtype=tf.float32
      )
    else:
      optimal_frequency = None
    (optimization_lower_bound, optimization_upper_bound) = (
        self._get_optimization_bounds(
            spend=rounded_spend,
            spend_constraint_lower=spend_constraint_lower,
            spend_constraint_upper=spend_constraint_upper,
            round_factor=round_factor,
            fixed_budget=fixed_budget,
        )
    )
    (spend_grid, incremental_impact_grid) = self._create_grids(
        spend=hist_spend,
        spend_bound_lower=optimization_lower_bound,
        spend_bound_upper=optimization_upper_bound,
        step_size=step_size,
        selected_times=selected_time_dims,
        optimal_frequency=optimal_frequency,
        batch_size=batch_size,
    )
    optimal_spend = self._grid_search(
        spend_grid=spend_grid,
        incremental_impact_grid=incremental_impact_grid,
        budget=np.sum(rounded_spend),
        fixed_budget=fixed_budget,
        target_mroi=target_mroi,
        target_roi=target_roi,
    )

    constraints = {
        c.FIXED_BUDGET: fixed_budget,
    }
    if target_roi:
      constraints[c.TARGET_ROI] = target_roi
    elif target_mroi:
      constraints[c.TARGET_MROI] = target_mroi

    self._nonoptimized_data = self._create_budget_dataset(
        hist_spend=hist_spend,
        spend=rounded_spend,
        selected_times=selected_time_dims,
        batch_size=batch_size,
    )
    self._optimized_data = self._create_budget_dataset(
        hist_spend=spend,
        spend=optimal_spend,
        selected_times=selected_time_dims,
        optimal_frequency=optimal_frequency,
        attrs=constraints,
        batch_size=batch_size,
    )

  def output_optimization_summary(self, filename: str, filepath: str):
    """Generate and save the HTML optimization summary output."""
    if self.optimized_data:
      os.makedirs(filepath, exist_ok=True)
      with open(os.path.join(filepath, filename), 'w') as f:
        f.write(self._gen_optimization_summary())

  def plot_incremental_impact_delta(self) -> alt.Chart:
    """Plots a waterfall chart showing the change in incremental impact."""
    df = self._transform_impact_delta_data()
    base = (
        alt.Chart(df)
        .transform_window(
            sum_impact=f'sum({c.INCREMENTAL_IMPACT})',
            lead_channel=f'lead({c.CHANNEL})',
        )
        .transform_calculate(
            calc_lead=(
                'datum.lead_channel === null ? datum.channel :'
                ' datum.lead_channel'
            ),
            prev_sum=(
                f"datum.channel === '{c.OPTIMIZED}' ? 0 : datum.sum_impact -"
                ' datum.incremental_impact'
            ),
            calc_amount=(
                f"datum.channel === '{c.OPTIMIZED}' ?"
                f" {formatter.compact_number_expr('sum_impact')} :"
                f' {formatter.compact_number_expr(c.INCREMENTAL_IMPACT)}'
            ),
            text_y=(
                'datum.sum_impact < datum.prev_sum ? datum.prev_sum :'
                ' datum.sum_impact'
            ),
        )
        .encode(
            x=alt.X(
                f'{c.CHANNEL}:N',
                axis=alt.Axis(
                    ticks=False,
                    labelPadding=c.PADDING_10,
                    domainColor=c.GREY_300,
                    labelAngle=-45,
                ),
                title=None,
                sort=None,
                scale=alt.Scale(paddingOuter=c.SCALED_PADDING),
            )
        )
    )

    color_coding = {
        'condition': [
            {
                'test': (
                    f"datum.channel === '{c.CURRENT}' || datum.channel ==="
                    f" '{c.OPTIMIZED}'"
                ),
                'value': c.BLUE_500,
            },
            {'test': 'datum.incremental_impact < 0', 'value': c.RED_300},
        ],
        'value': c.CYAN_400,
    }

    # To show the details of the incremental impact delta, zoom into the plot by
    # adjusting the domain of the y-axis so that the incremental impact does not
    # start at 0. Calculate the total decrease in incremental impact to pad the
    # y-axis from the current total incremental impact value.
    sum_decr = sum(df[df.incremental_impact < 0].incremental_impact)
    y_padding = float(f'1e{int(math.log10(-sum_decr))}') if sum_decr < 0 else 2
    domain_scale = [
        self.nonoptimized_data.total_incremental_impact + sum_decr - y_padding,
        self.optimized_data.total_incremental_impact + y_padding,
    ]

    bar = base.mark_bar(
        size=c.BAR_SIZE, clip=True, cornerRadius=c.CORNER_RADIUS
    ).encode(
        y=alt.Y(
            'prev_sum:Q',
            axis=alt.Axis(
                title=summary_text.INC_SALES_LABEL,
                ticks=False,
                domain=False,
                tickCount=5,
                labelPadding=c.PADDING_10,
                labelExpr=formatter.compact_number_expr(),
                **formatter.Y_AXIS_TITLE_CONFIG,
            ),
            scale=alt.Scale(domain=domain_scale),
        ),
        y2='sum_impact:Q',
        color=color_coding,
        tooltip=[f'{c.CHANNEL}:N', f'{c.INCREMENTAL_IMPACT}:Q'],
    )

    text = base.mark_text(
        baseline='top', dy=-20, fontSize=c.AXIS_FONT_SIZE, color=c.GREY_800
    ).encode(
        text=alt.Text('calc_amount:N'),
        y='text_y:Q',
    )

    return (
        (bar + text)
        .properties(
            title=formatter.custom_title_params(
                summary_text.SALES_DELTA_CHART_TITLE
            ),
            width=(c.BAR_SIZE + c.PADDING_20) * len(df)
            + c.BAR_SIZE * 2 * c.SCALED_PADDING,
            height=400,
        )
        .configure_axis(**formatter.TEXT_CONFIG)
        .configure_view(strokeOpacity=0)
    )

  def plot_budget_allocation(self, optimized: bool = True) -> alt.Chart:
    """Plots a pie chart showing the spend split by channel.

    Args:
      optimized: If True, shows the optimized spend. Else, shows the
        non-optimized spend.

    Returns:
      An Altair pie chart showing the spend by channel.
    """
    data = self.optimized_data if optimized else self.nonoptimized_data
    df = data.spend.to_dataframe().reset_index()
    return (
        alt.Chart(df)
        .mark_arc(tooltip=True, padAngle=0.02)
        .encode(
            theta=f'{c.SPEND}:Q',
            color=alt.Color(
                f'{c.CHANNEL}:N',
                legend=alt.Legend(
                    title=None, rowPadding=c.PADDING_10, offset=-25
                ),
            ),
        )
        .configure_view(stroke=None)
        .properties(
            title=formatter.custom_title_params(
                summary_text.SPEND_ALLOCATION_CHART_TITLE
            )
        )
    )

  def plot_spend_delta(self) -> alt.Chart:
    """Plots a bar chart showing the optimized change in spend per channel."""
    df = self._get_delta_data(c.SPEND)
    base = (
        alt.Chart(df)
        .transform_calculate(
            text_value=f'{formatter.compact_number_expr(c.SPEND, 2)}',
            text_y='datum.spend < 0 ? 0 : datum.spend',
        )
        .encode(
            x=alt.X(
                f'{c.CHANNEL}:N',
                sort=None,
                axis=alt.Axis(
                    title=None, labelAngle=-45, **formatter.AXIS_CONFIG
                ),
                scale=alt.Scale(padding=c.BAR_SIZE),
            ),
            y=alt.Y(
                f'{c.SPEND}:Q',
                axis=alt.Axis(
                    title='$',
                    domain=False,
                    labelExpr=formatter.compact_number_expr(n_sig_digits=1),
                    **formatter.AXIS_CONFIG,
                    **formatter.Y_AXIS_TITLE_CONFIG,
                ),
            ),
        )
    )

    bar_plot = base.mark_bar(
        tooltip=True, size=c.BAR_SIZE, cornerRadiusEnd=c.CORNER_RADIUS
    ).encode(
        color=alt.condition(
            alt.expr.datum.spend > 0,
            alt.value(c.CYAN_400),
            alt.value(c.RED_300),
        ),
    )

    text = base.mark_text(
        baseline='top', dy=-20, fontSize=c.AXIS_FONT_SIZE, color=c.GREY_800
    ).encode(
        text=alt.Text('text_value:N'),
        y='text_y:Q',
    )

    return (
        (bar_plot + text)
        .configure_view(stroke=None)
        .properties(
            title=formatter.custom_title_params(
                summary_text.SPEND_DELTA_CHART_TITLE
            ),
            width=formatter.bar_chart_width(len(df) + 2),
            height=400,
        )
        .configure_axis(**formatter.TEXT_CONFIG)
    )

  def plot_response_curves(
      self, n_top_channels: int | None = None
  ) -> alt.Chart:
    """Plots the response curves, with spend constraints, for each channel.

    Args:
      n_top_channels: Optional number of top channels by spend to include. By
        default, all geos are included.

    Returns:
      An Altair plot showing the response curves with optimization details.
    """
    df = self._get_response_curves_data(n_top_channels=n_top_channels)
    base = (
        alt.Chart(df)
        .transform_calculate(
            spend_constraint=(
                'datum.spend_multiplier >= datum.lower_bound &&'
                ' datum.spend_multiplier <= datum.upper_bound ?'
                ' "Within spend constraint" : "Outside spend constraint"'
            ),
        )
        .encode(
            x=alt.X(
                f'{c.SPEND}:Q',
                title='Spend',
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.AXIS_CONFIG,
                ),
            ),
            y=alt.Y(
                f'{c.MEAN}:Q',
                title=summary_text.INC_SALES_LABEL,
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.AXIS_CONFIG,
                    **formatter.Y_AXIS_TITLE_CONFIG,
                ),
            ),
            color=alt.Color(f'{c.CHANNEL}:N', legend=None),
        )
    )
    curve_below_constraint = base.mark_line(
        strokeDash=list(c.STROKE_DASH)
    ).transform_filter(
        (alt.datum.spend_multiplier)
        & (alt.datum.spend_multiplier <= alt.datum.lower_bound)
    )
    curve_at_constraint_and_above = (
        base.mark_line()
        .encode(
            strokeDash=alt.StrokeDash(
                f'{c.SPEND_CONSTRAINT}:N',
                sort='descending',
                legend=alt.Legend(title=None),
            )
        )
        .transform_filter(
            (alt.datum.spend_multiplier)
            & (alt.datum.spend_multiplier >= alt.datum.lower_bound)
        )
    )
    points = (
        base.mark_point(filled=True, opacity=1, size=c.POINT_SIZE, tooltip=True)
        .encode(
            shape=alt.Shape(f'{c.SPEND_LEVEL}:N', legend=alt.Legend(title=None))
        )
        .transform_filter(alt.datum.spend_level)
    )

    sorter = list(df[c.CHANNEL].unique()) if n_top_channels else None
    return (
        alt.layer(curve_below_constraint, curve_at_constraint_and_above, points)
        .facet(
            facet=alt.Facet(f'{c.CHANNEL}:N', title=None, sort=sorter),
            columns=3,
        )
        .resolve_scale(y='independent', x='independent')
        .configure_axis(**formatter.TEXT_CONFIG)
    )

  def _get_top_channels_by_spend(self, n_channels: int) -> Sequence[str]:
    """Gets the top channels by spend."""
    data = self.optimized_data
    if n_channels > data[c.CHANNEL].size:
      raise ValueError(
          f'Top number of channels ({n_channels}) by spend must be less than'
          f' the total number of channels ({data[c.CHANNEL].size})'
      )
    return list(
        data[c.SPEND]
        .to_dataframe()
        .sort_values(by=c.SPEND, ascending=False)
        .reset_index()[c.CHANNEL][:n_channels]
    )

  def _get_response_curves_data(
      self, n_top_channels: int | None
  ) -> pd.DataFrame:
    """Calculates the response curve data, specific to the optimization."""
    if self._spend_bounds is None:
      raise OptimizationNotRunError(
          'Optimization response curves are only available after running the'
          ' optimization.'
      )
    channels = self.optimized_data.channel.values
    selected_times = self._get_selected_time_dims(
        (self.optimized_data.start_date, self.optimized_data.end_date)
    )
    lower_bound = (
        self._spend_bounds[0].repeat(len(channels))
        if len(self._spend_bounds[0]) == 1
        else self._spend_bounds[0]
    )
    upper_bound = (
        self._spend_bounds[1].repeat(len(channels))
        if len(self._spend_bounds[1]) == 1
        else self._spend_bounds[1]
    )
    spend_constraints_df = pd.DataFrame({
        c.CHANNEL: channels,
        c.LOWER_BOUND: lower_bound,
        c.UPPER_BOUND: upper_bound,
    })

    # Get the upper limit for plotting the response curves. Default to 2 or the
    # max upper spend constraint + padding.
    upper_limit = max(max(upper_bound) + c.SPEND_CONSTRAINT_PADDING, 2)
    spend_multiplier = np.arange(0, upper_limit, c.RESPONSE_CURVE_STEP_SIZE)
    response_curves_ds = self._analyzer.response_curves(
        spend_multipliers=spend_multiplier,
        selected_times=selected_times,
        by_reach=True,
    )
    response_curves_df = (
        response_curves_ds.to_dataframe()
        .reset_index()
        .pivot(
            index=[
                c.CHANNEL,
                c.SPEND,
                c.SPEND_MULTIPLIER,
            ],
            columns=c.METRIC,
            values=c.INCREMENTAL_IMPACT,
        )
        .reset_index()
    )
    response_curves_df[c.SPEND_LEVEL] = np.where(
        response_curves_df[c.SPEND_MULTIPLIER] == 1.0,
        summary_text.CURRENT_SPEND_LABEL,
        pd.NA,
    )

    optimal_points_df = (
        self.optimized_data[[c.SPEND, c.INCREMENTAL_IMPACT]]
        .to_dataframe()
        .reset_index()
        .rename(columns={c.INCREMENTAL_IMPACT: c.MEAN})
    )
    optimal_points_df[c.SPEND_LEVEL] = summary_text.OPTIMIZED_SPEND_LABEL

    concat_df = pd.concat([response_curves_df, optimal_points_df])
    merged_df = concat_df.merge(spend_constraints_df, on=c.CHANNEL)
    if n_top_channels:
      top_channels = self._get_top_channels_by_spend(n_top_channels)
      merged_df[c.CHANNEL] = merged_df[c.CHANNEL].astype('category')
      merged_df[c.CHANNEL] = merged_df[c.CHANNEL].cat.set_categories(
          top_channels
      )
      return merged_df[merged_df[c.CHANNEL].isin(top_channels)].sort_values(
          by=c.CHANNEL
      )
    else:
      return merged_df

  def _get_delta_data(self, metric: str) -> pd.DataFrame:
    """Calculates and sorts the optimized delta for the specified metric."""
    delta = self.optimized_data[metric] - self.nonoptimized_data[metric]
    df = delta.to_dataframe().reset_index()
    return pd.concat([
        df[df[metric] < 0].sort_values([metric]),
        df[df[metric] >= 0].sort_values([metric], ascending=False),
    ]).reset_index(drop=True)

  def _transform_impact_delta_data(self) -> pd.DataFrame:
    """Calculates the incremental impact delta after optimization."""
    sorted_df = self._get_delta_data(c.INCREMENTAL_IMPACT)
    sorted_df.loc[len(sorted_df)] = [c.OPTIMIZED, 0]
    sorted_df.loc[-1] = [
        c.CURRENT,
        self.nonoptimized_data.total_incremental_impact,
    ]
    sorted_df.sort_index(inplace=True)
    return sorted_df

  def _validate_budget(
      self, fixed_budget: bool, budget: float | None, target_roi: float | None
  ):
    """Validates the budget scenario requirements."""
    if fixed_budget:
      if target_roi:
        raise ValueError(
            '`target_roi` is only used for flexible budget scenarios.'
        )
      if budget and budget <= 0:
        raise ValueError('`budget` must be greater than zero.')
    else:
      if budget:
        raise ValueError('`budget` is only used for fixed budget scenarios.')

  def _get_selected_time_dims(
      self, selected_times: tuple[str, str] | None
  ) -> Sequence[str] | None:
    """Validates and returns the time dimensions based on the selected times.

    Args:
      selected_times: Tuple of the start and end times. If None, all time
        dimensions are returned.

    Returns:
      DataArray of the time dimensions.
    """
    all_times = self._meridian.input_data.time.sortby(c.TIME)
    all_times_list = all_times.values.tolist()
    all_times_range = (min(all_times_list), max(all_times_list))
    if not selected_times or selected_times == all_times_range:
      return None

    if any(selected_time not in all_times for selected_time in selected_times):
      raise ValueError(
          '`selected_times` should match the time dimensions from '
          'meridian.input_data.'
      )

    start_index = np.where(all_times == selected_times[0])[0]
    end_index = np.where(all_times == selected_times[1])[0]
    start = selected_times[0] if start_index < end_index else selected_times[1]
    end = selected_times[1] if start_index < end_index else selected_times[0]
    return all_times.sel(time=slice(start, end)).values.tolist()

  def _get_hist_spend(self, selected_times: Sequence[str]) -> np.ndarray:
    """Gets the historical spend for all channels based on the time period."""
    dim_kwargs = {
        'selected_geos': None,
        'selected_times': selected_times,
        'aggregate_geos': True,
        'aggregate_times': True,
    }
    all_media = tf.convert_to_tensor(
        self._meridian.input_data.get_all_media_and_rf(), dtype=tf.float32
    )[:, -self._meridian.n_times :, :]
    all_spend = self._meridian.total_spend

    if all_spend.ndim == 3:
      return self._analyzer.filter_and_aggregate_geos_and_times(
          all_spend,
          **dim_kwargs,
      ).numpy()
    else:
      # Calculates CPM over all time if spend does not have a time dimension.
      hist_media = self._analyzer.filter_and_aggregate_geos_and_times(
          all_media,
          **dim_kwargs,
      )
      imputed_cpmu = tf.math.divide_no_nan(
          all_spend,
          np.sum(all_media, (0, 1)),
      )
      return (hist_media * imputed_cpmu).numpy()

  def _validate_pct_of_spend(
      self, hist_spend: np.ndarray, pct_of_spend: Sequence[float] | None
  ) -> np.ndarray:
    """Validates and returns the percent of spend."""
    if pct_of_spend is not None:
      if len(pct_of_spend) != len(self._meridian.input_data.get_all_channels()):
        raise ValueError('Percent of spend must be specified for all channels.')
      if not math.isclose(np.sum(pct_of_spend), 1.0, abs_tol=0.001):
        raise ValueError('Percent of spend must sum to one.')
      return np.array(pct_of_spend)
    else:
      return hist_spend / np.sum(hist_spend)

  def _validate_spend_constraints(
      self,
      fixed_budget: bool,
      const_lower: _SpendConstraint | None,
      const_upper: _SpendConstraint | None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Validates and returns the spend constraint requirements."""

    def get_const_array(const: _SpendConstraint) -> np.ndarray:
      if not const:
        const = np.array([0.3]) if fixed_budget else np.array([1.0])
      elif isinstance(const, (float, int)):
        const = np.array([const])
      else:
        const = np.array(const)
      return const

    const_lower = get_const_array(const_lower)
    const_upper = get_const_array(const_upper)

    if any(
        len(const) not in (1, len(self._meridian.input_data.get_all_channels()))
        for const in [const_lower, const_upper]
    ):
      raise ValueError(
          'Spend constraints must be either a single constraint or be specified'
          ' for all channels.'
      )

    for const in const_lower:
      if not 0.0 <= const <= 1.0:
        raise ValueError(
            'The lower spend constraint must be between 0 and 1 inclusive.'
        )
    for const in const_upper:
      if const < 0:
        raise ValueError('The upper spend constraint must be positive.')

    return (const_lower, const_upper)

  def _get_incremental_impact_tensors(
      self,
      hist_spend: np.ndarray,
      spend: np.ndarray,
      optimal_frequency: Sequence[float] | None = None,
  ) -> tuple[
      tf.Tensor | None,
      tf.Tensor | None,
      tf.Tensor | None,
      tf.Tensor | None,
      tf.Tensor | None,
  ]:
    """Gets the tensors for incremental impact, based on spend data.

    This function is used to get the tensor data used when calling
    incremental_impact() for creating budget data. new_media is calculated
    assuming a constant cpm between historical spend and optimization spend.
    new_reach and new_frequency are calculated by first multiplying them
    together and getting rf_media(impressions), and then calculating
    new_rf_media given the same formula for new_media. new_frequency is
    optimal_frequency if optimal_frequency is not none, and
    self._meridian.frequency otherwise. new_reach is calculated using
    (new_rf_media / new_frequency). new_spend and new_rf_spend are taken from
    their respective indexes in spend.

    Args:
      hist_spend: historical spend data.
      spend: new optimized spend data.
      optimal_frequency: xr.DataArray with dimension `n_rf_channels`, containing
        the optimal frequency per channel, that maximizes posterior mean roi.
        Value is `None` if the model does not contain reach and frequency data,
        or if the model does contain reach and frequency data, but historical
        frequency is used for the optimization scenario.

    Returns:
      Tuple of tf.tensors (new_media, new_media_spend, new_reach, new_frequency,
      new_rf_spend).
    """
    if self._meridian.n_media_channels > 0:
      new_media = (
          tf.math.divide_no_nan(
              spend[: self._meridian.n_media_channels],
              hist_spend[: self._meridian.n_media_channels],
          )
          * self._meridian.media
      )
      new_media_spend = tf.convert_to_tensor(
          spend[: self._meridian.n_media_channels]
      )
    else:
      new_media = None
      new_media_spend = None
    if self._meridian.n_rf_channels > 0:
      rf_media = self._meridian.reach * self._meridian.frequency
      new_rf_media = (
          tf.math.divide_no_nan(
              spend[-self._meridian.n_rf_channels :],
              hist_spend[-self._meridian.n_rf_channels :],
          )
          * rf_media
      )
      frequency = (
          self._meridian.frequency
          if optimal_frequency is None
          else optimal_frequency
      )
      new_reach = tf.math.divide_no_nan(new_rf_media, frequency)
      new_frequency = tf.math.divide_no_nan(new_rf_media, new_reach)
      new_rf_spend = tf.convert_to_tensor(
          spend[-self._meridian.n_rf_channels :]
      )
    else:
      new_reach = None
      new_frequency = None
      new_rf_spend = None

    return (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend)

  def _create_budget_dataset(
      self,
      hist_spend: np.ndarray,
      spend: np.ndarray,
      selected_times: Sequence[str] | None = None,
      optimal_frequency: Sequence[float] | None = None,
      attrs: Mapping[str, Any] | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Creates the budget dataset."""
    spend = tf.convert_to_tensor(spend, dtype=tf.float32)
    hist_spend = tf.convert_to_tensor(hist_spend, dtype=tf.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self._get_incremental_impact_tensors(
            hist_spend, spend, optimal_frequency
        )
    )

    incremental_impact = tf.math.reduce_mean(
        self._analyzer.incremental_impact(
            new_media=new_media,
            new_reach=new_reach,
            new_frequency=new_frequency,
            selected_times=selected_times,
            batch_size=batch_size,
        ),
        axis=(0, 1),
    )
    roi = incremental_impact / spend
    marginal_roi = tf.math.reduce_mean(
        self._analyzer.marginal_roi(
            new_media=new_media,
            new_reach=new_reach,
            new_frequency=new_frequency,
            new_media_spend=new_media_spend,
            new_rf_spend=new_rf_spend,
            selected_times=selected_times,
            batch_size=batch_size,
            by_reach=True,
        ),
        axis=(0, 1),
    )
    budget = np.sum(spend)
    total_incremental_impact = np.sum(incremental_impact)
    all_times = self._meridian.input_data.time.values.tolist()
    return xr.Dataset(
        data_vars={
            c.SPEND: ([c.CHANNEL], spend),
            c.PCT_OF_SPEND: ([c.CHANNEL], spend / sum(spend)),
            c.INCREMENTAL_IMPACT: ([c.CHANNEL], incremental_impact),
            c.ROI: ([c.CHANNEL], roi),
            c.MROI: ([c.CHANNEL], marginal_roi),
        },
        coords={
            c.CHANNEL: (
                [c.CHANNEL],
                self._meridian.input_data.get_all_channels(),
            ),
        },
        attrs={
            c.BUDGET: budget,
            c.PROFIT: total_incremental_impact - budget,
            c.TOTAL_INCREMENTAL_IMPACT: total_incremental_impact,
            c.TOTAL_ROI: total_incremental_impact / budget,
            c.START_DATE: (
                min(selected_times) if selected_times else all_times[0]
            ),
            c.END_DATE: (
                max(selected_times) if selected_times else all_times[-1]
            ),
        }
        | (attrs or {}),
    )

  def _get_optimization_bounds(
      self,
      spend: np.ndarray,
      spend_constraint_lower: _SpendConstraint | None,
      spend_constraint_upper: _SpendConstraint | None,
      round_factor: int,
      fixed_budget: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Get optimization bounds from spend and spend constraints.

    Args:
      spend: np.ndarray with size `n_total_channels` containing media-level
        spend for all media and RF channels.
      spend_constraint_lower: Numeric list of size `n_total_channels` or float
        (same constraint for all media) indicating the lower bound of
        media-level spend. The lower bound of media-level spend is `(1 -
        spend_constraint_lower) * budget * allocation)`. The value must be
        between 0-1. Defaults to 0.3 for fixed budget and 1 for flexible.
      spend_constraint_upper: Numeric list of size `n_total_channels` or float
        (same constraint for all media) indicating the upper bound of
        media-level spend. The upper bound of media-level spend is `(1 +
        spend_constraint_upper) * budget * allocation)`. Defaults to 0.3 for
        fixed budget and 1 for flexible.
      round_factor: Integer number of digits to round optimization bounds.
      fixed_budget: Boolean indicating whether it's a fixed budget optimization
        or flexible budget optimization. Defaults to True. If False, must
        specify either `target_roi` or `target_mroi`.

    Returns:
      lower_bound: np.ndarray of size `n_total_channels` containing the lower
      bound
      spend
        for each media and RF channel.
      upper_bound: np.ndarray of size `n_total_channels` containing the upper
      bound
      spend
        for each media and RF channel.
    """
    (spend_const_lower, spend_const_upper) = self._validate_spend_constraints(
        fixed_budget, spend_constraint_lower, spend_constraint_upper
    )
    self._spend_bounds = (
        np.maximum((1 - spend_const_lower), 0),
        (1 + spend_const_upper),
    )

    lower_bound = np.round(
        (self._spend_bounds[0] * spend),
        round_factor,
    ).astype(int)
    upper_bound = np.round(self._spend_bounds[1] * spend, round_factor).astype(
        int
    )
    return (lower_bound.squeeze(), upper_bound.squeeze())

  def _update_incremental_impact_grid(
      self,
      i: int,
      incremental_impact_grid: np.ndarray,
      multipliers_grid: tf.Tensor,
      selected_times: Sequence[str],
      optimal_frequency: xr.DataArray | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ):
    """Updates incremental_impact_grid for each channel.

    Args:
      i: Row index used in updating incremental_impact_grid.
      incremental_impact_grid: Discrete two-dimensional grid with the number of
        rows determined by the `spend_constraints` and `step_size`, and the
        number of columns is equal to the number of total channels, containing
        incremental impact by channel.
      multipliers_grid: A grid derived from spend.
      selected_times: Sequence of strings representing the time dimensions in
        `meridian.input_data.time` to use for optimization.
      optimal_frequency: xr.DataArray with dimension `n_rf_channels`, containing
        the optimal frequency per channel, that maximizes posterior mean roi.
        Value is `None` if the model does not contain reach and frequency data,
        or if the model does contain reach and frequency data, but historical
        frequency is used for the optimization scenario.
      batch_size: Max draws per chain in each batch. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.
    """
    if self._meridian.n_media_channels > 0:
      new_media = (
          multipliers_grid[i, : self._meridian.n_media_channels]
          * self._meridian.media
      )
    else:
      new_media = None

    if self._meridian.n_rf_channels == 0:
      new_frequency = None
      new_reach = None
    elif optimal_frequency is not None:
      new_frequency = tf.ones_like(self._meridian.frequency) * optimal_frequency
      new_reach = tf.math.divide_no_nan(
          multipliers_grid[i, -self._meridian.n_rf_channels :]
          * self._meridian.reach
          * self._meridian.frequency,
          new_frequency,
      )
    else:
      new_frequency = self._meridian.frequency
      new_reach = (
          multipliers_grid[i, -self._meridian.n_rf_channels :]
          * self._meridian.reach
      )

    # incremental_impact returns a three dimensional tensor with dims
    # (n_chains x n_draws x n_total_channels). Incremental_impact_grid requires
    # incremental impact by channel.
    incremental_impact_grid[i, :] = np.mean(
        self._analyzer.incremental_impact(
            new_media=new_media,
            new_reach=new_reach,
            new_frequency=new_frequency,
            selected_times=selected_times,
            batch_size=batch_size,
        ),
        (c.CHAINS_DIMENSION, c.DRAWS_DIMENSION),
        dtype=np.float64,
    )

  def _create_grids(
      self,
      spend: np.ndarray,
      spend_bound_lower: np.ndarray,
      spend_bound_upper: np.ndarray,
      step_size: int,
      selected_times: Sequence[str],
      optimal_frequency: xr.DataArray | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Creates spend and incremental impact grids for optimization algorithm.

    Args:
      spend: np.ndarray with actual spend per media or RF channel.
      spend_bound_lower: np.ndarray of dimension (`n_total_channels`) containing
        the lower constraint spend for each channel.
      spend_bound_upper: np.ndarray of dimension (`n_total_channels`) containing
        the upper constraint spend for each channel.
      step_size: Integer indicating the step size, or interval, between values
        in the spend grid. All media channels have the same step size.
      selected_times: Sequence of strings representing the time dimensions in
        `meridian.input_data.time` to use for optimization.
      optimal_frequency: xr.DataArray with dimension `n_rf_channels`, containing
        the optimal frequency per channel, that maximizes posterior mean roi.
        Value is `None` if the model does not contain reach and frequency data,
        or if the model does contain reach and frequency data, but historical
        frequency is used for the optimization scenario.
      batch_size: Max draws per chain in each batch. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      spend_grid: Discrete two-dimensional grid with the number of rows
        determined by the `spend_constraints` and `step_size`, and the number of
        columns is equal to the number of total channels, containing spend by
        channel.
      incremental_impact_grid: Discrete two-dimensional grid with the number of
        rows determined by the `spend_constraints` and `step_size`, and the
        number of columns is equal to the number of total channels, containing
        incremental impact by channel.
    """
    n_grid_rows = int(
        (np.max(np.subtract(spend_bound_upper, spend_bound_lower)) // step_size)
        + 1
    )
    n_grid_columns = len(self._meridian.input_data.get_all_channels())
    spend_grid = np.full([n_grid_rows, n_grid_columns], np.nan)
    for i in range(n_grid_columns):
      spend_grid_m = np.arange(
          spend_bound_lower[i],
          spend_bound_upper[i] + step_size,
          step_size,
      )
      spend_grid[: len(spend_grid_m), i] = spend_grid_m
    incremental_impact_grid = np.full([n_grid_rows, n_grid_columns], np.nan)
    multipliers_grid = tf.cast(
        tf.math.divide_no_nan(spend_grid, spend), dtype=tf.float32
    )
    for i in range(n_grid_rows):
      self._update_incremental_impact_grid(
          i,
          incremental_impact_grid,
          multipliers_grid,
          selected_times,
          optimal_frequency,
          batch_size=batch_size,
      )
    # In theory, for RF channels, incremental_impact/spend should always be
    # same despite of spend, But given the level of precision,
    # incremental_impact/spend could have very tiny difference in high
    # decimals. This tiny difference will cause issue in
    # np.unravel_index(np.nanargmax(iROAS_grid), iROAS_grid.shape). Therefore
    # we use the following code to fix it, and ensure incremental_impact/spend
    # is always same for RF channels.
    if self._meridian.n_rf_channels > 0:
      rf_incremental_impact_max = np.nanmax(
          incremental_impact_grid[:, -self._meridian.n_rf_channels :], axis=0
      )
      rf_spend_max = np.nanmax(
          spend_grid[:, -self._meridian.n_rf_channels :], axis=0
      )
      rf_roi = tf.math.divide_no_nan(rf_incremental_impact_max, rf_spend_max)
      incremental_impact_grid[:, -self._meridian.n_rf_channels :] = (
          rf_roi * spend_grid[:, -self._meridian.n_rf_channels :]
      )
    return (spend_grid, incremental_impact_grid)

  def _grid_search(
      self,
      spend_grid: np.ndarray,
      incremental_impact_grid: np.ndarray,
      budget: float,
      fixed_budget: bool,
      target_mroi: float | None = None,
      target_roi: float | None = None,
  ) -> np.ndarray:
    """Hill-climbing search algorithm for budget optimization.

    Args:
      spend_grid: Discrete grid with dimensions (`grid_length` x
        `n_total_channels`) containing spend by channel for all media and RF
        channels, used in the hill-climbing search algorithm.
      incremental_impact_grid: Discrete grid with dimensions (`grid_length` x
        `n_total_channels`) containing incremental impact by channel for all
        media and RF channels, used in the hill-climbing search algorithm.
      budget: Integer indicating the total budget.
      fixed_budget: Bool indicating whether it's a fixed budget optimization or
        flexible budget optimization.
      target_mroi: Optional float indicating the target marginal return on
        investment (mroi) constraint. This can be translated into "How much can
        I spend when I have flexible budget until the mroi of each channel hits
        the target mroi". It's still possible that the mroi of some channels
        will not be equal to the target mroi due to the feasible range of media
        spend. However, the mroi will effectively shrink toward the target mroi.
      target_roi: Optional float indicating the target return on investment
        (roi) constraint. This can be translated into "How much can I spend when
        I have a flexible budget until the roi of total media spend hits the
        target roi".

    Returns:
      optimal_spend: np.ndarry of dimension (`n_total_channels`) containing the
      media
        spend that maximizes incremental impact based on spend constraints for
        all media and RF channels.
      optimal_inc_impact: np.ndarry of dimension (`n_total_channels`) containing
      the
        post optimization incremental impact per channel for all media and RF
        channels.
    """
    spend = spend_grid[0, :].copy()
    incremental_impact = incremental_impact_grid[0, :].copy()
    spend_grid = spend_grid[1:, :]
    incremental_impact_grid = incremental_impact_grid[1:, :]
    iterative_roi_grid = np.round(
        tf.math.divide_no_nan(
            incremental_impact_grid - incremental_impact, spend_grid - spend
        ),
        decimals=8,
    )
    while True:
      spend_optimal = spend.astype(int)
      # If none of the exit criteria are met roi_grid will eventually be filled
      # with all nans.
      if np.isnan(iterative_roi_grid).all():
        break
      point = np.unravel_index(
          np.nanargmax(iterative_roi_grid), iterative_roi_grid.shape
      )
      row_idx = point[0]
      media_idx = point[1]
      spend[media_idx] = spend_grid[row_idx, media_idx]
      incremental_impact[media_idx] = incremental_impact_grid[
          row_idx, media_idx
      ]
      roi_grid_point = iterative_roi_grid[row_idx, media_idx]
      if _exceeds_optimization_constraints(
          fixed_budget,
          budget,
          spend,
          incremental_impact,
          roi_grid_point,
          target_mroi,
          target_roi,
      ):
        break

      iterative_roi_grid[0 : row_idx + 1, media_idx] = np.nan
      iterative_roi_grid[row_idx + 1 :, media_idx] = np.round(
          tf.math.divide_no_nan(
              incremental_impact_grid[row_idx + 1 :, media_idx]
              - incremental_impact_grid[row_idx, media_idx],
              spend_grid[row_idx + 1 :, media_idx]
              - spend_grid[row_idx, media_idx],
          ),
          decimals=8,
      )
    return spend_optimal

  def _gen_optimization_summary(self) -> str:
    """Generates HTML optimization summary output (as sanitized content str)."""
    self._template_env.globals[c.START_DATE] = self.optimized_data.start_date
    self._template_env.globals[c.END_DATE] = self.optimized_data.end_date

    html_template = self._template_env.get_template('summary.html.jinja')
    return html_template.render(
        title=summary_text.OPTIMIZATION_TITLE,
        cards=self._create_output_sections(),
    )

  def _create_output_sections(self) -> Sequence[str]:
    """Creates the HTML snippets for cards in the summary page."""
    return [
        self._create_scenario_plan_section(),
        self._create_budget_allocation_section(),
        self._create_response_curves_section(),
    ]

  # TODO(b/319501774): Update insights when there's varied spend bounds.
  def _create_scenario_plan_section(self) -> str:
    """Creates the HTML card snippet for the scenario plan section."""
    card_spec = formatter.CardSpec(
        id=summary_text.SCENARIO_PLAN_CARD_ID,
        title=summary_text.SCENARIO_PLAN_CARD_TITLE,
    )
    current_budget = formatter.StatsSpec(
        title=summary_text.CURRENT_BUDGET_LABEL,
        stat=formatter.format_monetary_num(self.nonoptimized_data.budget),
    )
    optimized_budget = formatter.StatsSpec(
        title=summary_text.OPTIMIZED_BUDGET_LABEL,
        stat=formatter.format_monetary_num(self.optimized_data.budget),
        note=summary_text.FIXED_BUDGET_LABEL
        if self.optimized_data.fixed_budget
        else summary_text.FLEXIBLE_BUDGET_LABEL,
    )

    roi_diff = round(
        self.optimized_data.total_roi - self.nonoptimized_data.total_roi, 1
    )
    roi_prefix = '+' if roi_diff > 0 else ''
    current_roi = formatter.StatsSpec(
        title=summary_text.CURRENT_ROI_LABEL,
        stat=round(self.nonoptimized_data.total_roi, 1),
    )
    optimized_roi = formatter.StatsSpec(
        title=summary_text.OPTIMIZED_ROI_LABEL,
        stat=round(self.optimized_data.total_roi, 1),
        delta=roi_prefix + str(roi_diff),
    )

    inc_impact_diff = (
        self.optimized_data.total_incremental_impact
        - self.nonoptimized_data.total_incremental_impact
    )
    inc_impact_prefix = '+' if inc_impact_diff > 0 else ''
    current_inc_impact = formatter.StatsSpec(
        title=summary_text.CURRENT_INC_SALES_LABEL,
        stat=formatter.format_monetary_num(
            self.nonoptimized_data.total_incremental_impact,
        ),
    )
    optimized_inc_impact = formatter.StatsSpec(
        title=summary_text.OPTIMIZED_INC_SALES_LABEL,
        stat=formatter.format_monetary_num(
            self.optimized_data.total_incremental_impact,
        ),
        delta=inc_impact_prefix
        + formatter.format_monetary_num(inc_impact_diff),
    )
    assert self._spend_bounds is not None
    return formatter.create_card_html(
        self._template_env,
        card_spec,
        summary_text.SCENARIO_PLAN_INSIGHTS_FORMAT.format(
            lower_bound=self._spend_bounds[0][0],
            upper_bound=self._spend_bounds[1][0],
            start_date=self.optimized_data.start_date,
            end_date=self.optimized_data.end_date,
        ),
        stats_specs=[
            current_budget,
            optimized_budget,
            current_roi,
            optimized_roi,
            current_inc_impact,
            optimized_inc_impact,
        ],
    )

  def _create_budget_allocation_section(self) -> str:
    """Creates the HTML card snippet for the budget allocation section."""
    card_spec = formatter.CardSpec(
        id=summary_text.BUDGET_ALLOCATION_CARD_ID,
        title=summary_text.BUDGET_ALLOCATION_CARD_TITLE,
    )
    spend_delta = formatter.ChartSpec(
        id=summary_text.SPEND_DELTA_CHART_ID,
        chart_json=self.plot_spend_delta().to_json(),
    )
    spend_allocation = formatter.ChartSpec(
        id=summary_text.SPEND_ALLOCATION_CHART_ID,
        chart_json=self.plot_budget_allocation().to_json(),
    )
    impact_delta = formatter.ChartSpec(
        id=summary_text.SALES_DELTA_CHART_ID,
        chart_json=self.plot_incremental_impact_delta().to_json(),
    )
    spend_allocation_table = formatter.TableSpec(
        id=summary_text.SPEND_ALLOCATION_TABLE_ID,
        title=summary_text.SPEND_ALLOCATION_CHART_TITLE,
        column_headers=[
            summary_text.CHANNEL_LABEL,
            summary_text.CURRENT_SPEND_LABEL,
            summary_text.OPTIMIZED_SPEND_LABEL,
        ],
        row_values=self._create_budget_allocation_table().values.tolist(),
    )

    return formatter.create_card_html(
        self._template_env,
        card_spec,
        summary_text.BUDGET_ALLOCATION_INSIGHTS,
        [spend_delta, spend_allocation, impact_delta, spend_allocation_table],
    )

  def _create_budget_allocation_table(self) -> pd.DataFrame:
    """Creates a table of the current vs optimized spend split by channel."""
    current = (
        self.nonoptimized_data[c.PCT_OF_SPEND]
        .to_dataframe()
        .reset_index()
        .rename(columns={c.PCT_OF_SPEND: c.CURRENT})
    )
    optimized = (
        self.optimized_data[c.PCT_OF_SPEND]
        .to_dataframe()
        .reset_index()
        .rename(columns={c.PCT_OF_SPEND: c.OPTIMIZED})
    )
    df = (
        current.merge(optimized, on=c.CHANNEL)
        .sort_values(by=c.OPTIMIZED, ascending=False)
        .reset_index(drop=True)
    )
    df[c.CURRENT] = df[c.CURRENT].apply(lambda x: f'{round(x * 100)}%')
    df[c.OPTIMIZED] = df[c.OPTIMIZED].apply(lambda x: f'{round(x * 100)}%')
    return df

  def _create_response_curves_section(self) -> str:
    """Creates the HTML card snippet for the response curves section."""
    card_spec = formatter.CardSpec(
        id=summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_ID,
        title=summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_TITLE,
    )
    n_channels = min(len(self.optimized_data.channel), 6)
    response_curves = formatter.ChartSpec(
        id=summary_text.OPTIMIZED_RESPONSE_CURVES_CHART_ID,
        chart_json=self.plot_response_curves(
            n_top_channels=n_channels
        ).to_json(),
    )
    return formatter.create_card_html(
        self._template_env,
        card_spec,
        summary_text.OPTIMIZED_RESPONSE_CURVES_INSIGHTS,
        [response_curves],
    )
