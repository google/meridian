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

"""Module to output budget optimization scenarios based on the model."""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import math
import os
from typing import Any, TypeAlias

import altair as alt
import jinja2
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.model import model
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr


__all__ = [
    'BudgetOptimizer',
    'OptimizationResults',
]

# Disable max row limitations in Altair.
alt.data_transformers.disable_max_rows()

_SpendConstraint: TypeAlias = float | Sequence[float]


@dataclasses.dataclass(frozen=True)
class OptimizationResults:
  """The optimized budget allocation.

  This is a dataclass object containing datasets output from `BudgetOptimizer`.
  These datasets include:

  - `nonoptimized_data`: The non-optimized budget metrics (based on historical
    frequency).
  - `nonoptimized_data_with_optimal_freq`: The non-optimized budget metrics
    based on optimal frequency.
  - `optimized_data`: The optimized budget metrics.
  - `optimization_grid`: The grid information used for optimization.

  The metrics (data variables) are: ROI, mROI, incremental outcome, CPIK.

  Additionally, some intermediate values and referecences to the source fitted
  model and analyzer are also stored here. These are useful for visualizing and
  debugging.

  Attributes:
    meridian: The fitted Meridian model that was used to create this budget
      allocation.
    analyzer: The analyzer bound to the model above.
    use_posterior: Whether the posterior distribution was used to optimize the
      budget. If `False`, the prior distribution was used.
    use_optimal_frequency: Whether optimal frequency was used to optimize the
      budget.
    spend_ratio: The spend ratio used to scale the non-optimized budget metrics
      to the optimized budget metrics.
    spend_bounds: The spend bounds used to scale the non-optimized budget
      metrics to the optimized budget metrics.
    nonoptimized_data: The non-optimized budget metrics (based on historical
      frequency).
    nonoptimized_data_with_optimal_freq: The non-optimized budget metrics based
      on optimal frequency.
    optimized_data: The optimized budget metrics.
    optimization_grid: The grid information used for optimization.
  """

  meridian: model.Meridian
  # The analyzer bound to the model above.
  analyzer: analyzer.Analyzer

  # The intermediate values used to derive the optimized budget allocation.
  use_posterior: bool
  use_optimal_frequency: bool
  spend_ratio: np.ndarray  # spend / historical spend
  spend_bounds: tuple[np.ndarray, np.ndarray]

  # The optimized budget allocation datasets. See: each @property pydocs below.
  _nonoptimized_data: xr.Dataset
  _nonoptimized_data_with_optimal_freq: xr.Dataset
  _optimized_data: xr.Dataset
  _optimization_grid: xr.Dataset

  # TODO: Move this, and the plotting methods, to a summarizer.
  @functools.cached_property
  def template_env(self) -> jinja2.Environment:
    """A shared template environment bound to this optimized budget."""
    return formatter.create_template_env()

  @property
  def _kpi_or_revenue(self) -> str:
    return (
        c.REVENUE
        if self.nonoptimized_data.attrs[c.IS_REVENUE_KPI]
        else c.KPI.upper()
    )

  @property
  def nonoptimized_data(self) -> xr.Dataset:
    """Dataset holding the non-optimized budget metrics.

    For channels that have reach and frequency data, their performance metrics
    (ROI, mROI, incremental outcome, CPIK) are based on historical frequency.

    The dataset contains the following:

      - Coordinates: `channel`
      - Data variables: `spend`, `pct_of_spend`, `roi`, `mroi`, `cpik`,
        `incremental_outcome`, `effectiveness`
      - Attributes: `start_date`, `end_date`, `budget`, `profit`,
        `total_incremental_outcome`, `total_roi`, `total_cpik`,
        `is_revenue_kpi`,
        `use_historical_budget`

    ROI and mROI are only included if `revenue_per_kpi` is known. Otherwise,
    CPIK is used.
    """
    return self._nonoptimized_data

  @property
  def nonoptimized_data_with_optimal_freq(self) -> xr.Dataset:
    """Dataset holding the non-optimized budget metrics.

    For channels that have reach and frequency data, their performance metrics
    (ROI, mROI, incremental outcome, CPIK) are based on optimal frequency.

    The dataset contains the following:

      - Coordinates: `channel`
      - Data variables: `spend`, `pct_of_spend`, `roi`, `mroi`, `cpik`,
        `incremental_outcome`, `effectiveness`
      - Attributes: `start_date`, `end_date`, `budget`, `profit`,
        `total_incremental_outcome`, `total_roi`, `total_cpik`,
        `is_revenue_kpi`, `use_historical_budget`
    """
    return self._nonoptimized_data_with_optimal_freq

  @property
  def optimized_data(self) -> xr.Dataset:
    """Dataset holding the optimized budget metrics.

    For channels that have reach and frequency data, their performance metrics
    (ROI, mROI, incremental outcome) are based on optimal frequency.

    The dataset contains the following:

      - Coordinates: `channel`
      - Data variables: `spend`, `pct_of_spend`, `roi`, `mroi`, `cpik`,
        `incremental_outcome`, `effectiveness`
      - Attributes: `start_date`, `end_date`, `budget`, `profit`,
        `total_incremental_outcome`, `total_roi`, `total_cpik`, `fixed_budget`,
        `is_revenue_kpi`, `use_historical_budget`
    """
    return self._optimized_data

  @property
  def optimization_grid(self) -> xr.Dataset:
    """Dataset holding the grid information used for optimization.

    The dataset contains the following:

      - Coordinates:  `grid_spend_index`, `channel`
      - Data variables: `spend_grid`, `incremental_outcome_grid`
      - Attributes: `spend_step_size`
    """
    return self._optimization_grid

  def output_optimization_summary(self, filename: str, filepath: str):
    """Generates and saves the HTML optimization summary output."""
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as f:
      f.write(self._gen_optimization_summary())

  def plot_incremental_outcome_delta(self) -> alt.Chart:
    """Plots a waterfall chart showing the change in incremental outcome."""
    outcome = self._kpi_or_revenue
    if outcome == c.REVENUE:
      y_axis_label = summary_text.INC_REVENUE_LABEL
    else:
      y_axis_label = summary_text.INC_KPI_LABEL
    df = self._transform_outcome_delta_data()
    base = (
        alt.Chart(df)
        .transform_window(
            sum_outcome=f'sum({c.INCREMENTAL_OUTCOME})',
            lead_channel=f'lead({c.CHANNEL})',
        )
        .transform_calculate(
            calc_lead=(
                'datum.lead_channel === null ? datum.channel :'
                ' datum.lead_channel'
            ),
            prev_sum=(
                f"datum.channel === '{c.OPTIMIZED}' ? 0 : datum.sum_outcome -"
                ' datum.incremental_outcome'
            ),
            calc_amount=(
                f"datum.channel === '{c.OPTIMIZED}' ?"
                f" {formatter.compact_number_expr('sum_outcome')} :"
                f' {formatter.compact_number_expr(c.INCREMENTAL_OUTCOME)}'
            ),
            text_y=(
                'datum.sum_outcome < datum.prev_sum ? datum.prev_sum :'
                ' datum.sum_outcome'
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
                    f"datum.channel === '{c.NON_OPTIMIZED}' || datum.channel"
                    f" === '{c.OPTIMIZED}'"
                ),
                'value': c.BLUE_500,
            },
            {'test': 'datum.incremental_outcome < 0', 'value': c.RED_300},
        ],
        'value': c.CYAN_400,
    }

    # To show the details of the incremental outcome delta, zoom into the plot
    # by adjusting the domain of the y-axis so that the incremental outcome does
    # not start at 0. Calculate the total decrease in incremental outcome to pad
    # the y-axis from the non-optimized total incremental outcome value.
    sum_decr = sum(df[df.incremental_outcome < 0].incremental_outcome)
    y_padding = float(f'1e{int(math.log10(-sum_decr))}') if sum_decr < 0 else 2
    domain_scale = [
        self.nonoptimized_data.total_incremental_outcome + sum_decr - y_padding,
        self.optimized_data.total_incremental_outcome + y_padding,
    ]
    bar = base.mark_bar(
        size=c.BAR_SIZE, clip=True, cornerRadius=c.CORNER_RADIUS
    ).encode(
        y=alt.Y(
            'prev_sum:Q',
            axis=alt.Axis(
                title=y_axis_label,
                ticks=False,
                domain=False,
                tickCount=5,
                labelPadding=c.PADDING_10,
                labelExpr=formatter.compact_number_expr(),
                **formatter.Y_AXIS_TITLE_CONFIG,
            ),
            scale=alt.Scale(domain=domain_scale),
        ),
        y2='sum_outcome:Q',
        color=color_coding,
        tooltip=[f'{c.CHANNEL}:N', f'{c.INCREMENTAL_OUTCOME}:Q'],
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
                summary_text.OUTCOME_DELTA_CHART_TITLE.format(outcome=outcome)
            ),
            width=(c.BAR_SIZE + c.PADDING_20) * len(df)
            + c.BAR_SIZE * 2 * c.SCALED_PADDING,
            height=400,
        )
        .configure_axis(**formatter.TEXT_CONFIG)
        .configure_view(strokeOpacity=0)
    )

  def plot_budget_allocation(self, optimized: bool = True) -> alt.Chart:
    """Plots a pie chart showing the spend allocated for each channel.

    Args:
      optimized: If `True`, shows the optimized spend. If `False`, shows the
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
                    labelExpr=formatter.compact_number_expr(),
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
        default, all channels are included.

    Returns:
      An Altair plot showing the response curves with optimization details.
    """
    outcome = self._kpi_or_revenue
    if outcome == c.REVENUE:
      title = summary_text.INC_REVENUE_LABEL
    else:
      title = summary_text.INC_KPI_LABEL
    df = self._get_plottable_response_curves_df(n_top_channels=n_top_channels)
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
                title=title,
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

  def get_response_curves(self) -> xr.Dataset:
    """Calculates response curves, per budget optimization scenario.

    This method is a wrapper for `Analyzer.response_curves()`, that sets the
    following arguments to be consistent with the budget optimization scenario
    specified in `BudgetOptimizer.optimize()` call that returned this result.
    In particular:

    1. `spend_multiplier` matches the discrete optimization grid, considering
       the grid step size and any channel-level constraint bounds.
    2. `selected_times`, `by_reach`, and `use_optimal_frequency` match the
       values set in `BudgetOptimizer.optimize()`.

    Returns:
      A dataset returned by `Analyzer.response_curves()`, per budget
      optimization scenario specified in `BudgetOptimizer.optimize()` call that
      returned this result.
    """
    channels = self.optimized_data.channel.values
    selected_times = self.meridian.expand_selected_time_dims(
        start_date=self.optimized_data.start_date,
        end_date=self.optimized_data.end_date,
    )
    _, ubounds = self.spend_bounds
    upper_bound = (
        ubounds.repeat(len(channels)) * self.spend_ratio
        if len(ubounds) == 1
        else ubounds * self.spend_ratio
    )

    # Get the upper limit for plotting the response curves. Default to 2 or the
    # max upper spend constraint + padding.
    upper_limit = max(max(upper_bound) + c.SPEND_CONSTRAINT_PADDING, 2)
    spend_multiplier = np.arange(0, upper_limit, c.RESPONSE_CURVE_STEP_SIZE)
    # WARN: If `selected_times` is not None (i.e. a subset time range), this
    # response curve computation might take a significant amount of time.
    return self.analyzer.response_curves(
        spend_multipliers=spend_multiplier,
        use_posterior=self.use_posterior,
        selected_times=selected_times,
        by_reach=True,
        use_optimal_frequency=self.use_optimal_frequency,
    )

  def _get_plottable_response_curves_df(
      self, n_top_channels: int | None = None
  ) -> pd.DataFrame:
    """Calculates the response curve data frame, for plotting.

    Args:
      n_top_channels: Optional number of top channels by spend to include. If
        None, include all channels.

    Returns:
      A dataframe containing the response curve data suitable for plotting.
    """
    channels = self.optimized_data.channel.values
    lbounds, ubounds = self.spend_bounds
    lower_bound = (
        lbounds.repeat(len(channels)) * self.spend_ratio
        if len(lbounds) == 1
        else lbounds * self.spend_ratio
    )
    upper_bound = (
        ubounds.repeat(len(channels)) * self.spend_ratio
        if len(ubounds) == 1
        else ubounds * self.spend_ratio
    )
    spend_constraints_df = pd.DataFrame({
        c.CHANNEL: channels,
        c.LOWER_BOUND: lower_bound,
        c.UPPER_BOUND: upper_bound,
    })

    response_curves_ds = self.get_response_curves()
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
            values=c.INCREMENTAL_OUTCOME,
        )
        .reset_index()
    )
    non_optimized_points_df = (
        self.nonoptimized_data_with_optimal_freq[
            [c.SPEND, c.INCREMENTAL_OUTCOME]
        ]
        .sel(metric=c.MEAN, drop=True)
        .to_dataframe()
        .reset_index()
        .rename(columns={c.INCREMENTAL_OUTCOME: c.MEAN})
    )
    non_optimized_points_df[c.SPEND_LEVEL] = (
        summary_text.NONOPTIMIZED_SPEND_LABEL
    )
    optimal_points_df = (
        self.optimized_data[[c.SPEND, c.INCREMENTAL_OUTCOME]]
        .sel(metric=c.MEAN, drop=True)
        .to_dataframe()
        .reset_index()
        .rename(columns={c.INCREMENTAL_OUTCOME: c.MEAN})
    )
    optimal_points_df[c.SPEND_LEVEL] = summary_text.OPTIMIZED_SPEND_LABEL

    concat_df = pd.concat(
        [response_curves_df, optimal_points_df, non_optimized_points_df]
    )
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
    if c.METRIC in delta.dims:
      delta = delta.sel(metric=c.MEAN, drop=True)
    df = delta.to_dataframe().reset_index()
    return pd.concat([
        df[df[metric] < 0].sort_values([metric]),
        df[df[metric] >= 0].sort_values([metric], ascending=False),
    ]).reset_index(drop=True)

  def _transform_outcome_delta_data(self) -> pd.DataFrame:
    """Calculates the incremental outcome delta after optimization."""
    sorted_df = self._get_delta_data(c.INCREMENTAL_OUTCOME)
    sorted_df.loc[len(sorted_df)] = [c.OPTIMIZED, 0]
    sorted_df.loc[-1] = [
        c.NON_OPTIMIZED,
        self.nonoptimized_data.total_incremental_outcome,
    ]
    sorted_df.sort_index(inplace=True)
    return sorted_df

  def _gen_optimization_summary(self) -> str:
    """Generates HTML optimization summary output (as sanitized content str)."""
    self.template_env.globals[c.START_DATE] = self.optimized_data.start_date
    self.template_env.globals[c.END_DATE] = self.optimized_data.end_date

    html_template = self.template_env.get_template('summary.html.jinja')
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

  def _create_scenario_plan_section(self) -> str:
    """Creates the HTML card snippet for the scenario plan section."""
    card_spec = formatter.CardSpec(
        id=summary_text.SCENARIO_PLAN_CARD_ID,
        title=summary_text.SCENARIO_PLAN_CARD_TITLE,
    )

    scenario_type = (
        summary_text.FIXED_BUDGET_LABEL.lower()
        if self.optimized_data.fixed_budget
        else summary_text.FLEXIBLE_BUDGET_LABEL
    )
    lbounds, ubounds = self.spend_bounds
    if len(lbounds) > 1 or len(ubounds) > 1:
      insights = summary_text.SCENARIO_PLAN_INSIGHTS_VARIED_SPEND_BOUNDS.format(
          scenario_type=scenario_type,
      )
    else:
      lower_bound = int((1 - lbounds[0]) * 100)
      upper_bound = int((ubounds[0] - 1) * 100)
      insights = (
          summary_text.SCENARIO_PLAN_INSIGHTS_UNIFORM_SPEND_BOUNDS.format(
              scenario_type=scenario_type,
              lower_bound=lower_bound,
              upper_bound=upper_bound,
          )
      )
    if self.nonoptimized_data.use_historical_budget:
      insights += (
          ' '
          + summary_text.SCENARIO_PLAN_INSIGHTS_HISTORICAL_BUDGET.format(
              start_date=self.optimized_data.start_date,
              end_date=self.optimized_data.end_date,
          )
      )
    else:
      insights += ' ' + summary_text.SCENARIO_PLAN_INSIGHTS_NEW_BUDGET.format(
          start_date=self.optimized_data.start_date,
          end_date=self.optimized_data.end_date,
      )
    return formatter.create_card_html(
        self.template_env,
        card_spec,
        insights,
        stats_specs=self._create_scenario_stats_specs(),
    )

  def _create_scenario_stats_specs(self) -> Sequence[formatter.StatsSpec]:
    """Creates the stats to fill the scenario plan section."""
    outcome = self._kpi_or_revenue
    budget_diff = self.optimized_data.budget - self.nonoptimized_data.budget
    budget_prefix = '+' if budget_diff > 0 else ''
    non_optimized_budget = formatter.StatsSpec(
        title=summary_text.NON_OPTIMIZED_BUDGET_LABEL,
        stat=formatter.format_monetary_num(self.nonoptimized_data.budget),
    )
    optimized_budget = formatter.StatsSpec(
        title=summary_text.OPTIMIZED_BUDGET_LABEL,
        stat=formatter.format_monetary_num(self.optimized_data.budget),
        delta=(budget_prefix + formatter.format_monetary_num(budget_diff)),
    )

    if outcome == c.REVENUE:
      diff = round(
          self.optimized_data.total_roi - self.nonoptimized_data.total_roi, 1
      )
      non_optimized_performance_title = summary_text.NON_OPTIMIZED_ROI_LABEL
      non_optimized_performance_stat = round(
          self.nonoptimized_data.total_roi, 1
      )
      optimized_performance_title = summary_text.OPTIMIZED_ROI_LABEL
      optimized_performance_stat = round(self.optimized_data.total_roi, 1)
      optimized_performance_diff = f'+{str(diff)}' if diff > 0 else str(diff)
    else:
      diff = self.optimized_data.total_cpik - self.nonoptimized_data.total_cpik
      non_optimized_performance_title = summary_text.NON_OPTIMIZED_CPIK_LABEL
      non_optimized_performance_stat = (
          f'${self.nonoptimized_data.total_cpik:.2f}'
      )
      optimized_performance_title = summary_text.OPTIMIZED_CPIK_LABEL
      optimized_performance_stat = f'${self.optimized_data.total_cpik:.2f}'
      optimized_performance_diff = formatter.compact_number(diff, 2, '$')
    non_optimized_performance = formatter.StatsSpec(
        title=non_optimized_performance_title,
        stat=non_optimized_performance_stat,
    )
    optimized_performance = formatter.StatsSpec(
        title=optimized_performance_title,
        stat=optimized_performance_stat,
        delta=optimized_performance_diff,
    )

    inc_outcome_diff = (
        self.optimized_data.total_incremental_outcome
        - self.nonoptimized_data.total_incremental_outcome
    )
    inc_outcome_prefix = '+' if inc_outcome_diff > 0 else ''
    non_optimized_inc_outcome = formatter.StatsSpec(
        title=summary_text.NON_OPTIMIZED_INC_OUTCOME_LABEL.format(
            outcome=outcome
        ),
        stat=formatter.format_monetary_num(
            self.nonoptimized_data.total_incremental_outcome,
        ),
    )
    optimized_inc_outcome = formatter.StatsSpec(
        title=summary_text.OPTIMIZED_INC_OUTCOME_LABEL.format(outcome=outcome),
        stat=formatter.format_monetary_num(
            self.optimized_data.total_incremental_outcome,
        ),
        delta=inc_outcome_prefix
        + formatter.format_monetary_num(inc_outcome_diff),
    )
    return [
        non_optimized_budget,
        optimized_budget,
        non_optimized_performance,
        optimized_performance,
        non_optimized_inc_outcome,
        optimized_inc_outcome,
    ]

  def _create_budget_allocation_section(self) -> str:
    """Creates the HTML card snippet for the budget allocation section."""
    outcome = self._kpi_or_revenue
    card_spec = formatter.CardSpec(
        id=summary_text.BUDGET_ALLOCATION_CARD_ID,
        title=summary_text.BUDGET_ALLOCATION_CARD_TITLE,
    )
    spend_delta = formatter.ChartSpec(
        id=summary_text.SPEND_DELTA_CHART_ID,
        description=summary_text.SPEND_DELTA_CHART_INSIGHTS,
        chart_json=self.plot_spend_delta().to_json(),
    )
    spend_allocation = formatter.ChartSpec(
        id=summary_text.SPEND_ALLOCATION_CHART_ID,
        chart_json=self.plot_budget_allocation().to_json(),
    )
    outcome_delta = formatter.ChartSpec(
        id=summary_text.OUTCOME_DELTA_CHART_ID,
        description=summary_text.OUTCOME_DELTA_CHART_INSIGHTS_FORMAT.format(
            outcome=outcome
        ),
        chart_json=self.plot_incremental_outcome_delta().to_json(),
    )
    spend_allocation_table = formatter.TableSpec(
        id=summary_text.SPEND_ALLOCATION_TABLE_ID,
        title=summary_text.SPEND_ALLOCATION_CHART_TITLE,
        column_headers=[
            summary_text.CHANNEL_LABEL,
            summary_text.NONOPTIMIZED_SPEND_LABEL,
            summary_text.OPTIMIZED_SPEND_LABEL,
        ],
        row_values=self._create_budget_allocation_table().values.tolist(),
    )

    return formatter.create_card_html(
        self.template_env,
        card_spec,
        summary_text.BUDGET_ALLOCATION_INSIGHTS,
        [spend_delta, spend_allocation, outcome_delta, spend_allocation_table],
    )

  def _create_budget_allocation_table(self) -> pd.DataFrame:
    """Creates a table of the non-optimized vs optimized spend allocation."""
    non_optimized = (
        self.nonoptimized_data[c.PCT_OF_SPEND]
        .to_dataframe()
        .reset_index()
        .rename(columns={c.PCT_OF_SPEND: c.NON_OPTIMIZED})
    )
    optimized = (
        self.optimized_data[c.PCT_OF_SPEND]
        .to_dataframe()
        .reset_index()
        .rename(columns={c.PCT_OF_SPEND: c.OPTIMIZED})
    )
    df = (
        non_optimized.merge(optimized, on=c.CHANNEL)
        .sort_values(by=c.OPTIMIZED, ascending=False)
        .reset_index(drop=True)
    )
    df[c.NON_OPTIMIZED] = df[c.NON_OPTIMIZED].apply(
        lambda x: f'{round(x * 100)}%'
    )
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
        self.template_env,
        card_spec,
        summary_text.OPTIMIZED_RESPONSE_CURVES_INSIGHTS_FORMAT.format(
            outcome=self._kpi_or_revenue,
        ),
        [response_curves],
    )


class BudgetOptimizer:
  """Runs and outputs budget optimization scenarios on your model.

  Finds the optimal budget allocation that maximizes outcome based on various
  scenarios where the budget, data, and constraints can be customized. The
  results can be viewed as plots and as an HTML summary output page.
  """

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(self._meridian)

  def optimize(
      self,
      use_posterior: bool = True,
      selected_times: tuple[str | None, str | None] | None = None,
      fixed_budget: bool = True,
      budget: float | None = None,
      pct_of_spend: Sequence[float] | None = None,
      spend_constraint_lower: _SpendConstraint | None = None,
      spend_constraint_upper: _SpendConstraint | None = None,
      target_roi: float | None = None,
      target_mroi: float | None = None,
      gtol: float = 0.0001,
      use_optimal_frequency: bool = True,
      use_kpi: bool = False,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> OptimizationResults:
    """Finds the optimal budget allocation that maximizes outcome.

    Outcome is typically revenue, but when the KPI is not revenue and "revenue
    per KPI" data is not available, then Meridian defines the Outcome to be the
    KPI itself.

    Args:
      use_posterior: Boolean. If `True`, then the budget is optimized based on
        the posterior distribution of the model. Otherwise, the prior
        distribution is used.
      selected_times: Tuple containing the start and end time dimension
        coordinates for the duration to run the optimization on. Selected time
        values should align with the Meridian time dimension coordinates in the
        underlying model. By default, all times periods are used. Either start
        or end time component can be `None` to represent the first or the last
        time coordinate, respectively.
      fixed_budget: Boolean indicating whether it's a fixed budget optimization
        or flexible budget optimization. Defaults to `True`. If `False`, must
        specify either `target_roi` or `target_mroi`.
      budget: Number indicating the total budget for the fixed budget scenario.
        Defaults to the historical budget.
      pct_of_spend: Numeric list of size `n_paid_channels` containing the
        percentage allocation for spend for all media and RF channels. The order
        must match `(InputData.media + InputData.reach)` with values between
        0-1, summing to 1. By default, the historical allocation is used. Budget
        and allocation are used in conjunction to determine the non-optimized
        media-level spend, which is used to calculate the non-optimized
        performance metrics (for example, ROI) and construct the feasible range
        of media-level spend with the spend constraints. Consider using
        `InputData.get_paid_channels_argument_builder()` to construct this
        argument.
      spend_constraint_lower: Numeric list of size `n_paid_channels` or float
        (same constraint for all channels) indicating the lower bound of
        media-level spend. If given as a channel-indexed array, the order must
        match `(InputData.media + InputData.reach)`. The lower bound of
        media-level spend is `(1 - spend_constraint_lower) * budget *
        allocation)`. The value must be between 0-1. Defaults to `0.3` for fixed
        budget and `1` for flexible. Consider using
        `InputData.get_paid_channels_argument_builder()` to construct this
        argument.
      spend_constraint_upper: Numeric list of size `n_paid_channels` or float
        (same constraint for all channels) indicating the upper bound of
        media-level spend. If given as a channel-indexed array, the order must
        match `(InputData.media + InputData.reach)`. The upper bound of
        media-level spend is `(1 + spend_constraint_upper) * budget *
        allocation)`. Defaults to `0.3` for fixed budget and `1` for flexible.
        Consider using `InputData.get_paid_channels_argument_builder()` to
        construct this argument.
      target_roi: Float indicating the target ROI constraint. Only used for
        flexible budget scenarios. The budget is constrained to when the ROI of
        the total spend hits `target_roi`.
      target_mroi: Float indicating the target marginal ROI constraint. Only
        used for flexible budget scenarios. The budget is constrained to when
        the marginal ROI of the total spend hits `target_mroi`.
      gtol: Float indicating the acceptable relative error for the budget used
        in the grid setup. The budget will be rounded by `10*n`, where `n` is
        the smallest integer such that `(budget - rounded_budget)` is less than
        or equal to `(budget * gtol)`. `gtol` must be less than 1.
      use_optimal_frequency: If `True`, uses `optimal_frequency` calculated by
        trained Meridian model for optimization. If `False`, uses historical
        frequency.
      use_kpi: If `True`, runs the optimization on KPI. Defaults to revenue.
      confidence_level: The threshold for computing the confidence intervals.
      batch_size: Maximum draws per chain in each batch. The calculation is run
        in batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      An `OptimizationResults` object containing optimized budget allocation
      datasets, along with some of the intermediate values used to derive them.
    """
    dist_type = c.POSTERIOR if use_posterior else c.PRIOR
    if dist_type not in self._meridian.inference_data.groups():
      raise model.NotFittedModelError(
          'Running budget optimization scenarios requires fitting the model.'
      )
    self._validate_budget(fixed_budget, budget, target_roi, target_mroi)
    if selected_times is not None:
      start_date, end_date = selected_times
      selected_time_dims = self._meridian.expand_selected_time_dims(
          start_date=start_date,
          end_date=end_date,
      )
    else:
      selected_time_dims = None

    hist_spend = self._analyzer.get_historical_spend(
        selected_time_dims,
        include_media=self._meridian.n_media_channels > 0,
        include_rf=self._meridian.n_rf_channels > 0,
    ).data

    use_historical_budget = budget is None or round(budget) == round(
        np.sum(hist_spend)
    )
    budget = budget or np.sum(hist_spend)
    pct_of_spend = self._validate_pct_of_spend(hist_spend, pct_of_spend)
    spend = budget * pct_of_spend
    round_factor = _get_round_factor(budget, gtol)
    step_size = 10 ** (-round_factor)
    rounded_spend = np.round(spend, round_factor).astype(int)
    spend_ratio = np.divide(
        spend,
        hist_spend,
        out=np.zeros_like(hist_spend, dtype=float),
        where=hist_spend != 0,
    )
    if self._meridian.n_rf_channels > 0 and use_optimal_frequency:
      optimal_frequency = tf.convert_to_tensor(
          self._analyzer.optimal_freq(
              use_posterior=use_posterior,
              selected_times=selected_time_dims,
              use_kpi=use_kpi,
          ).optimal_frequency,
          dtype=tf.float32,
      )
    else:
      optimal_frequency = None

    (optimization_lower_bound, optimization_upper_bound, spend_bounds) = (
        self._get_optimization_bounds(
            spend=rounded_spend,
            spend_constraint_lower=spend_constraint_lower,
            spend_constraint_upper=spend_constraint_upper,
            round_factor=round_factor,
            fixed_budget=fixed_budget,
        )
    )
    (spend_grid, incremental_outcome_grid) = self._create_grids(
        spend=hist_spend,
        spend_bound_lower=optimization_lower_bound,
        spend_bound_upper=optimization_upper_bound,
        step_size=step_size,
        selected_times=selected_time_dims,
        use_posterior=use_posterior,
        use_kpi=use_kpi,
        optimal_frequency=optimal_frequency,
        batch_size=batch_size,
    )
    optimal_spend = self._grid_search(
        spend_grid=spend_grid,
        incremental_outcome_grid=incremental_outcome_grid,
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

    nonoptimized_data = self._create_budget_dataset(
        use_posterior=use_posterior,
        use_kpi=use_kpi,
        hist_spend=hist_spend,
        spend=rounded_spend,
        selected_times=selected_time_dims,
        confidence_level=confidence_level,
        batch_size=batch_size,
        use_historical_budget=use_historical_budget,
    )
    nonoptimized_data_with_optimal_freq = self._create_budget_dataset(
        use_posterior=use_posterior,
        use_kpi=use_kpi,
        hist_spend=hist_spend,
        spend=rounded_spend,
        selected_times=selected_time_dims,
        optimal_frequency=optimal_frequency,
        confidence_level=confidence_level,
        batch_size=batch_size,
        use_historical_budget=use_historical_budget,
    )
    optimized_data = self._create_budget_dataset(
        use_posterior=use_posterior,
        use_kpi=use_kpi,
        hist_spend=hist_spend,
        spend=optimal_spend,
        selected_times=selected_time_dims,
        optimal_frequency=optimal_frequency,
        attrs=constraints,
        confidence_level=confidence_level,
        batch_size=batch_size,
        use_historical_budget=use_historical_budget,
    )

    optimization_grid = self._create_optimization_grid(
        spend_grid=spend_grid,
        spend_step_size=step_size,
        incremental_outcome_grid=incremental_outcome_grid,
    )

    return OptimizationResults(
        meridian=self._meridian,
        analyzer=self._analyzer,
        use_posterior=use_posterior,
        use_optimal_frequency=use_optimal_frequency,
        spend_ratio=spend_ratio,
        spend_bounds=spend_bounds,
        _nonoptimized_data=nonoptimized_data,
        _nonoptimized_data_with_optimal_freq=nonoptimized_data_with_optimal_freq,
        _optimized_data=optimized_data,
        _optimization_grid=optimization_grid,
    )

  def _create_optimization_grid(
      self,
      spend_grid: np.ndarray,
      spend_step_size: float,
      incremental_outcome_grid: np.ndarray,
  ) -> xr.Dataset:
    """Creates the optimization grid dataset.

    Args:
      spend_grid: Discrete two-dimensional grid with the number of rows equal to
        the maximum number of spend points among all channels, and the number of
        columns is equal to the number of total channels, containing spend by
        channel.
      spend_step_size: The step size of the spend grid.
      incremental_outcome_grid: Discrete two-dimensional grid with the size same
        as the `spend_grid` containing incremental outcome by channel.

    Returns:
      The optimization grid dataset. The dataset contains the following:
        - Coordinates:  `grid_spend_index`, `channel`
        - Data variables: `spend_grid`, `incremental_outcome_grid`
        - Attributes: `spend_step_size`
    """
    data_vars = {
        c.SPEND_GRID: ([c.GRID_SPEND_INDEX, c.CHANNEL], spend_grid),
        c.INCREMENTAL_OUTCOME_GRID: (
            [c.GRID_SPEND_INDEX, c.CHANNEL],
            incremental_outcome_grid,
        ),
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            c.GRID_SPEND_INDEX: (
                [c.GRID_SPEND_INDEX],
                np.arange(0, len(spend_grid)),
            ),
            c.CHANNEL: (
                [c.CHANNEL],
                self._meridian.input_data.get_all_paid_channels(),
            ),
        },
        attrs={c.SPEND_STEP_SIZE: spend_step_size},
    )

  def _validate_budget(
      self,
      fixed_budget: bool,
      budget: float | None,
      target_roi: float | None,
      target_mroi: float | None,
  ):
    """Validates the budget optimization arguments."""
    if fixed_budget:
      if target_roi is not None:
        raise ValueError(
            '`target_roi` is only used for flexible budget scenarios.'
        )
      if target_mroi is not None:
        raise ValueError(
            '`target_mroi` is only used for flexible budget scenarios.'
        )
      if budget is not None and budget <= 0:
        raise ValueError('`budget` must be greater than zero.')
    else:
      if budget is not None:
        raise ValueError('`budget` is only used for fixed budget scenarios.')
      if target_roi is None and target_mroi is None:
        raise ValueError(
            'Must specify either `target_roi` or `target_mroi` for flexible'
            ' budget optimization.'
        )
      if target_roi is not None and target_mroi is not None:
        raise ValueError(
            'Must specify only one of `target_roi` or `target_mroi` for'
            'flexible budget optimization.'
        )

  def _validate_pct_of_spend(
      self, hist_spend: np.ndarray, pct_of_spend: Sequence[float] | None
  ) -> np.ndarray:
    """Validates and returns the percent of spend."""
    if pct_of_spend is not None:
      if len(pct_of_spend) != len(
          self._meridian.input_data.get_all_paid_channels()
      ):
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

    def get_const_array(const: _SpendConstraint | None) -> np.ndarray:
      if const is None:
        const = (
            np.array([c.SPEND_CONSTRAINT_DEFAULT_FIXED_BUDGET])
            if fixed_budget
            else np.array([c.SPEND_CONSTRAINT_DEFAULT_FLEXIBLE_BUDGET])
        )
      elif isinstance(const, (float, int)):
        const = np.array([const])
      else:
        const = np.array(const)
      return const

    const_lower = get_const_array(const_lower)
    const_upper = get_const_array(const_upper)

    if any(
        len(const)
        not in (1, len(self._meridian.input_data.get_all_paid_channels()))
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

  def _get_incremental_outcome_tensors(
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
    """Gets the tensors for incremental outcome, based on spend data.

    This function is used to get the tensor data used when calling
    incremental_outcome() for creating budget data. new_media is calculated
    assuming a constant cpm between historical spend and optimization spend.
    new_reach and new_frequency are calculated by first multiplying them
    together and getting rf_media(impressions), and then calculating
    new_rf_media given the same formula for new_media. new_frequency is
    optimal_frequency if optimal_frequency is not none, and
    self._meridian.rf_tensors.frequency otherwise. new_reach is calculated using
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
          * self._meridian.media_tensors.media
      )
      new_media_spend = tf.convert_to_tensor(
          spend[: self._meridian.n_media_channels]
      )
    else:
      new_media = None
      new_media_spend = None
    if self._meridian.n_rf_channels > 0:
      rf_media = (
          self._meridian.rf_tensors.reach * self._meridian.rf_tensors.frequency
      )
      new_rf_media = (
          tf.math.divide_no_nan(
              spend[-self._meridian.n_rf_channels :],
              hist_spend[-self._meridian.n_rf_channels :],
          )
          * rf_media
      )
      frequency = (
          self._meridian.rf_tensors.frequency
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
      use_posterior: bool = True,
      use_kpi: bool = False,
      selected_times: Sequence[str] | None = None,
      optimal_frequency: Sequence[float] | None = None,
      attrs: Mapping[str, Any] | None = None,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
      use_historical_budget: bool = True,
  ) -> xr.Dataset:
    """Creates the budget dataset."""
    spend = tf.convert_to_tensor(spend, dtype=tf.float32)
    hist_spend = tf.convert_to_tensor(hist_spend, dtype=tf.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self._get_incremental_outcome_tensors(
            hist_spend, spend, optimal_frequency
        )
    )
    budget = np.sum(spend)
    all_times = self._meridian.input_data.time.values.tolist()

    # incremental_outcome here is a tensor with the shape
    # (n_chains, n_draws, n_channels)
    incremental_outcome = self._analyzer.incremental_outcome(
        use_posterior=use_posterior,
        new_data=analyzer.DataTensors(
            media=new_media,
            reach=new_reach,
            frequency=new_frequency,
        ),
        selected_times=selected_times,
        use_kpi=use_kpi,
        batch_size=batch_size,
        include_non_paid_channels=False,
    )
    # incremental_outcome_with_mean_median_and_ci here is an ndarray with the
    # shape (n_channels, n_metrics) where n_metrics = 4 for (mean, median,
    # ci_lo, and ci_hi)
    incremental_outcome_with_mean_median_and_ci = (
        analyzer.get_central_tendency_and_ci(
            data=incremental_outcome,
            confidence_level=confidence_level,
            include_median=True,
        )
    )
    # Total of `mean` column.
    total_incremental_outcome = np.sum(
        incremental_outcome_with_mean_median_and_ci[:, 0]
    )

    # expected_outcome here is a tensor with the shape (n_chains, n_draws)
    expected_outcome = self._analyzer.expected_outcome(
        use_posterior=use_posterior,
        new_data=analyzer.DataTensors(
            media=new_media,
            reach=new_reach,
            frequency=new_frequency,
        ),
        selected_times=selected_times,
        use_kpi=use_kpi,
        batch_size=batch_size,
    )
    mean_expected_outcome = tf.reduce_mean(expected_outcome, (0, 1))  # a scalar

    pct_contrib = incremental_outcome / mean_expected_outcome[..., None] * 100
    pct_contrib_with_mean_median_and_ci = analyzer.get_central_tendency_and_ci(
        data=pct_contrib,
        confidence_level=confidence_level,
        include_median=True,
    )

    aggregated_impressions = self._analyzer.get_aggregated_impressions(
        selected_times=selected_times,
        selected_geos=None,
        aggregate_times=True,
        aggregate_geos=True,
        optimal_frequency=optimal_frequency,
        include_non_paid_channels=False,
    )
    effectiveness = incremental_outcome / aggregated_impressions
    effectiveness_with_mean_median_and_ci = (
        analyzer.get_central_tendency_and_ci(
            data=effectiveness,
            confidence_level=confidence_level,
            include_median=True,
        )
    )

    roi = analyzer.get_central_tendency_and_ci(
        data=tf.math.divide_no_nan(incremental_outcome, spend),
        confidence_level=confidence_level,
        include_median=True,
    )
    marginal_roi = analyzer.get_central_tendency_and_ci(
        data=self._analyzer.marginal_roi(
            use_posterior=use_posterior,
            new_data=analyzer.DataTensors(
                media=new_media,
                reach=new_reach,
                frequency=new_frequency,
                media_spend=new_media_spend,
                rf_spend=new_rf_spend,
            ),
            selected_times=selected_times,
            batch_size=batch_size,
            by_reach=True,
            use_kpi=use_kpi,
        ),
        confidence_level=confidence_level,
        include_median=True,
    )

    cpik = analyzer.get_central_tendency_and_ci(
        data=tf.math.divide_no_nan(spend, incremental_outcome),
        confidence_level=confidence_level,
        include_median=True,
    )
    total_inc_outcome = np.sum(incremental_outcome, -1)
    total_cpik = np.mean(
        tf.math.divide_no_nan(budget, total_inc_outcome),
        axis=(0, 1),
    )

    total_spend = np.sum(spend) if np.sum(spend) > 0 else 1
    data_vars = {
        c.SPEND: ([c.CHANNEL], spend),
        c.PCT_OF_SPEND: ([c.CHANNEL], spend / total_spend),
        c.INCREMENTAL_OUTCOME: (
            [c.CHANNEL, c.METRIC],
            incremental_outcome_with_mean_median_and_ci,
        ),
        c.PCT_OF_CONTRIBUTION: (
            [c.CHANNEL, c.METRIC],
            pct_contrib_with_mean_median_and_ci,
        ),
        c.EFFECTIVENESS: (
            [c.CHANNEL, c.METRIC],
            effectiveness_with_mean_median_and_ci,
        ),
        c.ROI: ([c.CHANNEL, c.METRIC], roi),
        c.MROI: ([c.CHANNEL, c.METRIC], marginal_roi),
        c.CPIK: ([c.CHANNEL, c.METRIC], cpik),
    }

    attributes = {
        c.START_DATE: min(selected_times) if selected_times else all_times[0],
        c.END_DATE: max(selected_times) if selected_times else all_times[-1],
        c.BUDGET: budget,
        c.PROFIT: total_incremental_outcome - budget,
        c.TOTAL_INCREMENTAL_OUTCOME: total_incremental_outcome,
        c.TOTAL_ROI: total_incremental_outcome / budget,
        c.TOTAL_CPIK: total_cpik,
        c.IS_REVENUE_KPI: (
            self._meridian.input_data.kpi_type == c.REVENUE or not use_kpi
        ),
        c.CONFIDENCE_LEVEL: confidence_level,
        c.USE_HISTORICAL_BUDGET: use_historical_budget,
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            c.CHANNEL: (
                [c.CHANNEL],
                self._meridian.input_data.get_all_paid_channels(),
            ),
            c.METRIC: (
                [c.METRIC],
                [c.MEAN, c.MEDIAN, c.CI_LO, c.CI_HI],
            ),
        },
        attrs=attributes | (attrs or {}),
    )

  def _get_optimization_bounds(
      self,
      spend: np.ndarray,
      spend_constraint_lower: _SpendConstraint | None,
      spend_constraint_upper: _SpendConstraint | None,
      round_factor: int,
      fixed_budget: bool,
  ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Get optimization bounds from spend and spend constraints.

    Args:
      spend: np.ndarray with size `n_total_channels` containing media-level
        spend for all media and RF channels.
      spend_constraint_lower: Numeric list of size `n_total_channels` or float
        (same constraint for all media) indicating the lower bound of
        media-level spend. The lower bound of media-level spend is `(1 -
        spend_constraint_lower) * budget * allocation)`. The value must be
        between 0-1.
      spend_constraint_upper: Numeric list of size `n_total_channels` or float
        (same constraint for all media) indicating the upper bound of
        media-level spend. The upper bound of media-level spend is `(1 +
        spend_constraint_upper) * budget * allocation)`.
      round_factor: Integer number of digits to round optimization bounds.
      fixed_budget: Boolean indicating whether it's a fixed budget optimization
        or flexible budget optimization.

    Returns:
      lower_bound: np.ndarray of size `n_total_channels` containing the treated
        lower bound spend for each media and RF channel.
      upper_bound: np.ndarray of size `n_total_channels` containing the treated
        upper bound spend for each media and RF channel.
      spend_bounds: tuple of np.ndarray of size `n_total_channels` containing
        the untreated lower and upper bound spend for each media and RF channel.
    """
    (spend_const_lower, spend_const_upper) = self._validate_spend_constraints(
        fixed_budget, spend_constraint_lower, spend_constraint_upper
    )
    spend_bounds = (
        np.maximum((1 - spend_const_lower), 0),
        (1 + spend_const_upper),
    )

    lower_bound = np.round(
        (spend_bounds[0] * spend),
        round_factor,
    ).astype(int)
    upper_bound = np.round(spend_bounds[1] * spend, round_factor).astype(int)
    return (lower_bound, upper_bound, spend_bounds)

  def _update_incremental_outcome_grid(
      self,
      i: int,
      incremental_outcome_grid: np.ndarray,
      multipliers_grid: tf.Tensor,
      selected_times: Sequence[str],
      use_posterior: bool = True,
      use_kpi: bool = False,
      optimal_frequency: xr.DataArray | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ):
    """Updates incremental_outcome_grid for each channel.

    Args:
      i: Row index used in updating incremental_outcome_grid.
      incremental_outcome_grid: Discrete two-dimensional grid with the number of
        rows determined by the `spend_constraints` and `step_size`, and the
        number of columns is equal to the number of total channels, containing
        incremental outcome by channel.
      multipliers_grid: A grid derived from spend.
      selected_times: Sequence of strings representing the time dimensions in
        `meridian.input_data.time` to use for optimization.
      use_posterior: Boolean. If `True`, then the incremental outcome is derived
        from the posterior distribution of the model. Otherwise, the prior
        distribution is used.
      use_kpi: Boolean. If `True`, then the incremental outcome is derived from
        the KPI impact. Otherwise, the incremental outcome is derived from the
        revenue impact.
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
          * self._meridian.media_tensors.media
      )
    else:
      new_media = None

    if self._meridian.n_rf_channels == 0:
      new_frequency = None
      new_reach = None
    elif optimal_frequency is not None:
      new_frequency = (
          tf.ones_like(self._meridian.rf_tensors.frequency) * optimal_frequency
      )
      new_reach = tf.math.divide_no_nan(
          multipliers_grid[i, -self._meridian.n_rf_channels :]
          * self._meridian.rf_tensors.reach
          * self._meridian.rf_tensors.frequency,
          new_frequency,
      )
    else:
      new_frequency = self._meridian.rf_tensors.frequency
      new_reach = (
          multipliers_grid[i, -self._meridian.n_rf_channels :]
          * self._meridian.rf_tensors.reach
      )

    # incremental_outcome returns a three dimensional tensor with dims
    # (n_chains x n_draws x n_total_channels). Incremental_outcome_grid requires
    # incremental outcome by channel.
    incremental_outcome_grid[i, :] = np.mean(
        self._analyzer.incremental_outcome(
            use_posterior=use_posterior,
            new_data=analyzer.DataTensors(
                media=new_media,
                reach=new_reach,
                frequency=new_frequency,
            ),
            selected_times=selected_times,
            use_kpi=use_kpi,
            include_non_paid_channels=False,
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
      use_posterior: bool = True,
      use_kpi: bool = False,
      optimal_frequency: xr.DataArray | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Creates spend and incremental outcome grids for optimization algorithm.

    Args:
      spend: `np.ndarray` with actual spend per media or RF channel.
      spend_bound_lower: `np.ndarray` of dimension (`n_total_channels`)
        containing the lower constraint spend for each channel.
      spend_bound_upper: `np.ndarray` of dimension (`n_total_channels`)
        containing the upper constraint spend for each channel.
      step_size: Integer indicating the step size, or interval, between values
        in the spend grid. All media channels have the same step size.
      selected_times: Sequence of strings representing the time dimensions in
        `meridian.input_data.time` to use for optimization.
      use_posterior: Boolean. If `True`, then the incremental outcome is derived
        from the posterior distribution of the model. Otherwise, the prior
        distribution is used.
      use_kpi: Boolean. If `True`, then the incremental outcome is derived from
        the KPI impact. Otherwise, the incremental outcome is derived from the
        revenue impact.
      optimal_frequency: `xr.DataArray` with dimension `n_rf_channels`,
        containing the optimal frequency per channel, that maximizes mean ROI
        over the corresponding prior/posterior distribution. Value is `None` if
        the model does not contain reach and frequency data, or if the model
        does contain reach and frequency data, but historical frequency is used
        for the optimization scenario.
      batch_size: Max draws per chain in each batch. The calculation is run in
        batches to avoid memory exhaustion. If a memory error occurs, try
        reducing `batch_size`. The calculation will generally be faster with
        larger `batch_size` values.

    Returns:
      spend_grid: Discrete two-dimensional grid with the number of rows
        determined by the `spend_bound_**` and `step_size`, and the number of
        columns is equal to the number of total channels, containing spend by
        channel.
      incremental_outcome_grid: Discrete two-dimensional grid with the number of
        rows determined by the `spend_bound_**` and `step_size`, and the
        number of columns is equal to the number of total channels, containing
        incremental outcome by channel.
    """
    n_grid_rows = int(
        (np.max(np.subtract(spend_bound_upper, spend_bound_lower)) // step_size)
        + 1
    )
    n_grid_columns = len(self._meridian.input_data.get_all_paid_channels())
    spend_grid = np.full([n_grid_rows, n_grid_columns], np.nan)
    for i in range(n_grid_columns):
      spend_grid_m = np.arange(
          spend_bound_lower[i],
          spend_bound_upper[i] + step_size,
          step_size,
      )
      spend_grid[: len(spend_grid_m), i] = spend_grid_m
    incremental_outcome_grid = np.full([n_grid_rows, n_grid_columns], np.nan)
    multipliers_grid_base = tf.cast(
        tf.math.divide_no_nan(spend_grid, spend), dtype=tf.float32
    )
    multipliers_grid = np.where(
        np.isnan(spend_grid), np.nan, multipliers_grid_base
    )
    for i in range(n_grid_rows):
      self._update_incremental_outcome_grid(
          i=i,
          incremental_outcome_grid=incremental_outcome_grid,
          multipliers_grid=multipliers_grid,
          selected_times=selected_times,
          use_posterior=use_posterior,
          use_kpi=use_kpi,
          optimal_frequency=optimal_frequency,
          batch_size=batch_size,
      )
    # In theory, for RF channels, incremental_outcome/spend should always be
    # same despite of spend, But given the level of precision,
    # incremental_outcome/spend could have very tiny difference in high
    # decimals. This tiny difference will cause issue in
    # np.unravel_index(np.nanargmax(iROAS_grid), iROAS_grid.shape). Therefore
    # we use the following code to fix it, and ensure incremental_outcome/spend
    # is always same for RF channels.
    if self._meridian.n_rf_channels > 0:
      rf_incremental_outcome_max = np.nanmax(
          incremental_outcome_grid[:, -self._meridian.n_rf_channels :], axis=0
      )
      rf_spend_max = np.nanmax(
          spend_grid[:, -self._meridian.n_rf_channels :], axis=0
      )
      rf_roi = tf.math.divide_no_nan(rf_incremental_outcome_max, rf_spend_max)
      incremental_outcome_grid[:, -self._meridian.n_rf_channels :] = (
          rf_roi * spend_grid[:, -self._meridian.n_rf_channels :]
      )
    return (spend_grid, incremental_outcome_grid)

  def _grid_search(
      self,
      spend_grid: np.ndarray,
      incremental_outcome_grid: np.ndarray,
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
      incremental_outcome_grid: Discrete grid with dimensions (`grid_length` x
        `n_total_channels`) containing incremental outcome by channel for all
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
        media spend that maximizes incremental outcome based on spend
        constraints for all media and RF channels.
      optimal_inc_outcome: np.ndarry of dimension (`n_total_channels`)
        containing the post optimization incremental outcome per channel for all
        media and RF channels.
    """
    spend = spend_grid[0, :].copy()
    incremental_outcome = incremental_outcome_grid[0, :].copy()
    spend_grid = spend_grid[1:, :]
    incremental_outcome_grid = incremental_outcome_grid[1:, :]
    iterative_roi_grid = np.round(
        tf.math.divide_no_nan(
            incremental_outcome_grid - incremental_outcome, spend_grid - spend
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
      incremental_outcome[media_idx] = incremental_outcome_grid[
          row_idx, media_idx
      ]
      roi_grid_point = iterative_roi_grid[row_idx, media_idx]
      if _exceeds_optimization_constraints(
          fixed_budget,
          budget,
          spend,
          incremental_outcome,
          roi_grid_point,
          target_mroi,
          target_roi,
      ):
        break

      iterative_roi_grid[0 : row_idx + 1, media_idx] = np.nan
      iterative_roi_grid[row_idx + 1 :, media_idx] = np.round(
          tf.math.divide_no_nan(
              incremental_outcome_grid[row_idx + 1 :, media_idx]
              - incremental_outcome_grid[row_idx, media_idx],
              spend_grid[row_idx + 1 :, media_idx]
              - spend_grid[row_idx, media_idx],
          ),
          decimals=8,
      )
    return spend_optimal


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
    incremental_outcome: np.ndarray,
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
    incremental_outcome: np.ndarray with dimensions (`n_total_channels`)
      containing incremental outcome per channel for all media and RF channels.
    roi_grid_point: float roi for non-optimized optimation step.
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
    bool indicating whether optimal spend and incremental outcome have been
      found, given the optimization constraints.
  """
  if fixed_budget:
    return np.sum(spend) > budget
  elif target_roi is not None:
    return (np.sum(incremental_outcome) / np.sum(spend)) < target_roi
  else:
    return roi_grid_point < target_mroi
