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

"""Plotting and visualization functions for Model Quality Checks."""

import warnings

import altair as alt
from meridian.analysis.review import calibration_details_plots
from meridian.analysis.review import calibration_overview_plots
from meridian.analysis.review import constants
from meridian.analysis.review import results
import numpy as np
import pandas as pd

__all__ = [
    "build_calibration_details_chart",
    "build_calibration_overview_chart",
    "generate_calibration_details_chart_json",
    "generate_calibration_overview_chart_json",
    "generate_high_variance_chart_json",
    "generate_implausible_roi_chart_json",
    "generate_potential_bias_chart_json",
]


def generate_implausible_roi_chart_json(
    result: results.ImplausibleROICheckResult | None,
) -> str | None:
  """Generates the single-chart scaled Altair chart JSON for Implausible ROI.

  Args:
    result: ImplausibleROICheckResult | None.

  Returns:
    The serialized JSON string for the chart, or None if not applicable.
  """
  if result is None or not result.channel_results:
    return None

  # The plotted layout is fixed: [0, GAP) is the bottom band, GAP is the axis
  # break where ROIs in [lower_break, upper_break) are clustered, and
  # [GAP, MAX] is the top band. The mapping from real ROI to plotted position
  # is derived from the configured thresholds.
  gap_plotted = constants.IMPLAUSIBLE_ROI_GAP_PLOTTED
  max_plotted = constants.IMPLAUSIBLE_ROI_MAX_PLOTTED
  lower_break = result.plot_cluster_lower_bound
  upper_break = result.plot_cluster_upper_bound
  max_real = result.roi_upper_bound * constants.IMPLAUSIBLE_ROI_MAX_RATIO
  bottom_scale = gap_plotted / lower_break if lower_break > 0 else 0.0
  top_scale = (
      (max_plotted - gap_plotted) / (max_real - upper_break)
      if max_real > upper_break
      else 1.0
  )

  def _is_bottom(y: float) -> bool:
    return y < lower_break

  def _is_top(y: float) -> bool:
    return not _is_bottom(y) and y >= upper_break

  def _scale_roi(y: float) -> float:
    if _is_bottom(y):
      return y * bottom_scale
    elif not _is_top(y):
      return gap_plotted
    else:
      return gap_plotted + (min(y, max_real) - upper_break) * top_scale

  rows = []
  for idx, cr in enumerate(result.channel_results, start=1):
    legend_label = (
        f"{cr.channel_name} (Spend = {cr.spend_share * 100:.1f}%, ROI ="
        f" {cr.roi_mean:.1f})"
    )
    rows.append({
        constants.CHANNEL_ID: str(idx),
        constants.CHANNEL_NAME: cr.channel_name,
        constants.SPEND_SHARE: cr.spend_share,
        constants.ROI_MEAN: cr.roi_mean,
        constants.Y_PLOTTED: _scale_roi(cr.roi_mean),
        constants.LEGEND_LABEL: legend_label,
    })
  df = pd.DataFrame(rows)

  legend_df = pd.DataFrame([
      {
          constants.LEGEND_LABEL: (
              f"{cr.channel_name} (Spend = {cr.spend_share * 100:.1f}%, ROI ="
              f" {cr.roi_mean:.1f})"
          ),
          constants.SPEND_SHARE: 0.0,
          constants.ROI_MEAN: constants.IMPLAUSIBLE_ROI_GAP_PLOTTED,
          constants.Y_PLOTTED: constants.IMPLAUSIBLE_ROI_GAP_PLOTTED,
      }
      for cr in result.channel_results
  ])

  legend_labels = df[constants.LEGEND_LABEL].tolist()
  channel_color_scale = alt.Scale(
      domain=legend_labels, range=constants.CHANNEL_COLORS
  )
  channel_legend = alt.Legend(
      title=constants.CHANNELS_LEGEND_TITLE,
      orient="bottom",
      columns=2,
      labelLimit=0,
      symbolSize=100,
      labelFontSize=11,
      titleFontSize=12,
  )

  is_bottom = df[constants.ROI_MEAN].map(_is_bottom).astype(bool)
  is_top = df[constants.ROI_MEAN].map(_is_top).astype(bool)
  df_top = df[is_top].copy()
  df_bottom = df[is_bottom].copy()
  df_gap = df[~is_bottom & ~is_top].copy()

  x_curve = np.linspace(0.01, 1.0, 100)
  y_upper_true = result.roi_upper_bound / x_curve
  upper_spend_shares = np.concatenate(([0.0], x_curve))
  upper_y_plotted = np.concatenate((
      [
          constants.IMPLAUSIBLE_ROI_MAX_PLOTTED,
      ],
      [_scale_roi(y) for y in y_upper_true],
  ))
  upper_region = pd.DataFrame({
      constants.SPEND_SHARE: upper_spend_shares,
      constants.Y_PLOTTED: upper_y_plotted,
      constants.Y2_PLOTTED: np.full_like(
          upper_spend_shares, constants.IMPLAUSIBLE_ROI_MAX_PLOTTED
      ),
      constants.REGION: (
          [constants.IMPLAUSIBLE_HIGH_ROI] * len(upper_spend_shares)
      ),
  })

  x_lower = np.linspace(0.0, 1.0, 100)
  y_lower_true = result.roi_lower_bound * x_lower
  lower_region = pd.DataFrame({
      constants.SPEND_SHARE: x_lower,
      constants.Y_PLOTTED: np.zeros_like(x_lower),
      constants.Y2_PLOTTED: [_scale_roi(y) for y in y_lower_true],
      constants.REGION: [constants.IMPLAUSIBLE_LOW_ROI] * len(x_lower),
  })

  region_color_scale = alt.Scale(
      domain=[constants.IMPLAUSIBLE_HIGH_ROI, constants.IMPLAUSIBLE_LOW_ROI],
      range=[
          constants.IMPLAUSIBLE_ROI_UPPER_COLOR,
          constants.IMPLAUSIBLE_ROI_LOWER_COLOR,
      ],
  )
  region_legend = alt.Legend(
      title=constants.DIAGNOSTIC_THRESHOLDS_TITLE,
      orient="right",
      titleFontSize=11,
      labelFontSize=11,
      symbolType="square",
  )

  unified_y_scale = alt.Scale(domain=[0.0, max_plotted], clamp=True)
  y_ticks: list[tuple[float, str]] = [(0.0, "0.0")]
  if lower_break > 0:
    for ratio in constants.IMPLAUSIBLE_ROI_LOWER_TICK_RATIOS:
      tick_roi = ratio * result.roi_lower_bound
      if _is_bottom(tick_roi):
        y_ticks.append((_scale_roi(tick_roi), f"{tick_roi:g}"))
  y_ticks.append((gap_plotted, constants.BREAK_MARK_TEXT))
  num_upper_ticks = constants.IMPLAUSIBLE_ROI_NUM_UPPER_TICKS
  for k in range(1, num_upper_ticks + 1):
    tick_roi = k * result.roi_upper_bound
    if not _is_top(tick_roi):
      continue
    label = f"{tick_roi:g}"
    if k == num_upper_ticks:
      label += "+"
    y_ticks.append((_scale_roi(tick_roi), label))
  y_ticks = [(round(float(pos), 6), label) for pos, label in y_ticks]
  label_expr = "".join(
      f"datum.value == {pos} ? '{label}' : " for pos, label in y_ticks
  )
  unified_y_axis = alt.Axis(
      values=[pos for pos, _ in y_ticks],
      labelExpr=f"{label_expr}''",
  )

  area_upper = (
      alt.Chart(upper_region)
      .mark_area(opacity=0.15, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q",
              scale=alt.Scale(domain=[0, 1.0]),
              axis=alt.Axis(
                  format="%", title=constants.SPEND_PERCENT_TITLE, grid=True
              ),
          ),
          y=alt.Y(
              f"{constants.Y_PLOTTED}:Q",
              scale=unified_y_scale,
              axis=unified_y_axis,
              title=constants.ROI_TITLE,
          ),
          y2=alt.Y2(f"{constants.Y2_PLOTTED}:Q"),
          color=alt.Color(
              f"{constants.REGION}:N",
              scale=region_color_scale,
              legend=region_legend,
          ),
      )
  )

  line_upper = (
      alt.Chart(upper_region)
      .mark_line(
          color=constants.IMPLAUSIBLE_ROI_UPPER_COLOR,
          strokeDash=[4, 4],
          clip=True,
      )
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
      )
  )

  area_lower = (
      alt.Chart(lower_region)
      .mark_area(opacity=0.15, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          y2=alt.Y2(f"{constants.Y2_PLOTTED}:Q"),
          color=alt.Color(
              f"{constants.REGION}:N", scale=region_color_scale, legend=None
          ),
      )
  )

  line_lower = (
      alt.Chart(lower_region)
      .mark_line(
          color=constants.IMPLAUSIBLE_ROI_LOWER_LINE_COLOR,
          strokeDash=[4, 4],
          clip=True,
      )
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y2_PLOTTED}:Q", scale=unified_y_scale),
      )
  )

  points_bottom = (
      alt.Chart(df_bottom)
      .mark_point(filled=True, size=60, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          color=alt.Color(
              f"{constants.LEGEND_LABEL}:N",
              scale=channel_color_scale,
              legend=None,
          ),
          tooltip=[
              constants.CHANNEL_ID,
              constants.CHANNEL_NAME,
              constants.SPEND_SHARE,
              constants.ROI_MEAN,
          ],
      )
  )

  points_top = (
      alt.Chart(df_top)
      .mark_point(filled=True, size=60, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          color=alt.Color(
              f"{constants.LEGEND_LABEL}:N",
              scale=channel_color_scale,
              legend=None,
          ),
          tooltip=[
              constants.CHANNEL_ID,
              constants.CHANNEL_NAME,
              constants.SPEND_SHARE,
              constants.ROI_MEAN,
          ],
      )
  )

  points_gap = (
      alt.Chart(df_gap)
      .mark_point(filled=True, size=60, clip=False)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          color=alt.Color(
              f"{constants.LEGEND_LABEL}:N",
              scale=channel_color_scale,
              legend=None,
          ),
          tooltip=[
              constants.CHANNEL_ID,
              constants.CHANNEL_NAME,
              constants.SPEND_SHARE,
              constants.ROI_MEAN,
          ],
      )
  )

  break_mark_single = (
      alt.Chart(
          pd.DataFrame({
              constants.SPEND_SHARE: [0.0],
              constants.Y_PLOTTED: [constants.IMPLAUSIBLE_ROI_GAP_PLOTTED],
              constants.TEXT: [constants.BREAK_MARK_TEXT],
          })
      )
      .mark_text(
          align="center",
          baseline="middle",
          size=14,
          fontWeight="bold",
          color=constants.IMPLAUSIBLE_ROI_BREAK_TEXT_COLOR,
          dy=-1,
          dx=-1,
          clip=False,
      )
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          text=f"{constants.TEXT}:N",
      )
  )

  legend_layer = (
      alt.Chart(legend_df)
      .mark_point(filled=True, size=0, opacity=0)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
          color=alt.Color(
              f"{constants.LEGEND_LABEL}:N",
              scale=channel_color_scale,
              legend=channel_legend,
          ),
      )
  )

  chart = (
      alt.layer(
          area_upper,
          line_upper,
          area_lower,
          line_lower,
          points_bottom,
          points_top,
          points_gap,
          break_mark_single,
          legend_layer,
      )
      .properties(width=400, height=300)
      .resolve_scale(color="independent")
  )

  return chart.to_json()


def generate_high_variance_chart_json(
    result: results.HighVarianceCheckResult | None,
) -> str | None:
  """Generates the Altair chart JSON for High Variance ROI.

  Args:
    result: HighVarianceCheckResult | None.

  Returns:
    The serialized JSON string for the chart, or None if not applicable.
  """
  if result is None or not result.channel_results:
    return None

  rows = []
  for idx, cr in enumerate(result.channel_results, start=1):
    legend_label = (
        f"{cr.channel_name} (Spend = {cr.spend_share * 100:.1f}%, RCI ="
        f" {cr.relative_width_ratio:.2f})"
    )
    rows.append({
        constants.CHANNEL_ID: str(idx),
        constants.CHANNEL_NAME: cr.channel_name,
        constants.SPEND_SHARE: cr.spend_share,
        constants.RELATIVE_WIDTH: cr.relative_width_ratio,
        constants.LEGEND_LABEL: legend_label,
    })
  df = pd.DataFrame(rows)

  legend_labels = df[constants.LEGEND_LABEL].tolist()
  channel_color_scale = alt.Scale(
      domain=legend_labels, range=constants.CHANNEL_COLORS
  )
  channel_legend = alt.Legend(
      title=constants.CHANNELS_LEGEND_TITLE,
      orient="bottom",
      columns=2,
      labelLimit=0,
      symbolSize=100,
      labelFontSize=11,
      titleFontSize=12,
  )

  x_curve = np.linspace(0.01, 1.0, 100)
  threshold = result.high_variance_threshold
  y_upper_curve = threshold / x_curve
  upper_spend_shares = np.concatenate(([0.0], x_curve))
  upper_y_plotted = np.concatenate((
      [constants.HIGH_VARIANCE_RCI_MAX_PLOTTED],
      [min(y, constants.HIGH_VARIANCE_RCI_MAX_PLOTTED) for y in y_upper_curve],
  ))
  upper_region = pd.DataFrame({
      constants.SPEND_SHARE: upper_spend_shares,
      constants.Y_PLOTTED: upper_y_plotted,
      constants.Y2_PLOTTED: np.full_like(
          upper_spend_shares, constants.HIGH_VARIANCE_RCI_MAX_PLOTTED
      ),
      constants.REGION: [constants.HIGH_VARIANCE_ROI] * len(upper_spend_shares),
  })

  region_color_scale = alt.Scale(
      domain=[constants.HIGH_VARIANCE_ROI],
      range=[constants.HIGH_VARIANCE_UPPER_COLOR],
  )
  region_legend = alt.Legend(
      title=constants.DIAGNOSTIC_THRESHOLDS_TITLE,
      orient="right",
      titleFontSize=11,
      labelFontSize=11,
      symbolType="square",
  )

  unified_y_scale = alt.Scale(
      domain=[0.0, constants.HIGH_VARIANCE_RCI_MAX_PLOTTED], clamp=True
  )

  area_upper = (
      alt.Chart(upper_region)
      .mark_area(opacity=0.15, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q",
              scale=alt.Scale(domain=[0, 1.0]),
              axis=alt.Axis(
                  format="%", title=constants.SPEND_PERCENT_TITLE, grid=True
              ),
          ),
          y=alt.Y(
              f"{constants.Y_PLOTTED}:Q",
              scale=unified_y_scale,
              title=constants.RCI_TITLE,
          ),
          y2=alt.Y2(f"{constants.Y2_PLOTTED}:Q"),
          color=alt.Color(
              f"{constants.REGION}:N",
              scale=region_color_scale,
              legend=region_legend,
          ),
      )
  )

  line_upper = (
      alt.Chart(upper_region)
      .mark_line(
          color=constants.HIGH_VARIANCE_UPPER_COLOR,
          strokeDash=[4, 4],
          clip=True,
      )
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.Y_PLOTTED}:Q", scale=unified_y_scale),
      )
  )

  points = (
      alt.Chart(df)
      .mark_point(filled=True, size=60, clip=True)
      .encode(
          x=alt.X(
              f"{constants.SPEND_SHARE}:Q", scale=alt.Scale(domain=[0, 1.0])
          ),
          y=alt.Y(f"{constants.RELATIVE_WIDTH}:Q", scale=unified_y_scale),
          color=alt.Color(
              f"{constants.LEGEND_LABEL}:N",
              scale=channel_color_scale,
              legend=channel_legend,
          ),
          tooltip=[
              constants.CHANNEL_ID,
              constants.CHANNEL_NAME,
              constants.SPEND_SHARE,
              constants.RELATIVE_WIDTH,
          ],
      )
  )

  chart = (
      alt.layer(area_upper, line_upper, points)
      .properties(width=400, height=300)
      .resolve_scale(color="independent")
  )

  return chart.to_json()


def generate_potential_bias_chart_json(
    result: results.PotentialBiasCheckResult | None,
) -> str | None:
  """Generates the Altair chart JSON for Potential Bias.

  Args:
    result: PotentialBiasCheckResult | None.

  Returns:
    The serialized JSON string for the chart, or None if not applicable.
  """
  if (
      result is None
      or result.correlation_matrix is None
      or getattr(result.correlation_matrix, "ndim", 0) == 0
      or getattr(result.correlation_matrix, "size", 0) == 0
  ):
    return None

  try:
    df = result.correlation_matrix.to_dataframe(
        name=constants.CORRELATION
    ).reset_index()
  except (ValueError, KeyError, AttributeError, TypeError):
    return None

  if (
      df.empty
      or constants.CORRELATION not in df.columns
      or constants.CHANNEL not in df.columns
      or constants.CONTROL_VARIABLE not in df.columns
  ):
    return None

  df[constants.PAIR] = (
      df[constants.CHANNEL] + " - " + df[constants.CONTROL_VARIABLE]
  )

  df[constants.ABS_CORRELATION] = df[constants.CORRELATION].abs()
  idx = df.groupby([constants.CHANNEL, constants.CONTROL_VARIABLE])[
      constants.ABS_CORRELATION
  ].idxmax()
  df_max = df.loc[idx].copy()
  df_max[constants.IS_MAX] = True

  df = df.merge(
      df_max[[
          constants.GEO,
          constants.CHANNEL,
          constants.CONTROL_VARIABLE,
          constants.IS_MAX,
      ]],
      on=[constants.GEO, constants.CHANNEL, constants.CONTROL_VARIABLE],
      how="left",
  )
  # After the left merge, the column holds True or NaN; map NaN to False.
  df[constants.IS_MAX] = df[constants.IS_MAX].eq(True)

  threshold = result.correlation_threshold
  max_abs_corr = (
      float(df[constants.CORRELATION].abs().max()) if not df.empty else 0.0
  )
  x_limit = max(threshold, max_abs_corr) + 0.05

  rect_df = pd.DataFrame([{
      constants.X1: -threshold,
      constants.X2: threshold,
  }])

  rect = (
      alt.Chart(rect_df)
      .mark_rect(color=constants.POTENTIAL_BIAS_RECT_COLOR, opacity=0.08)
      .encode(x=f"{constants.X1}:Q", x2=f"{constants.X2}:Q")
  )

  rule_left = (
      alt.Chart(pd.DataFrame([{constants.X: -threshold}]))
      .mark_rule(
          color=constants.POTENTIAL_BIAS_THRESHOLD_LINE_COLOR,
          strokeDash=[4, 4],
      )
      .encode(x=f"{constants.X}:Q")
  )

  rule_right = (
      alt.Chart(pd.DataFrame([{constants.X: threshold}]))
      .mark_rule(
          color=constants.POTENTIAL_BIAS_THRESHOLD_LINE_COLOR,
          strokeDash=[4, 4],
      )
      .encode(x=f"{constants.X}:Q")
  )

  points_geos = (
      alt.Chart(df)
      .mark_point(
          filled=True,
          size=40,
          color=constants.POTENTIAL_BIAS_GEO_POINT_COLOR,
          opacity=0.6,
          clip=True,
      )
      .encode(
          x=alt.X(
              f"{constants.CORRELATION}:Q",
              scale=alt.Scale(domain=[-x_limit, x_limit]),
              title=constants.PEARSON_CORRELATION_TITLE,
          ),
          y=alt.Y(f"{constants.PAIR}:N", title=None),
          tooltip=[
              constants.CHANNEL,
              constants.CONTROL_VARIABLE,
              constants.GEO,
              constants.CORRELATION,
          ],
      )
  )

  df_max[constants.FILL_COLOR] = np.where(
      df_max[constants.ABS_CORRELATION] < threshold,
      constants.POTENTIAL_BIAS_REVIEW_FILL_COLOR,
      constants.POTENTIAL_BIAS_PASS_FILL_COLOR,
  )
  df_max[constants.STROKE_COLOR] = np.where(
      df_max[constants.ABS_CORRELATION] < threshold,
      constants.POTENTIAL_BIAS_REVIEW_STROKE_COLOR,
      constants.POTENTIAL_BIAS_PASS_STROKE_COLOR,
  )
  points_max = (
      alt.Chart(df_max)
      .mark_point(shape="diamond", size=120, strokeWidth=2, clip=True)
      .encode(
          x=alt.X(
              f"{constants.CORRELATION}:Q",
              scale=alt.Scale(domain=[-x_limit, x_limit]),
          ),
          y=alt.Y(f"{constants.PAIR}:N"),
          fill=alt.Fill(f"{constants.FILL_COLOR}:N", scale=None),
          stroke=alt.Stroke(f"{constants.STROKE_COLOR}:N", scale=None),
          tooltip=[
              constants.CHANNEL,
              constants.CONTROL_VARIABLE,
              constants.CORRELATION,
          ],
      )
  )

  legend_df = pd.DataFrame([
      {constants.LABEL: constants.INDIVIDUAL_GEO_CORRELATION, constants.X: 0.0},
      {constants.LABEL: constants.MAX_ABS_CORRELATION, constants.X: 0.0},
  ])
  legend_layer = (
      alt.Chart(legend_df)
      .mark_circle(size=0, opacity=0)
      .encode(
          x=alt.X(
              f"{constants.X}:Q", scale=alt.Scale(domain=[-x_limit, x_limit])
          ),
          shape=alt.Shape(
              f"{constants.LABEL}:N",
              scale=alt.Scale(
                  domain=[
                      constants.INDIVIDUAL_GEO_CORRELATION,
                      constants.MAX_ABS_CORRELATION,
                  ],
                  range=["circle", "diamond"],
              ),
              legend=alt.Legend(title=None, symbolSize=100),
          ),
          color=alt.Color(
              f"{constants.LABEL}:N",
              scale=alt.Scale(
                  domain=[
                      constants.INDIVIDUAL_GEO_CORRELATION,
                      constants.MAX_ABS_CORRELATION,
                  ],
                  range=[
                      constants.POTENTIAL_BIAS_GEO_POINT_COLOR,
                      constants.POTENTIAL_BIAS_MAX_POINT_COLOR,
                  ],
              ),
              legend=alt.Legend(title=None),
          ),
      )
  )

  chart = alt.layer(
      rect, rule_left, rule_right, points_geos, points_max, legend_layer
  ).properties(width=400, height=300)

  return chart.to_json()


build_calibration_details_chart = (
    calibration_details_plots.build_calibration_details_chart
)


def generate_calibration_details_chart_json(
    ch_data: results.CalibrationOverviewChannelData | None,
) -> str | None:
  """Generates the Altair chart JSON for a calibration details channel chart."""
  try:
    chart = build_calibration_details_chart(ch_data)
    if chart is None:
      return None
    return chart.to_json()
  except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
    warnings.warn(
        "Failed to generate calibration details chart for channel"
        f" '{ch_data.channel_name if ch_data is not None else 'unknown'}': {e}",
        RuntimeWarning,
    )
    return None


build_calibration_overview_chart = (
    calibration_overview_plots.build_calibration_overview_chart
)


def generate_calibration_overview_chart_json(
    ch_data: results.CalibrationOverviewChannelData | None,
) -> str | None:
  """Generates the Altair chart JSON for a calibration overview channel chart."""
  try:
    chart = build_calibration_overview_chart(ch_data)
    if chart is None:
      return None
    return chart.to_json()
  except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
    warnings.warn(
        "Failed to generate calibration overview chart for channel"
        f" '{getattr(ch_data, 'channel_name', 'unknown')}': {e}",
        RuntimeWarning,
    )
    return None
