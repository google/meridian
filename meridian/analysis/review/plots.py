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

from typing import cast
import warnings

import altair as alt
from meridian import backend
from meridian.analysis.review import constants
from meridian.analysis.review import results
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
import numpy as np
import pandas as pd


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

  def _scale_roi(y: float) -> float:
    if y < constants.IMPLAUSIBLE_ROI_THRESHOLD_LOWER:
      return y * constants.IMPLAUSIBLE_ROI_SCALE_FACTOR
    elif y < constants.IMPLAUSIBLE_ROI_GAP_PLOTTED:
      return constants.IMPLAUSIBLE_ROI_GAP_PLOTTED
    else:
      return min(y, constants.IMPLAUSIBLE_ROI_MAX_PLOTTED)

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

  df_top = df[
      df[constants.ROI_MEAN] >= constants.IMPLAUSIBLE_ROI_GAP_PLOTTED
  ].copy()
  df_bottom = df[
      df[constants.ROI_MEAN] < constants.IMPLAUSIBLE_ROI_THRESHOLD_LOWER
  ].copy()
  df_gap = df[
      (df[constants.ROI_MEAN] >= constants.IMPLAUSIBLE_ROI_THRESHOLD_LOWER)
      & (df[constants.ROI_MEAN] < constants.IMPLAUSIBLE_ROI_GAP_PLOTTED)
  ].copy()

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

  unified_y_scale = alt.Scale(
      domain=[0.0, constants.IMPLAUSIBLE_ROI_MAX_PLOTTED], clamp=True
  )
  unified_y_axis = alt.Axis(
      values=[
          0,
          5,
          10,
          constants.IMPLAUSIBLE_ROI_GAP_PLOTTED,
          20,
          40,
          60,
          80,
          constants.IMPLAUSIBLE_ROI_MAX_PLOTTED,
      ],
      labelExpr=(
          "datum.value == 0 ? '0.0' : datum.value == 5 ? '0.2' : datum.value"
          " == 10 ? '0.4' : datum.value =="
          f" {constants.IMPLAUSIBLE_ROI_GAP_PLOTTED} ?"
          f" '{constants.BREAK_MARK_TEXT}' : datum.value == 20 ? '20' :"
          " datum.value == 40 ? '40' : datum.value == 60 ? '60' : datum.value"
          " == 80 ? '80' : datum.value =="
          f" {constants.IMPLAUSIBLE_ROI_MAX_PLOTTED} ? '100+' : ''"
      ),
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
  threshold = 1.0
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
  df[constants.IS_MAX] = df[constants.IS_MAX].fillna(False)

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


def _compute_experiment_adjustment_stages(
    experiment: calibration_base.CalibratedExperiment,
) -> list[tuple[str, float, float]]:
  """Computes (stage_name, mean, se) tuples across all adjustment stages."""
  mu = experiment.raw_experiment_result.point_estimate
  se = experiment.raw_experiment_result.standard_error
  tau_s = experiment.tau_spend
  gamma_d = experiment.gamma_duration
  tau_d = experiment.tau_duration
  tau_r = experiment.tau_recency
  gamma_u = experiment.user_point_estimate_adjustment
  tau_u = experiment.user_standard_error_adjustment
  final_mu = experiment.adjusted_experiment_result.point_estimate
  final_se = experiment.adjusted_experiment_result.standard_error

  if tau_s < -1.0:
    raise ValueError(f"`tau_spend` must be >= -1.0, got {tau_s}.")

  stages = [
      (eda_constants.STAGE_UNADJUSTED_RAW, mu, se),
      (
          eda_constants.STAGE_SPEND_ADJUSTED,
          mu,
          se * np.sqrt(max(0.0, 1.0 + tau_s)),
      ),
      (
          eda_constants.STAGE_SPEND_DURATION_ADJUSTED,
          gamma_d * mu,
          se * np.sqrt(max(0.0, 1.0 + tau_s + tau_d)),
      ),
      (
          eda_constants.STAGE_SPEND_DURATION_RECENCY_ADJUSTED,
          gamma_d * mu,
          se * np.sqrt(max(0.0, 1.0 + tau_s + tau_d + tau_r)),
      ),
  ]
  if gamma_u is not None or tau_u is not None:
    gu_val = gamma_u if gamma_u is not None else 0.0
    tu_val = tau_u if tau_u is not None else 0.0
    stages.append((
        eda_constants.STAGE_SPEND_DURATION_RECENCY_USER_ADJUSTED,
        (gamma_d + gu_val) * mu,
        se * np.sqrt(max(0.0, 1.0 + tau_s + tau_d + tau_r + tu_val)),
    ))
  stages.append((eda_constants.STAGE_FINAL_ADJUSTED, final_mu, final_se))
  return stages


def _format_experiment_label(
    exp_idx: int,
    source_type: calibration_base.SourceType,
) -> str:
  """Formats an experiment label with its 1-based index and source type suffix."""
  label_suffix = calibration_plots.get_experiment_label_suffix(source_type)
  return f"{eda_constants.EXPERIMENT_LABEL_PREFIX} {exp_idx}{label_suffix}"


def _prepare_experiment_adjustment_df_for_channel(
    experiment: calibration_base.CalibratedExperiment,
    exp_idx: int,
) -> pd.DataFrame:
  """Processes experiment adjustment data into a DataFrame for Altair errorbar plotting."""
  exp_label = _format_experiment_label(exp_idx, experiment.source_type)
  stages = _compute_experiment_adjustment_stages(experiment)
  baseline_mean, baseline_se = stages[0][1], stages[0][2]
  num_stages = len(stages)
  rows = []
  for stage_idx, (stage_name, mean, se) in enumerate(stages):
    if stage_idx == 0:
      label_text = f"Baseline\nM: {mean:.2f}\nSE: {se:.2f}"
    elif stage_idx == num_stages - 1:
      label_text = f"Final\nM: {mean:.2f}\nSE: {se:.2f}"
    else:
      delta_m = mean - baseline_mean
      delta_se = se - baseline_se
      label_text = f"ΔM: {delta_m:+.2f}\nΔSE: {delta_se:+.2f}"
    rows.append({
        eda_constants.VARIABLE: exp_label,
        eda_constants.STAGE: stage_name,
        eda_constants.POINT_ESTIMATE: mean,
        eda_constants.STANDARD_ERROR: se,
        eda_constants.CI_LOWER: mean - se,
        eda_constants.CI_UPPER: mean + se,
        eda_constants.LABEL_TEXT: label_text,
    })
  return pd.DataFrame(rows)


def _build_single_experiment_adjustment_chart(
    df: pd.DataFrame,
    title: str,
    color_hex: str,
) -> alt.LayerChart:
  """Constructs and layers Altair components for a single experiment adjustment."""
  x_encoding = alt.X(
      f"{eda_constants.STAGE}:N",
      sort=None,
      title=None,
      axis=alt.Axis(
          labelAngle=-20,
          labelFontSize=11,
          labelExpr=r"split(datum.label, '\n')",
      ),
  )
  y_encoding = alt.Y(
      f"{eda_constants.POINT_ESTIMATE}:Q",
      title=eda_constants.MEAN_ROI_PLUS_MINUS_SE,
      scale=alt.Scale(zero=False, padding=70),
  )
  color_encoding = alt.value(color_hex)

  tooltips = [
      alt.Tooltip(f"{eda_constants.STAGE}:N", title="Stage"),
      alt.Tooltip(f"{eda_constants.VARIABLE}:N", title="Experiment"),
      alt.Tooltip(
          f"{eda_constants.POINT_ESTIMATE}:Q",
          title="Point Estimate (Mean)",
          format=".4f",
      ),
      alt.Tooltip(
          f"{eda_constants.STANDARD_ERROR}:Q",
          title="Standard Error (SE)",
          format=".4f",
      ),
      alt.Tooltip(
          f"{eda_constants.CI_LOWER}:Q", title="Mean - SE", format=".4f"
      ),
      alt.Tooltip(
          f"{eda_constants.CI_UPPER}:Q", title="Mean + SE", format=".4f"
      ),
  ]

  rules = (
      alt.Chart(df)
      .mark_rule(strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(
              f"{eda_constants.CI_LOWER}:Q",
              title=eda_constants.MEAN_ROI_PLUS_MINUS_SE,
              scale=alt.Scale(zero=False, padding=70),
          ),
          y2=alt.Y2(f"{eda_constants.CI_UPPER}:Q"),
          color=color_encoding,
          tooltip=tooltips,
      )
  )
  ticks_lower = (
      alt.Chart(df)
      .mark_tick(size=12, strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(f"{eda_constants.CI_LOWER}:Q"),
          color=color_encoding,
          tooltip=tooltips,
      )
  )
  ticks_upper = (
      alt.Chart(df)
      .mark_tick(size=12, strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(f"{eda_constants.CI_UPPER}:Q"),
          color=color_encoding,
          tooltip=tooltips,
      )
  )
  points = (
      alt.Chart(df)
      .mark_point(filled=False, size=90, strokeWidth=2)
      .encode(
          x=x_encoding,
          y=y_encoding,
          color=color_encoding,
          tooltip=tooltips,
      )
  )

  text_layers = []
  for count, dy in [(2, -38), (1, -26)]:
    sub_df = (
        df[df[eda_constants.LABEL_TEXT].str.count("\n") >= count]
        if count == 2
        else df[df[eda_constants.LABEL_TEXT].str.count("\n") == 1]
    )
    if not sub_df.empty:
      text_layers.append(
          alt.Chart(sub_df)
          .mark_text(
              align="center",
              baseline="top",
              dy=dy,
              fontSize=10,
              fontWeight="bold",
              lineBreak="\n",
          )
          .encode(
              x=x_encoding,
              y=alt.Y(f"{eda_constants.CI_UPPER}:Q"),
              text=f"{eda_constants.LABEL_TEXT}:N",
              color=color_encoding,
          )
      )

  return alt.layer(
      rules, ticks_lower, ticks_upper, points, *text_layers
  ).properties(
      title=alt.TitleParams(
          text=title,
          anchor="start",
          fontSize=12,
          fontWeight="bold",
      ),
      width=400,
      height=220,
  )


def build_calibration_details_chart(
    ch_data: results.CalibrationOverviewChannelData | None,
) -> alt.Chart | None:
  """Builds the experiment adjustments grid chart for a single channel."""
  if (
      ch_data is None
      or ch_data.calibrated_output is None
      or not ch_data.calibrated_output.experiments
  ):
    return None

  experiments_to_plot = calibration_plots.filter_and_sort_experiments(
      ch_data.calibrated_output.experiments,
      lambda exp: exp.adjusted_experiment_result.standard_error,
      limit_experiments=constants.MAX_EXPERIMENTS_FOR_DETAILS_CARD,
      sort_experiments=True,
  )
  if not experiments_to_plot:
    return None

  sub_charts = []
  for idx, (exp_idx, exp) in enumerate(experiments_to_plot):
    exp_name = _format_experiment_label(exp_idx, exp.source_type)
    sub_title = f"Experiment Adjustments: {ch_data.channel_name} ({exp_name})"
    color_hex = eda_constants.EXPERIMENT_COLORS[
        idx % len(eda_constants.EXPERIMENT_COLORS)
    ]
    exp_df = _prepare_experiment_adjustment_df_for_channel(exp, exp_idx)
    sub_charts.append(
        _build_single_experiment_adjustment_chart(exp_df, sub_title, color_hex)
    )

  return alt.hconcat(*sub_charts) if len(sub_charts) > 1 else sub_charts[0]  # pyrefly: ignore[bad-return]


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
