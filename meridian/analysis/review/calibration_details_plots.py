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

"""Calibration details chart builders for the Model Quality Checks."""

import altair as alt
from meridian.analysis.review import constants
from meridian.analysis.review import results
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
import numpy as np
import pandas as pd

__all__ = [
    "build_calibration_details_chart",
]


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
