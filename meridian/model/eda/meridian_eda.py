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

"""Meridian Exploratory Data Analysis (EDA) visualization tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import os
import re
from typing import Literal, TYPE_CHECKING

import altair as alt
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_engine as eda_engine_module
from meridian.model.eda import eda_outcome
from meridian.model.eda import sampling_eda_engine
from meridian.templates import formatter
import numpy as np
import pandas as pd
import xarray as xr

_CalibrationResultsByChannel = dict[
    str,
    tuple[
        calibration_base.CalibrationOutput,
        backend.tfd.Distribution,
    ],
]


_make_calibration_plot_df = calibration_plots.make_calibration_plot_df
_create_roi_grid = calibration_plots.create_roi_grid
_filter_and_sort_experiments = calibration_plots.filter_and_sort_experiments
_CalibrationPlotData = calibration_plots.CalibrationPlotData
_prepare_calibration_data = calibration_plots.prepare_calibration_data
_make_density_line_chart = calibration_plots.make_density_line_chart


def _get_total_spend_by_channel(
    eda_engine: eda_engine_module.EDAEngine,
) -> xr.DataArray | None:
  """Computes total spend by channel from the EDA engine."""
  if eda_engine.national_all_spend_ds is None:
    return None
  stacked_spend = eda_engine_module.stack_variables(
      eda_engine.national_all_spend_ds, constants.CHANNEL
  )
  return stacked_spend.sum(dim=constants.TIME)


_create_interactive_hover_layers = (
    calibration_plots.create_interactive_hover_layers
)
_build_calibration_chart = calibration_plots.build_calibration_chart
_plot_single_channel_calibration = (
    calibration_plots.plot_single_channel_calibration
)


def _build_experiment_adjustment_chart_layers(
    df: pd.DataFrame, channel_title: str | Sequence[str]
) -> alt.Chart:
  """Constructs and layers the Altair components for experiment adjustment grid."""
  x_encoding = alt.X(
      f'{eda_constants.STAGE}:N',
      sort=None,
      title=None,
      axis=alt.Axis(
          labelAngle=-20,
          labelFontSize=11,
          labelExpr=r"split(datum.label, '\n')",
      ),
  )
  y_encoding = alt.Y(
      f'{eda_constants.POINT_ESTIMATE}:Q',
      title=eda_constants.MEAN_ROI_PLUS_MINUS_SE,
      scale=alt.Scale(zero=False, padding=70),
  )
  color_encoding = alt.Color(
      f'{eda_constants.VARIABLE}:N',
      legend=None,
  )

  tooltips = [
      alt.Tooltip(f'{eda_constants.STAGE}:N', title='Stage'),
      alt.Tooltip(f'{eda_constants.VARIABLE}:N', title='Experiment'),
      alt.Tooltip(
          f'{eda_constants.POINT_ESTIMATE}:Q',
          title='Point Estimate (Mean)',
          format='.4f',
      ),
      alt.Tooltip(
          f'{eda_constants.STANDARD_ERROR}:Q',
          title='Standard Error (SE)',
          format='.4f',
      ),
      alt.Tooltip(
          f'{eda_constants.CI_LOWER}:Q', title='Mean - SE', format='.4f'
      ),
      alt.Tooltip(
          f'{eda_constants.CI_UPPER}:Q', title='Mean + SE', format='.4f'
      ),
  ]

  rules = (
      alt.Chart(df)
      .mark_rule(strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(
              f'{eda_constants.CI_LOWER}:Q',
              title=eda_constants.MEAN_ROI_PLUS_MINUS_SE,
              scale=alt.Scale(zero=False, padding=70),
          ),
          y2=alt.Y2(f'{eda_constants.CI_UPPER}:Q'),
          color=color_encoding,
          tooltip=tooltips,
      )
  )

  ticks_lower = (
      alt.Chart(df)
      .mark_tick(size=12, strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(f'{eda_constants.CI_LOWER}:Q'),
          color=color_encoding,
          tooltip=tooltips,
      )
  )

  ticks_upper = (
      alt.Chart(df)
      .mark_tick(size=12, strokeWidth=2)
      .encode(
          x=x_encoding,
          y=alt.Y(f'{eda_constants.CI_UPPER}:Q'),
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

  df_3_lines = df[df[eda_constants.LABEL_TEXT].str.count('\n') >= 2]
  df_2_lines = df[df[eda_constants.LABEL_TEXT].str.count('\n') == 1]

  text_layer_3 = (
      alt.Chart(df_3_lines)
      .mark_text(
          align='center',
          baseline='top',
          dy=-38,
          fontSize=10,
          fontWeight='bold',
          lineBreak='\n',
      )
      .encode(
          x=x_encoding,
          y=alt.Y(f'{eda_constants.CI_UPPER}:Q'),
          text=f'{eda_constants.LABEL_TEXT}:N',
          color=color_encoding,
      )
  )

  text_layer_2 = (
      alt.Chart(df_2_lines)
      .mark_text(
          align='center',
          baseline='top',
          dy=-26,
          fontSize=10,
          fontWeight='bold',
          lineBreak='\n',
      )
      .encode(
          x=x_encoding,
          y=alt.Y(f'{eda_constants.CI_UPPER}:Q'),
          text=f'{eda_constants.LABEL_TEXT}:N',
          color=color_encoding,
      )
  )

  return alt.layer(  # pyrefly: ignore[bad-return]
      rules, ticks_lower, ticks_upper, points, text_layer_3, text_layer_2
  ).properties(
      title=alt.TitleParams(
          text=channel_title,
          anchor='start',
          fontSize=12,
          fontWeight='bold',
      ),
      width=400,
      height=220,
  )


def _format_experiment_label(
    exp_idx: int,
    source_type: calibration_base.SourceType,
) -> str:
  """Formats an experiment label with its 1-based index and source type suffix."""
  label_suffix = calibration_plots.get_experiment_label_suffix(source_type)
  return f'{eda_constants.EXPERIMENT_LABEL_PREFIX} {exp_idx}{label_suffix}'


def _prepare_experiment_adjustment_df(
    indexed_exp_data_list: Sequence[
        tuple[int, eda_outcome.ExperimentAdjustmentData]
    ],
) -> pd.DataFrame:
  """Processes experiment adjustment data into a DataFrame for Altair plotting."""
  rows = []
  for exp_idx, exp_data in indexed_exp_data_list:
    exp_label = _format_experiment_label(
        exp_idx, calibration_base.SourceType(exp_data.source_type)
    )

    baseline_mean = exp_data.stages[0].point_estimate
    baseline_se = exp_data.stages[0].standard_error
    num_stages = len(exp_data.stages)

    for stage_idx, stage_data in enumerate(exp_data.stages):
      stage_name = stage_data.stage.value
      mean = stage_data.point_estimate
      se = stage_data.standard_error

      if stage_idx == 0:
        label_text = f'Baseline\nM: {mean:.2f}\nSE: {se:.2f}'
      elif stage_idx == num_stages - 1:
        label_text = f'Final\nM: {mean:.2f}\nSE: {se:.2f}'
      else:
        delta_m = mean - baseline_mean
        delta_se = se - baseline_se
        label_text = f'ΔM: {delta_m:+.2f}\nΔSE: {delta_se:+.2f}'

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


def _build_experiment_adjustment_grid(
    ch_name: str,
    indexed_exp_data_list: Sequence[
        tuple[int, eda_outcome.ExperimentAdjustmentData]
    ],
    spend: float | None = None,
    show_spend_in_title: bool = True,
) -> alt.Chart | None:
  """Builds the Altair errorbar grid chart for a single channel's experiment adjustments."""
  chart_title = eda_constants.EXPERIMENT_ADJUSTMENTS_PLOT_TITLE_TEMPLATE.format(
      ch_name=ch_name
  )
  if show_spend_in_title and spend is not None:
    channel_title = f'{chart_title} (Total Spend: ${spend:,.0f})'
  else:
    channel_title = chart_title

  sub_charts = []
  for exp_idx, exp_data in indexed_exp_data_list:
    exp_df = _prepare_experiment_adjustment_df([(exp_idx, exp_data)])
    exp_name = _format_experiment_label(
        exp_idx, calibration_base.SourceType(exp_data.source_type)
    )
    sub_title = [
        f'{channel_title}',
        f'({exp_name})',
    ]
    sub_chart = _build_experiment_adjustment_chart_layers(exp_df, sub_title)
    sub_charts.append(sub_chart)

  if not sub_charts:
    return None
  return alt.hconcat(*sub_charts) if len(sub_charts) > 1 else sub_charts[0]  # pyrefly: ignore[bad-return]


if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order,g-import-not-at-top


__all__ = [
    'MeridianEDA',
    'BoxplotSortingConfig',
]

Geos = int | Sequence[str] | Literal[eda_constants.NATIONALIZE]  # pyrefly: ignore[invalid-literal]

_BULLET_POINT_RE = re.compile(r'(?m)^\* ')


@dataclasses.dataclass(frozen=True)
class BoxplotSortingConfig:
  """Configuration for sorting variables in boxplots.

  Attributes:
    std_artifact: StandardDeviationArtifact containing std dev and outlier info.
    n_vars: The maximum number of variables to display. If None, displays all
      variables.
  """

  std_artifact: eda_outcome.StandardDeviationArtifact
  n_vars: int | None = None

  def __post_init__(self):
    if self.n_vars is not None and self.n_vars <= 0:
      raise ValueError(
          f'n_vars must be a positive integer, but got {self.n_vars}.'
      )


def _prepare_boxplot_sorting_metrics(
    sorting_config: BoxplotSortingConfig,
    unique_variables: Sequence[str],
) -> pd.DataFrame:
  """Pre-computes sorting metrics for boxplots.

  This function takes a BoxplotSortingConfig and a list of unique variables
  and prepares a DataFrame used to sort variables in boxplots. The sorting
  is based on standard deviation without outliers and the maximum absolute
  outlier value.

  Args:
    sorting_config: Configuration containing the StandardDeviationArtifact.
    unique_variables: A sequence of variable names to include in the sorting
      metrics.

  Returns:
    A pandas DataFrame indexed by variable (and potentially geo) containing
    'std_without_outliers' and 'abs_outliers' columns, sorted in ascending
    order by 'std_without_outliers' and descending order by 'abs_outliers'.

  Raises:
    ValueError: If the variable dimension cannot be found in the
      `std_artifact.std_ds`, or if any of the `unique_variables` are not
      present in the `std_artifact`.
  """
  std_ds = sorting_config.std_artifact.std_ds
  std_without_outliers_da = std_ds[eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME]
  outlier_df = sorting_config.std_artifact.outlier_df

  var_dim_name = next(
      (dim for dim in std_without_outliers_da.dims if dim != constants.GEO),
      None,
  )
  if var_dim_name is None:
    raise ValueError('Could not determine variable dimension name in std_ds.')

  # Check missing variables
  artifact_vars = std_without_outliers_da.coords[var_dim_name].values
  missing_vars = set(unique_variables) - set(artifact_vars)
  if missing_vars:
    raise ValueError(
        'The following variables are missing from the std_artifact:'
        f' {missing_vars}.'
    )

  filtered_std_da = std_without_outliers_da.sel(
      {var_dim_name: list(unique_variables)}
  )
  std_df = filtered_std_da.to_dataframe()

  group_keys = (
      [var_dim_name, constants.GEO]
      if constants.GEO in std_df.index.names
      else [var_dim_name]
  )

  # Filter outlier_df to only include unique_variables in a single assignment
  # Also safely guards against empty DataFrames without that index.
  filtered_outlier_df = outlier_df[
      outlier_df.index.get_level_values(var_dim_name).isin(unique_variables)
  ]

  outlier_info = filtered_outlier_df.groupby(group_keys)[
      eda_constants.ABS_OUTLIERS_COL_NAME
  ].max()

  return (
      std_df.join(outlier_info, how='left')
      .fillna(0.0)
      .sort_values(
          by=[
              eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME,
              eda_constants.ABS_OUTLIERS_COL_NAME,
          ],
          ascending=[True, False],
      )
  )


def _get_sorted_variables(
    sorting_metrics: pd.DataFrame,
    geo_to_plot: str,
    n_vars: int | None,
) -> list[str]:
  """Slices pre-computed metrics and returns sorted variables."""
  if constants.GEO in sorting_metrics.index.names:
    # Filter for the specific geo
    metrics_slice = sorting_metrics.xs(geo_to_plot, level=constants.GEO)
  else:
    metrics_slice = sorting_metrics

  sorted_vars = metrics_slice.index.tolist()

  if n_vars is None or n_vars >= len(sorted_vars):
    return sorted_vars

  return sorted_vars[:n_vars]


class MeridianEDA:
  """Handles EDA report generation and visualization for a Meridian model."""

  def __init__(
      self,
      meridian: model.Meridian,
      *,
      n_draws_prior: int = eda_constants.DEFAULT_PRIOR_N_DRAW,
      seed: int = eda_constants.DEFAULT_PRIOR_SEED,
  ):
    """Initializes the instance.

    Args:
      meridian: Meridian object.
      n_draws_prior: Number of draws for prior distribution check. Only used if
        no prior samples exist in `meridian.inference_data`.
      seed: Seed for prior distribution check. Only used if no prior samples
        exist in `meridian.inference_data`.
    """

    self._meridian = meridian
    self._template_env = formatter.create_template_env()

    if constants.PRIOR not in self._meridian.inference_data.groups():
      self._meridian.sample_prior(n_draws=n_draws_prior, seed=seed)

    analyzer = analyzer_module.Analyzer(
        model_context=self._meridian.model_context,
        inference_data=self._meridian.inference_data,
    )
    # TODO: Use EDAEngine when EDAEngine can use Analyzer without
    # circular dependencies.
    self._eda_engine = sampling_eda_engine.SamplingEDAEngine(
        analyzer,
        self._meridian.eda_spec,
    )

  @property
  # TODO: Use EDAEngine when EDAEngine can use Analyzer without
  # circular dependencies.
  def eda_engine(self) -> sampling_eda_engine.SamplingEDAEngine:
    return self._eda_engine

  @functools.cached_property
  def critical_outcomes(self) -> eda_outcome.CriticalCheckEDAOutcomes:
    return self.eda_engine.run_all_critical_checks()

  @functools.cached_property
  def national_cost_per_media_unit_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.CostPerMediaUnitArtifact]:
    return self.eda_engine.check_national_cost_per_media_unit()

  @functools.cached_property
  def geo_cost_per_media_unit_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.CostPerMediaUnitArtifact]:
    return self.eda_engine.check_geo_cost_per_media_unit()

  @functools.cached_property
  def national_pairwise_correlation_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    if self._meridian.is_national:
      return self._dataset_level_pairwise_correlation_check_outcome
    return self.eda_engine.check_national_pairwise_corr()

  @functools.cached_property
  def geo_pairwise_correlation_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    if not self._meridian.is_national:
      return self._dataset_level_pairwise_correlation_check_outcome
    return self.eda_engine.check_geo_pairwise_corr()

  @functools.cached_property
  def national_stdev_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    return self.eda_engine.check_national_std()

  @functools.cached_property
  def geo_stdev_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    return self.eda_engine.check_geo_std()

  @functools.cached_property
  def _dataset_level_pairwise_correlation_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PairwiseCorrArtifact]:
    return self.critical_outcomes.pairwise_correlation

  @functools.cached_property
  def _dataset_level_vif_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.VIFArtifact]:
    return self.critical_outcomes.multicollinearity

  @functools.cached_property
  def _overall_kpi_invariability_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.KpiInvariabilityArtifact]:
    return self.critical_outcomes.kpi_invariability

  @functools.cached_property
  def _dataset_level_stdev_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.StandardDeviationArtifact]:
    return self.eda_engine.check_std()

  @functools.cached_property
  def _dataset_level_cost_per_media_unit_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.CostPerMediaUnitArtifact]:
    return self.eda_engine.check_cost_per_media_unit()

  @functools.cached_property
  def _dataset_level_population_raw_media_correlation_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PopulationCorrelationArtifact]:
    return self.eda_engine.check_population_corr_raw_media()

  @functools.cached_property
  def _dataset_level_population_treatment_correlation_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PopulationCorrelationArtifact]:
    return self.eda_engine.check_population_corr_scaled_treatment_control()

  @functools.cached_property
  def _dataset_level_prior_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PriorProbabilityArtifact]:
    return self.eda_engine.check_prior_probability()

  @functools.cached_property
  def _dataset_level_prior_quality_check_outcome(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PriorQualityArtifact] | None:
    """Runs and caches the dataset-level prior quality check outcome.

    Returns:
      An EDAOutcome containing a PriorQualityArtifact, or None if unavailable.
    """
    try:
      return self.eda_engine.check_prior_quality()
    except (AttributeError, ValueError):
      return None

  def generate_and_save_report(self, filename: str, filepath: str) -> None:
    """Generates and saves a HTML report containing EDA findings.

    Args:
      filename: The filename for the generated HTML output.
      filepath: The path to the directory where the file will be saved.
    """
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as f:
      f.write(self._generate_report_html())

  def _generate_report_html(self) -> str:
    """Generates the HTML content for the EDA report.

    This method orchestrates the creation of various sections of the EDA report,
    including spend and media unit, response variables, and relationship among
    variables. It also compiles a summary card based on the findings.

    Returns:
      A string containing the complete HTML content of the EDA report.
    """

    card_severity_counts = {
        card: _initialize_severity_counts()
        for card in eda_constants.CATEGORY_TO_MESSAGE_BY_STATUS.keys()
    }
    card_generators = [
        (
            self._generate_spend_and_media_unit_card,
            eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
        ),
        (
            self._generate_response_variables_card,
            eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
        ),
        (
            self._generate_population_scaling_card,
            eda_constants.POPULATION_SCALING_CARD_TITLE,
        ),
        (
            self._generate_relationship_among_variables_card,
            eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
        ),
        (
            self._generate_prior_specifications_card,
            eda_constants.PRIOR_SPECIFICATIONS_CARD_TITLE,
        ),
    ]
    card_htmls = []
    for generate_function, card in card_generators:
      card_html, counts = generate_function()
      if not card_html:
        _ = card_severity_counts.pop(card, None)
        continue

      card_htmls.append(card_html)
      for severity, count in counts.items():
        card_severity_counts[card][severity] += count
        card_severity_counts[eda_constants.SUMMARY_CARD_TITLE][
            severity
        ] += count

    return formatter.create_summary_html(
        self._template_env,
        title=eda_constants.REPORT_TITLE,
        cards=[self._generate_summary_card(card_severity_counts)] + card_htmls,
    )

  def plot_national_kpi_with_knots_time_series(self) -> alt.Chart:
    """Plots the national scaled KPI with knots time series plot."""
    nat_scaled_kpi_da = self.eda_engine.national_kpi_scaled_da

    kpi_df = nat_scaled_kpi_da.to_dataframe(
        name=eda_constants.VALUE
    ).reset_index()

    selected_knots = self._meridian.knot_info.knot_locations
    if len(selected_knots) < 2:
      raise ValueError(
          'This feature requires more than one knot. With a single'
          ' knot, the model uses a single, common intercept across all time'
          ' periods.'
      )
    time_at_knots = nat_scaled_kpi_da.time.values[selected_knots]
    kpi_at_knots = nat_scaled_kpi_da.values[selected_knots]

    knots_df = pd.DataFrame({
        constants.TIME: time_at_knots,
        eda_constants.VALUE: kpi_at_knots,
        eda_constants.LABEL: 'knots',
    })

    line_chart = _plot_time_series(
        data=kpi_df,
        title='Time Series of scaled national KPI with Knots',
        x_axis_title=constants.TIME,
        y_axis_title=constants.NATIONAL + ' ' + constants.KPI_SCALED,
    )

    knot_markers = (
        alt.Chart(knots_df)
        .mark_point(shape='triangle-up', size=50, color='red')
        .encode(
            x=alt.X(f'{constants.TIME}:T'),
            y=alt.Y(f'{eda_constants.VALUE}:Q'),
            color=alt.Color(
                f'{eda_constants.LABEL}:N',
                scale=alt.Scale(range=['red']),
                legend=alt.Legend(title=None),
            ),
        )
    )
    return (
        (line_chart + knot_markers)
        .properties(width=600, height=400)
        .resolve_scale(color='independent')
        .configure_title(anchor='start')
        .configure_axis(labelOverlap='parity')
    )

  def _get_calibration_results_by_channel(
      self,
  ) -> _CalibrationResultsByChannel:
    """Maps calibrated channel names to their output and 1D prior distribution."""
    calibrated_priors = self.eda_engine.get_calibrated_priors()
    if not calibrated_priors:
      return {}

    calibration_results = {}
    for joint_prior in calibrated_priors:
      for idx, output in enumerate(joint_prior.calibration_outputs):
        if output is not None:
          calibration_results[output.channel_name] = (
              output,
              joint_prior.distributions[idx],
          )
    return calibration_results

  def _sort_channels_by_spend(self, channel_names: Sequence[str]) -> list[str]:
    """Sorts the channel names by total national spend in descending order."""
    total_spend_da = _get_total_spend_by_channel(self.eda_engine)
    if total_spend_da is None:
      return list(channel_names)

    available_channels = set(total_spend_da[constants.CHANNEL].values)
    for c in channel_names:
      if c not in available_channels:
        raise ValueError(f"Channel '{c}' is not present in the spend dataset.")

    return sorted(
        channel_names,
        key=lambda c: float(total_spend_da.sel({constants.CHANNEL: c}).values),
        reverse=True,
    )

  def _get_plotting_calibrated_channels(
      self,
      channel_name: str | None,
      limit_channels: int | None,
      sort_channels: bool,
      calibrated_channels: Sequence[str],
  ) -> Sequence[str]:
    """Resolves, sorts, and limits the channels to plot.

    Args:
      channel_name: Optional name of a specific channel. If provided, only this
        channel is returned.
      limit_channels: Optional maximum number of channels to return.
      sort_channels: If True, sorts channels by total spend.
      calibrated_channels: List of all calibrated channel names.

    Returns:
      Sequence of channel names to plot.

    Raises:
      ValueError: If `channel_name` is provided but not in
        `calibrated_channels`, or if `sort_channels` is True and any
        calibrated channel is not present in the spend dataset.
    """
    if channel_name is not None:
      if channel_name not in calibrated_channels:
        raise ValueError(
            f"Channel '{channel_name}' is not calibrated or has no calibration"
            ' outputs.'
        )
      return [channel_name]

    if not calibrated_channels:
      return []

    sorted_channels = (
        self._sort_channels_by_spend(calibrated_channels)
        if sort_channels
        else list(calibrated_channels)
    )
    return (
        sorted_channels[:limit_channels]
        if limit_channels is not None
        else sorted_channels
    )

  # TODO: UX review and Eng fix on color codes.
  def plot_calibration(
      self,
      channel_name: str | None = None,
      limit_channels: int | None = None,
      limit_experiments: int | None = None,
      sort_channels: bool = True,
      sort_experiments: bool = True,
  ) -> alt.Chart | None:
    """Plots the prior calibration comparison charts (1x2 subplots).

    Args:
      channel_name: Optional name of a specific channel. If None, plots all
        calibrated channels.
      limit_channels: Optional maximum number of channels to display (sorted by
        spend).
      limit_experiments: Optional maximum number of experiments to display per
        channel (sorted by SE).
      sort_channels: If True, ranks channels by total spend.
      sort_experiments: If True, ranks experiments by adjusted SE (ascending).

    Returns:
      An Altair Chart containing the calibration plots, or None if no calibrated
      priors exist in the model.
    """
    if limit_channels is not None and limit_channels <= 0:
      raise ValueError('limit_channels must be greater than 0.')
    if limit_experiments is not None and limit_experiments <= 0:
      raise ValueError('limit_experiments must be greater than 0.')

    calibration_results = self._get_calibration_results_by_channel()
    if not calibration_results:
      return None

    # We assume only paid channels can be calibrated. Non-paid channels
    # (e.g., organic) are not supported for calibration plotting as they lack
    # spend data.
    channels_to_plot = self._get_plotting_calibrated_channels(
        channel_name=channel_name,
        limit_channels=limit_channels,
        sort_channels=sort_channels,
        calibrated_channels=list(calibration_results.keys()),
    )

    rng_handler = backend.RNGHandler(seed=eda_constants.DEFAULT_PRIOR_SEED)
    total_spend_da = _get_total_spend_by_channel(self.eda_engine)

    channel_charts = []
    for ch_name in channels_to_plot:
      calibrated_output, calibrated_prior_dist = calibration_results[ch_name]
      indexed_experiments = _filter_and_sort_experiments(
          experiments=calibrated_output.experiments,
          get_se_fn=lambda e: e.adjusted_experiment_result.standard_error,
          limit_experiments=limit_experiments,
          sort_experiments=sort_experiments,
      )
      channel_chart = _plot_single_channel_calibration(
          ch_name=ch_name,
          calibrated_output=calibrated_output,
          calibrated_prior_dist=calibrated_prior_dist,
          indexed_experiments=indexed_experiments,
          rng_handler=rng_handler,
          total_spend_da=total_spend_da,
          show_spend_in_title=(channel_name is None),
      )
      channel_charts.append(channel_chart)

    return (
        alt.vconcat(*channel_charts).resolve_scale(color='independent')
        if channel_charts
        else None
    )

  def plot_experiment_adjustments(
      self,
      channel_name: str | None = None,
      limit_channels: int | None = None,
      limit_experiments: int | None = None,
      sort_channels: bool = True,
      sort_experiments: bool = True,
  ) -> alt.Chart | None:
    """Plots the step-by-step experiment adjustments errorbar grid.

    Args:
      channel_name: Name of the specific channel to plot. Default is None. If
        None, plots all calibrated channels.
      limit_channels: Optional maximum number of channels to display (sorted by
        spend).
      limit_experiments: Optional maximum number of experiments to display per
        channel (sorted by Standard Error).
      sort_channels: If True, ranks channels by total spend.
      sort_experiments: If True, ranks experiments by adjusted Standard Error
        (ascending).

    Returns:
      An Altair Chart containing the experiment adjustment errorbar grid, or
      None
      if no calibrated priors exist in the model.
    """
    if limit_channels is not None and limit_channels <= 0:
      raise ValueError('limit_channels must be greater than 0.')
    if limit_experiments is not None and limit_experiments <= 0:
      raise ValueError('limit_experiments must be greater than 0.')

    outcome = self.eda_engine.check_experiment_adjustment()
    try:
      [artifact] = outcome.get_overall_artifacts()
    except ValueError:
      return None

    adjustment_data = artifact.adjustment_data

    channels_to_plot = self._get_plotting_calibrated_channels(
        channel_name=channel_name,
        limit_channels=limit_channels,
        sort_channels=sort_channels,
        calibrated_channels=list(adjustment_data.keys()),
    )
    if not channels_to_plot:
      return None

    total_spend_da = _get_total_spend_by_channel(self.eda_engine)

    channel_charts = []
    for ch_name in channels_to_plot:
      if ch_name not in adjustment_data:
        continue
      indexed_exp_data_list = _filter_and_sort_experiments(
          experiments=adjustment_data[ch_name],
          get_se_fn=lambda exp: exp.stages[-1].standard_error,
          limit_experiments=limit_experiments,
          sort_experiments=sort_experiments,
      )
      if not indexed_exp_data_list:
        continue

      if (
          channel_name is None
          and total_spend_da is not None
          and ch_name in total_spend_da[constants.CHANNEL].values
      ):
        spend = float(total_spend_da.sel({constants.CHANNEL: ch_name}).values)
      else:
        spend = None

      channel_chart = _build_experiment_adjustment_grid(
          ch_name=ch_name,
          indexed_exp_data_list=indexed_exp_data_list,
          spend=spend,
          show_spend_in_title=(channel_name is None),
      )
      if channel_chart is not None:
        channel_charts.append(channel_chart)

    if not channel_charts:
      return None
    return alt.vconcat(*channel_charts).resolve_scale(color='independent')

  def plot_cost_per_media_unit_time_series(
      self, geos: Geos = 1, channels: Sequence[str] | None = None
  ) -> alt.Chart:
    """Plots cost per media unit time series for paid media channels.

    Args:
        geos: The geos to plot.
        channels: The channels to plot. If None, plots all channels.

    Returns:
        The cost per media unit time series plot and the superimposed cost with
        media unit time series plot for each paid media channel.
    """
    geos_to_plot = self._validate_and_get_geos_to_plot(geos)
    use_national_data = (
        self._meridian.is_national or geos == eda_constants.NATIONALIZE
    )

    if use_national_data:
      cost_data = self.eda_engine.national_all_spend_ds
      media_unit_data = self.eda_engine.national_paid_raw_media_units_ds
      [cost_per_media_unit_artifact] = (
          self.national_cost_per_media_unit_check_outcome.get_national_artifacts()
      )
      select_data = lambda data, geo: data
    else:
      cost_data = self.eda_engine.all_spend_ds
      media_unit_data = self.eda_engine.paid_raw_media_units_ds
      [cost_per_media_unit_artifact] = (
          self.geo_cost_per_media_unit_check_outcome.get_geo_artifacts()
      )
      select_data = lambda data, geo: data.sel(geo=geo)

    final_charts = []
    for geo_to_plot in geos_to_plot:
      cost_plot_data = _process_stacked_ds(
          eda_engine_module.stack_variables(
              select_data(cost_data, geo_to_plot)
          ),
          include_time=True,
      )
      media_unit_plot_data = _process_stacked_ds(
          eda_engine_module.stack_variables(
              select_data(media_unit_data, geo_to_plot)
          ),
          include_time=True,
      )
      cost_per_media_unit_plot_data = _process_stacked_ds(
          select_data(
              cost_per_media_unit_artifact.cost_per_media_unit_da, geo_to_plot
          )
          .rename({constants.CHANNEL: eda_constants.VARIABLE})
          .rename(eda_constants.VALUE),
          include_time=True,
      )

      available_channels = cost_per_media_unit_plot_data[
          eda_constants.VARIABLE
      ].unique()

      channels_to_plot = (
          [c for c in channels if c in available_channels]
          if channels
          else available_channels
      )

      current_geo_charts = []
      for channel_to_plot in channels_to_plot:
        superimposed_chart_title = (
            f'Time series of {geo_to_plot} level {constants.SPEND} and'
            f' {constants.MEDIA_UNITS} for {channel_to_plot}'
        )
        cost_plot = _plot_time_series(
            data=cost_plot_data[
                cost_plot_data[eda_constants.VARIABLE] == channel_to_plot
            ],
            title=superimposed_chart_title,
            x_axis_title=constants.TIME,
            y_axis_title=constants.SPEND,
        )
        media_unit_plot = _plot_time_series(
            data=media_unit_plot_data[
                media_unit_plot_data[eda_constants.VARIABLE] == channel_to_plot
            ],
            title=superimposed_chart_title,
            x_axis_title=constants.TIME,
            y_axis_title=constants.MEDIA_UNITS,
            y_axis_orient='right',
            y_axis_offset=10,
        )
        cost_per_media_unit_plot = _plot_time_series(
            data=cost_per_media_unit_plot_data[
                cost_per_media_unit_plot_data[eda_constants.VARIABLE]
                == channel_to_plot
            ],
            title=(
                f'Time series of {geo_to_plot} level'
                f' {eda_constants.COST_PER_MEDIA_UNIT} for {channel_to_plot}'
            ),
            x_axis_title=constants.TIME,
            y_axis_title=eda_constants.COST_PER_MEDIA_UNIT,
        )

        current_geo_charts.append(
            alt.vconcat(
                (cost_plot + media_unit_plot).resolve_scale(y='independent'),
                cost_per_media_unit_plot,
            )
        )
      final_charts.append(alt.vconcat(*current_geo_charts))
    return _apply_chart_config(
        alt.vconcat(*final_charts).resolve_legend(color='independent')
    )

  def plot_relative_spend_share_barchart(
      self,
      geos: Geos = 1,
      n_channels: int | None = None,
      ascending: bool = True,
  ) -> alt.Chart:
    """Plots a bar chart of the relative spend share per paid media channel.

    Args:
      geos: The geos to plot.
      n_channels: The number of channels to display by spend share. If None,
        displays all channels.
      ascending: Whether to sort the channels in ascending order by spend share.

    Returns:
      The bar chart of the relative spend share of each paid media channel.
    """
    return self._plot_barcharts(
        geos=geos,
        title_prefix='Relative spend share of paid media channels',
        x_axis_title=eda_constants.SPEND_SHARE,
        y_axis_title=constants.CHANNEL,
        national_data_source=self.eda_engine.national_all_spend_ds,
        geo_data_source=self.eda_engine.all_spend_ds,
        processing_function=lambda data: _calculate_relative_shares(
            _process_stacked_ds(eda_engine_module.stack_variables(data))
        ),
        n_channels=n_channels,
        ascending=ascending,
    )

  def plot_relative_impression_share_barchart(
      self,
      geos: Geos = 1,
      n_channels: int | None = None,
      ascending: bool = True,
  ) -> alt.Chart:
    """Plots a bar chart of the relative impression share per media channel.

    Args:
      geos: The geos to plot.
      n_channels: The number of channels to display by impression share. If
        None, displays all channels.
      ascending: Whether to sort the channels in ascending order by impression
        share.

    Returns:
      The bar chart of the relative impression share of each media channel.
    """
    return self._plot_barcharts(
        geos=geos,
        title_prefix=(
            'Relative scaled impression share of paid and organic media'
            ' channels'
        ),
        x_axis_title=eda_constants.IMPRESSION_SHARE_SCALED,
        y_axis_title=constants.CHANNEL,
        national_data_source=self.eda_engine.national_treatments_without_non_media_scaled_ds,
        geo_data_source=self.eda_engine.treatments_without_non_media_scaled_ds,
        processing_function=lambda data: _calculate_relative_shares(
            _process_stacked_ds(eda_engine_module.stack_variables(data))
        ),
        n_channels=n_channels,
        ascending=ascending,
    )

  def plot_kpi_boxplot(self, geos: Geos = 1) -> alt.Chart:
    """Plots the boxplot for KPI variation."""
    return self._plot_boxplots(
        geos=geos,
        title_prefix='Boxplots of scaled KPI',
        x_axis_title=constants.KPI,
        y_axis_title=constants.KPI_SCALED,
        national_data_source=self.eda_engine.national_kpi_scaled_da,
        geo_data_source=self.eda_engine.kpi_scaled_da,
        processing_function=lambda data: data.to_dataframe()
        .reset_index()
        .rename(columns={data.name: constants.VALUE})
        .assign(**{eda_constants.VARIABLE: constants.KPI}),
    )

  def _get_std_artifact(
      self, geos: Geos, variable: str
  ) -> eda_outcome.StandardDeviationArtifact:
    """Retrieves the StandardDeviationArtifact for a given variable and geos.

    Args:
      geos: The geos level to check (e.g., 'nationalize', or a specific number
        or sequence of geos).
      variable: The name of the variable to find the artifact for.

    Returns:
      The matching StandardDeviationArtifact.

    Raises:
      ValueError: If the StandardDeviationArtifact for the given variable
        cannot be found.
    """
    if self._meridian.is_national or geos == eda_constants.NATIONALIZE:
      outcome = self.national_stdev_check_outcome
    else:
      outcome = self.geo_stdev_check_outcome

    std_artifact = next(
        (a for a in outcome.analysis_artifacts if a.variable == variable),
        None,
    )
    if std_artifact is None:
      raise ValueError(
          f'Could not find StandardDeviationArtifact for {variable}'
      )
    return std_artifact

  def plot_treatments_without_non_media_boxplot(
      self, geos: Geos = 1, max_vars: int | None = None
  ) -> alt.Chart:
    """Plots the boxplot for treatments variation, excluding non-media."""
    treatment_control_var = (
        constants.NATIONAL_TREATMENT_CONTROL_SCALED
        if self._meridian.is_national or geos == eda_constants.NATIONALIZE
        else constants.TREATMENT_CONTROL_SCALED
    )
    std_artifact = self._get_std_artifact(geos, treatment_control_var)
    sorting_config = BoxplotSortingConfig(
        std_artifact=std_artifact, n_vars=max_vars
    )

    return self._plot_boxplots(
        geos=geos,
        title_prefix='Boxplots of paid and organic scaled impressions',
        x_axis_title=eda_constants.VARIABLE,
        y_axis_title=eda_constants.MEDIA_IMPRESSIONS_SCALED,
        national_data_source=self.eda_engine.national_treatments_without_non_media_scaled_ds,
        geo_data_source=self.eda_engine.treatments_without_non_media_scaled_ds,
        processing_function=lambda data: _process_stacked_ds(
            eda_engine_module.stack_variables(data)  # pyrefly: ignore[bad-argument-type]
        ),
        sorting_config=sorting_config,
    )

  def plot_controls_and_non_media_boxplot(
      self, geos: Geos = 1, max_vars: int | None = None
  ) -> alt.Chart:
    """Plots the boxplots for controls and non-media treatments variation."""
    treatment_control_var = (
        constants.NATIONAL_TREATMENT_CONTROL_SCALED
        if self._meridian.is_national or geos == eda_constants.NATIONALIZE
        else constants.TREATMENT_CONTROL_SCALED
    )
    std_artifact = self._get_std_artifact(geos, treatment_control_var)
    sorting_config = BoxplotSortingConfig(
        std_artifact=std_artifact, n_vars=max_vars
    )

    return self._plot_boxplots(
        geos=geos,
        title_prefix='Boxplots of scaled controls and non-media treatments',
        x_axis_title=eda_constants.VARIABLE,
        y_axis_title=f'{constants.CONTROLS_SCALED}/{constants.NON_MEDIA_TREATMENTS_SCALED}',
        national_data_source=self.eda_engine.national_controls_and_non_media_scaled_ds,
        geo_data_source=self.eda_engine.controls_and_non_media_scaled_ds,
        processing_function=lambda data: _process_stacked_ds(
            eda_engine_module.stack_variables(data)  # pyrefly: ignore[bad-argument-type]
        ),
        sorting_config=sorting_config,
    )

  def _plot_barcharts(
      self,
      geos: Geos,
      title_prefix: str,
      x_axis_title: str,
      y_axis_title: str,
      national_data_source: xr.Dataset | xr.DataArray,
      geo_data_source: xr.Dataset | xr.DataArray,
      processing_function: Callable[[xr.Dataset], pd.DataFrame],
      n_channels: int | None = None,
      ascending: bool = False,
      sort_abs: bool = False,
      bar_size: int = 40,
      width: int = 600,
      **encoding_options,
  ) -> alt.Chart:
    """Helper function for plotting bar charts.

    Args:
      geos: The geos to plot.
      title_prefix: The prefix of the title of the chart.
      x_axis_title: The title of the x-axis of the chart.
      y_axis_title: The title of the y-axis of the chart.
      national_data_source: The national data source to plot.
      geo_data_source: The geo data source to plot.
      processing_function: The function to process the data. Returns a pandas
        DataFrame that can be plotted in Altair.
      n_channels: The number of channels to display. If None, displays all
        channels.
      ascending: Whether to sort the channels in ascending order.
      sort_abs: Whether to sort the channels in absolute order.
      bar_size: The size of the bars in the chart.
      width: The width of the chart.
      **encoding_options: dict, Encodings to pass to the chart.

    Returns:
      Altair barchart of the provided data.
    """
    geos_to_plot = self._validate_and_get_geos_to_plot(geos)
    charts = []
    use_national_data = (
        self._meridian.is_national or geos == eda_constants.NATIONALIZE
    )
    if n_channels is None:
      channels_text = '(all channels)'
    else:
      order = 'lowest' if ascending else 'top'
      abs_text = ' by abs value' if sort_abs else ''
      channels_text = f'(display limit {n_channels} {order} channels{abs_text})'
    for geo_to_plot in geos_to_plot:
      title = f'{title_prefix} {channels_text} for {geo_to_plot}'

      data_to_process = (
          national_data_source
          if use_national_data
          else geo_data_source.sel(geo=geo_to_plot)
      )

      processed_data = processing_function(data_to_process).sort_values(  # pyrefly: ignore[bad-argument-type]
          eda_constants.VALUE,
          ascending=ascending,
          key=abs if sort_abs else None,
      )
      if n_channels is not None:
        plot_data = processed_data.head(n_channels).sort_values(
            eda_constants.VALUE,
            ascending=ascending,
        )  # reset the abs sorting after getting top/lowest n_channels
      else:
        plot_data = processed_data
      custom_encodings = encoding_options.get('encoding_kwargs')
      if custom_encodings:
        encodings = custom_encodings
      else:
        sort_order = plot_data[eda_constants.VARIABLE].tolist()
        encodings = {
            'x': alt.X(f'{eda_constants.VALUE}:Q', title=x_axis_title),
            'y': alt.Y(
                f'{eda_constants.VARIABLE}:N',
                sort=sort_order,
                title=y_axis_title,
                scale=alt.Scale(paddingInner=0.025),
            ),
            'color': alt.Color(
                f'{eda_constants.VARIABLE}:N',
                legend=alt.Legend(title=None, orient='bottom'),
            ),
        }

      charts.append((
          alt.Chart(plot_data)
          .mark_bar(size=bar_size)
          .encode(
              **encodings,
              tooltip=[
                  f'{eda_constants.VARIABLE}:N',
                  alt.Tooltip(
                      f'{eda_constants.VALUE}:Q',
                      title=x_axis_title,
                      format='.3f',
                  ),
              ],
          )
          .properties(
              title=title, width=width, height=min(len(plot_data) * 50, 600)
          )
      ))

    return _apply_chart_config(
        alt.vconcat(*charts).resolve_legend(color='independent')
    )

  def _plot_boxplots(
      self,
      geos: Geos,
      title_prefix: str,
      x_axis_title: str,
      y_axis_title: str,
      national_data_source: xr.DataArray | xr.Dataset | None,
      geo_data_source: xr.DataArray | xr.Dataset | None,
      processing_function: Callable[[xr.DataArray | xr.Dataset], pd.DataFrame],
      sorting_config: BoxplotSortingConfig | None = None,
  ) -> alt.Chart:
    """Helper function for plotting boxplots."""
    geos_to_plot = self._validate_and_get_geos_to_plot(geos)

    use_national_data = (
        self._meridian.is_national or geos == eda_constants.NATIONALIZE
    )
    data_source = national_data_source if use_national_data else geo_data_source
    if data_source is None:
      raise ValueError(
          'There is no data to plot! Make sure your InputData contains the'
          ' component you are trying to plot.'
      )
    # When plotting Datasets, there are channel types in the coords/dims
    # (e.g. media, organic media, rf, etc.) In these cases, we want to assign
    # channels different colors based on their respective channel types.
    # Note: there may be DataArrays which contain multiple channel types
    # (e.g. scaled_reach_da contains paid and organic values), but the current
    # implementation will not color them differently. Currently, we don't plot
    # any DataArrays which contain multiple channel types, but in the future if
    # we do, we will need to make that distinction here.
    channel_to_type = {}
    if isinstance(data_source, xr.Dataset):
      for channel_type, coord in data_source.coords.items():
        if channel_type not in (constants.GEO, constants.TIME):
          for channel_name in coord.values:
            channel_to_type[channel_name] = channel_type

    # Optimization: Slice the xarray Dataset by geos_to_plot before converting
    # to DataFrame, which can be slow for models with many geos.
    if not use_national_data:
      # sel requires a list if multiple values are provided
      data_to_process = data_source.sel(geo=list(geos_to_plot))
    else:
      data_to_process = data_source

    plot_data_df = processing_function(data_to_process)
    unique_variables = plot_data_df[eda_constants.VARIABLE].unique()

    if sorting_config and len(unique_variables) > 1:
      sorting_metrics = _prepare_boxplot_sorting_metrics(
          sorting_config, unique_variables
      )
      n_vars = sorting_config.n_vars
    else:
      sorting_metrics = None
      n_vars = None

    charts = []
    for geo_to_plot in geos_to_plot:
      title = f'{title_prefix} for {geo_to_plot}'

      if sorting_metrics is not None:
        sort_order = _get_sorted_variables(sorting_metrics, geo_to_plot, n_vars)
      else:
        sort_order = unique_variables

      sorting_mask = plot_data_df[eda_constants.VARIABLE].isin(sort_order)
      if use_national_data:
        geo_mask = pd.Series(True, index=plot_data_df.index)
      else:
        geo_mask = plot_data_df[constants.GEO] == geo_to_plot
      plot_data = plot_data_df[sorting_mask & geo_mask]

      color_variable = eda_constants.VARIABLE
      color_scale = None

      if channel_to_type:
        plot_data[constants.CHANNEL] = plot_data[eda_constants.VARIABLE].map(
            channel_to_type
        )
        present_categories = plot_data[constants.CHANNEL].unique()

        if all(
            channel_type in eda_constants.CHANNEL_TYPE_TO_COLOR
            for channel_type in present_categories
        ):
          color_variable = constants.CHANNEL
          color_scale = alt.Scale(
              domain=present_categories,
              range=[
                  eda_constants.CHANNEL_TYPE_TO_COLOR[channel_type]
                  for channel_type in present_categories
              ],
          )
      color_legend = alt.Legend(
          title=None, orient='bottom', symbolType='square'
      )
      if color_scale is not None:
        color_encoding = alt.Color(
            f'{color_variable}:N',
            scale=color_scale,
            legend=color_legend,
        )
      else:
        color_encoding = alt.Color(
            f'{color_variable}:N',
            legend=color_legend,
        )
      charts.append((
          alt.Chart(plot_data)
          .mark_boxplot(ticks={'size': 20}, size=40, extent=1.5)
          .encode(
              x=alt.X(
                  f'{eda_constants.VARIABLE}:N',
                  title=x_axis_title,
                  sort=sort_order,
                  scale=alt.Scale(paddingInner=0.02),
              ),
              y=alt.Y(
                  f'{eda_constants.VALUE}:Q',
                  title=y_axis_title,
                  scale=alt.Scale(zero=True),
              ),
              color=color_encoding,
          )
          .properties(title=title, width=600, height=250)
      ))

    return _apply_chart_config(
        alt.vconcat(*charts).resolve_legend(color='independent')
    )

  def plot_pairwise_correlation(
      self, geos: Geos = 1, max_vars: int | None = None
  ) -> alt.Chart:
    """Plots the Pairwise Correlation data.

    Args:
      geos: Defines which geos to plot. - int: The number of top geos to plot,
        ranked by population. - Sequence[str]: A specific sequence of geo names
        to plot. - 'nationalize': Aggregates all geos into a single national
        view. Defaults to 1 (plotting the top geo). If the data is already at a
        national level, this parameter is ignored and a national plot is
        generated.
      max_vars: The maximum number of variables to include in the correlation
        heatmap. Variables are selected based on their absolute correlation
        values. If None, all variables are included.

    Returns:
      Altair chart(s) of the Pairwise Correlation data.

    Raises:
      ValueError: If `max_vars` is less than or equal to 1.
    """
    if max_vars is not None and max_vars <= 1:
      raise ValueError('max_vars must be greater than 1.')

    geos_to_plot = self._validate_and_get_geos_to_plot(geos)
    is_national = self._meridian.is_national
    nationalize_geos = geos == eda_constants.NATIONALIZE
    use_national_data = is_national or nationalize_geos
    if use_national_data:
      [pairwise_corr_artifact] = (
          self.national_pairwise_correlation_check_outcome.get_national_artifacts()
      )
    else:
      [pairwise_corr_artifact] = (
          self.geo_pairwise_correlation_check_outcome.get_geo_artifacts()
      )
    full_corr_matrix = pairwise_corr_artifact.corr_matrix

    all_variables = list(
        full_corr_matrix.coords[eda_constants.VARIABLE_1].values
    )

    df_full = (
        # We need to get the upper triangle of the correlation matrix so that
        # variable 1 is on the x-axis and variable 2 is on the y-axis.
        eda_engine_module.get_triangle_corr_mat(full_corr_matrix, lower=False)
        .to_dataframe(name=eda_constants.CORRELATION)
        .dropna()
        .reset_index()
        .assign(abs_corr=lambda x: x[eda_constants.CORRELATION].abs())
    )

    charts = []
    for geo_to_plot in geos_to_plot:
      plot_data, unique_variables = _get_plot_data_for_heatmap(
          df_full
          if use_national_data
          else df_full[df_full[constants.GEO] == geo_to_plot],
          all_variables,
          max_vars,
      )

      n_total_vars = len(all_variables)
      if len(unique_variables) < n_total_vars:
        title = (
            f'Top {len(unique_variables)} variables by pairwise correlations'
            f' among treatments and controls for {geo_to_plot}'
        )
      else:
        title = (
            'Pairwise correlations among all treatments and controls for'
            f' {geo_to_plot}'
        )

      charts.append(self._plot_2d_heatmap(plot_data, title, unique_variables))
    return _apply_chart_config(
        alt.vconcat(*charts).resolve_legend(color='independent')
    )

  def _plot_2d_heatmap(
      self, data: pd.DataFrame, title: str, unique_variables: Sequence[str]
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
        .properties(title=formatter.custom_title_params(title))
    )

    # Heatmap layer (rectangles)
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            f'{eda_constants.CORRELATION}:Q',
            scale=eda_constants.PAIRWISE_CORR_COLOR_SCALE,
            legend=alt.Legend(title=eda_constants.CORRELATION_LEGEND_TITLE),
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

    return (heatmap + text).properties(
        title=formatter.custom_title_params(title), width=550, height=450
    )

  def plot_population_raw_media_correlation(self) -> alt.Chart:
    """Plots Spearman correlation between population and raw media units.

    Returns:
      A bar chart showing the correlation.

    Raises:
      eda_engine_module.GeoLevelCheckOnNationalModelError: If the Meridian model
      is national.
    """

    if self._meridian.is_national:
      raise eda_engine_module.GeoLevelCheckOnNationalModelError(
          'Population raw media correlation is not supported for national'
          ' models.'
      )

    [artifact] = (
        self._dataset_level_population_raw_media_correlation_check_outcome.get_overall_artifacts()
    )

    return self._plot_barcharts(
        geos=eda_constants.NATIONALIZE,
        title_prefix=(
            'Correlation between population and raw paid and organic media'
            ' variables'
        ),
        x_axis_title=constants.CHANNEL,
        y_axis_title=eda_constants.CORRELATION,
        national_data_source=artifact.correlation_ds,
        geo_data_source=artifact.correlation_ds,
        processing_function=lambda data: _process_stacked_ds(
            eda_engine_module.stack_variables(data)
        ),
        n_channels=eda_constants.POPULATION_CORRELATION_BARCHART_LIMIT,
        ascending=True,
        bar_size=60,
        width=750,
        encoding_kwargs=eda_constants.POPULATION_RAW_MEDIA_CORRELATION_ENCODINGS,
    )

  def plot_population_treatment_correlation(self) -> alt.Chart:
    """Plots the Spearman correlation of population vs treatments or controls.

    Returns:
      An Altair chart showing the correlation.

    Raises:
      eda_engine_module.GeoLevelCheckOnNationalModelError: If the Meridian model
      is national.
    """
    if self._meridian.is_national:
      raise eda_engine_module.GeoLevelCheckOnNationalModelError(
          'Population treatment correlation is not supported for national'
          ' models.'
      )

    [artifact] = (
        self._dataset_level_population_treatment_correlation_check_outcome.get_overall_artifacts()
    )

    return self._plot_barcharts(
        geos=eda_constants.NATIONALIZE,
        title_prefix=(
            'Correlation between population and scaled treatment/scaled'
            ' controls'
        ),
        x_axis_title=constants.CHANNEL,
        y_axis_title=eda_constants.CORRELATION,
        national_data_source=artifact.correlation_ds,
        geo_data_source=artifact.correlation_ds,
        processing_function=lambda data: _process_stacked_ds(
            eda_engine_module.stack_variables(data)
        ),
        n_channels=eda_constants.POPULATION_CORRELATION_BARCHART_LIMIT,
        ascending=False,
        sort_abs=True,
        bar_size=60,
        width=750,
        encoding_kwargs=eda_constants.POPULATION_TREATMENT_CORRELATION_ENCODINGS,
    )

  def plot_prior_mean(self) -> alt.Chart:
    """Plots the prior mean of contribution by channel.

    Returns:
      A bar chart showing the prior mean of contribution by channel.
    """
    [artifact] = self._dataset_level_prior_check_outcome.get_overall_artifacts()

    return self._plot_barcharts(
        geos=eda_constants.NATIONALIZE,
        title_prefix='Prior mean of contribution by channel',
        x_axis_title=constants.CHANNEL,
        y_axis_title=eda_constants.PRIOR_CONTRIBUTION,
        national_data_source=artifact.mean_prior_contribution_da,
        geo_data_source=artifact.mean_prior_contribution_da,
        processing_function=lambda data: _process_stacked_ds(
            data.rename({constants.CHANNEL: eda_constants.VARIABLE})  # pyrefly: ignore[bad-argument-type]
        ),
        n_channels=eda_constants.PRIOR_MEAN_BARCHART_LIMIT,
        bar_size=60,
        width=750,
        encoding_kwargs=eda_constants.PRIOR_MEAN_ENCODINGS,
    )

  def _generate_summary_card(
      self, card_severity_counts: dict[str, dict[eda_outcome.EDASeverity, int]]
  ) -> str:
    """Creates the HTML snippet for the Summary section."""
    rows = []

    for (
        category,
        metrics,
    ) in card_severity_counts.items():
      n_fails = metrics[eda_outcome.EDASeverity.FAIL]
      n_reviews = metrics[eda_outcome.EDASeverity.REVIEW]
      finding_tags = []
      if n_fails > 0:
        finding_tags.append(
            formatter.create_finding_html(
                self._template_env, f'{n_fails} fail(s)', 'fail'
            )
        )
      if n_reviews > 0:
        finding_tags.append(
            formatter.create_finding_html(
                self._template_env, f'{n_reviews} review(s)', 'review'
            )
        )
      finding_tag = (
          ''.join(finding_tags)
          if finding_tags
          else formatter.create_finding_html(self._template_env, 'Info', 'info')
      )

      rows.append({
          eda_constants.CATEGORY: category,
          eda_constants.FINDING: finding_tag,
          eda_constants.RECOMMENDED_NEXT_STEP: (
              eda_constants.CATEGORY_TO_MESSAGE_BY_STATUS[category][
                  bool(n_fails or n_reviews)
              ]
          ),
      })

    formatted_table = pd.DataFrame(rows)
    return formatter.create_card_html(
        self._template_env,
        formatter.CardSpec(
            id=eda_constants.SUMMARY_CARD_ID,
            title=eda_constants.SUMMARY_CARD_TITLE,
        ),
        chart_specs=[
            formatter.TableSpec(
                id=eda_constants.SUMMARY_TABLE_ID,
                title='',
                column_headers=formatted_table.columns.tolist(),
                row_values=formatted_table.values.tolist(),
            )
        ],
    )

  def _generate_spend_and_media_unit_card(
      self,
  ) -> tuple[str, dict[eda_outcome.EDASeverity, int]]:
    """Creates the HTML snippet for the Spend and Media Unit section."""
    all_charts = []
    total_severity_counts = _initialize_severity_counts()

    relative_spend_share_chart = formatter.ChartSpec(
        id=eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
        chart_json=self.plot_relative_spend_share_barchart(
            eda_constants.NATIONALIZE, eda_constants.DISPLAY_LIMIT
        ).to_json(),
        infos=[eda_constants.RELATIVE_SPEND_SHARE_INFO],
    )
    all_charts.append(relative_spend_share_chart)

    callout_generation_functions = [
        self._generate_data_adequacy_callouts,
        self._generate_spend_and_media_unit_callouts,
    ]

    for func in callout_generation_functions:
      callouts, card_severity_counts = func()
      all_charts.extend(callouts)
      for severity, count in card_severity_counts.items():
        total_severity_counts[severity] += count

    return (
        formatter.create_card_html(
            self._template_env,
            formatter.CardSpec(
                id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
                title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
            ),
            chart_specs=all_charts,
        ),
        total_severity_counts,
    )

  def _generate_data_adequacy_callouts(
      self,
  ) -> tuple[
      Sequence[formatter.ChartSpec | formatter.TableSpec],
      dict[eda_outcome.EDASeverity, int],
  ]:
    """Generates the data adequacy callout table spec and severity counts."""
    callouts = []
    card_severity_counts = _initialize_severity_counts()
    data_param_outcome = self.eda_engine.check_data_param_ratio()
    finding = _get_finding(
        data_param_outcome,
        eda_outcome.FindingCause.NONE,
        eda_outcome.EDASeverity.INFO,
    )

    if finding:
      if finding.severity in card_severity_counts:
        card_severity_counts[finding.severity] += 1
      explanation = _format_explanation_for_html(finding.explanation)
      artifact = finding.associated_artifact

      row_values = []
      if isinstance(artifact, eda_outcome.DataParameterRatioArtifact):
        row_values = [
            ['Ratio', f'{artifact.ratio:.2f}'],
            ['n_geos', str(artifact.n_geos)],
            ['n_times', str(artifact.n_times)],
            ['n_knots', str(artifact.n_knots)],
            ['n_controls', str(artifact.n_controls)],
            ['n_treatments', str(artifact.n_treatments)],
            ['n_parameters', str(artifact.n_parameters)],
            ['n_data_points', str(artifact.n_data_points)],
        ]

      callouts.append(
          formatter.TableSpec(
              id=eda_constants.DATA_ADEQUACY_TABLE_ID,
              title='',
              column_headers=['Metric', 'Value'] if row_values else [],
              row_values=row_values,
              infos=[explanation],
          )
      )

    return callouts, card_severity_counts

  def _generate_spend_and_media_unit_callouts(
      self,
  ) -> tuple[
      Sequence[formatter.ChartSpec | formatter.TableSpec],
      dict[eda_outcome.EDASeverity, int],
  ]:
    """Generates error/warning callouts for the Spend and Media Unit card."""
    callouts = []
    card_severity_counts = _initialize_severity_counts()
    inconsistency_finding = _get_finding(
        self._dataset_level_cost_per_media_unit_check_outcome,
        eda_outcome.FindingCause.INCONSISTENT_DATA,
        eda_outcome.EDASeverity.REVIEW,
    )
    outlier_finding = _get_finding(
        self._dataset_level_cost_per_media_unit_check_outcome,
        eda_outcome.FindingCause.OUTLIER,
        eda_outcome.EDASeverity.REVIEW,
    )

    outlier_table_spec, outlier_channels = (
        self._create_cost_per_media_unit_callout(outlier_finding)
    )
    inconsistency_table_spec, inconsistency_channels = (
        self._create_cost_per_media_unit_callout(inconsistency_finding)
    )

    display_channels = {**outlier_channels, **inconsistency_channels}

    if display_channels:
      callouts.append(
          formatter.ChartSpec(
              id=eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              chart_json=self.plot_cost_per_media_unit_time_series(
                  eda_constants.NATIONALIZE,
                  list(display_channels)[: eda_constants.TIME_SERIES_LIMIT],
              ).to_json(),
              infos=[eda_constants.SPEND_PER_MEDIA_UNIT_INFO],
          )
      )
    if inconsistency_table_spec:
      callouts.append(inconsistency_table_spec)
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1
    if outlier_table_spec:
      callouts.append(outlier_table_spec)
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1

    return callouts, card_severity_counts

  def _create_cost_per_media_unit_callout(
      self,
      finding: eda_outcome.EDAFinding | None,
  ) -> tuple[formatter.TableSpec | None, dict[str, bool]]:
    """Creates a cost per media unit callout table spec from the finding.

    Args:
      finding: The EDAFinding to create the callout from. Can be None if no
        finding of the specified type exists.

    Returns:
      A tuple containing:
        - A formatter.TableSpec if a valid finding is provided, otherwise None.
        - A dictionary of channels found in the artifact (mapping channel name
        to
          True), preserving the order they appear in the table.
    """
    if not finding or not isinstance(
        finding.associated_artifact,
        eda_outcome.CostPerMediaUnitArtifact,
    ):
      return None, {}

    if finding.finding_cause == eda_outcome.FindingCause.OUTLIER:
      display_table = finding.associated_artifact.outlier_df
      formatted_table = _format_display_table(
          display_table,
          [
              eda_constants.ABS_COST_PER_MEDIA_UNIT,
              constants.CHANNEL,
          ],
          [False, True],
      )
      table_id = eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID
      to_review_prefix = 'outliers'
    else:
      display_table = (
          finding.associated_artifact.cost_media_unit_inconsistency_df
      )
      formatted_table = _format_display_table(
          display_table,
          [constants.MEDIA_UNITS, constants.SPEND, constants.CHANNEL],
          [False, False, True],
      )
      table_id = eda_constants.INCONSISTENT_DATA_TABLE_ID
      to_review_prefix = 'inconsistencies'

    display_limit_message = _create_display_limit_message(
        display_table,
        'EDAEngine.check_cost_per_media_unit()',
        to_review_prefix,
        n_channels=display_table.index.unique(level=constants.CHANNEL).size,
        n_times=display_table.index.unique(level=constants.TIME).size,
        n_geos=display_table.index.unique(level=constants.GEO).size
        if not self._meridian.is_national
        else None,
    )

    # Use dict to store the sorted channels since set does not guarantee order.
    found_channels = {
        channel: True for channel in formatted_table[constants.CHANNEL].unique()
    }

    return (
        formatter.TableSpec(
            id=table_id,
            title='',
            column_headers=formatter.format_col_names(
                formatted_table.columns.tolist()
            ),
            row_values=formatted_table.values.tolist(),
            warnings=[
                _format_explanation_for_html(finding.explanation)
                + f' {display_limit_message}'
            ],
        ),
        found_channels,
    )

  def _generate_response_variables_card(
      self,
  ) -> tuple[str, dict[eda_outcome.EDASeverity, int]]:
    """Creates HTML for Individual Explanatory/Response Variables section."""
    all_charts = []
    all_charts.append(
        formatter.ChartSpec(
            id=eda_constants.TREATMENTS_CHART_ID,
            chart_json=self.plot_treatments_without_non_media_boxplot(
                geos=eda_constants.NATIONALIZE,
                max_vars=eda_constants.BOXPLOT_VAR_NUMBER_LIMIT,
            ).to_json(),
            infos=[eda_constants.VARIABILITY_PLOT_INFO],
        )
    )

    if self.eda_engine.national_controls_and_non_media_scaled_ds is not None:
      all_charts.append(
          formatter.ChartSpec(
              id=eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              chart_json=self.plot_controls_and_non_media_boxplot(
                  geos=eda_constants.NATIONALIZE,
                  max_vars=eda_constants.BOXPLOT_VAR_NUMBER_LIMIT,
              ).to_json(),
          )
      )

    callouts, card_severity_counts = (
        self._generate_response_variables_callouts()
    )
    all_charts.extend(callouts)

    return (
        formatter.create_card_html(
            self._template_env,
            formatter.CardSpec(
                id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
                title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
            ),
            chart_specs=all_charts,
        ),
        card_severity_counts,
    )

  def _generate_response_variables_callouts(
      self,
  ) -> tuple[
      Sequence[formatter.ChartSpec | formatter.TableSpec],
      dict[eda_outcome.EDASeverity, int],
  ]:
    """Generates callouts for the Individual Explanatory/Response Variables."""
    callouts = []
    card_severity_counts = _initialize_severity_counts()
    kpi_error_finding = _get_finding(
        self._overall_kpi_invariability_check_outcome,
        eda_outcome.FindingCause.VARIABILITY,
        eda_outcome.EDASeverity.FAIL,
    )
    kpi_attention_finding = _get_finding(
        self._dataset_level_stdev_check_outcome,
        eda_outcome.FindingCause.VARIABILITY,
        eda_outcome.EDASeverity.REVIEW,
        constants.NATIONAL_KPI_SCALED
        if self._meridian.is_national
        else constants.KPI_SCALED,
    )
    treatment_control_var = (
        constants.NATIONAL_TREATMENT_CONTROL_SCALED
        if self._meridian.is_national
        else constants.TREATMENT_CONTROL_SCALED
    )
    treatment_control_variability_finding = _get_finding(
        self._dataset_level_stdev_check_outcome,
        eda_outcome.FindingCause.VARIABILITY,
        eda_outcome.EDASeverity.REVIEW,
        treatment_control_var,
    )
    treatment_control_outlier_finding = _get_finding(
        self._dataset_level_stdev_check_outcome,
        eda_outcome.FindingCause.OUTLIER,
        eda_outcome.EDASeverity.REVIEW,
        treatment_control_var,
    )

    if kpi_error_finding:
      kpi_chart_errors = [
          _format_explanation_for_html(kpi_error_finding.explanation)
      ]
    else:
      kpi_chart_errors = None

    if kpi_attention_finding and not kpi_error_finding:
      kpi_chart_warnings = [
          _format_explanation_for_html(kpi_attention_finding.explanation)
      ]
    else:
      kpi_chart_warnings = None

    kpi_chart = formatter.ChartSpec(
        id=eda_constants.KPI_CHART_ID,
        chart_json=self.plot_kpi_boxplot(eda_constants.NATIONALIZE).to_json(),
        errors=kpi_chart_errors,
        warnings=kpi_chart_warnings,
    )
    callouts.append(kpi_chart)

    variability_callout = self._create_stdev_callout(
        treatment_control_variability_finding
    )
    if variability_callout:
      callouts.append(variability_callout)
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1

    outlier_callout = self._create_stdev_callout(
        treatment_control_outlier_finding
    )
    if outlier_callout:
      callouts.append(outlier_callout)
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1
    if kpi_error_finding:
      card_severity_counts[eda_outcome.EDASeverity.FAIL] += 1
    if kpi_attention_finding and not kpi_error_finding:
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1
    return callouts, card_severity_counts

  def _create_stdev_callout(
      self, finding: eda_outcome.EDAFinding | None
  ) -> formatter.TableSpec | None:
    """Creates a standard deviation callout table spec from the finding."""
    if not finding or not isinstance(
        finding.associated_artifact, eda_outcome.StandardDeviationArtifact
    ):
      return None

    if finding.finding_cause == eda_outcome.FindingCause.VARIABILITY:
      std_spec = self.eda_engine.spec.std_spec
      finding_level = finding.associated_artifact.level
      std_threshold = (
          std_spec.national_std_threshold
          if finding_level == eda_outcome.AnalysisLevel.NATIONAL
          else std_spec.geo_std_threshold
      )
      display_table = finding.associated_artifact.std_ds.to_dataframe().loc[
          lambda df: df[eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME]
          < std_threshold
      ]
      to_review_prefix = 'zero variability'
      table_id = eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID
      n_times = None
      sort_columns = [
          eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME,
          eda_constants.STD_WITH_OUTLIERS_VAR_NAME,
          eda_constants.VARIABLE,
      ]
      ascending_orders = [True, True, True]
    else:
      display_table = finding.associated_artifact.outlier_df
      to_review_prefix = 'outliers'
      table_id = eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID
      n_times = display_table.index.unique(level=constants.TIME).size
      sort_columns = [
          eda_constants.ABS_OUTLIERS_COL_NAME,
          eda_constants.VARIABLE,
      ]
      ascending_orders = [False, True]

    formatted_table = _format_display_table(
        display_table,
        sort_columns,
        ascending_orders,
    )

    display_limit_message = _create_display_limit_message(
        display_table,
        'EDAEngine.check_std()',
        to_review_prefix,
        n_channels=display_table.index.unique(
            level=eda_constants.VARIABLE
        ).size,
        n_times=n_times,
        n_geos=display_table.index.unique(level=constants.GEO).size
        if not self._meridian.is_national
        else None,
    )

    return formatter.TableSpec(
        id=table_id,
        title='',
        column_headers=formatter.format_col_names(
            formatted_table.columns.tolist()
        ),
        row_values=formatted_table.values.tolist(),
        warnings=[
            f'{_format_explanation_for_html(finding.explanation)}'
            f'{display_limit_message}'
        ],
    )

  def _generate_population_scaling_card(
      self,
  ) -> tuple[str | None, dict[eda_outcome.EDASeverity, int]]:
    """Creates HTML for Population Scaling of Explanatory Variables section."""
    if self._meridian.is_national:
      return None, _initialize_severity_counts()

    all_charts = []

    population_raw_media_chart = formatter.ChartSpec(
        id=eda_constants.POPULATION_RAW_MEDIA_CHART_ID,
        chart_json=self.plot_population_raw_media_correlation().to_json(),
        infos=[eda_constants.POPULATION_CORRELATION_RAW_MEDIA_INFO],
    )
    all_charts.append(population_raw_media_chart)

    finding = _get_finding(
        self._dataset_level_population_treatment_correlation_check_outcome,
        eda_outcome.FindingCause.NONE,
        eda_outcome.EDASeverity.INFO,
    )
    if not finding:
      # This should never happen as the check is always performed and should
      # always produce a finding of severity INFO.
      raise ValueError(
          'No finding found for dataset level population treatment correlation'
          ' check.'
      )
    explanation = _format_explanation_for_html(finding.explanation)

    population_treatment_chart = formatter.ChartSpec(
        id=eda_constants.POPULATION_TREATMENT_CHART_ID,
        chart_json=self.plot_population_treatment_correlation().to_json(),
        infos=[explanation],
    )
    all_charts.append(population_treatment_chart)

    return (
        formatter.create_card_html(
            self._template_env,
            formatter.CardSpec(
                id=eda_constants.POPULATION_SCALING_CARD_ID,
                title=eda_constants.POPULATION_SCALING_CARD_TITLE,
            ),
            chart_specs=all_charts,
        ),
        _initialize_severity_counts(),
    )

  def _generate_relationship_among_variables_card(
      self,
  ) -> tuple[str, dict[eda_outcome.EDASeverity, int]]:
    """Creates the HTML snippet for the Relationship Among Variables section."""
    all_charts = []

    pairwise_correlation_chart = formatter.ChartSpec(
        id=eda_constants.PAIRWISE_CORRELATION_CHART_ID,
        chart_json=self.plot_pairwise_correlation(
            eda_constants.NATIONALIZE,
            max_vars=eda_constants.PAIRWISE_CORRELATION_VAR_NUMBER_LIMIT,
        ).to_json(),
        infos=[eda_constants.PAIRWISE_CORRELATION_CHECK_INFO],
    )
    all_charts.append(pairwise_correlation_chart)

    callouts, card_severity_counts = (
        self._generate_relationship_among_variables_callouts()
    )
    all_charts.extend(callouts)

    if not self._meridian.is_national:
      all_charts.extend(self._generate_r_squared_tables())

    return (
        formatter.create_card_html(
            self._template_env,
            formatter.CardSpec(
                id=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID,
                title=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
            ),
            chart_specs=all_charts,
        ),
        card_severity_counts,
    )

  def _combine_extreme_vif_with_extreme_corr_pairs(
      self, extreme_vif_df: pd.DataFrame, severity: eda_outcome.EDASeverity
  ) -> pd.DataFrame:
    """Combines the extreme VIF table with extreme correlation pairs table."""
    display_table = extreme_vif_df.assign(
        **{eda_constants.EXTREME_CORRELATION_WITH: ''}
    )
    finding = _get_finding(
        self._dataset_level_pairwise_correlation_check_outcome,
        eda_outcome.FindingCause.MULTICOLLINEARITY,
        severity,
    )
    if not finding or not isinstance(
        finding.associated_artifact, eda_outcome.PairwiseCorrArtifact
    ):
      return display_table

    extreme_corr_pairs = finding.associated_artifact.extreme_corr_var_pairs

    corr_partners = {}
    for corr_var_pair in extreme_corr_pairs.index:
      *geo, v1, v2 = corr_var_pair
      key1 = (geo[0], v1) if geo else v1
      key2 = (geo[0], v2) if geo else v2

      if key1 in display_table.index:
        corr_partners.setdefault(key1, []).append(v2)
      if key2 in display_table.index:
        corr_partners.setdefault(key2, []).append(v1)

    for key, partners in corr_partners.items():
      display_table.at[key, eda_constants.EXTREME_CORRELATION_WITH] = ', '.join(
          partners
      )
    return display_table

  def _generate_relationship_among_variables_callouts(
      self,
  ) -> tuple[Sequence[formatter.TableSpec], dict[eda_outcome.EDASeverity, int]]:
    """Generates callouts for the Relationship Among Variables card."""
    callouts = []
    card_severity_counts = _initialize_severity_counts()
    failed_variables = set()
    vif_outcome = self._dataset_level_vif_check_outcome
    error_finding = _get_finding(
        vif_outcome,
        eda_outcome.FindingCause.MULTICOLLINEARITY,
        eda_outcome.EDASeverity.FAIL,
    )
    attention_finding = _get_finding(
        vif_outcome,
        eda_outcome.FindingCause.MULTICOLLINEARITY,
        eda_outcome.EDASeverity.REVIEW,
    )

    error_callout = self._create_vif_callout(
        error_finding,
        failed_variables,
    )
    if error_callout:
      callouts.append(error_callout)
      card_severity_counts[eda_outcome.EDASeverity.FAIL] += 1

    attention_callout = self._create_vif_callout(
        attention_finding,
        failed_variables,
    )
    if attention_callout:
      callouts.append(attention_callout)
      card_severity_counts[eda_outcome.EDASeverity.REVIEW] += 1

    return callouts, card_severity_counts

  def _create_vif_callout(
      self,
      finding: eda_outcome.EDAFinding | None,
      failed_variables: set[str],
  ) -> formatter.TableSpec | None:
    """Creates a VIF callout table spec from a finding."""
    if not finding or not isinstance(
        finding.associated_artifact, eda_outcome.VIFArtifact
    ):
      return None

    severity = finding.severity
    extreme_vif_df = finding.associated_artifact.outlier_df
    vif_spec = self.eda_engine.spec.vif_spec

    if severity == eda_outcome.EDASeverity.FAIL:
      failed_variables.update(extreme_vif_df.index)
      threshold = (
          vif_spec.national_threshold
          if self._meridian.is_national
          else vif_spec.overall_threshold
      )
      aggregation = (
          eda_constants.TIME_AGGREGATION
          if self._meridian.is_national
          else eda_constants.TIME_AND_GEO_AGGREGATION
      )
      message_template = eda_constants.MULTICOLLINEARITY_FAIL
      message_kwargs = {'threshold': threshold, 'aggregation': aggregation}
      table_id = eda_constants.EXTREME_VIF_FAIL_TABLE_ID
      df_to_process = extreme_vif_df
    else:
      current_vars = extreme_vif_df.index.get_level_values(
          eda_constants.VARIABLE
      )
      df_to_process = extreme_vif_df[~current_vars.isin(failed_variables)]
      message_template = eda_constants.MULTICOLLINEARITY_REVIEW
      message_kwargs = {'threshold': vif_spec.geo_threshold}
      table_id = eda_constants.EXTREME_VIF_REVIEW_TABLE_ID

    display_table = self._combine_extreme_vif_with_extreme_corr_pairs(
        df_to_process, severity
    )

    if severity == eda_outcome.EDASeverity.FAIL:
      n_geos = None
    else:
      n_geos = display_table.index.unique(level=constants.GEO).size

    formatted_table = _format_display_table(
        display_table,
        [eda_constants.VIF_COL_NAME, eda_constants.VARIABLE],
        [False, True],
    )

    display_limit_message = _create_display_limit_message(
        display_table,
        'EDAEngine.check_vif()',
        'extreme VIF',
        n_channels=display_table.index.unique(
            level=eda_constants.VARIABLE
        ).size,
        n_geos=n_geos,
    )
    message_kwargs['additional_info'] = display_limit_message  # pyrefly: ignore[unsupported-operation]
    final_message = message_template.format(**message_kwargs)

    return formatter.TableSpec(
        id=table_id,
        title='',
        column_headers=formatter.format_col_names(
            formatted_table.columns.tolist()
        ),
        row_values=formatted_table.values.tolist(),
        errors=(
            [final_message]
            if severity == eda_outcome.EDASeverity.FAIL
            else None
        ),
        warnings=(
            [final_message]
            if severity == eda_outcome.EDASeverity.REVIEW
            else None
        ),
    )

  def _generate_r_squared_tables(self) -> Sequence[formatter.TableSpec]:
    """Generates R-squared tables for the Relationship Among Variables card."""
    outcome = self.eda_engine.check_variable_geo_time_collinearity()
    [artifact] = outcome.get_overall_artifacts()
    format_df = lambda df: [[col[0], f'{col[1]:.3f}'] for col in df.values]

    tables = []
    configs = [
        (
            eda_constants.RSQUARED_TIME,
            'R-squared time',
            eda_constants.R_SQUARED_TIME_TABLE_ID,
            eda_constants.R_SQUARED_TIME_INFO,
        ),
        (
            eda_constants.RSQUARED_GEO,
            'R-squared geo',
            eda_constants.R_SQUARED_GEO_TABLE_ID,
            eda_constants.R_SQUARED_GEO_INFO,
        ),
    ]

    for rsquared_key, to_review_prefix, table_id, info in configs:
      display_table = artifact.rsquared_ds[rsquared_key].to_dataframe()
      df = (
          display_table.reset_index()
          .sort_values(by=rsquared_key, ascending=False)
          .head(eda_constants.DISPLAY_LIMIT)
      )
      display_limit_message = _create_display_limit_message(
          display_table,
          'EDAEngine.check_variable_geo_time_collinearity()',
          to_review_prefix,
          n_channels=len(display_table),
      )
      tables.append(
          formatter.TableSpec(
              id=table_id,
              title='',
              column_headers=formatter.format_col_names(
                  [constants.CHANNEL, constants.R_SQUARED]
              ),
              row_values=format_df(df),
              infos=[f'{info}{display_limit_message}'],
          )
      )
    return tables

  def _generate_prior_quality_table(self) -> formatter.TableSpec | None:
    """Generates a TableSpec summarizing prior quality indicators.

    Returns:
      A TableSpec containing the prior quality table, or None if unavailable.
    """
    outcome = self._dataset_level_prior_quality_check_outcome
    if outcome is None:
      return None

    try:
      artifacts = outcome.get_overall_artifacts()
    except ValueError:
      return None

    if not artifacts or not artifacts[0].prior_quality_data:
      return None
    quality_data = artifacts[0].prior_quality_data

    column_headers = [
        constants.CHANNEL,
        eda_constants.BASELINE_PRIOR_TYPE,
        eda_constants.PRIOR_BIMODALITY_STATISTIC,
        eda_constants.PRIOR_WIDTH_RATIO,
        eda_constants.INTERMEDIARY_AND_PARAMETERIZED_OVERLAP_PERCENTAGE,
        eda_constants.NEGATIVE_POINT_ESTIMATE_EXPERIMENTS,
    ]

    row_values = []
    bimodal_channels = []
    baseline_type_bimodal_channels = []
    width_channels = []
    overlap_channels = []
    neg_exp_channels = []

    for data in quality_data:
      bimodal_str = (
          eda_constants.NOT_AVAILABLE
          if data.bimodal_statistic is None
          else f'{data.bimodal_statistic:.4f}'
      )
      width_str = f'{data.prior_width_ratio:.2f}'
      overlap_str = f'{data.overlap_percentage * 100:.0f}%'
      neg_count_str = str(data.n_negative_experiments)

      if data.bimodal_statistic is None:
        baseline_type_bimodal_channels.append(data.channel_name)
      elif (
          data.bimodal_statistic > eda_constants.BIMODALITY_STATISTIC_THRESHOLD
      ):
        bimodal_channels.append(data.channel_name)

      if data.prior_width_ratio > eda_constants.HIGH_VARIANCE_THRESHOLD:
        width_channels.append(data.channel_name)
      if data.overlap_percentage < eda_constants.OVERLAP_PERCENTAGE_THRESHOLD:
        overlap_channels.append(data.channel_name)
      if data.n_negative_experiments > 0:
        neg_exp_channels.append(data.channel_name)

      row_values.append([
          data.channel_name,
          data.baseline_prior_type,
          bimodal_str,
          width_str,
          overlap_str,
          neg_count_str,
      ])

    warnings = []
    if bimodal_channels:
      channels_str = ', '.join(bimodal_channels)
      warnings.append(
          eda_constants.PRIOR_BIMODALITY_WARNING.format(channels=channels_str)
      )
    if baseline_type_bimodal_channels:
      channels_str = ', '.join(baseline_type_bimodal_channels)
      warnings.append(
          eda_constants.BASELINE_PRIOR_TYPE_BIMODALITY_WARNING.format(
              channels=channels_str
          )
      )
    if width_channels:
      channels_str = ', '.join(width_channels)
      warnings.append(
          eda_constants.UNINFORMATIVE_CALIBRATED_PRIOR_WARNING.format(
              channels=channels_str
          )
      )
    if overlap_channels:
      channels_str = ', '.join(overlap_channels)
      warnings.append(
          eda_constants.LOW_OVERLAP_PRIORS_WARNING.format(channels=channels_str)
      )
    if neg_exp_channels:
      channels_str = ', '.join(neg_exp_channels)
      warnings.append(
          eda_constants.NEGATIVE_POINT_ESTIMATE_EXPERIMENTS_WARNING.format(
              channels=channels_str
          )
      )

    return formatter.TableSpec(
        id=eda_constants.PRIOR_QUALITY_TABLE_ID,
        title='Prior Quality Summary',
        column_headers=column_headers,
        row_values=row_values,
        warnings=warnings if warnings else None,
        infos=[eda_constants.PRIOR_QUALITY_TABLE_INFO]
        if not warnings
        else None,
    )

  def _generate_prior_specifications_card(
      self,
  ) -> tuple[str, dict[eda_outcome.EDASeverity, int]]:
    """Creates the HTML snippet for the Prior Specifications section.

    Returns:
      A tuple containing:
        - A string of the HTML snippet for the Prior Specifications card.
        - A dictionary of severity counts for the card.
    """
    [artifact] = self._dataset_level_prior_check_outcome.get_overall_artifacts()
    prior_probability = artifact.prior_negative_baseline_prob

    prior_chart = self._generate_prior_mean_chart_spec(prior_probability)
    chart_specs: list[formatter.ChartSpec | formatter.TableSpec] = [prior_chart]

    card_severity_counts = _initialize_severity_counts()
    calibration_chart_spec = self._generate_prior_calibration_chart_spec()
    if calibration_chart_spec is not None:
      chart_specs.append(calibration_chart_spec)
    experiment_adjustments_chart_spec = (
        self._generate_experiment_adjustments_chart_spec()
    )
    if experiment_adjustments_chart_spec is not None:
      chart_specs.append(experiment_adjustments_chart_spec)
    prior_quality_table_spec = self._generate_prior_quality_table()
    if prior_quality_table_spec is not None:
      chart_specs.append(prior_quality_table_spec)
    outcome = self._dataset_level_prior_quality_check_outcome
    if outcome is not None:
      for finding in outcome.findings:
        if finding.severity in card_severity_counts:
          card_severity_counts[finding.severity] += 1

    return (
        formatter.create_card_html(
            self._template_env,
            formatter.CardSpec(
                id=eda_constants.PRIOR_SPECIFICATIONS_CARD_ID,
                title=eda_constants.PRIOR_SPECIFICATIONS_CARD_TITLE,
            ),
            chart_specs=chart_specs,
        ),
        card_severity_counts,
    )

  def _generate_prior_mean_chart_spec(
      self, prior_probability: float
  ) -> formatter.ChartSpec:
    """Generates the ChartSpec for the default prior mean chart."""
    return formatter.ChartSpec(
        id=eda_constants.PRIOR_CHART_ID,
        chart_json=self.plot_prior_mean().to_json(),
        infos=[
            f'{eda_constants.PRIOR_PROBABILITY_REPORT_INFO}Prior Probability of'
            f' negative baseline: {prior_probability}'
        ],
    )

  def _generate_prior_calibration_chart_spec(
      self,
  ) -> formatter.ChartSpec | None:
    """Generates the ChartSpec for the prior calibration comparison chart."""
    channel_limit = eda_constants.PRIOR_CALIBRATION_REPORT_CHANNEL_LIMIT
    exp_limit = eda_constants.PRIOR_CALIBRATION_REPORT_EXPERIMENT_LIMIT
    calibration_chart = self.plot_calibration(
        limit_channels=channel_limit,
        limit_experiments=exp_limit,
    )
    if calibration_chart is None:
      return None

    calibration_results = self._get_calibration_results_by_channel()
    calibrated_channels = list(calibration_results.keys())

    if len(calibrated_channels) > channel_limit:
      channels_text = f'your {channel_limit} highest-spend calibrated channels'
      banner_text = eda_constants.PRIOR_CALIBRATION_BANNER_TEXT_TEMPLATE.format(
          channels=channels_text
      ) + eda_constants.PRIOR_CALIBRATION_LIMIT_MESSAGE.format(
          limit=channel_limit
      )
    else:
      channels_text = 'all calibrated channels'
      banner_text = eda_constants.PRIOR_CALIBRATION_BANNER_TEXT_TEMPLATE.format(
          channels=channels_text
      )

    return formatter.ChartSpec(
        id=eda_constants.PRIOR_CALIBRATION_CHART_ID,
        chart_json=calibration_chart.to_json(),
        infos=[
            banner_text,
            eda_constants.PRIOR_CALIBRATION_EXPERIMENT_ORDER_INFO,
        ],
    )

  def _generate_experiment_adjustments_chart_spec(
      self,
  ) -> formatter.ChartSpec | None:
    """Generates the ChartSpec for the experiment adjustments chart."""
    channel_limit = eda_constants.EXPERIMENT_ADJUSTMENTS_REPORT_CHANNEL_LIMIT
    exp_limit = eda_constants.EXPERIMENT_ADJUSTMENTS_REPORT_EXPERIMENT_LIMIT
    adjustments_chart = self.plot_experiment_adjustments(
        limit_channels=channel_limit,
        limit_experiments=exp_limit,
    )
    if adjustments_chart is None:
      return None

    outcome = self.eda_engine.check_experiment_adjustment()
    try:
      [artifact] = outcome.get_overall_artifacts()
    except ValueError:
      return None

    if not artifact.adjustment_data:
      return None

    calibrated_channels = list(artifact.adjustment_data.keys())

    if len(calibrated_channels) > channel_limit:
      channels_text = f'your {channel_limit} highest-spend calibrated channels'
      banner_text = (
          eda_constants.EXPERIMENT_ADJUSTMENTS_BANNER_TEXT_TEMPLATE.format(
              channels=channels_text
          )
          + eda_constants.EXPERIMENT_ADJUSTMENTS_LIMIT_MESSAGE.format(
              limit=channel_limit
          )
      )
    else:
      channels_text = 'all calibrated channels'
      banner_text = (
          eda_constants.EXPERIMENT_ADJUSTMENTS_BANNER_TEXT_TEMPLATE.format(
              channels=channels_text
          )
      )

    return formatter.ChartSpec(
        id=eda_constants.EXPERIMENT_ADJUSTMENTS_CHART_ID,
        chart_json=adjustments_chart.to_json(),
        infos=[banner_text],
    )

  def _validate_and_get_geos_to_plot(self, geos: Geos) -> Sequence[str]:
    """Validates and returns the geos to plot."""
    is_national = self._meridian.is_national
    if is_national or geos == eda_constants.NATIONALIZE:
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
        not is_national and geos != eda_constants.NATIONALIZE
    ):  # if national then geos_to_plot will be ignored
      for geo in geos_to_plot:
        if geo not in self._meridian.input_data.geo:
          raise ValueError(f'Geo {geo} does not exist in the data.')
      if len(geos_to_plot) != len(set(geos_to_plot)):
        raise ValueError('geos must not contain duplicate values.')

    return geos_to_plot


def _calculate_relative_shares(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates the relative shares of each variable to plot for Altair."""
  return (
      df.groupby(eda_constants.VARIABLE)[eda_constants.VALUE]
      .sum()
      .pipe(lambda s: s / s.sum())
      .reset_index(name=eda_constants.VALUE)
  )


def _process_stacked_ds(
    data: xr.DataArray, include_time: bool = False
) -> pd.DataFrame:
  """Processes a stacked Dataset so it can be plotted by Altair."""
  df = data.rename(eda_constants.VALUE).to_dataframe().reset_index()
  if include_time or constants.TIME not in df.columns:
    return df
  return df.drop(columns=[constants.TIME])


def _plot_time_series(
    data: pd.DataFrame,
    *,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    y_axis_orient: str = 'left',
    y_axis_offset: int = 0,
) -> alt.Chart:
  """Helper function for plotting time series charts.

  Args:
    data: The DataFrame containing the time series data.
    title: The title of the chart.
    x_axis_title: The title for the x-axis.
    y_axis_title: The title for the y-axis.
    y_axis_orient: The orientation of the y-axis labels and title. Can be 'left'
      or 'right'.
    y_axis_offset: The offset in pixels to shift the y-axis. Useful when
      plotting multiple y-axes.

  Returns:
    An Altair time series chart.
  """
  color = eda_constants.CHANNEL_TYPE_TO_COLOR.get(
      y_axis_title, eda_constants.DEFAULT_CHART_COLOR
  )
  return (
      alt.Chart(data)
      .mark_line(opacity=1, color=color)
      .encode(
          x=alt.X(
              f'{constants.TIME}:T',
              axis=alt.Axis(grid=True, title=x_axis_title, format='%Y-%m'),
          ),
          y=alt.Y(
              f'{eda_constants.VALUE}:Q',
              axis=alt.Axis(
                  grid=True,
                  title=y_axis_title,
                  titleColor=color,
                  minExtent=60,
                  orient=y_axis_orient,  # pyrefly: ignore[bad-argument-type]
                  offset=y_axis_offset,
              ),
              scale=alt.Scale(zero=False),
          ),
          tooltip=[
              alt.Tooltip(
                  f'{constants.TIME}:T', title='Time', format='%Y-%m-%d'
              ),
              alt.Tooltip(
                  f'{eda_constants.VALUE}:Q',
                  title=y_axis_title,
                  format='.3e',
              ),
          ],
      )
      .properties(
          title=alt.TitleParams(text=title, anchor='start'),
          width=600,
          height=300,
      )
  )


def _apply_chart_config(chart: alt.Chart) -> alt.Chart:
  """Applies the final chart configurations for concatenated charts."""
  return (
      chart.configure_axis(labelAngle=315)
      .configure_title(anchor='start')
      .configure_view(stroke=None)
  )


def _format_display_table(
    df: pd.DataFrame,
    sort_columns: Sequence[str],
    ascending_orders: Sequence[bool],
) -> pd.DataFrame:
  """Returns the formatted report card tables sorted and filtered."""
  return (
      df.reset_index()
      .sort_values(by=sort_columns, ascending=ascending_orders)
      .head(eda_constants.DISPLAY_LIMIT)
      .apply(lambda x: x.map('{:.3f}'.format) if x.dtype.kind == 'f' else x)
  )


def _get_finding(
    outcome: eda_outcome.EDAOutcome,
    finding_cause: eda_outcome.FindingCause,
    severity: eda_outcome.EDASeverity,
    variable: str | None = None,
) -> eda_outcome.EDAFinding | None:
  """Retrieves a unique finding matching specific criteria.

  This function filters EDA findings from an outcome based on `finding_cause`
  and `severity`. If `variable` is provided, it further filters for findings
  associated with a `StandardDeviationArtifact` where the artifact's variable
  matches the given `variable`.

  The function expects at most one finding to match the criteria.

  Args:
    outcome: The EDAOutcome containing findings.
    finding_cause: The cause of the finding to filter by.
    severity: The severity of the finding to filter by.
    variable: If provided, filters for findings associated with a
      `StandardDeviationArtifact` matching this variable name.

  Returns:
    The single matching `EDAFinding` if one is found, otherwise `None`.

  Raises:
    ValueError: If more than one finding matches the specified criteria.
  """
  findings = outcome.get_findings_by_cause_and_severity(finding_cause, severity)
  if variable:
    findings = [
        finding
        for finding in findings
        if isinstance(
            finding.associated_artifact, eda_outcome.StandardDeviationArtifact
        )
        and finding.associated_artifact.variable == variable
    ]
  if len(findings) > 1:
    raise ValueError(
        f'Expected at most one finding for {finding_cause.name} with severity'
        f' {severity.name}, but found {len(findings)}.'
    )
  return findings[0] if findings else None


def _create_display_limit_message(
    display_table: pd.DataFrame,
    function: str,
    to_review_prefix: str,
    n_channels: int,
    n_times: int | None = None,
    n_geos: int | None = None,
) -> str:
  """Creates a formatted display limit message from the given arguments."""
  if len(display_table) <= eda_constants.DISPLAY_LIMIT:
    return ''

  to_review_suffixes = []
  if n_times:
    to_review_suffixes.append(f'{n_times} times')
  if n_geos:
    to_review_suffixes.append(f'{n_geos} geos')

  to_review_suffix = (
      (' in ' + ' and '.join(to_review_suffixes)) if to_review_suffixes else ''
  )
  to_review = f'{to_review_prefix} for {n_channels} channels{to_review_suffix}'
  return eda_constants.DISPLAY_LIMIT_MESSAGE.format(
      function=function, to_review=to_review
  )


def _initialize_severity_counts() -> dict[eda_outcome.EDASeverity, int]:
  """Initializes a dictionary of severity counts."""
  return {
      eda_outcome.EDASeverity.REVIEW: 0,
      eda_outcome.EDASeverity.FAIL: 0,
  }


def _get_plot_data_for_heatmap(
    df: pd.DataFrame, all_variables: Sequence[str], max_vars: int | None = None
) -> tuple[pd.DataFrame, Sequence[str]]:
  """Prepares the data for the heatmap plot.

  Args:
    df: A pandas DataFrame representing one of the triangles (upper or lower) of
      the correlation matrix, with columns VARIABLE_1, VARIABLE_2, and
      CORRELATION.
    all_variables: A sequence of all unique variable names present in the
      correlation matrix.
    max_vars: The maximum number of variables to include in the correlation
      heatmap. Variables are selected based on their absolute correlation
      values. If None, all variables are included.

  Returns:
    A tuple (triangle_plot_data, plot_variables), where triangle_plot_data
    is a pandas DataFrame representing the corresponding triangle of the
    correlation matrix, with columns VARIABLE_1, VARIABLE_2, and CORRELATION,
    and plot_variables is a sequence of unique variable names included in the
    plot.

  Raises:
    ValueError: If `max_vars` is provided and is less than or equal to 1.
  """
  if max_vars is not None and max_vars <= 1:
    raise ValueError('max_vars must be greater than 1.')

  n_total_vars = len(all_variables)

  if max_vars is not None and max_vars < n_total_vars:
    selected_vars = set()

    # Sort by absolute correlation value. Use stable sort to ensure
    # consistent results with tied correlations.
    sorted_pairs = df.sort_values('abs_corr', ascending=False, kind='mergesort')

    stop_evaluating_below_corr = None
    for _, row in sorted_pairs.iterrows():
      abs_c = row['abs_corr']

      # If adding a previous pair would have exceeded max_vars, its
      # correlation was recorded as `stop_evaluating_below_corr`. We continue
      # evaluating subsequent pairs only if they are tied with this
      # correlation, as a tied pair might still fit if it reuses one of the
      # already-selected variables in its pair.
      # Once we encounter a strictly lower correlation, we stop completely.
      if stop_evaluating_below_corr is not None and not np.isclose(
          abs_c, stop_evaluating_below_corr
      ):
        break

      v1 = row[eda_constants.VARIABLE_1]
      v2 = row[eda_constants.VARIABLE_2]

      if len(selected_vars | {v1, v2}) <= max_vars:
        selected_vars.update({v1, v2})
      elif stop_evaluating_below_corr is None:
        stop_evaluating_below_corr = abs_c

      if len(selected_vars) == max_vars:
        break
  else:
    selected_vars = set(all_variables)

  # Maintain the original order of variables.
  plot_variables = [v for v in all_variables if v in selected_vars]

  return (
      df[
          df[eda_constants.VARIABLE_1].isin(plot_variables)
          & df[eda_constants.VARIABLE_2].isin(plot_variables)
      ],
      plot_variables,
  )


def _format_explanation_for_html(explanation: str) -> str:
  """Formats an explanation string for HTML rendering using <br> tags."""
  explanation = _BULLET_POINT_RE.sub('&#160;&#160;&#8226; ', explanation)
  return explanation.replace('\n', '<br/>')
