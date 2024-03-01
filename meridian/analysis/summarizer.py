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

"""Summarization module that creates a 2-page HTML report."""

from collections.abc import Sequence
import datetime as dt
import functools
import os

import jinja2
from meridian import constants as c
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.analysis import visualizer
from meridian.model import model
import pandas as pd
import xarray as xr


MODEL_FIT_CARD_SPEC = formatter.CardSpec(
    id=summary_text.MODEL_FIT_CARD_ID,
    title=summary_text.MODEL_FIT_CARD_TITLE,
)
REVENUE_CONTRIB_CARD_SPEC = formatter.CardSpec(
    id=summary_text.IMPACT_CONTRIB_CARD_ID,
    title=summary_text.IMPACT_CONTRIB_CARD_TITLE.format(
        impact=c.REVENUE.title()
    ),
)
KPI_CONTRIB_CARD_SPEC = formatter.CardSpec(
    id=summary_text.IMPACT_CONTRIB_CARD_ID,
    title=summary_text.IMPACT_CONTRIB_CARD_TITLE.format(impact=c.KPI.upper()),
)
ROI_BREAKDOWN_CARD_SPEC = formatter.CardSpec(
    id=summary_text.ROI_BREAKDOWN_CARD_ID,
    title=summary_text.ROI_BREAKDOWN_CARD_TITLE,
)
BUDGET_OPTIMIZATION_CARD_SPEC = formatter.CardSpec(
    id=summary_text.BUDGET_OPTIMIZATION_CARD_ID,
    title=summary_text.BUDGET_OPTIMIZATION_CARD_TITLE,
)


class Summarizer:
  """Generates HTML summary visualizations from the model fitting."""

  def __init__(self, meridian: model.Meridian):
    """Initialize the visualizer classes that are not time-dependent."""
    self._meridian = meridian

  @functools.cached_property
  def _model_fit(self):
    return visualizer.ModelFit(self._meridian)

  @functools.cached_property
  def _model_diagnostics(self):
    return visualizer.ModelDiagnostics(self._meridian)

  @functools.cached_property
  def _response_time_values(self) -> Sequence[dt.datetime]:
    """The response data array's `time` coordinates as `datetime` objects."""
    return [
        dt.datetime.strptime(value, c.DATE_FORMAT)
        for value in self._meridian.input_data.time.values
    ]

  def output_model_results_summary(
      self,
      filename: str,
      filepath: str,
      start_date: str | None = None,
      end_date: str | None = None,
  ):
    """Generates and saves the HTML results summary output.

    Args:
      filename: The filename for the generated HTML output.
      filepath: The path to the directory where the file will be saved.
      start_date: Optional start date selector, in _yyyy-mm-dd_ format.
      end_date: Optional end date selector, in _yyyy-mm-dd_ format.
    """
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as f:
      f.write(
          self._gen_model_results_summary(
              (
                  dt.datetime.strptime(start_date, c.DATE_FORMAT)
                  if start_date
                  else None
              ),
              (
                  dt.datetime.strptime(end_date, c.DATE_FORMAT)
                  if end_date
                  else None
              ),
          )
      )

  def _gen_model_results_summary(
      self,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
  ) -> str:
    """Generate HTML results summary output (as sanitized content str)."""
    start_date = start_date or min(self._response_time_values)
    end_date = end_date or max(self._response_time_values)

    if start_date not in self._response_time_values:
      raise ValueError(
          f'start_date ({start_date.strftime(c.DATE_FORMAT)}) must be'
          ' in the time coordinates!'
      )
    if end_date not in self._response_time_values:
      raise ValueError(
          f'end_date ({end_date.strftime(c.DATE_FORMAT)}) must be'
          ' in the time coordinates!'
      )
    if start_date > end_date:
      raise ValueError(
          f'start_date ({start_date.strftime(c.DATE_FORMAT)}) must be before'
          f' end_date ({end_date.strftime(c.DATE_FORMAT)})!'
      )

    template_env = formatter.create_template_env()
    template_env.globals[c.START_DATE] = start_date.strftime(
        c.SUMMARY_DATE_FORMAT
    )
    template_env.globals[c.END_DATE] = end_date.strftime(c.SUMMARY_DATE_FORMAT)

    html_template = template_env.get_template('summary.html.jinja')
    cards_htmls = self._create_cards_htmls(
        template_env,
        selected_times=self._get_selected_times(start_date, end_date),
    )

    return html_template.render(
        title=summary_text.MODEL_RESULTS_TITLE, cards=cards_htmls
    )

  def _get_selected_times(
      self,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
  ) -> Sequence[str]:
    selected_times = [
        date
        for date in self._response_time_values
        if start_date <= date <= end_date
    ]
    return [time.strftime(c.DATE_FORMAT) for time in selected_times]

  def _create_cards_htmls(
      self,
      template_env: jinja2.Environment,
      selected_times: Sequence[str],
  ) -> Sequence[str]:
    """Creates the HTML snippets for cards in the summary page."""
    media_summary = visualizer.MediaSummary(
        self._meridian, selected_times=selected_times
    )
    media_effects = visualizer.MediaEffects(self._meridian)
    reach_frequency = (
        visualizer.ReachAndFrequency(
            self._meridian, selected_times=selected_times
        )
        if self._meridian.n_rf_channels > 0
        else None
    )
    cards = [
        self._create_model_fit_card_html(
            template_env, selected_times=selected_times
        ),
        self._create_impact_contrib_card_html(template_env, media_summary),
        self._create_budget_optimization_card_html(
            template_env=template_env,
            selected_times=selected_times,
            media_summary=media_summary,
            media_effects=media_effects,
            reach_frequency=reach_frequency,
        ),
    ]
    # If Meridian's `revenue_per_kpi` exists, add into the card HTML in-between
    # the KPI/Revenue contribution and Optimization sections.

    if self._meridian.input_data.revenue_per_kpi is not None:
      cards.insert(
          2, self._create_roi_breakdown_card_html(template_env, media_summary)
      )
    return cards

  def _create_model_fit_card_html(
      self, template_env: jinja2.Environment, **kwargs
  ) -> str:
    """Creates the HTML snippet for the Model Fit card."""
    model_fit = self._model_fit
    expected_actual_impact_chart = formatter.ChartSpec(
        id=summary_text.EXPECTED_ACTUAL_IMPACT_CHART_ID,
        description=summary_text.EXPECTED_ACTUAL_IMPACT_CHART_DESCRIPTION,
        chart_json=model_fit.plot_model_fit(**kwargs).to_json(),
    )

    predictive_accuracy_table = self._predictive_accuracy_table_spec(**kwargs)
    insights = summary_text.MODEL_FIT_INSIGHTS_FORMAT

    return formatter.create_card_html(
        template_env,
        MODEL_FIT_CARD_SPEC,
        insights,
        [expected_actual_impact_chart, predictive_accuracy_table],
    )

  def _predictive_accuracy_table_spec(self, **kwargs) -> formatter.TableSpec:
    """Creates the HTML snippet for the predictive accuracy table."""
    impact = self._kpi_or_revenue()
    model_diag = self._model_diagnostics
    table = model_diag.predictive_accuracy_table(column_var=c.METRIC, **kwargs)

    # Only take the national stats, even if geo ones exist.
    national_table = table[table[c.GEO_GRANULARITY] == c.NATIONAL]

    # Translate column names into human-presentable ones.
    column_names = [
        summary_text.DATASET_LABEL,
        summary_text.R_SQUARED_LABEL,
        summary_text.MAPE_LABEL,
        summary_text.WMAPE_LABEL,
    ]

    if c.EVALUATION_SET_VAR in list(national_table.columns):

      def _slice_table_by_evaluation_set(eval_set: str) -> Sequence[str]:
        """Slices table by the given evaluation set."""
        sliced_table_by_eval_set = national_table[
            national_table[c.EVALUATION_SET_VAR] == eval_set
        ]
        row_values = [
            '{:.2f}'.format(
                float(sliced_table_by_eval_set[c.R_SQUARED].iloc[0])
            ),
            '{:.0%}'.format(float(sliced_table_by_eval_set[c.MAPE].iloc[0])),
            '{:.0%}'.format(float(sliced_table_by_eval_set[c.WMAPE].iloc[0])),
        ]
        return row_values

      # The dataset has holdout_id that distinguish training and test data sets.
      training_row = [summary_text.TRAINING_DATA_LABEL]
      training_row.extend(_slice_table_by_evaluation_set(c.TRAIN))
      testing_row = [summary_text.TESTING_DATA_LABEL]
      testing_row.extend(_slice_table_by_evaluation_set(c.TEST))
      all_data_row = [summary_text.ALL_DATA_LABEL]
      all_data_row.extend(_slice_table_by_evaluation_set(c.ALL_DATA))

      row_values = [training_row, testing_row, all_data_row]
    else:  # No holdout_id present, so metrics are taken from 'All Data'.
      row_values = [[
          summary_text.ALL_DATA_LABEL,
          '{:.2f}'.format(float(national_table[c.R_SQUARED].iloc[0])),
          '{:.0%}'.format(float(national_table[c.MAPE].iloc[0])),
          '{:.0%}'.format(float(national_table[c.WMAPE].iloc[0])),
      ]]

    return formatter.TableSpec(
        id=summary_text.PREDICTIVE_ACCURACY_TABLE_ID,
        title=summary_text.PREDICTIVE_ACCURACY_TABLE_TITLE.format(
            impact=impact
        ),
        description=summary_text.PREDICTIVE_ACCURACY_TABLE_DESCRIPTION.format(
            impact=impact
        ),
        column_headers=column_names,
        row_values=row_values,
    )

  def _create_impact_contrib_card_html(
      self,
      template_env: jinja2.Environment,
      media_summary: visualizer.MediaSummary,
  ) -> str:
    """Creates the HTML snippet for the Impact Contrib card."""
    impact = self._kpi_or_revenue()
    channel_drivers_chart = formatter.ChartSpec(
        id=summary_text.CHANNEL_DRIVERS_CHART_ID,
        description=summary_text.CHANNEL_DRIVERS_CHART_DESCRIPTION.format(
            impact=impact
        ),
        chart_json=media_summary.plot_contribution_waterfall_chart().to_json(),
    )
    lead_channels = self._get_sorted_posterior_mean_metrics_df(
        media_summary, [c.INCREMENTAL_IMPACT]
    )[c.CHANNEL][:2]
    spend_impact_chart = formatter.ChartSpec(
        id=summary_text.SPEND_IMPACT_CHART_ID,
        description=summary_text.SPEND_IMPACT_CHART_DESCRIPTION.format(
            impact=impact
        ) if impact == c.REVENUE else '',
        chart_json=media_summary.plot_spend_vs_contribution().to_json(),
    )
    impact_contribution_chart = formatter.ChartSpec(
        id=summary_text.IMPACT_CONTRIBUTION_CHART_ID,
        description=summary_text.IMPACT_CONTRIBUTION_CHART_DESCRIPTION.format(
            impact=impact
        ),
        chart_json=media_summary.plot_contribution_pie_chart().to_json(),
    )
    if impact == c.REVENUE:
      roi_df = self._get_sorted_posterior_mean_metrics_df(
          media_summary, [c.ROI]
      )
      insights = summary_text.IMPACT_CONTRIB_INSIGHTS_FORMAT.format(
          impact=impact,
          lead_channels=' and '.join(lead_channels),
          lead_roi_channel=roi_df[c.CHANNEL][0],
          lead_roi_ratio=roi_df[c.ROI][0],
      )
      impact_contrib_card_spec = REVENUE_CONTRIB_CARD_SPEC
      card_string = formatter.create_card_html(
          template_env=template_env,
          card_spec=impact_contrib_card_spec,
          insights=insights,
          chart_specs=[
              channel_drivers_chart,
              spend_impact_chart,
              impact_contribution_chart,
          ],
      )
      return card_string
    else:
      impact_contrib_card_spec = KPI_CONTRIB_CARD_SPEC
      card_string = formatter.create_card_html(
          template_env=template_env,
          card_spec=impact_contrib_card_spec,
          insights='',
          chart_specs=[
              channel_drivers_chart,
              spend_impact_chart,
              impact_contribution_chart,
          ],
      )
      return card_string

  def _get_sorted_posterior_mean_metrics_df(
      self,
      media_summary: visualizer.MediaSummary,
      metrics: Sequence[str],
  ) -> pd.DataFrame:
    return (
        media_summary.media_summary_metrics[metrics]
        .sel(distribution=c.POSTERIOR, metric=c.MEAN)
        .drop_sel(channel=c.ALL_CHANNELS)
        .to_dataframe()
        .drop(columns=[c.METRIC, c.DISTRIBUTION])
        .sort_values(by=metrics, ascending=False)
        .reset_index()
    )

  def _create_roi_breakdown_card_html(
      self,
      template_env: jinja2.Environment,
      media_summary: visualizer.MediaSummary,
  ) -> str:
    """Creates the HTML snippet for the ROI Breakdown card."""
    impact = self._kpi_or_revenue()
    roi_effectiveness_chart = formatter.ChartSpec(
        id=summary_text.ROI_EFFECTIVENESS_CHART_ID,
        description=summary_text.ROI_EFFECTIVENESS_CHART_DESCRIPTION.format(
            impact=impact
        ),
        chart_json=media_summary.plot_roi_vs_effectiveness().to_json(),
    )
    roi_marginal_chart = formatter.ChartSpec(
        id=summary_text.ROI_MARGINAL_CHART_ID,
        description=summary_text.ROI_MARGINAL_CHART_DESCRIPTION,
        chart_json=media_summary.plot_roi_vs_mroi().to_json(),
    )
    roi_channel_chart = formatter.ChartSpec(
        id=summary_text.ROI_CHANNEL_CHART_ID,
        description=summary_text.ROI_CHANNEL_CHART_DESCRIPTION,
        chart_json=media_summary.plot_roi_bar_chart().to_json(),
    )
    effectiveness_df = self._get_sorted_posterior_mean_metrics_df(
        media_summary, [c.EFFECTIVENESS]
    )
    mroi_df = self._get_sorted_posterior_mean_metrics_df(
        media_summary, [c.MROI]
    )
    insights = summary_text.ROI_BREAKDOWN_INSIGHTS_FORMAT.format(
        lead_effectiveness_channel=effectiveness_df[c.CHANNEL][0],
        lead_marginal_roi_channel=mroi_df[c.CHANNEL][0],
        lead_marginal_roi_channel_value=mroi_df[c.MROI][0],
    )
    return formatter.create_card_html(
        template_env,
        ROI_BREAKDOWN_CARD_SPEC,
        insights,
        [roi_effectiveness_chart, roi_marginal_chart, roi_channel_chart],
    )

  def _create_budget_optimization_card_html(
      self,
      template_env: jinja2.Environment,
      selected_times: Sequence[str],
      media_summary: visualizer.MediaSummary,
      media_effects: visualizer.MediaEffects,
      reach_frequency: visualizer.ReachAndFrequency | None,
  ) -> str:
    """Creates the HTML snippet for the Optimal Analyst card."""
    impact = self._kpi_or_revenue()
    charts = []
    charts.append(
        formatter.ChartSpec(
            id=summary_text.RESPONSE_CURVES_CHART_ID,
            description=summary_text.RESPONSE_CURVES_CHART_DESCRIPTION.format(
                impact=impact
            ),
            chart_json=media_effects.plot_response_curves(
                confidence_level=0.9,
                selected_times=frozenset(selected_times),
                plot_separately=False,
                include_ci=False,
                num_channels_displayed=7,
            ).to_json(),
        )
    )

    if reach_frequency is None:
      insights = summary_text.BUDGET_OPTIMIZATION_INSIGHTS_NO_RF
    else:
      assert self._meridian.n_rf_channels > 0
      optimal_rf = self._select_optimal_rf_data(media_summary, reach_frequency)
      channel_name = optimal_rf[c.RF_CHANNEL].values.item()
      opt_freq = '{:.1f}'.format(optimal_rf.values.item())

      charts.append(
          formatter.ChartSpec(
              id=summary_text.OPTIMAL_FREQUENCY_CHART_ID,
              description=summary_text.OPTIMAL_FREQUENCY_CHART_DESCRIPTION,
              chart_json=reach_frequency.plot_optimal_frequency(
                  selected_channels=[channel_name],
              ).to_json(),
          )
      )

      insights = summary_text.BUDGET_OPTIMIZATION_INSIGHTS_FORMAT.format(
          rf_channel=channel_name,
          opt_freq=opt_freq,
      )

    return formatter.create_card_html(
        template_env, BUDGET_OPTIMIZATION_CARD_SPEC, insights, charts
    )

  def _select_optimal_rf_data(
      self,
      media_summary: visualizer.MediaSummary,
      reach_frequency: visualizer.ReachAndFrequency,
  ) -> xr.DataArray:
    """Selects and returns the `optimal_frequency` DataArray--if any.

    The `optimal_frequency` data is a subset of the Dataset property
    `optimal_frequency_data` of visualizer.ReachAndFrequency.

    Assumes that there is at least 1 RF channel in the model, else ValueError.
    Returns:
      DataArray of the optimal_frequency data for the channel with the highest
      spend value (per MediaSummary).
    """
    # Select the optimal frequency channel with the most spend.
    # This raises ValueError if there is no RF channel in the model.
    rf_channels = reach_frequency.optimal_frequency_data.rf_channel
    assert rf_channels.size > 0
    # This will raise KeyError if not all `rf_channels` can be found in here:
    rf_channel_spends = media_summary.media_summary_metrics[c.SPEND].sel(
        channel=rf_channels
    )
    most_spend_rf_channel = rf_channel_spends.idxmax()

    return reach_frequency.optimal_frequency_data.sel(
        rf_channel=most_spend_rf_channel
    ).optimal_frequency

  def _kpi_or_revenue(self) -> str:
    if self._meridian.input_data.revenue_per_kpi is not None:
      impact_str = c.REVENUE
    else:
      impact_str = c.KPI.upper()
    return impact_str
