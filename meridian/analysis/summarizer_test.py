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

import datetime as dt
import os
import tempfile
from unittest import mock
from xml.etree import ElementTree as ET

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import summarizer
from meridian.analysis import summary_text
from meridian.analysis import test_utils
from meridian.data import input_data
from meridian.data import test_utils as data_test_utils
from meridian.data import time_coordinates as tc
from meridian.model import model
import xarray as xr


_EARLIEST_DATE = dt.datetime(2022, 1, 1)
_NUM_WEEKS = 80

_TIME_COORDS = [
    _EARLIEST_DATE + dt.timedelta(weeks=i) for i in range(_NUM_WEEKS)
]
_TIME_COORDS_STRINGS = [t.strftime(c.DATE_FORMAT) for t in _TIME_COORDS]

_LATEST_DATE = _TIME_COORDS[-1]


class SummarizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.input_data = mock.create_autospec(input_data.InputData, instance=True)
    self.input_data_2 = mock.create_autospec(
        input_data.InputData, instance=True
    )
    n_times = 149
    n_geos = 10
    self.revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=2.2
    )
    self.mock_meridian_revenue = mock.create_autospec(
        model.Meridian, instance=True, input_data=self.input_data
    )
    self.mock_meridian_revenue.input_data.kpi_type = c.REVENUE
    self.mock_meridian_revenue.input_data.revenue_per_kpi = self.revenue_per_kpi

    self.mock_meridian_kpi = mock.create_autospec(
        model.Meridian, instance=True, input_data=self.input_data_2
    )
    self.mock_meridian_kpi.input_data.kpi_type = c.NON_REVENUE
    self.mock_meridian_kpi.input_data.revenue_per_kpi = None

    self.analyzer_patcher = mock.patch.object(
        analyzer,
        'Analyzer',
    )
    self.analyzer = self.analyzer_patcher.start()()

    self.model_fit_patcher = mock.patch.object(
        summarizer.visualizer,
        'ModelFit',
    )
    self.model_fit = self.model_fit_patcher.start()()

    self.model_diagnostics_patcher = mock.patch.object(
        summarizer.visualizer,
        'ModelDiagnostics',
    )
    self.model_diagnostics = self.model_diagnostics_patcher.start()()

    self.media_summary_patcher = mock.patch.object(
        summarizer.visualizer,
        'MediaSummary',
    )
    self.media_summary_class = self.media_summary_patcher.start()
    self.media_summary = self.media_summary_class()

    self.media_effects_patcher = mock.patch.object(
        summarizer.visualizer,
        'MediaEffects',
    )
    self.media_effects_class = self.media_effects_patcher.start()
    self.media_effects = self.media_effects_class()

    self.reach_frequency_patcher = mock.patch.object(
        summarizer.visualizer,
        'ReachAndFrequency',
    )
    self.reach_frequency_class = self.reach_frequency_patcher.start()
    self.reach_frequency = self.reach_frequency_class()

    self.summarizer_revenue = summarizer.Summarizer(self.mock_meridian_revenue)
    self.summarizer_kpi = summarizer.Summarizer(self.mock_meridian_kpi)

    self._stub_plotters()
    self._stub_for_insights()

  def tearDown(self):
    super().tearDown()

    self.analyzer_patcher.stop()
    self.model_fit_patcher.stop()
    self.model_diagnostics_patcher.stop()
    self.media_summary_patcher.stop()
    self.media_effects_patcher.stop()
    self.reach_frequency_patcher.stop()

  def _stub_plotters(self):
    self.model_fit.plot_model_fit().to_json.return_value = '{}'

    self._stub_media_summary_plotters(self.media_summary)

    self.model_diagnostics.predictive_accuracy_table.return_value = (
        test_utils.generate_predictive_accuracy_table(
            with_holdout=True, column_var=c.METRIC
        )
    )

    self.media_effects.plot_response_curves().to_json.return_value = '{}'

    self.reach_frequency.plot_optimal_frequency().to_json.return_value = '{}'

  def _stub_media_summary_plotters(self, media_summary):
    media_summary.plot_channel_contribution_area_chart().to_json.return_value = (
        '{}'
    )
    media_summary.plot_contribution_waterfall_chart().to_json.return_value = (
        '{}'
    )
    media_summary.plot_contribution_pie_chart().to_json.return_value = '{}'
    media_summary.plot_spend_vs_contribution().to_json.return_value = '{}'
    media_summary.plot_roi_vs_effectiveness().to_json.return_value = '{}'
    media_summary.plot_roi_vs_mroi().to_json.return_value = '{}'
    media_summary.plot_roi_bar_chart().to_json.return_value = '{}'
    media_summary.plot_cpik().to_json.return_value = '{}'

  def _stub_for_insights(self):
    self.media_metrics = test_utils.generate_paid_summary_metrics()
    self.media_summary.get_paid_summary_metrics = mock.MagicMock(
        return_value=self.media_metrics
    )

    frequency_data = test_utils.generate_optimal_frequency_data(
        channel_prefix='rf_ch', num_channels=2
    )
    self.reach_frequency.optimal_frequency_data = frequency_data
    self.mock_meridian_revenue.n_rf_channels = 2
    self.mock_meridian_kpi.n_rf_channels = 2

    # Stub `input_data.time` property.
    response = xr.DataArray(
        data=None,
        dims=[c.TIME],
        coords={
            c.TIME: list(_TIME_COORDS_STRINGS),
        },
    )
    for mock_meridian in (self.mock_meridian_revenue, self.mock_meridian_kpi):
      mock_meridian.input_data.time = response[c.TIME]
      mock_meridian.input_data.time_coordinates = tc.TimeCoordinates.from_dates(
          response[c.TIME]
      )
      mock_meridian.expand_selected_time_dims.return_value = (
          _TIME_COORDS_STRINGS
      )

  def _get_output_model_results_summary_html_dom(
      self,
      summarizer_outcome: summarizer.Summarizer,
      start_date: dt.datetime | None = _EARLIEST_DATE,
      end_date: dt.datetime | None = _LATEST_DATE,
  ) -> ET.Element:
    outfile_path = tempfile.mkdtemp() + '/summary'
    outfile_name = 'sum.html'
    fpath = os.path.join(outfile_path, outfile_name)

    try:
      summarizer_outcome.output_model_results_summary(
          filename=outfile_name,
          filepath=outfile_path,
          start_date=start_date.strftime(c.DATE_FORMAT) if start_date else None,
          end_date=end_date.strftime(c.DATE_FORMAT) if end_date else None,
      )

      with open(fpath, 'r') as f:
        written_html_dom = ET.parse(f)
    finally:
      os.remove(fpath)
      os.removedirs(outfile_path)

    root = written_html_dom.getroot()
    self.assertEqual(root.tag, 'html')
    return root

  @parameterized.parameters([
      (
          _LATEST_DATE,
          _EARLIEST_DATE,
          'start_date (2023-07-08) must be before end_date (2022-01-01)!',
      ),
      (
          _EARLIEST_DATE + dt.timedelta(days=1),
          None,
          'start_date (2022-01-02) must be in the time coordinates!',
      ),
      (
          None,
          _LATEST_DATE - dt.timedelta(days=1),
          'end_date (2023-07-07) must be in the time coordinates!',
      ),
  ])
  def test_output_invalid_date_range(
      self, start_date, end_date, expected_error_message
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_message):
      self._get_output_model_results_summary_html_dom(
          summarizer_outcome=self.summarizer_revenue,
          start_date=start_date,
          end_date=end_date,
      )

  def test_output_html_title(self):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    title = summary_html_dom.find('head/title')

    self.assertIsNotNone(title)
    title_text = title.text
    self.assertIsNotNone(title_text)
    self.assertEqual(title_text.strip(), summary_text.MODEL_RESULTS_TITLE)

  def test_output_header_section(self):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    header_div = test_utils.get_child_element(
        summary_html_dom, 'body/div', {'class': 'header'}
    )
    _ = test_utils.get_child_element(header_div, 'div', {'class': 'logo'})
    header_title_div = test_utils.get_child_element(
        header_div, 'div', {'class': 'title'}
    )

    header_title_div_text = header_title_div.text
    self.assertIsNotNone(header_title_div_text)
    self.assertEqual(
        header_title_div_text.strip(), summary_text.MODEL_RESULTS_TITLE
    )

  def test_output_chips(self):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    chips_node = summary_html_dom.find('body/chips')
    self.assertIsNotNone(chips_node)
    chip_nodes = chips_node.findall('chip')

    self.assertLen(chip_nodes, 1)
    self.assertSequenceEqual(
        [chip.text.strip() for chip in chip_nodes if chip.text is not None],
        [
            'Time period: Jan 1, 2022 - Jul 15, 2023',
        ],
    )

  def test_output_no_date_range(self):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
        start_date=None,
        end_date=None,
    )
    chip_nodes = summary_html_dom.findall('body/chips/chip')

    self.assertSequenceEqual(
        [chip.text.strip() for chip in chip_nodes if chip.text is not None],
        [
            'Time period: Jan 1, 2022 - Jul 15, 2023',
        ],
    )

  def test_output_chips_one_week(self):
    start_date, end_date = dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 1)

    self.summarizer_revenue._meridian.expand_selected_time_dims.return_value = [
        start_date.strftime(c.DATE_FORMAT)
    ]

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
        start_date=start_date,
        end_date=end_date,
    )
    chips_node = summary_html_dom.find('body/chips')
    self.assertIsNotNone(chips_node)
    chip_nodes = chips_node.findall('chip')

    self.assertLen(chip_nodes, 1)
    self.assertSequenceEqual(
        [chip.text.strip() for chip in chip_nodes if chip.text is not None],
        ['Time period: Jan 1, 2022 - Jan 8, 2022'],
    )

  def test_output_card_structure(self):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    cards_node = test_utils.get_child_element(summary_html_dom, 'body/cards')

    self.assertLen(cards_node, 4)
    for card in cards_node:
      self.assertEqual(card.tag, 'card')
      self.assertTrue(
          {child_node.tag for child_node in card}.issubset({
              'card-title',
              'card-insights-icon',
              'card-insights',
              'charts',
          })
      )

  @parameterized.parameters(
      summarizer.MODEL_FIT_CARD_SPEC,
      summarizer.CHANNEL_CONTRIB_CARD_SPEC,
      summarizer.PERFORMANCE_BREAKDOWN_CARD_SPEC,
      summarizer.RESPONSE_CURVES_CARD_SPEC,
  )
  def test_output_card_static_chart_spec(self, card_spec):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    card = test_utils.get_child_element(
        summary_html_dom, 'body/cards/card', attribs={'id': card_spec.id}
    )
    card_title_text = test_utils.get_child_element(card, 'card-title').text
    self.assertIsNotNone(card_title_text)
    self.assertEqual(card_title_text.strip(), card_spec.title)

  @parameterized.parameters([
      (
          summary_text.MODEL_FIT_CARD_ID,
          [
              (
                  summary_text.EXPECTED_ACTUAL_OUTCOME_CHART_ID,
                  summary_text.EXPECTED_ACTUAL_OUTCOME_CHART_DESCRIPTION_FORMAT.format(
                      outcome=c.REVENUE
                  ),
              ),
          ],
      ),
      (
          summary_text.CHANNEL_CONTRIB_CARD_ID,
          [
              (
                  summary_text.CHANNEL_CONTRIB_BY_TIME_CHART_ID,
                  summary_text.CHANNEL_CONTRIB_BY_TIME_CHART_DESCRIPTION.format(
                      outcome=c.REVENUE
                  ),
              ),
              (
                  summary_text.CHANNEL_DRIVERS_CHART_ID,
                  summary_text.CHANNEL_DRIVERS_CHART_DESCRIPTION.format(
                      outcome=c.REVENUE
                  ),
              ),
              (
                  summary_text.SPEND_OUTCOME_CHART_ID,
                  summary_text.SPEND_OUTCOME_CHART_DESCRIPTION.format(
                      outcome=c.REVENUE
                  ),
              ),
              (
                  summary_text.OUTCOME_CONTRIBUTION_CHART_ID,
                  summary_text.OUTCOME_CONTRIBUTION_CHART_DESCRIPTION.format(
                      outcome=c.REVENUE
                  ),
              ),
          ],
      ),
      (
          summary_text.PERFORMANCE_BREAKDOWN_CARD_ID,
          [
              (
                  summary_text.ROI_EFFECTIVENESS_CHART_ID,
                  summary_text.ROI_EFFECTIVENESS_CHART_DESCRIPTION,
              ),
              (
                  summary_text.ROI_MARGINAL_CHART_ID,
                  summary_text.ROI_MARGINAL_CHART_DESCRIPTION,
              ),
              (summary_text.ROI_CHANNEL_CHART_ID,),
              (
                  summary_text.CPIK_CHANNEL_CHART_ID,
                  summary_text.CPIK_CHANNEL_CHART_DESCRIPTION,
              ),
          ],
      ),
      (
          summary_text.RESPONSE_CURVES_CARD_ID,
          [
              (
                  summary_text.RESPONSE_CURVES_CHART_ID,
                  summary_text.RESPONSE_CURVES_CHART_DESCRIPTION_FORMAT.format(
                      outcome=c.REVENUE
                  ),
              ),
              (
                  summary_text.OPTIMAL_FREQUENCY_CHART_ID,
                  summary_text.OPTIMAL_FREQ_CHART_DESCRIPTION,
              ),
          ],
      ),
  ])
  def test_card_chart_info(self, card_id, expected_chart_tuples):
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        summarizer_outcome=self.summarizer_revenue,
    )
    card = test_utils.get_child_element(
        summary_html_dom, 'body/cards/card', attribs={'id': card_id}
    )

    charts = []
    for chart in card.findall('charts/chart'):
      chart_embed = test_utils.get_child_element(chart, 'chart-embed')
      chart_id = chart_embed.attrib['id']
      try:
        chart_description_text = test_utils.get_child_element(
            chart, 'chart-description'
        ).text
      except AssertionError:
        chart_description_text = None
      if chart_description_text is None:
        charts.append((chart_id,))
      else:
        self.assertIsNotNone(chart_description_text)
        chart_description_text = chart_description_text.strip()
        charts.append((chart_id, chart_description_text))

    self.assertEqual(set(expected_chart_tuples), set(charts))

  def test_model_fit_card_custom_date_range(self):
    model_fit = self.model_fit

    mock_spec = 'model_fit'

    with mock.patch.object(model_fit, 'plot_model_fit') as plot:
      plot().to_json.return_value = f'["{mock_spec}"]'

      self.summarizer_revenue._meridian.expand_selected_time_dims.return_value = [
          '2022-06-04',
          '2022-06-11',
          '2022-06-18',
          '2022-06-25',
          '2022-07-02',
          '2022-07-09',
          '2022-07-16',
          '2022-07-23',
          '2022-07-30',
          '2022-08-06',
          '2022-08-13',
          '2022-08-20',
          '2022-08-27',
      ]

      summary_html_dom = self._get_output_model_results_summary_html_dom(
          self.summarizer_revenue,
          start_date=dt.datetime(2022, 6, 4),
          end_date=dt.datetime(2022, 8, 27),
      )

      self.mock_meridian_revenue.expand_selected_time_dims.assert_called_once_with(
          start_date=dt.datetime(2022, 6, 4).date(),
          end_date=dt.datetime(2022, 8, 27).date(),
      )

      plot.assert_called_with(
          selected_times=[
              '2022-06-04',
              '2022-06-11',
              '2022-06-18',
              '2022-06-25',
              '2022-07-02',
              '2022-07-09',
              '2022-07-16',
              '2022-07-23',
              '2022-07-30',
              '2022-08-06',
              '2022-08-13',
              '2022-08-20',
              '2022-08-27',
          ]
      )
      plot().to_json.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.MODEL_FIT_CARD_ID},
    )
    script_texts = [
        script.text.strip()
        for script in card.findall('charts/script')
        if script.text is not None
    ]

    self.assertTrue(
        any([mock_spec in script_text for script_text in script_texts])
    )

  def test_model_diagnostics_table_no_holdout(self):
    model_diag = self.model_diagnostics

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue
    )
    model_diag.predictive_accuracy_table.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.MODEL_FIT_CARD_ID},
    )

    chart_table = test_utils.get_child_element(card, 'charts/chart-table')
    self.assertEqual(
        chart_table.attrib['id'], summary_text.PREDICTIVE_ACCURACY_TABLE_ID
    )
    title_text = test_utils.get_child_element(
        chart_table, 'div', attribs={'class': 'chart-table-title'}
    ).text
    self.assertEqual(
        title_text,
        summary_text.PREDICTIVE_ACCURACY_TABLE_TITLE.format(outcome=c.REVENUE),
    )
    description_text = test_utils.get_child_element(
        chart_table, 'div', attribs={'class': 'chart-table-description'}
    ).text
    self.assertEqual(
        description_text,
        summary_text.PREDICTIVE_ACCURACY_TABLE_DESCRIPTION.format(
            outcome=c.REVENUE
        ),
    )

    table = test_utils.get_child_element(chart_table, 'div/table')

    header_row = test_utils.get_child_element(
        table, 'tr', attribs={'class': 'chart-table-column-headers'}
    )
    header_values = test_utils.get_table_row_values(header_row, 'th')
    self.assertSequenceEqual(
        header_values,
        [
            summary_text.DATASET_LABEL,
            summary_text.R_SQUARED_LABEL,
            summary_text.MAPE_LABEL,
            summary_text.WMAPE_LABEL,
        ],
    )

    value_rows = table.findall('tr')[1:]
    self.assertLen(value_rows, 3)
    values = test_utils.get_table_row_values(value_rows[0])
    self.assertLen(values, 4)
    self.assertEqual(values[0], summary_text.TRAINING_DATA_LABEL)

  def test_model_diagnostics_table_with_holdout(self):
    model_diag = self.model_diagnostics

    model_diag.predictive_accuracy_table.return_value = (
        test_utils.generate_predictive_accuracy_table(
            with_holdout=True, column_var=c.METRIC
        )
    )

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )
    model_diag.predictive_accuracy_table.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.MODEL_FIT_CARD_ID},
    )

    chart_table = test_utils.get_child_element(card, 'charts/chart-table')
    self.assertEqual(
        chart_table.attrib['id'], summary_text.PREDICTIVE_ACCURACY_TABLE_ID
    )
    title_text = test_utils.get_child_element(
        chart_table, 'div', attribs={'class': 'chart-table-title'}
    ).text
    self.assertEqual(
        title_text,
        summary_text.PREDICTIVE_ACCURACY_TABLE_TITLE.format(outcome=c.REVENUE),
    )
    description_text = test_utils.get_child_element(
        chart_table, 'div', attribs={'class': 'chart-table-description'}
    ).text
    self.assertEqual(
        description_text,
        summary_text.PREDICTIVE_ACCURACY_TABLE_DESCRIPTION.format(
            outcome=c.REVENUE
        ),
    )

    table = test_utils.get_child_element(chart_table, 'div/table')

    header_row = test_utils.get_child_element(
        table, 'tr', attribs={'class': 'chart-table-column-headers'}
    )
    header_values = test_utils.get_table_row_values(header_row, 'th')
    self.assertSequenceEqual(
        header_values,
        [
            summary_text.DATASET_LABEL,
            summary_text.R_SQUARED_LABEL,
            summary_text.MAPE_LABEL,
            summary_text.WMAPE_LABEL,
        ],
    )

    value_rows = table.findall('tr')[1:]
    self.assertLen(value_rows, 3)

    value_rows = [test_utils.get_table_row_values(row) for row in value_rows]
    for row_values in value_rows:
      self.assertLen(row_values, 4)

    rows_labels = [row[0] for row in value_rows]
    self.assertSequenceEqual(
        rows_labels,
        [
            summary_text.TRAINING_DATA_LABEL,
            summary_text.TESTING_DATA_LABEL,
            summary_text.ALL_DATA_LABEL,
        ],
    )

  def test_media_summary_with_custom_date_range(self):
    self.summarizer_revenue._meridian.expand_selected_time_dims.return_value = [
        '2022-06-04',
        '2022-06-11',
        '2022-06-18',
        '2022-06-25',
        '2022-07-02',
        '2022-07-09',
        '2022-07-16',
        '2022-07-23',
        '2022-07-30',
        '2022-08-06',
        '2022-08-13',
        '2022-08-20',
        '2022-08-27',
    ]

    _ = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
        start_date=dt.datetime(2022, 6, 4),
        end_date=dt.datetime(2022, 8, 27),
    )

    self.media_summary_class.assert_called_with(
        self.mock_meridian_revenue,
        selected_times=[
            '2022-06-04',
            '2022-06-11',
            '2022-06-18',
            '2022-06-25',
            '2022-07-02',
            '2022-07-09',
            '2022-07-16',
            '2022-07-23',
            '2022-07-30',
            '2022-08-06',
            '2022-08-13',
            '2022-08-20',
            '2022-08-27',
        ],
    )

  def test_media_effects_with_custom_date_range(self):
    media_effects = self.media_effects
    mock_spec_1 = 'response_curves'

    with mock.patch.object(media_effects, 'plot_response_curves') as plot:
      plot().to_json.return_value = f'["{mock_spec_1}"]'

      self.summarizer_revenue._meridian.expand_selected_time_dims.return_value = [
          '2022-06-04',
          '2022-06-11',
          '2022-06-18',
          '2022-06-25',
          '2022-07-02',
          '2022-07-09',
          '2022-07-16',
          '2022-07-23',
          '2022-07-30',
      ]

      _ = self._get_output_model_results_summary_html_dom(
          self.summarizer_revenue,
          start_date=dt.datetime(2022, 6, 4),
          end_date=dt.datetime(2022, 7, 30),
      )
      self.media_effects_class.assert_called_with(self.mock_meridian_revenue)
      plot.assert_called_with(
          confidence_level=c.DEFAULT_CONFIDENCE_LEVEL,
          selected_times=frozenset([
              '2022-06-04',
              '2022-06-11',
              '2022-06-18',
              '2022-06-25',
              '2022-07-02',
              '2022-07-09',
              '2022-07-16',
              '2022-07-23',
              '2022-07-30',
          ]),
          plot_separately=False,
          include_ci=False,
          num_channels_displayed=7,
      )
      plot().to_json.assert_called_once()

  def test_reach_frequency_with_custom_date_range(self):
    self.summarizer_revenue._meridian.expand_selected_time_dims.return_value = [
        '2022-06-04',
        '2022-06-11',
        '2022-06-18',
        '2022-06-25',
        '2022-07-02',
        '2022-07-09',
        '2022-07-16',
        '2022-07-23',
        '2022-07-30',
    ]

    _ = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
        start_date=dt.datetime(2022, 6, 4),
        end_date=dt.datetime(2022, 7, 30),
    )

    self.reach_frequency_class.assert_called_with(
        self.mock_meridian_revenue,
        selected_times=[
            '2022-06-04',
            '2022-06-11',
            '2022-06-18',
            '2022-06-25',
            '2022-07-02',
            '2022-07-09',
            '2022-07-16',
            '2022-07-23',
            '2022-07-30',
        ],
    )

  def test_channel_contrib_card_plotters_called(self):
    media_summary = self.media_summary

    mock_spec_1 = 'revenue_waterfall'
    media_summary.plot_contribution_waterfall_chart().to_json.return_value = (
        f'["{mock_spec_1}"]'
    )
    mock_spec_2 = 'spend_contrib'
    media_summary.plot_spend_vs_contribution().to_json.return_value = (
        f'["{mock_spec_2}"]'
    )
    mock_spec_3 = 'revenue_contrib_pie'
    media_summary.plot_contribution_pie_chart().to_json.return_value = (
        f'["{mock_spec_3}"]'
    )
    mock_spec_4 = 'revenue_area_chart'
    media_summary.plot_channel_contribution_area_chart().to_json.return_value = (
        f'["{mock_spec_4}"]'
    )

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )
    for mock_plot in [
        media_summary.plot_contribution_waterfall_chart(),
        media_summary.plot_spend_vs_contribution(),
        media_summary.plot_contribution_pie_chart(),
        media_summary.plot_channel_contribution_area_chart(),
    ]:
      mock_plot.to_json.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.CHANNEL_CONTRIB_CARD_ID},
    )
    script_texts = [
        script.text.strip()
        for script in card.findall('charts/script')
        if script.text is not None
    ]

    mock_spec_1_exists = any(
        [mock_spec_1 in script_text for script_text in script_texts]
    )
    mock_spec_2_exists = any(
        [mock_spec_2 in script_text for script_text in script_texts]
    )
    mock_spec_3_exists = any(
        [mock_spec_3 in script_text for script_text in script_texts]
    )
    mock_spec_4_exists = any(
        [mock_spec_4 in script_text for script_text in script_texts]
    )
    self.assertTrue(
        all([
            mock_spec_1_exists,
            mock_spec_2_exists,
            mock_spec_3_exists,
            mock_spec_4_exists,
        ])
    )

  def test_channel_contrib_card_insights(self):
    self.media_metrics[c.INCREMENTAL_OUTCOME].loc[{
        c.CHANNEL: 'rf_ch_1',
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }] = 999999  # largest outcome
    self.media_metrics[c.INCREMENTAL_OUTCOME].loc[{
        c.CHANNEL: 'ch_0',
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }] = 888888  # 2nd largest outcome

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.CHANNEL_CONTRIB_CARD_ID},
    )
    insights_text = test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text

    self.assertIn('Rf_Ch_1 and Ch_0 drove the most', insights_text)

  def test_performance_breakdown_card_plotters_called(self):
    media_summary = self.media_summary

    mock_spec_1 = 'roi_effectiveness'
    media_summary.plot_roi_vs_effectiveness().to_json.return_value = (
        f'["{mock_spec_1}"]'
    )
    mock_spec_2 = 'roi_mroi'
    media_summary.plot_roi_vs_mroi().to_json.return_value = f'["{mock_spec_2}"]'
    mock_spec_3 = 'roi_bar'
    media_summary.plot_roi_bar_chart().to_json.return_value = (
        f'["{mock_spec_3}"]'
    )
    mock_spec_4 = 'cpik_bar'
    media_summary.plot_cpik().to_json.return_value = f'["{mock_spec_4}"]'

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )
    for mock_plot in [
        media_summary.plot_roi_vs_effectiveness(),
        media_summary.plot_roi_vs_mroi(),
        media_summary.plot_roi_bar_chart(),
        media_summary.plot_cpik(),
    ]:
      mock_plot.to_json.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.PERFORMANCE_BREAKDOWN_CARD_ID},
    )
    script_texts = [script.text for script in card.findall('charts/script')]

    mock_spec_1_exists = any(
        [mock_spec_1 in script_text for script_text in script_texts]
    )
    mock_spec_2_exists = any(
        [mock_spec_2 in script_text for script_text in script_texts]
    )
    mock_spec_3_exists = any(
        [mock_spec_3 in script_text for script_text in script_texts]
    )
    mock_spec_4_exists = any(
        [mock_spec_4 in script_text for script_text in script_texts]
    )
    self.assertTrue(
        all([
            mock_spec_1_exists,
            mock_spec_2_exists,
            mock_spec_3_exists,
            mock_spec_4_exists,
        ])
    )

  def test_performance_breakdown_card_insights(self):
    high_roi = high_mroi = 999999
    low_cpik = 0.01
    self.media_metrics[c.ROI].loc[{
        c.CHANNEL: 'ch_0',
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }] = high_roi
    self.media_metrics[c.MROI].loc[{
        c.CHANNEL: 'ch_0',
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }] = high_mroi
    self.media_metrics[c.CPIK].loc[{
        c.CHANNEL: 'ch_1',
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEDIAN,
    }] = low_cpik
    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.PERFORMANCE_BREAKDOWN_CARD_ID},
    )
    insights_text = test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    insights_text = insights_text.strip()

    self.assertIn(
        f'Ch_0 drove the highest ROI at {high_roi:.1f}', insights_text
    )
    self.assertIn('Rf_Ch_0 had the highest effectiveness', insights_text)
    self.assertIn(
        f'Ch_0 had the highest marginal\nROI at {high_mroi:.2f}',
        insights_text,
    )
    self.assertIn(
        f'Ch_1 drove the lowest CPIK\nat ${low_cpik:.2f}',
        insights_text,
    )
    self.assertIn(
        f'For every KPI unit, you spent ${low_cpik:.2f}',
        insights_text,
    )

  def test_response_curves_card_plotters_called(self):
    media_effects = self.media_effects
    reach_frequency = self.reach_frequency

    mock_spec_1 = 'response_curves'
    mock_spec_2 = 'optimal_frequency'
    reach_frequency.plot_optimal_frequency().to_json.return_value = (
        f'["{mock_spec_2}"]'
    )

    with mock.patch.object(media_effects, 'plot_response_curves') as plot:
      plot().to_json.return_value = f'["{mock_spec_1}"]'

      summary_html_dom = self._get_output_model_results_summary_html_dom(
          self.summarizer_revenue,
      )

      plot.assert_called_with(
          confidence_level=c.DEFAULT_CONFIDENCE_LEVEL,
          selected_times=frozenset(
              self.summarizer_revenue._meridian.input_data.time.values
          ),
          plot_separately=False,
          include_ci=False,
          num_channels_displayed=7,
      )
      plot().to_json.assert_called_once()

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.RESPONSE_CURVES_CARD_ID},
    )
    script_texts = [
        script.text.strip()
        for script in card.findall('charts/script')
        if script.text is not None
    ]
    mock_spec_1_exists = any(
        [mock_spec_1 in script_text for script_text in script_texts]
    )
    mock_spec_2_exists = any(
        [mock_spec_2 in script_text for script_text in script_texts]
    )
    self.assertTrue(all([mock_spec_1_exists, mock_spec_2_exists]))

  def test_response_curves_card_insights_multiple_channels(self):
    media_summary = self.media_summary
    reach_frequency = self.reach_frequency

    media_summary.get_paid_summary_metrics().spend.data = [
        100,  # 'ch_0'
        200,  # 'ch_1'
        300,  # 'ch_2'
        400,  # 'rf_ch_0'
        500,  # 'rf_ch_1'
        1500,  # 'All Channels'
    ]
    reach_frequency.optimal_frequency_data.optimal_frequency.data = [
        1.23,  # 'rf_ch_0'
        2.34,  # 'rf_ch_1' << this should be selected for plotting
    ]

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.RESPONSE_CURVES_CARD_ID},
    )
    insights_text = test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertIn('for rf_ch_1 is 2.3 ', insights_text.replace('\n', ' '))
    self.assertIn('to maximize ROI', insights_text.replace('\n', ' '))

  def test_response_curves_card_insights_no_rf(self):
    self.mock_meridian_revenue.n_rf_channels = 0
    reach_frequency = self.reach_frequency
    reach_frequency.optimal_frequency_data = xr.Dataset(
        data_vars={
            c.OPTIMAL_FREQUENCY: (
                [c.RF_CHANNEL],
                [],
            ),
        },
        coords={
            c.FREQUENCY: [],
            c.RF_CHANNEL: [],
        },
    )

    summary_html_dom = self._get_output_model_results_summary_html_dom(
        self.summarizer_revenue,
    )

    card = test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.RESPONSE_CURVES_CARD_ID},
    )
    insights_text = test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIn(
        summary_text.RESPONSE_CURVES_INSIGHTS_FORMAT.format(outcome=c.REVENUE),
        insights_text,
    )


if __name__ == '__main__':
  absltest.main()
