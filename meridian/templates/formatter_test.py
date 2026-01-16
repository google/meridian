# Copyright 2025 The Meridian Authors.
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

from xml.etree import ElementTree as ET

from absl.testing import absltest
from absl.testing import parameterized
from meridian.templates import formatter


class FormatterTest(parameterized.TestCase):

  def test_custom_title_params_correct(self):
    title_params = formatter.custom_title_params('test title')
    self.assertEqual(
        title_params.to_dict(),
        {
            'anchor': 'start',
            'color': '#3C4043',
            'font': 'Google Sans Display',
            'fontSize': 18,
            'fontWeight': 'normal',
            'offset': 10,
            'text': 'test title',
        },
    )

  def test_bar_chart_width(self):
    num_bars = 3
    width = formatter.bar_chart_width(num_bars)
    self.assertEqual(width, 186)

  @parameterized.named_parameters(
      ('zero_percent', 0.0, '0%'),
      ('less_than_one_percent', 0.0005, '0.05%'),
      ('one_percent', 0.01, '1%'),
      ('greater_than_one_percent', 0.4257, '43%'),
  )
  def test_format_percent_correct(self, percent, expected):
    formatted_percent = formatter.format_percent(percent)
    self.assertEqual(formatted_percent, expected)

  def test_compact_number_expr_default(self):
    expr = formatter.compact_number_expr()
    self.assertEqual(expr, "replace(format(datum.value, '.3~s'), 'G', 'B')")

  def test_compact_number_expr_params(self):
    expr = formatter.compact_number_expr('other', 2)
    self.assertEqual(expr, "replace(format(datum.other, '.2~s'), 'G', 'B')")

  @parameterized.named_parameters(
      ('rounded_up_percent', 0.4257, 15, '42.6% (15)'),
      ('rounded_down_percent', 0.4251, 15, '42.5% (15)'),
      ('thousand_value', 0.42, 2e4, '42.0% (20k)'),
      ('million_value', 0.42, 3e7, '42.0% (30M)'),
      ('billion_value', 0.42, 4e9, '42.0% (4B)'),
  )
  def test_format_number_text_correct(self, percent, value, expected):
    formatted_text = formatter.format_number_text(percent, value)
    self.assertEqual(formatted_text, expected)

  @parameterized.named_parameters(
      ('zero_precision_thousands', 12345, '$', '$12k'),
      ('round_up_thousands', 14900, '€', '€15k'),
      ('million_value', 3.21e6, '£', '£3.2M'),
      ('billion_value_round_up', 4.28e9, '¥', '¥4.3B'),
      ('negative', -12345, '₮', '-₮12k'),
  )
  def test_format_monetary_num_correct(self, num, currency, expected):
    formatted_number = formatter.format_monetary_num(num, currency)
    self.assertEqual(formatted_number, expected)

  @parameterized.named_parameters(
      ('decimals', -0.1234, 2, '$', '-$0.12'),
      ('zero_precision_thousands', 12345, 0, '', '12k'),
      ('round_up_thousands', 14900, 0, '$', '$15k'),
      ('million_value', 3.21e6, 2, '$', '$3.21M'),
      ('negative', -12345, 0, '$', '-$12k'),
  )
  def test_compact_number_correct(self, num, precision, currency, expected):
    formatted_number = formatter.compact_number(num, precision, currency)
    self.assertEqual(formatted_number, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_snake_case',
          input_headers=['finding_cause'],
          expected=['Finding Cause'],
      ),
      dict(
          testcase_name='multiple_columns',
          input_headers=['geo', 'time_index', 'channel_name'],
          expected=['Geo', 'Time Index', 'Channel Name'],
      ),
      dict(
          testcase_name='preserves_acronyms',
          input_headers=['VIF_score', 'national_KPI'],
          expected=['VIF Score', 'National KPI'],
      ),
      dict(
          testcase_name='handles_tuples_input',
          input_headers=('row_id', 'value'),
          expected=['Row Id', 'Value'],
      ),
      dict(
          testcase_name='empty_input',
          input_headers=[],
          expected=[],
      ),
  )
  def test_format_col_names(self, input_headers, expected):
    self.assertEqual(formatter.format_col_names(input_headers), expected)

  def test_create_summary_html(self):
    template_env = formatter.create_template_env()
    title = 'Integration Test Report'
    cards = ['<card>Card 1</card>', '<card>Card 2</card>']

    html_result = formatter.create_summary_html(template_env, title, cards)

    # Since summary.html contains DOCTYPE (which breaks ElementTree XML parser),
    # we verify the output using string assertions.
    self.assertIn('<!DOCTYPE html>', html_result)
    self.assertIn(title, html_result)
    self.assertIn('<card>Card 1</card>', html_result)
    self.assertIn('<card>Card 2</card>', html_result)

  def test_create_card_html_structure(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    stats_spec = formatter.StatsSpec(title='stats_title', stat='test_stat')
    chart_spec = formatter.ChartSpec(
        'test_chart_id', 'test_chart_json', 'test_chart_description'
    )

    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', [chart_spec], [stats_spec]
        )
    )
    self.assertEqual(card_html.tag, 'card')
    self.assertLen(card_html, 4)
    self.assertEqual(card_html[0].tag, 'card-title')
    self.assertEqual(card_html[1].tag, 'card-insights')
    self.assertEqual(card_html[2].tag, 'stats-section')
    self.assertEqual(card_html[3].tag, 'charts')

  def test_create_card_html_text(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    chart_spec = formatter.ChartSpec(
        'test_chart_id', 'test_chart_json', 'test_chart_description'
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', [chart_spec]
        )
    )
    self.assertContainsSubset('test_title', card_html[0].text)
    self.assertContainsSubset('test_insights', card_html[1][1].text)

  def test_create_card_html_multiple_charts(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    chart_spec1 = formatter.ChartSpec(
        'test_chart_id1', 'test_chart_json1', 'test_chart_description1'
    )
    chart_spec2 = formatter.ChartSpec(
        'test_chart_id2', 'test_chart_json2', 'test_chart_description2'
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', [chart_spec1, chart_spec2]
        )
    )
    charts = card_html[2]
    self.assertLen(charts, 4)  # Each chart has 2 items, chart and script.

  def test_create_card_html_chart_structure(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    chart_spec = formatter.ChartSpec(
        'test_chart_id', 'test_chart_json', 'test_chart_description'
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', [chart_spec]
        )
    )
    chart_html = card_html[2]
    self.assertEqual(chart_html.tag, 'charts')
    self.assertEqual(chart_html[0].tag, 'chart')
    self.assertEqual(chart_html[0][0].tag, 'chart-embed')
    self.assertEqual(chart_html[0][1].tag, 'chart-description')
    self.assertContainsSubset('test_chart_description', chart_html[0][1].text)
    self.assertEqual(chart_html[1].tag, 'script')
    self.assertContainsSubset('test_chart_json', chart_html[1].text)

  def test_create_card_html_mulitple_stats(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    stat1 = formatter.StatsSpec(title='stats_title1', stat='test_stat1')
    stat2 = formatter.StatsSpec(title='stats_title2', stat='test_stat2')
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', stats_specs=[stat1, stat2]
        )
    )
    stats = card_html[2]
    self.assertLen(stats, 2)

  def test_create_card_html_stats_structure(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    stats_spec = formatter.StatsSpec(
        title='stats_title', stat='test_stat', delta='+0.3'
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, 'test_insights', stats_specs=[stats_spec]
        )
    )
    stats_html = card_html[2]
    self.assertEqual(stats_html.tag, 'stats-section')
    self.assertEqual(stats_html[0].tag, 'stats')
    self.assertEqual(stats_html[0][0].tag, 'stats-title')
    self.assertEqual(stats_html[0][0].text, 'stats_title')
    self.assertEqual(stats_html[0][1].tag, 'stat')
    self.assertEqual(stats_html[0][1].text, 'test_stat')
    self.assertEqual(stats_html[0][2].tag, 'delta')
    self.assertContainsSubset('+0.3', stats_html[0][2].text)

  def test_create_card_html_no_insights(self):
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    stats_spec = formatter.StatsSpec(title='stats_title', stat='test_stat')

    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, insights=None, stats_specs=[stats_spec]
        )
    )

    self.assertEqual(card_html.tag, 'card')
    self.assertIsNone(card_html.find('card-insights'))
    self.assertIsNotNone(card_html.find('stats-section'))

  def test_create_card_html_chart_findings(self):
    """Tests that errors, warnings, and infos render inside a chart."""
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    chart_spec = formatter.ChartSpec(
        id='id',
        chart_json='{}',
        errors=['Chart Error'],
        warnings=['Chart Warning'],
        infos=['Chart Info'],
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, insights=None, chart_specs=[chart_spec]
        )
    )

    charts_elem = card_html.find('charts')
    self.assertIsNotNone(charts_elem)
    chart_elem = charts_elem.find('chart')
    self.assertIsNotNone(chart_elem)

    error_elem = chart_elem.find('errors')
    self.assertIsNotNone(error_elem)
    error_p = error_elem.find('p')
    self.assertIsNotNone(error_p)
    self.assertIn('Chart Error', error_p.text)

    warning_elem = chart_elem.find('warnings')
    self.assertIsNotNone(warning_elem)
    warning_p = warning_elem.find('p')
    self.assertIsNotNone(warning_p)
    self.assertIn('Chart Warning', warning_p.text)

    info_elem = chart_elem.find('infos')
    self.assertIsNotNone(info_elem)
    info_p = info_elem.find('p')
    self.assertIsNotNone(info_p)
    self.assertIn('Chart Info', info_p.text)

  def test_create_card_html_table_findings(self):
    """Tests that errors, warnings, and infos render inside a table."""
    template_env = formatter.create_template_env()
    card_spec = formatter.CardSpec(id='test_id', title='test_title')
    table_spec = formatter.TableSpec(
        id='table_id',
        title='Table Title',
        column_headers=['Col1'],
        row_values=[['Val1']],
        errors=['Table Error'],
        warnings=['Table Warning'],
        infos=['Table Info'],
    )
    card_html = ET.fromstring(
        formatter.create_card_html(
            template_env, card_spec, insights=None, chart_specs=[table_spec]
        )
    )

    charts_elem = card_html.find('charts')
    self.assertIsNotNone(charts_elem)
    table_elem = charts_elem.find('chart-table')
    self.assertIsNotNone(table_elem)

    error_elem = table_elem.find('errors')
    self.assertIsNotNone(error_elem)
    error_p = error_elem.find('p')
    self.assertIsNotNone(error_p)
    self.assertIn('Table Error', error_p.text)

    warning_elem = table_elem.find('warnings')
    self.assertIsNotNone(warning_elem)
    warning_p = warning_elem.find('p')
    self.assertIsNotNone(warning_p)
    self.assertIn('Table Warning', warning_p.text)

    info_elem = table_elem.find('infos')
    self.assertIsNotNone(info_elem)
    info_p = info_elem.find('p')
    self.assertIsNotNone(info_p)
    self.assertIn('Table Info', info_p.text)


if __name__ == '__main__':
  absltest.main()
