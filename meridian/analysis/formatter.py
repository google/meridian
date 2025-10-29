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

"""Functions for formatting analysis outputs."""

from collections.abc import Sequence
import dataclasses
import math
import os

import altair as alt
import immutabledict
import jinja2
from meridian import constants as c


__all__ = [
    'CardSpec',
    'ChartSpec',
    'TableSpec',
    'StatsSpec',
    'create_template_env',
    'create_card_html',
]


@dataclasses.dataclass(frozen=True)
class CardSpec:
  id: str
  title: str


@dataclasses.dataclass(frozen=True)
class ChartSpec:
  id: str
  chart_json: str
  description: str | None = None


@dataclasses.dataclass(frozen=True)
class TableSpec:
  id: str
  title: str
  column_headers: Sequence[str]
  row_values: Sequence[Sequence[str]]
  description: str | None = None


@dataclasses.dataclass(frozen=True)
class StatsSpec:
  title: str
  stat: str
  delta: str | None = None


TEXT_CONFIG = immutabledict.immutabledict({
    'titleFont': c.FONT_ROBOTO,
    'labelFont': c.FONT_ROBOTO,
    'titleFontWeight': 'normal',
    'titleFontSize': c.AXIS_FONT_SIZE,
    'labelFontSize': c.AXIS_FONT_SIZE,
    'titleColor': c.GREY_700,
    'labelColor': c.GREY_700,
})

Y_AXIS_TITLE_CONFIG = immutabledict.immutabledict({
    'titleAngle': 0,
    'titleAlign': 'left',
    'titleY': -20,
})

AXIS_CONFIG = immutabledict.immutabledict({
    'ticks': False,
    'labelPadding': c.PADDING_10,
    'domainColor': c.GREY_300,
})


_template_loader = jinja2.FileSystemLoader(
    os.path.abspath(os.path.dirname(__file__)) + '/templates'
)


def custom_title_params(title: str) -> alt.TitleParams:
  """Formats the title to be at the top left of the plot."""
  return alt.TitleParams(
      text=title,
      anchor='start',
      fontSize=c.TITLE_FONT_SIZE,
      font=c.FONT_GOOGLE_SANS_DISPLAY,
      fontWeight='normal',
      offset=c.PADDING_10,
      color=c.GREY_800,
  )


def bar_chart_width(num_bars: int) -> int:
  """Returns the width for a bar chart based on the number of bars."""
  return (c.BAR_SIZE + c.PADDING_20) * num_bars


def format_percent(percent: float) -> str:
  """Formats a percentage value into a string format.

  Percentage values between 0 and 1 are formatted with 1 decimal place.
  Percentage values greater than 1 are formatted with 0 decimal places.

  Args:
    percent: The percentage value to format.

  Returns:
    A formatted string.
  """
  if percent >= 0.01:
    return '{:.0%}'.format(percent)
  else:
    return '{:.1g}%'.format(percent * 100)


def compact_number(n: float, precision: int = 0, currency: str = '') -> str:
  """Formats a number into a compact notation to the specified precision.

  Ex. $15M

  Args:
    n: The number to format.
    precision: The number of decimals to use when rounding.
    currency: Optional string currency character. This is added at the beginning
      of the formatted string.

  Returns:
    A formatted string.
  """
  millnames = ['', 'k', 'M', 'B', 'T']
  millidx = max(
      0,
      min(
          len(millnames) - 1,
          int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
      ),
  )
  result = '{:.{precision}f}'.format(
      n / 10 ** (3 * millidx), precision=precision
  )
  suffixed = '{0}{dx}'.format(result, dx=millnames[millidx])

  # For negative numbers, add the currency after the negative sign.
  if n < 0:
    return suffixed[0] + currency + suffixed[1:]
  return currency + suffixed


def compact_number_expr(value: str = 'value', n_sig_digits: int = 3) -> str:
  """Returns the Vega expression to format the datum value with SI-prefixes.

  The scientific notation prefixes (k, M, G, T) are used for numeric values to
  make the numbers easier to read. Note, G for giga is replaced with B for
  billion.

  Args:
    value: Datum value to format.
    n_sig_digits: The number of significant digits for the formatted number.

  Returns: The Vega expression string to format the text into a compact form.
  """
  return f"replace(format(datum.{value}, '.{n_sig_digits}~s'), 'G', 'B')"


def format_number_text(percent_value: float, actual_value: float) -> str:
  """Formats the percent and actual value into a human-readable compact string.

  Ex. 40.2% (15M)

  Args:
    percent_value: Float between 0 and 1 representing a percentage value.
    actual_value: Float to display next to the percentage value.

  Returns:
    String with the percentage and the compact notation of the actual value.
  """
  return f'{round(percent_value * 100, 1)}% ({compact_number(actual_value)})'


def format_monetary_num(num: float, currency: str) -> str:
  """Formats a number into a readable monetary value (ex. $15M, $1.2B)."""
  precision = 1 if num != 0 and int(math.log10(abs(num))) % 3 == 0 else 0
  return compact_number(num, precision=precision, currency=currency)


def create_template_env() -> jinja2.Environment:
  """Creates a Jinja2 template environment."""
  return jinja2.Environment(
      loader=_template_loader,
      autoescape=jinja2.select_autoescape(),
  )


def create_card_html(
    template_env: jinja2.Environment,
    card_spec: CardSpec,
    insights: str,
    chart_specs: Sequence[ChartSpec | TableSpec] | None = None,
    stats_specs: Sequence[StatsSpec] | None = None,
) -> str:
  """Creates a card's HTML snippet that includes given card and chart specs."""
  insights_html = template_env.get_template('insights.html.jinja').render(
      text_html=insights
  )
  card_params = dataclasses.asdict(card_spec)
  card_params[c.CARD_CHARTS] = (
      _create_charts_htmls(template_env, chart_specs) if chart_specs else None
  )
  card_params[c.CARD_INSIGHTS] = insights_html
  card_params[c.CARD_STATS] = (
      _create_stats_htmls(template_env, stats_specs) if stats_specs else None
  )
  return template_env.get_template('card.html.jinja').render(card_params)


def _create_stats_htmls(
    template_env: jinja2.Environment, specs: Sequence[StatsSpec]
) -> Sequence[str]:
  """Creates a list of stats HTML snippets given a list of stats specs."""
  stats_htmls = []
  for spec in specs:
    stats_htmls.append(
        template_env.get_template('stats.html.jinja').render(
            dataclasses.asdict(spec)
        )
    )
  return stats_htmls


def _create_charts_htmls(
    template_env: jinja2.Environment,
    specs: Sequence[ChartSpec | TableSpec],
) -> Sequence[str]:
  """Creates a list of chart HTML snippets given a list of chart specs."""
  chart_template = template_env.get_template('chart.html.jinja')
  table_template = template_env.get_template('table.html.jinja')
  htmls = []
  for spec in specs:
    if isinstance(spec, ChartSpec):
      htmls.append(chart_template.render(dataclasses.asdict(spec)))
    else:
      htmls.append(table_template.render(dataclasses.asdict(spec)))
  return htmls
