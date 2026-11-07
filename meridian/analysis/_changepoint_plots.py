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

"""Plots of a metric of each changepoint-defined time interval, such as the ROI.

A channel with changepoints has a different effect in each changepoint-defined
time interval, and the same effect throughout an interval. The functions in this
module turn a metric of each time interval into a DataFrame and plot it over
time as a staircase, with one subplot for each channel. They are private helpers
of `visualizer.MediaSummary`.
"""

import altair as alt
from meridian import constants as c
from meridian.analysis import summary_text
from meridian.data import time_coordinates as tc
from meridian.templates import formatter
import numpy as np
import pandas as pd
import xarray as xr

# Columns of the DataFrame of a metric of each interval: the end of the
# interval's last time period, which is the day after its last day, and the
# part of a channel's staircase that the interval is on.
INTERVAL_END = 'interval_end'
SEGMENT = 'segment'
# Columns for the two ends of each interval that the staircase line goes
# through: which end it is (start or end) and its date.
STEP_POINT = 'step_point'
STEP_DATE = 'step_date'


def interval_df(
    metric_by_interval: xr.Dataset,
    metric: str,
    time_coordinates: tc.TimeCoordinates,
) -> pd.DataFrame:
  """Returns a metric of each changepoint-defined time interval, one row each.

  Intervals whose metric isn't finite (NaN or infinite), such as the ROI of an
  interval without spend, have NaN values. Each of them starts a new `segment`
  of the channel, so the staircase breaks at them.

  Args:
    metric_by_interval: A dataset like `Analyzer._roi_by_changepoint_interval`
      returns. The data variable named `metric` has the mean and credible
      interval of each channel and interval, and the `start_date` and `end_date`
      coordinates have the first and last time periods of each interval. A
      channel with fewer intervals than the others has empty dates for the extra
      intervals.
    metric: The name of the data variable in `metric_by_interval`, such as
      `roi`.
    time_coordinates: The time coordinates of the model's training data.

  Returns:
    A DataFrame with the columns `channel`, `start_date` and `end_date` (the
    first and last days of the interval), `interval_end` (the end of the
    interval's last time period), `segment`, `mean`, `ci_lo` and `ci_hi`. The
    rows are in the order of the channels in `metric_by_interval`, then of their
    intervals.
  """
  rows = []
  for channel in metric_by_interval[c.CHANNEL].values:
    segment = 0
    for interval in metric_by_interval[c.CHANGEPOINT_INTERVAL].values:
      selected = metric_by_interval.sel(
          {c.CHANNEL: channel, c.CHANGEPOINT_INTERVAL: interval}
      )
      # A channel with fewer intervals than the others has empty dates for the
      # intervals it doesn't have.
      if not selected[c.START_DATE].item():
        break
      values = selected[metric]
      mean = float(values.sel({c.METRIC: c.MEAN}))
      ci_lo = float(values.sel({c.METRIC: c.CI_LO}))
      ci_hi = float(values.sel({c.METRIC: c.CI_HI}))
      # The metric can be NaN or infinite, for example the ROI of an interval
      # without spend. The row stays, with NaN values, so that a channel whose
      # metric isn't finite in any interval still has a subplot.
      if not np.isfinite(mean):
        mean = ci_lo = ci_hi = np.nan
        segment += 1
      end = pd.Timestamp(
          time_coordinates.period_end(str(selected[c.END_DATE].item()))
      )
      rows.append({
          c.CHANNEL: str(channel),
          c.START_DATE: pd.Timestamp(str(selected[c.START_DATE].item())),
          c.END_DATE: end - pd.Timedelta(days=1),
          INTERVAL_END: end,
          SEGMENT: segment,
          c.MEAN: mean,
          c.CI_LO: ci_lo,
          c.CI_HI: ci_hi,
      })
  return pd.DataFrame(
      rows,
      columns=[
          c.CHANNEL,
          c.START_DATE,
          c.END_DATE,
          INTERVAL_END,
          SEGMENT,
          c.MEAN,
          c.CI_LO,
          c.CI_HI,
      ],
  )


def _time_axis(field: str, time_coordinates: tc.TimeCoordinates) -> alt.X:
  """Returns the time axis over all time periods of the training data."""
  dates = time_coordinates.all_dates
  return alt.X(
      f'{field}:T',
      title='Time period',
      axis=alt.Axis(format=c.DATE_FORMAT, grid=False, domainColor=c.GREY_300),
      # Timestamps without a time zone, like the data, so that the axis and
      # the steps use the same time zone.
      scale=alt.Scale(
          domain=[
              pd.Timestamp(dates[0]),
              pd.Timestamp(time_coordinates.period_end(dates[-1])),
          ]
      ),
  )


def plot_interval_metric(
    metric_by_interval: xr.Dataset,
    metric: str,
    label: str,
    title: str,
    time_coordinates: tc.TimeCoordinates,
) -> alt.FacetChart:
  """Plots a metric of each time interval, by channel.

  There is one subplot for each channel, in the order of `metric_by_interval`,
  and the subplots share the time axis. A channel's changepoints partition the
  training period into time intervals; each flat step shows the time interval's
  mean and shaded credible interval, from its first time period to the end of
  its last, with vertical lines connecting consecutive steps at changepoints.
  Intervals whose metric isn't finite (NaN or infinite), such as the ROI of an
  interval without spend, are left blank, and a channel whose metric isn't
  finite in any interval has an empty subplot. The time axis covers all time
  periods of the training data.

  Args:
    metric_by_interval: The metric of each interval, as in `interval_df`.
    metric: The name of the data variable in `metric_by_interval`, such as
      `roi`.
    label: The name of the metric shown on the plot, such as "ROI".
    title: The title of the plot.
    time_coordinates: The time coordinates of the model's training data.

  Returns:
    An Altair plot of the metric of each changepoint-defined time interval.
  """
  y_axis = alt.Axis(gridDash=[3, 2])
  tooltip = [
      alt.Tooltip(f'{c.CHANNEL}:N', title='Channel'),
      alt.Tooltip(f'{c.START_DATE}:T', title='Start', format=c.DATE_FORMAT),
      alt.Tooltip(f'{c.END_DATE}:T', title='End', format=c.DATE_FORMAT),
      alt.Tooltip(f'{c.MEAN}:Q', title=label, format='.2f'),
      alt.Tooltip(
          f'{c.CI_LO}:Q', title=summary_text.CI_LOW_LABEL, format='.2f'
      ),
      alt.Tooltip(
          f'{c.CI_HI}:Q', title=summary_text.CI_HIGH_LABEL, format='.2f'
      ),
  ]
  credible_intervals = (
      alt.Chart()
      .mark_rect(color=c.BLUE_300, opacity=0.3)
      .encode(
          x=_time_axis(c.START_DATE, time_coordinates),
          x2=alt.X2(f'{INTERVAL_END}:T'),
          y=alt.Y(f'{c.CI_LO}:Q', title=label, axis=y_axis),
          y2=alt.Y2(f'{c.CI_HI}:Q'),
          tooltip=tooltip,
      )
  )
  # One line through the start and the end of each interval, in time order.
  # The end of an interval is the start of the next, so the line goes up or
  # down there. It has no tooltip, because a line shows the tooltip of its
  # first point wherever it is hovered; the band around it has the tooltip of
  # each interval.
  steps = (
      alt.Chart()
      .transform_fold([c.START_DATE, INTERVAL_END], as_=[STEP_POINT, STEP_DATE])
      .mark_line(color=c.BLUE_600, strokeWidth=2)
      .encode(
          x=_time_axis(STEP_DATE, time_coordinates),
          y=alt.Y(f'{c.MEAN}:Q', title=label, axis=y_axis),
          # At a changepoint, the end of an interval comes before the start of
          # the next.
          order=[
              alt.Order(f'{STEP_DATE}:T'),
              alt.Order(f'{c.START_DATE}:T'),
          ],
          # The line doesn't join across intervals whose metric isn't finite.
          detail=f'{SEGMENT}:N',
      )
  )
  data = interval_df(metric_by_interval, metric, time_coordinates)
  return (
      alt.layer(credible_intervals, steps, data=data)
      .properties(width=c.VEGALITE_FACET_EXTRA_LARGE_WIDTH, height=120)
      .facet(
          row=alt.Row(
              f'{c.CHANNEL}:N',
              title=None,
              header=alt.Header(labelAngle=0, labelAlign='left'),
              # An explicit order, because Vega-Lite sorts the subplots by
              # channel name even with `sort=None`.
              sort=[str(ch) for ch in metric_by_interval[c.CHANNEL].values],
          )
      )
      .resolve_scale(y=c.INDEPENDENT)
      .properties(title=formatter.custom_title_params(title))
      .configure_axis(titlePadding=c.PADDING_10, **formatter.TEXT_CONFIG)
  )
