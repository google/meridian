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

from absl.testing import absltest
import altair as alt
from meridian import constants as c
from meridian.analysis import _changepoint_plots
from meridian.analysis import test_utils
from meridian.data import time_coordinates as tc
import numpy as np
import pandas as pd


def _weekly() -> tc.TimeCoordinates:
  """Five weekly time coordinates from 2024-01-01 to 2024-01-29."""
  return tc.TimeCoordinates.from_dates(
      pd.date_range("2024-01-01", periods=5, freq="W-MON")
  )


# The channels aren't in alphabetical order. TV's middle interval has no spend,
# YouTube has two intervals, and radio has no spend in its only interval.
_INTERVALS_BY_CHANNEL = {
    "youtube": [
        ("2024-01-01", "2024-01-15", 1.0),
        ("2024-01-22", "2024-01-29", 4.0),
    ],
    "tv": [
        ("2024-01-01", "2024-01-08", 2.0),
        ("2024-01-15", "2024-01-22", None),
        ("2024-01-29", "2024-01-29", 3.0),
    ],
    "radio": [("2024-01-01", "2024-01-29", None)],
}
_ROI_BY_INTERVAL = test_utils.generate_metric_by_changepoint_interval(
    _INTERVALS_BY_CHANNEL
)


class IntervalDfTest(absltest.TestCase):

  def test_interval_df(self):
    pd.testing.assert_frame_equal(
        _changepoint_plots.interval_df(_ROI_BY_INTERVAL, c.ROI, _weekly()),
        pd.DataFrame({
            c.CHANNEL: ["youtube", "youtube", "tv", "tv", "tv", "radio"],
            c.START_DATE: pd.to_datetime([
                "2024-01-01",
                "2024-01-22",
                "2024-01-01",
                "2024-01-15",
                "2024-01-29",
                "2024-01-01",
            ]),
            c.END_DATE: pd.to_datetime([
                "2024-01-21",
                "2024-02-04",
                "2024-01-14",
                "2024-01-28",
                "2024-02-04",
                "2024-02-04",
            ]),
            # Each interval ends where the next one starts.
            _changepoint_plots.INTERVAL_END: pd.to_datetime([
                "2024-01-22",
                "2024-02-05",
                "2024-01-15",
                "2024-01-29",
                "2024-02-05",
                "2024-02-05",
            ]),
            # TV's staircase breaks at its interval without spend.
            _changepoint_plots.SEGMENT: [0, 0, 0, 1, 1, 1],
            # Intervals without spend have NaN values.
            c.MEAN: [1.0, 4.0, 2.0, np.nan, 3.0, np.nan],
            c.CI_LO: [0.0, 3.0, 1.0, np.nan, 2.0, np.nan],
            c.CI_HI: [2.0, 5.0, 3.0, np.nan, 4.0, np.nan],
        }),
    )

  def test_interval_df_with_monthly_time_periods(self):
    time_coordinates = tc.TimeCoordinates.from_dates(
        ["2024-01-01", "2024-02-01", "2024-03-01"]
    )
    roi_by_interval = test_utils.generate_metric_by_changepoint_interval(
        {"tv": [("2024-02-01", "2024-03-01", 2.0)]}
    )

    df = _changepoint_plots.interval_df(
        roi_by_interval, c.ROI, time_coordinates
    )

    self.assertEqual(df[c.END_DATE].tolist(), [pd.Timestamp("2024-03-31")])
    self.assertEqual(
        df[_changepoint_plots.INTERVAL_END].tolist(),
        [pd.Timestamp("2024-04-01")],
    )

  def test_interval_df_with_an_infinite_metric(self):
    # A metric can be infinite, for example the ROI of an interval without
    # spend when earlier media still adds incremental outcome in it, as in
    # `Analyzer.roi(selected_times=...)`. -inf is blank too.
    roi_by_interval = test_utils.generate_metric_by_changepoint_interval({
        "tv": [
            ("2024-01-01", "2024-01-08", 2.0),
            ("2024-01-15", "2024-01-22", 0.0),
            ("2024-01-29", "2024-01-29", 3.0),
        ],
        "youtube": [("2024-01-01", "2024-01-29", 0.0)],
    })
    roi_by_interval[c.ROI].loc[
        {c.CHANNEL: "tv", c.CHANGEPOINT_INTERVAL: 1}
    ] = np.inf
    roi_by_interval[c.ROI].loc[
        {c.CHANNEL: "youtube", c.CHANGEPOINT_INTERVAL: 0}
    ] = -np.inf

    df = _changepoint_plots.interval_df(roi_by_interval, c.ROI, _weekly())

    # Intervals whose ROI isn't finite are blank, and break the staircase.
    pd.testing.assert_frame_equal(
        df[[_changepoint_plots.SEGMENT, c.MEAN, c.CI_LO, c.CI_HI]],
        pd.DataFrame({
            _changepoint_plots.SEGMENT: [0, 1, 1, 1],
            c.MEAN: [2.0, np.nan, 3.0, np.nan],
            c.CI_LO: [1.0, np.nan, 2.0, np.nan],
            c.CI_HI: [3.0, np.nan, 4.0, np.nan],
        }),
    )


class PlotIntervalMetricTest(absltest.TestCase):

  def test_plot_interval_metric(self):
    cpik_by_interval = test_utils.generate_metric_by_changepoint_interval(
        _INTERVALS_BY_CHANNEL, metric=c.CPIK
    )

    chart = _changepoint_plots.plot_interval_metric(
        cpik_by_interval,
        metric=c.CPIK,
        label="CPIK",
        title="CPIK by changepoint-defined time interval",
        time_coordinates=_weekly(),
    )

    self.assertIsInstance(chart, alt.FacetChart)
    pd.testing.assert_frame_equal(
        chart.data,
        _changepoint_plots.interval_df(cpik_by_interval, c.CPIK, _weekly()),
    )
    spec = chart.to_dict()
    self.assertEqual(
        spec["title"]["text"], "CPIK by changepoint-defined time interval"
    )
    # One subplot for each channel, in the order of the dataset.
    self.assertEqual(spec["facet"]["row"]["field"], c.CHANNEL)
    self.assertEqual(spec["facet"]["row"]["sort"], ["youtube", "tv", "radio"])
    self.assertEqual(spec["resolve"]["scale"]["y"], c.INDEPENDENT)
    # Intervals whose metric isn't finite have no values, so they are blank.
    (rows,) = spec["datasets"].values()
    self.assertEqual(
        [row[c.MEAN] for row in rows if row[c.CHANNEL] == "tv"],
        [2.0, None, 3.0],
    )
    credible_intervals, steps = spec["spec"]["layer"]
    for layer in (credible_intervals, steps):
      # The y axis title is along the axis of each subplot.
      self.assertEqual(layer["encoding"]["y"]["title"], "CPIK")
      self.assertEqual(layer["encoding"]["y"]["axis"], {"gridDash": [3, 2]})
      # The time axis covers all time periods of the training data, in the same
      # time zone as the data.
      self.assertEqual(
          layer["encoding"]["x"]["scale"]["domain"],
          ["2024-01-01T00:00:00", "2024-02-05T00:00:00"],
      )
      # Dates as year-month-day, and no vertical grid lines.
      self.assertEqual(
          layer["encoding"]["x"]["axis"],
          {"format": c.DATE_FORMAT, "grid": False, "domainColor": c.GREY_300},
      )
    # A shaded band for the credible interval of each interval, from its first
    # time period to the end of its last.
    self.assertEqual(
        credible_intervals["mark"],
        {"type": "rect", "color": c.BLUE_300, "opacity": 0.3},
    )
    self.assertEqual(credible_intervals["encoding"]["x"]["field"], c.START_DATE)
    self.assertEqual(
        credible_intervals["encoding"]["x2"],
        {"field": _changepoint_plots.INTERVAL_END},
    )
    self.assertEqual(credible_intervals["encoding"]["y"]["field"], c.CI_LO)
    self.assertEqual(credible_intervals["encoding"]["y2"]["field"], c.CI_HI)
    # One line through the start and the end of each interval, in time order,
    # at the mean. The end of an interval is the start of the next, so
    # consecutive steps are joined by a vertical line.
    self.assertEqual(
        steps["mark"], {"type": "line", "color": c.BLUE_600, "strokeWidth": 2}
    )
    self.assertEqual(
        steps["transform"],
        [{
            "fold": [c.START_DATE, _changepoint_plots.INTERVAL_END],
            "as": [_changepoint_plots.STEP_POINT, _changepoint_plots.STEP_DATE],
        }],
    )
    self.assertEqual(
        steps["encoding"]["x"]["field"], _changepoint_plots.STEP_DATE
    )
    self.assertEqual(steps["encoding"]["y"]["field"], c.MEAN)
    self.assertEqual(
        steps["encoding"]["order"],
        [
            {"field": _changepoint_plots.STEP_DATE, "type": "temporal"},
            {"field": c.START_DATE, "type": "temporal"},
        ],
    )
    # The line breaks at intervals whose metric isn't finite.
    self.assertEqual(
        steps["encoding"]["detail"],
        {"field": _changepoint_plots.SEGMENT, "type": "nominal"},
    )
    # Only the band has a tooltip, for each interval.
    self.assertNotIn("tooltip", steps["encoding"])
    self.assertEqual(
        credible_intervals["encoding"]["tooltip"],
        [
            {"field": c.CHANNEL, "title": "Channel", "type": "nominal"},
            {
                "field": c.START_DATE,
                "format": c.DATE_FORMAT,
                "title": "Start",
                "type": "temporal",
            },
            {
                "field": c.END_DATE,
                "format": c.DATE_FORMAT,
                "title": "End",
                "type": "temporal",
            },
            {
                "field": c.MEAN,
                "format": ".2f",
                "title": "CPIK",
                "type": "quantitative",
            },
            {
                "field": c.CI_LO,
                "format": ".2f",
                "title": "CI Low",
                "type": "quantitative",
            },
            {
                "field": c.CI_HI,
                "format": ".2f",
                "title": "CI High",
                "type": "quantitative",
            },
        ],
    )


if __name__ == "__main__":
  absltest.main()
