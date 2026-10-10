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

"""Tests for plotting and visualization functions."""

import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.analysis.review import constants
from meridian.analysis.review import plots
from meridian.analysis.review import results
import xarray as xr


class PlotsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("implausible_roi", plots.generate_implausible_roi_chart_json),
      ("high_variance", plots.generate_high_variance_chart_json),
      ("potential_bias", plots.generate_potential_bias_chart_json),
      (
          "calibration_details",
          plots.generate_calibration_details_chart_json,
      ),
      (
          "calibration_overview",
          plots.generate_calibration_overview_chart_json,
      ),
  )
  def test_generate_chart_json_none_input(self, generate_chart_json_fn):
    self.assertIsNone(generate_chart_json_fn(None))

  def test_generate_implausible_roi_chart_json_valid(self):
    mock_result = results.ImplausibleROICheckResult(
        case=results.ImplausibleROIAggregateCases.REVIEW,
        channel_results=[
            results.ImplausibleROIChannelResult(
                case=results.ImplausibleROIChannelCases.ROI_HIGH,
                channel_name="ch1",
                spend_share=0.5,
                roi_mean=30.0,
                spend_weighted_roi=15.0,
            ),
        ],
        high_roi_channels=["ch1"],
        low_roi_channels=[],
        aggregate_details={},
    )
    chart_json = plots.generate_implausible_roi_chart_json(mock_result)
    self.assertIsNotNone(chart_json)
    chart_dict = json.loads(chart_json)
    self.assertIn("$schema", chart_dict)
    self.assertIn(constants.IMPLAUSIBLE_HIGH_ROI, chart_json)
    self.assertIn(constants.IMPLAUSIBLE_LOW_ROI, chart_json)
    self.assertIn(constants.CHANNELS_LEGEND_TITLE, chart_json)
    self.assertIn(constants.DIAGNOSTIC_THRESHOLDS_TITLE, chart_json)

  def _implausible_roi_result(
      self, roi_means: Sequence[float], **kwargs
  ) -> results.ImplausibleROICheckResult:
    return results.ImplausibleROICheckResult(
        case=results.ImplausibleROIAggregateCases.REVIEW,
        channel_results=[
            results.ImplausibleROIChannelResult(
                case=results.ImplausibleROIChannelCases.ROI_PASS,
                channel_name=f"ch{i}",
                spend_share=0.1,
                roi_mean=roi_mean,
                spend_weighted_roi=0.1 * roi_mean,
            )
            for i, roi_mean in enumerate(roi_means)
        ],
        high_roi_channels=[],
        low_roi_channels=[],
        aggregate_details={},
        **kwargs,
    )

  def _load_chart(self, chart_json: str | None) -> Any:
    self.assertIsNotNone(chart_json)
    assert chart_json is not None
    return json.loads(chart_json)

  def _get_y_axis(self, chart_dict: Any) -> Any:
    return chart_dict["layer"][0]["encoding"]["y"]["axis"]

  def _get_plotted_roi_by_channel(self, chart_dict: Any) -> dict[str, float]:
    plotted = {}
    for rows in chart_dict["datasets"].values():
      for row in rows:
        if constants.CHANNEL_NAME in row and constants.ROI_MEAN in row:
          plotted[row[constants.CHANNEL_NAME]] = row[constants.Y_PLOTTED]
    return plotted

  def test_generate_implausible_roi_chart_json_default_axis(self):
    chart_dict = self._load_chart(
        plots.generate_implausible_roi_chart_json(
            self._implausible_roi_result([0.2, 10.0, 40.0])
        )
    )
    axis = self._get_y_axis(chart_dict)
    self.assertSequenceAlmostEqual(
        axis["values"],
        [
            0.0,
            0.2 * 19 / 0.6,
            0.4 * 19 / 0.6,
            19.0,
            20.0,
            40.0,
            60.0,
            80.0,
            100.0,
        ],
        places=5,
    )
    for label in ["'0.2'", "'0.4'", "'//'", "'20'", "'40'", "'100+'"]:
      self.assertIn(label, axis["labelExpr"])
    plotted = self._get_plotted_roi_by_channel(chart_dict)
    self.assertAlmostEqual(plotted["ch0"], 0.2 * 19 / 0.6)
    self.assertAlmostEqual(plotted["ch1"], 19.0)
    self.assertAlmostEqual(plotted["ch2"], 40.0)

  def test_generate_implausible_roi_chart_json_custom_thresholds_axis(self):
    chart_dict = self._load_chart(
        plots.generate_implausible_roi_chart_json(
            self._implausible_roi_result(
                [30.0, 100.0, 1000.0], roi_lower_bound=1.0, roi_upper_bound=50.0
            )
        )
    )
    axis = self._get_y_axis(chart_dict)
    label_expr = axis["labelExpr"]
    for label in ["'0.4'", "'0.8'", "'//'", "'50'", "'100'", "'250+'"]:
      self.assertIn(label, label_expr)
    for label in ["'0.2'", "'20'", "'100+'"]:
      self.assertNotIn(label, label_expr)
    self.assertIn(19.0, axis["values"])
    self.assertEqual(max(axis["values"]), 100.0)
    plotted = self._get_plotted_roi_by_channel(chart_dict)
    # ROI 30 is below the upper threshold, so it is clustered at the break.
    self.assertAlmostEqual(plotted["ch0"], 19.0)
    # ROI 100 is plotted at the position labelled '100'.
    plotted_100 = round(plotted["ch1"], 6)
    self.assertGreater(plotted_100, 19.0)
    self.assertIn(plotted_100, axis["values"])
    self.assertIn(f"datum.value == {plotted_100} ? '100'", label_expr)
    # ROIs above the top of the axis are capped.
    self.assertAlmostEqual(plotted["ch2"], 100.0)

  def test_generate_high_variance_chart_json_uses_threshold(self):
    mock_result = results.HighVarianceCheckResult(
        case=results.HighVarianceAggregateCases.REVIEW,
        channel_results=[
            results.HighVarianceChannelResult(
                case=results.HighVarianceChannelCases.HIGH_VARIANCE,
                channel_name="ch1",
                spend_share=0.5,
                relative_width_ratio=2.5,
            ),
        ],
        high_variance_channels=["ch1"],
        high_variance_threshold=2.0,
    )
    chart_dict = self._load_chart(
        plots.generate_high_variance_chart_json(mock_result)
    )
    y_at_full_spend = []
    for dataset_rows in chart_dict["datasets"].values():
      for row in dataset_rows:
        if (
            row.get(constants.REGION) == constants.HIGH_VARIANCE_ROI
            and row[constants.SPEND_SHARE] == 1.0
        ):
          y_at_full_spend.append(row[constants.Y_PLOTTED])
    self.assertNotEmpty(y_at_full_spend)
    self.assertAlmostEqual(y_at_full_spend[0], 2.0)

  def test_generate_high_variance_chart_json_valid(self):
    mock_result = results.HighVarianceCheckResult(
        case=results.HighVarianceAggregateCases.REVIEW,
        channel_results=[
            results.HighVarianceChannelResult(
                case=results.HighVarianceChannelCases.HIGH_VARIANCE,
                channel_name="ch1",
                spend_share=0.5,
                relative_width_ratio=2.5,
            ),
        ],
        high_variance_channels=["ch1"],
    )
    chart_json = plots.generate_high_variance_chart_json(mock_result)
    self.assertIsNotNone(chart_json)
    chart_dict = json.loads(chart_json)
    self.assertIn("$schema", chart_dict)
    self.assertIn(constants.HIGH_VARIANCE_ROI, chart_json)
    self.assertIn(constants.CHANNELS_LEGEND_TITLE, chart_json)
    self.assertIn(constants.DIAGNOSTIC_THRESHOLDS_TITLE, chart_json)

  def test_generate_potential_bias_chart_json_valid(self):
    da = xr.DataArray(
        [[0.8, 0.1]],
        dims=["geo", "channel_control"],
        coords={
            "geo": ["geo1"],
            "channel_control": ["ch1 - ctrl1", "ch1 - ctrl2"],
            "channel": ("channel_control", ["ch1", "ch1"]),
            "control_variable": ("channel_control", ["ctrl1", "ctrl2"]),
        },
    )
    mock_result = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.REVIEW,
        channel_results=[
            results.PotentialBiasChannelResult(
                case=results.PotentialBiasChannelCases.LOW_CORRELATION,
                channel_name="ch1",
                max_abs_correlation=0.1,
            ),
        ],
        low_correlation_channels=["ch1"],
        correlation_matrix=da,
    )
    chart_json = plots.generate_potential_bias_chart_json(mock_result)
    self.assertIsNotNone(chart_json)
    chart_dict = json.loads(chart_json)
    self.assertIn("$schema", chart_dict)
    self.assertIn(constants.INDIVIDUAL_GEO_CORRELATION, chart_json)
    self.assertIn(constants.MAX_ABS_CORRELATION, chart_json)
    self.assertIn(constants.PEARSON_CORRELATION_TITLE, chart_json)

  def test_generate_potential_bias_chart_json_uses_threshold(self):
    da = xr.DataArray(
        [[0.2, 0.8]],
        dims=["geo", "channel_control"],
        coords={
            "geo": ["geo1"],
            "channel_control": ["ch1 - ctrl1", "ch1 - ctrl2"],
            "channel": ("channel_control", ["ch1", "ch1"]),
            "control_variable": ("channel_control", ["ctrl1", "ctrl2"]),
        },
    )
    mock_result = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.PASS,
        channel_results=[
            results.PotentialBiasChannelResult(
                case=results.PotentialBiasChannelCases.ROI_PASS,
                channel_name="ch1",
                max_abs_correlation=0.8,
            ),
        ],
        low_correlation_channels=[],
        correlation_matrix=da,
        correlation_threshold=0.3,
    )
    chart_json = plots.generate_potential_bias_chart_json(mock_result)
    self.assertIsNotNone(chart_json)
    chart_dict = json.loads(chart_json)
    rows = []
    for dataset_rows in chart_dict["datasets"].values():
      rows.extend(dataset_rows)

    self.assertIn({constants.X1: -0.3, constants.X2: 0.3}, rows)
    rule_xs = sorted(
        row[constants.X] for row in rows if row.keys() == {constants.X}
    )
    self.assertEqual(rule_xs, [-0.3, 0.3])
    fill_colors = {
        row[constants.CONTROL_VARIABLE]: row[constants.FILL_COLOR]
        for row in rows
        if constants.FILL_COLOR in row
    }
    self.assertEqual(
        fill_colors,
        {
            "ctrl1": constants.POTENTIAL_BIAS_REVIEW_FILL_COLOR,
            "ctrl2": constants.POTENTIAL_BIAS_PASS_FILL_COLOR,
        },
    )

  @parameterized.parameters(ValueError, KeyError, AttributeError, TypeError)
  def test_generate_potential_bias_chart_json_handled_exception(self, exc_type):
    mock_matrix = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_matrix.ndim = 2
    mock_matrix.size = 4
    mock_matrix.to_dataframe.side_effect = exc_type("Handled exception")
    mock_result = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.REVIEW,
        channel_results=[],
        low_correlation_channels=["ch1"],
        correlation_matrix=mock_matrix,
    )
    self.assertIsNone(plots.generate_potential_bias_chart_json(mock_result))

  def test_generate_potential_bias_chart_json_unhandled_exception(self):
    mock_matrix = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_matrix.ndim = 2
    mock_matrix.size = 4
    mock_matrix.to_dataframe.side_effect = RuntimeError("Unhandled exception")
    mock_result = results.PotentialBiasCheckResult(
        case=results.PotentialBiasAggregateCases.REVIEW,
        channel_results=[],
        low_correlation_channels=["ch1"],
        correlation_matrix=mock_matrix,
    )
    with self.assertRaises(RuntimeError):
      plots.generate_potential_bias_chart_json(mock_result)


class CalibrationDetailsPlotsTest(parameterized.TestCase):

  @parameterized.parameters(
      ValueError, KeyError, AttributeError, TypeError, IndexError
  )
  def test_generate_calibration_details_chart_json_warning_on_error(
      self, exc_type
  ):
    mock_data = results.CalibrationOverviewChannelData(
        channel_name="err_channel",
        spend=100.0,
    )
    with mock.patch.object(
        plots,
        "build_calibration_details_chart",
        side_effect=exc_type("test error"),
        autospec=True,
        spec_set=True,
    ):
      with self.assertWarns(RuntimeWarning):
        self.assertIsNone(
            plots.generate_calibration_details_chart_json(mock_data)
        )


class BuildCalibrationOverviewChartTest(parameterized.TestCase):

  @parameterized.parameters(
      ValueError, KeyError, AttributeError, TypeError, IndexError
  )
  def test_generate_calibration_overview_chart_json_warning_on_error(
      self, exc_type
  ):
    mock_data = results.CalibrationOverviewChannelData(
        channel_name="err_channel",
        spend=100.0,
    )

    with mock.patch.object(
        plots,
        "build_calibration_overview_chart",
        side_effect=exc_type("test error"),
        autospec=True,
        spec_set=True,
    ):
      with self.assertWarns(RuntimeWarning):
        self.assertIsNone(
            plots.generate_calibration_overview_chart_json(mock_data)
        )


if __name__ == "__main__":
  absltest.main()
