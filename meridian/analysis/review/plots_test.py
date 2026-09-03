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

from collections.abc import Sequence
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import altair as alt
from meridian import backend
from meridian.analysis.review import constants
from meridian.analysis.review import plots
from meridian.analysis.review import results
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
import numpy as np
import pandas as pd
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


def _create_mock_experiment(
    point_estimate: float = 2.0,
    standard_error: float = 0.4,
    adjusted_point_estimate: float = 1.8,
    adjusted_standard_error: float = 0.5,
    source_type: calibration_base.SourceType = calibration_base.SourceType.MERIDIAN_GEOX,
    tau_spend: float = 0.0,
    gamma_duration: float = 1.0,
    tau_duration: float = 0.0,
    tau_recency: float = 0.0,
    user_point_estimate_adjustment: float | None = None,
    user_standard_error_adjustment: float | None = None,
) -> calibration_base.CalibratedExperiment:
  return calibration_base.CalibratedExperiment(
      source_type=source_type,
      raw_experiment_result=calibration_base.ExperimentResult(
          point_estimate=point_estimate, standard_error=standard_error
      ),
      adjusted_experiment_result=calibration_base.ExperimentResult(
          point_estimate=adjusted_point_estimate,
          standard_error=adjusted_standard_error,
      ),
      tau_spend=tau_spend,
      gamma_duration=gamma_duration,
      tau_duration=tau_duration,
      tau_recency=tau_recency,
      user_point_estimate_adjustment=user_point_estimate_adjustment,
      user_standard_error_adjustment=user_standard_error_adjustment,
  )


def _create_mock_distribution(
    prob_val: float = 0.1, quantile_val: float = 5.0
) -> backend.tfd.Distribution:
  mock_dist = mock.create_autospec(
      backend.tfd.Distribution, instance=True, spec_set=True
  )
  mock_dist.quantile.return_value = np.array(quantile_val)
  mock_dist.prob.side_effect = lambda x: np.ones_like(x) * prob_val
  mock_dist.sample.return_value = np.array([1.0, 2.0, 3.0])
  return mock_dist


def _create_mock_channel_data(
    channel_name: str = "test_channel",
    spend: float = 500.0,
    experiments: Sequence[calibration_base.CalibratedExperiment] | None = None,
    has_calibrated_output: bool = True,
    has_calibrated_prior_dist: bool = True,
    has_baseline_prior: bool = True,
    posterior_samples: np.ndarray | None = None,
) -> results.CalibrationOverviewChannelData:
  """Helper to construct realistic mock CalibrationOverviewChannelData."""
  mock_dist = (
      _create_mock_distribution(prob_val=0.1, quantile_val=5.0)
      if has_calibrated_prior_dist
      else None
  )
  mock_output = None
  if has_calibrated_output:
    mock_prior = _create_mock_distribution(prob_val=0.05, quantile_val=5.0)
    if experiments is None:
      experiments = [
          _create_mock_experiment(
              adjusted_point_estimate=2.0, adjusted_standard_error=0.5
          )
      ]

    mock_output = calibration_base.CalibrationOutput(
        channel_name=channel_name,
        baseline_prior=mock_prior if has_baseline_prior else None,
        intermediary_prior=mock_prior,
        experiments=list(experiments),
    )

  return results.CalibrationOverviewChannelData(
      channel_name=channel_name,
      spend=spend,
      calibrated_output=mock_output,
      calibrated_prior_dist=mock_dist,
      posterior_samples=(
          posterior_samples if posterior_samples is not None else np.array([])
      ),
  )


_FILTERING_AND_SORTING_EXP_SPECS = (
    (1.0, 1.0, calibration_base.SourceType.GENERIC),
    (2.0, 0.1, calibration_base.SourceType.MERIDIAN_GEOX),
    (3.0, 0.9, calibration_base.SourceType.GENERIC),
    (4.0, 0.2, calibration_base.SourceType.MERIDIAN_GEOX),
    (5.0, 0.8, calibration_base.SourceType.GENERIC),
    (6.0, 0.3, calibration_base.SourceType.MERIDIAN_GEOX),
    (7.0, 1.5, calibration_base.SourceType.GENERIC),
)

_EXPECTED_TOP5_FILTERED_EXP_LABELS = (
    "Experiment 2 (Meridian GeoX)",
    "Experiment 3 (Incrementality)",
    "Experiment 4 (Meridian GeoX)",
    "Experiment 5 (Incrementality)",
    "Experiment 6 (Meridian GeoX)",
)


def _create_mock_experiments_for_filtering(
    count: int = 7,
) -> list[calibration_base.CalibratedExperiment]:
  return [
      _create_mock_experiment(
          point_estimate=pe,
          standard_error=se,
          adjusted_point_estimate=pe,
          adjusted_standard_error=se,
          source_type=st,
      )
      for pe, se, st in _FILTERING_AND_SORTING_EXP_SPECS[:count]
  ]


class CalibrationDetailsPlotsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "generic",
          calibration_base.SourceType.GENERIC,
          " (Incrementality)",
      ),
      (
          "geox",
          calibration_base.SourceType.MERIDIAN_GEOX,
          " (Meridian GeoX)",
      ),
  )
  def test_get_experiment_label_suffix(
      self,
      source_type: calibration_base.SourceType,
      expected_suffix: str,
  ):
    self.assertEqual(
        calibration_plots.get_experiment_label_suffix(source_type),
        expected_suffix,
    )

  @parameterized.named_parameters(
      ("none_input", None),
      (
          "none_calibrated_output",
          _create_mock_channel_data(has_calibrated_output=False),
      ),
      ("empty_experiments", _create_mock_channel_data(experiments=[])),
  )
  def test_build_calibration_details_chart_none_or_empty(self, ch_data):
    self.assertIsNone(plots.build_calibration_details_chart(ch_data))
    self.assertIsNone(plots.generate_calibration_details_chart_json(ch_data))

  @parameterized.named_parameters(
      (
          "single_experiment",
          [
              _create_mock_experiment(
                  tau_spend=0.1,
                  gamma_duration=0.9,
                  tau_duration=0.05,
                  tau_recency=0.02,
              )
          ],
          "Experiment Adjustments: test_channel (Experiment 1 (Meridian GeoX))",
          [
              eda_constants.STAGE_UNADJUSTED_RAW,
              eda_constants.STAGE_SPEND_ADJUSTED,
              eda_constants.STAGE_SPEND_DURATION_ADJUSTED,
              eda_constants.STAGE_SPEND_DURATION_RECENCY_ADJUSTED,
              eda_constants.STAGE_FINAL_ADJUSTED,
          ],
          [],
          1,
      ),
      (
          "user_adjustments",
          [
              _create_mock_experiment(
                  source_type=calibration_base.SourceType.GENERIC,
                  tau_spend=0.1,
                  gamma_duration=0.9,
                  tau_duration=0.05,
                  tau_recency=0.02,
                  user_point_estimate_adjustment=0.1,
                  user_standard_error_adjustment=0.05,
              )
          ],
          (
              "Experiment Adjustments: test_channel (Experiment 1"
              " (Incrementality))"
          ),
          [
              eda_constants.STAGE_UNADJUSTED_RAW,
              eda_constants.STAGE_SPEND_ADJUSTED,
              eda_constants.STAGE_SPEND_DURATION_ADJUSTED,
              eda_constants.STAGE_SPEND_DURATION_RECENCY_ADJUSTED,
              eda_constants.STAGE_SPEND_DURATION_RECENCY_USER_ADJUSTED,
              eda_constants.STAGE_FINAL_ADJUSTED,
          ],
          [],
          1,
      ),
      (
          "multiple_experiments",
          [
              _create_mock_experiment(
                  source_type=calibration_base.SourceType.MERIDIAN_GEOX
              ),
              _create_mock_experiment(
                  point_estimate=3.0,
                  standard_error=0.6,
                  adjusted_point_estimate=2.5,
                  adjusted_standard_error=0.3,
                  source_type=calibration_base.SourceType.GENERIC,
              ),
          ],
          None,
          [],
          ["(Meridian GeoX)", "(Incrementality)"],
          2,
      ),
      (
          "zero_gamma_and_sorting",
          [
              _create_mock_experiment(
                  point_estimate=3.0,
                  standard_error=0.6,
                  adjusted_point_estimate=2.5,
                  adjusted_standard_error=0.8,
                  source_type=calibration_base.SourceType.GENERIC,
              ),
              _create_mock_experiment(
                  point_estimate=2.0,
                  standard_error=0.4,
                  adjusted_point_estimate=0.0,
                  adjusted_standard_error=0.5,
                  gamma_duration=0.0,
                  source_type=calibration_base.SourceType.MERIDIAN_GEOX,
              ),
          ],
          "Experiment 1 (Incrementality)",
          [],
          [],
          2,
      ),
  )
  def test_build_calibration_details_chart_scenarios(
      self,
      experiments,
      expected_title_contains,
      expected_stages,
      expected_json_substrings,
      expected_num_charts,
  ):
    ch_data = _create_mock_channel_data(experiments=experiments)
    chart = plots.build_calibration_details_chart(ch_data)
    self.assertIsNotNone(chart)
    chart_dict = chart.to_dict()

    if expected_num_charts == 1:
      self.assertEqual(chart_dict["title"]["text"], expected_title_contains)
      self.assertIn("layer", chart_dict)
    else:
      self.assertIsInstance(chart, alt.HConcatChart)
      self.assertIn("hconcat", chart_dict)
      self.assertLen(chart_dict["hconcat"], expected_num_charts)
      if expected_title_contains:
        self.assertIn(
            expected_title_contains,
            chart_dict["hconcat"][0]["title"]["text"],
        )

    chart_json = plots.generate_calibration_details_chart_json(ch_data)
    self.assertIsNotNone(chart_json)
    for substr in expected_json_substrings:
      self.assertIn(substr, chart_json)

    if expected_stages:
      parsed_json = json.loads(chart_json)
      self.assertIn("datasets", parsed_json)
      stages = []
      for dataset in parsed_json["datasets"].values():
        for row in dataset:
          if "stage" in row:
            stages.append(row["stage"])
      for stage in expected_stages:
        self.assertIn(stage, stages)

  def test_build_calibration_details_chart_filtering_and_sorting(self):
    ch_data = _create_mock_channel_data(
        experiments=_create_mock_experiments_for_filtering()
    )

    chart = plots.build_calibration_details_chart(ch_data)
    self.assertIsNotNone(chart)
    self.assertIsInstance(chart, alt.HConcatChart)
    chart_dict = chart.to_dict()
    # Limit is MAX_EXPERIMENTS_FOR_DETAILS_CARD (5). Top 5 by lowest SE are
    # 2, 4, 6, 5, 3. Preserving 1-based original index order: 2, 3, 4, 5, 6.
    self.assertLen(chart_dict["hconcat"], 5)
    for i, exp_label in enumerate(_EXPECTED_TOP5_FILTERED_EXP_LABELS):
      expected_title = (
          f"Experiment Adjustments: test_channel ({exp_label})"
      )
      self.assertEqual(
          chart_dict["hconcat"][i]["title"]["text"], expected_title
      )

  def test_compute_experiment_adjustment_stages_invalid_tau_spend(self):
    exp = _create_mock_experiment(tau_spend=-1.5)
    with self.assertRaisesRegex(ValueError, "`tau_spend` must be >= -1.0"):
      plots._compute_experiment_adjustment_stages(exp)

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


def _is_line_layer(
    layer: dict[str, object], stroke_dash: Sequence[int] | None = None
) -> bool:
  mark = layer.get("mark")
  mark_type = (
      mark
      if isinstance(mark, str)
      else (mark.get("type") if isinstance(mark, dict) else None)
  )
  layer_dash = (
      mark.get("strokeDash")
      if isinstance(mark, dict)
      else layer.get("strokeDash")
  )
  if mark_type != "line":
    return False
  if stroke_dash is None:
    return layer_dash is None
  return (
      list(layer_dash) == list(stroke_dash)
      if isinstance(layer_dash, (list, tuple))
      else False
  )


def _get_layer_dataset_labels(
    chart_dict: dict[str, object], layer: dict[str, object]
) -> set[str]:
  data_obj = layer.get("data")
  if not isinstance(data_obj, dict):
    return set()
  data_name = data_obj.get("name")
  datasets = chart_dict.get("datasets", {})
  if not isinstance(datasets, dict):
    return set()
  rows = datasets.get(data_name, [])
  if not isinstance(rows, list):
    return set()
  labels: set[str] = set()
  for row in rows:
    if isinstance(row, dict) and eda_constants.LABEL in row:
      val = row[eda_constants.LABEL]
      if isinstance(val, str):
        labels.add(val)
  return labels


def _get_color_scale(
    subplot: dict[str, object],
) -> tuple[list[str], list[str]]:
  layers = subplot.get("layer")
  if isinstance(layers, list):
    for layer in layers:
      if isinstance(layer, dict):
        scale = layer.get("encoding", {}).get("color", {}).get("scale")
        if isinstance(scale, dict):
          domain = scale.get("domain", [])
          range_ = scale.get("range", [])
          return (
              list(domain) if isinstance(domain, (list, tuple)) else [],
              list(range_) if isinstance(range_, (list, tuple)) else [],
          )
  return ([], [])


class BuildCalibrationOverviewChartTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("with_posterior", np.array([1.2, 2.3])),
      ("without_posterior", np.array([])),
  )
  def test_build_calibration_overview_chart_structure(
      self, posterior_samples: np.ndarray
  ):
    ch_data = _create_mock_channel_data(
        channel_name="overview_channel",
        posterior_samples=posterior_samples,
    )
    chart = plots.build_calibration_overview_chart(ch_data)
    self.assertIsNotNone(chart)
    self.assertIsInstance(chart, alt.HConcatChart)
    chart_dict = chart.to_dict()
    self.assertNotIn("title", chart_dict)
    self.assertIn("hconcat", chart_dict)
    self.assertLen(chart_dict["hconcat"], 3)
    self.assertEqual(
        chart_dict["hconcat"][0]["title"]["text"],
        constants.CALIBRATION_LEFT_PLOT_TITLE,
    )
    self.assertEqual(
        chart_dict["hconcat"][1]["title"]["text"],
        constants.CALIBRATION_MIDDLE_PLOT_TITLE,
    )
    self.assertEqual(
        chart_dict["hconcat"][2]["title"]["text"],
        constants.CALIBRATION_RIGHT_PLOT_TITLE,
    )
    self.assertEqual(
        chart_dict["hconcat"][0]["layer"][0]["encoding"]["color"]["legend"][
            "symbolType"
        ],
        "square",
    )
    for subplot in chart_dict["hconcat"]:
      layer_scales = [
          layer.get("encoding", {}).get("x", {}).get("scale", {})
          for layer in subplot.get("layer", [])
          if "encoding" in layer and "x" in layer["encoding"]
      ]
      for scale in layer_scales:
        if scale:
          self.assertEqual(scale.get("domainMin"), 0)
          self.assertTrue(scale.get("clamp"))

    chart_json = plots.generate_calibration_overview_chart_json(ch_data)
    self.assertIsNotNone(chart_json)
    self.assertIsInstance(chart_json, str)

  @parameterized.named_parameters(
      ("none_input", None),
      (
          "none_calibrated_output",
          _create_mock_channel_data(has_calibrated_output=False),
      ),
      (
          "none_calibrated_prior_dist",
          _create_mock_channel_data(has_calibrated_prior_dist=False),
      ),
      (
          "none_output_and_dist",
          _create_mock_channel_data(
              has_calibrated_output=False, has_calibrated_prior_dist=False
          ),
      ),
  )
  def test_build_calibration_overview_chart_none_returns_none(self, ch_data):
    self.assertIsNone(plots.build_calibration_overview_chart(ch_data))
    self.assertIsNone(plots.generate_calibration_overview_chart_json(ch_data))

  @parameterized.named_parameters(
      ("single_geox", [calibration_base.SourceType.MERIDIAN_GEOX]),
      (
          "multi_experiment",
          [
              calibration_base.SourceType.MERIDIAN_GEOX,
              calibration_base.SourceType.GENERIC,
          ],
      ),
      ("no_experiments", []),
  )
  def test_build_calibration_overview_chart_experiments(
      self, source_types: list[calibration_base.SourceType]
  ):
    experiments = [
        _create_mock_experiment(
            adjusted_point_estimate=2.0 + idx,
            adjusted_standard_error=0.5 + idx * 0.2,
            source_type=st,
        )
        for idx, st in enumerate(source_types)
    ]
    ch_data = _create_mock_channel_data(
        channel_name="exp_test_channel",
        experiments=experiments,
        has_baseline_prior=True,
        posterior_samples=np.array([1.2, 2.3]),
    )
    chart = plots.build_calibration_overview_chart(ch_data)
    self.assertIsNotNone(chart)
    chart_dict = chart.to_dict()
    left_subplot = chart_dict["hconcat"][0]

    domain, range_ = _get_color_scale(left_subplot)
    self.assertEqual(len(domain), len(range_))
    chart_json = chart.to_json()

    exp_line_layers = [
        layer
        for layer in left_subplot["layer"]
        if _is_line_layer(layer, stroke_dash=None)
    ]
    left_hover_labels = _get_layer_dataset_labels(
        chart_dict, left_subplot["layer"][-2]
    )

    self.assertIn(constants.CALIBRATED_PRIOR_COLOR, range_)
    self.assertIn(constants.INTERMEDIARY_PRIOR_COLOR, range_)
    self.assertIn(constants.CALIBRATED_MERIDIAN_PRIOR, domain)
    self.assertIn(constants.INTERMEDIARY_PRIOR, domain)
    self.assertNotIn("Prior Calibration:", chart_json)

    if source_types:
      self.assertLen(exp_line_layers, 1)
      self.assertLen(left_subplot["layer"], 6)
      for idx, st in enumerate(source_types):
        suffix = (
            " (Meridian GeoX)"
            if st == calibration_base.SourceType.MERIDIAN_GEOX
            else " (Incrementality)"
        )
        exp_label = f"Experiment {idx + 1}{suffix}"
        self.assertIn(exp_label, domain)
        expected_color = constants.CALIBRATION_EXPERIMENT_COLORS[
            idx % len(constants.CALIBRATION_EXPERIMENT_COLORS)
        ]
        self.assertEqual(range_[domain.index(exp_label)], expected_color)
        self.assertIn(exp_label, chart_json)
        self.assertIn(expected_color, chart_json)
        self.assertIn(exp_label, left_hover_labels)
    else:
      self.assertEmpty(exp_line_layers)
      self.assertLen(left_subplot["layer"], 5)
      for label in domain:
        self.assertFalse(label.startswith("Experiment"))
      self.assertNotIn("Experiment 1", chart_json)
      for label in left_hover_labels:
        self.assertFalse(label.startswith("Experiment"))

  def test_build_calibration_overview_chart_filtering_and_sorting(self):
    ch_data = _create_mock_channel_data(
        experiments=_create_mock_experiments_for_filtering(),
        has_baseline_prior=True,
    )

    chart = plots.build_calibration_overview_chart(ch_data)
    self.assertIsNotNone(chart)
    chart_dict = chart.to_dict()
    left_subplot = chart_dict["hconcat"][0]
    domain, _ = _get_color_scale(left_subplot)

    # Limit is MAX_EXPERIMENTS_FOR_OVERVIEW_CARD (5). Top 5 by lowest SE are
    # 2 (0.1), 4 (0.2), 6 (0.3), 5 (0.8), 3 (0.9).
    # Preserving 1-based original index order: 2, 3, 4, 5, 6.
    for exp_label in _EXPECTED_TOP5_FILTERED_EXP_LABELS:
      self.assertIn(exp_label, domain)

    self.assertNotIn("Experiment 1 (Incrementality)", domain)
    self.assertNotIn("Experiment 7 (Incrementality)", domain)

    exp_labels_in_domain = [l for l in domain if l.startswith("Experiment ")]
    self.assertSequenceEqual(
        exp_labels_in_domain, _EXPECTED_TOP5_FILTERED_EXP_LABELS
    )

  @parameterized.named_parameters(
      ("presence", np.array([1.2, 2.3]), True),
      ("absence_empty_array", np.array([]), False),
      ("absence_none", None, False),
  )
  def test_build_calibration_overview_chart_posterior(
      self, posterior_samples: np.ndarray | None, expected_present: bool
  ):
    ch_data = _create_mock_channel_data(posterior_samples=posterior_samples)
    chart = plots.build_calibration_overview_chart(ch_data)
    self.assertIsNotNone(chart)
    chart_dict = chart.to_dict()

    left_subplot = chart_dict["hconcat"][0]
    domain, range_ = _get_color_scale(left_subplot)
    self.assertEqual(len(domain), len(range_))
    chart_json = chart.to_json()

    right_subplot = chart_dict["hconcat"][2]
    right_bar_labels = _get_layer_dataset_labels(
        chart_dict, right_subplot["layer"][0]
    )
    right_hover_labels = _get_layer_dataset_labels(
        chart_dict, right_subplot["layer"][-2]
    )

    if expected_present:
      self.assertIn(constants.MERIDIAN_POSTERIOR, domain)
      self.assertIn(constants.POSTERIOR_HISTOGRAM_COLOR, range_)
      self.assertEqual(
          range_[domain.index(constants.MERIDIAN_POSTERIOR)],
          constants.POSTERIOR_HISTOGRAM_COLOR,
      )
      self.assertIn(constants.POSTERIOR_HISTOGRAM_COLOR, chart_json)
      self.assertIn(constants.MERIDIAN_POSTERIOR, right_bar_labels)
      self.assertNotIn(constants.INTERMEDIARY_PRIOR, right_bar_labels)
      self.assertIn(constants.MERIDIAN_POSTERIOR, right_hover_labels)
    else:
      self.assertNotIn(constants.MERIDIAN_POSTERIOR, domain)
      self.assertNotIn(constants.POSTERIOR_HISTOGRAM_COLOR, range_)
      self.assertNotIn(constants.POSTERIOR_HISTOGRAM_COLOR, chart_json)
      self.assertIn(constants.INTERMEDIARY_PRIOR, right_bar_labels)
      self.assertNotIn(constants.MERIDIAN_POSTERIOR, right_bar_labels)
      self.assertNotIn(constants.MERIDIAN_POSTERIOR, right_hover_labels)
      self.assertIn(constants.INTERMEDIARY_PRIOR, right_hover_labels)

  @parameterized.named_parameters(
      ("presence", True),
      ("absence", False),
  )
  def test_build_calibration_overview_chart_baseline_prior(
      self, has_baseline_prior: bool
  ):
    ch_data = _create_mock_channel_data(
        has_baseline_prior=has_baseline_prior,
        posterior_samples=np.array([1.2, 2.3]),
    )
    chart = plots.build_calibration_overview_chart(ch_data)
    self.assertIsNotNone(chart)
    chart_dict = chart.to_dict()
    left_subplot = chart_dict["hconcat"][0]

    domain, range_ = _get_color_scale(left_subplot)
    self.assertEqual(len(domain), len(range_))

    baseline_layers = [
        layer
        for layer in left_subplot["layer"]
        if _is_line_layer(layer, stroke_dash=[5, 5])
    ]
    left_hover_labels = _get_layer_dataset_labels(
        chart_dict, left_subplot["layer"][-2]
    )
    chart_json = chart.to_json()

    if has_baseline_prior:
      self.assertLen(baseline_layers, 1)
      self.assertLen(left_subplot["layer"], 6)
      self.assertIn(constants.BASELINE_PRIOR, domain)
      self.assertIn(constants.BASELINE_PRIOR_COLOR, range_)
      self.assertEqual(
          range_[domain.index(constants.BASELINE_PRIOR)],
          constants.BASELINE_PRIOR_COLOR,
      )
      self.assertIn(constants.BASELINE_PRIOR, chart_json)
      self.assertIn(constants.BASELINE_PRIOR_COLOR, chart_json)
      self.assertIn(constants.BASELINE_PRIOR, left_hover_labels)
    else:
      self.assertEmpty(baseline_layers)
      self.assertLen(left_subplot["layer"], 5)
      self.assertNotIn(constants.BASELINE_PRIOR, domain)
      self.assertNotIn(constants.BASELINE_PRIOR_COLOR, range_)
      self.assertNotIn(constants.BASELINE_PRIOR, chart_json)
      self.assertNotIn(constants.BASELINE_PRIOR_COLOR, chart_json)
      self.assertNotIn(constants.BASELINE_PRIOR, left_hover_labels)

  @parameterized.named_parameters(
      ("empty_baseline_df", "baseline"),
      ("empty_posterior_df", "posterior"),
  )
  def test_build_calibration_overview_chart_empty_df_fallback(
      self, target: str
  ):
    ch_data = _create_mock_channel_data(
        has_baseline_prior=True,
        posterior_samples=np.array([1.2, 2.3]),
    )
    if target == "baseline":
      empty_plot_data = calibration_plots.CalibrationPlotData(
          baseline_df=pd.DataFrame(),
          exp_dfs=[],
          intermediary_df=pd.DataFrame({
              constants.ROI: [1.0],
              eda_constants.DENSITY: [0.5],
              eda_constants.LABEL: [constants.INTERMEDIARY_PRIOR],
          }),
          calibrated_df=pd.DataFrame({
              constants.ROI: [1.0],
              eda_constants.DENSITY: [0.5],
              eda_constants.LABEL: [constants.CALIBRATED_MERIDIAN_PRIOR],
          }),
      )
      patch_ctx = mock.patch.object(
          calibration_plots,
          "prepare_calibration_data",
          return_value=empty_plot_data,
          autospec=True,
          spec_set=True,
      )
    else:
      orig_make_df = calibration_plots.make_calibration_plot_df
      patch_ctx = mock.patch.object(
          calibration_plots,
          "make_calibration_plot_df",
          side_effect=lambda x, y, label: (
              pd.DataFrame()
              if label == constants.MERIDIAN_POSTERIOR
              else orig_make_df(x, y, label)
          ),
          autospec=True,
          spec_set=True,
      )

    with patch_ctx:
      chart = plots.build_calibration_overview_chart(ch_data)
      self.assertIsNotNone(chart)
      chart_dict = chart.to_dict()
      domain, range_ = _get_color_scale(chart_dict["hconcat"][0])

      if target == "baseline":
        self.assertNotIn(constants.BASELINE_PRIOR, domain)
        self.assertNotIn(constants.BASELINE_PRIOR_COLOR, range_)
        baseline_layers = [
            layer
            for layer in chart_dict["hconcat"][0]["layer"]
            if _is_line_layer(layer, stroke_dash=[5, 5])
        ]
        self.assertEmpty(baseline_layers)
      else:
        self.assertNotIn(constants.MERIDIAN_POSTERIOR, domain)
        self.assertNotIn(constants.POSTERIOR_HISTOGRAM_COLOR, range_)
        right_bar_labels = _get_layer_dataset_labels(
            chart_dict, chart_dict["hconcat"][2]["layer"][0]
        )
        self.assertIn(constants.INTERMEDIARY_PRIOR, right_bar_labels)
        self.assertNotIn(constants.MERIDIAN_POSTERIOR, right_bar_labels)

  def test_create_roi_grid_with_negative_experiments(self):
    mock_dist = _create_mock_distribution(quantile_val=5.0)
    exp = _create_mock_experiment(
        adjusted_point_estimate=-2.0, adjusted_standard_error=4.0
    )
    grid = plots._create_roi_grid(mock_dist, [exp])
    self.assertLess(grid[0], 0.0)
    self.assertGreater(grid[-1], 0.0)

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
