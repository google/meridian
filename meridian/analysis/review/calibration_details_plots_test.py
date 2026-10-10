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

"""Tests for the calibration details chart builders."""

from collections.abc import Sequence
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import altair as alt
from meridian import backend
from meridian.analysis.review import calibration_details_plots
from meridian.analysis.review import results
from meridian.model.calibration import base as calibration_base
from meridian.model.eda import calibration_plots
from meridian.model.eda import constants as eda_constants
import numpy as np


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
    self.assertIsNone(
        calibration_details_plots.build_calibration_details_chart(ch_data)
    )

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
    chart = calibration_details_plots.build_calibration_details_chart(ch_data)
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

    chart_json = chart.to_json()
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

    chart = calibration_details_plots.build_calibration_details_chart(ch_data)
    self.assertIsNotNone(chart)
    self.assertIsInstance(chart, alt.HConcatChart)
    chart_dict = chart.to_dict()
    # Limit is MAX_EXPERIMENTS_FOR_DETAILS_CARD (5). Top 5 by lowest SE are
    # 2, 4, 6, 5, 3. Preserving 1-based original index order: 2, 3, 4, 5, 6.
    self.assertLen(chart_dict["hconcat"], 5)
    for i, exp_label in enumerate(_EXPECTED_TOP5_FILTERED_EXP_LABELS):
      expected_title = f"Experiment Adjustments: test_channel ({exp_label})"
      self.assertEqual(
          chart_dict["hconcat"][i]["title"]["text"], expected_title
      )

  def test_compute_experiment_adjustment_stages_invalid_tau_spend(self):
    exp = _create_mock_experiment(tau_spend=-1.5)
    with self.assertRaisesRegex(ValueError, "`tau_spend` must be >= -1.0"):
      calibration_details_plots._compute_experiment_adjustment_stages(exp)


if __name__ == "__main__":
  absltest.main()
