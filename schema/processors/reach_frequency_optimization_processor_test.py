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

"""Unit tests for reach_frequency_optimization_processor.py."""

import dataclasses
import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.analysis import analyzer
from meridian.data import time_coordinates as tc
from meridian.model import model
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import estimate_pb2
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_pb
from mmm.v1.marketing import marketing_data_pb2
from mmm.v1.marketing.analysis import kpi_outcome_pb2
from mmm.v1.marketing.analysis import marketing_analysis_pb2
from mmm.v1.marketing.analysis import media_analysis_pb2
from mmm.v1.marketing.analysis import response_curve_pb2
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from schema.processors import common
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor as rfop
import numpy as np
import tensorflow as tf
import xarray as xr

from google.type import date_pb2
from tensorflow.python.util.protobuf import compare


_MEDIA_CHANNELS = ["media_ch_0", "media_ch_1"]
_RF_CHANNELS = ["rf_ch_0", "rf_ch_1"]
_ALL_CHANNELS = _MEDIA_CHANNELS + _RF_CHANNELS

_ALL_TIMES = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
]

_SPEND_MULTIPLIERS = [1, 2, 3]

_METRIC_BASE = np.array([[4.57, 1.60, 7.52], [6.60, 1.69, 11.70]])

_FREQUENCY_RANGE = [
    49.0,
    49.1,
    49.2,
    49.3,
    49.4,
    49.5,
    49.6,
    49.7,
    49.8,
    49.9,
    50.0,
]

_OPTIMAL_FREQUENCY = np.array([1.0, 1.0])

_OPTIMIZED_INCREMENTAL_OUTCOME = np.array([
    [313.0, 130.1, 495.3],
    [454.7, 159.1, 761.4],
])

_OPTIMIZED_EFFECTIVENESS = np.array([
    [0.000427, 0.000177, 0.000676],
    [0.000618, 0.000216, 0.001036],
])

_OPTIMIZED_ROI = np.array([
    [1.150, 0.478, 1.819],
    [1.579, 0.552, 2.644],
])

_OPTIMIZED_MROI_BY_REACH = np.array(
    [[0.724, 0.573, 0.619], [0.574, 0.593, 0.641]]
)

_OPTIMIZED_MROI_BY_FREQUENCY = np.array(
    [[0.118, 0.095, 0.139], [0.152, 0.141, 0.164]]
)

_OPTIMIZED_CPIK = np.array([[3.191, 3.475, 3.812], [3.575, 3.552, 3.648]])

_MEDIA_SPEND = np.array([[[1.5, 2.8], [4.1, 3.3], [2.7, 0.9], [3.2, 1.3]]])
_RF_SPEND = np.array([[[1.5, 2.8], [4.1, 3.3], [2.7, 0.9], [3.2, 1.3]]])
_ALL_SPEND = np.concat([_MEDIA_SPEND, _RF_SPEND], axis=-1)

_SPEC_1_TIME_INDEX = slice(0, 2)
_AGGREGATED_SPEND_1 = xr.DataArray(
    data=np.sum(_ALL_SPEND[:, _SPEC_1_TIME_INDEX, :], axis=(0, 1)),
    dims=[constants.CHANNEL],
    coords={constants.CHANNEL: _ALL_CHANNELS},
)

_SPEC_2_TIME_INDEX = slice(2, 4)
_AGGREGATED_SPEND_2 = xr.DataArray(
    data=np.sum(_ALL_SPEND[:, _SPEC_2_TIME_INDEX, :], axis=(0, 1)),
    dims=[constants.CHANNEL],
    coords={constants.CHANNEL: _ALL_CHANNELS},
)


def _create_rf_spend_info_list(aggregated_spend: xr.DataArray):
  spend_info_list = []
  rf_spend = aggregated_spend.sel({constants.CHANNEL: _RF_CHANNELS}).data
  total_spend = np.sum(aggregated_spend.data)

  for spend in rf_spend:
    spend_info_list.append(
        media_analysis_pb2.SpendInfo(
            spend=spend,
            spend_share=100 * spend / total_spend,
        )
    )

  return spend_info_list


_EXPECTED_RF_SPEND_INFOS = [
    _create_rf_spend_info_list(_AGGREGATED_SPEND_1),
    _create_rf_spend_info_list(_AGGREGATED_SPEND_2),
]

_EXPECTED_RESPONSE_CURVES = [
    [
        response_curve_pb2.ResponsePoint(
            input_value=0.6,
            incremental_kpi=0.75,
        ),
        response_curve_pb2.ResponsePoint(
            input_value=0.5,
            incremental_kpi=0.7,
        ),
        response_curve_pb2.ResponsePoint(
            input_value=0.48,
            incremental_kpi=0.85,
        ),
    ],
    [
        response_curve_pb2.ResponsePoint(
            input_value=0.7,
            incremental_kpi=0.68,
        ),
        response_curve_pb2.ResponsePoint(
            input_value=0.72,
            incremental_kpi=0.65,
        ),
        response_curve_pb2.ResponsePoint(
            input_value=0.61,
            incremental_kpi=0.8,
        ),
    ],
]

_OPTIMIZATION_NAME_1 = "RF optimization 1"
_OPTIMIZATION_NAME_2 = "RF optimization 2"
_MEDIA_UNIT = "impressions"
_GRID_NAME = "frequency outcome grid"
_GROUP_ID = "group_id"
_START_DATE_SPEC_1 = datetime.date(2024, 1, 1)
_END_DATE_SPEC_1 = datetime.date(2024, 1, 15)
_START_PROTO_DATE_SPEC_1 = date_pb2.Date(year=2024, month=1, day=1)
_END_PROTO_DATE_SPEC_1 = date_pb2.Date(year=2024, month=1, day=15)

_START_DATE_SPEC_2 = datetime.date(2024, 1, 15)
_END_DATE_SPEC_2 = datetime.date(2024, 1, 29)

_MODEL_MAX_FREQUENCY = 50.0
_PRECISION = 3
_STEP_SIZE = 0.1

_FREQUENCY = tf.convert_to_tensor(
    [[[_MODEL_MAX_FREQUENCY, 10.0], [20.0, 25.0], [15.0, 30.0], [40.0, 20.0]]]
)

_REACH = tf.convert_to_tensor(
    [[[10.0, 2.0], [5.0, 10.0], [3.0, 6.0], [4.0, 4.0]]]
)


def _create_expected_frequency_outcome_grid_proto(
    frequencies: list[float], confidence_level: float, spec_number: int
) -> rf_pb.FrequencyOutcomeGrid:
  channel_cells = []

  # TODO: Simplify the logic for creating frequency outcome grid.
  for i, channel in enumerate(_RF_CHANNELS):
    cells = []
    for freq in frequencies:
      impressions = 0.0
      for j, _ in enumerate(_FREQUENCY.numpy()[0]):
        impressions += _FREQUENCY.numpy()[0][j][i] * _REACH.numpy()[0][j][i]
      cell = rf_pb.FrequencyOutcomeGrid.Cell(
          outcome=estimate_pb2.Estimate(
              value=freq * _METRIC_BASE[i][0] * spec_number,
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=confidence_level,
                      lowerbound=freq * _METRIC_BASE[i][1] * spec_number,
                      upperbound=freq * _METRIC_BASE[i][2] * spec_number,
                  )
              ],
          ),
          reach_frequency=marketing_data_pb2.ReachFrequency(
              reach=int(impressions / freq),
              average_frequency=freq,
          ),
      )
      cells.append(cell)

    channel_cell = rf_pb.FrequencyOutcomeGrid.ChannelCells(
        channel_name=channel,
        cells=cells,
    )
    channel_cells.append(channel_cell)

  return rf_pb.FrequencyOutcomeGrid(
      name=_GRID_NAME,
      frequency_step_size=_STEP_SIZE,
      channel_cells=channel_cells,
  )


def _create_expected_media_analysis_proto(
    is_revenue_type: bool,
    spec_number: int,
    channel_index: int,
    spec: rfop.ReachFrequencyOptimizationSpec,
) -> media_analysis_pb2.MediaAnalysis:
  expected_response_points = _EXPECTED_RESPONSE_CURVES[channel_index]

  outcome = kpi_outcome_pb2.KpiOutcome(
      kpi_type=(
          kpi_type_pb.KpiType.REVENUE
          if is_revenue_type
          else kpi_type_pb.KpiType.NON_REVENUE
      ),
      contribution=kpi_outcome_pb2.Contribution(
          value=estimate_pb2.Estimate(
              value=_OPTIMIZED_INCREMENTAL_OUTCOME[channel_index][0]
              * spec_number,
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=spec.confidence_level,
                      lowerbound=_OPTIMIZED_INCREMENTAL_OUTCOME[channel_index][
                          1
                      ]
                      * spec_number,
                      upperbound=_OPTIMIZED_INCREMENTAL_OUTCOME[channel_index][
                          2
                      ]
                      * spec_number,
                  )
              ],
          ),
      ),
      effectiveness=kpi_outcome_pb2.Effectiveness(
          media_unit=_MEDIA_UNIT,
          value=estimate_pb2.Estimate(
              value=_OPTIMIZED_EFFECTIVENESS[channel_index][0] * spec_number,
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=spec.confidence_level,
                      lowerbound=_OPTIMIZED_EFFECTIVENESS[channel_index][1]
                      * spec_number,
                      upperbound=_OPTIMIZED_EFFECTIVENESS[channel_index][2]
                      * spec_number,
                  )
              ],
          ),
      ),
      roi=estimate_pb2.Estimate(
          value=_OPTIMIZED_ROI[channel_index][0] * spec_number,
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=spec.confidence_level,
                  lowerbound=_OPTIMIZED_ROI[channel_index][1] * spec_number,
                  upperbound=_OPTIMIZED_ROI[channel_index][2] * spec_number,
              )
          ],
      ),
      marginal_roi=estimate_pb2.Estimate(
          value=_OPTIMIZED_MROI_BY_FREQUENCY[channel_index][0] * spec_number,
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=spec.confidence_level,
                  lowerbound=_OPTIMIZED_MROI_BY_FREQUENCY[channel_index][1]
                  * spec_number,
                  upperbound=_OPTIMIZED_MROI_BY_FREQUENCY[channel_index][2]
                  * spec_number,
              )
          ],
      ),
      cost_per_contribution=estimate_pb2.Estimate(
          value=_OPTIMIZED_CPIK[channel_index][0] * spec_number,
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=spec.confidence_level,
                  lowerbound=_OPTIMIZED_CPIK[channel_index][1] * spec_number,
                  upperbound=_OPTIMIZED_CPIK[channel_index][2] * spec_number,
              )
          ],
      ),
  )

  analysis = media_analysis_pb2.MediaAnalysis(
      channel_name=_RF_CHANNELS[channel_index],
      media_outcomes=[outcome],
      spend_info=_EXPECTED_RF_SPEND_INFOS[spec_number - 1][channel_index],
      response_curve=response_curve_pb2.ResponseCurve(
          input_name=constants.SPEND,
          response_points=expected_response_points,
      ),
  )

  return analysis


def _create_expected_optimization_result_proto(
    is_revenue_type: bool,
    spec_number: int,
    spec: rfop.ReachFrequencyOptimizationSpec,
    frequencies: list[float],
) -> rf_pb.ReachFrequencyOptimizationResult:
  channel_protos = []
  for index, channel in enumerate(_RF_CHANNELS):
    channel_protos.append(
        rf_pb.OptimizedChannelFrequency(
            channel_name=channel,
            optimal_average_frequency=_OPTIMAL_FREQUENCY[index] * spec_number,
        )
    )

  max_freq = (
      spec.max_frequency
      if spec.max_frequency is not None
      else _MODEL_MAX_FREQUENCY
  )
  spec_proto = rf_pb.ReachFrequencyOptimizationSpec(
      date_interval=date_interval_pb2.DateInterval(
          start_date=date_pb2.Date(
              year=spec.start_date.year,
              month=spec.start_date.month,
              day=spec.start_date.day,
          ),
          end_date=date_pb2.Date(
              year=spec.end_date.year,
              month=spec.end_date.month,
              day=spec.end_date.day,
          ),
      ),
      rf_channel_constraints=[
          rf_pb.RfChannelConstraint(
              channel_name=_RF_CHANNELS[0],
              frequency_constraint=constraints_pb.FrequencyConstraint(
                  min_frequency=spec.min_frequency,
                  max_frequency=max_freq,
              ),
          ),
          rf_pb.RfChannelConstraint(
              channel_name=_RF_CHANNELS[1],
              frequency_constraint=constraints_pb.FrequencyConstraint(
                  min_frequency=spec.min_frequency,
                  max_frequency=max_freq,
              ),
          ),
      ],
      objective=target_pb.TargetMetric.ROI,
      kpi_type=(
          kpi_type_pb.KpiType.REVENUE
          if is_revenue_type
          else kpi_type_pb.KpiType.NON_REVENUE
      ),
  )

  marketing_analysis_proto = marketing_analysis_pb2.MarketingAnalysis(
      date_interval=date_interval_pb2.DateInterval(
          start_date=date_pb2.Date(
              year=spec.start_date.year,
              month=spec.start_date.month,
              day=spec.start_date.day,
          ),
          end_date=date_pb2.Date(
              year=spec.end_date.year,
              month=spec.end_date.month,
              day=spec.end_date.day,
          ),
      ),
  )

  for index, _ in enumerate(_RF_CHANNELS):
    marketing_analysis_proto.media_analyses.append(
        _create_expected_media_analysis_proto(
            is_revenue_type,
            spec_number,
            index,
            spec,
        )
    )

  mock_result = rf_pb.ReachFrequencyOptimizationResult(
      name=spec.optimization_name,
      spec=spec_proto,
      optimized_channel_frequencies=channel_protos,
      optimized_marketing_analysis=marketing_analysis_proto,
      frequency_outcome_grid=_create_expected_frequency_outcome_grid_proto(
          frequencies,
          spec.confidence_level,
          spec_number,
      ),
  )
  if spec.group_id:
    mock_result.group_id = spec.group_id
  return mock_result


def _create_response_curve_data() -> xr.Dataset:
  xr_dims_spend = (
      constants.SPEND_MULTIPLIER,
      constants.CHANNEL,
  )
  xr_dims_incremental_outcome = (
      constants.SPEND_MULTIPLIER,
      constants.CHANNEL,
      constants.METRIC,
  )
  xr_coords = {
      constants.CHANNEL: ([constants.CHANNEL], _RF_CHANNELS),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.CI_LO, constants.CI_HI],
      ),
      constants.SPEND_MULTIPLIER: (
          [constants.SPEND_MULTIPLIER],
          _SPEND_MULTIPLIERS,
      ),
  }

  spend = [[0.6, 0.7], [0.5, 0.72], [0.48, 0.61]]
  incremental_outcome = [
      [[0.75, 0.62, 0.96], [0.68, 0.6, 0.91]],
      [[0.7, 0.6, 0.95], [0.65, 0.55, 0.84]],
      [[0.85, 0.75, 0.94], [0.8, 0.7, 0.88]],
  ]

  xr_data = {
      constants.SPEND: (
          xr_dims_spend,
          spend,
      ),
      constants.INCREMENTAL_OUTCOME: (
          xr_dims_incremental_outcome,
          incremental_outcome,
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_metric(frequencies: list[float]) -> np.array:
  metric = np.empty((len(frequencies), 2, 3))
  for i, freq in enumerate(frequencies):
    metric[i] = _METRIC_BASE * freq

  return metric


def _create_optimal_freq_data(
    spec_number: int,
    frequencies: list[float],
    is_revenue_kpi: bool,
) -> xr.Dataset:
  xr_coords = {
      constants.FREQUENCY: ([constants.FREQUENCY], frequencies),
      constants.RF_CHANNEL: (
          [constants.RF_CHANNEL],
          _RF_CHANNELS,
      ),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.CI_LO, constants.CI_HI],
      ),
  }
  xr_attrs = {
      constants.CONFIDENCE_LEVEL: constants.DEFAULT_CONFIDENCE_LEVEL,
      constants.USE_POSTERIOR: True,
      constants.IS_REVENUE_KPI: is_revenue_kpi,
  }
  xr_data_vars = {
      constants.ROI: (
          [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
          _create_metric(frequencies) * spec_number,
      ),
      constants.OPTIMAL_FREQUENCY: (
          [constants.RF_CHANNEL],
          _OPTIMAL_FREQUENCY * spec_number,
      ),
      constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_INCREMENTAL_OUTCOME * spec_number,
      ),
      constants.OPTIMIZED_EFFECTIVENESS: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_EFFECTIVENESS * spec_number,
      ),
      constants.OPTIMIZED_ROI: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_ROI * spec_number,
      ),
      constants.OPTIMIZED_MROI_BY_REACH: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_MROI_BY_REACH * spec_number,
      ),
      constants.OPTIMIZED_MROI_BY_FREQUENCY: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_MROI_BY_FREQUENCY * spec_number,
      ),
      constants.OPTIMIZED_CPIK: (
          [constants.RF_CHANNEL, constants.METRIC],
          _OPTIMIZED_CPIK * spec_number,
      ),
  }

  return xr.Dataset(xr_data_vars, coords=xr_coords, attrs=xr_attrs)


class ReachFrequencyOptimizationSpecTest(parameterized.TestCase):

  def test_selected_times_returns_tuple_with_start_date_and_end_date(self):
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
    )
    self.assertEqual(
        spec.selected_times,
        ("2024-01-01", "2024-01-15"),
    )

  def test_validate_raises_error_when_end_date_is_before_start_date(self):
    with self.assertRaisesRegex(
        ValueError,
        "Start date must be before end date.",
    ):
      rfop.ReachFrequencyOptimizationSpec(
          start_date=_END_DATE_SPEC_1,
          end_date=_START_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          group_id=_GROUP_ID,
      )

  def test_validate_raises_error_when_confidence_level_is_below_zero(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be between 0 and 1.",
    ):
      rfop.ReachFrequencyOptimizationSpec(
          start_date=_START_DATE_SPEC_1,
          end_date=_END_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          confidence_level=-0.1,
      )

  def test_validate_raises_error_when_confidence_level_is_above_one(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be between 0 and 1.",
    ):
      rfop.ReachFrequencyOptimizationSpec(
          start_date=_START_DATE_SPEC_1,
          end_date=_END_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          confidence_level=5.0,
      )

  def test_validate_raises_error_when_min_freq_is_below_zero(self):
    with self.assertRaisesRegex(
        ValueError,
        "Min frequency must be non-negative.",
    ):
      rfop.ReachFrequencyOptimizationSpec(
          start_date=_START_DATE_SPEC_1,
          end_date=_END_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          min_frequency=-5.0,
      )

  def test_validate_raises_error_when_max_freq_is_below_min_freq(self):
    with self.assertRaisesRegex(
        ValueError,
        "Max frequency must be greater than min frequency.",
    ):
      rfop.ReachFrequencyOptimizationSpec(
          start_date=_START_DATE_SPEC_1,
          end_date=_END_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          min_frequency=5.0,
          max_frequency=3.0,
      )

  def test_validates_successfully_when_spec_is_valid(self):
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        confidence_level=0.8,
        min_frequency=0.5,
        max_frequency=3.0,
    )

    self.assertEqual(spec.start_date, _START_DATE_SPEC_1)
    self.assertEqual(spec.end_date, _END_DATE_SPEC_1)
    self.assertEqual(spec.optimization_name, _OPTIMIZATION_NAME_1)
    self.assertEqual(spec.grid_name, _GRID_NAME)
    self.assertEqual(spec.group_id, _GROUP_ID)
    self.assertEqual(spec.confidence_level, 0.8)
    self.assertEqual(spec.min_frequency, 0.5)
    self.assertEqual(spec.max_frequency, 3.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="revenue_kpi",
          kpi_type=common.KpiType.REVENUE,
      ),
      dict(
          testcase_name="non_revenue_kpi",
          kpi_type=common.KpiType.NON_REVENUE,
      ),
  )
  def test_to_proto_returns_reach_frequency_optimization_spec_proto(
      self, kpi_type: common.KpiType
  ):
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        confidence_level=0.8,
        min_frequency=0.5,
        max_frequency=3.0,
        kpi_type=kpi_type,
    )

    expected_date_interval = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(
            year=spec.start_date.year,
            month=spec.start_date.month,
            day=spec.start_date.day,
        ),
        end_date=date_pb2.Date(
            year=spec.end_date.year,
            month=spec.end_date.month,
            day=spec.end_date.day,
        ),
    )

    expected_proto = rf_pb.ReachFrequencyOptimizationSpec(
        date_interval=expected_date_interval,
        rf_channel_constraints=[
            rf_pb.RfChannelConstraint(
                channel_name=_RF_CHANNELS[0],
                frequency_constraint=constraints_pb.FrequencyConstraint(
                    min_frequency=0.5,
                    max_frequency=3.0,
                ),
            ),
            rf_pb.RfChannelConstraint(
                channel_name=_RF_CHANNELS[1],
                frequency_constraint=constraints_pb.FrequencyConstraint(
                    min_frequency=0.5,
                    max_frequency=3.0,
                ),
            ),
        ],
        objective=target_pb.TargetMetric.ROI,
        kpi_type=(
            kpi_type_pb.KpiType.REVENUE
            if kpi_type == common.KpiType.REVENUE
            else kpi_type_pb.KpiType.NON_REVENUE
        ),
    )

    compare.assertProto2Equal(
        self,
        dataclasses.replace(spec, rf_channels=_RF_CHANNELS).to_proto(),
        expected_proto,
    )


class ReachFrequencyOptimizationProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mock_meridian_model = self.enter_context(
        mock.patch.object(model, "Meridian", autospec=True)
    )
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.input_data.time = xr.DataArray(
        data=np.array(_ALL_TIMES),
        dims=[constants.TIME],
        coords={constants.TIME: ([constants.TIME], _ALL_TIMES)},
    )
    self.mock_meridian_model.input_data.rf_channel = xr.DataArray(
        data=np.array(_RF_CHANNELS),
        dims=[constants.RF_CHANNEL],
        coords={constants.RF_CHANNEL: ([constants.RF_CHANNEL], _RF_CHANNELS)},
    )
    self.mock_meridian_model.rf_tensors.frequency = _FREQUENCY
    self.mock_meridian_model.rf_tensors.reach = _REACH
    self.mock_meridian_model.rf_tensors.rf_spend = tf.convert_to_tensor(
        _RF_SPEND
    )

    # Time related setup.
    self.time_coordinates = tc.TimeCoordinates.from_dates(_ALL_TIMES)
    self.mock_meridian_model.input_data.time_coordinates = self.time_coordinates

    self.mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, "Analyzer", autospec=True)
    )
    # Assume the input is using spec1.
    self.mock_analyzer.get_historical_spend.return_value = _AGGREGATED_SPEND_1

    self.mock_analyzer.response_curves.return_value = (
        _create_response_curve_data()
    )

    # Assume the input is using spec1.
    def optimal_freq_side_effect(use_kpi=False, **_):
      if use_kpi:
        return _create_optimal_freq_data(
            spec_number=1, frequencies=_FREQUENCY_RANGE, is_revenue_kpi=False
        )
      else:
        return _create_optimal_freq_data(
            spec_number=1, frequencies=_FREQUENCY_RANGE, is_revenue_kpi=True
        )

    self.mock_analyzer.optimal_freq.side_effect = optimal_freq_side_effect

    self.mock_trained_model = self.enter_context(
        mock.patch.object(model_processor, "TrainedModel", autospec=True)
    )
    self.mock_trained_model.mmm = self.mock_meridian_model
    self.mock_trained_model.internal_analyzer = self.mock_analyzer

    self.mock_ensure_trained_model = self.enter_context(
        mock.patch.object(
            model_processor, "ensure_trained_model", autospec=True
        )
    )
    self.mock_ensure_trained_model.return_value = self.mock_trained_model

  def test_spec_type_returns_reach_frequency_optimization_spec(self):
    self.assertEqual(
        rfop.ReachFrequencyOptimizationProcessor.spec_type(),
        rfop.ReachFrequencyOptimizationSpec,
    )

  def test_output_type_returns_reach_frequency_optimization_proto(self):
    self.assertEqual(
        rfop.ReachFrequencyOptimizationProcessor.output_type(),
        rf_pb.ReachFrequencyOptimization,
    )

  def test_raises_error_when_model_is_missing_rf_channels(self):
    with self.assertRaisesRegex(
        ValueError,
        "RF channels must be set in the model.",
    ):
      self.mock_meridian_model.input_data.rf_channel = None
      spec = rfop.ReachFrequencyOptimizationSpec(
          start_date=_START_DATE_SPEC_1,
          end_date=_END_DATE_SPEC_1,
          optimization_name=_OPTIMIZATION_NAME_1,
          grid_name=_GRID_NAME,
          min_frequency=49.0,
      )

      rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model).execute(
          [spec]
      )

  def test_execute_when_min_freq_is_not_set(self):
    expected_freq_grid = [1.0, 1.1, 1.2]
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        max_frequency=expected_freq_grid[-1],
    )

    actual = (
        rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model)
        .execute([spec])
        .results[0]
    )

    for rf_c_constraint in actual.spec.rf_channel_constraints:
      self.assertEqual(
          rf_c_constraint.frequency_constraint.min_frequency,
          expected_freq_grid[0],
      )

    self.mock_analyzer.optimal_freq.assert_called_once_with(
        selected_times=_ALL_TIMES[_SPEC_1_TIME_INDEX],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        freq_grid=expected_freq_grid,
        use_kpi=False,
    )

  def test_execute_when_max_freq_is_not_set(self):
    expected_freq_grid = [49.8, 49.9, 50.0]  # 50.0 is the model max frequency.
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        min_frequency=expected_freq_grid[0],
    )

    actual = (
        rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model)
        .execute([spec])
        .results[0]
    )

    for rf_c_constraint in actual.spec.rf_channel_constraints:
      self.assertEqual(
          rf_c_constraint.frequency_constraint.max_frequency,
          expected_freq_grid[-1],
      )

    self.mock_analyzer.optimal_freq.assert_called_once_with(
        selected_times=_ALL_TIMES[_SPEC_1_TIME_INDEX],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        freq_grid=expected_freq_grid,
        use_kpi=False,
    )

  def test_execute_when_rounding_min_and_max_frequency(self):
    min_freq = 1.23456789
    max_freq = 2.16789
    expected_freq_grid = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]

    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        min_frequency=min_freq,
        max_frequency=max_freq,
    )

    actual = (
        rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model)
        .execute([spec])
        .results[0]
    )

    for rf_c_constraint in actual.spec.rf_channel_constraints:
      self.assertEqual(
          rf_c_constraint.frequency_constraint.min_frequency,
          expected_freq_grid[0],
      )
      self.assertEqual(
          rf_c_constraint.frequency_constraint.max_frequency,
          expected_freq_grid[-1],
      )

    self.mock_analyzer.optimal_freq.assert_called_once_with(
        selected_times=_ALL_TIMES[_SPEC_1_TIME_INDEX],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        freq_grid=expected_freq_grid,
        use_kpi=False,
    )

  def test_execute_with_one_spec_revenue_type(self):
    self.mock_meridian_model.input_data.revenue_per_kpi = True

    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        min_frequency=_FREQUENCY_RANGE[0],
        max_frequency=_FREQUENCY_RANGE[-1],
    )

    results = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([spec])

    expected_results = rf_pb.ReachFrequencyOptimization(
        results=[
            _create_expected_optimization_result_proto(
                is_revenue_type=True,
                spec_number=1,
                spec=spec,
                frequencies=_FREQUENCY_RANGE,
            )
        ]
    )

    compare.assertProto2Equal(
        self, expected_results, results, precision=_PRECISION
    )

    self.mock_analyzer.optimal_freq.assert_called_once_with(
        selected_times=_ALL_TIMES[_SPEC_1_TIME_INDEX],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        freq_grid=_FREQUENCY_RANGE,
        use_kpi=False,
    )

  def test_execute_when_default_start_and_end_dates(self):
    actual = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([
        rfop.ReachFrequencyOptimizationSpec(
            optimization_name=_OPTIMIZATION_NAME_1,
            grid_name=_GRID_NAME,
        )
    ])

    self.assertEqual(
        actual.results[0].spec.date_interval,
        date_interval_pb2.DateInterval(
            start_date=date_pb2.Date(year=2024, month=1, day=1),
            end_date=date_pb2.Date(year=2024, month=1, day=29),
        ),
    )

  def test_execute_when_no_start_date_defined(self):
    actual = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([
        rfop.ReachFrequencyOptimizationSpec(
            end_date=_END_DATE_SPEC_1,
            optimization_name=_OPTIMIZATION_NAME_1,
            grid_name=_GRID_NAME,
        )
    ])

    self.assertEqual(
        actual.results[0].spec.date_interval,
        date_interval_pb2.DateInterval(
            start_date=_START_PROTO_DATE_SPEC_1,
            end_date=_END_PROTO_DATE_SPEC_1,
        ),
    )

  def test_execute_when_no_end_date_defined(self):
    actual = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([
        rfop.ReachFrequencyOptimizationSpec(
            start_date=_START_DATE_SPEC_1,
            optimization_name=_OPTIMIZATION_NAME_1,
            grid_name=_GRID_NAME,
        )
    ])

    self.assertEqual(
        actual.results[0].spec.date_interval,
        date_interval_pb2.DateInterval(
            start_date=_START_PROTO_DATE_SPEC_1,
            end_date=date_pb2.Date(year=2024, month=1, day=29),
        ),
    )

  def test_execute_with_one_spec_nonrevenue_type(self):
    spec = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        min_frequency=49.0,
        kpi_type=common.KpiType.NON_REVENUE,
    )

    actual = (
        rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model)
        .execute([spec])
        .results[0]
    )

    for c_analysis in actual.optimized_marketing_analysis.media_analyses:
      self.assertEqual(
          c_analysis.media_outcomes[0].kpi_type,
          kpi_type_pb.KpiType.NON_REVENUE,
      )

  def test_execute_with_multiple_specs_in_revenue_type(self):
    frequencies = [
        19.0,
        19.1,
        19.2,
        19.3,
        19.4,
        19.5,
        19.6,
        19.7,
        19.8,
        19.9,
        20.0,
    ]
    self.mock_analyzer.optimal_freq.side_effect = [
        _create_optimal_freq_data(
            spec_number=1, frequencies=frequencies, is_revenue_kpi=True
        ),
        _create_optimal_freq_data(
            spec_number=2, frequencies=frequencies, is_revenue_kpi=True
        ),
    ]
    self.mock_meridian_model.input_data.revenue_per_kpi = True
    self.mock_analyzer.get_historical_spend.side_effect = [
        _AGGREGATED_SPEND_1,
        _AGGREGATED_SPEND_2,
    ]

    spec_1 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id="group",
        min_frequency=19,
        max_frequency=20,
    )
    spec_2 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_2,
        end_date=_END_DATE_SPEC_2,
        optimization_name=_OPTIMIZATION_NAME_2,
        grid_name=_GRID_NAME,
        min_frequency=19,
        max_frequency=20,
    )

    results = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([spec_1, spec_2])

    expected_results = rf_pb.ReachFrequencyOptimization(
        results=[
            _create_expected_optimization_result_proto(
                is_revenue_type=True,
                spec_number=1,
                spec=spec_1,
                frequencies=frequencies,
            ),
            _create_expected_optimization_result_proto(
                is_revenue_type=True,
                spec_number=2,
                spec=spec_2,
                frequencies=frequencies,
            ),
        ]
    )

    compare.assertProto2Equal(
        self, expected_results, results, precision=_PRECISION
    )

  def test_execute_multiple_specs_duplicate_group_id(self):
    spec_1 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id="dupe-group",
        min_frequency=19,
        max_frequency=20,
    )
    spec_2 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_2,
        end_date=_END_DATE_SPEC_2,
        optimization_name=_OPTIMIZATION_NAME_2,
        grid_name=_GRID_NAME,
        group_id="dupe-group",
        min_frequency=19,
        max_frequency=20,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Specified group_id must be unique",
    ):
      rfop.ReachFrequencyOptimizationProcessor(self.mock_trained_model).execute(
          [spec_1, spec_2]
      )

  def test_execute_multiple_specs_empty_group_id_counts_as_unique(self):
    frequencies = [
        19.0,
        19.1,
        19.2,
        19.3,
        19.4,
        19.5,
        19.6,
        19.7,
        19.8,
        19.9,
        20.0,
    ]
    self.mock_analyzer.optimal_freq.side_effect = [
        _create_optimal_freq_data(
            spec_number=1, frequencies=frequencies, is_revenue_kpi=True
        ),
        _create_optimal_freq_data(
            spec_number=2, frequencies=frequencies, is_revenue_kpi=True
        ),
    ]
    self.mock_meridian_model.input_data.revenue_per_kpi = True
    self.mock_analyzer.get_historical_spend.side_effect = [
        _AGGREGATED_SPEND_1,
        _AGGREGATED_SPEND_2,
    ]

    spec_1 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_1,
        end_date=_END_DATE_SPEC_1,
        optimization_name=_OPTIMIZATION_NAME_1,
        grid_name=_GRID_NAME,
        group_id="group",
        min_frequency=19,
        max_frequency=20,
    )
    spec_2 = rfop.ReachFrequencyOptimizationSpec(
        start_date=_START_DATE_SPEC_2,
        end_date=_END_DATE_SPEC_2,
        optimization_name=_OPTIMIZATION_NAME_2,
        grid_name=_GRID_NAME,
        min_frequency=19,
        max_frequency=20,
    )

    results = rfop.ReachFrequencyOptimizationProcessor(
        self.mock_trained_model
    ).execute([spec_1, spec_2])

    expected_results = rf_pb.ReachFrequencyOptimization(
        results=[
            _create_expected_optimization_result_proto(
                is_revenue_type=True,
                spec_number=1,
                spec=spec_1,
                frequencies=frequencies,
            ),
            _create_expected_optimization_result_proto(
                is_revenue_type=True,
                spec_number=2,
                spec=spec_2,
                frequencies=frequencies,
            ),
        ]
    )

    compare.assertProto2Equal(
        self, expected_results, results, precision=_PRECISION
    )


if __name__ == "__main__":
  absltest.main()
