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

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.data import time_coordinates
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_pb
from mmm.v1.marketing.analysis import kpi_outcome_pb2 as kpi_outcome_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_analysis_pb
from mmm.v1.marketing.analysis import response_curve_pb2 as response_curve_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from schema.processors import budget_optimization_processor as proc
from schema.processors import common
from schema.processors import model_processor
import numpy as np
import tensorflow as tf
import xarray as xr

from google.type import date_pb2
from tensorflow.python.util.protobuf import compare


_TAG = "YEAR_2024"

_START_DATE = datetime.date(2024, 1, 1)
_START_DATE_STR = "2024-01-01"
_START_DATE_PROTO = date_pb2.Date(year=2024, month=1, day=1)

# Open / exclusive end date.
_END_DATE = datetime.date(2024, 12, 30)
_END_DATE_STR = "2024-12-30"
_END_DATE_PROTO = date_pb2.Date(year=2024, month=12, day=30)

# Closed / inclusive end date.
_ADJUSTED_END_DATE_STR = "2024-12-23"

_OPTIMIZATION_NAME = "optimization_name"
_GRID_NAME = "grid_name"
_GROUP_ID = "group_id"
_BUDGET_AMOUNT = 300_000

_NEW_DATA = analyzer.DataTensors(
    media=tf.convert_to_tensor([[
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ]]),
    revenue_per_kpi=tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]]),
    time=tf.convert_to_tensor(
        ["2024-01-29", "2024-02-05", "2024-02-12", "2024-02-19"]
    ),
)

_BASE_BUDGET_OPT_INPUT_SPEC = proc.BudgetOptimizationSpec(
    start_date=_START_DATE,
    end_date=_END_DATE,
    optimization_name=_OPTIMIZATION_NAME,
    grid_name=_GRID_NAME,
    group_id=_GROUP_ID,
    scenario=optimizer.FixedBudgetScenario(total_budget=_BUDGET_AMOUNT),
    kpi_type=common.KpiType.REVENUE,
)

_HIST_SPEND_BY_CHANNEL = {
    "channel1": 100_000,
    "channel2": 200_000,
}

_CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO = (
    proc.CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO
)
_CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO = (
    proc.CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO
)

_BASE_BUDGET_OPT_OUTPUT_SPEC = dataclasses.replace(
    _BASE_BUDGET_OPT_INPUT_SPEC,
    # Resolve to absolute budget constraint values from default ratios above.
    constraints=[
        proc.ChannelConstraintAbs(
            channel_name="channel1",
            abs_lowerbound=(
                _HIST_SPEND_BY_CHANNEL["channel1"]
                * (1 - _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO)
            ),
            abs_upperbound=(
                _HIST_SPEND_BY_CHANNEL["channel1"]
                * (1 + _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO)
            ),
        ),
        proc.ChannelConstraintAbs(
            channel_name="channel2",
            abs_lowerbound=(
                _HIST_SPEND_BY_CHANNEL["channel2"]
                * (1 - _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO)
            ),
            abs_upperbound=(
                _HIST_SPEND_BY_CHANNEL["channel2"]
                * (1 + _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO)
            ),
        ),
    ],
)

# Expected output proto transcription of the above dataclass spec.
_BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC = budget_pb.BudgetOptimizationSpec(
    date_interval=date_interval_pb2.DateInterval(
        start_date=_START_DATE_PROTO,
        end_date=_END_DATE_PROTO,
    ),
    objective=target_pb.TargetMetric.KPI,
    fixed_budget_scenario=budget_pb.FixedBudgetScenario(
        total_budget=_BUDGET_AMOUNT
    ),
    channel_constraints=[
        budget_pb.ChannelConstraint(
            channel_name="channel1",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=(
                    _HIST_SPEND_BY_CHANNEL["channel1"]
                    * (1 - _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO)
                ),
                max_budget=(
                    _HIST_SPEND_BY_CHANNEL["channel1"]
                    * (1 + _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO)
                ),
            ),
        ),
        budget_pb.ChannelConstraint(
            channel_name="channel2",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=(
                    _HIST_SPEND_BY_CHANNEL["channel2"]
                    * (1 - _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO)
                ),
                max_budget=(
                    _HIST_SPEND_BY_CHANNEL["channel2"]
                    * (1 + _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO)
                ),
            ),
        ),
    ],
    kpi_type=kpi_type_pb.KpiType.REVENUE,
)


class BudgetOptimizationSpecTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._spec_date = date_interval_pb2.DateInterval(
        start_date=_START_DATE_PROTO,
        end_date=_END_DATE_PROTO,
    )

    self._dated_spec_resolver = mock.MagicMock()
    self._dated_spec_resolver.resolve_to_date_interval_open_end.return_value = (
        _START_DATE,
        _END_DATE,
    )
    self._dated_spec_resolver.resolve_to_date_interval_proto.return_value = (
        self._spec_date
    )

  def test_invalid_spec_new_data_no_times(self):
    with self.assertRaisesRegex(
        ValueError, "`time` must be provided in `new_data`."
    ):
      proc.BudgetOptimizationSpec(
          start_date=_START_DATE,
          end_date=_END_DATE,
          optimization_name=_OPTIMIZATION_NAME,
          grid_name=_GRID_NAME,
          group_id=_GROUP_ID,
          scenario=optimizer.FixedBudgetScenario(total_budget=_BUDGET_AMOUNT),
          kpi_type=common.KpiType.REVENUE,
          new_data=_NEW_DATA.filter_fields([c.MEDIA, c.REVENUE_PER_KPI]),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="revenue",
          kpi_type=common.KpiType.REVENUE,
      ),
      dict(
          testcase_name="non_revenue",
          kpi_type=common.KpiType.NON_REVENUE,
      ),
  )
  def test_to_proto_no_channel_constraints(self, kpi_type: common.KpiType):
    spec = dataclasses.replace(_BASE_BUDGET_OPT_INPUT_SPEC, kpi_type=kpi_type)
    expected_proto = budget_pb.BudgetOptimizationSpec()
    expected_proto.CopyFrom(_BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC)
    expected_proto.kpi_type = (
        kpi_type_pb.REVENUE
        if kpi_type == common.KpiType.REVENUE
        else kpi_type_pb.KpiType.NON_REVENUE
    )
    expected_proto.ClearField("channel_constraints")

    compare.assertProto2Equal(self, spec.to_proto(), expected_proto)

  def test_to_proto_fixed_budget(self):
    # With default channel constraints.
    compare.assertProto2Equal(
        self,
        _BASE_BUDGET_OPT_OUTPUT_SPEC.to_proto(),
        _BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC,
    )

  def test_to_proto_raises_error_on_default_scenario(self):
    spec = proc.BudgetOptimizationSpec(
        start_date=_START_DATE,
        end_date=_END_DATE,
        optimization_name=_OPTIMIZATION_NAME,
        grid_name=_GRID_NAME,
        kpi_type=common.KpiType.REVENUE,
    )

    with self.assertRaises(ValueError):
      spec.to_proto()

  def test_to_proto_flexible_budget_roi(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_OUTPUT_SPEC,
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.ROI,
            target_value=1.0,
        ),
    )

    expected_proto = budget_pb.BudgetOptimizationSpec()
    expected_proto.CopyFrom(_BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC)
    expected_proto.fixed_budget_scenario.Clear()
    expected_proto.flexible_budget_scenario.CopyFrom(
        budget_pb.FlexibleBudgetScenario(
            target_metric_constraints=[
                constraints_pb.TargetMetricConstraint(
                    target_metric=target_pb.TargetMetric.ROI,
                    target_value=1.0,
                )
            ],
        )
    )

    compare.assertProto2Equal(self, spec.to_proto(), expected_proto)

  def test_to_proto_flexible_budget_mroi(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_OUTPUT_SPEC,
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.MROI,
            target_value=1.0,
        ),
    )

    expected_proto = budget_pb.BudgetOptimizationSpec()
    expected_proto.CopyFrom(_BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC)
    expected_proto.fixed_budget_scenario.Clear()
    expected_proto.flexible_budget_scenario.CopyFrom(
        budget_pb.FlexibleBudgetScenario(
            target_metric_constraints=[
                constraints_pb.TargetMetricConstraint(
                    target_metric=target_pb.TargetMetric.MARGINAL_ROI,
                    target_value=1.0,
                ),
            ],
        )
    )

    compare.assertProto2Equal(self, spec.to_proto(), expected_proto)

  def test_to_proto_with_tag(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_OUTPUT_SPEC,
        date_interval_tag=_TAG,
    )

    expected_proto = budget_pb.BudgetOptimizationSpec()
    expected_proto.CopyFrom(_BASE_BUDGET_OPT_OUTPUT_PROTO_SPEC)
    expected_proto.date_interval.tag = _TAG

    compare.assertProto2Equal(self, spec.to_proto(), expected_proto)

  @parameterized.named_parameters(
      dict(
          testcase_name="inverted_dates",
          kwargs=dict(
              start_date=_END_DATE,
              end_date=_START_DATE,
              scenario=optimizer.FixedBudgetScenario(total_budget=300_000),
          ),
      ),
      dict(
          testcase_name="invalid_confidence_level",
          kwargs=dict(
              start_date=_START_DATE,
              end_date=_END_DATE,
              scenario=optimizer.FixedBudgetScenario(total_budget=300_000),
              confidence_level=2.0,
          ),
      ),
      dict(
          testcase_name="invalid_confidence_level_negative",
          kwargs=dict(
              start_date=_START_DATE,
              end_date=_END_DATE,
              scenario=optimizer.FixedBudgetScenario(total_budget=300_000),
              constraints=[],
              confidence_level=-0.5,
          ),
      ),
  )
  def test_invalid_input_spec(self, kwargs):
    with self.assertRaises(ValueError):
      dataclasses.replace(_BASE_BUDGET_OPT_INPUT_SPEC, **kwargs)

    with self.assertRaises(ValueError):
      _ = dataclasses.replace(_BASE_BUDGET_OPT_OUTPUT_SPEC, **kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_lowerbound",
          kwargs=dict(
              channel_name="channel1",
              spend_constraint_lower=-0.1,
              spend_constraint_upper=0.2,
          ),
      ),
      dict(
          testcase_name="invalid_upperbound",
          kwargs=dict(
              channel_name="channel1",
              spend_constraint_lower=0.1,
              spend_constraint_upper=-0.2,
          ),
      ),
      dict(
          testcase_name="lowerbound_greater_than_1",
          kwargs=dict(
              channel_name="channel1",
              spend_constraint_lower=10,
              spend_constraint_upper=10,
          ),
      ),
  )
  def test_invalid_relative_channel_constraint(self, kwargs):
    with self.assertRaises(ValueError):
      proc.ChannelConstraintRel(**kwargs)


def _create_budget_data(
    spend: np.ndarray,
    inc_outcome: np.ndarray,
    pct_contrib: np.ndarray,
    effectiveness: np.ndarray,
    mroi: np.ndarray,
    channels: np.ndarray | None = None,
    attrs: Mapping[str, Any] | None = None,
    confidence_level: float = 0.9,
) -> xr.Dataset:
  channels = (
      [f"channel{i + 1}" for i in range(len(spend))]
      if channels is None
      else channels
  )
  data_vars = {
      c.SPEND: ([c.CHANNEL], spend),
      c.PCT_OF_SPEND: ([c.CHANNEL], spend / sum(spend)),
      c.INCREMENTAL_OUTCOME: ([c.CHANNEL, c.METRIC], inc_outcome),
      c.PCT_OF_CONTRIBUTION: ([c.CHANNEL, c.METRIC], pct_contrib),
      c.EFFECTIVENESS: ([c.CHANNEL, c.METRIC], effectiveness),
      c.CPIK: ([c.CHANNEL, c.METRIC], spend.reshape(-1, 1) / inc_outcome),
      c.ROI: (
          [c.CHANNEL, c.METRIC],
          tf.math.divide_no_nan(inc_outcome, spend.reshape(-1, 1)),
      ),
      c.MROI: ([c.CHANNEL, c.METRIC], mroi),
  }
  attributes = {
      c.START_DATE: _START_DATE_STR,
      c.END_DATE: _END_DATE_STR,
      c.BUDGET: sum(spend),
      c.PROFIT: sum(inc_outcome[:, 0]) - sum(spend),
      c.TOTAL_INCREMENTAL_OUTCOME: sum(inc_outcome[:, 0]),
      c.CONFIDENCE_LEVEL: confidence_level,
      c.TOTAL_CPIK: sum(spend) / sum(inc_outcome[:, 0]),
      c.TOTAL_ROI: sum(inc_outcome[:, 0]) / sum(spend),
  }
  return xr.Dataset(
      data_vars=data_vars,
      coords={
          c.CHANNEL: ([c.CHANNEL], channels),
          c.METRIC: ([c.METRIC], [c.MEAN, c.CI_LO, c.CI_HI]),
      },
      attrs=attributes | (attrs or {}),
  )


_SAMPLE_OPTIMIZED_DATA_REVENUE = _create_budget_data(
    spend=np.array([200_000, 100_000]),
    # Shape (n_media, n_CI_metrics)
    inc_outcome=np.array([
        [350, 349, 351],
        [210, 209, 211],
    ]),
    # Shape (n_media, n_CI_metrics)
    pct_contrib=np.array([
        [0.1, 0.05, 0.2],
        [0.2, 0.1, 0.3],
    ]),
    # Shape (n_media, n_CI_metrics)
    effectiveness=np.array([
        [0.16, 0.039, 0.29],
        [0.32, 0.12, 0.53],
    ]),
    # Shape (n_media, n_CI_metrics)
    mroi=np.array([
        [1.4, 1.4, 1.4],
        [1.5, 1.5, 1.5],
    ]),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: True,
    },
)

_SAMPLE_NONOPTIMIZED_DATA_REVENUE = _create_budget_data(
    spend=np.array([100_000, 200_000]),
    # Shape (n_media, n_CI_metrics)
    inc_outcome=np.array([
        [450, 449, 451],
        [310, 309, 311],
    ]),
    # Shape (n_media, n_CI_metrics)
    pct_contrib=np.array([
        [0.2, 0.15, 0.3],
        [0.3, 0.2, 0.4],
    ]),
    # Shape (n_media, n_CI_metrics)
    effectiveness=np.array([
        [0.26, 0.049, 0.39],
        [0.42, 0.22, 0.63],
    ]),
    # Shape (n_media, n_CI_metrics)
    mroi=np.array([
        [2.4, 2.4, 2.4],
        [2.5, 2.5, 2.5],
    ]),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: True,
    },
)

_SAMPLE_OPTIMIZED_DATA_NONREVENUE = _create_budget_data(
    spend=np.array([200_000, 100_000]),
    # Shape (n_media, n_CI_metrics)
    inc_outcome=np.array([
        [350, 349, 351],
        [210, 209, 211],
    ]),
    # Shape (n_media, n_CI_metrics)
    pct_contrib=np.array([
        [0.1, 0.05, 0.2],
        [0.2, 0.1, 0.3],
    ]),
    # Shape (n_media, n_CI_metrics)
    effectiveness=np.array([
        [0.16, 0.039, 0.29],
        [0.32, 0.12, 0.53],
    ]),
    # Shape (n_media, n_CI_metrics)
    mroi=np.array([
        [2.4, 2.4, 2.4],
        [2.5, 2.5, 2.5],
    ]),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: False,
    },
)

_SAMPLE_NONOPTIMIZED_DATA_NONREVENUE = _create_budget_data(
    spend=np.array([100_000, 200_000]),
    # Shape (n_media, n_CI_metrics)
    inc_outcome=np.array([
        [450, 449, 451],
        [310, 309, 311],
    ]),
    # Shape (n_media, n_CI_metrics)
    pct_contrib=np.array([
        [0.2, 0.06, 0.3],
        [0.3, 0.2, 0.4],
    ]),
    # Shape (n_media, n_CI_metrics)
    effectiveness=np.array([
        [0.26, 0.049, 0.39],
        [0.42, 0.22, 0.63],
    ]),
    # Shape (n_media, n_CI_metrics)
    mroi=np.array([
        [3.4, 3.4, 3.4],
        [3.5, 3.5, 3.5],
    ]),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: False,
    },
)


def _create_response_curves_dataset(channels: Sequence[str]) -> xr.Dataset:
  # A response curves dataset has: (channel, metric, and spend_multiplier)
  # dimension coordinates, and (spend, incremental_outcome) data variables.
  spend_multipliers = np.arange(0, 2, 0.5)  # 4 values

  spends = []
  n_channels = len(channels)
  for mult in spend_multipliers:
    ch_spends = [10 * i * mult for i in range(1, n_channels + 1)]
    spends.append(ch_spends)

  outcomes = []
  for mult in spend_multipliers:
    mult_outcomes = []
    for _, _ in enumerate(channels):
      mean = 0.1 + mult
      mult_outcomes.append([mean])
    outcomes.append(mult_outcomes)

  data_vars = {
      c.SPEND: ([c.SPEND_MULTIPLIER, c.CHANNEL], spends),
      c.INCREMENTAL_OUTCOME: (
          [c.SPEND_MULTIPLIER, c.CHANNEL, c.METRIC],
          outcomes,
      ),
  }
  return xr.Dataset(
      data_vars=data_vars,
      coords={
          c.CHANNEL: ([c.CHANNEL], channels),
          c.METRIC: ([c.METRIC], [c.MEAN]),
          c.SPEND_MULTIPLIER: ([c.SPEND_MULTIPLIER], spend_multipliers),
      },
  )


_CHANNELS = ["channel1", "channel2"]

_SAMPLE_RESPONSE_CURVES_DATA = _create_response_curves_dataset(_CHANNELS)

_SPEND_STEP_SIZE = 200

_SPEND_GRID = np.array(
    [
        [500.0, 900.0],
        [700.0, 1100.0],
        [900.0, np.nan],
        [1100.0, np.nan],
        [1300.0, np.nan],
        [1500.0, np.nan],
    ],
)
_INCREMENTAL_OUTCOME_GRID = np.array(
    [
        [1.0, 0.81818182],
        [1.0, 1.0],
        [1.0, np.nan],
        [1.0, np.nan],
        [1.0, np.nan],
        [1.0, np.nan],
    ],
)


def _create_optimization_grid_dataset(
    spend_grid: np.ndarray = _SPEND_GRID,
    incremental_outcome_grid: np.ndarray = _INCREMENTAL_OUTCOME_GRID,
    spend_step_size: float = _SPEND_STEP_SIZE,
    channels: Sequence[str] | None = None,
) -> xr.Dataset:
  data_vars = {
      c.SPEND_GRID: ([c.GRID_SPEND_INDEX, c.CHANNEL], spend_grid),
      c.INCREMENTAL_OUTCOME_GRID: (
          [c.GRID_SPEND_INDEX, c.CHANNEL],
          incremental_outcome_grid,
      ),
  }
  return xr.Dataset(
      data_vars=data_vars,
      coords={
          c.GRID_SPEND_INDEX: (
              [c.GRID_SPEND_INDEX],
              np.arange(0, len(spend_grid)),
          ),
          c.CHANNEL: ([c.CHANNEL], channels or _CHANNELS),
      },
      attrs={c.SPEND_STEP_SIZE: spend_step_size},
  )


def _create_optimization_result_spec(
    start_date_proto: date_pb2.Date = _START_DATE_PROTO,
    end_date_proto: date_pb2.Date = _END_DATE_PROTO,
    total_budget: float = _BUDGET_AMOUNT,
    kpi_type: kpi_type_pb.KpiType = kpi_type_pb.KpiType.REVENUE,
):
  channel_constraints = [
      budget_pb.ChannelConstraint(
          channel_name="channel1",
          budget_constraint=constraints_pb.BudgetConstraint(
              min_budget=0,
              max_budget=300_000,
          ),
      ),
      budget_pb.ChannelConstraint(
          channel_name="channel2",
          budget_constraint=constraints_pb.BudgetConstraint(
              min_budget=0,
              max_budget=600_000,
          ),
      ),
  ]

  return budget_pb.BudgetOptimizationSpec(
      date_interval=date_interval_pb2.DateInterval(
          start_date=start_date_proto,
          end_date=end_date_proto,
      ),
      objective=target_pb.TargetMetric.KPI,
      fixed_budget_scenario=budget_pb.FixedBudgetScenario(
          total_budget=total_budget
      ),
      channel_constraints=channel_constraints,
      kpi_type=kpi_type,
  )


_BASE_BUDGET_OPT_RESULT_SPEC_PROTO = _create_optimization_result_spec()


class BudgetOptimizationProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_optimizer = mock.MagicMock()
    self.mock_mmm = mock.MagicMock()
    self.trained_model = mock.MagicMock()
    self.trained_model.mmm = self.mock_mmm

    self.mock_mmm.input_data.get_all_paid_channels.return_value = [
        "channel1",
        "channel2",
    ]
    self.all_dates = [
        _START_DATE + datetime.timedelta(weeks=x)
        for x in range(int((_END_DATE - _START_DATE).days / 7))
    ]
    self.mock_mmm.input_data.revenue_per_kpi = None

    self.mock_analyzer = mock.MagicMock()
    self.mock_optimizer = mock.MagicMock()
    self.trained_model.internal_analyzer = self.mock_analyzer
    self.trained_model.internal_optimizer = self.mock_optimizer
    self.trained_model.time_coordinates = (
        time_coordinates.TimeCoordinates.from_dates(self.all_dates)
    )

    self.mock_ensure_trained_model = self.enter_context(
        mock.patch.object(
            model_processor, "ensure_trained_model", autospec=True
        )
    )
    self.mock_ensure_trained_model.return_value = self.trained_model

    self.processor = proc.BudgetOptimizationProcessor(self.trained_model)

    self.mock_optimization_results = mock.MagicMock()
    self.mock_optimizer.optimize.return_value = self.mock_optimization_results
    self.mock_optimization_results.get_response_curves.return_value = (
        _SAMPLE_RESPONSE_CURVES_DATA
    )
    self.mock_optimization_results.optimization_grid.grid_dataset = (
        _create_optimization_grid_dataset()
    )
    self.mock_optimization_results.optimized_data.attrs = {
        c.BUDGET: _BUDGET_AMOUNT
    }
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    )
    self.mock_optimization_results.optimization_grid.round_factor = -2

  def test_spec_type(self):
    self.assertEqual(
        proc.BudgetOptimizationSpec,
        proc.BudgetOptimizationProcessor.spec_type(),
    )

  def test_output_type(self):
    self.assertEqual(
        budget_pb.BudgetOptimization,
        proc.BudgetOptimizationProcessor.output_type(),
    )

  def test_execute_output_spec_channel_constraints(self):
    opt = self.processor.execute([_BASE_BUDGET_OPT_INPUT_SPEC])
    result = opt.results[0]
    output_spec_proto = result.spec

    compare.assertProto2Equal(
        self, output_spec_proto, _BASE_BUDGET_OPT_RESULT_SPEC_PROTO
    )

  def test_execute_output_spec_invalid_extra_channel_constraints(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        constraints=[
            proc.ChannelConstraintRel(
                channel_name="channel1",
                spend_constraint_lower=0.1,
                spend_constraint_upper=0.2,
            ),
            proc.ChannelConstraintRel(
                channel_name="channel3",  # "channel3" is not in the mock model.
                spend_constraint_lower=0.3,
                spend_constraint_upper=0.4,
            ),
        ],
    )
    with self.assertRaisesRegex(
        ValueError,
        "Channel constraints must have channel names that are in the model",
    ):
      self.processor.execute([spec])

  def test_execute_populate_spec_correctly(self):
    spec = _BASE_BUDGET_OPT_INPUT_SPEC

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)

    opt_result = opt.results[0]

    with self.subTest("Name"):
      self.assertEqual(
          opt_result.name, _BASE_BUDGET_OPT_OUTPUT_SPEC.optimization_name
      )
      self.assertEqual(
          opt_result.group_id, _BASE_BUDGET_OPT_OUTPUT_SPEC.group_id
      )

    with self.subTest("Spec"):
      compare.assertProto2Equal(
          self, opt_result.spec, _BASE_BUDGET_OPT_RESULT_SPEC_PROTO
      )

    marketing_analysis = opt_result.optimized_marketing_analysis

    with self.subTest("DateInterval"):
      compare.assertProto2Equal(
          self,
          marketing_analysis.date_interval,
          date_interval_pb2.DateInterval(
              start_date=_START_DATE_PROTO,
              end_date=_END_DATE_PROTO,
          ),
      )

    with self.subTest("OptimizationGridName"):
      self.assertEqual(
          opt_result.incremental_outcome_grid.name,
          _BASE_BUDGET_OPT_OUTPUT_SPEC.grid_name,
      )

  def test_execute_without_response_curves(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        include_response_curves=False,
    )
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)
    opt_result = opt.results[0]
    media_analyses = opt_result.optimized_marketing_analysis.media_analyses
    nonoptimized_media_analyses = (
        opt_result.nonoptimized_marketing_analysis.media_analyses
    )
    for media_analysis in media_analyses:
      compare.assertProto2Equal(
          self,
          media_analysis.response_curve,
          response_curve_pb.ResponseCurve(),
      )
    for media_analysis in nonoptimized_media_analyses:
      compare.assertProto2Equal(
          self,
          media_analysis.response_curve,
          response_curve_pb.ResponseCurve(),
      )

  def test_execute_populate_spec_when_start_date_is_none(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        start_date=None,
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)
    opt_result = opt.results[0]

    with self.subTest("Spec"):
      compare.assertProto2Equal(
          self, opt_result.spec, _BASE_BUDGET_OPT_RESULT_SPEC_PROTO
      )

  def test_execute_populate_spec_when_end_date_is_none(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        end_date=None,
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)
    opt_result = opt.results[0]

    with self.subTest("Spec"):
      compare.assertProto2Equal(
          self, opt_result.spec, _BASE_BUDGET_OPT_RESULT_SPEC_PROTO
      )

  def test_execute_fixed_budget_no_constraints(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        constraints=[],
    )

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    compare.assertProto2Equal(
        self, opt.results[0].spec, _BASE_BUDGET_OPT_RESULT_SPEC_PROTO
    )

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=True,
        budget=300_000,
        confidence_level=0.9,
        # Expect that default channel constraints were synthesized.
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="revenue",
          kpi_type=common.KpiType.REVENUE,
          use_kpi=False,
      ),
      dict(
          testcase_name="non_revenue",
          kpi_type=common.KpiType.NON_REVENUE,
          use_kpi=True,
      ),
  )
  def test_execute_default_scenario_no_constraints(
      self, kpi_type: common.KpiType, use_kpi: bool
  ):
    spec = proc.BudgetOptimizationSpec(
        start_date=_START_DATE,
        end_date=_END_DATE,
        optimization_name=_OPTIMIZATION_NAME,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        # no `scenario` specified explicitly
        kpi_type=kpi_type,
    )

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    spec_result = opt.results[0].spec

    expected_spec = _create_optimization_result_spec(
        kpi_type=(
            kpi_type_pb.KpiType.REVENUE
            if kpi_type == common.KpiType.REVENUE
            else kpi_type_pb.KpiType.NON_REVENUE
        ),
    )
    compare.assertProto2Equal(self, expected_spec, spec_result)

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=True,
        # No `budget` kwarg specified.
        confidence_level=0.9,
        # Expect that default channel constraints were synthesized.
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=use_kpi,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_flexible_budget_roi_no_constraints(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.ROI,
            target_value=1.0,
        ),
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)
    output_spec = opt.results[0].spec

    expected_output_spec = budget_pb.BudgetOptimizationSpec()
    expected_output_spec.CopyFrom(_BASE_BUDGET_OPT_RESULT_SPEC_PROTO)
    expected_output_spec.fixed_budget_scenario.Clear()
    expected_output_spec.flexible_budget_scenario.CopyFrom(
        budget_pb.FlexibleBudgetScenario(
            target_metric_constraints=[
                constraints_pb.TargetMetricConstraint(
                    target_metric=target_pb.TargetMetric.ROI,
                    target_value=1.0,
                )
            ],
        )
    )
    compare.assertProto2Equal(self, output_spec, expected_output_spec)

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=False,
        target_roi=1.0,
        confidence_level=0.9,
        # Expect that default channel constraints were synthesized.
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_default_scenario_new_data(self):
    spec = proc.BudgetOptimizationSpec(
        start_date=datetime.date(2024, 2, 5),
        end_date=datetime.date(2024, 2, 19),
        optimization_name=_OPTIMIZATION_NAME,
        grid_name=_GRID_NAME,
        group_id=_GROUP_ID,
        # no `scenario` specified explicitly
        kpi_type=common.KpiType.NON_REVENUE,
        new_data=_NEW_DATA,
    )

    _ = self.processor.execute([spec])

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date="2024-02-05",
        end_date="2024-02-12",
        fixed_budget=True,
        # No `budget` kwarg specified.
        confidence_level=0.9,
        # Expect that default channel constraints were synthesized.
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=True,
        optimization_grid=None,
        new_data=_NEW_DATA,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_channel_name",
          constraints=[
              proc.ChannelConstraintRel(
                  channel_name="channel3",
                  spend_constraint_lower=0.1,
                  spend_constraint_upper=0.2,
              ),
          ],
      ),
      dict(
          testcase_name="extra_channel_name",
          constraints=[
              proc.ChannelConstraintRel(
                  channel_name="channel2",
                  spend_constraint_lower=0.1,
                  spend_constraint_upper=0.2,
              ),
              proc.ChannelConstraintRel(
                  channel_name="channel1",
                  spend_constraint_lower=0.3,
                  spend_constraint_upper=0.4,
              ),
              proc.ChannelConstraintRel(
                  channel_name="channel3",
                  spend_constraint_lower=0.3,
                  spend_constraint_upper=0.4,
              ),
          ],
      ),
  )
  def test_execute_fixed_budget_with_invalid_channels(self, constraints):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        constraints=constraints,
        # default confidence level = 0.9
    )
    with self.assertRaisesRegex(
        ValueError,
        "Channel constraints must have channel names that are in the model",
    ):
      self.processor.execute([spec])

  def test_execute_flexible_budget_mroi_with_constraints(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.MROI,
            target_value=1.0,
        ),
        constraints=[
            proc.ChannelConstraintRel(
                channel_name="channel1",
                spend_constraint_lower=0.1,
                spend_constraint_upper=0.2,
            ),
            proc.ChannelConstraintRel(
                channel_name="channel2",
                spend_constraint_lower=0.3,
                spend_constraint_upper=0.4,
            ),
        ],
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)
    output_spec = opt.results[0].spec

    expected_output_spec = budget_pb.BudgetOptimizationSpec()
    expected_output_spec.CopyFrom(_BASE_BUDGET_OPT_RESULT_SPEC_PROTO)
    expected_output_spec.fixed_budget_scenario.Clear()
    expected_output_spec.flexible_budget_scenario.CopyFrom(
        budget_pb.FlexibleBudgetScenario(
            target_metric_constraints=[
                constraints_pb.TargetMetricConstraint(
                    target_metric=target_pb.TargetMetric.MARGINAL_ROI,
                    target_value=1.0,
                )
            ],
        )
    )
    expected_constraints = [
        budget_pb.ChannelConstraint(
            channel_name="channel1",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=90_000.0, max_budget=120_000.0
            ),
        ),
        budget_pb.ChannelConstraint(
            channel_name="channel2",
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=140_000.0, max_budget=280_000.0
            ),
        ),
    ]
    expected_output_spec.channel_constraints.clear()
    expected_output_spec.channel_constraints.extend(expected_constraints)
    # absolute channel constraints resolution from historical spend should
    # already be included in the base output spec prototype.
    compare.assertProto2Equal(self, expected_output_spec, output_spec)

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=False,
        confidence_level=0.9,
        target_mroi=1.0,
        spend_constraint_lower=[0.1, 0.3],
        spend_constraint_upper=[0.2, 0.4],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_build_constraints_kwargs(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        constraints=[
            proc.ChannelConstraintRel(
                channel_name="channel1",
                spend_constraint_lower=0.1,
                spend_constraint_upper=0.2,
            ),
            proc.ChannelConstraintRel(
                channel_name="channel2",
                spend_constraint_lower=0.3,
                spend_constraint_upper=0.4,
            ),
        ],
        confidence_level=0.85,
    )

    self.processor.execute([spec])

    # `_build_kwargs()` is fed into the internal optimizer's `optimize()` call.
    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=True,
        confidence_level=0.85,
        budget=300_000,
        spend_constraint_lower=[0.1, 0.3],
        spend_constraint_upper=[0.2, 0.4],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_build_constraints_kwargs_partial(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        constraints=[
            proc.ChannelConstraintRel(
                channel_name="channel1",
                spend_constraint_lower=0.1,
                spend_constraint_upper=0.2,
            ),
            # "channel2" is left intentionally unspecified.
        ],
        confidence_level=0.85,
    )

    opt = self.processor.execute([spec])
    self.assertLen(opt.results, 1)

    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=True,
        confidence_level=0.85,
        budget=300_000,
        # assert that both channels are present in the constraint kwargs:
        # channel2 lower and upperbound values should be default values.
        spend_constraint_lower=[
            0.1,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            0.2,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_optional_time_dates(self):
    spec = proc.BudgetOptimizationSpec(
        optimization_name="Budget optimization",
        grid_name="Outcome grid",
        scenario=optimizer.FixedBudgetScenario(total_budget=300_000),
        constraints=[],
        confidence_level=0.85,
        kpi_type=common.KpiType.REVENUE,
    )

    opt = self.processor.execute([spec])

    expected_output_spec = dataclasses.replace(
        spec,
        start_date=self.all_dates[0],
        end_date=(self.all_dates[-1] + datetime.timedelta(days=7)),
    )
    compare.assertProto2Equal(
        self,
        opt.results[0].spec,
        expected_output_spec.to_proto(),
        ignored_fields=["channel_constraints"],
    )
    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=None,
        end_date=None,
        fixed_budget=True,
        confidence_level=0.85,
        budget=300_000,
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=False,
        optimization_grid=None,
        new_data=None,
    )

  def test_execute_build_constraints_kwargs_with_grid(self):
    optimization_grid = optimizer.OptimizationGrid(
        _grid_dataset=mock.MagicMock(),
        historical_spend=np.array([0, 0, 0]),
        use_kpi=False,
        use_posterior=True,
        use_optimal_frequency=False,
        start_date=None,
        end_date=None,
        gtol=0.1,
        round_factor=1,
        optimal_frequency=None,
        selected_times=mock.MagicMock(),
    )
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        grid=optimization_grid,
    )

    self.processor.execute([spec])

    # `_build_kwargs()` is fed into the internal optimizer's `optimize()` call.
    self.mock_optimizer.optimize.assert_called_once_with(
        start_date=_START_DATE_STR,
        end_date=_ADJUSTED_END_DATE_STR,
        fixed_budget=True,
        confidence_level=0.9,
        budget=300_000,
        # Expect that default channel constraints were synthesized.
        spend_constraint_lower=[
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
        ],
        spend_constraint_upper=[
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
            _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
        ],
        use_kpi=False,
        optimization_grid=optimization_grid,
        new_data=None,
    )

  def test_execute_multiple_specs_mmm_output_sets_optimization_results(self):
    spec1 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        # default empty constraints
        group_id="group1",
    )
    # A second spec with different group name and slightly offset dates.
    spec2 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        start_date=datetime.date(2024, 1, 8),
        end_date=datetime.date(2024, 12, 23),
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.MROI,
            target_value=1.0,
        ),
        # default empty constraints
        confidence_level=0.95,
        group_id="group2",
    )

    mmm_output = pb.Mmm()
    self.processor(  # this invokes both execute() and _set_output() together
        [spec1, spec2],
        mmm_output,
    )
    self.assertTrue(mmm_output.HasField("marketing_optimization"))
    self.assertLen(
        mmm_output.marketing_optimization.budget_optimization.results,
        2,
    )

    opt = mmm_output.marketing_optimization.budget_optimization

    expected_spec1_output_proto = _BASE_BUDGET_OPT_RESULT_SPEC_PROTO

    expected_spec2_output_proto = _create_optimization_result_spec(
        start_date_proto=date_pb2.Date(year=2024, month=1, day=8),
        end_date_proto=date_pb2.Date(year=2024, month=12, day=23),
    )
    expected_spec2_output_proto.fixed_budget_scenario.Clear()
    expected_spec2_output_proto.flexible_budget_scenario.CopyFrom(
        budget_pb.FlexibleBudgetScenario(
            target_metric_constraints=[
                constraints_pb.TargetMetricConstraint(
                    target_metric=target_pb.TargetMetric.MARGINAL_ROI,
                    target_value=1.0,
                )
            ],
        )
    )

    compare.assertProto2Equal(
        self, opt.results[0].spec, expected_spec1_output_proto
    )
    compare.assertProto2Equal(
        self, opt.results[1].spec, expected_spec2_output_proto
    )

    marketing_analysis_1 = opt.results[0].optimized_marketing_analysis
    compare.assertProto2Equal(
        self,
        marketing_analysis_1.date_interval,
        date_interval_pb2.DateInterval(
            start_date=_START_DATE_PROTO,
            end_date=_END_DATE_PROTO,
        ),
    )

    marketing_analysis_2 = opt.results[1].optimized_marketing_analysis
    compare.assertProto2Equal(
        self,
        marketing_analysis_2.date_interval,
        date_interval_pb2.DateInterval(
            start_date=date_pb2.Date(
                year=2024,
                month=1,
                day=8,
            ),
            end_date=date_pb2.Date(
                year=2024,
                month=12,
                day=23,
            ),
        ),
    )

    self.mock_optimizer.optimize.assert_has_calls(
        [
            mock.call(
                start_date=_START_DATE_STR,
                end_date=_ADJUSTED_END_DATE_STR,
                fixed_budget=True,
                budget=300_000,
                confidence_level=0.9,
                spend_constraint_lower=[
                    _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
                    _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
                ],
                spend_constraint_upper=[
                    _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
                    _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
                ],
                use_kpi=False,
                optimization_grid=None,
                new_data=None,
            ),
            mock.call(
                # End date in the internal optimizer is inclusive and so 7 days
                # earlier.
                start_date="2024-01-08",
                end_date="2024-12-16",
                fixed_budget=False,
                target_mroi=1.0,
                confidence_level=0.95,
                spend_constraint_lower=[
                    _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
                    _CHANNEL_CONSTRAINT_LOWERBOUND_DEFAULT_RATIO,
                ],
                spend_constraint_upper=[
                    _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
                    _CHANNEL_CONSTRAINT_UPPERBOUND_DEFAULT_RATIO,
                ],
                use_kpi=False,
                optimization_grid=None,
                new_data=None,
            ),
        ],
        any_order=True,
    )

  def test_execute_multiple_specs_duplicate_group_id_raises_error(
      self,
  ):
    spec1 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        # default empty constraints
        group_id="dupe-group",
    )
    spec2 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        start_date=_START_DATE + datetime.timedelta(days=60),
        end_date=_END_DATE + datetime.timedelta(days=60),
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.MROI,
            target_value=1.0,
        ),
        # default empty constraints
        confidence_level=0.95,
        group_id="dupe-group",
    )

    with self.assertRaisesRegex(
        ValueError,
        "Specified group_id must be unique",
    ):
      self.processor.execute([spec1, spec2])

  def test_execute_multiple_specs_no_group_id(self):
    spec1 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        # default empty constraints
        group_id=None,
    )
    spec2 = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        start_date=_START_DATE + datetime.timedelta(days=60),
        end_date=_END_DATE + datetime.timedelta(days=60),
        scenario=optimizer.FlexibleBudgetScenario(
            target_metric=c.ROI,
            target_value=1.0,
        ),
        # default empty constraints
        confidence_level=0.95,
        group_id=None,
    )

    mmm_output = pb.Mmm()
    self.processor(  # this invokes both execute() and _set_output() together
        [spec1, spec2],
        mmm_output,
    )
    self.assertTrue(mmm_output.HasField("marketing_optimization"))
    self.assertLen(
        mmm_output.marketing_optimization.budget_optimization.results,
        2,
    )

  def test_execute_media_analyses_date_interval_tag(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        date_interval_tag=_TAG,
    )

    self.trained_model.mmm.input_data.revenue_per_kpi = True
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])
    self.assertEqual(
        opt.results[0].optimized_marketing_analysis.date_interval.tag, _TAG
    )

  def test_execute_media_analyses_channel_names(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    media_analyses = opt.results[0].optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)
    self.assertEqual(media_analyses[0].channel_name, "channel1")
    self.assertEqual(media_analyses[1].channel_name, "channel2")

  def test_media_analyses_channel_names_nonoptimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    media_analyses = opt.results[
        0
    ].nonoptimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)
    self.assertEqual(media_analyses[0].channel_name, "channel1")
    self.assertEqual(media_analyses[1].channel_name, "channel2")

  def test_execute_media_analyses_channel_spend_info_optimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    # There are 2 channels in the sample data
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    media_analyses = opt.results[0].optimized_marketing_analysis.media_analyses
    spend_infos = [
        media_analysis.spend_info for media_analysis in media_analyses
    ]
    self.assertLen(spend_infos, 2)
    compare.assertProto2Equal(
        self,
        spend_infos[0],
        media_analysis_pb.SpendInfo(
            spend=200_000, spend_share=200_000 / 300_000
        ),
    )
    compare.assertProto2Equal(
        self,
        spend_infos[1],
        media_analysis_pb.SpendInfo(
            spend=100_000, spend_share=100_000 / 300_000
        ),
    )

  def test_execute_media_analyses_channel_spend_info_nonoptimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    # There are 2 channels in the sample data
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    media_analyses = opt.results[
        0
    ].nonoptimized_marketing_analysis.media_analyses
    spend_infos = [
        media_analysis.spend_info for media_analysis in media_analyses
    ]
    self.assertLen(spend_infos, 2)
    compare.assertProto2Equal(
        self,
        spend_infos[0],
        media_analysis_pb.SpendInfo(
            spend=100_000, spend_share=100_000 / 300_000
        ),
    )
    compare.assertProto2Equal(
        self,
        spend_infos[1],
        media_analysis_pb.SpendInfo(
            spend=200_000, spend_share=200_000 / 300_000
        ),
    )

  def test_execute_media_analyses_media_outcomes_revenue_optimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        confidence_level=0.85,
    )
    optdata = _SAMPLE_OPTIMIZED_DATA_REVENUE.copy()
    optdata.attrs[c.CONFIDENCE_LEVEL] = 0.85
    # There are 2 channels in the sample data
    self.mock_optimization_results.optimized_data = optdata

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    spec_result = opt.results[0]
    media_analyses = spec_result.optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    for media_analysis in media_analyses:
      self.assertLen(media_analysis.media_outcomes, 1)
      outcome = media_analysis.media_outcomes[0]
      self.assertEqual(outcome.kpi_type, kpi_type_pb.KpiType.REVENUE)
      for field in ("cost_per_contribution", "roi", "marginal_roi"):
        self.assertTrue(outcome.HasField(field))
        # Each estimate value has 1 uncertainty
        self.assertLen(getattr(outcome, field).uncertainties, 1)
        uncertainty = getattr(outcome, field).uncertainties[0]
        # All uncertainties have the same probability
        self.assertEqual(uncertainty.probability, 0.85)
      # Each media analysis should have a contribution field
      self.assertTrue(outcome.HasField("contribution"))

  def test_execute_media_analyses_media_outcomes_revenue_nonoptimized_data(
      self,
  ):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    # There are 2 channels in the sample data
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    spec_result = opt.results[0]
    media_analyses = spec_result.nonoptimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    for media_analysis in media_analyses:
      self.assertLen(media_analysis.media_outcomes, 1)
      outcome = media_analysis.media_outcomes[0]
      self.assertEqual(outcome.kpi_type, kpi_type_pb.REVENUE)
      for field in ("cost_per_contribution", "roi", "marginal_roi"):
        self.assertTrue(outcome.HasField(field))
        # Each estimate value has 1 uncertainty
        self.assertLen(getattr(outcome, field).uncertainties, 1)
        uncertainty = getattr(outcome, field).uncertainties[0]
        # All uncertainties have the same probability
        self.assertEqual(uncertainty.probability, 0.9)
      # Each media analysis should have a contribution field
      self.assertTrue(outcome.HasField("contribution"))

  def test_media_analyses_media_outcomes_kpi_contribution_optimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    # There are 2 channels in the sample data
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    media_analyses = spec_result.optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    contribution_proto = {}
    expected_incr_outcomes = {}
    expected_pct_contribs = {}
    for idx, channel in enumerate(["channel1", "channel2"]):
      media_analysis = media_analyses[idx]
      contribution_proto[channel] = media_analysis.media_outcomes[
          0
      ].contribution
      expected_incr_outcome: xr.DataArray = (
          _SAMPLE_OPTIMIZED_DATA_REVENUE.incremental_outcome.sel(
              channel=channel
          )
      )
      expected_incr_outcomes[channel] = {
          metric: expected_incr_outcome.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
      expected_pct_contrib: xr.DataArray = (
          _SAMPLE_OPTIMIZED_DATA_REVENUE.pct_of_contribution.sel(
              channel=channel
          )
      )
      expected_pct_contribs[channel] = {
          metric: expected_pct_contrib.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
    for channel in ("channel1", "channel2"):
      compare.assertProto2Equal(
          self,
          contribution_proto[channel],
          kpi_outcome_pb.Contribution(
              value=estimate_pb.Estimate(
                  value=expected_incr_outcomes[channel][c.MEAN],
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          lowerbound=expected_incr_outcomes[channel][c.CI_LO],
                          upperbound=expected_incr_outcomes[channel][c.CI_HI],
                          probability=0.9,
                      )
                  ],
              ),
          ),
      )

  def test_media_analyses_media_outcomes_kpi_contribution_nonoptimized_data(
      self,
  ):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    # There are 2 channels in the sample data
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    media_analyses = spec_result.nonoptimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    contribution_proto = {}
    expected_incr_outcomes = {}
    expected_pct_contribs = {}
    for idx, channel in enumerate(["channel1", "channel2"]):
      media_analysis = media_analyses[idx]
      contribution_proto[channel] = media_analysis.media_outcomes[
          0
      ].contribution
      expected_incr_outcome: xr.DataArray = (
          _SAMPLE_NONOPTIMIZED_DATA_REVENUE.incremental_outcome.sel(
              channel=channel
          )
      )
      expected_incr_outcomes[channel] = {
          metric: expected_incr_outcome.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
      expected_pct_contrib: xr.DataArray = (
          _SAMPLE_NONOPTIMIZED_DATA_REVENUE.pct_of_contribution.sel(
              channel=channel
          )
      )
      expected_pct_contribs[channel] = {
          metric: expected_pct_contrib.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
    for channel in ("channel1", "channel2"):
      compare.assertProto2Equal(
          self,
          contribution_proto[channel],
          kpi_outcome_pb.Contribution(
              value=estimate_pb.Estimate(
                  value=expected_incr_outcomes[channel][c.MEAN],
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          lowerbound=expected_incr_outcomes[channel][c.CI_LO],
                          upperbound=expected_incr_outcomes[channel][c.CI_HI],
                          probability=0.9,
                      )
                  ],
              ),
          ),
      )

  def test_execute_media_analyses_media_outcomes_effectiveness_optimized_data(
      self,
  ):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        confidence_level=0.85,
    )
    opt_data = _SAMPLE_OPTIMIZED_DATA_REVENUE.copy()
    opt_data.attrs[c.CONFIDENCE_LEVEL] = 0.85
    # There are 2 channels in the sample data
    self.mock_optimization_results.optimized_data = opt_data

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    media_analyses = spec_result.optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    effectiveness_proto = {}
    expected_effectiveness = {}
    for idx, channel in enumerate(["channel1", "channel2"]):
      media_analysis = media_analyses[idx]
      effectiveness_proto[channel] = media_analysis.media_outcomes[
          0
      ].effectiveness
      expected: xr.DataArray = opt_data.effectiveness.sel(channel=channel)
      expected_effectiveness[channel] = {
          metric: expected.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
    for channel in ("channel1", "channel2"):
      compare.assertProto2Equal(
          self,
          effectiveness_proto[channel],
          kpi_outcome_pb.Effectiveness(
              media_unit=c.IMPRESSIONS,
              value=estimate_pb.Estimate(
                  value=expected_effectiveness[channel][c.MEAN],
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          lowerbound=expected_effectiveness[channel][c.CI_LO],
                          upperbound=expected_effectiveness[channel][c.CI_HI],
                          probability=0.85,
                      )
                  ],
              ),
          ),
      )

  def test_execute_media_analyses_media_outcomes_effectiveness_nonoptimized_data(
      self,
  ):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        confidence_level=0.85,
    )
    nonopt_data = _SAMPLE_NONOPTIMIZED_DATA_REVENUE
    # There are 2 channels in the sample data
    self.mock_optimization_results.nonoptimized_data = nonopt_data

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    media_analyses = spec_result.nonoptimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    effectiveness_proto = {}
    expected_effectiveness = {}
    for idx, channel in enumerate(["channel1", "channel2"]):
      media_analysis = media_analyses[idx]
      effectiveness_proto[channel] = media_analysis.media_outcomes[
          0
      ].effectiveness
      expected: xr.DataArray = nonopt_data.effectiveness.sel(channel=channel)
      expected_effectiveness[channel] = {
          metric: expected.sel(metric=metric).item()
          for metric in (c.MEAN, c.CI_LO, c.CI_HI)
      }
    for channel in ("channel1", "channel2"):
      compare.assertProto2Equal(
          self,
          effectiveness_proto[channel],
          kpi_outcome_pb.Effectiveness(
              media_unit=c.IMPRESSIONS,
              value=estimate_pb.Estimate(
                  value=expected_effectiveness[channel][c.MEAN],
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          lowerbound=expected_effectiveness[channel][c.CI_LO],
                          upperbound=expected_effectiveness[channel][c.CI_HI],
                          probability=0.9,
                      )
                  ],
              ),
          ),
      )

  def test_media_analyses_media_outcomes_nonrevenue_optimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        confidence_level=0.85,
    )
    optdata = _SAMPLE_OPTIMIZED_DATA_NONREVENUE.copy()
    optdata.attrs[c.CONFIDENCE_LEVEL] = 0.85
    # There are 2 channels in the sample data
    self.mock_optimization_results.optimized_data = optdata

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    spec_result = opt.results[0]
    media_analyses = spec_result.optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    # Each media channel analysis should have one nonrevenue outcome
    for media_analysis in media_analyses:
      self.assertLen(media_analysis.media_outcomes, 1)
      outcome = media_analysis.media_outcomes[0]
      self.assertEqual(outcome.kpi_type, kpi_type_pb.KpiType.NON_REVENUE)
      for field in ("cost_per_contribution", "roi", "marginal_roi"):
        self.assertTrue(outcome.HasField(field))
        # Each estimate value has 1 uncertainty
        self.assertLen(getattr(outcome, field).uncertainties, 1)
        uncertainty = getattr(outcome, field).uncertainties[0]
        # All uncertainties have the same probability
        self.assertEqual(uncertainty.probability, 0.85)

  def test_media_analyses_media_outcomes_nonrevenue_nonoptimized_data(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        confidence_level=0.85,
    )
    # There are 3 channels in the sample data
    self.mock_optimization_results.nonoptimized_data = (
        _SAMPLE_NONOPTIMIZED_DATA_NONREVENUE
    )

    opt = self.processor.execute([spec])

    self.assertLen(opt.results, 1)
    spec_result = opt.results[0]
    media_analyses = spec_result.nonoptimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses

    # Each media channel analysis should have one nonrevenue outcome
    for media_analysis in media_analyses:
      self.assertLen(media_analysis.media_outcomes, 1)
      outcome = media_analysis.media_outcomes[0]
      self.assertEqual(outcome.kpi_type, kpi_type_pb.NON_REVENUE)
      for field in ("cost_per_contribution", "roi", "marginal_roi"):
        self.assertTrue(outcome.HasField(field))
        # Each estimate value has 1 uncertainty
        self.assertLen(getattr(outcome, field).uncertainties, 1)
        uncertainty = getattr(outcome, field).uncertainties[0]
        # All uncertainties have the same probability
        self.assertEqual(uncertainty.probability, 0.9)

  def test_media_analyses_response_curves(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
    )
    self.mock_optimization_results.optimized_data = (
        _SAMPLE_OPTIMIZED_DATA_REVENUE
    )

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    media_analyses = spec_result.optimized_marketing_analysis.media_analyses
    self.assertLen(media_analyses, 2)  # 2 channels = 2 media analyses
    # Each media analysis should have one response curve
    for media_analysis in media_analyses:
      # Since the response curves data have 4 spend multiplier dimensions per
      # channel, expect 4 response point per channel media analysis.
      self.assertLen(media_analysis.response_curve.response_points, 4)
    # For channel1
    expected_points = [(0.0, 0.1), (5.0, 0.6), (10.0, 1.1), (15.0, 1.6)]
    compare.assertProto2Equal(
        self,
        media_analyses[0].response_curve,
        response_curve_pb.ResponseCurve(
            input_name=c.SPEND,
            response_points=[
                response_curve_pb.ResponsePoint(
                    input_value=input,
                    incremental_kpi=kpi,
                )
                for input, kpi in expected_points
            ],
        ),
    )
    # For channel2
    expected_points = [(0.0, 0.1), (10.0, 0.6), (20.0, 1.1), (30.0, 1.6)]
    compare.assertProto2Equal(
        self,
        media_analyses[1].response_curve,
        response_curve_pb.ResponseCurve(
            input_name=c.SPEND,
            response_points=[
                response_curve_pb.ResponsePoint(
                    input_value=input,
                    incremental_kpi=kpi,
                )
                for input, kpi in expected_points
            ],
        ),
    )

  def test_execute_budget_opt_result_incremental_outcome_grid(self):
    spec = dataclasses.replace(
        _BASE_BUDGET_OPT_INPUT_SPEC,
        optimization_name="opt_name",
        grid_name="test_grid",
    )

    opt = self.processor.execute([spec])

    spec_result = opt.results[0]
    self.assertEqual(spec_result.name, "opt_name")
    incremental_outcome_grid = spec_result.incremental_outcome_grid
    self.assertEqual(incremental_outcome_grid.name, "test_grid")
    self.assertEqual(incremental_outcome_grid.spend_step_size, 200)
    self.assertLen(incremental_outcome_grid.channel_cells, 2)  # 2 channels

    expected_spend_outcomes = [
        (500, 1.0),
        (700, 1.0),
        (900, 1.0),
        (1100, 1.0),
        (1300, 1.0),
        (1500, 1.0),
    ]
    compare.assertProto2Equal(
        self,
        incremental_outcome_grid.channel_cells[0],
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name="channel1",
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=spend,
                    incremental_outcome=estimate_pb.Estimate(value=outcome),
                )
                for spend, outcome in expected_spend_outcomes
            ],
        ),
    )
    expected_spend_outcomes = [
        (900, 0.81818182),
        (1100, 1.0),
    ]
    compare.assertProto2Equal(
        self,
        incremental_outcome_grid.channel_cells[1],
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name="channel2",
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=spend,
                    incremental_outcome=estimate_pb.Estimate(value=outcome),
                )
                for spend, outcome in expected_spend_outcomes
            ],
        ),
    )


if __name__ == "__main__":
  absltest.main()
