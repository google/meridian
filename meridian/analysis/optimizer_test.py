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

"""Optimization tests using mocked data.

The unit tests generally follow this procedure:
1) Load InferenceData MCMC results from disk.
2) Run an optimization scenario based on the InferenceData.
3) Compare optimization results against the values in the constants below. These
  values are obtained from a previous optimization run that is assumed to be
  correct.
"""

from collections.abc import Mapping
import dataclasses
import math
import os
import tempfile
from typing import Any
from xml.etree import ElementTree as ET

from absl.testing import absltest
from absl.testing import parameterized
import altair as alt
import arviz as az
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import optimizer
from meridian.analysis import summary_text
from meridian.analysis import test_utils as analysis_test_utils
from meridian.data import input_data
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


mock = absltest.mock


# Path to the stored inference data generated with
# .../google_internal/generate_test_data.ipynb
_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'model', 'test_data'
)
# Constants below are used for mocking, the values are based on the test data.

# For mocking computed aggregated impressions (per channel).
_AGGREGATED_IMPRESSIONS = np.array([1000, 2000, 3000, 2500, 1500])

# Expected incremental outcome under the actual historical budget scenario,
# where the model contains both R&F and non-R&F channels. The name
# "nonoptimized" refers to an attribute name of the Optimizer class.
_NONOPTIMIZED_INCREMENTAL_OUTCOME = np.array(
    [335.65, 531.11, 791.67, 580.1, 242.4]
)
_NONOPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI = np.array([
    [335.65, 335.65, 335.65, 335.65],
    [531.11, 531.11, 531.11, 531.11],
    [791.67, 791.67, 791.67, 791.67],
    [580.1, 580.1, 580.1, 580.1],
    [242.4, 242.4, 242.4, 242.4],
])
_NONOPTIMIZED_EFFECTIVENESS_WITH_CI = np.array([
    [0.33565, 0.33565, 0.33565, 0.33565],
    [0.26555499, 0.26555499, 0.26555499, 0.26555499],
    [0.26389, 0.26389, 0.26389, 0.26389],
    [0.23203999, 0.23203999, 0.23203999, 0.23203999],
    [0.16159999, 0.16159999, 0.16159999, 0.16159999],
])
# Actual historical spend. The name "nonoptimized" refers to an attribute name
# of the Optimizer class.
_NONOPTIMIZED_SPEND = np.array([294.0, 279.0, 256.0, 272.0, 288.0])
# Correct incremental outcome for a fixed budget scenario, where the model
# contains both R&F and non-R&F channels.
_OPTIMIZED_INCREMENTAL_OUTCOME = np.array([528.7, 648.4, 427.4, 1178.6, 889.0])
_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI = np.array([
    [528.7, 528.7, 528.7, 528.7],
    [648.4, 648.4, 648.4, 648.4],
    [427.4, 427.4, 427.4, 427.4],
    [1178.6, 1178.6, 1178.6, 1178.6],
    [889.0, 889.0, 889.0, 889.0],
])
_OPTIMIZED_EFFECTIVENESS_WITH_CI = np.array([
    [0.52869999, 0.52869999, 0.52869999, 0.52869999],
    [0.3242, 0.3242, 0.3242, 0.3242],
    [0.14246666, 0.14246666, 0.14246666, 0.14246666],
    [0.47143999, 0.47143999, 0.47143999, 0.47143999],
    [0.59266669, 0.59266669, 0.59266669, 0.59266669],
])
# Correct optimal spend allocation for fixed budget scenario, where the model
# contains both R&F and non-R&F channels.
_OPTIMIZED_SPEND = np.array([233.0, 222.0, 206.0, 354.0, 374.0])
# Correct optimal spend allocation for fixed budget scenario, where the model
# contains only non-R&F channels.
_OPTIMIZED_MEDIA_ONLY_SPEND = np.array([289.0, 278.0, 262.0])
# Correct optimal spend allocation for fixed budget scenario, where the model
# contains only R&F channels.
_OPTIMIZED_RF_ONLY_SPEND = np.array([354.0, 206.0])
# Correct optimal spend allocation for a flexible budget optimization scenario
# using a target mROI value.
_TARGET_MROI_SPEND = np.array([0.0, 0.0, 0.0, 544.0, 576.0])
# Correct optimal spend allocation for a flexible budget optimization scenario
# using a target ROI value.
_TARGET_ROI_SPEND = np.array([588.0, 558.0, 512.0, 544.0, 576.0])
# Currently zero due to numerator being zero with mocked constant inc_outcome
_BUDGET_MROI_WITH_CI = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
)

_N_GEOS = 5
_N_TIMES = 49
_N_MEDIA_TIMES = 52
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2
_N_ORGANIC_MEDIA_CHANNELS = 4
_N_ORGANIC_RF_CHANNELS = 1
_N_NON_MEDIA_CHANNELS = 4
_N_CONTROLS = 2
_N_CHAINS = 1
_N_DRAWS = 1


def _create_budget_data(
    spend: np.ndarray,
    inc_outcome: np.ndarray,
    effectiveness: np.ndarray,
    mroi: np.ndarray | None = None,
    explicit_cpik: np.ndarray | None = None,
    channels: np.ndarray | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
  channels = (
      [f'channel {i}' for i in range(len(spend))]
      if channels is None
      else channels
  )
  data_vars = {
      c.SPEND: ([c.CHANNEL], spend),
      c.PCT_OF_SPEND: ([c.CHANNEL], spend / sum(spend)),
      c.INCREMENTAL_OUTCOME: ([c.CHANNEL, c.METRIC], inc_outcome),
      c.EFFECTIVENESS: ([c.CHANNEL, c.METRIC], effectiveness),
      c.ROI: (
          [c.CHANNEL, c.METRIC],
          tf.transpose(tf.math.divide_no_nan(tf.transpose(inc_outcome), spend)),
      ),
      c.MROI: ([c.CHANNEL, c.METRIC], mroi),
  }

  if explicit_cpik is not None:
    data_vars[c.CPIK] = ([c.CHANNEL, c.METRIC], explicit_cpik)
  else:
    data_vars[c.CPIK] = (
        [c.CHANNEL, c.METRIC],
        tf.transpose(tf.math.divide_no_nan(spend, tf.transpose(inc_outcome))),
    )

  attributes = {
      c.START_DATE: '2020-01-05',
      c.END_DATE: '2020-06-28',
      c.BUDGET: sum(spend),
      c.PROFIT: sum(inc_outcome[:, 0]) - sum(spend),
      c.TOTAL_INCREMENTAL_OUTCOME: sum(inc_outcome[:, 0]),
      c.TOTAL_CPIK: sum(spend) / sum(inc_outcome[:, 0]),
      c.TOTAL_ROI: sum(inc_outcome[:, 0]) / sum(spend),
      c.CONFIDENCE_LEVEL: c.DEFAULT_CONFIDENCE_LEVEL,
      c.USE_HISTORICAL_BUDGET: True,
  }

  return xr.Dataset(
      data_vars=data_vars,
      coords={
          c.CHANNEL: ([c.CHANNEL], channels),
          c.METRIC: ([c.METRIC], [c.MEAN, c.MEDIAN, c.CI_LO, c.CI_HI]),
      },
      attrs=attributes | (attrs or {}),
  )


def _verify_actual_vs_expected_budget_data(
    actual_data: xr.Dataset, expected_data: xr.Dataset
):
  xr.testing.assert_allclose(actual_data, expected_data, atol=0.1, rtol=0.01)
  np.testing.assert_allclose(actual_data.budget, expected_data.budget, atol=0.1)
  np.testing.assert_allclose(actual_data.profit, expected_data.profit, atol=0.1)
  np.testing.assert_allclose(
      actual_data.total_incremental_outcome,
      expected_data.total_incremental_outcome,
      atol=0.1,
  )
  np.testing.assert_allclose(
      actual_data.total_roi, expected_data.total_roi, atol=0.1
  )
  if c.FIXED_BUDGET in expected_data.attrs:
    np.testing.assert_equal(
        actual_data.fixed_budget, expected_data.fixed_budget
    )
  if c.TARGET_ROI in expected_data.attrs:
    np.testing.assert_equal(actual_data.target_roi, expected_data.target_roi)
  if c.TARGET_MROI in expected_data.attrs:
    np.testing.assert_equal(actual_data.target_mroi, expected_data.target_mroi)


_SAMPLE_NON_OPTIMIZED_DATA = _create_budget_data(
    spend=np.array([200, 100, 300]),
    inc_outcome=np.array(
        [[280, 280, 280, 280], [150, 150, 150, 150], [330, 330, 330, 330]]
    ),
    effectiveness=np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ]),
    mroi=np.array(
        [[1.2, 1.2, 1.2, 1.2], [1.3, 1.3, 1.3, 1.3], [1.4, 1.4, 1.4, 1.4]]
    ),
    attrs={c.IS_REVENUE_KPI: True},
)
_SAMPLE_OPTIMIZED_DATA = _create_budget_data(
    spend=np.array([220, 140, 240]),
    inc_outcome=np.array(
        [[350, 350, 349, 351], [210, 210, 209, 211], [270, 270, 269, 271]]
    ),
    effectiveness=np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ]),
    mroi=np.array(
        [[1.4, 1.4, 1.4, 1.4], [1.5, 1.5, 1.5, 1.5], [1.6, 1.6, 1.6, 1.6]]
    ),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: True,
    },
)
_SAMPLE_NON_OPTIMIZED_DATA_KPI = _create_budget_data(
    spend=np.array([200, 100, 300]),
    inc_outcome=np.array(
        [[280, 280, 279, 281], [150, 150, 149, 151], [330, 330, 329, 331]]
    ),
    effectiveness=np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ]),
    mroi=np.array(
        [[2.4, 2.4, 2.4, 2.4], [2.5, 2.5, 2.5, 2.5], [2.6, 2.6, 2.6, 2.6]]
    ),
    attrs={c.IS_REVENUE_KPI: False},
)
_SAMPLE_OPTIMIZED_DATA_KPI = _create_budget_data(
    spend=np.array([220, 140, 240]),
    inc_outcome=np.array(
        [[350, 350, 349, 351], [210, 210, 209, 211], [270, 270, 269, 271]]
    ),
    effectiveness=np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ]),
    mroi=np.array(
        [[3.4, 3.4, 3.4, 3.4], [3.5, 3.5, 3.5, 3.5], [3.6, 3.6, 3.6, 3.6]]
    ),
    attrs={
        c.FIXED_BUDGET: True,
        c.IS_REVENUE_KPI: False,
    },
)


class OptimizerAlgorithmTest(parameterized.TestCase):
  # TODO: Update the sample datasets to span over 1 year.
  def setUp(self):
    super(OptimizerAlgorithmTest, self).setUp()

    self.input_data_media_and_rf = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_controls=_N_CONTROLS,
            seed=0,
        )
    )
    self.input_data_media_only = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_controls=_N_CONTROLS,
            seed=0,
        )
    )
    self.input_data_rf_only = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_rf_channels=_N_RF_CHANNELS,
            n_controls=_N_CONTROLS,
            seed=0,
        )
    )
    self.input_data_all_channels = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            n_controls=_N_CONTROLS,
            seed=0,
        )
    )

    self.inference_data_media_and_rf = az.InferenceData(
        prior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_prior_media_and_rf.nc')
        ),
        posterior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_posterior_media_and_rf.nc')
        ),
    )
    self.inference_data_media_only = az.InferenceData(
        prior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_prior_media_only.nc')
        ),
        posterior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_posterior_media_only.nc')
        ),
    )
    self.inference_data_rf_only = az.InferenceData(
        prior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_prior_rf_only.nc')
        ),
        posterior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_posterior_rf_only.nc')
        ),
    )
    self.inference_data_all_channels = az.InferenceData(
        prior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_prior_non_paid.nc')
        ),
        posterior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_posterior_non_paid.nc')
        ),
    )

    self.meridian_media_and_rf = model.Meridian(
        input_data=self.input_data_media_and_rf
    )
    self.meridian_media_only = model.Meridian(
        input_data=self.input_data_media_only
    )
    self.meridian_rf_only = model.Meridian(input_data=self.input_data_rf_only)
    self.meridian_all_channels = model.Meridian(
        input_data=self.input_data_all_channels
    )

    self.budget_optimizer_media_and_rf = optimizer.BudgetOptimizer(
        self.meridian_media_and_rf
    )
    self.budget_optimizer_media_only = optimizer.BudgetOptimizer(
        self.meridian_media_only
    )
    self.budget_optimizer_rf_only = optimizer.BudgetOptimizer(
        self.meridian_rf_only
    )
    self.budget_optimizer_all_channels = optimizer.BudgetOptimizer(
        self.meridian_all_channels
    )

    self.enter_context(
        mock.patch.object(
            model.Meridian,
            'inference_data',
            new=property(lambda unused_self: self.inference_data_media_and_rf),
        )
    )
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'summary_metrics',
            return_value=analysis_test_utils.generate_paid_summary_metrics(),
        )
    )

  @parameterized.parameters([True, False])
  def test_not_fitted_meridian_model_raises_exception(
      self, use_posterior: bool
  ):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    not_fitted_mmm.inference_data = az.InferenceData()
    budget_optimizer = optimizer.BudgetOptimizer(not_fitted_mmm)
    with self.assertRaisesRegex(
        model.NotFittedModelError,
        'Running budget optimization scenarios requires fitting the model.',
    ):
      budget_optimizer.create_optimization_grid(
          spend=mock.Mock(),
          spend_bound_lower=mock.Mock(),
          spend_bound_upper=mock.Mock(),
          selected_times=None,
          round_factor=1,
          use_posterior=use_posterior,
      )

  def test_fixed_budget_target_roi_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`target_roi` is only used for flexible budget scenarios.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=True, target_roi=1.2
      )

  def test_fixed_budget_target_mroi_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`target_mroi` is only used for flexible budget scenarios.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=True, target_mroi=1.2
      )

  def test_fixed_budget_negative_budget_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`budget` must be greater than zero.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=True, budget=-100
      )

  def test_flexible_budget_with_budget_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`budget` is only used for fixed budget scenarios.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=False, budget=10000
      )

  def test_flexible_budget_without_targets_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'Must specify either `target_roi` or `target_mroi`',
    ):
      self.budget_optimizer_media_and_rf.optimize(fixed_budget=False)

  def test_flexible_budget_with_multiple_targets_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'Must specify only one of `target_roi` or `target_mroi`',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=False,
          target_roi=1.2,
          target_mroi=1.2,
      )

  def test_pct_of_spend_incorrect_length_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'Percent of spend must be specified for all channels.',
    ):
      self.budget_optimizer_media_and_rf.optimize(pct_of_spend=[0.3, 0.7])

  def test_pct_of_spend_incorrect_sum_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'Percent of spend must sum to one.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          pct_of_spend=[0.1, 0.1, 0.1, 0.1, 0.1]
      )

  @parameterized.named_parameters(
      ('incorrect_lower', [0.3, 0.3], 0.3),
      ('incorrect_upper', 0.3, [0.3, 0.3]),
  )
  def test_spend_constraint_incorrect_length_raises_exception(
      self, lower_constraint, upper_constraint
  ):
    with self.assertRaisesRegex(
        ValueError,
        'Spend constraints must be either a single constraint or be specified'
        ' for all channels.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          spend_constraint_lower=lower_constraint,
          spend_constraint_upper=upper_constraint,
      )

  @parameterized.named_parameters(
      ('negative_constraint', -0.3), ('greater_than_1_lower_constraint', 1.2)
  )
  def test_lower_spend_constraint_out_of_bounds_raises_exception(
      self, lower_constraint
  ):
    with self.assertRaisesRegex(
        ValueError,
        'The lower spend constraint must be between 0 and 1 inclusive.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          spend_constraint_lower=lower_constraint
      )

  def test_negative_upper_spend_constraint_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        'The upper spend constraint must be positive.',
    ):
      self.budget_optimizer_media_and_rf.optimize(spend_constraint_upper=-0.3)

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_default_selected_times_all_times(
      self,
      mock_incremental_outcome,
  ):
    mock_incremental_outcome.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))

    optimization_results = self.budget_optimizer_media_and_rf.optimize()

    expected_times = self.input_data_media_and_rf.time.values.tolist()
    self.assertEqual(
        optimization_results.optimized_data.start_date,
        expected_times[0],
    )
    self.assertEqual(
        optimization_results.optimized_data.end_date,
        expected_times[-1],
    )
    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertIsNone(mock_kwargs['selected_times'])

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_selected_times_all_times(self, mock_incremental_outcome):
    mock_incremental_outcome.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))

    expected_times = self.input_data_media_and_rf.time.values.tolist()
    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        selected_times=None
    )

    self.assertEqual(
        optimization_results.optimized_data.start_date,
        expected_times[0],
    )
    self.assertEqual(
        optimization_results.optimized_data.end_date,
        expected_times[-1],
    )
    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertIsNone(mock_kwargs['selected_times'])

  @parameterized.named_parameters(
      {
          'testcase_name': 'selected_times_range',
          'selected_times_arg': ['2021-05-17', '2021-06-14'],
          'expected_times': [
              '2021-05-17',
              '2021-05-24',
              '2021-05-31',
              '2021-06-07',
              '2021-06-14',
          ],
      },
      {
          'testcase_name': 'end_none',
          'selected_times_arg': ['2021-02-08', None],
          'expected_times': (
              pd.date_range(start='2021-2-8', end='2021-12-27', freq='7D')
              .strftime('%Y-%m-%d')
              .tolist()
          ),
      },
      {
          'testcase_name': 'start_none',
          'selected_times_arg': [None, '2021-11-15'],
          'expected_times': (
              pd.date_range(start='2021-01-25', end='2021-11-15', freq='7D')
              .strftime('%Y-%m-%d')
              .tolist()
          ),
      },
      {
          'testcase_name': 'none_tuple',
          'selected_times_arg': [None, None],
          'expected_times': None,
      },
      {
          'testcase_name': 'none',
          'selected_times_arg': None,
          'expected_times': None,
      },
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_selected_times_used_correctly(
      self, mock_incremental_outcome, selected_times_arg, expected_times
  ):
    mock_incremental_outcome.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))

    self.budget_optimizer_media_and_rf.optimize(
        selected_times=selected_times_arg
    )

    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertEqual(mock_kwargs['selected_times'], expected_times)

  def test_default_hist_spend_with_time_geo_dims(self):
    expected_spend = np.round(
        np.sum(self.meridian_media_and_rf.total_spend, axis=(0, 1))
    )

    optimization_results = self.budget_optimizer_media_and_rf.optimize()

    self.assertEqual(self.meridian_media_and_rf.total_spend.ndim, 3)
    np.testing.assert_array_equal(
        optimization_results.nonoptimized_data.spend,
        expected_spend,
    )
    self.assertEqual(
        optimization_results.nonoptimized_data.budget,
        np.sum(expected_spend),
    )

  def test_spend_ratio(self):
    optimization_results = self.budget_optimizer_media_and_rf.optimize()
    np.testing.assert_allclose(
        optimization_results.spend_ratio,
        [1, 1, 1, 1, 1],
        atol=0.01,
    )

  def test_hist_spend_with_imputed_cpm(self):
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'incremental_outcome',
            autospec=True,
            spec_set=True,
            return_value=tf.convert_to_tensor(
                [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
            ),
        )
    )

    # TODO: Remove this mock once the bug is fixed.
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'marginal_roi',
            autospec=True,
            spec_set=True,
            return_value=tf.convert_to_tensor(
                [[[1.0, 1.0, 1.0, 1.0, 1.0]]], tf.float32
            ),
        )
    )

    media_spend_da = data_test_utils.random_media_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=0,
    )
    media_spend_da.values = np.array([120.0, 150.0, 652.0])

    rf_spend_da = data_test_utils.random_rf_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    rf_spend_da.values = np.array([645.0, 654.0])

    self.input_data_media_and_rf.media_spend = media_spend_da
    self.input_data_media_and_rf.rf_spend = rf_spend_da

    # To make the above mock work as intended, we have to recreate the model and
    # optimizer objects, since the `total_spend` property in the shared SUT objs
    # had by now been locked in.
    meridian_media_and_rf = model.Meridian(
        input_data=self.input_data_media_and_rf
    )
    budget_optimizer_media_and_rf = optimizer.BudgetOptimizer(
        meridian_media_and_rf
    )

    optimization_results = budget_optimizer_media_and_rf.optimize(
        selected_times=('2021-01-25', '2021-03-08'),
        # TODO: set optimal frequency back to true once the bug is
        # fixed.
        use_optimal_frequency=False,
    )
    expected_spend = [19.0, 24.0, 104.0, 94.0, 95.0]
    self.assertIsNotNone(meridian_media_and_rf.media_tensors.media_spend)
    assert meridian_media_and_rf.media_tensors.media_spend is not None
    self.assertEqual(meridian_media_and_rf.media_tensors.media_spend.ndim, 1)
    self.assertIsNotNone(meridian_media_and_rf.rf_tensors.rf_spend)
    assert meridian_media_and_rf.rf_tensors.rf_spend is not None
    self.assertEqual(meridian_media_and_rf.rf_tensors.rf_spend.ndim, 1)
    np.testing.assert_array_equal(
        optimization_results.nonoptimized_data.spend,
        expected_spend,
    )
    self.assertEqual(
        optimization_results.nonoptimized_data.budget,
        np.sum(expected_spend),
    )

  def test_spend_ratio_handles_zero_hist_spend(self):
    """Tests that spend_ratio is 0 when hist_spend is 0."""
    budget_optimizer = self.budget_optimizer_media_and_rf
    with mock.patch.object(
        budget_optimizer._analyzer,
        'get_historical_spend',
        return_value=mock.MagicMock(data=np.array([0, 0, 0, 0, 0])),
    ):
      optimization_results = budget_optimizer.optimize(
          budget=1000, pct_of_spend=[0, 0.25, 0.25, 0.25, 0.25]
      )

      np.testing.assert_array_equal(
          optimization_results.spend_ratio,
          np.zeros_like(optimization_results.spend_ratio),
      )

  def test_incremental_outcome_tensors(self):
    spend = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    hist_spend = np.array([350, 400, 200, 50, 500], dtype=np.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self.budget_optimizer_media_and_rf._get_incremental_outcome_tensors(
            hist_spend, spend
        )
    )
    expected_media = (
        self.meridian_media_and_rf.media_tensors.media
        * tf.math.divide_no_nan(new_media_spend, hist_spend[:_N_MEDIA_CHANNELS])
    )
    expected_media_spend = spend[:_N_MEDIA_CHANNELS]
    expected_reach = (
        self.meridian_media_and_rf.rf_tensors.reach
        * tf.math.divide_no_nan(new_rf_spend, hist_spend[-_N_RF_CHANNELS:])
    )
    expected_frequency = self.meridian_media_and_rf.rf_tensors.frequency
    expected_rf_spend = spend[-_N_RF_CHANNELS:]
    np.testing.assert_allclose(new_media, expected_media)
    np.testing.assert_allclose(new_media_spend, expected_media_spend)
    np.testing.assert_allclose(new_reach, expected_reach)
    np.testing.assert_allclose(new_frequency, expected_frequency)
    np.testing.assert_allclose(new_rf_spend, expected_rf_spend)

  def test_incremental_outcome_tensors_with_optimal_frequency(self):
    spend = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    hist_spend = np.array([350, 400, 200, 50, 500], dtype=np.float32)
    optimal_frequency = np.array([2, 2], dtype=np.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self.budget_optimizer_media_and_rf._get_incremental_outcome_tensors(
            hist_spend=hist_spend,
            spend=spend,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_media = (
        self.meridian_media_and_rf.media_tensors.media
        * tf.math.divide_no_nan(new_media_spend, hist_spend[:_N_MEDIA_CHANNELS])
    )
    expected_media_spend = spend[:_N_MEDIA_CHANNELS]
    rf_media = (
        self.meridian_media_and_rf.rf_tensors.reach
        * self.meridian_media_and_rf.rf_tensors.frequency
    )
    expected_reach = tf.math.divide_no_nan(
        rf_media
        * tf.math.divide_no_nan(
            new_rf_spend,
            hist_spend[-_N_RF_CHANNELS:],
        ),
        optimal_frequency,
    )
    expected_frequency = (
        tf.ones_like(self.meridian_media_and_rf.rf_tensors.frequency)
        * optimal_frequency
    )
    expected_rf_spend = spend[-_N_RF_CHANNELS:]
    np.testing.assert_allclose(new_media, expected_media)
    np.testing.assert_allclose(new_media_spend, expected_media_spend)
    np.testing.assert_allclose(new_reach, expected_reach)
    np.testing.assert_allclose(new_frequency, expected_frequency)
    np.testing.assert_allclose(new_rf_spend, expected_rf_spend)

  @mock.patch.object(optimizer.BudgetOptimizer, '_create_grids', autospec=True)
  @mock.patch.object(optimizer, '_get_round_factor', autospec=True)
  def test_optimization_grid(self, mock_get_round_factor, mock_create_grids):
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0, 800.0, 900.0],
            [700.0, 800.0, 900.0, 1000.0, 1100.0],
            [900.0, 1000.0, 1100.0, 1200.0, np.nan],
            [1100.0, 1200.0, 1300.0, np.nan, np.nan],
            [1300.0, 1400.0, np.nan, np.nan, np.nan],
            [1500.0, np.nan, np.nan, np.nan, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [1.0, 1.0, 1.0, 0.66666667, 0.81818182],
            [1.0, 1.0, 1.0, 0.83333333, 1.0],
            [1.0, 1.0, 1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
        ],
    )
    mock_create_grids.return_value = [
        expected_spend_grid,
        expected_incremental_outcome_grid,
    ]
    # This should correspond to a computed (spend) step size of 100.
    mock_get_round_factor.return_value = -int(math.log10(20)) - 1

    expected_data = xr.Dataset(
        data_vars={
            c.SPEND_GRID: (
                [c.GRID_SPEND_INDEX, c.CHANNEL],
                expected_spend_grid,
            ),
            c.INCREMENTAL_OUTCOME_GRID: (
                [c.GRID_SPEND_INDEX, c.CHANNEL],
                expected_incremental_outcome_grid,
            ),
        },
        coords={
            c.GRID_SPEND_INDEX: (
                [c.GRID_SPEND_INDEX],
                np.arange(0, len(expected_spend_grid)),
            ),
            c.CHANNEL: (
                [c.CHANNEL],
                self.input_data_media_and_rf.get_all_channels(),
            ),
        },
    )

    optimization_results = self.budget_optimizer_media_and_rf.optimize()

    actual_data = optimization_results.optimization_grid
    self.assertEqual(actual_data, expected_data)
    self.assertEqual(actual_data.spend_step_size, 100.0)

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_nonoptimized_data_with_defaults_media_and_rf(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS]], tf.float32
    )

    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND,
        inc_outcome=_NONOPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI,
        effectiveness=_NONOPTIMIZED_EFFECTIVENESS_WITH_CI,
        mroi=_BUDGET_MROI_WITH_CI,
        channels=self.input_data_media_and_rf.get_all_channels(),
    )
    optimization_results = self.budget_optimizer_media_and_rf.optimize()

    actual_data = optimization_results.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_nonoptimized_data_with_defaults_media_only(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME[:_N_MEDIA_CHANNELS]]],
        tf.float32,
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS[:_N_MEDIA_CHANNELS]]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND[:_N_MEDIA_CHANNELS],
        inc_outcome=_NONOPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI[
            :_N_MEDIA_CHANNELS
        ],
        effectiveness=_NONOPTIMIZED_EFFECTIVENESS_WITH_CI[:_N_MEDIA_CHANNELS],
        mroi=_BUDGET_MROI_WITH_CI[:_N_MEDIA_CHANNELS],
        channels=self.input_data_media_only.get_all_channels(),
    )
    optimization_results = self.budget_optimizer_media_only.optimize()

    actual_data = optimization_results.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_nonoptimized_data_with_defaults_rf_only(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME[-_N_RF_CHANNELS:]]],
        tf.float32,
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS[-_N_RF_CHANNELS:]]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND[-_N_RF_CHANNELS:],
        inc_outcome=_NONOPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI[-_N_RF_CHANNELS:],
        effectiveness=_NONOPTIMIZED_EFFECTIVENESS_WITH_CI[-_N_RF_CHANNELS:],
        mroi=_BUDGET_MROI_WITH_CI[-_N_RF_CHANNELS:],
        channels=self.input_data_rf_only.get_all_channels(),
    )
    optimization_results = self.budget_optimizer_rf_only.optimize()

    actual_data = optimization_results.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimized_data_with_defaults_media_and_rf(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_SPEND,
        inc_outcome=_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI,
        effectiveness=_OPTIMIZED_EFFECTIVENESS_WITH_CI,
        mroi=_BUDGET_MROI_WITH_CI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )
    optimization_results = self.budget_optimizer_media_and_rf.optimize()

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        optimization_results.nonoptimized_data.budget,
    )

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimized_data_with_defaults_media_only(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME[:_N_MEDIA_CHANNELS]]],
        tf.float32,
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS[:_N_MEDIA_CHANNELS]]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_MEDIA_ONLY_SPEND,
        inc_outcome=_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI[:_N_MEDIA_CHANNELS],
        effectiveness=_OPTIMIZED_EFFECTIVENESS_WITH_CI[:_N_MEDIA_CHANNELS],
        mroi=_BUDGET_MROI_WITH_CI[:_N_MEDIA_CHANNELS],
        channels=self.input_data_media_only.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )

    optimization_results = self.budget_optimizer_media_only.optimize()

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        optimization_results.nonoptimized_data.budget,
    )

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimized_data_with_defaults_rf_only(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME[-_N_RF_CHANNELS:]]],
        tf.float32,
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS[-_N_RF_CHANNELS:]]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_RF_ONLY_SPEND,
        inc_outcome=_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI[-_N_RF_CHANNELS:],
        effectiveness=_OPTIMIZED_EFFECTIVENESS_WITH_CI[-_N_RF_CHANNELS:],
        mroi=_BUDGET_MROI_WITH_CI[-self.meridian_rf_only.n_rf_channels :],
        channels=self.input_data_rf_only.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )

    optimization_results = self.budget_optimizer_rf_only.optimize()

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        optimization_results.nonoptimized_data.budget,
    )

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimized_data_with_target_mroi(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_TARGET_MROI_SPEND,
        inc_outcome=_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI,
        effectiveness=_OPTIMIZED_EFFECTIVENESS_WITH_CI,
        mroi=_BUDGET_MROI_WITH_CI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: False, c.TARGET_MROI: 1},
    )

    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        fixed_budget=False, target_mroi=1
    )

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(
      analyzer.Analyzer, 'get_aggregated_impressions', autospec=True
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimized_data_with_target_roi(
      self,
      mock_incremental_outcome,
      mock_get_aggregated_impressions,
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    mock_get_aggregated_impressions.return_value = tf.convert_to_tensor(
        [[_AGGREGATED_IMPRESSIONS]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_TARGET_ROI_SPEND,
        inc_outcome=_OPTIMIZED_INCREMENTAL_OUTCOME_WITH_CI,
        effectiveness=_OPTIMIZED_EFFECTIVENESS_WITH_CI,
        mroi=_BUDGET_MROI_WITH_CI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: False, c.TARGET_ROI: 1},
    )

    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        fixed_budget=False, target_roi=1
    )

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @parameterized.named_parameters(
      dict(
          testcase_name='optimal_frequency',
          use_optimal_frequency=True,
          expected_spend=np.array([206.0, 276.0, 179.0, 354.0, 374.0]),
          expected_incremental_outcome=np.array([
              [236.0762, 231.97923, 58.01262, 424.9303],
              [410.90778, 402.0758, 163.52602, 672.65576],
              [86.25695, 85.316986, 42.073544, 133.43192],
              [1619.1354, 1625.3555, 567.3521, 2663.7307],
              [2493.0112, 2461.1328, 645.39844, 4406.3643],
          ]),
          expected_effectiveness=np.array([
              [0.16258691, 0.15976532, 0.03995359, 0.29265174],
              [0.29862484, 0.29220626, 0.11884159, 0.48884866],
              [0.0677588, 0.06702042, 0.0330507, 0.1048169],
              [0.00220999, 0.00221848, 0.00077439, 0.00363578],
              [0.00339354, 0.00335014, 0.00087853, 0.00599803],
          ]),
          expected_mroi=np.array([
              [1.0190495, 1.001245, 0.24795413, 1.8372412],
              [1.1924459, 1.1633278, 0.32793596, 2.1029253],
              [0.38008744, 0.3760929, 0.16844606, 0.60487074],
              [4.5738173, 4.5914817, 1.6026927, 7.524566],
              [6.6658034, 6.5806994, 1.7257135, 11.7819195],
          ]),
          expected_cpik=np.array([
              [2.014588, 2.0058842, 0.48478606, 3.5509517],
              [1.0528584, 1.0574462, 0.41031447, 1.687805],
              [2.8098052, 2.8149421, 1.3415098, 4.2544556],
              [0.3714944, 0.36298645, 0.13289629, 0.6239512],
              [0.33170277, 0.33070716, 0.08487724, 0.579487],
          ]),
      ),
      dict(
          testcase_name='historical_frequency',
          use_optimal_frequency=False,
          expected_spend=np.array([206.0, 307.0, 179.0, 323.0, 374.0]),
          expected_incremental_outcome=np.array([
              [236.0762, 231.97923, 58.01262, 424.9303],
              [446.08163, 436.3723, 172.97798, 734.9176],
              [86.25695, 85.316986, 42.073544, 133.43192],
              [372.6778, 372.97467, 154.88652, 589.69165],
              [595.451, 588.6903, 210.37312, 995.23],
          ]),
          expected_effectiveness=np.array([
              [1.6258691e-01, 1.5976532e-01, 3.9953593e-02, 2.9265174e-01],
              [3.2418722e-01, 3.1713104e-01, 1.2571074e-01, 5.3409714e-01],
              [6.7758799e-02, 6.7020416e-02, 3.3050705e-02, 1.0481690e-01],
              [1.1391223e-04, 1.1400296e-04, 4.7342415e-05, 1.8024442e-04],
              [1.7027026e-04, 1.6833706e-04, 6.0156573e-05, 2.8458779e-04],
          ]),
          expected_mroi=np.array([
              [1.0190499, 1.0012858, 0.24795783, 1.837228],
              [1.1488259, 1.120242, 0.30020398, 2.0423303],
              [0.38008675, 0.37608653, 0.16845182, 0.6048635],
              [1.1502323, 1.1511508, 0.47801477, 1.8200259],
              [1.5921059, 1.5740317, 0.56248915, 2.6610544],
          ]),
          expected_cpik=np.array([
              [2.014588, 2.0058842, 0.48478606, 3.5509517],
              [1.0979072, 1.1025534, 0.41657582, 1.7717088],
              [2.8098052, 2.8149421, 1.3415098, 4.2544556],
              [1.3132117, 1.305907, 0.54943967, 2.0918553],
              [1.0760998, 1.0721365, 0.3757926, 1.7777934],
          ]),
      ),
  )
  def test_optimized_data_use_optimal_frequency(
      self,
      use_optimal_frequency,
      expected_spend,
      expected_incremental_outcome,
      expected_effectiveness,
      expected_mroi,
      expected_cpik,
  ):
    expected_data = _create_budget_data(
        spend=expected_spend,
        inc_outcome=expected_incremental_outcome,
        effectiveness=expected_effectiveness,
        mroi=expected_mroi,
        explicit_cpik=expected_cpik,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )

    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        use_optimal_frequency=use_optimal_frequency
    )

    actual_data = optimization_results.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  def test_get_round_factor_gtol_raise_error(self):
    with self.assertRaisesWithLiteralMatch(
        expected_exception=ValueError,
        expected_exception_message='gtol must be less than one.',
    ):
      self.budget_optimizer_media_and_rf.optimize(gtol=1.0)

  def test_get_round_factor_budget_raise_error(self):
    with self.assertRaisesWithLiteralMatch(
        expected_exception=ValueError,
        expected_exception_message='`budget` must be greater than zero.',
    ):
      self.budget_optimizer_media_and_rf.optimize(budget=-10_000)

  def test_get_optimization_bounds_correct(self):
    (lower_bound, upper_bound, _) = (
        self.budget_optimizer_media_and_rf._get_optimization_bounds(
            spend=np.array([10642.5, 22222.0, 33333.0, 44444.0, 55555.0]),
            spend_constraint_lower=[0.5, 0.4, 0.3, 0.2, 0.1],
            spend_constraint_upper=[0.2544, 0.4, 0.5, 0.6, 0.7],
            round_factor=-2,
            fixed_budget=True,
        )
    )
    np.testing.assert_array_equal(
        lower_bound,
        np.array([5300, 13300, 23300, 35600, 50000]),
    )
    np.testing.assert_array_equal(
        upper_bound,
        np.array([13300, 31100, 50000, 71100, 94400]),
    )

  def test_optimization_grid_media_and_rf_correct(self):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            )),
        )
    )
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    spend = np.array([1000, 1000, 1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700, 800, 900])
    spend_bound_upper = np.array([1500, 1400, 1300, 1200, 1100])
    selected_times = ('2021-01-25', '2021-02-01')
    optimization_grid = (
        self.budget_optimizer_media_and_rf.create_optimization_grid(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            selected_times=selected_times,
            round_factor=-2,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0, 800.0, 900.0],
            [600.0, 700.0, 800.0, 900.0, 1000.0],
            [700.0, 800.0, 900.0, 1000.0, 1100.0],
            [800.0, 900.0, 1000.0, 1100.0, np.nan],
            [900.0, 1000.0, 1100.0, 1200.0, np.nan],
            [1000.0, 1100.0, 1200.0, np.nan, np.nan],
            [1100.0, 1200.0, 1300.0, np.nan, np.nan],
            [1200.0, 1300.0, np.nan, np.nan, np.nan],
            [1300.0, 1400.0, np.nan, np.nan, np.nan],
            [1400.0, np.nan, np.nan, np.nan, np.nan],
            [1500.0, np.nan, np.nan, np.nan, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [1.0, 1.0, 1.0, 0.67, 0.82],
            [1.0, 1.0, 1.0, 0.75, 0.91],
            [1.0, 1.0, 1.0, 0.83, 1.0],
            [1.0, 1.0, 1.0, 0.92, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
        ],
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        use_kpi=False,
        batch_size=c.DEFAULT_BATCH_SIZE,
        include_non_paid_channels=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_outcome.call_args
    np.testing.assert_allclose(
        mock_kwargs['new_data'].frequency,
        self.meridian_media_and_rf.rf_tensors.frequency,
    )
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_optimization_grid_media_only_correct(self):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_only._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS)),
        )
    )
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    spend = np.array([1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700])
    spend_bound_upper = np.array([1500, 1400, 1300])
    selected_times = ('2021-01-25', '2021-02-01')
    optimization_grid = (
        self.budget_optimizer_media_only.create_optimization_grid(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            selected_times=selected_times,
            round_factor=-2,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0],
            [600.0, 700.0, 800.0],
            [700.0, 800.0, 900.0],
            [800.0, 900.0, 1000.0],
            [900.0, 1000.0, 1100.0],
            [1000.0, 1100.0, 1200.0],
            [1100.0, 1200.0, 1300.0],
            [1200.0, 1300.0, np.nan],
            [1300.0, 1400.0, np.nan],
            [1400.0, np.nan, np.nan],
            [1500.0, np.nan, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
        include_non_paid_channels=False,
    )
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_optimization_grid_rf_only_correct(self):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_rf_only._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_RF_CHANNELS,
            )),
        )
    )
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    spend = np.array([1000, 1000])
    spend_bound_lower = np.array([500, 600])
    spend_bound_upper = np.array([1500, 1400])
    selected_times = ('2021-01-25', '2021-02-01')
    optimization_grid = self.budget_optimizer_rf_only.create_optimization_grid(
        spend=spend,
        spend_bound_lower=spend_bound_lower,
        spend_bound_upper=spend_bound_upper,
        selected_times=selected_times,
        round_factor=-2,
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0],
            [600.0, 700.0],
            [700.0, 800.0],
            [800.0, 900.0],
            [900.0, 1000.0],
            [1000.0, 1100.0],
            [1100.0, 1200.0],
            [1200.0, 1300.0],
            [1300.0, 1400.0],
            [1400.0, np.nan],
            [1500.0, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [0.33, 0.43],
            [0.4, 0.5],
            [0.47, 0.57],
            [0.53, 0.64],
            [0.6, 0.71],
            [0.67, 0.79],
            [0.73, 0.86],
            [0.8, 0.93],
            [0.87, 1.0],
            [0.93, np.nan],
            [1.0, np.nan],
        ],
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
        include_non_paid_channels=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_outcome.call_args
    np.testing.assert_allclose(
        mock_kwargs['new_data'].frequency,
        self.meridian_media_and_rf.rf_tensors.frequency,
    )
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_optimization_grid_with_optimal_frequency_media_and_rf_correct(self):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            )),
        )
    )
    spend = np.array([1000, 1000, 1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700, 800, 900])
    spend_bound_upper = np.array([1500, 1400, 1300, 1200, 1100])
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])
    optimization_grid = (
        self.budget_optimizer_media_and_rf.create_optimization_grid(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            selected_times=selected_times,
            round_factor=-2,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0, 800.0, 900.0],
            [600.0, 700.0, 800.0, 900.0, 1000.0],
            [700.0, 800.0, 900.0, 1000.0, 1100.0],
            [800.0, 900.0, 1000.0, 1100.0, np.nan],
            [900.0, 1000.0, 1100.0, 1200.0, np.nan],
            [1000.0, 1100.0, 1200.0, np.nan, np.nan],
            [1100.0, 1200.0, 1300.0, np.nan, np.nan],
            [1200.0, 1300.0, np.nan, np.nan, np.nan],
            [1300.0, 1400.0, np.nan, np.nan, np.nan],
            [1400.0, np.nan, np.nan, np.nan, np.nan],
            [1500.0, np.nan, np.nan, np.nan, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [1.0, 1.0, 1.0, 0.67, 0.82],
            [1.0, 1.0, 1.0, 0.75, 0.91],
            [1.0, 1.0, 1.0, 0.83, 1.0],
            [1.0, 1.0, 1.0, 0.92, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
        ],
    )
    new_frequency = (
        tf.ones_like(self.meridian_media_and_rf.rf_tensors.frequency)
        * optimal_frequency
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
        include_non_paid_channels=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_outcome.call_args
    np.testing.assert_allclose(mock_kwargs['new_data'].frequency, new_frequency)
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_optimization_grid_with_optimal_frequency_media_only_correct(
      self,
  ):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_only._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS)),
        )
    )
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    spend = np.array([1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700])
    spend_bound_upper = np.array([1500, 1400, 1300])
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])

    optimization_grid = (
        self.budget_optimizer_media_only.create_optimization_grid(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            selected_times=selected_times,
            round_factor=-2,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0],
            [600.0, 700.0, 800.0],
            [700.0, 800.0, 900.0],
            [800.0, 900.0, 1000.0],
            [900.0, 1000.0, 1100.0],
            [1000.0, 1100.0, 1200.0],
            [1100.0, 1200.0, 1300.0],
            [1200.0, 1300.0, np.nan],
            [1300.0, 1400.0, np.nan],
            [1400.0, np.nan, np.nan],
            [1500.0, np.nan, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
        include_non_paid_channels=False,
    )
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_optimization_grid_with_optimal_frequency_rf_only_correct(
      self,
  ):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_rf_only._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((_N_CHAINS, _N_DRAWS, _N_RF_CHANNELS)),
        )
    )
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    spend = np.array([1000, 1000])
    spend_bound_lower = np.array([500, 600])
    spend_bound_upper = np.array([1500, 1400])
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])
    optimization_grid = self.budget_optimizer_rf_only.create_optimization_grid(
        spend=spend,
        spend_bound_lower=spend_bound_lower,
        spend_bound_upper=spend_bound_upper,
        selected_times=selected_times,
        round_factor=-2,
        optimal_frequency=optimal_frequency,
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0],
            [600.0, 700.0],
            [700.0, 800.0],
            [800.0, 900.0],
            [900.0, 1000.0],
            [1000.0, 1100.0],
            [1100.0, 1200.0],
            [1200.0, 1300.0],
            [1300.0, 1400.0],
            [1400.0, np.nan],
            [1500.0, np.nan],
        ],
    )
    expected_incremental_outcome_grid = np.array(
        [
            [0.33, 0.43],
            [0.4, 0.5],
            [0.47, 0.57],
            [0.53, 0.64],
            [0.6, 0.71],
            [0.67, 0.79],
            [0.73, 0.86],
            [0.8, 0.93],
            [0.87, 1.0],
            [0.93, np.nan],
            [1.0, np.nan],
        ],
    )
    new_frequency = (
        tf.ones_like(self.meridian_media_and_rf.rf_tensors.frequency)
        * optimal_frequency
    )
    mock_incremental_outcome.assert_called_with(
        use_posterior=True,
        new_data=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
        include_non_paid_channels=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertEqual(optimization_grid.spend_step_size, 100)
    np.testing.assert_allclose(mock_kwargs['new_data'].frequency, new_frequency)
    np.testing.assert_allclose(
        optimization_grid.spend_grid, expected_spend_grid, equal_nan=True
    )
    np.testing.assert_allclose(
        optimization_grid.incremental_outcome_grid,
        expected_incremental_outcome_grid,
        equal_nan=True,
        atol=0.01,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'default_budget_scenario',
          'fixed_budget': True,
          'target_roi': None,
          'target_mroi': None,
          'expected_optimal_spend': np.array([800, 900, 1000, 1200, 1100]),
      },
      {
          'testcase_name': 'flexible_budget_target_roi_scenario',
          'fixed_budget': False,
          'target_roi': 2.0,
          'target_mroi': None,
          'expected_optimal_spend': np.array([500, 600, 700, 800, 900]),
      },
      {
          'testcase_name': 'flexible_budget_target_mroi_scenario',
          'fixed_budget': False,
          'target_roi': 2.0,
          'target_mroi': None,
          'expected_optimal_spend': np.array([500, 600, 700, 800, 900]),
      },
  )
  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_grid_search_correct(
      self,
      mock_incremental_outcome,
      fixed_budget,
      target_roi,
      target_mroi,
      expected_optimal_spend,
  ):
    mock_incremental_outcome.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    spend = np.array([1000, 1000, 1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700, 800, 900])
    spend_bound_upper = np.array([1500, 1400, 1300, 1200, 1100])
    step_size = 100

    selected_times = (
        self.budget_optimizer_media_and_rf._meridian.expand_selected_time_dims()
    )
    (spend_grid, incremental_outcome_grid) = (
        self.budget_optimizer_media_and_rf._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
        )
    )
    optimal_spend = self.budget_optimizer_media_and_rf._grid_search(
        spend_grid=spend_grid,
        incremental_outcome_grid=incremental_outcome_grid,
        budget=np.sum(spend),
        fixed_budget=fixed_budget,
        target_roi=target_roi,
        target_mroi=target_mroi,
    )
    np.testing.assert_array_equal(optimal_spend, expected_optimal_spend)

  def test_optimization_grid_nans_match(self):
    self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'get_historical_spend',
            return_value=mock.MagicMock(data=np.array([100, 200, 0, 400, 0])),
        )
    )

    opt_results = self.budget_optimizer_media_and_rf.optimize()
    np.testing.assert_array_equal(
        np.isnan(opt_results.optimization_grid.spend_grid),
        np.isnan(opt_results.optimization_grid.incremental_outcome_grid),
    )

  def test_grid_search_with_target_mroi_correct(self):
    spend_grid = np.array([
        [0, 0, 0, 0, 65900000, 40300000],
        [100000, 100000, 100000, 100000, 66000000, 40400000],
        [200000, 200000, 200000, 200000, 66100000, 40500000],
        [300000, 300000, 300000, 300000, 66200000, 40600000],
        [400000, 400000, 400000, 400000, 66300000, 40700000],
        [500000, 500000, 500000, 500000, 66400000, 40800000],
    ])

    incremental_outcome_grid = np.array([
        [0, 0, 0, 0, 31020000, 16160000],
        [220000, 360000, 180000, 390000, 31070000, 16200000],
        [430000, 710000, 350000, 780000, 31120000, 16240000],
        [650000, 1060000, 530000, 1160000, 31170000, 16280000],
        [860000, 1400000, 700000, 1520000, 31220000, 16320000],
        [1060000, 1730000, 870000, 1880000, 31270000, 16360000],
    ])

    budget = 225_000_000
    fixed_budget = False
    target_roi = 1.0
    target_mroi = None

    optimal_spend = self.budget_optimizer_media_and_rf._grid_search(
        spend_grid=spend_grid,
        incremental_outcome_grid=incremental_outcome_grid,
        budget=budget,
        fixed_budget=fixed_budget,
        target_roi=target_roi,
        target_mroi=target_mroi,
    )

    expected_optimal_spend = np.array(
        [500000, 500000, 500000, 500000, 66400000, 40300000]
    )

    np.testing.assert_array_equal(optimal_spend, expected_optimal_spend)

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_optimizer_budget_with_specified_budget(
      self, mock_incremental_outcome
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    budget = 2000

    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        budget=budget, fixed_budget=True
    )

    self.assertEqual(
        optimization_results.nonoptimized_data.budget,
        budget,
    )
    self.assertEqual(
        optimization_results.optimized_data.budget,
        budget,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_budget_data_with_specified_pct_of_spend(
      self, mock_incremental_outcome
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    expected_pct_of_spend = [0.1, 0.2, 0.3, 0.3, 0.1]

    idata = self.budget_optimizer_media_and_rf._meridian.input_data
    paid_channels = list(idata.get_all_paid_channels())
    pct_of_spend = idata.get_paid_channels_argument_builder()(**{
        paid_channels[0]: 0.1,
        paid_channels[1]: 0.2,
        paid_channels[2]: 0.3,
        paid_channels[3]: 0.3,
        paid_channels[4]: 0.1,
    })

    optimization_results = self.budget_optimizer_media_and_rf.optimize(
        pct_of_spend=pct_of_spend, fixed_budget=True
    )

    actual_spend = optimization_results.nonoptimized_data.spend
    actual_budget = optimization_results.nonoptimized_data.budget
    np.testing.assert_almost_equal(
        optimization_results.nonoptimized_data.pct_of_spend,
        expected_pct_of_spend,
        decimal=7,
    )
    np.testing.assert_array_equal(
        optimization_results.nonoptimized_data.budget,
        optimization_results.optimized_data.budget,
    )
    np.testing.assert_almost_equal(
        expected_pct_of_spend,
        actual_spend / actual_budget,
        decimal=7,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_budget_data_with_specified_pct_of_spend_size_incorrect(
      self, mock_incremental_outcome
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Percent of spend must be specified for all channels.',
    ):
      pct_of_spend = np.array([0.1, 0.2, 0.3, 0.3])
      self.budget_optimizer_media_and_rf.optimize(
          pct_of_spend=pct_of_spend, fixed_budget=True
      )

  @mock.patch.object(analyzer.Analyzer, 'incremental_outcome', autospec=True)
  def test_budget_data_with_specified_pct_of_spend_value_incorrect(
      self, mock_incremental_outcome
  ):
    mock_incremental_outcome.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Percent of spend must sum to one.',
    ):
      pct_of_spend = np.array([0.1, 0.2, 0.3, 0.3, 0.5], dtype=np.float32)
      self.budget_optimizer_media_and_rf.optimize(
          pct_of_spend=pct_of_spend, fixed_budget=True
      )

  def test_batch_size(self):
    batch_size = 500
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.convert_to_tensor(
                [[_NONOPTIMIZED_INCREMENTAL_OUTCOME]], tf.float32
            ),
        )
    )
    self.budget_optimizer_media_and_rf.optimize(batch_size=batch_size)
    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertEqual(mock_kwargs['batch_size'], batch_size)

  def test_optimize_with_non_paid_channels_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            'inference_data',
            new=property(lambda unused_self: self.inference_data_all_channels),
        )
    )
    optimization_results = self.budget_optimizer_all_channels.optimize()
    self.assertLen(
        optimization_results.spend_ratio, _N_MEDIA_CHANNELS + _N_RF_CHANNELS
    )


class OptimizerPlotsTest(absltest.TestCase):

  def setUp(self):
    super(OptimizerPlotsTest, self).setUp()
    mock_data = mock.create_autospec(input_data.InputData, instance=True)
    mock_time = analysis_test_utils.generate_selected_times('2020-01-05', 52)
    type(mock_data).time = mock.PropertyMock(
        return_value=xr.DataArray(data=mock_time, coords=dict(time=mock_time))
    )
    meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=mock_data
    )
    n_times = 149
    n_geos = 10
    self.revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=1.0
    )
    meridian.input_data.kpi_type = c.REVENUE
    meridian.input_data.revenue_per_kpi = self.revenue_per_kpi

    self.meridian = meridian
    self.budget_optimizer = optimizer.BudgetOptimizer(meridian)
    self.optimization_grid = optimizer.OptimizationGrid(
        _grid_dataset=mock.MagicMock(),
        spend=np.array([0, 0, 0]),
        use_kpi=False,
        use_posterior=True,
        use_optimal_frequency=False,
        round_factor=1,
        optimal_frequency=None,
        selected_times=self.meridian.expand_selected_time_dims(),
    )
    self.optimization_results = optimizer.OptimizationResults(
        meridian=self.budget_optimizer._meridian,
        analyzer=self.budget_optimizer._analyzer,
        spend_ratio=np.array([1.0, 1.0, 1.0]),
        spend_bounds=(
            np.array([0.7, 0.5, 0.7]),
            np.array([1.3]),
        ),
        _nonoptimized_data=_SAMPLE_NON_OPTIMIZED_DATA,
        _nonoptimized_data_with_optimal_freq=_SAMPLE_NON_OPTIMIZED_DATA,
        _optimized_data=_SAMPLE_OPTIMIZED_DATA,
        _optimization_grid=self.optimization_grid,
    )

    spend_multiplier = np.arange(0, 2, 0.01)
    self.mock_response_curves = self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'response_curves',
            return_value=analysis_test_utils.generate_response_curve_data(
                n_channels=3, spend_multiplier=spend_multiplier
            ),
        )
    )

  def test_outcome_waterfall_chart_data_correct(self):
    plot = self.optimization_results.plot_incremental_outcome_delta()
    df = plot.data
    self.assertEqual(list(df.columns), [c.CHANNEL, c.INCREMENTAL_OUTCOME])
    self.assertEqual(
        list(df.channel),
        ['non_optimized', 'channel 2', 'channel 0', 'channel 1', 'optimized'],
    )

  def test_outcome_waterfall_chart_correct_config(self):
    plot = self.optimization_results.plot_incremental_outcome_delta()
    config = plot.config.to_dict()
    self.assertEqual(
        config['axis'],
        {
            'labelColor': c.GREY_700,
            'labelFont': c.FONT_ROBOTO,
            'labelFontSize': c.AXIS_FONT_SIZE,
            'titleColor': c.GREY_700,
            'titleFont': c.FONT_ROBOTO,
            'titleFontSize': c.AXIS_FONT_SIZE,
            'titleFontWeight': 'normal',
        },
    )
    self.assertEqual(config['view'], {'strokeOpacity': 0})

  def test_outcome_waterfall_chart_correct_mark(self):
    plot = self.optimization_results.plot_incremental_outcome_delta()
    self.assertEqual(
        plot.layer[0].mark.to_dict(),
        {
            'clip': True,
            'cornerRadius': c.CORNER_RADIUS,
            'size': c.BAR_SIZE,
            'type': 'bar',
        },
    )
    self.assertEqual(
        plot.layer[1].mark.to_dict(),
        {
            'baseline': 'top',
            'color': c.GREY_800,
            'dy': -20,
            'fontSize': c.AXIS_FONT_SIZE,
            'type': 'text',
        },
    )

  def test_outcome_waterfall_chart_correct_bar_encoding(self):
    plot = self.optimization_results.plot_incremental_outcome_delta()
    encoding = plot.layer[0].encoding.to_dict()
    self.assertEqual(
        encoding['color'],
        {
            'condition': [
                {
                    'value': c.BLUE_500,
                    'test': (
                        f"datum.channel === '{c.NON_OPTIMIZED}' ||"
                        f" datum.channel === '{c.OPTIMIZED}'"
                    ),
                },
                {'value': c.RED_300, 'test': 'datum.incremental_outcome < 0'},
            ],
            'value': c.CYAN_400,
        },
    )
    self.assertEqual(
        encoding['y'],
        {
            'axis': {
                'domain': False,
                'labelExpr': "replace(format(datum.value, '.3~s'), 'G', 'B')",
                'labelPadding': c.PADDING_10,
                'tickCount': 5,
                'ticks': False,
                'title': summary_text.INC_REVENUE_LABEL,
                'titleAlign': 'left',
                'titleAngle': 0,
                'titleY': -20,
            },
            'field': 'prev_sum',
            'scale': {'domain': [690.0, 840.0]},
            'type': 'quantitative',
        },
    )
    self.assertEqual(encoding['y2'], {'field': 'sum_outcome'})
    self.assertEqual(
        encoding['tooltip'],
        [
            {'field': c.CHANNEL, 'type': 'nominal'},
            {'field': c.INCREMENTAL_OUTCOME, 'type': 'quantitative'},
        ],
    )
    self.assertEqual(
        plot.title.text,
        summary_text.OUTCOME_DELTA_CHART_TITLE.format(outcome=c.REVENUE),
    )

  def test_outcome_waterfall_chart_correct_text_encoding(self):
    plot = self.optimization_results.plot_incremental_outcome_delta()
    encoding = plot.layer[1].encoding.to_dict()
    text = encoding['text']
    self.assertEqual(
        encoding['x'],
        {
            'axis': {
                'domainColor': c.GREY_300,
                'labelAngle': -45,
                'labelPadding': c.PADDING_10,
                'ticks': False,
            },
            'field': c.CHANNEL,
            'scale': {'paddingOuter': c.SCALED_PADDING},
            'sort': None,
            'title': None,
            'type': 'nominal',
        },
    )
    self.assertEqual(text, {'field': 'calc_amount', 'type': 'nominal'})
    self.assertEqual(encoding['y'], {'field': 'text_y', 'type': 'quantitative'})

  def test_budget_allocation_optimized_data(self):
    plot = self.optimization_results.plot_budget_allocation()
    df = plot.data
    self.assertEqual(list(df.columns), [c.CHANNEL, c.SPEND])
    self.assertEqual(df.channel.size, 3)
    self.assertEqual(list(df.spend), list(_SAMPLE_OPTIMIZED_DATA.spend.values))

  def test_budget_allocation_nonoptimized_data(self):
    plot = self.optimization_results.plot_budget_allocation(optimized=False)
    df = plot.data
    self.assertEqual(
        list(df.spend), list(_SAMPLE_NON_OPTIMIZED_DATA.spend.values)
    )

  def test_budget_allocation_correct_encoding(self):
    plot = self.optimization_results.plot_budget_allocation()
    encoding = plot.encoding.to_dict()
    config = plot.config.to_dict()
    mark = plot.mark.to_dict()

    self.assertEqual(
        encoding,
        {
            'color': {
                'field': c.CHANNEL,
                'legend': {
                    'offset': -25,
                    'rowPadding': c.PADDING_10,
                    'title': None,
                },
                'type': 'nominal',
            },
            'theta': {'field': c.SPEND, 'type': 'quantitative'},
        },
    )
    self.assertEqual(config, {'view': {'stroke': None}})
    self.assertEqual(mark, {'padAngle': 0.02, 'tooltip': True, 'type': 'arc'})
    self.assertEqual(plot.title.text, summary_text.SPEND_ALLOCATION_CHART_TITLE)

  def test_plot_spend_delta_correct_data(self):
    plot = self.optimization_results.plot_spend_delta()
    df = plot.data
    self.assertEqual(list(df.columns), [c.CHANNEL, c.SPEND])
    self.assertEqual(
        list(df.channel),
        [
            'channel 2',
            'channel 1',
            'channel 0',
        ],
    )

  def test_plot_spend_delta_correct_config(self):
    plot = self.optimization_results.plot_spend_delta()
    axis_config = plot.config.axis.to_dict()
    view_config = plot.config.view.to_dict()

    self.assertEqual(axis_config, formatter.TEXT_CONFIG)
    self.assertEqual(view_config, {'stroke': None})
    self.assertEqual(plot.width, formatter.bar_chart_width(5))  # n_channels + 2
    self.assertEqual(plot.height, 400)

  def test_plot_spend_delta_correct_mark(self):
    plot = self.optimization_results.plot_spend_delta()
    bar_mark = plot.layer[0].mark.to_dict()
    text_mark = plot.layer[1].mark.to_dict()
    self.assertEqual(
        bar_mark,
        {
            'cornerRadiusEnd': c.CORNER_RADIUS,
            'size': c.BAR_SIZE,
            'tooltip': True,
            'type': 'bar',
        },
    )
    self.assertEqual(
        text_mark,
        {
            'baseline': 'top',
            'color': c.GREY_800,
            'dy': -20,
            'fontSize': c.AXIS_FONT_SIZE,
            'type': 'text',
        },
    )

  def test_plot_spend_delta_correct_bar_encoding(self):
    plot = self.optimization_results.plot_spend_delta()
    encoding = plot.layer[0].encoding.to_dict()
    self.assertEqual(
        encoding['color'],
        {
            'condition': {'test': '(datum.spend > 0)', 'value': c.CYAN_400},
            'value': c.RED_300,
        },
    )
    self.assertEqual(
        encoding['x'],
        {
            'axis': {
                'labelAngle': -45,
                'title': None,
                **formatter.AXIS_CONFIG,
            },
            'field': c.CHANNEL,
            'scale': {'padding': c.BAR_SIZE},
            'sort': None,
            'type': 'nominal',
        },
    )
    self.assertEqual(
        encoding['y'],
        {
            'axis': {
                'domain': False,
                'labelExpr': formatter.compact_number_expr(),
                'title': '$',
                **formatter.AXIS_CONFIG,
                **formatter.Y_AXIS_TITLE_CONFIG,
            },
            'field': c.SPEND,
            'type': 'quantitative',
        },
    )
    self.assertEqual(plot.title.text, summary_text.SPEND_DELTA_CHART_TITLE)

  def test_plot_spend_delta_correct_text_encoding(self):
    plot = self.optimization_results.plot_spend_delta()
    encoding = plot.layer[1].encoding.to_dict()
    text_encoding = encoding['text']
    self.assertEqual(text_encoding, {'field': 'text_value', 'type': 'nominal'})
    self.assertEqual(encoding['y'], {'field': 'text_y', 'type': 'quantitative'})
    self.assertEqual(
        encoding['x'],
        {
            'axis': {'labelAngle': -45, 'title': None, **formatter.AXIS_CONFIG},
            'field': c.CHANNEL,
            'scale': {'padding': c.BAR_SIZE},
            'sort': None,
            'type': 'nominal',
        },
    )

  def test_get_response_curves(self):
    ds = self.optimization_results.get_response_curves()
    self.assertEqual(
        list(ds.to_dataframe().reset_index().columns),
        [
            c.SPEND_MULTIPLIER,
            c.CHANNEL,
            c.METRIC,
            c.SPEND,
            c.INCREMENTAL_OUTCOME,
        ],
    )

    _, mock_kwargs = self.meridian.expand_selected_time_dims.call_args
    self.assertEqual(
        mock_kwargs,
        {
            'start_date': self.optimization_results.optimized_data.start_date,
            'end_date': self.optimization_results.optimized_data.end_date,
        },
    )

    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.mock_response_curves.call_args
    # Check that the spend multiplier max is 2.
    multiplier = np.arange(0, 2, 0.01)
    np.testing.assert_array_equal(mock_kwargs['spend_multipliers'], multiplier)
    self.assertEqual(mock_kwargs['by_reach'], True)

  def test_plot_response_curves_correct_data(self):
    plot = self.optimization_results.plot_response_curves()
    df = plot.data
    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.mock_response_curves.call_args
    # Check that the spend multiplier max is 2.
    multiplier = np.arange(0, 2, 0.01)
    np.testing.assert_array_equal(mock_kwargs['spend_multipliers'], multiplier)
    self.assertEqual(mock_kwargs['by_reach'], True)
    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.SPEND,
            c.SPEND_MULTIPLIER,
            c.CI_HI,
            c.CI_LO,
            c.MEAN,
            c.SPEND_LEVEL,
            c.LOWER_BOUND,
            c.UPPER_BOUND,
        ],
    )
    # Check that the current and optimal points for each channel are included.
    self.assertEqual(
        set(df[c.SPEND_LEVEL].dropna()),
        {
            summary_text.OPTIMIZED_SPEND_LABEL,
            summary_text.NONOPTIMIZED_SPEND_LABEL,
        },
    )
    self.assertLen(
        df[df[c.SPEND_LEVEL] == summary_text.OPTIMIZED_SPEND_LABEL], 3
    )
    self.assertLen(
        df[df[c.SPEND_LEVEL] == summary_text.NONOPTIMIZED_SPEND_LABEL], 3
    )
    self.assertContainsSubset(
        self.optimization_results.spend_bounds[0], df[c.LOWER_BOUND]
    )
    self.assertContainsSubset(
        self.optimization_results.spend_bounds[1], df[c.UPPER_BOUND]
    )

  def test_plot_response_curves_modified_bounds(self):
    optimization_results = dataclasses.replace(
        self.optimization_results,
        spend_bounds=(np.array([0.7]), np.array([1.2, 1.3, 1.4])),
    )
    plot = optimization_results.plot_response_curves()
    df = plot.data
    self.assertContainsSubset(
        optimization_results.spend_bounds[0], df[c.LOWER_BOUND]
    )
    self.assertContainsSubset(
        optimization_results.spend_bounds[1], df[c.UPPER_BOUND]
    )

  def test_plot_response_curves_invalid_n_channels(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Top number of channels (5) by spend must be less than the total'
        ' number of channels (3)',
    ):
      self.optimization_results.plot_response_curves(5)

  def test_plot_response_curves_correct_selected_times(self):
    self.optimization_results.plot_response_curves()
    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.meridian.expand_selected_time_dims.call_args
    self.assertEqual(
        mock_kwargs,
        {
            'start_date': self.optimization_results.optimized_data.start_date,
            'end_date': self.optimization_results.optimized_data.end_date,
        },
    )

  def test_plot_response_curves_n_top_channels(self):
    plot = self.optimization_results.plot_response_curves(2)
    channels = list(plot.data.channel.unique())
    self.assertEqual(channels, ['channel 2', 'channel 0'])

  def test_plot_response_curves_upper_limit(self):
    optimization_results = dataclasses.replace(
        self.optimization_results,
        spend_bounds=(np.array([0]), np.array([2])),
    )

    optimization_results.plot_response_curves()

    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.mock_response_curves.call_args
    # Check that the spend multiplier max is the upper limit of the upper spend
    # bound + spend constraint padding.
    multiplier = np.arange(0, 2.5, 0.01)
    np.testing.assert_array_equal(mock_kwargs['spend_multipliers'], multiplier)
    self.assertEqual(mock_kwargs['by_reach'], True)

  def test_plot_response_curves_correct_config(self):
    plot = self.optimization_results.plot_response_curves()
    self.assertEqual(plot.config.axis.to_dict(), formatter.TEXT_CONFIG)

  def test_plot_response_curves_correct_facet_properties(self):
    plot = self.optimization_results.plot_response_curves()
    self.assertEqual(
        plot.facet.to_dict(),
        {'field': c.CHANNEL, 'title': None, 'type': 'nominal', 'sort': None},
    )
    self.assertLen(plot.spec.layer, 3)
    self.assertEqual(plot.columns, 3)
    self.assertEqual(plot.resolve.scale.x, 'independent')
    self.assertEqual(plot.resolve.scale.y, 'independent')

  def test_plot_response_curves_correct_encoding(self):
    plot = self.optimization_results.plot_response_curves()
    base_encoding = {
        'color': {'field': c.CHANNEL, 'legend': None, 'type': 'nominal'},
        'x': {
            'axis': {
                'labelExpr': formatter.compact_number_expr(),
                **formatter.AXIS_CONFIG,
            },
            'field': c.SPEND,
            'title': 'Spend',
            'type': 'quantitative',
        },
        'y': {
            'field': c.MEAN,
            'title': summary_text.INC_REVENUE_LABEL,
            'type': 'quantitative',
            'axis': {
                'labelExpr': formatter.compact_number_expr(),
                **formatter.AXIS_CONFIG,
                **formatter.Y_AXIS_TITLE_CONFIG,
            },
        },
    }
    constraint_and_above_curve_encoding = {
        'strokeDash': {
            'field': c.SPEND_CONSTRAINT,
            'legend': {'title': None},
            'sort': 'descending',
            'type': 'nominal',
        }
    }
    points_encoding = {
        'shape': {
            'field': c.SPEND_LEVEL,
            'legend': {'title': None},
            'type': 'nominal',
        }
    }
    self.assertEqual(plot.spec.layer[0].encoding.to_dict(), base_encoding)
    self.assertEqual(
        plot.spec.layer[1].encoding.to_dict(),
        base_encoding | constraint_and_above_curve_encoding,
    )
    self.assertEqual(
        plot.spec.layer[2].encoding.to_dict(), base_encoding | points_encoding
    )

  def test_plot_response_curves_below_constraint_line_properties(self):
    plot = self.optimization_results.plot_response_curves()
    below_constraint_line = plot.spec.layer[0]
    self.assertEqual(
        below_constraint_line.mark.to_dict(),
        {'strokeDash': list(c.STROKE_DASH), 'type': 'line'},
    )
    self.assertEqual(
        below_constraint_line.transform[1].to_dict(),
        {
            'filter': (
                '(datum.spend_multiplier && (datum.spend_multiplier <='
                ' datum.lower_bound))'
            )
        },
    )

  def test_plot_response_curves_constraint_and_above_line_properties(self):
    plot = self.optimization_results.plot_response_curves()
    constraint_and_above_line = plot.spec.layer[1]
    self.assertEqual(constraint_and_above_line.mark, 'line')
    self.assertEqual(
        constraint_and_above_line.transform[0].to_dict(),
        {
            'as': c.SPEND_CONSTRAINT,
            'calculate': (
                'datum.spend_multiplier >= datum.lower_bound &&'
                ' datum.spend_multiplier <= datum.upper_bound ? "Within spend'
                ' constraint" : "Outside spend constraint"'
            ),
        },
    )
    self.assertEqual(
        constraint_and_above_line.transform[1].to_dict(),
        {
            'filter': (
                '(datum.spend_multiplier && (datum.spend_multiplier >='
                ' datum.lower_bound))'
            )
        },
    )

  def test_plot_response_curves_points_properties(self):
    plot = self.optimization_results.plot_response_curves()
    points = plot.spec.layer[2]
    self.assertEqual(
        points.mark.to_dict(),
        {
            'opacity': 1,
            'size': c.POINT_SIZE,
            'type': 'point',
            'filled': True,
            'tooltip': True,
        },
    )
    self.assertEqual(
        points.transform[1].to_dict(),
        {'filter': 'datum.spend_level'},
    )


class OptimizerOutputTest(parameterized.TestCase):

  def setUp(self):
    super(OptimizerOutputTest, self).setUp()
    mock_data = mock.create_autospec(input_data.InputData, instance=True)
    mock_data_kpi_output = mock.create_autospec(
        input_data.InputData, instance=True
    )
    meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=mock_data
    )
    meridian_kpi_output = mock.create_autospec(
        model.Meridian, instance=True, input_data=mock_data_kpi_output
    )
    n_times = 149
    n_geos = 10
    self.revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=1.0
    )
    meridian.input_data.kpi_type = c.REVENUE
    meridian.input_data.revenue_per_kpi = self.revenue_per_kpi
    meridian_kpi_output.input_data.kpi_type = c.NON_REVENUE
    meridian_kpi_output.input_data.revenue_per_kpi = None

    self.budget_optimizer = optimizer.BudgetOptimizer(meridian)
    self.budget_optimizer_kpi_output = optimizer.BudgetOptimizer(
        meridian_kpi_output
    )
    self.optimization_grid = optimizer.OptimizationGrid(
        _grid_dataset=mock.MagicMock(),
        spend=np.array([0, 0, 0]),
        use_kpi=False,
        use_posterior=True,
        use_optimal_frequency=False,
        round_factor=1,
        optimal_frequency=None,
        selected_times=mock.MagicMock(),
    )
    self.optimization_results = optimizer.OptimizationResults(
        meridian=self.budget_optimizer._meridian,
        analyzer=self.budget_optimizer._analyzer,
        spend_ratio=np.array([1.0, 1.0, 1.0]),
        spend_bounds=(np.array([0.7]), np.array([1.3])),
        _nonoptimized_data=_SAMPLE_NON_OPTIMIZED_DATA,
        _optimized_data=_SAMPLE_OPTIMIZED_DATA,
        _nonoptimized_data_with_optimal_freq=mock.MagicMock(),
        _optimization_grid=self.optimization_grid,
    )
    self.optimization_results_kpi_output = optimizer.OptimizationResults(
        meridian=self.budget_optimizer_kpi_output._meridian,
        analyzer=self.budget_optimizer_kpi_output._analyzer,
        spend_ratio=np.array([1.0, 1.0, 1.0]),
        spend_bounds=(np.array([0.7]), np.array([1.3])),
        _nonoptimized_data=_SAMPLE_NON_OPTIMIZED_DATA_KPI,
        _optimized_data=_SAMPLE_OPTIMIZED_DATA_KPI,
        _nonoptimized_data_with_optimal_freq=mock.MagicMock(),
        _optimization_grid=self.optimization_grid,
    )
    self.mock_spend_delta = self.enter_context(
        mock.patch.object(
            optimizer.OptimizationResults,
            'plot_spend_delta',
            return_value=self._mock_chart(),
        )
    )
    self.mock_budget_allocation = self.enter_context(
        mock.patch.object(
            optimizer.OptimizationResults,
            'plot_budget_allocation',
            return_value=self._mock_chart(),
        )
    )
    self.mock_outcome_delta = self.enter_context(
        mock.patch.object(
            optimizer.OptimizationResults,
            'plot_incremental_outcome_delta',
            return_value=self._mock_chart(),
        )
    )
    self.mock_response_curves = self.enter_context(
        mock.patch.object(
            optimizer.OptimizationResults,
            'plot_response_curves',
            return_value=self._mock_chart(),
        )
    )

  def _mock_chart(self) -> alt.Chart:
    return alt.Chart(pd.DataFrame()).mark_point()

  def _get_output_summary_html_dom(
      self, optimization_results: optimizer.OptimizationResults
  ) -> ET.Element:
    outfile_path = tempfile.mkdtemp() + '/optimization'
    outfile_name = 'optimization.html'
    fpath = os.path.join(outfile_path, outfile_name)

    try:
      optimization_results.output_optimization_summary(
          outfile_name, outfile_path
      )
      with open(fpath, 'r') as f:
        written_html_dom = ET.parse(f)
    finally:
      os.remove(fpath)
      os.removedirs(outfile_path)

    root = written_html_dom.getroot()
    self.assertEqual(root.tag, 'html')
    return root

  def test_output_html_title(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    title = summary_html_dom.find('head/title')

    self.assertIsNotNone(title)
    title_text = title.text
    self.assertIsNotNone(title_text)
    self.assertEqual(title_text.strip(), summary_text.OPTIMIZATION_TITLE)

  def test_output_header_section(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    header_div = analysis_test_utils.get_child_element(
        summary_html_dom, 'body/div', {'class': 'header'}
    )
    _ = analysis_test_utils.get_child_element(
        header_div, 'div', {'class': 'logo'}
    )
    header_title_div = analysis_test_utils.get_child_element(
        header_div, 'div', {'class': 'title'}
    )

    header_title_div_text = header_title_div.text
    self.assertIsNotNone(header_title_div_text)
    self.assertEqual(
        header_title_div_text.strip(), summary_text.OPTIMIZATION_TITLE
    )

  def test_output_chips(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    chips_node = summary_html_dom.find('body/chips')
    self.assertIsNotNone(chips_node)
    chip_nodes = chips_node.findall('chip')

    self.assertLen(chip_nodes, 1)
    self.assertSequenceEqual(
        [chip.text.strip() for chip in chip_nodes if chip.text is not None],
        [
            'Time period: 2020-01-05 - 2020-06-28',
        ],
    )

  def test_output_scenario_plan_card_text(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    card_title_text = analysis_test_utils.get_child_element(
        card, 'card-title'
    ).text
    self.assertIsNotNone(card_title_text)
    self.assertEqual(
        card_title_text.strip(), summary_text.SCENARIO_PLAN_CARD_TITLE
    )
    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(),
        summary_text.SCENARIO_PLAN_INSIGHTS_UNIFORM_SPEND_BOUNDS.format(
            scenario_type='fixed',
            lower_bound=30,
            upper_bound=30,
        )
        + ' '
        + summary_text.SCENARIO_PLAN_INSIGHTS_HISTORICAL_BUDGET.format(
            start_date='2020-01-05', end_date='2020-06-28'
        ),
    )

  def test_output_scenario_plan_card_text_new_budget(self):
    new_non_optimized_data = _SAMPLE_NON_OPTIMIZED_DATA.copy()
    new_non_optimized_data.attrs[c.USE_HISTORICAL_BUDGET] = False
    new_budget_optimization_results = dataclasses.replace(
        self.optimization_results,
        _nonoptimized_data=new_non_optimized_data,
    )
    summary_html_dom = self._get_output_summary_html_dom(
        new_budget_optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    card_title_text = analysis_test_utils.get_child_element(
        card, 'card-title'
    ).text
    self.assertIsNotNone(card_title_text)
    self.assertEqual(
        card_title_text.strip(), summary_text.SCENARIO_PLAN_CARD_TITLE
    )
    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(),
        summary_text.SCENARIO_PLAN_INSIGHTS_UNIFORM_SPEND_BOUNDS.format(
            scenario_type='fixed',
            lower_bound=30,
            upper_bound=30,
        )
        + ' '
        + summary_text.SCENARIO_PLAN_INSIGHTS_NEW_BUDGET.format(
            start_date='2020-01-05', end_date='2020-06-28'
        ),
    )

  def test_output_scenario_plan_card_custom_spend_constraint_upper(self):
    optimization_results = dataclasses.replace(
        self.optimization_results,
        spend_bounds=(np.array([0.7, 0.6, 0.7, 0.6, 0.7]), np.array([1.3])),
    )
    summary_html_dom = self._get_output_summary_html_dom(optimization_results)
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(),
        summary_text.SCENARIO_PLAN_INSIGHTS_VARIED_SPEND_BOUNDS.format(
            scenario_type='fixed',
        )
        + ' '
        + summary_text.SCENARIO_PLAN_INSIGHTS_HISTORICAL_BUDGET.format(
            start_date='2020-01-05',
            end_date='2020-06-28',
        ),
    )

  def test_output_scenario_plan_card_custom_spend_constraint_lower(self):
    optimization_results = dataclasses.replace(
        self.optimization_results,
        spend_bounds=(np.array([0.7]), np.array([1.3, 1.4, 1.3, 1.4, 1.3])),
    )

    summary_html_dom = self._get_output_summary_html_dom(optimization_results)
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(),
        summary_text.SCENARIO_PLAN_INSIGHTS_VARIED_SPEND_BOUNDS.format(
            scenario_type='fixed',
        )
        + ' '
        + summary_text.SCENARIO_PLAN_INSIGHTS_HISTORICAL_BUDGET.format(
            start_date='2020-01-05',
            end_date='2020-06-28',
        ),
    )

  def test_output_scenario_card_use_cpik_no_revenue_per_kpi(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results_kpi_output
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    stats_section = analysis_test_utils.get_child_element(card, 'stats-section')
    stats = stats_section.findall('stats')
    self.assertLen(stats, 6)

    title = analysis_test_utils.get_child_element(stats[0], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Non-optimized budget')
    title = analysis_test_utils.get_child_element(stats[1], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized budget')
    title = analysis_test_utils.get_child_element(stats[2], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Non-optimized CPIK')
    title = analysis_test_utils.get_child_element(stats[3], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized CPIK')
    title = analysis_test_utils.get_child_element(stats[4], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Non-optimized incremental KPI')
    title = analysis_test_utils.get_child_element(stats[5], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized incremental KPI')

  @parameterized.named_parameters(
      (
          'non_optimized_budget',
          0,
          summary_text.NON_OPTIMIZED_BUDGET_LABEL,
          '$600',
          None,
      ),
      (
          'optimization_results',
          1,
          summary_text.OPTIMIZED_BUDGET_LABEL,
          '$600',
          '$0',
      ),
      (
          'non_optimized_roi',
          2,
          summary_text.NON_OPTIMIZED_ROI_LABEL,
          '1.3',
          None,
      ),
      ('optimized_roi', 3, summary_text.OPTIMIZED_ROI_LABEL, '1.4', '+0.1'),
      (
          'non_optimized_inc_outcome',
          4,
          summary_text.NON_OPTIMIZED_INC_OUTCOME_LABEL.format(
              outcome=c.REVENUE
          ),
          '$760',
          None,
      ),
      (
          'optimized_inc_outcome',
          5,
          summary_text.OPTIMIZED_INC_OUTCOME_LABEL.format(outcome=c.REVENUE),
          '$830',
          '+$70',
      ),
  )
  def test_output_scenario_plan_card_stats_text(
      self, index, expected_title, expected_stat, expected_delta
  ):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.SCENARIO_PLAN_CARD_ID},
    )
    stats_section = analysis_test_utils.get_child_element(card, 'stats-section')
    stats = stats_section.findall('stats')
    self.assertLen(stats, 6)

    title = analysis_test_utils.get_child_element(
        stats[index], 'stats-title'
    ).text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), expected_title)
    stat = analysis_test_utils.get_child_element(stats[index], 'stat').text
    self.assertIsNotNone(stat)
    self.assertEqual(stat.strip(), expected_stat)
    if expected_delta:
      delta = analysis_test_utils.get_child_element(stats[index], 'delta').text
      self.assertIsNotNone(delta)
      self.assertEqual(delta.strip(), expected_delta)

  def test_output_budget_allocation_card_text(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.BUDGET_ALLOCATION_CARD_ID},
    )
    card_title_text = analysis_test_utils.get_child_element(
        card, 'card-title'
    ).text
    self.assertIsNotNone(card_title_text)
    self.assertEqual(
        card_title_text.strip(), summary_text.BUDGET_ALLOCATION_CARD_TITLE
    )

    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(), summary_text.BUDGET_ALLOCATION_INSIGHTS
    )

  def test_output_budget_allocation_charts(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    self.mock_spend_delta.assert_called_once()
    self.mock_budget_allocation.assert_called_once()
    self.mock_outcome_delta.assert_called_once()

    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.BUDGET_ALLOCATION_CARD_ID},
    )
    charts = card.findall('charts/chart')
    self.assertLen(charts, 3)

    spend_delta_description_text = analysis_test_utils.get_child_element(
        charts[0], 'chart-description'
    ).text
    self.assertIsNotNone(spend_delta_description_text)
    self.assertEqual(
        spend_delta_description_text.strip(),
        summary_text.SPEND_DELTA_CHART_INSIGHTS,
    )

    outcome_delta_description_text = analysis_test_utils.get_child_element(
        charts[2], 'chart-description'
    ).text
    self.assertIsNotNone(outcome_delta_description_text)
    self.assertEqual(
        outcome_delta_description_text.strip(),
        summary_text.OUTCOME_DELTA_CHART_INSIGHTS_FORMAT.format(
            outcome=c.REVENUE
        ),
    )

  def test_output_budget_allocation_table(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.BUDGET_ALLOCATION_CARD_ID},
    )
    chart_table = analysis_test_utils.get_child_element(
        card, 'charts/chart-table'
    )
    self.assertEqual(
        chart_table.attrib['id'], summary_text.SPEND_ALLOCATION_TABLE_ID
    )
    title_text = analysis_test_utils.get_child_element(
        chart_table, 'div', attribs={'class': 'chart-table-title'}
    ).text
    self.assertEqual(title_text, summary_text.SPEND_ALLOCATION_CHART_TITLE)

    table = analysis_test_utils.get_child_element(chart_table, 'div/table')
    header_row = analysis_test_utils.get_child_element(
        table, 'tr', attribs={'class': 'chart-table-column-headers'}
    )
    header_values = analysis_test_utils.get_table_row_values(header_row, 'th')
    self.assertSequenceEqual(
        header_values,
        [
            summary_text.CHANNEL_LABEL,
            summary_text.NONOPTIMIZED_SPEND_LABEL,
            summary_text.OPTIMIZED_SPEND_LABEL,
        ],
    )

    value_rows = table.findall('tr')[1:]
    self.assertLen(value_rows, 3)  # Equal to the number of channels.
    row1 = analysis_test_utils.get_table_row_values(value_rows[0])
    row2 = analysis_test_utils.get_table_row_values(value_rows[1])
    row3 = analysis_test_utils.get_table_row_values(value_rows[2])
    self.assertLen(row1, 3)  # Equal to the number of columns.
    self.assertEqual(row1, ['channel 2', '50%', '40%'])
    self.assertEqual(row2, ['channel 0', '33%', '37%'])
    self.assertEqual(row3, ['channel 1', '17%', '23%'])

  def test_output_response_curves_card_text(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_ID},
    )
    card_title_text = analysis_test_utils.get_child_element(
        card, 'card-title'
    ).text
    self.assertIsNotNone(card_title_text)
    self.assertEqual(
        card_title_text.strip(),
        summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_TITLE,
    )

    insights_text = analysis_test_utils.get_child_element(
        card, 'card-insights/p', {'class': 'insights-text'}
    ).text
    self.assertIsNotNone(insights_text)
    self.assertEqual(
        insights_text.strip(),
        summary_text.OPTIMIZED_RESPONSE_CURVES_INSIGHTS_FORMAT.format(
            outcome=c.REVENUE
        ),
    )

  def test_output_response_curves_chart(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.optimization_results
    )
    self.mock_response_curves.assert_called_once_with(n_top_channels=3)

    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_ID},
    )

    charts = card.findall('charts/chart')
    self.assertLen(charts, 1)


class OptimizerHelperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'tolerance_one_hundredth',
          'budget': 10_000,
          'gtol': 0.000001,
          'expected_round_factor': 0,
      },
      {
          'testcase_name': 'tolerance_one_tenth',
          'budget': 10_000,
          'gtol': 0.00001,
          'expected_round_factor': 0,
      },
      {
          'testcase_name': 'tolerance_1',
          'budget': 10_000,
          'gtol': 0.0001,
          'expected_round_factor': -1,
      },
      {
          'testcase_name': 'tolerance_10',
          'budget': 10_000,
          'gtol': 0.001,
          'expected_round_factor': -2,
      },
      {
          'testcase_name': 'tolerance_100',
          'budget': 10_000,
          'gtol': 0.01,
          'expected_round_factor': -3,
      },
  )
  def test_round_factor_correct(
      self,
      budget,
      gtol,
      expected_round_factor,
  ):
    round_factor = optimizer._get_round_factor(budget, gtol)
    self.assertEqual(round_factor, expected_round_factor)

  @parameterized.named_parameters(
      {
          'testcase_name': 'gtol_equals_one',
          'budget': 10_000,
          'gtol': 1.0,
          'error_message': 'gtol must be less than one.',
      },
      {
          'testcase_name': 'gtol_greater_than_one',
          'budget': 10_000,
          'gtol': 1.5,
          'error_message': 'gtol must be less than one.',
      },
      {
          'testcase_name': 'zero_budget',
          'budget': 0.0,
          'gtol': 0.01,
          'error_message': '`budget` must be greater than zero.',
      },
      {
          'testcase_name': 'negative_budget',
          'budget': -10_000.0,
          'gtol': 0.01,
          'error_message': '`budget` must be greater than zero.',
      },
  )
  def test_round_factor_raises_error(
      self,
      budget,
      gtol,
      error_message,
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      optimizer._get_round_factor(budget, gtol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'fixed_budget_under_budget',
          'fixed_budget': True,
          'budget': 1_000.0,
          'spend': np.array([100, 100]),
          'incremental_outcome': np.array([100, 100]),
          'roi_grid_point': 0.0,
          'target_mroi': None,
          'target_roi': None,
          'expected_output': False,
      },
      {
          'testcase_name': 'fixed_budget_over_budget',
          'fixed_budget': True,
          'budget': 1_000.0,
          'spend': np.array([500, 600]),
          'incremental_outcome': np.array([100, 100]),
          'roi_grid_point': 0.0,
          'target_mroi': None,
          'target_roi': None,
          'expected_output': True,
      },
      {
          'testcase_name': 'fixed_budget_under_target_roi',
          'fixed_budget': False,
          'budget': 1_000.0,
          'spend': np.array([500, 500]),
          'incremental_outcome': np.array([100, 100]),
          'roi_grid_point': 0.0,
          'target_mroi': None,
          'target_roi': 2.0,
          'expected_output': True,
      },
      {
          'testcase_name': 'fixed_budget_over_target_roi',
          'fixed_budget': False,
          'budget': 1_000.0,
          'spend': np.array([500, 500]),
          'incremental_outcome': np.array([1_000, 1_000]),
          'roi_grid_point': 0.0,
          'target_mroi': None,
          'target_roi': 2.0,
          'expected_output': False,
      },
      {
          'testcase_name': 'fixed_budget_under_target_mroi',
          'fixed_budget': False,
          'budget': 1_000.0,
          'spend': np.array([500, 500]),
          'incremental_outcome': np.array([1_000, 1_000]),
          'roi_grid_point': 1.8,
          'target_mroi': 2.0,
          'target_roi': None,
          'expected_output': True,
      },
      {
          'testcase_name': 'fixed_budget_over_target_mroi',
          'fixed_budget': False,
          'budget': 1_000.0,
          'spend': np.array([500, 500]),
          'incremental_outcome': np.array([1_000, 1_000]),
          'roi_grid_point': 2.1,
          'target_mroi': 2.0,
          'target_roi': None,
          'expected_output': False,
      },
  )
  def test_exceeds_optimization_constraints(
      self,
      fixed_budget,
      budget,
      spend,
      incremental_outcome,
      roi_grid_point,
      target_mroi,
      target_roi,
      expected_output,
  ):
    exceeds = optimizer._exceeds_optimization_constraints(
        fixed_budget,
        budget,
        spend,
        incremental_outcome,
        roi_grid_point,
        target_mroi,
        target_roi,
    )
    self.assertEqual(exceeds, expected_output)


class OptimizerKPITest(parameterized.TestCase):

  def setUp(self):
    super(OptimizerKPITest, self).setUp()
    # Input data resulting in KPI computation.
    self.input_data_media_and_rf_kpi = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    self.input_data_non_revenue_revenue_per_kpi = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    custom_model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Normal(0.0, 5.0, name=c.KNOT_VALUES),
            roi_m=tfp.distributions.LogNormal(0.2, 0.8, name=c.ROI_M),
            roi_rf=tfp.distributions.LogNormal(0.2, 0.8, name=c.ROI_RF),
        )
    )
    self.inference_data_media_and_rf_kpi = az.InferenceData(
        prior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_prior_media_and_rf.nc')
        ),
        posterior=xr.open_dataset(
            os.path.join(_TEST_DATA_DIR, 'sample_posterior_media_and_rf.nc')
        ),
    )
    self.meridian_media_and_rf_kpi = model.Meridian(
        input_data=self.input_data_media_and_rf_kpi,
        model_spec=custom_model_spec,
    )
    self.meridian_non_revenue_revenue_per_kpi = model.Meridian(
        input_data=self.input_data_non_revenue_revenue_per_kpi,
    )
    self.budget_optimizer_media_and_rf_kpi = optimizer.BudgetOptimizer(
        self.meridian_media_and_rf_kpi
    )
    self.budget_optimizer_non_revenue_revenue_per_kpi = (
        optimizer.BudgetOptimizer(self.meridian_non_revenue_revenue_per_kpi)
    )
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            'inference_data',
            new=property(
                lambda unused_self: self.inference_data_media_and_rf_kpi
            ),
        )
    )
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'summary_metrics',
            return_value=analysis_test_utils.generate_paid_summary_metrics(),
        )
    )

  @parameterized.parameters([True, False])
  def test_incremental_outcome_called_correct_optimize(
      self, use_posterior: bool
  ):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf_kpi._analyzer,
            'incremental_outcome',
            autospec=True,
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            )),
        )
    )

    self.budget_optimizer_media_and_rf_kpi.optimize(
        use_posterior=use_posterior, use_kpi=True
    )

    mock_incremental_outcome.assert_called_with(
        # marginal roi computation in the analyzer transitively calls
        # incremental_outcome() with the following arguments.
        selected_geos=None,
        selected_times=None,
        aggregate_geos=True,
        aggregate_times=True,
        inverse_transform_outcome=True,
        # Note that the above arguments also happen to be their default values.
        # All other direct incremental_outcome() calls use the following args.
        use_kpi=True,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_posterior=use_posterior,
        new_data=mock.ANY,
        include_non_paid_channels=False,
    )

  def test_results_kpi_only(self):
    optimization_results = self.budget_optimizer_media_and_rf_kpi.optimize(
        use_kpi=True
    )
    for var in (c.ROI, c.MROI, c.CPIK, c.EFFECTIVENESS):
      self.assertIsNotNone(optimization_results.optimized_data[var])
      self.assertIsNotNone(optimization_results.nonoptimized_data[var])
      self.assertIsNotNone(
          optimization_results.nonoptimized_data_with_optimal_freq[var]
      )
    for attr in (c.TOTAL_ROI, c.TOTAL_CPIK):
      self.assertIsNotNone(optimization_results.optimized_data.attrs[attr])
      self.assertIsNotNone(optimization_results.nonoptimized_data.attrs[attr])
      self.assertIsNotNone(
          optimization_results.nonoptimized_data_with_optimal_freq.attrs[attr]
      )
    self.assertFalse(
        optimization_results.optimized_data.attrs[c.IS_REVENUE_KPI]
    )
    self.assertFalse(
        optimization_results.nonoptimized_data.attrs[c.IS_REVENUE_KPI]
    )
    self.assertFalse(
        optimization_results.nonoptimized_data_with_optimal_freq.attrs[
            c.IS_REVENUE_KPI
        ]
    )

  @parameterized.parameters([True, False])
  def test_use_kpi_sets_is_revenue_kpi(self, use_kpi: bool):
    optimization_results = (
        self.budget_optimizer_non_revenue_revenue_per_kpi.optimize(
            use_kpi=use_kpi
        )
    )

    self.assertEqual(
        optimization_results.optimized_data.attrs[c.IS_REVENUE_KPI], not use_kpi
    )
    self.assertEqual(
        optimization_results.nonoptimized_data.attrs[c.IS_REVENUE_KPI],
        not use_kpi,
    )
    self.assertEqual(
        optimization_results.nonoptimized_data_with_optimal_freq.attrs[
            c.IS_REVENUE_KPI
        ],
        not use_kpi,
    )

  def test_optimize_no_use_kpi_no_revenue_per_kpi_raises_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Revenue analysis is not available when `revenue_per_kpi` is unknown.'
        ' Set `use_kpi=True` to perform KPI analysis instead.',
    ):
      self.budget_optimizer_media_and_rf_kpi.optimize(use_kpi=False)


if __name__ == '__main__':
  absltest.main()
