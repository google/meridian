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

from collections.abc import Mapping
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

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'model', 'test_data'
)
_NONOPTIMIZED_INCREMENTAL_IMPACT = np.array(
    [335.65, 531.11, 791.67, 580.1, 242.4]
)
_NONOPTIMIZED_SPEND = np.array([325.0, 311.0, 287.0, 307.0, 322.0])
_OPTIMIZED_INCREMENTAL_IMPACT = np.array(
    [528.6618, 648.44446, 427.40665, 1178.5515, 888.9901]
)
_OPTIMIZED_SPEND = np.array([257.0, 247.0, 230.0, 399.0, 419.0])
_OPTIMIZED_MEDIA_ONLY_SPEND = np.array([320.0, 310.0, 293.0])
_OPTIMIZED_RF_ONLY_SPEND = np.array([399.0, 230.0])
_TARGET_MROI_SPEND = np.array([0.0, 0.0, 0.0, 614.0, 644.0])
_TARGET_ROI_SPEND = np.array([650.0, 622.0, 574.0, 614.0, 644.0])
# Currently zero due to numerator being zero with mocked constant inc_impact
_BUDGET_MROI = np.array([0, 0, 0, 0, 0])

_N_GEOS = 5
_N_TIMES = 55  # 49 in the current sample datasets.
_N_MEDIA_TIMES = 58  # 52 in the current sample datasets.
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2
_N_CONTROLS = 2
_N_CHAINS = 1
_N_DRAWS = 1


def _create_budget_data(
    spend: np.ndarray,
    inc_impact: np.ndarray,
    mroi: np.ndarray | None = None,
    channels: np.ndarray | None = None,
    attrs: Mapping[str, Any] | None = None,
    use_kpi: bool = False,
) -> xr.Dataset:
  channels = (
      [f'channel {i}' for i in range(len(spend))]
      if channels is None
      else channels
  )
  data_vars = {
      c.SPEND: ([c.CHANNEL], spend),
      c.PCT_OF_SPEND: ([c.CHANNEL], spend / sum(spend)),
      c.INCREMENTAL_IMPACT: ([c.CHANNEL], inc_impact),
  }
  attributes = {
      c.START_DATE: '2020-01-05',
      c.END_DATE: '2020-06-28',
      c.BUDGET: sum(spend),
      c.PROFIT: sum(inc_impact) - sum(spend),
      c.TOTAL_INCREMENTAL_IMPACT: sum(inc_impact),
  }
  if use_kpi:
    data_vars[c.CPIK] = ([c.CHANNEL], spend / inc_impact)
    attributes[c.TOTAL_CPIK] = sum(spend) / sum(inc_impact)
  else:
    data_vars[c.ROI] = ([c.CHANNEL], inc_impact / spend)
    data_vars[c.MROI] = ([c.CHANNEL], mroi)
    attributes[c.TOTAL_ROI] = sum(inc_impact) / sum(spend)
  return xr.Dataset(
      data_vars=data_vars,
      coords={
          c.CHANNEL: ([c.CHANNEL], channels),
      },
      attrs=attributes | (attrs or {}),
  )


def _verify_actual_vs_expected_budget_data(
    actual_data: xr.Dataset, expected_data: xr.Dataset
):
  xr.testing.assert_allclose(actual_data, expected_data, atol=0.1)
  np.testing.assert_allclose(actual_data.budget, expected_data.budget, atol=0.1)
  np.testing.assert_allclose(actual_data.profit, expected_data.profit, atol=0.1)
  np.testing.assert_allclose(
      actual_data.total_incremental_impact,
      expected_data.total_incremental_impact,
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
    inc_impact=np.array([280, 150, 330]),
    mroi=np.array([1.2, 1.3, 1.4]),
)
_SAMPLE_OPTIMIZED_DATA = _create_budget_data(
    spend=np.array([220, 140, 240]),
    inc_impact=np.array([350, 210, 270]),
    mroi=np.array([1.4, 1.5, 1.6]),
    attrs={c.FIXED_BUDGET: True},
)
_SAMPLE_NON_OPTIMIZED_DATA_KPI = _create_budget_data(
    spend=np.array([200, 100, 300]),
    inc_impact=np.array([280, 150, 330]),
    use_kpi=True,
)
_SAMPLE_OPTIMIZED_DATA_KPI = _create_budget_data(
    spend=np.array([220, 140, 240]),
    inc_impact=np.array([350, 210, 270]),
    use_kpi=True,
    attrs={c.FIXED_BUDGET: True},
)


class OptimizerAlgorithmTest(parameterized.TestCase):
  # TODO(b/302713435): Update the sample datasets to span over 1 year.
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
    self.meridian_media_and_rf = model.Meridian(
        input_data=self.input_data_media_and_rf
    )
    self.meridian_media_only = model.Meridian(
        input_data=self.input_data_media_only
    )
    self.meridian_rf_only = model.Meridian(input_data=self.input_data_rf_only)
    self.budget_optimizer_media_and_rf = optimizer.BudgetOptimizer(
        self.meridian_media_and_rf
    )
    self.budget_optimizer_media_only = optimizer.BudgetOptimizer(
        self.meridian_media_only
    )
    self.budget_optimizer_rf_only = optimizer.BudgetOptimizer(
        self.meridian_rf_only
    )
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            'inference_data',
            new=property(lambda unused_self: self.inference_data_media_and_rf),
        )
    )

  def test_not_fitted_meridian_model_raises_exception(self):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    not_fitted_mmm.inference_data = az.InferenceData()
    budget_optimizer = optimizer.BudgetOptimizer(not_fitted_mmm)
    with self.assertRaisesRegex(
        model.NotFittedModelError,
        'Running budget optimization scenarios requires fitting the model.',
    ):
      budget_optimizer.optimize()

  def test_incorrect_selected_times_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`selected_times` should match the time dimensions from '
        'meridian.input_data.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          selected_times=('2020-01-01', '2021-01-01')
      )

  def test_fixed_budget_target_roi_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        '`target_roi` is only used for flexible budget scenarios.',
    ):
      self.budget_optimizer_media_and_rf.optimize(
          fixed_budget=True, target_roi=1.2
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

  def test_accessing_data_raises_exception_before_running_optimize(self):
    with self.assertRaisesRegex(
        optimizer.OptimizationNotRunError,
        'Non-optimized data is only available after running optimize().',
    ):
      _ = self.budget_optimizer_media_and_rf.nonoptimized_data
    with self.assertRaisesRegex(
        optimizer.OptimizationNotRunError,
        'Optimized data is only available after running optimize().',
    ):
      _ = self.budget_optimizer_media_and_rf.optimized_data

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_default_selected_times_all_times(
      self,
      mock_incremental_impact,
  ):
    mock_incremental_impact.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    self.budget_optimizer_media_and_rf.optimize()
    expected_times = self.input_data_media_and_rf.time.values.tolist()
    self.assertEqual(
        self.budget_optimizer_media_and_rf.optimized_data.start_date,
        expected_times[0],
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.optimized_data.end_date,
        expected_times[-1],
    )
    _, mock_kwargs = mock_incremental_impact.call_args
    self.assertIsNone(mock_kwargs['selected_times'])

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_selected_times_all_times(self, mock_incremental_impact):
    mock_incremental_impact.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    expected_times = self.input_data_media_and_rf.time.values.tolist()
    self.budget_optimizer_media_and_rf.optimize(
        selected_times=(expected_times[0], expected_times[-1])
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.optimized_data.start_date,
        expected_times[0],
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.optimized_data.end_date,
        expected_times[-1],
    )
    _, mock_kwargs = mock_incremental_impact.call_args
    self.assertIsNone(mock_kwargs['selected_times'])

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_selected_times_used_correctly(self, mock_incremental_impact):
    mock_incremental_impact.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    selected_time_dims = [
        '2021-05-17',
        '2021-05-24',
        '2021-05-31',
        '2021-06-07',
        '2021-06-14',
    ]
    selected_time = ('2021-05-17', '2021-06-14')
    self.budget_optimizer_media_and_rf.optimize(selected_times=selected_time)
    _, mock_kwargs = mock_incremental_impact.call_args
    self.assertEqual(mock_kwargs['selected_times'], selected_time_dims)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_selected_times_reversed_order(
      self,
      mock_incremental_impact,
  ):
    mock_incremental_impact.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    selected_time_dims = [
        '2021-05-17',
        '2021-05-24',
        '2021-05-31',
        '2021-06-07',
        '2021-06-14',
    ]
    selected_time = ('2021-06-14', '2021-05-17')
    self.budget_optimizer_media_and_rf.optimize(selected_times=selected_time)
    self.assertEqual(
        selected_time_dims,
        self.budget_optimizer_media_and_rf._get_selected_time_dims(
            selected_times=selected_time
        ),
    )

  def test_default_hist_spend_with_time_geo_dims(self):
    expected_spend = np.round(
        np.sum(self.meridian_media_and_rf.total_spend, axis=(0, 1))
    )
    self.budget_optimizer_media_and_rf.optimize()
    self.assertEqual(self.meridian_media_and_rf.total_spend.ndim, 3)
    np.testing.assert_array_equal(
        self.budget_optimizer_media_and_rf.nonoptimized_data.spend,
        expected_spend,
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.nonoptimized_data.budget,
        np.sum(expected_spend),
    )

  def test_spend_ratio(self):
    self.budget_optimizer_media_and_rf.optimize()
    np.testing.assert_array_equal(
        self.budget_optimizer_media_and_rf._spend_ratio, [1, 1, 1, 1, 1]
    )

  def test_hist_spend_with_imputed_cpm(self):
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'incremental_impact',
            autospec=True,
            spec_set=True,
            return_value=tf.convert_to_tensor(
                [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
            ),
        )
    )
    self.enter_context(
        mock.patch.object(
            input_data.InputData,
            'get_total_spend',
            autospec=True,
            spec_set=True,
            return_value=tf.convert_to_tensor(
                [120.0, 150.0, 652.0, 645.0, 654.0],
                dtype=tf.float32,
            ),
        )
    )
    self.budget_optimizer_media_and_rf.optimize(
        selected_times=('2021-01-25', '2021-03-08')
    )
    expected_spend = [18.0, 20.0, 81.0, 81.0, 75.0]
    self.assertEqual(self.meridian_media_and_rf.total_spend.ndim, 1)
    np.testing.assert_array_equal(
        self.budget_optimizer_media_and_rf.nonoptimized_data.spend,
        expected_spend,
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.nonoptimized_data.budget,
        np.sum(expected_spend),
    )

  def test_incremental_impact_tensors(self):
    spend = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    hist_spend = np.array([350, 400, 200, 50, 500], dtype=np.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self.budget_optimizer_media_and_rf._get_incremental_impact_tensors(
            hist_spend, spend
        )
    )
    expected_media = self.meridian_media_and_rf.media * tf.math.divide_no_nan(
        new_media_spend, hist_spend[:_N_MEDIA_CHANNELS]
    )
    expected_media_spend = spend[:_N_MEDIA_CHANNELS]
    expected_reach = self.meridian_media_and_rf.reach * tf.math.divide_no_nan(
        new_rf_spend, hist_spend[-_N_RF_CHANNELS:]
    )
    expected_frequency = self.meridian_media_and_rf.frequency
    expected_rf_spend = spend[-_N_RF_CHANNELS:]
    np.testing.assert_allclose(new_media, expected_media)
    np.testing.assert_allclose(new_media_spend, expected_media_spend)
    np.testing.assert_allclose(new_reach, expected_reach)
    np.testing.assert_allclose(new_frequency, expected_frequency)
    np.testing.assert_allclose(new_rf_spend, expected_rf_spend)

  def test_incremental_impact_tensors_with_optimal_frequency(self):
    spend = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    hist_spend = np.array([350, 400, 200, 50, 500], dtype=np.float32)
    optimal_frequency = np.array([2, 2], dtype=np.float32)
    (new_media, new_media_spend, new_reach, new_frequency, new_rf_spend) = (
        self.budget_optimizer_media_and_rf._get_incremental_impact_tensors(
            hist_spend=hist_spend,
            spend=spend,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_media = self.meridian_media_and_rf.media * tf.math.divide_no_nan(
        new_media_spend, hist_spend[:_N_MEDIA_CHANNELS]
    )
    expected_media_spend = spend[:_N_MEDIA_CHANNELS]
    rf_media = (
        self.meridian_media_and_rf.reach * self.meridian_media_and_rf.frequency
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
        tf.ones_like(self.meridian_media_and_rf.frequency) * optimal_frequency
    )
    expected_rf_spend = spend[-_N_RF_CHANNELS:]
    np.testing.assert_allclose(new_media, expected_media)
    np.testing.assert_allclose(new_media_spend, expected_media_spend)
    np.testing.assert_allclose(new_reach, expected_reach)
    np.testing.assert_allclose(new_frequency, expected_frequency)
    np.testing.assert_allclose(new_rf_spend, expected_rf_spend)

  def test_nonoptimized_data_before_optimization_raises_error(self):
    with self.assertRaisesWithLiteralMatch(
        optimizer.OptimizationNotRunError,
        'Non-optimized data is only available after running optimize().',
    ):
      _ = self.budget_optimizer_media_and_rf.nonoptimized_data

  def test_optimized_data_before_optimization_raises_error(self):
    with self.assertRaisesWithLiteralMatch(
        optimizer.OptimizationNotRunError,
        'Optimized data is only available after running optimize().',
    ):
      _ = self.budget_optimizer_media_and_rf.optimized_data

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_nonoptimized_data_with_defaults_media_and_rf(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND,
        inc_impact=_NONOPTIMIZED_INCREMENTAL_IMPACT,
        mroi=_BUDGET_MROI,
        channels=self.input_data_media_and_rf.get_all_channels(),
    )
    self.budget_optimizer_media_and_rf.optimize()
    actual_data = self.budget_optimizer_media_and_rf.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_nonoptimized_data_with_defaults_media_only(
      self,
      mock_incremental_impact,
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT[:_N_MEDIA_CHANNELS]]],
        tf.float32,
    )
    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND[:_N_MEDIA_CHANNELS],
        inc_impact=_NONOPTIMIZED_INCREMENTAL_IMPACT[:_N_MEDIA_CHANNELS],
        mroi=_BUDGET_MROI[:_N_MEDIA_CHANNELS],
        channels=self.input_data_media_only.get_all_channels(),
    )
    self.budget_optimizer_media_only.optimize()
    actual_data = self.budget_optimizer_media_only.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_nonoptimized_data_with_defaults_rf_only(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT[-_N_RF_CHANNELS:]]],
        tf.float32,
    )
    expected_data = _create_budget_data(
        spend=_NONOPTIMIZED_SPEND[-_N_RF_CHANNELS:],
        inc_impact=_NONOPTIMIZED_INCREMENTAL_IMPACT[-_N_RF_CHANNELS:],
        mroi=_BUDGET_MROI[-_N_RF_CHANNELS:],
        channels=self.input_data_rf_only.get_all_channels(),
    )
    self.budget_optimizer_rf_only.optimize()
    actual_data = self.budget_optimizer_rf_only.nonoptimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimized_data_with_defaults_media_and_rf(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_SPEND,
        inc_impact=_OPTIMIZED_INCREMENTAL_IMPACT,
        mroi=_BUDGET_MROI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )
    self.budget_optimizer_media_and_rf.optimize()
    actual_data = self.budget_optimizer_media_and_rf.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        self.budget_optimizer_media_and_rf.nonoptimized_data.budget,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimized_data_with_defaults_media_only(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT[:_N_MEDIA_CHANNELS]]],
        tf.float32,
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_MEDIA_ONLY_SPEND,
        inc_impact=_OPTIMIZED_INCREMENTAL_IMPACT[:_N_MEDIA_CHANNELS],
        mroi=_BUDGET_MROI[:_N_MEDIA_CHANNELS],
        channels=self.input_data_media_only.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )
    self.budget_optimizer_media_only.optimize()
    actual_data = self.budget_optimizer_media_only.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        self.budget_optimizer_media_only.nonoptimized_data.budget,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimized_data_with_defaults_rf_only(self, mock_incremental_impact):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT[-_N_RF_CHANNELS:]]],
        tf.float32,
    )
    expected_data = _create_budget_data(
        spend=_OPTIMIZED_RF_ONLY_SPEND,
        inc_impact=_OPTIMIZED_INCREMENTAL_IMPACT[-_N_RF_CHANNELS:],
        mroi=_BUDGET_MROI[-self.meridian_rf_only.n_rf_channels :],
        channels=self.input_data_rf_only.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )
    self.budget_optimizer_rf_only.optimize()
    actual_data = self.budget_optimizer_rf_only.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)
    self.assertEqual(
        actual_data.budget,
        self.budget_optimizer_rf_only.nonoptimized_data.budget,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimized_data_with_target_mroi(self, mock_incremental_impact):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_TARGET_MROI_SPEND,
        inc_impact=_OPTIMIZED_INCREMENTAL_IMPACT,
        mroi=_BUDGET_MROI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: False, c.TARGET_MROI: 1},
    )
    self.budget_optimizer_media_and_rf.optimize(
        fixed_budget=False, target_mroi=1
    )
    actual_data = self.budget_optimizer_media_and_rf.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimized_data_with_target_roi(self, mock_incremental_impact):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    expected_data = _create_budget_data(
        spend=_TARGET_ROI_SPEND,
        inc_impact=_OPTIMIZED_INCREMENTAL_IMPACT,
        mroi=_BUDGET_MROI,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: False, c.TARGET_ROI: 1},
    )
    self.budget_optimizer_media_and_rf.optimize(
        fixed_budget=False, target_roi=1
    )
    actual_data = self.budget_optimizer_media_and_rf.optimized_data
    _verify_actual_vs_expected_budget_data(actual_data, expected_data)

  @parameterized.named_parameters(
      dict(
          testcase_name='optimal_frequency',
          expected_incremental_impact=np.array(
              [544.2629, 1122.9128, 411.21844, 1196.8931, 1037.8674]
          ),
          expected_spend=np.array([227.0, 306.0, 201.0, 399.0, 419.0]),
          expected_mroi=np.array([1.329, 1.94, 1.156, 2.928, 2.44]),
          use_optimal_frequency=True,
      ),
      dict(
          testcase_name='historical_frequency',
          expected_incremental_impact=np.array(
              [572.5, 1291.739, 411.2, 664.9, 369.8]
          ),
          expected_spend=np.array([249.0, 404.0, 201.0, 399.0, 299.0]),
          expected_mroi=np.array([1.231, 1.519, 1.156, 1.666, 1.237]),
          use_optimal_frequency=False,
      ),
  )
  def test_optimized_data_use_optimal_frequency(
      self,
      expected_incremental_impact,
      expected_spend,
      expected_mroi,
      use_optimal_frequency,
  ):
    expected_data = _create_budget_data(
        spend=expected_spend,
        inc_impact=expected_incremental_impact,
        mroi=expected_mroi,
        channels=self.input_data_media_and_rf.get_all_channels(),
        attrs={c.FIXED_BUDGET: True},
    )
    self.budget_optimizer_media_and_rf.optimize(
        use_optimal_frequency=use_optimal_frequency
    )
    actual_data = self.budget_optimizer_media_and_rf.optimized_data
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
    (lower_bound, upper_bound) = (
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

  def test_create_grids_media_and_rf_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_media_and_rf._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
        )
    )
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
    expected_incremental_impact_grid = np.array(
        [
            [1.0, 1.0, 1.0, 0.66666667, 0.81818182],
            [1.0, 1.0, 1.0, 0.83333333, 1.0],
            [1.0, 1.0, 1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
        ],
    )
    mock_incremental_impact.assert_called_with(
        new_media=mock.ANY,
        new_reach=mock.ANY,
        new_frequency=mock.ANY,
        selected_times=selected_times,
        use_kpi=False,
        batch_size=c.DEFAULT_BATCH_SIZE,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_impact.call_args
    np.testing.assert_allclose(
        mock_kwargs['new_frequency'], self.meridian_media_and_rf.frequency
    )
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_create_grids_media_only_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_only._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_media_only._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0],
            [700.0, 800.0, 900.0],
            [900.0, 1000.0, 1100.0],
            [1100.0, 1200.0, 1300.0],
            [1300.0, 1400.0, np.nan],
            [1500.0, np.nan, np.nan],
        ],
    )
    expected_incremental_impact_grid = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    mock_incremental_impact.assert_called_with(
        new_media=mock.ANY,
        new_reach=None,
        new_frequency=None,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
    )
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_create_grids_rf_only_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_rf_only._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_rf_only._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0],
            [700.0, 800.0],
            [900.0, 1000.0],
            [1100.0, 1200.0],
            [1300.0, 1400.0],
            [1500.0, np.nan],
        ],
    )
    expected_incremental_impact_grid = np.array(
        [
            [0.33333333, 0.42857143],
            [0.46666667, 0.57142857],
            [0.6, 0.71428571],
            [0.73333333, 0.85714286],
            [0.86666667, 1.0],
            [1.0, np.nan],
        ],
    )
    mock_incremental_impact.assert_called_with(
        new_media=None,
        new_reach=mock.ANY,
        new_frequency=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_impact.call_args
    np.testing.assert_allclose(
        mock_kwargs['new_frequency'], self.meridian_media_and_rf.frequency
    )
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_create_grids_with_optimal_frequency_media_and_rf_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])
    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_media_and_rf._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
            optimal_frequency=optimal_frequency,
        )
    )
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
    expected_incremental_impact_grid = np.array(
        [
            [1.0, 1.0, 1.0, 0.66666667, 0.81818182],
            [1.0, 1.0, 1.0, 0.83333333, 1.0],
            [1.0, 1.0, 1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, np.nan, np.nan],
        ],
    )
    new_frequency = (
        tf.ones_like(self.meridian_media_and_rf.frequency) * optimal_frequency
    )
    mock_incremental_impact.assert_called_with(
        new_media=mock.ANY,
        new_reach=mock.ANY,
        new_frequency=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_impact.call_args
    np.testing.assert_allclose(mock_kwargs['new_frequency'], new_frequency)
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_create_grids_with_optimal_frequency_media_only_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_only._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])

    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_media_only._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0, 700.0],
            [700.0, 800.0, 900.0],
            [900.0, 1000.0, 1100.0],
            [1100.0, 1200.0, 1300.0],
            [1300.0, 1400.0, np.nan],
            [1500.0, np.nan, np.nan],
        ],
    )
    expected_incremental_impact_grid = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    )
    mock_incremental_impact.assert_called_with(
        new_media=mock.ANY,
        new_reach=None,
        new_frequency=None,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
    )
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  def test_create_grids_with_optimal_frequency_rf_only_correct(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_rf_only._analyzer,
            'incremental_impact',
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
    step_size = 200
    selected_times = ('2021-01-25', '2021-02-01')
    optimal_frequency = xr.DataArray(data=[2.5, 3.1])
    (spend_grid, incremental_impact_grid) = (
        self.budget_optimizer_rf_only._create_grids(
            spend=spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
            optimal_frequency=optimal_frequency,
        )
    )
    expected_spend_grid = np.array(
        [
            [500.0, 600.0],
            [700.0, 800.0],
            [900.0, 1000.0],
            [1100.0, 1200.0],
            [1300.0, 1400.0],
            [1500.0, np.nan],
        ],
    )
    expected_incremental_impact_grid = np.array(
        [
            [0.33333333, 0.42857143],
            [0.46666667, 0.57142857],
            [0.6, 0.71428571],
            [0.73333333, 0.85714286],
            [0.86666667, 1.0],
            [1.0, np.nan],
        ],
    )
    new_frequency = (
        tf.ones_like(self.meridian_media_and_rf.frequency) * optimal_frequency
    )
    mock_incremental_impact.assert_called_with(
        new_media=None,
        new_reach=mock.ANY,
        new_frequency=mock.ANY,
        selected_times=selected_times,
        batch_size=c.DEFAULT_BATCH_SIZE,
        use_kpi=False,
    )
    # Using `assert_called_with` doesn't work with array comparison.
    _, mock_kwargs = mock_incremental_impact.call_args
    np.testing.assert_allclose(mock_kwargs['new_frequency'], new_frequency)
    np.testing.assert_allclose(spend_grid, expected_spend_grid, equal_nan=True)
    np.testing.assert_allclose(
        incremental_impact_grid,
        expected_incremental_impact_grid,
        equal_nan=True,
        atol=0.01,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'default_budget_scenario',
          'fixed_budget': True,
          'target_roi': None,
          'target_mroi': None,
          'expected_optimal_spend': np.array([900, 800, 900, 1200, 1100]),
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
  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_grid_search_correct(
      self,
      mock_incremental_impact,
      fixed_budget,
      target_roi,
      target_mroi,
      expected_optimal_spend,
  ):
    mock_incremental_impact.return_value = tf.ones((
        _N_CHAINS,
        _N_DRAWS,
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    ))
    spend = np.array([1000, 1000, 1000, 1000, 1000])
    spend_bound_lower = np.array([500, 600, 700, 800, 900])
    spend_bound_upper = np.array([1500, 1400, 1300, 1200, 1100])
    step_size = 200
    selected_times = self.budget_optimizer_media_and_rf._get_selected_time_dims(
        selected_times=None
    )
    (spend_grid, incremental_impact_grid) = (
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
        incremental_impact_grid=incremental_impact_grid,
        budget=np.sum(spend),
        fixed_budget=fixed_budget,
        target_roi=target_roi,
        target_mroi=target_mroi,
    )
    np.testing.assert_array_equal(optimal_spend, expected_optimal_spend)

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_optimizer_budget_with_specified_budget(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_OPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    budget = 2000
    self.budget_optimizer_media_and_rf.optimize(
        budget=budget, fixed_budget=True
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.nonoptimized_data.budget,
        budget,
    )
    self.assertEqual(
        self.budget_optimizer_media_and_rf.optimized_data.budget,
        budget,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_budget_data_with_specified_pct_of_spend(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    expected_pct_of_spend = [0.1, 0.2, 0.3, 0.3, 0.1]
    self.budget_optimizer_media_and_rf.optimize(
        pct_of_spend=expected_pct_of_spend, fixed_budget=True
    )
    actual_spend = self.budget_optimizer_media_and_rf.nonoptimized_data.spend
    actual_budget = self.budget_optimizer_media_and_rf.nonoptimized_data.budget
    np.testing.assert_almost_equal(
        self.budget_optimizer_media_and_rf.nonoptimized_data.pct_of_spend,
        expected_pct_of_spend,
        decimal=7,
    )
    np.testing.assert_array_equal(
        self.budget_optimizer_media_and_rf.nonoptimized_data.budget,
        self.budget_optimizer_media_and_rf.optimized_data.budget,
    )
    np.testing.assert_almost_equal(
        expected_pct_of_spend,
        actual_spend / actual_budget,
        decimal=7,
    )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_budget_data_with_specified_pct_of_spend_size_incorrect(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Percent of spend must be specified for all channels.',
    ):
      pct_of_spend = np.array([0.1, 0.2, 0.3, 0.3])
      self.budget_optimizer_media_and_rf.optimize(
          pct_of_spend=pct_of_spend, fixed_budget=True
      )

  @mock.patch.object(analyzer.Analyzer, 'incremental_impact', autospec=True)
  def test_budget_data_with_specified_pct_of_spend_value_incorrect(
      self, mock_incremental_impact
  ):
    mock_incremental_impact.return_value = tf.convert_to_tensor(
        [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
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
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf._analyzer,
            'incremental_impact',
            autospec=True,
            return_value=tf.convert_to_tensor(
                [[_NONOPTIMIZED_INCREMENTAL_IMPACT]], tf.float32
            ),
        )
    )
    self.budget_optimizer_media_and_rf.optimize(batch_size=batch_size)
    _, mock_kwargs = mock_incremental_impact.call_args
    self.assertEqual(mock_kwargs['batch_size'], batch_size)

  def test_optimization_output_before_optimization_raises_error(self):
    with self.assertRaisesWithLiteralMatch(
        optimizer.OptimizationNotRunError,
        'Optimized data is only available after running optimize().',
    ):
      self.budget_optimizer_media_and_rf.output_optimization_summary(
          'test', 'test'
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

    self.budget_optimizer = optimizer.BudgetOptimizer(meridian)
    self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'nonoptimized_data',
            new=property(lambda unused_self: _SAMPLE_NON_OPTIMIZED_DATA),
        )
    )
    self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'nonoptimized_data_with_optimal_freq',
            new=property(lambda unused_self: _SAMPLE_NON_OPTIMIZED_DATA),
        )
    )
    self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'optimized_data',
            new=property(lambda unused_self: _SAMPLE_OPTIMIZED_DATA),
        )
    )
    self.budget_optimizer._spend_bounds = (
        np.array([0.7, 0.5, 0.7]),
        np.array([1.3]),
    )
    self.budget_optimizer._spend_ratio = np.array([1.0, 1.0, 1.0])
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

  def test_impact_waterfall_chart_data_correct(self):
    plot = self.budget_optimizer.plot_incremental_impact_delta()
    df = plot.data
    self.assertEqual(list(df.columns), [c.CHANNEL, c.INCREMENTAL_IMPACT])
    self.assertEqual(
        list(df.channel),
        ['current', 'channel 2', 'channel 0', 'channel 1', 'optimized'],
    )

  def test_impact_waterfall_chart_correct_config(self):
    plot = self.budget_optimizer.plot_incremental_impact_delta()
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

  def test_impact_waterfall_chart_correct_mark(self):
    plot = self.budget_optimizer.plot_incremental_impact_delta()
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

  def test_impact_waterfall_chart_correct_bar_encoding(self):
    plot = self.budget_optimizer.plot_incremental_impact_delta()
    encoding = plot.layer[0].encoding.to_dict()
    self.assertEqual(
        encoding['color'],
        {
            'condition': [
                {
                    'value': c.BLUE_500,
                    'test': (
                        f"datum.channel === '{c.CURRENT}' || datum.channel ==="
                        f" '{c.OPTIMIZED}'"
                    ),
                },
                {'value': c.RED_300, 'test': 'datum.incremental_impact < 0'},
            ],
            'value': c.CYAN_400,
        },
    )
    self.assertEqual(
        encoding['y'],
        {
            'axis': {
                'domain': False,
                'labelExpr': "replace(format(datum.value, '.3s'), 'G', 'B')",
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
    self.assertEqual(encoding['y2'], {'field': 'sum_impact'})
    self.assertEqual(
        encoding['tooltip'],
        [
            {'field': c.CHANNEL, 'type': 'nominal'},
            {'field': c.INCREMENTAL_IMPACT, 'type': 'quantitative'},
        ],
    )
    self.assertEqual(
        plot.title.text,
        summary_text.IMPACT_DELTA_CHART_TITLE.format(impact=c.REVENUE),
    )

  def test_impact_waterfall_chart_correct_text_encoding(self):
    plot = self.budget_optimizer.plot_incremental_impact_delta()
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
    plot = self.budget_optimizer.plot_budget_allocation()
    df = plot.data
    self.assertEqual(list(df.columns), [c.CHANNEL, c.SPEND])
    self.assertEqual(df.channel.size, 3)
    self.assertEqual(list(df.spend), list(_SAMPLE_OPTIMIZED_DATA.spend.values))

  def test_budget_allocation_nonoptimized_data(self):
    plot = self.budget_optimizer.plot_budget_allocation(optimized=False)
    df = plot.data
    self.assertEqual(
        list(df.spend), list(_SAMPLE_NON_OPTIMIZED_DATA.spend.values)
    )

  def test_budget_allocation_correct_encoding(self):
    plot = self.budget_optimizer.plot_budget_allocation()
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
    plot = self.budget_optimizer.plot_spend_delta()
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
    plot = self.budget_optimizer.plot_spend_delta()
    axis_config = plot.config.axis.to_dict()
    view_config = plot.config.view.to_dict()

    self.assertEqual(axis_config, formatter.TEXT_CONFIG)
    self.assertEqual(view_config, {'stroke': None})
    self.assertEqual(plot.width, formatter.bar_chart_width(5))  # n_channels + 2
    self.assertEqual(plot.height, 400)

  def test_plot_spend_delta_correct_mark(self):
    plot = self.budget_optimizer.plot_spend_delta()
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
    plot = self.budget_optimizer.plot_spend_delta()
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
                'labelExpr': formatter.compact_number_expr(n_sig_digits=1),
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
    plot = self.budget_optimizer.plot_spend_delta()
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

  def test_plot_response_curves_correct_data(self):
    plot = self.budget_optimizer.plot_response_curves()
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
        self.budget_optimizer._spend_bounds[0], df[c.LOWER_BOUND]
    )
    self.assertContainsSubset(
        self.budget_optimizer._spend_bounds[1], df[c.UPPER_BOUND]
    )

  def test_plot_response_curves_modified_bounds(self):
    self.budget_optimizer._spend_bounds = (
        np.array([0.7]),
        np.array([1.2, 1.3, 1.4]),
    )
    plot = self.budget_optimizer.plot_response_curves()
    df = plot.data
    self.assertContainsSubset(
        self.budget_optimizer._spend_bounds[0], df[c.LOWER_BOUND]
    )
    self.assertContainsSubset(
        self.budget_optimizer._spend_bounds[1], df[c.UPPER_BOUND]
    )

  def test_plot_response_curves_invalid_n_channels(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Top number of channels (5) by spend must be less than the total'
        ' number of channels (3)',
    ):
      self.budget_optimizer.plot_response_curves(5)

  def test_plot_response_curves_correct_selected_times(self):
    expected_times = analysis_test_utils.generate_selected_times(
        '2020-01-05', 26
    )
    self.budget_optimizer.plot_response_curves()
    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.mock_response_curves.call_args
    self.assertEqual(mock_kwargs['selected_times'], expected_times)

  def test_plot_response_curves_n_top_channels(self):
    plot = self.budget_optimizer.plot_response_curves(2)
    channels = list(plot.data.channel.unique())
    self.assertEqual(channels, ['channel 2', 'channel 0'])

  def test_plot_response_curves_upper_limit(self):
    self.budget_optimizer._spend_bounds = (np.array([0]), np.array([2]))
    self.budget_optimizer._spend_ratio = np.array([1.0, 1.0, 1.0])
    self.budget_optimizer.plot_response_curves()
    self.mock_response_curves.assert_called_once()
    _, mock_kwargs = self.mock_response_curves.call_args
    # Check that the spend multiplier max is the upper limit of the upper spend
    # bound + spend constraint padding.
    multiplier = np.arange(0, 2.5, 0.01)
    np.testing.assert_array_equal(mock_kwargs['spend_multipliers'], multiplier)
    self.assertEqual(mock_kwargs['by_reach'], True)

  def test_plot_response_curves_correct_config(self):
    plot = self.budget_optimizer.plot_response_curves()
    self.assertEqual(plot.config.axis.to_dict(), formatter.TEXT_CONFIG)

  def test_plot_response_curves_correct_facet_properties(self):
    plot = self.budget_optimizer.plot_response_curves()
    self.assertEqual(
        plot.facet.to_dict(),
        {'field': c.CHANNEL, 'title': None, 'type': 'nominal', 'sort': None},
    )
    self.assertLen(plot.spec.layer, 3)
    self.assertEqual(plot.columns, 3)
    self.assertEqual(plot.resolve.scale.x, 'independent')
    self.assertEqual(plot.resolve.scale.y, 'independent')

  def test_plot_response_curves_correct_encoding(self):
    plot = self.budget_optimizer.plot_response_curves()
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
    plot = self.budget_optimizer.plot_response_curves()
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
    plot = self.budget_optimizer.plot_response_curves()
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
    plot = self.budget_optimizer.plot_response_curves()
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

    self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'nonoptimized_data',
            new=property(lambda unused_self: _SAMPLE_NON_OPTIMIZED_DATA),
        )
    )
    self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'optimized_data',
            new=property(lambda unused_self: _SAMPLE_OPTIMIZED_DATA),
        )
    )
    self.budget_optimizer._spend_bounds = (np.array([0.7]), np.array([1.3]))
    self.budget_optimizer_kpi_output._spend_bounds = (
        np.array([0.7]),
        np.array([1.3]),
    )
    self.mock_spend_delta = self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'plot_spend_delta',
            return_value=self._mock_chart(),
        )
    )
    self.mock_budget_allocation = self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'plot_budget_allocation',
            return_value=self._mock_chart(),
        )
    )
    self.mock_impact_delta = self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'plot_incremental_impact_delta',
            return_value=self._mock_chart(),
        )
    )
    self.mock_response_curves = self.enter_context(
        mock.patch.object(
            optimizer.BudgetOptimizer,
            'plot_response_curves',
            return_value=self._mock_chart(),
        )
    )

  def test_output_html_title(self):
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
    title = summary_html_dom.find('head/title')

    self.assertIsNotNone(title)
    title_text = title.text
    self.assertIsNotNone(title_text)
    self.assertEqual(title_text.strip(), summary_text.OPTIMIZATION_TITLE)

  def test_output_header_section(self):
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
        summary_text.SCENARIO_PLAN_INSIGHTS_FORMAT.format(
            scenario_type='fixed',
            lower_bound=30,
            upper_bound=30,
            start_date='2020-01-05',
            end_date='2020-06-28',
        ),
    )

  def test_output_scenario_plan_card_custom_spend_constraint_upper(self):
    self.budget_optimizer._spend_bounds = (
        np.array([0.7, 0.6, 0.7, 0.6, 0.7]),
        np.array([1.3]),
    )
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
        summary_text.SCENARIO_PLAN_BASE_INSIGHTS_FORMAT.format(
            scenario_type='fixed',
            start_date='2020-01-05',
            end_date='2020-06-28',
        ),
    )

  def test_output_scenario_plan_card_custom_spend_constraint_lower(self):
    self.budget_optimizer._spend_bounds = (
        np.array([0.7]),
        np.array([1.3, 1.4, 1.3, 1.4, 1.3]),
    )
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
        summary_text.SCENARIO_PLAN_BASE_INSIGHTS_FORMAT.format(
            scenario_type='fixed',
            start_date='2020-01-05',
            end_date='2020-06-28',
        ),
    )

  @mock.patch.object(
      optimizer.BudgetOptimizer,
      'nonoptimized_data',
      new=property(lambda unused_self: _SAMPLE_NON_OPTIMIZED_DATA_KPI),
  )
  @mock.patch.object(
      optimizer.BudgetOptimizer,
      'optimized_data',
      new=property(lambda unused_self: _SAMPLE_OPTIMIZED_DATA_KPI),
  )
  def test_output_scenario_card_use_cpik_no_revenue_per_kpi(self):
    summary_html_dom = self._get_output_summary_html_dom(
        self.budget_optimizer_kpi_output
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
    self.assertEqual(title.strip(), 'Current budget')
    title = analysis_test_utils.get_child_element(stats[1], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized budget')
    title = analysis_test_utils.get_child_element(stats[2], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Current CPIK')
    title = analysis_test_utils.get_child_element(stats[3], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized CPIK')
    title = analysis_test_utils.get_child_element(stats[4], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Current incremental KPI')
    title = analysis_test_utils.get_child_element(stats[5], 'stats-title').text
    self.assertIsNotNone(title)
    self.assertEqual(title.strip(), 'Optimized incremental KPI')

  @parameterized.named_parameters(
      ('current_budget', 0, summary_text.CURRENT_BUDGET_LABEL, '$600', None),
      (
          'optimized_budget',
          1,
          summary_text.OPTIMIZED_BUDGET_LABEL,
          '$600',
          '$0',
      ),
      ('current_roi', 2, summary_text.CURRENT_ROI_LABEL, '1.3', None),
      ('optimized_roi', 3, summary_text.OPTIMIZED_ROI_LABEL, '1.4', '+0.1'),
      (
          'current_inc_revenue',
          4,
          summary_text.CURRENT_INC_IMPACT_LABEL.format(impact=c.REVENUE),
          '$760',
          None,
      ),
      (
          'optimized_inc_revenue',
          5,
          summary_text.OPTIMIZED_INC_IMPACT_LABEL.format(impact=c.REVENUE),
          '$830',
          '+$70',
      ),
  )
  def test_output_scenario_plan_card_stats_text(
      self, index, expected_title, expected_stat, expected_delta
  ):
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
    self.mock_spend_delta.assert_called_once()
    self.mock_budget_allocation.assert_called_once()
    self.mock_impact_delta.assert_called_once()

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

    impact_delta_description_text = analysis_test_utils.get_child_element(
        charts[2], 'chart-description'
    ).text
    self.assertIsNotNone(impact_delta_description_text)
    self.assertEqual(
        impact_delta_description_text.strip(),
        summary_text.IMPACT_DELTA_CHART_INSIGHTS_FORMAT.format(
            impact=c.REVENUE
        ),
    )

  def test_output_budget_allocation_table(self):
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
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
            impact=c.REVENUE
        ),
    )

  def test_output_response_curves_chart(self):
    summary_html_dom = self._get_output_summary_html_dom(self.budget_optimizer)
    self.mock_response_curves.assert_called_once_with(n_top_channels=3)

    card = analysis_test_utils.get_child_element(
        summary_html_dom,
        'body/cards/card',
        attribs={'id': summary_text.OPTIMIZED_RESPONSE_CURVES_CARD_ID},
    )

    charts = card.findall('charts/chart')
    self.assertLen(charts, 1)

  def _mock_chart(self) -> alt.Chart:
    return alt.Chart(pd.DataFrame()).mark_point()

  def _get_output_summary_html_dom(
      self, budget_optimizer: optimizer.BudgetOptimizer
  ) -> ET.Element:
    outfile_path = tempfile.mkdtemp() + '/optimization'
    outfile_name = 'optimization.html'
    fpath = os.path.join(outfile_path, outfile_name)

    try:
      budget_optimizer.output_optimization_summary(outfile_name, outfile_path)
      with open(fpath, 'r') as f:
        written_html_dom = ET.parse(f)
    finally:
      os.remove(fpath)
      os.removedirs(outfile_path)

    root = written_html_dom.getroot()
    self.assertEqual(root.tag, 'html')
    return root


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
          'incremental_impact': np.array([100, 100]),
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
          'incremental_impact': np.array([100, 100]),
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
          'incremental_impact': np.array([100, 100]),
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
          'incremental_impact': np.array([1_000, 1_000]),
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
          'incremental_impact': np.array([1_000, 1_000]),
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
          'incremental_impact': np.array([1_000, 1_000]),
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
      incremental_impact,
      roi_grid_point,
      target_mroi,
      target_roi,
      expected_output,
  ):
    exceeds = optimizer._exceeds_optimization_constraints(
        fixed_budget,
        budget,
        spend,
        incremental_impact,
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
    self.budget_optimizer_media_and_rf_kpi = optimizer.BudgetOptimizer(
        self.meridian_media_and_rf_kpi
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

  def test_incremental_impact_called_correct_optimize(self):
    mock_incremental_impact = self.enter_context(
        mock.patch.object(
            self.budget_optimizer_media_and_rf_kpi._analyzer,
            'incremental_impact',
            autospec=True,
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            )),
        )
    )
    self.budget_optimizer_media_and_rf_kpi.optimize()
    mock_incremental_impact.assert_called_with(
        new_media=mock.ANY,
        new_reach=mock.ANY,
        new_frequency=mock.ANY,
        selected_times=None,
        use_kpi=True,
        batch_size=c.DEFAULT_BATCH_SIZE,
    )

  def test_roi_and_mroi_not_existing_no_revenue_per_kpi_budget_dataset(self):
    hist_spend = np.array([1000, 1000, 1000, 1000, 1000])
    rounded_spend = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    selected_time_dims = [
        '2021-05-17',
        '2021-05-24',
        '2021-05-31',
        '2021-06-07',
        '2021-06-14',
    ]
    budget_dataset = (
        self.budget_optimizer_media_and_rf_kpi._create_budget_dataset(
            hist_spend=hist_spend,
            spend=rounded_spend,
            selected_times=selected_time_dims,
            batch_size=c.DEFAULT_BATCH_SIZE,
        )
    )
    self.assertEqual(
        list(budget_dataset.data_vars),
        [c.SPEND, c.PCT_OF_SPEND, c.INCREMENTAL_IMPACT, c.CPIK],
    )
    self.assertIn(c.TOTAL_CPIK, list(budget_dataset.attrs.keys()))
    self.assertNotIn(c.TOTAL_ROI, list(budget_dataset.attrs.keys()))


if __name__ == '__main__':
  absltest.main()
