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

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.analysis import test_utils as analysis_test_utils
from meridian.analysis import weekly_optimization_grid
from meridian.common import errors as common_errors
from meridian.data import test_utils as data_test_utils
from meridian.model import model
import numpy as np
import xarray as xr

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'model', 'test_data'
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


def _verify_actual_vs_expected_budget_data(
    actual_data: xr.Dataset,
    expected_data: xr.Dataset,
    *,
    atol: float = 0.1,
    rtol: float = 0.01,
) -> None:
  xr.testing.assert_allclose(actual_data, expected_data, atol=atol, rtol=rtol)
  np.testing.assert_allclose(
      actual_data.budget, expected_data.budget, atol=atol, rtol=rtol
  )
  np.testing.assert_allclose(
      actual_data.profit, expected_data.profit, atol=atol, rtol=rtol
  )
  np.testing.assert_allclose(
      actual_data.total_incremental_outcome,
      expected_data.total_incremental_outcome,
      atol=atol,
      rtol=rtol,
  )
  np.testing.assert_allclose(
      actual_data.total_roi, expected_data.total_roi, atol=atol, rtol=rtol
  )
  if c.FIXED_BUDGET in expected_data.attrs:
    np.testing.assert_equal(
        actual_data.attrs[c.FIXED_BUDGET],
        expected_data.attrs[c.FIXED_BUDGET],
    )


class WeeklyOptimizationGridTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

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

    self.meridian_media_and_rf = model.Meridian(
        input_data=self.input_data_media_and_rf
    )
    self.meridian_media_only = model.Meridian(
        input_data=self.input_data_media_only
    )

    self.budget_optimizer_media_and_rf = optimizer.BudgetOptimizer(
        self.meridian_media_and_rf
    )
    self.budget_optimizer_media_only = optimizer.BudgetOptimizer(
        self.meridian_media_only
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
            'inference_data',
            new=property(lambda unused_self: self.inference_data_media_and_rf),
        )
    )
    self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            'summary_metrics',
            return_value=analysis_test_utils.generate_paid_summary_metrics(),
            autospec=True,
        )
    )

  def test_create(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsInstance(
        weekly_grid, weekly_optimization_grid.WeeklyOptimizationGrid
    )
    assert weekly_grid is not None
    self.assertIsInstance(weekly_grid.nonoptimized_spend, xr.DataArray)
    self.assertEqual(
        list(weekly_grid.nonoptimized_spend.dims),
        [c.CHANNEL, c.TIME],
    )
    self.assertIsInstance(weekly_grid.incremental_outcome, xr.DataArray)
    self.assertEqual(
        list(weekly_grid.incremental_outcome.dims),
        [c.CHANNEL, c.SPEND_MULTIPLIER, c.TIME],
    )

  def test_create_with_new_data(self):
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    new_times = model_times
    new_data = analyzer.DataTensors(
        media=self.meridian_media_only.media_tensors.media,
        media_spend=self.meridian_media_only.media_tensors.media_spend,
        revenue_per_kpi=self.meridian_media_only.revenue_per_kpi,
        time=new_times,
    )
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        new_data=new_data,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsInstance(
        weekly_grid, weekly_optimization_grid.WeeklyOptimizationGrid
    )
    assert weekly_grid is not None
    self.assertEqual(
        list(weekly_grid.incremental_outcome.dims),
        [c.CHANNEL, c.SPEND_MULTIPLIER, c.TIME],
    )
    self.assertEqual(
        weekly_grid.incremental_outcome[c.TIME].data.tolist(),
        new_times,
    )

  def test_weekly_grid_constraint_validation_warns(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        max_constraint_variation=0.1,
        use_posterior=True,
    )
    assert weekly_grid is not None
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    first_time = model_times[0]
    last_time = model_times[-1]

    grid = weekly_grid.to_optimization_grid(
        start_date=first_time,
        end_date=last_time,
    )
    with self.assertWarnsRegex(
        UserWarning,
        'Optimization called with bounds that are not within the grid',
    ):
      self.budget_optimizer_media_only.optimize(
          optimization_grid=grid,
          spend_constraint_lower=0.2,
          spend_constraint_upper=0.0,
          start_date=first_time,
          end_date=last_time,
      )

    with self.assertWarnsRegex(
        UserWarning,
        'Optimization called with bounds that are not within the grid',
    ):
      self.budget_optimizer_media_only.optimize(
          optimization_grid=grid,
          spend_constraint_lower=0.0,
          spend_constraint_upper=0.2,
          start_date=first_time,
          end_date=last_time,
      )

  def test_weekly_grid_budget_validation_warns(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        max_budget_percent_decrease=0.1,
        max_budget_percent_increase=0.1,
        use_posterior=True,
    )
    assert weekly_grid is not None
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    first_time = model_times[0]
    last_time = model_times[-1]

    grid = weekly_grid.to_optimization_grid(
        start_date=first_time,
        end_date=last_time,
    )
    assert grid is not None
    hist_spend = np.sum(grid.historical_spend)
    with self.assertWarnsRegex(
        UserWarning,
        'Optimization called with bounds that are not within the grid',
    ):
      results = self.budget_optimizer_media_only.optimize(
          optimization_grid=grid,
          start_date=first_time,
          end_date=last_time,
          budget=hist_spend * 2.0,
      )
    self.assertIsNotNone(results)

  def test_weekly_grid_channels_property(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    assert weekly_grid is not None
    expected_channels = list(
        self.meridian_media_only.input_data.get_all_paid_channels()
    )
    self.assertEqual(weekly_grid.channels, expected_channels)

  def test_weekly_grid_time_property(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    assert weekly_grid is not None
    expected_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    self.assertEqual(weekly_grid.time, expected_times)

  def test_weekly_grid_validate_dates(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    assert weekly_grid is not None
    self.assertTrue(weekly_grid._validate_dates())

  def test_weekly_grid_date_validation_warns(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    assert weekly_grid is not None
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    first_time = model_times[0]
    last_time = model_times[-1]

    # Shorten time by removing the first week so start date is missing
    new_da_start = weekly_grid.incremental_outcome.sel(
        time=weekly_grid.incremental_outcome.time[1:]
    )
    invalid_weekly_grid_start = dataclasses.replace(
        weekly_grid, incremental_outcome=new_da_start
    )
    with self.assertWarnsRegex(
        UserWarning,
        'Given weekly grid does not cover start_date',
    ):
      grid = invalid_weekly_grid_start.to_optimization_grid(
          start_date=first_time,
      )
    results = self.budget_optimizer_media_only.optimize(
        optimization_grid=grid,
    )
    self.assertIsNotNone(results)

    # Shorten time by removing the last week so end date is missing
    new_da_end = weekly_grid.incremental_outcome.sel(
        time=weekly_grid.incremental_outcome.time[:-1]
    )
    invalid_weekly_grid_end = dataclasses.replace(
        weekly_grid, incremental_outcome=new_da_end
    )
    with self.assertWarnsRegex(
        UserWarning,
        'Given weekly grid does not cover end_date',
    ):
      grid = invalid_weekly_grid_end.to_optimization_grid(
          end_date=last_time,
      )
    results = self.budget_optimizer_media_only.optimize(
        optimization_grid=grid,
    )
    self.assertIsNotNone(results)

  def test_create_with_rf_succeeds(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    self.assertEqual(weekly_grid.n_rf_channels, _N_RF_CHANNELS)
    grid = weekly_grid.to_optimization_grid()
    self.assertIsNotNone(grid)
    results = self.budget_optimizer_media_and_rf.optimize(
        optimization_grid=grid,
    )
    self.assertIsNotNone(results)

  def test_to_optimization_grid_rf_stabilization(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    grid = weekly_grid.to_optimization_grid()
    self.assertIsNotNone(grid)

    spend_grid = grid.spend_grid.values
    outcome_grid = grid.incremental_outcome_grid.values

    for i in range(-_N_RF_CHANNELS, 0):
      spend_col = spend_grid[:, i]
      outcome_col = outcome_grid[:, i]
      valid_mask = ~np.isnan(spend_col)

      valid_spend = spend_col[valid_mask]
      valid_outcome = outcome_col[valid_mask]

      non_zero_mask = valid_spend != 0
      if np.sum(non_zero_mask) > 1:
        slopes = valid_outcome[non_zero_mask] / valid_spend[non_zero_mask]
        # All slopes should be almost identical due to stabilization
        np.testing.assert_allclose(slopes, slopes[0], rtol=1e-5)

  def test_create_with_rf_use_optimal_frequency(self):
    # Case 1: use_optimal_frequency=True (default)
    weekly_grid_opt = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        use_optimal_frequency=True,
    )
    self.assertIsNotNone(weekly_grid_opt)
    self.assertTrue(weekly_grid_opt.use_optimal_frequency)
    self.assertIsNotNone(weekly_grid_opt.opt_freq_ds)
    self.assertIsInstance(weekly_grid_opt.opt_freq_ds, xr.Dataset)

    # Case 2: use_optimal_frequency=False
    weekly_grid_no_opt = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        use_optimal_frequency=False,
    )
    self.assertIsNotNone(weekly_grid_no_opt)
    self.assertFalse(weekly_grid_no_opt.use_optimal_frequency)
    self.assertIsNone(weekly_grid_no_opt.opt_freq_ds)

  @parameterized.named_parameters(
      dict(
          testcase_name='negative_decrease',
          max_budget_percent_decrease=-0.1,
      ),
      dict(
          testcase_name='decrease_equals_one',
          max_budget_percent_decrease=1.0,
      ),
      dict(
          testcase_name='decrease_greater_than_one',
          max_budget_percent_decrease=1.5,
      ),
  )
  def test_create_invalid_max_budget_percent_decrease_raises_error(
      self, max_budget_percent_decrease
  ):
    with self.assertRaisesRegex(
        ValueError,
        '`max_budget_percent_decrease` must be in the range \\[0, 1\\)\\.',
    ):
      weekly_optimization_grid.WeeklyOptimizationGrid.create(
          self.budget_optimizer_media_only._analyzer,
          max_budget_percent_decrease=max_budget_percent_decrease,
      )

  def test_create_invalid_max_budget_percent_increase_raises_error(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        '`max_budget_percent_increase` must be non-negative\\.',
    ):
      weekly_optimization_grid.WeeklyOptimizationGrid.create(
          self.budget_optimizer_media_only._analyzer,
          max_budget_percent_increase=-0.1,
      )

  def test_create_invalid_max_constraint_variation_raises_error(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        '`max_constraint_variation` must be non-negative\\.',
    ):
      weekly_optimization_grid.WeeklyOptimizationGrid.create(
          self.budget_optimizer_media_only._analyzer,
          max_constraint_variation=-0.1,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='default_dates_default_budget',
          start_offset=0,
          end_offset=-1,
          budget_factor=1.0,
      ),
      dict(
          testcase_name='custom_dates_reduced_budget',
          start_offset=5,
          end_offset=-5,
          budget_factor=0.9,
      ),
      dict(
          testcase_name='custom_dates_increased_budget',
          start_offset=10,
          end_offset=-10,
          budget_factor=1.1,
      ),
  )
  def test_compare_weekly_and_standard_grid_optimization(
      self, start_offset: int, end_offset: int, budget_factor: float
  ):
    times = self.meridian_media_only.input_data.time.to_numpy().tolist()
    start_date = times[start_offset] if start_offset > 0 else None
    end_date = times[end_offset] if end_offset < -1 else None

    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.0001,
        use_posterior=True,
        batch_size=10,
    )
    assert weekly_grid is not None
    hist_spend = np.sum(
        self.budget_optimizer_media_only.create_optimization_grid(
            start_date=start_date, end_date=end_date
        ).historical_spend
    )
    budget = hist_spend * budget_factor

    results_weekly = None
    with mock.patch.object(
        weekly_optimization_grid.WeeklyOptimizationGrid,
        '_create_grids',
        wraps=weekly_optimization_grid.WeeklyOptimizationGrid._create_grids,
        autospec=True,
    ) as mock_create_grids:
      grid = weekly_grid.to_optimization_grid(
          start_date=start_date,
          end_date=end_date,
      )
      results_weekly = self.budget_optimizer_media_only.optimize(
          optimization_grid=grid,
          spend_constraint_lower=0.1,
          spend_constraint_upper=0.1,
          start_date=start_date,
          end_date=end_date,
          budget=budget,
      )
      mock_create_grids.assert_called_once()
    results_standard = self.budget_optimizer_media_only.optimize(
        spend_constraint_lower=0.1,
        spend_constraint_upper=0.1,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
    )
    _verify_actual_vs_expected_budget_data(
        results_weekly.optimized_data,
        results_standard.optimized_data,
        rtol=0.05,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='default_dates_default_budget',
          start_offset=0,
          end_offset=-1,
          budget_factor=1.0,
      ),
      dict(
          testcase_name='custom_dates_reduced_budget',
          start_offset=1,
          end_offset=-2,
          budget_factor=0.9,
      ),
      dict(
          testcase_name='custom_dates_increased_budget',
          start_offset=2,
          end_offset=-1,
          budget_factor=1.1,
      ),
  )
  def test_compare_weekly_and_standard_grid_optimization_with_new_data(
      self, start_offset: int, end_offset: int, budget_factor: float
  ):
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    new_times = model_times
    start_date = new_times[start_offset] if start_offset > 0 else None
    end_date = new_times[end_offset] if end_offset < -1 else None
    new_data = analyzer.DataTensors(
        media=self.meridian_media_only.media_tensors.media,
        media_spend=self.meridian_media_only.media_tensors.media_spend,
        revenue_per_kpi=self.meridian_media_only.revenue_per_kpi,
        time=new_times,
    )
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        new_data=new_data,
        multiplier_step=0.0001,
        use_posterior=True,
        batch_size=10,
    )
    assert weekly_grid is not None
    hist_spend = np.sum(
        self.budget_optimizer_media_only.create_optimization_grid(
            new_data=new_data, start_date=start_date, end_date=end_date
        ).historical_spend
    )
    budget = hist_spend * budget_factor

    results_weekly = None
    with mock.patch.object(
        weekly_optimization_grid.WeeklyOptimizationGrid,
        '_create_grids',
        wraps=weekly_optimization_grid.WeeklyOptimizationGrid._create_grids,
        autospec=True,
    ) as mock_create_grids:
      grid = weekly_grid.to_optimization_grid(
          start_date=start_date,
          end_date=end_date,
      )
      results_weekly = self.budget_optimizer_media_only.optimize(
          new_data=new_data,
          optimization_grid=grid,
          spend_constraint_lower=0.1,
          spend_constraint_upper=0.1,
          start_date=start_date,
          end_date=end_date,
          budget=budget,
      )
      mock_create_grids.assert_called_once()
    results_standard = self.budget_optimizer_media_only.optimize(
        new_data=new_data,
        spend_constraint_lower=0.1,
        spend_constraint_upper=0.1,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
    )
    _verify_actual_vs_expected_budget_data(
        results_weekly.optimized_data,
        results_standard.optimized_data,
        rtol=0.1,
    )

  def test_create_not_fitted_model_raises_error(self):
    with mock.patch.object(
        self.budget_optimizer_media_only._analyzer.inference_data,
        'groups',
        return_value=[],
        autospec=True,
    ):
      with self.assertRaisesRegex(
          common_errors.NotFittedModelError,
          'Running budget optimization scenarios requires fitting the model.',
      ):
        weekly_optimization_grid.WeeklyOptimizationGrid.create(
            self.budget_optimizer_media_only._analyzer,
        )

  def test_create_default_multiplier_step(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)
    self.assertIsNotNone(weekly_grid.multiplier_step)

  def test_create_prior(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=False,
    )
    self.assertIsNotNone(weekly_grid)
    self.assertFalse(weekly_grid.use_posterior)

  def test_create_no_revenue_per_kpi(self):
    with mock.patch.object(
        model.Meridian, '_validate_injected_inference_data', autospec=True
    ):
      input_data_non_revenue = (
          data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
              n_geos=_N_GEOS,
              n_times=_N_TIMES,
              n_media_times=_N_MEDIA_TIMES,
              n_controls=_N_CONTROLS,
              n_media_channels=_N_MEDIA_CHANNELS,
              n_rf_channels=0,
              seed=0,
          )
      )
      meridian_no_rev = model.Meridian(input_data=input_data_non_revenue)
    meridian_analyzer = analyzer.Analyzer(
        model_context=meridian_no_rev.model_context,
        inference_data=self.inference_data_media_only,
    )
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        meridian_analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)

  def test_create_media_transformer_none(self):
    new_media_tensors = dataclasses.replace(
        self.budget_optimizer_media_only._analyzer.model_context.media_tensors,
        media_transformer=None,
    )
    with mock.patch.object(
        self.budget_optimizer_media_only._analyzer.model_context.__class__,
        'media_tensors',
        new_callable=mock.PropertyMock,
        return_value=new_media_tensors,
    ):
      weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
          self.budget_optimizer_media_only._analyzer,
          multiplier_step=0.5,
          use_posterior=True,
      )
      self.assertIsNotNone(weekly_grid)

  def test_create_grids_channel_spend_zero(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)

    # Set spend of first channel to 0
    zero_spend = weekly_grid.nonoptimized_spend.copy()
    zero_spend[0, :] = 0.0
    weekly_grid_zero = dataclasses.replace(
        weekly_grid, nonoptimized_spend=zero_spend
    )

    grid = weekly_grid_zero.to_optimization_grid()
    self.assertIsNotNone(grid)

    spend_grid = grid.spend_grid.values
    outcome_grid = grid.incremental_outcome_grid.values

    valid_mask_zero_spend = ~np.isnan(spend_grid[:, 0])
    self.assertTrue(np.any(valid_mask_zero_spend))
    self.assertTrue(np.any(np.isnan(spend_grid[:, 0])))
    np.testing.assert_array_equal(outcome_grid[valid_mask_zero_spend, 0], 0.0)
    np.testing.assert_array_equal(np.isnan(outcome_grid), np.isnan(spend_grid))

  def test_to_optimization_grid_custom_budget_and_constraints(self):
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.1,
        use_posterior=True,
    )
    self.assertIsNotNone(weekly_grid)

    # Valid custom budget and constraints
    grid = weekly_grid.to_optimization_grid(
        budget=1000.0,
        spend_constraint_lower=0.1,
        spend_constraint_upper=0.1,
    )
    self.assertIsNotNone(grid)

    # Constraints exceed grid coverage
    with self.assertWarnsRegex(
        UserWarning,
        'Bounds are not within the grid',
    ):
      invalid_grid = weekly_grid.to_optimization_grid(
          budget=100000.0,
          spend_constraint_lower=0.1,
          spend_constraint_upper=0.1,
      )
      self.assertIsNone(invalid_grid)

  def test_create_with_start_end_date(self):
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    start_date = model_times[0]
    end_date = model_times[3]
    weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=start_date,
        end_date=end_date,
    )
    self.assertIsNotNone(weekly_grid)
    expected_times = model_times[:4]
    self.assertEqual(weekly_grid.time, expected_times)
    self.assertEqual(
        weekly_grid.nonoptimized_spend[c.TIME].data.tolist(), expected_times
    )

  def test_combine_weekly_grids(self):
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    grid1 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[0],
        end_date=model_times[2],
    )
    grid2 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[3],
        end_date=model_times[5],
    )
    combined = weekly_optimization_grid.WeeklyOptimizationGrid.combine(
        [grid1, grid2]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertIsNotNone(combined)
    expected_times = model_times[:6]
    self.assertEqual(combined.time, expected_times)
    self.assertEqual(
        combined.nonoptimized_spend[c.TIME].data.tolist(), expected_times
    )

    # Test empty grids
    with self.assertRaisesRegex(ValueError, 'must not be empty'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine([])

    # Test single grid
    self.assertEqual(
        weekly_optimization_grid.WeeklyOptimizationGrid.combine([grid1]), grid1  # pyrefly: ignore[bad-argument-type]
    )

    # Test duplicate dates
    with self.assertRaisesRegex(ValueError, 'duplicate dates'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine([grid1, grid1])  # pyrefly: ignore[bad-argument-type]

    # Test non-contiguous gap
    grid_gap = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[4],
        end_date=model_times[5],
    )
    with self.assertRaisesRegex(ValueError, 'contiguous'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine([grid1, grid_gap])  # pyrefly: ignore[bad-argument-type]

    # Test mismatched attributes
    grid_mismatch = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.2,
        use_posterior=True,
        start_date=model_times[3],
        end_date=model_times[5],
    )
    with self.assertRaisesRegex(ValueError, 'multiplier_step'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine(
          [grid1, grid_mismatch]  # pyrefly: ignore[bad-argument-type]
      )

  def test_combine_three_weekly_grids(self):
    model_times = (
        np.asarray(self.meridian_media_only.input_data.time)
        .astype(str)
        .tolist()
    )
    grid1 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[0],
        end_date=model_times[1],
    )
    grid2 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[2],
        end_date=model_times[3],
    )
    grid3 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_only._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[4],
        end_date=model_times[5],
    )
    combined = weekly_optimization_grid.WeeklyOptimizationGrid.combine(
        [grid1, grid2, grid3]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertIsNotNone(combined)
    expected_times = model_times[:6]
    self.assertEqual(combined.time, expected_times)
    self.assertEqual(
        combined.nonoptimized_spend[c.TIME].data.tolist(), expected_times
    )

  def test_combine_weekly_grids_with_rf(self):
    model_times = (
        np.asarray(self.meridian_media_and_rf.input_data.time)
        .astype(str)
        .tolist()
    )
    grid1 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[0],
        end_date=model_times[2],
    )
    grid2 = weekly_optimization_grid.WeeklyOptimizationGrid.create(
        self.budget_optimizer_media_and_rf._analyzer,
        multiplier_step=0.5,
        use_posterior=True,
        start_date=model_times[3],
        end_date=model_times[5],
    )
    self.assertIsNotNone(grid1.opt_freq_ds)
    self.assertIsNotNone(grid2.opt_freq_ds)

    combined = weekly_optimization_grid.WeeklyOptimizationGrid.combine(
        [grid1, grid2]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertIsNotNone(combined)
    expected_times = model_times[:6]
    self.assertEqual(combined.time, expected_times)
    self.assertEqual(
        combined.nonoptimized_spend[c.TIME].data.tolist(), expected_times
    )
    self.assertIsNotNone(combined.opt_freq_ds)
    self.assertTrue(combined.opt_freq_ds.equals(grid1.opt_freq_ds))

    # Test mismatched opt_freq_ds presence: second grid missing opt_freq_ds
    grid2_no_opt_freq_ds = dataclasses.replace(grid2, opt_freq_ds=None)
    with self.assertRaisesRegex(ValueError, 'different opt_freq_ds presence'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine(
          [grid1, grid2_no_opt_freq_ds]  # pyrefly: ignore[bad-argument-type]
      )

    # Test mismatched opt_freq_ds presence: first grid missing opt_freq_ds
    grid1_no_opt_freq_ds = dataclasses.replace(grid1, opt_freq_ds=None)
    with self.assertRaisesRegex(ValueError, 'different opt_freq_ds presence'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine(
          [grid1_no_opt_freq_ds, grid2]  # pyrefly: ignore[bad-argument-type]
      )

    # Test mismatched opt_freq_ds values
    diff_opt_freq_ds = grid2.opt_freq_ds.copy(deep=True)
    diff_opt_freq_ds = diff_opt_freq_ds.assign(
        dummy_var=(['channel'], np.ones(grid2.n_rf_channels))
    )
    grid2_diff_opt_freq_ds = dataclasses.replace(
        grid2, opt_freq_ds=diff_opt_freq_ds
    )
    with self.assertRaisesRegex(ValueError, 'different opt_freq_ds values'):
      weekly_optimization_grid.WeeklyOptimizationGrid.combine(
          [grid1, grid2_diff_opt_freq_ds]  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == '__main__':
  absltest.main()
