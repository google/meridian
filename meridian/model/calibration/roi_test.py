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

from __future__ import annotations

from collections.abc import Callable, Sequence
import datetime
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants as meridian_constants
from meridian.backend import test_utils
from meridian.model.calibration import base
from meridian.model.calibration import roi
import numpy as np
from scipy import stats


def _get_seed():
  return backend.RNGHandler(42).get_next_seed()


class RoiTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="equal_rates",
          channel_avg_daily_spend=100.0,
          experiment_avg_daily_spend=100.0,
          expected_tau=0.0,
      ),
      dict(
          testcase_name="experiment_lower_rate",
          channel_avg_daily_spend=200.0,
          experiment_avg_daily_spend=100.0,
          expected_tau=1.0,
      ),
      dict(
          testcase_name="experiment_higher_rate",
          channel_avg_daily_spend=100.0,
          experiment_avg_daily_spend=200.0,
          expected_tau=1.0,
      ),
  )
  def test_get_spend_adjustment(
      self,
      channel_avg_daily_spend: float,
      experiment_avg_daily_spend: float,
      expected_tau: float,
  ) -> None:
    tau = roi._get_spend_adjustment(
        channel_avg_daily_spend=channel_avg_daily_spend,
        experiment_avg_daily_spend=experiment_avg_daily_spend,
        channel_name="Search",
    )
    self.assertAlmostEqual(tau, expected_tau)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_channel_spend",
          channel_avg_daily_spend=0.0,
      ),
      dict(
          testcase_name="negative_channel_spend",
          channel_avg_daily_spend=-10.0,
      ),
  )
  def test_get_spend_adjustment_invalid_channel_spend_raises_value_error(
      self,
      channel_avg_daily_spend: float,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError,
        "Average daily channel spend must be positive.*for channel 'Search'",
    ):
      roi._get_spend_adjustment(
          channel_avg_daily_spend=channel_avg_daily_spend,
          experiment_avg_daily_spend=100.0,
          channel_name="Search",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_experiment_spend",
          experiment_avg_daily_spend=0.0,
      ),
      dict(
          testcase_name="negative_experiment_spend",
          experiment_avg_daily_spend=-5.0,
      ),
  )
  def test_get_spend_adjustment_invalid_experiment_spend_raises_value_error(
      self,
      experiment_avg_daily_spend: float,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError,
        "Average daily experiment spend must be positive.*for channel 'Search'",
    ):
      roi._get_spend_adjustment(
          channel_avg_daily_spend=100.0,
          experiment_avg_daily_spend=experiment_avg_daily_spend,
          channel_name="Search",
      )

  @parameterized.named_parameters(
      # Specific geometric decay cases.
      dict(
          testcase_name="geometric_max_lag",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=8.0,
          expected_factor=1.00196,
      ),
      dict(
          testcase_name="geometric_half_duration",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=4.0,
          expected_factor=1.06458,
      ),
      dict(
          testcase_name="geometric_custom_rate",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.8,
          duration=4.0,
          expected_factor=1.46643,
      ),
      # Specific binomial decay cases.
      dict(
          testcase_name="binomial_half_duration",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.5,
          duration=4.0,
          expected_factor=1.50000,
      ),
      # Specific custom rate cases.
      dict(
          testcase_name="binomial_custom_rate",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.8,
          duration=4.0,
          expected_factor=1.97114,
      ),
      dict(
          testcase_name="binomial_decay_rate_zero",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.0,
          duration=3.0,
          expected_factor=1.0,
      ),
      dict(
          testcase_name="geometric_duration_greater_than_max_lag",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=10.0,
          expected_factor=0.99902,
      ),
      dict(
          testcase_name="binomial_duration_greater_than_max_lag",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.5,
          duration=10.0,
          expected_factor=1.0,
      ),
  )
  def test_duration_adjustment_gamma(
      self,
      adstock_decay_function: str,
      alpha: float,
      duration: float,
      expected_factor: float,
  ) -> None:
    gamma_duration, _ = roi._duration_adjustment(
        duration=duration,
        max_lag=8,
        adstock_decay_function=adstock_decay_function,
        alpha=alpha,
    )
    self.assertAlmostEqual(gamma_duration, expected_factor, places=5)

  @parameterized.named_parameters(
      # Specific geometric decay cases.
      dict(
          testcase_name="geometric_max_lag",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=8.0,
          expected_factor=0.00196,
      ),
      dict(
          testcase_name="geometric_half_duration",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=4.0,
          expected_factor=0.06458,
      ),
      # Specific binomial decay cases.
      dict(
          testcase_name="binomial_half_duration",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.5,
          duration=4.0,
          expected_factor=0.50000,
      ),
      dict(
          testcase_name="geometric_duration_greater_than_max_lag",
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          duration=10.0,
          expected_factor=0.0,
      ),
      dict(
          testcase_name="binomial_duration_greater_than_max_lag",
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.5,
          duration=10.0,
          expected_factor=0.0,
      ),
  )
  def test_duration_adjustment_tau(
      self,
      adstock_decay_function: str,
      alpha: float,
      duration: float,
      expected_factor: float,
  ) -> None:
    _, tau_duration = roi._duration_adjustment(
        duration=duration,
        max_lag=8,
        adstock_decay_function=adstock_decay_function,
        alpha=alpha,
    )
    self.assertAlmostEqual(tau_duration, expected_factor, places=5)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_no_adjustment",
          standard_error_adjustment=None,
          point_estimate_adjustment=None,
          gamma=1.0,
          tau=2.0,
          expected_mean=2.5,
          expected_std=0.6928,
      ),
      dict(
          testcase_name="with_standard_error_adjustment",
          standard_error_adjustment=1.0,
          point_estimate_adjustment=None,
          gamma=1.0,
          tau=2.0,
          expected_mean=2.5,
          expected_std=0.8,
      ),
      dict(
          testcase_name="with_point_estimate_adjustment",
          standard_error_adjustment=None,
          point_estimate_adjustment=2.0,
          gamma=1.0,
          tau=2.0,
          expected_mean=7.5,
          expected_std=0.6928,
      ),
      dict(
          testcase_name="with_gamma_scaling",
          standard_error_adjustment=None,
          point_estimate_adjustment=None,
          gamma=3.0,
          tau=2.0,
          expected_mean=7.5,
          expected_std=0.6928,
      ),
      dict(
          testcase_name="with_both_adjustments",
          standard_error_adjustment=1.0,
          point_estimate_adjustment=2.0,
          gamma=1.0,
          tau=2.0,
          expected_mean=7.5,
          expected_std=0.8,
      ),
  )
  def test_get_adjusted_mean_and_std(
      self,
      standard_error_adjustment: float | None,
      point_estimate_adjustment: float | None,
      gamma: float,
      tau: float,
      expected_mean: float,
      expected_std: float,
  ):
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.4)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2025, 1, 1),
        experiment_end_date=datetime.date(2025, 1, 10),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        standard_error_adjustment=standard_error_adjustment,
        point_estimate_adjustment=point_estimate_adjustment,
    )
    mean, std = roi._get_adjusted_mean_and_std(cfg, gamma=gamma, tau=tau)
    self.assertEqual(mean, expected_mean)
    self.assertAlmostEqual(std, expected_std, places=4)

  @parameterized.named_parameters(
      dict(
          testcase_name="negative_tau",
          standard_error_adjustment=-4.0,
          tau=2.0,
      ),
      dict(
          testcase_name="minus_one_tau",
          standard_error_adjustment=-3.0,
          tau=2.0,
      ),
  )
  def test_get_adjusted_mean_and_std_raises_value_error(
      self, standard_error_adjustment: float, tau: float
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.4)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2025, 1, 1),
        experiment_end_date=datetime.date(2025, 1, 10),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        standard_error_adjustment=standard_error_adjustment,
    )
    with self.assertRaisesRegex(ValueError, "Tau must be greater than -1.0"):
      roi._get_adjusted_mean_and_std(cfg, gamma=1.0, tau=tau)

  def test_get_calibrated_roi_prior_spend_exceeds_channel_spend_succeeds(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # 7 days experiment.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 8),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=500.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 1, 8),
        model_duration_days=7,
    )
    self.assertEqual(output.experiments[0].tau_spend, 1.0)

  def test_merge_distributions_raises_value_error_for_zero_or_negative_standard_errors(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "Standard errors must be positive"):
      roi._merge_distributions(
          means=[2.0], stds=[0.0], baseline_prior=None, channel_name="Search"
      )

    with self.assertRaisesRegex(ValueError, "Standard errors must be positive"):
      roi._merge_distributions(
          means=[2.0], stds=[-1.0], baseline_prior=None, channel_name="Search"
      )

  def test_get_calibrated_roi_prior_empty_configs_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, "No calibration data provided for channel 'Search'"
    ):
      roi.get_calibrated_roi_prior(
          calibration_data=[],
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          max_lag=meridian_constants.DEFAULT_MAX_LAG,
          total_channel_spend=4000.0,
          channel_name="Search",
          interval_days=7,
          last_modeled_date=datetime.date(2026, 1, 8),
          model_duration_days=63,
      )

  def test_get_calibrated_roi_prior_single_config_calculates_best_fit(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )

    prior, _ = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    self.assertIsInstance(prior, backend.tfd.Gamma)
    self.assertAlmostEqual(float(prior.concentration), 6.7, places=1)
    self.assertAlmostEqual(float(prior.rate), 2.7, places=1)

  def test_get_calibrated_roi_prior_multiple_configs_merges_distributions(
      self,
  ) -> None:
    obs_1 = base.ExperimentResult(point_estimate=2.0, standard_error=0.4)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    info_1 = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg_1 = base.CalibrationData(
        experiment_result=obs_1, experiment_info=info_1
    )

    obs_2 = base.ExperimentResult(point_estimate=3.0, standard_error=0.6)
    info_2 = base.ExperimentInfo(
        total_spend=2000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg_2 = base.CalibrationData(
        experiment_result=obs_2, experiment_info=info_2
    )

    prior, _ = roi.get_calibrated_roi_prior(
        calibration_data=[cfg_1, cfg_2],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    self.assertIsInstance(prior, backend.tfd.Gamma)
    self.assertAlmostEqual(float(prior.concentration), 18.4, places=1)
    self.assertAlmostEqual(float(prior.rate), 7.4, places=1)

  def test_get_calibrated_roi_prior_with_baseline_prior_regularizes(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.0, standard_error=0.4)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )

    baseline = backend.tfd.Normal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(1.0, backend.float_dtype),
    )

    prior, _ = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
        baseline_prior=baseline,
    )

    w_cfg = 1.0 / (0.4 * np.sqrt(1.0 + 3.0)) ** 2
    w_baseline = 1.0 / 1.0**2
    expected_mean = (w_cfg * 2.0 + w_baseline * 1.0) / (w_cfg + w_baseline)
    expected_std = 1.0 / np.sqrt(w_cfg + w_baseline)

    self.assertIsInstance(prior, backend.tfd.Normal)
    self.assertAlmostEqual(float(prior.loc), expected_mean, places=2)
    self.assertAlmostEqual(float(prior.scale), expected_std, places=1)

  def test_get_calibrated_roi_prior_with_adjustments(self) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        point_estimate_adjustment=0.5,
        standard_error_adjustment=1.0,
    )

    prior, _ = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    self.assertIsInstance(prior, backend.tfd.Gamma)
    self.assertAlmostEqual(float(prior.concentration), 11.5, places=1)
    self.assertAlmostEqual(float(prior.rate), 3.1, places=1)

  def test_get_calibrated_roi_prior_populates_calibration_output_metadata(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        source_type=base.SourceType.MERIDIAN_GEOX,
    )
    baseline = backend.tfd.Normal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(1.0, backend.float_dtype),
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=8,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
        baseline_prior=baseline,
    )

    self.assertEqual(output.channel_name, "Search")
    self.assertEqual(output.baseline_prior, baseline)
    self.assertIsInstance(output.intermediary_prior, roi.GridDistribution)
    self.assertLen(output.experiments, 1)

  def test_get_calibrated_roi_prior_populates_experiment_adjustments(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        source_type=base.SourceType.MERIDIAN_GEOX,
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=8,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    self.assertEqual(
        output.adstock_decay_spec, meridian_constants.GEOMETRIC_DECAY
    )
    self.assertEqual(output.max_lag, 8)
    exp_out = output.experiments[0]
    self.assertEqual(exp_out.tau_spend, 3.0)
    self.assertEqual(exp_out.tau_recency, 0.0)
    self.assertEqual(exp_out.tau_duration, 0.0)
    self.assertEqual(exp_out.gamma_duration, 1.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly",
          interval_days=7,
          max_lag=8,
          duration=4.0,
      ),
      dict(
          testcase_name="daily",
          interval_days=1,
          max_lag=30,
          duration=28.0,
      ),
  )
  def test_get_calibrated_roi_prior_populates_experiment_adjustments_active_duration(
      self, interval_days: int, max_lag: int, duration: float
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # 28 days experiment
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 29),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        source_type=base.SourceType.MERIDIAN_GEOX,
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=max_lag,
        interval_days=interval_days,
        channel_name="Search",
        total_channel_spend=4000.0,
        last_modeled_date=datetime.date(2026, 1, 29),
        model_duration_days=28,
    )

    exp_out = output.experiments[0]
    expected_gamma, expected_tau = roi._duration_adjustment(
        duration=duration,
        max_lag=max_lag,
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
    )
    self.assertAlmostEqual(exp_out.gamma_duration, expected_gamma)
    self.assertAlmostEqual(exp_out.tau_duration, expected_tau)

  def test_get_calibrated_roi_prior_populates_experiment_results(self) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        source_type=base.SourceType.MERIDIAN_GEOX,
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=8,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    exp_out = output.experiments[0]
    self.assertEqual(exp_out.source_type, base.SourceType.MERIDIAN_GEOX)
    self.assertEqual(exp_out.raw_experiment_result, obs)
    self.assertAlmostEqual(
        exp_out.adjusted_experiment_result.point_estimate, 2.5, places=2
    )
    self.assertAlmostEqual(
        exp_out.adjusted_experiment_result.standard_error, 1.0, places=2
    )

  def test_get_calibrated_roi_prior_populates_user_adjustments(self) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        point_estimate_adjustment=0.5,
        standard_error_adjustment=1.0,
        source_type=base.SourceType.GENERIC,
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=8,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
    )

    exp_out = output.experiments[0]
    self.assertEqual(exp_out.user_point_estimate_adjustment, 0.5)
    self.assertEqual(exp_out.user_standard_error_adjustment, 1.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="negative_adjusted_tau",
          standard_error_adjustment=-5.0,
      ),
      dict(
          testcase_name="minus_one_adjusted_tau",
          standard_error_adjustment=-4.0,
      ),
  )
  def test_get_calibrated_roi_prior_negative_adjusted_tau_raises_value_error(
      self, standard_error_adjustment: float
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs,
        experiment_info=experiment_info,
        standard_error_adjustment=standard_error_adjustment,
    )
    with self.assertRaisesRegex(ValueError, "Tau must be greater than -1.0"):
      roi.get_calibrated_roi_prior(
          calibration_data=[cfg],
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          max_lag=meridian_constants.DEFAULT_MAX_LAG,
          channel_name="Search",
          total_channel_spend=4000.0,
          interval_days=7,
          last_modeled_date=datetime.date(2026, 3, 5),
          model_duration_days=63,
      )

  def test_get_calibrated_roi_prior_selects_gamma_for_skewed_posterior(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.0, standard_error=0.9)
    # Duration is exactly 9 weeks (max_lag + 1) so duration adjustments are
    # inactive.
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 3, 5),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )

    baseline = backend.tfd.LogNormal(
        loc=backend.cast(0.2, backend.float_dtype),
        scale=backend.cast(0.9, backend.float_dtype),
    )

    prior, _ = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 3, 5),
        model_duration_days=63,
        baseline_prior=baseline,
    )

    self.assertIsInstance(prior, backend.tfd.Gamma)

  def test_fit_distribution_selects_lognormal_for_skewed_data(self) -> None:
    grid = np.linspace(0.01, 10.0, 1000)
    dist = backend.tfd.LogNormal(
        loc=backend.cast(0.5, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    pdf = np.asarray(dist.prob(grid))
    dx = (10.0 - 0.01) / 999.0
    pdf = pdf / (np.sum(pdf) * dx)

    fitted = roi._fit_distribution(
        grid_np=grid,
        pdf_np=pdf,
        dx=dx,
        channel_name="Search",
    )
    self.assertIsInstance(fitted, backend.tfd.LogNormal)

  def test_get_calibrated_roi_prior_with_disjoint_prior_and_likelihoods_raises_value_error(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.1)
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 8),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )
    mock_baseline = mock.create_autospec(
        backend.tfd.Distribution, instance=True, spec_set=True
    )
    mock_baseline.mean.return_value = backend.to_tensor(1.0)
    mock_baseline.variance.return_value = backend.to_tensor(1.0)
    mock_baseline.log_prob.return_value = backend.to_tensor([-np.inf])

    with self.assertRaisesRegex(ValueError, "probability mass"):
      roi.get_calibrated_roi_prior(
          calibration_data=[cfg],
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          max_lag=meridian_constants.DEFAULT_MAX_LAG,
          channel_name="Search",
          total_channel_spend=4000.0,
          interval_days=7,
          last_modeled_date=datetime.date(2026, 1, 8),
          model_duration_days=7,
          baseline_prior=mock_baseline,
      )

  def test_fit_distribution_fails_on_nan_pdf(self) -> None:
    grid_np = np.array([0.1, 0.2, 0.3])
    pdf_np = np.array([np.nan, np.nan, np.nan])
    with self.assertRaisesRegex(ValueError, "Failed to fit any candidate"):
      roi._fit_distribution(
          grid_np=grid_np,
          pdf_np=pdf_np,
          dx=0.1,
          channel_name="Search",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="normal_zero_scale",
          loss_fn=roi._normal_loss,
          p=[1.0, 0.0],
      ),
      dict(
          testcase_name="normal_negative_scale",
          loss_fn=roi._normal_loss,
          p=[1.0, -0.5],
      ),
      dict(
          testcase_name="lognormal_zero_scale",
          loss_fn=roi._lognormal_loss,
          p=[1.0, 0.0],
      ),
      dict(
          testcase_name="gamma_zero_shape",
          loss_fn=roi._gamma_loss,
          p=[0.0, 1.0],
      ),
      dict(
          testcase_name="gamma_zero_rate",
          loss_fn=roi._gamma_loss,
          p=[1.0, 0.0],
      ),
      dict(
          testcase_name="gamma_negative_params",
          loss_fn=roi._gamma_loss,
          p=[-0.5, -0.5],
      ),
      dict(
          testcase_name="gamma_negative_shape",
          loss_fn=roi._gamma_loss,
          p=[-0.5, 1.0],
      ),
      dict(
          testcase_name="gamma_negative_rate",
          loss_fn=roi._gamma_loss,
          p=[1.0, -0.5],
      ),
  )
  def test_loss_boundary(
      self, loss_fn: Callable[..., float], p: Sequence[float]
  ) -> None:
    grid = np.array([0.1, 0.2, 0.3])
    pdf = np.array([0.2, 0.5, 0.3])
    self.assertEqual(loss_fn(p, grid_np=grid, pdf_np=pdf, dx=0.1), np.inf)

  def test_compute_grid_bounds_zero_probability_mass_raises_value_error(
      self,
  ) -> None:
    prior = roi.ImproperUniformPrior()
    mock_reduce_sum = mock.create_autospec(roi.backend.reduce_sum)
    mock_reduce_sum.return_value = backend.to_tensor(np.nan)
    with mock.patch.object(roi.backend, "reduce_sum", mock_reduce_sum):
      with self.assertRaisesRegex(ValueError, "Scouting pass resulted in zero"):
        roi._compute_grid_bounds(prior=prior, likelihoods=[])

  def test_compute_grid_bounds_calculates_correct_range(self) -> None:
    prior = backend.tfd.Normal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    grid_min, grid_max = roi._compute_grid_bounds(prior=prior, likelihoods=[])
    self.assertAlmostEqual(grid_min, -0.545, delta=0.02)
    self.assertAlmostEqual(grid_max, 2.545, delta=0.02)

  def test_is_finite_backend(self) -> None:
    res = backend.is_finite(backend.to_tensor(1.5))  # pyrefly: ignore[bad-argument-type]
    self.assertTrue(bool(np.all(np.asarray(res))))

  def test_evaluate_posterior_nan_sum_raises_value_error(self) -> None:
    prior = roi.ImproperUniformPrior()
    mock_reduce_sum = mock.create_autospec(roi.backend.reduce_sum)
    mock_reduce_sum.return_value = backend.to_tensor(np.nan)
    with mock.patch.object(roi.backend, "reduce_sum", mock_reduce_sum):
      with self.assertRaisesRegex(ValueError, "non-finite probability mass"):
        roi._evaluate_posterior(
            prior=prior,
            likelihoods=[],
            grid_min=0.0,
            grid_max=2.0,
        )

  def test_evaluate_posterior_zero_probability_mass_raises_value_error(
      self,
  ) -> None:
    prior = roi.ImproperUniformPrior()
    with self.assertRaisesRegex(ValueError, "zero probability mass everywhere"):
      roi._evaluate_posterior(
          prior=prior,
          likelihoods=[],
          grid_min=-2.0,
          grid_max=-1.0,
          num_points=3,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="positive_support",
          has_negative_support=False,
          target_class="LogNormal",
      ),
      dict(
          testcase_name="negative_support",
          has_negative_support=True,
          target_class="Normal",
      ),
  )
  def test_fit_distribution_handles_candidate_exception(
      self, has_negative_support: bool, target_class: str
  ) -> None:
    grid_np = np.array([0.1, 0.2, 0.3])
    pdf_np = np.array([0.2, 0.6, 0.2])

    mock_target = mock.create_autospec(
        getattr(backend.tfd, target_class),
        side_effect=ValueError("Test exception"),
    )
    with mock.patch.object(backend.tfd, target_class, mock_target):
      best_dist = roi._fit_distribution(
          grid_np=grid_np,
          pdf_np=pdf_np,
          dx=0.1,
          has_negative_support=has_negative_support,
          channel_name="Search",
      )
      self.assertIsInstance(best_dist, backend.tfd.Gamma)

  def test_fit_distribution_large_negative_mass_fraction_fits_normal(
      self,
  ) -> None:
    grid_np = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    pdf_np = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    best_dist = roi._fit_distribution(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=0.5,
        has_negative_support=True,
        channel_name="Search",
    )
    self.assertIsInstance(best_dist, backend.tfd.Normal)

  def test_compute_grid_bounds_exp_scale(self) -> None:
    prior = backend.tfd.Normal(loc=1.0, scale=1.0)
    mock_exp = mock.create_autospec(roi.backend.exp)
    mock_exp.side_effect = lambda x: backend.to_tensor(  # pylint: disable=unnecessary-lambda
        np.power(2.718281828459045, np.asarray(x)), dtype=x.dtype
    )
    with mock.patch.object(roi.backend, "exp", mock_exp):
      roi._compute_grid_bounds(prior=prior, likelihoods=[])
      called_args = mock_exp.call_args[0][0]
      self.assertAlmostEqual(
          float(np.max(np.asarray(called_args))), 0.0, places=5
      )

  def test_evaluate_posterior_exp_scale(self) -> None:
    prior = backend.tfd.Normal(loc=1.0, scale=1.0)
    mock_exp = mock.create_autospec(roi.backend.exp)
    mock_exp.side_effect = lambda x: backend.to_tensor(  # pylint: disable=unnecessary-lambda
        np.power(2.718281828459045, np.asarray(x)), dtype=x.dtype
    )
    with mock.patch.object(roi.backend, "exp", mock_exp):
      roi._evaluate_posterior(
          prior=prior,
          likelihoods=[],
          grid_min=0.0,
          grid_max=2.0,
          num_points=3,
      )
      called_args = mock_exp.call_args[0][0]
      self.assertAlmostEqual(
          float(np.max(np.asarray(called_args))), 0.0, places=5
      )

  def test_evaluate_posterior_calculates_correct_values(self) -> None:
    prior = backend.tfd.Normal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    grid, pdf, dx = roi._evaluate_posterior(
        prior=prior,
        likelihoods=[],
        grid_min=0.0,
        grid_max=2.0,
        num_points=3,
    )
    self.assertEqual(dx, 1.0)
    np.testing.assert_array_almost_equal(grid, [0.0, 1.0, 2.0])
    self.assertAlmostEqual(float(np.sum(pdf) * dx), 1.0, places=5)

  def test_fit_distribution_normal_success(self) -> None:
    grid_min, grid_max = -5.0, 5.0
    grid_np = np.linspace(grid_min, grid_max, 1000)
    dx = (grid_max - grid_min) / 999
    pdf_np = np.exp(-0.5 * grid_np**2) / np.sqrt(2.0 * np.pi)
    best_dist = roi._fit_distribution(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=dx,
        has_negative_support=True,
        channel_name="Search",
    )
    self.assertIsInstance(best_dist, backend.tfd.Normal)
    self.assertAlmostEqual(float(best_dist.loc), 0.0, places=2)
    self.assertAlmostEqual(float(best_dist.scale), 1.0, places=2)

  @parameterized.named_parameters(
      dict(
          testcase_name="none",
          prior=None,
          expected=False,
      ),
      dict(
          testcase_name="improper_uniform_prior",
          prior=lambda: roi.ImproperUniformPrior(),  # pylint: disable=unnecessary-lambda
          expected=False,
      ),
      dict(
          testcase_name="normal",
          prior=lambda: backend.tfd.Normal(1.0, 1.0),  # pylint: disable=unnecessary-lambda
          expected=True,
      ),
      dict(
          testcase_name="lognormal",
          prior=lambda: backend.tfd.LogNormal(0.2, 0.9),  # pylint: disable=unnecessary-lambda
          expected=False,
      ),
      dict(
          testcase_name="invalid_mock",
          prior=None,
          expected=False,
          use_mock=True,
      ),
  )
  def test_has_negative_support(
      self,
      prior: Any,
      expected: bool,
      use_mock: bool = False,
  ) -> None:
    if use_mock:
      prior = mock.create_autospec(
          backend.tfd.Distribution, instance=True, spec_set=True
      )
    elif callable(prior):
      prior = prior()
    self.assertEqual(roi._has_negative_support(prior), expected)

  def test_fit_distribution_gamma_success(self) -> None:
    grid_min, grid_max = 0.01, 10.0
    grid_np = np.linspace(grid_min, grid_max, 1000)
    dx = (grid_max - grid_min) / 999
    pdf_np = stats.gamma.pdf(grid_np, a=2.0, scale=0.5)
    pdf_np /= np.sum(pdf_np) * dx
    best_dist = roi._fit_distribution(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=dx,
        channel_name="Search",
    )
    self.assertIsInstance(best_dist, backend.tfd.Gamma)
    self.assertAlmostEqual(float(best_dist.concentration), 2.0, places=2)
    self.assertAlmostEqual(float(best_dist.rate), 2.0, places=2)

  def test_fit_normal_helper(self) -> None:
    grid_min, grid_max = -5.0, 5.0
    grid_np = np.linspace(grid_min, grid_max, 100)
    dx = (grid_max - grid_min) / 99
    pdf_np = np.exp(-0.5 * grid_np**2) / np.sqrt(2.0 * np.pi)
    dist, loss = roi._fit_normal(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=dx,
        mean_emp=0.0,
        std_emp=1.0,
    )
    self.assertIsInstance(dist, backend.tfd.Normal)
    self.assertAlmostEqual(float(dist.loc), 0.0, places=2)
    self.assertGreaterEqual(loss, 0.0)

  def test_fit_lognormal_helper(self) -> None:
    grid_min, grid_max = 0.1, 10.0
    grid_np = np.linspace(grid_min, grid_max, 100)
    dx = (grid_max - grid_min) / 99

    pdf_np = stats.lognorm.pdf(grid_np, s=0.5, scale=np.exp(1.0))
    pdf_np /= np.sum(pdf_np) * dx
    dist, loss = roi._fit_lognormal(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=dx,
        mean_emp=1.0,
        var_emp=0.5,
    )
    self.assertIsInstance(dist, backend.tfd.LogNormal)
    self.assertGreaterEqual(loss, 0.0)

  def test_fit_gamma_helper(self) -> None:
    grid_min, grid_max = 0.1, 10.0
    grid_np = np.linspace(grid_min, grid_max, 100)
    dx = (grid_max - grid_min) / 99

    pdf_np = stats.gamma.pdf(grid_np, a=2.0, scale=0.5)
    pdf_np /= np.sum(pdf_np) * dx
    dist, loss = roi._fit_gamma(
        grid_np=grid_np,
        pdf_np=pdf_np,
        dx=dx,
        mean_emp=1.0,
        var_emp=0.5,
    )
    self.assertIsInstance(dist, backend.tfd.Gamma)
    self.assertGreaterEqual(loss, 0.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="experiment_in_past",
          experiment_end_date=datetime.date(2026, 1, 2),
          last_modeled_date=datetime.date(2027, 1, 1),
          expected_tau=1.0,
      ),
      dict(
          testcase_name="experiment_ends_exactly_on_last_modeled_date",
          experiment_end_date=datetime.date(2026, 1, 8),
          last_modeled_date=datetime.date(2026, 1, 8),
          expected_tau=0.0,
      ),
      dict(
          testcase_name="experiment_in_future",
          experiment_end_date=datetime.date(2026, 1, 15),
          last_modeled_date=datetime.date(2026, 1, 8),
          expected_tau=0.0,
      ),
  )
  def test_get_recency_adjustment(
      self,
      experiment_end_date: datetime.date,
      last_modeled_date: datetime.date,
      expected_tau: float,
  ) -> None:
    tau = roi._get_recency_adjustment(
        experiment_end_date=experiment_end_date,
        last_modeled_date=last_modeled_date,
    )
    self.assertAlmostEqual(tau, expected_tau)

  def test_get_calibrated_roi_prior_future_experiment_recency_adjustment_is_zero(
      self,
  ) -> None:
    obs = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    # Experiment ends after the last modeled date (future experiment).
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 15),
    )
    cfg = base.CalibrationData(
        experiment_result=obs, experiment_info=experiment_info
    )

    _, output = roi.get_calibrated_roi_prior(
        calibration_data=[cfg],
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
        max_lag=meridian_constants.DEFAULT_MAX_LAG,
        channel_name="Search",
        total_channel_spend=4000.0,
        interval_days=7,
        last_modeled_date=datetime.date(2026, 1, 8),
        model_duration_days=56,
    )

    exp_out = output.experiments[0]
    self.assertEqual(exp_out.tau_recency, 0.0)

  @mock.patch.object(roi.adstock_hill, "compute_decay_weights", autospec=True)
  def test_duration_adjustment_calls_compute_decay_weights_with_normalize_false(
      self, mock_compute_decay_weights
  ) -> None:
    mock_compute_decay_weights.return_value = backend.ones(
        [8], dtype=backend.float_dtype
    )
    _ = roi._duration_adjustment(
        duration=4.0,
        max_lag=8,
        adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
        alpha=0.5,
    )
    mock_compute_decay_weights.assert_called_once()
    _, kwargs = mock_compute_decay_weights.call_args
    self.assertIn("normalize", kwargs)
    self.assertFalse(kwargs["normalize"])

  @parameterized.named_parameters(
      dict(
          testcase_name="weekly_granularity",
          experiment_days=28,
          max_lag=8,
          interval_days=7,
          expected_duration=4.0,
      ),
      dict(
          testcase_name="daily_granularity",
          experiment_days=28,
          max_lag=30,
          interval_days=1,
          expected_duration=28.0,
      ),
      dict(
          testcase_name="fractional_duration_rounds_internally",
          experiment_days=10,
          max_lag=8,
          interval_days=7,
          expected_duration=10.0 / 7.0,
      ),
  )
  def test_get_duration_adjustment_and_scaling_granularity(
      self,
      experiment_days: int,
      max_lag: int,
      interval_days: int,
      expected_duration: float,
  ) -> None:
    start_date = datetime.date(2026, 1, 1)
    end_date = start_date + datetime.timedelta(days=experiment_days)

    gamma, tau = roi._get_duration_adjustment_and_scaling(
        experiment_start_date=start_date,
        experiment_end_date=end_date,
        max_lag=max_lag,
        interval_days=interval_days,
        alpha=0.5,
    )
    expected_gamma, expected_tau = roi._duration_adjustment(
        duration=expected_duration,
        max_lag=max_lag,
        alpha=0.5,
    )
    self.assertAlmostEqual(gamma, expected_gamma)
    self.assertAlmostEqual(tau, expected_tau)

  @parameterized.named_parameters(
      dict(
          testcase_name="rounded_duration_zero_or_negative",
          duration=0.0,
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          expected_p=1e-6,
      ),
      dict(
          testcase_name="decay_rate_zero",
          duration=4.0,
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.0,
          expected_p=1.0,
      ),
      dict(
          testcase_name="geometric_decay_longer_duration",
          duration=10.0,
          adstock_decay_function=meridian_constants.GEOMETRIC_DECAY,
          alpha=0.5,
          expected_p=1.0009785,
      ),
      dict(
          testcase_name="binomial_decay_longer_duration",
          duration=10.0,
          adstock_decay_function=meridian_constants.BINOMIAL_DECAY,
          alpha=0.5,
          expected_p=1.0,
      ),
  )
  def test_calculate_capture_proportion(
      self,
      duration: float,
      adstock_decay_function: str,
      alpha: float,
      expected_p: float,
  ) -> None:
    p = roi._calculate_capture_proportion(
        duration=duration,
        max_lag=8,
        adstock_decay_function=adstock_decay_function,
        alpha=alpha,
    )
    self.assertAlmostEqual(p, expected_p, places=5)

  def test_calculate_capture_proportion_invalid_decay_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "Unsupported decay function:"):
      roi._calculate_capture_proportion(
          duration=4.0,
          max_lag=8,
          adstock_decay_function="unknown_decay",
          alpha=0.5,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="with_negative_support",
          has_negative_support=True,
      ),
      dict(
          testcase_name="without_negative_support",
          has_negative_support=False,
      ),
  )
  def test_fit_distribution_raises_value_error_with_channel_name(
      self, has_negative_support: bool
  ) -> None:
    grid_np = np.linspace(0.1, 1.0, 100)
    pdf_np = np.ones_like(grid_np)

    mock_fit_normal = mock.create_autospec(roi._fit_normal, spec_set=True)
    mock_fit_normal.return_value = (None, np.inf)

    mock_fit_lognormal = mock.create_autospec(roi._fit_lognormal, spec_set=True)
    mock_fit_lognormal.return_value = (None, np.inf)

    mock_fit_gamma = mock.create_autospec(roi._fit_gamma, spec_set=True)
    mock_fit_gamma.return_value = (None, np.inf)

    with mock.patch.object(
        roi, "_fit_normal", mock_fit_normal
    ), mock.patch.object(
        roi, "_fit_lognormal", mock_fit_lognormal
    ), mock.patch.object(
        roi, "_fit_gamma", mock_fit_gamma
    ):
      with self.assertRaisesRegex(
          ValueError,
          r"Failed to fit any candidate distribution shape for channel"
          r" 'Search'\.",
      ):
        roi._fit_distribution(
            grid_np=grid_np,
            pdf_np=pdf_np,
            dx=0.01,
            has_negative_support=has_negative_support,
            channel_name="Search",
        )

  def test_compute_grid_bounds_non_tensor_coverage(self) -> None:
    prior = backend.tfd.Normal(loc=1.0, scale=0.5)

    mock_to_tensor = mock.create_autospec(roi.backend.to_tensor, spec_set=True)
    mock_to_tensor.return_value = np.linspace(-1.0, 3.0, 2000)

    mock_reduce_max = mock.create_autospec(
        roi.backend.reduce_max, spec_set=True
    )
    mock_reduce_max.return_value = 0.0

    mock_where = mock.create_autospec(roi.backend.where, spec_set=True)
    mock_where.side_effect = lambda cond, x, y: x

    mock_exp = mock.create_autospec(roi.backend.exp, spec_set=True)
    mock_exp.return_value = np.ones(2000)

    mock_reduce_sum = mock.create_autospec(
        roi.backend.reduce_sum, spec_set=True
    )
    mock_reduce_sum.return_value = 2000.0

    mock_log_prob = mock.create_autospec(prior.log_prob, spec_set=True)
    mock_log_prob.return_value = np.ones(2000)

    with mock.patch.object(
        roi.backend, "to_tensor", mock_to_tensor
    ), mock.patch.object(
        roi.backend, "reduce_max", mock_reduce_max
    ), mock.patch.object(
        roi.backend, "where", mock_where
    ), mock.patch.object(
        roi.backend, "exp", mock_exp
    ), mock.patch.object(
        roi.backend, "reduce_sum", mock_reduce_sum
    ), mock.patch.object(
        prior, "log_prob", mock_log_prob
    ):
      grid_min, grid_max = roi._compute_grid_bounds(prior=prior, likelihoods=[])
      self.assertAlmostEqual(grid_min, -1.0, places=2)
      self.assertAlmostEqual(grid_max, 3.0, places=2)


class ImproperUniformPriorTest(absltest.TestCase):

  def test_improper_uniform_prior_batch_and_event_shape_properties(
      self,
  ) -> None:
    prior = roi.ImproperUniformPrior()
    self.assertEqual(tuple(prior.batch_shape), ())
    self.assertEqual(tuple(prior.event_shape), ())

  def test_improper_uniform_prior_batch_and_event_shape_methods(self) -> None:
    prior = roi.ImproperUniformPrior()
    self.assertEqual(tuple(prior._batch_shape()), ())
    self.assertEqual(tuple(prior._event_shape()), ())

  def test_improper_uniform_prior_batch_and_event_shape_tensors(self) -> None:
    prior = roi.ImproperUniformPrior()
    self.assertEqual(list(np.asarray(prior.batch_shape_tensor())), [])
    self.assertEqual(list(np.asarray(prior.event_shape_tensor())), [])
    self.assertEqual(list(np.asarray(prior._batch_shape_tensor())), [])
    self.assertEqual(list(np.asarray(prior._event_shape_tensor())), [])

  def test_improper_uniform_prior_parameters(self) -> None:
    prior = roi.ImproperUniformPrior()
    self.assertEqual(prior.parameters["name"], "improper_uniform_prior")
    self.assertEqual(prior.parameter_properties(), {})

  def test_improper_uniform_prior_sample_raises_not_implemented_error(
      self,
  ) -> None:
    prior = roi.ImproperUniformPrior()
    with self.assertRaises(NotImplementedError):
      prior.sample(5, seed=_get_seed())

  def test_improper_uniform_prior_log_prob_and_probability(self) -> None:
    prior = roi.ImproperUniformPrior()
    self.assertEqual(float(prior.log_prob(1.5)), 0.0)
    self.assertEqual(float(prior.log_prob(-0.5)), -np.inf)
    self.assertEqual(float(prior.prob(1.5)), 1.0)
    self.assertEqual(float(prior.prob(-0.5)), 0.0)

  def test_improper_uniform_prior_custom_dtype(self) -> None:
    prior_default = roi.ImproperUniformPrior()
    self.assertEqual(
        backend.standardize_dtype(prior_default.dtype),
        backend.standardize_dtype(backend.to_tensor(1.0).dtype),
    )

    prior_explicit = roi.ImproperUniformPrior(dtype="float64")
    self.assertEqual(prior_explicit.dtype, "float64")


class GridDistributionTest(test_utils.MeridianTestCase):

  def test_grid_distribution_normalizes_pdf(self) -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    # Unnormalized PDF: sum is 3.0, dx is 1.0, so total mass is 3.0
    pdf = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dist = roi.GridDistribution(grid=grid, pdf=pdf, dx=1.0)
    # Expected normalized PDF: [1/3, 1/3, 1/3]
    self.assertAlmostEqual(np.sum(dist._pdf) * dist._dx, 1.0, places=5)  # pyrefly: ignore[no-matching-overload]
    np.testing.assert_allclose(
        dist._pdf, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], rtol=1e-5
    )

  def test_grid_distribution_sample_shape_and_limits(self) -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pdf = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    dist = roi.GridDistribution(grid=grid, pdf=pdf, dx=1.0)
    # n=100000 to reduce sampling variance/standard error
    samples = np.asarray(self.sample(dist, 100000))
    self.assertEqual(samples.shape, (100000,))
    self.assertTrue(np.all(samples >= 0.0))
    self.assertTrue(np.all(samples <= 2.0))
    # Expected mean = 0.0 * 0.2 + 1.0 * 0.5 + 2.0 * 0.3 = 1.1
    self.assertAlmostEqual(np.mean(samples), 1.1, delta=0.01)

  def test_grid_distribution_empty_arrays(self) -> None:
    dist = roi.GridDistribution(grid=np.array([]), pdf=np.array([]), dx=1.0)
    self.assertEmpty(dist._grid)
    self.assertEmpty(dist._pdf)

  def test_grid_distribution_log_prob_raises_not_implemented_error(
      self,
  ) -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pdf = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    dist = roi.GridDistribution(grid=grid, pdf=pdf, dx=1.0)
    with self.assertRaises(NotImplementedError):
      dist.log_prob(0.5)

  def test_grid_distribution_parameter_properties(self) -> None:
    properties = roi.GridDistribution.parameter_properties()
    self.assertIn("grid", properties)
    self.assertIn("pdf", properties)
    self.assertIn("dx", properties)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_total_mass",
          pdf=np.array([0.0, 0.0, 0.0]),
      ),
      dict(
          testcase_name="nan_total_mass",
          pdf=np.array([1.0, np.nan, 1.0]),
      ),
  )
  def test_grid_distribution_invalid_pdf_raises_error(
      self, pdf: np.ndarray
  ) -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError, "Invalid PDF: total mass must be finite and positive"
    ):
      roi.GridDistribution(grid=grid, pdf=pdf, dx=1.0)

  def test_grid_distribution_tf_function_compatible(self) -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pdf = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    dist = roi.GridDistribution(grid=grid, pdf=pdf, dx=1.0)

    seed = _get_seed()

    @backend.function(static_argnums=(0,))
    def run_sample(n):
      return dist.sample(n, seed=seed)

    samples = np.asarray(run_sample(10))
    self.assertEqual(samples.shape, (10,))


if __name__ == "__main__":
  absltest.main()
