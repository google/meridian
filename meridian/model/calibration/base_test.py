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

import datetime
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian.backend import test_utils
from meridian.model import prior_distribution
from meridian.model.calibration import base
import numpy as np
import tensorflow as tf


def _make_dummy_calibration_output(channel_name: str) -> base.CalibrationOutput:
  """Creates a dummy CalibrationOutput instance for testing."""
  return base.CalibrationOutput(
      channel_name=channel_name,
      baseline_prior=backend.tfd.Normal(0.0, 1.0),
      intermediary_prior=backend.tfd.Normal(0.0, 1.0),
  )


class SourceTypeTest(absltest.TestCase):

  def test_source_type_values(self) -> None:
    self.assertEqual(base.SourceType.MERIDIAN_GEOX, "MeridianGeoX")
    self.assertEqual(base.SourceType.GENERIC, "Generic")


class CalibratedExperimentTest(absltest.TestCase):

  def test_calibrated_experiment_creation(self) -> None:
    raw_res = base.ExperimentResult(point_estimate=2.0, standard_error=0.5)
    adj_res = base.ExperimentResult(point_estimate=2.5, standard_error=0.6)
    experiment = base.CalibratedExperiment(
        source_type=base.SourceType.MERIDIAN_GEOX,
        raw_experiment_result=raw_res,
        adjusted_experiment_result=adj_res,
        tau_spend=1.1,
        tau_recency=1.0,
        tau_duration=0.9,
        gamma_duration=1.2,
        user_point_estimate_adjustment=0.2,
        user_standard_error_adjustment=0.1,
    )
    expected_experiment = base.CalibratedExperiment(
        source_type=base.SourceType.MERIDIAN_GEOX,
        raw_experiment_result=raw_res,
        adjusted_experiment_result=adj_res,
        tau_spend=1.1,
        tau_recency=1.0,
        tau_duration=0.9,
        gamma_duration=1.2,
        user_point_estimate_adjustment=0.2,
        user_standard_error_adjustment=0.1,
    )
    self.assertEqual(experiment, expected_experiment)

  def test_calibrated_experiment_defaults(self) -> None:
    raw_res = base.ExperimentResult(point_estimate=2.0, standard_error=0.5)
    adj_res = base.ExperimentResult(point_estimate=2.5, standard_error=0.6)
    experiment = base.CalibratedExperiment(
        source_type=base.SourceType.MERIDIAN_GEOX,
        raw_experiment_result=raw_res,
        adjusted_experiment_result=adj_res,
        tau_spend=1.1,
        tau_recency=1.0,
        tau_duration=0.9,
        gamma_duration=1.2,
    )
    self.assertIsNone(experiment.user_point_estimate_adjustment)
    self.assertIsNone(experiment.user_standard_error_adjustment)


class CalibrationOutputTest(absltest.TestCase):

  def test_calibration_output_creation(self) -> None:
    raw_res = base.ExperimentResult(point_estimate=2.0, standard_error=0.5)
    adj_res = base.ExperimentResult(point_estimate=2.5, standard_error=0.6)
    exp = base.CalibratedExperiment(
        source_type=base.SourceType.MERIDIAN_GEOX,
        raw_experiment_result=raw_res,
        adjusted_experiment_result=adj_res,
        tau_spend=1.1,
        tau_recency=1.0,
        tau_duration=0.9,
        gamma_duration=1.2,
        user_point_estimate_adjustment=0.2,
        user_standard_error_adjustment=0.1,
    )
    base_prior = backend.tfd.Normal(0.0, 1.0)
    output = base.CalibrationOutput(
        channel_name="search",
        experiments=[exp],
        baseline_prior=base_prior,
        intermediary_prior=base_prior,
    )
    expected_output = base.CalibrationOutput(
        channel_name="search",
        experiments=[exp],
        baseline_prior=base_prior,
        intermediary_prior=base_prior,
    )
    self.assertEqual(output, expected_output)
    self.assertEqual(output.experiments, [exp])

  def test_calibration_output_defaults(self) -> None:
    base_prior = backend.tfd.Normal(0.0, 1.0)
    output = base.CalibrationOutput(
        channel_name="search",
        intermediary_prior=base_prior,
    )
    self.assertEqual(output.experiments, [])
    self.assertIsNone(output.baseline_prior)
    self.assertEqual(output.adstock_decay_spec, "geometric")
    self.assertEqual(output.max_lag, 8)

  def test_calibration_output_custom_params(self) -> None:
    base_prior = backend.tfd.Normal(0.0, 1.0)
    output = base.CalibrationOutput(
        channel_name="search",
        intermediary_prior=base_prior,
        adstock_decay_spec="binomial",
        max_lag=4,
    )
    self.assertEqual(output.adstock_decay_spec, "binomial")
    self.assertEqual(output.max_lag, 4)


class ExperimentResultTest(parameterized.TestCase):

  def test_experiment_result_creation(self) -> None:
    result = base.ExperimentResult(point_estimate=10.0, standard_error=2.5)
    self.assertEqual(result.point_estimate, 10.0)
    self.assertEqual(result.standard_error, 2.5)

  @parameterized.named_parameters(
      dict(
          testcase_name="negative_standard_error",
          standard_error=-0.5,
          expected_msg="Standard error must be positive\\. Got: -0\\.5",
      ),
      dict(
          testcase_name="zero_standard_error",
          standard_error=0.0,
          expected_msg="Standard error must be positive\\. Got: 0\\.0",
      ),
      dict(
          testcase_name="none_standard_error",
          standard_error=None,
          expected_msg="Standard error must be positive\\. Got: None",
      ),
  )
  def test_experiment_result_creation_invalid_standard_error_raises_value_error(
      self,
      standard_error: float | None,
      expected_msg: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_msg):
      base.ExperimentResult(
          point_estimate=10.0,
          standard_error=standard_error,  # pyrefly: ignore[bad-argument-type]
      )

  def test_experiment_result_creation_violates_significantly_negative(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError,
        "The experiment shows statistically significant negative lift",
    ):
      base.ExperimentResult(point_estimate=-5.0, standard_error=0.5)


class ExperimentInfoTest(parameterized.TestCase):

  def test_experiment_info_creation(self) -> None:
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2025, 1, 1),
        experiment_end_date=datetime.date(2025, 1, 10),
    )
    expected_experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2025, 1, 1),
        experiment_end_date=datetime.date(2025, 1, 10),
    )
    self.assertEqual(experiment_info, expected_experiment_info)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_spend",
          total_spend=0.0,
          expected_msg="Total spend must be positive\\. Got: 0\\.0",
      ),
      dict(
          testcase_name="negative_spend",
          total_spend=-100.0,
          expected_msg="Total spend must be positive\\. Got: -100\\.0",
      ),
      dict(
          testcase_name="none_spend",
          total_spend=None,
          expected_msg="Total spend must be positive\\. Got: None",
      ),
  )
  def test_experiment_info_invalid_spend_raises_value_error(
      self,
      total_spend: float | None,
      expected_msg: str,
  ) -> None:
    with self.assertRaisesRegex(ValueError, expected_msg):
      base.ExperimentInfo(
          total_spend=total_spend,  # pyrefly: ignore[bad-argument-type]
          experiment_start_date=datetime.date(2025, 1, 1),
          experiment_end_date=datetime.date(2025, 1, 10),
      )

  def test_experiment_info_start_date_after_end_date_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "start date must be before"):
      base.ExperimentInfo(
          total_spend=1000.0,
          experiment_start_date=datetime.date(2025, 1, 10),
          experiment_end_date=datetime.date(2025, 1, 1),
      )

  def test_experiment_info_start_date_equal_end_date_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "start date must be before"):
      base.ExperimentInfo(
          total_spend=1000.0,
          experiment_start_date=datetime.date(2025, 1, 1),
          experiment_end_date=datetime.date(2025, 1, 1),
      )

  def test_avg_daily_spend_raises_value_error_for_non_positive_duration(
      self,
  ) -> None:
    experiment_info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2025, 1, 1),
        experiment_end_date=datetime.date(2025, 1, 10),
    )
    object.__setattr__(
        experiment_info, "experiment_end_date", datetime.date(2025, 1, 1)
    )
    with self.assertRaisesRegex(
        ValueError, "Experiment duration must be at least 1 day"
    ):
      _ = experiment_info.avg_daily_spend

  def test_experiment_info_future_end_date_raises_warning(self) -> None:
    future_date = datetime.date.today() + datetime.timedelta(days=1)
    with self.assertWarnsRegex(
        UserWarning,
        f"Experiment end date \\({future_date}\\) is in the future\\. "
        "Calibration results using incomplete experiments may be unreliable "
        "or incorrect\\.",
    ):
      base.ExperimentInfo(
          total_spend=1000.0,
          experiment_start_date=datetime.date.today(),
          experiment_end_date=future_date,
      )


class CalibrationDataTest(absltest.TestCase):

  def test_calibration_data_creation(self) -> None:
    result = base.ExperimentResult(point_estimate=1.5, standard_error=0.2)
    experiment_info = base.ExperimentInfo(
        total_spend=5000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config = base.CalibrationData(
        experiment_result=result, experiment_info=experiment_info
    )
    self.assertEqual(config.experiment_result, result)
    self.assertEqual(config.experiment_info, experiment_info)
    self.assertEqual(config.source_type, base.SourceType.GENERIC)

  def test_calibration_data_creation_with_source_type(self) -> None:
    result = base.ExperimentResult(point_estimate=1.5, standard_error=0.2)
    experiment_info = base.ExperimentInfo(
        total_spend=5000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config = base.CalibrationData(
        experiment_result=result,
        experiment_info=experiment_info,
        source_type=base.SourceType.MERIDIAN_GEOX,
    )
    self.assertEqual(config.source_type, base.SourceType.MERIDIAN_GEOX)


class CalibrationInputTest(parameterized.TestCase):

  def test_calibration_input_creation_with_default_baseline_prior(self) -> None:
    ci = base.CalibrationInput(channel_name="Search", total_spend=4000.0)

    self.assertEqual(ci.channel_name, "Search")
    self.assertEqual(ci.total_spend, 4000.0)
    self.assertIsNone(ci.baseline_prior)
    self.assertEmpty(ci.configs)

  def test_calibration_input_creation_with_custom_baseline_prior(self) -> None:
    mock_prior = mock.create_autospec(
        backend.tfd.Distribution, instance=True, spec_set=True
    )
    mock_prior.mean.return_value = tf.constant(1.0)
    mock_prior.variance.return_value = tf.constant(1.0)
    mock_prior.log_prob.return_value = tf.constant([0.0])
    ci = base.CalibrationInput(
        channel_name="YouTube", total_spend=4000.0, baseline_prior=mock_prior
    )

    self.assertEqual(ci.channel_name, "YouTube")
    self.assertEqual(ci.total_spend, 4000.0)
    self.assertEqual(ci.baseline_prior, mock_prior)
    self.assertEmpty(ci.configs)

  @parameterized.named_parameters(
      dict(
          testcase_name="nan_mean",
          mean_val=np.nan,
          var_val=1.0,
          log_prob_val=[0.0],
      ),
      dict(
          testcase_name="inf_variance",
          mean_val=1.0,
          var_val=np.inf,
          log_prob_val=[0.0],
      ),
      dict(
          testcase_name="nan_log_prob",
          mean_val=1.0,
          var_val=1.0,
          log_prob_val=[np.nan],
      ),
      dict(
          testcase_name="not_implemented_mean",
          mean_val=None,
          var_val=1.0,
          log_prob_val=[0.0],
          raise_not_implemented=True,
      ),
  )
  def test_calibration_input_creation_with_invalid_baseline_prior_raises_value_error(
      self,
      mean_val,
      var_val,
      log_prob_val,
      raise_not_implemented=False,
  ) -> None:
    mock_baseline = mock.create_autospec(
        backend.tfd.Distribution, instance=True, spec_set=True
    )
    if raise_not_implemented:
      mock_baseline.mean.side_effect = NotImplementedError
    else:
      mock_baseline.mean.return_value = tf.cast(mean_val, tf.float32)
    mock_baseline.variance.return_value = tf.cast(var_val, tf.float32)
    mock_baseline.log_prob.return_value = tf.cast(log_prob_val, tf.float32)

    with self.assertRaisesRegex(ValueError, "invalid"):
      base.CalibrationInput(
          channel_name="Search",
          total_spend=4000.0,
          baseline_prior=mock_baseline,
      )

  def test_add_calibration_data_valid_succeeds(self) -> None:
    ci = base.CalibrationInput(channel_name="Search", total_spend=10000.0)
    result = base.ExperimentResult(point_estimate=1.5, standard_error=0.2)
    experiment_info = base.ExperimentInfo(
        total_spend=5000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    dummy_config = base.CalibrationData(
        experiment_result=result, experiment_info=experiment_info
    )

    ci.add_calibration_data(dummy_config)
    self.assertEqual(ci.configs, [dummy_config])

  def test_add_calibration_data_spend_exceeds_channel_spend_raises_value_error(
      self,
  ) -> None:
    ci = base.CalibrationInput(channel_name="Search", total_spend=4000.0)
    result = base.ExperimentResult(point_estimate=1.5, standard_error=0.2)
    experiment_info = base.ExperimentInfo(
        total_spend=5000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config = base.CalibrationData(
        experiment_result=result, experiment_info=experiment_info
    )

    with self.assertRaisesRegex(
        ValueError, "cannot exceed total channel spend"
    ):
      ci.add_calibration_data(config)

  def test_calibration_input_repr(self) -> None:
    ci = base.CalibrationInput(channel_name="Search", total_spend=4000.0)
    self.assertEqual(
        repr(ci),
        "CalibrationInput(channel_name='Search', total_spend=4000.0,"
        " baseline_prior=None, adstock_decay_spec='geometric',"
        " alpha=0.5)",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_spend",
          total_spend=None,
          error_msg="Total channel spend is required.",
      ),
      dict(
          testcase_name="zero_spend",
          total_spend=0.0,
          error_msg="Total channel spend must be positive.",
      ),
      dict(
          testcase_name="negative_spend",
          total_spend=-100.0,
          error_msg="Total channel spend must be positive.",
      ),
  )
  def test_calibration_input_creation_invalid_spend_raises(
      self, total_spend: Any, error_msg: str
  ) -> None:
    with self.assertRaisesRegex(ValueError, error_msg):
      base.CalibrationInput(
          channel_name="Search", total_spend=total_spend  # pyrefly: ignore[bad-argument-type]
      )

  def test_calibration_input_creation_with_non_default_decay_args(self) -> None:
    ci = base.CalibrationInput(
        channel_name="Search",
        adstock_decay_spec="binomial",
        alpha=0.8,
        total_spend=5000.0,
    )
    self.assertEqual(ci.channel_name, "Search")
    self.assertEqual(ci.adstock_decay_spec, "binomial")
    self.assertEqual(ci.alpha, 0.8)
    self.assertEqual(ci.total_spend, 5000.0)

  def test_add_multiple_calibration_data(self) -> None:
    ci = base.CalibrationInput(channel_name="Search", total_spend=4000.0)
    result1 = base.ExperimentResult(point_estimate=1.5, standard_error=0.2)
    info1 = base.ExperimentInfo(
        total_spend=100.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config1 = base.CalibrationData(
        experiment_result=result1, experiment_info=info1
    )

    result2 = base.ExperimentResult(point_estimate=2.5, standard_error=0.3)
    info2 = base.ExperimentInfo(
        total_spend=200.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config2 = base.CalibrationData(
        experiment_result=result2, experiment_info=info2
    )

    ci.add_calibration_data(config1)
    self.assertEqual(ci.configs, [config1])

    ci.add_calibration_data(config2)
    self.assertEqual(ci.configs, [config1, config2])


class CalibrationRegistryTest(absltest.TestCase):

  def test_calibration_registry_empty_init(self) -> None:
    registry = base.CalibrationRegistry()
    self.assertEmpty(registry._inputs)

  def test_calibration_registry_repr(self) -> None:
    registry = base.CalibrationRegistry()
    self.assertEqual(
        repr(registry),
        "CalibrationRegistry()",
    )

  def test_add_multiple_inputs(self) -> None:
    registry = base.CalibrationRegistry()
    ci_1 = base.CalibrationInput(channel_name="Search", total_spend=4000.0)
    ci_2 = base.CalibrationInput(channel_name="YouTube", total_spend=5000.0)

    registry.add_input(ci_1)
    self.assertEqual(registry._inputs, [ci_1])

    registry.add_input(ci_2)
    self.assertEqual(registry._inputs, [ci_1, ci_2])

  def test_get_roi_distributions_by_channel_success(self) -> None:
    registry = base.CalibrationRegistry()
    ci = base.CalibrationInput(channel_name="Search", total_spend=4000.0)
    result = base.ExperimentResult(point_estimate=2.5, standard_error=0.5)
    info = base.ExperimentInfo(
        total_spend=1000.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
    )
    config = base.CalibrationData(
        experiment_result=result, experiment_info=info
    )
    ci.add_calibration_data(config)
    registry.add_input(ci)

    result = registry.get_roi_distributions_by_channel(
        last_modeled_date=datetime.date(2026, 1, 10),
        max_lag=8,
        interval_days=7,
        model_duration_days=9,
    )
    roi_dists = result.distributions
    calibration_outputs = result.outputs
    self.assertIn("Search", roi_dists)
    self.assertIsInstance(roi_dists["Search"], backend.tfd.Gamma)

    self.assertIn("Search", calibration_outputs)
    self.assertEqual(calibration_outputs["Search"].channel_name, "Search")
    self.assertLen(calibration_outputs["Search"].experiments, 1)

  def test_get_roi_distributions_by_channel_with_baseline_prior_and_no_sources_raises_value_error(
      self,
  ) -> None:
    registry = base.CalibrationRegistry()
    mock_baseline = mock.create_autospec(
        backend.tfd.Distribution, instance=True, spec_set=True
    )
    mock_baseline.mean.return_value = tf.constant(1.0)
    mock_baseline.variance.return_value = tf.constant(1.0)
    mock_baseline.log_prob.return_value = tf.constant([0.0])

    ci = base.CalibrationInput(
        channel_name="Search",
        total_spend=4000.0,
        baseline_prior=mock_baseline,
    )
    registry.add_input(ci)

    with self.assertRaisesRegex(
        ValueError,
        (
            "Baseline prior was provided for channel 'Search', but no"
            " experiments were found\\. A baseline prior can only be used to"
            " regularize active experiment results\\."
        ),
    ):
      registry.get_roi_distributions_by_channel(
          last_modeled_date=datetime.date(2026, 1, 10),
          max_lag=8,
          interval_days=7,
          model_duration_days=10,
      )

  def test_get_roi_distributions_by_channel_is_stateless(self) -> None:
    registry = base.CalibrationRegistry()
    result = base.ExperimentResult(
        point_estimate=2.0,
        standard_error=1.0,
    )
    info = base.ExperimentInfo(
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
        total_spend=2000.0,
    )
    config = base.CalibrationData(
        experiment_result=result, experiment_info=info
    )
    ci = base.CalibrationInput(
        channel_name="Search",
        total_spend=4000.0,
    )
    ci.add_calibration_data(config)
    registry.add_input(ci)

    # First call with date 1
    res_1 = registry.get_roi_distributions_by_channel(
        last_modeled_date=datetime.date(2026, 1, 10),
        max_lag=8,
        interval_days=7,
        model_duration_days=9,
    )
    dists_1 = res_1.distributions
    outputs_1 = res_1.outputs
    # Second call with date 2
    res_2 = registry.get_roi_distributions_by_channel(
        last_modeled_date=datetime.date(2026, 3, 5),
        max_lag=8,
        interval_days=7,
        model_duration_days=63,
    )
    dists_2 = res_2.distributions
    outputs_2 = res_2.outputs

    self.assertIn("Search", dists_1)
    self.assertIn("Search", dists_2)
    self.assertIn("Search", outputs_1)
    self.assertIn("Search", outputs_2)
    self.assertFalse(hasattr(registry, "_outputs"))


class TestCalibratedDistribution(test_utils.MeridianTestCase):

  def test_parameter_properties(self):
    properties = base.CalibratedDistribution._parameter_properties(np.float32)
    self.assertIn("distributions", properties)
    self.assertIsInstance(
        properties["distributions"], backend.util.BatchedComponentProperties
    )

  def test_default_name(self):
    distributions = [
        backend.tfd.Normal(0, 1, name="Normal1"),
        backend.tfd.Normal([0, 0], [1, 1], name="Normal2"),
    ]
    dist = base.CalibratedDistribution(
        distributions, is_calibrated=[False, False, False]
    )
    self.assertTrue(dist.name.startswith("Calibrated"))
    self.assertIn("Normal1", dist.name)
    self.assertIn("Normal2", dist.name)

  def test_default_name_single_distribution(self):
    dist_single = backend.tfd.Normal(0, 1, name="MyNormal")
    dist = base.CalibratedDistribution(
        dist_single, is_calibrated=[False, False, False]
    )
    self.assertTrue(dist.name.startswith("Calibrated"))
    # BatchBroadcast might change the name or append to it, but it should
    # contain the original name
    self.assertIn("MyNormal", dist.name)

  def test_initialization_default(self):
    distributions = [
        backend.tfd.Normal(0, 1),
        backend.tfd.Normal([0, 0], [1, 1]),
    ]
    dist = base.CalibratedDistribution(
        distributions, is_calibrated=[False, False, False]
    )
    self.assertEqual(
        dist.get_calibration_status(),
        (False, False, False),
    )
    self.assertEqual(dist.calibration_outputs, (None, None, None))

  def test_initialization_custom(self):
    distributions = [
        backend.tfd.Normal(0, 1),
        backend.tfd.Normal([0, 0], [1, 1]),
    ]
    is_calibrated = [True, False, True]
    outputs = [
        _make_dummy_calibration_output("channel1"),
        None,
        _make_dummy_calibration_output("channel2"),
    ]
    dist = base.CalibratedDistribution(
        distributions,
        is_calibrated=is_calibrated,
        calibration_outputs=outputs,
    )
    self.assertEqual(dist.get_calibration_status(), tuple(is_calibrated))
    self.assertEqual(dist.calibration_outputs, tuple(outputs))
    self.assertIn("is_calibrated", dist.parameters)
    self.assertIn("calibration_outputs", dist.parameters)

  def test_initialization_single_scalar_distribution(self):
    distribution = backend.tfd.Normal(0, 1)
    is_calibrated = [True, False, True]
    dist = base.CalibratedDistribution(
        distribution,
        is_calibrated=is_calibrated,
    )
    self.assertEqual(dist.get_calibration_status(), tuple(is_calibrated))
    self.assertEqual(dist.batch_shape, (3,))

  def test_initialization_single_broadcasted_distribution(self):
    distribution = backend.tfd.Normal([0, 0, 0], [1, 1, 1])
    is_calibrated = [True, False, True]
    dist = base.CalibratedDistribution(
        distribution,
        is_calibrated=is_calibrated,
    )
    self.assertEqual(dist.get_calibration_status(), tuple(is_calibrated))
    self.assertEqual(dist.batch_shape, (3,))

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_is_calibrated_length",
          is_calibrated=[True, False],
          calibration_outputs=None,
          expected_error_regex=(
              r"is_calibrated length \(2\) must match total number of channels"
              r" \(3\)\."
          ),
      ),
      dict(
          testcase_name="invalid_calibration_outputs_length",
          is_calibrated=[True, False, True],
          calibration_outputs=[
              "channel1",
              None,
          ],
          expected_error_regex=(
              r"calibration_outputs length \(2\) must match total number of"
              r" channels \(3\)\."
          ),
      ),
      dict(
          testcase_name="inconsistent_calibration_metadata",
          is_calibrated=[True, False, True],
          calibration_outputs=[
              "channel1",
              "channel2",
              "channel3",
          ],
          expected_error_regex=(
              r"Channel 1 has a non-None calibration output but is_calibrated"
              r" is False\."
          ),
      ),
  )
  def test_invalid_initialization_raises_error(
      self, is_calibrated, calibration_outputs, expected_error_regex
  ):
    if calibration_outputs is not None:
      calibration_outputs = [
          _make_dummy_calibration_output(c) if c is not None else None
          for c in calibration_outputs
      ]
    distributions = [
        backend.tfd.Normal(0, 1),
        backend.tfd.Normal([0, 0], [1, 1]),
    ]
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      _ = base.CalibratedDistribution(
          distributions,
          is_calibrated=is_calibrated,
          calibration_outputs=calibration_outputs,
      )

  def test_sampling_and_log_prob(self):
    distributions = [
        backend.tfd.Normal(0, 1),
        backend.tfd.Normal([0, 0], [1, 1]),
    ]
    dist = base.CalibratedDistribution(
        distributions, is_calibrated=[False, False, False]
    )
    sample = self.sample(dist)
    self.assertEqual(sample.shape, (3,))
    log_prob = dist.log_prob([0.0, 0.0, 0.0])
    self.assertFalse(np.isnan(log_prob).any())

  def test_distributions_are_equal(self):
    distributions1 = [backend.tfd.Normal(0.0, 1.0)]
    dist1 = base.CalibratedDistribution(
        distributions1,
        is_calibrated=[True],
        calibration_outputs=[_make_dummy_calibration_output("channel1")],
    )
    dist2 = base.CalibratedDistribution(
        distributions1,
        is_calibrated=[True],
        calibration_outputs=[_make_dummy_calibration_output("channel1")],
    )
    self.assertTrue(prior_distribution.distributions_are_equal(dist1, dist2))

    dist_diff_calib = base.CalibratedDistribution(
        distributions1,
        is_calibrated=[True],
        calibration_outputs=[_make_dummy_calibration_output("channel2")],
    )
    # Mathematical equality ignores calibration metadata.
    self.assertTrue(
        prior_distribution.distributions_are_equal(dist1, dist_diff_calib)
    )

    dist_diff_is_calibrated = base.CalibratedDistribution(
        distributions1,
        is_calibrated=[False],
        calibration_outputs=[None],
    )
    # Mathematical equality ignores calibration metadata.
    self.assertTrue(
        prior_distribution.distributions_are_equal(
            dist1, dist_diff_is_calibrated
        )
    )

    dist_diff_math = base.CalibratedDistribution(
        [backend.tfd.Normal(0, 2)],
        is_calibrated=[True],
    )
    self.assertFalse(
        prior_distribution.distributions_are_equal(dist1, dist_diff_math)
    )


if __name__ == "__main__":
  absltest.main()
