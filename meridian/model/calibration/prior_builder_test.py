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

from collections.abc import Sequence
import datetime
import sys
from typing import Any, TYPE_CHECKING, get_type_hints
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.data import arg_builder
from meridian.data import input_data
from meridian.data import time_coordinates
from meridian.model import prior_distribution
from meridian.model.calibration import base

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=g-bad-import-order
  # pylint: disable=unused-import
  from meridian_geox import api as geox_api
  # pylint: enable=g-import-not-at-top
  # pylint: enable=g-bad-import-order
  # pylint: enable=unused-import
  _HAS_GEOX = True
else:  # pylint: disable=unreachable
  try:
    # pylint: disable=g-import-not-at-top
    # pylint: disable=g-bad-import-order
    # pylint: disable=unused-import
    from meridian_geox import api as geox_api
    # pylint: enable=g-import-not-at-top
    # pylint: enable=g-bad-import-order
    # pylint: enable=unused-import
    _HAS_GEOX = True
  except ImportError:

    class DummyContainer:

      def __init__(self, **kwargs):
        for k, v in kwargs.items():
          setattr(self, k, v)

    # pylint: disable=invalid-name
    class GeoxApiModuleSpec:
      Estimate = DummyContainer
      DescriptiveMetrics = DummyContainer
      AnalysisMetrics = DummyContainer
      AnalysisConfig = DummyContainer
      AnalysisResult = DummyContainer

    # pylint: enable=invalid-name

    mock_api = mock.create_autospec(
        GeoxApiModuleSpec, instance=True, spec_set=True
    )
    mock_api.Estimate = DummyContainer
    mock_api.DescriptiveMetrics = DummyContainer
    mock_api.AnalysisMetrics = DummyContainer
    mock_api.AnalysisConfig = DummyContainer
    mock_api.AnalysisResult = DummyContainer

    mock_parent = mock.MagicMock()
    mock_parent.api = mock_api

    sys.modules["meridian_geox"] = mock_parent
    sys.modules["meridian_geox.api"] = mock_api
    geox_api = mock_api
    _HAS_GEOX = True

# pylint: disable=g-import-not-at-top
from meridian.model.calibration import constants as calibration_constants
from meridian.model.calibration import prior_builder
import meridian.model.calibration.adapters.meridian_geox as geox_adapter

# pylint: enable=g-import-not-at-top
import numpy as np
import xarray as xr


def _create_mock_geox_result() -> geox_api.AnalysisResult:
  """Creates a mock AnalysisResult for testing.

  Returns:
    A mock `meridian_geox.api.AnalysisResult` object.
  """
  estimate = geox_api.Estimate(
      point_estimate=2.5,
      lower_bound=1.5,
      upper_bound=3.5,
      standard_deviation=0.5,
      p_value=0.05,
  )
  desc_metrics = geox_api.DescriptiveMetrics(estimated_bau_spend=1000.0)

  lift_estimate = geox_api.Estimate(
      point_estimate=10.0,
      lower_bound=5.0,
      upper_bound=15.0,
      standard_deviation=2.5,
      p_value=0.05,
  )
  cell_metrics = geox_api.AnalysisMetrics(
      lift=lift_estimate,
      percent_lift=lift_estimate,
      icpd=estimate,
      descriptive_metrics=desc_metrics,
  )

  analysis_config = mock.create_autospec(
      geox_api.AnalysisConfig, instance=True, spec_set=True
  )
  type(analysis_config).analysis_start_date = mock.PropertyMock(
      return_value=datetime.datetime(2026, 1, 1)
  )
  type(analysis_config).analysis_end_date = mock.PropertyMock(
      return_value=datetime.datetime(2026, 1, 10)
  )

  return geox_api.AnalysisResult(
      results={"cell_1": cell_metrics},
      analysis_config=analysis_config,
  )


def _get_underlying_distributions(
    joint_prior: prior_distribution.IndependentMultivariateDistribution,
) -> Sequence[backend.tfd.Distribution]:
  """Extracts underlying univariate distributions from a joint prior.

  Args:
    joint_prior: The joint prior distribution.

  Returns:
    A sequence of univariate distributions.
  """
  return [
      d.distribution if hasattr(d, "distribution") else d
      for d in joint_prior._distributions
  ]


class CalibrationBuilderTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.mock_input_data = mock.create_autospec(
        input_data.InputData, instance=True, spec_set=True
    )
    mock_time_coords = mock.create_autospec(
        time_coordinates.TimeCoordinates, instance=True, spec_set=True
    )
    mock_time_coords.all_dates = [
        datetime.date(2026, 1, 3),
        datetime.date(2026, 1, 10),
        datetime.date(2026, 1, 17),
    ]
    type(mock_time_coords).interval_days = mock.PropertyMock(return_value=7)
    type(self.mock_input_data).time_coordinates = mock.PropertyMock(
        return_value=mock_time_coords
    )
    type(self.mock_input_data).kpi_type = mock.PropertyMock(
        return_value=constants.REVENUE
    )
    self.mock_input_data.revenue_per_kpi = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    self.mock_input_data.revenue_per_kpi.mean.return_value = 2.0
    self.mock_input_data.get_all_paid_channels.return_value = [
        "Search",
        "YouTube",
    ]

    mock_media_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_media_channel.values = ["Search", "YouTube"]
    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=mock_media_channel
    )

    mock_rf_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_rf_channel.values = []
    type(self.mock_input_data).rf_channel = mock.PropertyMock(
        return_value=mock_rf_channel
    )

    mock_media_spend = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_media_spend.sel.return_value.sum.return_value = 4000.0
    type(self.mock_input_data).media_spend = mock.PropertyMock(
        return_value=mock_media_spend
    )

  def test_init_without_registry_creates_default_registry(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    self.assertIsInstance(builder._registry, base.CalibrationRegistry)

  def test_init_invalid_adstock_decay_spec_string_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "Invalid 'adstock_decay_spec'"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data, adstock_decay_spec="invalid_decay"
      )

  def test_init_invalid_adstock_decay_spec_mapping_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "Invalid 'adstock_decay_spec'"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          adstock_decay_spec={"Search": "invalid_decay"},
      )

  def test_init_invalid_adstock_decay_spec_type_raises_type_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        TypeError, "must be either a string or a Mapping"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          adstock_decay_spec=123,  # pytype: disable=wrong-arg-types
      )

  def test_init_adstock_decay_spec_invalid_channel_mapping_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, "Invalid channels in 'adstock_decay_spec' mapping"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          adstock_decay_spec={"InvalidChannel": "geometric"},
      )

  def test_init_alpha_invalid_channel_mapping_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, "Invalid channels in 'alpha' mapping"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data, alpha={"InvalidChannel": 0.5}
      )

  def test_init_invalid_max_lag_type_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "'max_lag' must be a non-negative integer"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          max_lag="invalid",  # pytype: disable=wrong-arg-types
      )

    with self.assertRaisesRegex(
        ValueError, "'max_lag' must be a non-negative integer"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          max_lag=True,  # pytype: disable=wrong-arg-types
      )

  def test_init_invalid_max_lag_bounds_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, "'max_lag' must be a non-negative integer"
    ):
      prior_builder.CalibrationBuilder(self.mock_input_data, max_lag=-1)

  def test_init_invalid_alpha_value_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "Alpha must be between 0.0 and 1.0"
    ):
      prior_builder.CalibrationBuilder(self.mock_input_data, alpha=1.5)

  def test_init_invalid_alpha_mapping_value_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "Alpha must be between 0.0 and 1.0"
    ):
      prior_builder.CalibrationBuilder(
          self.mock_input_data, alpha={"Search": -0.5}
      )

  def test_init_invalid_alpha_mapping_type_raises_type_error(self) -> None:
    with self.assertRaisesRegex(TypeError, "Alpha values must be numeric"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          alpha={"Search": "invalid"},  # pytype: disable=wrong-arg-types
      )

  def test_init_invalid_alpha_type_raises_type_error(self) -> None:
    with self.assertRaisesRegex(TypeError, "alpha must be either a float"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          alpha=[0.5],  # pytype: disable=wrong-arg-types
      )

  def test_init_alpha_boolean_value_raises_type_error(self) -> None:
    with self.assertRaisesRegex(TypeError, "alpha must be either a float"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          alpha=True,  # pytype: disable=wrong-arg-types
      )

  def test_init_alpha_mapping_boolean_value_raises_type_error(self) -> None:
    with self.assertRaisesRegex(TypeError, "Alpha values must be numeric"):
      prior_builder.CalibrationBuilder(
          self.mock_input_data,
          alpha={"Search": False},  # pytype: disable=wrong-arg-types
      )

  def _create_mock_distribution(self) -> mock.Mock:
    """Helper to create a configured mock Distribution instance."""
    mock_dist = mock.create_autospec(
        backend.tfd.Distribution, instance=True, spec_set=True
    )
    mock_dist.mean.return_value = np.array(1.0)
    mock_dist.variance.return_value = np.array(1.0)
    mock_dist.log_prob.return_value = np.array([0.0])
    return mock_dist

  @parameterized.named_parameters(
      dict(
          testcase_name="geometric_decay",
          decay_function="geometric",
          decay_rate=0.8,
          max_lag=10,
      ),
      dict(
          testcase_name="binomial_decay",
          decay_function="binomial",
          decay_rate=0.4,
          max_lag=6,
      ),
  )
  def test_builder_initializes_with_decay_parameters(
      self,
      decay_function: str,
      decay_rate: float,
      max_lag: int,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        adstock_decay_spec=decay_function,
        alpha=decay_rate,
        max_lag=max_lag,
    )
    self.assertEqual(builder._max_lag, max_lag)
    self.assertEqual(builder._adstock_decay_spec, decay_function)
    self.assertEqual(builder._alpha, decay_rate)

  def test_builder_resolves_alphas_mapping_with_fallback(self) -> None:
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        alpha={"Search": 0.8},
    )
    self.assertEqual(builder._resolved_alphas["Search"], 0.8)
    self.assertEqual(
        builder._resolved_alphas["YouTube"],
        calibration_constants.DEFAULT_ALPHA,
    )

  def test_builder_resolves_decay_specs_mapping_with_fallback(self) -> None:
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        adstock_decay_spec={"Search": "binomial"},
    )
    self.assertEqual(
        builder._resolved_adstock_decay_specs["Search"], "binomial"
    )
    self.assertEqual(
        builder._resolved_adstock_decay_specs["YouTube"],
        constants.GEOMETRIC_DECAY,
    )

  def test_builder_initializes_with_channels_baseline_prior_map(self) -> None:
    mock_baseline = self._create_mock_distribution()
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        baseline_prior={"Search": mock_baseline},
    )
    container = builder._get_or_create_channel_container("Search")
    self.assertEqual(container.baseline_prior, mock_baseline)

  def test_builder_initializes_with_single_baseline_prior(self) -> None:
    mock_baseline = self._create_mock_distribution()
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        baseline_prior=mock_baseline,
    )
    container = builder._get_or_create_channel_container("Search")
    self.assertEqual(container.baseline_prior, mock_baseline)

  def test_builder_initializes_with_single_custom_prior(self) -> None:
    search_custom = backend.tfd.LogNormal(
        loc=backend.cast(0.5, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        custom_prior=search_custom,
    )
    self.assertEqual(builder._resolved_custom_priors["Search"], search_custom)
    self.assertEqual(builder._resolved_custom_priors["YouTube"], search_custom)

  # ============================================================================
  # SECTION 1: GeoX Experiments Feature Block
  # ============================================================================

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_adds_source(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res = _create_mock_geox_result()

    result = builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res,
        experiment_kpi_types=constants.REVENUE,
    )

    self.assertEqual(result, builder)
    container = builder._container_by_channel["Search"]
    self.assertLen(container.configs, 1)
    self.assertIsInstance(container.configs[0], base.CalibrationData)
    self.assertEqual(builder._registry._inputs, [container])

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_sequence_adds_source(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res_1 = _create_mock_geox_result()
    mock_res_2 = _create_mock_geox_result()

    result = builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=[mock_res_1, mock_res_2],
        experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
    )

    self.assertEqual(result, builder)
    container = builder._container_by_channel["Search"]
    self.assertLen(container.configs, 2)
    self.assertEqual(
        [type(s) for s in container.configs],
        [base.CalibrationData, base.CalibrationData],
    )
    self.assertEqual(builder._registry._inputs, [container])

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_multiple_calls_append_sources(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res_1 = _create_mock_geox_result()
    mock_res_2 = _create_mock_geox_result()

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res_1,
        experiment_kpi_types=constants.REVENUE,
    )

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res_2,
        experiment_kpi_types=constants.NON_REVENUE,
    )

    container = builder._container_by_channel["Search"]
    self.assertLen(
        container.configs,
        2,
        msg="Calibration builder container should store exactly 2 configs.",
    )

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_multiple_calls_preserve_attributes(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res_1 = _create_mock_geox_result()
    mock_res_2 = _create_mock_geox_result()

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res_1,
        experiment_kpi_types=constants.REVENUE,
    )

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res_2,
        experiment_kpi_types=constants.NON_REVENUE,
    )

    container = builder._container_by_channel["Search"]
    config1 = container.configs[0]
    self.assertIsInstance(config1, base.CalibrationData)
    self.assertEqual(
        config1.experiment_result.point_estimate,
        2.5,
        msg="The first config should match REVENUE.",
    )
    config2 = container.configs[1]
    self.assertIsInstance(config2, base.CalibrationData)
    self.assertEqual(
        config2.experiment_result.point_estimate,
        5.0,
        msg="The second config should match NON_REVENUE.",
    )

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_invalid_type_raises_error(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    invalid_results: Any = 123
    with self.assertRaisesRegex(
        ValueError,
        "`geox_results` must be a single `AnalysisResult` or sequence.",
    ):
      builder.with_meridian_geox_experiment_result(
          channel_name="Search",
          geox_results=invalid_results,
          experiment_kpi_types=constants.REVENUE,
      )

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_empty_results_raises_error(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res = _create_mock_geox_result()
    empty_res = geox_api.AnalysisResult(
        results={},
        analysis_config=mock_res.analysis_config,
    )

    with self.assertRaisesRegex(
        geox_adapter.InvalidGeoXResultError,
        "GeoX AnalysisResult contains no results.",
    ):
      builder.with_meridian_geox_experiment_result(
          channel_name="Search",
          geox_results=empty_res,
          experiment_kpi_types=constants.REVENUE,
      )

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_mixed_kpi_types_add_sources(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res_1 = _create_mock_geox_result()
    mock_res_2 = _create_mock_geox_result()

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=[mock_res_1, mock_res_2],
        experiment_kpi_types=[constants.REVENUE, constants.NON_REVENUE],
    )

    container = builder._container_by_channel["Search"]
    self.assertLen(container.configs, 2)
    config1 = container.configs[0]
    self.assertIsInstance(config1, base.CalibrationData)
    self.assertEqual(config1.experiment_result.point_estimate, 2.5)
    config2 = container.configs[1]
    self.assertIsInstance(config2, base.CalibrationData)
    self.assertEqual(config2.experiment_result.point_estimate, 5.0)

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_integration_resolves_configs(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)

    estimate = geox_api.Estimate(
        point_estimate=2.5,
        lower_bound=1.5,
        upper_bound=3.5,
        standard_deviation=0.5,
        p_value=0.05,
    )
    desc_metrics = geox_api.DescriptiveMetrics(estimated_bau_spend=1000.0)

    lift_estimate = geox_api.Estimate(
        point_estimate=10.0,
        lower_bound=5.0,
        upper_bound=15.0,
        standard_deviation=2.5,
        p_value=0.05,
    )
    cell_metrics = geox_api.AnalysisMetrics(
        lift=lift_estimate,
        percent_lift=lift_estimate,
        icpd=estimate,
        descriptive_metrics=desc_metrics,
    )

    analysis_config = mock.create_autospec(
        geox_api.AnalysisConfig, instance=True, spec_set=True
    )
    type(analysis_config).analysis_start_date = mock.PropertyMock(
        return_value=datetime.datetime(2026, 1, 1)
    )
    type(analysis_config).analysis_end_date = mock.PropertyMock(
        return_value=datetime.datetime(2026, 1, 10)
    )

    mock_res = geox_api.AnalysisResult(
        results={"cell_1": cell_metrics},
        analysis_config=analysis_config,
    )

    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res,
        experiment_kpi_types=constants.REVENUE,
    )

    container = builder._container_by_channel["Search"]
    resolved = container.configs

    expected = base.CalibrationData(
        experiment_result=base.ExperimentResult(
            point_estimate=2.5, standard_error=0.5
        ),
        experiment_info=base.ExperimentInfo(
            total_spend=1000.0,
            experiment_start_date=datetime.date(2026, 1, 1),
            experiment_end_date=datetime.date(2026, 1, 10),
        ),
        source_type=base.SourceType.MERIDIAN_GEOX,
    )
    self.assertEqual(resolved, [expected])

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_with_meridian_geox_experiment_result_adjustments_are_preserved(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res = _create_mock_geox_result()
    builder.with_meridian_geox_experiment_result(
        channel_name="Search",
        geox_results=mock_res,
        experiment_kpi_types=constants.REVENUE,
        point_estimate_adjustments=0.5,
        standard_error_adjustments=0.6,
    )
    container = builder._container_by_channel["Search"]
    resolved = container.configs
    self.assertLen(resolved, 1)
    self.assertEqual(resolved[0].point_estimate_adjustment, 0.5)
    self.assertEqual(resolved[0].standard_error_adjustment, 0.6)

  def test_with_meridian_geox_experiment_result_type_hints_resolve(
      self,
  ) -> None:
    # Verifies type hints can be resolved at runtime
    # (e.g. by builders or doc-generators)
    # without NameError when optional modules are not imported.
    hints = get_type_hints(
        prior_builder.CalibrationBuilder.with_meridian_geox_experiment_result
    )
    self.assertIn("geox_results", hints)

  # ============================================================================
  # SECTION 2: Incrementality Experiments Block
  # ============================================================================

  # Happy path tests for incrementality experiment results.
  @parameterized.named_parameters(
      dict(
          testcase_name="single_point_estimate",
          point_estimates=0.8,
          standard_errors=0.1,
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=1000.0,
          experiment_start_dates="2026-01-01",
          experiment_end_dates="2026-01-10",
          expected_source_count=1,
      ),
      dict(
          testcase_name="multiple_point_estimates",
          point_estimates=[0.8, 0.9],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[1000.0, 2000.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          expected_source_count=2,
      ),
  )
  def test_with_incrementality_experiment_result_adds_sources(
      self,
      point_estimates,
      standard_errors,
      experiment_kpi_types,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_dates,
      expected_source_count,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    result = builder.with_incrementality_experiment_result(
        channel_name="YouTube",
        point_estimates=point_estimates,
        standard_errors=standard_errors,
        experiment_kpi_types=experiment_kpi_types,
        experiment_total_spends=experiment_total_spends,
        experiment_start_dates=experiment_start_dates,
        experiment_end_dates=experiment_end_dates,
    )

    self.assertEqual(result, builder)
    container = builder._container_by_channel["YouTube"]
    self.assertLen(container.configs, expected_source_count)
    self.assertEqual(
        [type(s) for s in container.configs],
        [base.CalibrationData] * expected_source_count,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_experiment_total_spends",
          experiment_total_spends=None,
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          expected_error="Expected float, int, or sequence",
      ),
      dict(
          testcase_name="none_element_in_experiment_total_spends",
          experiment_total_spends=[100.0, None],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          expected_error=(
              "None/Booleans are not allowed for required float fields"
          ),
      ),
  )
  def test_with_incrementality_experiment_result_none_args_raises_type_error(
      self,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_dates,
      expected_error,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(TypeError, expected_error):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=experiment_total_spends,  # pyrefly: ignore[bad-argument-type]
          experiment_start_dates=experiment_start_dates,  # pyrefly: ignore[bad-argument-type]
          experiment_end_dates=experiment_end_dates,  # pyrefly: ignore[bad-argument-type]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_experiment_start_dates",
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=None,
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          expected_error="Unsupported date value type: .*NoneType.* for None",
      ),
      dict(
          testcase_name="none_experiment_end_dates",
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=None,
          expected_error="Unsupported date value type: .*NoneType.* for None",
      ),
      dict(
          testcase_name="none_element_in_experiment_start_dates",
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", None],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          expected_error="Unsupported date value type: .*NoneType.* for None",
      ),
      dict(
          testcase_name="none_element_in_experiment_end_dates",
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=[None, "2026-01-11"],
          expected_error="Unsupported date value type: .*NoneType.* for None",
      ),
  )
  def test_with_incrementality_experiment_result_none_args_raises_value_error(
      self,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_dates,
      expected_error,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, expected_error):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=experiment_total_spends,  # pyrefly: ignore[bad-argument-type]
          experiment_start_dates=experiment_start_dates,  # pyrefly: ignore[bad-argument-type]
          experiment_end_dates=experiment_end_dates,  # pyrefly: ignore[bad-argument-type]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_string_dates",
          point_estimates=1.0,
          standard_errors=0.1,
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=100.0,
          experiment_start_dates="2026-01-01",
          experiment_end_dates="2026-01-05",
          expected_source_count=1,
      ),
      dict(
          testcase_name="single_datetime_dates",
          point_estimates=1.0,
          standard_errors=0.1,
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=100.0,
          experiment_start_dates=datetime.date(2026, 1, 1),
          experiment_end_dates=datetime.date(2026, 1, 5),
          expected_source_count=1,
      ),
      dict(
          testcase_name="list_of_string_dates",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-05", "2026-01-07"],
          expected_source_count=2,
      ),
      dict(
          testcase_name="list_of_datetime_dates",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=[
              datetime.date(2026, 1, 1),
              datetime.date(2026, 1, 2),
          ],
          experiment_end_dates=[
              datetime.date(2026, 1, 5),
              datetime.date(2026, 1, 7),
          ],
          expected_source_count=2,
      ),
      dict(
          testcase_name="mixed_string_and_datetime_dates",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=[
              datetime.date(2026, 1, 1),
              "2026-01-02",
          ],
          experiment_end_dates=[
              "2026-01-05",
              datetime.date(2026, 1, 7),
          ],
          expected_source_count=2,
      ),
  )
  def test_with_incrementality_experiment_result_date_formats_add_sources(
      self,
      point_estimates,
      standard_errors,
      experiment_kpi_types,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_dates,
      expected_source_count,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    result = builder.with_incrementality_experiment_result(
        channel_name="Search",
        point_estimates=point_estimates,
        standard_errors=standard_errors,
        experiment_kpi_types=experiment_kpi_types,
        experiment_total_spends=experiment_total_spends,
        experiment_start_dates=experiment_start_dates,
        experiment_end_dates=experiment_end_dates,
    )

    self.assertEqual(result, builder)
    container = builder._container_by_channel["Search"]
    self.assertLen(container.configs, expected_source_count)
    self.assertEqual(
        [type(s) for s in container.configs],
        [base.CalibrationData] * expected_source_count,
    )

  def test_with_incrementality_experiment_result_integration_resolves_configs(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)

    mock_revenue_per_kpi = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_revenue_per_kpi.mean.return_value = 2.0
    self.mock_input_data.revenue_per_kpi = mock_revenue_per_kpi

    builder.with_incrementality_experiment_result(
        channel_name="Search",
        point_estimates=[2.5],
        standard_errors=[0.5],
        experiment_kpi_types=[constants.NON_REVENUE],
        experiment_total_spends=[1000.0],
        experiment_start_dates=["2026-01-01"],
        experiment_end_dates=["2026-01-10"],
    )

    container = builder._container_by_channel["Search"]
    resolved = container.configs

    expected = base.CalibrationData(
        experiment_result=base.ExperimentResult(
            point_estimate=5.0, standard_error=1.0
        ),
        experiment_info=base.ExperimentInfo(
            total_spend=1000.0,
            experiment_start_date=datetime.date(2026, 1, 1),
            experiment_end_date=datetime.date(2026, 1, 10),
        ),
    )
    self.assertEqual(resolved, [expected])

  # Sad path tests for incrementality experiment results.
  def test_with_incrementality_experiment_result_invalid_floats_raises_error(
      self,
  ):
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "must be positive"):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, -0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_spends",
          experiment_total_spends=[100.0, 0.0],
          expected_error="Total spend must be positive",
      ),
      dict(
          testcase_name="negative_spends",
          experiment_total_spends=[100.0, -100.0],
          expected_error="Total spend must be positive",
      ),
      dict(
          testcase_name="exceeding_channel_spend",
          experiment_total_spends=[5000.0, 100.0],
          expected_error="cannot exceed total channel spend",
      ),
  )
  def test_with_incrementality_experiment_result_invalid_spends_raises_error(
      self,
      experiment_total_spends,
      expected_error,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, expected_error):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=experiment_total_spends,
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
      )

  def test_with_incrementality_experiment_result_end_date_before_start_date_raises_error(
      self,
  ):
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(
        ValueError, "start date must be before the experiment end date"
    ):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0],
          standard_errors=[0.1],
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=[1000.0],
          experiment_start_dates="2026-01-05",
          experiment_end_dates="2026-01-01",
      )

  def test_with_incrementality_experiment_result_zero_stderror_raises_value_error(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(
        ValueError,
        "Standard error must be positive\\. Got: 0\\.0",
    ):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0],
          standard_errors=[0.0],
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=[1000.0],
          experiment_start_dates=["2026-01-01"],
          experiment_end_dates=["2026-01-10"],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_in_point_estimates",
          point_estimates=[1.0, None],
          standard_errors=[0.1, 0.1],
          experiment_total_spends=[100.0, 200.0],
      ),
      dict(
          testcase_name="none_in_standard_errors",
          point_estimates=[1.0],
          standard_errors=[None],
          experiment_total_spends=[100.0],
      ),
      dict(
          testcase_name="none_in_total_spends",
          point_estimates=[1.0],
          standard_errors=[0.1],
          experiment_total_spends=[None],
      ),
  )
  def test_incrementality_none_or_boolean_coercion_raises_type_error(
      self,
      point_estimates,
      standard_errors,
      experiment_total_spends,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)

    dates_len = len(point_estimates)
    start_dates = ["2026-01-01"] * dates_len
    end_dates = ["2026-01-10"] * dates_len
    kpis = [constants.REVENUE] * dates_len

    with self.assertRaisesRegex(TypeError, "None/Booleans are not allowed"):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=point_estimates,  # pyrefly: ignore[bad-argument-type]
          standard_errors=standard_errors,  # pyrefly: ignore[bad-argument-type]
          experiment_kpi_types=kpis,
          experiment_total_spends=experiment_total_spends,  # pyrefly: ignore[bad-argument-type]
          experiment_start_dates=start_dates,
          experiment_end_dates=end_dates,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_date_type_list",
          point_estimates=[1.0],
          standard_errors=[0.1],
          experiment_kpi_types=[constants.REVENUE],
          experiment_total_spends=[100.0],
          experiment_start_dates=["123456"],
          experiment_end_dates=["2026-01-10"],
          expected_error="does not match format",
      ),
      dict(
          testcase_name="invalid_date_format_str",
          point_estimates=[1.0],
          standard_errors=[0.1],
          experiment_kpi_types=[constants.REVENUE],
          experiment_total_spends=[100.0],
          experiment_start_dates="01-01-2026",
          experiment_end_dates="2026-01-10",
          expected_error="does not match format",
      ),
      dict(
          testcase_name="invalid_date_type_sequence",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=[datetime.date(2026, 1, 1), "123456"],
          experiment_end_dates=["2026-01-10", datetime.date(2026, 1, 7)],
          expected_error="does not match format",
      ),
  )
  def test_incrementality_raises_error_for_invalid_date_formats_and_types(
      self,
      point_estimates,
      standard_errors,
      experiment_kpi_types,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_dates,
      expected_error,
  ):
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, expected_error):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=point_estimates,
          standard_errors=standard_errors,
          experiment_kpi_types=experiment_kpi_types,
          experiment_total_spends=experiment_total_spends,
          experiment_start_dates=experiment_start_dates,
          experiment_end_dates=experiment_end_dates,
      )

  def test_with_incrementality_experiment_result_adjustments_are_preserved(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    builder.with_incrementality_experiment_result(
        channel_name="Search",
        point_estimates=[2.0, 3.0],
        standard_errors=[0.5, 0.6],
        experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
        experiment_total_spends=[1000.0, 2000.0],
        experiment_start_dates=["2026-01-01", "2026-01-02"],
        experiment_end_dates=["2026-01-10", "2026-01-11"],
        point_estimate_adjustments=[0.1, 0.2],
        standard_error_adjustments=[0.3, 0.4],
    )
    container = builder._container_by_channel["Search"]
    resolved = container.configs
    self.assertLen(resolved, 2)
    self.assertEqual(resolved[0].point_estimate_adjustment, 0.1)
    self.assertEqual(resolved[0].standard_error_adjustment, 0.3)
    self.assertEqual(resolved[1].point_estimate_adjustment, 0.2)
    self.assertEqual(resolved[1].standard_error_adjustment, 0.4)

  # ============================================================================
  # SECTION 3: Mismatched Length Sequence Validations (GeoX & Incrementality)
  # ============================================================================

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_geox_raises_error_for_length_mismatch(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_res = mock.create_autospec(
        geox_api.AnalysisResult, instance=True, spec_set=True
    )
    with self.assertRaisesRegex(ValueError, "must have the same length"):
      builder.with_meridian_geox_experiment_result(
          channel_name="Search",
          # The length of geox_results is 2, which does not match the length of
          # experiment_kpi_types (1).
          geox_results=[mock_res, mock_res],
          experiment_kpi_types=[constants.REVENUE],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="mismatched_standard_errors",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_date_list=["2026-01-10", "2026-01-11"],
      ),
      dict(
          testcase_name="mismatched_experiment_total_spends",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_total_spends=[100.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_date_list=["2026-01-10", "2026-01-11"],
      ),
      dict(
          testcase_name="mismatched_start_dates",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01"],
          experiment_end_date_list=["2026-01-10", "2026-01-11"],
      ),
      dict(
          testcase_name="mismatched_end_dates",
          point_estimates=[1.0, 1.2],
          standard_errors=[0.1, 0.1],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_date_list=["2026-01-10"],
      ),
  )
  def test_incrementality_raises_error_for_length_mismatch(
      self,
      point_estimates,
      standard_errors,
      experiment_total_spends,
      experiment_start_dates,
      experiment_end_date_list,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "must have the same length"):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=point_estimates,
          standard_errors=standard_errors,
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=experiment_total_spends,
          experiment_start_dates=experiment_start_dates,
          experiment_end_dates=experiment_end_date_list,
      )

  def test_with_incrementality_experiment_result_adjustments_length_mismatch_raises_error(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "must have the same length"):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[2.0, 3.0],
          standard_errors=[0.5, 0.6],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[1000.0, 2000.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
          point_estimate_adjustments=[0.1],
      )

  # ============================================================================
  # SECTION 4: Validator Helpers Block.
  # ============================================================================

  # Tests for _get_model_duration_days()
  @parameterized.named_parameters(
      dict(
          testcase_name="weekly",
          all_dates=[
              datetime.date(2026, 1, 1),
              datetime.date(2026, 1, 8),
          ],
          interval_days=7,
          expected_duration=14,
      ),
      dict(
          testcase_name="daily",
          all_dates=[
              datetime.date(2026, 1, 1),
              datetime.date(2026, 1, 2),
              datetime.date(2026, 1, 3),
          ],
          interval_days=1,
          expected_duration=3,
      ),
  )
  def test_get_model_duration_days_success(
      self,
      all_dates: list[datetime.date],
      interval_days: int,
      expected_duration: int,
  ) -> None:
    mock_time_coords = mock.create_autospec(
        time_coordinates.TimeCoordinates, instance=True, spec_set=True
    )
    mock_time_coords.all_dates = all_dates
    type(mock_time_coords).interval_days = mock.PropertyMock(
        return_value=interval_days
    )

    duration = prior_builder._get_model_duration_days(mock_time_coords)
    self.assertEqual(duration, expected_duration)

  def test_get_model_duration_days_raises_value_error_for_non_positive_duration(
      self,
  ) -> None:
    mock_time_coords = mock.create_autospec(
        time_coordinates.TimeCoordinates, instance=True, spec_set=True
    )
    mock_time_coords.all_dates = [
        datetime.date(2026, 1, 2),
        datetime.date(2026, 1, 1),
    ]
    type(mock_time_coords).interval_days = mock.PropertyMock(return_value=-5)

    with self.assertRaisesRegex(
        ValueError, "Model duration in days must be positive"
    ):
      prior_builder._get_model_duration_days(mock_time_coords)

  # Validation tests for channel name. _validate_channel_name()
  def test_channel_name_validation_valid_succeeds(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    # "Search" is a valid channel.
    builder._validate_channel_name("Search")

  def test_channel_name_validation_invalid_raises_error(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "not found in the model"):
      builder._validate_channel_name("Invalid_Channel")

  def test_incrementality_channel_name_validation_invalid_raises_error(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "not found in the model"):
      builder.with_incrementality_experiment_result(
          channel_name="Invalid_Channel",
          point_estimates=1.0,
          standard_errors=0.1,
          experiment_kpi_types=constants.REVENUE,
          experiment_total_spends=100.0,
          experiment_start_dates="2026-01-01",
          experiment_end_dates="2026-01-10",
      )

  @unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed")
  def test_geox_channel_name_validation_invalid_raises_error(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_geox_results = mock.create_autospec(
        geox_api.AnalysisResult, instance=True, spec_set=True
    )
    with self.assertRaisesRegex(ValueError, "not found in the model"):
      builder.with_meridian_geox_experiment_result(
          channel_name="Invalid_Channel",
          geox_results=mock_geox_results,
          experiment_kpi_types=constants.REVENUE,
      )

  def test_geox_experiment_result_raises_import_error_when_geox_not_installed(
      self,
  ) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    mock_geox_class = geox_api.AnalysisResult if _HAS_GEOX else object
    mock_geox_results = mock.create_autospec(
        mock_geox_class, instance=True, spec_set=True
    )
    with mock.patch.object(geox_adapter, "HAS_GEOX", False):
      with self.assertRaisesRegex(
          ImportError,
          "GeoX calibration is not available because the 'meridian_geox'"
          " library is not installed.",
      ):
        builder.with_meridian_geox_experiment_result(
            channel_name="Search",
            geox_results=mock_geox_results,
            experiment_kpi_types=constants.REVENUE,
        )
      with self.assertRaisesRegex(
          ImportError,
          "GeoX calibration is not available because the 'meridian_geox'"
          " library is not installed.",
      ):
        geox_adapter.resolve_meridian_geox_source(
            result=mock_geox_results,
            kpi_type=constants.REVENUE,
            revenue_per_kpi=None,
        )

  # ============================================================================
  # SECTION 5: KPI Compatibility Validations
  # ============================================================================

  @parameterized.named_parameters(
      dict(
          testcase_name="revenue_model_non_revenue_experiment",
          model_kpi=constants.REVENUE,
          experiment_kpi_type=constants.NON_REVENUE,
          expected_error_regex=(
              "Experiment calibration for models with revenue as KPI and"
              " experiments with non-revenue KPIs requires `revenue_per_kpi`"
              " to be passed in `meridian.data.input_data.InputData`."
          ),
      ),
      dict(
          testcase_name="non_revenue_model_revenue_experiment",
          model_kpi=constants.NON_REVENUE,
          experiment_kpi_type=constants.REVENUE,
          expected_error_regex=(
              "Experiment calibration for models where the outcome is not in"
              " terms of revenue is not supported. Pass `revenue_per_kpi` in"
              " `meridian.data.input_data.InputData`."
          ),
      ),
  )
  def test_kpi_compatibility_missing_revenue_per_kpi_raises_error(
      self, model_kpi: str, experiment_kpi_type: str, expected_error_regex: str
  ) -> None:
    type(self.mock_input_data).kpi_type = mock.PropertyMock(
        return_value=model_kpi
    )
    self.mock_input_data.revenue_per_kpi = None
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=1.5,
          standard_errors=0.2,
          experiment_kpi_types=experiment_kpi_type,
          experiment_total_spends=100.0,
          experiment_start_dates="2026-01-01",
          experiment_end_dates="2026-01-10",
      )

  # ============================================================================
  # SECTION 6: Coercion Helpers Block (_coerce_*)
  # ============================================================================

  def test_coerce_optional_floats_valid_cases(self):
    # Single float/int input.
    self.assertEqual(prior_builder._coerce_optional_floats(12.5), [12.5])
    self.assertEqual(prior_builder._coerce_optional_floats(10), [10])
    self.assertEqual(prior_builder._coerce_optional_floats([10]), [10])
    self.assertEmpty(prior_builder._coerce_optional_floats(None))

    # List with float and None.
    self.assertEqual(
        prior_builder._coerce_optional_floats([12.5, None, 8.0]),  # pyrefly: ignore[bad-argument-type]
        [12.5, None, 8.0],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="boolean_direct",
          value=True,
          expected_regex="Expected float, int, None, or sequence",
      ),
      dict(
          testcase_name="boolean_in_sequence",
          value=[12.5, False],
          expected_regex="Booleans are not allowed",
      ),
      dict(
          testcase_name="string_in_sequence",
          value=[12.5, "abc"],
          expected_regex="Expected float, int, or None",
      ),
      dict(
          testcase_name="string_direct",
          value="abc",
          expected_regex="Expected float, int, None, or sequence",
      ),
  )
  def test_coerce_optional_floats_raises_type_error(
      self, value, expected_regex
  ):
    # Verify raises TypeError on booleans or invalid types.
    with self.assertRaisesRegex(TypeError, expected_regex):
      prior_builder._coerce_optional_floats(value)  # pyrefly: ignore[bad-argument-type]

  def test_coerce_required_floats_valid_cases(self):
    # Single float/int input.
    self.assertEqual(prior_builder._coerce_required_floats(12.5), [12.5])
    self.assertEqual(prior_builder._coerce_required_floats(10), [10])
    self.assertEqual(prior_builder._coerce_required_floats([10]), [10])

    # List of floats.
    self.assertEqual(
        prior_builder._coerce_required_floats([12.5, 8.0]),
        [12.5, 8.0],
    )

  def test_coerce_required_floats_raises_on_none(self):
    with self.assertRaisesRegex(TypeError, "Expected float, int, or sequence"):
      prior_builder._coerce_required_floats(None)  # type: ignore

  def test_coerce_required_floats_raises_on_invalid_element(self):
    with self.assertRaisesRegex(TypeError, "Expected float or int"):
      prior_builder._coerce_required_floats(["invalid"])  # type: ignore

  def test_incrementality_required_floats_distinction(
      self,
  ):
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)

    # Required float: point_estimates containing 'None' raises TypeError.
    with self.assertRaisesRegex(TypeError, "None/Booleans are not allowed"):
      builder.with_incrementality_experiment_result(
          channel_name="Search",
          point_estimates=[1.0, None],  # type: ignore
          standard_errors=[0.1, 0.1],
          experiment_kpi_types=[constants.REVENUE, constants.REVENUE],
          experiment_total_spends=[100.0, 200.0],
          experiment_start_dates=["2026-01-01", "2026-01-02"],
          experiment_end_dates=["2026-01-10", "2026-01-11"],
      )

  def test_coerce_kpi_types_valid_cases(self) -> None:
    self.assertEqual(
        prior_builder._coerce_kpi_types(constants.REVENUE), [constants.REVENUE]
    )
    self.assertEqual(
        prior_builder._coerce_kpi_types([constants.NON_REVENUE]),
        [constants.NON_REVENUE],
    )

  def test_coerce_kpi_types_raises_on_invalid_type(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "Invalid 'experiment_kpi_types': 'invalid'"
    ):
      prior_builder._coerce_kpi_types("invalid")

    with self.assertRaisesRegex(
        ValueError, "Invalid 'experiment_kpi_types': 'invalid'"
    ):
      prior_builder._coerce_kpi_types([constants.REVENUE, "invalid"])

  def test_coerce_kpi_types_raises_on_none(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "must be passed as a string or sequence"
    ):
      prior_builder._coerce_kpi_types(None)  # type: ignore

  def test_coerce_required_dates_expression_handling(self) -> None:
    # Coercing a single string date.
    self.assertEqual(
        prior_builder._coerce_required_dates("2026-05-26"),
        [datetime.date(2026, 5, 26)],
    )
    # Coercing sequence of string and datetime.date values.
    self.assertEqual(
        prior_builder._coerce_required_dates(
            ["2026-05-26", datetime.date(2026, 5, 27)]
        ),
        [datetime.date(2026, 5, 26), datetime.date(2026, 5, 27)],
    )
    # Coercing datetime.datetime value.
    self.assertEqual(
        prior_builder._coerce_required_dates(
            datetime.datetime(2026, 5, 26, 12, 0, 0)
        ),
        [datetime.date(2026, 5, 26)],
    )
    # Coercing np.datetime64 value.
    self.assertEqual(
        prior_builder._coerce_required_dates(np.datetime64("2026-05-26")),
        [datetime.date(2026, 5, 26)],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="none",
          dates=None,
          error_msg="Unsupported date value type: .*NoneType.* for None",
      ),
      dict(
          testcase_name="sequence_with_none",
          dates=["2026-05-26", None],
          error_msg="Unsupported date value type: .*NoneType.* for None",
      ),
      dict(
          testcase_name="invalid_type_int",
          dates=12345,
          error_msg="Unsupported date value type: <class 'int'> for 12345",
      ),
      dict(
          testcase_name="sequence_with_boolean",
          dates=["2026-05-26", True],
          error_msg="Unsupported date value type: <class 'bool'> for True",
      ),
  )
  def test_coerce_required_dates_raises_on_none_and_invalid(
      self, dates: Any, error_msg: str
  ) -> None:
    with self.assertRaisesRegex(ValueError, error_msg):
      prior_builder._coerce_required_dates(dates)  # type: ignore

  def test_revenue_per_kpi_returns_mean_value(self) -> None:
    mock_revenue = mock.create_autospec(xr.DataArray, instance=True)
    mock_revenue.mean.return_value = 2.5
    type(self.mock_input_data).revenue_per_kpi = mock.PropertyMock(
        return_value=mock_revenue
    )
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    self.assertEqual(builder._revenue_per_kpi, 2.5)

  def test_revenue_per_kpi_handles_attribute_error_returns_none(self) -> None:
    type(self.mock_input_data).revenue_per_kpi = mock.PropertyMock(
        return_value=object()
    )
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    self.assertIsNone(builder._revenue_per_kpi)

  def test_revenue_per_kpi_is_none_returns_none(self) -> None:
    type(self.mock_input_data).revenue_per_kpi = mock.PropertyMock(
        return_value=None
    )
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    self.assertIsNone(builder._revenue_per_kpi)

  def test_get_channel_total_spend_caches_and_returns_media_spend(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    spend = builder._get_channel_total_spend("Search")
    self.assertEqual(spend, 4000.0)

    # Calling again should use cached value without hitting input_data mocks.
    with mock.patch.object(self.mock_input_data, "media_channel") as mock_chan:
      spend_cached = builder._get_channel_total_spend("Search")
      self.assertEqual(spend_cached, 4000.0)
      mock_chan.assert_not_called()

  def test_get_channel_total_spend_caches_and_returns_rf_spend(self) -> None:
    mock_rf_channel = mock.create_autospec(xr.DataArray, instance=True)
    mock_rf_channel.values = ["RF_Channel"]
    type(self.mock_input_data).rf_channel = mock.PropertyMock(
        return_value=mock_rf_channel
    )
    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=None
    )

    mock_rf_spend = mock.create_autospec(xr.DataArray, instance=True)
    mock_rf_spend.sel.return_value.sum.return_value = 1500.0
    type(self.mock_input_data).rf_spend = mock.PropertyMock(
        return_value=mock_rf_spend
    )

    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    self.assertEqual(builder._get_channel_total_spend("RF_Channel"), 1500.0)

  def test_get_channel_total_spend_raises_on_invalid_channel(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(ValueError, "not found in the model"):
      builder._get_channel_total_spend("InvalidChannel")

  def test_get_channel_total_spend_missing_spend_data_raises_value_error(
      self,
  ) -> None:
    # 1. Media Channel with media_spend is None
    type(self.mock_input_data).media_spend = mock.PropertyMock(
        return_value=None
    )
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(
        ValueError,
        "Channel 'Search' spend is required for calibration but is not"
        " available.",
    ):
      builder._get_channel_total_spend("Search")

    # 2. RF Channel with rf_spend is None
    mock_rf_channel = mock.create_autospec(xr.DataArray, instance=True)
    mock_rf_channel.values = ["RF_Channel"]
    type(self.mock_input_data).rf_channel = mock.PropertyMock(
        return_value=mock_rf_channel
    )
    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=None
    )
    type(self.mock_input_data).rf_spend = mock.PropertyMock(return_value=None)
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)
    with self.assertRaisesRegex(
        ValueError,
        "Channel 'RF_Channel' spend is required for calibration but is not"
        " available.",
    ):
      builder._get_channel_total_spend("RF_Channel")

  # ============================================================================
  # SECTION 7: CalibrationBuilder build() and Health Checks Block
  # ============================================================================
  def test_build_success_with_calibrated_experiments(self) -> None:
    builder = prior_builder.CalibrationBuilder(self.mock_input_data)

    mock_media_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_media_channel.values = ["Search", "YouTube"]
    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=mock_media_channel
    )
    type(self.mock_input_data).rf_channel = mock.PropertyMock(return_value=None)

    media_builder = arg_builder.OrderedListArgumentBuilder(
        ["Search", "YouTube"]
    )
    self.mock_input_data.get_paid_media_channels_argument_builder.return_value = (
        media_builder
    )

    builder.with_incrementality_experiment_result(
        channel_name="Search",
        point_estimates=[2.0],
        standard_errors=[0.4],
        experiment_kpi_types=constants.REVENUE,
        experiment_total_spends=[1000.0],
        experiment_start_dates=["2026-01-08"],
        experiment_end_dates=["2026-01-15"],
    )

    priors = builder.build()
    self.assertIsNotNone(priors.roi_m)

    roi_dists = _get_underlying_distributions(priors.roi_m)
    self.assertIsInstance(roi_dists[0], backend.tfd.Gamma)
    self.assertIsInstance(roi_dists[1], backend.tfd.LogNormal)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_priors",
          channel_type="media",
          channels=["Search", "YouTube"],
          prior_key="roi_m",
      ),
      dict(
          testcase_name="rf_priors",
          channel_type="rf",
          channels=["RF_Target"],
          prior_key="roi_rf",
      ),
  )
  def test_build_success_with_no_experiments_returns_default_priors(
      self,
      channel_type: str,
      channels: Sequence[str],
      prior_key: str,
  ) -> None:
    mock_registry = mock.create_autospec(
        base.CalibrationRegistry, instance=True, spec_set=True
    )
    mock_registry.get_roi_distributions_by_channel.return_value = (
        base.CalibrationRegistryResult(distributions={}, outputs={})
    )
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data, registry=mock_registry
    )

    mock_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_channel.values = list(channels)
    arg_builder_obj = arg_builder.OrderedListArgumentBuilder(list(channels))

    if channel_type == "media":
      type(self.mock_input_data).media_channel = mock.PropertyMock(
          return_value=mock_channel
      )
      type(self.mock_input_data).rf_channel = mock.PropertyMock(
          return_value=None
      )
      self.mock_input_data.get_paid_media_channels_argument_builder.return_value = (
          arg_builder_obj
      )
    else:
      type(self.mock_input_data).media_channel = mock.PropertyMock(
          return_value=None
      )
      type(self.mock_input_data).rf_channel = mock.PropertyMock(
          return_value=mock_channel
      )
      self.mock_input_data.get_paid_rf_channels_argument_builder.return_value = (
          arg_builder_obj
      )

    priors = builder.build()

    self.assertIsNotNone(getattr(priors, prior_key))
    roi_dists = _get_underlying_distributions(getattr(priors, prior_key))
    default_prior_val = getattr(
        calibration_constants.get_default_prior(), prior_key
    )
    self.assertEqual(len(roi_dists), len(channels))
    for d in roi_dists:
      self.assertTrue(
          prior_distribution.distributions_are_equal(d, default_prior_val)
      )

  def test_build_prior_override_precedence(self) -> None:
    # Custom priors set on Search & YouTube.
    search_custom = backend.tfd.LogNormal(
        loc=backend.cast(0.5, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    youtube_custom = backend.tfd.LogNormal(
        loc=backend.cast(0.5, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )

    # Calibration registry returns calibrated prior on Search.
    search_calibrated = backend.tfd.LogNormal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(1.0, backend.float_dtype),
    )
    mock_registry = mock.create_autospec(
        base.CalibrationRegistry, instance=True, spec_set=True
    )
    mock_registry.get_roi_distributions_by_channel.return_value = (
        base.CalibrationRegistryResult(
            distributions={"Search": search_calibrated}, outputs={}
        )
    )
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data,
        custom_prior={"Search": search_custom, "YouTube": youtube_custom},
        registry=mock_registry,
    )

    mock_media_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_media_channel.values = ["Search", "YouTube", "Third_Channel"]
    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=mock_media_channel
    )
    type(self.mock_input_data).rf_channel = mock.PropertyMock(return_value=None)

    media_builder = arg_builder.OrderedListArgumentBuilder(
        ["Search", "YouTube", "Third_Channel"]
    )
    self.mock_input_data.get_paid_media_channels_argument_builder.return_value = (
        media_builder
    )

    with self.assertWarnsRegex(
        UserWarning,
        "Custom prior for channel\\(s\\) Search will be overwritten",
    ):
      priors = builder.build()

    # Verify the combined prior overrides are correctly returned:
    # 1. "Search" has both custom & calibrated -> Calibrated overrides custom.
    # 2. "YouTube" has only custom -> Custom is used.
    # 3. "Third_Channel" has neither -> Fallback default ROI prior.
    self.assertIsNotNone(priors.roi_m)
    roi_m_dists = _get_underlying_distributions(priors.roi_m)
    expected_dists = [
        search_calibrated,
        youtube_custom,
        calibration_constants.get_default_prior().roi_m,
    ]
    self.assertEqual(len(roi_m_dists), len(expected_dists))
    for actual, expected in zip(roi_m_dists, expected_dists):
      self.assertTrue(
          prior_distribution.distributions_are_equal(actual, expected)
      )

  def test_build_calculates_max_lag_weeks_for_daily_data(self) -> None:
    mock_time_coords = mock.create_autospec(
        time_coordinates.TimeCoordinates, instance=True, spec_set=True
    )
    mock_time_coords.all_dates = [
        datetime.date(2026, 1, 1),
        datetime.date(2026, 1, 2),
        datetime.date(2026, 1, 3),
    ]
    type(mock_time_coords).interval_days = mock.PropertyMock(return_value=1)

    mock_input_daily = mock.create_autospec(
        input_data.InputData, instance=True, spec_set=True
    )
    type(mock_input_daily).time_coordinates = mock.PropertyMock(
        return_value=mock_time_coords
    )
    type(mock_input_daily).kpi_type = mock.PropertyMock(
        return_value=constants.REVENUE
    )
    mock_input_daily.get_all_paid_channels.return_value = ["Search"]
    mock_media_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_media_channel.values = ["Search"]
    type(mock_input_daily).media_channel = mock.PropertyMock(
        return_value=mock_media_channel
    )
    type(mock_input_daily).rf_channel = mock.PropertyMock(return_value=None)
    mock_input_daily.get_paid_media_channels_argument_builder.return_value = (
        arg_builder.OrderedListArgumentBuilder(["Search"])
    )

    mock_registry = mock.create_autospec(
        base.CalibrationRegistry, instance=True, spec_set=True
    )
    mock_registry.get_roi_distributions_by_channel.return_value = (
        base.CalibrationRegistryResult(distributions={}, outputs={})
    )

    # max_lag is set to 28 periods (days in this daily model)
    builder = prior_builder.CalibrationBuilder(
        mock_input_daily,
        max_lag=28,
        registry=mock_registry,
    )

    builder.build()

    mock_registry.get_roi_distributions_by_channel.assert_called_once_with(
        last_modeled_date=datetime.date(2026, 1, 3),
        max_lag=28,
        interval_days=1,
        model_duration_days=3,
    )

  def test_build_returns_calibrated_distributions_media(self) -> None:
    mock_registry = mock.create_autospec(
        base.CalibrationRegistry, instance=True, spec_set=True
    )
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data, registry=mock_registry
    )
    mock_dist = backend.tfd.LogNormal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    mock_output = base.CalibrationOutput(
        channel_name="Search",
        intermediary_prior=mock_dist,
        experiments=[],
    )
    mock_registry.get_roi_distributions_by_channel.return_value = (
        base.CalibrationRegistryResult(
            distributions={"Search": mock_dist},
            outputs={"Search": mock_output},
        )
    )

    media_channels = ["Search", "YouTube"]
    arg_builder_obj = arg_builder.OrderedListArgumentBuilder(media_channels)
    mock_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_channel.values = media_channels

    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=mock_channel
    )
    type(self.mock_input_data).rf_channel = mock.PropertyMock(return_value=None)
    self.mock_input_data.get_paid_media_channels_argument_builder.return_value = (
        arg_builder_obj
    )

    priors = builder.build()

    self.assertIsInstance(priors, base.CalibratedPriors)
    self.assertIsNone(priors.roi_rf)

    roi_m = priors.roi_m
    self.assertIsInstance(roi_m, base.CalibratedDistribution)

    with self.subTest("calibration_status_checks"):
      self.assertEqual(roi_m.get_calibration_status(), (True, False))

    with self.subTest("calibration_outputs_checks"):
      output = roi_m.calibration_outputs[0]
      self.assertIsNotNone(output)
      self.assertEqual(output.channel_name, "Search")
      self.assertEqual(output.adstock_decay_spec, constants.GEOMETRIC_DECAY)
      self.assertEqual(output.max_lag, 8)
      self.assertIsNone(roi_m.calibration_outputs[1])

  def test_build_returns_calibrated_distributions_rf(self) -> None:
    mock_registry = mock.create_autospec(
        base.CalibrationRegistry, instance=True, spec_set=True
    )
    builder = prior_builder.CalibrationBuilder(
        self.mock_input_data, registry=mock_registry
    )
    mock_dist = backend.tfd.LogNormal(
        loc=backend.cast(1.0, backend.float_dtype),
        scale=backend.cast(0.5, backend.float_dtype),
    )
    mock_output = base.CalibrationOutput(
        channel_name="RF_Target",
        intermediary_prior=mock_dist,
        experiments=[],
    )
    mock_registry.get_roi_distributions_by_channel.return_value = (
        base.CalibrationRegistryResult(
            distributions={"RF_Target": mock_dist},
            outputs={"RF_Target": mock_output},
        )
    )

    rf_channels = ["RF_Target"]
    arg_builder_obj = arg_builder.OrderedListArgumentBuilder(rf_channels)
    mock_channel = mock.create_autospec(
        xr.DataArray, instance=True, spec_set=True
    )
    mock_channel.values = rf_channels

    type(self.mock_input_data).media_channel = mock.PropertyMock(
        return_value=None
    )
    type(self.mock_input_data).rf_channel = mock.PropertyMock(
        return_value=mock_channel
    )
    self.mock_input_data.get_paid_rf_channels_argument_builder.return_value = (
        arg_builder_obj
    )

    priors = builder.build()

    self.assertIsInstance(priors, base.CalibratedPriors)
    self.assertIsNone(priors.roi_m)

    roi_rf = priors.roi_rf
    self.assertIsInstance(roi_rf, base.CalibratedDistribution)

    with self.subTest("calibration_status_checks"):
      self.assertEqual(roi_rf.get_calibration_status(), (True,))

    with self.subTest("calibration_outputs_checks"):
      output = roi_rf.calibration_outputs[0]
      self.assertIsNotNone(output)
      self.assertEqual(output.channel_name, "RF_Target")


if __name__ == "__main__":
  absltest.main()
