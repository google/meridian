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

from collections.abc import Callable, Mapping
import datetime
import sys
from typing import TYPE_CHECKING
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model.calibration import base
import pandas as pd

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=g-bad-import-order
  from meridian_geox import api as geox_api
  # pylint: enable=g-import-not-at-top
  # pylint: enable=g-bad-import-order
  _HAS_GEOX = True
else:  # pylint: disable=unreachable
  try:
    # pylint: disable=g-import-not-at-top
    # pylint: disable=g-bad-import-order
    from meridian_geox import api as geox_api
    # pylint: enable=g-import-not-at-top
    # pylint: enable=g-bad-import-order

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

# pylint: disable=g-import-not-at-top,g-bad-import-order
from meridian.model.calibration.adapters import meridian_geox as geox_adapter

# pylint: enable=g-import-not-at-top,g-bad-import-order


def _create_metrics(
    point_estimate: float = 2.5,
    standard_deviation: float = 0.5,
    total_spend: float | None = 1000.0,
    icpd_available: bool = True,
    descriptive_metrics_available: bool = True,
) -> geox_api.AnalysisMetrics:
  """Creates a mock AnalysisMetrics object for testing.

  Args:
    point_estimate: The point estimate of the KPI.
    standard_deviation: The standard deviation of the KPI.
    total_spend: The total spend of the media channel, or None.
    icpd_available: Whether the KPI estimate is populated.
    descriptive_metrics_available: Whether descriptive metrics are available.

  Returns:
    A mock `meridian_geox.api.AnalysisMetrics` object.
  """
  if icpd_available:
    estimate = geox_api.Estimate(
        point_estimate=point_estimate,
        lower_bound=point_estimate - 1.0,
        upper_bound=point_estimate + 1.0,
        standard_deviation=standard_deviation,
        p_value=0.05,
    )
  else:
    estimate = None

  descriptive_metrics = (
      geox_api.DescriptiveMetrics(estimated_bau_spend=total_spend)
      if descriptive_metrics_available
      else None
  )
  dummy_estimate = geox_api.Estimate(
      point_estimate=0.0,
      lower_bound=0.0,
      upper_bound=0.0,
      standard_deviation=0.0,
      p_value=0.05,
  )
  dummy_dataframe = pd.DataFrame()
  return geox_api.AnalysisMetrics(
      lift=dummy_estimate,
      percent_lift=dummy_estimate,
      cumulative_lift=dummy_dataframe,
      counterfactual_conversions=dummy_dataframe,
      pointwise_difference=dummy_dataframe,
      icpd=estimate,
      cumulative_icpd=None,
      descriptive_metrics=descriptive_metrics,
  )


def _create_analysis_result(
    results: Mapping[str, geox_api.AnalysisMetrics],
) -> geox_api.AnalysisResult:
  """Creates a mock AnalysisResult object for testing.

  Args:
    results: Mapping of cell names to mock `AnalysisMetrics`.

  Returns:
    A mock `meridian_geox.api.AnalysisResult` object.
  """
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
      results=dict(results),
      analysis_config=analysis_config,
  )


@unittest.skipIf(not _HAS_GEOX, "meridian_geox is not installed.")
class MeridianGeoXAdapterTest(parameterized.TestCase):

  def test_resolve_revenue_kpi_success(self) -> None:
    cell_metrics = _create_metrics()
    result = _create_analysis_result({"cell_1": cell_metrics})
    resolved = geox_adapter.resolve_meridian_geox_source(
        result=result,
        kpi_type=constants.REVENUE,
        revenue_per_kpi=None,
    )

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
    self.assertEqual(resolved, expected)

  def test_resolve_non_revenue_kpi_success(self) -> None:
    cell_metrics = _create_metrics()
    result = _create_analysis_result({"cell_1": cell_metrics})
    resolved = geox_adapter.resolve_meridian_geox_source(
        result=result,
        kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=2.0,
    )

    expected = base.CalibrationData(
        experiment_result=base.ExperimentResult(
            point_estimate=5.0, standard_error=1.0
        ),
        experiment_info=base.ExperimentInfo(
            total_spend=1000.0,
            experiment_start_date=datetime.date(2026, 1, 1),
            experiment_end_date=datetime.date(2026, 1, 10),
        ),
        source_type=base.SourceType.MERIDIAN_GEOX,
    )
    self.assertEqual(resolved, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_results",
          results_factory=dict,
          error_msg="GeoX AnalysisResult contains no results.",
      ),
      dict(
          testcase_name="no_icpd",
          results_factory=lambda: {
              "cell_1": _create_metrics(icpd_available=False)
          },
          error_msg=(
              "Meridian GeoX experiment result for cell 'cell_1' does not "
              "contain the required iCPD estimate."
          ),
      ),
      dict(
          testcase_name="multiple_results",
          results_factory=lambda: {
              "cell_1": _create_metrics(),
              "cell_2": _create_metrics(),
          },
          error_msg=(
              "The GeoX AnalysisResult contains results from a multi cell"
              " study. Only single cell studies are supported."
          ),
      ),
      dict(
          testcase_name="missing_descriptive_metrics",
          results_factory=lambda: {
              "cell_1": _create_metrics(
                  total_spend=None, descriptive_metrics_available=False
              )
          },
          error_msg="must contain the required estimated_bau_spend estimate",
      ),
      dict(
          testcase_name="missing_estimated_bau_spend",
          results_factory=lambda: {
              "cell_1": _create_metrics(
                  total_spend=None, descriptive_metrics_available=True
              )
          },
          error_msg="must contain the required estimated_bau_spend estimate",
      ),
  )
  def test_resolve_invalid_geox_result_raises_invalid_geox_result_error(
      self,
      results_factory: Callable[[], Mapping[str, geox_api.AnalysisMetrics]],
      error_msg: str,
  ) -> None:
    result = _create_analysis_result(results_factory())

    with self.assertRaisesRegex(
        geox_adapter.InvalidGeoXResultError,
        error_msg,
    ):
      geox_adapter.resolve_meridian_geox_source(
          result=result,
          kpi_type=constants.REVENUE,
          revenue_per_kpi=None,
      )

  def test_resolve_non_revenue_no_revenue_per_kpi_raises_value_error(
      self,
  ) -> None:
    cell_metrics = _create_metrics()
    result = _create_analysis_result({"cell_1": cell_metrics})

    with self.assertRaisesRegex(
        ValueError,
        "Experiment has `non-revenue` kpi, but provided"
        " `revenue_per_kpi` is None.",
    ):
      geox_adapter.resolve_meridian_geox_source(
          result=result,
          kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=None,
      )

  def test_init_raises_import_error_when_geox_not_installed(self) -> None:
    cell_metrics = _create_metrics()
    result = _create_analysis_result({"cell_1": cell_metrics})
    with mock.patch.object(geox_adapter, "HAS_GEOX", False):
      with self.assertRaisesRegex(
          ImportError,
          "GeoX calibration is not available because the 'meridian_geox'"
          " library is not installed.",
      ):
        geox_adapter.resolve_meridian_geox_source(
            result=result,
            kpi_type=constants.REVENUE,
            revenue_per_kpi=None,
        )


if __name__ == "__main__":
  absltest.main()
