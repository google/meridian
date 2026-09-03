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

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model.calibration import base
from meridian.model.calibration.adapters import experiment as experiment_adapter


class ExperimentAdapterTest(parameterized.TestCase):

  def test_resolve_revenue_kpi_success(self) -> None:
    resolved = experiment_adapter.resolve_experiment_source(
        point_estimate=2.5,
        standard_error=0.5,
        kpi_type=constants.REVENUE,
        total_spend=1000.0,
        revenue_per_kpi=None,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
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
    )
    self.assertEqual(resolved, expected)

  def test_resolve_non_revenue_kpi_success(self) -> None:
    resolved = experiment_adapter.resolve_experiment_source(
        point_estimate=2.5,
        standard_error=0.5,
        kpi_type=constants.NON_REVENUE,
        total_spend=1000.0,
        revenue_per_kpi=2.0,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
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
    )
    self.assertEqual(resolved, expected)

  def test_resolve_non_revenue_no_revenue_per_kpi_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError,
        "Experiment has `non-revenue` kpi, but provided"
        " `revenue_per_kpi` is None.",
    ):
      experiment_adapter.resolve_experiment_source(
          point_estimate=2.5,
          standard_error=0.5,
          kpi_type=constants.NON_REVENUE,
          total_spend=1000.0,
          revenue_per_kpi=None,
          experiment_start_date=datetime.date(2026, 1, 1),
          experiment_end_date=datetime.date(2026, 1, 10),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="generic",
          source_type=base.SourceType.GENERIC,
      ),
      dict(
          testcase_name="meridian_geox",
          source_type=base.SourceType.MERIDIAN_GEOX,
      ),
  )
  def test_resolve_experiment_source_propagates_source_type(
      self, source_type: base.SourceType
  ) -> None:
    resolved = experiment_adapter.resolve_experiment_source(
        point_estimate=2.5,
        standard_error=0.5,
        kpi_type=constants.REVENUE,
        total_spend=1000.0,
        revenue_per_kpi=None,
        experiment_start_date=datetime.date(2026, 1, 1),
        experiment_end_date=datetime.date(2026, 1, 10),
        source_type=source_type,
    )
    self.assertEqual(resolved.source_type, source_type)


class TransformToRevenueTest(parameterized.TestCase):

  def test_transform_to_revenue_revenue_kpi_returns_unchanged(self) -> None:
    point_est, std_err = experiment_adapter._transform_to_revenue(
        point_estimate=2.5,
        standard_error=0.5,
        experiment_kpi_type=constants.REVENUE,
        revenue_per_kpi=None,
    )
    self.assertEqual(point_est, 2.5)
    self.assertEqual(std_err, 0.5)

  def test_transform_to_revenue_non_revenue_kpi_scales_correctly(self) -> None:
    point_est, std_err = experiment_adapter._transform_to_revenue(
        point_estimate=2.5,
        standard_error=0.5,
        experiment_kpi_type=constants.NON_REVENUE,
        revenue_per_kpi=2.0,
    )
    self.assertEqual(point_est, 5.0)
    self.assertEqual(std_err, 1.0)

  def test_transform_to_revenue_non_revenue_kpi_missing_revenue_per_kpi_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError,
        "Experiment has `non-revenue` kpi, but provided"
        " `revenue_per_kpi` is None.",
    ):
      experiment_adapter._transform_to_revenue(
          point_estimate=2.5,
          standard_error=0.5,
          experiment_kpi_type=constants.NON_REVENUE,
          revenue_per_kpi=None,
      )


if __name__ == "__main__":
  absltest.main()
