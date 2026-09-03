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

"""Generic incrementality experiment adapter for calibration."""

from __future__ import annotations

import datetime

from meridian import constants
from meridian.model.calibration import base


def _transform_to_revenue(
    *,
    point_estimate: float,
    standard_error: float,
    experiment_kpi_type: str,
    revenue_per_kpi: float | None,
) -> tuple[float, float]:
  """Transforms the point estimate and standard error of a non-revenue experiment to revenue.

  Args:
    point_estimate: The point estimate of the observed effect.
    standard_error: The standard error of the observed effect.
    experiment_kpi_type: The KPI type string, either `'revenue'` or
      `'non-revenue'`.
    revenue_per_kpi: The revenue per KPI for converting `non-revenue` experiment
      results to `revenue` values.

  Returns:
    A tuple containing the scaled `(point_estimate, standard_error)`.

  Raises:
    ValueError: If kpi_type is `non-revenue` but `revenue_per_kpi` is None.
  """
  if experiment_kpi_type == constants.NON_REVENUE:
    if revenue_per_kpi is None:
      raise ValueError(
          "Experiment has `non-revenue` kpi, but provided"
          " `revenue_per_kpi` is None."
      )
    return (
        point_estimate * revenue_per_kpi,
        standard_error * revenue_per_kpi,
    )

  return point_estimate, standard_error


def resolve_experiment_source(
    *,
    point_estimate: float,
    standard_error: float,
    kpi_type: str,
    revenue_per_kpi: float | None = None,
    total_spend: float,
    experiment_start_date: datetime.date,
    experiment_end_date: datetime.date,
    point_estimate_adjustment: float | None = None,
    standard_error_adjustment: float | None = None,
    source_type: base.SourceType = base.SourceType.GENERIC,
) -> base.CalibrationData:
  """Resolves incrementality experiment data into `meridian.model.calibration.base.CalibrationData`.

  This adapter translates point estimates and standard errors into generic
  `meridian.model.calibration.base.CalibrationData` objects and converts
  non-revenue experiment KPIs to revenue values, if necessary.

  Args:
    point_estimate: The point estimate of the incrementality experiment result.
    standard_error: The standard error of the incrementality experiment result.
    kpi_type: A string denoting whether the KPI of the incrementality experiment
      result is of a `'revenue'` or `'non-revenue'` type.
    revenue_per_kpi: The revenue per KPI for converting `non-revenue` experiment
      results to `revenue` values when necessary.
    total_spend: The total spend covered by the incrementality experiment.
    experiment_start_date: The start date of the incrementality experiment. The
      first day the experiment was active.
    experiment_end_date: The end date of the incrementality experiment. The last
      day the experiment was active.
    point_estimate_adjustment: The optional point estimate (gamma) adjustment.
    standard_error_adjustment: The optional standard error (tau) adjustment.
    source_type: The source type representing the origin of the experiment
      result. Default is `base.SourceType.GENERIC`.

  Returns:
    A `meridian.model.calibration.base.CalibrationData` object representing
    the incrementality experiment data.

  Raises:
    ValueError: If any of the following are true:
      - `kpi_type` is `'non-revenue'` but `revenue_per_kpi` is `None`.
      - The experiment result is statistically significantly negative.
      - The standard error of the experiment result is not positive.
  """
  point_estimate_revenue, standard_error_revenue = _transform_to_revenue(
      point_estimate=point_estimate,
      standard_error=standard_error,
      experiment_kpi_type=kpi_type,
      revenue_per_kpi=revenue_per_kpi,
  )

  return base.CalibrationData(
      experiment_result=base.ExperimentResult(
          point_estimate=point_estimate_revenue,
          standard_error=standard_error_revenue,
      ),
      experiment_info=base.ExperimentInfo(
          total_spend=total_spend,
          experiment_start_date=experiment_start_date,
          experiment_end_date=experiment_end_date,
      ),
      point_estimate_adjustment=point_estimate_adjustment,
      standard_error_adjustment=standard_error_adjustment,
      source_type=source_type,
  )
