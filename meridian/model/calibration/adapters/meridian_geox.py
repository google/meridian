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

"""Meridian GeoX adapter for calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from meridian.model.calibration import base
from meridian.model.calibration.adapters import experiment as experiment_adapter

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top
  from meridian_geox import api as geox_api
  # pylint: enable=g-import-not-at-top

  HAS_GEOX = True
else:  # pylint: disable=unreachable
  try:
    # pylint: disable=g-import-not-at-top
    from meridian_geox import api as geox_api
    # pylint: enable=g-import-not-at-top
    HAS_GEOX = True
  except ImportError:
    geox_api = None
    HAS_GEOX = False


class InvalidGeoXResultError(ValueError):
  """GeoX analysis result that violates modeling assumptions."""


def _validate_and_extract_metrics(
    result: geox_api.AnalysisResult,
) -> geox_api.AnalysisMetrics:
  """Validates the GeoX AnalysisResult and extracts metrics for the single cell.

  Args:
    result: The `meridian_geox.api.AnalysisResult` containing the experiment
      result.

  Returns:
    The `meridian_geox.api.AnalysisMetrics` extracted for the single cell
    experiment result.

  Raises:
    InvalidGeoXResultError: When any of the following are true:
      - The results are empty.
      - The results contain multi-cell study results.
      - The KPI estimate is missing.
  """
  if not result.results:
    raise InvalidGeoXResultError("GeoX AnalysisResult contains no results.")

  if len(result.results) > 1:
    raise InvalidGeoXResultError(
        "The GeoX AnalysisResult contains results from a multi cell study. Only"
        " single cell studies are supported."
    )

  cell_id, metrics = next(iter(result.results.items()))
  if metrics.icpd is None:
    raise InvalidGeoXResultError(
        f"Meridian GeoX experiment result for cell '{cell_id}' does not"
        " contain the required iCPD estimate."
    )
  return metrics


def resolve_meridian_geox_source(
    *,
    result: geox_api.AnalysisResult,
    kpi_type: str,
    revenue_per_kpi: float | None = None,
    point_estimate_adjustment: float | None = None,
    standard_error_adjustment: float | None = None,
) -> base.CalibrationData:
  """Resolves Meridian GeoX experiment results into generic calibration data.

  This adapter translates `meridian_geox.api.AnalysisResult` objects and KPI
  type information into generic
  `meridian.model.calibration.base.CalibrationData` objects and performs
  necessary KPI conversions.

  Args:
    result: The `meridian_geox.api.AnalysisResult` containing the Meridian GeoX
      experiment result.
    kpi_type: A string denoting whether the KPI of the incrementality experiment
      result is of a `'revenue'` or `'non-revenue'` type.
    revenue_per_kpi: The revenue per KPI for converting `non-revenue` experiment
      results to `revenue` values when necessary.
    point_estimate_adjustment: The optional point estimate (gamma) adjustment.
    standard_error_adjustment: The optional standard error (tau) adjustment.

  Returns:
    A `meridian.model.calibration.base.CalibrationData` object representing
    the Meridian GeoX experiment data.

  Raises:
    ImportError: If the 'meridian_geox' library is not installed.
    InvalidGeoXResultError: If the result is invalid because of any of the
        following:
        - The GeoX analysis result contains no results.
        - The results contain multi-cell study results.
        - The result is missing the required iCPD estimate.
        - The result is missing the required estimated_bau_spend estimate.
    ValueError: If any of the following are true:
        - `kpi_type` is `'non-revenue'` but `revenue_per_kpi` is `None`.
        - The experiment result is statistically significantly negative.
        - The standard error of the experiment result is not positive.
  """
  if not HAS_GEOX:
    raise ImportError(
        "GeoX calibration is not available because the 'meridian_geox'"
        " library is not installed. Install it via pip:"
        " `pip install google-meridian[geox]`."
    )

  metrics = _validate_and_extract_metrics(result)

  icpd = metrics.icpd
  if icpd is None:
    raise InvalidGeoXResultError(
        "Meridian GeoX experiment metrics must contain the required iCPD"
        " estimate in the icpd field."
    )

  if (
      metrics.descriptive_metrics is None
      or metrics.descriptive_metrics.estimated_bau_spend is None
  ):
    raise InvalidGeoXResultError(
        "Meridian GeoX experiment metrics must contain the required"
        " estimated_bau_spend estimate in the descriptive_metrics field for"
        " running calibration."
    )

  return experiment_adapter.resolve_experiment_source(
      point_estimate=icpd.point_estimate,
      standard_error=icpd.standard_deviation,
      kpi_type=kpi_type,
      revenue_per_kpi=revenue_per_kpi,
      total_spend=metrics.descriptive_metrics.estimated_bau_spend,
      experiment_start_date=result.analysis_config.analysis_start_date.date(),
      experiment_end_date=result.analysis_config.analysis_end_date.date(),
      point_estimate_adjustment=point_estimate_adjustment,
      standard_error_adjustment=standard_error_adjustment,
      source_type=base.SourceType.MERIDIAN_GEOX,
  )
