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

"""Constants specific to Meridian calibration module."""

from meridian.model import prior_distribution

# Constants for prior builder input argument names.
GEOX_RESULTS = "geox_results"
POINT_ESTIMATES = "point_estimates"
STANDARD_ERRORS = "standard_errors"
EXPERIMENT_KPI_TYPES = "experiment_kpi_types"
EXPERIMENT_TOTAL_SPENDS = "experiment_total_spends"
EXPERIMENT_START_DATES = "experiment_start_dates"
EXPERIMENT_END_DATES = "experiment_end_dates"
POINT_ESTIMATE_ADJUSTMENTS = "point_estimate_adjustments"
STANDARD_ERROR_ADJUSTMENTS = "standard_error_adjustments"


def get_default_prior() -> prior_distribution.PriorDistribution:
  """Returns the default prior distribution."""
  return prior_distribution.PriorDistribution()


# The default alpha parameter used in duration adjustments.
DEFAULT_ALPHA = 0.5

# The Z-score multiplier for a 95% confidence interval.
CONFIDENCE_LEVEL_Z_SCORE_95 = 1.96
