# Copyright 2025 The Meridian Authors.
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

"""Meridian EDA Outcome."""

import dataclasses
import enum
import pandas as pd
import xarray as xr


@enum.unique
class EDASeverity(enum.Enum):
  """Enumeration for the severity of an EDA check's finding."""

  # For the non-critical findings.
  INFO = enum.auto()
  # For the non-critical findings that require user attention.
  ATTENTION = enum.auto()
  # For unacceptable, model-blocking data errors.
  ERROR = enum.auto()


@dataclasses.dataclass(frozen=True)
class EDAFinding:
  """Encapsulates a single, specific finding from an EDA check.

  Attributes:
      severity: The severity level of the finding.
      explanation: A human-readable description about the EDA check and a
        potential actionable guidance on how to address or interpret this
        specific finding.
  """

  severity: EDASeverity
  explanation: str


@dataclasses.dataclass(frozen=True)
class EDAOutcome:
  """Base dataclass for the outcomes of a single EDA check function.

  An EDA check function can discover multiple issues. This object groups all of
  those individual issues, reported as a list of EDAFinding objects. Specific
  EDA checks should inherit from this class to store check-specific results
  for downstream processing (e.g., plotting).

  Attributes:
      findings: A list of all individual issues discovered by the check.
  """

  findings: list[EDAFinding]


@enum.unique
class AnalysisLevel(enum.Enum):
  """Enumeration for the level of an analysis.

  Attributes:
    OVERALL: Computed across all geos and time.
    NATIONAL: Computed across time for data aggregated to the national level.
    GEO: Computed across time, for each geo.
  """

  OVERALL = enum.auto()
  NATIONAL = enum.auto()
  GEO = enum.auto()


@dataclasses.dataclass(frozen=True)
class PairwiseCorrResult:
  """Encapsulates results from a single pairwise correlation analysis.

  Attributes:
    level: The level of the correlation analysis.
    corr_matrix: Pairwise correlation matrix.
    extreme_corr_var_pairs: DataFrame of variable pairs exceeding the
      correlation threshold.
    extreme_corr_threshold: The threshold used to identify extreme correlation
      pairs.
  """

  level: AnalysisLevel
  corr_matrix: xr.DataArray
  extreme_corr_var_pairs: pd.DataFrame
  extreme_corr_threshold: float


@dataclasses.dataclass(frozen=True)
class PairwiseCorrOutcome(EDAOutcome):
  """Encapsulates results from the pairwise correlation EDA check.

  Attributes:
    findings: A list of all individual findings related to pairwise
      correlations.
    pairwise_corr_results: A list of pairwise correlation results.
  """

  pairwise_corr_results: list[PairwiseCorrResult]


@dataclasses.dataclass(frozen=True)
class StandardDeviationResult:
  """Encapsulates results from a standard deviation analysis.

  Attributes:
    variable: The variable for which standard deviation is calculated.
    level: The level of the analysis.
    std_ds: Dataset with stdev_with_outliers and stdev_without_outliers.
    outlier_df: DataFrame with outliers.
  """

  variable: str
  level: AnalysisLevel
  std_ds: xr.Dataset
  outlier_df: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class StandardDeviationOutcome(EDAOutcome):
  """Encapsulates results from the standard deviation EDA check.

  Attributes:
    findings: A list of all individual findings related to standard deviation.
    std_results: A list of standard deviation results.
  """

  std_results: list[StandardDeviationResult]


@dataclasses.dataclass(frozen=True)
class VIFResult:
  """Encapsulates results from a single VIF analysis.

  Attributes:
    level: The level of the VIF analysis.
    vif_da: DataArray with VIF values.
    outlier_df: DataFrame with extreme VIF values.
  """

  level: AnalysisLevel
  vif_da: xr.DataArray
  outlier_df: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class VIFOutcome(EDAOutcome):
  """Encapsulates results from the VIF EDA check.

  Attributes:
    findings: A list of all individual findings related to VIF.
    vif_results: A list of VIF results.
  """

  vif_results: list[VIFResult]
