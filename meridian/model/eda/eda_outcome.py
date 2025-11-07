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
import typing
import pandas as pd
import xarray as xr

__all__ = [
    "EDASeverity",
    "EDAFinding",
    "AnalysisLevel",
    "AnalysisArtifact",
    "PairwiseCorrArtifact",
    "StandardDeviationArtifact",
    "VIFArtifact",
    "KpiInvariabilityArtifact",
    "EDACheckType",
    "ArtifactType",
    "EDAOutcome",
]


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


@enum.unique
class AnalysisLevel(enum.Enum):
  """Enumeration for the level of an analysis.

  Attributes:
    OVERALL: Computed across all geos and time. When the analysis is performed
      on national data, this level is equivalent to the NATIONAL level.
    NATIONAL: Computed across time for data aggregated to the national level.
      When the analysis is performed on national data, this level is equivalent
      to the OVERALL level.
    GEO: Computed across time, for each geo.
  """

  OVERALL = enum.auto()
  NATIONAL = enum.auto()
  GEO = enum.auto()


@dataclasses.dataclass(frozen=True)
class AnalysisArtifact:
  """Base dataclass for analysis artifacts.

  Specific EDA artifacts should inherit from this class to store check-specific
  data for downstream processing (e.g., plotting).

  Attributes:
    level: The level of the analysis.
  """

  level: AnalysisLevel


@dataclasses.dataclass(frozen=True)
class PairwiseCorrArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a single pairwise correlation analysis.

  Attributes:
    corr_matrix: Pairwise correlation matrix.
    extreme_corr_var_pairs: DataFrame of variable pairs exceeding the
      correlation threshold.
    extreme_corr_threshold: The threshold used to identify extreme correlation
      pairs.
  """

  corr_matrix: xr.DataArray
  extreme_corr_var_pairs: pd.DataFrame
  extreme_corr_threshold: float


@dataclasses.dataclass(frozen=True)
class StandardDeviationArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a standard deviation analysis.

  Attributes:
    variable: The variable for which standard deviation is calculated.
    std_ds: Dataset with stdev_with_outliers and stdev_without_outliers.
    outlier_df: DataFrame with outliers.
  """

  variable: str
  std_ds: xr.Dataset
  outlier_df: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class VIFArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a single VIF analysis.

  Attributes:
    vif_da: DataArray with VIF values.
    outlier_df: DataFrame with extreme VIF values.
  """

  vif_da: xr.DataArray
  outlier_df: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class KpiInvariabilityArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a KPI invariability analysis.

  Attributes:
    kpi_da: DataArray of the KPI that is examined for variability.
    kpi_stdev: The standard deviation of the KPI, which is used to test the KPI
      invariability.
  """

  kpi_da: xr.DataArray
  kpi_stdev: xr.DataArray


@enum.unique
class EDACheckType(enum.Enum):
  """Enumeration for the type of an EDA check."""

  PAIRWISE_CORRELATION = enum.auto()
  STANDARD_DEVIATION = enum.auto()
  MULTICOLLINEARITY = enum.auto()
  KPI_INVARIABILITY = enum.auto()


ArtifactType = typing.TypeVar("ArtifactType", bound="AnalysisArtifact")


@dataclasses.dataclass(frozen=True)
class EDAOutcome(typing.Generic[ArtifactType]):
  """A dataclass for the outcomes of a single EDA check function.

  An EDA check function can discover multiple issues. This object groups all of
  those individual issues, reported as a list of `EDAFinding` objects.

  Attributes:
    check_type: The type of the EDA check that is being performed.
    findings: A list of all individual issues discovered by the check.
    analysis_artifacts: A list of analysis artifacts from the EDA check.
  """

  check_type: EDACheckType
  findings: list[EDAFinding]
  analysis_artifacts: list[ArtifactType]

  @property
  def get_geo_artifact(self) -> ArtifactType | None:
    """Returns the geo-level analysis artifact."""
    for artifact in self.analysis_artifacts:
      if artifact.level == AnalysisLevel.GEO:
        return artifact
    return None

  @property
  def get_national_artifact(self) -> ArtifactType | None:
    """Returns the national-level analysis artifact."""
    for artifact in self.analysis_artifacts:
      if artifact.level == AnalysisLevel.NATIONAL:
        return artifact
    return None
