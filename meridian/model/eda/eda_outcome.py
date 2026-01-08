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

from collections.abc import Sequence
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
    "FindingCause",
    "PairwiseCorrArtifact",
    "StandardDeviationArtifact",
    "VIFArtifact",
    "KpiInvariabilityArtifact",
    "CostPerMediaUnitArtifact",
    "VariableGeoTimeCollinearityArtifact",
    "EDACheckType",
    "ArtifactType",
    "EDAOutcome",
    "CriticalCheckEDAOutcomes",
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


@enum.unique
class FindingCause(enum.Enum):
  """Enumeration for the type of finding, mapping to specific data tables.

  Attributes:
    NONE: For informational findings that do not indicate a data issue.
    MULTICOLLINEARITY: For findings related to multicollinearity between
      variables (e.g. from VIF or pairwise correlation checks).
    VARIABILITY: For findings related to variables with extreme variability
      issues, such as no variation (e.g. KPI invariability check or standard
      deviation checks).
    INCONSISTENT_DATA: For findings related to inconsistent data points (e.g.
      zero cost with positive media units, from cost per media unit check).
    RUNTIME_ERROR: For findings that indicate a runtime error during an EDA
      check.
    OUTLIER: For findings related to outliers in data (e.g. cost per media unit
      outlier check).
  """

  NONE = enum.auto()
  MULTICOLLINEARITY = enum.auto()
  VARIABILITY = enum.auto()
  INCONSISTENT_DATA = enum.auto()
  RUNTIME_ERROR = enum.auto()
  OUTLIER = enum.auto()


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class EDAFinding:
  """Encapsulates a single, specific finding from an EDA check.

  Attributes:
      severity: The severity level of the finding.
      explanation: A human-readable description about the EDA check and a
        potential actionable guidance on how to address or interpret this
        specific finding.
      finding_cause: The type of finding, mapping to specific data tables.
      associated_artifact: The artifact associated with the finding, if any.
  """

  __hash__ = None

  severity: EDASeverity
  explanation: str
  finding_cause: FindingCause
  associated_artifact: AnalysisArtifact | None = None


@dataclasses.dataclass(frozen=True)
class PairwiseCorrArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a single pairwise correlation analysis.

  Attributes:
    corr_matrix: Pairwise correlation matrix.
    extreme_corr_var_pairs: DataFrame of variable pairs exceeding the
      correlation threshold. Includes 'correlation' and 'abs_correlation'
      columns, and is sorted by 'abs_correlation' in descending order.
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
  # TODO: change this naming
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


@dataclasses.dataclass(frozen=True)
class CostPerMediaUnitArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a Cost per Media Unit analysis.

  Attributes:
    cost_per_media_unit_da: DataArray of cost per media unit.
    cost_media_unit_inconsistency_df: DataFrame of time periods where cost and
      media units are inconsistent (e.g., zero cost with positive media units,
      or positive cost with zero media units).
    outlier_df: DataFrame with outliers of cost per media unit.
  """

  cost_per_media_unit_da: xr.DataArray
  cost_media_unit_inconsistency_df: pd.DataFrame
  outlier_df: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class VariableGeoTimeCollinearityArtifact(AnalysisArtifact):
  """Encapsulates artifacts from a Geo/Time Collinearity analysis for Treatment/Control variables.

  Attributes:
    rsquared_ds: Dataset containing adjusted R-squared values for treatments and
      controls regressed against 'geo' and 'time'.
  """

  rsquared_ds: xr.Dataset


@enum.unique
class EDACheckType(enum.Enum):
  """Enumeration for the type of an EDA check."""

  PAIRWISE_CORRELATION = enum.auto()
  STANDARD_DEVIATION = enum.auto()
  MULTICOLLINEARITY = enum.auto()
  KPI_INVARIABILITY = enum.auto()
  COST_PER_MEDIA_UNIT = enum.auto()
  VARIABLE_GEO_TIME_COLLINEARITY = enum.auto()


ArtifactType = typing.TypeVar("ArtifactType", bound=AnalysisArtifact)


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

  def _get_artifacts_by_level(self, level: AnalysisLevel) -> list[ArtifactType]:
    """Helper method to retrieve artifacts by level.

    Args:
      level: The AnalysisLevel to filter artifacts by.

    Returns:
      A list of AnalysisArtifacts at the specified level.

    Raises:
      ValueError: If no artifacts of the specified level are found.
    """
    artifacts = [
        artifact
        for artifact in self.analysis_artifacts
        if artifact.level == level
    ]

    if not artifacts:
      raise ValueError(
          f"The EDAOutcome for {self.check_type.name} check does not have "
          f"{level.name.lower()} artifacts."
      )
    return artifacts

  def get_geo_artifacts(self) -> list[ArtifactType]:
    """Returns the geo-level analysis artifacts.

    Returns a list to account for checks that produce multiple artifacts
    at the same level (e.g. Standard Deviation check).
    """
    return self._get_artifacts_by_level(AnalysisLevel.GEO)

  def get_national_artifacts(self) -> list[ArtifactType]:
    """Returns the national-level analysis artifacts.

    Returns a list to account for checks that produce multiple artifacts
    at the same level.
    """
    return self._get_artifacts_by_level(AnalysisLevel.NATIONAL)

  def get_overall_artifacts(self) -> list[ArtifactType]:
    """Returns the overall-level analysis artifacts.

    Returns a list to account for checks that produce multiple artifacts
    at the same level.
    """
    return self._get_artifacts_by_level(AnalysisLevel.OVERALL)

  def get_findings_by_cause_and_severity(
      self, finding_cause: FindingCause, severity: EDASeverity
  ) -> Sequence[EDAFinding]:
    """Helper method to retrieve findings by cause and severity."""
    return [
        finding
        for finding in self.findings
        if finding.finding_cause == finding_cause
        and finding.severity == severity
    ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CriticalCheckEDAOutcomes:
  """Encapsulates the outcomes of all critical EDA checks.

  Attributes:
    kpi_invariability: Outcome of the KPI invariability check.
    multicollinearity: Outcome of the multicollinearity (VIF) check.
    pairwise_correlation: Outcome of the pairwise correlation check.
  """

  kpi_invariability: EDAOutcome[KpiInvariabilityArtifact]
  multicollinearity: EDAOutcome[VIFArtifact]
  pairwise_correlation: EDAOutcome[PairwiseCorrArtifact]
