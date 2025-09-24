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
from typing import List
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

  findings: List[EDAFinding]


@dataclasses.dataclass(frozen=True)
class PairwiseCorrEDAOutcome(EDAOutcome):
  """Encapsulates results from the pairwise correlation EDA check.

  Attributes:
      findings: A list of all individual findings related to pairwise
        correlations.
      overall_corr_mat: Pairwise correlation matrix computed across all geos and
        time.
      geo_corr_mat: Pairwise correlation matrix computed across time, for each
        geo.
      overall_extreme_corr_var_pairs: DataFrame of variable pairs exceeding the
        correlation threshold across all geos and time. Columns are 'var1',
        'var2', and 'correlation'.
      geo_extreme_corr_var_pairs: DataFrame of variable pairs exceeding the
        correlation threshold within specific geos. Contains 'geo', 'var1',
        'var2', and 'correlation' information, typically with 'geo' in the
        index.
  """

  overall_corr_mat: xr.DataArray
  geo_corr_mat: xr.DataArray
  overall_extreme_corr_var_pairs: pd.DataFrame
  geo_extreme_corr_var_pairs: pd.DataFrame
