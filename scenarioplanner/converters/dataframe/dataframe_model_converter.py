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

"""An output converter that denormalizes into flat data frame tables."""

from collections.abc import Mapping, Sequence

from scenarioplanner.converters import mmm_converter
from scenarioplanner.converters.dataframe import budget_opt_converters
from scenarioplanner.converters.dataframe import converter
from scenarioplanner.converters.dataframe import marketing_analyses_converters
from scenarioplanner.converters.dataframe import rf_opt_converters
import pandas as pd


__all__ = ["DataFrameModelConverter"]


class DataFrameModelConverter(mmm_converter.ModelConverter[pd.DataFrame]):
  """Converts a bound `Mmm` model into denormalized flat data frame tables.

  The denormalized, two-dimensional data frame tables are intended to be
  directly compiled into sheets in a Google Sheets file to be used as a data
  source for a Looker Studio dashboard.

  These data frame tables are:

  *   "ModelDiagnostics"
  *   "ModelFit"
  *   "MediaOutcome"
  *   "MediaSpend"
  *   "MediaROI"
  *   (Named Incremental Outcome Grids)
  *   "budget_opt_specs"
  *   "budget_opt_results"
  *   "response_curves"
  *   (Named R&F ROI Grids)
  *   "rf_opt_specs"
  *   "rf_opt_results"
  """

  _converters: Sequence[type[converter.Converter]] = (
      marketing_analyses_converters.CONVERTERS
      + budget_opt_converters.CONVERTERS
      + rf_opt_converters.CONVERTERS
  )

  def __call__(self, **kwargs) -> Mapping[str, pd.DataFrame]:
    """Converts bound `Mmm` model proto to named, flat data frame tables."""
    output = {}

    for converter_class in self._converters:
      converter_instance = converter_class(self.mmm)  # pytype: disable=not-instantiable
      for table_name, table_data in converter_instance():
        if output.get(table_name) is not None:
          raise ValueError(f"Duplicate table name: {table_name}")
        output[table_name] = table_data

    return output
