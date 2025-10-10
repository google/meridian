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

"""Converters for `Mmm` protos to flat dataframes.

This package provides a set of tools for transforming data from `Mmm`
protos into flat dataframes. This conversion makes the data easier to analyze,
visualize, and use in other data processing pipelines.
"""

from scenarioplanner.converters.dataframe import budget_opt_converters
from scenarioplanner.converters.dataframe import common
from scenarioplanner.converters.dataframe import constants
from scenarioplanner.converters.dataframe import converter
from scenarioplanner.converters.dataframe import dataframe_model_converter
from scenarioplanner.converters.dataframe import marketing_analyses_converters
from scenarioplanner.converters.dataframe import rf_opt_converters
