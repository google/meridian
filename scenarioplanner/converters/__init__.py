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

"""Provides tools for converting and wrapping MMM schema data.

This package contains modules to transform Marketing Mix Modeling (MMM) protocol
buffer data into other formats and provides high-level wrappers for easier data
manipulation, analysis, and reporting.
"""

from scenarioplanner.converters import dataframe
from scenarioplanner.converters import mmm
from scenarioplanner.converters import mmm_converter
from scenarioplanner.converters import sheets
