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

"""Generates Meridian Scenario Planner Dashboards in Looker Studio.

This package provides tools to create and manage Meridian dashboards. It helps
transform data from the MMM (Marketing Mix Modeling) schema into a custom
Looker Studio dashboard, which can be shared via a URL.

The typical workflow is:

  1. Analyze MMM data into the appropriate schema.
  2. Generate UI-specific proto messages from this data using
     `mmm_ui_proto_generator`.
  3. Build a Looker Studio URL that embeds this UI proto data using
     `linkingapi`.

Key functionalities include:

  - `linkingapi`: Builds Looker Studio report URLs with embedded data sources.
    This allows for the creation of pre-configured reports.
  - `mmm_ui_proto_generator`: Generates a `Mmm` proto message for the Meridian
    Scenario Planner UI. It takes structured MMM data and transforms it into the
    specific proto format that the dashboard frontend expects.
  - `converters`: Provides utilities to convert and transform analyzed model
    data into a data format that Looker Studio expects.
"""

from scenarioplanner import converters
from scenarioplanner import linkingapi
from scenarioplanner import mmm_ui_proto_generator
