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

"""Builds Looker Studio report URLs.

This package provides tools to construct URLs for Looker Studio reports that
embed data directly within the URL itself. This is achieved through the creation
of shareable, pre-configured reports without requiring a separate, pre-existing
data source.

The primary functionality is exposed through the `url_generator` module.

Typical Usage:

  1. Use `url_generator.create_report_url()` to create the complete URL, based
  on a `sheets.Spreadsheet` object.

Example:

```python
from lookerstudio.linkingapi import url_generator
from lookerstudio.converters import sheets

# Generate the URL
looker_studio_report_url = url_generator.create_report_url(
    url="some_url",
    id="some_id",
    sheet_id_by_tab_name={},
)
# The `looker_studio_report_url` can now be shared to open a pre-populated
# report.
```
"""

from scenarioplanner.linkingapi import constants
from scenarioplanner.linkingapi import url_generator
