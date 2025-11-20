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

"""Utility library for compiling Google sheets.

This library requires authentication.

*   If you're developing locally, set up Application Default Credentials (ADC)
in
    your local environment:

    <https://cloud.google.com/docs/authentication/application-default-credentials>

*   If you're working in Colab, run the following command in a cell to
    authenticate:

    ```python
    from google.colab import auth
    auth.authenticate_user()
    ```

    This command opens a window where you can complete the authentication.
"""

from collections.abc import Mapping
import dataclasses

import google.auth
from googleapiclient import discovery
import numpy as np
import pandas as pd


__all__ = [
    "Spreadsheet",
    "upload_to_gsheet",
]


# https://developers.google.com/sheets/api/scopes#sheets-scopes
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_DEFAULT_SHEET_ID = 0
_ADD_SHEET_REQUEST_NAME = "addSheet"
_DELETE_SHEET_REQUEST_NAME = "deleteSheet"
_DEFAULT_SPREADSHEET_NAME = "Meridian Looker Studio Data"


@dataclasses.dataclass(frozen=True)
class Spreadsheet:
  """Spreadsheet data class.

  Attributes:
    url: URL of the spreadsheet.
    id: ID of the spreadsheet.
    sheet_id_by_tab_name: Mapping of sheet tab names to sheet IDs.
  """

  url: str
  id: str
  sheet_id_by_tab_name: Mapping[str, int]


def upload_to_gsheet(
    data: Mapping[str, pd.DataFrame],
    credentials: google.auth.credentials.Credentials | None = None,
    spreadsheet_name: str = _DEFAULT_SPREADSHEET_NAME,
) -> Spreadsheet:
  """Creates new spreadsheet.

  Loads pre-authorized user credentials from the environment.

  Args:
    data: Mapping of tab name to dataframe.
    credentials: Optional credentials from the user.
    spreadsheet_name: Name of the spreadsheet.

  Returns:
    Spreadsheet data class.
  """
  if credentials is None:
    credentials, _ = google.auth.default(scopes=_SCOPES)
  service = discovery.build("sheets", "v4", credentials=credentials)
  spreadsheet = (
      service.spreadsheets()
      .create(body={"properties": {"title": spreadsheet_name}})
      .execute()
  )
  spreadsheet_id = spreadsheet["spreadsheetId"]

  # Build requests to add a worksheets and fill them in.
  tab_requests = []
  values_request_body = {
      "data": [],
      "valueInputOption": "USER_ENTERED",
  }
  data = {
      k: v.replace([np.inf, -np.inf, np.nan], None) for k, v in data.items()
  }
  for tab_name, dataframe in data.items():
    tab_requests.append(
        {_ADD_SHEET_REQUEST_NAME: {"properties": {"title": tab_name}}}
    )
    values_request_body["data"].append({
        "values": (
            [dataframe.columns.values.tolist()] + dataframe.values.tolist()
        ),
        "range": f"{tab_name}!A1",
    })
  # Delete first default tab
  tab_requests.append(
      {_DELETE_SHEET_REQUEST_NAME: {"sheetId": _DEFAULT_SHEET_ID}}
  )

  created_tab_objects = (
      service.spreadsheets()
      .batchUpdate(
          spreadsheetId=spreadsheet_id, body={"requests": tab_requests}
      )
      .execute()
  )

  sheet_id_by_tab_name = dict()
  for tab in created_tab_objects["replies"]:
    if _ADD_SHEET_REQUEST_NAME not in tab:
      continue
    add_sheet_response_properties = tab.get(_ADD_SHEET_REQUEST_NAME).get(
        "properties"
    )
    tab_name = add_sheet_response_properties.get("title")
    sheet_id = add_sheet_response_properties.get("sheetId")
    sheet_id_by_tab_name[tab_name] = sheet_id

  # Fill in the data.
  service.spreadsheets().values().batchUpdate(
      spreadsheetId=spreadsheet_id, body=values_request_body
  ).execute()

  return Spreadsheet(
      url=spreadsheet.get("spreadsheetUrl"),
      id=spreadsheet_id,
      sheet_id_by_tab_name=sheet_id_by_tab_name,
  )
