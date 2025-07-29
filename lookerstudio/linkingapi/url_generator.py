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

"""Utility library for generating a Looker Studio report and outputting the URL.

This library requires authentication.

*   If you're developing locally, set up Application Default Credentials (ADC)
in
    your local environment:

    https://cloud.google.com/docs/authentication/application-default-credentials

*   If you're working in Colab, run the following command in a cell to
    authenticate:

    ```python
    from google.colab import auth
    auth.authenticate_user()
    ```

    This command opens a window where you can complete the authentication.
"""

import urllib.parse
import warnings
from lookerstudio.converters import sheets
from lookerstudio.converters.dataframe import constants as dc
from lookerstudio.linkingapi import constants


def create_report_url(spreadsheet: sheets.Spreadsheet) -> str:
  """Creates a Looker Studio report URL based on the given spreadsheet.

  If there are some sheet tabs that are not in `spreadsheet`, the report will
  display its demo data.

  Args:
    spreadsheet: The spreadsheet object that contains the data to visualize in a
      Looker Studio report.

  Returns:
    The URL of the Looker Studio report.
  """
  params = []

  encoded_sheet_url = urllib.parse.quote_plus(spreadsheet.url)

  params.append(f'c.reportId={constants.REPORT_TEMPLATE_ID}')
  params.append(f'r.measurementId={constants.GA4_MEASUREMENT_ID}')

  if dc.OPTIMIZATION_SPECS in spreadsheet.sheet_id_by_tab_name:
    params.append(f'ds.dscc.connector={constants.COMMUNITY_CONNECTOR_NAME}')
    params.append(f'ds.dscc.connectorId={constants.COMMUNITY_CONNECTOR_ID}')
    params.append(f'ds.dscc.spreadsheetUrl={encoded_sheet_url}')
  else:
    warnings.warn(
        'No optimization specs found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  params.append('ds.*.refreshFields=false')
  params.append('ds.*.keepDatasourceName=true')
  params.append(f'ds.*.connector={constants.SHEETS_CONNECTOR_NAME}')
  params.append(f'ds.*.spreadsheetId={spreadsheet.id}')

  if dc.MODEL_FIT in spreadsheet.sheet_id_by_tab_name:
    worksheet_id = spreadsheet.sheet_id_by_tab_name[dc.MODEL_FIT]
    params.append(f'ds.ds_model_fit.worksheetId={worksheet_id}')
  else:
    warnings.warn(
        'No model fit found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  if dc.MODEL_DIAGNOSTICS in spreadsheet.sheet_id_by_tab_name:
    worksheet_id = spreadsheet.sheet_id_by_tab_name[dc.MODEL_DIAGNOSTICS]
    params.append(f'ds.ds_model_diag.worksheetId={worksheet_id}')
  else:
    warnings.warn(
        'No model diagnostics found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  if dc.MEDIA_OUTCOME in spreadsheet.sheet_id_by_tab_name:
    worksheet_id = spreadsheet.sheet_id_by_tab_name[dc.MEDIA_OUTCOME]
    params.append(f'ds.ds_outcome.worksheetId={worksheet_id}')
  else:
    warnings.warn(
        'No media outcome found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  if dc.MEDIA_SPEND in spreadsheet.sheet_id_by_tab_name:
    worksheet_id = spreadsheet.sheet_id_by_tab_name[dc.MEDIA_SPEND]
    params.append(f'ds.ds_spend.worksheetId={worksheet_id}')
  else:
    warnings.warn(
        'No media spend found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  if dc.MEDIA_ROI in spreadsheet.sheet_id_by_tab_name:
    worksheet_id = spreadsheet.sheet_id_by_tab_name[dc.MEDIA_ROI]
    params.append(f'ds.ds_roi.worksheetId={worksheet_id}')
  else:
    warnings.warn(
        'No media ROI found in the spreadsheet. The report will'
        ' display its demo data.'
    )

  joined_params = '&'.join(params)
  report_url = (
      'https://lookerstudio.google.com/reporting/create?' + joined_params
  )

  return report_url
