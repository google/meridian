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

from unittest import mock

from absl.testing import absltest
from google.auth import credentials as auth_credentials
import google.auth.transport.requests
from googleapiclient import discovery
from scenarioplanner.converters import sheets
import pandas as pd


class SheetsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock auth.
    self.mock_auth_default = self.enter_context(
        mock.patch.object(
            google.auth,
            'default',
            autospec=True,
        )
    )
    mock_creds = mock.create_autospec(
        auth_credentials.Credentials, instance=True
    )
    self.mock_auth_default.return_value = mock_creds, None
    self.api_mock = mock.Mock()

    # Mock discovery build.
    self.build_mock = self.enter_context(
        mock.patch.object(
            discovery,
            'build',
            return_value=self.api_mock,
            autospec=True,
        )
    )

    # Mock spreadsheet create.
    self.api_create_mock = self.api_mock.spreadsheets().create
    self.api_create_mock.return_value.execute.return_value = {
        'spreadsheetId': 'test_id',
        'title': 'Untitled spreadsheet',
        'spreadsheetUrl': 'test_url',
    }

    # Mock sheet values batch update.
    self.api_batch_update_mock = self.api_mock.spreadsheets().batchUpdate
    self.api_batch_update_mock.return_value.execute.return_value = {
        'replies': [
            {
                'addSheet': {
                    'properties': {
                        'sheetId': 1,
                        'title': 'Tab1',
                    }
                }
            },
            {
                'addSheet': {
                    'properties': {
                        'sheetId': 2,
                        'title': 'Tab2',
                    }
                }
            },
            {'deleteSheet': {'sheetId': 0}},
        ]
    }
    self.api_values_batch_update_mock = (
        self.api_mock.spreadsheets().values().batchUpdate
    )

  def test_upload_to_gsheet_creates_spreadsheet_with_name(self):
    spreadsheet_name = 'test_title'
    sheets.upload_to_gsheet({}, spreadsheet_name=spreadsheet_name)
    self.api_create_mock.assert_called_once_with(
        body={'properties': {'title': spreadsheet_name}},
    )

  def test_upload_to_gsheet_output_is_correct(self):
    spreadsheet = sheets.upload_to_gsheet({})
    self.assertEqual(
        spreadsheet,
        sheets.Spreadsheet(
            id='test_id',
            url='test_url',
            sheet_id_by_tab_name={
                'Tab1': 1,
                'Tab2': 2,
            },
        ),
    )

  def test_upload_to_gsheet_has_called_batch_update_with_correct_requests(self):
    dict_of_dataframes = {
        'Tab1': pd.DataFrame({
            'NumberColumn': [1, 2, 3],
            'StringColumn': ['abc', '', 'def'],
            'NoneColumn': [None, None, None],
        }),
        'Tab2': pd.DataFrame({
            'FloatColumn': [7.1, 8.2],
            'DateColumn': ['2021-02-01,2021-02-22', '2021-02-01,2021-02-22'],
        }),
    }
    sheets.upload_to_gsheet(dict_of_dataframes)

    self.api_batch_update_mock.assert_has_calls([
        mock.call(
            spreadsheetId='test_id',
            body={
                'requests': [
                    {'addSheet': {'properties': {'title': 'Tab1'}}},
                    {'addSheet': {'properties': {'title': 'Tab2'}}},
                    {'deleteSheet': {'sheetId': 0}},
                ]
            },
        ),
        mock.call().execute(),
    ])
    values_request_body = {
        'data': [
            {
                'values': [
                    ['NumberColumn', 'StringColumn', 'NoneColumn'],
                    [1, 'abc', None],
                    [2, '', None],
                    [3, 'def', None],
                ],
                'range': 'Tab1!A1',
            },
            {
                'values': [
                    ['FloatColumn', 'DateColumn'],
                    [7.1, '2021-02-01,2021-02-22'],
                    [8.2, '2021-02-01,2021-02-22'],
                ],
                'range': 'Tab2!A1',
            },
        ],
        'valueInputOption': 'USER_ENTERED',
    }
    self.api_values_batch_update_mock.assert_has_calls([
        mock.call(
            spreadsheetId='test_id',
            body=values_request_body,
        ),
        mock.call().execute(),
    ])


if __name__ == '__main__':
  absltest.main()
