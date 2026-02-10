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

import dataclasses
from unittest import mock

from absl.testing import absltest
from scenarioplanner.converters import sheets
from scenarioplanner.converters.dataframe import constants as dc
from scenarioplanner.linkingapi import constants
from scenarioplanner.linkingapi import url_generator


_SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/test_id"
_ENCODED_SPREADSHEET_URL = (
    "https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Ftest_id"
)
_SPREADSHEET_ID = "test_id"

_REPORT_TEMPLATE_ID = "report_template_id"
_COMMUNITY_CONNECTOR_NAME = "community"
_COMMUNITY_CONNECTOR_ID = "cc_id"
_SHEETS_CONNECTOR_NAME = "google_sheets"
_GA4_MEASUREMENT_ID = "ga4_measurement_id"


class UrlGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_spreadsheet = sheets.Spreadsheet(
        url=_SPREADSHEET_URL,
        id=_SPREADSHEET_ID,
        sheet_id_by_tab_name={},
    )
    self.base_url = (
        "https://lookerstudio.google.com/reporting/create?"
        + f"c.reportId={_REPORT_TEMPLATE_ID}&"
        + f"r.measurementId={_GA4_MEASUREMENT_ID}&"
    )

    self.enter_context(
        mock.patch.object(
            constants,
            "REPORT_TEMPLATE_ID",
            new=_REPORT_TEMPLATE_ID,
        )
    )
    self.enter_context(
        mock.patch.object(
            constants,
            "COMMUNITY_CONNECTOR_NAME",
            new=_COMMUNITY_CONNECTOR_NAME,
        )
    )
    self.enter_context(
        mock.patch.object(
            constants,
            "COMMUNITY_CONNECTOR_ID",
            new=_COMMUNITY_CONNECTOR_ID,
        )
    )
    self.enter_context(
        mock.patch.object(
            constants,
            "SHEETS_CONNECTOR_NAME",
            new=_SHEETS_CONNECTOR_NAME,
        )
    )
    self.enter_context(
        mock.patch.object(
            constants,
            "GA4_MEASUREMENT_ID",
            new=_GA4_MEASUREMENT_ID,
        )
    )

  def test_create_report_url_when_empty_spreadsheet(self):
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}"
    )

    self.assertEqual(
        url_generator.create_report_url(self.base_spreadsheet), expected
    )

  def test_create_report_url_with_community_connector(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.OPTIMIZATION_SPECS: 1,
            dc.OPTIMIZATION_RESULTS: 2,
            dc.OPTIMIZATION_RESPONSE_CURVES: 3,
            dc.RF_OPTIMIZATION_SPECS: 4,
            dc.RF_OPTIMIZATION_RESULTS: 5,
        },
    )
    expected = (
        self.base_url
        + f"ds.dscc.connector={_COMMUNITY_CONNECTOR_NAME}&"
        + f"ds.dscc.connectorId={_COMMUNITY_CONNECTOR_ID}&"
        + f"ds.dscc.spreadsheetUrl={_ENCODED_SPREADSHEET_URL}&"
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_community_connector(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No optimization specs found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)

  def test_create_report_url_with_model_fit(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.MODEL_FIT: 1,
        },
    )
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}&"
        + "ds.ds_model_fit.worksheetId=1"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_model_fit(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No model fit found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)

  def test_create_report_url_with_model_diagnostics(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.MODEL_DIAGNOSTICS: 1,
        },
    )
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}&"
        + "ds.ds_model_diag.worksheetId=1"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_model_diagnostics(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No model diagnostics found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)

  def test_create_report_url_with_media_outcome(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.MEDIA_OUTCOME: 1,
        },
    )
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}&"
        + "ds.ds_outcome.worksheetId=1"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_media_outcome(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No media outcome found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)

  def test_create_report_url_with_media_spend(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.MEDIA_SPEND: 1,
        },
    )
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}&"
        + "ds.ds_spend.worksheetId=1"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_media_spend(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No media spend found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)

  def test_create_report_url_with_media_roi(self):
    spreadsheet = dataclasses.replace(
        self.base_spreadsheet,
        sheet_id_by_tab_name={
            dc.MEDIA_ROI: 1,
        },
    )
    expected = (
        self.base_url
        + "ds.*.refreshFields=false&"
        + "ds.*.keepDatasourceName=true&"
        + f"ds.*.connector={_SHEETS_CONNECTOR_NAME}&"
        + f"ds.*.spreadsheetId={_SPREADSHEET_ID}&"
        + "ds.ds_roi.worksheetId=1"
    )

    self.assertEqual(url_generator.create_report_url(spreadsheet), expected)

  def test_create_report_url_warns_when_no_media_roi(self):
    with self.assertWarnsRegex(
        UserWarning,
        "No media ROI found in the spreadsheet. The report will"
        " display its demo data.",
    ):
      url_generator.create_report_url(self.base_spreadsheet)


if __name__ == "__main__":
  absltest.main()
