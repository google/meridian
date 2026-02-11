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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.model import media
from meridian.model import model
from meridian.model import spec
from mmm.v1.marketing import marketing_data_pb2 as marketing_pb
from meridian.schema.serde import marketing_data
from meridian.schema.serde import test_data
import numpy as np
import xarray.testing as xrt

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format


class MarketingDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, 'Analyzer', autospec=True)
    )
    self._mock_visualizer = self.enter_context(
        mock.patch.object(visualizer, 'ModelDiagnostics', autospec=True)
    )

    self.serde = marketing_data.MarketingDataSerde()

  def _mock_meridian(self) -> mock.MagicMock:
    """Creates a mock MMM object with InferenceData based on given flags.

    Returns:
      A mock MMM object with InferenceData.
    """
    return mock.MagicMock(
        spec=model.Meridian,
        controls_scaled=backend.to_tensor(
            np.full((2, 3), 5.0), dtype=backend.float32
        ),
        kpi_scaled=backend.to_tensor(np.full((4,), 6.0), dtype=backend.float32),
        media_tensors=media.MediaTensors(),
        rf_tensors=media.RfTensors(),
        inference_data=az.InferenceData(),
        model_spec=spec.ModelSpec(),
    )

  def _setup_meridian(self):
    self._mock_meridian = self._mock_meridian()

  @parameterized.named_parameters(
      dict(
          testcase_name='national_media_and_rf_non_revenue',
          input_data=test_data.MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE,
          expected_proto=test_data.MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='national_media_and_rf_non_revenue_no_controls',
          input_data=test_data.MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS,
          expected_proto=test_data.MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='media_paid_expanded_lagged',
          input_data=test_data.MOCK_INPUT_DATA_MEDIA_PAID_EXPANDED_LAGGED,
          expected_proto=test_data.MOCK_PROTO_MEDIA_PAID_EXPANDED_LAGGED,
          n_geos=2,
          n_times=2,
      ),
      dict(
          testcase_name='media_paid_granular_not_lagged',
          input_data=test_data.MOCK_INPUT_DATA_MEDIA_PAID_GRANULAR_NOT_LAGGED,
          expected_proto=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='rf_paid_expanded_lagged',
          input_data=test_data.MOCK_INPUT_DATA_RF_PAID_EXPANDED_LAGGED,
          expected_proto=test_data.MOCK_PROTO_RF_PAID_EXPANDED_LAGGED,
          n_geos=2,
          n_times=2,
      ),
      dict(
          testcase_name='rf_paid_granular_not_lagged',
          input_data=test_data.MOCK_INPUT_DATA_RF_PAID_GRANULAR_NOT_LAGGED,
          expected_proto=test_data.MOCK_PROTO_RF_PAID_GRANULAR_NOT_LAGGED,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='media_organic_expanded_lagged',
          input_data=test_data.MOCK_INPUT_DATA_MEDIA_ORGANIC_EXPANDED_LAGGED,
          expected_proto=test_data.MOCK_PROTO_MEDIA_ORGANIC_EXPANDED_LAGGED,
          n_geos=2,
          n_times=2,
      ),
      dict(
          testcase_name='media_organic_granular_not_lagged',
          input_data=test_data.MOCK_INPUT_DATA_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED,
          expected_proto=test_data.MOCK_PROTO_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='rf_organic_expanded_lagged',
          input_data=test_data.MOCK_INPUT_DATA_RF_ORGANIC_EXPANDED_LAGGED,
          expected_proto=test_data.MOCK_PROTO_RF_ORGANIC_EXPANDED_LAGGED,
          n_geos=2,
          n_times=2,
      ),
      dict(
          testcase_name='rf_organic_granular_not_lagged',
          input_data=test_data.MOCK_INPUT_DATA_RF_ORGANIC_GRANULAR_NOT_LAGGED,
          expected_proto=test_data.MOCK_PROTO_RF_ORGANIC_GRANULAR_NOT_LAGGED,
          n_geos=1,
          n_times=2,
      ),
      dict(
          testcase_name='non_media_treatments',
          input_data=test_data.MOCK_INPUT_DATA_NON_MEDIA_TREATMENTS,
          expected_proto=test_data.MOCK_PROTO_NON_MEDIA_TREATMENTS,
          n_geos=2,
          n_times=2,
      ),
      dict(
          testcase_name='no_revenue_per_kpi',
          input_data=test_data.MOCK_INPUT_DATA_NO_REVENUE_PER_KPI,
          expected_proto=test_data.MOCK_PROTO_NO_REVENUE_PER_KPI,
          n_geos=2,
          n_times=2,
      ),
  )
  def test_serialize_marketing_data(
      self, input_data, expected_proto, n_geos, n_times
  ):
    self._setup_meridian()
    self._mock_meridian.n_geos = n_geos
    self._mock_meridian.n_times = n_times
    self._mock_meridian.input_data = input_data

    actual = self.serde.serialize(input_data)

    compare.assertProtoEqual(self, expected_proto, actual)

  def test_serialize_metadata_unknown_channel_data_name(self):
    input_data = test_data.MOCK_INPUT_DATA_MEDIA_PAID_EXPANDED_LAGGED
    unknown_channel_name = 'unknown_channel'
    input_data.media.name = unknown_channel_name

    with self.assertRaisesRegex(
        ValueError, f'Unknown channel data name: {unknown_channel_name}.'
    ):
      self.serde.serialize(input_data)

  @parameterized.named_parameters(
      dict(
          testcase_name='national_media_and_rf_non_revenue_no_controls',
          marketing_data_proto=test_data.MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS,
          expected_input_data=test_data.MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE_NO_CONTROLS,
      ),
      dict(
          testcase_name='media_and_rf_non_revenue',
          marketing_data_proto=test_data.MOCK_PROTO_NATIONAL_MEDIA_RF_NON_REVENUE,
          expected_input_data=test_data.MOCK_INPUT_DATA_NATIONAL_MEDIA_RF_NON_REVENUE,
      ),
      dict(
          testcase_name='media_paid_expanded_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_MEDIA_PAID_EXPANDED_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_MEDIA_PAID_EXPANDED_LAGGED,
      ),
      dict(
          testcase_name='media_paid_granular_not_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_MEDIA_PAID_GRANULAR_NOT_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_MEDIA_PAID_GRANULAR_NOT_LAGGED,
      ),
      dict(
          testcase_name='rf_paid_expanded_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_RF_PAID_EXPANDED_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_RF_PAID_EXPANDED_LAGGED,
      ),
      dict(
          testcase_name='rf_paid_granular_not_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_RF_PAID_GRANULAR_NOT_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_RF_PAID_GRANULAR_NOT_LAGGED,
      ),
      dict(
          testcase_name='media_organic_expanded_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_MEDIA_ORGANIC_EXPANDED_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_MEDIA_ORGANIC_EXPANDED_LAGGED,
      ),
      dict(
          testcase_name='media_organic_granular_not_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_MEDIA_ORGANIC_GRANULAR_NOT_LAGGED,
      ),
      dict(
          testcase_name='rf_organic_expanded_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_RF_ORGANIC_EXPANDED_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_RF_ORGANIC_EXPANDED_LAGGED,
      ),
      dict(
          testcase_name='rf_organic_granular_not_lagged',
          marketing_data_proto=test_data.MOCK_PROTO_RF_ORGANIC_GRANULAR_NOT_LAGGED,
          expected_input_data=test_data.MOCK_INPUT_DATA_RF_ORGANIC_GRANULAR_NOT_LAGGED,
      ),
      dict(
          testcase_name='non_media_treatments',
          marketing_data_proto=test_data.MOCK_PROTO_NON_MEDIA_TREATMENTS,
          expected_input_data=test_data.MOCK_INPUT_DATA_NON_MEDIA_TREATMENTS,
      ),
      dict(
          testcase_name='no_revenue_per_kpi',
          marketing_data_proto=test_data.MOCK_PROTO_NO_REVENUE_PER_KPI,
          expected_input_data=test_data.MOCK_INPUT_DATA_NO_REVENUE_PER_KPI,
      ),
  )
  def test_deserialize_marketing_data_proto(
      self, marketing_data_proto, expected_input_data
  ):
    deserialized_data = self.serde.deserialize(marketing_data_proto)
    xrt.assert_allclose(
        deserialized_data.population,
        expected_input_data.population,
        atol=0.5,
        rtol=0,
    )
    self.assertEqual(deserialized_data.kpi_type, expected_input_data.kpi_type)
    xrt.assert_allclose(
        deserialized_data.kpi,
        expected_input_data.kpi,
    )
    if expected_input_data.revenue_per_kpi is None:
      self.assertIsNone(
          deserialized_data.revenue_per_kpi,
          'Expected revenue_per_kpi to be None',
      )
    else:
      xrt.assert_allclose(
          deserialized_data.revenue_per_kpi,
          expected_input_data.revenue_per_kpi,
      )
    if expected_input_data.controls is None:
      self.assertIsNone(
          deserialized_data.controls,
          'Expected controls to be None',
      )
    else:
      xrt.assert_allclose(
          deserialized_data.controls,
          expected_input_data.controls,
      )
    if expected_input_data.media is None:
      self.assertIsNone(deserialized_data.media, 'Expected media to be None')
    else:
      xrt.assert_allclose(
          deserialized_data.media,
          expected_input_data.media,
      )

    if expected_input_data.media_spend is None:
      self.assertIsNone(
          deserialized_data.media, 'Expected media_spend to be None'
      )
    else:
      xrt.assert_allclose(
          deserialized_data.media_spend,
          expected_input_data.media_spend,
      )

    if expected_input_data.reach is None:
      self.assertIsNone(deserialized_data.reach, 'Expected reach to be None')
    else:
      xrt.assert_allclose(deserialized_data.reach, expected_input_data.reach)

    if expected_input_data.frequency is None:
      self.assertIsNone(
          deserialized_data.frequency, 'Expected frequency to be None'
      )
    else:
      xrt.assert_allclose(
          deserialized_data.frequency, expected_input_data.frequency
      )

    if expected_input_data.rf_spend is None:
      self.assertIsNone(
          deserialized_data.rf_spend, 'Expected rf_spend to be None'
      )
    else:
      xrt.assert_allclose(
          deserialized_data.rf_spend,
          expected_input_data.rf_spend,
      )

    if expected_input_data.organic_media is None:
      self.assertIsNone(
          deserialized_data.organic_media, 'Expected organic_media to be None'
      )
    else:
      xrt.assert_allclose(
          deserialized_data.organic_media, expected_input_data.organic_media
      )

    if expected_input_data.organic_reach is None:
      self.assertIsNone(
          deserialized_data.organic_reach, 'Expected organic_reach to be None'
      )
    else:
      xrt.assert_allclose(
          deserialized_data.organic_reach, expected_input_data.organic_reach
      )

    if expected_input_data.organic_frequency is None:
      self.assertIsNone(
          deserialized_data.organic_frequency,
          'Expected organic_frequency to be None',
      )
    else:
      xrt.assert_allclose(
          deserialized_data.organic_frequency,
          expected_input_data.organic_frequency,
      )

    if expected_input_data.non_media_treatments is None:
      self.assertIsNone(
          deserialized_data.non_media_treatments,
          'Expected non_media_treatments to be None',
      )
    else:
      xrt.assert_allclose(
          deserialized_data.non_media_treatments,
          expected_input_data.non_media_treatments,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='inconsistent_kpi_type',
          marketing_data_proto=text_format.Parse(
              """
            marketing_data_points {
              geo_info { geo_id: "geo_0" }
              date_interval {
                start_date { year: 2023 month: 1 day: 1 }
                end_date { year: 2023 month: 1 day: 8 }
              }
              kpi { revenue { value: 10 } }
            }
            marketing_data_points {
              geo_info { geo_id: "geo_1" }
              date_interval {
                start_date { year: 2023 month: 1 day: 8 }
                end_date { year: 2023 month: 1 day: 15 }
              }
              kpi { non_revenue { value: 5 } }
            }
            """,
              marketing_pb.MarketingData(),
          ),
          expected_error_message='Inconsistent kpi_type found in the data.',
      ),
      dict(
          testcase_name='missing_kpi_type',
          marketing_data_proto=text_format.Parse(
              """
            marketing_data_points {
              geo_info { geo_id: "geo_0" }
              date_interval {
                start_date { year: 2023 month: 1 day: 1 }
                end_date { year: 2023 month: 1 day: 8 }
              }
            }
            """,
              marketing_pb.MarketingData(),
          ),
          expected_error_message='kpi_type not found in the data.',
      ),
  )
  def test_extract_kpi_type_errors(
      self, marketing_data_proto, expected_error_message
  ):
    with self.assertRaisesRegex(ValueError, expected_error_message):
      self.serde.deserialize(marketing_data_proto)

  def test_extract_controls_missing_data_error(self):
    marketing_data_proto = text_format.Parse(
        """
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 8
            }
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
            media_spend: 123.0
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 8
            }
            end_date {
              year: 2021
              month: 2
              day: 15
            }
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 40.0
            }
            media_spend: 125.0
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.1
            }
          }
        }
        metadata {
          time_dimensions {
            name: "time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          time_dimensions {
            name: "media_time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          channel_dimensions {
            name: "media"
            channels: "ch_paid_0"
          }
          kpi_type: "revenue"
        }
        """,
        marketing_pb.MarketingData(),
    )

    input_data = self.serde.deserialize(marketing_data_proto)
    self.assertIsNone(input_data.controls)

  def test_deserialize_skip_non_rf_channel_in_extract_frequency(self):
    marketing_data_proto = text_format.Parse(
        """
          metadata {
            time_dimensions {
              name: "time"
              dates { year: 2023 month: 1 day: 1 }
              dates { year: 2023 month: 1 day: 8 }
            }
            time_dimensions {
              name: "media_time"
              dates { year: 2023 month: 1 day: 1 }
              dates { year: 2023 month: 1 day: 8 }
            }
            channel_dimensions {
              name: "media"
              channels: "media_channel1"
            }
            channel_dimensions {
              name: "reach"
              channels: "rf_channel1"
            }
            channel_dimensions {
              name: "frequency"
              channels: "rf_channel1"
            }
          }
          marketing_data_points {
            geo_info { geo_id: "geo_0" }
            date_interval {
              start_date { year: 2023 month: 1 day: 1 }
              end_date { year: 2023 month: 1 day: 8 }
            }
            kpi { non_revenue { value: 10 } }
            control_variables {
              name: "control_0"
              value: 31.0
            }
            media_variables {
              channel_name: "media_channel1"
            }
            reach_frequency_variables {
              channel_name: "media_channel1"
              reach: 1
              average_frequency: 2
            }
          }
          marketing_data_points {
            geo_info { geo_id: "geo_0" }
            date_interval {
              start_date { year: 2023 month: 1 day: 8 }
              end_date { year: 2023 month: 1 day: 15 }
            }
            kpi { non_revenue { value: 10 } }
            control_variables {
              name: "control_0"
              value: 31.0
            }
            media_variables {
              channel_name: "media_channel1"
            }
            reach_frequency_variables {
              channel_name: "media_channel1"
              reach: 1
              average_frequency: 2
            }
          }
        """,
        marketing_pb.MarketingData(),
    )
    deserialized_data = self.serde.deserialize(marketing_data_proto)
    self.assertIsNone(deserialized_data.frequency)

  def test_deserialize_time_dimension_with_no_dates(self):
    marketing_data_proto = text_format.Parse(
        """
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 8
            }
          }
          control_variables {
            name: "control_0"
            value: 31.0
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
            media_spend: 123.0
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        metadata {
          time_dimensions {
            name: "time"
          }
          time_dimensions {
            name: "media_time"
          }
          channel_dimensions {
            name: "media"
            channels: "ch_paid_0"
          }
          control_names: "control_0"
          kpi_type: "revenue"
        }
        """,
        marketing_pb.MarketingData(),
    )
    with self.assertRaisesRegex(
        ValueError, 'TimeDimension proto must have at least one date.'
    ):
      self.serde.deserialize(marketing_data_proto)

  def test_deserialize_aggregated_spend_incorrect_start_date_interval(self):
    # Create a MarketingData proto with an incorrectly defined aggregated
    # spend data point (no geo_info, but also wrong date interval).
    marketing_data_proto = text_format.Parse(
        """
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 8
            }
          }
          control_variables {
            name: "control_0"
            value: 31.0
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 8
            }
            end_date {
              year: 2021
              month: 2
              day: 15
            }
          }
          control_variables {
            name: "control_0"
            value: 31.0
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        marketing_data_points {
          date_interval {
            start_date {
              year: 2021
              month: 1
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 22
            }
          }
          media_variables {
            channel_name: "ch_paid_0"
            media_spend: 123.0
          }
        }
        metadata {
          time_dimensions {
            name: "time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          time_dimensions {
            name: "media_time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          channel_dimensions {
            name: "media"
            channels: "ch_paid_0"
          }
          control_names: "control_0"
          kpi_type: "revenue"
        }
        """,
        marketing_pb.MarketingData(),
    )

    deserializer = marketing_data.MarketingDataSerde()
    deserialized_input_data = deserializer.deserialize(marketing_data_proto)

    # This should be granular since the date interval doesn't match.
    self.assertEqual(
        deserialized_input_data.media_spend.sizes,
        {
            c.GEO: 1,
            c.TIME: 1,
            c.MEDIA_CHANNEL: 1,
        },
        'media_spend should have dimensions (geo=1, time=1, media_channel=1)'
        ' when treated as granular',
    )

  def test_deserialize_aggregated_spend_incorrect_end_date_interval(self):
    # Create a MarketingData proto with an incorrectly defined aggregated
    # spend data point (no geo_info, but also wrong date interval).
    marketing_data_proto = text_format.Parse(
        """
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 8
            }
          }
          control_variables {
            name: "control_0"
            value: 31.0
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        marketing_data_points {
          geo_info {
            geo_id: "geo_0"
            population: 11
          }
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 8
            }
            end_date {
              year: 2021
              month: 2
              day: 15
            }
          }
          control_variables {
            name: "control_0"
            value: 31.0
          }
          media_variables {
            channel_name: "ch_paid_0"
            scalar_metric {
              name: "impressions"
              value: 39.0
            }
          }
          kpi {
            name: "revenue"
            revenue {
              value: 1.0
            }
          }
        }
        marketing_data_points {
          date_interval {
            start_date {
              year: 2021
              month: 2
              day: 1
            }
            end_date {
              year: 2021
              month: 2
              day: 16
            }
          }
          media_variables {
            channel_name: "ch_paid_0"
            media_spend: 123.0
          }
        }
        metadata {
          time_dimensions {
            name: "time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          time_dimensions {
            name: "media_time"
            dates {
              year: 2021
              month: 2
              day: 1
            }
            dates {
              year: 2021
              month: 2
              day: 8
            }
          }
          channel_dimensions {
            name: "media"
            channels: "ch_paid_0"
          }
          control_names: "control_0"
          kpi_type: "revenue"
        }
        """,
        marketing_pb.MarketingData(),
    )

    deserializer = marketing_data.MarketingDataSerde()
    deserialized_input_data = deserializer.deserialize(marketing_data_proto)

    # This should be granular since the date interval doesn't match.
    self.assertEqual(
        deserialized_input_data.media_spend.sizes,
        {
            c.GEO: 1,
            c.TIME: 1,
            c.MEDIA_CHANNEL: 1,
        },
        'media_spend should have dimensions (geo=1, time=1, media_channel=1)'
        ' when treated as granular',
    )

  def test_deserialize_aggregated_spend_incorrect_geo_info(self):
    # Create a MarketingData proto with an incorrectly defined aggregated
    # spend data point (geo_info set, and correct date interval).
    marketing_data_proto = text_format.Parse(
        """
          marketing_data_points {
            geo_info {
              geo_id: "geo_0"
              population: 11
            }
            date_interval {
              start_date {
                year: 2021
                month: 2
                day: 1
              }
              end_date {
                year: 2021
                month: 2
                day: 8
              }
            }
            control_variables {
              name: "control_0"
              value: 31.0
            }
            media_variables {
              channel_name: "ch_paid_0"
              scalar_metric {
                name: "impressions"
                value: 39.0
              }
            }
            kpi {
              name: "revenue"
              revenue {
                value: 1.0
              }
            }
          }
          marketing_data_points {
            geo_info {
              geo_id: "geo_0"
              population: 11
            }
            date_interval {
              start_date {
                year: 2021
                month: 2
                day: 8
              }
              end_date {
                year: 2021
                month: 2
                day: 15
              }
            }
            control_variables {
              name: "control_0"
              value: 31.0
            }
            media_variables {
              channel_name: "ch_paid_0"
              scalar_metric {
                name: "impressions"
                value: 39.0
              }
            }
            kpi {
              name: "revenue"
              revenue {
                value: 1.0
              }
            }
          }
          marketing_data_points {
            geo_info {
              geo_id: "geo_0"
              population: 11
            }
            date_interval {
              start_date {
                year: 2021
                month: 2
                day: 1
              }
              end_date {
                year: 2021
                month: 2
                day: 15
              }
            }
            media_variables {
              channel_name: "ch_paid_0"
              media_spend: 123.0
            }
          }
          metadata {
            time_dimensions {
              name: "time"
              dates {
                year: 2021
                month: 2
                day: 1
              }
              dates {
                year: 2021
                month: 2
                day: 8
              }
            }
            time_dimensions {
              name: "media_time"
              dates {
                year: 2021
                month: 2
                day: 1
              }
              dates {
                year: 2021
                month: 2
                day: 8
              }
            }
            channel_dimensions {
              name: "media"
              channels: "ch_paid_0"
            }
            control_names: "control_0"
            kpi_type: "revenue"
          }
          """,
        marketing_pb.MarketingData(),
    )

    deserializer = marketing_data.MarketingDataSerde()
    deserialized_input_data = deserializer.deserialize(marketing_data_proto)

    # This should be granular since the geo_info is set.
    self.assertEqual(
        deserialized_input_data.media_spend.sizes,
        {
            c.GEO: 1,
            c.TIME: 1,
            c.MEDIA_CHANNEL: 1,
        },
        'media_spend should have dimensions (geo=1, time=1, media_channel=1)'
        ' when treated as granular',
    )


if __name__ == '__main__':
  absltest.main()
