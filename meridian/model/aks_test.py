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

import os
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import data_frame_input_data_builder
# from meridian.data import input_data
# from meridian.data import test_utils
from meridian.model import aks
import pandas as pd


class AKS(parameterized.TestCase):

  # @parameterized.named_parameters(
  #     dict(
  #         testcase_name='national_geo_',
  #         data=test_utils.sample_input_data_from_dataset(
  #             test_utils.random_dataset(
  #                 n_geos=20,
  #                 n_times=50,
  #                 n_media_times=50,
  #                 n_controls=20,
  #                 n_media_channels=30,
  #                 seed=111,
  #             ),
  #             'non_revenue',
  #         ),
  #     )
  # )
  def test_aks(self):
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'test_data',
            '1_800_Flowers_com_impressions_coarse_20.csv',
        )
    )
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.NON_REVENUE,
        default_media_time_column='media_time',
    )
    data = (
        builder.with_kpi(df, 'conversion_pings', 'date_week', 'dma_no')
        .with_controls(
            df,
            ['control_gqv_branded', 'control_gqv_generic'],
            'date_week',
            'dma_no',
        )
        .with_population(df, 'population', 'dma_no')
        .with_media(
            df,
            [
                'impressions_youtube',
                'impressions_search',
                'impressions_pmax',
                'impressions_display',
                'impressions_demandgen',
            ],
            [
                'spend_youtube',
                'spend_search',
                'spend_pmax',
                'spend_display',
                'spend_demandgen',
            ],
            ['youtube', 'search', 'pmax', 'display', 'demandgen'],
            'date_week',
            'dma_no',
        )
        .build()
    )
    aks_obj = aks.AKS()
    actual_knots, actual_model = aks_obj.automatic_knot_selection(data)
    self.assertListEqual(
        actual_knots,
        [
            3,
            4,
            5,
            6,
            7,
            13,
            14,
            15,
            16,
            17,
            18,
            21,
            23,
            24,
            45,
            46,
            50,
            51,
            55,
            56,
            57,
            58,
            59,
            64,
            65,
            66,
            68,
            69,
            70,
            71,
            78,
            97,
            98,
            102,
            103,
            107,
            108,
            109,
            110,
            111,
            114,
        ],
    )
    self.assertIsNotNone(actual_model)


if __name__ == '__main__':
  absltest.main()
