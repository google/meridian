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

from absl.testing import absltest
from absl.testing import parameterized
from meridian.data import test_utils
from meridian.model import model
from meridian.model.eda import meridian_eda
import pandas as pd


class MeridianEdaTest(parameterized.TestCase):

  def test_plot_pairwise_correlation_on_geo_data(self):
    data = test_utils.sample_input_data_revenue(
        n_geos=10, n_times=20, n_media_channels=3, n_controls=2
    )
    plots = meridian_eda.MeridianEDA(
        model.Meridian(data)
    ).plot_pairwise_correlation(geos=2)
    self.assertLen(plots.vconcat, 2, 'Should concatenate charts for 2 geos.')
    self.assertEqual(
        plots.vconcat[0].layer[0].title,
        'Pairwise correlations among all treatments and controls for geo_3',
    )
    self.assertEqual(
        plots.vconcat[1].layer[0].title,
        'Pairwise correlations among all treatments and controls for geo_4',
    )

    expected_data = pd.DataFrame({
        'var1': [
            'ch_0',
            'ch_0',
            'ch_0',
            'ch_0',
            'ch_1',
            'ch_1',
            'ch_1',
            'ch_2',
            'ch_2',
            'control_0',
        ],
        'var2': [
            'ch_1',
            'ch_2',
            'control_0',
            'control_1',
            'ch_2',
            'control_0',
            'control_1',
            'control_0',
            'control_1',
            'control_1',
        ],
        'correlation': [
            0.089829296,
            -0.000296344,
            0.57335,
            0.4404598,
            -0.08075117,
            0.30931523,
            0.34205836,
            0.3145167,
            0.49990183,
            0.33091488,
        ],
        'idx1': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        'idx2': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
    }).astype({'correlation': 'float32'})

    actual_data = plots.vconcat[0].data

    pd.testing.assert_frame_equal(
        actual_data.reset_index(drop=True),
        expected_data.reset_index(drop=True),
        atol=1e-5,
    )

    expected_data['correlation'] = [
        -0.22606656,
        0.3749229,
        0.19961049,
        0.22778077,
        -0.20215248,
        0.25613737,
        0.3708803,
        0.3394406,
        0.20705077,
        0.5799423,
    ]
    expected_data = expected_data.astype({'correlation': 'float32'})
    pd.testing.assert_frame_equal(
        plots.vconcat[1].data.reset_index(drop=True),
        expected_data.reset_index(drop=True),
        atol=1e-5,
    )

  def test_plot_pairwise_correlation_on_national_data(self):
    data = test_utils.sample_input_data_revenue(
        n_geos=1, n_times=20, n_media_channels=3, n_controls=2
    )
    plots = meridian_eda.MeridianEDA(
        model.Meridian(data)
    ).plot_pairwise_correlation()
    self.assertLen(plots.vconcat, 1, 'Should concatenate charts for 1 geo.')
    self.assertEqual(
        plots.vconcat[0].layer[0].title,
        'Pairwise correlations among all treatments and controls for'
        ' national_geo',
    )

    expected_data = pd.DataFrame({
        'var1': [
            'ch_0',
            'ch_0',
            'ch_0',
            'ch_0',
            'ch_1',
            'ch_1',
            'ch_1',
            'ch_2',
            'ch_2',
            'control_0',
        ],
        'var2': [
            'ch_1',
            'ch_2',
            'control_0',
            'control_1',
            'ch_2',
            'control_0',
            'control_1',
            'control_0',
            'control_1',
            'control_1',
        ],
        'correlation': [
            -0.1960723,
            0.028662149,
            0.4714015,
            0.18931182,
            0.02225872,
            0.018320927,
            0.28400287,
            0.4606406,
            0.36200395,
            0.18966967,
        ],
        'idx1': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        'idx2': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
    }).astype({'correlation': 'float32'})

    actual_data = plots.data
    pd.testing.assert_frame_equal(
        actual_data.reset_index(drop=True),
        expected_data.reset_index(drop=True),
        atol=1e-5,
    )

  def test_plot_pairwise_correlation_on_nationalized_geos(self):
    data = test_utils.sample_input_data_revenue(
        n_geos=5, n_times=20, n_media_channels=3, n_controls=2
    )
    plots = meridian_eda.MeridianEDA(
        model.Meridian(data)
    ).plot_pairwise_correlation('nationalize')
    self.assertLen(plots.vconcat, 1, 'Should concatenate charts for 1 geo.')
    self.assertEqual(
        plots.vconcat[0].layer[0].title,
        'Pairwise correlations among all treatments and controls for'
        ' national_geo',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='geo_does_not_exist',
          geos=['fake_geo'],
          expected_error='Geo fake_geo does not exist in the data.',
      ),
      dict(
          testcase_name='duplicate_geos',
          geos=['geo_0', 'geo_0'],
          expected_error='geos must not contain duplicate values.',
      ),
      dict(
          testcase_name='geos_too_large',
          geos=11,
          expected_error=(
              'geos must be a positive integer less than or equal to the number'
              ' of geos in the data.'
          ),
      ),
      dict(
          testcase_name='geos_negative',
          geos=-1,
          expected_error=(
              'geos must be a positive integer less than or equal to the number'
              ' of geos in the data.'
          ),
      ),
  )
  def test_plot_pairwise_correlation_error_scenarios(
      self, expected_error: str, geos: int | list[str]
  ):
    data = test_utils.sample_input_data_revenue(
        n_geos=10, n_times=20, n_media_channels=3, n_controls=2
    )
    with self.assertRaisesRegex(ValueError, expected_error):
      meridian_eda.MeridianEDA(model.Meridian(data)).plot_pairwise_correlation(
          geos
      )


if __name__ == '__main__':
  absltest.main()
