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

"""Tests for the simulator `test_utils.sample_input_data_with_changepoint`."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants as c
from meridian.data import test_utils
import numpy as np
import xarray as xr

_TV = 'TV'
_SEARCH = 'Search'
# Explicit simulator settings, so the expected intervals don't depend on the
# simulator's defaults.
_N_TIMES = 60
_CHANGEPOINT_INDEX = 25


class SampleInputDataWithChangepointTest(parameterized.TestCase):

  def test_data_has_expected_dimensions_and_coordinates(self):
    # Fewer time periods than the 8 lags of the carryover.
    data, _ = test_utils.sample_input_data_with_changepoint(
        n_geos=3, n_times=5, changepoint_index=2
    )
    assert data.media is not None
    assert data.media_spend is not None
    assert data.media_channel is not None
    assert data.revenue_per_kpi is not None

    self.assertEqual(data.kpi_type, c.REVENUE)
    self.assertEqual(data.kpi.dims, (c.GEO, c.TIME))
    self.assertEqual(data.kpi.shape, (3, 5))
    self.assertEqual(data.media.dims, (c.GEO, c.MEDIA_TIME, c.MEDIA_CHANNEL))
    self.assertEqual(data.media.shape, (3, 5, 2))
    self.assertEqual(data.media_spend.dims, (c.GEO, c.TIME, c.MEDIA_CHANNEL))
    self.assertEqual(data.media_spend.shape, (3, 5, 2))
    self.assertEqual(list(data.geo.values), ['geo_0', 'geo_1', 'geo_2'])
    self.assertEqual(list(data.media_channel.values), [_TV, _SEARCH])
    self.assertEqual(
        list(data.time.values),
        ['2021-01-25', '2021-02-01', '2021-02-08', '2021-02-15', '2021-02-22'],
    )
    self.assertEqual(list(data.media_time.values), list(data.time.values))
    self.assertIsNone(data.controls)
    # `InputData` sets a revenue per KPI of 1 for a revenue KPI.
    np.testing.assert_array_equal(data.revenue_per_kpi.values, 1.0)
    self.assertTrue(np.all(np.isfinite(data.kpi.values)))

  def test_same_seed_gives_the_same_data(self):
    data_1, roi_1 = test_utils.sample_input_data_with_changepoint(seed=3)
    data_2, roi_2 = test_utils.sample_input_data_with_changepoint(seed=3)

    xr.testing.assert_identical(data_1.kpi, data_2.kpi)
    xr.testing.assert_identical(data_1.media, data_2.media)
    xr.testing.assert_identical(data_1.media_spend, data_2.media_spend)
    xr.testing.assert_identical(data_1.population, data_2.population)
    self.assertEqual(roi_1, roi_2)

  def test_different_seeds_give_different_data(self):
    data_1, _ = test_utils.sample_input_data_with_changepoint(seed=3)
    data_2, _ = test_utils.sample_input_data_with_changepoint(seed=4)

    self.assertFalse(np.allclose(data_1.kpi.values, data_2.kpi.values))

  @parameterized.named_parameters(
      dict(
          testcase_name='default',
          kwargs={},
          expected={_TV: (1.0, 2.0), _SEARCH: (1.5,)},
      ),
      dict(
          testcase_name='custom',
          kwargs=dict(changepoint_index=30, tv_roi=(3.0, 0.5), search_roi=0.8),
          expected={_TV: (3.0, 0.5), _SEARCH: (0.8,)},
      ),
  )
  def test_true_roi_is_the_given_roi(self, kwargs, expected):
    _, true_roi = test_utils.sample_input_data_with_changepoint(**kwargs)

    self.assertSameElements(true_roi, expected)
    for channel, roi in expected.items():
      np.testing.assert_allclose(true_roi[channel], roi, rtol=1e-9)

  @parameterized.named_parameters(
      dict(
          testcase_name='tv',
          channel=_TV,
          without_channel=dict(tv_roi=(0.0, 0.0)),
          intervals=[(0, _CHANGEPOINT_INDEX), (_CHANGEPOINT_INDEX, _N_TIMES)],
      ),
      dict(
          testcase_name='search',
          channel=_SEARCH,
          without_channel=dict(search_roi=0.0),
          intervals=[(0, _N_TIMES)],
      ),
  )
  def test_kpi_includes_the_incremental_revenue_of_each_interval(
      self, channel, without_channel, intervals
  ):
    settings = dict(n_times=_N_TIMES, changepoint_index=_CHANGEPOINT_INDEX)
    data, true_roi = test_utils.sample_input_data_with_changepoint(**settings)
    # The random draws don't depend on the ROIs, so the difference between the
    # KPIs is the channel's incremental revenue.
    data_without_channel, _ = test_utils.sample_input_data_with_changepoint(
        **settings, **without_channel
    )
    assert data.media_spend is not None
    incremental_revenue = data.kpi.values - data_without_channel.kpi.values
    spend = data.media_spend.sel(media_channel=channel).values

    roi = [
        incremental_revenue[:, start:end].sum() / spend[:, start:end].sum()
        for start, end in intervals
    ]

    np.testing.assert_allclose(roi, true_roi[channel], rtol=1e-6)

  def test_tv_effect_changes_only_from_the_changepoint(self):
    settings = dict(n_times=_N_TIMES, changepoint_index=_CHANGEPOINT_INDEX)
    data, _ = test_utils.sample_input_data_with_changepoint(**settings)
    data_with_flat_tv_roi, _ = test_utils.sample_input_data_with_changepoint(
        **settings, tv_roi=(1.0, 1.0)
    )
    difference = data.kpi.values - data_with_flat_tv_roi.kpi.values

    # Raising the ROI of the second interval leaves the first one unchanged.
    np.testing.assert_allclose(
        difference[:, :_CHANGEPOINT_INDEX], 0.0, atol=1e-6
    )
    self.assertTrue(np.all(difference[:, _CHANGEPOINT_INDEX:] > 0))

  @parameterized.parameters(0, 10, -1)
  def test_changepoint_index_outside_the_time_periods_raises(self, index):
    with self.assertRaisesRegex(
        ValueError, f'must be between 1 and 9, got {index}'
    ):
      test_utils.sample_input_data_with_changepoint(
          n_times=10, changepoint_index=index
      )


if __name__ == '__main__':
  absltest.main()
