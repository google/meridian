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

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.model import changepoints
import numpy as np

_LOG_NORMAL = constants.MEDIA_EFFECTS_LOG_NORMAL
_NORMAL = constants.MEDIA_EFFECTS_NORMAL


def _tensor(x: np.ndarray) -> backend.Tensor:
  return backend.to_tensor(x, dtype=backend.float_dtype)


class IntervalWeightsTest(parameterized.TestCase):

  def test_two_intervals(self):
    weights = changepoints.interval_weights(5, [0, 2])
    np.testing.assert_array_equal(weights, [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
    self.assertEqual(weights.dtype, backend.np_float_dtype)

  def test_three_intervals_columns_sum_to_one(self):
    weights = changepoints.interval_weights(6, (0, 1, 5))
    np.testing.assert_array_equal(
        weights,
        [[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]],
    )
    np.testing.assert_array_equal(weights.sum(axis=0), np.ones(6))

  def test_single_interval(self):
    np.testing.assert_array_equal(
        changepoints.interval_weights(3, [0]), [[1, 1, 1]]
    )

  @parameterized.named_parameters(
      ('empty', 4, [], 'starts with 0'),
      ('not_starting_at_zero', 4, [1, 2], 'starts with 0'),
      ('not_increasing', 4, [0, 2, 2], 'strictly increasing'),
      ('decreasing', 4, [0, 3, 1], 'strictly increasing'),
      ('start_at_n_times', 4, [0, 4], 'less than `n_times`'),
      ('no_times', 0, [0], '`n_times` must be positive'),
  )
  def test_invalid_input_raises(self, n_times, starts, message):
    with self.assertRaisesRegex(ValueError, message):
      changepoints.interval_weights(n_times, starts)


class GetChangepointInfoTest(parameterized.TestCase):

  def test_no_channel_with_changepoints_returns_none(self):
    self.assertIsNone(
        changepoints.get_changepoint_info(
            n_times=5,
            changepoints={'YouTube': [2]},
            channel_names=['TV', 'Search'],
        )
    )

  def test_padded_layout_follows_channel_order(self):
    info = changepoints.get_changepoint_info(
        n_times=6,
        # Dict order differs from the channel order on purpose.
        changepoints={'Search': [1, 4], 'TV': [3], 'YouTube': [2]},
        channel_names=['TV', 'Display', 'Search'],
    )
    assert info is not None
    self.assertEqual(info.channel_names, ('TV', 'Search'))
    np.testing.assert_array_equal(info.channel_indices, [0, 2])
    self.assertEqual(info.n_channels, 2)
    self.assertEqual(info.n_channels_total, 3)
    self.assertEqual(info.max_intervals, 3)
    self.assertEqual(info.interval_starts, ((0, 3), (0, 1, 4)))
    np.testing.assert_array_equal(
        info.mask, [[True, True, False], [True, True, True]]
    )
    np.testing.assert_array_equal(
        info.weights[0],
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        info.weights[1], changepoints.interval_weights(6, [0, 1, 4])
    )
    self.assertEqual(info.weights.dtype, backend.np_float_dtype)

  def test_unsorted_changepoints_are_sorted(self):
    info = changepoints.get_changepoint_info(
        n_times=6, changepoints={'TV': [4, 1]}, channel_names=['TV']
    )
    assert info is not None
    self.assertEqual(info.interval_starts, ((0, 1, 4),))

  def test_max_intervals_pads_the_interval_axis(self):
    info = changepoints.get_changepoint_info(
        n_times=4,
        changepoints={'TV': [2]},
        channel_names=['TV'],
        max_intervals=4,
    )
    assert info is not None
    self.assertEqual(info.max_intervals, 4)
    self.assertEqual(info.weights.shape, (1, 4, 4))
    np.testing.assert_array_equal(info.mask, [[True, True, False, False]])
    np.testing.assert_array_equal(info.weights[0, 2:], np.zeros((2, 4)))

  def test_every_time_period_starts_an_interval(self):
    info = changepoints.get_changepoint_info(
        n_times=3, changepoints={'TV': [1, 2]}, channel_names=['TV']
    )
    assert info is not None
    np.testing.assert_array_equal(info.weights[0], np.eye(3))

  def test_merge_indices(self):
    info = changepoints.get_changepoint_info(
        n_times=5,
        changepoints={'B': [2], 'D': [3]},
        channel_names=['A', 'B', 'C', 'D'],
    )
    assert info is not None
    # Columns 4 and 5 of `concat([x, sub])` hold the replacements for B, D.
    np.testing.assert_array_equal(info.merge_indices, [0, 4, 2, 5])
    np.testing.assert_array_equal(
        info.is_changepoint_channel, [False, True, False, True]
    )

  def test_last_interval_one_hot(self):
    info = changepoints.get_changepoint_info(
        n_times=6,
        changepoints={'TV': [3], 'Search': [1, 4]},
        channel_names=['TV', 'Search'],
    )
    assert info is not None
    np.testing.assert_array_equal(
        info.last_interval_one_hot, [[0, 1, 0], [0, 0, 1]]
    )

  @parameterized.named_parameters(
      ('no_changepoints', {'TV': []}, None, 'has no changepoints'),
      ('duplicate', {'TV': [2, 2]}, None, 'duplicate changepoints'),
      ('first_time_period', {'TV': [0, 2]}, None, 'between 1 and 4'),
      ('negative', {'TV': [-1]}, None, 'between 1 and 4'),
      ('past_the_end', {'TV': [5]}, None, 'between 1 and 4'),
      ('max_intervals_too_small', {'TV': [1, 2]}, 2, 'at least 3'),
  )
  def test_invalid_input_raises(self, points, max_intervals, message):
    with self.assertRaisesRegex(ValueError, message):
      changepoints.get_changepoint_info(
          n_times=5,
          changepoints=points,
          channel_names=['TV'],
          max_intervals=max_intervals,
      )


class ChangepointKnotOverlapsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'exact_overlap',
          [0, 10, 20],
          {'TV': [10, 15], 'Search': [5]},
          {'TV': (10,)},
      ),
      ('no_overlap', [0, 10, 20], {'TV': [9, 11]}, {}),
      ('first_time_period_is_ignored', [0, 10], {'TV': [0]}, {}),
      ('single_knot_never_overlaps', [10], {'TV': [10]}, {}),
      (
          'several_channels_sorted',
          np.array([0, 5, 10, 15]),
          {'TV': [15, 5], 'YouTube': [10]},
          {'TV': (5, 15), 'YouTube': (10,)},
      ),
  )
  def test_overlaps(self, knot_locations, points, expected):
    self.assertEqual(
        changepoints.changepoint_knot_overlaps(knot_locations, points),
        expected,
    )


class SharedChangepointsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_shared_point', {'TV': [5, 10], 'Search': [15]}, {}),
      ('one_channel', {'TV': [5, 10]}, {}),
      (
          'two_channels',
          {'TV': [10, 15], 'Search': [10]},
          {10: ('TV', 'Search')},
      ),
      (
          'channel_order_and_sorted_points',
          {'YouTube': [20, 5], 'TV': [5], 'Search': [20, 5]},
          {5: ('YouTube', 'TV', 'Search'), 20: ('YouTube', 'Search')},
      ),
      ('no_channels', {}, {}),
  )
  def test_shared_changepoints(self, points, expected):
    self.assertEqual(changepoints.shared_changepoints(points), expected)


class TimeVaryingMathTest(test_utils.MeridianTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)
    info = changepoints.get_changepoint_info(
        n_times=6,
        changepoints={'TV': [3], 'Search': [1, 4]},
        channel_names=['TV', 'Display', 'Search'],
    )
    assert info is not None
    self._info: changepoints.ChangepointInfo = info

  def _normal(self, *shape: int) -> np.ndarray:
    return self._rng.normal(size=shape).astype(backend.np_float_dtype)

  def test_expand_intervals_uses_each_periods_interval(self):
    # (chains=2, n_channels=2, max_intervals=3).
    varphi = self._normal(2, 2, 3)
    phi = changepoints.expand_intervals(_tensor(varphi), self._info.weights)
    self.assertEqual(tuple(phi.shape), (2, 6, 2))
    # TV starts intervals at periods 0, 3; Search at periods 0, 1, 4.
    expected = np.stack(
        [
            varphi[:, 0, [0, 0, 0, 1, 1, 1]],
            varphi[:, 1, [0, 1, 1, 1, 2, 2]],
        ],
        axis=-1,
    )
    test_utils.assert_allclose(phi, expected)

  def test_expand_intervals_ignores_padded_intervals(self):
    varphi = self._normal(2, 3)
    padded = varphi.copy()
    padded[0, 2] = 100.0  # TV has only 2 intervals.
    test_utils.assert_allclose(
        changepoints.expand_intervals(_tensor(padded), self._info.weights),
        changepoints.expand_intervals(_tensor(varphi), self._info.weights),
    )

  @parameterized.named_parameters(
      ('log_normal', _LOG_NORMAL), ('normal', _NORMAL)
  )
  def test_time_varying_coefficients(self, dist):
    beta_gx = np.abs(self._normal(2, 4, 2))  # (chains, n_geos, n_channels)
    zeta = np.abs(self._normal(2, 2))
    phi = self._normal(2, 6, 2)
    result = changepoints.time_varying_coefficients(
        beta_gx=_tensor(beta_gx),
        zeta=_tensor(zeta),
        phi=_tensor(phi),
        media_effects_dist=dist,
    )
    time_effect = zeta[:, np.newaxis, np.newaxis, :] * phi[:, np.newaxis]
    if dist == _NORMAL:
      expected = beta_gx[:, :, np.newaxis, :] + time_effect
    else:
      expected = beta_gx[:, :, np.newaxis, :] * np.exp(time_effect)
    self.assertEqual(tuple(result.shape), (2, 4, 6, 2))
    test_utils.assert_allclose(result, expected, rtol=1e-5)

  @parameterized.named_parameters(
      ('log_normal', _LOG_NORMAL), ('normal', _NORMAL)
  )
  def test_time_varying_coefficients_without_time_effect(self, dist):
    beta_gx = np.abs(self._normal(4, 2))
    result = changepoints.time_varying_coefficients(
        beta_gx=_tensor(beta_gx),
        zeta=_tensor(np.zeros(2)),
        phi=_tensor(self._normal(6, 2)),
        media_effects_dist=dist,
    )
    test_utils.assert_allclose(
        result, np.broadcast_to(beta_gx[:, np.newaxis, :], (4, 6, 2))
    )

  def test_merge_channels(self):
    x = self._normal(2, 3)  # (chains, all 3 channels)
    sub = self._normal(2, 2)  # (chains, TV and Search)
    result = changepoints.merge_channels(
        x=_tensor(x), sub=_tensor(sub), info=self._info
    )
    expected = x.copy()
    expected[:, [0, 2]] = sub
    test_utils.assert_allclose(result, expected)

  def test_fill_channels(self):
    x = self._normal(2, 3)  # (chains, all 3 channels)
    result = changepoints.fill_channels(_tensor(x), value=1.0, info=self._info)
    expected = x.copy()
    expected[:, [0, 2]] = 1.0
    test_utils.assert_allclose(result, expected)

  def test_interval_adjustments_distribution(self):
    distribution = changepoints.interval_adjustments_distribution(
        self._info, constants.VARPHI_M
    )
    self.assertEqual(distribution.name, constants.VARPHI_M)
    self.assertEqual(
        list(distribution.event_shape),
        [self._info.n_channels, self._info.max_intervals],
    )
    self.assertEqual(list(distribution.batch_shape), [])


if __name__ == '__main__':
  absltest.main()
