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
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import adstock_hill
from meridian.model import changepoints
from meridian.model import context
from meridian.model import equations
from meridian.model import model_test_data
from meridian.model import spec
import numpy as np


class ComputeAdstockHillsTest(
    test_utils.MeridianTestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.mock_context = mock.create_autospec(
        context.ModelContext,
        instance=True,
        spec_set=True,
    )
    self.equations = equations.ModelEquations(model_context=self.mock_context)

  def test_adstock_hill_media_missing_required_n_times_output(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `media` has a number of time periods equal to `n_media_times`.",
    ):
      self.equations.adstock_hill_media(
          media=media[:, :-8, :],
          alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
      )

  def test_adstock_hill_media_n_times_output(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autospec=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = media

      self.equations.adstock_hill_media(
          media=media,
          alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          n_times_output=8,
      )

      calls = mock_adstock_cls.call_args_list
      _, mock_kwargs = calls[0]
      self.assertEqual(mock_kwargs["n_times_output"], 8)

  @parameterized.named_parameters(
      dict(
          testcase_name="adstock_first",
          hill_before_adstock=False,
          expected_called_names=["mock_adstock", "mock_hill"],
      ),
      dict(
          testcase_name="hill_first",
          hill_before_adstock=True,
          expected_called_names=["mock_hill", "mock_adstock"],
      ),
  )
  def test_adstock_hill_media(
      self,
      hill_before_adstock,
      expected_called_names,
  ):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec(
        hill_before_adstock=hill_before_adstock,
    )
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=media,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=media,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    self.equations.adstock_hill_media(
        media=media,
        alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
        decay_functions=constants.GEOMETRIC_DECAY,
    )

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  @parameterized.named_parameters(
      dict(
          testcase_name="hill_first",
          hill_before_adstock=True,
          expected_channel_0_func=lambda mock_hill, media: mock_hill[..., 0],
          expected_channel_1_func=lambda mock_hill, media: media[..., 1],
      ),
      dict(
          testcase_name="adstock_first",
          hill_before_adstock=False,
          expected_channel_0_func=lambda mock_adstock, media: mock_adstock[
              ..., 0
          ]
          * 3.0,
          expected_channel_1_func=lambda mock_adstock, media: mock_adstock[
              ..., 1
          ],
      ),
  )
  def test_adstock_hill_media_selective_saturation(
      self,
      hill_before_adstock,
      expected_channel_0_func,
      expected_channel_1_func,
  ):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec(
        hill_before_adstock=hill_before_adstock,
    )
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    n_channels = self._N_MEDIA_CHANNELS
    alpha = backend.ones(shape=(n_channels,))
    ec = backend.ones(shape=(n_channels,))
    slope = backend.ones(shape=(n_channels,))

    mock_output = media * 2.0

    if hill_before_adstock:
      self.enter_context(
          mock.patch.object(
              adstock_hill.HillTransformer,
              "forward",
              autospec=True,
              return_value=mock_output,
          )
      )
      self.enter_context(
          mock.patch.object(
              adstock_hill.AdstockTransformer,
              "forward",
              autospec=True,
              side_effect=lambda self, x: x,
          )
      )
    else:
      self.enter_context(
          mock.patch.object(
              adstock_hill.AdstockTransformer,
              "forward",
              autospec=True,
              return_value=mock_output,
          )
      )
      self.enter_context(
          mock.patch.object(
              adstock_hill.HillTransformer,
              "forward",
              autospec=True,
              side_effect=lambda self, x: x * 3.0,
          )
      )

    saturation_spec = [constants.HILL, "none"] + [constants.HILL] * (
        n_channels - 2
    )

    result = self.equations.adstock_hill_media(
        media=media,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=constants.GEOMETRIC_DECAY,
        saturation_spec=saturation_spec,
    )

    test_utils.assert_allclose(
        result[..., 0], expected_channel_0_func(mock_output, media)
    )
    test_utils.assert_allclose(
        result[..., 1], expected_channel_1_func(mock_output, media)
    )

  def test_adstock_hill_media_invalid_saturation_spec_string(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation_spec: invalid. Must be 'hill' or 'none'.",
    ):
      self.equations.adstock_hill_media(
          media=media,
          alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          saturation_spec="invalid",
      )

  def test_adstock_hill_media_invalid_saturation_spec_sequence(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    saturation_spec = ["invalid"] + [constants.HILL] * (
        self._N_MEDIA_CHANNELS - 1
    )
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation function in saturation_spec: invalid. Must be"
        " 'hill' or 'none'.",
    ):
      self.equations.adstock_hill_media(
          media=media,
          alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          saturation_spec=saturation_spec,
      )

  def test_adstock_hill_rf_missing_required_n_times_output(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `reach` has a number of time periods equal to `n_media_times`.",
    ):
      self.equations.adstock_hill_rf(
          reach=reach[:, :-8, :],
          frequency=frequency,
          alpha=backend.ones(shape=(self._N_RF_CHANNELS,)),
          ec=backend.ones(shape=(self._N_RF_CHANNELS,)),
          slope=backend.ones(shape=(self._N_RF_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
      )

  def test_adstock_hill_rf_n_times_output(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autospec=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = media

      self.equations.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=backend.ones(shape=(self._N_RF_CHANNELS,)),
          ec=backend.ones(shape=(self._N_RF_CHANNELS,)),
          slope=backend.ones(shape=(self._N_RF_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          n_times_output=8,
      )

      calls = mock_adstock_cls.call_args_list
      _, mock_kwargs = calls[0]
      self.assertEqual(mock_kwargs["n_times_output"], 8)

  def test_adstock_hill_rf(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=frequency,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=reach * frequency,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    self.equations.adstock_hill_rf(
        reach=reach,
        frequency=frequency,
        alpha=backend.ones(shape=(self._N_RF_CHANNELS,)),
        ec=backend.ones(shape=(self._N_RF_CHANNELS,)),
        slope=backend.ones(shape=(self._N_RF_CHANNELS,)),
        decay_functions=constants.GEOMETRIC_DECAY,
    )

    expected_called_names = ["mock_hill", "mock_adstock"]

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_adstock_hill_rf_selective_saturation(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    n_channels = self._N_RF_CHANNELS
    alpha = backend.ones(shape=(n_channels,))
    ec = backend.ones(shape=(n_channels,))
    slope = backend.ones(shape=(n_channels,))

    self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            side_effect=lambda self, x: x * 2.0,
        )
    )

    self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            side_effect=lambda self, x: x,
        )
    )

    saturation_spec = [constants.HILL, "none"] + [constants.HILL] * (
        n_channels - 2
    )

    result = self.equations.adstock_hill_rf(
        reach=reach,
        frequency=frequency,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=constants.GEOMETRIC_DECAY,
        saturation_spec=saturation_spec,
    )

    expected_channel_0 = reach[..., 0] * (frequency[..., 0] * 2.0)
    expected_channel_1 = reach[..., 1] * frequency[..., 1]

    test_utils.assert_allclose(result[..., 0], expected_channel_0)
    test_utils.assert_allclose(result[..., 1], expected_channel_1)

  def test_adstock_hill_rf_invalid_saturation_spec_string(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation_spec: invalid. Must be 'hill' or 'none'.",
    ):
      self.equations.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=backend.ones(shape=(self._N_RF_CHANNELS,)),
          ec=backend.ones(shape=(self._N_RF_CHANNELS,)),
          slope=backend.ones(shape=(self._N_RF_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          saturation_spec="invalid",
      )

  def test_adstock_hill_rf_invalid_saturation_spec_sequence(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float_dtype)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float_dtype)
    saturation_spec = ["invalid"] + [constants.HILL] * (self._N_RF_CHANNELS - 1)
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation function in saturation_spec: invalid. Must be"
        " 'hill' or 'none'.",
    ):
      self.equations.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=backend.ones(shape=(self._N_RF_CHANNELS,)),
          ec=backend.ones(shape=(self._N_RF_CHANNELS,)),
          slope=backend.ones(shape=(self._N_RF_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
          saturation_spec=saturation_spec,
      )

  def test_get_saturation_mask(self):
    hill_mask = self.equations.get_saturation_mask(
        saturation_spec=constants.HILL, n_channels=3
    )
    np.testing.assert_array_equal(
        np.asarray(hill_mask), np.array([True, True, True])
    )
    none_mask = self.equations.get_saturation_mask(
        saturation_spec=constants.NONE, n_channels=3
    )
    np.testing.assert_array_equal(
        np.asarray(none_mask), np.array([False, False, False])
    )
    mixed_mask = self.equations.get_saturation_mask(
        saturation_spec=[constants.HILL, constants.NONE, constants.HILL],
        n_channels=3,
    )
    self.assertIsNotNone(mixed_mask)
    np.testing.assert_array_equal(
        np.asarray(mixed_mask), np.array([True, False, True])
    )
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation_spec: invalid. Must be 'hill' or 'none'.",
    ):
      self.equations.get_saturation_mask(
          saturation_spec="invalid", n_channels=3
      )
    with self.assertRaisesRegex(
        ValueError,
        "Invalid saturation function in saturation_spec: invalid. Must be"
        " 'hill' or 'none'.",
    ):
      self.equations.get_saturation_mask(
          saturation_spec=[constants.HILL, "invalid", constants.NONE],
          n_channels=3,
      )

  def test_prepare_adstock_hill_media_missing_required_n_times_output(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `media` has a number of time periods equal to `n_media_times`.",
    ):
      self.equations.prepare_adstock_hill_media(
          media=media[:, :-8, :],
          alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="adstock_first_hill_all",
          hill_before_adstock=False,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
      ),
      dict(
          testcase_name="adstock_first_zero_multiplier",
          hill_before_adstock=False,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.0, 0.0, 0.0],
      ),
      dict(
          testcase_name="adstock_first_saturation_none",
          hill_before_adstock=False,
          saturation_spec=constants.NONE,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
      ),
      dict(
          testcase_name="adstock_first_saturation_mixed_binomial",
          hill_before_adstock=False,
          saturation_spec=[constants.HILL, constants.NONE, constants.HILL],
          decay_functions=constants.BINOMIAL_DECAY,
          multipliers=[0.0, 1.2, 2.5],
      ),
      dict(
          testcase_name="adstock_first_with_time_indices",
          hill_before_adstock=False,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
          time_indices=[1, 3, 5],
      ),
      dict(
          testcase_name="hill_first_hill_all",
          hill_before_adstock=True,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
      ),
      dict(
          testcase_name="hill_first_zero_multiplier",
          hill_before_adstock=True,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.0, 0.0, 0.0],
      ),
      dict(
          testcase_name="hill_first_saturation_none",
          hill_before_adstock=True,
          saturation_spec=constants.NONE,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
      ),
      dict(
          testcase_name="hill_first_saturation_mixed_binomial",
          hill_before_adstock=True,
          saturation_spec=[constants.HILL, constants.NONE, constants.HILL],
          decay_functions=constants.BINOMIAL_DECAY,
          multipliers=[0.0, 1.2, 2.5],
      ),
      dict(
          testcase_name="hill_first_with_time_indices",
          hill_before_adstock=True,
          saturation_spec=constants.HILL,
          decay_functions=constants.GEOMETRIC_DECAY,
          multipliers=[0.5, 1.5, 2.0],
          time_indices=[1, 3, 5],
      ),
  )
  def test_prepare_and_apply_adstock_hill_media_matches_adstock_hill_media(
      self,
      hill_before_adstock,
      saturation_spec,
      decay_functions,
      multipliers,
      time_indices=None,
  ):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec(
        hill_before_adstock=hill_before_adstock,
    )
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float_dtype)
    alpha = backend.to_tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.6, 0.7]], dtype=backend.float_dtype
    )
    ec = backend.to_tensor(
        [[0.8, 1.2, 1.5], [1.0, 0.9, 1.3]], dtype=backend.float_dtype
    )
    slope = backend.to_tensor(
        [[1.0, 2.0, 1.5], [0.8, 1.3, 2.2]], dtype=backend.float_dtype
    )
    multiplier_tensor = backend.to_tensor(
        multipliers, dtype=backend.float_dtype
    )
    time_indices_tensor = (
        backend.to_tensor(time_indices) if time_indices is not None else None
    )

    media_base, base_t1, t2 = self.equations.prepare_adstock_hill_media(
        media=media,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=self._N_TIMES,
        time_indices=time_indices_tensor,
    )
    saturation_mask = self.equations.get_saturation_mask(
        saturation_spec=saturation_spec,
        n_channels=self._N_MEDIA_CHANNELS,
    )
    actual = self.equations.apply_adstock_hill_media(
        multiplier=multiplier_tensor,
        alpha=alpha,
        slope=slope,
        media_base=media_base,
        base_t1=base_t1,
        t2=t2,
        saturation_mask=saturation_mask,
        n_times_output=self._N_TIMES,
        decay_functions=decay_functions,
        time_indices=time_indices_tensor,
    )

    expected = self.equations.adstock_hill_media(
        media=media * multiplier_tensor,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=self._N_TIMES,
        saturation_spec=saturation_spec,
    )
    if time_indices_tensor is not None:
      expected = backend.gather(expected, time_indices_tensor, axis=-2)
    test_utils.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class CalculateBetaXTest(
    test_utils.MeridianTestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples
  _N_GEOS_SMALL = 3
  _N_TIMES_SMALL = 5
  _N_MEDIA_TIMES_SMALL = 6

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.small_data = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=self._N_GEOS_SMALL,
            n_times=self._N_TIMES_SMALL,
            n_media_times=self._N_MEDIA_TIMES_SMALL,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.small_data_no_revenue_per_kpi = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_SMALL,
            n_times=self._N_TIMES_SMALL,
            n_media_times=self._N_MEDIA_TIMES_SMALL,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            seed=0,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="normal",
          input_data_name="small_data",
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          is_non_media=False,
          expected_coef=[[0.004037, 0.004037, 0.004037]],
      ),
      dict(
          testcase_name="normal_no_revenue_per_kpi",
          input_data_name="small_data_no_revenue_per_kpi",
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          is_non_media=False,
          expected_coef=[[0.001286, 0.001286, 0.001286]],
      ),
      dict(
          testcase_name="log_normal",
          input_data_name="small_data",
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
          is_non_media=False,
          expected_coef=[[-5.512325, -5.512325, -5.512325]],
      ),
      dict(
          testcase_name="non_media_normal",
          input_data_name="small_data_no_revenue_per_kpi",
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
          is_non_media=True,
          expected_coef=[[0.001286, 0.001286]],
      ),
  )
  def test_calculate_beta_x(
      self,
      *,
      input_data_name: str,
      media_effects_dist: str,
      is_non_media: bool,
      expected_coef: np.ndarray,
  ):
    data = getattr(self, input_data_name)
    model_spec = spec.ModelSpec(media_effects_dist=media_effects_dist)
    model_context = context.ModelContext(
        input_data=data,
        model_spec=model_spec,
    )
    eqn = equations.ModelEquations(model_context=model_context)
    n_channels = (
        self._N_NON_MEDIA_CHANNELS if is_non_media else self._N_MEDIA_CHANNELS
    )
    eta_x = backend.to_tensor([[0.0] * n_channels], dtype=backend.float_dtype)
    beta_gx_dev = backend.zeros(
        (1, self._N_GEOS_SMALL, n_channels), dtype=backend.float_dtype
    )
    linear_predictor_counterfactual_difference = backend.to_tensor(
        backend.ones((1, self._N_GEOS_SMALL, self._N_TIMES_SMALL, n_channels)),
        dtype=backend.float_dtype,
    )
    incremental_outcome_x = backend.to_tensor(
        [[1.0] * n_channels], dtype=backend.float_dtype
    )

    calculated_beta_x = eqn.calculate_beta_x(
        is_non_media=is_non_media,
        incremental_outcome_x=incremental_outcome_x,
        linear_predictor_counterfactual_difference=linear_predictor_counterfactual_difference,
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
    )

    test_utils.assert_allclose(
        calculated_beta_x,
        backend.to_tensor(expected_coef, dtype=backend.float_dtype),
        rtol=1e-4,
    )


def _other_media_effects_dist(media_effects_dist: str) -> str:
  if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL:
    return constants.MEDIA_EFFECTS_LOG_NORMAL
  return constants.MEDIA_EFFECTS_NORMAL


class CalculateTimeVaryingBetaXTest(
    test_utils.MeridianTestCase, parameterized.TestCase
):

  # pylint: disable=protected-access

  _N_GEOS = 4
  _N_TIMES = 6
  _N_CHANNELS = 2

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)

    def normal(*shape: int) -> np.ndarray:
      return rng.normal(size=shape).astype(backend.np_float_dtype)

    self._inputs = dict(
        incremental_outcome_x=np.abs(normal(3, self._N_CHANNELS)) + 1.0,
        linear_predictor_counterfactual_difference=np.abs(
            normal(3, self._N_GEOS, self._N_TIMES, self._N_CHANNELS)
        ),
        eta_x=np.abs(normal(3, self._N_CHANNELS)) * 0.3,
        beta_gx_dev=normal(3, self._N_GEOS, self._N_CHANNELS),
        zeta_x=np.abs(normal(3, self._N_CHANNELS)) * 0.3,
        phi_tx=normal(3, self._N_TIMES, self._N_CHANNELS),
    )
    self._population = np.abs(normal(self._N_GEOS)) + 1.0
    self._population_scaled_stdev = 1.7
    self._revenue_per_kpi = np.abs(normal(self._N_GEOS, self._N_TIMES)) + 0.5

  def _equations(
      self,
      media_effects_dist: str,
      with_revenue_per_kpi: bool = True,
      changepoint_media_effects_dist: str | None = None,
  ) -> equations.ModelEquations:
    model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )
    model_context.media_effects_dist = media_effects_dist
    model_context._changepoint_media_effects_dist = (
        changepoint_media_effects_dist or media_effects_dist
    )
    model_context.revenue_per_kpi = (
        backend.to_tensor(self._revenue_per_kpi, dtype=backend.float_dtype)
        if with_revenue_per_kpi
        else None
    )
    model_context.n_geos = self._N_GEOS
    model_context.n_times = self._N_TIMES
    model_context.population = backend.to_tensor(
        self._population, dtype=backend.float_dtype
    )
    model_context.kpi_transformer = mock.Mock(
        population_scaled_stdev=backend.to_tensor(
            self._population_scaled_stdev, dtype=backend.float_dtype
        )
    )
    return equations.ModelEquations(model_context)

  def _tensors(self, **overrides: np.ndarray) -> dict[str, backend.Tensor]:
    return {
        name: backend.to_tensor(value, dtype=backend.float_dtype)
        for name, value in (self._inputs | overrides).items()
    }

  @parameterized.product(
      media_effects_dist=(
          constants.MEDIA_EFFECTS_LOG_NORMAL,
          constants.MEDIA_EFFECTS_NORMAL,
      ),
      with_revenue_per_kpi=(True, False),
  )
  def test_matches_the_incremental_outcome(
      self, media_effects_dist, with_revenue_per_kpi
  ):
    # The other channels' distribution doesn't apply.
    beta_x = self._equations(
        _other_media_effects_dist(media_effects_dist),
        with_revenue_per_kpi,
        changepoint_media_effects_dist=media_effects_dist,
    )._calculate_time_varying_beta_x(**self._tensors())
    self.assertEqual(tuple(beta_x.shape), (3, self._N_CHANNELS))

    # Rebuild every geo and time coefficient from the solved `beta_x`, then
    # recompute the incremental outcome over all geos and time periods.
    inputs = self._inputs
    effect = (
        np.asarray(beta_x)[:, np.newaxis, np.newaxis, :]
        + (inputs["eta_x"][:, np.newaxis, :] * inputs["beta_gx_dev"])[
            :, :, np.newaxis, :
        ]
        + (inputs["zeta_x"][:, np.newaxis, :] * inputs["phi_tx"])[
            :, np.newaxis, :, :
        ]
    )
    beta_gtx = (
        effect
        if media_effects_dist == constants.MEDIA_EFFECTS_NORMAL
        else np.exp(effect)
    )
    revenue_per_kpi = (
        self._revenue_per_kpi
        if with_revenue_per_kpi
        else np.ones((self._N_GEOS, self._N_TIMES))
    )
    incremental_outcome = (
        np.einsum(
            "cgtx,gt,g,cgtx->cx",
            inputs["linear_predictor_counterfactual_difference"],
            revenue_per_kpi,
            self._population,
            beta_gtx,
        )
        * self._population_scaled_stdev
    )
    test_utils.assert_allclose(
        incremental_outcome, inputs["incremental_outcome_x"], rtol=1e-4
    )

  @parameterized.named_parameters(
      ("log_normal", constants.MEDIA_EFFECTS_LOG_NORMAL),
      ("normal", constants.MEDIA_EFFECTS_NORMAL),
  )
  def test_without_time_effect_matches_calculate_beta_x(
      self, media_effects_dist
  ):
    eqn = self._equations(media_effects_dist)
    tensors = self._tensors(zeta_x=np.zeros((3, self._N_CHANNELS)))
    expected = eqn.calculate_beta_x(
        is_non_media=False,
        incremental_outcome_x=tensors["incremental_outcome_x"],
        linear_predictor_counterfactual_difference=tensors[
            "linear_predictor_counterfactual_difference"
        ],
        eta_x=tensors["eta_x"],
        beta_gx_dev=tensors["beta_gx_dev"],
    )
    test_utils.assert_allclose(
        eqn._calculate_time_varying_beta_x(**tensors), expected, rtol=1e-5
    )

  def test_beta_x_with_changepoints_combines_both_calculations(self):
    # Channel "b" has changepoints and normal effects; channel "a" is
    # log-normal.
    info = changepoints.get_changepoint_info(
        n_times=self._N_TIMES, changepoints={"b": [3]}, channel_names=["a", "b"]
    )
    assert info is not None
    eqn = self._equations(
        constants.MEDIA_EFFECTS_LOG_NORMAL,
        changepoint_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
    )
    # With log-normal effects, these values would give infinite results.
    incremental_outcome_x = self._inputs["incremental_outcome_x"].copy()
    incremental_outcome_x[:, 1] = 0.0
    beta_gx_dev = self._inputs["beta_gx_dev"].copy()
    beta_gx_dev[..., 1] = 1e3
    tensors = self._tensors(
        incremental_outcome_x=incremental_outcome_x, beta_gx_dev=beta_gx_dev
    )
    zeta_x = tensors["zeta_x"][:, 1:]
    varphi_x = backend.to_tensor(
        np.random.default_rng(1).normal(size=(3, 1, 2)),
        dtype=backend.float_dtype,
    )
    regular_inputs = dict(
        is_non_media=False,
        incremental_outcome_x=tensors["incremental_outcome_x"],
        linear_predictor_counterfactual_difference=tensors[
            "linear_predictor_counterfactual_difference"
        ],
        eta_x=tensors["eta_x"],
        beta_gx_dev=tensors["beta_gx_dev"],
    )
    expected_regular = eqn.calculate_beta_x(**regular_inputs)
    expected_time_varying = eqn._calculate_time_varying_beta_x(
        incremental_outcome_x=tensors["incremental_outcome_x"][:, 1:],
        linear_predictor_counterfactual_difference=tensors[
            "linear_predictor_counterfactual_difference"
        ][..., 1:],
        eta_x=tensors["eta_x"][:, 1:],
        beta_gx_dev=tensors["beta_gx_dev"][..., 1:],
        zeta_x=zeta_x,
        phi_tx=changepoints.expand_intervals(varphi_x, info.weights),
    )

    with mock.patch.object(
        eqn,
        "calculate_beta_x",
        autospec=True,
        side_effect=eqn.calculate_beta_x,
    ) as calculate_beta_x:
      beta_x = eqn._calculate_beta_x_with_changepoints(
          changepoint_info=info,
          incremental_outcome_x=tensors["incremental_outcome_x"],
          linear_predictor_counterfactual_difference=tensors[
              "linear_predictor_counterfactual_difference"
          ],
          eta_x=tensors["eta_x"],
          beta_gx_dev=tensors["beta_gx_dev"],
          zeta_x=zeta_x,
          varphi_x=varphi_x,
      )

    test_utils.assert_allclose(beta_x[:, 0], expected_regular[:, 0], rtol=1e-5)
    test_utils.assert_allclose(
        beta_x[:, 1], expected_time_varying[:, 0], rtol=1e-5
    )
    # The calculation for the other channels gets placeholder values for the
    # channel with changepoints, so that its unused results stay finite.
    regular_call = calculate_beta_x.call_args.kwargs
    test_utils.assert_allclose(regular_call["incremental_outcome_x"][:, 1], 1.0)
    test_utils.assert_allclose(regular_call["beta_gx_dev"][..., 1], 0.0)


class SplitTimeVaryingOutcomeTest(
    test_utils.MeridianTestCase, parameterized.TestCase
):

  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("log_normal", constants.MEDIA_EFFECTS_LOG_NORMAL),
      ("normal", constants.MEDIA_EFFECTS_NORMAL),
  )
  def test_outcome_and_remaining_coefficients(self, media_effects_dist):
    n_draws, n_geos, n_times = 2, 4, 6
    # Channel "a" has 2 intervals, padded to 3; channel "c" has 3; "b" has none.
    info = changepoints.get_changepoint_info(
        n_times=n_times,
        changepoints={"a": [2], "c": [1, 4]},
        channel_names=["a", "b", "c"],
    )
    assert info is not None
    rng = np.random.default_rng(0)

    def normal(*shape: int) -> np.ndarray:
      return rng.normal(size=shape).astype(backend.np_float_dtype)

    media_transformed = np.abs(normal(n_draws, n_geos, n_times, 3))
    beta_gx = np.abs(normal(n_draws, n_geos, 3)) + 0.5
    zeta = np.abs(normal(n_draws, 2)) * 0.3
    varphi = normal(n_draws, 2, 3)
    # The padded interval of "a" must not reach the outcome.
    varphi[:, 0, 2] = 1e3
    model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )
    # The other channels' distribution doesn't apply.
    model_context.media_effects_dist = _other_media_effects_dist(
        media_effects_dist
    )
    model_context._changepoint_media_effects_dist = media_effects_dist

    def to_tensor(x: np.ndarray) -> backend.Tensor:
      return backend.to_tensor(x, dtype=backend.float_dtype)

    outcome, beta_gx_without = equations.ModelEquations(
        model_context
    )._split_time_varying_outcome(
        changepoint_info=info,
        media_transformed=to_tensor(media_transformed),
        beta_gx=to_tensor(beta_gx),
        zeta_x=to_tensor(zeta),
        varphi_x=to_tensor(varphi),
    )

    t = np.arange(n_times)
    phi = {
        0: np.where(t < 2, varphi[:, 0, 0:1], varphi[:, 0, 1:2]),
        2: np.select(
            [t < 1, t < 4],
            [varphi[:, 1, 0:1], varphi[:, 1, 1:2]],
            varphi[:, 1, 2:3],
        ),
    }
    expected_outcome = np.zeros((n_draws, n_geos, n_times))
    for slot, channel in enumerate((0, 2)):
      time_effect = (zeta[:, slot, np.newaxis] * phi[channel])[:, np.newaxis, :]
      base = beta_gx[:, :, channel, np.newaxis]
      coefficient = (
          base * np.exp(time_effect)
          if media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL
          else base + time_effect
      )
      expected_outcome += media_transformed[..., channel] * coefficient
    test_utils.assert_allclose(outcome, expected_outcome, rtol=1e-5)
    expected_beta_gx_without = beta_gx.copy()
    expected_beta_gx_without[..., [0, 2]] = 0.0
    test_utils.assert_allequal(beta_gx_without, expected_beta_gx_without)

  @parameterized.named_parameters(
      ("log_normal", constants.MEDIA_EFFECTS_LOG_NORMAL),
      ("normal", constants.MEDIA_EFFECTS_NORMAL),
  )
  def test_coefficients_by_time(self, media_effects_dist):
    n_draws, n_geos, n_times = 2, 4, 6
    # Channel "a" has changepoints at time periods 2 and 4; "b" has none.
    info = changepoints.get_changepoint_info(
        n_times=n_times, changepoints={"a": [2, 4]}, channel_names=["a", "b"]
    )
    assert info is not None
    rng = np.random.default_rng(0)
    beta_gx = rng.normal(size=(n_draws, n_geos, 2)).astype(
        backend.np_float_dtype
    )
    zeta = np.abs(rng.normal(size=(n_draws, 1))).astype(backend.np_float_dtype)
    varphi = rng.normal(size=(n_draws, 1, 3)).astype(backend.np_float_dtype)
    model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )
    model_context._changepoint_media_effects_dist = media_effects_dist

    coefficients = equations.ModelEquations(
        model_context
    )._coefficients_by_time(
        changepoint_info=info,
        beta_gx=backend.to_tensor(beta_gx),
        zeta_x=backend.to_tensor(zeta),
        varphi_x=backend.to_tensor(varphi),
    )

    # The time effect of each draw and time period.
    time_effect = zeta[:, 0, np.newaxis] * varphi[:, 0, [0, 0, 1, 1, 2, 2]]
    time_effect = time_effect[:, np.newaxis, :]
    expected = np.repeat(beta_gx[:, :, np.newaxis, :], n_times, axis=2)
    if media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL:
      expected[..., 0] *= np.exp(time_effect)
    else:
      expected[..., 0] += time_effect
    test_utils.assert_allclose(coefficients, expected, rtol=1e-5)
    # The channel without changepoints has the same coefficient in every time
    # period.
    test_utils.assert_allequal(
        np.asarray(coefficients)[..., 1], expected[..., 1]
    )


class CoefficientsFromRandomEffectsTest(
    test_utils.MeridianTestCase, parameterized.TestCase
):

  # pylint: disable=protected-access

  def _coefficients(
      self,
      beta_eta_combined: np.ndarray,
      *,
      media_effects_dist: str,
      changepoint_media_effects_dist: str,
      changepoint_info: changepoints.ChangepointInfo | None,
  ) -> np.ndarray:
    model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )
    model_context.media_effects_dist = media_effects_dist
    model_context._changepoint_media_effects_dist = (
        changepoint_media_effects_dist
    )
    return np.asarray(
        equations.ModelEquations(
            model_context
        )._coefficients_from_random_effects(
            backend.to_tensor(beta_eta_combined, dtype=backend.float_dtype),
            changepoint_info,
        )
    )

  @parameterized.product(
      media_effects_dist=(
          constants.MEDIA_EFFECTS_LOG_NORMAL,
          constants.MEDIA_EFFECTS_NORMAL,
      ),
      changepoint_media_effects_dist=(
          constants.MEDIA_EFFECTS_LOG_NORMAL,
          constants.MEDIA_EFFECTS_NORMAL,
      ),
      with_changepoints=(True, False),
  )
  def test_each_channel_uses_its_distribution(
      self,
      media_effects_dist,
      changepoint_media_effects_dist,
      with_changepoints,
  ):
    beta_eta_combined = (
        np.random.default_rng(0)
        .normal(size=(2, 4, 3))
        .astype(backend.np_float_dtype)
    )
    info: changepoints.ChangepointInfo | None = None
    if with_changepoints:
      info = changepoints.get_changepoint_info(
          n_times=6, changepoints={"c": [2]}, channel_names=["a", "b", "c"]
      )

    coefficients = self._coefficients(
        beta_eta_combined,
        media_effects_dist=media_effects_dist,
        changepoint_media_effects_dist=changepoint_media_effects_dist,
        changepoint_info=info,
    )

    def transform(x: np.ndarray, dist: str) -> np.ndarray:
      return x if dist == constants.MEDIA_EFFECTS_NORMAL else np.exp(x)

    expected = transform(beta_eta_combined, media_effects_dist)
    if with_changepoints:
      expected[..., 2] = transform(
          beta_eta_combined[..., 2], changepoint_media_effects_dist
      )
    test_utils.assert_allclose(coefficients, expected, rtol=1e-6)

  def test_large_normal_effect_does_not_overflow(self):
    info = changepoints.get_changepoint_info(
        n_times=6, changepoints={"b": [2]}, channel_names=["a", "b"]
    )
    beta_eta_combined = np.array([[0.5, 1e3]], dtype=backend.np_float_dtype)

    coefficients = self._coefficients(
        beta_eta_combined,
        media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
        changepoint_media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
        changepoint_info=info,
    )

    test_utils.assert_allclose(coefficients, [[np.exp(0.5), 1e3]], rtol=1e-6)


class LinearPredictorCounterfactualDifferenceTest(
    test_utils.MeridianTestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.mock_context = mock.create_autospec(
        context.ModelContext,
        instance=True,
        spec_set=True,
    )
    self.equations = equations.ModelEquations(model_context=self.mock_context)

  def test_linear_predictor_counterfactual_difference_media_no_counterfactual(
      self,
  ):
    self.mock_context.media_tensors = mock.Mock()
    self.mock_context.media_tensors.prior_media_scaled_counterfactual = None
    media_transformed = backend.to_tensor([1.0, 2.0], dtype=backend.float_dtype)

    result = self.equations.linear_predictor_counterfactual_difference_media(
        media_transformed=media_transformed,
        alpha_m=mock.Mock(),
        ec_m=mock.Mock(),
        slope_m=mock.Mock(),
    )
    test_utils.assert_allequal(result, media_transformed)

  def test_linear_predictor_counterfactual_difference_media_with_counterfactual(
      self,
  ):
    self.mock_context.media_tensors = mock.Mock()
    prior_media_counterfactual = mock.Mock()
    self.mock_context.media_tensors.prior_media_scaled_counterfactual = (
        prior_media_counterfactual
    )
    self.mock_context.adstock_decay_spec = mock.Mock()
    self.mock_context.adstock_decay_spec.media = "geometric"

    media_transformed = backend.to_tensor([10.0], dtype=backend.float_dtype)
    counterfactual_result = backend.to_tensor([4.0], dtype=backend.float_dtype)

    with mock.patch.object(
        self.equations, "adstock_hill_media", return_value=counterfactual_result
    ) as mock_adstock:
      alpha_m = mock.Mock()
      ec_m = mock.Mock()
      slope_m = mock.Mock()

      result = self.equations.linear_predictor_counterfactual_difference_media(
          media_transformed=media_transformed,
          alpha_m=alpha_m,
          ec_m=ec_m,
          slope_m=slope_m,
      )

      mock_adstock.assert_called_once_with(
          media=prior_media_counterfactual,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
          decay_functions="geometric",
      )
      test_utils.assert_allclose(
          result, backend.to_tensor([6.0], dtype=backend.float_dtype)
      )

  def test_linear_predictor_counterfactual_difference_rf_no_counterfactual(
      self,
  ):
    self.mock_context.rf_tensors = mock.Mock()
    self.mock_context.rf_tensors.prior_reach_scaled_counterfactual = None
    rf_transformed = backend.to_tensor([1.0, 2.0], dtype=backend.float_dtype)

    result = self.equations.linear_predictor_counterfactual_difference_rf(
        rf_transformed=rf_transformed,
        alpha_rf=mock.Mock(),
        ec_rf=mock.Mock(),
        slope_rf=mock.Mock(),
    )
    test_utils.assert_allequal(result, rf_transformed)

  def test_linear_predictor_counterfactual_difference_rf_with_counterfactual(
      self,
  ):
    self.mock_context.rf_tensors = mock.Mock()
    prior_reach_counterfactual = mock.Mock()
    self.mock_context.rf_tensors.prior_reach_scaled_counterfactual = (
        prior_reach_counterfactual
    )
    self.mock_context.rf_tensors.frequency = mock.Mock()
    self.mock_context.adstock_decay_spec = mock.Mock()
    self.mock_context.adstock_decay_spec.rf = "geometric"

    rf_transformed = backend.to_tensor([10.0], dtype=backend.float_dtype)
    counterfactual_result = backend.to_tensor([4.0], dtype=backend.float_dtype)

    with mock.patch.object(
        self.equations, "adstock_hill_rf", return_value=counterfactual_result
    ) as mock_adstock:
      alpha_rf = mock.Mock()
      ec_rf = mock.Mock()
      slope_rf = mock.Mock()

      result = self.equations.linear_predictor_counterfactual_difference_rf(
          rf_transformed=rf_transformed,
          alpha_rf=alpha_rf,
          ec_rf=ec_rf,
          slope_rf=slope_rf,
      )

      mock_adstock.assert_called_once_with(
          reach=prior_reach_counterfactual,
          frequency=self.mock_context.rf_tensors.frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
          decay_functions="geometric",
      )
      test_utils.assert_allclose(
          result, backend.to_tensor([6.0], dtype=backend.float_dtype)
      )


class ComputeNonMediaTreatmentsBaselineTest(
    test_utils.MeridianTestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_compute_non_media_treatments_baseline_wrong_baseline_values_shape_raises_exception(
      self,
  ):
    data = self.input_data_non_media_and_organic
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The number of non-media channels (2) does not match the number of"
        " baseline values (3).",
    ):
      model_spec = spec.ModelSpec(
          non_media_baseline_values=["min", "max", "min"]
      )
      model_context_instance = context.ModelContext(data, model_spec)
      eqs = equations.ModelEquations(model_context=model_context_instance)
      _ = eqs.compute_non_media_treatments_baseline()

  def test_compute_non_media_treatments_baseline_fails_with_wrong_baseline_type(
      self,
  ):
    data = self.input_data_non_media_and_organic
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid non_media_baseline_values value: 'wrong'. Only"
        " float numbers and strings 'min' and 'max' are supported.",
    ):
      model_spec = spec.ModelSpec(
          non_media_baseline_values=[
              "max",
              "wrong",
          ]
      )
      model_context_instance = context.ModelContext(data, model_spec)
      eqs = equations.ModelEquations(model_context=model_context_instance)
      _ = eqs.compute_non_media_treatments_baseline()

  def test_compute_non_media_treatments_baseline_default(self):
    """Tests default baseline calculation (all 'min')."""
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(non_media_baseline_values=None)
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    expected_baseline = backend.reduce_min(non_media_treatments, axis=[0, 1])  # pyrefly: ignore[bad-argument-type]
    actual_baseline = eqs.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_strings(self):
    """Tests baseline calculation with 'min' and 'max' strings."""
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(non_media_baseline_values=["min", "max"])
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0], axis=[0, 1]  # pyrefly: ignore[unsupported-operation]
    )
    expected_baseline_max = backend.reduce_max(
        non_media_treatments[..., 1], axis=[0, 1]  # pyrefly: ignore[unsupported-operation]
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_max], axis=-1
    )
    actual_baseline = eqs.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_floats(self):
    """Tests baseline calculation with float values."""
    data = self.input_data_non_media_and_organic
    baseline_values = [10.5, -2.3]
    model_spec = spec.ModelSpec(non_media_baseline_values=baseline_values)
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    expected_baseline = backend.to_tensor(
        baseline_values, dtype=backend.float_dtype
    )
    actual_baseline = eqs.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_mixed_float_and_string(
      self,
  ) -> None:
    """Tests baseline calculation with mixed float and string values."""
    data = self.input_data_non_media_and_organic
    baseline_values = ["min", 5.0]
    model_spec = spec.ModelSpec(
        non_media_baseline_values=baseline_values,
    )
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    _, baseline_value_float = baseline_values
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0], axis=[0, 1]  # pyrefly: ignore[unsupported-operation]
    )
    expected_baseline_float = backend.to_tensor(
        baseline_value_float, dtype=backend.float_dtype
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1  # pyrefly: ignore[bad-argument-type]
    )
    test_utils.assert_allclose(
        expected_baseline, eqs.compute_non_media_treatments_baseline()
    )

  def test_compute_non_media_treatments_baseline_mixed_with_population_scaling(
      self,
  ) -> None:
    """Tests baseline calculation with population scaling."""
    data = self.input_data_non_media_and_organic
    baseline_values = ["min", 5.0]
    model_spec = spec.ModelSpec(
        non_media_baseline_values=baseline_values,
        non_media_population_scaling_id=backend.to_tensor([True, False]),  # pyrefly: ignore[bad-argument-type]
    )
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    _, baseline_value_float = baseline_values
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0]  # pyrefly: ignore[unsupported-operation]
        / eqs._context.population[:, backend.newaxis],
        axis=[0, 1],
    )
    expected_baseline_float = backend.to_tensor(
        baseline_value_float, dtype=backend.float_dtype
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1  # pyrefly: ignore[bad-argument-type]
    )
    actual_baseline = eqs.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_mapping(self) -> None:
    """Tests baseline calculation with a mapping of channel names to values."""
    data = self.input_data_non_media_and_organic
    assert data.non_media_channel is not None
    channels = list(data.non_media_channel.values)
    baseline_map = {channels[0]: "min", channels[1]: 5.0}
    model_spec = spec.ModelSpec(non_media_baseline_values=baseline_map)
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0], axis=[0, 1]  # pyrefly: ignore[unsupported-operation]
    )
    expected_baseline_float = backend.to_tensor(5.0, dtype=backend.float_dtype)
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1  # pyrefly: ignore[bad-argument-type]
    )
    test_utils.assert_allclose(
        expected_baseline, eqs.compute_non_media_treatments_baseline()
    )

  def test_compute_non_media_treatments_baseline_mapping_argument(self) -> None:
    """An explicit mapping argument resolves the same as the equivalent list."""
    data = self.input_data_non_media_and_organic
    assert data.non_media_channel is not None
    channels = list(data.non_media_channel.values)
    model_context_instance = context.ModelContext(data, spec.ModelSpec())
    eqs = equations.ModelEquations(model_context=model_context_instance)
    test_utils.assert_allclose(
        eqs.compute_non_media_treatments_baseline(
            non_media_baseline_values={channels[1]: 5.0}
        ),
        # The mapping names only the channel that differs from the default.
        eqs.compute_non_media_treatments_baseline(
            non_media_baseline_values=[constants.NON_MEDIA_BASELINE_MIN, 5.0]
        ),
    )

  def test_compute_non_media_treatments_baseline_mapping_argument_unknown_channel_fails(
      self,
  ) -> None:
    data = self.input_data_non_media_and_organic
    model_context_instance = context.ModelContext(data, spec.ModelSpec())
    eqs = equations.ModelEquations(model_context=model_context_instance)
    with self.assertRaisesRegex(
        ValueError,
        r"`non_media_baseline_values` refers to non-media channels that are not"
        r" in the input data: \['unknown_channel'\]",
    ):
      eqs.compute_non_media_treatments_baseline(
          non_media_baseline_values={"unknown_channel": "min"}
      )

  def test_compute_non_media_treatments_baseline_mapping_unknown_channel_fails(
      self,
  ) -> None:
    data = self.input_data_non_media_and_organic
    baseline_map = {"unknown_channel": "min"}
    model_spec = spec.ModelSpec(non_media_baseline_values=baseline_map)
    with self.assertRaisesRegex(
        ValueError,
        r"`non_media_baseline_values` refers to non-media channels that are not"
        r" in the input data: \['unknown_channel'\]",
    ):
      context.ModelContext(data, model_spec)


if __name__ == "__main__":
  absltest.main()
