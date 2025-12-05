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
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.model import adstock_hill
from meridian.model import context
from meridian.model import equations
from meridian.model import model_test_data
from meridian.model import spec
import numpy as np


class ModelEquationsTest(
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
    )
    self.equations = equations.ModelEquations(model_context=self.mock_context)

  def test_adstock_hill_media_missing_required_n_times_output(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float32)
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `media` has a number of time periods equal to `n_media_times`.",
    ):
      self.equations.adstock_hill_media(
          media=media[:, :-8, :],
          alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
      )

  def test_adstock_hill_media_n_times_output(self):
    data = self.input_data_with_media_only
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float32)
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autospec=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = media

      self.equations.adstock_hill_media(
          media=media,
          alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
          slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
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

    media = backend.to_tensor(data.media, dtype=backend.float32)
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
        alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        decay_functions=constants.GEOMETRIC_DECAY,
    )

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_adstock_hill_rf_missing_required_n_times_output(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    reach = backend.to_tensor(data.reach, dtype=backend.float32)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float32)
    with self.assertRaisesRegex(
        ValueError,
        "n_times_output is required. This argument is only optional when"
        " `reach` has a number of time periods equal to `n_media_times`.",
    ):
      self.equations.adstock_hill_rf(
          reach=reach[:, :-8, :],
          frequency=frequency,
          alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
          ec=np.ones(shape=(self._N_RF_CHANNELS,)),
          slope=np.ones(shape=(self._N_RF_CHANNELS,)),
          decay_functions=constants.GEOMETRIC_DECAY,
      )

  def test_adstock_hill_rf_n_times_output(self):
    data = self.input_data_with_media_and_rf
    self.mock_context.input_data = data
    self.mock_context.model_spec = spec.ModelSpec()
    self.mock_context.n_media_times = self._N_MEDIA_TIMES
    self.mock_context.n_times = self._N_TIMES

    media = backend.to_tensor(data.media, dtype=backend.float32)
    reach = backend.to_tensor(data.reach, dtype=backend.float32)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float32)
    with mock.patch.object(
        adstock_hill, "AdstockTransformer", autospec=True
    ) as mock_adstock_cls:
      mock_adstock_cls.return_value.forward.return_value = media

      self.equations.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
          ec=np.ones(shape=(self._N_RF_CHANNELS,)),
          slope=np.ones(shape=(self._N_RF_CHANNELS,)),
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

    reach = backend.to_tensor(data.reach, dtype=backend.float32)
    frequency = backend.to_tensor(data.frequency, dtype=backend.float32)
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
        alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
        ec=np.ones(shape=(self._N_RF_CHANNELS,)),
        slope=np.ones(shape=(self._N_RF_CHANNELS,)),
        decay_functions=constants.GEOMETRIC_DECAY,
    )

    expected_called_names = ["mock_hill", "mock_adstock"]

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_compute_non_media_treatments_baseline(self):
    pass

  def test_linear_predictor_counterfactual_difference_media(self):
    pass

  def test_linear_predictor_counterfactual_difference_rf(self):
    pass

  def test_calculate_beta_x(self):
    pass


if __name__ == "__main__":
  absltest.main()
