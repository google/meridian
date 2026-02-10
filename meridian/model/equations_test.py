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

    media = backend.to_tensor(data.media, dtype=backend.float32)
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

    media = backend.to_tensor(data.media, dtype=backend.float32)
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
        alpha=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=backend.ones(shape=(self._N_MEDIA_CHANNELS,)),
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
    eta_x = backend.to_tensor([[0.0] * n_channels], dtype=backend.float32)
    beta_gx_dev = backend.zeros(
        (1, self._N_GEOS_SMALL, n_channels), dtype=backend.float32
    )
    linear_predictor_counterfactual_difference = backend.to_tensor(
        backend.ones((1, self._N_GEOS_SMALL, self._N_TIMES_SMALL, n_channels)),
        dtype=backend.float32,
    )
    incremental_outcome_x = backend.to_tensor(
        [[1.0] * n_channels], dtype=backend.float32
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
        backend.to_tensor(expected_coef, dtype=backend.float32),
        rtol=1e-4,
    )


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
    media_transformed = backend.to_tensor([1.0, 2.0], dtype=backend.float32)

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

    media_transformed = backend.to_tensor([10.0], dtype=backend.float32)
    counterfactual_result = backend.to_tensor([4.0], dtype=backend.float32)

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
          result, backend.to_tensor([6.0], dtype=backend.float32)
      )

  def test_linear_predictor_counterfactual_difference_rf_no_counterfactual(
      self,
  ):
    self.mock_context.rf_tensors = mock.Mock()
    self.mock_context.rf_tensors.prior_reach_scaled_counterfactual = None
    rf_transformed = backend.to_tensor([1.0, 2.0], dtype=backend.float32)

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

    rf_transformed = backend.to_tensor([10.0], dtype=backend.float32)
    counterfactual_result = backend.to_tensor([4.0], dtype=backend.float32)

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
          result, backend.to_tensor([6.0], dtype=backend.float32)
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
    expected_baseline = backend.reduce_min(non_media_treatments, axis=[0, 1])
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
        non_media_treatments[..., 0], axis=[0, 1]
    )
    expected_baseline_max = backend.reduce_max(
        non_media_treatments[..., 1], axis=[0, 1]
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
        baseline_values, dtype=backend.float32
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
        non_media_treatments[..., 0], axis=[0, 1]
    )
    expected_baseline_float = backend.to_tensor(
        baseline_value_float, dtype=backend.float32
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1
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
        non_media_population_scaling_id=backend.to_tensor([True, False]),
    )
    model_context_instance = context.ModelContext(data, model_spec)
    eqs = equations.ModelEquations(model_context=model_context_instance)
    non_media_treatments = eqs._context.non_media_treatments
    _, baseline_value_float = baseline_values
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0]
        / eqs._context.population[:, backend.newaxis],
        axis=[0, 1],
    )
    expected_baseline_float = backend.to_tensor(
        baseline_value_float, dtype=backend.float32
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1
    )
    actual_baseline = eqs.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)


if __name__ == "__main__":
  absltest.main()
