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

import dataclasses
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import equations
from meridian.model import knots as knots_module
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
import numpy as np


class ModelTest(
    test_utils.MeridianTestCase,
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

  def test_custom_priors_not_passed_in_ok(self):
    data = self.input_data_non_revenue_no_revenue_per_kpi
    meridian = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(
            media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
        ),
    )
    # Compare input data.
    self.assertEqual(meridian.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(
        media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    )

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_custom_priors_okay_with_array_params(self):
    prior = prior_distribution.PriorDistribution(
        roi_m=backend.tfd.LogNormal([1, 1], [1, 1])
    )
    data = self.input_data_non_revenue_no_revenue_per_kpi
    meridian = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior),
    )
    # Compare input data.
    self.assertEqual(meridian.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec(prior=prior)

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_get_knot_info_fails(self):
    error_msg = "Knots must be all non-negative."
    with mock.patch.object(
        knots_module,
        "get_knot_info",
        autospec=True,
        side_effect=ValueError(error_msg),
    ):
      with self.assertRaisesWithLiteralMatch(ValueError, error_msg):
        _ = model.Meridian(
            input_data=self.input_data_with_media_only,
            model_spec=spec.ModelSpec(knots=4),
        ).knot_info

  def test_init_with_default_parameters_works(self):
    data = self.input_data_with_media_only
    meridian = model.Meridian(input_data=data)

    # Compare input data.
    self.assertEqual(meridian.input_data, data)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  @parameterized.named_parameters(
      dict(
          testcase_name="with_default_spec",
          eda_spec_kwargs={},
          expected_eda_spec=eda_spec_module.EDASpec(),
      ),
      dict(
          testcase_name="with_custom_spec",
          eda_spec_kwargs={
              "eda_spec": eda_spec_module.EDASpec(
                  vif_spec=eda_spec_module.VIFSpec(geo_threshold=500.0)
              )
          },
          expected_eda_spec=eda_spec_module.EDASpec(
              vif_spec=eda_spec_module.VIFSpec(geo_threshold=500.0)
          ),
      ),
  )
  def test_eda_engine_and_spec_initialization(
      self, eda_spec_kwargs, expected_eda_spec
  ):
    meridian = model.Meridian(
        input_data=self.input_data_with_media_only, **eda_spec_kwargs
    )

    self.assertIsInstance(meridian.eda_engine, eda_engine.EDAEngine)
    self.assertEqual(meridian.eda_spec, expected_eda_spec)

  def test_equations_initialization(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_only)
    self.assertIsInstance(meridian.equations, equations.ModelEquations)

  def test_base_geo_properties(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertIsNotNone(meridian.eda_engine)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  def test_base_national_properties(self):
    meridian = model.Meridian(input_data=self.national_input_data_media_only)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertIsNotNone(meridian.eda_engine)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  @parameterized.named_parameters(
      dict(
          testcase_name="rf_prior_type_roi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              '`kpi_scaled` cannot be constant with `rf_prior_type` = "roi".'
          ),
      ),
      dict(
          testcase_name="media_prior_type_mroi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with"
              ' `media_prior_type` = "mroi".'
          ),
      ),
      dict(
          testcase_name="organic_media_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with"
              ' `organic_media_prior_type` = "contribution".'
          ),
      ),
      dict(
          testcase_name="organic_rf_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with"
              ' `organic_rf_prior_type` = "contribution".'
          ),
      ),
      dict(
          testcase_name="non_media_treatments_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          error_msg=(
              "`kpi_scaled` cannot be constant with"
              ' `non_media_treatments_prior_type` = "contribution".'
          ),
      ),
  )
  def test_init_validate_kpi_transformer_geo_model(
      self,
      media_prior_type: str,
      rf_prior_type: str,
      organic_media_prior_type: str,
      organic_rf_prior_type: str,
      non_media_treatments_prior_type: str,
      error_msg: str,
  ):
    valid_input_data = self.input_data_non_media_and_organic
    kpi = valid_input_data.kpi
    kpi.data = np.zeros_like(kpi.data)
    zero_kpi_input_data = dataclasses.replace(
        valid_input_data,
        kpi=kpi,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      model.Meridian(
          input_data=zero_kpi_input_data,
          model_spec=spec.ModelSpec(
              media_prior_type=media_prior_type,
              rf_prior_type=rf_prior_type,
              organic_media_prior_type=organic_media_prior_type,
              organic_rf_prior_type=organic_rf_prior_type,
              non_media_treatments_prior_type=non_media_treatments_prior_type,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="media_prior_type_roi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              '`kpi_scaled` cannot be constant with `media_prior_type` = "roi".'
          ),
      ),
      dict(
          testcase_name="media_prior_type_mroi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with `media_prior_type` ="
              ' "mroi".'
          ),
      ),
      dict(
          testcase_name="rf_prior_type_roi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              '`kpi_scaled` cannot be constant with `rf_prior_type` = "roi".'
          ),
      ),
      dict(
          testcase_name="rf_prior_type_mroi",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              '`kpi_scaled` cannot be constant with `rf_prior_type` = "mroi".'
          ),
      ),
      dict(
          testcase_name="organic_media_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with `organic_media_prior_type`"
              ' = "contribution".'
          ),
      ),
      dict(
          testcase_name="organic_rf_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          error_msg=(
              "`kpi_scaled` cannot be constant with `organic_rf_prior_type` ="
              ' "contribution".'
          ),
      ),
      dict(
          testcase_name="non_media_treatments_prior_type_contribution",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          error_msg=(
              "`kpi_scaled` cannot be constant with"
              ' `non_media_treatments_prior_type` = "contribution".'
          ),
      ),
  )
  def test_init_validate_kpi_transformer_national_model(
      self,
      media_prior_type: str,
      rf_prior_type: str,
      organic_media_prior_type: str,
      organic_rf_prior_type: str,
      non_media_treatments_prior_type: str,
      error_msg: str,
  ):
    valid_input_data = self.national_input_data_non_media_and_organic
    kpi = valid_input_data.kpi
    kpi.data = np.zeros_like(kpi.data)
    zero_kpi_input_data = dataclasses.replace(
        valid_input_data,
        kpi=kpi,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        error_msg,
    ):
      model.Meridian(
          input_data=zero_kpi_input_data,
          model_spec=spec.ModelSpec(
              media_prior_type=media_prior_type,
              rf_prior_type=rf_prior_type,
              organic_media_prior_type=organic_media_prior_type,
              organic_rf_prior_type=organic_rf_prior_type,
              non_media_treatments_prior_type=non_media_treatments_prior_type,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          input_data_type="geo",
      ),
      dict(
          testcase_name="national",
          input_data_type="national",
      ),
  )
  def test_init_validate_kpi_transformer_ok(self, input_data_type):
    valid_input_data = (
        self.national_input_data_non_media_and_organic
        if input_data_type == "national"
        else self.input_data_non_media_and_organic
    )
    kpi = valid_input_data.kpi
    kpi.data = np.zeros_like(kpi.data)
    zero_kpi_input_data = dataclasses.replace(
        valid_input_data,
        kpi=kpi,
    )

    prior_type = constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    meridian = model.Meridian(
        input_data=zero_kpi_input_data,
        model_spec=spec.ModelSpec(
            media_prior_type=prior_type,
            rf_prior_type=prior_type,
            organic_media_prior_type=prior_type,
            organic_rf_prior_type=prior_type,
            non_media_treatments_prior_type=prior_type,
        ),
    )
    self.assertIsNotNone(meridian)

  @parameterized.named_parameters(
      dict(testcase_name="geo", is_national=False),
      dict(testcase_name="national", is_national=True),
  )
  def test_validate_kpi_transformer_with_kpi_variability(
      self, is_national: bool
  ):
    valid_input_data = (
        self.national_input_data_non_media_and_organic
        if is_national
        else self.input_data_non_media_and_organic
    )
    meridian = model.Meridian(
        input_data=valid_input_data,
        model_spec=spec.ModelSpec(),
    )
    self.assertIsNotNone(meridian)

  def test_broadcast_prior_distribution_compute_property(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_and_rf)
    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        meridian.prior_broadcast.tau_g_excl_baseline.batch_shape,
        (meridian.n_geos - 1,),
    )

    # Validate `n_knots` shape distributions.
    self.assertEqual(
        meridian.prior_broadcast.knot_values.batch_shape,
        (meridian.knot_info.n_knots,),
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        meridian.prior_broadcast.beta_m,
        meridian.prior_broadcast.eta_m,
        meridian.prior_broadcast.alpha_m,
        meridian.prior_broadcast.ec_m,
        meridian.prior_broadcast.slope_m,
        meridian.prior_broadcast.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_media_channels,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        meridian.prior_broadcast.beta_rf,
        meridian.prior_broadcast.eta_rf,
        meridian.prior_broadcast.alpha_rf,
        meridian.prior_broadcast.ec_rf,
        meridian.prior_broadcast.slope_rf,
        meridian.prior_broadcast.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_rf_channels,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        meridian.prior_broadcast.gamma_c,
        meridian.prior_broadcast.xi_c,
    ]
    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_controls,))

    # Validate sigma -- unique_sigma_for_each_geo is False by default, so sigma
    # should be a scalar batch.
    self.assertEqual(meridian.prior_broadcast.sigma.batch_shape, ())

  # TODO: Move to `equations_test`.
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
    mmm = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(media_effects_dist=media_effects_dist),
    )
    n_channels = (
        self._N_NON_MEDIA_CHANNELS if is_non_media else self._N_MEDIA_CHANNELS
    )
    eta_x = backend.to_tensor([[0.0] * n_channels], dtype=backend.float32)
    beta_gx_dev = backend.zeros(
        (1, self._N_GEOS_SMALL, n_channels), dtype=backend.float32
    )
    linear_predictor_counterfactual_difference = backend.to_tensor(
        np.ones((1, self._N_GEOS_SMALL, self._N_TIMES_SMALL, n_channels)),
        dtype=backend.float32,
    )
    incremental_outcome_x = backend.to_tensor(
        [[1.0] * n_channels], dtype=backend.float32
    )

    calculated_beta_x = mmm.calculate_beta_x(
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

  def test_run_model_fitting_guardrail_error_message(self):
    # Create mock EDA outcomes with ERROR severity findings
    mock_finding1 = mock.Mock()
    mock_finding1.severity = eda_outcome.EDASeverity.ERROR
    mock_finding1.explanation = "Error explanation for PAIRWISE_CORR 1."

    mock_finding2 = mock.Mock()
    mock_finding2.severity = eda_outcome.EDASeverity.ERROR
    mock_finding2.explanation = "Error explanation for PAIRWISE_CORR 2."

    mock_finding3 = mock.Mock()
    mock_finding3.severity = eda_outcome.EDASeverity.ERROR
    mock_finding3.explanation = "Error explanation for MULTICOLLINEARITY 1."

    mock_outcome1 = mock.Mock()
    mock_outcome1.check_type = eda_outcome.EDACheckType.PAIRWISE_CORRELATION
    mock_outcome1.findings = [mock_finding1, mock_finding2]

    mock_outcome2 = mock.Mock()
    mock_outcome2.check_type = eda_outcome.EDACheckType.MULTICOLLINEARITY
    mock_outcome2.findings = [mock_finding3]

    mock_eda_outcomes = self.enter_context(
        mock.patch(
            "meridian.model.model.Meridian.eda_outcomes",
            new_callable=mock.PropertyMock,
        )
    )
    mock_eda_outcomes.return_value = [mock_outcome1, mock_outcome2]
    meridian = model.Meridian(input_data=self.input_data_with_media_only)

    expected_error_message = (
        "Model has critical EDA issues. Please fix before running"
        " `sample_posterior`.\n\nCheck type: PAIRWISE_CORRELATION\n- Error"
        " explanation for PAIRWISE_CORR 1.\n- Error explanation for"
        " PAIRWISE_CORR 2.\nCheck type: MULTICOLLINEARITY\n- Error explanation"
        " for MULTICOLLINEARITY 1.\nFor further details, please refer to"
        " `Meridian.eda_outcomes`."
    )
    with self.assertRaisesWithLiteralMatch(
        model.ModelFittingError, expected_error_message
    ):
      meridian.sample_posterior(n_chains=1, n_adapt=1, n_burnin=1, n_keep=1)


class ModelPersistenceTest(
    test_utils.MeridianTestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_save_and_load_works(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, "joblib")
    mmm = model.Meridian(input_data=self.input_data_with_media_and_rf)
    model.save_mmm(mmm, str(file_path))
    self.assertTrue(os.path.exists(file_path))
    new_mmm = model.load_mmm(file_path)
    for attr in dir(mmm):
      if isinstance(getattr(mmm, attr), (int, bool)):
        with self.subTest(name=attr):
          self.assertEqual(getattr(mmm, attr), getattr(new_mmm, attr))
      elif isinstance(getattr(mmm, attr), backend.Tensor):
        with self.subTest(name=attr):
          test_utils.assert_allclose(getattr(mmm, attr), getattr(new_mmm, attr))

  def test_load_error(self):
    with self.assertRaisesWithLiteralMatch(
        FileNotFoundError, "No such file or directory: this/path/does/not/exist"
    ):
      model.load_mmm("this/path/does/not/exist")


class NonPaidModelTest(
    test_utils.MeridianTestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_base_geo_properties(self):
    data = self.input_data_non_media_and_organic
    meridian = model.Meridian(input_data=data)
    self.assertEqual(meridian.n_geos, self._N_GEOS)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_non_media_channels, self._N_NON_MEDIA_CHANNELS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertFalse(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertIsNotNone(meridian.eda_engine)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  def test_base_national_properties(self):
    data = self.national_input_data_non_media_and_organic
    meridian = model.Meridian(input_data=data)
    self.assertEqual(meridian.n_geos, self._N_GEOS_NATIONAL)
    self.assertEqual(meridian.n_controls, self._N_CONTROLS)
    self.assertEqual(meridian.n_non_media_channels, self._N_NON_MEDIA_CHANNELS)
    self.assertEqual(meridian.n_times, self._N_TIMES)
    self.assertEqual(meridian.n_media_times, self._N_MEDIA_TIMES)
    self.assertTrue(meridian.is_national)
    self.assertIsNotNone(meridian.prior_broadcast)
    self.assertIsNotNone(meridian.inference_data)
    self.assertIsNotNone(meridian.eda_engine)
    self.assertNotIn(constants.PRIOR, meridian.inference_data.attrs)
    self.assertNotIn(constants.POSTERIOR, meridian.inference_data.attrs)

  @parameterized.named_parameters(
      dict(
          testcase_name="media_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="rf_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
      dict(
          testcase_name="media_rf_non_media_and_organic",
          data=data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
              n_media_channels=input_data_samples._N_MEDIA_CHANNELS,
              n_rf_channels=input_data_samples._N_RF_CHANNELS,
              n_non_media_channels=input_data_samples._N_NON_MEDIA_CHANNELS,
              n_organic_media_channels=input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              n_organic_rf_channels=input_data_samples._N_ORGANIC_RF_CHANNELS,
          ),
      ),
  )
  def test_input_data_tensor_properties(self, data):
    meridian = model.Meridian(input_data=data)
    test_utils.assert_allequal(
        backend.to_tensor(data.kpi, dtype=backend.float32),
        meridian.kpi,
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.revenue_per_kpi, dtype=backend.float32),
        meridian.revenue_per_kpi,
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.controls, dtype=backend.float32),
        meridian.controls,
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.non_media_treatments, dtype=backend.float32),
        meridian.non_media_treatments,
    )
    test_utils.assert_allequal(
        backend.to_tensor(data.population, dtype=backend.float32),
        meridian.population,
    )
    if data.media is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.media, dtype=backend.float32),
          meridian.media_tensors.media,
      )
    if data.media_spend is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.media_spend, dtype=backend.float32),
          meridian.media_tensors.media_spend,
      )
    if data.reach is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.reach, dtype=backend.float32),
          meridian.rf_tensors.reach,
      )
    if data.frequency is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.frequency, dtype=backend.float32),
          meridian.rf_tensors.frequency,
      )
    if data.rf_spend is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.rf_spend, dtype=backend.float32),
          meridian.rf_tensors.rf_spend,
      )
    if data.organic_media is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_media, dtype=backend.float32),
          meridian.organic_media_tensors.organic_media,
      )
    if data.organic_reach is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_reach, dtype=backend.float32),
          meridian.organic_rf_tensors.organic_reach,
      )
    if data.organic_frequency is not None:
      test_utils.assert_allequal(
          backend.to_tensor(data.organic_frequency, dtype=backend.float32),
          meridian.organic_rf_tensors.organic_frequency,
      )
    if data.media_spend is not None and data.rf_spend is not None:
      test_utils.assert_allclose(
          backend.concatenate(
              [
                  backend.to_tensor(data.media_spend, dtype=backend.float32),
                  backend.to_tensor(data.rf_spend, dtype=backend.float32),
              ],
              axis=-1,
          ),
          meridian.total_spend,
      )
    elif data.media_spend is not None:
      test_utils.assert_allclose(
          backend.to_tensor(data.media_spend, dtype=backend.float32),
          meridian.total_spend,
      )
    else:
      test_utils.assert_allclose(
          backend.to_tensor(data.rf_spend, dtype=backend.float32),
          meridian.total_spend,
      )

  def test_broadcast_prior_distribution_is_called_in_meridian_init(self):
    data = self.input_data_non_media_and_organic
    meridian = model.Meridian(input_data=data)
    # Validate `tau_g_excl_baseline` distribution.
    self.assertEqual(
        meridian.prior_broadcast.tau_g_excl_baseline.batch_shape,
        (meridian.n_geos - 1,),
    )

    # Validate `n_knots` shape distributions.
    self.assertEqual(
        meridian.prior_broadcast.knot_values.batch_shape,
        (meridian.knot_info.n_knots,),
    )

    # Validate `n_media_channels` shape distributions.
    n_media_channels_distributions_list = [
        meridian.prior_broadcast.beta_m,
        meridian.prior_broadcast.eta_m,
        meridian.prior_broadcast.alpha_m,
        meridian.prior_broadcast.ec_m,
        meridian.prior_broadcast.slope_m,
        meridian.prior_broadcast.roi_m,
    ]
    for broad in n_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_media_channels,))

    # Validate `n_rf_channels` shape distributions.
    n_rf_channels_distributions_list = [
        meridian.prior_broadcast.beta_rf,
        meridian.prior_broadcast.eta_rf,
        meridian.prior_broadcast.alpha_rf,
        meridian.prior_broadcast.ec_rf,
        meridian.prior_broadcast.slope_rf,
        meridian.prior_broadcast.roi_rf,
    ]
    for broad in n_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_rf_channels,))

    # Validate `n_organic_media_channels` shape distributions.
    n_organic_media_channels_distributions_list = [
        meridian.prior_broadcast.beta_om,
        meridian.prior_broadcast.eta_om,
        meridian.prior_broadcast.alpha_om,
        meridian.prior_broadcast.ec_om,
        meridian.prior_broadcast.slope_om,
    ]
    for broad in n_organic_media_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_organic_media_channels,))

    # Validate `n_organic_rf_channels` shape distributions.
    n_organic_rf_channels_distributions_list = [
        meridian.prior_broadcast.beta_orf,
        meridian.prior_broadcast.eta_orf,
        meridian.prior_broadcast.alpha_orf,
        meridian.prior_broadcast.ec_orf,
        meridian.prior_broadcast.slope_orf,
    ]
    for broad in n_organic_rf_channels_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_organic_rf_channels,))

    # Validate `n_controls` shape distributions.
    n_controls_distributions_list = [
        meridian.prior_broadcast.gamma_c,
        meridian.prior_broadcast.xi_c,
    ]
    for broad in n_controls_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_controls,))

    # Validate `n_non_media_channels` shape distributions.
    n_non_media_distributions_list = [
        meridian.prior_broadcast.gamma_n,
        meridian.prior_broadcast.xi_n,
    ]
    for broad in n_non_media_distributions_list:
      self.assertEqual(broad.batch_shape, (meridian.n_non_media_channels,))

    # Validate sigma -- unique_sigma_for_each_geo is False by default, so sigma
    # should be a scalar batch.
    self.assertEqual(meridian.prior_broadcast.sigma.batch_shape, ())

  def test_scaled_data_shape(self):
    data = self.input_data_non_media_and_organic
    controls = data.controls
    meridian = model.Meridian(input_data=data)
    self.assertIsNotNone(meridian.controls_scaled)
    self.assertIsNotNone(controls)
    test_utils.assert_allequal(
        meridian.controls_scaled.shape,  # pytype: disable=attribute-error
        controls.shape,
        err_msg=(
            "Shape of `_controls_scaled` does not match the shape of `controls`"
            " from the input data."
        ),
    )
    self.assertIsNotNone(meridian.non_media_treatments_normalized)
    self.assertIsNotNone(data.non_media_treatments)
    # pytype: disable=attribute-error
    test_utils.assert_allequal(
        meridian.non_media_treatments_normalized.shape,
        data.non_media_treatments.shape,
        err_msg=(
            "Shape of `_non_media_treatments_scaled` does not match the shape"
            " of `non_media_treatments` from the input data."
        ),
    )
    # pytype: enable=attribute-error
    test_utils.assert_allequal(
        meridian.kpi_scaled.shape,
        data.kpi.shape,
        err_msg=(
            "Shape of `_kpi_scaled` does not match the shape of"
            " `kpi` from the input data."
        ),
    )

  def test_population_scaled_non_media_transformer_set(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(
        non_media_population_scaling_id=backend.to_tensor(
            [True for _ in data.non_media_channel]
        )
    )
    meridian = model.Meridian(input_data=data, model_spec=model_spec)
    self.assertIsNotNone(meridian.non_media_transformer)
    # pytype: disable=attribute-error
    self.assertIsNotNone(
        meridian.non_media_transformer._population_scaling_factors,
        msg=(
            "`_population_scaling_factors` not set for the non-media"
            " transformer."
        ),
    )
    test_utils.assert_allequal(
        meridian.non_media_transformer._population_scaling_factors.shape,
        [
            len(data.geo),
            len(data.non_media_channel),
        ],
        err_msg=(
            "Shape of"
            " `non_media_transformer._population_scaling_factors` does"
            " not match (`n_geos`, `n_non_media_channels`)."
        ),
    )
    # pytype: enable=attribute-error

  def test_scaled_data_inverse_is_identity(self):
    data = self.input_data_non_media_and_organic
    meridian = model.Meridian(input_data=data)

    # With the default tolerance of eps * 10 the test fails due to rounding
    # errors.
    atol = np.finfo(np.float32).eps * 100
    test_utils.assert_allclose(
        meridian.controls_transformer.inverse(meridian.controls_scaled),  # pytype: disable=attribute-error
        data.controls,
        atol=atol,
    )
    self.assertIsNotNone(meridian.non_media_transformer)
    # pytype: disable=attribute-error
    test_utils.assert_allclose(
        meridian.non_media_transformer.inverse(
            meridian.non_media_treatments_normalized
        ),
        data.non_media_treatments,
        atol=atol,
    )
    # pytype: enable=attribute-error
    test_utils.assert_allclose(
        meridian.kpi_transformer.inverse(meridian.kpi_scaled),
        data.kpi,
        atol=atol,
    )

  # TODO: Move this integration test to a separate module.
  def test_get_joint_dist_constants(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=backend.tfd.Deterministic(0),
            tau_g_excl_baseline=backend.tfd.Deterministic(0),
            beta_m=backend.tfd.Deterministic(0),
            beta_rf=backend.tfd.Deterministic(0),
            beta_om=backend.tfd.Deterministic(0),
            beta_orf=backend.tfd.Deterministic(0),
            contribution_m=backend.tfd.Deterministic(0),
            contribution_rf=backend.tfd.Deterministic(0),
            contribution_om=backend.tfd.Deterministic(0),
            contribution_orf=backend.tfd.Deterministic(0),
            contribution_n=backend.tfd.Deterministic(0),
            eta_m=backend.tfd.Deterministic(0),
            eta_rf=backend.tfd.Deterministic(0),
            eta_om=backend.tfd.Deterministic(0),
            eta_orf=backend.tfd.Deterministic(0),
            gamma_c=backend.tfd.Deterministic(0),
            xi_c=backend.tfd.Deterministic(0),
            gamma_n=backend.tfd.Deterministic(0),
            xi_n=backend.tfd.Deterministic(0),
            alpha_m=backend.tfd.Deterministic(0),
            alpha_rf=backend.tfd.Deterministic(0),
            alpha_om=backend.tfd.Deterministic(0),
            alpha_orf=backend.tfd.Deterministic(0),
            sigma=backend.tfd.Deterministic(0),
            roi_m=backend.tfd.Deterministic(0),
            roi_rf=backend.tfd.Deterministic(0),
        ),
        media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_non_media,
        model_spec=model_spec,
    )
    sample = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(
            self._N_DRAWS, seed=self.get_next_rng_seed_or_key()
        )
    )
    test_utils.assert_allequal(
        sample.y,
        backend.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

  # TODO: Move this integration test to a separate module.
  @parameterized.named_parameters(
      dict(
          testcase_name="default_normal_failing",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
      dict(
          testcase_name="mixed_log_normal_ok",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
      ),
      dict(
          testcase_name="mixed_normal_failing",
          media_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          rf_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          organic_media_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          organic_rf_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
          non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
          media_effects_dist=constants.MEDIA_EFFECTS_NORMAL,
      ),
  )
  def test_get_joint_dist_with_log_prob_non_media(
      self,
      media_prior_type: str,
      rf_prior_type: str,
      organic_media_prior_type: str,
      organic_rf_prior_type: str,
      non_media_treatments_prior_type: str,
      media_effects_dist: str,
  ):
    model_spec = spec.ModelSpec(
        media_prior_type=media_prior_type,
        rf_prior_type=rf_prior_type,
        organic_media_prior_type=organic_media_prior_type,
        organic_rf_prior_type=organic_rf_prior_type,
        non_media_treatments_prior_type=non_media_treatments_prior_type,
        media_effects_dist=media_effects_dist,
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_non_media_and_organic,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist_unpinned().sample(
            1, seed=self.get_next_rng_seed_or_key()
        )
    )
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) outcome
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) outcome data is "pinned" as "y".
    log_prob_parts_structtuple = (
        meridian.posterior_sampler_callable._get_joint_dist().log_prob_parts(
            par
        )
    )
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.BETA_GRF,
        constants.BETA_GOM,
        constants.BETA_GORF,
        constants.GAMMA_GC,
        constants.GAMMA_GN,
        constants.MU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.ETA_RF,
        constants.ETA_OM,
        constants.ETA_ORF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.XI_N,
        constants.ALPHA_M,
        constants.ALPHA_RF,
        constants.ALPHA_OM,
        constants.ALPHA_ORF,
        constants.EC_M,
        constants.EC_RF,
        constants.EC_OM,
        constants.EC_ORF,
        constants.SLOPE_M,
        constants.SLOPE_RF,
        constants.SLOPE_OM,
        constants.SLOPE_ORF,
        constants.SIGMA,
    ]
    if media_prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.ROI_M)
    elif media_prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.MROI_M)
    elif media_prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      derived_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.CONTRIBUTION_M)
    elif media_prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      prior_distribution_params.append(constants.BETA_M)
    else:
      raise ValueError(f"Unsupported media prior type: {media_prior_type}")

    if rf_prior_type == constants.TREATMENT_PRIOR_TYPE_ROI:
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.ROI_RF)
    elif rf_prior_type == constants.TREATMENT_PRIOR_TYPE_MROI:
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.MROI_RF)
    elif rf_prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      derived_params.append(constants.BETA_RF)
      prior_distribution_params.append(constants.CONTRIBUTION_RF)
    elif rf_prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      prior_distribution_params.append(constants.BETA_RF)
    else:
      raise ValueError(f"Unsupported RF prior type: {rf_prior_type}")

    if organic_media_prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      derived_params.append(constants.BETA_OM)
      prior_distribution_params.append(constants.CONTRIBUTION_OM)
    elif organic_media_prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      prior_distribution_params.append(constants.BETA_OM)
    else:
      raise ValueError(
          f"Unsupported organic media prior type: {organic_media_prior_type}"
      )

    if organic_rf_prior_type == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      derived_params.append(constants.BETA_ORF)
      prior_distribution_params.append(constants.CONTRIBUTION_ORF)
    elif organic_rf_prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
      prior_distribution_params.append(constants.BETA_ORF)
    else:
      raise ValueError(
          f"Unsupported organic RF prior type: {organic_rf_prior_type}"
      )

    if (
        non_media_treatments_prior_type
        == constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
    ):
      derived_params.append(constants.GAMMA_N)
      prior_distribution_params.append(constants.CONTRIBUTION_N)
    elif (
        non_media_treatments_prior_type
        == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    ):
      prior_distribution_params.append(constants.GAMMA_N)
    else:
      raise ValueError(
          "Unsupported non-media treatments prior type:"
          f" {non_media_treatments_prior_type}"
      )

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      test_utils.assert_allequal(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = backend.reduce_sum(
          getattr(meridian.prior_broadcast, parname).log_prob(par[parname])
      )
      test_utils.assert_allclose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.BETA_GRF_DEV,
        constants.BETA_GOM_DEV,
        constants.BETA_GORF_DEV,
        constants.GAMMA_GC_DEV,
        constants.GAMMA_GN_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = backend.reduce_sum(
          backend.tfd.Normal(0, 1).log_prob(par[parname])
      )
      test_utils.assert_allclose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
        decay_functions=meridian.adstock_decay_spec.media,
    )[0, :, :, :]
    transformed_reach = meridian.adstock_hill_rf(
        reach=meridian.rf_tensors.reach_scaled,
        frequency=meridian.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
        decay_functions=meridian.adstock_decay_spec.rf,
    )[0, :, :, :]
    transformed_organic_media = meridian.adstock_hill_media(
        media=meridian.organic_media_tensors.organic_media_scaled,
        alpha=par[constants.ALPHA_OM],
        ec=par[constants.EC_OM],
        slope=par[constants.SLOPE_OM],
        decay_functions=meridian.adstock_decay_spec.organic_media,
    )[0, :, :, :]
    transformed_organic_reach = meridian.adstock_hill_rf(
        reach=meridian.organic_rf_tensors.organic_reach_scaled,
        frequency=meridian.organic_rf_tensors.organic_frequency,
        alpha=par[constants.ALPHA_ORF],
        ec=par[constants.EC_ORF],
        slope=par[constants.SLOPE_ORF],
        decay_functions=meridian.adstock_decay_spec.organic_rf,
    )[0, :, :, :]
    combined_transformed_media = backend.concatenate(
        [
            transformed_media,
            transformed_reach,
            transformed_organic_media,
            transformed_organic_reach,
        ],
        axis=-1,
    )

    combined_beta = backend.concatenate(
        [
            par[constants.BETA_GM][0, :, :],
            par[constants.BETA_GRF][0, :, :],
            par[constants.BETA_GOM][0, :, :],
            par[constants.BETA_GORF][0, :, :],
        ],
        axis=-1,
    )
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.MU_T][0, None, :]
        + backend.einsum(
            "gtm,gm->gt", combined_transformed_media, combined_beta
        )
        + backend.einsum(
            "gtc,gc->gt",
            meridian.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
        + backend.einsum(
            "gtn,gn->gt",
            meridian.non_media_treatments_normalized,
            par[constants.GAMMA_GN][0, :, :],
        )
    )
    y_means_logprob = backend.reduce_sum(
        backend.tfd.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.kpi_scaled
        )
    )
    test_utils.assert_allclose(
        y_means_logprob, log_prob_parts["pinned"]["y"][0]
    )

    tau_g_logprob = backend.reduce_sum(
        getattr(
            meridian.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    test_utils.assert_allclose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    test_utils.assert_allclose(
        posterior_unnormalized_logprob,
        meridian.posterior_sampler_callable._get_joint_dist().log_prob(par)[0],
        rtol=1e-3,
    )

  # TODO: Move this integration test to a separate module.
  def test_inference_data_non_paid_correct_dims(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec()
    mmm = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    n_draws = 7
    prior_draws = mmm.prior_sampler_callable._sample_prior(n_draws, seed=1)
    # Create Arviz InferenceData for prior draws.
    prior_coords = mmm.create_inference_data_coords(1, n_draws)
    prior_dims = mmm.create_inference_data_dims()

    for param, tensor in prior_draws.items():
      self.assertIn(param, prior_dims)
      dims = prior_dims[param]
      self.assertEqual(
          len(tensor.shape),
          len(dims),
          f"Parameter {param} has expected dimension {dims} but prior-drawn"
          f" tensor for this parameter has shape {tensor.shape}",
      )
      for dim, shape_dim in zip(dims, tensor.shape):
        self.assertIn(dim, prior_coords)
        self.assertLen(prior_coords[dim], shape_dim)

  # TODO: Move this integration test to a separate module.
  def test_inference_data_with_unique_sigma_geo_correct_dims(self):
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(unique_sigma_for_each_geo=True)
    mmm = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    n_draws = 7
    prior_draws = mmm.prior_sampler_callable._sample_prior(n_draws, seed=1)
    # Create Arviz InferenceData for prior draws.
    prior_coords = mmm.create_inference_data_coords(1, n_draws)
    prior_dims = mmm.create_inference_data_dims()

    for param, tensor in prior_draws.items():
      self.assertIn(param, prior_dims)
      dims = prior_dims[param]
      self.assertEqual(
          len(tensor.shape),
          len(dims),
          f"Parameter {param} has expected dimension {dims} but prior-drawn"
          f" tensor for this parameter has shape {tensor.shape}",
      )
      for dim, shape_dim in zip(dims, tensor.shape):
        self.assertIn(dim, prior_coords)
        self.assertLen(prior_coords[dim], shape_dim)

  # TODO: Move this integration test to a separate module.
  def test_validate_injected_inference_data_correct_shapes(self):
    """Checks validation passes with correct shapes."""
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    n_chains = 1
    n_draws = 10
    prior_samples = meridian.prior_sampler_callable._sample_prior(
        n_draws, seed=1
    )
    prior_coords = meridian.create_inference_data_coords(n_chains, n_draws)
    prior_dims = meridian.create_inference_data_dims()
    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    # This should not raise an error
    meridian_with_inference_data = model.Meridian(
        input_data=data,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  # TODO: Move this integration test to a separate module.
  @parameterized.named_parameters(
      dict(
          testcase_name="non_media_channels",
          coord=constants.NON_MEDIA_CHANNEL,
          mismatched_priors={
              constants.GAMMA_GN: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
              constants.GAMMA_N: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
              constants.XI_N: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
              constants.CONTRIBUTION_N: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_NON_MEDIA_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_NON_MEDIA_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_NON_MEDIA_CHANNELS,
      ),
      dict(
          testcase_name="organic_rf_channels",
          coord=constants.ORGANIC_RF_CHANNEL,
          mismatched_priors={
              constants.ALPHA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.BETA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.BETA_GORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.EC_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.ETA_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.SLOPE_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
              constants.CONTRIBUTION_ORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_ORGANIC_RF_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_ORGANIC_RF_CHANNELS,
      ),
  )
  def test_validate_injected_inference_data_prior_incorrect_coordinates(
      self, coord, mismatched_priors, mismatched_coord_size, expected_coord_size
  ):
    """Checks validation fails with incorrect coordinates."""
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    prior_samples = meridian.prior_sampler_callable._sample_prior(
        self._N_DRAWS, seed=1
    )
    prior_coords = meridian.create_inference_data_coords(1, self._N_DRAWS)
    prior_dims = meridian.create_inference_data_dims()

    prior_samples = dict(prior_samples)
    for param in mismatched_priors:
      prior_samples[param] = backend.zeros(mismatched_priors[param])
    prior_coords = dict(prior_coords)
    prior_coords[coord] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Injected inference data prior has incorrect coordinate"
        f" '{coord}': expected"
        f" {expected_coord_size}, got"
        f" {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=self.input_data_non_media_and_organic,
          model_spec=model_spec,
          inference_data=inference_data,
      )

  # TODO: Move this integration test to a separate module.
  @parameterized.named_parameters(
      dict(
          testcase_name="sigma_dims_unique_sigma",
          coord=constants.GEO,
          mismatched_priors={
              constants.BETA_GOM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              ),
              constants.BETA_GORF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_RF_CHANNELS,
              ),
              constants.GAMMA_GN: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_NON_MEDIA_CHANNELS,
              ),
              constants.GAMMA_GC: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G_EXCL_BASELINE: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
              ),
              constants.BETA_GM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_MEDIA_CHANNELS,
              ),
              constants.BETA_GRF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_RF_CHANNELS,
              ),
              constants.BETA_GOM_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_MEDIA_CHANNELS,
              ),
              constants.BETA_GORF_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_ORGANIC_RF_CHANNELS,
              ),
              constants.GAMMA_GN_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_NON_MEDIA_CHANNELS,
              ),
              constants.GAMMA_GC_DEV: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.SIGMA: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_GEOS + 1,
          expected_coord_size=input_data_samples._N_GEOS,
      ),
  )
  def test_validate_injected_inference_data_prior_incorrect_sigma_coordinates(
      self,
      coord,
      mismatched_priors,
      mismatched_coord_size,
      expected_coord_size,
  ):
    """Checks validation fails with incorrect coordinates for sigma."""
    data = self.input_data_non_media_and_organic
    model_spec = spec.ModelSpec(unique_sigma_for_each_geo=True)
    meridian = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    prior_samples = meridian.prior_sampler_callable._sample_prior(
        self._N_DRAWS, seed=1
    )
    prior_coords = meridian.create_inference_data_coords(1, self._N_DRAWS)
    prior_dims = meridian.create_inference_data_dims()

    prior_samples = dict(prior_samples)
    for param in mismatched_priors:
      prior_samples[param] = backend.zeros(mismatched_priors[param])
    prior_coords = dict(prior_coords)
    prior_coords[coord] = np.arange(mismatched_coord_size)
    prior_coords[constants.GEO] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Injected inference data prior has incorrect coordinate"
        f" '{coord}': expected"
        f" {expected_coord_size}, got"
        f" {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=data,
          model_spec=model_spec,
          inference_data=inference_data,
      )

  def test_compute_non_media_treatments_baseline_wrong_baseline_values_shape_raises_exception(
      self,
  ):
    data = self.input_data_non_media_and_organic
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The number of non-media channels (2) does not match the number of"
        " baseline values (3).",
    ):
      mmm = model.Meridian(
          input_data=data,
          model_spec=spec.ModelSpec(
              non_media_baseline_values=["min", "max", "min"]
          ),
      )
      _ = mmm.compute_non_media_treatments_baseline()

  def test_compute_non_media_treatments_baseline_fails_with_wrong_baseline_type(
      self,
  ):
    data = self.input_data_non_media_and_organic
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid non_media_baseline_values value: 'wrong'. Only"
        " float numbers and strings 'min' and 'max' are supported.",
    ):
      mmm = model.Meridian(
          input_data=data,
          model_spec=spec.ModelSpec(
              non_media_baseline_values=[
                  "max",
                  "wrong",
              ]
          ),
      )
      _ = mmm.compute_non_media_treatments_baseline()

  def test_compute_non_media_treatments_baseline_default(self):
    """Tests default baseline calculation (all 'min')."""
    data = self.input_data_non_media_and_organic
    meridian = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(non_media_baseline_values=None),
    )
    non_media_treatments = meridian.non_media_treatments
    expected_baseline = backend.reduce_min(non_media_treatments, axis=[0, 1])
    actual_baseline = meridian.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_strings(self):
    """Tests baseline calculation with 'min' and 'max' strings."""
    data = self.input_data_non_media_and_organic
    meridian = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(non_media_baseline_values=["min", "max"]),
    )
    non_media_treatments = meridian.non_media_treatments
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0], axis=[0, 1]
    )
    expected_baseline_max = backend.reduce_max(
        non_media_treatments[..., 1], axis=[0, 1]
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_max], axis=-1
    )
    actual_baseline = meridian.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)

  def test_compute_non_media_treatments_baseline_floats(self):
    """Tests baseline calculation with float values."""
    data = self.input_data_non_media_and_organic
    baseline_values = [10.5, -2.3]
    meridian = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(non_media_baseline_values=baseline_values),
    )
    expected_baseline = backend.to_tensor(
        baseline_values, dtype=backend.float32
    )
    actual_baseline = meridian.compute_non_media_treatments_baseline()
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
    meridian = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    non_media_treatments = meridian.non_media_treatments
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
        expected_baseline, meridian.compute_non_media_treatments_baseline()
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
    meridian = model.Meridian(
        input_data=data,
        model_spec=model_spec,
    )
    non_media_treatments = meridian.non_media_treatments
    _, baseline_value_float = baseline_values
    expected_baseline_min = backend.reduce_min(
        non_media_treatments[..., 0] / meridian.population[:, np.newaxis],
        axis=[0, 1],
    )
    expected_baseline_float = backend.to_tensor(
        baseline_value_float, dtype=backend.float32
    )
    expected_baseline = backend.stack(
        [expected_baseline_min, expected_baseline_float], axis=-1
    )
    actual_baseline = meridian.compute_non_media_treatments_baseline()
    test_utils.assert_allclose(expected_baseline, actual_baseline)


if __name__ == "__main__":
  absltest.main()
