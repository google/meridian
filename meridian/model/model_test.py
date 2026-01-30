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
from meridian.backend import config as backend_config
from meridian.backend import test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import equations
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
    self.assertIsInstance(meridian.model_equations, equations.ModelEquations)

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
      (
          "tensorflow",
          backend_config.ComputationBackend.TENSORFLOW,
          "TENSORFLOW",
      ),
      ("jax", backend_config.ComputationBackend.JAX, "JAX"),
  )
  def test_computation_backend_property(self, backend_enum, expected_name):
    with mock.patch.object(
        backend, "computation_backend", return_value=backend_enum
    ):
      meridian = model.Meridian(input_data=self.input_data_with_media_only)
      self.assertEqual(meridian.computation_backend, expected_name)

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

  def test_run_model_fitting_guardrail_error_message(self):
    finding_corr_1 = eda_outcome.EDAFinding(
        severity=eda_outcome.EDASeverity.ERROR,
        explanation="Error explanation for PAIRWISE_CORR 1.",
        finding_cause=eda_outcome.FindingCause.MULTICOLLINEARITY,
    )
    finding_corr_2 = eda_outcome.EDAFinding(
        severity=eda_outcome.EDASeverity.ERROR,
        explanation="Error explanation for PAIRWISE_CORR 2.",
        finding_cause=eda_outcome.FindingCause.MULTICOLLINEARITY,
    )
    finding_vif_1 = eda_outcome.EDAFinding(
        severity=eda_outcome.EDASeverity.ERROR,
        explanation="Error explanation for MULTICOLLINEARITY 1.",
        finding_cause=eda_outcome.FindingCause.MULTICOLLINEARITY,
    )

    outcome_corr = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        findings=[finding_corr_1, finding_corr_2],
        analysis_artifacts=[],
    )
    outcome_vif = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.MULTICOLLINEARITY,
        findings=[finding_vif_1],
        analysis_artifacts=[],
    )
    outcome_kpi = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.KPI_INVARIABILITY,
        findings=[],
        analysis_artifacts=[],
    )

    critical_outcomes = eda_outcome.CriticalCheckEDAOutcomes(
        kpi_invariability=outcome_kpi,
        multicollinearity=outcome_vif,
        pairwise_correlation=outcome_corr,
    )

    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "eda_outcomes",
            new_callable=mock.PropertyMock,
            return_value=critical_outcomes,
        )
    )
    meridian = model.Meridian(input_data=self.input_data_with_media_only)

    # The error message order is deterministic based on
    # `eda_outcome.CriticalCheckEDAOutcomes` dataclass field order:
    # 1. kpi (empty)
    # 2. multicollinearity
    # 3. pairwise_correlation
    expected_error_message = (
        "Model has critical EDA issues. Please fix before running"
        " `sample_posterior`.\n\n"
        "Check type: MULTICOLLINEARITY\n"
        "- Error explanation for MULTICOLLINEARITY 1.\n"
        "Check type: PAIRWISE_CORRELATION\n"
        "- Error explanation for PAIRWISE_CORR 1.\n"
        "- Error explanation for PAIRWISE_CORR 2.\n"
        "For further details, please refer to `Meridian.eda_outcomes`."
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

  def setUp(self):
    super().setUp()
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    self.file_path = os.path.join(self.create_tempdir().full_path, "joblib")

  def test_save_and_load_works(self):
    mmm = model.Meridian(input_data=self.input_data_with_media_and_rf)
    model.save_mmm(mmm, str(self.file_path))
    self.assertTrue(os.path.exists(self.file_path))
    new_mmm = model.load_mmm(self.file_path)
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

  def test_save_and_load_warning(self):
    mmm = model.Meridian(input_data=self.input_data_with_media_and_rf)
    with self.assertWarns(DeprecationWarning):
      model.save_mmm(mmm, str(self.file_path))
    with self.assertWarns(DeprecationWarning):
      model.load_mmm(self.file_path)

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
    prior_samples = meridian.prior_sampler_callable(n_draws, seed=1)
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
    prior_samples = meridian.prior_sampler_callable(self._N_DRAWS, seed=1)
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
    prior_samples = meridian.prior_sampler_callable(self._N_DRAWS, seed=1)
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


class ModelEquationsDelegateTest(test_utils.MeridianTestCase):

  def setUp(self):
    super().setUp()
    self.mock_equations = mock.create_autospec(
        equations.ModelEquations, instance=True, spec_set=True
    )
    # Patch ModelEquations to return our mock
    self.enter_context(
        mock.patch.object(
            equations, "ModelEquations", return_value=self.mock_equations
        )
    )
    # We need a minimal Meridian object. We can mock input_data.
    self.mock_input_data = mock.create_autospec(
        model.data.InputData, instance=True, spec_set=True
    )
    # We need to mock ModelContext as well since Meridian init creates it
    self.mock_context = mock.create_autospec(
        model.context.ModelContext, instance=True, spec_set=True
    )
    self.mock_context.is_national = False
    self.enter_context(
        mock.patch.object(
            model.context, "ModelContext", return_value=self.mock_context
        )
    )

    # Patch EDAEngine to avoid initialization errors
    self.mock_eda_engine = mock.create_autospec(
        eda_engine.EDAEngine, instance=True, spec_set=True
    )
    self.mock_eda_engine.kpi_has_variability = True
    self.enter_context(
        mock.patch.object(
            eda_engine, "EDAEngine", return_value=self.mock_eda_engine
        )
    )

    self.mmm = model.Meridian(input_data=self.mock_input_data)
    # Verify equations was set to our mock (sanity check of the setup)
    self.assertIs(self.mmm.model_equations, self.mock_equations)

  def test_adstock_hill_media_delegates(self):
    media_val = mock.Mock()
    alpha = mock.Mock()
    ec = mock.Mock()
    slope = mock.Mock()
    decay_functions = "geometric"
    n_times_output = 10

    self.mmm.adstock_hill_media(
        media=media_val,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

    self.mock_equations.adstock_hill_media.assert_called_once_with(
        media=media_val,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

  def test_adstock_hill_rf_delegates(self):
    reach = mock.Mock()
    frequency = mock.Mock()
    alpha = mock.Mock()
    ec = mock.Mock()
    slope = mock.Mock()
    decay_functions = "geometric"
    n_times_output = 10

    self.mmm.adstock_hill_rf(
        reach=reach,
        frequency=frequency,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

    self.mock_equations.adstock_hill_rf.assert_called_once_with(
        reach=reach,
        frequency=frequency,
        alpha=alpha,
        ec=ec,
        slope=slope,
        decay_functions=decay_functions,
        n_times_output=n_times_output,
    )

  def test_calculate_beta_x_delegates(self):
    is_non_media = True
    incremental_outcome_x = mock.Mock()
    linear_predictor_diff = mock.Mock()
    eta_x = mock.Mock()
    beta_gx_dev = mock.Mock()

    self.mmm.calculate_beta_x(
        is_non_media=is_non_media,
        incremental_outcome_x=incremental_outcome_x,
        linear_predictor_counterfactual_difference=linear_predictor_diff,
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
    )

    self.mock_equations.calculate_beta_x.assert_called_once_with(
        is_non_media=is_non_media,
        incremental_outcome_x=incremental_outcome_x,
        linear_predictor_counterfactual_difference=linear_predictor_diff,
        eta_x=eta_x,
        beta_gx_dev=beta_gx_dev,
    )

  def test_linear_predictor_counterfactual_difference_media_delegates(self):
    media_transformed = mock.Mock()
    alpha_m = mock.Mock()
    ec_m = mock.Mock()
    slope_m = mock.Mock()

    self.mmm.linear_predictor_counterfactual_difference_media(
        media_transformed=media_transformed,
        alpha_m=alpha_m,
        ec_m=ec_m,
        slope_m=slope_m,
    )

    self.mock_equations.linear_predictor_counterfactual_difference_media.assert_called_once_with(
        media_transformed=media_transformed,
        alpha_m=alpha_m,
        ec_m=ec_m,
        slope_m=slope_m,
    )

  def test_linear_predictor_counterfactual_difference_rf_delegates(self):
    rf_transformed = mock.Mock()
    alpha_rf = mock.Mock()
    ec_rf = mock.Mock()
    slope_rf = mock.Mock()

    self.mmm.linear_predictor_counterfactual_difference_rf(
        rf_transformed=rf_transformed,
        alpha_rf=alpha_rf,
        ec_rf=ec_rf,
        slope_rf=slope_rf,
    )

    self.mock_equations.linear_predictor_counterfactual_difference_rf.assert_called_once_with(
        rf_transformed=rf_transformed,
        alpha_rf=alpha_rf,
        ec_rf=ec_rf,
        slope_rf=slope_rf,
    )


if __name__ == "__main__":
  absltest.main()
