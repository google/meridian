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
import arviz as az
from meridian.__version__ import __version__
from meridian import constants
from meridian.analysis import analyzer
from meridian.data import test_utils
from meridian.mlflow import autolog
import mlflow
from meridian.model import model
from meridian.model import posterior_sampler
from meridian.model import prior_sampler
from meridian.model import spec


INPUT_DATA = test_utils.sample_input_data_revenue(n_media_channels=1)


def _create_prior_distribution_log_entry(
    field_name: str, dist: str
) -> tuple[str, str]:
  """Returns a loggable representation of a TensorFlow Distribution."""
  key = f"prior.{field_name}"
  value = (
      f'tfp.distributions.{dist}("{field_name}", batch_shape=[],'
      " event_shape=[], dtype=float32)"
  )
  return key, value


DEFAULT_EXPECTED_CALLS = [
    ("meridian_version", __version__),
    ("arviz_version", az.__version__),
    ("spec.media_effects_dist", constants.MEDIA_EFFECTS_LOG_NORMAL),
    ("spec.hill_before_adstock", False),
    ("spec.max_lag", 8),
    ("spec.unique_sigma_for_each_geo", False),
    ("spec.media_prior_type", None),
    ("spec.rf_prior_type", None),
    ("spec.paid_media_prior_type", None),
    ("spec.roi_calibration_period", None),
    ("spec.rf_roi_calibration_period", None),
    (
        "spec.organic_media_prior_type",
        constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
    ),
    ("spec.organic_rf_prior_type", constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION),
    (
        "spec.non_media_treatments_prior_type",
        constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
    ),
    ("spec.non_media_baseline_values", None),
    ("spec.knots", None),
    ("spec.baseline_geo", None),
    ("spec.holdout_id", None),
    ("spec.control_population_scaling_id", None),
    ("spec.non_media_population_scaling_id", None),
    (_create_prior_distribution_log_entry("knot_values", "Normal")),
    (_create_prior_distribution_log_entry("tau_g_excl_baseline", "Normal")),
    (_create_prior_distribution_log_entry("beta_m", "HalfNormal")),
    (_create_prior_distribution_log_entry("beta_rf", "HalfNormal")),
    (_create_prior_distribution_log_entry("beta_om", "HalfNormal")),
    (_create_prior_distribution_log_entry("beta_orf", "HalfNormal")),
    (_create_prior_distribution_log_entry("eta_m", "HalfNormal")),
    (_create_prior_distribution_log_entry("eta_rf", "HalfNormal")),
    (_create_prior_distribution_log_entry("eta_om", "HalfNormal")),
    (_create_prior_distribution_log_entry("eta_orf", "HalfNormal")),
    (_create_prior_distribution_log_entry("gamma_c", "Normal")),
    (_create_prior_distribution_log_entry("gamma_n", "Normal")),
    (_create_prior_distribution_log_entry("xi_c", "HalfNormal")),
    (_create_prior_distribution_log_entry("xi_n", "HalfNormal")),
    (_create_prior_distribution_log_entry("alpha_m", "Uniform")),
    (_create_prior_distribution_log_entry("alpha_rf", "Uniform")),
    (_create_prior_distribution_log_entry("alpha_om", "Uniform")),
    (_create_prior_distribution_log_entry("alpha_orf", "Uniform")),
    (_create_prior_distribution_log_entry("ec_m", "TruncatedNormal")),
    (_create_prior_distribution_log_entry("ec_rf", "TransformedDistribution")),
    (_create_prior_distribution_log_entry("ec_om", "TruncatedNormal")),
    (_create_prior_distribution_log_entry("ec_orf", "TransformedDistribution")),
    (_create_prior_distribution_log_entry("slope_m", "Deterministic")),
    (_create_prior_distribution_log_entry("slope_rf", "LogNormal")),
    (_create_prior_distribution_log_entry("slope_om", "Deterministic")),
    (_create_prior_distribution_log_entry("slope_orf", "LogNormal")),
    (_create_prior_distribution_log_entry("sigma", "HalfNormal")),
    (_create_prior_distribution_log_entry("roi_m", "LogNormal")),
    (_create_prior_distribution_log_entry("roi_rf", "LogNormal")),
    (_create_prior_distribution_log_entry("mroi_m", "LogNormal")),
    (_create_prior_distribution_log_entry("mroi_rf", "LogNormal")),
    (_create_prior_distribution_log_entry("contribution_m", "Beta")),
    (_create_prior_distribution_log_entry("contribution_rf", "Beta")),
    (_create_prior_distribution_log_entry("contribution_om", "Beta")),
    (_create_prior_distribution_log_entry("contribution_orf", "Beta")),
    (_create_prior_distribution_log_entry("contribution_n", "TruncatedNormal")),
    ("sample_prior.seed", 1),
    ("sample_prior.n_draws", 100),
    ("sample_posterior.n_chains", 1),
    ("sample_posterior.n_adapt", 1),
    ("sample_posterior.n_burnin", 1),
    ("sample_posterior.n_keep", 1),
    ("sample_posterior.init_step_size", None),
    ("sample_posterior.dual_averaging_kwargs", None),
    ("sample_posterior.max_tree_depth", 10),
    ("sample_posterior.max_energy_diff", 500.0),
    ("sample_posterior.unrolled_leapfrog_steps", 1),
    ("sample_posterior.parallel_iterations", 10),
    ("sample_posterior.seed", None),
]


class AutologTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AutologTest, cls).setUpClass()
    cls.mock_log_param = cls.enter_context(
        mock.patch.object(mlflow, "log_param", autospec=True)
    )
    cls.mock_log_metric = cls.enter_context(
        mock.patch.object(mlflow, "log_metric", autospec=True)
    )
    cls.mock_analyzer_method = cls.enter_context(
        mock.patch.object(analyzer.Analyzer, "predictive_accuracy")
    )
    prior_sampler.PriorDistributionSampler.__call__ = cls.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "__call__",
            autospec=True,
            return_value=az.InferenceData(),
        )
    )
    posterior_sampler.PosteriorMCMCSampler.__call__ = cls.enter_context(
        mock.patch.object(
            posterior_sampler.PosteriorMCMCSampler,
            "__call__",
            autospec=True,
            return_value=az.InferenceData(),
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="default_model_spec_positional_args",
          model_spec=spec.ModelSpec(),
          sample_prior={"args": [100, 1], "kwargs": {}},
          sample_posterior={"args": [1, 1, 1, 1], "kwargs": {}},
          expected_log_param_calls=DEFAULT_EXPECTED_CALLS,
      ),
      dict(
          testcase_name="default_model_spec_keyword_args",
          model_spec=spec.ModelSpec(),
          sample_prior={"args": [], "kwargs": {"n_draws": 100, "seed": 1}},
          sample_posterior={"args": [1, 1, 1, 1], "kwargs": {}},
          expected_log_param_calls=DEFAULT_EXPECTED_CALLS,
      ),
      dict(
          testcase_name="default_model_spec_mixed_args",
          model_spec=spec.ModelSpec(),
          sample_prior={"args": [100], "kwargs": {"seed": 1}},
          sample_posterior={"args": [1, 1, 1], "kwargs": {"n_keep": 1}},
          expected_log_param_calls=DEFAULT_EXPECTED_CALLS,
      ),
      dict(
          testcase_name="no_model_spec_positional_args",
          model_spec=None,
          sample_prior={"args": [100, 1], "kwargs": {}},
          sample_posterior={"args": [1, 1, 1, 1], "kwargs": {}},
          expected_log_param_calls=DEFAULT_EXPECTED_CALLS,
      ),
  )
  def test_autolog(
      self,
      model_spec,
      sample_prior,
      sample_posterior,
      expected_log_param_calls,
  ):
    autolog.autolog()
    mmm = model.Meridian(input_data=INPUT_DATA, model_spec=model_spec)
    mmm.sample_prior(*sample_prior["args"], **sample_prior["kwargs"])
    mmm.sample_posterior(
        *sample_posterior["args"], **sample_posterior["kwargs"]
    )

    for key, value in expected_log_param_calls:
      self.mock_log_param.assert_any_call(key, value)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_model_spec_log_metrics_enabled",
          sample_prior={"args": [100, 1], "kwargs": {}},
          sample_posterior={"args": [1, 1, 1, 1], "kwargs": {}},
          expected_log_metric_calls=["R_Squared", "MAPE", "wMAPE"],
      )
  )
  def test_autolog_log_metrics_enabled(
      self, sample_prior, sample_posterior, expected_log_metric_calls
  ):
    autolog.autolog(log_metrics=True)
    mmm = model.Meridian(input_data=INPUT_DATA)
    mmm.sample_prior(*sample_prior["args"], **sample_prior["kwargs"])
    mmm.sample_posterior(
        *sample_posterior["args"], **sample_posterior["kwargs"]
    )
    for metric in expected_log_metric_calls:
      self.mock_log_metric.assert_any_call(metric, mock.ANY)

  def test_autolog_disabled_after_initially_enabled(self):
    autolog.autolog()
    model.Meridian(input_data=INPUT_DATA)
    self.mock_log_param.assert_called()

    autolog.autolog(disable=True)
    self.mock_log_param.reset_mock()

    model.Meridian(input_data=INPUT_DATA)
    self.mock_log_param.assert_not_called()


if __name__ == "__main__":
  absltest.main()
