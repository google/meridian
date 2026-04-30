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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.backend import test_utils
from meridian.model import context
from meridian.model import model_test_data
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
from meridian.model.eda import sampling_eda_engine
import numpy as np


class SamplingEdaEngineTest(
    test_utils.MeridianTestCase, model_test_data.WithInputDataSamples
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.mock_model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )
    self.mock_model_context.input_data = self.input_data_with_media_only
    self.eda_spec = eda_spec_module.EDASpec()
    self.mock_analyzer = mock.create_autospec(
        analyzer_module.Analyzer,
        instance=True,
        spec_set=True,
        model_context=self.mock_model_context,
        inference_data=az.from_dict(prior={'x': np.ones((1, 100))}),
    )
    self.mock_analyzer.negative_baseline_probability.return_value = 0.1
    # Shape: (n_chains, n_draws, n_media_channels)
    self.default_shape = (self._N_CHAINS, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    self.mock_analyzer.incremental_outcome.return_value = (
        self._create_incremental_outcome(self.default_shape)
    )

  def _create_incremental_outcome(
      self, shape: tuple[int, int, int]
  ) -> backend.Tensor:
    return backend.to_tensor(
        np.arange(np.prod(shape)).reshape(shape).astype(float)
    )

  def test_initialization_success(self):
    engine = sampling_eda_engine.SamplingEDAEngine(
        analyzer=self.mock_analyzer, spec=self.eda_spec
    )
    self.assertIsInstance(engine, sampling_eda_engine.SamplingEDAEngine)

  def test_initialization_default_spec(self):
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertIsInstance(engine.spec, eda_spec_module.EDASpec)
    self.assertEqual(engine.spec, eda_spec_module.EDASpec())

  def test_initialization_no_prior_raises_error(self):
    mock_analyzer_no_prior = mock.create_autospec(
        analyzer_module.Analyzer,
        instance=True,
        spec_set=True,
        model_context=self.mock_model_context,
        inference_data=az.from_dict(posterior={'x': np.ones((1, 100))}),
    )

    with self.assertRaisesRegex(
        ValueError, "Analyzer instance must have 'prior' in its inference_data."
    ):
      sampling_eda_engine.SamplingEDAEngine(
          analyzer=mock_analyzer_no_prior, spec=self.eda_spec
      )

  def test_check_prior_probability_valid_inputs(self):
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    result = engine.check_prior_probability()

    input_data = self.input_data_with_media_only
    revenue_per_kpi = input_data.revenue_per_kpi
    self.assertIsNotNone(revenue_per_kpi)
    expected_total_outcome = np.sum(
        input_data.kpi.values * revenue_per_kpi.values
    )
    expected_mean = np.mean(
        self.mock_analyzer.incremental_outcome.return_value
        / expected_total_outcome,
        axis=(0, 1),
    )
    media_channel = self.input_data_with_media_only.media_channel
    self.assertIsNotNone(media_channel)

    with self.subTest('check_type'):
      self.assertEqual(
          result.check_type, eda_outcome.EDACheckType.PRIOR_PROBABILITY
      )
    with self.subTest('artifact_level'):
      self.assertEqual(
          result.analysis_artifacts[0].level, eda_outcome.AnalysisLevel.OVERALL
      )
    with self.subTest('prior_negative_baseline_prob'):
      self.assertEqual(
          result.analysis_artifacts[0].prior_negative_baseline_prob, 0.1
      )
    with self.subTest('mean_prior_contribution_da'):
      self.assertEqual(
          result.analysis_artifacts[0].mean_prior_contribution_da.shape,
          (self._N_MEDIA_CHANNELS,),
      )
      self.assertSequenceEqual(
          result.analysis_artifacts[0].mean_prior_contribution_da.dims,
          (constants.CHANNEL,),
      )
      self.assertSequenceEqual(
          result.analysis_artifacts[0]
          .mean_prior_contribution_da.coords[constants.CHANNEL]
          .values.tolist(),
          media_channel.values.tolist(),
      )
      np.testing.assert_array_almost_equal(
          result.analysis_artifacts[0].mean_prior_contribution_da.values,
          expected_mean,
      )
    with self.subTest('findings'):
      self.assertLen(result.findings, 1)
      finding = result.findings[0]
      self.assertEqual(finding.severity, eda_outcome.EDASeverity.INFO)
      self.assertEqual(
          finding.explanation, eda_constants.PRIOR_PROBABILITY_INFO
      )
      self.assertEqual(finding.finding_cause, eda_outcome.FindingCause.NONE)

  def test_check_prior_probability_no_revenue_per_kpi(self):
    self.mock_model_context.input_data = (
        self.input_data_non_revenue_no_revenue_per_kpi
    )

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    result = engine.check_prior_probability()

    input_data = self.input_data_non_revenue_no_revenue_per_kpi
    expected_total_outcome = np.sum(input_data.kpi.values)
    expected_mean = np.mean(
        self.mock_analyzer.incremental_outcome.return_value
        / expected_total_outcome,
        axis=(0, 1),
    )

    np.testing.assert_array_almost_equal(
        result.analysis_artifacts[0].mean_prior_contribution_da.values,
        expected_mean,
    )

  def test_check_prior_probability_zero_total_outcome(self):
    input_data = self.input_data_with_media_only
    kpi = input_data.kpi.copy(deep=True)
    kpi.values = np.zeros_like(kpi.values)
    self.mock_model_context.input_data = dataclasses.replace(
        input_data, kpi=kpi
    )

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    result = engine.check_prior_probability()

    with self.subTest('mean_prior_contribution_da'):
      self.assertTrue(
          np.all(
              np.isinf(
                  result.analysis_artifacts[0].mean_prior_contribution_da.values
              )
          )
      )
      self.assertEqual(
          result.analysis_artifacts[0].mean_prior_contribution_da.shape,
          (self._N_MEDIA_CHANNELS,),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='n_chains=1',
          shape=(1, 10, input_data_samples._N_MEDIA_CHANNELS),
      ),
      dict(
          testcase_name='n_draws=5',
          shape=(2, 5, input_data_samples._N_MEDIA_CHANNELS),
      ),
  )
  def test_check_prior_probability_different_shapes(self, shape):
    incremental_outcome = self._create_incremental_outcome(shape)
    self.mock_analyzer.incremental_outcome.return_value = incremental_outcome

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    result = engine.check_prior_probability()

    input_data = self.input_data_with_media_only
    revenue_per_kpi = input_data.revenue_per_kpi
    self.assertIsNotNone(revenue_per_kpi)
    expected_total_outcome = np.sum(
        input_data.kpi.values * revenue_per_kpi.values
    )
    expected_mean = np.mean(
        incremental_outcome / expected_total_outcome, axis=(0, 1)
    )
    np.testing.assert_array_almost_equal(
        result.analysis_artifacts[0].mean_prior_contribution_da.values,
        expected_mean,
    )

  def test_check_prior_probability_negative_baseline_probability_uses_prior(
      self,
  ):
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    engine.check_prior_probability()

    with self.subTest('negative_baseline_probability_call'):
      self.mock_analyzer.negative_baseline_probability.assert_called_once_with(
          use_posterior=False
      )
    with self.subTest('incremental_outcome_call'):
      self.mock_analyzer.incremental_outcome.assert_called_once_with(
          use_posterior=False
      )


if __name__ == '__main__':
  absltest.main()
