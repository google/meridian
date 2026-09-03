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

from collections.abc import Sequence
import dataclasses
import math
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
from meridian.model import prior_distribution
from meridian.model.calibration import base as calibration_base
from meridian.model.calibration import roi
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
from meridian.model.eda import sampling_eda_engine
import numpy as np


def _create_experiment_result(
    point_estimate: float, standard_error: float
) -> calibration_base.ExperimentResult:
  try:
    return calibration_base.ExperimentResult(
        point_estimate=point_estimate, standard_error=standard_error
    )
  except ValueError:
    res = object.__new__(calibration_base.ExperimentResult)
    object.__setattr__(res, 'point_estimate', point_estimate)
    object.__setattr__(res, 'standard_error', standard_error)
    return res


def _create_calibrated_experiment(
    point_estimate: float = 1.5,
    standard_error: float = 0.3,
    adjusted_point_estimate: float = 1.65,
    adjusted_standard_error: float = 0.38,
    source_type: calibration_base.SourceType = calibration_base.SourceType.MERIDIAN_GEOX,
    tau_spend: float = 0.1,
    tau_duration: float = 0.2,
    tau_recency: float = 0.3,
    gamma_duration: float = 1.1,
    user_point_estimate_adjustment: float | None = None,
    user_standard_error_adjustment: float | None = None,
) -> calibration_base.CalibratedExperiment:
  raw_res = _create_experiment_result(point_estimate, standard_error)
  adj_res = _create_experiment_result(
      adjusted_point_estimate, adjusted_standard_error
  )
  return calibration_base.CalibratedExperiment(
      source_type=source_type,
      raw_experiment_result=raw_res,
      adjusted_experiment_result=adj_res,
      tau_spend=tau_spend,
      tau_duration=tau_duration,
      tau_recency=tau_recency,
      gamma_duration=gamma_duration,
      user_point_estimate_adjustment=user_point_estimate_adjustment,
      user_standard_error_adjustment=user_standard_error_adjustment,
  )


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

  def test_get_named_calibrated_priors_non_calibrated_returns_empty(self):
    mock_prior = prior_distribution.PriorDistribution()
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(engine.get_named_calibrated_priors(), {})

  def test_get_calibrated_priors_non_calibrated_returns_empty(self):
    mock_prior = prior_distribution.PriorDistribution()
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(engine.get_calibrated_priors(), [])

  def test_get_calibration_outputs_non_calibrated_returns_empty(self):
    mock_prior = prior_distribution.PriorDistribution()
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(engine.get_calibration_outputs(), [])

  def test_get_named_calibrated_priors_calibrated_roi_m_returns_dict(self):
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
    )
    mock_prior = prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(
        engine.get_named_calibrated_priors(), {'roi_m': calibrated_dist}
    )

  def test_get_named_calibrated_priors_calibrated_roi_rf_returns_dict(self):
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
    )
    mock_prior = prior_distribution.PriorDistribution(roi_rf=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(
        engine.get_named_calibrated_priors(), {'roi_rf': calibrated_dist}
    )

  def test_get_calibrated_priors_calibrated_roi_m_returns_list(self):
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
    )
    mock_prior = prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(engine.get_calibrated_priors(), [calibrated_dist])

  def test_get_calibrated_priors_calibrated_roi_rf_returns_list(self):
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
    )
    mock_prior = prior_distribution.PriorDistribution(roi_rf=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(engine.get_calibrated_priors(), [calibrated_dist])

  def test_get_calibration_outputs_returns_non_none_outputs(self):
    intermediary_dist = backend.tfd.Normal(
        backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
    )
    mock_output_1 = calibration_base.CalibrationOutput(
        channel_name='ch1', intermediary_prior=intermediary_dist
    )
    mock_output_2 = calibration_base.CalibrationOutput(
        channel_name='ch2', intermediary_prior=intermediary_dist
    )
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True, False, True],
        calibration_outputs=[
            mock_output_1,
            None,
            mock_output_2,
        ],
    )
    mock_prior = prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    self.assertEqual(
        engine.get_calibration_outputs(), [mock_output_1, mock_output_2]
    )

  def _run_check_experiment_adjustment(
      self,
      mock_exp: calibration_base.CalibratedExperiment,
      channel_name: str = 'facebook_spend',
  ) -> eda_outcome.EDAOutcome[eda_outcome.ExperimentAdjustmentArtifact]:
    intermediary_dist = backend.tfd.Normal(
        backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
    )
    mock_output = calibration_base.CalibrationOutput(
        channel_name=channel_name,
        intermediary_prior=intermediary_dist,
        experiments=(mock_exp,),
    )
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
        calibration_outputs=[mock_output],
    )

    mock_prior = prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    self.mock_model_context.model_spec.prior = mock_prior

    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    return engine.check_experiment_adjustment()

  def _get_stage_data(
      self,
      stages: Sequence[eda_outcome.ExperimentAdjustmentStageData],
      target_stage: eda_outcome.CalibrationExperimentAdjustmentStage,
  ) -> eda_outcome.ExperimentAdjustmentStageData:
    for stage_data in stages:
      if stage_data.stage == target_stage:
        return stage_data
    self.fail(f'Stage {target_stage} not found in output stages: {stages}')

  def test_check_experiment_adjustment_stage1(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        tau_spend=0.1,
        tau_duration=0.2,
        tau_recency=0.3,
        gamma_duration=1.1,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    self.assertEqual(
        outcome.check_type, eda_outcome.EDACheckType.EXPERIMENT_ADJUSTMENT
    )
    self.assertEqual(outcome.findings[0].severity, eda_outcome.EDASeverity.INFO)

    artifacts = outcome.get_overall_artifacts()
    self.assertLen(artifacts, 1)
    artifact = artifacts[0]
    self.assertIsInstance(artifact, eda_outcome.ExperimentAdjustmentArtifact)
    self.assertLen(artifact.adjustment_data, 1)

    exp_list = artifact.adjustment_data.get('facebook_spend')
    self.assertIsNotNone(exp_list)
    self.assertLen(exp_list, 1)

    exp_data = exp_list[0]
    self.assertEqual(exp_data.source_type, 'MeridianGeoX')

    stage1 = self._get_stage_data(
        exp_data.stages,
        eda_outcome.CalibrationExperimentAdjustmentStage.UNADJUSTED_RAW,
    )
    self.assertAlmostEqual(stage1.point_estimate, 1.5)
    self.assertAlmostEqual(stage1.standard_error, 0.3)

  def test_check_experiment_adjustment_stage2(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        tau_spend=0.1,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    stage2 = self._get_stage_data(
        exp_list[0].stages,  # pyrefly: ignore[unsupported-operation]
        eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_ADJUSTED,
    )
    self.assertAlmostEqual(stage2.point_estimate, 1.5)
    self.assertAlmostEqual(stage2.standard_error, 0.3 * math.sqrt(1.1))

  def test_check_experiment_adjustment_invalid_tau_spend_raises(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        tau_spend=-1.5,
    )
    with self.assertRaisesRegex(
        ValueError, r'`tau_spend` must be >= -1.0, got -1.5.'
    ):
      self._run_check_experiment_adjustment(mock_exp)

  def test_check_experiment_adjustment_stage3(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        tau_spend=0.1,
        tau_duration=0.2,
        gamma_duration=1.1,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    stage3 = self._get_stage_data(
        exp_list[0].stages,  # pyrefly: ignore[unsupported-operation]
        eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_ADJUSTED,
    )
    self.assertAlmostEqual(stage3.point_estimate, 1.5 * 1.1)
    self.assertAlmostEqual(stage3.standard_error, 0.3 * math.sqrt(1.3))

  def test_check_experiment_adjustment_stage4(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        tau_spend=0.1,
        tau_duration=0.2,
        tau_recency=0.3,
        gamma_duration=1.1,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    stage4 = self._get_stage_data(
        exp_list[0].stages,  # pyrefly: ignore[unsupported-operation]
        eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_RECENCY_ADJUSTED,
    )
    self.assertAlmostEqual(stage4.point_estimate, 1.5 * 1.1)
    self.assertAlmostEqual(stage4.standard_error, 0.3 * math.sqrt(1.6))

  def test_check_experiment_adjustment_invalid_tau_recency_raises(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        tau_spend=0.1,
        tau_duration=0.2,
        tau_recency=-1.5,
    )
    with self.assertRaisesRegex(
        ValueError,
        r'`1.0 \+ tau_spend \+ tau_duration \+ tau_recency` must be >= 0',
    ):
      self._run_check_experiment_adjustment(mock_exp)

  def test_check_experiment_adjustment_invalid_tau_duration_raises(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        tau_spend=0.1,
        tau_duration=-1.5,
    )
    with self.assertRaisesRegex(
        ValueError,
        r'`1.0 \+ tau_spend \+ tau_duration` must be >= 0',
    ):
      self._run_check_experiment_adjustment(mock_exp)

  def test_check_experiment_adjustment_stage5(self):
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.5,
        standard_error=0.3,
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        tau_spend=0.1,
        tau_duration=0.2,
        tau_recency=0.3,
        gamma_duration=1.1,
        adjusted_point_estimate=1.65,
        adjusted_standard_error=0.38,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    stage5 = self._get_stage_data(
        exp_list[0].stages,  # pyrefly: ignore[unsupported-operation]
        eda_outcome.CalibrationExperimentAdjustmentStage.FINAL_ADJUSTED,
    )
    self.assertAlmostEqual(stage5.point_estimate, 1.65)
    self.assertAlmostEqual(stage5.standard_error, 0.38)

  def test_check_experiment_adjustment_includes_user_adjustment_stage(
      self,
  ) -> None:
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.0,
        standard_error=0.2,
        gamma_duration=1.1,
        tau_spend=0.3,
        tau_duration=0.2,
        tau_recency=0.5,
        user_point_estimate_adjustment=0.15,
        user_standard_error_adjustment=0.25,
        adjusted_point_estimate=1.25,
        adjusted_standard_error=0.3,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    self.assertIsNotNone(exp_list)
    assert exp_list is not None
    self.assertLen(exp_list[0].stages, 6)

    stage_user = self._get_stage_data(
        exp_list[0].stages,
        eda_outcome.CalibrationExperimentAdjustmentStage.SPEND_DURATION_RECENCY_USER_ADJUSTED,
    )
    self.assertAlmostEqual(stage_user.point_estimate, 1.25)
    self.assertAlmostEqual(stage_user.standard_error, 0.3)

  def test_check_experiment_adjustment_omits_user_adjustment_stage_when_none(
      self,
  ) -> None:
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.0,
        standard_error=0.2,
        gamma_duration=1.1,
        tau_spend=0.3,
        tau_duration=0.2,
        tau_recency=0.5,
        user_point_estimate_adjustment=None,
        user_standard_error_adjustment=None,
        adjusted_point_estimate=1.1,
        adjusted_standard_error=0.28,
    )
    outcome = self._run_check_experiment_adjustment(mock_exp)

    exp_list = outcome.get_overall_artifacts()[0].adjustment_data.get(
        'facebook_spend'
    )
    self.assertIsNotNone(exp_list)
    assert exp_list is not None
    self.assertLen(exp_list[0].stages, 5)

  def test_check_experiment_adjustment_negative_sum_tau_raises_value_error(
      self,
  ) -> None:
    mock_exp = _create_calibrated_experiment(
        point_estimate=1.0,
        standard_error=0.2,
        gamma_duration=1.0,
        tau_spend=0.0,
        tau_duration=0.0,
        tau_recency=0.0,
        user_point_estimate_adjustment=0.0,
        user_standard_error_adjustment=-1.5,
    )
    with self.assertRaisesRegex(
        ValueError,
        'user_standard_error_adjustment` must be >= 0',
    ):
      self._run_check_experiment_adjustment(mock_exp)

  def _create_mock_calibrated_prior(
      self,
      channel_name: str = 'facebook_spend',
      baseline_prior: backend.tfd.Distribution | None = None,
      calibrated_prior: backend.tfd.Distribution | None = None,
      intermediary_prior: backend.tfd.Distribution | None = None,
      experiments: Sequence[calibration_base.CalibratedExperiment] = (),
  ) -> None:
    effective_calibrated_prior = (
        calibrated_prior
        if calibrated_prior is not None
        else backend.tfd.LogNormal(0.2, 0.9)
    )
    raw_intermediary = (
        intermediary_prior
        if intermediary_prior is not None
        else effective_calibrated_prior
    )
    if isinstance(raw_intermediary, roi.GridDistribution):
      intermediary_dist = raw_intermediary
    else:
      grid = np.linspace(0.01, 20.0, 1000, dtype=backend.np_float_dtype)
      dx = float(grid[1] - grid[0])
      pdf = np.exp(
          np.asarray(
              raw_intermediary.log_prob(grid), dtype=backend.np_float_dtype
          )
      )
      pdf = np.where(np.isfinite(pdf), pdf, 0.0)
      intermediary_dist = roi.GridDistribution(
          grid=grid, pdf=pdf, dx=dx
      )
    mock_output = calibration_base.CalibrationOutput(
        channel_name=channel_name,
        baseline_prior=baseline_prior,
        intermediary_prior=intermediary_dist,
        experiments=list(experiments),
    )
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=[effective_calibrated_prior],
        is_calibrated=[True],
        calibration_outputs=[mock_output],
    )
    self.mock_model_context.model_spec.prior = (
        prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    )

  def _create_mock_experiment(
      self, point_estimate: float = 0.5, standard_error: float = 0.1
  ) -> calibration_base.CalibratedExperiment:
    adj_result = _create_experiment_result(point_estimate, standard_error)
    return calibration_base.CalibratedExperiment(
        source_type=calibration_base.SourceType.MERIDIAN_GEOX,
        raw_experiment_result=adj_result,
        adjusted_experiment_result=adj_result,
        tau_spend=0.0,
        tau_duration=0.0,
        tau_recency=0.0,
        gamma_duration=1.0,
    )

  def test_check_prior_quality(self):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='facebook_spend',
        baseline_prior=backend.tfd.Normal(
            loc=backend.np_float_dtype(0.2), scale=backend.np_float_dtype(0.9)
        ),
        calibrated_prior=backend.tfd.LogNormal(
            loc=backend.np_float_dtype(0.2), scale=backend.np_float_dtype(0.9)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    with self.subTest('outcome_metadata'):
      self.assertEqual(
          outcome.check_type, eda_outcome.EDACheckType.PRIOR_QUALITY
      )
      self.assertEqual(
          outcome.findings[0].severity, eda_outcome.EDASeverity.INFO
      )

    with self.subTest('artifact_data'):
      artifacts = outcome.get_overall_artifacts()
      self.assertLen(artifacts, 1)
      artifact = artifacts[0]
      self.assertIsInstance(artifact, eda_outcome.PriorQualityArtifact)
      self.assertLen(artifact.prior_quality_data, 1)
      self.assertEqual(
          artifact.prior_quality_data[0].channel_name, 'facebook_spend'
      )
      self.assertAlmostEqual(
          artifact.prior_quality_data[0].prior_width_ratio, 1.0, places=1
      )
      self.assertEqual(artifact.prior_quality_data[0].bimodal_statistic, 0.0)
      self.assertGreater(artifact.prior_quality_data[0].overlap_percentage, 0.0)
      self.assertEqual(artifact.prior_quality_data[0].n_negative_experiments, 0)

  def test_check_prior_quality_unflagged_is_info(self):
    self._create_mock_calibrated_prior(
        channel_name='facebook_spend',
        baseline_prior=backend.tfd.Normal(
            loc=backend.np_float_dtype(0.2), scale=backend.np_float_dtype(0.9)
        ),
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(0.5)
        ),
        experiments=[],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    self.assertEqual(outcome.findings[0].severity, eda_outcome.EDASeverity.INFO)

  def test_check_prior_quality_counts_negative_experiments(self):
    exp = self._create_mock_experiment(point_estimate=-0.25, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='channel_neg',
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]
    self.assertEqual(artifact.prior_quality_data[0].n_negative_experiments, 1)

  def test_check_prior_quality_high_width_ratio(self):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='channel_high_variance',
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(1.8)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]
    self.assertGreater(artifact.prior_quality_data[0].prior_width_ratio, 1.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='log_concave_baseline_returns_zero_bimodality',
          baseline_dist_cls=backend.tfd.Normal,
          baseline_params=(0.2, 0.9),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='lognormal_unimodal_returns_zero_bimodality',
          baseline_dist_cls=backend.tfd.LogNormal,
          baseline_params=(0.2, 0.9),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='lognormal_bimodal_returns_one_bimodality',
          baseline_dist_cls=backend.tfd.LogNormal,
          baseline_params=(-3.0, 1.5),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[(5.0, 0.5)],
          expected_bimodal_stat=1.0,
      ),
      dict(
          testcase_name='no_experiments_returns_zero_bimodality',
          baseline_dist_cls=backend.tfd.LogNormal,
          baseline_params=(0.2, 0.9),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='none_baseline_prior_returns_zero_bimodality',
          baseline_dist_cls=None,
          baseline_params=None,
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='unsupported_baseline_prior_returns_none_bimodality',
          baseline_dist_cls=backend.tfd.StudentT,
          baseline_params=(3.0, 0.0, 1.0),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=None,
      ),
  )
  def test_check_prior_quality_scenarios(
      self,
      baseline_dist_cls: type[backend.tfd.Distribution] | None,
      baseline_params: tuple[float, ...] | None,
      calibrated_dist_cls: type[backend.tfd.Distribution],
      calibrated_params: tuple[float, float],
      exp_params: Sequence[tuple[float, float]],
      expected_bimodal_stat: float | None,
  ):
    if baseline_dist_cls is not None and baseline_params is not None:
      baseline_prior = baseline_dist_cls(
          *(backend.np_float_dtype(p) for p in baseline_params)
      )
    else:
      baseline_prior = None
    calibrated_prior = calibrated_dist_cls(
        *(backend.np_float_dtype(p) for p in calibrated_params)
    )
    experiments = [
        self._create_mock_experiment(pe, se) for pe, se in exp_params
    ]
    self._create_mock_calibrated_prior(
        channel_name='facebook_spend',
        baseline_prior=baseline_prior,
        calibrated_prior=calibrated_prior,
        experiments=experiments,
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    if expected_bimodal_stat is None:
      self.assertIsNone(artifact.prior_quality_data[0].bimodal_statistic)
    else:
      self.assertIsNotNone(artifact.prior_quality_data[0].bimodal_statistic)
      self.assertAlmostEqual(
          artifact.prior_quality_data[0].bimodal_statistic,
          expected_bimodal_stat,
          places=2,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='colab_bimodal_case_1',
          prior_mean=0.2,
          prior_std=6.1,
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=1.0,
      ),
      dict(
          testcase_name='colab_bimodal_case_2',
          prior_mean=1.8,
          prior_std=2.1,
          exp_params=[(14.0, 4.0)],
          expected_bimodal_stat=1.0,
      ),
      dict(
          testcase_name='colab_unimodal_case_1',
          prior_mean=5.0,
          prior_std=8.1,
          exp_params=[(10.0, 4.0)],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='colab_unimodal_case_2',
          prior_mean=5.0,
          prior_std=0.1,
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=0.0,
      ),
      dict(
          testcase_name='colab_borderline_bimodal_case_1',
          prior_mean=1.0,
          prior_std=2.1,
          exp_params=[(18.0, 4.0)],
          expected_bimodal_stat=1.0,
      ),
      dict(
          testcase_name='colab_borderline_bimodal_case_2',
          prior_mean=1.0,
          prior_std=3.1,
          exp_params=[(18.0, 4.0)],
          expected_bimodal_stat=1.0,
      ),
      dict(
          testcase_name='colab_borderline_bimodal_case_3',
          prior_mean=1.0,
          prior_std=9.1,
          exp_params=[(0.5, 0.1)],
          expected_bimodal_stat=1.0,
      ),
  )
  def test_check_prior_quality_lognormal_analytical_bimodality_scenarios(
      self,
      prior_mean: float,
      prior_std: float,
      exp_params: Sequence[tuple[float, float]],
      expected_bimodal_stat: float,
  ):
    baseline_prior = prior_distribution.lognormal_dist_from_mean_std(
        prior_mean, prior_std
    )
    calibrated_prior = backend.tfd.LogNormal(
        backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
    )
    experiments = [
        self._create_mock_experiment(pe, se) for pe, se in exp_params
    ]
    self._create_mock_calibrated_prior(
        channel_name='facebook_spend',
        baseline_prior=baseline_prior,
        calibrated_prior=calibrated_prior,
        experiments=experiments,
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertIsNotNone(artifact.prior_quality_data[0].bimodal_statistic)
    self.assertAlmostEqual(
        artifact.prior_quality_data[0].bimodal_statistic,
        expected_bimodal_stat,
        places=2,
    )

  def test_check_prior_quality_high_width_ratio_flags_attention(self):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='channel_wide',
        baseline_prior=backend.tfd.Normal(
            loc=backend.np_float_dtype(0.2), scale=backend.np_float_dtype(0.9)
        ),
        calibrated_prior=backend.tfd.LogNormal(
            loc=backend.np_float_dtype(0.2), scale=backend.np_float_dtype(1.5)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertGreater(artifact.prior_quality_data[0].prior_width_ratio, 1.0)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.REVIEW
    )

  def test_check_prior_quality_lognormal_bimodal_flags_attention(self):
    exp = self._create_mock_experiment(point_estimate=5.0, standard_error=0.5)
    self._create_mock_calibrated_prior(
        channel_name='channel_bimodal',
        baseline_prior=backend.tfd.LogNormal(
            loc=backend.np_float_dtype(-3.0), scale=backend.np_float_dtype(1.5)
        ),
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertEqual(artifact.prior_quality_data[0].bimodal_statistic, 1.0)
    self.assertEqual(
        artifact.prior_quality_data[0].baseline_prior_type, 'LogNormal'
    )
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.REVIEW
    )

  def test_check_prior_quality_unsupported_baseline_prior_flags_attention(self):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='channel_student_t',
        baseline_prior=backend.tfd.StudentT(
            df=backend.np_float_dtype(3.0),
            loc=backend.np_float_dtype(0.0),
            scale=backend.np_float_dtype(1.0),
        ),
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertIsNone(artifact.prior_quality_data[0].bimodal_statistic)
    self.assertEqual(
        artifact.prior_quality_data[0].baseline_prior_type, 'StudentT'
    )
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.REVIEW
    )

  def test_check_prior_quality_improper_uniform_baseline_prior_type(self):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    self._create_mock_calibrated_prior(
        channel_name='channel_none',
        baseline_prior=None,
        calibrated_prior=backend.tfd.LogNormal(
            backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertEqual(artifact.prior_quality_data[0].bimodal_statistic, 0.0)
    self.assertEqual(
        artifact.prior_quality_data[0].baseline_prior_type, 'None'
    )

  def test_check_prior_quality_disjoint_posterior_grid_sets_zero_overlap(
      self,
  ):
    exp = self._create_mock_experiment(point_estimate=0.5, standard_error=0.1)
    disjoint_grid = np.linspace(50.0, 100.0, 1000, dtype=backend.np_float_dtype)
    disjoint_dx = float(disjoint_grid[1] - disjoint_grid[0])
    disjoint_pdf = np.ones(
        1000, dtype=backend.np_float_dtype
    ) / backend.np_float_dtype(50.0)
    disjoint_prior = roi.GridDistribution(
        grid=disjoint_grid, pdf=disjoint_pdf, dx=disjoint_dx
    )
    self._create_mock_calibrated_prior(
        channel_name='channel_disjoint',
        baseline_prior=backend.tfd.Uniform(
            low=backend.np_float_dtype(50.0),
            high=backend.np_float_dtype(100.0),
        ),
        intermediary_prior=disjoint_prior,
        calibrated_prior=backend.tfd.Uniform(
            low=backend.np_float_dtype(0.0), high=backend.np_float_dtype(10.0)
        ),
        experiments=[exp],
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertEqual(artifact.prior_quality_data[0].overlap_percentage, 0.0)
    self.assertEqual(
        outcome.findings[0].severity, eda_outcome.EDASeverity.REVIEW
    )

  def test_check_prior_quality_skips_uncalibrated_channels(self):
    grid = np.linspace(0.01, 20.0, 1000, dtype=np.float32)
    dx = float(grid[1] - grid[0])
    pdf = np.exp(
        np.asarray(
            backend.tfd.LogNormal(
                backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
            ).log_prob(grid),
            dtype=backend.np_float_dtype,
        )
    )
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    mock_output = calibration_base.CalibrationOutput(
        channel_name='channel_0',
        baseline_prior=backend.tfd.LogNormal(0.2, 0.9),
        intermediary_prior=roi.GridDistribution(grid=grid, pdf=pdf, dx=dx),
        experiments=[],
    )
    calibrated_dist = calibration_base.CalibratedDistribution(
        distributions=[
            backend.tfd.LogNormal(
                backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
            ),
            backend.tfd.LogNormal(
                backend.np_float_dtype(0.2), backend.np_float_dtype(0.9)
            ),
        ],
        is_calibrated=[True, False],
        calibration_outputs=[mock_output, None],
    )
    self.mock_model_context.model_spec.prior = (
        prior_distribution.PriorDistribution(roi_m=calibrated_dist)
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    self.assertLen(artifact.prior_quality_data, 1)
    self.assertEqual(artifact.prior_quality_data[0].channel_name, 'channel_0')

  @parameterized.named_parameters(
      dict(
          testcase_name='overlap_percentage_computed',
          baseline_dist_cls=backend.tfd.Normal,
          baseline_params=(0.5, 0.1),
          exp_params=[(0.5, 0.1)],
          intermediary_dist_cls=backend.tfd.Normal,
          intermediary_params=(0.5, 0.1),
          calibrated_dist_cls=backend.tfd.Normal,
          calibrated_params=(0.5, 0.1),
          expected_overlap_min=0.0,
          expected_overlap_max=1.0,
          expected_severity=eda_outcome.EDASeverity.INFO,
      ),
      dict(
          testcase_name='low_overlap_flags_attention',
          baseline_dist_cls=backend.tfd.LogNormal,
          baseline_params=(0.0, 1.0),
          exp_params=[(0.1, 0.01)],
          intermediary_dist_cls=backend.tfd.LogNormal,
          intermediary_params=(0.0, 0.1),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(2.0, 0.1),
          expected_overlap_min=0.0,
          expected_overlap_max=eda_constants.OVERLAP_PERCENTAGE_THRESHOLD,
          expected_severity=eda_outcome.EDASeverity.REVIEW,
      ),
      dict(
          testcase_name='inexact_overlap_passes_threshold',
          baseline_dist_cls=backend.tfd.Normal,
          baseline_params=(1.0, 0.2),
          exp_params=[(1.05, 0.25)],
          intermediary_dist_cls=backend.tfd.Normal,
          intermediary_params=(1.05, 0.2),
          calibrated_dist_cls=backend.tfd.Normal,
          calibrated_params=(1.02, 0.15),
          expected_overlap_min=eda_constants.OVERLAP_PERCENTAGE_THRESHOLD,
          expected_overlap_max=1.0,
          expected_severity=eda_outcome.EDASeverity.INFO,
      ),
      dict(
          testcase_name='no_experiments_overlap_is_one',
          baseline_dist_cls=backend.tfd.Normal,
          baseline_params=(0.2, 0.9),
          exp_params=[],
          intermediary_dist_cls=backend.tfd.LogNormal,
          intermediary_params=(0.2, 0.9),
          calibrated_dist_cls=backend.tfd.LogNormal,
          calibrated_params=(0.2, 0.9),
          expected_overlap_min=1.0,
          expected_overlap_max=1.0,
          expected_severity=eda_outcome.EDASeverity.INFO,
      ),
  )
  def test_check_prior_quality_overlap_scenarios(
      self,
      baseline_dist_cls: type[backend.tfd.Distribution],
      baseline_params: tuple[float, float],
      exp_params: Sequence[tuple[float, float]],
      intermediary_dist_cls: type[backend.tfd.Distribution],
      intermediary_params: tuple[float, float],
      calibrated_dist_cls: type[backend.tfd.Distribution],
      calibrated_params: tuple[float, float],
      expected_overlap_min: float,
      expected_overlap_max: float,
      expected_severity: eda_outcome.EDASeverity,
  ):
    experiments = [
        self._create_mock_experiment(pe, se) for pe, se in exp_params
    ]
    self._create_mock_calibrated_prior(
        channel_name='channel_overlap',
        baseline_prior=baseline_dist_cls(
            *(backend.np_float_dtype(p) for p in baseline_params)
        ),
        intermediary_prior=intermediary_dist_cls(
            *(backend.np_float_dtype(p) for p in intermediary_params)
        ),
        calibrated_prior=calibrated_dist_cls(
            *(backend.np_float_dtype(p) for p in calibrated_params)
        ),
        experiments=experiments,
    )
    engine = sampling_eda_engine.SamplingEDAEngine(analyzer=self.mock_analyzer)
    outcome = engine.check_prior_quality()
    artifact = outcome.get_overall_artifacts()[0]

    overlap = artifact.prior_quality_data[0].overlap_percentage
    self.assertGreaterEqual(overlap, expected_overlap_min)
    self.assertLessEqual(overlap, expected_overlap_max)
    self.assertEqual(outcome.findings[0].severity, expected_severity)


if __name__ == '__main__':
  absltest.main()
