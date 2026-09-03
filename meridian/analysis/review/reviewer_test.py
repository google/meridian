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
import immutabledict
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import constants as review_constants
from meridian.analysis.review import results
from meridian.analysis.review import reviewer
from meridian.data import input_data
from meridian.model import context
from meridian.model import prior_distribution
from meridian.model import spec as model_spec_module
from meridian.model.calibration import base as calibration_base
import numpy as np
import xarray as xr


class ReviewerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(analyzer_module, 'Analyzer', autospec=True)
    )
    self._model_context = mock.create_autospec(
        context.ModelContext, spec_set=True, instance=True
    )
    type(self._model_context).n_media_channels = mock.PropertyMock(
        return_value=1
    )
    type(self._model_context).n_rf_channels = mock.PropertyMock(return_value=1)
    type(self._model_context).is_roi_prior = mock.PropertyMock(
        return_value=True
    )
    type(self._model_context).is_custom_roi_prior = mock.PropertyMock(
        return_value=True
    )
    self._inference_data = mock.create_autospec(
        az.InferenceData, spec_set=True, instance=True
    )

    convergence_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.ConvergenceCheck'
    )
    self._mock_convergence_check_cls = self.enter_context(
        convergence_check_cls_patcher
    )
    self._mock_convergence_check = self._mock_convergence_check_cls.return_value
    self._mock_convergence_result = mock.create_autospec(
        spec=results.ConvergenceCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_convergence_result.config = configs.ConvergenceConfig()
    self._mock_convergence_check.run.return_value = (
        self._mock_convergence_result
    )
    self._mock_convergence_check_cls.__name__ = 'ConvergenceCheck'

    roi_consistency_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.ROIConsistencyCheck'
    )
    self._mock_roi_consistency_check_cls = self.enter_context(
        roi_consistency_check_cls_patcher
    )
    self._mock_roi_consistency_check = (
        self._mock_roi_consistency_check_cls.return_value
    )
    self._mock_roi_consistency_result = mock.create_autospec(
        spec=results.ROIConsistencyCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_roi_consistency_result.config = configs.ROIConsistencyConfig()
    self._mock_roi_consistency_result.channel_results = [
        mock.create_autospec(
            spec=results.ROIConsistencyChannelResult,
            instance=True,
            spec_set=False,
            case=results.ROIConsistencyChannelCases.ROI_PASS,
        )
    ]
    self._mock_roi_consistency_check.run.return_value = (
        self._mock_roi_consistency_result
    )
    self._mock_roi_consistency_check_cls.__name__ = 'ROIConsistencyCheck'

    baseline_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.BaselineCheck'
    )
    self._mock_baseline_check_cls = self.enter_context(
        baseline_check_cls_patcher
    )
    self._mock_baseline_check = self._mock_baseline_check_cls.return_value
    self._mock_baseline_result = mock.create_autospec(
        spec=results.BaselineCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_baseline_result.config = configs.BaselineConfig()
    self._mock_baseline_result.negative_baseline_prob = 0.05
    self._mock_baseline_check.run.return_value = self._mock_baseline_result
    self._mock_baseline_check_cls.__name__ = 'BaselineCheck'

    bayesian_ppp_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.BayesianPPPCheck'
    )
    self._mock_bayesian_ppp_check_cls = self.enter_context(
        bayesian_ppp_check_cls_patcher
    )
    self._mock_bayesian_ppp_check = (
        self._mock_bayesian_ppp_check_cls.return_value
    )
    self._mock_bayesian_ppp_result = mock.create_autospec(
        spec=results.BayesianPPPCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_bayesian_ppp_result.config = configs.BayesianPPPConfig()
    self._mock_bayesian_ppp_result.bayesian_ppp = 0.1
    self._mock_bayesian_ppp_check.run.return_value = (
        self._mock_bayesian_ppp_result
    )
    self._mock_bayesian_ppp_check_cls.__name__ = 'BayesianPPPCheck'

    gof_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.GoodnessOfFitCheck'
    )
    self._mock_gof_check_cls = self.enter_context(gof_check_cls_patcher)
    self._mock_gof_check = self._mock_gof_check_cls.return_value
    self._mock_gof_result = mock.create_autospec(
        spec=results.GoodnessOfFitCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_gof_result.config = configs.GoodnessOfFitConfig()
    self._mock_gof_result.metrics = results.GoodnessOfFitMetrics(
        r_squared=1.0, mape=0.1, wmape=0.2
    )
    self._mock_gof_check.run.return_value = self._mock_gof_result
    self._mock_gof_check_cls.__name__ = 'GoodnessOfFitCheck'

    prior_posterior_shift_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.PriorPosteriorShiftCheck'
    )
    self._mock_pps_check_cls = self.enter_context(
        prior_posterior_shift_cls_patcher
    )
    self._mock_pps_check = self._mock_pps_check_cls.return_value
    self._mock_pps_result = mock.create_autospec(
        spec=results.PriorPosteriorShiftCheckResult,
        instance=True,
        spec_set=False,
    )
    self._mock_pps_result.config = configs.PriorPosteriorShiftConfig()
    self._mock_pps_result.no_shift_channels = []
    self._mock_pps_result.channel_results = [
        mock.create_autospec(
            spec=results.PriorPosteriorShiftChannelResult,
            instance=True,
            spec_set=False,
        ),
        mock.create_autospec(
            spec=results.PriorPosteriorShiftChannelResult,
            instance=True,
            spec_set=False,
        ),
    ]
    self._mock_pps_check.run.return_value = self._mock_pps_result
    self._mock_pps_check_cls.__name__ = 'PriorPosteriorShiftCheck'
    self._mock_convergence_check_cls.is_relevant.return_value = True
    self._mock_baseline_check_cls.is_relevant.return_value = True
    self._mock_bayesian_ppp_check_cls.is_relevant.return_value = True
    self._mock_gof_check_cls.is_relevant.return_value = True
    self._mock_pps_check_cls.is_relevant.side_effect = (
        lambda model_context, inf_data=None: model_context.is_roi_prior
    )
    self._mock_roi_consistency_check_cls.is_relevant.side_effect = (
        lambda model_context, inf_data=None: model_context.is_custom_roi_prior
    )

    patcher = mock.patch.object(
        reviewer,
        '_POST_CONVERGENCE_CHECKS',
        new=immutabledict.immutabledict({
            self._mock_baseline_check_cls: configs.BaselineConfig(),
            self._mock_bayesian_ppp_check_cls: configs.BayesianPPPConfig(),
            self._mock_gof_check_cls: configs.GoodnessOfFitConfig(),
            self._mock_pps_check_cls: configs.PriorPosteriorShiftConfig(),
            self._mock_roi_consistency_check_cls: (
                configs.ROIConsistencyConfig()
            ),
        }),
    )
    patcher.start()
    self.addCleanup(patcher.stop)

    mock_map_by_name = {
        'BaselineCheck': self._mock_baseline_check_cls,
        'BayesianPPPCheck': self._mock_bayesian_ppp_check_cls,
        'GoodnessOfFitCheck': self._mock_gof_check_cls,
        'PriorPosteriorShiftCheck': self._mock_pps_check_cls,
        'ROIConsistencyCheck': self._mock_roi_consistency_check_cls,
    }

    new_components = []
    for comp in reviewer._HEALTH_SCORE_COMPONENTS:
      if comp.check_type.__name__ in mock_map_by_name:
        new_components.append(
            dataclasses.replace(
                comp, check_type=mock_map_by_name[comp.check_type.__name__]
            )
        )
      else:
        new_components.append(comp)

    patcher_health = mock.patch.object(
        reviewer, '_HEALTH_SCORE_COMPONENTS', tuple(new_components)
    )
    patcher_health.start()
    self.addCleanup(patcher_health.stop)

  @parameterized.named_parameters(
      dict(
          testcase_name='perfect_score',
          baseline_prob=0.05,
          bayesian_ppp=0.1,
          gof_r2=1.0,
          pps_no_shift=0,
          pps_total=2,
          roi_review=0,
          roi_total=1,
          expected_score=100.0,
      ),
      dict(
          testcase_name='worst_score',
          baseline_prob=0.9,
          bayesian_ppp=0.01,
          gof_r2=0.0,
          pps_no_shift=2,
          pps_total=2,
          roi_review=1,
          roi_total=1,
          expected_score=0.0,
      ),
      dict(
          testcase_name='mixed_score',
          baseline_prob=0.5,
          bayesian_ppp=0.1,
          gof_r2=0.5,
          pps_no_shift=1,
          pps_total=2,
          roi_review=1,
          roi_total=2,
          expected_score=72.7,
      ),
      dict(
          testcase_name='edge_cases',
          baseline_prob=0.2,
          bayesian_ppp=0.05,
          gof_r2=0.6,
          pps_no_shift=0,
          pps_total=1,
          roi_review=1,
          roi_total=1,
          expected_score=53.2,
      ),
  )
  def test_health_score_value_correct(
      self,
      baseline_prob,
      bayesian_ppp,
      gof_r2,
      pps_no_shift,
      pps_total,
      roi_review,
      roi_total,
      expected_score,
  ):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_baseline_result.negative_baseline_prob = baseline_prob
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_bayesian_ppp_result.bayesian_ppp = bayesian_ppp
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )

    self._mock_roi_consistency_result.channel_results = []
    for _ in range(roi_review):
      self._mock_roi_consistency_result.channel_results.append(
          mock.create_autospec(
              spec=results.ROIConsistencyChannelResult,
              instance=True,
              spec_set=False,
              case=mock.create_autospec(
                  spec=results.ROIConsistencyChannelCases,
                  instance=True,
                  spec_set=False,
                  status=results.Status.REVIEW,
              ),
          )
      )
    for _ in range(roi_total - roi_review):
      self._mock_roi_consistency_result.channel_results.append(
          mock.create_autospec(
              spec=results.ROIConsistencyChannelResult,
              instance=True,
              spec_set=False,
              case=mock.create_autospec(
                  spec=results.ROIConsistencyChannelCases,
                  instance=True,
                  spec_set=False,
                  status=results.Status.PASS,
              ),
          )
      )

    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_gof_result.metrics = results.GoodnessOfFitMetrics(
        r_squared=gof_r2, mape=0.1, wmape=0.1
    )
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS
    self._mock_pps_result.no_shift_channels = ['ch'] * pps_no_shift
    self._mock_pps_result.channel_results = [
        mock.create_autospec(
            spec=results.PriorPosteriorShiftChannelResult,
            instance=True,
            spec_set=False,
        )
    ] * pps_total

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertAlmostEqual(summary.health_score, expected_score, places=1)

  def test_health_score_empty_channel_results(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_baseline_result.negative_baseline_prob = 0.05
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_bayesian_ppp_result.bayesian_ppp = 0.5
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_gof_result.metrics = results.GoodnessOfFitMetrics(
        r_squared=1.0, mape=0.1, wmape=0.1
    )
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_roi_consistency_result.channel_results = []
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS
    self._mock_pps_result.no_shift_channels = []
    self._mock_pps_result.channel_results = []

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()
    self.assertAlmostEqual(summary.health_score, 100.0, places=1)

  def test_run_pass_with_roi_consistency_review(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.REVIEW
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertEqual(
        summary.summary_message, 'Passed with reviews: Review is needed.'
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_pass_with_gof_review(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.REVIEW
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertEqual(
        summary.summary_message, 'Passed with reviews: Review is needed.'
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_with_custom_convergence_config(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    custom_conv_config = configs.ConvergenceConfig(convergence_threshold=1.1)
    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        convergence_check_config=custom_conv_config,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self._mock_convergence_check_cls.assert_called_once_with(
        model_context=self._model_context,
        inference_data=self._inference_data,
        analyzer=review._analyzer,
        config=custom_conv_config,
        selected_geos=None,
        selected_times=None,
    )

  def test_run_pass_with_pps_review(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = (
        results.PriorPosteriorShiftAggregateCases.REVIEW
    )

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertEqual(
        summary.summary_message, 'Passed with reviews: Review is needed.'
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_fail_not_converged_skips_other_checks(self):
    self._mock_convergence_result.case = results.ConvergenceCases.NOT_CONVERGED

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        'Failed: Model did not converge. Other checks were skipped.',
    )
    self.assertEqual(summary.health_score, 0.0)
    self.assertLen(summary.results, 1)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_not_called()
    self._mock_bayesian_ppp_check_cls.assert_not_called()
    self._mock_roi_consistency_check_cls.assert_not_called()
    self._mock_gof_check_cls.assert_not_called()
    self._mock_pps_check_cls.assert_not_called()

  def test_run_fail_not_fully_converged(self):
    self._mock_convergence_result.case = (
        results.ConvergenceCases.NOT_FULLY_CONVERGED
    )
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        (
            'Failed: Quality issues were detected in your model. Address failed'
            ' checks before proceeding.'
        ),
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_fail_with_reviews(self):
    self._mock_convergence_result.case = (
        results.ConvergenceCases.NOT_FULLY_CONVERGED
    )
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.REVIEW
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        (
            'Failed: Quality issues were detected in your model. Follow'
            ' recommendations to address any failed checks and review'
            ' results to determine if further action is needed.'
        ),
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_converged_with_fail_and_review(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.FAIL
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.REVIEW
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        (
            'Failed: Quality issues were detected in your model. Follow'
            ' recommendations to address any failed checks and review'
            ' results to determine if further action is needed.'
        ),
    )
    self.assertLen(summary.results, 6)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_fail_baseline(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.FAIL
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        (
            'Failed: Quality issues were detected in your model. Address failed'
            ' checks before proceeding.'
        ),
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_pass_with_baseline_review(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.REVIEW
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertEqual(
        summary.summary_message, 'Passed with reviews: Review is needed.'
    )
    self.assertLen(summary.results, 6)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self.assertEqual(summary.results[4], self._mock_pps_result)
    self.assertEqual(summary.results[5], self._mock_roi_consistency_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_called_once()

  def test_run_with_custom_post_convergence_checks(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS

    custom_checks = immutabledict.immutabledict({
        self._mock_baseline_check_cls: configs.BaselineConfig(),
        self._mock_bayesian_ppp_check_cls: configs.BayesianPPPConfig(),
        self._mock_gof_check_cls: configs.GoodnessOfFitConfig(),
    })

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=custom_checks,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertLen(summary.results, 4)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_roi_consistency_check_cls.assert_not_called()
    self._mock_pps_check_cls.assert_not_called()

  def test_run_with_selected_geos_and_times(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    review.run(selected_geos=['geo1'], selected_times=['time1'])

    self._mock_convergence_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.ConvergenceConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_baseline_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.BaselineConfig(),
        selected_geos=['geo1'],
        selected_times=['time1'],
    )
    self._mock_bayesian_ppp_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.BayesianPPPConfig(),
        selected_geos=['geo1'],
        selected_times=['time1'],
    )
    self._mock_roi_consistency_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.ROIConsistencyConfig(),
        selected_geos=['geo1'],
        selected_times=['time1'],
    )
    self._mock_gof_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.GoodnessOfFitConfig(),
        selected_geos=['geo1'],
        selected_times=['time1'],
    )
    self._mock_pps_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.PriorPosteriorShiftConfig(),
        selected_geos=['geo1'],
        selected_times=['time1'],
    )

  def test_run_skip_checks_with_custom_roi_priors(self):
    type(self._model_context).is_roi_prior = mock.PropertyMock(
        return_value=False
    )
    type(self._model_context).is_custom_roi_prior = mock.PropertyMock(
        return_value=False
    )
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertLen(summary.results, 4)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_not_called()
    self._mock_roi_consistency_check_cls.assert_not_called()

  def test_run_skip_checks_with_non_roi_priors(self):
    type(self._model_context).is_roi_prior = mock.PropertyMock(
        return_value=False
    )
    type(self._model_context).is_custom_roi_prior = mock.PropertyMock(
        return_value=False
    )
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertLen(summary.results, 4)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_bayesian_ppp_result)
    self.assertEqual(summary.results[3], self._mock_gof_result)
    self._mock_convergence_check_cls.assert_called_once()
    self._mock_baseline_check_cls.assert_called_once()
    self._mock_bayesian_ppp_check_cls.assert_called_once()
    self._mock_gof_check_cls.assert_called_once()
    self._mock_pps_check_cls.assert_not_called()
    self._mock_roi_consistency_check_cls.assert_not_called()

  def test_run_with_default_configs(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_baseline_result.negative_baseline_prob = 0.05
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_bayesian_ppp_result.bayesian_ppp = 0.1
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_roi_consistency_result.channel_results = [
        mock.create_autospec(
            spec=results.ROIConsistencyChannelResult,
            instance=True,
            spec_set=False,
            case=results.ROIConsistencyChannelCases.ROI_PASS,
        )
    ]
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_gof_result.metrics = results.GoodnessOfFitMetrics(
        r_squared=0.7, mape=0.1, wmape=0.1
    )
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS
    self._mock_pps_result.no_shift_channels = []
    self._mock_pps_result.channel_results = [
        mock.create_autospec(
            spec=results.PriorPosteriorShiftChannelResult,
            instance=True,
            spec_set=False,
        ),
        mock.create_autospec(
            spec=results.PriorPosteriorShiftChannelResult,
            instance=True,
            spec_set=False,
        ),
    ]

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    review.run()

    self._mock_convergence_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.ConvergenceConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_baseline_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.BaselineConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_bayesian_ppp_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.BayesianPPPConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_roi_consistency_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.ROIConsistencyConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_gof_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.GoodnessOfFitConfig(),
        selected_geos=None,
        selected_times=None,
    )
    self._mock_pps_check_cls.assert_called_once_with(
        model_context=mock.ANY,
        inference_data=mock.ANY,
        analyzer=mock.ANY,
        config=configs.PriorPosteriorShiftConfig(),
        selected_geos=None,
        selected_times=None,
    )

  def test_run_missing_required_check_raises_error(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_baseline_result.negative_baseline_prob = 0.05
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_roi_consistency_result.channel_results = [
        mock.create_autospec(
            spec=results.ROIConsistencyChannelResult,
            instance=True,
            spec_set=False,
            case=results.ROIConsistencyChannelCases.ROI_PASS,
        )
    ]
    custom_checks = immutabledict.immutabledict({
        checks.BaselineCheck: configs.BaselineConfig(
            negative_baseline_prob_review_threshold=0.5,
            negative_baseline_prob_fail_threshold=0.9,
        ),
        checks.ROIConsistencyCheck: configs.ROIConsistencyConfig(
            prior_lower_quantile=0.05,
            prior_upper_quantile=0.95,
        ),
    })

    with mock.patch.object(
        reviewer, '_POST_CONVERGENCE_CHECKS', new=custom_checks
    ):
      review = reviewer.ModelReviewer(
          model_context=self._model_context,
          inference_data=self._inference_data,
      )
      with self.assertRaisesRegex(
          ValueError,
          r'The following required checks results are missing: '
          r'\[\'BayesianPPPCheck\', \'GoodnessOfFitCheck\'\].',
      ):
        review.run()

  def test_run_twice_clears_results(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary1 = review.run()
    summary2 = review.run()

    self.assertLen(summary1.results, 6)
    self.assertLen(summary2.results, 6)
    self.assertEqual(summary1.overall_status, results.Status.PASS)
    self.assertEqual(summary2.overall_status, results.Status.PASS)

  def test_checks_status(self):
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.REVIEW
    self._mock_pps_result.case = (
        results.PriorPosteriorShiftAggregateCases.REVIEW
    )

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
    )
    summary = review.run()

    self.assertEqual(
        summary.checks_status,
        {
            'ConvergenceCheckResult': 'PASS',
            'BaselineCheckResult': 'PASS',
            'BayesianPPPCheckResult': 'PASS',
            'GoodnessOfFitCheckResult': 'REVIEW',
            'PriorPosteriorShiftCheckResult': 'REVIEW',
            'ROIConsistencyCheckResult': 'PASS',
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_revenue_per_kpi_skips_roi_checks',
          revenue_per_kpi=None,
          n_media_channels=1,
          n_rf_channels=0,
          has_inference_data=True,
          has_posterior=True,
          posterior_coords=[constants.MEDIA_CHANNEL],
          check_class=checks.ImplausibleROICheck,
          should_skip=True,
      ),
      dict(
          testcase_name='has_revenue_per_kpi_runs_roi_checks',
          revenue_per_kpi=mock.create_autospec(xr.DataArray, instance=True),
          n_media_channels=1,
          n_rf_channels=0,
          has_inference_data=True,
          has_posterior=True,
          posterior_coords=[constants.MEDIA_CHANNEL],
          check_class=checks.ImplausibleROICheck,
          should_skip=False,
      ),
      dict(
          testcase_name='no_channels_skips_all',
          revenue_per_kpi=mock.create_autospec(xr.DataArray, instance=True),
          n_media_channels=0,
          n_rf_channels=0,
          has_inference_data=True,
          has_posterior=True,
          posterior_coords=[],
          check_class=checks.PotentialBiasCheck,
          should_skip=True,
      ),
      dict(
          testcase_name='no_inference_data_skips_all',
          revenue_per_kpi=mock.create_autospec(xr.DataArray, instance=True),
          n_media_channels=1,
          n_rf_channels=0,
          has_inference_data=False,
          has_posterior=False,
          posterior_coords=[],
          check_class=checks.PotentialBiasCheck,
          should_skip=True,
      ),
      dict(
          testcase_name='no_posterior_skips_all',
          revenue_per_kpi=mock.create_autospec(xr.DataArray, instance=True),
          n_media_channels=1,
          n_rf_channels=0,
          has_inference_data=True,
          has_posterior=False,
          posterior_coords=[],
          check_class=checks.PotentialBiasCheck,
          should_skip=True,
      ),
      dict(
          testcase_name='no_matching_coords_skips_all',
          revenue_per_kpi=mock.create_autospec(xr.DataArray, instance=True),
          n_media_channels=1,
          n_rf_channels=0,
          has_inference_data=True,
          has_posterior=True,
          posterior_coords=['foo'],
          check_class=checks.PotentialBiasCheck,
          should_skip=True,
      ),
  )
  def test_should_skip_calibration_checks(
      self,
      revenue_per_kpi,
      n_media_channels,
      n_rf_channels,
      has_inference_data,
      has_posterior,
      posterior_coords,
      check_class,
      should_skip,
  ):
    self._model_context.input_data.revenue_per_kpi = revenue_per_kpi
    self._model_context.n_media_channels = n_media_channels
    self._model_context.n_rf_channels = n_rf_channels

    if has_inference_data:
      mock_inference_data = mock.create_autospec(
          az.InferenceData, instance=True
      )
      if has_posterior:
        mock_posterior = mock.create_autospec(xr.Dataset, instance=True)
        mock_posterior.coords = posterior_coords
        mock_inference_data.posterior = mock_posterior
      else:
        if hasattr(mock_inference_data, 'posterior'):
          delattr(mock_inference_data, 'posterior')
    else:
      mock_inference_data = None

    # ModelReviewer.__init__ prevents inference_data=None.
    # However, _should_skip_calibration_checks has a check for it.
    # We test the logic by bypassing __init__ validation if necessary,
    # or we can test it on the object after initialization if we can modify it.
    # Since we want to test _should_skip_calibration_checks directly:
    if not has_inference_data:
      # Bypass init validation for this specific unit test case
      with mock.patch.object(
          reviewer.ModelReviewer, '__init__', return_value=None
      ):
        rev = reviewer.ModelReviewer()
        rev._model_context = self._model_context
        rev._inference_data = None
    else:
      rev = reviewer.ModelReviewer(
          model_context=self._model_context,
          inference_data=mock_inference_data,
          post_convergence_checks=immutabledict.immutabledict({
              check_class: configs.BaseConfig(),
          }),
      )

    self.assertEqual(
        rev._should_skip_calibration_checks(check_class), should_skip
    )


class CalibrationRecommendationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='all_uncalibrated',
          media_calibration=[False, False],
          rf_calibration=[False],
          expected_status={
              'ch1': False,
              'ch2': False,
              'rf1': False,
          },
      ),
      dict(
          testcase_name='all_calibrated',
          media_calibration=[True, True],
          rf_calibration=[True],
          expected_status={
              'ch1': True,
              'ch2': True,
              'rf1': True,
          },
      ),
      dict(
          testcase_name='mixed',
          media_calibration=[True, False],
          rf_calibration=[False],
          expected_status={
              'ch1': True,
              'ch2': False,
              'rf1': False,
          },
      ),
  )
  def test_get_calibration_status_by_channel(
      self, media_calibration, rf_calibration, expected_status
  ):
    model_context = mock.create_autospec(
        context.ModelContext, spec_set=True, instance=True
    )
    type(model_context).n_media_channels = mock.PropertyMock(
        return_value=len(media_calibration)
    )
    type(model_context).n_rf_channels = mock.PropertyMock(
        return_value=len(rf_calibration)
    )

    input_data_mock = mock.create_autospec(
        input_data.InputData, spec_set=True, instance=True
    )
    input_data_mock.media_channel.values = np.array(
        [f'ch{i+1}' for i in range(len(media_calibration))]
    )
    input_data_mock.rf_channel.values = np.array(
        [f'rf{i+1}' for i in range(len(rf_calibration))]
    )
    model_context.input_data = input_data_mock

    roi_m = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=media_calibration,
    )

    roi_rf = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=rf_calibration,
    )

    prior = prior_distribution.PriorDistribution(
        roi_m=roi_m,
        roi_rf=roi_rf,
    )
    model_spec = model_spec_module.ModelSpec(prior=prior)
    model_context.model_spec = model_spec

    rev = reviewer.ModelReviewer(
        model_context=model_context,
        inference_data=mock.create_autospec(
            az.InferenceData, spec_set=True, instance=True
        ),
    )

    status = rev._get_calibration_status_by_channel()
    self.assertEqual(status, expected_status)


class CalibrationRecommendationReviewerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model_context = mock.create_autospec(
        context.ModelContext, spec_set=True, instance=True
    )
    self._input_data = mock.create_autospec(
        input_data.InputData, spec_set=True, instance=True
    )
    self._input_data.get_all_paid_channels.return_value = np.array(
        ['calibrated_channel', 'uncalibrated_channel']
    )
    mock_media_channel = mock.create_autospec(xr.DataArray, instance=True)
    type(mock_media_channel).values = mock.PropertyMock(
        return_value=np.array(['calibrated_channel', 'uncalibrated_channel'])
    )
    type(self._input_data).media_channel = mock.PropertyMock(
        return_value=mock_media_channel
    )
    self._input_data.rf_channel = None
    self._input_data.media_spend = None
    self._input_data.rf_spend = None

    type(self._model_context).input_data = mock.PropertyMock(
        return_value=self._input_data
    )
    type(self._model_context).n_media_channels = mock.PropertyMock(
        return_value=2
    )
    type(self._model_context).n_rf_channels = mock.PropertyMock(return_value=0)
    self._model_spec = mock.create_autospec(
        model_spec_module.ModelSpec, spec_set=False, instance=True
    )
    self._prior = mock.create_autospec(
        prior_distribution.PriorDistribution, spec_set=False, instance=True
    )
    calibrated_output = calibration_base.CalibrationOutput(
        channel_name='calibrated_channel',
        baseline_prior=None,
        intermediary_prior=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
    )
    self._roi_m = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True, False],
        calibration_outputs=[calibrated_output, None],
    )
    self._prior.roi_m = self._roi_m
    self._prior.roi_rf = None
    self._model_spec.prior = self._prior
    self._model_spec.effective_media_prior_type = (
        constants.TREATMENT_PRIOR_TYPE_ROI
    )
    self._model_spec.effective_rf_prior_type = (
        constants.TREATMENT_PRIOR_TYPE_ROI
    )
    type(self._model_context).model_spec = mock.PropertyMock(
        return_value=self._model_spec
    )

    self._inference_data = mock.create_autospec(az.InferenceData, instance=True)
    mock_posterior = mock.create_autospec(xr.Dataset, instance=True)
    mock_posterior.coords = [constants.MEDIA_CHANNEL]
    self._inference_data.posterior = mock_posterior

  def test_calibration_recommendation_execution(self):
    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({
            checks.ImplausibleROICheck: configs.ImplausibleROIConfig(),
        }),
    )

    with mock.patch.object(
        checks.ImplausibleROICheck, 'run', autospec=True
    ) as mock_run:
      mock_run.return_value = results.ImplausibleROICheckResult(
          case=results.ImplausibleROIAggregateCases.PASS,
          channel_results=[],
          high_roi_channels=[],
          low_roi_channels=[],
          aggregate_details={},
      )
      with mock.patch.object(
          checks.ConvergenceCheck, 'run', autospec=True
      ) as mock_conv:
        mock_conv.return_value = results.ConvergenceCheckResult(
            case=results.ConvergenceCases.CONVERGED,
            config=configs.ConvergenceConfig(),
            max_r_hat=1.0,
            max_parameter='mock',
        )
        with mock.patch.object(
            reviewer.ModelReviewer, '_compute_health_score', autospec=True
        ) as mock_health:
          mock_health.return_value = 100.0
          summary = review.run()

    self.assertEqual(summary.calibrated_channel_names, ['calibrated_channel'])
    self.assertEqual(
        summary.channel_calibration_status,
        {
            'calibrated_channel': True,
            'uncalibrated_channel': False,
        },
    )
    self.assertEqual(
        summary.channel_calibration_recommendations,
        [
            {
                review_constants.CHANNEL_NAME: 'calibrated_channel',
                review_constants.IS_CALIBRATED: True,
                review_constants.CALIBRATION_SCORE: (
                    review_constants.CALIBRATED_CHANNEL_SCORE
                ),
            },
            {
                review_constants.CHANNEL_NAME: 'uncalibrated_channel',
                review_constants.IS_CALIBRATED: False,
                review_constants.CALIBRATION_SCORE: (
                    review_constants.CALIBRATED_CHANNEL_SCORE
                ),
                review_constants.HIGH_ROI_STATUS: results.Status.PASS,
                review_constants.LOW_ROI_STATUS: results.Status.PASS,
                review_constants.HIGH_VARIANCE_STATUS: results.Status.PASS,
                review_constants.POTENTIAL_BIAS_STATUS: results.Status.PASS,
            },
        ],
    )

  def test_get_calibrated_channels_with_experiments_rf(self):
    # Setup RF channels
    mock_rf_channel = mock.create_autospec(xr.DataArray, instance=True)
    type(mock_rf_channel).values = mock.PropertyMock(
        return_value=np.array(['calibrated_rf_channel'])
    )
    self._input_data.rf_channel = mock_rf_channel
    type(self._model_context).n_rf_channels = mock.PropertyMock(return_value=1)

    # Setup Calibrated Distribution for RF
    rf_output = calibration_base.CalibrationOutput(
        channel_name='calibrated_rf_channel',
        baseline_prior=None,
        intermediary_prior=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
    )
    mock_roi_rf = calibration_base.CalibratedDistribution(
        distributions=backend.tfd.Normal(
            backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
        ),
        is_calibrated=[True],
        calibration_outputs=[rf_output],
    )
    self._prior.roi_rf = mock_roi_rf

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({}),
    )

    calibrated_channels = review._get_calibrated_channels_with_experiments()
    self.assertIn('calibrated_rf_channel', calibrated_channels)

  def test_get_calibrated_channels_with_experiments_no_prior(self):
    setattr(self._model_spec, constants.PRIOR, None)
    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({}),
    )
    calibrated_channels = review._get_calibrated_channels_with_experiments()
    self.assertEqual(calibrated_channels, [])

  def test_run_skips_roi_checks_with_non_roi_priors(self):
    self._model_spec.effective_media_prior_type = (
        constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    )
    self._model_spec.effective_rf_prior_type = (
        constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
    )
    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({
            checks.ImplausibleROICheck: configs.ImplausibleROIConfig(),
            checks.HighVarianceCheck: configs.HighVarianceConfig(),
        }),
    )
    with mock.patch.object(
        checks.ImplausibleROICheck, 'run', autospec=True
    ) as mock_implausible_run, mock.patch.object(
        checks.HighVarianceCheck, 'run', autospec=True
    ) as mock_high_var_run, mock.patch.object(
        checks.ConvergenceCheck, 'run', autospec=True
    ) as mock_conv, mock.patch.object(
        reviewer.ModelReviewer, '_compute_health_score', autospec=True
    ) as mock_health:
      mock_conv.return_value = results.ConvergenceCheckResult(
          case=results.ConvergenceCases.CONVERGED,
          config=configs.ConvergenceConfig(),
          max_r_hat=1.0,
          max_parameter='mock',
      )
      mock_health.return_value = 100.0
      summary = review.run()

    mock_implausible_run.assert_not_called()
    mock_high_var_run.assert_not_called()
    self.assertLen(summary.results, 1)
    self.assertEqual(summary.results[0], mock_conv.return_value)


class ScoreFunctionsTest(absltest.TestCase):

  def test_get_pps_score_empty_channel_results_returns_100(self):
    mock_result = results.PriorPosteriorShiftCheckResult(
        case=results.PriorPosteriorShiftAggregateCases.PASS,
        channel_results=[],
        no_shift_channels=[],
    )
    self.assertEqual(reviewer._get_pps_score(mock_result), 100.0)

  def test_get_roi_consistency_score_empty_channel_results_returns_100(self):
    mock_result = results.ROIConsistencyCheckResult(
        case=results.ROIConsistencyAggregateCases.PASS,
        channel_results=[],
        aggregate_details={},
    )
    self.assertEqual(reviewer._get_roi_consistency_score(mock_result), 100.0)


class CalibrationOverviewDataTest(CalibrationRecommendationReviewerTest):

  @parameterized.named_parameters(
      dict(
          testcase_name='none_prior',
          modify_setup=lambda self: setattr(
              self._model_spec, constants.PRIOR, None
          ),
      ),
      dict(
          testcase_name='none_posterior',
          modify_setup=lambda self: setattr(
              self._inference_data, constants.POSTERIOR, None
          ),
      ),
      dict(
          testcase_name='none_spend',
          modify_setup=lambda self: setattr(
              self._input_data, 'media_spend', None
          ),
      ),
      dict(
          testcase_name='uncalibrated_distribution',
          modify_setup=lambda self: setattr(self._prior, 'roi_m', None),
      ),
  )
  def test_get_calibration_overview_data_empty(self, modify_setup):
    modify_setup(self)
    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({}),
    )
    self.assertEqual(review._get_calibration_overview_data(), [])

  @parameterized.named_parameters(
      dict(
          testcase_name='media_channel_2d_spend',
          media_channels=['calibrated_channel', 'uncalibrated_channel'],
          media_spend=xr.DataArray(
              [[100.0, 200.0]],
              coords={
                  'time': ['t1'],
                  constants.MEDIA_CHANNEL: [
                      'calibrated_channel',
                      'uncalibrated_channel',
                  ],
              },
              dims=['time', constants.MEDIA_CHANNEL],
          ),
          media_calibrated_indices=[0],
          media_posterior=[[[1.0, 2.0]]],
          rf_channels=None,
          rf_spend=None,
          rf_calibrated_indices=None,
          rf_posterior=None,
          expected_channel_names=['calibrated_channel'],
          expected_spends=[100.0],
          expected_posterior_samples=[[1.0]],
      ),
      dict(
          testcase_name='rf_channel_2d_spend',
          media_channels=None,
          media_spend=None,
          media_calibrated_indices=None,
          media_posterior=None,
          rf_channels=['rf_channel'],
          rf_spend=xr.DataArray(
              [[300.0]],
              coords={
                  'time': ['t1'],
                  constants.RF_CHANNEL: ['rf_channel'],
              },
              dims=['time', constants.RF_CHANNEL],
          ),
          rf_calibrated_indices=[0],
          rf_posterior=[[[1.5]]],
          expected_channel_names=['rf_channel'],
          expected_spends=[300.0],
          expected_posterior_samples=[[1.5]],
      ),
      dict(
          testcase_name='sorting_by_spend_1d_spend',
          media_channels=['channel_0', 'channel_1'],
          media_spend=xr.DataArray(
              [100.0, 500.0],
              coords={constants.MEDIA_CHANNEL: ['channel_0', 'channel_1']},
              dims=[constants.MEDIA_CHANNEL],
          ),
          media_calibrated_indices=[0, 1],
          media_posterior=[[[1.0, 2.0]]],
          rf_channels=None,
          rf_spend=None,
          rf_calibrated_indices=None,
          rf_posterior=None,
          expected_channel_names=['channel_1', 'channel_0'],
          expected_spends=[500.0, 100.0],
          expected_posterior_samples=[[2.0], [1.0]],
      ),
      dict(
          testcase_name='geo_time_3d_spend',
          media_channels=['channel_0'],
          media_spend=xr.DataArray(
              [[[10.0]], [[20.0]]],
              coords={
                  'geo': ['geo1', 'geo2'],
                  'time': ['time1'],
                  constants.MEDIA_CHANNEL: ['channel_0'],
              },
              dims=['geo', 'time', constants.MEDIA_CHANNEL],
          ),
          media_calibrated_indices=[0],
          media_posterior=[[[1.0]]],
          rf_channels=None,
          rf_spend=None,
          rf_calibrated_indices=None,
          rf_posterior=None,
          expected_channel_names=['channel_0'],
          expected_spends=[30.0],
          expected_posterior_samples=[[1.0]],
      ),
      dict(
          testcase_name='both_media_and_rf_sorted',
          media_channels=['media_ch'],
          media_spend=xr.DataArray(
              [100.0],
              coords={constants.MEDIA_CHANNEL: ['media_ch']},
              dims=[constants.MEDIA_CHANNEL],
          ),
          media_calibrated_indices=[0],
          media_posterior=[[[1.0]]],
          rf_channels=['rf_ch'],
          rf_spend=xr.DataArray(
              [400.0],
              coords={constants.RF_CHANNEL: ['rf_ch']},
              dims=[constants.RF_CHANNEL],
          ),
          rf_calibrated_indices=[0],
          rf_posterior=[[[2.0]]],
          expected_channel_names=['rf_ch', 'media_ch'],
          expected_spends=[400.0, 100.0],
          expected_posterior_samples=[[2.0], [1.0]],
      ),
  )
  def test_get_calibration_overview_data(
      self,
      media_channels,
      media_spend,
      media_calibrated_indices,
      media_posterior,
      rf_channels,
      rf_spend,
      rf_calibrated_indices,
      rf_posterior,
      expected_channel_names,
      expected_spends,
      expected_posterior_samples,
  ):
    expected_outputs = {}
    expected_dists = {}
    posterior_dict = {}

    for is_rf, channels, spend, cal_indices, post_vals in [
        (
            False,
            media_channels,
            media_spend,
            media_calibrated_indices,
            media_posterior,
        ),
        (
            True,
            rf_channels,
            rf_spend,
            rf_calibrated_indices,
            rf_posterior,
        ),
    ]:
      coord_name = constants.RF_CHANNEL if is_rf else constants.MEDIA_CHANNEL
      param_name = constants.ROI_RF if is_rf else constants.ROI_M
      if channels is None:
        if is_rf:
          self._input_data.rf_channel = None
          self._input_data.rf_spend = None
          self._prior.roi_rf = None
        else:
          self._input_data.media_channel = None
          self._input_data.media_spend = None
          self._prior.roi_m = None
        continue

      coord_da = xr.DataArray(
          channels, coords={coord_name: channels}, dims=[coord_name]
      )
      if is_rf:
        type(self._input_data).rf_channel = mock.PropertyMock(
            return_value=coord_da
        )
        self._input_data.rf_spend = spend
      else:
        type(self._input_data).media_channel = mock.PropertyMock(
            return_value=coord_da
        )
        self._input_data.media_spend = spend

      outputs = []
      dists = []
      for idx, ch in enumerate(channels):
        if idx in (cal_indices or []):
          prior_dist = backend.tfd.Normal(
              backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
          )
          mock_out = calibration_base.CalibrationOutput(
              channel_name=ch,
              baseline_prior=None,
              intermediary_prior=prior_dist,
              experiments=[
                  calibration_base.CalibratedExperiment(
                      source_type=calibration_base.SourceType.MERIDIAN_GEOX,
                      raw_experiment_result=calibration_base.ExperimentResult(
                          point_estimate=1.0, standard_error=0.2
                      ),
                      adjusted_experiment_result=(
                          calibration_base.ExperimentResult(
                              point_estimate=1.0, standard_error=0.2
                          )
                      ),
                      tau_spend=0.0,
                      tau_recency=0.0,
                      tau_duration=0.0,
                      gamma_duration=1.0,
                  )
              ],
          )
          channel_dist = backend.tfd.Normal(
              backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
          )
          outputs.append(mock_out)
          dists.append(channel_dist)
          expected_outputs[ch] = mock_out
        else:
          dummy_dist = backend.tfd.Normal(
              backend.np_float_dtype(0.0), backend.np_float_dtype(1.0)
          )
          outputs.append(None)
          dists.append(dummy_dist)

      is_calibrated = [
          idx in (cal_indices or []) for idx in range(len(channels))
      ]
      mock_roi_dist = calibration_base.CalibratedDistribution(
          distributions=dists,
          is_calibrated=is_calibrated,
          calibration_outputs=outputs,
      )
      for idx, ch in enumerate(channels):
        if idx in (cal_indices or []):
          expected_dists[ch] = mock_roi_dist.distributions[idx]

      param_name = constants.ROI_RF if is_rf else constants.ROI_M
      setattr(self._prior, param_name, mock_roi_dist)

      posterior_dims = ['chain', 'draw', coord_name][: np.ndim(post_vals)]
      posterior_dict[param_name] = xr.DataArray(
          post_vals,
          coords={coord_name: channels},
          dims=posterior_dims,
      )

    mock_posterior = mock.create_autospec(
        xr.Dataset, instance=True, spec_set=True
    )
    mock_posterior.__getitem__.side_effect = lambda key: posterior_dict[key]
    type(self._inference_data).posterior = mock.PropertyMock(
        return_value=mock_posterior
    )

    review = reviewer.ModelReviewer(
        model_context=self._model_context,
        inference_data=self._inference_data,
        post_convergence_checks=immutabledict.immutabledict({}),
    )
    data = review._get_calibration_overview_data()

    self.assertIsInstance(data, list)
    self.assertLen(data, len(expected_channel_names))
    for item, exp_name, exp_spend, exp_post in zip(
        data,
        expected_channel_names,
        expected_spends,
        expected_posterior_samples,
    ):
      self.assertEqual(item.channel_name, exp_name)
      self.assertEqual(item.spend, exp_spend)
      self.assertEqual(item.calibrated_output, expected_outputs[exp_name])
      self.assertEqual(item.calibrated_prior_dist, expected_dists[exp_name])
      self.assertIsNotNone(item.chart_json)
      self.assertIn(
          review_constants.CALIBRATION_LEFT_PLOT_TITLE, item.chart_json
      )
      np.testing.assert_array_equal(item.posterior_samples, exp_post)

if __name__ == '__main__':
  absltest.main()

