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
import immutabledict
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import results
from meridian.analysis.review import reviewer


class ReviewerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(analyzer_module, 'Analyzer', autospec=True)
    )
    self._meridian = mock.MagicMock()
    type(self._meridian).n_media_channels = mock.PropertyMock(return_value=1)
    type(self._meridian).n_rf_channels = mock.PropertyMock(return_value=1)
    type(self._meridian).is_roi_prior = mock.PropertyMock(return_value=True)
    type(self._meridian).is_custom_roi_prior = mock.PropertyMock(
        return_value=True
    )
    self.enter_context(
        mock.patch.object(
            reviewer.ModelReviewer,
            '_uses_roi_priors',
            side_effect=lambda: self._meridian.is_roi_prior,
        )
    )
    self.enter_context(
        mock.patch.object(
            reviewer.ModelReviewer,
            '_has_custom_roi_priors',
            side_effect=lambda: self._meridian.is_custom_roi_prior,
        )
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
          expected_score=74.4,
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
        meridian=self._meridian,
    )
    summary = review.run()

    self.assertAlmostEqual(summary.health_score, expected_score, places=1)

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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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

  def test_run_skip_checks_with_custom_roi_priors(self):
    type(self._meridian).is_roi_prior = mock.PropertyMock(return_value=False)
    type(self._meridian).is_custom_roi_prior = mock.PropertyMock(
        return_value=False
    )
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS

    review = reviewer.ModelReviewer(
        meridian=self._meridian,
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
    type(self._meridian).is_roi_prior = mock.PropertyMock(return_value=False)
    type(self._meridian).is_custom_roi_prior = mock.PropertyMock(
        return_value=False
    )
    self._mock_convergence_result.case = results.ConvergenceCases.CONVERGED
    self._mock_baseline_result.case = results.BaselineCases.PASS
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS

    review = reviewer.ModelReviewer(
        meridian=self._meridian,
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
        meridian=self._meridian,
    )
    review.run()

    self._mock_convergence_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.ConvergenceConfig()
    )
    self._mock_baseline_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.BaselineConfig()
    )
    self._mock_bayesian_ppp_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.BayesianPPPConfig()
    )
    self._mock_roi_consistency_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.ROIConsistencyConfig()
    )
    self._mock_gof_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.GoodnessOfFitConfig()
    )
    self._mock_pps_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.PriorPosteriorShiftConfig()
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
      review = reviewer.ModelReviewer(meridian=self._meridian)
      with self.assertRaisesRegex(
          ValueError,
          r'The following required checks results are missing: '
          r"\['BayesianPPPCheck', 'GoodnessOfFitCheck'\].",
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
        meridian=self._meridian,
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
        meridian=self._meridian,
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


if __name__ == '__main__':
  absltest.main()
