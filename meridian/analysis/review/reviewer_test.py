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
import immutabledict
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import results
from meridian.analysis.review import reviewer


class ReviewerTest(absltest.TestCase):

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
    self._mock_convergence_result = mock.Mock(
        spec=results.ConvergenceCheckResult, case=mock.PropertyMock()
    )
    self._mock_convergence_result.__class__ = results.ConvergenceCheckResult
    self._mock_convergence_check.run.return_value = (
        self._mock_convergence_result
    )

    roi_consistency_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.ROIConsistencyCheck'
    )
    self._mock_roi_consistency_check_cls = self.enter_context(
        roi_consistency_check_cls_patcher
    )
    self._mock_roi_consistency_check = (
        self._mock_roi_consistency_check_cls.return_value
    )
    self._mock_roi_consistency_result = mock.Mock(
        spec=results.ROIConsistencyCheckResult, case=mock.PropertyMock()
    )
    self._mock_roi_consistency_result.__class__ = (
        results.ROIConsistencyCheckResult
    )
    self._mock_roi_consistency_check.run.return_value = (
        self._mock_roi_consistency_result
    )

    baseline_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.BaselineCheck'
    )
    self._mock_baseline_check_cls = self.enter_context(
        baseline_check_cls_patcher
    )
    self._mock_baseline_check = self._mock_baseline_check_cls.return_value
    self._mock_baseline_result = mock.Mock(
        spec=results.BaselineCheckResult, case=mock.PropertyMock()
    )
    self._mock_baseline_result.__class__ = results.BaselineCheckResult
    self._mock_baseline_check.run.return_value = self._mock_baseline_result

    bayesian_ppp_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.BayesianPPPCheck'
    )
    self._mock_bayesian_ppp_check_cls = self.enter_context(
        bayesian_ppp_check_cls_patcher
    )
    self._mock_bayesian_ppp_check = (
        self._mock_bayesian_ppp_check_cls.return_value
    )
    self._mock_bayesian_ppp_result = mock.Mock(
        spec=results.BayesianPPPCheckResult, case=mock.PropertyMock()
    )
    self._mock_bayesian_ppp_result.__class__ = results.BayesianPPPCheckResult
    self._mock_bayesian_ppp_check.run.return_value = (
        self._mock_bayesian_ppp_result
    )

    gof_check_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.GoodnessOfFitCheck'
    )
    self._mock_gof_check_cls = self.enter_context(gof_check_cls_patcher)
    self._mock_gof_check = self._mock_gof_check_cls.return_value
    self._mock_gof_result = mock.Mock(
        spec=results.GoodnessOfFitCheckResult, case=mock.PropertyMock()
    )
    self._mock_gof_result.__class__ = results.GoodnessOfFitCheckResult
    self._mock_gof_check.run.return_value = self._mock_gof_result

    prior_posterior_shift_cls_patcher = mock.patch(
        'meridian.analysis.review.checks.PriorPosteriorShiftCheck'
    )
    self._mock_pps_check_cls = self.enter_context(
        prior_posterior_shift_cls_patcher
    )
    self._mock_pps_check = self._mock_pps_check_cls.return_value
    self._mock_pps_result = mock.Mock(
        spec=results.PriorPosteriorShiftCheckResult, case=mock.PropertyMock()
    )
    self._mock_pps_result.__class__ = results.PriorPosteriorShiftCheckResult
    self._mock_pps_check.run.return_value = self._mock_pps_result

    self._default_post_convergence_checks = immutabledict.immutabledict({
        self._mock_baseline_check_cls: configs.BaselineConfig(),
        self._mock_bayesian_ppp_check_cls: configs.BayesianPPPConfig(),
        self._mock_gof_check_cls: configs.GoodnessOfFitConfig(),
        self._mock_pps_check_cls: configs.PriorPosteriorShiftConfig(),
        self._mock_roi_consistency_check_cls: configs.ROIConsistencyConfig(),
    })

  def test_run_all_pass(self):
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
        post_convergence_checks=self._default_post_convergence_checks,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.PASS)
    self.assertEqual(
        summary.summary_message,
        'Passed: No major quality issues were identified.',
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
    )
    summary = review.run()

    self.assertEqual(summary.overall_status, results.Status.FAIL)
    self.assertEqual(
        summary.summary_message,
        'Failed: Model did not converge. Other checks were skipped.',
    )
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
    self._mock_bayesian_ppp_result.case = results.BayesianPPPCases.PASS
    self._mock_roi_consistency_result.case = (
        results.ROIConsistencyAggregateCases.PASS
    )
    self._mock_gof_result.case = results.GoodnessOfFitCases.PASS
    self._mock_pps_result.case = results.PriorPosteriorShiftAggregateCases.PASS

    review = reviewer.ModelReviewer(
        meridian=self._meridian,
        post_convergence_checks=self._default_post_convergence_checks,
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

  def test_run_with_custom_configs(self):
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
    review = reviewer.ModelReviewer(
        meridian=self._meridian, post_convergence_checks=custom_checks
    )
    summary = review.run()

    self._mock_convergence_check_cls.assert_called_once_with(
        mock.ANY, mock.ANY, configs.ConvergenceConfig()
    )
    self._mock_baseline_check_cls.assert_called_once_with(
        mock.ANY,
        mock.ANY,
        configs.BaselineConfig(
            negative_baseline_prob_review_threshold=0.5,
            negative_baseline_prob_fail_threshold=0.9,
        ),
    )
    self._mock_roi_consistency_check_cls.assert_called_once_with(
        mock.ANY,
        mock.ANY,
        configs.ROIConsistencyConfig(
            prior_lower_quantile=0.05,
            prior_upper_quantile=0.95,
        ),
    )
    self._mock_gof_check_cls.assert_not_called()
    self._mock_bayesian_ppp_check_cls.assert_not_called()
    self._mock_pps_check_cls.assert_not_called()
    self.assertLen(summary.results, 3)
    self.assertEqual(summary.results[0], self._mock_convergence_result)
    self.assertEqual(summary.results[1], self._mock_baseline_result)
    self.assertEqual(summary.results[2], self._mock_roi_consistency_result)

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
        post_convergence_checks=self._default_post_convergence_checks,
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
        post_convergence_checks=self._default_post_convergence_checks,
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
