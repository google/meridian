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

from collections.abc import Callable, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis.review import checks
from meridian.analysis.review import configs
from meridian.analysis.review import results
from meridian.model import model
import numpy as np
import xarray as xr


class ConvergenceCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)

  @parameterized.named_parameters(
      dict(
          testcase_name="not_converged_high_rhat",
          rhat_mock_value=11.0,
          expected_case=results.ConvergenceCases.NOT_CONVERGED,
      ),
      dict(
          testcase_name="needs_review_medium_rhat",
          rhat_mock_value=9.0,
          expected_case=results.ConvergenceCases.NOT_FULLY_CONVERGED,
      ),
      dict(
          testcase_name="converged_low_rhat",
          rhat_mock_value=1.1,
          expected_case=results.ConvergenceCases.CONVERGED,
      ),
  )
  def test_convergence_check(
      self,
      rhat_mock_value: float,
      expected_case: results.ConvergenceCases,
  ):
    self.analyzer.get_rhat.return_value = {
        "mock_var": np.array([rhat_mock_value])
    }

    config = configs.ConvergenceConfig()
    convergence_check = checks.ConvergenceCheck(
        meridian=self.meridian,
        analyzer=self.analyzer,
        config=config,
    )
    result = convergence_check.run()
    self.assertEqual(result.case, expected_case)

    if result.case == results.ConvergenceCases.CONVERGED:
      self.assertEqual(
          result.recommendation,
          "The model has likely converged, as all parameters have R-hat values"
          " < 1.2.",
      )
    elif result.case == results.ConvergenceCases.NOT_FULLY_CONVERGED:
      self.assertEqual(
          result.recommendation,
          "The model hasn't fully converged, and the `max_r_hat` for parameter"
          " `mock_var` is 9.00. "
          + results.NOT_FULLY_CONVERGED_RECOMMENDATION,
      )
    elif result.case == results.ConvergenceCases.NOT_CONVERGED:
      self.assertEqual(
          result.recommendation,
          "The model hasn't converged, and the `max_r_hat` for parameter"
          " `mock_var` is 11.00. "
          + results.NOT_CONVERGED_RECOMMENDATION,
      )

  def test_convergence_check_with_nan_rhats(self):
    self.analyzer.get_rhat.return_value = {
        "mock_var": np.array([np.nan, np.nan])
    }

    config = configs.ConvergenceConfig()
    convergence_check = checks.ConvergenceCheck(
        meridian=self.meridian,
        analyzer=self.analyzer,
        config=config,
    )
    result = convergence_check.run()
    self.assertEqual(result.case, results.ConvergenceCases.CONVERGED)
    self.assertTrue(np.isnan(result.details[results.constants.RHAT]))
    self.assertTrue(np.isnan(result.details[results.constants.PARAMETER]))
    self.assertEqual(
        result.details[results.constants.CONVERGENCE_THRESHOLD],
        config.convergence_threshold,
    )
    self.assertEqual(
        result.recommendation,
        "The model has likely converged, as all parameters have R-hat values"
        " < 1.2.",
    )


class ROIConsistencyCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)
    self.config = configs.ROIConsistencyConfig(
        prior_lower_quantile=0.01, prior_upper_quantile=0.99
    )

  def _get_quantile_side_effect(
      self, num_channels: int, return_scalar: bool = False
  ) -> Callable[..., np.ndarray]:
    """Returns a side effect function for mocking quantile calculations."""

    def side_effect(q):
      if q == 0.01:
        return 1.0 if return_scalar else np.full((num_channels,), 1.0)
      elif q == 0.99:
        return 10.0 if return_scalar else np.full((num_channels,), 10.0)
      else:
        raise ValueError(f"Unexpected quantile: {q}")

    return side_effect

  def _run_roi_consistency_check(
      self,
      media_channel_names: Sequence[str] | None = None,
      posterior_means: Sequence[float] | None = None,
      rf_channel_names: Sequence[str] | None = None,
      rf_posterior_means: Sequence[float] | None = None,
  ) -> results.ROIConsistencyCheckResult:
    """Runs the ROI consistency check with mocked channel data.

    Args:
      media_channel_names: A sequence of media channel names.
      posterior_means: A sequence of posterior means corresponding to the media
        channels.
      rf_channel_names: A sequence of RF channel names.
      rf_posterior_means: A sequence of posterior means corresponding to the RF
        channels.

    Returns:
      The `ROIConsistencyCheckResult` object from running the check.
    """
    coords = []
    if media_channel_names:
      self.meridian.inference_data.posterior.media_channel.values = (
          media_channel_names
      )
      self.meridian.inference_data.posterior.roi_m = np.array(
          posterior_means, dtype=float
      )[np.newaxis, np.newaxis, :]
      coords.append(constants.MEDIA_CHANNEL)

    if rf_channel_names:
      self.meridian.inference_data.posterior.rf_channel.values = (
          rf_channel_names
      )
      self.meridian.inference_data.posterior.roi_rf = np.array(
          rf_posterior_means, dtype=float
      )[np.newaxis, np.newaxis, :]
      coords.append(constants.RF_CHANNEL)

    self.meridian.inference_data.posterior.coords = coords

    check = checks.ROIConsistencyCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=self.config
    )
    return check.run()

  @parameterized.named_parameters(
      dict(
          testcase_name="all_pass",
          media_channel_names=["ch1"],
          posterior_means=[5.0],
          rf_channel_names=["rf1"],
          rf_posterior_means=[6.0],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.PASS,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_PASS,
              results.ROIConsistencyChannelCases.ROI_PASS,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": "",
          },
      ),
      dict(
          testcase_name="high_low_pass",
          media_channel_names=["ch1", "ch2"],
          posterior_means=[10.1, 5.0],
          rf_channel_names=["rf1"],
          rf_posterior_means=[0.9],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.REVIEW,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_HIGH,
              results.ROIConsistencyChannelCases.ROI_PASS,
              results.ROIConsistencyChannelCases.ROI_LOW,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": (
                  "We've detected an unusually low ROI estimate (for channel"
                  " `rf1`) and an unusually high ROI estimate (for channel"
                  " `ch1`) where the posterior point estimate falls into the"
                  " extreme tail of your custom prior."
              ),
          },
      ),
      dict(
          testcase_name="high_low",
          media_channel_names=["ch1"],
          posterior_means=[10.1],
          rf_channel_names=["rf1", "rf2"],
          rf_posterior_means=[0.9, 0.8],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.REVIEW,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_HIGH,
              results.ROIConsistencyChannelCases.ROI_LOW,
              results.ROIConsistencyChannelCases.ROI_LOW,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": (
                  "We've detected an unusually low ROI estimate (for channels"
                  " `rf1`, `rf2`) and an unusually high ROI estimate (for"
                  " channel `ch1`) where the posterior point estimate falls"
                  " into the extreme tail of your custom prior."
              ),
          },
      ),
      dict(
          testcase_name="only_high_media",
          media_channel_names=["ch1", "ch2"],
          posterior_means=[10.1, 11.1],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.REVIEW,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_HIGH,
              results.ROIConsistencyChannelCases.ROI_HIGH,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": (
                  "We've detected an unusually high ROI estimate (for channels"
                  " `ch1`, `ch2`) where the posterior point estimate falls into"
                  " the extreme tail of your custom prior."
              ),
          },
      ),
      dict(
          testcase_name="low_pass_rf",
          rf_channel_names=["rf1", "rf2"],
          rf_posterior_means=[0.9, 5.0],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.REVIEW,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_LOW,
              results.ROIConsistencyChannelCases.ROI_PASS,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": (
                  "We've detected an unusually low ROI estimate (for channel"
                  " `rf1`) where the posterior point estimate falls into the"
                  " extreme tail of your custom prior."
              ),
          },
      ),
      dict(
          testcase_name="all_pass_global_prior",
          media_channel_names=["ch1"],
          posterior_means=[5.0],
          rf_channel_names=["rf1"],
          rf_posterior_means=[6.0],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.PASS,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_PASS,
              results.ROIConsistencyChannelCases.ROI_PASS,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": "",
          },
          global_prior=True,
      ),
      dict(
          testcase_name="high_low_pass_global_prior",
          media_channel_names=["ch1", "ch2"],
          posterior_means=[10.1, 5.0],
          rf_channel_names=["rf1"],
          rf_posterior_means=[0.9],
          expected_aggregate_case=results.ROIConsistencyAggregateCases.REVIEW,
          expected_channel_cases=[
              results.ROIConsistencyChannelCases.ROI_HIGH,
              results.ROIConsistencyChannelCases.ROI_PASS,
              results.ROIConsistencyChannelCases.ROI_LOW,
          ],
          expected_details={
              "quantile_not_defined_msg": "",
              "inf_channels_msg": "",
              "low_high_channels_msg": (
                  "We've detected an unusually low ROI estimate (for channel"
                  " `rf1`) and an unusually high ROI estimate (for channel"
                  " `ch1`) where the posterior point estimate falls into the"
                  " extreme tail of your custom prior."
              ),
          },
          global_prior=True,
      ),
  )
  def test_roi_consistency_check(
      self,
      expected_aggregate_case,
      expected_channel_cases,
      expected_details,
      media_channel_names=None,
      posterior_means=None,
      rf_channel_names=None,
      rf_posterior_means=None,
      global_prior=False,
  ):
    if media_channel_names:
      self.meridian.model_spec.prior.roi_m.quantile.side_effect = (
          self._get_quantile_side_effect(
              len(media_channel_names), return_scalar=global_prior
          )
      )
    if rf_channel_names:
      self.meridian.model_spec.prior.roi_rf.quantile.side_effect = (
          self._get_quantile_side_effect(
              len(rf_channel_names), return_scalar=global_prior
          )
      )

    all_channels = []
    if media_channel_names:
      all_channels.extend(media_channel_names)
    if rf_channel_names:
      all_channels.extend(rf_channel_names)
    result = self._run_roi_consistency_check(
        media_channel_names=media_channel_names,
        posterior_means=posterior_means,
        rf_channel_names=rf_channel_names,
        rf_posterior_means=rf_posterior_means,
    )
    self.assertEqual(result.case, expected_aggregate_case)
    self.assertEqual(result.details, expected_details)
    self.assertEqual(
        [res.case for res in result.channel_results], expected_channel_cases
    )
    self.assertEqual(
        [res.channel_name for res in result.channel_results], all_channels
    )

  def test_roi_consistency_check_infinite_roi_prior(self):
    media_channel_names = ["ch1"]
    posterior_means = [5.0]
    rf_channel_names = ["rf1"]
    rf_posterior_means = [6.0]

    def get_m_quantile_side_effect(q, **kwargs):
      del kwargs
      if q == 0.01:
        return np.array([-np.inf])
      elif q == 0.99:
        return np.array([10.0])
      else:
        raise ValueError(f"Unexpected quantile: {q}")

    def get_rf_quantile_side_effect(q, **kwargs):
      del kwargs
      if q == 0.01:
        return np.array([1.0])
      elif q == 0.99:
        return np.array([np.inf])
      else:
        raise ValueError(f"Unexpected quantile: {q}")

    self.meridian.model_spec.prior.roi_m.quantile.side_effect = (
        get_m_quantile_side_effect
    )
    self.meridian.model_spec.prior.roi_rf.quantile.side_effect = (
        get_rf_quantile_side_effect
    )
    result = self._run_roi_consistency_check(
        media_channel_names=media_channel_names,
        posterior_means=posterior_means,
        rf_channel_names=rf_channel_names,
        rf_posterior_means=rf_posterior_means,
    )
    all_channels = media_channel_names + rf_channel_names
    self.assertEqual(result.case, results.ROIConsistencyAggregateCases.REVIEW)
    expected_details = {
        "quantile_not_defined_msg": "",
        "inf_channels_msg": (
            "Prior ROI quantiles are infinite for channels: ch1, rf1"
        ),
        "low_high_channels_msg": "",
    }
    self.assertEqual(result.details, expected_details)
    self.assertEqual(
        [res.case for res in result.channel_results],
        [
            results.ROIConsistencyChannelCases.PRIOR_ROI_QUANTILE_INF,
            results.ROIConsistencyChannelCases.PRIOR_ROI_QUANTILE_INF,
        ],
    )
    self.assertEqual(
        [res.channel_name for res in result.channel_results], all_channels
    )

  def test_roi_consistency_check_quantile_not_defined_rf_channel(self):
    media_channel_names = ["ch1"]
    posterior_means = [5.0]
    rf_channel_names = ["rf1"]
    rf_posterior_means = [6.0]

    self.meridian.model_spec.prior.roi_m.quantile.side_effect = (
        self._get_quantile_side_effect(1)
    )
    self.meridian.model_spec.prior.roi_rf = backend.tfd.Deterministic(loc=6.0)
    result = self._run_roi_consistency_check(
        media_channel_names=media_channel_names,
        posterior_means=posterior_means,
        rf_channel_names=rf_channel_names,
        rf_posterior_means=rf_posterior_means,
    )
    param = backend.tfd.Deterministic(loc=6.0)

    self.assertEqual(result.case, results.ROIConsistencyAggregateCases.REVIEW)
    expected_details = {
        "quantile_not_defined_msg": (
            "The quantile method is not defined for the following parameters:"
            f" {[param]}. The ROI Consistency check cannot be performed for"
            " these parameters."
        ),
        "inf_channels_msg": "",
        "low_high_channels_msg": "",
    }
    self.assertEqual(result.details, expected_details)
    self.assertEqual(
        [res.case for res in result.channel_results],
        [
            results.ROIConsistencyChannelCases.QUANTILE_NOT_DEFINED,
            results.ROIConsistencyChannelCases.ROI_PASS,
        ],
    )
    self.assertEqual(
        [res.channel_name for res in result.channel_results], ["rf1", "ch1"]
    )

  def test_roi_consistency_check_quantile_not_defined_all_channels(self):
    media_channel_names = ["ch1"]
    posterior_means = [5.0]
    rf_channel_names = ["rf1"]
    rf_posterior_means = [6.0]

    self.meridian.model_spec.prior.roi_m = backend.tfd.Deterministic(loc=5.0)
    self.meridian.model_spec.prior.roi_rf = backend.tfd.Deterministic(loc=6.0)
    all_channels = media_channel_names + rf_channel_names
    result = self._run_roi_consistency_check(
        media_channel_names=media_channel_names,
        posterior_means=posterior_means,
        rf_channel_names=rf_channel_names,
        rf_posterior_means=rf_posterior_means,
    )
    params = [
        backend.tfd.Deterministic(loc=5.0),
        backend.tfd.Deterministic(loc=6.0),
    ]
    self.assertEqual(result.case, results.ROIConsistencyAggregateCases.REVIEW)
    expected_details = {
        "quantile_not_defined_msg": (
            "The quantile method is not defined for the following parameters:"
            f" {params}. The ROI Consistency check cannot be performed for"
            " these parameters."
        ),
        "inf_channels_msg": "",
        "low_high_channels_msg": "",
    }
    self.assertEqual(result.details, expected_details)
    self.assertEqual(
        [res.case for res in result.channel_results],
        [
            results.ROIConsistencyChannelCases.QUANTILE_NOT_DEFINED,
            results.ROIConsistencyChannelCases.QUANTILE_NOT_DEFINED,
        ],
    )
    self.assertEqual(
        [res.channel_name for res in result.channel_results], all_channels
    )


class PriorPosteriorShiftCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)
    self.config = configs.PriorPosteriorShiftConfig(
        n_bootstraps=100, alpha=0.05, seed=0
    )

  def _run_prior_posterior_shift_check(
      self,
      media_channel_names: Sequence[str] | None = None,
      posterior_media_samples: np.ndarray | None = None,
      rf_channel_names: Sequence[str] | None = None,
      posterior_rf_samples: np.ndarray | None = None,
      quantile_not_defined: bool = False,
  ) -> results.PriorPosteriorShiftCheckResult:
    """Runs the PriorPosteriorShiftCheck with mocked data."""
    posterior_vars = {}
    posterior_coords = {}
    if media_channel_names is not None and posterior_media_samples is not None:
      n_channels = len(media_channel_names)
      n_chains, n_draws, _ = posterior_media_samples.shape
      posterior_coords.update({
          constants.CHAIN: range(n_chains),
          constants.DRAW: range(n_draws),
          constants.MEDIA_CHANNEL: media_channel_names,
      })
      self.meridian.inference_data.posterior.media_channel.values = (
          media_channel_names
      )
      posterior_vars[constants.ROI_M] = mock.Mock(
          values=posterior_media_samples
      )

      dist_m = mock.MagicMock()
      dist_m.mean.return_value = np.zeros(n_channels)

      if quantile_not_defined:
        dist_m.quantile.side_effect = NotImplementedError
      else:

        def quantile_m(q):
          if q == 0.5:
            return np.zeros(n_channels)
          elif q == 0.25:
            return np.full(n_channels, -0.67448975)
          elif q == 0.75:
            return np.full(n_channels, 0.67448975)

        dist_m.quantile.side_effect = quantile_m
      self.meridian.model_spec.prior.roi_m = dist_m

    if rf_channel_names is not None and posterior_rf_samples is not None:
      n_channels = len(rf_channel_names)
      n_chains, n_draws, _ = posterior_rf_samples.shape
      posterior_coords.update({
          constants.CHAIN: range(n_chains),
          constants.DRAW: range(n_draws),
          constants.RF_CHANNEL: rf_channel_names,
      })
      self.meridian.inference_data.posterior.rf_channel.values = (
          rf_channel_names
      )
      posterior_vars[constants.ROI_RF] = mock.Mock(values=posterior_rf_samples)

      dist_rf = mock.MagicMock()
      dist_rf.mean.return_value = np.zeros(n_channels)

      if quantile_not_defined:
        dist_rf.quantile.side_effect = NotImplementedError
      else:

        def quantile_rf(q):
          if q == 0.5:
            return np.zeros(n_channels)
          elif q == 0.25:
            return np.full(n_channels, -0.67448975)
          elif q == 0.75:
            return np.full(n_channels, 0.67448975)

        dist_rf.quantile.side_effect = quantile_rf
      self.meridian.model_spec.prior.roi_rf = dist_rf

    def getitem_side_effect(key):
      return getattr(self.meridian.inference_data.posterior, key)

    self.meridian.inference_data.posterior.__getitem__.side_effect = (
        getitem_side_effect
    )
    self.meridian.inference_data.posterior.variables = posterior_vars
    self.meridian.inference_data.posterior.coords = posterior_coords

    check = checks.PriorPosteriorShiftCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=self.config
    )
    return check.run()

  @parameterized.named_parameters(
      dict(
          testcase_name="all_shifted_media",
          media_channel_names=["ch1", "ch2"],
          posterior_medians=[5.0, 5.0],
          expected_aggregate_case=(
              results.PriorPosteriorShiftAggregateCases.PASS
          ),
          expected_channel_cases=[
              results.PriorPosteriorShiftChannelCases.SHIFT,
              results.PriorPosteriorShiftChannelCases.SHIFT,
          ],
          expected_details={},
      ),
      dict(
          testcase_name="one_not_shifted_media",
          media_channel_names=["ch1", "ch2"],
          posterior_medians=[5.0, 0.0],
          expected_aggregate_case=(
              results.PriorPosteriorShiftAggregateCases.REVIEW
          ),
          expected_channel_cases=[
              results.PriorPosteriorShiftChannelCases.SHIFT,
              results.PriorPosteriorShiftChannelCases.NO_SHIFT,
          ],
          expected_details={"channels_str": "`ch2`"},
      ),
      dict(
          testcase_name="one_shifted_one_not_rf",
          rf_channel_names=["rf1", "rf2"],
          posterior_medians=[5.0, 0.0],
          expected_aggregate_case=(
              results.PriorPosteriorShiftAggregateCases.REVIEW
          ),
          expected_channel_cases=[
              results.PriorPosteriorShiftChannelCases.SHIFT,
              results.PriorPosteriorShiftChannelCases.NO_SHIFT,
          ],
          expected_details={"channels_str": "`rf2`"},
      ),
      dict(
          testcase_name="mixed_channels_one_not_shifted",
          media_channel_names=["ch1"],
          posterior_medians_media=[5.0],
          rf_channel_names=["rf1"],
          posterior_medians_rf=[0.0],
          expected_aggregate_case=(
              results.PriorPosteriorShiftAggregateCases.REVIEW
          ),
          expected_channel_cases=[
              results.PriorPosteriorShiftChannelCases.SHIFT,
              results.PriorPosteriorShiftChannelCases.NO_SHIFT,
          ],
          expected_details={"channels_str": "`rf1`"},
      ),
  )
  def test_prior_posterior_shift_check(
      self,
      expected_aggregate_case,
      expected_channel_cases,
      expected_details,
      media_channel_names=None,
      posterior_medians=None,
      posterior_medians_media=None,
      rf_channel_names=None,
      posterior_medians_rf=None,
  ):
    np.random.seed(0)
    prior_samples = np.random.normal(0, 1, size=(1, 1000, 1))
    post_samples_shifted = np.random.normal(5.0, 1, size=(1, 100, 1))
    post_samples_not_shifted = prior_samples[:, :100, :]

    post_media = None
    if media_channel_names:
      if posterior_medians_media is None:
        posterior_medians_media = posterior_medians
      post_media_parts = []
      for median in posterior_medians_media:
        if median == 0.0:
          post_media_parts.append(post_samples_not_shifted)
        else:
          post_media_parts.append(post_samples_shifted)
      post_media = np.concatenate(post_media_parts, axis=2)

    post_rf = None
    if rf_channel_names:
      if posterior_medians_rf is None:
        posterior_medians_rf = posterior_medians
      post_rf_parts = []
      for median in posterior_medians_rf:
        if median == 0.0:
          post_rf_parts.append(post_samples_not_shifted)
        else:
          post_rf_parts.append(post_samples_shifted)
      post_rf = np.concatenate(post_rf_parts, axis=2)

    result = self._run_prior_posterior_shift_check(
        media_channel_names=media_channel_names,
        posterior_media_samples=post_media,
        rf_channel_names=rf_channel_names,
        posterior_rf_samples=post_rf,
    )
    all_channels = []
    if media_channel_names:
      all_channels.extend(media_channel_names)
    if rf_channel_names:
      all_channels.extend(rf_channel_names)

    self.assertEqual(result.case, expected_aggregate_case)
    self.assertEqual(result.details, expected_details)
    self.assertLen(result.channel_results, len(expected_channel_cases))
    self.assertEqual(
        [res.case for res in result.channel_results], expected_channel_cases
    )
    self.assertEqual(
        [res.channel_name for res in result.channel_results], all_channels
    )

  def test_prior_posterior_shift_check_quantile_not_defined(self):
    np.random.seed(0)
    post_samples_shifted = np.random.normal(5.0, 1, size=(1, 100, 1))

    result = self._run_prior_posterior_shift_check(
        media_channel_names=["ch1"],
        posterior_media_samples=post_samples_shifted,
        rf_channel_names=None,
        posterior_rf_samples=None,
        quantile_not_defined=True,
    )
    # With quantile throwing NotImplementedError, we only check for shift in
    # MEAN.
    # The posterior is N(5,1) and prior is N(0,1), so mean is different and
    # shift should be detected.
    self.assertEqual(
        result.case, results.PriorPosteriorShiftAggregateCases.PASS
    )
    self.assertEqual(result.details, {})
    self.assertLen(result.channel_results, 1)
    self.assertEqual(
        result.channel_results[0].case,
        results.PriorPosteriorShiftChannelCases.SHIFT,
    )
    self.assertEqual(result.channel_results[0].channel_name, "ch1")


class BaselineCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)
    self.config = configs.BaselineConfig(
        negative_baseline_prob_review_threshold=0.2,
        negative_baseline_prob_fail_threshold=0.8,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          prob=0.1,
          expected_case=results.BaselineCases.PASS,
      ),
      dict(
          testcase_name="review",
          prob=0.5,
          expected_case=results.BaselineCases.REVIEW,
      ),
      dict(
          testcase_name="fail",
          prob=0.9,
          expected_case=results.BaselineCases.FAIL,
      ),
  )
  def test_baseline_check(
      self, prob: float, expected_case: results.BaselineCases
  ):
    self.analyzer.negative_baseline_probability.return_value = prob
    check = checks.BaselineCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=self.config
    )
    result = check.run()
    self.assertEqual(result.case, expected_case)
    if expected_case == results.BaselineCases.PASS:
      self.assertEqual(
          result.recommendation,
          "The posterior probability that the baseline is negative is 0.10. "
          + results._BASELINE_PASS_RECOMMENDATION,
      )
    elif expected_case == results.BaselineCases.REVIEW:
      self.assertEqual(
          result.recommendation,
          "The posterior probability that the baseline is negative is 0.50. "
          + results._BASELINE_REVIEW_RECOMMENDATION,
      )
    else:
      self.assertEqual(
          result.recommendation,
          "The posterior probability that the baseline is negative is 0.90. "
          + results._BASELINE_FAIL_RECOMMENDATION,
      )


class BayesianPPPCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)
    self.config = configs.BayesianPPPConfig(ppp_threshold=0.05)

  @parameterized.named_parameters(
      dict(
          testcase_name="pass",
          kpi=np.array([10, 20]),
          revenue_per_kpi=None,
          expected_outcome=np.array([25, 35]),
          expected_case=results.BayesianPPPCases.PASS,
          expected_ppp=1.0,
      ),
      dict(
          testcase_name="fail",
          kpi=np.array([10, 30]),
          revenue_per_kpi=None,
          expected_outcome=np.array([29, 31]),
          expected_case=results.BayesianPPPCases.FAIL,
          expected_ppp=0.0,
      ),
      dict(
          testcase_name="pass_with_revenue_per_kpi",
          kpi=np.array([5.0, 10.0]),
          revenue_per_kpi=2.0,
          expected_outcome=np.array([25, 35]),
          expected_case=results.BayesianPPPCases.PASS,
          expected_ppp=1.0,
      ),
  )
  def test_bayesian_ppp_check(
      self, kpi, revenue_per_kpi, expected_outcome, expected_case, expected_ppp
  ):
    self.meridian.kpi = kpi
    self.meridian.revenue_per_kpi = revenue_per_kpi
    self.analyzer.expected_outcome.return_value = expected_outcome

    check = checks.BayesianPPPCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=self.config
    )
    result = check.run()

    self.assertEqual(result.case, expected_case)
    self.assertAlmostEqual(
        result.details[results.constants.BAYESIAN_PPP], expected_ppp
    )


class GoodnessOfFitCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.meridian = mock.MagicMock(spec=model.Meridian)
    self.analyzer = mock.MagicMock(spec=analyzer_module.Analyzer)

  def _get_gof_dataset(
      self,
      r_squared: float,
      mape: float,
      wmape: float,
      is_national: bool = False,
  ) -> xr.Dataset:
    dims = (
        constants.METRIC,
        constants.EVALUATION_SET_VAR,
        constants.GEO_GRANULARITY,
    )
    if is_national:
      data = np.array([
          [[1.0, r_squared], [1.0, 1.0], [1.0, 0.1]],
          [[1.0, mape], [1.0, 1.0], [1.0, 0.2]],
          [[1.0, wmape], [1.0, 1.0], [1.0, 0.3]],
      ])
    else:
      data = np.array([
          [[r_squared, 1.0], [1.0, 1.0], [0.1, 1.0]],
          [[mape, 1.0], [1.0, 1.0], [0.2, 1.0]],
          [[wmape, 1.0], [1.0, 1.0], [0.3, 1.0]],
      ])
    coords = {
        constants.METRIC: [
            constants.R_SQUARED,
            constants.MAPE,
            constants.WMAPE,
        ],
        constants.GEO_GRANULARITY: [constants.GEO, constants.NATIONAL],
        constants.EVALUATION_SET_VAR: [
            constants.ALL_DATA,
            constants.TRAIN,
            constants.TEST,
        ],
    }
    return xr.Dataset(
        data_vars={constants.VALUE: (dims, data)},
        coords=coords,
    )

  def _get_gof_dataset_no_holdout(
      self,
      r_squared: float,
      mape: float,
      wmape: float,
      is_national: bool = False,
  ) -> xr.Dataset:
    dims = (
        constants.METRIC,
        constants.GEO_GRANULARITY,
    )
    if is_national:
      data = np.array([
          [1.0, r_squared],
          [1.0, mape],
          [1.0, wmape],
      ])
    else:
      data = np.array([
          [r_squared, 1.0],
          [mape, 1.0],
          [wmape, 1.0],
      ])
    coords = {
        constants.METRIC: [
            constants.R_SQUARED,
            constants.MAPE,
            constants.WMAPE,
        ],
        constants.GEO_GRANULARITY: [constants.GEO, constants.NATIONAL],
    }
    return xr.Dataset(
        data_vars={constants.VALUE: (dims, data)},
        coords=coords,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="pass_geo",
          r_squared=0.5,
          mape=0.1,
          wmape=0.1,
          is_national=False,
          expected_case=results.GoodnessOfFitCases.PASS,
      ),
      dict(
          testcase_name="review_zero_geo",
          r_squared=0.0,
          mape=0.1,
          wmape=0.1,
          is_national=False,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
      dict(
          testcase_name="review_negative_geo",
          r_squared=-0.1,
          mape=0.1,
          wmape=0.1,
          is_national=False,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
      dict(
          testcase_name="pass_national",
          r_squared=0.5,
          mape=0.1,
          wmape=0.1,
          is_national=True,
          expected_case=results.GoodnessOfFitCases.PASS,
      ),
      dict(
          testcase_name="review_zero_national",
          r_squared=0.0,
          mape=0.1,
          wmape=0.1,
          is_national=True,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
      dict(
          testcase_name="review_negative_national",
          r_squared=-0.1,
          mape=0.1,
          wmape=0.1,
          is_national=True,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
  )
  def test_goodness_of_fit_check(
      self, r_squared, mape, wmape, is_national, expected_case
  ):
    self.meridian.n_geos = 1 if is_national else 2
    gof_dataset = self._get_gof_dataset(r_squared, mape, wmape, is_national)
    self.analyzer.predictive_accuracy.return_value = gof_dataset
    config = configs.GoodnessOfFitConfig()
    gof_check = checks.GoodnessOfFitCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=config
    )
    result = gof_check.run()
    self.assertEqual(result.case, expected_case)
    self.assertIn(f"R-squared = {r_squared:.4f}", result.recommendation)
    self.assertIn(f"MAPE = {mape:.4f}", result.recommendation)
    self.assertIn(f"wMAPE = {wmape:.4f}", result.recommendation)
    if expected_case == results.GoodnessOfFitCases.PASS:
      self.assertIn(
          results._GOODNESS_OF_FIT_PASS_RECOMMENDATION, result.recommendation
      )
    else:
      self.assertIn(
          results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION, result.recommendation
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="pass_geo_no_holdout",
          r_squared=0.5,
          mape=0.1,
          wmape=0.1,
          is_national=False,
          expected_case=results.GoodnessOfFitCases.PASS,
      ),
      dict(
          testcase_name="review_negative_geo_no_holdout",
          r_squared=-0.1,
          mape=0.1,
          wmape=0.1,
          is_national=False,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
      dict(
          testcase_name="pass_national_no_holdout",
          r_squared=0.5,
          mape=0.1,
          wmape=0.1,
          is_national=True,
          expected_case=results.GoodnessOfFitCases.PASS,
      ),
      dict(
          testcase_name="review_negative_national_no_holdout",
          r_squared=-0.1,
          mape=0.1,
          wmape=0.1,
          is_national=True,
          expected_case=results.GoodnessOfFitCases.REVIEW,
      ),
  )
  def test_goodness_of_fit_check_no_holdout(
      self, r_squared, mape, wmape, is_national, expected_case
  ):
    self.meridian.n_geos = 1 if is_national else 2
    gof_dataset = self._get_gof_dataset_no_holdout(
        r_squared, mape, wmape, is_national
    )
    self.analyzer.predictive_accuracy.return_value = gof_dataset
    config = configs.GoodnessOfFitConfig()
    gof_check = checks.GoodnessOfFitCheck(
        meridian=self.meridian, analyzer=self.analyzer, config=config
    )
    result = gof_check.run()
    self.assertEqual(result.case, expected_case)
    self.assertIn(f"R-squared = {r_squared:.4f}", result.recommendation)
    self.assertIn(f"MAPE = {mape:.4f}", result.recommendation)
    self.assertIn(f"wMAPE = {wmape:.4f}", result.recommendation)
    if expected_case == results.GoodnessOfFitCases.PASS:
      self.assertEndsWith(
          result.recommendation, results._GOODNESS_OF_FIT_PASS_RECOMMENDATION
      )
    else:
      self.assertIn(
          results._GOODNESS_OF_FIT_REVIEW_RECOMMENDATION, result.recommendation
      )


if __name__ == "__main__":
  absltest.main()
