# Copyright 2024 The Meridian Authors.
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
import os
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "test_data"
)
_TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH = os.path.join(
    _TEST_DIR,
    "sample_prior_media_and_rf.nc",
)
_TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH = os.path.join(
    _TEST_DIR,
    "sample_prior_media_only.nc",
)
_TEST_SAMPLE_PRIOR_RF_ONLY_PATH = os.path.join(
    _TEST_DIR,
    "sample_prior_rf_only.nc",
)
_TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH = os.path.join(
    _TEST_DIR,
    "sample_posterior_media_and_rf.nc",
)
_TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH = os.path.join(
    _TEST_DIR,
    "sample_posterior_media_only.nc",
)
_TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH = os.path.join(
    _TEST_DIR,
    "sample_posterior_rf_only.nc",
)
_TEST_SAMPLE_PRIOR_NON_PAID_PATH = os.path.join(
    _TEST_DIR,
    "sample_prior_non_paid.nc",
)
_TEST_SAMPLE_POSTERIOR_NON_PAID_PATH = os.path.join(
    _TEST_DIR,
    "sample_posterior_non_paid.nc",
)
_TEST_SAMPLE_PRIOR_NATIONAL_PATH = os.path.join(
    _TEST_DIR,
    "sample_prior_national.nc",
)
_TEST_SAMPLE_POSTERIOR_NATIONAL_PATH = os.path.join(
    _TEST_DIR,
    "sample_posterior_national.nc",
)
_TEST_SAMPLE_TRACE_PATH = os.path.join(
    _TEST_DIR,
    "sample_trace.nc",
)

# Data dimensions for sample input.
_N_CHAINS = 2
_N_KEEP = 10
_N_DRAWS = 10
_N_GEOS = 5
_N_TIMES = 49
_N_MEDIA_TIMES = 52
_N_CONTROLS = 2
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2
_N_NON_MEDIA_CHANNELS = 4
_N_ORGANIC_MEDIA_CHANNELS = 4
_N_ORGANIC_RF_CHANNELS = 1


def _convert_with_swap(array: xr.DataArray) -> tf.Tensor:
  """Converts DataArray to tf.Tensor and swaps first two dimensions."""
  tensor = tf.convert_to_tensor(array)
  perm = [1, 0] + [i for i in range(2, len(tensor.shape))]
  return tf.transpose(tensor, perm=perm)


def _build_inference_data(
    prior_path: str, posterior_path: str
) -> az.InferenceData:
  inference_data = az.InferenceData(
      prior=xr.open_dataset(prior_path),
      posterior=xr.open_dataset(posterior_path),
  )
  inference_data.groups = lambda: [
      constants.PRIOR,
      constants.POSTERIOR,
  ]
  return inference_data


class AnalyzerTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerTest, cls).setUpClass()
    # Input data resulting in revenue computation.
    cls.input_data_media_and_rf = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    cls.input_data_national = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=1,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_media_and_rf = model.Meridian(
        input_data=cls.input_data_media_and_rf, model_spec=model_spec
    )
    cls.analyzer_media_and_rf = analyzer.Analyzer(cls.meridian_media_and_rf)

    cls.inference_data_media_and_rf = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH,
    )
    cls.inference_data_national = _build_inference_data(
        _TEST_SAMPLE_PRIOR_NATIONAL_PATH,
        _TEST_SAMPLE_POSTERIOR_NATIONAL_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_media_and_rf),
        )
    )

  def test_get_central_tendency_and_ci(self):
    data = np.array([[[10.0, 7, 4], [3, 2, 1]], [[1, 2, 3], [4, 5, 6.0]]])
    result = analyzer.get_central_tendency_and_ci(data, confidence_level=0.9)
    np.testing.assert_allclose(
        result,
        np.array([[4.5, 1.3, 9.1], [4.0, 2.0, 6.7], [3.5, 1.3, 5.7]]),
        atol=0.1,
    )

  def test_get_central_tendency_and_ci_returns_median(self):
    data = np.array([[[10.0, 7, 4], [3, 2, 1]], [[1, 2, 3], [4, 5, 6.0]]])
    result = analyzer.get_central_tendency_and_ci(
        data, confidence_level=0.9, include_median=True
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [[4.5, 3.5, 1.3, 9.1], [4.0, 3.5, 2.0, 6.7], [3.5, 3.5, 1.3, 5.7]]
        ),
        atol=0.1,
    )

  def test_expected_outcome_new_revenue_per_kpi_raises_warning(self):
    with warnings.catch_warnings(record=True) as w:
      self.analyzer_media_and_rf.expected_outcome(
          new_data=analyzer.DataTensors(
              revenue_per_kpi=self.meridian_media_and_rf.revenue_per_kpi
          ),
      )

      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "A `revenue_per_kpi` value was passed in the `new_data` argument to"
          " the `expected_outcome()` method. This is currently not supported"
          " and will be ignored.",
          str(w[0].message),
      )

  def test_expected_outcome_wrong_controls_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_controls.shape must match controls.shape",
    ):
      self.analyzer_media_and_rf.expected_outcome(
          new_data=analyzer.DataTensors(
              controls=self.meridian_media_and_rf.population
          ),
      )

  def test_expected_outcome_wrong_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_media.shape must match media.shape",
    ):
      self.analyzer_media_and_rf.expected_outcome(
          new_data=analyzer.DataTensors(
              media=self.meridian_media_and_rf.population
          ),
      )

  def test_expected_outcome_wrong_reach_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_reach.shape must match reach.shape",
    ):
      self.analyzer_media_and_rf.expected_outcome(
          new_data=analyzer.DataTensors(
              reach=self.meridian_media_and_rf.population
          ),
      )

  def test_expected_outcome_wrong_frequency_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_frequency.shape must match frequency.shape",
    ):
      self.analyzer_media_and_rf.expected_outcome(
          new_data=analyzer.DataTensors(
              frequency=self.meridian_media_and_rf.population
          ),
      )

  def test_expected_outcome_wrong_kpi_transformation(self):
    with self.assertRaisesRegex(
        ValueError,
        "use_kpi=False is only supported when inverse_transform_outcome=True.",
    ):
      self.analyzer_media_and_rf.expected_outcome(
          inverse_transform_outcome=False, use_kpi=False
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_outcome_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    outcome = self.analyzer_media_and_rf.expected_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=geos_to_include,
        selected_times=times_to_include,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(geos_to_include),) if geos_to_include is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(times_to_include),)
          if times_to_include is not None
          else (_N_TIMES,)
      )
    self.assertEqual(outcome.shape, expected_shape)

  def test_incremental_outcome_new_controls_raises_warning(self):
    with warnings.catch_warnings(record=True) as w:
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(
              controls=self.meridian_media_and_rf.controls
          ),
      )

      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "A `controls` value was passed in the `new_data` argument to the"
          " `incremental_outcome()` method. This has no effect on the output"
          " and will be ignored.",
          str(w[0].message),
      )

  @parameterized.named_parameters(
      (
          "wrong_media_dims",
          analyzer.DataTensors(media=tf.ones((5, 1))),
          "New media params must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_reach_dims",
          analyzer.DataTensors(reach=tf.ones((5, 1))),
          "New media params must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_frequency_dims",
          analyzer.DataTensors(frequency=tf.ones((5, 1))),
          "New media params must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_revenue_per_kpi_dims",
          analyzer.DataTensors(revenue_per_kpi=tf.ones((5))),
          "new_revenue_per_kpi must have 2 dimension(s). Found 1 dimension(s).",
      ),
  )
  def test_incremental_outcome_wrong_media_param_dims_raises_exception(
      self,
      new_param: analyzer.DataTensors,
      expected_error_message: str,
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_message):
      self.analyzer_media_and_rf.incremental_outcome(new_data=new_param)

  @parameterized.named_parameters(
      ("missing_media", "media"),
      ("missing_reach", "reach"),
      ("missing_frequency", "frequency"),
      ("missing_revenue_per_kpi", "revenue_per_kpi"),
  )
  def test_incremental_outcome_missing_new_param_raises_exception(
      self, missing_param: str
  ):
    new_data_dict = {
        "media": tf.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        "reach": tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        "frequency": tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        "revenue_per_kpi": tf.ones((_N_GEOS, 10)),
    }
    with self.assertRaisesRegex(
        ValueError,
        "If new_media, new_reach, new_frequency, new_organic_media,"
        " new_organic_reach, new_organic_frequency, or new_revenue_per_kpi is"
        " provided with a different number of time periods than in `InputData`,"
        " then all new parameters originally in `InputData` must be provided"
        " with the same number of time periods.",
    ):
      new_data_dict.pop(missing_param)
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(**new_data_dict)
      )

  def test_incremental_outcome_negative_scaling_factor0(self):
    with self.assertRaisesRegex(
        ValueError,
        "scaling_factor0 must be non-negative.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(scaling_factor0=-0.01)

  def test_incremental_outcome_negative_scaling_factor1(self):
    with self.assertRaisesRegex(
        ValueError, "scaling_factor1 must be non-negative."
    ):
      self.analyzer_media_and_rf.incremental_outcome(scaling_factor1=-0.01)

  def test_incremental_outcome_scaling_factor1_less_than_scaling_factor0(self):
    with self.assertRaisesRegex(
        ValueError,
        "scaling_factor1 must be greater than scaling_factor0. Got"
        " scaling_factor1=1.0 and scaling_factor0=1.1.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          scaling_factor0=1.1, scaling_factor1=1.0
      )

  def test_incremental_outcome_flexible_times_selected_times_wrong_type(self):
    with self.assertRaisesRegex(
        ValueError,
        "If new_media, new_reach, new_frequency, new_organic_media,"
        " new_organic_reach, new_organic_frequency, or new_revenue_per_kpi is"
        " provided with a different number of time periods than in `InputData`,"
        " then `selected_times` and `media_selected_times` must be a list of"
        " booleans with length equal to the number of time periods in the new"
        " data.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(
              media=tf.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
              reach=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              frequency=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              revenue_per_kpi=tf.ones((_N_GEOS, 10)),
          ),
          selected_times=["2021-04-19", "2021-09-13", "2021-12-13"],
      )

  def test_incremental_outcome_flexible_times_media_selected_times_wrong_type(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "If new_media, new_reach, new_frequency, new_organic_media,"
        " new_organic_reach, new_organic_frequency, or new_revenue_per_kpi is"
        " provided with a different number of time periods than in `InputData`,"
        " then `selected_times` and `media_selected_times` must be a list of"
        " booleans with length equal to the number of time periods in the new"
        " data.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(
              media=tf.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
              reach=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              frequency=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              revenue_per_kpi=tf.ones((_N_GEOS, 10)),
          ),
          media_selected_times=["2021-04-19", "2021-09-13", "2021-12-13"],
      )

  def test_incremental_outcome_media_selected_times_wrong_length(self):
    with self.assertRaisesRegex(
        ValueError,
        "Boolean `media_selected_times` must have the same number of elements "
        "as there are time period coordinates in the media tensors.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          media_selected_times=[False] * (_N_MEDIA_TIMES - 10) + [True],
      )

  def test_incremental_outcome_media_selected_times_wrong_time_dim_names(self):
    with self.assertRaisesRegex(
        ValueError,
        "`media_selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          media_selected_times=["random_time"],
      )

  def test_incremental_outcome_incorrect_media_selected_times_type(self):
    with self.assertRaisesRegex(
        ValueError,
        "`media_selected_times` must be a list of strings or a list of"
        " booleans.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          media_selected_times=["random_time", False, True],
      )

  def test_incremental_outcome_new_params_diff_time_dims_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_revenue_per_kpi must have the same number of time periods as the "
        "other media tensor arguments.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(
              media=tf.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
              reach=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              frequency=tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              revenue_per_kpi=tf.ones((_N_GEOS, 8)),
          )
      )

  def test_incremental_outcome_wrong_new_param_n_geos_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_media is expected to have 5 geos. Found 4 geos.",
    ):
      shape = (_N_GEOS - 1, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS)
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(media=tf.ones(shape))
      )

  def test_incremental_outcome_wrong_n_channels_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_reach is expected to have third dimension of size 2. Actual size "
        "is 1.",
    ):
      shape = (_N_GEOS, _N_MEDIA_TIMES, _N_RF_CHANNELS - 1)
      self.analyzer_media_and_rf.incremental_outcome(
          new_data=analyzer.DataTensors(reach=tf.ones(shape))
      )

  def test_incremental_outcome_wrong_kpi_transformation(self):
    with self.assertRaisesRegex(
        ValueError,
        "use_kpi=False is only supported when inverse_transform_outcome=True.",
    ):
      self.analyzer_media_and_rf.incremental_outcome(
          inverse_transform_outcome=False, use_kpi=False
      )

  def test_incremental_outcome_new_revenue_per_kpi_correct_shape(self):
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        new_data=analyzer.DataTensors(
            revenue_per_kpi=tf.ones((_N_GEOS, _N_TIMES))
        ),
    )
    self.assertEqual(
        outcome.shape, (_N_CHAINS, _N_KEEP, _N_MEDIA_CHANNELS + _N_RF_CHANNELS)
    )

  def test_incremental_outcome_media_selected_times_all_false_returns_zero(
      self,
  ):
    no_media_times = self.analyzer_media_and_rf.incremental_outcome(
        media_selected_times=[False] * _N_MEDIA_TIMES
    )
    self.assertAllEqual(no_media_times, tf.zeros_like(no_media_times))

  def test_incremental_outcome_no_overlap_between_media_and_selected_times(
      self,
  ):
    # If for any time period where media_selected_times is True, selected_times
    # is False for this time period and the following `max_lag` time periods,
    # then the incremental outcome should be zero.
    max_lag = self.meridian_media_and_rf.model_spec.max_lag
    media_selected_times = [
        self.meridian_media_and_rf.input_data.media_time.values[0]
    ]
    selected_times = [False] * (max_lag + 1) + [True] * (_N_TIMES - max_lag - 1)
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        selected_times=selected_times,
        media_selected_times=media_selected_times,
    )
    self.assertAllEqual(outcome, tf.zeros_like(outcome))

  def test_incremental_outcome_media_and_selected_times_overlap_non_zero(self):
    # Incremental outcome should be non-zero when there is at least one time
    # period of overlap between media_selected_times and selected_times. In this
    # case, media_selected_times is True for week 1 and selected_times is True
    # for week `max_lag+1` and the following weeks.
    max_lag = self.meridian_media_and_rf.model_spec.max_lag
    excess_times = _N_MEDIA_TIMES - _N_TIMES
    media_selected_times = [True] + [False] * (_N_MEDIA_TIMES - 1)
    selected_times = [False] * (max_lag - excess_times) + [True] * (
        _N_TIMES - max_lag + excess_times
    )
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        selected_times=selected_times,
        media_selected_times=media_selected_times,
    )
    mean_inc_outcome = tf.reduce_mean(outcome, axis=(0, 1))
    self.assertNotAllEqual(mean_inc_outcome, tf.zeros_like(mean_inc_outcome))

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[
          None,
          ["2021-04-19", "2021-09-13", "2021-12-13"],
          [False] * (_N_TIMES - 3) + [True] * 3,
      ],
  )
  def test_incremental_outcome_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      if selected_times is not None:
        if all(isinstance(time, bool) for time in selected_times):
          n_times = sum(selected_times)
        else:
          n_times = len(selected_times)
      else:
        n_times = _N_TIMES
      expected_shape += (n_times,)
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(outcome.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_outcome=test_utils.INC_OUTCOME_MEDIA_AND_RF_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=test_utils.INC_OUTCOME_MEDIA_AND_RF_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_media_and_rf(
      self,
      use_posterior: bool,
      expected_outcome: np.ndarray,
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(expected_outcome),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_compute_incremental_outcome_aggregate_media_and_rf(self):
    mock_incremental_outcome = np.ones(
        (_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS + _N_RF_CHANNELS)
    )
    self.enter_context(
        mock.patch.object(
            self.analyzer_media_and_rf,
            "incremental_outcome",
            return_value=mock_incremental_outcome,
        )
    )
    incremental_outcome_with_totals = np.full(
        (_N_CHAINS, _N_DRAWS, 1), _N_MEDIA_CHANNELS + _N_RF_CHANNELS
    )
    outcome = self.analyzer_media_and_rf.compute_incremental_outcome_aggregate(
        use_posterior=True
    )
    self.assertAllClose(
        outcome,
        tf.concat(
            [mock_incremental_outcome, incremental_outcome_with_totals], -1
        ),
    )

  # The purpose of this test is to prevent accidental logic change.
  def test_incremental_outcome_media_and_rf_new_params(self):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    outcome = self.analyzer_media_and_rf.incremental_outcome(
        new_data=analyzer.DataTensors(
            media=self.meridian_media_and_rf.media_tensors.media[..., -10:, :],
            reach=self.meridian_media_and_rf.rf_tensors.reach[..., -10:, :],
            frequency=self.meridian_media_and_rf.rf_tensors.frequency[
                ..., -10:, :
            ],
            revenue_per_kpi=self.meridian_media_and_rf.revenue_per_kpi[
                ..., -10:
            ],
        ),
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(test_utils.INC_OUTCOME_MEDIA_AND_RF_NEW_PARAMS),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
  ):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    mroi = self.analyzer_media_and_rf.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(mroi.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          by_reach=False,
          expected_mroi=test_utils.MROI_MEDIA_AND_RF_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=test_utils.MROI_MEDIA_AND_RF_USE_PRIOR_BY_REACH,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=test_utils.MROI_MEDIA_AND_RF_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=test_utils.MROI_MEDIA_AND_RF_USE_POSTERIOR_BY_REACH,
      ),
  )
  def test_marginal_roi_media_and_rf(
      self,
      use_posterior: bool,
      by_reach: bool,
      expected_mroi: tuple[float, ...],
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    mroi = self.analyzer_media_and_rf.marginal_roi(
        by_reach=by_reach,
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        mroi,
        tf.convert_to_tensor(expected_mroi),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_roi_wrong_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_data.media.shape must match media.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_data=analyzer.DataTensors(
              media=self.meridian_media_and_rf.population
          ),
      )

  def test_roi_wrong_media_spend_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "new_data.media_spend.shape: (5,) must match either (3,) or (5,"
        " 49, 3).",
    ):
      self.analyzer_media_and_rf.roi(
          new_data=analyzer.DataTensors(
              media_spend=self.meridian_media_and_rf.population
          ),
      )

  def test_roi_wrong_reach_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_data.reach.shape must match reach.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_data=analyzer.DataTensors(
              reach=self.meridian_media_and_rf.population
          ),
      )

  def test_roi_wrong_frequency_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_data.frequency.shape must match frequency.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_data=analyzer.DataTensors(
              frequency=self.meridian_media_and_rf.population
          ),
      )

  def test_roi_wrong_rf_spend_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "new_data.rf_spend.shape: (5,) must match either (2,) or (5, 49, 2).",
    ):
      self.analyzer_media_and_rf.roi(
          new_data=analyzer.DataTensors(
              rf_spend=self.meridian_media_and_rf.population
          ),
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    roi = self.analyzer_media_and_rf.roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(roi.shape, expected_shape)

  def test_roi_media_and_rf_default_returns_correct_value(self):
    roi = self.analyzer_media_and_rf.roi()
    total_spend = (
        self.analyzer_media_and_rf.filter_and_aggregate_geos_and_times(
            self.meridian_media_and_rf.total_spend
        )
    )
    expected_roi = (
        self.analyzer_media_and_rf.incremental_outcome() / total_spend
    )
    self.assertAllClose(expected_roi, roi)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_cpik_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    cpik = self.analyzer_media_and_rf.cpik(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(cpik.shape, expected_shape)

  def test_cpik_media_and_rf_default_returns_correct_value(self):
    cpik = self.analyzer_media_and_rf.cpik()
    total_spend = (
        self.analyzer_media_and_rf.filter_and_aggregate_geos_and_times(
            self.meridian_media_and_rf.total_spend
        )
    )
    expected_cpik = (
        total_spend
        / self.analyzer_media_and_rf.incremental_outcome(use_kpi=True)
    )
    self.assertAllClose(expected_cpik, cpik)

  def test_hill_histogram_column_names(self):
    hill_histogram_table = (
        self.analyzer_media_and_rf._get_hill_histogram_dataframe(n_bins=25)
    )
    self.assertEqual(
        list(hill_histogram_table.columns),
        [
            constants.CHANNEL,
            constants.CHANNEL_TYPE,
            constants.SCALED_COUNT_HISTOGRAM,
            constants.COUNT_HISTOGRAM,
            constants.START_INTERVAL_HISTOGRAM,
            constants.END_INTERVAL_HISTOGRAM,
        ],
    )

  def test_hill_calculation_dataframe_properties(self):
    hill_table = self.analyzer_media_and_rf.hill_curves()

    self.assertEqual(
        list(hill_table.columns),
        [
            constants.CHANNEL,
            constants.MEDIA_UNITS,
            constants.DISTRIBUTION,
            constants.CI_HI,
            constants.CI_LO,
            constants.MEAN,
            constants.CHANNEL_TYPE,
            constants.SCALED_COUNT_HISTOGRAM,
            constants.COUNT_HISTOGRAM,
            constants.START_INTERVAL_HISTOGRAM,
            constants.END_INTERVAL_HISTOGRAM,
        ],
    )
    self.assertAllInSet(
        list(set(hill_table[constants.CHANNEL])),
        ["ch_0", "ch_2", "ch_1", "rf_ch_0", "rf_ch_1"],
    )
    ci_lo_col = list(hill_table[constants.CI_LO].notna())
    ci_hi_col = list(hill_table[constants.CI_HI].notna())
    mean_col = list(hill_table[constants.MEAN].notna())

    for i, e in enumerate(mean_col):
      ci_lo_val = ci_lo_col[i]
      ci_hi_val = ci_hi_col[i]
      self.assertGreaterEqual(e, ci_lo_val)
      self.assertLessEqual(e, ci_hi_val)

  def test_hill_calculation_curve_data_correct(self):
    hill_table = self.analyzer_media_and_rf.hill_curves()
    self.assertAllClose(
        list(hill_table[constants.CI_HI])[:5],
        np.array(test_utils.HILL_CURVES_CI_HI),
        atol=1e-5,
    )
    self.assertAllClose(
        list(hill_table[constants.CI_LO])[:5],
        np.array(test_utils.HILL_CURVES_CI_LO),
        atol=1e-5,
    )
    self.assertAllClose(
        list(hill_table[constants.MEAN])[:5],
        np.array(test_utils.HILL_CURVES_MEAN),
        atol=1e-5,
    )

  def test_hill_calculation_histogram_data_correct(self):
    hill_table = self.analyzer_media_and_rf.hill_curves()
    # The Histogram data is in the bottom portion of the DataFrame for
    # Altair plotting purposes.
    self.assertAllClose(
        list(hill_table[constants.SCALED_COUNT_HISTOGRAM])[-5:],
        np.array(test_utils.HILL_CURVES_SCALED_COUNT_HISTOGRAM),
        atol=1e-5,
    )
    self.assertAllClose(
        list(hill_table[constants.COUNT_HISTOGRAM])[-5:],
        np.array(test_utils.HILL_CURVES_COUNT_HISTOGRAM),
        atol=1e-5,
    )
    self.assertAllClose(
        list(hill_table[constants.START_INTERVAL_HISTOGRAM])[-5:],
        np.array(test_utils.HILL_CURVES_START_INTERVAL_HISTOGRAM),
        atol=1e-5,
    )
    self.assertAllClose(
        list(hill_table[constants.END_INTERVAL_HISTOGRAM])[-5:],
        np.array(test_utils.HILL_CURVES_END_INTERVAL_HISTOGRAM),
        atol=1e-5,
    )

  def test_media_summary_returns_correct_values(self):
    media_summary = self.analyzer_media_and_rf.summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertEqual(
        list(media_summary.data_vars.keys()),
        [
            constants.IMPRESSIONS,
            constants.PCT_OF_IMPRESSIONS,
            constants.SPEND,
            constants.PCT_OF_SPEND,
            constants.CPM,
            constants.INCREMENTAL_OUTCOME,
            constants.PCT_OF_CONTRIBUTION,
            constants.ROI,
            constants.EFFECTIVENESS,
            constants.MROI,
            constants.CPIK,
        ],
    )
    self.assertAllClose(
        media_summary.impressions, test_utils.SAMPLE_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.pct_of_impressions, test_utils.SAMPLE_PCT_OF_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.spend, test_utils.SAMPLE_SPEND, atol=1e-4, rtol=1e-4
    )
    self.assertAllClose(
        media_summary.pct_of_spend,
        test_utils.SAMPLE_PCT_OF_SPEND,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertAllClose(
        media_summary.cpm,
        test_utils.SAMPLE_CPM,
    )
    self.assertAllClose(
        media_summary.incremental_outcome,
        test_utils.SAMPLE_INCREMENTAL_OUTCOME,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.pct_of_contribution,
        test_utils.SAMPLE_PCT_OF_CONTRIBUTION,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.roi, test_utils.SAMPLE_ROI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.effectiveness,
        test_utils.SAMPLE_EFFECTIVENESS,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertAllClose(
        media_summary.mroi, test_utils.SAMPLE_MROI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.cpik, test_utils.SAMPLE_CPIK, atol=1e-3, rtol=1e-3
    )

  def test_media_summary_with_new_data_returns_correct_values(self):
    data1 = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        seed=1,
    )
    new_data = analyzer.DataTensors(
        media=tf.convert_to_tensor(data1.media, dtype=tf.float32),
        reach=tf.convert_to_tensor(data1.reach, dtype=tf.float32),
        media_spend=tf.convert_to_tensor(data1.media_spend, dtype=tf.float32),
    )
    media_summary = self.analyzer_media_and_rf.summary_metrics(
        new_data=new_data,
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertEqual(
        list(media_summary.data_vars.keys()),
        [
            constants.IMPRESSIONS,
            constants.PCT_OF_IMPRESSIONS,
            constants.SPEND,
            constants.PCT_OF_SPEND,
            constants.CPM,
            constants.INCREMENTAL_OUTCOME,
            constants.PCT_OF_CONTRIBUTION,
            constants.ROI,
            constants.EFFECTIVENESS,
            constants.MROI,
            constants.CPIK,
        ],
    )
    self.assertNotAllClose(
        media_summary.impressions, test_utils.SAMPLE_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.impressions, test_utils.SAMPLE_IMPRESSIONS_NEW_DATA
    )
    self.assertNotAllClose(
        media_summary.pct_of_impressions, test_utils.SAMPLE_PCT_OF_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.pct_of_impressions,
        test_utils.SAMPLE_PCT_OF_IMPRESSIONS_NEW_DATA,
    )
    self.assertNotAllClose(media_summary.spend, test_utils.SAMPLE_SPEND)
    self.assertAllClose(
        media_summary.spend,
        test_utils.SAMPLE_SPEND_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertNotAllClose(
        media_summary.pct_of_spend,
        test_utils.SAMPLE_PCT_OF_SPEND,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertAllClose(
        media_summary.pct_of_spend,
        test_utils.SAMPLE_PCT_OF_SPEND_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertNotAllClose(media_summary.cpm, test_utils.SAMPLE_CPM)
    self.assertAllClose(
        media_summary.cpm,
        test_utils.SAMPLE_CPM_NEW_DATA,
    )
    self.assertNotAllClose(
        media_summary.incremental_outcome,
        test_utils.SAMPLE_INCREMENTAL_OUTCOME,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.incremental_outcome,
        test_utils.SAMPLE_INCREMENTAL_OUTCOME_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertNotAllClose(
        media_summary.pct_of_contribution,
        test_utils.SAMPLE_PCT_OF_CONTRIBUTION,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.pct_of_contribution,
        test_utils.SAMPLE_PCT_OF_CONTRIBUTION_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertNotAllClose(
        media_summary.roi, test_utils.SAMPLE_ROI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.roi, test_utils.SAMPLE_ROI_NEW_DATA, atol=1e-3, rtol=1e-3
    )
    self.assertNotAllClose(
        media_summary.effectiveness,
        test_utils.SAMPLE_EFFECTIVENESS,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertAllClose(
        media_summary.effectiveness,
        test_utils.SAMPLE_EFFECTIVENESS_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertNotAllClose(
        media_summary.mroi, test_utils.SAMPLE_MROI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.mroi,
        test_utils.SAMPLE_MROI_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertNotAllClose(
        media_summary.cpik, test_utils.SAMPLE_CPIK, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.cpik,
        test_utils.SAMPLE_CPIK_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )

  def test_baseline_summary_returns_correct_values(self):
    baseline_summary = self.analyzer_media_and_rf.baseline_summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertIsNotNone(baseline_summary.baseline_outcome)
    self.assertIsNotNone(baseline_summary.pct_of_contribution)
    self.assertAllClose(
        baseline_summary.baseline_outcome,
        test_utils.SAMPLE_BASELINE_EXPECTED_OUTCOME,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        baseline_summary.pct_of_contribution,
        test_utils.SAMPLE_BASELINE_PCT_OF_CONTRIBUTION,
        atol=1e-2,
        rtol=1e-2,
    )

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_media_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    analyzer_ = self.analyzer_media_and_rf
    num_channels = _N_MEDIA_CHANNELS + _N_RF_CHANNELS

    media_summary = analyzer_.summary_metrics(
        confidence_level=0.8,
        marginal_roi_by_reach=False,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_channel_shape = ()
    if not aggregate_geos:
      expected_channel_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_channel_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # (ch_1, ch_2, ..., All_Channels, [mean, median, ci_lo, ci_hi],
    # [prior, posterior])
    expected_channel_shape += (num_channels + 1,)
    expected_shape = expected_channel_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.impressions.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.pct_of_impressions.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.pct_of_spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.cpm.shape, expected_channel_shape)
    self.assertEqual(media_summary.incremental_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)
    if aggregate_times:
      self.assertEqual(media_summary.roi.shape, expected_shape)
      self.assertEqual(media_summary.effectiveness.shape, expected_shape)
      self.assertEqual(media_summary.mroi.shape, expected_shape)
      self.assertEqual(media_summary.cpik.shape, expected_shape)
    else:
      self.assertNotIn(constants.ROI, media_summary.data_vars)
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)
      self.assertNotIn(constants.MROI, media_summary.data_vars)
      self.assertNotIn(constants.CPIK, media_summary.data_vars)

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_baseline_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    analyzer_ = self.analyzer_media_and_rf

    media_summary = analyzer_.baseline_summary_metrics(
        confidence_level=0.8,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_geo_and_time_shape = ()
    if not aggregate_geos:
      expected_geo_and_time_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_geo_and_time_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # ([mean, median, ci_lo, ci_hi], [prior, posterior])
    expected_shape = expected_geo_and_time_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.baseline_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)

  def test_optimal_frequency_data_media_and_rf_correct(self):
    actual = self.analyzer_media_and_rf.optimal_freq(
        freq_grid=[1.0, 2.0, 3.0],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        use_posterior=True,
    )
    expected = xr.Dataset(
        coords={
            constants.FREQUENCY: ([constants.FREQUENCY], [1.0, 2.0, 3.0]),
            constants.RF_CHANNEL: (
                [constants.RF_CHANNEL],
                ["rf_ch_0", "rf_ch_1"],
            ),
            constants.METRIC: (
                [constants.METRIC],
                [
                    constants.MEAN,
                    constants.MEDIAN,
                    constants.CI_LO,
                    constants.CI_HI,
                ],
            ),
        },
        data_vars={
            constants.ROI: (
                [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
                [
                    [
                        [4.57, 4.59, 1.6, 7.52],
                        [6.61, 6.52, 1.69, 11.7],
                    ],  # freq=1.0
                    [
                        [2.48, 2.48, 0.98, 3.97],
                        [3.66, 3.61, 1.1, 6.32],
                    ],  # freq=2.0
                    [
                        [1.73, 1.73, 0.72, 2.73],
                        [2.54, 2.50, 0.83, 4.3],
                    ],  # freq=3.0
                ],
            ),
            constants.OPTIMAL_FREQUENCY: ([constants.RF_CHANNEL], [1.0, 1.0]),
            constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1244.8, 1249.58, 436.19, 2047.89],
                    [1902.85, 1878.92, 487.06, 3368.2],
                ],
            ),
            constants.OPTIMIZED_EFFECTIVENESS: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.001699, 0.001705, 0.000595, 0.002795],
                    [0.00259, 0.002557, 0.000663, 0.004585],
                ],
            ),
            constants.OPTIMIZED_PCT_OF_CONTRIBUTION: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [5.16181, 5.18164, 1.80875, 8.49197],
                    [7.89052, 7.79130, 2.01976, 13.96686],
                ],
            ),
            constants.OPTIMIZED_ROI: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[4.57, 4.59, 1.6, 7.52], [6.61, 6.52, 1.69, 11.7]],
            ),
            constants.OPTIMIZED_MROI_BY_REACH: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[4.57, 4.59, 1.6, 7.52], [6.61, 6.52, 1.69, 11.7]],
            ),
            constants.OPTIMIZED_MROI_BY_FREQUENCY: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[0.54, 0.54, 0.45, 0.62], [1.31, 1.30, 0.69, 1.96]],
            ),
            constants.OPTIMIZED_CPIK: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[0.371, 0.371, 0.132, 0.623], [0.337, 0.337, 0.085, 0.591]],
            ),
        },
        attrs={
            constants.CONFIDENCE_LEVEL: constants.DEFAULT_CONFIDENCE_LEVEL,
            "use_posterior": True,
        },
    )

    xr.testing.assert_allclose(actual, expected, atol=0.1)
    xr.testing.assert_allclose(actual.frequency, expected.frequency)
    xr.testing.assert_allclose(actual.rf_channel, expected.rf_channel)
    xr.testing.assert_allclose(actual.metric, expected.metric)
    xr.testing.assert_allclose(actual.roi, expected.roi, atol=0.1)
    xr.testing.assert_allclose(
        actual.optimal_frequency, expected.optimal_frequency
    )
    xr.testing.assert_allclose(
        actual.optimized_incremental_outcome,
        expected.optimized_incremental_outcome,
        atol=0.1,
    )
    xr.testing.assert_allclose(
        actual.optimized_effectiveness,
        expected.optimized_effectiveness,
        atol=0.00001,
    )
    xr.testing.assert_allclose(
        actual.optimized_pct_of_contribution,
        expected.optimized_pct_of_contribution,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_roi,
        expected.optimized_roi,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_reach,
        expected.optimized_mroi_by_reach,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_frequency,
        expected.optimized_mroi_by_frequency,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_cpik,
        expected.optimized_cpik,
        atol=0.01,
    )
    self.assertEqual(actual.confidence_level, expected.confidence_level)
    self.assertEqual(actual.use_posterior, expected.use_posterior)

  def test_rhat_media_and_rf_correct(self):
    rhat = self.analyzer_media_and_rf.get_rhat()
    self.assertSetEqual(
        set(rhat.keys()),
        set(
            constants.COMMON_PARAMETER_NAMES
            + constants.MEDIA_PARAMETER_NAMES
            + constants.RF_PARAMETER_NAMES
        ),
    )

  def test_rhat_summary_media_and_rf_correct(self):
    rhat_summary = self.analyzer_media_and_rf.rhat_summary()
    self.assertEqual(rhat_summary.shape, (20, 7))
    self.assertSetEqual(
        set(rhat_summary.param),
        set(
            constants.COMMON_PARAMETER_NAMES
            + constants.MEDIA_PARAMETER_NAMES
            + constants.RF_PARAMETER_NAMES
        )
        - set([constants.SLOPE_M]),
    )

  def test_predictive_accuracy_without_holdout_id_columns_correct(self):
    predictive_accuracy_dataset = (
        self.analyzer_media_and_rf.predictive_accuracy()
    )

    self.assertListEqual(
        list(predictive_accuracy_dataset[constants.METRIC].values),
        [constants.R_SQUARED, constants.MAPE, constants.WMAPE],
    )
    self.assertListEqual(
        list(predictive_accuracy_dataset[constants.GEO_GRANULARITY].values),
        [constants.GEO, constants.NATIONAL],
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )
    self.assertListEqual(
        list(df.columns),
        [constants.METRIC, constants.GEO_GRANULARITY, constants.VALUE],
    )

  @mock.patch.object(
      model.Meridian, "is_national", new=property(lambda unused_self: True)
  )
  def test_predictive_accuracy_national(self):
    predictive_accuracy_dataset = (
        self.analyzer_media_and_rf.predictive_accuracy()
    )
    self.assertListEqual(
        list(predictive_accuracy_dataset[constants.GEO_GRANULARITY].values),
        [constants.NATIONAL],
    )

  @parameterized.product(
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_predictive_accuracy_without_holdout_id_parameterized(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    predictive_accuracy_dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
    }
    predictive_accuracy_dataset = (
        self.analyzer_media_and_rf.predictive_accuracy(
            **predictive_accuracy_dims_kwargs,
        )
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )

    if not selected_geos and not selected_times:
      self.assertAllClose(
          list(df[constants.VALUE]),
          test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_NO_GEOS_OR_TIMES,
          atol=1e-3,
      )
    elif selected_geos and not selected_times:
      self.assertAllClose(
          list(df[constants.VALUE]),
          test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_GEOS_NO_TIMES,
          atol=1e-3,
      )
    elif not selected_geos and selected_times:
      self.assertAllClose(
          list(df[constants.VALUE]),
          test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_NO_GEOS,
          atol=1e-3,
      )
    else:
      self.assertAllClose(
          list(df[constants.VALUE]),
          test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_AND_GEOS,
          atol=1e-3,
      )

  def test_predictive_accuracy_with_holdout_id_table_properties_correct(self):
    n_geos = self.meridian_media_and_rf.n_geos
    n_times = self.meridian_media_and_rf.n_times
    holdout_id = np.full([n_geos, n_times], False)
    for i in range(n_geos):
      holdout_id[i, np.random.choice(n_times, int(np.round(0.2 * n_times)))] = (
          True
      )
    model_spec = spec.ModelSpec(holdout_id=holdout_id)
    meridian = model.Meridian(
        model_spec=model_spec, input_data=self.input_data_media_and_rf
    )
    analyzer_holdout_id = analyzer.Analyzer(meridian)
    predictive_accuracy_dataset = analyzer_holdout_id.predictive_accuracy()
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )
    self.assertListEqual(
        list(df.columns),
        [
            constants.METRIC,
            constants.GEO_GRANULARITY,
            constants.EVALUATION_SET_VAR,
            constants.VALUE,
        ],
    )

  @parameterized.product(
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_predictive_accuracy_with_holdout_id_correct(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    n_geos = len(self.input_data_media_and_rf.geo)
    n_times = len(self.input_data_media_and_rf.time)
    holdout_id = np.full([n_geos, n_times], False)
    for i in range(n_geos):
      holdout_id[i, np.random.choice(n_times, int(np.round(0.2 * n_times)))] = (
          True
      )
    model_spec = spec.ModelSpec(holdout_id=holdout_id)  # Set holdout_id
    meridian = model.Meridian(
        model_spec=model_spec, input_data=self.input_data_media_and_rf
    )
    analyzer_holdout_id = analyzer.Analyzer(meridian)

    predictive_accuracy_dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
    }

    predictive_accuracy_dataset = analyzer_holdout_id.predictive_accuracy(
        **predictive_accuracy_dims_kwargs,
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )

    if not selected_geos and not selected_times:
      expected_values = (
          test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NO_GEOS_OR_TIMES
      )
    elif selected_geos and not selected_times:
      expected_values = test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_GEOS_NO_TIMES
    elif not selected_geos and selected_times:
      expected_values = test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_TIMES_NO_GEOS
    else:
      expected_values = test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_TIMES_AND_GEO

    self.assertAllClose(
        list(df[constants.VALUE]),
        expected_values,
        atol=2e-3,
    )

  @parameterized.product(
      selected_geos=[None, ["geo_0"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  @mock.patch.object(
      model.Meridian,
      "inference_data",
      new_callable=mock.PropertyMock,
  )
  def test_predictive_accuracy_with_holdout_id_national_correct(
      self, mock_inference_data, selected_geos, selected_times
  ):
    mock_inference_data.return_value = self.inference_data_national
    input_data = self.input_data_national
    n_times = len(input_data.time)
    holdout_id = np.full([n_times], False)
    holdout_id[np.random.choice(n_times, int(np.round(0.2 * n_times)))] = True
    model_spec = spec.ModelSpec(holdout_id=holdout_id)  # Set holdout_id
    meridian = model.Meridian(model_spec=model_spec, input_data=input_data)
    analyzer_holdout_id = analyzer.Analyzer(meridian)
    predictive_accuracy_dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
    }

    predictive_accuracy_dataset = analyzer_holdout_id.predictive_accuracy(
        **predictive_accuracy_dims_kwargs,
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )

    if not selected_times:
      expected_values = (
          test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NATIONAL_NO_TIMES
      )
    else:
      expected_values = test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NATIONAL_TIMES

    self.assertAllClose(
        list(df[constants.VALUE]),
        expected_values,
        atol=2e-3,
    )

  def test_response_curves_check_both_channel_types_returns_correct_spend(self):
    response_curve_data = self.analyzer_media_and_rf.response_curves(
        by_reach=False
    )
    response_data_spend = response_curve_data.spend.values

    media_summary_spend = self.analyzer_media_and_rf.summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
    ).spend[:-1]
    self.assertAllEqual(
        media_summary_spend * 2,
        response_data_spend[-1],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          aggregate_geos=False,
          aggregate_times=False,
          holdout_id=None,
          split_by_holdout_id=False,
          expected_shape=(
              _N_GEOS,
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
          ),
          expected_actual_shape=(_N_GEOS, _N_TIMES),
      ),
      dict(
          testcase_name="split_by_holdout_id_true_wo_holdout_id",
          aggregate_geos=False,
          aggregate_times=False,
          holdout_id=None,
          split_by_holdout_id=True,
          expected_shape=(
              _N_GEOS,
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
          ),
          expected_actual_shape=(_N_GEOS, _N_TIMES),
      ),
      dict(
          testcase_name="split_by_holdout_id_true_w_holdout_id",
          aggregate_geos=False,
          aggregate_times=False,
          holdout_id=np.random.choice([True, False], size=(_N_GEOS, _N_TIMES)),
          split_by_holdout_id=True,
          expected_shape=(
              _N_GEOS,
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
              3,  # [train, test, all]
          ),
          expected_actual_shape=(_N_GEOS, _N_TIMES),
      ),
      dict(
          testcase_name="split_by_holdout_id_false_w_holdout_id",
          aggregate_geos=False,
          aggregate_times=False,
          holdout_id=np.random.choice([True, False], size=(_N_GEOS, _N_TIMES)),
          split_by_holdout_id=False,
          expected_shape=(
              _N_GEOS,
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
          ),
          expected_actual_shape=(_N_GEOS, _N_TIMES),
      ),
      dict(
          testcase_name="aggregate_geos_wo_split",
          aggregate_geos=True,
          aggregate_times=False,
          holdout_id=None,
          split_by_holdout_id=False,
          expected_shape=(
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
          ),
          expected_actual_shape=(_N_TIMES,),
      ),
      dict(
          testcase_name="aggregate_geos_w_split",
          aggregate_geos=True,
          aggregate_times=False,
          holdout_id=np.random.choice([True, False], size=(_N_GEOS, _N_TIMES)),
          split_by_holdout_id=True,
          expected_shape=(
              _N_TIMES,
              3,  # [mean, ci_lo, ci_hi]
              3,  # [train, test, all]
          ),
          expected_actual_shape=(_N_TIMES,),
      ),
      dict(
          testcase_name="aggregate_times_wo_split",
          aggregate_geos=False,
          aggregate_times=True,
          holdout_id=None,
          split_by_holdout_id=False,
          expected_shape=(
              _N_GEOS,
              3,  # [mean, ci_lo, ci_hi]
          ),
          expected_actual_shape=(_N_GEOS,),
      ),
      dict(
          testcase_name="aggregate_times_w_split",
          aggregate_geos=False,
          aggregate_times=True,
          holdout_id=np.random.choice([True, False], size=(_N_GEOS, _N_TIMES)),
          split_by_holdout_id=True,
          expected_shape=(
              _N_GEOS,
              3,  # [mean, ci_lo, ci_hi]
              3,  # [train, test, all]
          ),
          expected_actual_shape=(_N_GEOS,),
      ),
      dict(
          testcase_name="aggregate_geos_and_times_wo_split",
          aggregate_geos=True,
          aggregate_times=True,
          holdout_id=None,
          split_by_holdout_id=False,
          expected_shape=(3,),  # [mean, ci_lo, ci_hi]
          expected_actual_shape=(),
      ),
      dict(
          testcase_name="aggregate_geos_and_times_w_split",
          aggregate_geos=True,
          aggregate_times=True,
          holdout_id=np.random.choice([True, False], size=(_N_GEOS, _N_TIMES)),
          split_by_holdout_id=True,
          expected_shape=(
              3,  # [mean, ci_lo, ci_hi]
              3,  # [train, test, all]
          ),
          expected_actual_shape=(),
      ),
  )
  def test_expected_vs_actual_correct_data(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      holdout_id: np.ndarray | None,
      split_by_holdout_id: bool,
      expected_shape: tuple[int, ...],
      expected_actual_shape: tuple[int, ...],
  ):
    model_spec = spec.ModelSpec(holdout_id=holdout_id)
    meridian = model.Meridian(
        model_spec=model_spec, input_data=self.input_data_media_and_rf
    )
    meridian_analyzer = analyzer.Analyzer(meridian)

    ds = meridian_analyzer.expected_vs_actual_data(
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        split_by_holdout_id=split_by_holdout_id,
    )

    expected_actual_values = (
        meridian.kpi
        if self.input_data_media_and_rf.revenue_per_kpi is None
        else meridian.kpi * self.input_data_media_and_rf.revenue_per_kpi
    )  # shape (n_geos, n_times)

    axis_to_sum = tuple(
        ([0] if aggregate_geos else []) + ([1] if aggregate_times else [])
    )
    expected_actual_values = np.sum(expected_actual_values, axis=axis_to_sum)

    if aggregate_geos:
      self.assertNotIn(constants.GEO, ds.coords)
    else:
      self.assertListEqual(
          list(ds.geo.values), list(self.input_data_media_and_rf.geo.values)
      )

    if aggregate_times:
      self.assertNotIn(constants.TIME, ds.coords)
    else:
      self.assertListEqual(
          list(ds.time.values), list(self.input_data_media_and_rf.time.values)
      )

    self.assertListEqual(
        list(ds.metric.values),
        [constants.MEAN, constants.CI_LO, constants.CI_HI],
    )

    self.assertEqual(ds.expected.shape, expected_shape)
    self.assertEqual(ds.baseline.shape, expected_shape)
    self.assertEqual(ds.actual.shape, expected_actual_shape)
    self.assertEqual(ds.confidence_level, constants.DEFAULT_CONFIDENCE_LEVEL)

    np.testing.assert_array_less(
        ds.expected.sel(metric=constants.MEAN),
        ds.expected.sel(metric=constants.CI_HI),
    )
    np.testing.assert_array_less(
        ds.expected.sel(metric=constants.CI_LO),
        ds.expected.sel(metric=constants.MEAN),
    )
    np.testing.assert_array_less(
        ds.baseline.sel(metric=constants.MEAN),
        ds.baseline.sel(metric=constants.CI_HI),
    )
    np.testing.assert_array_less(
        ds.baseline.sel(metric=constants.CI_LO),
        ds.baseline.sel(metric=constants.MEAN),
    )
    np.testing.assert_array_less(ds.baseline, ds.expected)

    # Test the math for a sample of the actual outcome metrics.
    self.assertAllClose(
        ds.actual.values,
        expected_actual_values,
        atol=1e-5,
    )

  def test_expected_vs_actual_warns_if_split_by_holdout_id_without_holdout_id(
      self,
  ):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.analyzer_media_and_rf.expected_vs_actual_data(
          split_by_holdout_id=True
      )

      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "`split_by_holdout_id` is True but `holdout_id` is `None`. Data will"
          " not be split.",
          str(w[0].message),
      )

  def test_adstock_decay_dataframe(self):
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )

    self.assertEqual(
        list(adstock_decay_dataframe.columns),
        [
            constants.CHANNEL,
            constants.TIME_UNITS,
            constants.DISTRIBUTION,
            constants.CI_HI,
            constants.CI_LO,
            constants.MEAN,
            constants.IS_INT_TIME_UNIT,
        ],
    )
    self.assertAllInSet(
        list(set(adstock_decay_dataframe[constants.CHANNEL])),
        ["rf_ch_0", "rf_ch_1", "ch_0", "ch_1", "ch_2"],
    )
    for i, e in enumerate(list(adstock_decay_dataframe[constants.MEAN])):
      self.assertGreaterEqual(
          e, list(adstock_decay_dataframe[constants.CI_LO])[i]
      )
      self.assertLessEqual(e, list(adstock_decay_dataframe[constants.CI_HI])[i])

  def test_adstock_decay_effect_values(self):
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )

    first_channel = adstock_decay_dataframe[constants.CHANNEL].iloc[0]
    first_channel_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.CHANNEL] == first_channel
    ]
    prior_df = first_channel_df[
        first_channel_df[constants.DISTRIBUTION] == constants.PRIOR
    ]
    posterior_df = first_channel_df[
        first_channel_df[constants.DISTRIBUTION] == constants.POSTERIOR
    ]

    mean_arr_prior = list(prior_df[constants.MEAN])
    ci_lo_arr_prior = list(prior_df[constants.CI_LO])
    ci_hi_arr_prior = list(prior_df[constants.CI_HI])

    mean_arr_posterior = list(posterior_df[constants.MEAN])
    ci_lo_arr_posterior = list(posterior_df[constants.CI_LO])
    ci_hi_arr_posterior = list(posterior_df[constants.CI_HI])

    # Make sure values are monotonically decreasing throughout DataFrame slice
    # for one channel.
    for i in range(len(mean_arr_prior) - 1):
      self.assertLessEqual(mean_arr_prior[i + 1], mean_arr_prior[i])
      self.assertLessEqual(ci_lo_arr_prior[i + 1], ci_lo_arr_prior[i])
      self.assertLessEqual(ci_hi_arr_prior[i + 1], ci_hi_arr_prior[i])

      self.assertLessEqual(mean_arr_posterior[i + 1], mean_arr_posterior[i])
      self.assertLessEqual(ci_lo_arr_posterior[i + 1], ci_lo_arr_posterior[i])
      self.assertLessEqual(ci_hi_arr_posterior[i + 1], ci_hi_arr_posterior[i])

  def test_adstock_decay_math_correct(self):
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )

    first_channel = adstock_decay_dataframe[constants.CHANNEL].iloc[0]
    first_channel_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.CHANNEL] == first_channel
    ]

    self.assertAllClose(
        list(first_channel_df[constants.CI_HI])[:5],
        test_utils.ADSTOCK_DECAY_CI_HI,
        atol=1e-3,
    )

    self.assertAllClose(
        list(first_channel_df[constants.CI_LO])[:5],
        test_utils.ADSTOCK_DECAY_CI_LO,
        atol=1e-3,
    )

    self.assertAllClose(
        list(first_channel_df[constants.MEAN])[:5],
        test_utils.ADSTOCK_DECAY_MEAN,
        atol=1e-3,
    )

  def test_adstock_decay_time_unit_integer_indication_correct(self):
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )
    is_true_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.IS_INT_TIME_UNIT]
    ]
    for i in range(len(is_true_df[constants.TIME_UNITS])):
      self.assertEqual(
          list(is_true_df[constants.TIME_UNITS])[i],
          int(list(is_true_df[constants.TIME_UNITS])[i]),
      )

  def test_get_historical_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer_media_and_rf.get_historical_spend(
        selected_times=None
    )
    expected_channel_names = (
        self.input_data_media_and_rf.get_all_paid_channels()
    )

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_historical_spend_correct_values(self):
    # Set it to None to avoid the dimension checks on inference data.
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: None),
        )
    )

    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=1,
        n_times=3,
        n_media_times=4,
        n_media_channels=2,
        n_rf_channels=1,
        seed=0,
    )

    # Avoid the pytype check complaint.
    assert data.media_channel is not None and data.rf_channel is not None

    data.media_spend = xr.DataArray(
        np.array([[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]]),
        dims=["geo", "time", "media_channel"],
        coords={
            "geo": data.geo.values,
            "time": data.time.values,
            "media_channel": data.media_channel.values,
        },
    )
    data.rf_spend = xr.DataArray(
        np.array([[[3.0], [3.1], [3.2]]]),
        dims=["geo", "time", "rf_channel"],
        coords={
            "geo": data.geo.values,
            "time": data.time.values,
            "rf_channel": data.rf_channel.values,
        },
    )

    model_spec = spec.ModelSpec(max_lag=15)
    meridian = model.Meridian(input_data=data, model_spec=model_spec)
    meridian_analyzer = analyzer.Analyzer(meridian)

    # All times are selected.
    selected_times = None
    actual_hist_spend = meridian_analyzer.get_historical_spend(selected_times)
    expected_all_spend = np.array([3.3, 6.3, 9.3])
    self.assertAllClose(expected_all_spend, actual_hist_spend.data)

  def test_get_historical_spend_selected_times_correct_values(self):
    # Set it to None to avoid the dimension checks on inference data.
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: None),
        )
    )

    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=1,
        n_times=3,
        n_media_times=4,
        n_media_channels=2,
        n_rf_channels=1,
        seed=0,
    )

    # Avoid the pytype check complaint.
    assert data.media_channel is not None and data.rf_channel is not None

    data.media_spend = xr.DataArray(
        np.array([[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]]),
        dims=["geo", "time", "media_channel"],
        coords={
            "geo": data.geo.values,
            "time": data.time.values,
            "media_channel": data.media_channel.values,
        },
    )
    data.rf_spend = xr.DataArray(
        np.array([[[3.0], [3.1], [3.2]]]),
        dims=["geo", "time", "rf_channel"],
        coords={
            "geo": data.geo.values,
            "time": data.time.values,
            "rf_channel": data.rf_channel.values,
        },
    )

    model_spec = spec.ModelSpec(max_lag=15)
    meridian = model.Meridian(input_data=data, model_spec=model_spec)
    meridian_analyzer = analyzer.Analyzer(meridian)

    # The first two times are selected.
    selected_times = ["2021-01-25", "2021-02-01"]

    actual_hist_spend = meridian_analyzer.get_historical_spend(selected_times)
    expected_all_spend = np.array([2.1, 4.1, 6.1])
    self.assertAllClose(expected_all_spend, actual_hist_spend.data)

  def test_get_historical_spend_with_single_dim_spends(self):
    seed = 0
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=seed,
    )
    data.media_spend = data_test_utils.random_media_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=seed,
    )
    data.rf_spend = data_test_utils.random_rf_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_rf_channels=_N_RF_CHANNELS,
        seed=seed,
    )

    model_spec = spec.ModelSpec(max_lag=15)
    meridian = model.Meridian(input_data=data, model_spec=model_spec)
    meridian_analyzer = analyzer.Analyzer(meridian)

    n_sub_times = 4

    selected_times = data.time.values[-n_sub_times:].tolist()
    actual_hist_spends = meridian_analyzer.get_historical_spend(selected_times)

    # The spend is interpolated based on the ratio of media execution in the
    # selected times to the media execution in the entire time period.
    # Shape (n_geos, n_times, n_media_channels + n_rf_channels)
    all_media_exe_values = data.get_all_media_and_rf()

    # Get the media execution values in the selected times.
    # Shape (n_media_channels + n_rf_channels)
    target_media_exe_values = np.sum(
        all_media_exe_values[:, -n_sub_times:], axis=(0, 1)
    )
    # Get the media execution values in the entire time period.
    # Shape (n_media_channels + n_rf_channels)
    all_media_exe_values = np.sum(
        all_media_exe_values[:, -meridian.n_times :], axis=(0, 1)
    )
    # The ratio will be used to interpolate the spend.
    ratio = target_media_exe_values / all_media_exe_values
    # Shape (n_media_channels + n_rf_channels)
    expected_all_spend = data.get_total_spend() * ratio

    self.assertAllClose(expected_all_spend, actual_hist_spends.data)

  def test_get_historical_spend_with_empty_times(self):
    selected_times = []
    actual = self.analyzer_media_and_rf.get_historical_spend(selected_times)
    self.assertAllEqual(
        actual.data, np.zeros((_N_MEDIA_CHANNELS + _N_RF_CHANNELS))
    )

  def test_get_historical_spend_with_no_channel_selected(self):
    selected_times = self.input_data_media_and_rf.time.values.tolist()

    with self.assertRaisesRegex(
        ValueError, "At least one of include_media or include_rf must be True."
    ):
      self.analyzer_media_and_rf.get_historical_spend(
          selected_times, include_media=False, include_rf=False
      )


class AnalyzerMediaOnlyTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerMediaOnlyTest, cls).setUpClass()

    cls.input_data_media_only = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_media_only = model.Meridian(
        input_data=cls.input_data_media_only, model_spec=model_spec
    )
    cls.analyzer_media_only = analyzer.Analyzer(cls.meridian_media_only)

    cls.inference_data_media_only = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH,
    )

    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_media_only),
        )
    )

  def test_filter_and_aggregate_geos_and_times_incorrect_n_dim(self):
    with self.assertRaisesRegex(
        ValueError,
        "The tensor must have at least 3 dimensions if `has_media_dim=True` or"
        " at least 2 dimensions if `has_media_dim=False`.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.population),
          flexible_time_dim=True,
          has_media_dim=False,
      )

  @parameterized.named_parameters(
      (
          "not_flexible_time_dim",
          False,
          True,
          (
              "The tensor must have shape [..., n_geos, n_times, n_channels] or"
              " [..., n_geos, n_times] if `flexible_time_dim=False`."
          ),
      ),
      (
          "flexible_time_dim_w_media",
          True,
          True,
          (
              "If `has_media_dim=True`, the tensor must have shape"
              " `[..., n_geos, n_times, n_channels]`, where the time dimension"
              " is flexible."
          ),
      ),
      (
          "flexible_time_dim_wo_media",
          True,
          False,
          (
              "If `has_media_dim=False`, the tensor must have shape"
              " `[..., n_geos, n_times]`, where the time dimension is flexible."
          ),
      ),
  )
  def test_filter_and_aggregate_geos_and_times_incorrect_tensor_shape(
      self, flexible_time_dim: bool, has_media_dim: bool, error_message: str
  ):
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(
              self.input_data_media_only.media_spend[..., :-1, :-1]
          ),
          flexible_time_dim=flexible_time_dim,
          has_media_dim=has_media_dim,
      )

  def test_filter_and_aggregate_geos_and_times_empty_geos(self):
    tensor = tf.convert_to_tensor(self.input_data_media_only.media_spend)
    modified_tensor = (
        self.analyzer_media_only.filter_and_aggregate_geos_and_times(
            tensor,
            selected_geos=[],
        )
    )
    self.assertAllEqual(modified_tensor, tf.zeros([3]))

  def test_filter_and_aggregate_geos_and_times_empty_times(self):
    tensor = tf.convert_to_tensor(self.input_data_media_only.media_spend)
    modified_tensor = (
        self.analyzer_media_only.filter_and_aggregate_geos_and_times(
            tensor,
            selected_times=[],
        )
    )
    self.assertAllEqual(modified_tensor, tf.zeros([3]))

  def test_filter_and_aggregate_geos_and_times_incorrect_geos(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_geos` must match the geo dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_geos=["random_geo"],
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_time_dim_names(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_times=["random_time"],
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_time_bool(self):
    with self.assertRaisesRegex(
        ValueError,
        "Boolean `selected_times` must have the same number of elements as "
        "there are time period coordinates in `tensor`.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_times=[True] + [False] * (_N_MEDIA_TIMES - 1),
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_selected_times_type(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must be a list of strings or a list of booleans.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_times=["random_time", False, True],
      )

  @parameterized.product(
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[
          None,
          ["2021-04-19", "2021-09-13", "2021-12-13"],
          [False] * (_N_TIMES - 3) + [True] * 3,
      ],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
  )
  def test_filter_and_aggregate_geos_and_times_returns_correct_shape(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      aggregate_geos: bool,
      aggregate_times: bool,
  ):
    tensor = tf.convert_to_tensor(self.input_data_media_only.media_spend)
    modified_tensor = (
        self.analyzer_media_only.filter_and_aggregate_geos_and_times(
            tensor,
            selected_geos=selected_geos,
            selected_times=selected_times,
            aggregate_geos=aggregate_geos,
            aggregate_times=aggregate_times,
        )
    )
    expected_shape = ()
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      if selected_times is not None:
        if all(isinstance(time, bool) for time in selected_times):
          n_times = sum(selected_times)
        else:
          n_times = len(selected_times)
      else:
        n_times = _N_TIMES
      expected_shape += (n_times,)
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(modified_tensor.shape, expected_shape)

  @parameterized.product(
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      optimal_frequency=[None, [1.0, 3.0]],
  )
  def test_get_aggregated_impressions_returns_correct_shape(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      aggregate_geos: bool,
      aggregate_times: bool,
      optimal_frequency: Sequence[float] | None,
  ):
    impressions = self.analyzer_media_only.get_aggregated_impressions(
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        optimal_frequency=optimal_frequency,
    )
    expected_shape = ()
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(impressions.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_outcome_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    outcome = self.analyzer_media_only.expected_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=geos_to_include,
        selected_times=times_to_include,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(geos_to_include),) if geos_to_include is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(times_to_include),)
          if times_to_include is not None
          else (_N_TIMES,)
      )
    self.assertEqual(outcome.shape, expected_shape)

  @parameterized.named_parameters(
      ("missing_media", "media"),
      ("missing_revenue_per_kpi", "revenue_per_kpi"),
  )
  def test_incremental_outcome_media_only_missing_new_param_raises_exception(
      self, missing_param: str
  ):
    new_data_dict = {
        "media": tf.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        "revenue_per_kpi": tf.ones((_N_GEOS, 10)),
    }

    with self.assertRaisesRegex(
        ValueError,
        "If new_media, new_reach, new_frequency, new_organic_media,"
        " new_organic_reach, new_organic_frequency, or new_revenue_per_kpi is"
        " provided with a different number of time periods than in `InputData`,"
        " then all new parameters originally in `InputData` must be provided"
        " with the same number of time periods.",
    ):
      new_data_dict.pop(missing_param)
      self.analyzer_media_only.incremental_outcome(
          new_data=analyzer.DataTensors(**new_data_dict)
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_incremental_outcome_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    outcome = self.analyzer_media_only.incremental_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(outcome.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_outcome=test_utils.INC_OUTCOME_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=test_utils.INC_OUTCOME_MEDIA_ONLY_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_media_only(
      self,
      use_posterior: bool,
      expected_outcome: tuple[float, ...],
  ):
    outcome = self.analyzer_media_only.incremental_outcome(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(expected_outcome),
        rtol=1e-3,
        atol=1e-3,
    )

  # The purpose of this test is to prevent accidental logic change.
  def test_incremental_outcome_media_only_new_params(self):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    outcome = self.analyzer_media_only.incremental_outcome(
        new_data=analyzer.DataTensors(
            media=self.meridian_media_only.media_tensors.media[..., -10:, :],
            revenue_per_kpi=self.meridian_media_only.revenue_per_kpi[..., -10:],
        ),
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(test_utils.INC_OUTCOME_MEDIA_ONLY_NEW_PARAMS),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
  ):
    mroi = self.analyzer_media_only.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(mroi.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          by_reach=False,
          expected_mroi=test_utils.MROI_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=test_utils.MROI_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=test_utils.MROI_MEDIA_ONLY_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=test_utils.MROI_MEDIA_ONLY_USE_POSTERIOR,
      ),
  )
  def test_marginal_roi_media_only(
      self, use_posterior: bool, by_reach: bool, expected_mroi: np.ndarray
  ):
    mroi = self.analyzer_media_only.marginal_roi(
        by_reach=by_reach,
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        mroi,
        tf.convert_to_tensor(expected_mroi),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    roi = self.analyzer_media_only.roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(roi.shape, expected_shape)

  def test_roi_media_only_spend_1d_returns_correct_value(self):
    total_spend = self.analyzer_media_only.filter_and_aggregate_geos_and_times(
        self.meridian_media_only.media_tensors.media_spend
    )
    roi = self.analyzer_media_only.roi(
        new_data=analyzer.DataTensors(media_spend=total_spend)
    )
    expected_roi = self.analyzer_media_only.incremental_outcome() / total_spend
    self.assertAllClose(roi, expected_roi)

  def test_roi_media_only_default_returns_correct_value(self):
    roi = self.analyzer_media_only.roi()
    total_spend = self.analyzer_media_only.filter_and_aggregate_geos_and_times(
        self.meridian_media_only.media_tensors.media_spend
    )
    expected_roi = self.analyzer_media_only.incremental_outcome() / total_spend
    self.assertAllClose(expected_roi, roi)

  def test_roi_zero_media_returns_zero(self):
    new_media = tf.zeros_like(
        self.meridian_media_only.media_tensors.media, dtype=tf.float32
    )
    roi = self.analyzer_media_only.roi(
        new_data=analyzer.DataTensors(media=new_media)
    )
    self.assertAllClose(
        roi, tf.zeros((_N_CHAINS, _N_KEEP, _N_MEDIA_CHANNELS)), atol=2e-6
    )

  def test_optimal_frequency_data_media_only_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "Must have at least one channel with reach and frequency data.",
    ):
      self.analyzer_media_only.optimal_freq()

  def test_rhat_media_only_correct(self):
    rhat = self.analyzer_media_only.get_rhat()
    self.assertSetEqual(
        set(rhat.keys()),
        set(constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES),
    )

  def test_rhat_summary_media_only_correct(self):
    rhat_summary = self.analyzer_media_only.rhat_summary()
    self.assertEqual(rhat_summary.shape, (13, 7))
    self.assertSetEqual(
        set(rhat_summary.param),
        set(constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES)
        - set([constants.SLOPE_M]),
    )

  def test_response_curves_returns_correct_data(self):
    response_curve_data = self.analyzer_media_only.response_curves()
    self.assertEqual(
        list(response_curve_data.coords.keys()),
        [constants.CHANNEL, constants.METRIC, constants.SPEND_MULTIPLIER],
    )
    self.assertEqual(
        list(response_curve_data.data_vars.keys()),
        [constants.SPEND, constants.INCREMENTAL_OUTCOME],
    )
    response_curves_df = (
        response_curve_data[[constants.SPEND, constants.INCREMENTAL_OUTCOME]]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.SPEND,
                constants.SPEND_MULTIPLIER,
            ],
            columns=constants.METRIC,
            values=constants.INCREMENTAL_OUTCOME,
        )
    ).reset_index()
    self.assertAllInSet(
        list(set(response_curves_df[constants.CHANNEL])),
        ["ch_0", "ch_2", "ch_1"],
    )
    self.assertEqual(max(response_curves_df[constants.SPEND_MULTIPLIER]), 2.0)
    for i, mean in enumerate(response_curves_df[constants.MEAN]):
      self.assertGreaterEqual(mean, response_curves_df[constants.CI_LO][i])
      self.assertLessEqual(mean, response_curves_df[constants.CI_HI][i])

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_media_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    model_analyzer = self.analyzer_media_only
    num_channels = _N_MEDIA_CHANNELS

    media_summary = model_analyzer.summary_metrics(
        confidence_level=0.8,
        marginal_roi_by_reach=False,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_channel_shape = ()
    if not aggregate_geos:
      expected_channel_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_channel_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # (ch_1, ch_2, ..., Total_media, [mean, median, ci_lo, ci_hi],
    # [prior, posterior])
    expected_channel_shape += (num_channels + 1,)
    expected_shape = expected_channel_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.impressions.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.pct_of_impressions.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.pct_of_spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.cpm.shape, expected_channel_shape)
    self.assertEqual(media_summary.incremental_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)
    if aggregate_times:
      self.assertEqual(media_summary.roi.shape, expected_shape)
      self.assertEqual(media_summary.effectiveness.shape, expected_shape)
      self.assertEqual(media_summary.mroi.shape, expected_shape)
      self.assertEqual(media_summary.cpik.shape, expected_shape)
    else:
      self.assertNotIn(constants.ROI, media_summary.data_vars)
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)
      self.assertNotIn(constants.MROI, media_summary.data_vars)
      self.assertNotIn(constants.CPIK, media_summary.data_vars)

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_baseline_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    analyzer_ = self.analyzer_media_only

    media_summary = analyzer_.baseline_summary_metrics(
        confidence_level=0.8,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_geo_and_time_shape = ()
    if not aggregate_geos:
      expected_geo_and_time_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_geo_and_time_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # ([mean, median, ci_lo, ci_hi], [prior, posterior])
    expected_shape = expected_geo_and_time_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.baseline_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)

  def test_get_historical_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer_media_only.get_historical_spend(
        selected_times=None, include_rf=False
    )
    expected_channel_names = self.input_data_media_only.get_all_paid_channels()

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_historical_spend_requests_rf_when_no_rf_throws_warning(self):
    selected_times = self.input_data_media_only.time.values.tolist()
    with self.assertWarnsRegex(
        UserWarning,
        "Requested spends for paid media channels with R&F data, but but the"
        " channels are not available.",
    ):
      self.analyzer_media_only.get_historical_spend(selected_times)

  def test_get_historical_spend_requests_rf_when_no_rf_outputs_empty_data_array(
      self,
  ):
    selected_times = self.input_data_media_only.time.values.tolist()
    actual = self.analyzer_media_only.get_historical_spend(
        selected_times, include_media=False
    )
    self.assertAllEqual(actual.data, [])


class AnalyzerRFOnlyTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerRFOnlyTest, cls).setUpClass()

    cls.input_data_rf_only = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_rf_only = model.Meridian(
        input_data=cls.input_data_rf_only, model_spec=model_spec
    )

    cls.analyzer_rf_only = analyzer.Analyzer(cls.meridian_rf_only)

    cls.inference_data_rf_only = _build_inference_data(
        _TEST_SAMPLE_PRIOR_RF_ONLY_PATH,
        _TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH,
    )

    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_rf_only),
        )
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_outcome_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    outcome = self.analyzer_rf_only.expected_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=geos_to_include,
        selected_times=times_to_include,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(geos_to_include),) if geos_to_include is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(times_to_include),)
          if times_to_include is not None
          else (_N_TIMES,)
      )
    self.assertEqual(outcome.shape, expected_shape)

  @parameterized.named_parameters(
      ("missing_reach", "reach"),
      ("missing_frequency", "frequency"),
      ("missing_revenue_per_kpi", "revenue_per_kpi"),
  )
  def test_incremental_outcome_rf_only_missing_new_param_raises_exception(
      self, missing_param: str
  ):
    new_data_dict = {
        "reach": tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        "frequency": tf.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        "revenue_per_kpi": tf.ones((_N_GEOS, 10)),
    }
    with self.assertRaisesRegex(
        ValueError,
        "If new_media, new_reach, new_frequency, new_organic_media,"
        " new_organic_reach, new_organic_frequency, or new_revenue_per_kpi is"
        " provided with a different number of time periods than in `InputData`,"
        " then all new parameters originally in `InputData` must be provided"
        " with the same number of time periods.",
    ):
      new_data_dict.pop(missing_param)
      self.analyzer_rf_only.incremental_outcome(
          new_data=analyzer.DataTensors(**new_data_dict)
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_incremental_outcome_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    outcome = self.analyzer_rf_only.incremental_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )
    expected_shape += (_N_RF_CHANNELS,)
    self.assertEqual(outcome.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_outcome=test_utils.INC_OUTCOME_RF_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=test_utils.INC_OUTCOME_RF_ONLY_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_rf_only(
      self,
      use_posterior: bool,
      expected_outcome: tuple[float, ...],
  ):
    outcome = self.analyzer_rf_only.incremental_outcome(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(expected_outcome),
        rtol=1e-3,
        atol=1e-3,
    )

  # The purpose of this test is to prevent accidental logic change.
  def test_incremental_outcome_rf_only_new_params(self):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    outcome = self.analyzer_rf_only.incremental_outcome(
        new_data=analyzer.DataTensors(
            reach=self.meridian_rf_only.rf_tensors.reach[..., -10:, :],
            frequency=self.meridian_rf_only.rf_tensors.frequency[..., -10:, :],
            revenue_per_kpi=self.meridian_rf_only.revenue_per_kpi[..., -10:],
        )
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(test_utils.INC_OUTCOME_RF_ONLY_NEW_PARAMS),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
  ):
    mroi = self.analyzer_rf_only.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_RF_CHANNELS,)
    self.assertEqual(mroi.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          by_reach=False,
          expected_mroi=test_utils.MROI_RF_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=test_utils.MROI_RF_ONLY_USE_PRIOR_BY_REACH,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=test_utils.MROI_RF_ONLY_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=test_utils.MROI_RF_ONLY_USE_POSTERIOR_BY_REACH,
      ),
  )
  def test_marginal_roi_rf_only(
      self,
      use_posterior: bool,
      by_reach: bool,
      expected_mroi: tuple[float, ...],
  ):
    mroi = self.analyzer_rf_only.marginal_roi(
        by_reach=by_reach,
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        mroi,
        tf.convert_to_tensor(expected_mroi),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    roi = self.analyzer_rf_only.roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_RF_CHANNELS,)
    self.assertEqual(roi.shape, expected_shape)

  def test_roi_rf_only_default_returns_correct_value(self):
    roi = self.analyzer_rf_only.roi()
    total_spend = self.analyzer_rf_only.filter_and_aggregate_geos_and_times(
        self.meridian_rf_only.rf_tensors.rf_spend
    )
    expeted_roi = self.analyzer_rf_only.incremental_outcome() / total_spend
    self.assertAllClose(expeted_roi, roi)

  def test_by_reach_returns_correct_values(self):
    mroi = self.analyzer_rf_only.marginal_roi(
        use_posterior=True,
        aggregate_geos=True,
        selected_geos=None,
        selected_times=None,
        by_reach=True,
    )
    roi = self.analyzer_rf_only.roi(
        use_posterior=True,
        aggregate_geos=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertAllClose(
        mroi,
        roi,
        atol=1e-2,
        rtol=1e-2,
    )

  def test_media_summary_warns_if_time_not_aggregated(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      media_summary = self.analyzer_rf_only.summary_metrics(
          confidence_level=0.8,
          marginal_roi_by_reach=False,
          aggregate_geos=True,
          aggregate_times=False,
          selected_geos=None,
          selected_times=None,
      )
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "ROI, mROI, Effectiveness, and CPIK are not reported because they do "
          "not have a clear interpretation by time period.",
          str(w[0].message),
      )
      self.assertNotIn(constants.ROI, media_summary.data_vars)
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)
      self.assertNotIn(constants.MROI, media_summary.data_vars)
      self.assertNotIn(constants.CPIK, media_summary.data_vars)

  def test_optimal_frequency_data_rf_only_correct(self):
    actual = self.analyzer_rf_only.optimal_freq(
        freq_grid=[1.0, 2.0, 3.0],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        use_posterior=True,
    )
    expected = xr.Dataset(
        coords={
            constants.FREQUENCY: ([constants.FREQUENCY], [1.0, 2.0, 3.0]),
            constants.RF_CHANNEL: (
                [constants.RF_CHANNEL],
                ["rf_ch_0", "rf_ch_1"],
            ),
            constants.METRIC: (
                [constants.METRIC],
                [
                    constants.MEAN,
                    constants.MEDIAN,
                    constants.CI_LO,
                    constants.CI_HI,
                ],
            ),
        },
        data_vars={
            constants.ROI: (
                [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
                [
                    [
                        [3.34, 1.81, 0.20, 10.17],
                        [7.15, 7.77, 0.42, 13.10],
                    ],  # freq=1.0
                    [
                        [4.76, 3.24, 0.66, 10.79],
                        [4.83, 5.25, 1.13, 7.86],
                    ],  # freq=2.0
                    [
                        [4.87, 3.70, 1.17, 9.60],
                        [3.72, 3.91, 1.47, 5.71],
                    ],  # freq=3.0
                ],
            ),
            constants.OPTIMAL_FREQUENCY: ([constants.RF_CHANNEL], [3.0, 1.0]),
            constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1326.76, 1008.16, 320.44, 2614.4],
                    [2060.22, 2238.75, 122.18, 3772.43],
                ],
            ),
            constants.OPTIMIZED_EFFECTIVENESS: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.000604, 0.000458, 0.000146, 0.001189],
                    [0.002804, 0.003047, 0.000166, 0.005135],
                ],
            ),
            constants.OPTIMIZED_PCT_OF_CONTRIBUTION: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.01613, 0.01225, 0.0039, 0.03179],
                    [0.02505, 0.02721, 0.00149, 0.04587],
                ],
            ),
            constants.OPTIMIZED_ROI: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[4.87, 3.70, 1.18, 9.61], [7.16, 7.77, 0.42, 13.10]],
            ),
            constants.OPTIMIZED_MROI_BY_REACH: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[4.87, 3.70, 1.18, 9.59], [7.16, 7.78, 0.42, 13.12]],
            ),
            constants.OPTIMIZED_MROI_BY_FREQUENCY: (
                [constants.RF_CHANNEL, constants.METRIC],
                [[4.21, 3.26, 2.58, 8.68], [3.11, 3.80, 1.14, 3.95]],
            ),
            constants.OPTIMIZED_CPIK: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.476, 0.48521742, 0.104, 0.849],
                    [0.698, 0.231271, 0.076, 2.403],
                ],
            ),
        },
        attrs={
            constants.CONFIDENCE_LEVEL: constants.DEFAULT_CONFIDENCE_LEVEL,
            "use_posterior": True,
        },
    )

    xr.testing.assert_allclose(actual, expected, atol=0.1)
    xr.testing.assert_allclose(actual.frequency, expected.frequency)
    xr.testing.assert_allclose(actual.rf_channel, expected.rf_channel)
    xr.testing.assert_allclose(actual.metric, expected.metric)
    xr.testing.assert_allclose(actual.roi, expected.roi, atol=0.01)
    xr.testing.assert_allclose(
        actual.optimal_frequency, expected.optimal_frequency
    )
    xr.testing.assert_allclose(
        actual.optimized_incremental_outcome,
        expected.optimized_incremental_outcome,
        atol=0.1,
    )
    xr.testing.assert_allclose(
        actual.optimized_effectiveness,
        expected.optimized_effectiveness,
        atol=0.00001,
    )
    xr.testing.assert_allclose(
        actual.optimized_pct_of_contribution,
        expected.optimized_pct_of_contribution,
        atol=0.0001,
    )
    xr.testing.assert_allclose(
        actual.optimized_roi,
        expected.optimized_roi,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_reach,
        expected.optimized_mroi_by_reach,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_frequency,
        expected.optimized_mroi_by_frequency,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_cpik,
        expected.optimized_cpik,
        atol=0.001,
    )
    self.assertEqual(actual.confidence_level, expected.confidence_level)
    self.assertEqual(actual.use_posterior, expected.use_posterior)

  def test_optimal_frequency_freq_grid(self):
    max_freq = np.max(
        np.array(self.analyzer_rf_only._meridian.rf_tensors.frequency)
    )
    freq_grid = list(np.arange(1, max_freq, 0.1))
    roi = np.zeros(
        (len(freq_grid), self.analyzer_rf_only._meridian.n_rf_channels, 3)
    )
    for i, freq in enumerate(freq_grid):
      new_frequency = (
          tf.ones_like(self.analyzer_rf_only._meridian.rf_tensors.frequency)
          * freq
      )
      new_reach = (
          self.analyzer_rf_only._meridian.rf_tensors.frequency
          * self.analyzer_rf_only._meridian.rf_tensors.reach
          / new_frequency
      )
      dim_kwargs = {
          "selected_geos": None,
          "selected_times": None,
          "aggregate_geos": True,
      }
      roi_temp = self.analyzer_rf_only.roi(
          new_data=analyzer.DataTensors(
              reach=new_reach, frequency=new_frequency
          ),
          use_posterior=True,
          **dim_kwargs,
      )[..., -self.analyzer_rf_only._meridian.n_rf_channels :]
      roi[i, :, 0] = np.mean(roi_temp, (0, 1))
      roi[i, :, 1] = np.quantile(roi_temp, (1 - 0.9) / 2, (0, 1))
      roi[i, :, 2] = np.quantile(roi_temp, (1 + 0.9) / 2, (0, 1))

      self.assertAllEqual(roi[i, :, 0], np.mean(roi_temp, (0, 1)))
      self.assertAllEqual(
          roi[i, :, 1], np.quantile(roi_temp, (1 - 0.9) / 2, (0, 1))
      )
      self.assertAllEqual(
          roi[i, :, 2], np.quantile(roi_temp, (1 + 0.9) / 2, (0, 1))
      )

  def test_rhat_rf_only_correct(self):
    rhat = self.analyzer_rf_only.get_rhat()
    self.assertSetEqual(
        set(rhat.keys()),
        set(constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES),
    )

  def test_rhat_summary_rf_only_correct(self):
    rhat_summary = self.analyzer_rf_only.rhat_summary()
    self.assertEqual(rhat_summary.shape, (14, 7))
    self.assertSetEqual(
        set(rhat_summary.param),
        set(constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES),
    )

  def test_response_curves_check_only_rf_channels_returns_correct_spend(self):
    response_curve_data = self.analyzer_rf_only.response_curves(by_reach=False)
    response_data_spend = response_curve_data.spend.values

    media_summary_spend = self.analyzer_rf_only.summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
    ).spend[:-1]
    self.assertAllEqual(
        media_summary_spend * 2,
        response_data_spend[-1],
    )

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_media_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    model_analyzer = self.analyzer_rf_only
    num_channels = _N_RF_CHANNELS

    media_summary = model_analyzer.summary_metrics(
        confidence_level=0.8,
        marginal_roi_by_reach=False,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_channel_shape = ()
    if not aggregate_geos:
      expected_channel_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_channel_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # (ch_1, ch_2, ..., Total_media, [mean, median,ci_lo, ci_hi],
    # [prior, posterior])
    expected_channel_shape += (num_channels + 1,)
    expected_shape = expected_channel_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.impressions.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.pct_of_impressions.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.pct_of_spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.cpm.shape, expected_channel_shape)
    self.assertEqual(media_summary.incremental_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)
    if aggregate_times:
      self.assertEqual(media_summary.roi.shape, expected_shape)
      self.assertEqual(media_summary.effectiveness.shape, expected_shape)
      self.assertEqual(media_summary.mroi.shape, expected_shape)
      self.assertEqual(media_summary.cpik.shape, expected_shape)
    else:
      self.assertNotIn(constants.ROI, media_summary.data_vars)
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)
      self.assertNotIn(constants.MROI, media_summary.data_vars)
      self.assertNotIn(constants.CPIK, media_summary.data_vars)

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_baseline_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    analyzer_ = self.analyzer_rf_only

    media_summary = analyzer_.baseline_summary_metrics(
        confidence_level=0.8,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    expected_geo_and_time_shape = ()
    if not aggregate_geos:
      expected_geo_and_time_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_geo_and_time_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # ([mean, median, ci_lo, ci_hi], [prior, posterior])
    expected_shape = expected_geo_and_time_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.baseline_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)

  def test_get_historical_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer_rf_only.get_historical_spend(
        selected_times=None, include_media=False
    )
    expected_channel_names = self.input_data_rf_only.get_all_paid_channels()

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_historical_spend_requests_media_when_no_media_throws_warning(
      self,
  ):
    selected_times = self.input_data_rf_only.time.values.tolist()
    with self.assertWarnsRegex(
        UserWarning,
        "Requested spends for paid media channels that do not have R&F data,"
        " but the channels are not available.",
    ):
      self.analyzer_rf_only.get_historical_spend(selected_times)

  def test_get_historical_spend_requests_rf_when_no_rf_outputs_empty_data_array(
      self,
  ):
    selected_times = self.input_data_rf_only.time.values.tolist()
    actual = self.analyzer_rf_only.get_historical_spend(
        selected_times, include_rf=False
    )
    self.assertAllEqual(actual.data, [])


class AnalyzerKpiTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerKpiTest, cls).setUpClass()

    input_data = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    cpik_prior = tfp.distributions.LogNormal(0.5, 0.5)
    roi_prior = tfp.distributions.TransformedDistribution(
        cpik_prior, tfp.bijectors.Reciprocal()
    )
    custom_prior = prior_distribution.PriorDistribution(
        roi_m=roi_prior, roi_rf=roi_prior
    )
    model_spec = spec.ModelSpec(prior=custom_prior)
    cls.meridian_kpi = model.Meridian(
        input_data=input_data, model_spec=model_spec
    )
    cls.analyzer_kpi = analyzer.Analyzer(cls.meridian_kpi)
    inference_data = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: inference_data),
        )
    )

  def test_use_kpi_expected_vs_actual_data_expected_outcome_correct_usage(self):
    mock_expected_outcome = self.enter_context(
        mock.patch.object(
            self.analyzer_kpi,
            "expected_outcome",
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_GEOS,
                _N_TIMES,
            )),
        )
    )
    self.analyzer_kpi.expected_vs_actual_data()
    _, mock_kwargs = mock_expected_outcome.call_args
    self.assertEqual(mock_kwargs["use_kpi"], True)

  def test_use_kpi_no_revenue_per_kpi_correct_usage_expected_vs_actual(self):
    expected_vs_actual = self.analyzer_kpi.expected_vs_actual_data(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )
    self.assertAllClose(
        list(expected_vs_actual.data_vars[constants.ACTUAL].values[:5]),
        list(self.meridian_kpi.kpi[:5]),
        atol=1e-3,
    )

  def test_use_kpi_no_revenue_per_kpi_correct_usage_media_summary_metrics(self):
    media_summary = self.analyzer_kpi.summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
        use_kpi=True,
    )
    self.assertEqual(
        list(media_summary.data_vars.keys()),
        [
            constants.IMPRESSIONS,
            constants.PCT_OF_IMPRESSIONS,
            constants.SPEND,
            constants.PCT_OF_SPEND,
            constants.CPM,
            constants.INCREMENTAL_OUTCOME,
            constants.PCT_OF_CONTRIBUTION,
            constants.ROI,
            constants.EFFECTIVENESS,
            constants.MROI,
            constants.CPIK,
        ],
    )
    # Check the metrics that differ when `use_kpi=True`.
    self.assertAllClose(
        media_summary.incremental_outcome,
        test_utils.SAMPLE_INC_OUTCOME_KPI,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        media_summary.roi, test_utils.SAMPLE_ROI_KPI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.effectiveness,
        test_utils.SAMPLE_EFFECTIVENESS_KPI,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.mroi, test_utils.SAMPLE_MROI_KPI, atol=1e-3, rtol=1e-3
    )

  def test_marginal_roi_no_revenue_data_use_kpi_false(self):
    with self.assertRaisesRegex(
        ValueError,
        "Revenue analysis is not available when `revenue_per_kpi` is unknown."
        " Set `use_kpi=True` to perform KPI analysis instead.",
    ):
      self.analyzer_kpi.marginal_roi(use_kpi=False)

  def test_roi_no_revenue_data_use_kpi_false(self):
    with self.assertRaisesRegex(
        ValueError,
        "Revenue analysis is not available when `revenue_per_kpi` is unknown."
        " Set `use_kpi=True` to perform KPI analysis instead.",
    ):
      self.analyzer_kpi.roi(use_kpi=False)

  def test_use_kpi_no_revenue_per_kpi_correct_usage_response_curves(self):
    mock_incremental_outcome = self.enter_context(
        mock.patch.object(
            self.analyzer_kpi,
            "incremental_outcome",
            return_value=tf.ones((
                _N_CHAINS,
                _N_DRAWS,
                _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            )),
        )
    )
    self.analyzer_kpi.response_curves()
    _, mock_kwargs = mock_incremental_outcome.call_args
    self.assertEqual(mock_kwargs["use_kpi"], True)

  def test_use_kpi_no_revenue_per_kpi_correct_usage_expected_outcome(self):
    with self.assertRaisesRegex(
        ValueError,
        "Revenue analysis is not available when `revenue_per_kpi` is unknown."
        " Set `use_kpi=True` to perform KPI analysis instead.",
    ):
      self.analyzer_kpi.expected_outcome()

  def test_expected_outcome_revenue_kpi_use_kpi_true_raises_warning(self):
    input_data_revenue = data_test_utils.sample_input_data_revenue(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    mmm_revenue = model.Meridian(input_data=input_data_revenue)
    analyzer_revenue = analyzer.Analyzer(mmm_revenue)
    with warnings.catch_warnings(record=True) as w:
      analyzer_revenue.expected_outcome(
          use_kpi=True,
      )

      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "Setting `use_kpi=True` has no effect when `kpi_type=REVENUE`"
          " since in this case, KPI is equal to revenue.",
          str(w[0].message),
      )

  def test_optimal_frequency_data_no_revenue_per_kpi_correct(self):
    actual = self.analyzer_kpi.optimal_freq(
        freq_grid=[1.0, 2.0, 3.0],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        use_posterior=True,
        use_kpi=True,
    )
    expected = xr.Dataset(
        coords={
            constants.FREQUENCY: ([constants.FREQUENCY], [1.0, 2.0, 3.0]),
            constants.RF_CHANNEL: (
                [constants.RF_CHANNEL],
                ["rf_ch_0", "rf_ch_1"],
            ),
            constants.METRIC: (
                [constants.METRIC],
                [
                    constants.MEAN,
                    constants.MEDIAN,
                    constants.CI_LO,
                    constants.CI_HI,
                ],
            ),
        },
        data_vars={
            constants.ROI: (
                [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
                [
                    [
                        [1.45663321, 1.46222901, 0.51041067, 2.39638948],
                        [2.12286782, 2.09572244, 0.54957443, 3.752141],
                    ],
                    [
                        [0.78908628, 0.79023874, 0.31107241, 1.2646476],
                        [1.17641282, 1.16203976, 0.35569778, 2.02744746],
                    ],
                    [
                        [0.54987603, 0.5502463, 0.22928859, 0.86951894],
                        [0.81452322, 0.80485409, 0.26908818, 1.38052571],
                    ],
                ],
            ),
            constants.OPTIMAL_FREQUENCY: ([constants.RF_CHANNEL], [1.0, 1.0]),
            constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [396.44, 397.96, 138.92, 652.21],
                    [611.12, 603.30, 158.21, 1080.15],
                ],
            ),
            constants.OPTIMIZED_PCT_OF_CONTRIBUTION: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [5.1566, 5.1764145, 1.8069, 8.48342],
                    [7.94896, 7.8473177, 2.05785, 14.04969],
                ],
            ),
            constants.OPTIMIZED_ROI: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1.45663321, 1.462229, 0.51041068, 2.39638944],
                    [2.12286782, 2.0957224, 0.54957444, 3.75214096],
                ],
            ),
            constants.OPTIMIZED_EFFECTIVENESS: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.000541, 0.000543, 0.00019, 0.00089],
                    [0.000832, 0.000821, 0.000215, 0.00147],
                ],
            ),
            constants.OPTIMIZED_MROI_BY_REACH: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1.4566362, 1.4622416, 0.5104058, 2.396392],
                    [2.1228676, 2.095724, 0.5495737, 3.7521646],
                ],
            ),
            constants.OPTIMIZED_MROI_BY_FREQUENCY: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.17261705, 0.17397353, 0.14313781, 0.19791752],
                    [0.4224712, 0.41871762, 0.22409489, 0.62984283],
                ],
            ),
            constants.OPTIMIZED_CPIK: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1.166, 1.139, 0.417, 1.959],
                    [1.042, 1.038, 0.267, 1.82],
                ],
            ),
        },
        attrs={
            constants.CONFIDENCE_LEVEL: constants.DEFAULT_CONFIDENCE_LEVEL,
            "use_posterior": True,
        },
    )

    xr.testing.assert_allclose(actual, expected, atol=0.1)
    xr.testing.assert_allclose(actual.frequency, expected.frequency)
    xr.testing.assert_allclose(actual.rf_channel, expected.rf_channel)
    xr.testing.assert_allclose(actual.metric, expected.metric)
    xr.testing.assert_allclose(actual.roi, expected.roi, atol=0.01)
    xr.testing.assert_allclose(
        actual.optimal_frequency, expected.optimal_frequency
    )
    xr.testing.assert_allclose(
        actual.optimized_incremental_outcome,
        expected.optimized_incremental_outcome,
        atol=0.1,
    )
    xr.testing.assert_allclose(
        actual.optimized_effectiveness,
        expected.optimized_effectiveness,
        atol=0.000001,
    )
    xr.testing.assert_allclose(
        actual.optimized_pct_of_contribution,
        expected.optimized_pct_of_contribution,
        atol=0.0001,
    )
    xr.testing.assert_allclose(
        actual.optimized_roi,
        expected.optimized_roi,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_reach,
        expected.optimized_mroi_by_reach,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_mroi_by_frequency,
        expected.optimized_mroi_by_frequency,
        atol=0.01,
    )
    xr.testing.assert_allclose(
        actual.optimized_cpik,
        expected.optimized_cpik,
        atol=0.001,
    )
    self.assertEqual(actual.confidence_level, expected.confidence_level)
    self.assertEqual(actual.use_posterior, expected.use_posterior)


class AnalyzerNonMediaTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerNonMediaTest, cls).setUpClass()
    # Input data resulting in revenue computation.
    cls.input_data_non_media = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            seed=0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_non_media = model.Meridian(
        input_data=cls.input_data_non_media, model_spec=model_spec
    )
    cls.analyzer_non_media = analyzer.Analyzer(cls.meridian_non_media)

    cls.inference_data_non_media = _build_inference_data(
        _TEST_SAMPLE_PRIOR_NON_PAID_PATH,
        _TEST_SAMPLE_POSTERIOR_NON_PAID_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_non_media),
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_USE_PRIOR,
          non_media_baseline_values=None,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_USE_POSTERIOR,
          non_media_baseline_values=None,
      ),
      dict(
          testcase_name="all_min",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_USE_POSTERIOR,
          non_media_baseline_values=["min", "min", "min", "min"],
      ),
      dict(
          testcase_name="all_max",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_MAX,
          non_media_baseline_values=["max", "max", "max", "max"],
      ),
      dict(
          testcase_name="mix",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_MIX,
          non_media_baseline_values=["min", "max", "max", "min"],
      ),
      dict(
          testcase_name="mix_as_floats",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_MIX,
          non_media_baseline_values=["min", 3.630, 2.448, "min"],
      ),
      dict(
          testcase_name="all_fixed",
          use_posterior=True,
          expected_result=test_utils.INC_OUTCOME_NON_MEDIA_FIXED,
          non_media_baseline_values=[45.2, 1.03, 0.24, 7.77],
      ),
  )
  def test_incremental_outcome_non_media(
      self,
      use_posterior: bool,
      expected_result: np.ndarray,
      non_media_baseline_values: Sequence[float | str] | None,
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_non_media
    )
    outcome = self.analyzer_non_media.incremental_outcome(
        use_posterior=use_posterior,
        non_media_baseline_values=non_media_baseline_values,
        include_non_paid_channels=True,
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(expected_result),
        rtol=1e-2,
        atol=1e-2,
    )

  def test_incremental_outcome_wrong_non_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_non_media_treatments.shape must match non_media_treatments.shape",
    ):
      self.analyzer_non_media.incremental_outcome(
          new_data=analyzer.DataTensors(
              non_media_treatments=self.meridian_non_media.population
          ),
      )

  def test_incremental_outcome_wrong_baseline_types_shape_raises_exception(
      self,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The number of non-media channels (4) does not match the number of"
        " baseline types (3).",
    ):
      self.analyzer_non_media.incremental_outcome(
          non_media_baseline_values=["min", "max", "min"],
          include_non_paid_channels=True,
      )

  def test_incremental_outcome_wrong_baseline_type_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid non_media_baseline_values value: 'wrong'. Only float numbers"
        " and strings 'min' and 'max' are supported.",
    ):
      self.analyzer_non_media.incremental_outcome(
          non_media_baseline_values=["min", "max", "max", "wrong"],
          include_non_paid_channels=True,
      )

  def test_response_curves_use_only_paid_channels(self):
    response_curve_data = self.analyzer_non_media.response_curves(
        by_reach=False
    )
    self.assertIn(constants.CHANNEL, response_curve_data.coords)
    self.assertLen(
        response_curve_data.coords[constants.CHANNEL],
        len(
            self.analyzer_non_media._meridian.input_data.get_all_paid_channels()
        ),
    )

  def test_summary_metrics_with_non_media_baseline_values(self):
    # Call summary_metrics with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer_non_media,
        "compute_incremental_outcome_aggregate",
        wraps=self.analyzer_non_media.compute_incremental_outcome_aggregate,
    ) as mock_compute_incremental_outcome_aggregate:
      self.analyzer_non_media.summary_metrics(
          include_non_paid_channels=True,
          non_media_baseline_values=[0.0, "max", 1.0, "min"],
      )

    # Assert that _compute_incremental_outcome_aggregate was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_compute_incremental_outcome_aggregate.call_count, 2)

    # Both calls with include_non_paid_channels=True and given
    # non_media_baseline_values
    for call in mock_compute_incremental_outcome_aggregate.call_args_list:
      _, kwargs = call
      self.assertEqual(kwargs["include_non_paid_channels"], True)
      self.assertEqual(
          kwargs["non_media_baseline_values"], [0.0, "max", 1.0, "min"]
      )

  def test_baseline_summary_metrics_with_non_media_baseline_values(self):
    # Call baseline_summary_metrics with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer_non_media,
        "_calculate_baseline_expected_outcome",
        wraps=self.analyzer_non_media._calculate_baseline_expected_outcome,
    ) as mock_calculate_baseline_expected_outcome:
      self.analyzer_non_media.baseline_summary_metrics(
          non_media_baseline_values=[0.0, "max", 1.0, "min"],
      )

    # Assert that _calculate_baseline_expected_outcome was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_calculate_baseline_expected_outcome.call_count, 2)
    for call in mock_calculate_baseline_expected_outcome.call_args_list:
      _, kwargs = call
      self.assertEqual(
          kwargs["non_media_baseline_values"], [0.0, "max", 1.0, "min"]
      )

  def test_expected_vs_actual_with_non_media_baseline_values(self):
    # Call expected_vs_actual with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer_non_media,
        "_calculate_baseline_expected_outcome",
        wraps=self.analyzer_non_media._calculate_baseline_expected_outcome,
    ) as mock_calculate_baseline_expected_outcome:
      self.analyzer_non_media.expected_vs_actual_data(
          non_media_baseline_values=[0.0, "max", 1.0, "min"],
      )

    # Assert that _calculate_baseline_expected_outcome was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_calculate_baseline_expected_outcome.call_count, 1)
    _, kwargs = mock_calculate_baseline_expected_outcome.call_args
    self.assertEqual(
        kwargs["non_media_baseline_values"], [0.0, "max", 1.0, "min"]
    )


class AnalyzerOrganicMediaTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerOrganicMediaTest, cls).setUpClass()
    # Input data resulting in revenue computation.
    cls.input_data_non_paid = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    cls.not_lagged_input_data_non_paid = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_non_paid = model.Meridian(
        input_data=cls.input_data_non_paid, model_spec=model_spec
    )
    cls.analyzer_non_paid = analyzer.Analyzer(cls.meridian_non_paid)

    cls.inference_data_non_paid = _build_inference_data(
        _TEST_SAMPLE_PRIOR_NON_PAID_PATH,
        _TEST_SAMPLE_POSTERIOR_NON_PAID_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_non_paid),
        )
    )

  @parameterized.product(
      new_tensors_names=[
          [],
          [constants.MEDIA, constants.REACH, constants.FREQUENCY],
          [
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.CONTROLS,
          ],
          [
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.ORGANIC_MEDIA,
              constants.ORGANIC_REACH,
              constants.ORGANIC_FREQUENCY,
              constants.NON_MEDIA_TREATMENTS,
          ],
          [
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.REVENUE_PER_KPI,
          ],
      ],
      require_non_paid_channels=[True, False],
      require_controls=[True, False],
      require_revenue_per_kpi=[True, False],
  )
  def test_fill_missing_data_tensors(
      self,
      new_tensors_names: Sequence[str],
      require_non_paid_channels: bool,
      require_controls: bool,
      require_revenue_per_kpi: bool,
  ):
    data1 = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        n_non_media_channels=_N_NON_MEDIA_CHANNELS,
        n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
        n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
        seed=1,
    )

    if not new_tensors_names:
      new_data = None
    else:
      tensors = {}
      for tensor_name in new_tensors_names:
        tensors[tensor_name] = getattr(data1, tensor_name)
      new_data = analyzer.DataTensors(**tensors)

    required_tensors_names = [
        constants.MEDIA,
        constants.REACH,
        constants.FREQUENCY,
    ]
    if require_controls:
      required_tensors_names.append(constants.CONTROLS)
    if require_non_paid_channels:
      required_tensors_names.append(constants.ORGANIC_MEDIA)
      required_tensors_names.append(constants.ORGANIC_REACH)
      required_tensors_names.append(constants.ORGANIC_FREQUENCY)
      required_tensors_names.append(constants.NON_MEDIA_TREATMENTS)
    if require_revenue_per_kpi:
      required_tensors_names.append(constants.REVENUE_PER_KPI)

    filled_tensors = self.analyzer_non_paid._fill_missing_data_tensors(
        new_data=new_data,
        required_tensors_names=required_tensors_names,
    )
    for tensor_name in required_tensors_names:
      if tensor_name in new_tensors_names:
        self.assertAllClose(
            getattr(filled_tensors, tensor_name),
            getattr(data1, tensor_name),
            rtol=1e-4,
            atol=1e-4,
        )
      else:
        self.assertAllClose(
            getattr(filled_tensors, tensor_name),
            getattr(self.input_data_non_paid, tensor_name),
            rtol=1e-4,
            atol=1e-4,
        )

  @parameterized.product(
      selected_geos=[["geo_1", "geo_3"]],
      selected_times=[
          ["2021-04-19", "2021-09-13", "2021-12-13"],
      ],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      # (media, rf, non_media_treatments, organic_media, organic_rf)
      # (3, 0, 0, 0, 0) -> 3
      # (0, 2, 0, 0, 0) -> 2
      # (3, 2, 0, 0, 0) -> 5
      # (3, 2, 4, 4, 1) -> 14
      selected_channels=[
          (_N_MEDIA_CHANNELS, 0, 0, 0, 0),
          (0, _N_RF_CHANNELS, 0, 0, 0),
          (_N_MEDIA_CHANNELS, _N_RF_CHANNELS, 0, 0, 0),
          (
              _N_MEDIA_CHANNELS,
              _N_RF_CHANNELS,
              _N_NON_MEDIA_CHANNELS,
              _N_ORGANIC_MEDIA_CHANNELS,
              _N_ORGANIC_RF_CHANNELS,
          ),
      ],
  )
  def test_filter_and_aggregate_geos_and_times_accepts_channel_shape(
      self,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_channels: tuple[int, int, int, int, int],
  ):
    (
        n_media_channels,
        n_rf_channels,
        n_non_media_channels,
        n_organic_media_channels,
        n_organic_rf_channels,
    ) = selected_channels
    tensors = []
    if n_media_channels > 0:
      tensors.append(
          tf.convert_to_tensor(self.not_lagged_input_data_non_paid.media)
      )
    if n_rf_channels > 0:
      tensors.append(
          tf.convert_to_tensor(self.not_lagged_input_data_non_paid.reach)
      )
    if n_non_media_channels > 0:
      tensors.append(
          tf.convert_to_tensor(
              self.not_lagged_input_data_non_paid.non_media_treatments
          )
      )
    if n_organic_media_channels > 0:
      tensors.append(
          tf.convert_to_tensor(
              self.not_lagged_input_data_non_paid.organic_media
          )
      )
    if n_organic_rf_channels > 0:
      tensors.append(
          tf.convert_to_tensor(
              self.not_lagged_input_data_non_paid.organic_reach
          )
      )
    tensor = tf.concat(tensors, axis=-1)
    modified_tensor = (
        self.analyzer_non_paid.filter_and_aggregate_geos_and_times(
            tensor,
            selected_geos=selected_geos,
            selected_times=selected_times,
            aggregate_geos=aggregate_geos,
            aggregate_times=aggregate_times,
            flexible_time_dim=False,
            has_media_dim=True,
        )
    )
    expected_shape = ()
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      if selected_times is not None:
        if all(isinstance(time, bool) for time in selected_times):
          n_times = sum(selected_times)
        else:
          n_times = len(selected_times)
      else:
        n_times = _N_TIMES
      expected_shape += (n_times,)
    expected_shape += (
        n_media_channels
        + n_rf_channels
        + n_non_media_channels
        + n_organic_media_channels
        + n_organic_rf_channels,
    )
    self.assertEqual(modified_tensor.shape, expected_shape)

  @parameterized.product(
      # (media, rf, non_media_treatments, organic_media, organic_rf[, all])
      # (3, 0, 0, 0, 0[, 1]) -> 3, 4
      # (0, 2, 0, 0, 0[, 1]) -> 2, 3
      # (3, 2, 0, 0, 0[, 1]) -> 5, 6
      # (3, 2, 4, 4, 1[, 1]) -> 14, 15
      num_channels=[1, 7, 8, 9, 10, 11, 12, 13],
  )
  def test_filter_and_aggregate_geos_and_times_wrong_channels_fails(
      self,
      num_channels=int,
  ):
    self.assertNotIn(
        num_channels,
        [
            _N_MEDIA_CHANNELS,
            _N_RF_CHANNELS,
            _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
            _N_MEDIA_CHANNELS
            + _N_RF_CHANNELS
            + _N_NON_MEDIA_CHANNELS
            + _N_ORGANIC_MEDIA_CHANNELS
            + _N_ORGANIC_RF_CHANNELS,
        ],
    )
    tensor = tf.concat(
        [
            self.not_lagged_input_data_non_paid.media,
            self.not_lagged_input_data_non_paid.reach,
            self.not_lagged_input_data_non_paid.non_media_treatments,
            self.not_lagged_input_data_non_paid.organic_media,
            self.not_lagged_input_data_non_paid.organic_reach,
        ],
        axis=-1,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The tensor must have shape [..., n_geos, n_times, n_channels] or [...,"
        " n_geos, n_times] if `flexible_time_dim=False`.",
    ):
      self.analyzer_non_paid.filter_and_aggregate_geos_and_times(
          tensor[..., :num_channels],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_outcome=test_utils.INC_OUTCOME_NON_PAID_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=test_utils.INC_OUTCOME_NON_PAID_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_organic_media(
      self,
      use_posterior: bool,
      expected_outcome: np.ndarray,
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_non_paid
    )
    outcome = self.analyzer_non_paid.incremental_outcome(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        outcome,
        tf.convert_to_tensor(expected_outcome),
        rtol=1e-2,
        atol=1e-2,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_outcome_non_paid_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    outcome = self.analyzer_non_paid.expected_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=geos_to_include,
        selected_times=times_to_include,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(geos_to_include),) if geos_to_include is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_shape += (
          (len(times_to_include),)
          if times_to_include is not None
          else (_N_TIMES,)
      )
    self.assertEqual(outcome.shape, expected_shape)

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      include_non_paid_channels=[False, True],
  )
  def test_all_channels_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      include_non_paid_channels: bool,
  ):
    model_analyzer = self.analyzer_non_paid
    channels = (
        self.analyzer_non_paid._meridian.input_data.get_all_channels()
        if include_non_paid_channels
        else self.analyzer_non_paid._meridian.input_data.get_all_paid_channels()
    )

    media_summary = model_analyzer.summary_metrics(
        confidence_level=0.8,
        marginal_roi_by_reach=False,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        include_non_paid_channels=include_non_paid_channels,
    )
    expected_channel_shape = ()
    if not aggregate_geos:
      expected_channel_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      expected_channel_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )

    # (ch_1, ch_2, ..., All_Channels, [mean, median, ci_lo, ci_hi],
    # [prior, posterior])
    expected_channel_shape += (len(channels) + 1,)
    expected_shape = expected_channel_shape + (
        4,
        2,
    )
    self.assertEqual(media_summary.incremental_outcome.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)
    if aggregate_times:
      self.assertEqual(media_summary.effectiveness.shape, expected_shape)
    else:
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)

  def test_baseline_summary_returns_correct_values(self):
    baseline_summary = self.analyzer_non_paid.baseline_summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertIsNotNone(baseline_summary.baseline_outcome)
    self.assertIsNotNone(baseline_summary.pct_of_contribution)
    self.assertAllClose(
        baseline_summary.baseline_outcome,
        test_utils.SAMPLE_BASELINE_EXPECTED_OUTCOME_NON_PAID,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        baseline_summary.pct_of_contribution,
        test_utils.SAMPLE_BASELINE_PCT_OF_CONTRIBUTION_NON_PAID,
        atol=1e-2,
        rtol=1e-2,
    )


class AnalyzerNotFittedTest(absltest.TestCase):

  def test_rhat_summary_media_and_rf_pre_fitting_raises_exception(self):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    type(not_fitted_mmm).inference_data = mock.PropertyMock(
        return_value=az.InferenceData()
    )
    not_fitted_analyzer = analyzer.Analyzer(not_fitted_mmm)
    with self.assertRaisesWithLiteralMatch(
        model.NotFittedModelError,
        "sample_posterior() must be called prior to calling this method.",
    ):
      not_fitted_analyzer.rhat_summary()


if __name__ == "__main__":
  absltest.main()
