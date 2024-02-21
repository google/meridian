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
from meridian.model import spec
import numpy as np
import tensorflow as tf
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

    cls.input_data_media_and_rf = data_test_utils.sample_input_data(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    cls.input_data_media_only = data_test_utils.sample_input_data(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=0,
    )
    cls.input_data_rf_only = data_test_utils.sample_input_data(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_media_and_rf = model.Meridian(
        input_data=cls.input_data_media_and_rf, model_spec=model_spec
    )
    cls.meridian_media_only = model.Meridian(
        input_data=cls.input_data_media_only, model_spec=model_spec
    )
    cls.meridian_rf_only = model.Meridian(
        input_data=cls.input_data_rf_only, model_spec=model_spec
    )

    cls.analyzer_media_and_rf = analyzer.Analyzer(cls.meridian_media_and_rf)
    cls.analyzer_media_only = analyzer.Analyzer(cls.meridian_media_only)
    cls.analyzer_rf_only = analyzer.Analyzer(cls.meridian_rf_only)

    cls.inference_data_media_and_rf = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH,
    )
    cls.inference_data_media_only = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH,
    )
    cls.inference_data_rf_only = _build_inference_data(
        _TEST_SAMPLE_PRIOR_RF_ONLY_PATH,
        _TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH,
    )

  def test_filter_and_aggregate_geos_and_times_incorrect_tensor_shape(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The tensor must have shape [..., n_geos, n_times, n_channels] or"
        " [..., n_geos, n_times].",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.population),
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_geos(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`selected_geos` must match the geo dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_geos=["random_geo"],
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_times(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tf.convert_to_tensor(self.input_data_media_only.media_spend),
          selected_times=["random_time"],
      )

  @parameterized.product(
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
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
      expected_shape += (
          (len(selected_times),) if selected_times is not None else (_N_TIMES,)
      )
    expected_shape += (_N_MEDIA_CHANNELS,)
    self.assertEqual(modified_tensor.shape, expected_shape)

  def test_expected_impact_wrong_controls_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_controls.shape must match controls.shape",
    ):
      self.analyzer_media_and_rf.expected_impact(
          new_controls=self.meridian_media_and_rf.population,
      )

  def test_expected_impact_wrong_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_media.shape must match media.shape",
    ):
      self.analyzer_media_and_rf.expected_impact(
          new_media=self.meridian_media_and_rf.population,
      )

  def test_expected_impact_wrong_reach_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_reach.shape must match reach.shape",
    ):
      self.analyzer_media_and_rf.expected_impact(
          new_reach=self.meridian_media_and_rf.population,
      )

  def test_expected_impact_wrong_frequency_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_frequency.shape must match frequency.shape",
    ):
      self.analyzer_media_and_rf.expected_impact(
          new_frequency=self.meridian_media_and_rf.population,
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_impact_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    impact = self.analyzer_media_and_rf.expected_impact(
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
    self.assertEqual(impact.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_impact_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    impact = self.analyzer_media_only.expected_impact(
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
    self.assertEqual(impact.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_impact_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    impact = self.analyzer_rf_only.expected_impact(
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
    self.assertEqual(impact.shape, expected_shape)

  def test_incremental_impact_wrong_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_media.shape must match media.shape",
    ):
      self.analyzer_media_and_rf.incremental_impact(
          new_media=self.meridian_media_and_rf.population,
      )

  def test_incremental_impact_wrong_reach_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_reach.shape must match reach.shape",
    ):
      self.analyzer_media_and_rf.incremental_impact(
          new_reach=self.meridian_media_and_rf.population,
      )

  def test_incremental_impact_wrong_frequency_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_frequency.shape must match frequency.shape",
    ):
      self.analyzer_media_and_rf.incremental_impact(
          new_frequency=self.meridian_media_and_rf.population,
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_incremental_impact_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    impact = self.analyzer_media_and_rf.incremental_impact(
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
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(impact.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_incremental_impact_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    impact = self.analyzer_media_only.incremental_impact(
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
    self.assertEqual(impact.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_incremental_impact_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    impact = self.analyzer_rf_only.incremental_impact(
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
    self.assertEqual(impact.shape, expected_shape)

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_impact=test_utils.INC_IMPACT_MEDIA_AND_RF_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_impact=test_utils.INC_IMPACT_MEDIA_AND_RF_USE_POSTERIOR,
      ),
  )
  def test_incremental_impact_media_and_rf(
      self,
      use_posterior: bool,
      expected_impact: tuple[float, ...],
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    impact = self.analyzer_media_and_rf.incremental_impact(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        impact,
        tf.convert_to_tensor(expected_impact),
        rtol=1e-3,
        atol=1e-3,
    )

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_impact=test_utils.INC_IMPACT_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_impact=test_utils.INC_IMPACT_MEDIA_ONLY_USE_POSTERIOR,
      ),
  )
  def test_incremental_impact_media_only(
      self,
      use_posterior: bool,
      expected_impact: tuple[float, ...],
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    impact = self.analyzer_media_only.incremental_impact(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        impact,
        tf.convert_to_tensor(expected_impact),
        rtol=1e-3,
        atol=1e-3,
    )

  # The purpose of this test is to prevent accidental logic change.
  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_impact=test_utils.INC_IMPACT_RF_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_impact=test_utils.INC_IMPACT_RF_ONLY_USE_POSTERIOR,
      ),
  )
  def test_incremental_impact_rf_only(
      self,
      use_posterior: bool,
      expected_impact: tuple[float, ...],
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    impact = self.analyzer_rf_only.incremental_impact(
        use_posterior=use_posterior,
    )
    self.assertAllClose(
        impact,
        tf.convert_to_tensor(expected_impact),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
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
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
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
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(mroi.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
  ):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    mroi = self.analyzer_media_only.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
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
    self.assertEqual(mroi.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
  )
  def test_marginal_roi_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
  ):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    mroi = self.analyzer_rf_only.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
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
  def test_marginal_roi_media_and_rf_use_prior(
      self, use_posterior: bool, by_reach: bool, expected_mroi: tuple[float]
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
      self, use_posterior: bool, by_reach: bool, expected_mroi: tuple[float]
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
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
      self, use_posterior: bool, by_reach: bool, expected_mroi: tuple[float]
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
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

  def test_roi_wrong_media_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_media.shape must match media.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_media=self.meridian_media_and_rf.population,
      )

  def test_roi_wrong_media_spend_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "new_media_spend.shape: (5,) must match either (3,) or (5, 49, 3).",
    ):
      self.analyzer_media_and_rf.roi(
          new_media_spend=self.meridian_media_and_rf.population,
      )

  def test_roi_wrong_reach_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_reach.shape must match reach.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_reach=self.meridian_media_and_rf.population,
      )

  def test_roi_wrong_frequency_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "new_frequency.shape must match frequency.shape",
    ):
      self.analyzer_media_and_rf.roi(
          new_frequency=self.meridian_media_and_rf.population,
      )

  def test_roi_wrong_rf_spend_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "new_rf_spend.shape: (5,) must match either (2,) or (5, 49, 2).",
    ):
      self.analyzer_media_and_rf.roi(
          new_rf_spend=self.meridian_media_and_rf.population,
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    roi = self.analyzer_media_and_rf.roi(
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
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(roi.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_media_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    roi = self.analyzer_media_only.roi(
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
    self.assertEqual(roi.shape, expected_shape)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_roi_rf_only_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    roi = self.analyzer_rf_only.roi(
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
    self.assertEqual(roi.shape, expected_shape)

  def test_roi_media_and_rf_default_returns_correct_value(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    roi = self.analyzer_media_and_rf.roi()
    total_spend = (
        self.analyzer_media_and_rf.filter_and_aggregate_geos_and_times(
            self.meridian_media_and_rf.total_spend
        )
    )
    expeted_roi = self.analyzer_media_and_rf.incremental_impact() / total_spend
    self.assertAllClose(expeted_roi, roi)

  def test_roi_media_only_default_returns_correct_value(self):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    roi = self.analyzer_media_only.roi()
    total_spend = self.analyzer_media_only.filter_and_aggregate_geos_and_times(
        self.meridian_media_only.media_spend
    )
    expected_roi = self.analyzer_media_only.incremental_impact() / total_spend
    self.assertAllClose(expected_roi, roi)

  def test_roi_rf_only_default_returns_correct_value(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    roi = self.analyzer_rf_only.roi()
    total_spend = self.analyzer_rf_only.filter_and_aggregate_geos_and_times(
        self.meridian_rf_only.rf_spend
    )
    expeted_roi = self.analyzer_rf_only.incremental_impact() / total_spend
    self.assertAllClose(expeted_roi, roi)

  def test_roi_zero_media_returns_zero(self):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    new_media = tf.zeros_like(self.meridian_media_only.media, dtype=tf.float32)
    roi = self.analyzer_media_only.roi(new_media=new_media)
    self.assertAllClose(
        roi, tf.zeros((_N_CHAINS, _N_KEEP, _N_MEDIA_CHANNELS)), atol=2e-6
    )

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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    media_summary = self.analyzer_media_and_rf.media_summary_metrics(
        confidence_level=0.9,
        marginal_roi_by_reach=False,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    media_summary_by_freq = self.analyzer_media_and_rf.media_summary_metrics(
        confidence_level=0.9,
        marginal_roi_by_reach=True,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertIsNotNone(media_summary.impressions)
    self.assertIsNotNone(media_summary.spend)
    self.assertIsNotNone(media_summary.pct_of_spend)
    self.assertIsNotNone(media_summary.spend_per_impression)
    self.assertIsNotNone(media_summary.incremental_impact)
    self.assertIsNotNone(media_summary.pct_of_contribution)
    self.assertIsNotNone(media_summary.roi)
    self.assertIsNotNone(media_summary.effectiveness)
    self.assertIsNotNone(media_summary.mroi)
    self.assertAllClose(
        media_summary.impressions, test_utils.SAMPLE_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.pct_of_impressions, test_utils.SAMPLE_PCT_OF_IMPRESSIONS
    )
    self.assertAllClose(
        media_summary.spend, test_utils.SAMPLE_SPEND, atol=1e-2, rtol=1e-2
    )
    self.assertAllClose(
        media_summary.pct_of_spend,
        test_utils.SAMPLE_PCT_OF_SPEND,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        media_summary.spend_per_impression,
        test_utils.SAMPLE_SPEND_PER_IMPRESSION,
    )
    self.assertAllClose(
        media_summary.incremental_impact,
        test_utils.SAMPLE_INCREMENTAL_IMPACT,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        media_summary.pct_of_contribution,
        test_utils.SAMPLE_PCT_OF_CONTRIBUTION,
        atol=1e-2,
        rtol=1e-2,
    )
    self.assertAllClose(
        media_summary.roi, test_utils.SAMPLE_ROI, atol=1e-3, rtol=1e-3
    )
    self.assertAllClose(
        media_summary.effectiveness,
        test_utils.SAMPLE_EFFECTIVENESS,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertAllClose(
        media_summary.mroi, test_utils.SAMPLE_MROI, atol=1e-3, rtol=1e-3
    )


  def test_by_reach_returns_correct_values(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    mroi = self.analyzer_rf_only.marginal_roi(
        use_posterior=True,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
        by_reach=True,
    )
    roi = self.analyzer_rf_only.roi(
        use_posterior=True,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertAllClose(
        mroi,
        roi,
        atol=1e-2,
        rtol=1e-2,
    )
    # media_summary = self.analyzer_rf_only.media_summary_metrics(
    #     confidence_level=0.9,
    #     marginal_roi_by_reach=True,
    #     aggregate_geos=True,
    #     aggregate_times=True,
    #     selected_geos=None,
    #     selected_times=None,
    # )
    # self.assertAllClose(
    #     media_summary.mroi,
    #     media_summary.roi,
    #     atol=1e-3,
    #     rtol=1e-3,
    # )

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      channel_types=["media_and_rf", "media_only", "rf_only"],
  )
  def test_media_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      channel_types: str,
  ):
    if channel_types == "media_and_rf":
      type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
          return_value=self.inference_data_media_and_rf
      )
      analyzer = self.analyzer_media_and_rf
      num_channels = _N_MEDIA_CHANNELS + _N_RF_CHANNELS
    elif channel_types == "media_only":
      type(self.meridian_media_only).inference_data = mock.PropertyMock(
          return_value=self.inference_data_media_only
      )
      analyzer = self.analyzer_media_only
      num_channels = _N_MEDIA_CHANNELS
    else:  # "rf_only"
      type(self.meridian_rf_only).inference_data = mock.PropertyMock(
          return_value=self.inference_data_rf_only
      )
      analyzer = self.analyzer_rf_only
      num_channels = _N_RF_CHANNELS

    media_summary = analyzer.media_summary_metrics(
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

    # (ch_1, ch_2, ..., Total_media, [mean, ci_lo, ci_hi], [prior, posterior])
    expected_channel_shape += (num_channels + 1,)
    expected_shape = expected_channel_shape + (
        3,
        2,
    )
    self.assertEqual(media_summary.impressions.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.pct_of_impressions.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.pct_of_spend.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.spend_per_impression.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.incremental_impact.shape, expected_shape)
    self.assertEqual(media_summary.pct_of_contribution.shape, expected_shape)
    self.assertEqual(media_summary.roi.shape, expected_shape)
    self.assertEqual(media_summary.effectiveness.shape, expected_shape)
    self.assertEqual(media_summary.mroi.shape, expected_shape)

  def test_media_summary_warns_if_time_not_aggregated(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      media_summary = self.analyzer_rf_only.media_summary_metrics(
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
          "ROI, mROI, and Effectiveness are not reported because they do not"
          " have a clear interpretation by time period.",
          str(w[0].message),
      )
      self.assertTrue((media_summary.roi.isnull()).all())
      self.assertTrue((media_summary.mroi.isnull()).all())
      self.assertTrue((media_summary.effectiveness.isnull()).all())

  def test_optimal_frequency_data_media_and_rf_correct(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    optimal_frequency_dataset = self.analyzer_media_and_rf.optimal_freq(
        freq_grid=list(np.arange(1, 7.1, 0.1)),
        confidence_level=0.9,
        use_posterior=True,
    )

    roi_by_frequency_df = (
        optimal_frequency_dataset[[constants.ROI]]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[constants.RF_CHANNEL, constants.FREQUENCY],
            columns=constants.METRIC,
            values=constants.ROI,
        )
        .reset_index()
    )
    roi_by_frequency_df.rename(
        columns={constants.MEAN: constants.ROI}, inplace=True
    )
    optimal_freq_df = (
        optimal_frequency_dataset[[constants.OPTIMAL_FREQUENCY]]
        .to_dataframe()
        .reset_index()
    )
    final_df = roi_by_frequency_df.merge(
        optimal_freq_df, on=constants.RF_CHANNEL
    )

    self.assertEqual(
        list(final_df.columns),
        [
            constants.RF_CHANNEL,
            constants.FREQUENCY,
            constants.CI_HI,
            constants.CI_LO,
            constants.ROI,
            constants.OPTIMAL_FREQUENCY,
        ],
    )

  def test_optimal_frequency_data_media_only_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "Must have at least one channel with reach and frequency data.",
    ):
      self.analyzer_media_only.optimal_freq()

  def test_optimal_frequency_data_rf_only_correct(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    optimal_frequency_dataset = self.analyzer_rf_only.optimal_freq(
        freq_grid=list(np.arange(1, 7.1, 0.1)),
        confidence_level=0.9,
        use_posterior=True,
    )

    roi_by_frequency_df = (
        optimal_frequency_dataset[[constants.ROI]]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[constants.RF_CHANNEL, constants.FREQUENCY],
            columns=constants.METRIC,
            values=constants.ROI,
        )
        .reset_index()
    )
    roi_by_frequency_df.rename(
        columns={constants.MEAN: constants.ROI}, inplace=True
    )
    optimal_freq_df = (
        optimal_frequency_dataset[[constants.OPTIMAL_FREQUENCY]]
        .to_dataframe()
        .reset_index()
    )
    final_df = roi_by_frequency_df.merge(
        optimal_freq_df, on=constants.RF_CHANNEL
    )

    self.assertEqual(
        list(final_df.columns),
        [
            constants.RF_CHANNEL,
            constants.FREQUENCY,
            constants.CI_HI,
            constants.CI_LO,
            constants.ROI,
            constants.OPTIMAL_FREQUENCY,
        ],
    )

  def test_optimal_frequency_freq_grid(self):
    max_freq = np.max(np.array(self.analyzer_rf_only._meridian.frequency))
    freq_grid = list(np.arange(1, max_freq, 0.1))
    roi = np.zeros(
        (len(freq_grid), self.analyzer_rf_only._meridian.n_rf_channels, 3)
    )
    for i, freq in enumerate(freq_grid):
      new_frequency = (
          tf.ones_like(self.analyzer_rf_only._meridian.frequency) * freq
      )
      new_reach = (
          self.analyzer_rf_only._meridian.frequency
          * self.analyzer_rf_only._meridian.reach
          / new_frequency
      )
      dim_kwargs = {
          "selected_geos": None,
          "selected_times": None,
          "aggregate_geos": True,
          "aggregate_times": True,
      }
      roi_temp = self.analyzer_rf_only.roi(
          new_reach=new_reach,
          new_frequency=new_frequency,
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

  def test_r_hat_summary_media_and_rf_pre_fitting_raises_exception(self):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    type(not_fitted_mmm).inference_data = mock.PropertyMock(
        return_value=az.InferenceData()
    )
    not_fitted_analyzer = analyzer.Analyzer(not_fitted_mmm)
    with self.assertRaisesWithLiteralMatch(
        model.NotFittedModelError,
        "sample_posterior() must be called prior to calling this method.",
    ):
      not_fitted_analyzer.r_hat_summary()

  def test_r_hat_summary_media_and_rf_correct(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    r_hat_summary = self.analyzer_media_and_rf.r_hat_summary()
    self.assertEqual(r_hat_summary.shape, (20, 7))
    self.assertSetEqual(
        set(r_hat_summary.param),
        set(
            constants.COMMON_PARAMETER_NAMES
            + constants.MEDIA_PARAMETER_NAMES
            + constants.RF_PARAMETER_NAMES
        )
        - set([constants.SLOPE_M]),
    )

  def test_r_hat_summary_media_only_correct(self):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    r_hat_summary = self.analyzer_media_only.r_hat_summary()
    self.assertEqual(r_hat_summary.shape, (13, 7))
    self.assertSetEqual(
        set(r_hat_summary.param),
        set(constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES)
        - set([constants.SLOPE_M]),
    )

  def test_r_hat_summary_rf_only_correct(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    r_hat_summary = self.analyzer_rf_only.r_hat_summary()
    self.assertEqual(r_hat_summary.shape, (14, 7))
    self.assertSetEqual(
        set(r_hat_summary.param),
        set(constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES),
    )

  def test_predictive_accuracy_without_holdout_id_columns_correct(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    predictive_accuracy_dataset = (
        self.analyzer_media_and_rf.predictive_accuracy()
    )
    self.assertAllEqual(
        list(predictive_accuracy_dataset[constants.METRIC].values),
        [constants.R_SQUARED, constants.MAPE, constants.WMAPE],
    )
    self.assertAllEqual(
        list(predictive_accuracy_dataset[constants.GEO_GRANULARITY].values),
        [constants.GEO, constants.NATIONAL],
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )
    self.assertAllEqual(
        df.columns,
        [constants.METRIC, constants.GEO_GRANULARITY, constants.VALUE],
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
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
    self.assertAllEqual(
        list(df.columns),
        [
            constants.METRIC,
            constants.GEO_GRANULARITY,
            constants.EVALUATION_SET_VAR,
            constants.VALUE,
        ],
    )

  def test_predictive_accuracy_holdout_id_values_correct(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    n_geos = len(self.input_data_media_and_rf.geo)
    n_times = len(self.input_data_media_and_rf.time)
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
    self.assertAllClose(
        list(df[constants.VALUE]),
        test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID,
        atol=2e-3,
    )

  def test_response_curves_returns_correct_data(self):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    response_curve_data = self.analyzer_media_only.response_curves(
        by_reach=False
    )
    self.assertEqual(
        list(response_curve_data.coords.keys()),
        [constants.CHANNEL, constants.METRIC, constants.SPEND_MULTIPLIER],
    )
    self.assertEqual(
        list(response_curve_data.data_vars.keys()),
        [constants.SPEND, constants.INCREMENTAL_IMPACT, constants.ROI],
    )
    response_curves_df = (
        response_curve_data[[constants.SPEND, constants.INCREMENTAL_IMPACT]]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                constants.CHANNEL,
                constants.SPEND,
                constants.SPEND_MULTIPLIER,
            ],
            columns=constants.METRIC,
            values=constants.INCREMENTAL_IMPACT,
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

  def test_response_curves_check_roi_change(
      self,
  ):
    type(self.meridian_media_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_only
    )
    spend_multipliers = list(np.arange(0, 2.2, 0.2))
    incimpact = np.zeros((
        len(spend_multipliers),
        len(self.analyzer_media_only._meridian.input_data.get_all_channels()),
        3,  # Last dimension = 3 for the mean, ci_lo and ci_hi.
    ))

    if (
        self.analyzer_media_only._meridian.n_media_channels > 0
        and self.analyzer_media_only._meridian.n_rf_channels > 0
    ):
      spend = tf.concat(
          [
              self.analyzer_media_only._meridian.media_spend,
              self.analyzer_media_only._meridian.rf_spend,
          ],
          axis=-1,
      )
    elif self.analyzer_media_only._meridian.n_media_channels > 0:
      spend = self.analyzer_media_only._meridian.media_spend
    else:
      spend = self.analyzer_media_only._meridian.rf_spend
    if tf.rank(spend).numpy() == 3:
      spend = self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          tensor=spend,
          selected_geos=None,
          selected_times=None,
          aggregate_geos=True,
          aggregate_times=True,
      )
    spend_einsum = tf.einsum("k,m->km", np.array(spend_multipliers), spend)
    response_curve_roi = self.analyzer_media_only.response_curves(
        by_reach=False
    ).roi
    self.assertNotAllEqual(
        incimpact / spend_einsum[:, :, None], response_curve_roi
    )

  def test_response_curves_check_both_channel_types_returns_correct_spend(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    response_curve_data = self.analyzer_media_and_rf.response_curves(
        by_reach=False
    )
    response_data_spend = response_curve_data.spend.values

    media_summary_spend = self.analyzer_media_and_rf.media_summary_metrics(
        confidence_level=0.9,
        marginal_roi_by_reach=False,
    ).spend[:-1]
    self.assertAllEqual(
        media_summary_spend * 2,
        response_data_spend[-1],
    )

  def test_response_curves_check_only_rf_channels_returns_correct_spend(self):
    type(self.meridian_rf_only).inference_data = mock.PropertyMock(
        return_value=self.inference_data_rf_only
    )
    response_curve_data = self.analyzer_rf_only.response_curves(by_reach=False)
    response_data_spend = response_curve_data.spend.values

    media_summary_spend = self.analyzer_rf_only.media_summary_metrics(
        confidence_level=0.9,
        marginal_roi_by_reach=False,
    ).spend[:-1]
    self.assertAllEqual(
        media_summary_spend * 2,
        response_data_spend[-1],
    )

  def test_expected_vs_actual_correct_data(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    ds = self.analyzer_media_and_rf.expected_vs_actual_data()

    self.assertListEqual(
        list(ds.geo.values), list(self.input_data_media_and_rf.geo.values)
    )
    self.assertListEqual(
        list(ds.time.values), list(self.input_data_media_and_rf.time.values)
    )
    self.assertListEqual(
        list(ds.metric.values),
        [constants.MEAN, constants.CI_LO, constants.CI_HI],
    )

    expected_shape = (
        _N_GEOS,
        _N_TIMES,
        3,  # [mean, ci_lo, ci_hi]
    )
    self.assertEqual(ds.expected.shape, expected_shape)
    self.assertEqual(ds.baseline.shape, expected_shape)
    self.assertEqual(ds.actual.shape, (_N_GEOS, _N_TIMES))  # No ci_lo, ci_hi.
    self.assertEqual(ds.confidence_level, 0.9)

    self.assertTrue(
        np.all(
            ds.expected.sel(metric=constants.CI_HI)
            >= ds.expected.sel(metric=constants.MEAN)
        )
    )
    self.assertTrue(
        np.all(
            ds.expected.sel(metric=constants.CI_LO)
            <= ds.expected.sel(metric=constants.MEAN)
        )
    )
    self.assertTrue(
        np.all(
            ds.baseline.sel(metric=constants.CI_HI)
            >= ds.baseline.sel(metric=constants.MEAN)
        )
    )
    self.assertTrue(
        np.all(
            ds.baseline.sel(metric=constants.CI_LO)
            <= ds.baseline.sel(metric=constants.MEAN)
        )
    )

    self.assertTrue(np.all(ds.baseline < ds.expected))

    # Test the math for a sample of the actual impact metrics.
    self.assertAllClose(
        list(ds.actual.values[0, :5]),
        [178.26823, 121.92347, 164.19545, 27.069845, 176.33078],
        atol=1e-5,
    )

  def test_adstock_decay_dataframe(self):
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=0.9
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=0.9
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=0.9
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
    type(self.meridian_media_and_rf).inference_data = mock.PropertyMock(
        return_value=self.inference_data_media_and_rf
    )
    adstock_decay_dataframe = self.analyzer_media_and_rf.adstock_decay(
        confidence_level=0.9
    )
    is_true_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.IS_INT_TIME_UNIT]
    ]
    for i in range(len(is_true_df[constants.TIME_UNITS])):
      self.assertEqual(
          list(is_true_df[constants.TIME_UNITS])[i],
          int(list(is_true_df[constants.TIME_UNITS])[i]),
      )


if __name__ == "__main__":
  absltest.main()
