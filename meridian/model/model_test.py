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

"""Unit tests for model.py."""

import collections
from collections.abc import Collection, Sequence
import os
from unittest import mock
import warnings
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import test_utils
from meridian.model import adstock_hill
from meridian.model import knots as knots_module
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


def _convert_with_swap(array: xr.DataArray, n_burnin: int) -> tf.Tensor:
  """Converts a DataArray to a tf.Tensor with the correct MCMC format.

  This function converts a DataArray to tf.Tensor, swaps first two dimensions
  and adds the burnin part. This is needed to properly mock the
  _xla_windowed_adaptive_nuts() function output in the sample_posterior
  tests.

  Args:
    array: The array to be converted.
    n_burnin: The number of extra draws to be padded with as the 'burnin' part.

  Returns:
    A tensor in the same format as returned by the _xla_windowed_adaptive_nuts()
    function.
  """
  tensor = tf.convert_to_tensor(array)
  perm = [1, 0] + [i for i in range(2, len(tensor.shape))]
  transposed_tensor = tf.transpose(tensor, perm=perm)

  # Add the "burnin" part to the mocked output of _xla_windowed_adaptive_nuts
  # to make sure sample_posterior returns the correct "keep" part.
  if array.dtype == bool:
    pad_value = False
  else:
    pad_value = 0.0 if array.dtype.kind == "f" else 0

  burnin = tf.fill([n_burnin] + transposed_tensor.shape[1:], pad_value)
  return tf.concat(
      [burnin, transposed_tensor],
      axis=0,
  )


class ModelTest(tf.test.TestCase, parameterized.TestCase):
  # TODO(b/302713435): Update the sample data to span over 1 or 2 year(s).
  _TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
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
  _N_ADAPT = 2
  _N_BURNIN = 5
  _N_KEEP = 10
  _N_DRAWS = 10
  _N_GEOS = 5
  _N_GEOS_NATIONAL = 1
  _N_TIMES = 200
  _N_TIMES_SHORT = 49
  _N_MEDIA_TIMES = 203
  _N_MEDIA_TIMES_SHORT = 52
  _N_MEDIA_CHANNELS = 3
  _N_RF_CHANNELS = 2
  _N_CONTROLS = 2
  _ROI_CALIBRATION_PERIOD = tf.cast(
      tf.ones((_N_MEDIA_TIMES_SHORT, _N_MEDIA_CHANNELS)),
      dtype=tf.bool,
  )
  _RF_ROI_CALIBRATION_PERIOD = tf.cast(
      tf.ones((_N_MEDIA_TIMES_SHORT, _N_RF_CHANNELS)),
      dtype=tf.bool,
  )

  def setUp(self):
    super().setUp()
    self.input_data_non_revenue_no_revenue_per_kpi = (
        test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )
    self.national_input_data_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_NATIONAL,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    self.national_input_data_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_NATIONAL,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            seed=0,
        )
    )

    test_prior_media_and_rf = xr.open_dataset(
        self._TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH
    )
    test_prior_media_only = xr.open_dataset(
        self._TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH
    )
    test_prior_rf_only = xr.open_dataset(self._TEST_SAMPLE_PRIOR_RF_ONLY_PATH)
    self.test_dist_media_and_rf = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_media_and_rf[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })
    self.test_dist_media_only = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_media_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    })
    self.test_dist_rf_only = collections.OrderedDict({
        param: tf.convert_to_tensor(test_prior_rf_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })

    test_posterior_media_and_rf = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH
    )
    test_posterior_media_only = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH
    )
    test_posterior_rf_only = xr.open_dataset(
        self._TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH
    )
    posterior_params_to_tensors_media_and_rf = {
        param: _convert_with_swap(
            test_posterior_media_and_rf[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    posterior_params_to_tensors_media_only = {
        param: _convert_with_swap(
            test_posterior_media_only[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    }
    posterior_params_to_tensors_rf_only = {
        param: _convert_with_swap(
            test_posterior_rf_only[param], n_burnin=self._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    self.test_posterior_states_media_and_rf = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_and_rf)
    self.test_posterior_states_media_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_only)
    self.test_posterior_states_rf_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_rf_only)

    test_trace = xr.open_dataset(self._TEST_SAMPLE_TRACE_PATH)
    self.test_trace = {
        param: _convert_with_swap(test_trace[param], n_burnin=self._N_BURNIN)
        for param in test_trace.data_vars
    }

  def test_init_with_default_parameters_works(self):
    meridian = model.Meridian(input_data=self.input_data_with_media_only)

    # Compare input data.
    self.assertEqual(meridian.input_data, self.input_data_with_media_only)

    # Create sample model spec for comparison
    sample_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(sample_spec))

  def test_init_with_default_national_parameters_works(self):
    meridian = model.Meridian(input_data=self.national_input_data_media_only)

    # Compare input data.
    self.assertEqual(meridian.input_data, self.national_input_data_media_only)

    # Create sample model spec for comparison
    expected_spec = spec.ModelSpec()

    # Compare model spec.
    self.assertEqual(repr(meridian.model_spec), repr(expected_spec))

  # TODO(b/295163156) Move this test to a higher-level public API unit test.
  @parameterized.named_parameters(
      dict(
          testcase_name="adstock_first",
          hill_before_adstock=False,
          expected_called_names=["mock_adstock", "mock_hill"],
      ),
      dict(
          testcase_name="hill_first",
          hill_before_adstock=True,
          expected_called_names=["mock_hill", "mock_adstock"],
      ),
  )
  def test_adstock_hill_media(
      self,
      hill_before_adstock,
      expected_called_names,
  ):
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_only.media,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_only.media,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    meridian = model.Meridian(
        input_data=self.input_data_with_media_only,
        model_spec=spec.ModelSpec(
            hill_before_adstock=hill_before_adstock,
        ),
    )
    meridian.adstock_hill_media(
        media=meridian.model_data.media_tensors.media,
        alpha=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        ec=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
        slope=np.ones(shape=(self._N_MEDIA_CHANNELS,)),
    )

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  # TODO(b/295163156) Move this test to a higher-level public API unit test.
  def test_adstock_hill_rf(
      self,
  ):
    mock_hill = self.enter_context(
        mock.patch.object(
            adstock_hill.HillTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_and_rf.frequency,
        )
    )
    mock_adstock = self.enter_context(
        mock.patch.object(
            adstock_hill.AdstockTransformer,
            "forward",
            autospec=True,
            return_value=self.input_data_with_media_and_rf.reach
            * self.input_data_with_media_and_rf.frequency,
        )
    )
    manager = mock.Mock()
    manager.attach_mock(mock_adstock, "mock_adstock")
    manager.attach_mock(mock_hill, "mock_hill")

    meridian = model.Meridian(
        input_data=self.input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )
    meridian.adstock_hill_rf(
        reach=meridian.model_data.rf_tensors.reach,
        frequency=meridian.model_data.rf_tensors.frequency,
        alpha=np.ones(shape=(self._N_RF_CHANNELS,)),
        ec=np.ones(shape=(self._N_RF_CHANNELS,)),
        slope=np.ones(shape=(self._N_RF_CHANNELS,)),
    )

    expected_called_names = ["mock_hill", "mock_adstock"]

    mock_hill.assert_called_once()
    mock_adstock.assert_called_once()

    mocks_called_names = [mc[0] for mc in manager.mock_calls]
    self.assertEqual(mocks_called_names, expected_called_names)

  def test_get_joint_dist_zeros(self):
    model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(
            knot_values=tfp.distributions.Deterministic(0),
            tau_g_excl_baseline=tfp.distributions.Deterministic(0),
            beta_m=tfp.distributions.Deterministic(0),
            beta_rf=tfp.distributions.Deterministic(0),
            eta_m=tfp.distributions.Deterministic(0),
            eta_rf=tfp.distributions.Deterministic(0),
            gamma_c=tfp.distributions.Deterministic(0),
            xi_c=tfp.distributions.Deterministic(0),
            alpha_m=tfp.distributions.Deterministic(0),
            alpha_rf=tfp.distributions.Deterministic(0),
            ec_m=tfp.distributions.Deterministic(0),
            ec_rf=tfp.distributions.Deterministic(0),
            slope_m=tfp.distributions.Deterministic(0),
            slope_rf=tfp.distributions.Deterministic(0),
            sigma=tfp.distributions.Deterministic(0),
            roi_m=tfp.distributions.Deterministic(0),
            roi_rf=tfp.distributions.Deterministic(0),
        )
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    sample = meridian._get_joint_dist_unpinned().sample(self._N_DRAWS)
    self.assertAllEqual(
        sample.y,
        tf.zeros(shape=(self._N_DRAWS, self._N_GEOS, self._N_TIMES_SHORT)),
    )

  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_only(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_only,
    )

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.EC_M,
        constants.SLOPE_M,
        constants.SIGMA,
        constants.ROI_M,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_M)
    else:
      prior_distribution_params.append(constants.BETA_M)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(meridian.model_data.prior_broadcast, parname).log_prob(
              par[parname]
          )
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=meridian.model_data.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    beta_m = par[constants.BETA_GM][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_media, beta_m)
        + tf.einsum(
            "gtc,gc->gt",
            meridian.model_data.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            meridian.model_data.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            meridian.model_data.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_rf_only(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_rf_only,
    )
    mdata = meridian.model_data

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_RF,
        constants.EC_RF,
        constants.SLOPE_RF,
        constants.SIGMA,
        constants.ROI_RF,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_RF)
    else:
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(mdata.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_reach = meridian.adstock_hill_rf(
        reach=mdata.rf_tensors.reach_scaled,
        frequency=mdata.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    beta_rf = par[constants.BETA_GRF][0, :, :]
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", transformed_reach, beta_rf)
        + tf.einsum(
            "gtc,gc->gt",
            mdata.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            mdata.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(mdata.prior_broadcast, constants.TAU_G_EXCL_BASELINE).log_prob(
            par[constants.TAU_G_EXCL_BASELINE]
        )
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  # TODO(b/307543975): Add test for holdout_id.
  @parameterized.product(
      use_roi_prior=[False, True],
      media_effects_dist=[
          constants.MEDIA_EFFECTS_NORMAL,
          constants.MEDIA_EFFECTS_LOG_NORMAL,
      ],
  )
  def test_get_joint_dist_with_log_prob_media_and_rf(
      self, use_roi_prior: bool, media_effects_dist: str
  ):
    model_spec = spec.ModelSpec(
        use_roi_prior=use_roi_prior, media_effects_dist=media_effects_dist
    )
    meridian = model.Meridian(
        model_spec=model_spec,
        input_data=self.short_input_data_with_media_and_rf,
    )
    mdata = meridian.model_data

    # Take a single draw of all parameters from the prior distribution.
    par_structtuple = meridian._get_joint_dist_unpinned().sample(1)
    par = par_structtuple._asdict()

    # Note that "y" is a draw from the prior predictive (transformed) impact
    # distribution. We drop it because "y" is already "pinned" in
    # meridian._get_joint_dist() and is not actually a parameter.
    del par["y"]

    # Note that the actual (transformed) impact data is "pinned" as "y".
    log_prob_parts_structtuple = meridian._get_joint_dist().log_prob_parts(par)
    log_prob_parts = {
        k: v._asdict() for k, v in log_prob_parts_structtuple._asdict().items()
    }

    derived_params = [
        constants.BETA_GM,
        constants.BETA_GRF,
        constants.GAMMA_GC,
        constants.TAU_T,
        constants.TAU_G,
    ]
    prior_distribution_params = [
        constants.KNOT_VALUES,
        constants.ETA_M,
        constants.ETA_RF,
        constants.GAMMA_C,
        constants.XI_C,
        constants.ALPHA_M,
        constants.ALPHA_RF,
        constants.EC_M,
        constants.EC_RF,
        constants.SLOPE_M,
        constants.SLOPE_RF,
        constants.SIGMA,
        constants.ROI_M,
        constants.ROI_RF,
    ]
    if use_roi_prior:
      derived_params.append(constants.BETA_M)
      derived_params.append(constants.BETA_RF)
    else:
      prior_distribution_params.append(constants.BETA_M)
      prior_distribution_params.append(constants.BETA_RF)

    # Parameters that are derived from other parameters via Deterministic()
    # should have zero contribution to log_prob.
    for parname in derived_params:
      self.assertAllEqual(log_prob_parts["unpinned"][parname][0], 0)

    prior_distribution_logprobs = {}
    for parname in prior_distribution_params:
      prior_distribution_logprobs[parname] = tf.reduce_sum(
          getattr(mdata.prior_broadcast, parname).log_prob(par[parname])
      )
      self.assertAllClose(
          prior_distribution_logprobs[parname],
          log_prob_parts["unpinned"][parname][0],
      )

    coef_params = [
        constants.BETA_GM_DEV,
        constants.BETA_GRF_DEV,
        constants.GAMMA_GC_DEV,
    ]
    coef_logprobs = {}
    for parname in coef_params:
      coef_logprobs[parname] = tf.reduce_sum(
          tfp.distributions.Normal(0, 1).log_prob(par[parname])
      )
      self.assertAllClose(
          coef_logprobs[parname], log_prob_parts["unpinned"][parname][0]
      )
    transformed_media = meridian.adstock_hill_media(
        media=mdata.media_tensors.media_scaled,
        alpha=par[constants.ALPHA_M],
        ec=par[constants.EC_M],
        slope=par[constants.SLOPE_M],
    )[0, :, :, :]
    transformed_reach = meridian.adstock_hill_rf(
        reach=mdata.rf_tensors.reach_scaled,
        frequency=mdata.rf_tensors.frequency,
        alpha=par[constants.ALPHA_RF],
        ec=par[constants.EC_RF],
        slope=par[constants.SLOPE_RF],
    )[0, :, :, :]
    combined_transformed_media = tf.concat(
        [transformed_media, transformed_reach], axis=-1
    )

    combined_beta = tf.concat(
        [par[constants.BETA_GM][0, :, :], par[constants.BETA_GRF][0, :, :]],
        axis=-1,
    )
    y_means = (
        par[constants.TAU_G][0, :, None]
        + par[constants.TAU_T][0, None, :]
        + tf.einsum("gtm,gm->gt", combined_transformed_media, combined_beta)
        + tf.einsum(
            "gtc,gc->gt",
            mdata.controls_scaled,
            par[constants.GAMMA_GC][0, :, :],
        )
    )
    y_means_logprob = tf.reduce_sum(
        tfp.distributions.Normal(y_means, par[constants.SIGMA]).log_prob(
            mdata.kpi_scaled
        )
    )
    self.assertAllClose(y_means_logprob, log_prob_parts["pinned"]["y"][0])

    tau_g_logprob = tf.reduce_sum(
        getattr(
            mdata.prior_broadcast, constants.TAU_G_EXCL_BASELINE
        ).log_prob(par[constants.TAU_G_EXCL_BASELINE])
    )
    self.assertAllClose(
        tau_g_logprob,
        log_prob_parts["unpinned"][constants.TAU_G_EXCL_BASELINE][0],
    )

    posterior_unnormalized_logprob = (
        sum(prior_distribution_logprobs.values())
        + sum(coef_logprobs.values())
        + y_means_logprob
        + tau_g_logprob
    )
    self.assertAllClose(
        posterior_unnormalized_logprob,
        meridian._get_joint_dist().log_prob(par)[0],
    )

  def test_sample_prior_seed_same_seed(self):
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=1)
    self.assertEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_sample_prior_different_seed(self):
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=2)

    self.assertNotEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_sample_prior_media_and_rf_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_media_and_rf,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(prior, attr) for attr in constants.RF_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_media_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_media_only,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_rf_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "_sample_prior_fn",
            autospec=True,
            return_value=self.test_dist_rf_only,
        )
    )

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_rf_only,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(prior, attr) for attr in constants.RF_PARAMETERS
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_posterior_media_and_rf_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_rf_only_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_rf_only,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_rf_only,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=self._N_CHAINS,
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    knots_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (self._N_CHAINS, self._N_KEEP, self._N_CONTROLS)
    rf_channel_shape = (self._N_CHAINS, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (self._N_CHAINS, self._N_KEEP, 1)
    )
    geo_shape = (self._N_CHAINS, self._N_KEEP, self._N_GEOS)
    time_shape = (self._N_CHAINS, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              self._N_CHAINS,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_media_and_rf_sequential_returns_correct_shape(self):
    mock_sample_posterior = self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            return_value=collections.namedtuple(
                "StatesAndTrace", ["all_states", "trace"]
            )(
                all_states=self.test_posterior_states_media_and_rf,
                trace=self.test_trace,
            ),
        )
    )
    model_spec = spec.ModelSpec(
        roi_calibration_period=self._ROI_CALIBRATION_PERIOD,
        rf_roi_calibration_period=self._RF_ROI_CALIBRATION_PERIOD,
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )

    meridian.sample_posterior(
        n_chains=[self._N_CHAINS, self._N_CHAINS],
        n_adapt=self._N_ADAPT,
        n_burnin=self._N_BURNIN,
        n_keep=self._N_KEEP,
    )
    mock_sample_posterior.assert_called_with(
        n_draws=self._N_BURNIN + self._N_KEEP,
        joint_dist=mock.ANY,
        n_chains=self._N_CHAINS,
        num_adaptation_steps=self._N_ADAPT,
        current_state=None,
        init_step_size=None,
        dual_averaging_kwargs=None,
        max_tree_depth=10,
        max_energy_diff=500.0,
        unrolled_leapfrog_steps=1,
        parallel_iterations=10,
        seed=None,
    )
    n_total_chains = self._N_CHAINS * 2
    knots_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    control_shape = (n_total_chains, self._N_KEEP, self._N_CONTROLS)
    media_channel_shape = (n_total_chains, self._N_KEEP, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (n_total_chains, self._N_KEEP, self._N_RF_CHANNELS)
    sigma_shape = (
        (n_total_chains, self._N_KEEP, self._N_GEOS)
        if meridian.model_data.unique_sigma_for_each_geo
        else (n_total_chains, self._N_KEEP, 1)
    )
    geo_shape = (n_total_chains, self._N_KEEP, self._N_GEOS)
    time_shape = (n_total_chains, self._N_KEEP, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    posterior = meridian.inference_data.posterior
    shape_to_params = {
        knots_shape: [
            getattr(posterior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        control_shape: [
            getattr(posterior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        media_channel_shape: [
            getattr(posterior, attr) for attr in constants.MEDIA_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.RF_PARAMETERS
        ],
        sigma_shape: [
            getattr(posterior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [
            getattr(posterior, attr) for attr in constants.GEO_PARAMETERS
        ],
        time_shape: [
            getattr(posterior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(posterior, attr)
            for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(posterior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    for attr in [
        constants.STEP_SIZE,
        constants.TARGET_LOG_PROBABILITY_ARVIZ,
        constants.DIVERGING,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.sample_stats, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )
    for attr in [
        constants.STEP_SIZE,
        constants.TUNE,
        constants.TARGET_LOG_PROBABILITY_TF,
        constants.DIVERGING,
        constants.ACCEPT_RATIO,
        constants.N_STEPS,
    ]:
      self.assertEqual(
          getattr(meridian.inference_data.trace, attr).shape,
          (
              n_total_chains,
              self._N_KEEP,
          ),
      )

  def test_sample_posterior_raises_oom_error_when_limits_exceeded(self):
    self.enter_context(
        mock.patch.object(
            model,
            "_xla_windowed_adaptive_nuts",
            autospec=True,
            side_effect=tf.errors.ResourceExhaustedError(
                None, None, "Resource exhausted"
            ),
        )
    )
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=spec.ModelSpec(),
    )

    with self.assertRaises(model.MCMCOOMError):
      meridian.sample_posterior(
          n_chains=self._N_CHAINS,
          n_adapt=self._N_ADAPT,
          n_burnin=self._N_BURNIN,
          n_keep=self._N_KEEP,
      )

  def test_save_and_load_works(self):
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    file_path = os.path.join(self.create_tempdir().full_path, "joblib")
    mmm = model.Meridian(input_data=self.input_data_with_media_and_rf)
    model.save_mmm(mmm, str(file_path))
    self.assertTrue(os.path.exists(file_path))
    new_mmm = model.load_mmm(file_path)
    for attr in dir(mmm):
      if isinstance(getattr(mmm, attr), (int, bool)):
        with self.subTest(name=attr):
          self.assertEqual(getattr(mmm, attr), getattr(new_mmm, attr))
      elif isinstance(getattr(mmm, attr), tf.Tensor):
        with self.subTest(name=attr):
          self.assertAllClose(getattr(mmm, attr), getattr(new_mmm, attr))

  def test_load_error(self):
    with self.assertRaisesWithLiteralMatch(
        FileNotFoundError, "No such file or directory: this/path/does/not/exist"
    ):
      model.load_mmm("this/path/does/not/exist")


if __name__ == "__main__":
  absltest.main()
