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

"""Shared test data samples."""

import collections
import os

from meridian import constants
from meridian.data import test_utils
import tensorflow as tf
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


class WithInputDataSamples:
  """A mixin to inject test data samples to a unit test class."""

  # TODO: Update the sample data to span over 1 or 2 year(s).
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
  _N_ORGANIC_MEDIA_CHANNELS = 4
  _N_ORGANIC_RF_CHANNELS = 1
  _N_NON_MEDIA_CHANNELS = 2

  def setup(self):
    """Sets up input data samples."""
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
    self.input_data_media_and_rf_non_revenue_no_revenue_per_kpi = (
        test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
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

    # The following are input data samples with non-paid channels.

    self.national_input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS_NATIONAL,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            n_organic_media_channels=self._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=self._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )

    self.input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES,
            n_media_times=self._N_MEDIA_TIMES,
            n_controls=self._N_CONTROLS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            n_organic_media_channels=self._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=self._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            n_organic_media_channels=self._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=self._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    self.short_input_data_non_media = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=self._N_GEOS,
            n_times=self._N_TIMES_SHORT,
            n_media_times=self._N_MEDIA_TIMES_SHORT,
            n_controls=self._N_CONTROLS,
            n_non_media_channels=self._N_NON_MEDIA_CHANNELS,
            n_media_channels=self._N_MEDIA_CHANNELS,
            n_rf_channels=self._N_RF_CHANNELS,
            n_organic_media_channels=0,
            n_organic_rf_channels=0,
            seed=0,
        )
    )
