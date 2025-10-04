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

"""Shared test data samples."""

import collections
import os
from typing import NamedTuple

from meridian import backend
from meridian import constants
from meridian.data import input_data
from meridian.data import test_utils
import xarray as xr


def _convert_with_swap(array: xr.DataArray, n_burnin: int) -> backend.Tensor:
  """Converts a DataArray to a backend.Tensor with the correct MCMC format.

  This function converts a DataArray to backend.Tensor, swaps first two
  dimensions and adds the burnin part. This is needed to properly mock the
  _xla_windowed_adaptive_nuts() function output in the sample_posterior
  tests.

  Args:
    array: The array to be converted.
    n_burnin: The number of extra draws to be padded with as the 'burnin' part.

  Returns:
    A tensor in the same format as returned by the _xla_windowed_adaptive_nuts()
    function.
  """
  tensor = backend.to_tensor(array)
  perm = [1, 0] + [i for i in range(2, len(tensor.shape))]
  transposed_tensor = backend.transpose(tensor, perm=perm)

  # Add the "burnin" part to the mocked output of _xla_windowed_adaptive_nuts
  # to make sure sample_posterior returns the correct "keep" part.
  if array.dtype == bool:
    pad_value = False
  else:
    pad_value = 0.0 if array.dtype.kind == "f" else 0

  burnin = backend.fill(
      [n_burnin] + list(transposed_tensor.shape[1:]), pad_value
  )
  return backend.concatenate(
      [burnin, transposed_tensor],
      axis=0,
  )


class WithInputDataSamples:
  """A mixin to inject test data samples to a unit test class.

  The `setup` method is a classmethod because loading and creating the data
  samples can be expensive. The properties return deep copies to ensure
  immutability for the data properties. As a result, it is recommended to use
  local variables to avoid re-loading the data samples multiple times.
  """

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
  _TEST_SAMPLE_PRIOR_MEDIA_ONLY_NO_CONTROLS_PATH = os.path.join(
      _TEST_DIR,
      "sample_prior_media_only_no_controls.nc",
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
  _TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_NO_CONTROLS_PATH = os.path.join(
      _TEST_DIR,
      "sample_posterior_media_only_no_controls.nc",
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
  _N_ORGANIC_MEDIA_CHANNELS = 4
  _N_ORGANIC_RF_CHANNELS = 1
  _N_NON_MEDIA_CHANNELS = 2

  _ROI_CALIBRATION_PERIOD: backend.Tensor
  _RF_ROI_CALIBRATION_PERIOD: backend.Tensor

  # Private class variables to hold the base test data.
  _input_data_non_revenue_no_revenue_per_kpi: input_data.InputData
  _input_data_media_and_rf_non_revenue_no_revenue_per_kpi: input_data.InputData
  _input_data_with_media_only: input_data.InputData
  _input_data_with_rf_only: input_data.InputData
  _input_data_with_media_and_rf: input_data.InputData
  _input_data_with_media_and_rf_no_controls: input_data.InputData
  _short_input_data_with_media_only: input_data.InputData
  _short_input_data_with_media_only_no_controls: input_data.InputData
  _short_input_data_with_rf_only: input_data.InputData
  _short_input_data_with_media_and_rf: input_data.InputData
  _national_input_data_media_only: input_data.InputData
  _national_input_data_media_and_rf: input_data.InputData
  _test_dist_media_and_rf: collections.OrderedDict[str, backend.Tensor]
  _test_dist_media_only: collections.OrderedDict[str, backend.Tensor]
  _test_dist_media_only_no_controls: collections.OrderedDict[
      str, backend.Tensor
  ]
  _test_dist_rf_only: collections.OrderedDict[str, backend.Tensor]
  _test_trace: dict[str, backend.Tensor]
  _national_input_data_non_media_and_organic: input_data.InputData
  _input_data_non_media_and_organic: input_data.InputData
  _short_input_data_non_media_and_organic: input_data.InputData
  _short_input_data_non_media: input_data.InputData
  _input_data_non_media_and_organic_same_time_dims: input_data.InputData
  _input_data_organic_only: input_data.InputData
  _national_input_data_organic_only: input_data.InputData

  # The following NamedTuples and their attributes are immutable, so they can
  # be accessed directly.
  test_posterior_states_media_and_rf: NamedTuple
  test_posterior_states_media_only: NamedTuple
  test_posterior_states_media_only_no_controls: NamedTuple
  test_posterior_states_rf_only: NamedTuple

  @classmethod
  def setup(cls):
    """Sets up input data samples."""
    cls._ROI_CALIBRATION_PERIOD = backend.cast(
        backend.ones((cls._N_MEDIA_TIMES_SHORT, cls._N_MEDIA_CHANNELS)),
        dtype=backend.bool_,
    )
    cls._RF_ROI_CALIBRATION_PERIOD = backend.cast(
        backend.ones((cls._N_MEDIA_TIMES_SHORT, cls._N_RF_CHANNELS)),
        dtype=backend.bool_,
    )

    cls._input_data_non_revenue_no_revenue_per_kpi = (
        test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_media_and_rf_non_revenue_no_revenue_per_kpi = (
        test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_with_media_and_rf_no_controls = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=None,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_with_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_with_media_only_no_controls = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=0,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_with_rf_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=cls._N_CONTROLS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_with_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )
    cls._national_input_data_media_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS_NATIONAL,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls._national_input_data_media_and_rf = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS_NATIONAL,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            seed=0,
        )
    )

    test_prior_media_and_rf = xr.open_dataset(
        cls._TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH
    )
    test_prior_media_only = xr.open_dataset(
        cls._TEST_SAMPLE_PRIOR_MEDIA_ONLY_PATH
    )
    test_prior_media_only_no_controls = xr.open_dataset(
        cls._TEST_SAMPLE_PRIOR_MEDIA_ONLY_NO_CONTROLS_PATH
    )
    test_prior_rf_only = xr.open_dataset(cls._TEST_SAMPLE_PRIOR_RF_ONLY_PATH)
    cls._test_dist_media_and_rf = collections.OrderedDict({
        param: backend.to_tensor(test_prior_media_and_rf[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })
    cls._test_dist_media_only = collections.OrderedDict({
        param: backend.to_tensor(test_prior_media_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    })
    cls._test_dist_media_only_no_controls = collections.OrderedDict({
        param: backend.to_tensor(test_prior_media_only_no_controls[param])
        for param in (
            set(
                constants.COMMON_PARAMETER_NAMES
                + constants.MEDIA_PARAMETER_NAMES
            )
            - set(
                constants.CONTROL_PARAMETERS + constants.GEO_CONTROL_PARAMETERS
            )
        )
    })
    cls._test_dist_rf_only = collections.OrderedDict({
        param: backend.to_tensor(test_prior_rf_only[param])
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    })

    test_posterior_media_and_rf = xr.open_dataset(
        cls._TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH
    )
    test_posterior_media_only = xr.open_dataset(
        cls._TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_PATH
    )
    test_posterior_media_only_no_controls = xr.open_dataset(
        cls._TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_NO_CONTROLS_PATH
    )
    test_posterior_rf_only = xr.open_dataset(
        cls._TEST_SAMPLE_POSTERIOR_RF_ONLY_PATH
    )
    posterior_params_to_tensors_media_and_rf = {
        param: _convert_with_swap(
            test_posterior_media_and_rf[param], n_burnin=cls._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    posterior_params_to_tensors_media_only = {
        param: _convert_with_swap(
            test_posterior_media_only[param], n_burnin=cls._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
    }
    posterior_params_to_tensors_media_only_no_controls = {
        param: _convert_with_swap(
            test_posterior_media_only_no_controls[param],
            n_burnin=cls._N_BURNIN,
        )
        for param in (
            set(
                constants.COMMON_PARAMETER_NAMES
                + constants.MEDIA_PARAMETER_NAMES
            )
            - set(
                constants.CONTROL_PARAMETERS + constants.GEO_CONTROL_PARAMETERS
            )
        )
    }
    posterior_params_to_tensors_rf_only = {
        param: _convert_with_swap(
            test_posterior_rf_only[param], n_burnin=cls._N_BURNIN
        )
        for param in constants.COMMON_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES
    }
    cls.test_posterior_states_media_and_rf = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES
        + constants.MEDIA_PARAMETER_NAMES
        + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_and_rf)
    cls.test_posterior_states_media_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.MEDIA_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_media_only)
    cls.test_posterior_states_media_only_no_controls = collections.namedtuple(
        "StructTuple",
        (
            set(
                constants.COMMON_PARAMETER_NAMES
                + constants.MEDIA_PARAMETER_NAMES
            )
            - set(
                constants.CONTROL_PARAMETERS + constants.GEO_CONTROL_PARAMETERS
            )
        ),
    )(**posterior_params_to_tensors_media_only_no_controls)
    cls.test_posterior_states_rf_only = collections.namedtuple(
        "StructTuple",
        constants.COMMON_PARAMETER_NAMES + constants.RF_PARAMETER_NAMES,
    )(**posterior_params_to_tensors_rf_only)

    test_trace = xr.open_dataset(cls._TEST_SAMPLE_TRACE_PATH)
    cls._test_trace = {
        param: _convert_with_swap(test_trace[param], n_burnin=cls._N_BURNIN)
        for param in test_trace.data_vars
    }

    # The following are input data samples with non-paid channels.

    cls._national_input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS_NATIONAL,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=cls._N_NON_MEDIA_CHANNELS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )

    cls._input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=cls._N_NON_MEDIA_CHANNELS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_non_media_and_organic = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=cls._N_NON_MEDIA_CHANNELS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    cls._short_input_data_non_media = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES_SHORT,
            n_media_times=cls._N_MEDIA_TIMES_SHORT,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=cls._N_NON_MEDIA_CHANNELS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            n_organic_media_channels=0,
            n_organic_rf_channels=0,
            seed=0,
        )
    )
    cls._input_data_non_media_and_organic_same_time_dims = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_TIMES,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=cls._N_NON_MEDIA_CHANNELS,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=cls._N_RF_CHANNELS,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    cls._input_data_organic_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=0,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=0,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )
    cls._national_input_data_organic_only = (
        test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=cls._N_GEOS_NATIONAL,
            n_times=cls._N_TIMES,
            n_media_times=cls._N_MEDIA_TIMES,
            n_controls=cls._N_CONTROLS,
            n_non_media_channels=0,
            n_media_channels=cls._N_MEDIA_CHANNELS,
            n_rf_channels=0,
            n_organic_media_channels=cls._N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=cls._N_ORGANIC_RF_CHANNELS,
            seed=0,
        )
    )

  @property
  def input_data_non_revenue_no_revenue_per_kpi(self) -> input_data.InputData:
    return self._input_data_non_revenue_no_revenue_per_kpi.copy(deep=True)

  @property
  def input_data_media_and_rf_non_revenue_no_revenue_per_kpi(
      self,
  ) -> input_data.InputData:
    return self._input_data_media_and_rf_non_revenue_no_revenue_per_kpi.copy(
        deep=True
    )

  @property
  def input_data_with_media_only(self) -> input_data.InputData:
    return self._input_data_with_media_only.copy(deep=True)

  @property
  def input_data_with_rf_only(self) -> input_data.InputData:
    return self._input_data_with_rf_only.copy(deep=True)

  @property
  def input_data_with_media_and_rf(self) -> input_data.InputData:
    return self._input_data_with_media_and_rf.copy(deep=True)

  @property
  def input_data_with_media_and_rf_no_controls(
      self,
  ) -> input_data.InputData:
    return self._input_data_with_media_and_rf_no_controls.copy(deep=True)

  @property
  def short_input_data_with_media_only(self) -> input_data.InputData:
    return self._short_input_data_with_media_only.copy(deep=True)

  @property
  def short_input_data_with_media_only_no_controls(
      self,
  ) -> input_data.InputData:
    return self._short_input_data_with_media_only_no_controls.copy(deep=True)

  @property
  def short_input_data_with_rf_only(self) -> input_data.InputData:
    return self._short_input_data_with_rf_only.copy(deep=True)

  @property
  def short_input_data_with_media_and_rf(
      self,
  ) -> input_data.InputData:
    return self._short_input_data_with_media_and_rf.copy(deep=True)

  @property
  def national_input_data_media_only(self) -> input_data.InputData:
    return self._national_input_data_media_only.copy(deep=True)

  @property
  def national_input_data_media_and_rf(self) -> input_data.InputData:
    return self._national_input_data_media_and_rf.copy(deep=True)

  @property
  def test_dist_media_and_rf(
      self,
  ) -> collections.OrderedDict[str, backend.Tensor]:
    return collections.OrderedDict(self._test_dist_media_and_rf)

  @property
  def test_dist_media_only(
      self,
  ) -> collections.OrderedDict[str, backend.Tensor]:
    return collections.OrderedDict(self._test_dist_media_only)

  @property
  def test_dist_media_only_no_controls(
      self,
  ) -> collections.OrderedDict[str, backend.Tensor]:
    return collections.OrderedDict(self._test_dist_media_only_no_controls)

  @property
  def test_dist_rf_only(self) -> collections.OrderedDict[str, backend.Tensor]:
    return collections.OrderedDict(self._test_dist_rf_only)

  @property
  def test_trace(self) -> dict[str, backend.Tensor]:
    return self._test_trace.copy()

  @property
  def national_input_data_non_media_and_organic(
      self,
  ) -> input_data.InputData:
    return self._national_input_data_non_media_and_organic.copy(deep=True)

  @property
  def input_data_non_media_and_organic(self) -> input_data.InputData:
    return self._input_data_non_media_and_organic.copy(deep=True)

  @property
  def short_input_data_non_media_and_organic(
      self,
  ) -> input_data.InputData:
    return self._short_input_data_non_media_and_organic.copy(deep=True)

  @property
  def short_input_data_non_media(self) -> input_data.InputData:
    return self._short_input_data_non_media.copy(deep=True)

  @property
  def input_data_non_media_and_organic_same_time_dims(
      self,
  ) -> input_data.InputData:
    return self._input_data_non_media_and_organic_same_time_dims.copy(deep=True)

  @property
  def input_data_organic_only(self) -> input_data.InputData:
    return self._input_data_organic_only.copy(deep=True)

  @property
  def national_input_data_organic_only(self) -> input_data.InputData:
    return self._national_input_data_organic_only.copy(deep=True)
