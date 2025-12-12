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

from collections.abc import Sequence
import dataclasses
import os
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import test_utils as analysis_test_utils
from meridian.backend import test_utils as backend_test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import model
from meridian.model import spec
import numpy as np
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


def _convert_with_swap(array: xr.DataArray) -> backend.Tensor:
  """Converts DataArray to backend.Tensor and swaps first two dimensions."""
  tensor = backend.to_tensor(array)
  perm = [1, 0] + [i for i in range(2, len(tensor.shape))]
  return backend.transpose(tensor, perm=perm)


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


class DataTensorsTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
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
    cls.meridian_media_and_rf = model.Meridian(
        input_data=cls.input_data_media_and_rf,
        model_spec=spec.ModelSpec(max_lag=15),
    )
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
    cls.meridian_media_only = model.Meridian(
        input_data=cls.input_data_media_only,
        model_spec=spec.ModelSpec(max_lag=15),
    )
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
    cls.meridian_rf_only = model.Meridian(
        input_data=cls.input_data_rf_only,
        model_spec=spec.ModelSpec(max_lag=15),
    )
    cls.input_data_organic_media = (
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
    cls.meridian_organic_media = model.Meridian(
        input_data=cls.input_data_organic_media,
        model_spec=spec.ModelSpec(max_lag=15),
    )
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
    cls.meridian_non_media = model.Meridian(
        input_data=cls.input_data_non_media,
        model_spec=spec.ModelSpec(max_lag=15),
    )

  def test_init_wrong_dims_controls(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "New `controls` must have 3 dimension(s). Found 2 dimension(s).",
    ):
      analyzer.DataTensors(controls=backend.ones((_N_GEOS, _N_TIMES)))

  @parameterized.named_parameters(
      (
          "wrong_media_dims",
          {constants.MEDIA: (_N_GEOS, _N_MEDIA_CHANNELS)},
          "New `media` must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_reach_dims",
          {constants.REACH: (_N_GEOS, _N_RF_CHANNELS)},
          "New `reach` must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_frequency_dims",
          {constants.FREQUENCY: (_N_GEOS, _N_RF_CHANNELS)},
          "New `frequency` must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "wrong_revenue_per_kpi_dims",
          {constants.REVENUE_PER_KPI: (_N_GEOS,)},
          (
              "New `revenue_per_kpi` must have 2 dimension(s). Found 1"
              " dimension(s)."
          ),
      ),
      (
          "wrong_media_spend_dims",
          {constants.MEDIA_SPEND: (_N_GEOS, _N_MEDIA_CHANNELS)},
          "New `media_spend` must have 1 or 3 dimensions. Found 2 dimensions.",
      ),
      (
          "wrong_rf_spend_dims",
          {constants.RF_SPEND: (_N_GEOS, _N_RF_CHANNELS)},
          "New `rf_spend` must have 1 or 3 dimensions. Found 2 dimensions.",
      ),
      (
          "organic_media",
          {constants.ORGANIC_MEDIA: (_N_GEOS, _N_ORGANIC_MEDIA_CHANNELS)},
          "New `organic_media` must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "organic_reach",
          {constants.ORGANIC_REACH: (_N_GEOS, _N_ORGANIC_RF_CHANNELS)},
          "New `organic_reach` must have 3 dimension(s). Found 2 dimension(s).",
      ),
      (
          "non_media_treatments",
          {constants.NON_MEDIA_TREATMENTS: (_N_GEOS,)},
          (
              "New `non_media_treatments` must have 3 dimension(s). Found 1"
              " dimension(s)."
          ),
      ),
  )
  def test_init_wrong_shape_new_param(
      self,
      new_param_shapes: dict[str, tuple[int, ...]],
      expected_error_message: str,
  ):
    new_param = {k: backend.ones(v) for k, v in new_param_shapes.items()}
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_message):
      analyzer.DataTensors(**new_param)

  def test_validate_wrong_geos_media(self):
    new_data = analyzer.DataTensors(
        media=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError, "New `media` is expected to have 5 geos. Found 6 geos."
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          meridian=self.meridian_media_and_rf,
      )

  def test_validate_wrong_geos_media_spend(self):
    new_data = analyzer.DataTensors(
        media_spend=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `media_spend` is expected to have 5 geos. Found 6 geos.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA_SPEND],
          meridian=self.meridian_media_and_rf,
      )

  def test_validate_wrong_times_media(self):
    new_data = analyzer.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `media` is expected to have 52 time periods. Found 10 time"
        " periods.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          meridian=self.meridian_media_and_rf,
          allow_modified_times=False,
      )

  def test_validate_wrong_channels_frequency(self):
    new_data = analyzer.DataTensors(
        frequency=backend.ones((_N_GEOS, _N_TIMES, 3))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `frequency` is expected to have 2 channels. Found 3 channels.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.FREQUENCY],
          meridian=self.meridian_media_and_rf,
      )

  def test_validate_wrong_channels_reach(self):
    new_data = analyzer.DataTensors(
        reach=backend.ones((_N_GEOS, _N_MEDIA_TIMES, _N_RF_CHANNELS - 1))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `reach` is expected to have 2 channels. Found 1 channels.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.REACH],
          meridian=self.meridian_media_and_rf,
      )

  @parameterized.parameters([
      constants.MEDIA,
      constants.REACH,
      constants.FREQUENCY,
      constants.REVENUE_PER_KPI,
  ])
  def test_validate_missing_new_param_flexible_times(self, missing_param: str):
    new_data_dict = {
        constants.MEDIA: backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        constants.REACH: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.FREQUENCY: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.REVENUE_PER_KPI: backend.ones((_N_GEOS, 10)),
    }
    new_data_dict.pop(missing_param)
    new_data = analyzer.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified, then"
        " all variables must be provided in `new_data`. The following variables"
        f" are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=list(new_data_dict.keys()) + [missing_param],
          meridian=self.meridian_media_and_rf,
      )

  def test_validate_new_params_diff_time_dims(self):
    new_data = analyzer.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        frequency=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        revenue_per_kpi=backend.ones((_N_GEOS, 8)),
    )
    with self.assertRaisesRegex(
        ValueError,
        "If the time dimension of any variable in `new_data` is modified, then"
        " all variables must be provided with the same number of time periods."
        " `revenue_per_kpi` has 8 time periods, which does not match the"
        " modified number of time periods, 10.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.REVENUE_PER_KPI,
          ],
          meridian=self.meridian_media_and_rf,
      )

  @parameterized.parameters([constants.MEDIA, constants.REVENUE_PER_KPI])
  def test_validate_media_only_missing_new_param(self, missing_param: str):
    new_data_dict = {
        constants.MEDIA: backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        constants.REVENUE_PER_KPI: backend.ones((_N_GEOS, 10)),
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = analyzer.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified,"
        " then all variables must be provided in `new_data`. The"
        f" following variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names,
          meridian=self.meridian_media_only,
      )

  def test_validate_media_only_invalid_new_data(self):
    new_data = analyzer.DataTensors(
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `reach` is not allowed because the input data to the Meridian"
        " model does not contain `reach`",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.REACH],
          meridian=self.meridian_media_only,
      )

  @parameterized.parameters([
      constants.REACH,
      constants.FREQUENCY,
      constants.REVENUE_PER_KPI,
  ])
  def test_validate_rf_only_missing_new_param(self, missing_param: str):
    new_data_dict = {
        constants.REACH: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.FREQUENCY: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.REVENUE_PER_KPI: backend.ones((_N_GEOS, 10)),
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = analyzer.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified, then"
        " all variables must be provided in `new_data`. The following"
        f" variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names, meridian=self.meridian_rf_only
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
    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
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

    tensors = {}
    for tensor_name in new_tensors_names:
      tensors[tensor_name] = getattr(data, tensor_name)
    new_data = analyzer.DataTensors(**tensors)

    required_tensors_names = [
        constants.MEDIA,
        constants.REACH,
        constants.FREQUENCY,
    ]
    if require_controls:
      required_tensors_names.append(constants.CONTROLS)
    if require_non_paid_channels:
      required_tensors_names.extend([
          constants.ORGANIC_MEDIA,
          constants.ORGANIC_REACH,
          constants.ORGANIC_FREQUENCY,
          constants.NON_MEDIA_TREATMENTS,
      ])
    if require_revenue_per_kpi:
      required_tensors_names.append(constants.REVENUE_PER_KPI)

    filled_tensors = new_data.validate_and_fill_missing_data(
        required_tensors_names, self.meridian_organic_media
    )
    for tensor_name in required_tensors_names:
      expected_source = (
          data
          if tensor_name in new_tensors_names
          else self.input_data_organic_media
      )
      backend_test_utils.assert_allclose(
          getattr(filled_tensors, tensor_name),
          getattr(expected_source, tensor_name),
          rtol=1e-4,
          atol=1e-4,
      )

  @parameterized.parameters([constants.MEDIA, constants.NON_MEDIA_TREATMENTS])
  def test_validate_organic_media_missing_new_param_flexible_times(
      self, missing_param: str
  ):
    new_data_dict = {
        constants.MEDIA: backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        constants.REACH: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.FREQUENCY: backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        constants.ORGANIC_MEDIA: backend.ones(
            (_N_GEOS, 10, _N_ORGANIC_MEDIA_CHANNELS)
        ),
        constants.ORGANIC_REACH: backend.ones(
            (_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)
        ),
        constants.ORGANIC_FREQUENCY: backend.ones(
            (_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)
        ),
        constants.NON_MEDIA_TREATMENTS: backend.ones(
            (_N_GEOS, 10, _N_NON_MEDIA_CHANNELS)
        ),
        constants.REVENUE_PER_KPI: backend.ones((_N_GEOS, 10)),
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = analyzer.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified,"
        " then all variables must be provided in `new_data`. The"
        f" following variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names,
          meridian=self.meridian_organic_media,
      )

  def test_validate_organic_media_new_param_not_matching_times(self):
    new_data = analyzer.DataTensors(
        media=backend.ones((_N_GEOS, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS)),
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        frequency=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        organic_media=backend.ones(
            (_N_GEOS, _N_MEDIA_TIMES, _N_ORGANIC_MEDIA_CHANNELS)
        ),
        organic_reach=backend.ones(
            (_N_GEOS, _N_MEDIA_TIMES, _N_ORGANIC_RF_CHANNELS)
        ),
        organic_frequency=backend.ones(
            (_N_GEOS, _N_MEDIA_TIMES, _N_ORGANIC_RF_CHANNELS)
        ),
        non_media_treatments=backend.ones(
            (_N_GEOS, _N_TIMES, _N_NON_MEDIA_CHANNELS)
        ),
        revenue_per_kpi=backend.ones((_N_GEOS, _N_TIMES)),
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of any variable in `new_data` is modified, then"
        " all variables must be provided with the same number of time periods."
        " `media` has 52 time periods, which does not match the modified number"
        " of time periods, 10.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.ORGANIC_MEDIA,
              constants.ORGANIC_REACH,
              constants.ORGANIC_FREQUENCY,
              constants.NON_MEDIA_TREATMENTS,
              constants.REVENUE_PER_KPI,
          ],
          meridian=self.meridian_organic_media,
      )

  @parameterized.named_parameters(
      (
          "media_spend",
          constants.MEDIA_SPEND,
          "A `media_spend` value was passed",
      ),
      (
          "controls",
          constants.CONTROLS,
          "A `controls` value was passed",
      ),
  )
  def test_validate_warns_on_unexpected_params(
      self, param_name: str, warning_msg: str
  ) -> None:
    if param_name == constants.CONTROLS:
      tensor = self.meridian_media_and_rf.controls
    elif param_name == constants.MEDIA_SPEND:
      tensor = self.meridian_media_and_rf.media_tensors.media_spend
    else:
      tensor = getattr(self.meridian_media_and_rf.input_data, param_name)

    new_data = analyzer.DataTensors(**{param_name: tensor})
    required = [constants.MEDIA]

    with self.assertWarnsRegex(UserWarning, warning_msg):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required, meridian=self.meridian_media_and_rf
      )

  def test_validate_non_media_missing_new_param_flexible_times(self) -> None:
    new_data = analyzer.DataTensors(
        non_media_treatments=self.meridian_non_media.non_media_treatments[
            :, :2, :
        ]
    )
    required = [
        constants.MEDIA,
        constants.REACH,
        constants.FREQUENCY,
        constants.REVENUE_PER_KPI,
        constants.NON_MEDIA_TREATMENTS,
    ]
    with self.assertRaisesRegex(
        ValueError, "If the time dimension .* missing: .*"
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required, meridian=self.meridian_non_media
      )


class AnalyzerTest(backend_test_utils.MeridianTestCase):

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
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_media_and_rf = model.Meridian(
        input_data=cls.input_data_media_and_rf, model_spec=model_spec
    )
    cls.analyzer_media_and_rf = analyzer.Analyzer(cls.meridian_media_and_rf)

    cls.inference_data_media_and_rf = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_AND_RF_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_AND_RF_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_media_and_rf),
        )
    )


class AnalyzerNationalTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerNationalTest, cls).setUpClass()
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
    n_times = len(cls.input_data_national.time)
    holdout_id = np.full([n_times], False)
    holdout_id[np.random.choice(n_times, int(np.round(0.2 * n_times)))] = True
    model_spec = spec.ModelSpec(holdout_id=holdout_id)
    cls.meridian_national = model.Meridian(
        input_data=cls.input_data_national, model_spec=model_spec
    )
    cls.analyzer_national = analyzer.Analyzer(cls.meridian_national)

    cls.inference_data_national = _build_inference_data(
        _TEST_SAMPLE_PRIOR_NATIONAL_PATH,
        _TEST_SAMPLE_POSTERIOR_NATIONAL_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data_national),
        )
    )

  def test_rhat_summary_national_correct(self):
    rhat_summary = self.analyzer_national.rhat_summary()
    self.assertEqual(rhat_summary.shape, (31, 7))
    self.assertSetEqual(
        set(rhat_summary.param),
        set(
            constants.COMMON_PARAMETER_NAMES
            + constants.MEDIA_PARAMETER_NAMES
            + constants.RF_PARAMETER_NAMES
            + constants.ORGANIC_MEDIA_PARAMETER_NAMES
            + constants.ORGANIC_RF_PARAMETER_NAMES
            + constants.NON_MEDIA_PARAMETER_NAMES
        )
        - set(constants.ALL_NATIONAL_DETERMINISTIC_PARAMETER_NAMES),
    )

  @parameterized.product(
      selected_geos=[None, ["geo_0"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_predictive_accuracy_with_holdout_id_national_correct(
      self, selected_geos, selected_times
  ):
    predictive_accuracy_dims_kwargs = {
        "selected_geos": selected_geos,
        "selected_times": selected_times,
    }
    predictive_accuracy_dataset = self.analyzer_national.predictive_accuracy(
        **predictive_accuracy_dims_kwargs,
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )

    if not selected_times:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NATIONAL_NO_TIMES
      )
    else:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NATIONAL_TIMES
      )

    backend_test_utils.assert_allclose(
        list(df[constants.VALUE]),
        expected_values,
        atol=2e-3,
    )


class AnalyzerMediaOnlyTest(backend_test_utils.MeridianTestCase):

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
          backend.to_tensor(self.input_data_media_only.population),
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
          backend.to_tensor(
              self.input_data_media_only.media_spend[..., :-1, :-1]
          ),
          flexible_time_dim=flexible_time_dim,
          has_media_dim=has_media_dim,
      )

  def test_filter_and_aggregate_geos_and_times_empty_geos(self):
    tensor = backend.to_tensor(self.input_data_media_only.media_spend)
    modified_tensor = (
        self.analyzer_media_only.filter_and_aggregate_geos_and_times(
            tensor,
            selected_geos=[],
        )
    )
    backend_test_utils.assert_allequal(modified_tensor, backend.zeros([3]))

  def test_filter_and_aggregate_geos_and_times_empty_times(self):
    tensor = backend.to_tensor(self.input_data_media_only.media_spend)
    modified_tensor = (
        self.analyzer_media_only.filter_and_aggregate_geos_and_times(
            tensor,
            selected_times=[],
        )
    )
    backend_test_utils.assert_allequal(modified_tensor, backend.zeros([3]))

  def test_filter_and_aggregate_geos_and_times_incorrect_geos(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_geos` must match the geo dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          backend.to_tensor(self.input_data_media_only.media_spend),
          selected_geos=["random_geo"],
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_time_dim_names(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          backend.to_tensor(self.input_data_media_only.media_spend),
          selected_times=["random_time"],
      )

  def test_filter_and_aggregate_geos_and_times_incorrect_time_bool(self):
    with self.assertRaisesRegex(
        ValueError,
        "Boolean `selected_times` must have the same number of elements as "
        "there are time period coordinates in `tensor`.",
    ):
      self.analyzer_media_only.filter_and_aggregate_geos_and_times(
          backend.to_tensor(self.input_data_media_only.media_spend),
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
          backend.to_tensor(self.input_data_media_only.media_spend),
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
    tensor = backend.to_tensor(self.input_data_media_only.media_spend)
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
          expected_outcome=analysis_test_utils.INC_OUTCOME_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=analysis_test_utils.INC_OUTCOME_MEDIA_ONLY_USE_POSTERIOR,
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
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(expected_outcome),
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
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(
            analysis_test_utils.INC_OUTCOME_MEDIA_ONLY_NEW_PARAMS
        ),
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
          expected_mroi=analysis_test_utils.MROI_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_MEDIA_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=analysis_test_utils.MROI_MEDIA_ONLY_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_MEDIA_ONLY_USE_POSTERIOR,
      ),
  )
  def test_marginal_roi_media_only(
      self, use_posterior: bool, by_reach: bool, expected_mroi: np.ndarray
  ):
    mroi = self.analyzer_media_only.marginal_roi(
        by_reach=by_reach,
        use_posterior=use_posterior,
    )
    backend_test_utils.assert_allclose(
        mroi,
        backend.to_tensor(expected_mroi),
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
    backend_test_utils.assert_allclose(roi, expected_roi)

  def test_roi_media_only_default_returns_correct_value(self):
    roi = self.analyzer_media_only.roi()
    total_spend = self.analyzer_media_only.filter_and_aggregate_geos_and_times(
        self.meridian_media_only.media_tensors.media_spend
    )
    expected_roi = self.analyzer_media_only.incremental_outcome() / total_spend
    backend_test_utils.assert_allclose(expected_roi, roi)

  def test_roi_zero_media_returns_zero(self):
    new_media = backend.zeros_like(
        self.meridian_media_only.media_tensors.media, dtype=backend.float32
    )
    roi = self.analyzer_media_only.roi(
        new_data=analyzer.DataTensors(media=new_media)
    )
    backend_test_utils.assert_allclose(
        roi, backend.zeros((_N_CHAINS, _N_KEEP, _N_MEDIA_CHANNELS)), atol=2e-6
    )

  def test_roi_zero_media_spend_returns_inf(self):
    new_media_spend = backend.zeros_like(
        self.meridian_media_only.media_tensors.media_spend,
        dtype=backend.float32,
    )
    roi = self.analyzer_media_only.roi(
        new_data=analyzer.DataTensors(media_spend=new_media_spend)
    )
    np.testing.assert_array_equal(np.isinf(roi), np.full(roi.shape, True))

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
    self.assertContainsSubset(
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

  def test_get_aggregated_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer_media_only.get_aggregated_spend(
        selected_times=None, include_rf=False
    )
    expected_channel_names = self.input_data_media_only.get_all_paid_channels()

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_aggregated_spend_requests_rf_when_no_rf_throws_warning(self):
    with self.assertWarnsRegex(
        UserWarning,
        "Requested spends for paid media channels with R&F data, but the"
        " channels are not available.",
    ):
      self.analyzer_media_only.get_aggregated_spend()

  def test_get_aggregated_spend_requests_rf_when_no_rf_outputs_empty_data_array(
      self,
  ):
    actual = self.analyzer_media_only.get_aggregated_spend(include_media=False)
    backend_test_utils.assert_allequal(actual.data, [])


class AnalyzerMediaOnlyNoControlsTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super(AnalyzerMediaOnlyNoControlsTest, cls).setUpClass()

    cls.input_data_media_only_no_controls = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=0,
            n_media_channels=_N_MEDIA_CHANNELS,
            seed=0,
        )
    )

    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian_media_only_no_controls = model.Meridian(
        input_data=cls.input_data_media_only_no_controls, model_spec=model_spec
    )
    cls.analyzer_media_only_no_controls = analyzer.Analyzer(
        cls.meridian_media_only_no_controls
    )

    cls.inference_data_media_only_no_controls = _build_inference_data(
        _TEST_SAMPLE_PRIOR_MEDIA_ONLY_NO_CONTROLS_PATH,
        _TEST_SAMPLE_POSTERIOR_MEDIA_ONLY_NO_CONTROLS_PATH,
    )

    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(
                lambda unused_self: cls.inference_data_media_only_no_controls
            ),
        )
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_expected_outcome_media_only_no_controls_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
  ):
    outcome = self.analyzer_media_only_no_controls.expected_outcome(
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


class AnalyzerRFOnlyTest(backend_test_utils.MeridianTestCase):

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
          expected_outcome=analysis_test_utils.INC_OUTCOME_RF_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=analysis_test_utils.INC_OUTCOME_RF_ONLY_USE_POSTERIOR,
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
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(expected_outcome),
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
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(analysis_test_utils.INC_OUTCOME_RF_ONLY_NEW_PARAMS),
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
          expected_mroi=analysis_test_utils.MROI_RF_ONLY_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_RF_ONLY_USE_PRIOR_BY_REACH,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=analysis_test_utils.MROI_RF_ONLY_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_RF_ONLY_USE_POSTERIOR_BY_REACH,
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
    backend_test_utils.assert_allclose(
        mroi,
        backend.to_tensor(expected_mroi),
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
    expected_roi = self.analyzer_rf_only.incremental_outcome() / total_spend
    backend_test_utils.assert_allclose(expected_roi, roi)

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
    backend_test_utils.assert_allclose(
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
      # TODO: Remove this once the warning is fixed.
      user_warnings = [
          warning
          for warning in w
          if not issubclass(warning.category, DeprecationWarning)
      ]
      self.assertLen(user_warnings, 1)
      self.assertTrue(issubclass(user_warnings[0].category, UserWarning))
      self.assertIn(
          "ROI, mROI, Effectiveness, and CPIK are not reported because they do "
          "not have a clear interpretation by time period.",
          str(user_warnings[0].message),
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
            constants.FREQUENCY: [1.0, 2.0, 3.0],
            constants.RF_CHANNEL: ["rf_ch_0", "rf_ch_1"],
            constants.METRIC: [
                constants.MEAN,
                constants.MEDIAN,
                constants.CI_LO,
                constants.CI_HI,
            ],
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
                    [4.055367e-04, 3.081534e-04, 9.794392e-05, 7.991136e-04],
                    [5.891235e-04, 6.401768e-04, 3.493646e-05, 1.078733e-03],
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
                    [1.496, 1.523, 0.329, 2.666],
                    [2.191, 0.726, 0.239, 7.546],
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
    xr.testing.assert_allclose(actual.roi, expected.roi, atol=0.05)
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
          backend.ones_like(
              self.analyzer_rf_only._meridian.rf_tensors.frequency
          )
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

      backend_test_utils.assert_allequal(
          roi[i, :, 0], np.mean(roi_temp, (0, 1))
      )
      backend_test_utils.assert_allequal(
          roi[i, :, 1], np.quantile(roi_temp, (1 - 0.9) / 2, (0, 1))
      )
      backend_test_utils.assert_allequal(
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
    backend_test_utils.assert_allequal(
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

  def test_get_aggregated_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer_rf_only.get_aggregated_spend(
        selected_times=None, include_media=False
    )
    expected_channel_names = self.input_data_rf_only.get_all_paid_channels()

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_aggregated_spend_requests_media_when_no_media_throws_warning(
      self,
  ):
    with self.assertWarnsRegex(
        UserWarning,
        "Requested spends for paid media channels that do not have R&F data,"
        " but the channels are not available.",
    ):
      self.analyzer_rf_only.get_aggregated_spend()

  def test_get_aggregated_spend_requests_rf_when_no_rf_outputs_empty_data_array(
      self,
  ):
    actual = self.analyzer_rf_only.get_aggregated_spend(include_rf=False)
    backend_test_utils.assert_allequal(actual.data, [])


class AnalyzerComprehensiveTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # This generates data with Media, RF, Non-Media, Organic Media, and Organic
    # RF
    cls.input_data = (
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
            nonzero_shift=1.0,
        )
    )
    cls.not_lagged_input_data = (
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
            nonzero_shift=1.0,
        )
    )
    model_spec = spec.ModelSpec(max_lag=15)
    cls.meridian = model.Meridian(
        input_data=cls.input_data, model_spec=model_spec
    )
    cls.analyzer = analyzer.Analyzer(cls.meridian)

    cls.inference_data = _build_inference_data(
        _TEST_SAMPLE_PRIOR_NON_PAID_PATH,
        _TEST_SAMPLE_POSTERIOR_NON_PAID_PATH,
    )
    cls.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: cls.inference_data),
        )
    )

  def test_use_kpi_direct_calls_non_revenue_with_revenue_per_kpi(self):
    # `use_kpi` is respected
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertTrue(self.analyzer._use_kpi(use_kpi=True))
      self.assertEmpty(w)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertFalse(self.analyzer._use_kpi(use_kpi=False))
      self.assertEmpty(w)

  def test_use_kpi_direct_calls_revenue(self):
    input_data_revenue = data_test_utils.sample_input_data_revenue(
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
    mmm_revenue = model.Meridian(input_data=input_data_revenue)
    analyzer_revenue = analyzer.Analyzer(mmm_revenue)

    with self.assertWarnsRegex(
        UserWarning,
        "Setting `use_kpi=True` has no effect when `kpi_type=REVENUE`",
    ):
      self.assertFalse(analyzer_revenue._use_kpi(use_kpi=True))

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertFalse(analyzer_revenue._use_kpi(use_kpi=False))
      self.assertEmpty(w)

  def test_use_kpi_direct_calls_non_revenue_no_revenue_per_kpi(self):
    input_data_non_revenue_no_revenue_per_kpi = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
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
    mmm_non_revenue = model.Meridian(
        input_data=input_data_non_revenue_no_revenue_per_kpi
    )
    analyzer_non_revenue = analyzer.Analyzer(mmm_non_revenue)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertTrue(analyzer_non_revenue._use_kpi(use_kpi=True))
      self.assertEmpty(w)

    with self.assertWarnsRegex(
        UserWarning,
        "Revenue analysis is not available when `revenue_per_kpi` is"
        " unknown. Defaulting to KPI analysis.",
    ):
      self.assertTrue(analyzer_non_revenue._use_kpi(use_kpi=False))

  @parameterized.named_parameters(
      ("marginal_roi", "marginal_roi"),
      ("roi", "roi"),
      ("summary_metrics", "summary_metrics"),
      ("predictive_accuracy", "predictive_accuracy"),
  )
  def test_no_revenue_data_use_kpi_false_warning_and_fallback(
      self, method_name
  ):
    input_data = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
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
    mmm = model.Meridian(input_data=input_data)
    type(mmm).inference_data = mock.PropertyMock(
        return_value=self.inference_data
    )
    analyzer_no_rev = analyzer.Analyzer(mmm)

    method = getattr(analyzer_no_rev, method_name)

    with self.assertWarnsRegex(
        UserWarning,
        "Revenue analysis is not available when `revenue_per_kpi` is"
        " unknown. Defaulting to KPI analysis.",
    ):
      result_false = method(use_kpi=False)

    result_true = method(use_kpi=True)

    if isinstance(result_false, xr.Dataset):
      xr.testing.assert_allclose(result_false, result_true)
    else:
      backend_test_utils.assert_allclose(result_false, result_true)

  def test_public_methods_revenue_type_use_kpi_true_has_no_effect(self):
    input_data = data_test_utils.sample_input_data_revenue(
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
    mmm = model.Meridian(input_data=input_data)
    type(mmm).inference_data = mock.PropertyMock(
        return_value=self.inference_data
    )
    analyzer_rev = analyzer.Analyzer(mmm)

    with self.assertWarnsRegex(
        UserWarning,
        "Setting `use_kpi=True` has no effect when `kpi_type=REVENUE`",
    ):
      res_true = analyzer_rev.expected_outcome(use_kpi=True)

    res_false = analyzer_rev.expected_outcome(use_kpi=False)
    backend_test_utils.assert_allclose(res_true, res_false)

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

  def test_expected_outcome_wrong_kpi_transformation(self):
    with self.assertRaisesRegex(
        ValueError,
        "use_kpi=False is only supported when inverse_transform_outcome=True.",
    ):
      self.analyzer.expected_outcome(
          inverse_transform_outcome=False, use_kpi=False
      )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      geos_to_include=[None, ["geo_1", "geo_3"]],
      times_to_include=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      use_kpi=[False, True],
  )
  def test_expected_outcome_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      geos_to_include: Sequence[str] | None,
      times_to_include: Sequence[str] | None,
      use_kpi: bool,
  ):
    outcome = self.analyzer.expected_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=geos_to_include,
        selected_times=times_to_include,
        use_kpi=use_kpi,
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

  def test_expected_outcome_new_data(self):
    expected = self.analyzer.expected_outcome()
    # Set new data with only certain param overrides matching the existing data.
    # The params not provided should use the existing data and the result will
    # be overridden. Using the same training data should result in the same
    # expected outcome as the original data.
    outcome = self.analyzer.expected_outcome(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media,
            frequency=self.meridian.rf_tensors.frequency,
            revenue_per_kpi=self.meridian.revenue_per_kpi,
        ),
    )
    backend_test_utils.assert_allclose(
        outcome,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )

  def test_expected_outcome_new_data_result(self):
    default = self.analyzer.expected_outcome()
    outcome = self.analyzer.expected_outcome(
        new_data=analyzer.DataTensors(
            revenue_per_kpi=self.meridian.revenue_per_kpi * 2.0,
        ),
        use_kpi=False,
    )
    backend_test_utils.assert_allclose(
        outcome,
        default * 2.0,
        rtol=1e-3,
        atol=1e-3,
    )
    backend_test_utils.assert_allclose(
        outcome,
        analysis_test_utils.EXP_OUTCOME_MEDIA_AND_RF,
        rtol=1e-3,
        atol=1e-3,
    )

  def test_incremental_outcome_negative_scaling_factor0(self):
    with self.assertRaisesRegex(
        ValueError,
        "scaling_factor0 must be non-negative.",
    ):
      self.analyzer.incremental_outcome(scaling_factor0=-0.01)

  def test_incremental_outcome_negative_scaling_factor1(self):
    with self.assertRaisesRegex(
        ValueError, "scaling_factor1 must be non-negative."
    ):
      self.analyzer.incremental_outcome(scaling_factor1=-0.01)

  def test_incremental_outcome_scaling_factor1_less_than_scaling_factor0(self):
    with self.assertRaisesRegex(
        ValueError,
        "scaling_factor1 must be greater than scaling_factor0. Got"
        " scaling_factor1=1.0 and scaling_factor0=1.1.",
    ):
      self.analyzer.incremental_outcome(
          scaling_factor0=1.1, scaling_factor1=1.0
      )

  def test_incremental_outcome_flexible_times_selected_times_wrong_type(self):
    with self.assertRaisesRegex(
        ValueError,
        "If `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,"
        " `organic_frequency`, `non_media_treatments`, or `revenue_per_kpi` is"
        " provided with a different number of time periods than in `InputData`,"
        r" then \(1\) `selected_times` must be a list of booleans with length"
        r" equal to the number of time periods in the new data, or \(2\)"
        " `selected_times` must be a list of strings and `new_time` must be"
        " provided and `selected_times` must be a subset of `new_time`.",
    ):
      self.analyzer.incremental_outcome(
          new_data=analyzer.DataTensors(
              media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
              reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              frequency=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              organic_media=backend.ones(
                  (_N_GEOS, 10, _N_ORGANIC_MEDIA_CHANNELS)
              ),
              organic_reach=backend.ones((_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)),
              organic_frequency=backend.ones(
                  (_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)
              ),
              non_media_treatments=backend.ones(
                  (_N_GEOS, 10, _N_NON_MEDIA_CHANNELS)
              ),
              revenue_per_kpi=backend.ones((_N_GEOS, 10)),
          ),
          selected_times=["2021-04-19", "2021-09-13", "2021-12-13"],
      )

  def test_incremental_outcome_flexible_times_media_selected_times_wrong_type(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "If `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,"
        " `organic_frequency`, `non_media_treatments`, or `revenue_per_kpi` is"
        " provided with a different number of time periods than in `InputData`,"
        r" then \(1\) `media_selected_times` must be a list of booleans with"
        r" length equal to the number of time periods in the new data, or \(2\)"
        " `media_selected_times` must be a list of strings and `new_time` must"
        " be provided and `media_selected_times` must be a subset of"
        " `new_time`.",
    ):
      self.analyzer.incremental_outcome(
          new_data=analyzer.DataTensors(
              media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
              reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              frequency=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
              organic_media=backend.ones(
                  (_N_GEOS, 10, _N_ORGANIC_MEDIA_CHANNELS)
              ),
              organic_reach=backend.ones((_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)),
              organic_frequency=backend.ones(
                  (_N_GEOS, 10, _N_ORGANIC_RF_CHANNELS)
              ),
              non_media_treatments=backend.ones(
                  (_N_GEOS, 10, _N_NON_MEDIA_CHANNELS)
              ),
              revenue_per_kpi=backend.ones((_N_GEOS, 10)),
          ),
          media_selected_times=["2021-04-19", "2021-09-13", "2021-12-13"],
      )

  def test_incremental_outcome_media_selected_times_wrong_length(self):
    with self.assertRaisesRegex(
        ValueError,
        "Boolean `media_selected_times` must have the same number of elements "
        "as there are time period coordinates in the media tensors.",
    ):
      self.analyzer.incremental_outcome(
          media_selected_times=[False] * (_N_MEDIA_TIMES - 10) + [True],
      )

  def test_incremental_outcome_media_selected_times_wrong_time_dim_names(self):
    with self.assertRaisesRegex(
        ValueError,
        "`media_selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer.incremental_outcome(
          media_selected_times=["random_time"],
      )

  def test_incremental_outcome_incorrect_media_selected_times_type(self):
    with self.assertRaisesRegex(
        ValueError,
        "`media_selected_times` must be a list of strings or a list of"
        " booleans.",
    ):
      self.analyzer.incremental_outcome(
          media_selected_times=["random_time", False, True],
      )

  def test_incremental_outcome_wrong_kpi_transformation(self):
    with self.assertRaisesRegex(
        ValueError,
        "use_kpi=False is only supported when inverse_transform_outcome=True.",
    ):
      self.analyzer.incremental_outcome(
          inverse_transform_outcome=False, use_kpi=False
      )

  def test_incremental_outcome_new_revenue_per_kpi_correct_shape(self):
    n_channels = (
        _N_MEDIA_CHANNELS
        + _N_RF_CHANNELS
        + _N_NON_MEDIA_CHANNELS
        + _N_ORGANIC_MEDIA_CHANNELS
        + _N_ORGANIC_RF_CHANNELS
    )
    outcome = self.analyzer.incremental_outcome(
        new_data=analyzer.DataTensors(
            revenue_per_kpi=backend.ones((_N_GEOS, _N_TIMES))
        ),
    )
    self.assertEqual(outcome.shape, (_N_CHAINS, _N_KEEP, n_channels))

  def test_incremental_outcome_media_selected_times_all_false_returns_zero(
      self,
  ):
    no_media_times = self.analyzer.incremental_outcome(
        media_selected_times=[False] * _N_MEDIA_TIMES,
        include_non_paid_channels=False,
    )
    backend_test_utils.assert_allequal(
        no_media_times, backend.zeros_like(no_media_times)
    )

  def test_incremental_outcome_no_overlap_between_media_and_selected_times(
      self,
  ):
    # If for any time period where media_selected_times is True, selected_times
    # is False for this time period and the following `max_lag` time periods,
    # then the incremental outcome should be zero.
    max_lag = self.meridian.model_spec.max_lag
    media_selected_times = [self.meridian.input_data.media_time.values[0]]
    selected_times = [False] * (max_lag + 1) + [True] * (_N_TIMES - max_lag - 1)
    outcome = self.analyzer.incremental_outcome(
        selected_times=selected_times,
        media_selected_times=media_selected_times,
        include_non_paid_channels=False,
    )
    backend_test_utils.assert_allequal(outcome, backend.zeros_like(outcome))

  def test_incremental_outcome_media_and_selected_times_overlap_non_zero(self):
    # Incremental outcome should be non-zero when there is at least one time
    # period of overlap between media_selected_times and selected_times. In this
    # case, media_selected_times is True for week 1 and selected_times is True
    # for week `max_lag+1` and the following weeks.
    max_lag = self.meridian.model_spec.max_lag
    excess_times = _N_MEDIA_TIMES - _N_TIMES
    media_selected_times = [True] + [False] * (_N_MEDIA_TIMES - 1)
    selected_times = [False] * (max_lag - excess_times) + [True] * (
        _N_TIMES - max_lag + excess_times
    )
    outcome = self.analyzer.incremental_outcome(
        selected_times=selected_times,
        media_selected_times=media_selected_times,
    )
    mean_inc_outcome = backend.reduce_mean(outcome, axis=(0, 1))
    self.assertFalse(np.all(np.array(mean_inc_outcome) == 0))

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
      use_kpi=[False, True],
  )
  def test_incremental_outcome_media_and_rf_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      use_kpi: bool,
  ):
    outcome = self.analyzer.incremental_outcome(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
        include_non_paid_channels=False,
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
          expected_outcome=analysis_test_utils.INC_OUTCOME_MEDIA_AND_RF_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=analysis_test_utils.INC_OUTCOME_MEDIA_AND_RF_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_media_and_rf(
      self,
      use_posterior: bool,
      expected_outcome: np.ndarray,
  ):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data
    )
    outcome = self.analyzer.incremental_outcome(
        use_posterior=use_posterior,
        include_non_paid_channels=False,
    )
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(expected_outcome),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_compute_incremental_outcome_aggregate_media_and_rf(self):
    mock_incremental_outcome = np.ones(
        (_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS + _N_RF_CHANNELS)
    )
    with mock.patch.object(
        self.analyzer,
        "incremental_outcome",
        return_value=mock_incremental_outcome,
    ):
      incremental_outcome_with_totals = np.full(
          (_N_CHAINS, _N_DRAWS, 1),
          _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
          dtype=np.float64,
      )
      outcome = self.analyzer.compute_incremental_outcome_aggregate(
          use_posterior=True,
          include_non_paid_channels=False,
      )
      backend_test_utils.assert_allclose(
          outcome,
          backend.concatenate(
              [
                  backend.to_tensor(mock_incremental_outcome),
                  backend.to_tensor(incremental_outcome_with_totals),
              ],
              -1,
          ),
      )

  # The purpose of this test is to prevent accidental logic change.
  def test_incremental_outcome_media_and_rf_new_params(self):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data
    )
    outcome = self.analyzer.incremental_outcome(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -10:, :],
            reach=self.meridian.rf_tensors.reach[..., -10:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -10:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -10:],
        ),
        include_non_paid_channels=False,
    )
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(
            analysis_test_utils.INC_OUTCOME_MEDIA_AND_RF_NEW_PARAMS
        ),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_incremental_outcome_media_and_rf_new_params_correct_shape(self):
    model.Meridian.inference_data = mock.PropertyMock(
        return_value=self.inference_data
    )
    outcome = self.analyzer.incremental_outcome(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -15:, :],
            reach=self.meridian.rf_tensors.reach[..., -15:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -15:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -15:],
        ),
        aggregate_times=False,
        include_non_paid_channels=False,
    )
    self.assertEqual(
        outcome.shape,
        (_N_CHAINS, _N_KEEP, 15, _N_MEDIA_CHANNELS + _N_RF_CHANNELS),
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
      tensors.append(backend.to_tensor(self.not_lagged_input_data.media))
    if n_rf_channels > 0:
      tensors.append(backend.to_tensor(self.not_lagged_input_data.reach))
    if n_non_media_channels > 0:
      tensors.append(
          backend.to_tensor(self.not_lagged_input_data.non_media_treatments)
      )
    if n_organic_media_channels > 0:
      tensors.append(
          backend.to_tensor(self.not_lagged_input_data.organic_media)
      )
    if n_organic_rf_channels > 0:
      tensors.append(
          backend.to_tensor(self.not_lagged_input_data.organic_reach)
      )
    tensor = backend.concatenate(tensors, axis=-1)
    modified_tensor = self.analyzer.filter_and_aggregate_geos_and_times(
        tensor,
        selected_geos=selected_geos,
        selected_times=selected_times,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        flexible_time_dim=False,
        has_media_dim=True,
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
    tensor = backend.concatenate(
        [
            backend.to_tensor(self.not_lagged_input_data.media),
            backend.to_tensor(self.not_lagged_input_data.reach),
            backend.to_tensor(self.not_lagged_input_data.non_media_treatments),
            backend.to_tensor(self.not_lagged_input_data.organic_media),
            backend.to_tensor(self.not_lagged_input_data.organic_reach),
        ],
        axis=-1,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The tensor must have shape [..., n_geos, n_times, n_channels] or [...,"
        " n_geos, n_times] if `flexible_time_dim=False`.",
    ):
      self.analyzer.filter_and_aggregate_geos_and_times(
          tensor[..., :num_channels],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_outcome=analysis_test_utils.INC_OUTCOME_NON_PAID_USE_PRIOR,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_outcome=analysis_test_utils.INC_OUTCOME_NON_PAID_USE_POSTERIOR,
      ),
  )
  def test_incremental_outcome_organic_media(
      self,
      use_posterior: bool,
      expected_outcome: np.ndarray,
  ):
    outcome = self.analyzer.incremental_outcome(
        use_posterior=use_posterior,
    )
    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(expected_outcome),
        rtol=1e-2,
        atol=1e-2,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_USE_PRIOR,
          non_media_baseline_values=None,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_USE_POSTERIOR,
          non_media_baseline_values=None,
      ),
      dict(
          testcase_name="all_min",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_MIN,
          non_media_baseline_values=[
              -7.229473,
              -7.1908092,
              -3.0269506,
              -6.3038673,
          ],
      ),
      dict(
          testcase_name="all_max",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_MAX,
          non_media_baseline_values=[
              14.567047,
              15.695609,
              11.22775,
              14.682818,
          ],
      ),
      dict(
          testcase_name="mix",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_MIX,
          non_media_baseline_values=[
              -7.229473,
              15.695609,
              11.22775,
              -6.3038673,
          ],
      ),
      dict(
          testcase_name="mix_as_floats",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_MIX,
          non_media_baseline_values=[
              -7.229473,
              15.695609,
              11.22775,
              -6.3038673,
          ],
      ),
      dict(
          testcase_name="all_fixed",
          use_posterior=True,
          expected_result=analysis_test_utils.INC_OUTCOME_NON_MEDIA_FIXED,
          non_media_baseline_values=[45.2, 1.03, 0.24, 7.77],
      ),
  )
  def test_incremental_outcome_non_media(
      self,
      use_posterior: bool,
      expected_result: np.ndarray,
      non_media_baseline_values: Sequence[float] | None,
  ):
    outcome = self.analyzer.incremental_outcome(
        use_posterior=use_posterior,
        include_non_paid_channels=True,
        non_media_baseline_values=non_media_baseline_values,
    )

    backend_test_utils.assert_allclose(
        outcome,
        backend.to_tensor(expected_result),
        rtol=2e-2,
        atol=1e-2,
    )

  def test_incremental_outcome_wrong_baseline_types_shape_raises_exception(
      self,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The number of non-media channels (4) does not match the number of"
        " baseline values (3).",
    ):
      self.analyzer.incremental_outcome(
          non_media_baseline_values=[13, -4, 2.8],
          include_non_paid_channels=True,
      )

  def test_incremental_outcome_wrong_baseline_type_raises_exception(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Invalid `non_media_baseline_values` value: 'min'. Only float"
        " numbers are supported.",
    ):
      self.analyzer.incremental_outcome(
          non_media_baseline_values=["min", "max", "max", 5.0],
          include_non_paid_channels=True,
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
    outcome = self.analyzer.expected_outcome(
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
    channels = (
        self.analyzer._meridian.input_data.get_all_channels()
        if include_non_paid_channels
        else self.analyzer._meridian.input_data.get_all_paid_channels()
    )

    media_summary = self.analyzer.summary_metrics(
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
    baseline_summary = self.analyzer.baseline_summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        aggregate_geos=True,
        aggregate_times=True,
        selected_geos=None,
        selected_times=None,
    )
    self.assertIsNotNone(baseline_summary.baseline_outcome)
    self.assertIsNotNone(baseline_summary.pct_of_contribution)
    backend_test_utils.assert_allclose(
        baseline_summary.baseline_outcome,
        analysis_test_utils.SAMPLE_BASELINE_EXPECTED_OUTCOME_NON_PAID,
        atol=1e-2,
        rtol=1e-2,
    )
    backend_test_utils.assert_allclose(
        baseline_summary.pct_of_contribution,
        analysis_test_utils.SAMPLE_BASELINE_PCT_OF_CONTRIBUTION_NON_PAID,
        atol=1e-2,
        rtol=1e-2,
    )

  def test_adstock_decay_dataframe(self):
    adstock_decay_dataframe = self.analyzer.adstock_decay(
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
    expected_channels = [
        "ch_0",
        "ch_1",
        "ch_2",
        "rf_ch_0",
        "rf_ch_1",
        "organic_media_0",
        "organic_media_1",
        "organic_media_2",
        "organic_media_3",
        "organic_rf_ch_0",
    ]
    self.assertSameElements(
        expected_channels, adstock_decay_dataframe[constants.CHANNEL].unique()
    )

    for i, e in enumerate(list(adstock_decay_dataframe[constants.MEAN])):
      self.assertGreaterEqual(
          e, list(adstock_decay_dataframe[constants.CI_LO])[i]
      )
      self.assertLessEqual(e, list(adstock_decay_dataframe[constants.CI_HI])[i])

  def test_adstock_decay_effect_values(self):
    adstock_decay_dataframe = self.analyzer.adstock_decay(
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
    """Verifies values for Paid Media channels."""
    adstock_decay_dataframe = self.analyzer.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )

    # Testing "ch_0" (Paid Media)
    first_channel_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.CHANNEL] == "ch_0"
    ]

    backend_test_utils.assert_allclose(
        list(first_channel_df[constants.CI_HI])[:5],
        analysis_test_utils.ADSTOCK_DECAY_CI_HI,
        atol=1e-3,
    )

    backend_test_utils.assert_allclose(
        list(first_channel_df[constants.CI_LO])[:5],
        analysis_test_utils.ADSTOCK_DECAY_CI_LO,
        atol=1e-3,
    )

    backend_test_utils.assert_allclose(
        list(first_channel_df[constants.MEAN])[:5],
        analysis_test_utils.ADSTOCK_DECAY_MEAN,
        atol=1e-3,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_media",
          channel_name="organic_media_0",
          expected_ci_hi=analysis_test_utils.ORGANIC_ADSTOCK_DECAY_CI_HI,
          expected_ci_lo=analysis_test_utils.ORGANIC_ADSTOCK_DECAY_CI_LO,
          expected_mean=analysis_test_utils.ORGANIC_ADSTOCK_DECAY_MEAN,
      ),
      dict(
          testcase_name="organic_rf",
          channel_name="organic_rf_ch_0",
          expected_ci_hi=analysis_test_utils.ORGANIC_RF_ADSTOCK_DECAY_CI_HI,
          expected_ci_lo=analysis_test_utils.ORGANIC_RF_ADSTOCK_DECAY_CI_LO,
          expected_mean=analysis_test_utils.ORGANIC_RF_ADSTOCK_DECAY_MEAN,
      ),
  )
  def test_adstock_decay_organic_channels_math_correct(
      self, channel_name, expected_ci_hi, expected_ci_lo, expected_mean
  ):
    adstock_decay_dataframe = self.analyzer.adstock_decay(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )

    channel_df = adstock_decay_dataframe[
        adstock_decay_dataframe[constants.CHANNEL] == channel_name
    ]

    channel_df = channel_df.sort_values(
        by=[constants.DISTRIBUTION, constants.TIME_UNITS],
        ascending=[False, True],
    )

    backend_test_utils.assert_allclose(
        list(channel_df[constants.CI_HI])[:5],
        expected_ci_hi,
        atol=1e-3,
    )

    backend_test_utils.assert_allclose(
        list(channel_df[constants.CI_LO])[:5],
        expected_ci_lo,
        atol=1e-3,
    )

    backend_test_utils.assert_allclose(
        list(channel_df[constants.MEAN])[:5],
        expected_mean,
        atol=1e-3,
    )

  def test_adstock_decay_time_unit_integer_indication_correct(self):
    adstock_decay_dataframe = self.analyzer.adstock_decay(
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

  def test_adstock_decay_index_is_standard_range_index(self):
    adstock_df = self.analyzer.adstock_decay()

    self.assertNotEmpty(adstock_df)
    self.assertEqual(adstock_df.index.start, 0)
    self.assertEqual(adstock_df.index.step, 1)
    self.assertLen(adstock_df, adstock_df.index.stop)

  def test_hill_histogram_column_names(self):
    hill_histogram_table = self.analyzer._get_hill_histogram_dataframe(
        n_bins=25
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
    hill_table = self.analyzer.hill_curves()
    hist_df = hill_table[hill_table[constants.COUNT_HISTOGRAM].notna()]

    expected_columns = [
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
    ]
    self.assertListEqual(list(hill_table.columns), expected_columns)

    all_channels_present = set(hill_table[constants.CHANNEL].unique())
    expected_paid_media = {"ch_0", "ch_1", "ch_2"}
    expected_rf = {"rf_ch_0", "rf_ch_1"}
    expected_organic_media = {
        "organic_media_0",
        "organic_media_1",
        "organic_media_2",
        "organic_media_3",
    }
    expected_organic_rf = {"organic_rf_ch_0"}
    self.assertTrue(expected_paid_media.issubset(all_channels_present))
    self.assertTrue(expected_rf.issubset(all_channels_present))
    self.assertTrue(expected_organic_media.issubset(all_channels_present))
    self.assertTrue(expected_organic_rf.issubset(all_channels_present))
    self.assertSetEqual(
        set(
            hill_table[hill_table[constants.CHANNEL_TYPE] == constants.MEDIA][
                constants.CHANNEL
            ].unique()
        ),
        expected_paid_media,
    )
    self.assertSetEqual(
        set(
            hill_table[hill_table[constants.CHANNEL_TYPE] == constants.RF][
                constants.CHANNEL
            ].unique()
        ),
        expected_rf,
    )
    self.assertSetEqual(
        set(
            hill_table[
                hill_table[constants.CHANNEL_TYPE] == constants.ORGANIC_MEDIA
            ][constants.CHANNEL].unique()
        ),
        expected_organic_media,
    )
    self.assertSetEqual(
        set(
            hill_table[
                hill_table[constants.CHANNEL_TYPE] == constants.ORGANIC_RF
            ][constants.CHANNEL].unique()
        ),
        expected_organic_rf,
    )
    self.assertTrue(hist_df.index.is_unique)

    # Check CI properties for valid rows (where MEAN is present)
    curve_df = hill_table[hill_table[constants.MEAN].notna()]
    ci_lo_col = list(curve_df[constants.CI_LO])
    ci_hi_col = list(curve_df[constants.CI_HI])
    mean_col = list(curve_df[constants.MEAN])

    for i, e in enumerate(mean_col):
      self.assertGreaterEqual(e, ci_lo_col[i])
      self.assertLessEqual(e, ci_hi_col[i])

  def test_hill_calculation_curve_data_correct(self):
    """Verifies Paid Media values."""
    hill_table = self.analyzer.hill_curves()
    hill_table = hill_table[
        hill_table[constants.CHANNEL_TYPE] == constants.MEDIA
    ]

    backend_test_utils.assert_allclose(
        list(hill_table[constants.CI_HI])[:5],
        np.array(analysis_test_utils.HILL_CURVES_CI_HI),
        atol=1e-5,
    )
    backend_test_utils.assert_allclose(
        list(hill_table[constants.CI_LO])[:5],
        np.array(analysis_test_utils.HILL_CURVES_CI_LO),
        atol=1e-5,
    )
    backend_test_utils.assert_allclose(
        list(hill_table[constants.MEAN])[:5],
        np.array(analysis_test_utils.HILL_CURVES_MEAN),
        atol=1e-5,
    )

  def test_hill_calculation_histogram_data_correct(self):
    """Verifies Paid Media values."""
    hill_table = self.analyzer.hill_curves()
    hill_table = hill_table[
        hill_table[constants.CHANNEL_TYPE] == constants.MEDIA
    ]

    # The Histogram data is in the bottom portion of the DataFrame for
    # Altair plotting purposes.
    backend_test_utils.assert_allclose(
        list(hill_table[constants.SCALED_COUNT_HISTOGRAM])[-5:],
        np.array(analysis_test_utils.HILL_CURVES_SCALED_COUNT_HISTOGRAM),
        atol=1e-5,
    )
    backend_test_utils.assert_allclose(
        list(hill_table[constants.COUNT_HISTOGRAM])[-5:],
        np.array(analysis_test_utils.HILL_CURVES_COUNT_HISTOGRAM),
        atol=1e-5,
    )
    backend_test_utils.assert_allclose(
        list(hill_table[constants.START_INTERVAL_HISTOGRAM])[-5:],
        np.array(analysis_test_utils.HILL_CURVES_START_INTERVAL_HISTOGRAM),
        atol=1e-5,
    )
    backend_test_utils.assert_allclose(
        list(hill_table[constants.END_INTERVAL_HISTOGRAM])[-5:],
        np.array(analysis_test_utils.HILL_CURVES_END_INTERVAL_HISTOGRAM),
        atol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_media_test",
          channel_name="organic_media_1",
          expected_channel_type=constants.ORGANIC_MEDIA,
      ),
      dict(
          testcase_name="organic_rf_test",
          channel_name="organic_rf_ch_0",
          expected_channel_type=constants.ORGANIC_RF,
      ),
  )
  def test_hill_curves_organic_curve_data_correct(
      self, channel_name, expected_channel_type
  ):
    hill_table = self.analyzer.hill_curves()

    df = (
        hill_table[
            (hill_table[constants.CHANNEL] == channel_name)
            & (hill_table[constants.DISTRIBUTION] == constants.POSTERIOR)
        ]
        .sort_values(constants.MEDIA_UNITS)
        .dropna(subset=[constants.MEAN])
    )

    self.assertFalse(df.empty)
    self.assertEqual(df[constants.CHANNEL_TYPE].iloc[0], expected_channel_type)
    self.assertTrue(df[constants.MEAN].is_monotonic_increasing)
    self.assertLessEqual(df[constants.MEAN].iloc[-1], 1.0)
    self.assertGreaterEqual(df[constants.MEAN].iloc[0], 0.0)
    self.assertLess(df[constants.MEDIA_UNITS].iloc[0], 0.1)
    self.assertAlmostEqual(df[constants.MEAN].iloc[0], 0.0, delta=1e-3)

  @parameterized.named_parameters(
      dict(
          testcase_name="organic_media_test",
          channel_name="organic_media_0",
          expected_channel_type=constants.ORGANIC_MEDIA,
      ),
      dict(
          testcase_name="organic_rf_test",
          channel_name="organic_rf_ch_0",
          expected_channel_type=constants.ORGANIC_RF,
      ),
  )
  def test_hill_curves_organic_histogram_data_correct(
      self, channel_name, expected_channel_type
  ):
    n_bins = 25
    hill_table = self.analyzer.hill_curves(n_bins=n_bins)

    hist_df = hill_table[hill_table[constants.CHANNEL] == channel_name].dropna(
        subset=[constants.COUNT_HISTOGRAM]
    )

    self.assertFalse(hist_df.empty)
    self.assertLen(hist_df, n_bins)
    self.assertEqual(
        hist_df[constants.CHANNEL_TYPE].iloc[0], expected_channel_type
    )
    self.assertTrue((hist_df[constants.COUNT_HISTOGRAM] >= 0).all())
    self.assertTrue((hist_df[constants.SCALED_COUNT_HISTOGRAM] <= 1.0001).all())
    np.testing.assert_allclose(
        hist_df[constants.START_INTERVAL_HISTOGRAM].iloc[1:].values,
        hist_df[constants.END_INTERVAL_HISTOGRAM].iloc[:-1].values,
        atol=1e-6,
        err_msg="Histogram bin start/end edges do not align correctly.",
    )

  def test_response_curves_use_only_paid_channels(self):
    response_curve_data = self.analyzer.response_curves(by_reach=False)
    self.assertIn(constants.CHANNEL, response_curve_data.coords)
    self.assertLen(
        response_curve_data.coords[constants.CHANNEL],
        len(self.analyzer._meridian.input_data.get_all_paid_channels()),
    )

  def test_summary_metrics_with_non_media_baseline_values(self):
    # Call summary_metrics with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer,
        "compute_incremental_outcome_aggregate",
        wraps=self.analyzer.compute_incremental_outcome_aggregate,
    ) as mock_compute_incremental_outcome_aggregate:
      self.analyzer.summary_metrics(
          include_non_paid_channels=True,
          non_media_baseline_values=[0.0, 7, 1.0, -1],
      )

    # Assert that _compute_incremental_outcome_aggregate was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_compute_incremental_outcome_aggregate.call_count, 4)

    # Both calls with include_non_paid_channels=True and given
    # non_media_baseline_values
    for call in mock_compute_incremental_outcome_aggregate.call_args_list:
      _, kwargs = call
      self.assertEqual(kwargs["include_non_paid_channels"], True)
      self.assertEqual(kwargs["non_media_baseline_values"], [0.0, 7, 1.0, -1])

  def test_baseline_summary_metrics_with_non_media_baseline_values(self):
    # Call baseline_summary_metrics with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer,
        "_calculate_baseline_expected_outcome",
        wraps=self.analyzer._calculate_baseline_expected_outcome,
    ) as mock_calculate_baseline_expected_outcome:
      self.analyzer.baseline_summary_metrics(
          non_media_baseline_values=[0.0, 3, 1.0, 4.5],
      )

    # Assert that _calculate_baseline_expected_outcome was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_calculate_baseline_expected_outcome.call_count, 2)
    for call in mock_calculate_baseline_expected_outcome.call_args_list:
      _, kwargs = call
      self.assertEqual(kwargs["non_media_baseline_values"], [0.0, 3, 1.0, 4.5])

  def test_expected_vs_actual_with_non_media_baseline_values(self):
    # Call expected_vs_actual with non-default value of
    # non_media_baseline_values argument.
    with mock.patch.object(
        self.analyzer,
        "_calculate_baseline_expected_outcome",
        wraps=self.analyzer._calculate_baseline_expected_outcome,
    ) as mock_calculate_baseline_expected_outcome:
      self.analyzer.expected_vs_actual_data(
          non_media_baseline_values=[0.0, 22, 1.0, 2.2],
      )

    # Assert that _calculate_baseline_expected_outcome was called the right
    # number of times with the right arguments.
    self.assertEqual(mock_calculate_baseline_expected_outcome.call_count, 1)
    _, kwargs = mock_calculate_baseline_expected_outcome.call_args
    self.assertEqual(kwargs["non_media_baseline_values"], [0.0, 22, 1.0, 2.2])

  def test_incremental_outcome_impl_without_non_media_baseline_raises_exception(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "`non_media_treatments_baseline_normalized` must be passed to"
        " `_incremental_outcome_impl` when `non_media_treatments` data is"
        " present.",
    ):
      self.analyzer._incremental_outcome_impl(
          data_tensors=analyzer.DataTensors(
              non_media_treatments=self.meridian.non_media_treatments
          ),
          dist_tensors=analyzer.DistributionTensors(),
      )

  def test_get_incremental_kpi_without_non_media_baseline_raises_exception(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "`non_media_treatments_baseline_normalized` must be passed to"
        " `_get_incremental_kpi` when `non_media_treatments` data is"
        " present.",
    ):
      self.analyzer._get_incremental_kpi(
          data_tensors=analyzer.DataTensors(
              non_media_treatments=self.meridian.non_media_treatments
          ),
          dist_tensors=analyzer.DistributionTensors(),
      )

  @mock.patch("meridian.analysis.analyzer.np.histogram")
  def test_hill_curves_scaled_histogram_avoids_nan_on_zero_counts(
      self, mock_np_histogram
  ):
    n_bins = 10
    mock_counts = np.zeros(n_bins)
    mock_buckets = np.linspace(0, 1, n_bins + 1)
    mock_np_histogram.return_value = (mock_counts, mock_buckets)

    hill_curves_df = self.analyzer.hill_curves(n_bins=n_bins)
    histogram_part = hill_curves_df[
        (hill_curves_df[constants.CHANNEL_TYPE] == constants.MEDIA)
        & (hill_curves_df[constants.COUNT_HISTOGRAM].notna())
    ].copy()

    has_nan = histogram_part[constants.SCALED_COUNT_HISTOGRAM].isna().any()
    self.assertFalse(
        has_nan,
        "NaN found in scaled_count_histogram.",
    )

  def test_get_historical_spend_deprecated_warning(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "`get_historical_spend` is deprecated. Please use"
        " `get_aggregated_spend` with `new_data=None` instead.",
    ):
      self.analyzer.get_historical_spend(
          selected_times=None, include_media=True, include_rf=True
      )

  def test_get_historical_spend_calls_get_aggregated_spend(self):
    with mock.patch.object(
        self.analyzer,
        "get_aggregated_spend",
        autospec=True,
    ) as mock_get_aggregated_spend:
      self.analyzer.get_historical_spend(
          selected_times=None, include_media=True, include_rf=True
      )
      mock_get_aggregated_spend.assert_called_once_with(
          selected_times=None, include_media=True, include_rf=True
      )

  def test_get_aggregated_spend_correct_channel_names(self):
    actual_hist_spend = self.analyzer.get_aggregated_spend()
    expected_channel_names = self.input_data.get_all_paid_channels()

    self.assertSameElements(
        expected_channel_names, actual_hist_spend.channel.data
    )

  def test_get_aggregated_spend_correct_values(self):
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
    actual_hist_spend = meridian_analyzer.get_aggregated_spend()
    expected_all_spend = np.array([3.3, 6.3, 9.3])
    backend_test_utils.assert_allclose(
        expected_all_spend, actual_hist_spend.data
    )

  def test_get_aggregated_spend_new_data_correct_values(self):
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

    model_spec = spec.ModelSpec()
    meridian = model.Meridian(input_data=data, model_spec=model_spec)
    meridian_analyzer = analyzer.Analyzer(meridian)

    # All times are selected.
    new_media_spend = backend.to_tensor([[[1, 2], [2, 3], [3, 4]]])
    actual_hist_spend = meridian_analyzer.get_aggregated_spend(
        new_data=analyzer.DataTensors(media_spend=new_media_spend)
    )
    expected_all_spend = np.array([6, 9, 9.3])
    backend_test_utils.assert_allclose(
        expected_all_spend, actual_hist_spend.data
    )

  def test_get_aggregated_spend_selected_times_correct_values(self):
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

    actual_hist_spend = meridian_analyzer.get_aggregated_spend(
        selected_times=selected_times
    )
    expected_all_spend = np.array([2.1, 4.1, 6.1])
    backend_test_utils.assert_allclose(
        expected_all_spend, actual_hist_spend.data
    )

  def test_get_aggregated_spend_selected_geos_correct_values(self):
    # Set it to None to avoid the dimension checks on inference data.
    self.enter_context(
        mock.patch.object(
            model.Meridian,
            "inference_data",
            new=property(lambda unused_self: None),
        )
    )

    data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=2,
        n_times=2,
        n_media_times=3,
        n_media_channels=1,
        n_rf_channels=1,
        seed=0,
    )

    # Avoid the pytype check complaint.
    assert data.media_channel is not None and data.rf_channel is not None

    data.media_spend = xr.DataArray(
        np.array([[[1.0], [1.1]], [[1.2], [1.3]]]),
        dims=["geo", "time", "media_channel"],
        coords={
            "geo": data.geo.values,
            "time": data.time.values,
            "media_channel": data.media_channel.values,
        },
    )
    data.rf_spend = xr.DataArray(
        np.array([[[2.0], [2.1]], [[2.2], [2.3]]]),
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

    # Select only first geo.
    selected_geos = ["geo_0"]

    actual_hist_spend = meridian_analyzer.get_aggregated_spend(
        selected_geos=selected_geos
    )
    expected_all_spend = np.array([1.0 + 1.1, 2.0 + 2.1])
    backend_test_utils.assert_allclose(
        expected_all_spend, actual_hist_spend.data
    )

  def test_get_aggregated_spend_with_single_dim_spends(self):
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

    # Patch validation to avoid errors due to mismatched inference data
    with mock.patch.object(model.Meridian, "_validate_injected_inference_data"):
      meridian = model.Meridian(input_data=data, model_spec=model_spec)
    meridian_analyzer = analyzer.Analyzer(meridian)

    n_sub_times = 4
    n_sub_geos = 3

    selected_geos = data.geo.values.tolist()[:n_sub_geos]
    selected_times = data.time.values[-n_sub_times:].tolist()
    actual_hist_spends = meridian_analyzer.get_aggregated_spend(
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

    # The spend is interpolated based on the ratio of media execution in the
    # selected times to the media execution in the entire time period.
    # Shape (n_geos, n_times, n_media_channels + n_rf_channels)
    all_media_exe_values = data.get_all_media_and_rf()

    # Get the media execution values in the selected times.
    # Shape (n_media_channels + n_rf_channels)
    target_media_exe_values = np.sum(
        all_media_exe_values[:n_sub_geos, -n_sub_times:], axis=(0, 1)
    )
    # Get the media execution values in the entire time period.
    # Shape (n_media_channels + n_rf_channels)
    total_media_exe_values = np.sum(
        all_media_exe_values[:, -meridian.n_times :], axis=(0, 1)
    )
    # The ratio will be used to interpolate the spend.
    ratio = target_media_exe_values / total_media_exe_values
    # Shape (n_media_channels + n_rf_channels)
    expected_all_spend = data.get_total_spend() * ratio

    backend_test_utils.assert_allclose(
        expected_all_spend, actual_hist_spends.data
    )

  def test_get_aggregated_spend_with_empty_times(self):
    actual = self.analyzer.get_aggregated_spend(selected_times=[])
    backend_test_utils.assert_allequal(
        actual.data, np.zeros((_N_MEDIA_CHANNELS + _N_RF_CHANNELS))
    )

  def test_get_aggregated_spend_with_no_channel_selected(self):
    selected_times = self.input_data.time.values.tolist()

    with self.assertRaisesRegex(
        ValueError, "At least one of include_media or include_rf must be True."
    ):
      self.analyzer.get_aggregated_spend(
          selected_times, include_media=False, include_rf=False
      )

  def test_optimal_frequency_data_correct(self):
    """Verifies that optimal_freq returns correct data."""
    actual = self.analyzer.optimal_freq(
        freq_grid=[1.0, 2.0, 3.0],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        use_posterior=True,
    )

    expected = xr.Dataset(
        coords={
            constants.FREQUENCY: [1.0, 2.0, 3.0],
            constants.RF_CHANNEL: ["rf_ch_0", "rf_ch_1"],
            constants.METRIC: [
                constants.MEAN,
                constants.MEDIAN,
                constants.CI_LO,
                constants.CI_HI,
            ],
        },
        data_vars={
            constants.ROI: (
                [constants.FREQUENCY, constants.RF_CHANNEL, constants.METRIC],
                [
                    [
                        [1.87995589, 1.89611471, 0.94143367, 2.77492642],
                        [1.54409409, 1.52347374, 1.24998319, 1.86007643],
                    ],
                    [
                        [2.14786601, 2.11210585, 1.53225815, 2.79395390],
                        [0.86134517, 0.84939533, 0.67261636, 1.06261110],
                    ],
                    [
                        [2.02160525, 2.01261830, 1.07597828, 2.98439527],
                        [0.60505420, 0.59656411, 0.46712375, 0.75177753],
                    ],
                ],
            ),
            constants.OPTIMAL_FREQUENCY: (
                [constants.RF_CHANNEL],
                [2.0, 1.0],
            ),
            constants.OPTIMIZED_INCREMENTAL_OUTCOME: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1110.80200195, 1092.30798340, 792.43090820, 1444.93640137],
                    [822.81188965, 811.82385254, 666.08703613, 991.19152832],
                ],
            ),
            constants.OPTIMIZED_ROI: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [2.14786601, 2.11210585, 1.53225815, 2.79395390],
                    [1.54409409, 1.52347374, 1.24998319, 1.86007643],
                ],
            ),
            constants.OPTIMIZED_EFFECTIVENESS: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [0.00027731, 0.00027270, 0.00019783, 0.00036073],
                    [0.00019437, 0.00019178, 0.00015735, 0.00023415],
                ],
            ),
            constants.OPTIMIZED_MROI_BY_REACH: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [2.14786744, 2.11213827, 1.53223789, 2.79396868],
                    [1.54409289, 1.52347136, 1.24995458, 1.86003268],
                ],
            ),
            constants.OPTIMIZED_MROI_BY_FREQUENCY: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [2.42458034, 2.36427331, 0.21445109, 4.69357109],
                    [0.27252775, 0.26730531, 0.13575560, 0.41464081],
                ],
            ),
            constants.OPTIMIZED_CPIK: (
                [constants.RF_CHANNEL, constants.METRIC],
                [
                    [1.58747256, 1.59502435, 1.12389696, 2.04926896],
                    [2.10803270, 2.12235498, 1.68811047, 2.51203609],
                ],
            ),
        },
        attrs={
            constants.CONFIDENCE_LEVEL: constants.DEFAULT_CONFIDENCE_LEVEL,
            "use_posterior": True,
        },
    )

    xr.testing.assert_allclose(actual, expected, atol=0.1)

  def test_optimal_frequency_kpi_returns_correct_structure(self):
    """Verifies that use_kpi=True for optimal_freq returns correct structure."""
    result = self.analyzer.optimal_freq(
        freq_grid=[1.0, 2.0, 3.0],
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        use_posterior=True,
        use_kpi=True,
    )

    self.assertIsInstance(result, xr.Dataset)
    self.assertEqual(
        list(result.data_vars.keys()),
        [
            constants.ROI,
            constants.OPTIMAL_FREQUENCY,
            constants.OPTIMIZED_INCREMENTAL_OUTCOME,
            constants.OPTIMIZED_ROI,
            constants.OPTIMIZED_EFFECTIVENESS,
            constants.OPTIMIZED_MROI_BY_REACH,
            constants.OPTIMIZED_MROI_BY_FREQUENCY,
            constants.OPTIMIZED_CPIK,
        ],
    )
    self.assertFalse(result.attrs[constants.IS_REVENUE_KPI])

  def test_optimal_freq_new_times_data_correct(self):
    max_lag = 15
    n_new_times = 15
    total_times = max_lag + n_new_times
    actual = self.analyzer.optimal_freq(
        new_data=analyzer.DataTensors(
            rf_impressions=self.meridian.rf_tensors.reach[..., -total_times:, :]
            * self.meridian.rf_tensors.frequency[..., -total_times:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -total_times:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -total_times:],
        ),
        freq_grid=[1.0, 2.0, 3.0],
        selected_times=[False] * max_lag + [True] * n_new_times,
    )
    expected = self.analyzer.optimal_freq(
        selected_times=list(self.input_data.time.values)[-n_new_times:],
        freq_grid=[1.0, 2.0, 3.0],
    )
    xr.testing.assert_allclose(actual, expected)

  @parameterized.product(
      use_posterior=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_negative_baseline_probability_returns_correct_shape(
      self,
      use_posterior: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    prob = self.analyzer.negative_baseline_probability(
        use_posterior=use_posterior,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

    self.assertEqual(prob.shape, ())

  @parameterized.named_parameters(
      dict(
          testcase_name="prior",
          use_posterior=False,
          selected_geos=None,
          selected_times=None,
          expected_value=0.5,
      ),
      dict(
          testcase_name="prior_selected_times",
          use_posterior=False,
          selected_geos=None,
          selected_times=("2021-04-19", "2021-09-13", "2021-12-13"),
          expected_value=0.5,
      ),
      dict(
          testcase_name="prior_selected_geos",
          use_posterior=False,
          selected_geos=("geo_1", "geo_3"),
          selected_times=None,
          expected_value=0.5,
      ),
      dict(
          testcase_name="prior_selected_geos_times",
          use_posterior=False,
          selected_geos=("geo_1", "geo_3"),
          selected_times=("2021-04-19", "2021-09-13", "2021-12-13"),
          expected_value=0.5,
      ),
      dict(
          testcase_name="posterior",
          use_posterior=True,
          selected_geos=None,
          selected_times=None,
          expected_value=1.0,
      ),
      dict(
          testcase_name="posterior_selected_times",
          use_posterior=True,
          selected_geos=None,
          selected_times=("2021-04-19", "2021-09-13", "2021-12-13"),
          expected_value=1.0,
      ),
      dict(
          testcase_name="posterior_selected_geos",
          use_posterior=True,
          selected_geos=("geo_1", "geo_3"),
          selected_times=None,
          expected_value=1.0,
      ),
      dict(
          testcase_name="posterior_selected_geos_times",
          use_posterior=True,
          selected_geos=("geo_1", "geo_3"),
          selected_times=("2021-04-19", "2021-09-13", "2021-12-13"),
          expected_value=1.0,
      ),
  )
  def test_negative_baseline_probability_returns_correct_values(
      self,
      use_posterior,
      selected_geos,
      selected_times,
      expected_value,
  ):
    """Exact output tests for `Analyzer.negative_baseline_probability`."""
    prob = self.analyzer.negative_baseline_probability(
        use_posterior=use_posterior,
        selected_geos=selected_geos,
        selected_times=selected_times,
    )
    backend_test_utils.assert_allclose(
        prob,
        expected_value,
        atol=1e-5,
        rtol=1e-5,
    )

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      by_reach=[False, True],
      use_kpi=[False, True],
  )
  def test_marginal_roi_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      by_reach: bool,
      use_kpi: bool,
  ):
    mroi = self.analyzer.marginal_roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
        by_reach=by_reach,
        use_kpi=use_kpi,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(mroi.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="use_prior",
          use_posterior=False,
          by_reach=False,
          expected_mroi=analysis_test_utils.MROI_MEDIA_AND_RF_USE_PRIOR,
      ),
      dict(
          testcase_name="use_prior_by_reach",
          use_posterior=False,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_MEDIA_AND_RF_USE_PRIOR_BY_REACH,
      ),
      dict(
          testcase_name="use_posterior",
          use_posterior=True,
          by_reach=False,
          expected_mroi=analysis_test_utils.MROI_MEDIA_AND_RF_USE_POSTERIOR,
      ),
      dict(
          testcase_name="use_posterior_by_reach",
          use_posterior=True,
          by_reach=True,
          expected_mroi=analysis_test_utils.MROI_MEDIA_AND_RF_USE_POSTERIOR_BY_REACH,
      ),
  )
  def test_marginal_roi_values(
      self,
      use_posterior: bool,
      by_reach: bool,
      expected_mroi: tuple[float, ...],
  ):
    mroi = self.analyzer.marginal_roi(
        by_reach=by_reach,
        use_posterior=use_posterior,
    )

    backend_test_utils.assert_allclose(
        mroi,
        backend.to_tensor(expected_mroi),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_marginal_roi_new_times_data_correct(self):
    max_lag = 15
    n_new_times = 15
    total_times = max_lag + n_new_times
    actual = self.analyzer.marginal_roi(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -total_times:, :],
            media_spend=self.meridian.media_tensors.media_spend[
                ..., -total_times:, :
            ],
            reach=self.meridian.rf_tensors.reach[..., -total_times:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -total_times:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -total_times:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -total_times:],
        ),
        selected_times=[False] * max_lag + [True] * n_new_times,
    )
    expected = self.analyzer.marginal_roi(
        selected_times=list(self.input_data.time.values)[-n_new_times:]
    )
    backend_test_utils.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      use_kpi=[False, True],
  )
  def test_roi_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      use_kpi: bool,
  ):
    roi = self.analyzer.roi(
        use_posterior=use_posterior,
        aggregate_geos=aggregate_geos,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
    )
    expected_shape = (_N_CHAINS, _N_KEEP) if use_posterior else (1, _N_DRAWS)
    if not aggregate_geos:
      expected_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    expected_shape += (_N_MEDIA_CHANNELS + _N_RF_CHANNELS,)
    self.assertEqual(roi.shape, expected_shape)

  def test_roi_default_returns_correct_value(self):
    roi = self.analyzer.roi()
    total_spend = self.analyzer.filter_and_aggregate_geos_and_times(
        self.meridian.total_spend
    )
    expected_roi = (
        self.analyzer.incremental_outcome(include_non_paid_channels=False)
        / total_spend
    )
    backend_test_utils.assert_allclose(expected_roi, roi)

  def test_roi_new_params_correct(self):
    max_lag = 15
    n_new_times = 15
    total_times = max_lag + n_new_times
    actual = self.analyzer.roi(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -total_times:, :],
            media_spend=self.meridian.media_tensors.media_spend[
                ..., -total_times:, :
            ],
            reach=self.meridian.rf_tensors.reach[..., -total_times:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -total_times:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -total_times:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -total_times:],
        ),
        selected_times=[False] * max_lag + [True] * n_new_times,
    )
    expected = self.analyzer.roi(
        selected_times=list(self.input_data.time.values)[-n_new_times:]
    )
    backend_test_utils.assert_allclose(actual, expected)

  @parameterized.product(
      use_posterior=[False, True],
      aggregate_geos=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
  )
  def test_cpik_returns_correct_shape(
      self,
      use_posterior: bool,
      aggregate_geos: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
  ):
    cpik = self.analyzer.cpik(
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

  def test_cpik_default_returns_correct_value(self):
    cpik = self.analyzer.cpik()
    total_spend = self.analyzer.filter_and_aggregate_geos_and_times(
        self.meridian.total_spend
    )
    expected_cpik = total_spend / self.analyzer.incremental_outcome(
        use_kpi=True, include_non_paid_channels=False
    )
    backend_test_utils.assert_allclose(expected_cpik, cpik)

  def test_cpik_new_times_data_correct(self):
    max_lag = 15
    n_new_times = 15
    total_times = max_lag + n_new_times
    actual = self.analyzer.cpik(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -total_times:, :],
            media_spend=self.meridian.media_tensors.media_spend[
                ..., -total_times:, :
            ],
            reach=self.meridian.rf_tensors.reach[..., -total_times:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -total_times:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -total_times:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -total_times:],
        ),
        selected_times=[False] * max_lag + [True] * n_new_times,
    )
    expected = self.analyzer.cpik(
        selected_times=list(self.input_data.time.values)[-n_new_times:]
    )
    backend_test_utils.assert_allclose(actual, expected)

  def test_media_summary_returns_correct_values(self):
    media_summary = self.analyzer.summary_metrics(
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
    backend_test_utils.assert_allclose(
        media_summary.impressions, analysis_test_utils.SAMPLE_IMPRESSIONS
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_impressions,
        analysis_test_utils.SAMPLE_PCT_OF_IMPRESSIONS,
    )
    backend_test_utils.assert_allclose(
        media_summary.spend,
        analysis_test_utils.SAMPLE_SPEND,
        atol=1e-4,
        rtol=1e-4,
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_spend,
        analysis_test_utils.SAMPLE_PCT_OF_SPEND,
        atol=1e-4,
        rtol=1e-4,
    )
    backend_test_utils.assert_allclose(
        media_summary.cpm,
        analysis_test_utils.SAMPLE_CPM,
    )
    backend_test_utils.assert_allclose(
        media_summary.incremental_outcome,
        analysis_test_utils.SAMPLE_INCREMENTAL_OUTCOME,
        atol=1e-3,
        rtol=1e-3,
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_contribution,
        analysis_test_utils.SAMPLE_PCT_OF_CONTRIBUTION,
        atol=1e-3,
        rtol=1e-3,
    )
    backend_test_utils.assert_allclose(
        media_summary.roi, analysis_test_utils.SAMPLE_ROI, atol=1e-3, rtol=1e-3
    )
    backend_test_utils.assert_allclose(
        media_summary.effectiveness,
        analysis_test_utils.SAMPLE_EFFECTIVENESS,
        atol=1e-4,
        rtol=1e-4,
    )
    backend_test_utils.assert_allclose(
        media_summary.mroi,
        analysis_test_utils.SAMPLE_MROI,
        atol=1e-3,
        rtol=1e-3,
    )
    backend_test_utils.assert_allclose(
        media_summary.cpik,
        analysis_test_utils.SAMPLE_CPIK,
        atol=1e-3,
        rtol=1e-3,
    )

  def test_summary_metrics_kpi_values(self):
    """Verifies that use_kpi=True produces different (and correct) metric values."""
    media_summary = self.analyzer.summary_metrics(
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

    backend_test_utils.assert_allclose(
        media_summary.incremental_outcome,
        analysis_test_utils.SAMPLE_INC_OUTCOME_KPI,
        atol=1e-2,
        rtol=1e-2,
    )
    backend_test_utils.assert_allclose(
        media_summary.roi,
        analysis_test_utils.SAMPLE_ROI_KPI,
        atol=1e-3,
        rtol=1e-3,
    )
    backend_test_utils.assert_allclose(
        media_summary.effectiveness,
        analysis_test_utils.SAMPLE_EFFECTIVENESS_KPI,
        atol=1e-3,
        rtol=1e-3,
    )
    backend_test_utils.assert_allclose(
        media_summary.mroi,
        analysis_test_utils.SAMPLE_MROI_KPI,
        atol=1e-3,
        rtol=1e-3,
    )

  def test_media_summary_with_new_data_returns_correct_values(self):
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
    new_data = analyzer.DataTensors(
        media=backend.to_tensor(data1.media, dtype=backend.float32),
        reach=backend.to_tensor(data1.reach, dtype=backend.float32),
        media_spend=backend.to_tensor(data1.media_spend, dtype=backend.float32),
        organic_media=backend.to_tensor(
            data1.organic_media, dtype=backend.float32
        ),
        organic_reach=backend.to_tensor(
            data1.organic_reach, dtype=backend.float32
        ),
    )
    media_summary = self.analyzer.summary_metrics(
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

    self.assertFalse(
        np.allclose(
            np.array(media_summary.impressions),
            analysis_test_utils.SAMPLE_IMPRESSIONS,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.impressions,
        analysis_test_utils.SAMPLE_IMPRESSIONS_NEW_DATA,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.pct_of_impressions),
            analysis_test_utils.SAMPLE_PCT_OF_IMPRESSIONS,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_impressions,
        analysis_test_utils.SAMPLE_PCT_OF_IMPRESSIONS_NEW_DATA,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.spend), analysis_test_utils.SAMPLE_SPEND
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.spend,
        analysis_test_utils.SAMPLE_SPEND_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.pct_of_spend),
            analysis_test_utils.SAMPLE_PCT_OF_SPEND,
            atol=1e-4,
            rtol=1e-4,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_spend,
        analysis_test_utils.SAMPLE_PCT_OF_SPEND_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertFalse(
        np.allclose(np.array(media_summary.cpm), analysis_test_utils.SAMPLE_CPM)
    )
    backend_test_utils.assert_allclose(
        media_summary.cpm,
        analysis_test_utils.SAMPLE_CPM_NEW_DATA,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.incremental_outcome),
            analysis_test_utils.SAMPLE_INCREMENTAL_OUTCOME,
            atol=1e-3,
            rtol=1e-3,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.incremental_outcome,
        analysis_test_utils.SAMPLE_INCREMENTAL_OUTCOME_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.pct_of_contribution),
            analysis_test_utils.SAMPLE_PCT_OF_CONTRIBUTION,
            atol=1e-3,
            rtol=1e-3,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.pct_of_contribution,
        analysis_test_utils.SAMPLE_PCT_OF_CONTRIBUTION_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.roi),
            analysis_test_utils.SAMPLE_ROI,
            atol=1e-3,
            rtol=1e-3,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.roi,
        analysis_test_utils.SAMPLE_ROI_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.effectiveness),
            analysis_test_utils.SAMPLE_EFFECTIVENESS,
            atol=1e-4,
            rtol=1e-4,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.effectiveness,
        analysis_test_utils.SAMPLE_EFFECTIVENESS_NEW_DATA,
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.mroi),
            analysis_test_utils.SAMPLE_MROI,
            atol=1e-3,
            rtol=1e-3,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.mroi,
        analysis_test_utils.SAMPLE_MROI_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )
    self.assertFalse(
        np.allclose(
            np.array(media_summary.cpik),
            analysis_test_utils.SAMPLE_CPIK,
            atol=1e-3,
            rtol=1e-3,
        )
    )
    backend_test_utils.assert_allclose(
        media_summary.cpik,
        analysis_test_utils.SAMPLE_CPIK_NEW_DATA,
        atol=1e-3,
        rtol=1e-3,
    )

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      use_kpi=[False, True],
  )
  def test_media_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      use_kpi: bool,
  ):
    analyzer_ = self.analyzer

    num_paid_channels = _N_MEDIA_CHANNELS + _N_RF_CHANNELS

    media_summary = analyzer_.summary_metrics(
        confidence_level=0.8,
        marginal_roi_by_reach=False,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
    )

    base_shape = ()
    if not aggregate_geos:
      base_shape += (
          (len(selected_geos),) if selected_geos is not None else (_N_GEOS,)
      )
    if not aggregate_times:
      base_shape += (
          (len(selected_times),)
          if selected_times is not None
          else (media_summary.sizes[constants.TIME],)
      )

    expected_channel_shape = base_shape + (num_paid_channels + 1,)
    expected_metric_shape = expected_channel_shape + (4, 2)

    self.assertEqual(media_summary.impressions.shape, expected_channel_shape)
    self.assertEqual(
        media_summary.pct_of_impressions.shape, expected_channel_shape
    )
    self.assertEqual(media_summary.spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.pct_of_spend.shape, expected_channel_shape)
    self.assertEqual(media_summary.cpm.shape, expected_channel_shape)

    self.assertEqual(
        media_summary.incremental_outcome.shape, expected_metric_shape
    )
    self.assertEqual(
        media_summary.pct_of_contribution.shape, expected_metric_shape
    )

    if aggregate_times:
      self.assertEqual(media_summary.roi.shape, expected_metric_shape)
      self.assertEqual(media_summary.mroi.shape, expected_metric_shape)
      self.assertEqual(media_summary.cpik.shape, expected_metric_shape)
      self.assertEqual(media_summary.effectiveness.shape, expected_metric_shape)
    else:
      self.assertNotIn(constants.ROI, media_summary.data_vars)
      self.assertNotIn(constants.EFFECTIVENESS, media_summary.data_vars)
      self.assertNotIn(constants.MROI, media_summary.data_vars)
      self.assertNotIn(constants.CPIK, media_summary.data_vars)

  def test_summary_metrics_new_times_data_returns_correct_variables(self):
    summary_metrics = self.analyzer.summary_metrics(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -15:, :],
            media_spend=self.meridian.media_tensors.media_spend[..., -15:, :],
            reach=self.meridian.rf_tensors.reach[..., -15:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -15:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -15:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -15:],
            organic_media=self.meridian.organic_media_tensors.organic_media[
                ..., -15:, :
            ],
            organic_reach=self.meridian.organic_rf_tensors.organic_reach[
                ..., -15:, :
            ],
            organic_frequency=self.meridian.organic_rf_tensors.organic_frequency[
                ..., -15:, :
            ],
        )
    )
    self.assertEqual(
        list(summary_metrics.data_vars.keys()),
        [
            constants.IMPRESSIONS,
            constants.PCT_OF_IMPRESSIONS,
            constants.SPEND,
            constants.PCT_OF_SPEND,
            constants.CPM,
            constants.INCREMENTAL_OUTCOME,
            constants.ROI,
            constants.EFFECTIVENESS,
            constants.MROI,
            constants.CPIK,
        ],
    )

  def test_summary_metrics_new_times_data_aggregate_times_false(self):
    summary_metrics = self.analyzer.summary_metrics(
        new_data=analyzer.DataTensors(
            media=self.meridian.media_tensors.media[..., -5:, :],
            media_spend=self.meridian.media_tensors.media_spend[..., -5:, :],
            reach=self.meridian.rf_tensors.reach[..., -5:, :],
            frequency=self.meridian.rf_tensors.frequency[..., -5:, :],
            rf_spend=self.meridian.rf_tensors.rf_spend[..., -5:, :],
            revenue_per_kpi=self.meridian.revenue_per_kpi[..., -5:],
            organic_media=self.meridian.organic_media_tensors.organic_media[
                ..., -5:, :
            ],
            organic_reach=self.meridian.organic_rf_tensors.organic_reach[
                ..., -5:, :
            ],
            organic_frequency=self.meridian.organic_rf_tensors.organic_frequency[
                ..., -5:, :
            ],
        ),
        selected_times=[False, False, True, True, False],
        aggregate_times=False,
    )
    self.assertEqual(list(summary_metrics.time), [2, 3])
    self.assertEqual(
        list(summary_metrics.data_vars.keys()),
        [
            "impressions",
            "pct_of_impressions",
            "spend",
            "pct_of_spend",
            "cpm",
            "incremental_outcome",
        ],
    )

  @parameterized.product(
      aggregate_geos=[False, True],
      aggregate_times=[False, True],
      selected_geos=[None, ["geo_1", "geo_3"]],
      selected_times=[None, ["2021-04-19", "2021-09-13", "2021-12-13"]],
      use_kpi=[False, True],
  )
  def test_baseline_summary_returns_correct_shapes(
      self,
      aggregate_geos: bool,
      aggregate_times: bool,
      selected_geos: Sequence[str] | None,
      selected_times: Sequence[str] | None,
      use_kpi: bool,
  ):
    analyzer_ = self.analyzer

    media_summary = analyzer_.baseline_summary_metrics(
        confidence_level=0.8,
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        selected_geos=selected_geos,
        selected_times=selected_times,
        use_kpi=use_kpi,
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

  def test_rhat_media_and_rf_correct(self):
    rhat = self.analyzer.get_rhat()
    expected_params = (
        set(constants.COMMON_PARAMETER_NAMES)
        | set(constants.MEDIA_PARAMETER_NAMES)
        | set(constants.RF_PARAMETER_NAMES)
        | set(constants.ORGANIC_MEDIA_PARAMETER_NAMES)
        | set(constants.ORGANIC_RF_PARAMETER_NAMES)
        | set(constants.NON_MEDIA_PARAMETER_NAMES)
    ) - {
        constants.CONTRIBUTION_OM,
        constants.CONTRIBUTION_ORF,
        constants.CONTRIBUTION_N,
    }

    self.assertSetEqual(set(rhat.keys()), expected_params)

  def test_rhat_summary_media_and_rf_correct(self):
    rhat_summary = self.analyzer.rhat_summary()
    all_params = (
        set(constants.COMMON_PARAMETER_NAMES)
        | set(constants.MEDIA_PARAMETER_NAMES)
        | set(constants.RF_PARAMETER_NAMES)
        | set(constants.ORGANIC_MEDIA_PARAMETER_NAMES)
        | set(constants.ORGANIC_RF_PARAMETER_NAMES)
        | set(constants.NON_MEDIA_PARAMETER_NAMES)
    )
    excluded_params = {
        constants.CONTRIBUTION_OM,
        constants.CONTRIBUTION_ORF,
        constants.CONTRIBUTION_N,
        constants.SLOPE_M,
        constants.SLOPE_OM,
    }
    expected_params = all_params - excluded_params

    self.assertEqual(rhat_summary.shape, (len(expected_params), 7))
    self.assertSetEqual(set(rhat_summary.param), expected_params)

  def test_predictive_accuracy_without_holdout_id_columns_correct(self):
    predictive_accuracy_dataset = self.analyzer.predictive_accuracy()

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

  def test_predictive_accuracy_kpi_returns_correct_structure(self):
    """Verifies predictive accuracy structure when use_kpi=True."""
    dataset = self.analyzer.predictive_accuracy(use_kpi=True)

    self.assertListEqual(
        list(dataset[constants.METRIC].values),
        [constants.R_SQUARED, constants.MAPE, constants.WMAPE],
    )
    self.assertFalse(np.isnan(dataset[constants.VALUE]).all())

  @mock.patch.object(
      model.Meridian, "is_national", new=property(lambda unused_self: True)
  )
  def test_predictive_accuracy_national(self):
    predictive_accuracy_dataset = self.analyzer.predictive_accuracy()
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
    predictive_accuracy_dataset = self.analyzer.predictive_accuracy(
        **predictive_accuracy_dims_kwargs,
    )
    df = (
        predictive_accuracy_dataset[constants.VALUE]
        .to_dataframe()
        .reset_index()
    )

    actual_values = list(df[constants.VALUE])
    if not selected_geos and not selected_times:
      backend_test_utils.assert_allclose(
          actual_values,
          analysis_test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_NO_GEOS_OR_TIMES,
          atol=1e-3,
      )
    elif selected_geos and not selected_times:
      backend_test_utils.assert_allclose(
          actual_values,
          analysis_test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_GEOS_NO_TIMES,
          atol=1e-3,
      )
    elif not selected_geos and selected_times:
      backend_test_utils.assert_allclose(
          actual_values,
          analysis_test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_NO_GEOS,
          atol=1e-3,
      )
    else:
      backend_test_utils.assert_allclose(
          actual_values,
          analysis_test_utils.PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_AND_GEOS,
          atol=1e-3,
      )

  def test_predictive_accuracy_with_holdout_id_table_properties_correct(self):
    n_geos = self.meridian.n_geos
    n_times = self.meridian.n_times
    holdout_id = np.full([n_geos, n_times], False)
    for i in range(n_geos):
      holdout_id[i, np.random.choice(n_times, int(np.round(0.2 * n_times)))] = (
          True
      )
    model_spec = spec.ModelSpec(holdout_id=holdout_id)
    meridian = model.Meridian(model_spec=model_spec, input_data=self.input_data)
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
    n_geos = len(self.input_data.geo)
    n_times = len(self.input_data.time)

    np.random.seed(0)

    holdout_id = np.full([n_geos, n_times], False)
    for i in range(n_geos):
      holdout_id[i, np.random.choice(n_times, int(np.round(0.2 * n_times)))] = (
          True
      )
    model_spec = spec.ModelSpec(holdout_id=holdout_id)  # Set holdout_id
    meridian = model.Meridian(model_spec=model_spec, input_data=self.input_data)
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

    actual_values = list(df[constants.VALUE])

    if not selected_geos and not selected_times:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_NO_GEOS_OR_TIMES
      )
    elif selected_geos and not selected_times:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_GEOS_NO_TIMES
      )
    elif not selected_geos and selected_times:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_TIMES_NO_GEOS
      )
    else:
      expected_values = (
          analysis_test_utils.PREDICTIVE_ACCURACY_HOLDOUT_ID_TIMES_AND_GEO
      )

    backend_test_utils.assert_allclose(
        actual_values,
        expected_values,
        atol=2e-3,
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
    meridian = model.Meridian(model_spec=model_spec, input_data=self.input_data)
    meridian_analyzer = analyzer.Analyzer(meridian)

    ds = meridian_analyzer.expected_vs_actual_data(
        aggregate_geos=aggregate_geos,
        aggregate_times=aggregate_times,
        split_by_holdout_id=split_by_holdout_id,
    )

    expected_actual_values = (
        meridian.kpi
        if self.input_data.revenue_per_kpi is None
        else meridian.kpi * self.input_data.revenue_per_kpi
    )  # shape (n_geos, n_times)

    axis_to_sum = tuple(
        ([0] if aggregate_geos else []) + ([1] if aggregate_times else [])
    )
    expected_actual_values = np.sum(expected_actual_values, axis=axis_to_sum)

    if aggregate_geos:
      self.assertNotIn(constants.GEO, ds.coords)
    else:
      self.assertListEqual(
          list(ds.geo.values), list(self.input_data.geo.values)
      )

    if aggregate_times:
      self.assertNotIn(constants.TIME, ds.coords)
    else:
      self.assertListEqual(
          list(ds.time.values), list(self.input_data.time.values)
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
    backend_test_utils.assert_allclose(
        ds.actual.values,
        expected_actual_values,
        atol=1e-5,
    )

  def test_expected_vs_actual_warns_if_split_by_holdout_id_without_holdout_id(
      self,
  ):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.analyzer.expected_vs_actual_data(split_by_holdout_id=True)

      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn(
          "`split_by_holdout_id` is True but `holdout_id` is `None`. Data will"
          " not be split.",
          str(w[0].message),
      )

  def test_response_curves_check_both_channel_types_returns_correct_spend(self):
    response_curve_data = self.analyzer.response_curves(by_reach=False)
    response_data_spend = response_curve_data.spend.values

    media_summary_spend = self.analyzer.summary_metrics(
        confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL,
        marginal_roi_by_reach=False,
    ).spend[:-1]
    backend_test_utils.assert_allequal(
        media_summary_spend * 2,
        response_data_spend[-1],
    )

  def test_response_curves_selected_times_wrong_type(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must be a list of strings or a list of booleans.",
    ):
      self.analyzer.response_curves(
          selected_times=["random_time", False, True],
      )

  def test_response_curves_selected_times_wrong_time_dim_names(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer.response_curves(selected_times=["random_time"])

  def test_response_curves_selected_times_no_new_data_wrong_time(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` must match the time dimension names from "
        "meridian.InputData.",
    ):
      self.analyzer.response_curves(
          selected_times=["2022-01-01"]  # Not in input_data.time
      )

  def test_response_curves_new_data_selected_times_wrong_time(self):
    n_new_times = 15
    new_data = analyzer.DataTensors(
        media=self.meridian.media_tensors.media[..., -n_new_times:, :],
        reach=self.meridian.rf_tensors.reach[..., -n_new_times:, :],
        frequency=self.meridian.rf_tensors.frequency[..., -n_new_times:, :],
        revenue_per_kpi=self.meridian.revenue_per_kpi[..., -n_new_times:],
        media_spend=self.meridian.media_tensors.media_spend[
            ..., -n_new_times:, :
        ],
        rf_spend=self.meridian.rf_tensors.rf_spend[..., -n_new_times:, :],
        organic_media=self.meridian.organic_media_tensors.organic_media[
            ..., -n_new_times:, :
        ],
        organic_reach=self.meridian.organic_rf_tensors.organic_reach[
            ..., -n_new_times:, :
        ],
        organic_frequency=self.meridian.organic_rf_tensors.organic_frequency[
            ..., -n_new_times:, :
        ],
        non_media_treatments=self.meridian.non_media_treatments[
            ..., -n_new_times:, :
        ],
        time=self.meridian.input_data.time.values[-n_new_times:],
    )
    with self.assertRaisesRegex(
        ValueError,
        "If `media`, `reach`, `frequency`, `organic_media`, `organic_reach`,"
        " `organic_frequency`, `non_media_treatments`, or `revenue_per_kpi` is"
        " provided with a different number of time periods than in `InputData`,"
        r" then \(1\) `selected_times` must be a list of booleans with length"
        r" equal to the number of time periods in the new data, or \(2\)"
        " `selected_times` must be a list of strings and `new_time` must be"
        " provided and `selected_times` must be a subset of `new_time`.",
    ):
      self.analyzer.response_curves(
          new_data=new_data, selected_times=["2022-01-01"]
      )

  @parameterized.named_parameters(
      dict(testcase_name="historical_frequency", use_optimal_frequency=False),
      dict(testcase_name="optimal_frequency", use_optimal_frequency=True),
  )
  def test_response_curves_new_times(self, use_optimal_frequency):
    n_new_times = 15
    new_data = analyzer.DataTensors(
        media=self.meridian.media_tensors.media[..., -n_new_times:, :],
        reach=self.meridian.rf_tensors.reach[..., -n_new_times:, :],
        frequency=self.meridian.rf_tensors.frequency[..., -n_new_times:, :],
        revenue_per_kpi=self.meridian.revenue_per_kpi[..., -n_new_times:],
        media_spend=self.meridian.media_tensors.media_spend[
            ..., -n_new_times:, :
        ],
        rf_spend=self.meridian.rf_tensors.rf_spend[..., -n_new_times:, :],
        organic_media=self.meridian.organic_media_tensors.organic_media[
            ..., -n_new_times:, :
        ],
        organic_reach=self.meridian.organic_rf_tensors.organic_reach[
            ..., -n_new_times:, :
        ],
        organic_frequency=self.meridian.organic_rf_tensors.organic_frequency[
            ..., -n_new_times:, :
        ],
        non_media_treatments=self.meridian.non_media_treatments[
            ..., -n_new_times:, :
        ],
        time=self.meridian.input_data.time.values[-n_new_times:],
    )
    response_curve_data = self.analyzer.response_curves(
        new_data=new_data,
        use_optimal_frequency=use_optimal_frequency,
    )

    # Check coords and shapes.
    self.assertEqual(
        list(response_curve_data.coords.keys()),
        [constants.CHANNEL, constants.METRIC, constants.SPEND_MULTIPLIER],
    )
    self.assertNotIn(constants.TIME, response_curve_data.coords)
    self.assertEqual(
        list(response_curve_data.data_vars.keys()),
        [constants.SPEND, constants.INCREMENTAL_OUTCOME],
    )
    self.assertEqual(
        response_curve_data.sizes[constants.CHANNEL],
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    )
    self.assertEqual(response_curve_data.sizes[constants.METRIC], 3)
    self.assertEqual(response_curve_data.sizes[constants.SPEND_MULTIPLIER], 11)

    # Spend for multiplier 1.0 should match the aggregated spend from new_data.
    expected_spend = self.analyzer.get_aggregated_spend(new_data=new_data)
    backend_test_utils.assert_allclose(
        response_curve_data[constants.SPEND].sel(spend_multiplier=1.0),
        expected_spend.data,
        rtol=1e-3,
        atol=1e-3,
    )

    # Incremental outcome for multiplier 1.0 should match the result of
    # incremental_outcome called directly with the new_data.
    if use_optimal_frequency:
      opt_freq_data = analyzer.DataTensors(
          media=new_data.media,
          rf_impressions=new_data.reach * new_data.frequency,
          media_spend=new_data.media_spend,
          rf_spend=new_data.rf_spend,
          revenue_per_kpi=new_data.revenue_per_kpi,
          organic_media=new_data.organic_media,
          organic_reach=new_data.organic_reach,
          organic_frequency=new_data.organic_frequency,
          non_media_treatments=new_data.non_media_treatments,
      )
      optimal_frequency = self.analyzer.optimal_freq(
          new_data=opt_freq_data,
      ).optimal_frequency
      frequency = backend.ones_like(new_data.frequency) * backend.to_tensor(
          optimal_frequency, dtype=backend.float32
      )
      reach = backend.divide_no_nan(
          new_data.reach * new_data.frequency, frequency
      )
      inc_outcome_new_data = analyzer.DataTensors(
          media=new_data.media,
          reach=reach,
          frequency=frequency,
          media_spend=new_data.media_spend,
          rf_spend=new_data.rf_spend,
          revenue_per_kpi=new_data.revenue_per_kpi,
          organic_media=new_data.organic_media,
          organic_reach=new_data.organic_reach,
          organic_frequency=new_data.organic_frequency,
          non_media_treatments=new_data.non_media_treatments,
      )
    else:
      inc_outcome_new_data = new_data
    expected_inc_outcome = self.analyzer.incremental_outcome(
        new_data=inc_outcome_new_data,
        include_non_paid_channels=False,
        aggregate_geos=True,
        aggregate_times=True,
    )
    inc_outcome_multiplier_1 = response_curve_data[
        constants.INCREMENTAL_OUTCOME
    ].sel(spend_multiplier=1.0, metric="mean")
    backend_test_utils.assert_allclose(
        np.mean(expected_inc_outcome, axis=(0, 1)),
        inc_outcome_multiplier_1.values,
        rtol=1e-3,
        atol=1e-3,
    )

  def test_response_curves_new_times_data_correct(self):
    max_lag = self.meridian.model_spec.max_lag
    n_new_times = 15
    total_times = max_lag + n_new_times
    selected_times_str = self.meridian.input_data.time.values[
        -n_new_times:
    ].tolist()
    new_data_times = self.meridian.input_data.time.values[
        -total_times:
    ].tolist()

    new_data = analyzer.DataTensors(
        media=self.meridian.media_tensors.media[..., -total_times:, :],
        media_spend=self.meridian.media_tensors.media_spend[
            ..., -total_times:, :
        ],
        reach=self.meridian.rf_tensors.reach[..., -total_times:, :],
        frequency=self.meridian.rf_tensors.frequency[..., -total_times:, :],
        rf_spend=self.meridian.rf_tensors.rf_spend[..., -total_times:, :],
        revenue_per_kpi=self.meridian.revenue_per_kpi[..., -total_times:],
        organic_media=self.meridian.organic_media_tensors.organic_media[
            ..., -total_times:, :
        ],
        organic_reach=self.meridian.organic_rf_tensors.organic_reach[
            ..., -total_times:, :
        ],
        organic_frequency=self.meridian.organic_rf_tensors.organic_frequency[
            ..., -total_times:, :
        ],
        non_media_treatments=self.meridian.non_media_treatments[
            ..., -total_times:, :
        ],
        time=new_data_times,
    )
    actual = self.analyzer.response_curves(
        new_data=new_data,
        selected_times=selected_times_str,
    )
    expected = self.analyzer.response_curves(selected_times=selected_times_str)
    xr.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name="historical_frequency", use_optimal_frequency=False),
      dict(testcase_name="optimal_frequency", use_optimal_frequency=True),
  )
  def test_response_curves_new_data_same_times(self, use_optimal_frequency):
    # Modify only the media tensor, keep other tensors the same.
    new_media = self.meridian.media_tensors.media * 2
    new_data = analyzer.DataTensors(
        media=new_media,
    )
    response_curve_data = self.analyzer.response_curves(
        new_data=new_data,
        use_optimal_frequency=use_optimal_frequency,
    )
    original_response_curve_data = self.analyzer.response_curves(
        use_optimal_frequency=use_optimal_frequency,
    )

    # Spend should be the same as original.
    backend_test_utils.assert_allclose(
        response_curve_data[constants.SPEND].values,
        original_response_curve_data[constants.SPEND].values,
    )

    # Incremental outcome for multiplier 1.0 should match the result of
    # incremental_outcome called directly with the new_data.
    if use_optimal_frequency:
      opt_freq_data = analyzer.DataTensors(
          media=new_data.media,
          rf_impressions=self.meridian.rf_tensors.reach
          * self.meridian.rf_tensors.frequency,
          media_spend=self.meridian.media_tensors.media_spend,
          rf_spend=self.meridian.rf_tensors.rf_spend,
          revenue_per_kpi=self.meridian.revenue_per_kpi,
          organic_media=self.meridian.organic_media_tensors.organic_media,
          organic_reach=self.meridian.organic_rf_tensors.organic_reach,
          organic_frequency=self.meridian.organic_rf_tensors.organic_frequency,
          non_media_treatments=self.meridian.non_media_treatments,
      )
      optimal_frequency = self.analyzer.optimal_freq(
          new_data=opt_freq_data,
      ).optimal_frequency
      frequency = backend.ones_like(
          self.meridian.rf_tensors.frequency
      ) * backend.to_tensor(optimal_frequency, dtype=backend.float32)
      reach = backend.divide_no_nan(
          self.meridian.rf_tensors.reach * self.meridian.rf_tensors.frequency,
          frequency,
      )
      inc_outcome_new_data = analyzer.DataTensors(
          media=new_data.media,
          reach=reach,
          frequency=frequency,
      )
    else:
      inc_outcome_new_data = new_data
    expected_inc_outcome = self.analyzer.incremental_outcome(
        new_data=inc_outcome_new_data,
        include_non_paid_channels=False,
        aggregate_geos=True,
        aggregate_times=True,
    )
    inc_outcome_multiplier_1 = response_curve_data[
        constants.INCREMENTAL_OUTCOME
    ].sel(spend_multiplier=1.0, metric="mean")
    backend_test_utils.assert_allclose(
        np.mean(expected_inc_outcome, axis=(0, 1)),
        inc_outcome_multiplier_1.values,
        rtol=1e-3,
        atol=1e-3,
    )

    # Check coords and shapes.
    self.assertEqual(
        list(response_curve_data.coords.keys()),
        [constants.CHANNEL, constants.METRIC, constants.SPEND_MULTIPLIER],
    )
    self.assertEqual(
        list(response_curve_data.data_vars.keys()),
        [constants.SPEND, constants.INCREMENTAL_OUTCOME],
    )
    self.assertEqual(
        response_curve_data.sizes[constants.CHANNEL],
        _N_MEDIA_CHANNELS + _N_RF_CHANNELS,
    )
    self.assertEqual(response_curve_data.sizes[constants.METRIC], 3)
    self.assertEqual(response_curve_data.sizes[constants.SPEND_MULTIPLIER], 11)


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


def helper_sample_joint_dist_unpinned_as_posterior(self, n_draws, seed=None):
  """A helper function to sample joint distribution unpinned as posterior.

  Calling the `sample` method on the `tfp.JointDistributionCoroutineAutobatched`
  object is much faster than running MCMC posterior sampling. The `sample`
  method draws from the prior, but we use these draws to verify that the
  calculation of the coefficient means (`beta_m`, `beta_rf`, `beta_om`,
  `beta_orf`, `gamma_n`) is correct when non-coefficient prior types are used.
  `Analyzer.incremental_outcome` uses the coefficient values (`beta_gm`,
  `beta_grf`, `beta_gom`, `beta_gorf`, gamma_gn), so if the results derived from
  this method match the sampled parameters (`roi_m`, `mroi_m`, `contribution_m`,
  etc.) for every draw, then this confirms that the corresponding coefficient
  mean calculation is implemented correctly.

  Args:
    self: The Meridian object to sample from.
    n_draws: The number of draws to sample.
    seed: Optional seed for sampling.
  """
  posterior_sampler = self.posterior_sampler_callable
  prior_draws = (
      posterior_sampler._get_joint_dist_unpinned()
      .sample([1, n_draws], seed=seed)
      ._asdict()
  )
  prior_draws = {
      k: v
      for k, v in prior_draws.items()
      if k not in constants.UNSAVED_PARAMETERS
  }
  # Create Arviz InferenceData for posterior draws.
  posterior_coords = self.create_inference_data_coords(1, n_draws)
  posterior_dims = self.create_inference_data_dims()
  infdata_posterior = az.convert_to_inference_data(
      prior_draws, coords=posterior_coords, dims=posterior_dims
  )

  self.inference_data.extend(infdata_posterior, join="right")


def check_treatment_parameters(mmm, use_posterior, rtol=1e-3, atol=1e-3):
  infdata = (
      mmm.inference_data.posterior
      if use_posterior
      else mmm.inference_data.prior
  )
  # Calculate total_outcome from input_data instead of using mmm.total_outcome.
  assert mmm.input_data.revenue_per_kpi is not None
  total_outcome = np.sum(
      mmm.input_data.kpi.values * mmm.input_data.revenue_per_kpi.values
  )
  mmm_analyzer = analyzer.Analyzer(mmm)
  n_m, n_rf, n_om, n_orf = (
      mmm.n_media_channels,
      mmm.n_rf_channels,
      mmm.n_organic_media_channels,
      mmm.n_organic_rf_channels,
  )
  incremental_outcome = mmm_analyzer.incremental_outcome(
      include_non_paid_channels=True, use_posterior=use_posterior
  )
  ii_m = incremental_outcome[:, :, :n_m]
  ii_rf = incremental_outcome[:, :, n_m : (n_m + n_rf)]
  ii_om = incremental_outcome[:, :, (n_m + n_rf) : (n_m + n_rf + n_om)]
  ii_orf = incremental_outcome[
      :, :, (n_m + n_rf + n_om) : (n_m + n_rf + n_om + n_orf)
  ]
  ii_n = incremental_outcome[:, :, (n_m + n_rf + n_om + n_orf) :]
  calculated_om = ii_om / total_outcome
  calculated_orf = ii_orf / total_outcome
  calculated_n = ii_n / total_outcome
  param_om = infdata.contribution_om
  param_orf = infdata.contribution_orf
  param_n = infdata.contribution_n

  media_prior_type = mmm.model_spec.media_prior_type
  assert media_prior_type in [
      constants.TREATMENT_PRIOR_TYPE_ROI,
      constants.TREATMENT_PRIOR_TYPE_MROI,
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
  ]
  if media_prior_type == "roi":
    param_m = infdata.roi_m
    if mmm.model_spec.roi_calibration_period is None:
      roi = mmm_analyzer.roi(use_posterior=use_posterior)
      calculated_m = roi[:, :, :n_m]
    else:
      calculated_m = np.zeros_like(param_m)
      for i in range(n_m):
        times = mmm.model_spec.roi_calibration_period[:, i]
        ii = mmm_analyzer.incremental_outcome(
            media_selected_times=times.tolist(),
            use_posterior=use_posterior,
        )
        spend = np.einsum(
            "gtm,t->m",
            mmm.input_data.media_spend,
            times[-mmm.n_times :],
        )
        calculated_m[:, :, i] = ii[:, :, i] / spend[i]
      calculated_m = backend.to_tensor(calculated_m)
  elif media_prior_type == "mroi":
    mroi = mmm_analyzer.marginal_roi(use_posterior=use_posterior)
    calculated_m = mroi[:, :, :n_m]
    param_m = infdata.mroi_m
  else:  # media_prior_type == "contribution"
    calculated_m = ii_m / total_outcome
    param_m = infdata.contribution_m

  rf_prior_type = mmm.model_spec.rf_prior_type
  assert rf_prior_type in [
      constants.TREATMENT_PRIOR_TYPE_ROI,
      constants.TREATMENT_PRIOR_TYPE_MROI,
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
  ]
  if rf_prior_type == "roi":
    param_rf = infdata.roi_rf
    if mmm.model_spec.rf_roi_calibration_period is None:
      roi = mmm_analyzer.roi(use_posterior=use_posterior)
      calculated_rf = roi[:, :, n_m:]
    else:
      calculated_rf = np.zeros_like(param_rf)
      for i in range(n_rf):
        times = mmm.model_spec.rf_roi_calibration_period[:, i]
        ii = mmm_analyzer.incremental_outcome(
            media_selected_times=times.tolist(),
            use_posterior=use_posterior,
        )
        spend = np.einsum(
            "gtm,t->m",
            mmm.input_data.rf_spend,
            times[-mmm.n_times :],
        )
        calculated_rf[:, :, i] = ii[:, :, n_m + i] / spend[i]
      calculated_rf = backend.to_tensor(calculated_rf)
  elif rf_prior_type == "mroi":
    mroi = mmm_analyzer.marginal_roi(use_posterior=use_posterior)
    calculated_rf = mroi[:, :, n_m : (n_m + n_rf)]
    param_rf = infdata.mroi_rf
  else:  # rf_prior_type == "contribution"
    calculated_rf = ii_rf / total_outcome
    param_rf = infdata.contribution_rf
  backend_test_utils.assert_allclose(
      calculated_m, param_m, rtol=rtol, atol=atol
  )
  backend_test_utils.assert_allclose(
      calculated_rf, param_rf, rtol=rtol, atol=atol
  )
  backend_test_utils.assert_allclose(
      calculated_om, param_om, rtol=rtol, atol=atol
  )
  backend_test_utils.assert_allclose(
      calculated_orf, param_orf, rtol=rtol, atol=atol
  )
  backend_test_utils.assert_allclose(
      calculated_n, param_n, rtol=rtol, atol=atol
  )


class AnalyzerDataShapeTest(backend_test_utils.MeridianTestCase):
  """Tests for Analyzer behavior with specific data shapes, independent of global setups."""

  def test_hill_curves_scaling_and_alignment_with_disparate_magnitudes(
      self,
  ):
    n_geos = 3
    n_times = 10
    n_media_times = 12
    n_draws = 50

    channel_names = ["Z-Search", "A-Social"]

    base_dataset = data_test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_controls=1,
        n_media_channels=2,
        explicit_media_channel_names=channel_names,
        constant_population_value=1.0,  # Key condition: Population=1
        revenue_per_kpi_value=1.0,
        seed=123,
    )

    rng = np.random.default_rng(seed=0)

    # Channel 1 ("Z-Search"): low impression values (e.g., 80k - 120k)
    search_impressions = (
        80_000 + rng.random(size=(n_geos, n_media_times)) * 40_000
    )

    # Channel 2 ("A-Social"): high impression values (e.g., 7M - 8M)
    social_impressions = (
        7_000_000 + rng.random(size=(n_geos, n_media_times)) * 1_000_000
    )

    actual_max_low = search_impressions.max()
    actual_max_high = social_impressions.max()

    media_values = np.stack([search_impressions, social_impressions], axis=-1)

    media_da = xr.DataArray(
        media_values,
        dims=["geo", "media_time", "media_channel"],
        coords={
            "geo": base_dataset.geo.values,
            "media_time": base_dataset.media_time.values,
            "media_channel": channel_names,
        },
        name="media",
    )

    search_spend = 80 + rng.random(size=(n_geos, n_times)) * 40
    social_spend = 7000 + rng.random(size=(n_geos, n_times)) * 1000
    spend_values = np.stack([search_spend, social_spend], axis=-1)

    media_spend_da = xr.DataArray(
        spend_values,
        dims=["geo", "time", "media_channel"],
        coords={
            "geo": base_dataset.geo.values,
            "time": base_dataset.time.values,
            "media_channel": channel_names,
        },
        name="media_spend",
    )
    base_input_data = data_test_utils.sample_input_data_from_dataset(
        base_dataset, kpi_type=constants.NON_REVENUE
    )
    input_data_obj = dataclasses.replace(
        base_input_data,
        media=media_da,
        media_spend=media_spend_da,
    )

    model_spec = spec.ModelSpec(max_lag=4)
    mmm = model.Meridian(input_data=input_data_obj, model_spec=model_spec)

    model.Meridian.sample_joint_dist_unpinned_as_posterior = (
        helper_sample_joint_dist_unpinned_as_posterior
    )
    mmm.sample_prior(n_draws, seed=self.get_next_rng_seed_or_key())
    mmm.sample_joint_dist_unpinned_as_posterior(
        n_draws, seed=self.get_next_rng_seed_or_key()
    )

    mmm_analyzer = analyzer.Analyzer(mmm)
    hill_df = mmm_analyzer.hill_curves()

    max_units_low = hill_df[
        (hill_df[constants.CHANNEL] == "Z-Search")
        & (hill_df[constants.MEAN].notna())
    ][constants.MEDIA_UNITS].max()

    max_units_high = hill_df[
        (hill_df[constants.CHANNEL] == "A-Social")
        & (hill_df[constants.MEAN].notna())
    ][constants.MEDIA_UNITS].max()

    np.testing.assert_allclose(
        max_units_low,
        actual_max_low,
        rtol=1e-5,
        err_msg=(
            "X-axis range for low-scale channel does not match input data max."
        ),
    )
    np.testing.assert_allclose(
        max_units_high,
        actual_max_high,
        rtol=1e-5,
        err_msg=(
            "X-axis range for high-scale channel does not match input data max."
        ),
    )

  def test_hill_curves_channel_alignment_correct_with_non_alphabetical_order(
      self,
  ):
    n_geos = 2
    n_times = 10
    n_media_times = 12
    n_draws = 50

    # Input Order (Tensor Order): ['Channel_B', 'Channel_A']
    channel_names = ["Channel_B", "Channel_A"]

    # Scales corresponding to Input Order:
    # Channel_B (Index 0): Mean 100 (Low)
    # Channel_A (Index 1): Mean 10000 (High)
    scale_b = (100, 10)
    scale_a = (10000, 1000)
    media_scales = [scale_b, scale_a]

    dataset = data_test_utils.random_dataset(
        n_geos=n_geos,
        n_times=n_times,
        n_media_times=n_media_times,
        n_controls=1,
        n_media_channels=2,
        explicit_media_channel_names=channel_names,
        constant_population_value=1.0,  # Key condition: Population=1
        media_value_scales=media_scales,
        revenue_per_kpi_value=1.0,
        seed=1,
    )
    input_data_obj = data_test_utils.sample_input_data_from_dataset(
        dataset, kpi_type=constants.NON_REVENUE
    )

    assert input_data_obj.media is not None, "media should not be None"
    actual_max_a = (
        input_data_obj.media.sel(media_channel="Channel_A").max().item()
    )
    actual_max_b = (
        input_data_obj.media.sel(media_channel="Channel_B").max().item()
    )

    model_spec = spec.ModelSpec(max_lag=2)
    mmm = model.Meridian(input_data=input_data_obj, model_spec=model_spec)

    model.Meridian.sample_joint_dist_unpinned_as_posterior = (
        helper_sample_joint_dist_unpinned_as_posterior
    )
    mmm.sample_prior(n_draws, seed=self.get_next_rng_seed_or_key())
    mmm.sample_joint_dist_unpinned_as_posterior(
        n_draws, seed=self.get_next_rng_seed_or_key()
    )

    mmm_analyzer = analyzer.Analyzer(mmm)
    hill_df = mmm_analyzer.hill_curves()

    output_channels = (
        hill_df[hill_df[constants.MEAN].notna()][constants.CHANNEL]
        .unique()
        .tolist()
    )

    max_units_a = hill_df[
        (hill_df[constants.CHANNEL] == "Channel_A")
        & (hill_df[constants.MEAN].notna())
    ][constants.MEDIA_UNITS].max()

    max_units_b = hill_df[
        (hill_df[constants.CHANNEL] == "Channel_B")
        & (hill_df[constants.MEAN].notna())
    ][constants.MEDIA_UNITS].max()

    self.assertListEqual(
        output_channels,
        channel_names,
        msg=(
            "Channel order in the output DataFrame does not match the input"
            " tensor order."
        ),
    )

    np.testing.assert_allclose(max_units_a, actual_max_a, rtol=1e-5)
    np.testing.assert_allclose(max_units_b, actual_max_b, rtol=1e-5)


class AnalyzerCustomPriorTest(backend_test_utils.MeridianTestCase):

  @parameterized.product(
      n_channels_per_treatment=[1, 2],
      media_prior_type=["roi", "mroi", "contribution"],
      rf_prior_type=["roi", "mroi", "contribution"],
      roi_calibration_times=[None, [5, 6, 7]],
      rf_roi_calibration_times=[None, [5, 6, 7]],
  )
  def test_treatment_parameter_accuracy(
      self,
      n_channels_per_treatment,
      media_prior_type,
      rf_prior_type,
      roi_calibration_times,
      rf_roi_calibration_times,
  ):
    input_data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=3,
        n_times=10,
        n_media_times=15,
        n_controls=1,
        n_media_channels=n_channels_per_treatment,
        n_rf_channels=n_channels_per_treatment,
        n_organic_media_channels=n_channels_per_treatment,
        n_organic_rf_channels=n_channels_per_treatment,
        n_non_media_channels=n_channels_per_treatment,
        seed=1,
        nonzero_shift=1.0,
    )

    # Scale each channel's spend to be between 4-6% of total revenue. (Otherwise
    # spend values can be so small that they cause numerical inaccuracies with
    # ROI priors.)
    # pytype: disable=attribute-error
    total_outcome = np.sum(
        input_data.kpi.values * input_data.revenue_per_kpi.values
    )
    # pytype: enable=attribute-error

    total_spend_m = np.sum(input_data.media_spend.values, (0, 1))
    total_spend_rf = np.sum(input_data.rf_spend.values, (0, 1))
    n_m = len(input_data.media_channel)
    n_rf = len(input_data.rf_channel)
    media_pcts = np.linspace(0.04, 0.06, n_m)
    input_data.media_spend *= media_pcts * total_outcome / total_spend_m
    rf_pcts = np.linspace(0.04, 0.06, n_rf)
    input_data.rf_spend *= rf_pcts * total_outcome / total_spend_rf

    # Set `roi_calibration_period` and assert error if media_prior_type !=
    # "roi".
    if roi_calibration_times is None:
      roi_calibration_period = None
    else:
      n_media_times = len(input_data.media_time)
      n_media_channels = len(input_data.media_channel)
      roi_calibration_period = np.full([n_media_times, n_media_channels], False)
      for time in roi_calibration_times:
        roi_calibration_period[time, :] = True
      if media_prior_type != "roi":
        with self.assertRaisesRegex(ValueError, "The `roi_calibration_period`"):
          spec.ModelSpec(
              media_prior_type=media_prior_type,
              roi_calibration_period=roi_calibration_period,
          )
        return
    # Set `rf_roi_calibration_period` and assert error if rf_prior_type !=
    # "roi".
    if rf_roi_calibration_times is None:
      rf_roi_calibration_period = None
    else:
      n_media_times = len(input_data.media_time)
      n_rf_channels = len(input_data.rf_channel)
      rf_roi_calibration_period = np.full([n_media_times, n_rf_channels], False)
      for time in rf_roi_calibration_times:
        rf_roi_calibration_period[time, :] = True
      if rf_prior_type != "roi":
        with self.assertRaisesRegex(
            ValueError, "The `rf_roi_calibration_period`"
        ):
          spec.ModelSpec(
              rf_prior_type=rf_prior_type,
              rf_roi_calibration_period=rf_roi_calibration_period,
          )
        return

    model_spec = spec.ModelSpec(
        media_prior_type=media_prior_type,
        rf_prior_type=rf_prior_type,
        roi_calibration_period=roi_calibration_period,
        rf_roi_calibration_period=rf_roi_calibration_period,
    )

    model.Meridian.sample_joint_dist_unpinned_as_posterior = (
        helper_sample_joint_dist_unpinned_as_posterior
    )

    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    mmm.sample_prior(100, seed=self.get_next_rng_seed_or_key())
    mmm.sample_joint_dist_unpinned_as_posterior(
        100, seed=self.get_next_rng_seed_or_key()
    )
    check_treatment_parameters(mmm, use_posterior=False)
    check_treatment_parameters(mmm, use_posterior=True)


if __name__ == "__main__":
  absltest.main()
