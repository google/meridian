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

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian import constants
from meridian.analysis import tensors
from meridian.backend import test_utils as backend_test_utils
from meridian.data import test_utils as data_test_utils
from meridian.model import equations
from meridian.model import model
from meridian.model import spec
import numpy as np

_N_GEOS = 5
_N_TIMES = 49
_N_MEDIA_TIMES = 52
_N_CONTROLS = 2
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2
_N_NON_MEDIA_CHANNELS = 4
_N_ORGANIC_MEDIA_CHANNELS = 4
_N_ORGANIC_RF_CHANNELS = 1


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
      tensors.DataTensors(controls=backend.ones((_N_GEOS, _N_TIMES)))

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
      tensors.DataTensors(**new_param)

  def test_validate_wrong_geos_media(self):
    new_data = tensors.DataTensors(
        media=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError, r"New `media` is expected to have 5 geos\. Found 6 geos\."
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_wrong_geos_media_spend(self):
    new_data = tensors.DataTensors(
        media_spend=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `media_spend` is expected to have 5 geos\. Found 6 geos\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA_SPEND],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_wrong_times_media(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `media` is expected to have 52 time periods\. Found 10 time"
        r" periods\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          model_context=self.meridian_media_and_rf.model_context,
          allow_modified_times=False,
      )

  def test_validate_wrong_channels_frequency(self):
    new_data = tensors.DataTensors(
        frequency=backend.ones((_N_GEOS, _N_TIMES, 3))
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `frequency` is expected to have 2 channels\. Found 3 channels\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.FREQUENCY],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_wrong_channels_reach(self):
    new_data = tensors.DataTensors(
        reach=backend.ones((_N_GEOS, _N_MEDIA_TIMES, _N_RF_CHANNELS - 1))
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `reach` is expected to have 2 channels\. Found 1 channels\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.REACH],
          model_context=self.meridian_media_and_rf.model_context,
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
    new_data = tensors.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified, then"
        " all variables must be provided in `new_data`. The following variables"
        f" are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=list(new_data_dict.keys()) + [missing_param],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_new_params_diff_time_dims(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        frequency=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        revenue_per_kpi=backend.ones((_N_GEOS, 8)),
    )
    with self.assertRaisesRegex(
        ValueError,
        "If the time dimension of any variable in `new_data` is modified, then"
        " all variables must be provided with the same number of time periods."
        r" `revenue_per_kpi` has 8 time periods, which does not match the"
        r" modified number of time periods, 10\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[
              constants.MEDIA,
              constants.REACH,
              constants.FREQUENCY,
              constants.REVENUE_PER_KPI,
          ],
          model_context=self.meridian_media_and_rf.model_context,
      )

  @parameterized.parameters([constants.MEDIA, constants.REVENUE_PER_KPI])
  def test_validate_media_only_missing_new_param(self, missing_param: str):
    new_data_dict = {
        constants.MEDIA: backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        constants.REVENUE_PER_KPI: backend.ones((_N_GEOS, 10)),
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = tensors.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified,"
        " then all variables must be provided in `new_data`. The"
        f" following variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names,
          model_context=self.meridian_media_only.model_context,
      )

  def test_validate_media_only_invalid_new_data(self):
    new_data = tensors.DataTensors(
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS))
    )
    with self.assertRaisesRegex(
        ValueError,
        "New `reach` is not allowed because the input data to the Meridian"
        " model does not contain `reach`",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.REACH],
          model_context=self.meridian_media_only.model_context,
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
    new_data = tensors.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified, then"
        " all variables must be provided in `new_data`. The following"
        f" variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names,
          model_context=self.meridian_rf_only.model_context,
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

    tensors_dict = {}
    for tensor_name in new_tensors_names:
      tensors_dict[tensor_name] = getattr(data, tensor_name)
    new_data = tensors.DataTensors(**tensors_dict)

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
        required_tensors_names=required_tensors_names,
        model_context=self.meridian_organic_media.model_context,
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
    new_data = tensors.DataTensors(**new_data_dict)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If the time dimension of a variable in `new_data` is modified,"
        " then all variables must be provided in `new_data`. The"
        f" following variables are missing: `['{missing_param}']`.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required_names,
          model_context=self.meridian_organic_media.model_context,
      )

  def test_validate_organic_media_new_param_not_matching_times(self):
    new_data = tensors.DataTensors(
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
          model_context=self.meridian_organic_media.model_context,
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

    new_data = tensors.DataTensors(**{param_name: tensor})
    required = [constants.MEDIA]

    with self.assertWarnsRegex(UserWarning, warning_msg):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required,
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_non_media_missing_new_param_flexible_times(self) -> None:
    new_data = tensors.DataTensors(
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
          required_tensors_names=required,
          model_context=self.meridian_non_media.model_context,
      )

  def test_get_model_context_with_model_context(self):
    dummy_context = self.meridian_media_only.model_context
    # pylint: disable=protected-access
    result = tensors._get_model_context(
        meridian=None, model_context=dummy_context
    )
    # pylint: enable=protected-access
    self.assertEqual(result, dummy_context)

  def test_get_model_context_with_meridian(self):
    dummy_meridian = self.meridian_media_only
    # pylint: disable=protected-access
    with self.assertWarnsRegex(DeprecationWarning, "meridian.*deprecated"):
      result = tensors._get_model_context(
          meridian=dummy_meridian, model_context=None
      )
    # pylint: enable=protected-access
    self.assertEqual(result, dummy_meridian.model_context)

  def test_get_model_context_both_none(self):
    # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "must be provided"):
      tensors._get_model_context(meridian=None, model_context=None)
    # pylint: enable=protected-access


class DataTensorsBuilderTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.input_data = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            n_organic_media_channels=_N_ORGANIC_MEDIA_CHANNELS,
            n_organic_rf_channels=_N_ORGANIC_RF_CHANNELS,
            n_non_media_channels=_N_NON_MEDIA_CHANNELS,
            seed=0,
        )
    )
    cls.meridian = model.Meridian(
        input_data=cls.input_data,
        model_spec=spec.ModelSpec(max_lag=15),
    )

  def test_build_scaled_inputs_resolves_indices(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Select a subset of geos and times
    selected_geos = [
        self.input_data.geo.values[0],
        self.input_data.geo.values[2],
    ]
    selected_times = [
        self.input_data.time.values[1],
        self.input_data.time.values[3],
    ]

    inputs = builder.build_scaled_inputs(
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

    self.assertIsInstance(inputs, tensors.AnalyzerInputs)
    self.assertIsInstance(inputs.tensors, tensors.DataTensors)

    # Check geo indices
    expected_geo_indices = [0, 2]
    backend_test_utils.assert_allclose(
        inputs.geo_indices,
        backend.to_tensor(expected_geo_indices, dtype=backend.int32),
    )

    # Check time indices
    expected_time_indices = [1, 3]
    backend_test_utils.assert_allclose(
        inputs.time_indices,
        backend.to_tensor(expected_time_indices, dtype=backend.int32),
    )

  def test_build_scaled_inputs_resolves_boolean_times(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Create a boolean mask for times
    selected_times = [False] * _N_TIMES
    selected_times[1] = True
    selected_times[3] = True

    inputs = builder.build_scaled_inputs(
        selected_times=selected_times,
    )

    expected_time_indices = [1, 3]
    backend_test_utils.assert_allclose(
        inputs.time_indices,
        backend.to_tensor(expected_time_indices, dtype=backend.int32),
    )

  def test_build_unscaled_inputs_defaults(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_unscaled_inputs()
    self.assertIsInstance(inputs, tensors.AnalyzerInputs)
    self.assertIsInstance(inputs.tensors, tensors.DataTensors)
    self.assertIsNone(inputs.tensors.reach)
    self.assertIsNone(inputs.tensors.frequency)

  def test_build_unscaled_inputs_fill_missing(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_unscaled_inputs(
        required_tensors_names=[
            constants.REACH,
            constants.FREQUENCY,
        ]
    )
    self.assertIsNotNone(inputs.tensors.reach)
    self.assertIsNotNone(inputs.tensors.frequency)

  def test_build_unscaled_inputs_optimal_frequency_scaling_list(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Only require paid RF, so organic is not filled and not scaled.
    required_names = [
        constants.RF_IMPRESSIONS,
        constants.REACH,
        constants.FREQUENCY,
    ]

    optimal_frequency = [2.0, 3.0]

    inputs = builder.build_unscaled_inputs(
        required_tensors_names=required_names,
        optimal_frequency=optimal_frequency,
    )

    # Verify paid RF is scaled
    historical_reach = self.meridian.model_context.rf_tensors.reach
    historical_frequency = self.meridian.model_context.rf_tensors.frequency
    historical_impressions = historical_reach * historical_frequency

    expected_frequency = backend.ones_like(
        historical_impressions
    ) * backend.to_tensor(optimal_frequency, dtype=backend.float_dtype)
    expected_reach = historical_impressions / expected_frequency

    backend_test_utils.assert_allclose(
        inputs.tensors.frequency,
        expected_frequency,
    )
    backend_test_utils.assert_allclose(
        inputs.tensors.reach,
        expected_reach,
    )

    # Organic should be None because it was not in required_names
    self.assertIsNone(inputs.tensors.organic_reach)
    self.assertIsNone(inputs.tensors.organic_frequency)

  def test_build_unscaled_inputs_optimal_frequency_scaling_scalar_with_organic(
      self,
  ):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Require both paid and organic RF
    required_names = [
        constants.RF_IMPRESSIONS,
        constants.REACH,
        constants.FREQUENCY,
        constants.ORGANIC_REACH,
        constants.ORGANIC_FREQUENCY,
    ]

    optimal_frequency = 2.0  # Scalar

    inputs = builder.build_unscaled_inputs(
        required_tensors_names=required_names,
        optimal_frequency=optimal_frequency,
    )

    # Verify paid RF is scaled
    historical_reach = self.meridian.model_context.rf_tensors.reach
    historical_frequency = self.meridian.model_context.rf_tensors.frequency
    historical_impressions = historical_reach * historical_frequency
    expected_frequency = (
        backend.ones_like(historical_impressions) * optimal_frequency
    )
    expected_reach = historical_impressions / expected_frequency

    backend_test_utils.assert_allclose(
        inputs.tensors.frequency, expected_frequency
    )
    backend_test_utils.assert_allclose(inputs.tensors.reach, expected_reach)

    # Verify organic RF is scaled
    historical_organic_reach = (
        self.meridian.model_context.organic_rf_tensors.organic_reach
    )
    historical_organic_frequency = (
        self.meridian.model_context.organic_rf_tensors.organic_frequency
    )

    expected_organic_frequency = (
        backend.ones_like(historical_organic_frequency) * optimal_frequency
    )
    expected_organic_reach = (
        historical_organic_reach * historical_organic_frequency
    ) / expected_organic_frequency

    backend_test_utils.assert_allclose(
        inputs.tensors.organic_frequency, expected_organic_frequency
    )
    backend_test_utils.assert_allclose(
        inputs.tensors.organic_reach, expected_organic_reach
    )

  def test_build_unscaled_inputs_insert_dummy_media(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    inputs = builder.build_unscaled_inputs(
        insert_dummy_media=True,
    )

    self.assertIsInstance(inputs, tensors.AnalyzerInputs)
    self.assertIsInstance(inputs.tensors, tensors.DataTensors)

    # Verify dummy media and media spend are inserted and have correct shapes
    expected_media_shape = (
        _N_GEOS,
        self.meridian.model_context.n_media_times,
        _N_MEDIA_CHANNELS,
    )
    expected_spend_shape = (
        _N_GEOS,
        self.meridian.model_context.n_times,
        _N_MEDIA_CHANNELS,
    )

    assert inputs.tensors.media is not None
    assert inputs.tensors.media_spend is not None
    self.assertEqual(inputs.tensors.media.shape, expected_media_shape)
    self.assertEqual(inputs.tensors.media_spend.shape, expected_spend_shape)

    # Verify they are all ones
    backend_test_utils.assert_allclose(
        inputs.tensors.media,
        backend.ones(expected_media_shape, dtype=backend.float_dtype),
    )
    backend_test_utils.assert_allclose(
        inputs.tensors.media_spend,
        backend.ones(expected_spend_shape, dtype=backend.float_dtype),
    )

  def test_build_unscaled_inputs_resolves_indices(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    selected_geos = [
        self.input_data.geo.values[0],
        self.input_data.geo.values[2],
    ]
    selected_times = [
        self.input_data.time.values[1],
        self.input_data.time.values[3],
    ]

    inputs = builder.build_unscaled_inputs(
        selected_geos=selected_geos,
        selected_times=selected_times,
    )

    # Check geo indices
    expected_geo_indices = [0, 2]
    backend_test_utils.assert_allclose(
        inputs.geo_indices,
        backend.to_tensor(expected_geo_indices, dtype=backend.int32),
    )

    # Check time indices
    expected_time_indices = [1, 3]
    backend_test_utils.assert_allclose(
        inputs.time_indices,
        backend.to_tensor(expected_time_indices, dtype=backend.int32),
    )


class DataTensorsBuilderCounterfactualTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.input_data = (
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
    cls.meridian = model.Meridian(
        input_data=cls.input_data,
        model_spec=spec.ModelSpec(max_lag=15),
    )

  def test_build_counterfactual_inputs_returns_correct_type(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_counterfactual_inputs()
    self.assertIsInstance(inputs, tensors.CounterfactualInputs)
    self.assertIsInstance(inputs.tensors, tensors.DataTensors)

  def test_build_counterfactual_inputs_resolves_media_selected_times_mask(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Test default (None) -> all True
    inputs_default = builder.build_counterfactual_inputs()
    self.assertEqual(
        inputs_default.media_selected_times_mask,
        tuple([True] * _N_MEDIA_TIMES),
    )

    # Test with string list
    selected_times = [
        self.input_data.media_time.values[1],
        self.input_data.media_time.values[3],
    ]
    inputs_str = builder.build_counterfactual_inputs(
        media_selected_times=selected_times
    )
    expected_mask = [False] * _N_MEDIA_TIMES
    expected_mask[1] = True
    expected_mask[3] = True
    self.assertEqual(
        inputs_str.media_selected_times_mask,
        tuple(expected_mask),
    )

  def test_build_counterfactual_inputs_scales_media_tensors(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    scaling_factor = 0.5

    # Select some times to scale
    selected_times = [
        self.input_data.media_time.values[1],
        self.input_data.media_time.values[3],
    ]

    inputs = builder.build_counterfactual_inputs(
        scaling_factor=scaling_factor,
        media_selected_times=selected_times,
    )

    # Verify scaled media
    original_media = self.meridian.model_context.media_tensors.media_scaled
    expected_media = np.array(original_media)
    # Indices 1 and 3 should be scaled by 0.5
    expected_media[:, 1, :] *= scaling_factor
    expected_media[:, 3, :] *= scaling_factor

    backend_test_utils.assert_allclose(
        inputs.tensors.media,
        backend.to_tensor(expected_media, dtype=backend.float_dtype),
    )

  def test_build_counterfactual_inputs_sets_non_media_baseline(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    non_media_baseline_values = [1.0] * _N_NON_MEDIA_CHANNELS

    # When is_baseline=False, non_media_treatments should be scaled
    # historical values.
    inputs_historical = builder.build_counterfactual_inputs(is_baseline=False)
    backend_test_utils.assert_allclose(
        inputs_historical.tensors.non_media_treatments,
        self.meridian.model_context.non_media_treatments_normalized,
    )

    # When is_baseline=True, non_media_treatments should be baseline
    inputs_baseline = builder.build_counterfactual_inputs(
        non_media_baseline_values=non_media_baseline_values,
        is_baseline=True,
    )

    # Assert to satisfy pytype
    assert self.meridian.model_context.non_media_transformer is not None
    assert inputs_baseline.tensors.non_media_treatments is not None

    # Compute expected baseline
    expected_baseline_scaled = equations.ModelEquations(
        self.meridian.model_context
    ).compute_non_media_treatments_baseline(
        non_media_baseline_values=non_media_baseline_values
    )
    expected_baseline_normalized = (
        self.meridian.model_context.non_media_transformer.forward(
            expected_baseline_scaled,
            apply_population_scaling=False,
        )
    )
    expected_baseline_tensor = backend.broadcast_to(
        backend.to_tensor(
            expected_baseline_normalized,
            dtype=backend.float_dtype,
        )[backend.newaxis, backend.newaxis, :],
        inputs_baseline.tensors.non_media_treatments.shape,
    )

    backend_test_utils.assert_allclose(
        inputs_baseline.tensors.non_media_treatments,
        expected_baseline_tensor,
    )

    # Also verify non_media_baseline_normalized is set correctly
    backend_test_utils.assert_allclose(
        inputs_baseline.non_media_baseline_normalized,
        backend.to_tensor(
            expected_baseline_normalized, dtype=backend.float_dtype
        ),
    )


class DataTensorsBuilderBaselineTest(backend_test_utils.MeridianTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.input_data = (
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
    cls.meridian = model.Meridian(
        input_data=cls.input_data,
        model_spec=spec.ModelSpec(max_lag=15),
    )

  def test_build_baseline_inputs_returns_correct_type(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_baseline_inputs()
    self.assertIsInstance(inputs, tensors.AnalyzerInputs)

  def test_build_baseline_inputs_zeros_out_media_and_rf(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_baseline_inputs()

    # Media should be zeroed if it exists
    if self.meridian.model_context.media_tensors.media is not None:
      self.assertIsNotNone(inputs.tensors.media)
      backend_test_utils.assert_allclose(
          inputs.tensors.media,
          backend.zeros_like(self.meridian.model_context.media_tensors.media),
      )
    else:
      self.assertIsNone(inputs.tensors.media)

    # Reach should be zeroed if it exists
    if self.meridian.model_context.rf_tensors.reach is not None:
      self.assertIsNotNone(inputs.tensors.reach)
      backend_test_utils.assert_allclose(
          inputs.tensors.reach,
          backend.zeros_like(self.meridian.model_context.rf_tensors.reach),
      )
    else:
      self.assertIsNone(inputs.tensors.reach)

    # Organic media should be zeroed if it exists
    if (
        self.meridian.model_context.organic_media_tensors.organic_media
        is not None
    ):
      self.assertIsNotNone(inputs.tensors.organic_media)
      backend_test_utils.assert_allclose(
          inputs.tensors.organic_media,
          backend.zeros_like(
              self.meridian.model_context.organic_media_tensors.organic_media
          ),
      )
    else:
      self.assertIsNone(inputs.tensors.organic_media)

    # Organic reach should be zeroed if it exists
    if self.meridian.model_context.organic_rf_tensors.organic_reach is not None:
      self.assertIsNotNone(inputs.tensors.organic_reach)
      backend_test_utils.assert_allclose(
          inputs.tensors.organic_reach,
          backend.zeros_like(
              self.meridian.model_context.organic_rf_tensors.organic_reach
          ),
      )
    else:
      self.assertIsNone(inputs.tensors.organic_reach)

  def test_build_baseline_inputs_omits_frequency(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    inputs = builder.build_baseline_inputs()
    self.assertIsNone(inputs.tensors.frequency)
    self.assertIsNone(inputs.tensors.organic_frequency)

  def test_build_baseline_inputs_computes_non_media_baseline(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    non_media_baseline_values = [1.0] * _N_NON_MEDIA_CHANNELS
    inputs = builder.build_baseline_inputs(
        non_media_baseline_values=non_media_baseline_values
    )

    # Assert to satisfy pytype
    assert inputs.tensors.non_media_treatments is not None

    # Compute expected baseline
    expected_baseline_scaled = equations.ModelEquations(
        self.meridian.model_context
    ).compute_non_media_treatments_baseline(
        non_media_baseline_values=non_media_baseline_values
    )
    expected_baseline_tensor = backend.broadcast_to(
        backend.to_tensor(
            expected_baseline_scaled,
            dtype=backend.float_dtype,
        )[backend.newaxis, backend.newaxis, :],
        inputs.tensors.non_media_treatments.shape,
    )

    # Apply population scaling if needed (matching the implementation)
    ctx = self.meridian.model_context
    if ctx.model_spec.non_media_population_scaling_id is not None:
      scaling_factors = backend.where(
          ctx.model_spec.non_media_population_scaling_id,
          ctx.population[:, backend.newaxis, backend.newaxis],
          backend.ones_like(ctx.population)[
              :, backend.newaxis, backend.newaxis
          ],
      )
    else:
      scaling_factors = backend.ones_like(ctx.population)[
          :, backend.newaxis, backend.newaxis
      ]
    expected_baseline_tensor = expected_baseline_tensor * scaling_factors

    backend_test_utils.assert_allclose(
        inputs.tensors.non_media_treatments,
        expected_baseline_tensor,
    )

  def test_build_baseline_inputs_raises_value_error_for_invalid_baseline_values(
      self,
  ):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Test invalid types
    with self.assertRaises(ValueError):
      builder.build_baseline_inputs(
          non_media_baseline_values=["invalid"]  # pytype: disable=wrong-arg-types
      )

    # Test invalid length
    invalid_length_values = [1.0] * (_N_NON_MEDIA_CHANNELS + 1)
    with self.assertRaises(ValueError):
      builder.build_baseline_inputs(
          non_media_baseline_values=invalid_length_values
      )


if __name__ == "__main__":
  absltest.main()
