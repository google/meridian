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
      tensors.DataTensors(**new_param)  # pyrefly: ignore[bad-argument-type]

  def test_validate_wrong_geos_media(self):
    new_data = tensors.DataTensors(
        media=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS)),
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
        media_spend=backend.ones((6, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS)),
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `media_spend` is expected to have 5 geos\. Found 6 geos\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA_SPEND],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_no_time_matches_historic_times(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS)),
    )
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names=[constants.MEDIA],
        model_context=self.meridian_media_and_rf.model_context,
    )
    self.assertIsNotNone(filled_data.time)
    self.assertLen(filled_data.time, _N_TIMES)

  def test_validate_no_time_wrong_historic_times(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`time` must be provided in `new_data` if any time dimension in"
        " `new_data` is modified.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_wrong_times_controls(self):
    new_data = tensors.DataTensors(
        controls=backend.ones((_N_GEOS, 10, _N_CONTROLS)),
    )
    with self.assertRaisesRegex(
        ValueError,
        r"New `controls` is expected to have 49 time periods\. Found 10 time"
        r" periods\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.CONTROLS],
          model_context=self.meridian_media_and_rf.model_context,
          allow_modified_times=False,
      )

  def test_validate_wrong_times_media(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 10, _N_MEDIA_CHANNELS)),
        time=list(self.meridian_media_and_rf.input_data.time.values[:10]),
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

  def test_validate_invalid_time_format(self):
    with self.assertRaisesRegex(
        ValueError,
        r"time data '2021/01/01' does not match format '%Y-%m-%d'",
    ):
      tensors.DataTensors(
          media=backend.ones((_N_GEOS, 2, _N_MEDIA_CHANNELS)),
          time=["2021/01/01", "2021/01/08"],
      )

  def test_validate_non_monotonic_time(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 2, _N_MEDIA_CHANNELS)),
        time=["2021-01-08", "2021-01-01"],
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Time coordinates must be strictly monotonically increasing\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_irregular_time_spacing(self):
    new_data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 3, _N_MEDIA_CHANNELS)),
        time=["2021-01-01", "2021-01-03", "2021-01-20"],
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Time coordinates must be regularly spaced\.",
    ):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=[constants.MEDIA],
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_wrong_channels_frequency(self):
    new_data = tensors.DataTensors(
        frequency=backend.ones((_N_GEOS, _N_MEDIA_TIMES, 3)),
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
        reach=backend.ones((_N_GEOS, _N_MEDIA_TIMES, _N_RF_CHANNELS - 1)),
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
        constants.TIME: self.meridian_media_and_rf.input_data.time.values[:10],
    }
    new_data_dict.pop(missing_param)
    new_data = tensors.DataTensors(**new_data_dict)  # pyrefly: ignore[bad-argument-type]
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
        time=list(self.meridian_media_and_rf.input_data.time.values[:10]),
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
        constants.TIME: list(
            self.meridian_media_only.input_data.time.values[:10]
        ),
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = tensors.DataTensors(**new_data_dict)  # pyrefly: ignore[bad-argument-type]
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
        reach=backend.ones((_N_GEOS, 10, _N_RF_CHANNELS)),
        time=list(self.meridian_media_only.input_data.time.values[:10]),
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
        constants.TIME: self.meridian_rf_only.input_data.time.values[:10],
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = tensors.DataTensors(**new_data_dict)  # pyrefly: ignore[bad-argument-type]
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
    if new_tensors_names:
      tensors_dict[constants.TIME] = data.time
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
        constants.TIME: self.meridian_organic_media.input_data.time.values[:10],
    }
    required_names = list(new_data_dict.keys())
    new_data_dict.pop(missing_param)
    new_data = tensors.DataTensors(**new_data_dict)  # pyrefly: ignore[bad-argument-type]
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
        time=list(self.meridian_organic_media.input_data.time.values[:10]),
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

    new_data = tensors.DataTensors(**{  # pyrefly: ignore[bad-argument-type]
        param_name: tensor,
    })
    required = [constants.MEDIA]

    with self.assertWarnsRegex(UserWarning, warning_msg):
      new_data.validate_and_fill_missing_data(
          required_tensors_names=required,
          model_context=self.meridian_media_and_rf.model_context,
      )

  def test_validate_non_media_missing_new_param_flexible_times(self) -> None:
    new_data = tensors.DataTensors(
        non_media_treatments=self.meridian_non_media.non_media_treatments[  # pyrefly: ignore[unsupported-operation]
            :, :2, :
        ],
        time=list(self.meridian_non_media.input_data.time.values[:2]),
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
    result = tensors.get_model_context(
        meridian=None, model_context=dummy_context
    )
    self.assertEqual(result, dummy_context)

  def test_get_model_context_with_meridian(self):
    dummy_meridian = self.meridian_media_only
    with self.assertWarnsRegex(DeprecationWarning, "meridian.*deprecated"):
      result = tensors.get_model_context(
          meridian=dummy_meridian, model_context=None
      )
    self.assertEqual(result, dummy_meridian.model_context)

  def test_get_model_context_both_none(self):
    with self.assertRaisesRegex(ValueError, "must be provided"):
      tensors.get_model_context(meridian=None, model_context=None)


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

  def test_build_scaled_inputs_rejects_boolean_times(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Create a boolean mask for times
    selected_times = [False] * _N_TIMES
    selected_times[1] = True
    selected_times[3] = True

    with self.assertRaisesRegex(
        ValueError, r"`selected_times` must be a list of strings\."
    ):
      builder.build_scaled_inputs(
          selected_times=selected_times,  # pyrefly: ignore[bad-argument-type]
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
    historical_impressions = historical_reach * historical_frequency  # pyrefly: ignore[unsupported-operation]

    expected_frequency = backend.ones_like(  # pyrefly: ignore[unsupported-operation]
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
    historical_impressions = historical_reach * historical_frequency  # pyrefly: ignore[unsupported-operation]
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
        backend.ones_like(historical_organic_frequency) * optimal_frequency  # pyrefly: ignore[bad-argument-type]
    )
    expected_organic_reach = (
        historical_organic_reach * historical_organic_frequency  # pyrefly: ignore[unsupported-operation]
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


class DataTensorsBuilderSpendAllocationTest(
    backend_test_utils.MeridianTestCase
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    input_data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    # Spend aggregated over the geo and time dimensions.
    input_data.media_spend = data_test_utils.random_media_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=0,
    )
    input_data.rf_spend = data_test_utils.random_rf_spend_nd_da(
        n_geos=None,
        n_times=None,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    cls.meridian_1d_spend = model.Meridian(
        input_data=input_data,
        model_spec=spec.ModelSpec(max_lag=15),
    )

    # Avoid the pytype check complaint.
    assert input_data.media is not None
    assert input_data.reach is not None
    assert input_data.frequency is not None
    assert input_data.media_spend is not None
    assert input_data.rf_spend is not None
    assert input_data.allocated_media_spend is not None
    assert input_data.allocated_rf_spend is not None

    cls.media = input_data.media.values
    cls.reach = input_data.reach.values
    cls.frequency = input_data.frequency.values
    cls.aggregated_media_spend = input_data.media_spend.values
    cls.aggregated_rf_spend = input_data.rf_spend.values
    cls.allocated_media_spend = input_data.allocated_media_spend.values
    cls.allocated_rf_spend = input_data.allocated_rf_spend.values

  def _build_spend(
      self,
      new_data: tensors.DataTensors | None = None,
      required_tensors_names: Sequence[str] = (
          constants.PAID_CHANNELS + constants.SPEND_DATA
      ),
  ):
    """Returns the filled `(media_spend, rf_spend)` as numpy arrays."""
    builder = tensors.DataTensorsBuilder(self.meridian_1d_spend.model_context)
    filled = builder.build_unscaled_inputs(
        new_data=new_data,
        required_tensors_names=required_tensors_names,
    ).tensors
    return np.asarray(filled.media_spend), np.asarray(filled.rf_spend)

  def test_allocates_aggregated_spend_to_geo_and_time(self):
    media_spend, rf_spend = self._build_spend()

    self.assertEqual(media_spend.shape, (_N_GEOS, _N_TIMES, _N_MEDIA_CHANNELS))
    self.assertEqual(rf_spend.shape, (_N_GEOS, _N_TIMES, _N_RF_CHANNELS))
    # Allocation redistributes spend without changing the channel totals.
    backend_test_utils.assert_allclose(
        np.sum(media_spend, axis=(0, 1)),
        self.aggregated_media_spend,
        rtol=1e-5,
    )
    backend_test_utils.assert_allclose(
        np.sum(rf_spend, axis=(0, 1)), self.aggregated_rf_spend, rtol=1e-5
    )

  def test_allocation_is_proportional_to_media_units(self):
    media_spend, _ = self._build_spend()

    media = self.media[:, -_N_TIMES:, :]
    expected = media / np.sum(media, axis=(0, 1)) * self.aggregated_media_spend
    backend_test_utils.assert_allclose(media_spend, expected, rtol=1e-5)

  def test_allocation_matches_input_data_allocated_spend(self):
    media_spend, rf_spend = self._build_spend()

    backend_test_utils.assert_allclose(
        media_spend, self.allocated_media_spend, rtol=1e-5
    )
    backend_test_utils.assert_allclose(
        rf_spend, self.allocated_rf_spend, rtol=1e-5
    )

  def test_allocation_uses_new_media(self):
    # Reweight media non-uniformly across geos and times so that the allocation
    # weights differ from the ones implied by the historical media.
    geo_weights = np.linspace(1.0, 5.0, _N_GEOS)[:, np.newaxis, np.newaxis]
    time_weights = np.linspace(1.0, 3.0, _N_MEDIA_TIMES)[
        np.newaxis, :, np.newaxis
    ]
    new_media = self.media * geo_weights * time_weights

    media_spend, _ = self._build_spend(
        new_data=tensors.DataTensors(
            media=backend.to_tensor(new_media, dtype=backend.float_dtype)
        )
    )

    # The allocation weights come from the new media, not the historical one.
    new_media_window = new_media[:, -_N_TIMES:, :]
    expected = (
        new_media_window
        / np.sum(new_media_window, axis=(0, 1))
        * self.aggregated_media_spend
    )
    backend_test_utils.assert_allclose(media_spend, expected, rtol=1e-5)
    # Sanity check: the reweighting actually moves spend across geos and times,
    # so the assertion above would fail if the historical media were used.
    self.assertFalse(
        np.allclose(media_spend, self.allocated_media_spend, rtol=1e-2)
    )

  def test_allocation_uses_new_reach_and_frequency(self):
    # Reweight reach non-uniformly so that the RF impressions, and therefore
    # the allocation weights, differ from the historical ones.
    geo_weights = np.linspace(1.0, 5.0, _N_GEOS)[:, np.newaxis, np.newaxis]
    time_weights = np.linspace(1.0, 3.0, _N_MEDIA_TIMES)[
        np.newaxis, :, np.newaxis
    ]
    new_reach = self.reach * geo_weights * time_weights

    _, rf_spend = self._build_spend(
        new_data=tensors.DataTensors(
            reach=backend.to_tensor(new_reach, dtype=backend.float_dtype)
        )
    )

    # The allocation weights come from the new reach times the historical
    # frequency, not from the historical impressions.
    new_impressions = (new_reach * self.frequency)[:, -_N_TIMES:, :]
    expected = (
        new_impressions
        / np.sum(new_impressions, axis=(0, 1))
        * self.aggregated_rf_spend
    )
    backend_test_utils.assert_allclose(rf_spend, expected, rtol=1e-5)
    # Sanity check: the reweighting actually moves spend across geos and times,
    # so the assertion above would fail if the historical reach were used.
    self.assertFalse(np.allclose(rf_spend, self.allocated_rf_spend, rtol=1e-2))

  def test_allocation_without_execution_values_uses_model_tensors(self):
    # Spend is the only required data, so the filled tensors carry no media,
    # reach or frequency and the allocation falls back to the model tensors.
    media_spend, rf_spend = self._build_spend(
        required_tensors_names=constants.SPEND_DATA
    )

    backend_test_utils.assert_allclose(
        media_spend, self.allocated_media_spend, rtol=1e-5
    )
    backend_test_utils.assert_allclose(
        rf_spend, self.allocated_rf_spend, rtol=1e-5
    )

  def test_allocation_with_partial_rf_data_uses_model_tensors(self):
    # `frequency` is not required, so only `reach` is available. Impressions
    # cannot be derived from reach alone, so the model tensors are used.
    _, rf_spend = self._build_spend(
        required_tensors_names=constants.SPEND_DATA + (constants.REACH,)
    )

    backend_test_utils.assert_allclose(
        rf_spend, self.allocated_rf_spend, rtol=1e-5
    )

  def test_three_dimensional_spend_is_unchanged(self):
    input_data = data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        seed=0,
    )
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=spec.ModelSpec(max_lag=15),
    )
    builder = tensors.DataTensorsBuilder(meridian.model_context)

    filled = builder.build_unscaled_inputs(
        required_tensors_names=(constants.MEDIA, constants.MEDIA_SPEND),
    ).tensors

    # Avoid the pytype check complaint.
    assert input_data.media_spend is not None
    backend_test_utils.assert_allclose(
        np.asarray(filled.media_spend), input_data.media_spend.values
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

  def test_build_counterfactual_inputs_raises_on_invalid_times(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    with self.assertRaises(ValueError):
      builder.build_counterfactual_inputs(selected_times=["1999-01-01"])

  def test_normalize_date_str_with_datetime64_and_strings(self):
    self.assertEqual(
        tensors.normalize_date_str("2021-01-25T12:34:56"), "2021-01-25"
    )
    self.assertEqual(
        tensors.normalize_date_str(np.datetime64("2021-01-25")), "2021-01-25"
    )

  def test_normalize_times_set(self):
    self.assertEqual(
        tensors.normalize_times_set(
            ["2021-01-25T12:34:56", np.datetime64("2021-01-25")]
        ),
        {"2021-01-25"},
    )

  def test_resolve_time_indices_date_normalization(self):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)
    input_times = self.input_data.time
    target_date = tensors.normalize_date_str(input_times.values[2])
    resolved = builder._resolve_time_indices(
        selected_times=[f"{target_date}T00:00:00"],
        input_times=input_times,
    )
    assert resolved is not None
    self.assertEqual(list(np.asarray(resolved)), [2])

  def test_data_tensors_time_normalization(self):
    data = tensors.DataTensors(
        media=backend.ones((_N_GEOS, 3, _N_MEDIA_CHANNELS)),
        time=[
            "2021-01-01T12:00:00",
            "2021-01-08",
            "2021-01-15",
        ],
    )
    self.assertEqual(data.time, ("2021-01-01", "2021-01-08", "2021-01-15"))
    self.assertIsInstance(data.time, tuple)

  def test_data_tensors_time_invalid_type_raises_error(self):
    with self.assertRaises(ValueError):
      tensors.DataTensors(
          media=backend.ones((_N_GEOS, 1, _N_MEDIA_CHANNELS)),
          time=[12345],  # pyrefly: ignore[bad-argument-type]
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
    if ctx.compiled_non_media_population_scaling_id is not None:
      scaling_factors = backend.where(
          ctx.compiled_non_media_population_scaling_id,
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

  def _non_media_baseline(
      self, model_spec: spec.ModelSpec
  ) -> backend.Tensor | None:
    """Returns the baseline non-media treatments for the given model spec."""
    meridian = model.Meridian(input_data=self.input_data, model_spec=model_spec)
    builder = tensors.DataTensorsBuilder(meridian.model_context)
    return builder.build_baseline_inputs().tensors.non_media_treatments

  def test_build_baseline_inputs_population_scaling_from_declarative_spec(self):
    """A declaratively named channel is population-scaled like a legacy one."""
    assert self.input_data.non_media_channel is not None
    channels = [str(c) for c in self.input_data.non_media_channel.values]
    legacy = np.zeros(len(channels), dtype=bool)
    legacy[0] = True

    declarative_baseline = self._non_media_baseline(
        spec.ModelSpec(
            max_lag=15, population_scaled_non_media_channels=[channels[0]]
        )
    )
    backend_test_utils.assert_allclose(
        declarative_baseline,
        self._non_media_baseline(
            spec.ModelSpec(max_lag=15, non_media_population_scaling_id=legacy)
        ),
    )
    # Scaling must actually have been applied, or the comparison above would
    # hold for any pair of specs.
    backend_test_utils.assert_not_allequal(
        declarative_baseline,
        self._non_media_baseline(spec.ModelSpec(max_lag=15)),
    )

  def test_build_baseline_inputs_raises_value_error_for_invalid_baseline_values(
      self,
  ):
    builder = tensors.DataTensorsBuilder(self.meridian.model_context)

    # Test invalid types
    with self.assertRaises(ValueError):
      builder.build_baseline_inputs(
          non_media_baseline_values=["invalid"]  # pyrefly: ignore[bad-argument-type]
      )

    # Test invalid length
    invalid_length_values = [1.0] * (_N_NON_MEDIA_CHANNELS + 1)
    with self.assertRaises(ValueError):
      builder.build_baseline_inputs(
          non_media_baseline_values=invalid_length_values
      )

  def test_time_coordinates_returns_none_when_time_is_none(self):
    data = tensors.DataTensors()
    self.assertIsNone(data.time_coordinates)

  def test_time_coordinates_returns_instance_when_time_is_set(self):
    times = ("2021-01-04", "2021-01-11", "2021-01-18")
    data = tensors.DataTensors(time=times)
    self.assertIsNotNone(data.time_coordinates)
    self.assertEqual(data.time_coordinates.all_dates_str, list(times))  # pyrefly: ignore[missing-attribute]

  def test_expand_selected_time_dims_returns_none_when_time_is_none(self):
    data = tensors.DataTensors()
    self.assertIsNone(
        data.expand_selected_time_dims(
            start_date="2021-01-04", end_date="2021-01-11"
        )
    )

  def test_expand_selected_time_dims_returns_none_when_no_dates_passed(self):
    times = ("2021-01-04", "2021-01-11", "2021-01-18")
    data = tensors.DataTensors(time=times)
    self.assertIsNone(data.expand_selected_time_dims())

  def test_expand_selected_time_dims_returns_subset(self):
    times = ("2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25")
    data = tensors.DataTensors(time=times)
    expanded = data.expand_selected_time_dims(
        start_date="2021-01-11", end_date="2021-01-18"
    )
    self.assertEqual(expanded, ["2021-01-11", "2021-01-18"])

  def test_expand_selected_time_dims_returns_none_for_full_range(self):
    times = ("2021-01-04", "2021-01-11", "2021-01-18")
    data = tensors.DataTensors(time=times)
    expanded = data.expand_selected_time_dims(
        start_date="2021-01-04", end_date="2021-01-18"
    )
    self.assertIsNone(expanded)

  def test_expand_selected_time_dims_raises_value_error_for_invalid_date(self):
    times = ("2021-01-04", "2021-01-11", "2021-01-18")
    data = tensors.DataTensors(time=times)
    with self.assertRaises(ValueError):
      data.expand_selected_time_dims(start_date="2020-01-01")


if __name__ == "__main__":
  absltest.main()
