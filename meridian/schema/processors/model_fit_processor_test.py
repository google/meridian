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

"""Unit tests for model_fit_processor.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.analysis import analyzer
from meridian.data import time_coordinates as tc
from meridian.model import model
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import estimate_pb2
from mmm.v1.fit import model_fit_pb2
from meridian.schema.processors import model_fit_processor
from meridian.schema.processors import model_processor
import numpy as np
import xarray as xr

from google.type import date_pb2
from tensorflow.python.util.protobuf import compare


_ALL_TIMES = xr.DataArray(
    np.array([
        "2024-01-01",
        "2024-01-08",
        "2024-01-15",
    ])
)

_EXPECTED_RESULT_PROTO_TRAIN = model_fit_pb2.Result(
    name=constants.TRAIN,
    performance=model_fit_pb2.Performance(
        r_squared=0.9, mape=0.88, weighted_mape=0.95
    ),
    predictions=[
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=1),
                end_date=date_pb2.Date(year=2024, month=1, day=8),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.75,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.62, upperbound=0.96
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.65,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.52, upperbound=0.86
                    )
                ],
            ),
            actual_value=0.75,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=8),
                end_date=date_pb2.Date(year=2024, month=1, day=15),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.7,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.6, upperbound=0.95
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.6,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.5, upperbound=0.85
                    )
                ],
            ),
            actual_value=0.7,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=15),
                end_date=date_pb2.Date(year=2024, month=1, day=22),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.85,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.75, upperbound=0.97
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.75,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.65, upperbound=0.87
                    )
                ],
            ),
            actual_value=0.85,
        ),
    ],
)

_EXPECTED_RESULT_PROTO_TEST = model_fit_pb2.Result(
    name=constants.TEST,
    performance=model_fit_pb2.Performance(
        r_squared=0.74, mape=0.68, weighted_mape=0.83
    ),
    predictions=[
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=1),
                end_date=date_pb2.Date(year=2024, month=1, day=8),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.75,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.65, upperbound=0.86
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.65,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.55, upperbound=0.76
                    )
                ],
            ),
            actual_value=0.62,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=8),
                end_date=date_pb2.Date(year=2024, month=1, day=15),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.65,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.6, upperbound=0.84
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.55,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.5, upperbound=0.74
                    )
                ],
            ),
            actual_value=0.6,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=15),
                end_date=date_pb2.Date(year=2024, month=1, day=22),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.85,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.7, upperbound=0.88
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.75,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.6, upperbound=0.78
                    )
                ],
            ),
            actual_value=0.75,
        ),
    ],
)

_EXPECTED_RESULT_PROTO_ALL_DATA = model_fit_pb2.Result(
    name=constants.ALL_DATA,
    performance=model_fit_pb2.Performance(
        r_squared=0.91, mape=0.87, weighted_mape=0.98
    ),
    predictions=[
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=1),
                end_date=date_pb2.Date(year=2024, month=1, day=8),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.9,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.83, upperbound=0.71
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.8,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.73, upperbound=0.61
                    )
                ],
            ),
            actual_value=0.96,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=8),
                end_date=date_pb2.Date(year=2024, month=1, day=15),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.83,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.75, upperbound=0.65
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.73,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.65, upperbound=0.55
                    )
                ],
            ),
            actual_value=0.95,
        ),
        model_fit_pb2.Prediction(
            date_interval=date_interval_pb2.DateInterval(
                start_date=date_pb2.Date(year=2024, month=1, day=15),
                end_date=date_pb2.Date(year=2024, month=1, day=22),
            ),
            predicted_outcome=estimate_pb2.Estimate(
                value=0.96,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.9, upperbound=0.77
                    )
                ],
            ),
            predicted_baseline=estimate_pb2.Estimate(
                value=0.86,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=0.9, lowerbound=0.8, upperbound=0.67
                    )
                ],
            ),
            actual_value=0.97,
        ),
    ],
)


def _create_expected_model_fit(split: bool) -> model_fit_pb2.ModelFit:
  if split:
    return model_fit_pb2.ModelFit(
        results=[
            _EXPECTED_RESULT_PROTO_TRAIN,
            _EXPECTED_RESULT_PROTO_TEST,
            _EXPECTED_RESULT_PROTO_ALL_DATA,
        ]
    )
  else:
    return model_fit_pb2.ModelFit(
        results=[
            _EXPECTED_RESULT_PROTO_ALL_DATA,
        ]
    )


def _create_expected_vs_actual_data(split: bool) -> xr.Dataset:
  xr_dims_expected = (
      constants.TIME,
      constants.METRIC,
  ) + ((constants.EVALUATION_SET_VAR,) if split else ())
  xr_dims_baseline = xr_dims_expected
  xr_dims_actual = (constants.TIME,) + (
      (constants.EVALUATION_SET_VAR,) if split else ()
  )
  xr_coords = {
      constants.TIME: (
          [constants.TIME],
          _ALL_TIMES.data,
      ),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.CI_LO, constants.CI_HI],
      ),
  }
  if split:
    xr_coords.update({
        constants.EVALUATION_SET_VAR: (
            [constants.EVALUATION_SET_VAR],
            list(constants.EVALUATION_SET),
        )
    })

  time_1_train = [0.75, 0.7, 0.85]
  time_1_test = [0.75, 0.65, 0.85]
  time_1_all_data = [0.9, 0.83, 0.96]

  time_2_train = [0.62, 0.6, 0.75]
  time_2_test = [0.65, 0.6, 0.7]
  time_2_all_data = [0.83, 0.75, 0.9]

  time_3_train = [0.96, 0.95, 0.97]
  time_3_test = [0.86, 0.84, 0.88]
  time_3_all_data = [0.71, 0.65, 0.77]

  stacked_train = np.stack([time_1_train, time_2_train, time_3_train], axis=-1)
  stacked_test = np.stack([time_1_test, time_2_test, time_3_test], axis=-1)
  stacked_all_data = np.stack(
      [time_1_all_data, time_2_all_data, time_3_all_data], axis=-1
  )
  stacked_total = np.stack(
      [stacked_train, stacked_test, stacked_all_data],
      axis=-1,
  )

  xr_data = {
      constants.EXPECTED: (
          xr_dims_expected,
          stacked_total if split else stacked_all_data,
      ),
      constants.BASELINE: (
          xr_dims_baseline,
          (stacked_total if split else stacked_all_data) - 0.1,
      ),
      constants.ACTUAL: (
          xr_dims_actual,
          stacked_train if split else time_3_train,
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_predictive_accuracy_data(split: bool) -> xr.Dataset:
  xr_dims = (
      constants.METRIC,
      constants.GEO_GRANULARITY,
  ) + ((constants.EVALUATION_SET_VAR,) if split else ())
  xr_coords = {
      constants.METRIC: (
          [constants.METRIC],
          [constants.R_SQUARED, constants.MAPE, constants.WMAPE],
      ),
      constants.GEO_GRANULARITY: (
          [constants.GEO_GRANULARITY],
          [constants.GEO, constants.NATIONAL],
      ),
  }
  if split:
    xr_coords.update({
        constants.EVALUATION_SET_VAR: (
            [constants.EVALUATION_SET_VAR],
            list(constants.EVALUATION_SET),
        )
    })

  geo_train = [0.8, 0.75, 0.85]
  national_train = [0.9, 0.88, 0.95]
  geo_test = [0.75, 0.65, 0.85]
  national_test = [0.74, 0.68, 0.83]
  geo_all_data = [0.93, 0.9, 0.96]
  national_all_data = [0.91, 0.87, 0.98]

  stacked_train = np.stack([geo_train, national_train], axis=-1)
  stacked_test = np.stack([geo_test, national_test], axis=-1)
  stacked_all_data = np.stack([geo_all_data, national_all_data], axis=-1)
  stacked_total = np.stack(
      [stacked_train, stacked_test, stacked_all_data], axis=-1
  )

  xr_data = {
      constants.VALUE: (xr_dims, stacked_total if split else stacked_all_data)
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


class ModelFitSpecTest(absltest.TestCase):

  def test_confidence_level_is_below_zero(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be greater than 0 and less than 1.",
    ):
      spec = model_fit_processor.ModelFitSpec(confidence_level=-1)
      spec.validate()

  def test_confidence_level_is_above_one(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be greater than 0 and less than 1.",
    ):
      spec = model_fit_processor.ModelFitSpec(confidence_level=1.5)
      spec.validate()

  def test_validates_successfully(self):
    spec = model_fit_processor.ModelFitSpec(split=True, confidence_level=0.95)

    spec.validate()

    self.assertEqual(spec.split, True)
    self.assertEqual(spec.confidence_level, 0.95)


class ModelFitProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mock_meridian_model = self.enter_context(
        mock.patch.object(model, "Meridian", autospec=True)
    )
    self.mock_meridian_model.input_data.time = _ALL_TIMES

    self.mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, "Analyzer", autospec=True)
    )

    self.mock_trained_model = self.enter_context(
        mock.patch.object(model_processor, "TrainedModel", autospec=True)
    )
    self.mock_trained_model.mmm = self.mock_meridian_model
    self.mock_trained_model.internal_analyzer = self.mock_analyzer
    self.mock_trained_model.time_coordinates = tc.TimeCoordinates.from_dates(
        _ALL_TIMES
    )

    self.mock_ensure_trained_model = self.enter_context(
        mock.patch.object(
            model_processor, "ensure_trained_model", autospec=True
        )
    )
    self.mock_ensure_trained_model.return_value = self.mock_trained_model

  def test_spec_type_returns_model_fit_spec(self):
    self.assertEqual(
        model_fit_processor.ModelFitProcessor.spec_type(),
        model_fit_processor.ModelFitSpec,
    )

  def test_output_type_returns_model_fit_proto(self):
    self.assertEqual(
        model_fit_processor.ModelFitProcessor.output_type(),
        model_fit_pb2.ModelFit,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="multiple_evaluation_sets",
          split=True,
      ),
      dict(testcase_name="single_evaluation_set", split=False),
  )
  def test_execute(self, split: bool):
    self.mock_analyzer.expected_vs_actual_data.return_value = (
        _create_expected_vs_actual_data(split)
    )
    self.mock_analyzer.predictive_accuracy.return_value = (
        _create_predictive_accuracy_data(split)
    )

    spec = model_fit_processor.ModelFitSpec(split)
    model_fit = model_fit_processor.ModelFitProcessor(
        trained_model=self.mock_trained_model,
    ).execute([spec])

    compare.assertProtoEqual(
        self, model_fit, _create_expected_model_fit(split)
    )


if __name__ == "__main__":
  absltest.main()
