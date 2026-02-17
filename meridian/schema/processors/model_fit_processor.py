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

"""Meridian module for analyzing model fit in a Meridian model.

This module provides a `ModelFitProcessor`, which assesses the goodness of fit
of a trained Meridian model. It compares the model's predictions against the
actual observed data, generating key performance metrics.

Key metrics generated include R-squared, MAPE, and Weighted MAPE. The output
also includes timeseries data of actual values versus predicted values (with
confidence intervals) and the predicted baseline.

The results are structured into a `ModelFit` protobuf message.

Key Classes:

-   `ModelFitSpec`: Dataclass to specify parameters for the model fit analysis,
    such as whether to split by train/test sets and the confidence level for
    intervals.
-   `ModelFitProcessor`: The processor class that performs the fit analysis.

Example Usage:

```python
from meridian.schema.processors import model_fit_processor
from meridian.schema.processors import model_processor

# Assuming 'mmm' is a trained Meridian model object
trained_model = model_processor.TrainedModel(mmm)

# Default spec: split results by train/test if holdout ID exists
spec = model_fit_processor.ModelFitSpec()

processor = model_fit_processor.ModelFitProcessor(trained_model)
# result is a model_fit_pb2.ModelFit proto
result = processor.execute([spec])

print("Model Fit Analysis Results:")
for res in result.results:
    print(f"  Dataset: {res.name}")
    print(f"    R-squared: {res.performance.r_squared:.3f}")
    print(f"    MAPE: {res.performance.mape:.3f}")
    print(f"    Weighted MAPE: {res.performance.weighted_mape:.3f}")
    # Prediction data is available in res.predictions
    # Each element in res.predictions corresponds to a time point.
    # e.g., res.predictions[0].actual_value
    # e.g., res.predictions[0].predicted_outcome.value
```

Note: Only one spec is supported per processor execution.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import warnings

from meridian import constants
from mmm.v1 import mmm_pb2
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import estimate_pb2
from mmm.v1.fit import model_fit_pb2
from meridian.schema.processors import model_processor
from meridian.schema.utils import time_record
import xarray as xr


__all__ = [
    "ModelFitSpec",
    "ModelFitProcessor",
]


@dataclasses.dataclass(frozen=True)
class ModelFitSpec(model_processor.Spec):
  """Stores parameters needed for generating ModelFit protos.

  Attributes:
    split: If `True` and Meridian model contains holdout IDs, results are
      generated for `'Train'`, `'Test'`, and `'All Data'` sets.
    confidence_level: Confidence level for prior and posterior credible
      intervals, represented as a value between zero and one.
  """

  split: bool = True
  confidence_level: float = constants.DEFAULT_CONFIDENCE_LEVEL

  def validate(self):
    if self.confidence_level <= 0 or self.confidence_level >= 1:
      raise ValueError(
          "Confidence level must be greater than 0 and less than 1."
      )


class ModelFitProcessor(
    model_processor.ModelProcessor[ModelFitSpec, model_fit_pb2.ModelFit]
):
  """Generates a ModelFit proto for a given trained Meridian model.

  The proto contains performance metrics for each dataset as well as a list of
  predictions.
  """

  def __init__(
      self,
      trained_model: model_processor.ModelType,
  ):
    trained_model = model_processor.ensure_trained_model(trained_model)
    self._analyzer = trained_model.internal_analyzer
    self._time_coordinates = trained_model.time_coordinates

  @classmethod
  def spec_type(cls) -> type[ModelFitSpec]:
    return ModelFitSpec

  @classmethod
  def output_type(cls) -> type[model_fit_pb2.ModelFit]:
    return model_fit_pb2.ModelFit

  def _set_output(self, output: mmm_pb2.Mmm, result: model_fit_pb2.ModelFit):
    output.model_fit.CopyFrom(result)

  def execute(self, specs: Sequence[ModelFitSpec]) -> model_fit_pb2.ModelFit:
    model_fit_spec = specs[0]
    if len(specs) > 1:
      warnings.warn(
          "Multiple specs were provided. Only the first one will be used."
      )

    expected_vs_actual = self._analyzer.expected_vs_actual_data(
        confidence_level=model_fit_spec.confidence_level,
        split_by_holdout_id=model_fit_spec.split,
        aggregate_geos=True,
    )
    metrics = self._analyzer.predictive_accuracy()
    time_to_date_interval = time_record.convert_times_to_date_intervals(
        self._time_coordinates.datetime_index
    )

    results: list[model_fit_pb2.Result] = []

    if constants.EVALUATION_SET_VAR in expected_vs_actual.coords:
      results.append(
          self._create_result(
              result_type=constants.TRAIN,
              expected_vs_actual=expected_vs_actual.sel(
                  evaluation_set=constants.TRAIN
              ),
              metrics=metrics.sel(evaluation_set=constants.TRAIN),
              model_fit_spec=model_fit_spec,
              time_to_date_interval=time_to_date_interval,
          )
      )
      results.append(
          self._create_result(
              result_type=constants.TEST,
              expected_vs_actual=expected_vs_actual.sel(
                  evaluation_set=constants.TEST
              ),
              metrics=metrics.sel(evaluation_set=constants.TEST),
              model_fit_spec=model_fit_spec,
              time_to_date_interval=time_to_date_interval,
          )
      )
      results.append(
          self._create_result(
              result_type=constants.ALL_DATA,
              expected_vs_actual=expected_vs_actual.sel(
                  evaluation_set=constants.ALL_DATA
              ),
              metrics=metrics.sel(evaluation_set=constants.ALL_DATA),
              model_fit_spec=model_fit_spec,
              time_to_date_interval=time_to_date_interval,
          )
      )
    else:
      results.append(
          self._create_result(
              result_type=constants.ALL_DATA,
              expected_vs_actual=expected_vs_actual,
              metrics=metrics,
              model_fit_spec=model_fit_spec,
              time_to_date_interval=time_to_date_interval,
          )
      )

    return model_fit_pb2.ModelFit(results=results)

  def _create_result(
      self,
      result_type: str,
      expected_vs_actual: xr.Dataset,
      metrics: xr.Dataset,
      model_fit_spec: ModelFitSpec,
      time_to_date_interval: Mapping[str, date_interval_pb2.DateInterval],
  ) -> model_fit_pb2.Result:
    """Creates a proto that stores the model fit results for an evaluation set.

    Args:
      result_type: The evaluation set (`"Train"`, `"Test"`, or `"All Data"`) for
        the result.
      expected_vs_actual: A dataset containing the expected and actual values
        for the model. This dataset is filtered by the evaluation set in the
        calling code.
      metrics: A dataset containing the performance metrics for the model. This
        dataset is filtered by the evaluation set in the calling code.
      model_fit_spec: An instance of ModelFitSpec.
      time_to_date_interval: A mapping of date strings (in YYYY-MM-DD format) to
        date interval protos.

    Returns:
      A proto containing the results of the model fit analysis for the given
      evaluation set.
    """

    predictions: list[model_fit_pb2.Prediction] = []

    for start_date in self._time_coordinates.all_dates_str:
      date_interval = time_to_date_interval[start_date]
      actual = (
          expected_vs_actual.data_vars[constants.ACTUAL]
          .sel(
              time=start_date,
          )
          .item()
      )
      expected_dataset = expected_vs_actual[constants.EXPECTED].sel(
          time=start_date,
      )
      expected = expected_dataset.sel(metric=constants.MEAN).item()
      expected_lowerbound = expected_dataset.sel(metric=constants.CI_LO).item()
      expected_upperbound = expected_dataset.sel(metric=constants.CI_HI).item()
      baseline_dataset = expected_vs_actual[constants.BASELINE].sel(
          time=start_date,
      )
      baseline = baseline_dataset.sel(metric=constants.MEAN).item()
      baseline_lowerbound = baseline_dataset.sel(metric=constants.CI_LO).item()
      baseline_upperbound = baseline_dataset.sel(metric=constants.CI_HI).item()

      prediction = self._create_prediction(
          model_fit_spec=model_fit_spec,
          date_interval=date_interval,
          actual_value=actual,
          estimated_value=expected,
          estimated_lower_bound=expected_lowerbound,
          estimated_upper_bound=expected_upperbound,
          baseline_value=baseline,
          baseline_lower_bound=baseline_lowerbound,
          baseline_upper_bound=baseline_upperbound,
      )
      predictions.append(prediction)

    performance = self._evaluate_model_fit(metrics)

    return model_fit_pb2.Result(
        name=result_type, predictions=predictions, performance=performance
    )

  def _create_prediction(
      self,
      model_fit_spec: ModelFitSpec,
      date_interval: date_interval_pb2.DateInterval,
      actual_value: float,
      estimated_value: float,
      estimated_lower_bound: float,
      estimated_upper_bound: float,
      baseline_value: float,
      baseline_lower_bound: float,
      baseline_upper_bound: float,
  ) -> model_fit_pb2.Prediction:
    """Creates a proto that stores the model's prediction for the given date.

    Args:
      model_fit_spec: An instance of ModelFitSpec.
      date_interval: A DateInterval proto containing the start date and end date
        for this prediction.
      actual_value: The model's actual value for this date.
      estimated_value: The model's estimated value for this date.
      estimated_lower_bound: The lower bound of the estimated value's confidence
        interval.
      estimated_upper_bound: The upper bound of the estimated value's confidence
        interval.
      baseline_value: The baseline value for this date.
      baseline_lower_bound: The lower bound of the baseline value's confidence
        interval.
      baseline_upper_bound: The upper bound of the baseline value's confidence
        interval.

    Returns:
      A proto containing the model's predicted value and actual value for the
      given date.
    """

    estimate = estimate_pb2.Estimate(value=estimated_value)
    estimate.uncertainties.add(
        probability=model_fit_spec.confidence_level,
        lowerbound=estimated_lower_bound,
        upperbound=estimated_upper_bound,
    )

    baseline_estimate = estimate_pb2.Estimate(value=baseline_value)
    baseline_estimate.uncertainties.add(
        probability=model_fit_spec.confidence_level,
        lowerbound=baseline_lower_bound,
        upperbound=baseline_upper_bound,
    )

    return model_fit_pb2.Prediction(
        date_interval=date_interval,
        predicted_outcome=estimate,
        predicted_baseline=baseline_estimate,
        actual_value=actual_value,
    )

  def _evaluate_model_fit(
      self,
      metrics: xr.Dataset,
  ) -> model_fit_pb2.Performance:
    """Creates a proto that stores the model's performance metrics.

    Args:
      metrics: A dataset containing the performance metrics for the model. This
        dataset is filtered by evaluation set before this function is called.

    Returns:
      A proto containing the model's performance metrics for a specific
      evaluation set.
    """

    performance = model_fit_pb2.Performance()
    performance.r_squared = (
        metrics[constants.VALUE]
        .sel(
            geo_granularity=constants.NATIONAL,
            metric=constants.R_SQUARED,
        )
        .item()
    )
    performance.mape = (
        metrics[constants.VALUE]
        .sel(
            geo_granularity=constants.NATIONAL,
            metric=constants.MAPE,
        )
        .item()
    )
    performance.weighted_mape = (
        metrics[constants.VALUE]
        .sel(
            geo_granularity=constants.NATIONAL,
            metric=constants.WMAPE,
        )
        .item()
    )

    return performance
