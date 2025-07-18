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

"""Unit tests for marketing_processor.py."""

import dataclasses
import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import test_utils
from meridian.data import time_coordinates
from meridian.model import model
from mmm.v1.common import date_interval_pb2
from mmm.v1.common import estimate_pb2
from mmm.v1.common import kpi_type_pb2
from mmm.v1.marketing.analysis import kpi_outcome_pb2
from mmm.v1.marketing.analysis import marketing_analysis_pb2
from mmm.v1.marketing.analysis import media_analysis_pb2
from mmm.v1.marketing.analysis import non_media_analysis_pb2
from mmm.v1.marketing.analysis import response_curve_pb2
from schema.processors import marketing_processor
from schema.processors import model_processor
import numpy as np
import tensorflow as tf
import xarray as xr

from google.type import date_pb2
from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format


_ALL_TIMES = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
]
_ALL_GEOS = ["geo_1", "geo_2"]
_ALL_CHANNELS = ["channel_1", "channel_2"]

_SPEND_MULTIPLIERS = [1, 2, 3]

_SPEND = [[0.48, 0.61], [0.5, 0.7], [0.6, 0.72]]

_REVENUE_PER_KPI = 5
_NUMBER_OF_GEO_POINTS = len(_ALL_GEOS)
_NUMBER_OF_TIMES = len(_ALL_TIMES)
_NUMBER_OF_TIME_POINTS = int(len(_ALL_TIMES) / 2)
_NUMBER_OF_CHANNELS = len(_ALL_CHANNELS) + 1  # +1 for the pseudo "all channels"

_REV_REVENUE_PER_KPI = xr.DataArray(
    data=np.ones((_NUMBER_OF_GEO_POINTS, _NUMBER_OF_TIME_POINTS)),
    dims=[constants.GEO, constants.TIME],
    coords={
        constants.GEO: _ALL_GEOS,
        constants.TIME: _ALL_TIMES[:_NUMBER_OF_TIME_POINTS],
    },
    name=constants.REVENUE_PER_KPI,
)
_NONREV_REVENUE_PER_KPI = xr.DataArray(
    data=(_REV_REVENUE_PER_KPI * _REVENUE_PER_KPI),
    dims=[constants.GEO, constants.TIME],
    coords={
        constants.GEO: _ALL_GEOS,
        constants.TIME: _ALL_TIMES[:_NUMBER_OF_TIME_POINTS],
    },
    name=constants.REVENUE_PER_KPI,
)

_CONFIDENCE_LEVEL = constants.DEFAULT_CONFIDENCE_LEVEL

_SPEND_DATA = test_utils.SAMPLE_SPEND[0:_NUMBER_OF_CHANNELS]
_PCT_OF_SPEND_DATA = test_utils.SAMPLE_PCT_OF_SPEND[0:_NUMBER_OF_CHANNELS]
_EFFECTIVENESS_DATA = test_utils.SAMPLE_EFFECTIVENESS[0:_NUMBER_OF_CHANNELS]
_INCREMENTAL_OUTCOME_DATA = test_utils.SAMPLE_INCREMENTAL_OUTCOME[
    0:_NUMBER_OF_CHANNELS
]
_PCT_OF_CONTRIBUTION_DATA = test_utils.SAMPLE_PCT_OF_CONTRIBUTION[
    0:_NUMBER_OF_CHANNELS
]
_ROI_DATA = test_utils.SAMPLE_ROI[0:_NUMBER_OF_CHANNELS]
_MROI_DATA = test_utils.SAMPLE_MROI[0:_NUMBER_OF_CHANNELS]
_CPIK_DATA = test_utils.SAMPLE_CPIK[0:_NUMBER_OF_CHANNELS]

_DATE_1 = datetime.date(2024, 1, 1)
_DATE_2 = datetime.date(2024, 1, 15)
_DATE_3 = datetime.date(2024, 1, 29)

_MEDIA_SUMMARY_SPEC_1 = marketing_processor.MarketingAnalysisSpec(
    media_summary_spec=marketing_processor.MediaSummarySpec(),
    response_curve_spec=marketing_processor.ResponseCurveSpec(),
    start_date=_DATE_1,
    end_date=_DATE_2,
)
_MEDIA_SUMMARY_SPEC_2 = marketing_processor.MarketingAnalysisSpec(
    media_summary_spec=marketing_processor.MediaSummarySpec(),
    # The response_curve_spec sub-spec should have default factory.
    start_date=_DATE_2,
    end_date=_DATE_3,
)
_MEDIA_SUMMARY_SPEC_NON_AGGREGATED = marketing_processor.MarketingAnalysisSpec(
    media_summary_spec=marketing_processor.MediaSummarySpec(
        aggregate_times=False
    ),
    response_curve_spec=marketing_processor.ResponseCurveSpec(),
    start_date=_DATE_1,
    end_date=_DATE_2,
)
_INCREMENTAL_OUTCOME_SPEC = marketing_processor.MarketingAnalysisSpec(
    incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(),
    response_curve_spec=marketing_processor.ResponseCurveSpec(),
    start_date=_DATE_1,
    end_date=_DATE_3,
)

_DATE_INTERVAL = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=1, day=1),
    end_date=date_pb2.Date(year=2024, month=1, day=29),
)
# Date interval corresponding to the first half of _DATE_INTERVAL.
_DATE_INTERVAL_FIRST_HALF = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=1, day=1),
    end_date=date_pb2.Date(year=2024, month=1, day=15),
)
# Date interval corresponding to the second half of _DATE_INTERVAL.
_DATE_INTERVAL_SECOND_HALF = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=1, day=15),
    end_date=date_pb2.Date(year=2024, month=1, day=29),
)
_NEW_DATE_INTERVAL = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=1, day=29),
    end_date=date_pb2.Date(year=2024, month=2, day=26),
)
# Date interval corresponding to the first time period of _NEW_DATE_INTERVAL.
_NEW_DATE_INTERVAL_1 = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=1, day=29),
    end_date=date_pb2.Date(year=2024, month=2, day=5),
)
# Date interval corresponding to the second time period of _NEW_DATE_INTERVAL.
_NEW_DATE_INTERVAL_2 = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=2, day=5),
    end_date=date_pb2.Date(year=2024, month=2, day=12),
)
# Date interval corresponding to the third time period of _NEW_DATE_INTERVAL.
_NEW_DATE_INTERVAL_3 = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=2, day=12),
    end_date=date_pb2.Date(year=2024, month=2, day=19),
)
# Date interval corresponding to the fourth time period of _NEW_DATE_INTERVAL.
_NEW_DATE_INTERVAL_4 = date_interval_pb2.DateInterval(
    start_date=date_pb2.Date(year=2024, month=2, day=19),
    end_date=date_pb2.Date(year=2024, month=2, day=26),
)
_NEW_DATA = analyzer.DataTensors(
    media=tf.convert_to_tensor([[
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ]]),
    revenue_per_kpi=tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]]),
    time=tf.convert_to_tensor(
        ["2024-01-29", "2024-02-05", "2024-02-12", "2024-02-19"]
    ),
)

_EXPECTED_RESPONSE_POINTS_1 = [
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[0][0],
        incremental_kpi=0.75,
    ),
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[1][0],
        incremental_kpi=0.7,
    ),
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[2][0],
        incremental_kpi=0.85,
    ),
]

_EXPECTED_RESPONSE_POINTS_2 = [
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[0][1],
        incremental_kpi=0.68,
    ),
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[1][1],
        incremental_kpi=0.65,
    ),
    response_curve_pb2.ResponsePoint(
        input_value=_SPEND[2][1],
        incremental_kpi=0.8,
    ),
]

_baseline_outcome = test_utils.SAMPLE_BASELINE_EXPECTED_OUTCOME[:, 1]
_baseline_contribution_share = (
    test_utils.SAMPLE_BASELINE_PCT_OF_CONTRIBUTION[:, 1] / 100
)
_BASELINE_CONTRIBUTION = kpi_outcome_pb2.Contribution(
    value=estimate_pb2.Estimate(
        value=_baseline_outcome[0],
        uncertainties=[
            estimate_pb2.Estimate.Uncertainty(
                probability=_CONFIDENCE_LEVEL,
                lowerbound=_baseline_outcome[2],
                upperbound=_baseline_outcome[3],
            )
        ],
    ),
    share=estimate_pb2.Estimate(
        value=_baseline_contribution_share[0],
        uncertainties=[
            estimate_pb2.Estimate.Uncertainty(
                probability=_CONFIDENCE_LEVEL,
                lowerbound=_baseline_contribution_share[2],
                upperbound=_baseline_contribution_share[3],
            ),
        ],
    ),
)


def _create_baseline_summary_metrics_data() -> xr.Dataset:
  xr_dims = (constants.METRIC, constants.DISTRIBUTION)
  xr_coords = {
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.MEDIAN, constants.CI_LO, constants.CI_HI],
      ),
      constants.DISTRIBUTION: (
          [constants.DISTRIBUTION],
          [constants.PRIOR, constants.POSTERIOR],
      ),
  }

  xr_data = {
      constants.PCT_OF_CONTRIBUTION: (
          xr_dims,
          test_utils.SAMPLE_BASELINE_PCT_OF_CONTRIBUTION,
      ),
      constants.BASELINE_OUTCOME: (
          xr_dims,
          test_utils.SAMPLE_BASELINE_EXPECTED_OUTCOME,
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_baseline_summary_metrics_data_with_time_breakdown() -> xr.Dataset:
  xr_dims = (
      constants.TIME,
      constants.METRIC,
      constants.DISTRIBUTION,
  )
  xr_coords = {
      constants.TIME: ([constants.TIME], _ALL_TIMES),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.MEDIAN, constants.CI_LO, constants.CI_HI],
      ),
      constants.DISTRIBUTION: (
          [constants.DISTRIBUTION],
          [constants.PRIOR, constants.POSTERIOR],
      ),
  }

  xr_data = {
      constants.PCT_OF_CONTRIBUTION: (
          xr_dims,
          np.full([_NUMBER_OF_TIMES, 4, 2], 100),
      ),
      constants.BASELINE_OUTCOME: (
          xr_dims,
          np.full([_NUMBER_OF_TIMES, 4, 2], 200),
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_response_curve_data() -> xr.Dataset:
  xr_dims_spend = (
      constants.SPEND_MULTIPLIER,
      constants.CHANNEL,
  )
  xr_dims_incremental_outcome = (
      constants.SPEND_MULTIPLIER,
      constants.CHANNEL,
      constants.METRIC,
  )
  xr_coords = {
      constants.CHANNEL: ([constants.CHANNEL], _ALL_CHANNELS),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.CI_LO, constants.CI_HI],
      ),
      constants.SPEND_MULTIPLIER: (
          [constants.SPEND_MULTIPLIER],
          _SPEND_MULTIPLIERS,
      ),
  }

  incremental_outcome_1_channel_1 = [0.75, 0.7, 0.85]
  incremental_outcome_1_channel_2 = [0.68, 0.65, 0.8]

  incremental_outcome_2_channel_1 = [0.62, 0.6, 0.75]
  incremental_outcome_2_channel_2 = [0.6, 0.55, 0.7]

  incremental_outcome_3_channel_1 = [0.96, 0.95, 0.94]
  incremental_outcome_3_channel_2 = [0.91, 0.84, 0.88]

  stacked_incremental_outcome_1 = np.stack(
      [incremental_outcome_1_channel_1, incremental_outcome_1_channel_2],
      axis=-1,
  )
  stacked_incremental_outcome_2 = np.stack(
      [incremental_outcome_2_channel_1, incremental_outcome_2_channel_2],
      axis=-1,
  )
  stacked_incremental_outcome_3 = np.stack(
      [incremental_outcome_3_channel_1, incremental_outcome_3_channel_2],
      axis=-1,
  )
  stacked_total = np.stack(
      [
          stacked_incremental_outcome_1,
          stacked_incremental_outcome_2,
          stacked_incremental_outcome_3,
      ],
      axis=-1,
  )

  xr_data = {
      constants.SPEND: (xr_dims_spend, _SPEND),
      constants.INCREMENTAL_OUTCOME: (
          xr_dims_incremental_outcome,
          stacked_total,
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_media_summary_metrics_aggregated_data(
    is_revenue_type: bool,
    spec_number: int,
) -> xr.Dataset:
  xr_dims = (constants.CHANNEL,)
  xr_dims_with_ci_and_distribution = (
      constants.CHANNEL,
      constants.METRIC,
      constants.DISTRIBUTION,
  )
  xr_coords_with_ci_and_distribution = {
      constants.CHANNEL: (
          [constants.CHANNEL],
          _ALL_CHANNELS + [constants.ALL_CHANNELS],
      ),
      constants.METRIC: (
          [constants.METRIC],
          [constants.MEAN, constants.MEDIAN, constants.CI_LO, constants.CI_HI],
      ),
      constants.DISTRIBUTION: (
          [constants.DISTRIBUTION],
          [constants.PRIOR, constants.POSTERIOR],
      ),
  }

  revenue_scalar = 1 if is_revenue_type else _REVENUE_PER_KPI
  scalar = revenue_scalar * spec_number

  spend = _SPEND_DATA * scalar
  pct_of_spend = _PCT_OF_SPEND_DATA * scalar
  effectiveness = _EFFECTIVENESS_DATA * scalar
  incremental_outcome = _INCREMENTAL_OUTCOME_DATA * scalar
  pct_of_contribution = _PCT_OF_CONTRIBUTION_DATA * scalar
  roi = _ROI_DATA * scalar
  mroi = _MROI_DATA * scalar
  cpik = _CPIK_DATA * scalar

  xr_data = {
      constants.SPEND: (
          xr_dims,
          spend,
      ),
      constants.PCT_OF_SPEND: (
          xr_dims,
          pct_of_spend,
      ),
      constants.EFFECTIVENESS: (
          xr_dims_with_ci_and_distribution,
          effectiveness,
      ),
      constants.INCREMENTAL_OUTCOME: (
          xr_dims_with_ci_and_distribution,
          incremental_outcome,
      ),
      constants.PCT_OF_CONTRIBUTION: (
          xr_dims_with_ci_and_distribution,
          pct_of_contribution,
      ),
      constants.ROI: (
          xr_dims_with_ci_and_distribution,
          roi,
      ),
      constants.MROI: (
          xr_dims_with_ci_and_distribution,
          mroi,
      ),
      constants.CPIK: (
          xr_dims_with_ci_and_distribution,
          cpik,
      ),
  }

  return xr.Dataset(
      data_vars=xr_data, coords=xr_coords_with_ci_and_distribution
  )


def _create_media_summary_metrics_data_with_time_breakdown() -> xr.Dataset:
  xr_dims = (constants.TIME, constants.CHANNEL)
  xr_dims_with_ci_and_distribution = (
      constants.TIME,
      constants.CHANNEL,
      constants.METRIC,
      constants.DISTRIBUTION,
  )
  xr_coords = {
      constants.TIME: ([constants.TIME], _ALL_TIMES),
      constants.CHANNEL: (
          [constants.CHANNEL],
          _ALL_CHANNELS + [constants.ALL_CHANNELS],
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
      constants.DISTRIBUTION: (
          [constants.DISTRIBUTION],
          [constants.PRIOR, constants.POSTERIOR],
      ),
  }
  xr_data = {
      constants.SPEND: (
          xr_dims,
          np.full([_NUMBER_OF_TIMES, _NUMBER_OF_CHANNELS], 10),
      ),
      constants.PCT_OF_SPEND: (
          xr_dims,
          np.full([_NUMBER_OF_TIMES, _NUMBER_OF_CHANNELS], 20),
      ),
      constants.INCREMENTAL_OUTCOME: (
          xr_dims_with_ci_and_distribution,
          np.full([_NUMBER_OF_TIMES, _NUMBER_OF_CHANNELS, 4, 2], 60),
      ),
      constants.PCT_OF_CONTRIBUTION: (
          xr_dims_with_ci_and_distribution,
          np.full([_NUMBER_OF_TIMES, _NUMBER_OF_CHANNELS, 4, 2], 70),
      ),
  }

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def _create_expected_media_analysis(
    index: int,
    scalar_multiplier: int,
    is_revenue_type: bool,
    # Only used in the case where we expect a media analysis for non-revenue
    # data with revenue-type KPI (i.e. `revenue_per_kpi` is defined).
    create_derived_non_revenue_kpi_outcome: bool = False,
) -> media_analysis_pb2.MediaAnalysis:
  sample_incremental_outcome = (
      test_utils.SAMPLE_INCREMENTAL_OUTCOME[index, :, 1] * scalar_multiplier
  )
  sample_contribution_share = (
      test_utils.SAMPLE_PCT_OF_CONTRIBUTION[index, :, 1]
      / 100
      * scalar_multiplier
  )
  sample_effectiveness = (
      test_utils.SAMPLE_EFFECTIVENESS[index, :, 1] * scalar_multiplier
  )
  sample_roi = test_utils.SAMPLE_ROI[index, :, 1] * scalar_multiplier
  sample_mroi = test_utils.SAMPLE_MROI[index, :, 1] * scalar_multiplier
  sample_cpik = test_utils.SAMPLE_CPIK[index, :, 1] * scalar_multiplier

  expected_response_points = (
      _EXPECTED_RESPONSE_POINTS_1 if index == 0 else _EXPECTED_RESPONSE_POINTS_2
  )

  kpi_outcome = kpi_outcome_pb2.KpiOutcome(
      kpi_type=(
          kpi_type_pb2.REVENUE if is_revenue_type else kpi_type_pb2.NON_REVENUE
      ),
      contribution=kpi_outcome_pb2.Contribution(
          value=estimate_pb2.Estimate(
              value=sample_incremental_outcome[0],
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=_CONFIDENCE_LEVEL,
                      lowerbound=sample_incremental_outcome[2],
                      upperbound=sample_incremental_outcome[3],
                  )
              ],
          ),
          share=estimate_pb2.Estimate(
              value=sample_contribution_share[0],
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=_CONFIDENCE_LEVEL,
                      lowerbound=sample_contribution_share[2],
                      upperbound=sample_contribution_share[3],
                  )
              ],
          ),
      ),
      roi=estimate_pb2.Estimate(
          value=sample_roi[0],
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=_CONFIDENCE_LEVEL,
                  lowerbound=sample_roi[2],
                  upperbound=sample_roi[3],
              )
          ],
      ),
      cost_per_contribution=estimate_pb2.Estimate(
          value=sample_cpik[1],  # Uses the median instead of the mean.
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=_CONFIDENCE_LEVEL,
                  lowerbound=sample_cpik[2],
                  upperbound=sample_cpik[3],
              )
          ],
      ),
  )

  # Effectiveness is excluded from KpiOutcome for the pseudo "all channels".
  if index != 2:
    effectiveness = kpi_outcome_pb2.Effectiveness(
        media_unit=constants.IMPRESSIONS,
        value=estimate_pb2.Estimate(
            value=sample_effectiveness[0],
            uncertainties=[
                estimate_pb2.Estimate.Uncertainty(
                    probability=_CONFIDENCE_LEVEL,
                    lowerbound=sample_effectiveness[2],
                    upperbound=sample_effectiveness[3],
                )
            ],
        ),
    )
    kpi_outcome.effectiveness.CopyFrom(effectiveness)

    # mROI is also excluded from KpiOutcome for the pseudo "all channels".
    mroi = estimate_pb2.Estimate(
        value=sample_mroi[0],
        uncertainties=[
            estimate_pb2.Estimate.Uncertainty(
                probability=_CONFIDENCE_LEVEL,
                lowerbound=sample_mroi[2],
                upperbound=sample_mroi[3],
            )
        ],
    )
    kpi_outcome.marginal_roi.CopyFrom(mroi)

  kpi_outcomes = [kpi_outcome]

  if create_derived_non_revenue_kpi_outcome:
    second_kpi_outcome = kpi_outcome_pb2.KpiOutcome(
        # The secondary outcome is always non-revenue KPI type.
        kpi_type=kpi_type_pb2.NON_REVENUE,
        contribution=kpi_outcome_pb2.Contribution(
            value=estimate_pb2.Estimate(
                value=sample_incremental_outcome[0] * _REVENUE_PER_KPI,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=_CONFIDENCE_LEVEL,
                        lowerbound=(
                            sample_incremental_outcome[2] * _REVENUE_PER_KPI
                        ),
                        upperbound=(
                            sample_incremental_outcome[3] * _REVENUE_PER_KPI
                        ),
                    )
                ],
            ),
            share=estimate_pb2.Estimate(
                value=sample_contribution_share[0] * _REVENUE_PER_KPI,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=_CONFIDENCE_LEVEL,
                        lowerbound=(
                            sample_contribution_share[2] * _REVENUE_PER_KPI
                        ),
                        upperbound=(
                            sample_contribution_share[3] * _REVENUE_PER_KPI
                        ),
                    )
                ],
            ),
        ),
        roi=estimate_pb2.Estimate(
            value=sample_roi[0] * _REVENUE_PER_KPI,
            uncertainties=[
                estimate_pb2.Estimate.Uncertainty(
                    probability=_CONFIDENCE_LEVEL,
                    lowerbound=sample_roi[2] * _REVENUE_PER_KPI,
                    upperbound=sample_roi[3] * _REVENUE_PER_KPI,
                )
            ],
        ),
        cost_per_contribution=estimate_pb2.Estimate(
            value=sample_cpik[1] * _REVENUE_PER_KPI,
            uncertainties=[
                estimate_pb2.Estimate.Uncertainty(
                    probability=_CONFIDENCE_LEVEL,
                    lowerbound=sample_cpik[2] * _REVENUE_PER_KPI,
                    upperbound=sample_cpik[3] * _REVENUE_PER_KPI,
                )
            ],
        ),
    )

    if index != 2:
      effectiveness = kpi_outcome_pb2.Effectiveness(
          media_unit=constants.IMPRESSIONS,
          value=estimate_pb2.Estimate(
              value=sample_effectiveness[0] * _REVENUE_PER_KPI,
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=_CONFIDENCE_LEVEL,
                      lowerbound=sample_effectiveness[2] * _REVENUE_PER_KPI,
                      upperbound=sample_effectiveness[3] * _REVENUE_PER_KPI,
                  )
              ],
          ),
      )
      second_kpi_outcome.effectiveness.CopyFrom(effectiveness)

      mroi = estimate_pb2.Estimate(
          value=sample_mroi[0] * _REVENUE_PER_KPI,
          uncertainties=[
              estimate_pb2.Estimate.Uncertainty(
                  probability=_CONFIDENCE_LEVEL,
                  lowerbound=sample_mroi[2] * _REVENUE_PER_KPI,
                  upperbound=sample_mroi[3] * _REVENUE_PER_KPI,
              )
          ],
      )
      second_kpi_outcome.marginal_roi.CopyFrom(mroi)

    kpi_outcomes.append(second_kpi_outcome)

  channel_name = constants.ALL_CHANNELS if index == 2 else _ALL_CHANNELS[index]

  media_analysis = media_analysis_pb2.MediaAnalysis(
      channel_name=channel_name,
      media_outcomes=kpi_outcomes,
      spend_info=media_analysis_pb2.SpendInfo(
          spend=test_utils.SAMPLE_SPEND[index] * scalar_multiplier,
          spend_share=(
              test_utils.SAMPLE_PCT_OF_SPEND[index] * scalar_multiplier / 100
          ),
      ),
  )

  if index < 2:
    media_analysis.response_curve.CopyFrom(
        response_curve_pb2.ResponseCurve(
            input_name=constants.SPEND,
            response_points=expected_response_points,
        )
    )

  return media_analysis


def _create_media_analysis_inc_outcome(
    channel_name: str,
    fill_value: float,
    # Only used in the case where we expect a media analysis for non-revenue
    # data with revenue-type KPI (i.e. `revenue_per_kpi` is defined).
    create_derived_non_revenue_kpi_outcome: bool = True,
) -> media_analysis_pb2.MediaAnalysis:
  kpi_outcome_revenue = kpi_outcome_pb2.KpiOutcome(
      kpi_type=kpi_type_pb2.REVENUE,
      contribution=kpi_outcome_pb2.Contribution(
          value=estimate_pb2.Estimate(
              value=fill_value,
              uncertainties=[
                  estimate_pb2.Estimate.Uncertainty(
                      probability=constants.DEFAULT_CONFIDENCE_LEVEL,
                      lowerbound=fill_value,
                      upperbound=fill_value,
                  )
              ],
          ),
      ),
  )
  kpi_outcomes = [kpi_outcome_revenue]
  if create_derived_non_revenue_kpi_outcome:
    kpi_outcome_non_revenue = kpi_outcome_pb2.KpiOutcome(
        kpi_type=kpi_type_pb2.NON_REVENUE,
        contribution=kpi_outcome_pb2.Contribution(
            value=estimate_pb2.Estimate(
                value=fill_value,
                uncertainties=[
                    estimate_pb2.Estimate.Uncertainty(
                        probability=constants.DEFAULT_CONFIDENCE_LEVEL,
                        lowerbound=fill_value,
                        upperbound=fill_value,
                    )
                ],
            ),
        ),
    )
    kpi_outcomes.append(kpi_outcome_non_revenue)
  return media_analysis_pb2.MediaAnalysis(
      channel_name=channel_name,
      media_outcomes=kpi_outcomes,
  )


def _create_expected_baseline_analysis(
    is_revenue_type: bool,
) -> non_media_analysis_pb2.NonMediaAnalysis:
  non_media_outcome = kpi_outcome_pb2.KpiOutcome(
      contribution=_BASELINE_CONTRIBUTION,
      kpi_type=(
          kpi_type_pb2.REVENUE if is_revenue_type else kpi_type_pb2.NON_REVENUE
      ),
  )

  return non_media_analysis_pb2.NonMediaAnalysis(
      non_media_name=constants.BASELINE,
      non_media_outcomes=[non_media_outcome],
  )


class MarketingAnalysisSpecTest(parameterized.TestCase):

  def test_start_date_is_after_end_date(self):
    with self.assertRaisesRegex(
        ValueError,
        "Start date must be before end date.",
    ):
      marketing_processor.MarketingAnalysisSpec(
          end_date=_DATE_1,
          start_date=_DATE_2,
      )

  def test_confidence_level_is_below_zero(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be greater than 0 and less than 1.",
    ):
      marketing_processor.MarketingAnalysisSpec(
          start_date=_DATE_1,
          end_date=_DATE_2,
          confidence_level=-1.0,
      )

  def test_confidence_level_is_above_one(self):
    with self.assertRaisesRegex(
        ValueError,
        "Confidence level must be greater than 0 and less than 1.",
    ):
      marketing_processor.MarketingAnalysisSpec(
          start_date=_DATE_1,
          end_date=_DATE_2,
          confidence_level=10.0,
      )

  def test_validate_incremental_outcome_spec_with_new_data_no_time(self):
    new_data = analyzer.DataTensors(media=tf.ones((3, 4, 5)))
    with self.assertRaisesRegex(
        ValueError,
        "`time` must be provided in `new_data`.",
    ):
      marketing_processor.MarketingAnalysisSpec(
          incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(
              new_data=new_data
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="both_specs_provided",
          media_summary_spec=marketing_processor.MediaSummarySpec(),
          incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(),
          error_message=(
              "Only one of `media_summary_spec` or `incremental_outcome_spec`"
              " can be provided."
          ),
      ),
      dict(
          testcase_name="none_of_the_specs_provided",
          media_summary_spec=None,
          incremental_outcome_spec=None,
          error_message=(
              "At least one of `media_summary_spec` or"
              " `incremental_outcome_spec` must be provided."
          ),
      ),
  )
  def test_validate_marketing_analysis_spec(
      self,
      media_summary_spec: marketing_processor.MediaSummarySpec,
      incremental_outcome_spec: marketing_processor.IncrementalOutcomeSpec,
      error_message: str,
  ):
    with self.assertRaisesRegex(ValueError, error_message):
      marketing_processor.MarketingAnalysisSpec(
          media_summary_spec=media_summary_spec,
          incremental_outcome_spec=incremental_outcome_spec,
      )

  def test_media_summary_spec_validates_successfully(self):
    media_summary_spec = marketing_processor.MediaSummarySpec()
    response_curve_spec = marketing_processor.ResponseCurveSpec()
    spec = marketing_processor.MarketingAnalysisSpec(
        media_summary_spec=media_summary_spec,
        response_curve_spec=response_curve_spec,
        start_date=_DATE_1,
        end_date=_DATE_2,
    )

    self.assertTrue(spec.media_summary_spec.marginal_roi_by_reach)
    self.assertTrue(spec.response_curve_spec.by_reach)
    self.assertEqual(spec.start_date, _DATE_1)
    self.assertEqual(spec.end_date, _DATE_2)
    self.assertEqual(spec.confidence_level, _CONFIDENCE_LEVEL)

  def test_incremental_outcome_spec_validates_successfully(self):
    incremental_outcome_spec = marketing_processor.IncrementalOutcomeSpec()
    response_curve_spec = marketing_processor.ResponseCurveSpec()
    spec = marketing_processor.MarketingAnalysisSpec(
        incremental_outcome_spec=incremental_outcome_spec,
        response_curve_spec=response_curve_spec,
        start_date=_DATE_1,
        end_date=_DATE_2,
    )

    self.assertTrue(spec.incremental_outcome_spec.aggregate_times)
    self.assertIsNone(spec.incremental_outcome_spec.new_data)
    self.assertIsNone(spec.incremental_outcome_spec.media_selected_times)
    self.assertTrue(spec.response_curve_spec.by_reach)
    self.assertEqual(spec.start_date, _DATE_1)
    self.assertEqual(spec.end_date, _DATE_2)
    self.assertEqual(spec.confidence_level, _CONFIDENCE_LEVEL)


class MarketingProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    all_geos_dataarray = xr.DataArray(
        data=np.array(_ALL_GEOS),
        dims=[constants.GEO],
        coords={constants.GEO: ([constants.GEO], _ALL_GEOS)},
    )

    all_times_dataarray = xr.DataArray(
        data=np.array(_ALL_TIMES),
        dims=[constants.TIME],
        coords={constants.TIME: ([constants.TIME], _ALL_TIMES)},
    )

    self.mock_meridian_model = self.enter_context(
        mock.patch.object(model, "Meridian", autospec=True)
    )
    self.mock_meridian_model.input_data.geo = all_geos_dataarray
    self.mock_meridian_model.input_data.time = all_times_dataarray
    self.mock_meridian_model.all_channel_names = _ALL_CHANNELS

    self.mock_analyzer = self.enter_context(
        mock.patch.object(analyzer, "Analyzer", autospec=True)
    )

    self.mock_trained_model = self.enter_context(
        mock.patch.object(model_processor, "TrainedModel", autospec=True)
    )
    self.mock_trained_model.mmm = self.mock_meridian_model
    self.mock_trained_model.internal_analyzer = self.mock_analyzer
    self.mock_trained_model.time_coordinates = (
        time_coordinates.TimeCoordinates.from_dates(_ALL_TIMES)
    )

    self.mock_ensure_trained_model = self.enter_context(
        mock.patch.object(
            model_processor, "ensure_trained_model", autospec=True
        )
    )
    self.mock_ensure_trained_model.return_value = self.mock_trained_model

    self.mock_analyzer.baseline_summary_metrics.return_value = (
        _create_baseline_summary_metrics_data()
    )
    self.mock_analyzer.response_curves.return_value = (
        _create_response_curve_data()
    )

  def test_spec_type_returns_marketing_analysis_spec(self):
    self.assertEqual(
        marketing_processor.MarketingProcessor.spec_type(),
        marketing_processor.MarketingAnalysisSpec,
    )

  def test_output_type_returns_marketing_analysis_list_proto(self):
    self.assertEqual(
        marketing_processor.MarketingProcessor.output_type(),
        marketing_analysis_pb2.MarketingAnalysisList,
    )

  @parameterized.named_parameters(
      dict(testcase_name="without_tag", tag=""),
      dict(testcase_name="with_tag", tag="tag"),
  )
  def test_execute_with_one_spec_revenue_metrics_revenue_per_kpi_exists(
      self,
      tag: str,
  ):
    date_interval = date_interval_pb2.DateInterval()
    date_interval.CopyFrom(_DATE_INTERVAL_FIRST_HALF)
    date_interval.tag = tag
    spec = dataclasses.replace(_MEDIA_SUMMARY_SPEC_1, date_interval_tag=tag)

    self.mock_analyzer.summary_metrics.side_effect = [
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=True, spec_number=1
        ),
    ]
    self.mock_meridian_model.input_data.kpi_type = constants.REVENUE
    self.mock_meridian_model.input_data.revenue_per_kpi = _REV_REVENUE_PER_KPI
    self.mock_meridian_model.revenue_per_kpi = _REV_REVENUE_PER_KPI
    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([spec])

    expected_media_analysis_1 = _create_expected_media_analysis(
        index=0,
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=False,
    )
    expected_media_analysis_2 = _create_expected_media_analysis(
        index=1,
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=False,
    )
    expected_media_analysis_3 = _create_expected_media_analysis(
        index=2,  # "All Channels"
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=False,
    )
    expected_baseline_analysis = _create_expected_baseline_analysis(
        is_revenue_type=True
    )
    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=date_interval,
        media_analyses=[
            expected_media_analysis_1,
            expected_media_analysis_2,
            expected_media_analysis_3,
        ],
        non_media_analyses=[expected_baseline_analysis],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=3,
    )

  def test_optional_time_coordinates(self):
    marketing_analysis_spec = marketing_processor.MarketingAnalysisSpec(
        media_summary_spec=marketing_processor.MediaSummarySpec(),
        confidence_level=0.9,
    )
    self.mock_meridian_model.input_data.revenue_per_kpi = None
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.revenue_per_kpi = None
    self.mock_meridian_model.input_data.revenue_per_kpi = None
    self.mock_analyzer.summary_metrics.side_effect = [
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=True, spec_number=1
        ),
    ]
    self.mock_analyzer.baseline_summary_metrics.side_effect = [
        _create_baseline_summary_metrics_data()
    ]
    processor = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    )
    processor.execute([marketing_analysis_spec])

    self.mock_analyzer.summary_metrics.assert_called_once_with(
        marginal_roi_by_reach=True,
        selected_times=None,
        aggregate_geos=True,
        aggregate_times=True,
        confidence_level=0.9,
        include_non_paid_channels=False,
        use_kpi=True,
    )
    self.mock_analyzer.baseline_summary_metrics.assert_called_once_with(
        confidence_level=0.9,
        aggregate_times=True,
        selected_times=None,
    )

  def test_end_date_only_in_time_coordinates(self):
    marketing_analysis_spec = marketing_processor.MarketingAnalysisSpec(
        end_date=_DATE_2,
        confidence_level=0.9,
        media_summary_spec=marketing_processor.MediaSummarySpec(),
    )
    self.mock_meridian_model.input_data.revenue_per_kpi = None
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.revenue_per_kpi = None
    self.mock_meridian_model.input_data.revenue_per_kpi = None
    self.mock_analyzer.summary_metrics.side_effect = [
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=True, spec_number=1
        ),
    ]
    self.mock_analyzer.baseline_summary_metrics.side_effect = [
        _create_baseline_summary_metrics_data()
    ]
    processor = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    )
    processor.execute([marketing_analysis_spec])

    # Mutant coverage
    last_date_str = marketing_analysis_spec.end_date
    self.assertEqual(last_date_str, _DATE_2)
    self.assertNotEqual(last_date_str, _DATE_3)
    self.assertNotEqual(last_date_str, _DATE_1)

  def test_execute_with_one_spec_non_revenue_metrics_no_revenue_per_kpi(self):
    self.mock_analyzer.summary_metrics.return_value = (
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=False, spec_number=1
        )
    )
    self.mock_meridian_model.revenue_per_kpi = None
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.input_data.revenue_per_kpi = None

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([_MEDIA_SUMMARY_SPEC_1])

    expected_media_analysis_1 = _create_expected_media_analysis(
        index=0,
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_media_analysis_2 = _create_expected_media_analysis(
        index=1,
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_media_analysis_3 = _create_expected_media_analysis(
        index=2,
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_baseline_analysis = _create_expected_baseline_analysis(
        is_revenue_type=False
    )
    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_DATE_INTERVAL_FIRST_HALF,
        media_analyses=[
            expected_media_analysis_1,
            expected_media_analysis_2,
            expected_media_analysis_3,
        ],
        non_media_analyses=[expected_baseline_analysis],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

  def test_execute_with_one_spec_non_revenue_metrics_with_revenue_per_kpi(self):
    self.mock_analyzer.summary_metrics.side_effect = [
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=True, spec_number=1
        ),
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=False, spec_number=1
        ),
    ]
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.revenue_per_kpi = _NONREV_REVENUE_PER_KPI
    self.mock_meridian_model.input_data.revenue_per_kpi = (
        _NONREV_REVENUE_PER_KPI
    )

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([_MEDIA_SUMMARY_SPEC_1])

    expected_media_analysis_1 = _create_expected_media_analysis(
        index=0,
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=True,
    )
    expected_media_analysis_2 = _create_expected_media_analysis(
        index=1,
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=True,
    )
    expected_media_analysis_3 = _create_expected_media_analysis(
        index=2,
        scalar_multiplier=1,
        is_revenue_type=True,
        create_derived_non_revenue_kpi_outcome=True,
    )
    expected_baseline_analysis = _create_expected_baseline_analysis(
        is_revenue_type=True
    )
    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_DATE_INTERVAL_FIRST_HALF,
        media_analyses=[
            expected_media_analysis_1,
            expected_media_analysis_2,
            expected_media_analysis_3,
        ],
        non_media_analyses=[expected_baseline_analysis],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=3,
    )
    self.mock_analyzer.summary_metrics.assert_has_calls([
        mock.call(
            marginal_roi_by_reach=True,
            selected_times=_ALL_TIMES[:2],
            aggregate_geos=True,
            aggregate_times=True,
            confidence_level=0.9,
            use_kpi=False,
            include_non_paid_channels=False,
        ),
        mock.call(
            marginal_roi_by_reach=True,
            selected_times=_ALL_TIMES[:2],
            aggregate_geos=True,
            aggregate_times=True,
            confidence_level=0.9,
            use_kpi=True,
            include_non_paid_channels=False,
        ),
    ])

  def test_execute_with_multiple_specs_non_revenue_metrics_no_revenue_per_kpi(
      self,
  ):
    all_times_dataarray = xr.DataArray(
        data=np.array(_ALL_TIMES),
        dims=[constants.TIME],
        coords={constants.TIME: ([constants.TIME], _ALL_TIMES)},
    )
    self.mock_meridian_model.input_data.time = all_times_dataarray

    self.mock_analyzer.summary_metrics.side_effect = [
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=False, spec_number=1
        ),
        _create_media_summary_metrics_aggregated_data(
            is_revenue_type=False, spec_number=2
        ),
    ]
    self.mock_meridian_model.revenue_per_kpi = None
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.input_data.revenue_per_kpi = None

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([_MEDIA_SUMMARY_SPEC_1, _MEDIA_SUMMARY_SPEC_2])

    expected_media_analysis_1_spec_1 = _create_expected_media_analysis(
        index=0,
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_media_analysis_2_spec_1 = _create_expected_media_analysis(
        index=1,
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_media_analysis_3_spec_1 = _create_expected_media_analysis(
        index=2,  # "All Channels"
        scalar_multiplier=5,
        is_revenue_type=False,
    )
    expected_baseline_analysis = _create_expected_baseline_analysis(
        is_revenue_type=False
    )
    expected_marketing_analysis_spec_1 = (
        marketing_analysis_pb2.MarketingAnalysis(
            date_interval=_DATE_INTERVAL_FIRST_HALF,
            media_analyses=[
                expected_media_analysis_1_spec_1,
                expected_media_analysis_2_spec_1,
                expected_media_analysis_3_spec_1,
            ],
            non_media_analyses=[expected_baseline_analysis],
        )
    )
    expected_media_analysis_1_spec_2 = _create_expected_media_analysis(
        index=0,
        scalar_multiplier=10,
        is_revenue_type=False,
    )
    expected_media_analysis_2_spec_2 = _create_expected_media_analysis(
        index=1,
        scalar_multiplier=10,
        is_revenue_type=False,
    )
    expected_media_analysis_3_spec_2 = _create_expected_media_analysis(
        index=2,
        scalar_multiplier=10,
        is_revenue_type=False,
    )
    expected_marketing_analysis_spec_2 = (
        marketing_analysis_pb2.MarketingAnalysis(
            date_interval=_DATE_INTERVAL_SECOND_HALF,
            media_analyses=[
                expected_media_analysis_1_spec_2,
                expected_media_analysis_2_spec_2,
                expected_media_analysis_3_spec_2,
            ],
            non_media_analyses=[expected_baseline_analysis],
        )
    )

    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[
                expected_marketing_analysis_spec_1,
                expected_marketing_analysis_spec_2,
            ]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

  def test_execute_non_aggregated_results(self):
    self.mock_analyzer.summary_metrics.return_value = (
        _create_media_summary_metrics_data_with_time_breakdown()
    )
    self.mock_analyzer.baseline_summary_metrics.return_value = (
        _create_baseline_summary_metrics_data_with_time_breakdown()
    )
    self.mock_meridian_model.revenue_per_kpi = None
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.input_data.revenue_per_kpi = None

    with self.assertWarnsRegex(
        UserWarning,
        expected_regex=(
            "Response curves are not computed for non-aggregated time periods."
        ),
    ):
      marketing_analysis_list = marketing_processor.MarketingProcessor(
          trained_model=self.mock_trained_model,
      ).execute([_MEDIA_SUMMARY_SPEC_NON_AGGREGATED])

    expected_marketing_analysis_list = text_format.Parse(
        """
        marketing_analyses {
          date_interval {
            start_date { year: 2024 month: 1 day: 1 }
            end_date { year: 2024 month: 1 day: 8 }
          }
          media_analyses {
            channel_name: "channel_1"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          media_analyses {
            channel_name: "channel_2"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          media_analyses {
            channel_name: "All Channels"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          non_media_analyses {
            non_media_name: "baseline"
            non_media_outcomes {
              contribution {
                value {
                  value: 200
                  uncertainties {
                    probability: 0.9
                    lowerbound: 200
                    upperbound: 200
                  }
                }
                share {
                  value: 1
                  uncertainties {
                    probability: 0.9
                    lowerbound: 1
                    upperbound: 1
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
        }
        marketing_analyses {
          date_interval {
            start_date { year: 2024 month: 1 day: 8 }
            end_date { year: 2024 month: 1 day: 15 }
          }
          media_analyses {
            channel_name: "channel_1"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          media_analyses {
            channel_name: "channel_2"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          media_analyses {
            channel_name: "All Channels"
            spend_info { spend: 10 spend_share: 0.2 }
            media_outcomes {
              contribution {
                value {
                  value: 60
                  uncertainties {
                    probability: 0.9
                    lowerbound: 60
                    upperbound: 60
                  }
                }
                share {
                  value: 0.7
                  uncertainties {
                    probability: 0.9
                    lowerbound: 0.7
                    upperbound: 0.7
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
          non_media_analyses {
            non_media_name: "baseline"
            non_media_outcomes {
              contribution {
                value {
                  value: 200
                  uncertainties {
                    probability: 0.9
                    lowerbound: 200
                    upperbound: 200
                  }
                }
                share {
                  value: 1
                  uncertainties {
                    probability: 0.9
                    lowerbound: 1
                    upperbound: 1
                  }
                }
              }
              kpi_type: NON_REVENUE
            }
          }
        }
        """,
        marketing_analysis_pb2.MarketingAnalysisList(),
    )

    self.assertLen(marketing_analysis_list.marketing_analyses, 2)
    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

  def test_execute_incremental_outcome(self):
    n_chains = 2
    n_draws = 3
    n_channels = _NUMBER_OF_CHANNELS
    self.mock_analyzer.compute_incremental_outcome_aggregate.return_value = (
        tf.convert_to_tensor(np.ones((n_chains, n_draws, n_channels)))
    )
    all_times_dataarray = xr.DataArray(
        data=np.array(_ALL_TIMES),
        dims=[constants.TIME],
        coords={constants.TIME: ([constants.TIME], _ALL_TIMES)},
    )
    self.mock_meridian_model.input_data.time = all_times_dataarray
    self.mock_meridian_model.input_data.get_all_paid_channels.return_value = (
        _ALL_CHANNELS
    )

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([_INCREMENTAL_OUTCOME_SPEC])

    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_DATE_INTERVAL,
        media_analyses=[
            _create_media_analysis_inc_outcome(
                channel_name="channel_1", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="channel_2", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="All Channels", fill_value=1.0
            ),
        ],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )
    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
    )

  def test_execute_incremental_outcome_non_revenue_metrics_no_revenue_per_kpi(
      self,
  ):
    n_chains = 2
    n_draws = 3
    n_channels = _NUMBER_OF_CHANNELS
    self.mock_analyzer.compute_incremental_outcome_aggregate.return_value = (
        tf.convert_to_tensor(np.ones((n_chains, n_draws, n_channels)))
    )
    all_times_dataarray = xr.DataArray(
        data=np.array(_ALL_TIMES),
        dims=[constants.TIME],
        coords={constants.TIME: ([constants.TIME], _ALL_TIMES)},
    )
    self.mock_meridian_model.input_data.time = all_times_dataarray
    self.mock_meridian_model.input_data.get_all_paid_channels.return_value = (
        _ALL_CHANNELS
    )
    self.mock_meridian_model.input_data.kpi_type = constants.REVENUE
    self.mock_meridian_model.input_data.revenue_per_kpi = _REV_REVENUE_PER_KPI
    self.mock_meridian_model.revenue_per_kpi = _REV_REVENUE_PER_KPI

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([_INCREMENTAL_OUTCOME_SPEC])

    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_DATE_INTERVAL,
        media_analyses=[
            _create_media_analysis_inc_outcome(
                channel_name="channel_1",
                fill_value=1.0,
                create_derived_non_revenue_kpi_outcome=False,
            ),
            _create_media_analysis_inc_outcome(
                channel_name="channel_2",
                fill_value=1.0,
                create_derived_non_revenue_kpi_outcome=False,
            ),
            _create_media_analysis_inc_outcome(
                channel_name="All Channels",
                fill_value=1.0,
                create_derived_non_revenue_kpi_outcome=False,
            ),
        ],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )
    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
    )

  def test_execute_incremental_outcome_with_new_data(self):
    n_chains = 2
    n_draws = 3
    n_channels = _NUMBER_OF_CHANNELS
    self.mock_analyzer.compute_incremental_outcome_aggregate.return_value = (
        tf.convert_to_tensor(np.ones((n_chains, n_draws, n_channels)))
    )
    self.mock_meridian_model.input_data.get_all_paid_channels.return_value = (
        _ALL_CHANNELS
    )
    spec = marketing_processor.MarketingAnalysisSpec(
        incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(
            new_data=_NEW_DATA, aggregate_times=True
        )
    )

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([spec])

    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_NEW_DATE_INTERVAL,
        media_analyses=[
            _create_media_analysis_inc_outcome(
                channel_name="channel_1", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="channel_2", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="All Channels", fill_value=1.0
            ),
        ],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

  def test_execute_incremental_outcome_with_new_data_not_aggregated(self):
    n_chains = 2
    n_draws = 3
    n_times = 4
    n_channels = _NUMBER_OF_CHANNELS
    self.mock_analyzer.compute_incremental_outcome_aggregate.return_value = (
        tf.convert_to_tensor(np.ones((n_chains, n_draws, n_times, n_channels)))
    )
    self.mock_meridian_model.input_data.get_all_paid_channels.return_value = (
        _ALL_CHANNELS
    )
    spec = marketing_processor.MarketingAnalysisSpec(
        incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(
            new_data=_NEW_DATA, aggregate_times=False
        )
    )

    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([spec])

    expected_media_analyses = [
        _create_media_analysis_inc_outcome(
            channel_name="channel_1", fill_value=1.0
        ),
        _create_media_analysis_inc_outcome(
            channel_name="channel_2", fill_value=1.0
        ),
        _create_media_analysis_inc_outcome(
            channel_name="All Channels", fill_value=1.0
        ),
    ]
    expected_marketing_analysis_1 = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_NEW_DATE_INTERVAL_1,
        media_analyses=expected_media_analyses,
    )
    expected_marketing_analysis_2 = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_NEW_DATE_INTERVAL_2,
        media_analyses=expected_media_analyses,
    )
    expected_marketing_analysis_3 = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_NEW_DATE_INTERVAL_3,
        media_analyses=expected_media_analyses,
    )
    expected_marketing_analysis_4 = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=_NEW_DATE_INTERVAL_4,
        media_analyses=expected_media_analyses,
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[
                expected_marketing_analysis_1,
                expected_marketing_analysis_2,
                expected_marketing_analysis_3,
                expected_marketing_analysis_4,
            ]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

  def test_execute_incremental_outcome_with_new_data_with_selected_times(self):
    self.mock_meridian_model.input_data.kpi_type = constants.NON_REVENUE
    self.mock_meridian_model.revenue_per_kpi = _NONREV_REVENUE_PER_KPI
    self.mock_meridian_model.input_data.revenue_per_kpi = (
        _NONREV_REVENUE_PER_KPI
    )
    self.mock_analyzer.compute_incremental_outcome_aggregate.return_value = (
        tf.convert_to_tensor(np.ones((2, 3, _NUMBER_OF_CHANNELS)))
    )
    self.mock_meridian_model.input_data.get_all_paid_channels.return_value = (
        _ALL_CHANNELS
    )

    spec = marketing_processor.MarketingAnalysisSpec(
        start_date=datetime.date(2024, 2, 5),
        incremental_outcome_spec=marketing_processor.IncrementalOutcomeSpec(
            new_data=_NEW_DATA
        ),
    )
    marketing_analysis_list = marketing_processor.MarketingProcessor(
        trained_model=self.mock_trained_model,
    ).execute([spec])

    expected_date_interval = date_interval_pb2.DateInterval(
        start_date=date_pb2.Date(year=2024, month=2, day=5),
        end_date=date_pb2.Date(year=2024, month=2, day=26),
    )
    expected_marketing_analysis = marketing_analysis_pb2.MarketingAnalysis(
        date_interval=expected_date_interval,
        media_analyses=[
            _create_media_analysis_inc_outcome(
                channel_name="channel_1", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="channel_2", fill_value=1.0
            ),
            _create_media_analysis_inc_outcome(
                channel_name="All Channels", fill_value=1.0
            ),
        ],
    )
    expected_marketing_analysis_list = (
        marketing_analysis_pb2.MarketingAnalysisList(
            marketing_analyses=[expected_marketing_analysis]
        )
    )

    compare.assertProto2Equal(
        self,
        marketing_analysis_list,
        expected_marketing_analysis_list,
        precision=4,
    )

    self.mock_analyzer.compute_incremental_outcome_aggregate.assert_has_calls([
        mock.call(
            new_data=_NEW_DATA,
            media_selected_times=None,
            use_posterior=True,
            use_kpi=False,
            include_non_paid_channels=False,
            selected_geos=None,
            selected_times=[False, True, True, True],
            aggregate_geos=True,
            aggregate_times=True,
            batch_size=100,
        ),
        mock.call(
            new_data=_NEW_DATA,
            media_selected_times=None,
            use_posterior=True,
            use_kpi=True,
            include_non_paid_channels=False,
            selected_geos=None,
            selected_times=[False, True, True, True],
            aggregate_geos=True,
            aggregate_times=True,
            batch_size=100,
        ),
    ])


if __name__ == "__main__":
  absltest.main()
