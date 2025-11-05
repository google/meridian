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

"""Shared test data."""

from collections.abc import Iterator, Sequence

from meridian import constants as c
from mmm.v1.common import date_interval_pb2 as date_interval_pb
from mmm.v1.common import estimate_pb2 as estimate_pb
from mmm.v1.common import kpi_type_pb2 as kpi_type_pb
from mmm.v1.common import target_metric_pb2 as target_metric_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing import marketing_data_pb2 as marketing_data_pb
from mmm.v1.marketing.analysis import marketing_analysis_pb2 as marketing_pb
from mmm.v1.marketing.analysis import media_analysis_pb2 as media_pb
from mmm.v1.marketing.analysis import non_media_analysis_pb2 as non_media_pb
from mmm.v1.marketing.analysis import outcome_pb2 as outcome_pb
from mmm.v1.marketing.analysis import response_curve_pb2 as response_curve_pb
from mmm.v1.marketing.optimization import budget_optimization_pb2 as budget_pb
from mmm.v1.marketing.optimization import constraints_pb2 as constraints_pb
from mmm.v1.marketing.optimization import reach_frequency_optimization_pb2 as rf_pb
from scenarioplanner.converters.dataframe import constants as cc

from google.type import date_pb2 as date_pb


DATES = [
    date_pb.Date(year=2024, month=1, day=1),
    date_pb.Date(year=2024, month=1, day=8),
    date_pb.Date(year=2024, month=1, day=15),
]
DATE_INTERVALS = [
    date_interval_pb.DateInterval(
        start_date=DATES[0],
        end_date=DATES[1],
        tag="Week1",
    ),
    date_interval_pb.DateInterval(
        start_date=DATES[1],
        end_date=DATES[2],
        tag="Week2",
    ),
]
ALL_DATE_INTERVAL = date_interval_pb.DateInterval(
    start_date=DATES[0],
    end_date=DATES[2],
    tag=cc.ANALYSIS_TAG_ALL,
)


GEO_INFOS = [
    marketing_data_pb.GeoInfo(
        geo_id="geo-1",
        population=100,
    ),
    marketing_data_pb.GeoInfo(
        geo_id="geo-2",
        population=200,
    ),
]


MEDIA_CHANNELS = [
    "Channel 1",
    "Channel 2",
]
RF_CHANNELS = [
    "RF Channel 1",
    "RF Channel 2",
]


BASE_MEDIA_SPEND = 100.0
BASE_RF_MEDIA_SPEND = 110.0


def _create_marketing_data(
    create_rf_data: bool = True,
) -> Iterator[marketing_data_pb.MarketingDataPoint]:
  """Generator for default `MarketingDataPoint`s for each geo and date interval defined above."""
  for geo_info in GEO_INFOS:
    for date_interval in DATE_INTERVALS:
      media_vars = []
      rf_vars = []
      for channel in MEDIA_CHANNELS:
        media_var = marketing_data_pb.MediaVariable(
            channel_name=channel,
            # For simplicity, set all media spend to be the same across all
            # channels and across all geo and time dimensions.
            # Add function parameters if more sophisticated test data
            # generator is warranted here.
            media_spend=BASE_MEDIA_SPEND,
        )
        media_vars.append(media_var)
      if create_rf_data:
        for channel in RF_CHANNELS:
          rf_media_var = marketing_data_pb.ReachFrequencyVariable(
              channel_name=channel,
              spend=BASE_RF_MEDIA_SPEND,
              reach=10_000,
              average_frequency=1.1,
          )
          rf_vars.append(rf_media_var)
      yield marketing_data_pb.MarketingDataPoint(
          date_interval=date_interval,
          geo_info=geo_info,
          media_variables=media_vars,
          reach_frequency_variables=rf_vars,
          # `kpi` and `control_variables` fields are not set, since no test
          # needs it just yet. Fill them in when needed.
      )


MARKETING_DATA = marketing_data_pb.MarketingData(
    marketing_data_points=list(_create_marketing_data()),
)


PERFORMANCE_TEST = fit_pb.Performance(
    r_squared=0.99,
    mape=67.7,
    weighted_mape=59.8,
    rmse=55.05,
)
PERFORMANCE_TRAIN = fit_pb.Performance(
    r_squared=0.91,
    mape=60.6,
    weighted_mape=55.5,
    rmse=59.87,
)
PERFORMANCE_ALL_DATA = fit_pb.Performance(
    r_squared=0.94,
    mape=60.0,
    weighted_mape=55.4,
    rmse=52.0,
)


def _create_model_fit_result(
    name: str,
    performance: fit_pb.Performance,
) -> fit_pb.Result:
  return fit_pb.Result(
      name=name,
      performance=performance,
      predictions=[
          fit_pb.Prediction(
              date_interval=DATE_INTERVALS[0],
              predicted_outcome=estimate_pb.Estimate(
                  value=100.0,
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          probability=0.9,
                          lowerbound=90.0,
                          upperbound=110.0,
                      )
                  ],
              ),
              predicted_baseline=estimate_pb.Estimate(
                  value=90.0,
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          probability=0.9,
                          lowerbound=89.0,
                          upperbound=111.0,
                      )
                  ],
              ),
              actual_value=105.0,
          ),
          fit_pb.Prediction(
              date_interval=DATE_INTERVALS[1],
              predicted_outcome=estimate_pb.Estimate(
                  value=110.0,
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          probability=0.9,
                          lowerbound=100.0,
                          upperbound=120.0,
                      )
                  ],
              ),
              predicted_baseline=estimate_pb.Estimate(
                  value=109.0,
                  uncertainties=[
                      estimate_pb.Estimate.Uncertainty(
                          probability=0.9,
                          lowerbound=90.0,
                          upperbound=125.0,
                      )
                  ],
              ),
              actual_value=115.0,
          ),
      ],
  )


MODEL_FIT_RESULT_TEST = _create_model_fit_result(
    name=c.TEST,
    performance=PERFORMANCE_TEST,
)
MODEL_FIT_RESULT_TRAIN = _create_model_fit_result(
    name=c.TRAIN,
    performance=PERFORMANCE_TRAIN,
)
MODEL_FIT_RESULT_ALL_DATA = _create_model_fit_result(
    name=c.ALL_DATA,
    performance=PERFORMANCE_ALL_DATA,
)


def create_outcome(
    incremental_outcome: float,
    pct_of_contribution: float,
    effectiveness: float,
    roi: float,
    mroi: float,
    cpik: float,
    is_revenue_type: bool,
) -> outcome_pb.Outcome:
  return outcome_pb.Outcome(
      kpi_type=(
          kpi_type_pb.REVENUE if is_revenue_type else kpi_type_pb.NON_REVENUE
      ),
      contribution=outcome_pb.Contribution(
          value=estimate_pb.Estimate(value=incremental_outcome),
          share=estimate_pb.Estimate(value=pct_of_contribution),
      ),
      effectiveness=outcome_pb.Effectiveness(
          media_unit=c.IMPRESSIONS,
          value=estimate_pb.Estimate(value=effectiveness),
      ),
      roi=estimate_pb.Estimate(
          value=roi,
          uncertainties=[
              estimate_pb.Estimate.Uncertainty(
                  probability=0.9,
                  lowerbound=roi * 0.9,
                  upperbound=roi * 1.1,
              )
          ],
      ),
      marginal_roi=estimate_pb.Estimate(value=mroi),
      cost_per_contribution=estimate_pb.Estimate(
          value=cpik,
          uncertainties=[
              estimate_pb.Estimate.Uncertainty(
                  probability=0.9,
                  lowerbound=cpik * 0.9,
                  upperbound=cpik * 1.1,
              )
          ],
      ),
  )


REVENUE_OUTCOME = create_outcome(
    incremental_outcome=100.0,
    pct_of_contribution=0.1,
    effectiveness=3.3,
    roi=1.0,
    mroi=10.0,
    cpik=5.0,
    is_revenue_type=True,
)

NON_REVENUE_OUTCOME = create_outcome(
    incremental_outcome=100.0,
    pct_of_contribution=0.1,
    effectiveness=4.4,
    roi=10.0,
    mroi=100.0,
    cpik=100.0,
    is_revenue_type=False,
)


SPENDS = {
    MEDIA_CHANNELS[0]: 75_000,
    MEDIA_CHANNELS[1]: 25_000,
    RF_CHANNELS[0]: 30_000,
    RF_CHANNELS[1]: 20_000,
}
TOTAL_SPEND = sum(SPENDS.values())
SPENDS[c.ALL_CHANNELS] = TOTAL_SPEND


def create_media_analysis(
    channel: str,
    multiplier: float = 1.0,
    make_revenue_outcome: bool = True,
    make_non_revenue_outcome: bool = True,
) -> media_pb.MediaAnalysis:
  """Creates a `MediaAnalysis` test proto."""
  # `multiplier` is used to create unique metric numbers for the given channel
  # from the base template metrics above.
  outcomes = []
  if make_revenue_outcome:
    outcomes.append(
        create_outcome(
            incremental_outcome=100.0 * multiplier,
            pct_of_contribution=0.1 * multiplier,
            effectiveness=2.2 * multiplier,
            roi=1.0 * multiplier,
            mroi=10.0 * multiplier,
            cpik=5.0 * multiplier,
            is_revenue_type=True,
        )
    )
  if make_non_revenue_outcome:
    outcomes.append(
        create_outcome(
            incremental_outcome=100.0 * multiplier,
            pct_of_contribution=0.1 * multiplier,
            effectiveness=5.5 * multiplier,
            roi=10.0 * multiplier,
            mroi=100.0 * multiplier,
            cpik=100.0 * multiplier,
            is_revenue_type=False,
        )
    )

  response_curve = response_curve_pb.ResponseCurve(
      input_name="Spend",
      response_points=[
          response_curve_pb.ResponsePoint(
              input_value=1 * multiplier,
              incremental_kpi=100.0 * multiplier,
          ),
          response_curve_pb.ResponsePoint(
              input_value=2 * multiplier,
              incremental_kpi=200.0 * multiplier,
          ),
      ],
  )
  return media_pb.MediaAnalysis(
      channel_name=channel,
      spend_info=media_pb.SpendInfo(
          spend=SPENDS[channel],
          spend_share=SPENDS[channel] / TOTAL_SPEND,
      ),
      media_outcomes=outcomes,
      response_curve=response_curve,
  )


MEDIA_ANALYSES_BOTH_OUTCOMES = [
    create_media_analysis(channel, (idx + 1))
    for (idx, channel) in enumerate(MEDIA_CHANNELS)
]
RF_ANALYSES_BOTH_OUTCOMES = [
    create_media_analysis(channel, (idx + 1))
    for (idx, channel) in enumerate(RF_CHANNELS)
]
MEDIA_ANALYSES_NONREVENUE = [
    create_media_analysis(
        channel,
        # use a different multiplier value to distinquish from the above
        (idx + 1.2),
        make_revenue_outcome=False,
    )
    for (idx, channel) in enumerate(MEDIA_CHANNELS)
]
RF_ANALYSES_NONREVENUE = [
    create_media_analysis(
        channel,
        # use a different multiplier value to distinquish from the above
        (idx + 1.2),
        make_revenue_outcome=False,
    )
    for (idx, channel) in enumerate(RF_CHANNELS)
]

ALL_CHANNELS_ANALYSIS_BOTH_OUTCOMES = create_media_analysis(
    c.ALL_CHANNELS, multiplier=10
)
ALL_CHANNELS_ANALYSIS_NONREVENUE = create_media_analysis(
    c.ALL_CHANNELS, multiplier=12, make_revenue_outcome=False
)

BASELINE_NONREVENUE_OUTCOME = create_outcome(
    incremental_outcome=40.0,
    pct_of_contribution=0.04,
    effectiveness=4.4,
    cpik=75.0,
    roi=7.0,
    mroi=70.0,
    is_revenue_type=False,
)
BASELINE_ANALYSIS_NONREVENUE = non_media_pb.NonMediaAnalysis(
    non_media_name=c.BASELINE,
    non_media_outcomes=[BASELINE_NONREVENUE_OUTCOME],
)

BASELINE_REVENUE_OUTCOME = create_outcome(
    incremental_outcome=50.0,
    pct_of_contribution=0.05,
    effectiveness=5.5,
    roi=1.0,
    mroi=10.0,
    cpik=0.5,
    is_revenue_type=True,
)
BASELINE_ANALYSIS_REVENUE = non_media_pb.NonMediaAnalysis(
    non_media_name=c.BASELINE,
    non_media_outcomes=[BASELINE_REVENUE_OUTCOME],
)

BASELINE_ANALYSIS_BOTH_OUTCOMES = non_media_pb.NonMediaAnalysis(
    non_media_name=c.BASELINE,
    non_media_outcomes=[BASELINE_NONREVENUE_OUTCOME, BASELINE_REVENUE_OUTCOME],
)


def create_marketing_analysis(
    date_interval: date_interval_pb.DateInterval,
    baseline_analysis: non_media_pb.NonMediaAnalysis = BASELINE_ANALYSIS_BOTH_OUTCOMES,
    explicit_channel_analyses: Sequence[media_pb.MediaAnalysis] | None = None,
    explicit_all_channels_analysis: media_pb.MediaAnalysis | None = None,
) -> marketing_pb.MarketingAnalysis:
  """Create a `MarketingAnalysis` for the given analysis period and tag."""
  media_analyses = (
      list(explicit_channel_analyses)
      if explicit_channel_analyses
      else (MEDIA_ANALYSES_BOTH_OUTCOMES + RF_ANALYSES_BOTH_OUTCOMES)
  )
  media_analyses.append(
      explicit_all_channels_analysis
      if explicit_all_channels_analysis
      else ALL_CHANNELS_ANALYSIS_BOTH_OUTCOMES
  )

  return marketing_pb.MarketingAnalysis(
      date_interval=date_interval,
      non_media_analyses=[baseline_analysis],
      media_analyses=media_analyses,
  )


# All of the below test analyses data contain both media and R&F channels.

ALL_TAG_MARKETING_ANALYSIS_BOTH_OUTCOMES = create_marketing_analysis(
    date_interval=ALL_DATE_INTERVAL,
    baseline_analysis=BASELINE_ANALYSIS_BOTH_OUTCOMES,
)
ALL_TAG_MARKETING_ANALYSIS_NONREVENUE = create_marketing_analysis(
    date_interval=ALL_DATE_INTERVAL,
    baseline_analysis=BASELINE_ANALYSIS_NONREVENUE,
    explicit_channel_analyses=(
        MEDIA_ANALYSES_NONREVENUE + RF_ANALYSES_NONREVENUE
    ),
)

DATED_MARKETING_ANALYSES_BOTH_OUTCOMES = [
    create_marketing_analysis(
        date_interval=date_interval,
        baseline_analysis=BASELINE_ANALYSIS_BOTH_OUTCOMES,
    )
    for date_interval in DATE_INTERVALS
]
DATED_MARKETING_ANALYSES_NONREVENUE = [
    create_marketing_analysis(
        date_interval=date_interval,
        baseline_analysis=BASELINE_ANALYSIS_NONREVENUE,
        explicit_channel_analyses=(
            MEDIA_ANALYSES_NONREVENUE + RF_ANALYSES_NONREVENUE
        ),
    )
    for date_interval in DATE_INTERVALS
]

MARKETING_ANALYSIS_LIST_BOTH_OUTCOMES = marketing_pb.MarketingAnalysisList(
    marketing_analyses=(
        [ALL_TAG_MARKETING_ANALYSIS_BOTH_OUTCOMES]
        + DATED_MARKETING_ANALYSES_BOTH_OUTCOMES
    ),
)

MARKETING_ANALYSIS_LIST_NONREVENUE = marketing_pb.MarketingAnalysisList(
    marketing_analyses=(
        [ALL_TAG_MARKETING_ANALYSIS_NONREVENUE]
        + DATED_MARKETING_ANALYSES_NONREVENUE
    ),
)


# Incremental outcome grids (budget) are only relevant for non-RF media
# channels.

INCREMENTAL_OUTCOME_GRID_FOO = budget_pb.IncrementalOutcomeGrid(
    name="incremental outcome grid foo",
    channel_cells=[
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name=MEDIA_CHANNELS[0],
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=10000.0,
                    incremental_outcome=estimate_pb.Estimate(value=100.0),
                ),
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=20000.0,
                    incremental_outcome=estimate_pb.Estimate(value=200.0),
                ),
            ],
        ),
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name=MEDIA_CHANNELS[1],
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=10000.0,
                    incremental_outcome=estimate_pb.Estimate(value=100.0),
                ),
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=20000.0,
                    incremental_outcome=estimate_pb.Estimate(value=200.0),
                ),
            ],
        ),
    ],
)

INCREMENTAL_OUTCOME_GRID_BAR = budget_pb.IncrementalOutcomeGrid(
    name="incremental outcome grid bar",
    channel_cells=[
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name=MEDIA_CHANNELS[0],
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=1000.0,
                    incremental_outcome=estimate_pb.Estimate(value=10.0),
                ),
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=2000.0,
                    incremental_outcome=estimate_pb.Estimate(value=20.0),
                ),
            ],
        ),
        budget_pb.IncrementalOutcomeGrid.ChannelCells(
            channel_name=MEDIA_CHANNELS[1],
            cells=[
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=1000.0,
                    incremental_outcome=estimate_pb.Estimate(value=10.0),
                ),
                budget_pb.IncrementalOutcomeGrid.Cell(
                    spend=2000.0,
                    incremental_outcome=estimate_pb.Estimate(value=20.0),
                ),
            ],
        ),
    ],
)

# A fixed budget scenario for the entire time interval in the test data above.
BUDGET_OPTIMIZATION_SPEC_FIXED_ALL_DATES = budget_pb.BudgetOptimizationSpec(
    date_interval=ALL_DATE_INTERVAL,
    objective=target_metric_pb.TargetMetric.ROI,
    fixed_budget_scenario=budget_pb.FixedBudgetScenario(total_budget=100000.0),
    # No individual channel constraints. Expect implicit constraints: max budget
    # applied for each channel.
)
BUDGET_OPTIMIZATION_RESULT_FIXED_BOTH_OUTCOMES = (
    budget_pb.BudgetOptimizationResult(
        name="budget optimization result foo",
        group_id="group-foo",
        optimized_marketing_analysis=ALL_TAG_MARKETING_ANALYSIS_BOTH_OUTCOMES,
        spec=BUDGET_OPTIMIZATION_SPEC_FIXED_ALL_DATES,
        incremental_outcome_grid=INCREMENTAL_OUTCOME_GRID_FOO,
    )
)

# A flexible budget scenario for the second time interval only.
BUDGET_OPTIMIZATION_SPEC_FLEX_SELECT_DATES = budget_pb.BudgetOptimizationSpec(
    date_interval=DATE_INTERVALS[1],
    objective=target_metric_pb.TargetMetric.KPI,
    flexible_budget_scenario=budget_pb.FlexibleBudgetScenario(
        total_budget_constraint=constraints_pb.BudgetConstraint(
            min_budget=1000.0,
            max_budget=2000.0,
        ),
        target_metric_constraints=[
            constraints_pb.TargetMetricConstraint(
                target_metric=target_metric_pb.COST_PER_INCREMENTAL_KPI,
                target_value=10.0,
            )
        ],
    ),
    # Define explicit channel constraints.
    channel_constraints=[
        budget_pb.ChannelConstraint(
            channel_name=MEDIA_CHANNELS[0],
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=1100.0,
                max_budget=1500.0,
            ),
        ),
        budget_pb.ChannelConstraint(
            channel_name=MEDIA_CHANNELS[1],
            budget_constraint=constraints_pb.BudgetConstraint(
                min_budget=1000.0,
                max_budget=1800.0,
            ),
        ),
    ],
)
BUDGET_OPTIMIZATION_RESULT_FLEX_NONREV = budget_pb.BudgetOptimizationResult(
    name="budget optimization result bar",
    group_id="group-bar",
    optimized_marketing_analysis=ALL_TAG_MARKETING_ANALYSIS_NONREVENUE,
    spec=BUDGET_OPTIMIZATION_SPEC_FLEX_SELECT_DATES,
    incremental_outcome_grid=INCREMENTAL_OUTCOME_GRID_BAR,
)


# Frequency outcome grids are only relevant for R&F media channels.

FREQUENCY_OUTCOME_GRID_FOO = rf_pb.FrequencyOutcomeGrid(
    name="frequency outcome grid foo",
    channel_cells=[
        rf_pb.FrequencyOutcomeGrid.ChannelCells(
            channel_name=RF_CHANNELS[0],
            cells=[
                rf_pb.FrequencyOutcomeGrid.Cell(
                    reach_frequency=marketing_data_pb.ReachFrequency(
                        reach=10000,
                        average_frequency=1.0,
                    ),
                    outcome=estimate_pb.Estimate(value=100.0),
                ),
                rf_pb.FrequencyOutcomeGrid.Cell(
                    reach_frequency=marketing_data_pb.ReachFrequency(
                        reach=20000,
                        average_frequency=2.0,
                    ),
                    outcome=estimate_pb.Estimate(value=200.0),
                ),
            ],
        ),
        rf_pb.FrequencyOutcomeGrid.ChannelCells(
            channel_name=RF_CHANNELS[1],
            cells=[
                rf_pb.FrequencyOutcomeGrid.Cell(
                    reach_frequency=marketing_data_pb.ReachFrequency(
                        reach=10000,
                        average_frequency=1.0,
                    ),
                    outcome=estimate_pb.Estimate(value=100.0),
                ),
                rf_pb.FrequencyOutcomeGrid.Cell(
                    reach_frequency=marketing_data_pb.ReachFrequency(
                        reach=20000,
                        average_frequency=2.0,
                    ),
                    outcome=estimate_pb.Estimate(value=200.0),
                ),
            ],
        ),
    ],
)

RF_OPTIMIZATION_SPEC_ALL_DATES = rf_pb.ReachFrequencyOptimizationSpec(
    date_interval=ALL_DATE_INTERVAL,
    objective=target_metric_pb.TargetMetric.KPI,
    total_budget_constraint=constraints_pb.BudgetConstraint(
        min_budget=100000.0,
        max_budget=200000.0,
    ),
    rf_channel_constraints=[
        rf_pb.RfChannelConstraint(
            channel_name=RF_CHANNELS[0],
            frequency_constraint=constraints_pb.FrequencyConstraint(
                max_frequency=5.0,
            ),
        ),
        rf_pb.RfChannelConstraint(
            channel_name=RF_CHANNELS[1],
            frequency_constraint=constraints_pb.FrequencyConstraint(
                min_frequency=1.3,
                max_frequency=6.6,
            ),
        ),
    ],
)

RF_OPTIMIZATION_RESULT_FOO = rf_pb.ReachFrequencyOptimizationResult(
    name="reach frequency optimization result foo",
    group_id="group-foo",
    spec=RF_OPTIMIZATION_SPEC_ALL_DATES,
    optimized_channel_frequencies=[
        rf_pb.OptimizedChannelFrequency(
            channel_name=RF_CHANNELS[0],
            optimal_average_frequency=3.3,
        ),
        rf_pb.OptimizedChannelFrequency(
            channel_name=RF_CHANNELS[1],
            optimal_average_frequency=5.6,
        ),
    ],
    optimized_marketing_analysis=ALL_TAG_MARKETING_ANALYSIS_BOTH_OUTCOMES,
    frequency_outcome_grid=FREQUENCY_OUTCOME_GRID_FOO,
)
