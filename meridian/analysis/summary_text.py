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

"""Defines text string constants used in the model outputs."""


# Model results text.
MODEL_RESULTS_TITLE = 'Marketing Mix Modeling Report'

MODEL_FIT_CARD_ID = 'model-fit'
MODEL_FIT_CARD_TITLE = 'Model fit'
MODEL_FIT_INSIGHTS_FORMAT = """Model fit is a measure of how well your MMM fits
your current data used to train the model."""

EXPECTED_ACTUAL_IMPACT_CHART_ID = 'expected-actual-impact-chart'
EXPECTED_ACTUAL_IMPACT_CHART_TITLE = 'Expected {impact} vs. actual {impact}'
EXPECTED_ACTUAL_IMPACT_CHART_DESCRIPTION_FORMAT = """Note: The baseline represents the
expected {impact} without any media execution. The shaded blue area represents
the 90% credible interval."""

PREDICTIVE_ACCURACY_TABLE_ID = 'model-fit-statistics-table-chart'
PREDICTIVE_ACCURACY_TABLE_TITLE = 'Model fit metrics'
PREDICTIVE_ACCURACY_TABLE_DESCRIPTION = """Note: R-squared measures the amount
of variation in the data that is explained by the model. The closer it is to 1,
the better the model fit. MAPE measures the mean absolute percentage difference
between the expected and the actual. The closer it is to 0, the better the
model fit. wMAPE is MAPE weighted by the actual {impact}."""

CHANNEL_CONTRIB_CARD_ID = 'channel-contrib'
CHANNEL_CONTRIB_CARD_TITLE = 'Channel contribution'
CHANNEL_CONTRIB_INSIGHTS_FORMAT = """Your channel contributions help you
understand what drove your {impact}. {lead_channels} drove the most overall
{impact}."""

CHANNEL_DRIVERS_CHART_ID = 'channel-drivers-chart'
CHANNEL_DRIVERS_CHART_TITLE = 'Contribution by baseline and marketing channels'
CHANNEL_DRIVERS_CHART_DESCRIPTION = """Note: This graphic encompasses all of
your {impact} drivers, but breaks down your marketing {impact} by the baseline
and all channels."""

SPEND_IMPACT_CHART_ID = 'spend-impact-chart'
SPEND_IMPACT_CHART_TITLE = (
    'Spend and {impact} contribution by marketing channel'
)
SPEND_IMPACT_CHART_DESCRIPTION = """Note: Return on investment is calculated by
dividing the {impact} attributed to a channel by marketing costs."""

IMPACT_CONTRIBUTION_CHART_ID = 'impact-contribution-chart'
CONTRIBUTION_CHART_TITLE = 'Contribution by baseline and marketing channels'
IMPACT_CONTRIBUTION_CHART_DESCRIPTION = """Note: This is a percentage breakdown
of all your {impact}."""

PERFORMANCE_BREAKDOWN_CARD_ID = 'performance-breakdown'
PERFORMANCE_BREAKDOWN_CARD_TITLE = 'Return on investment'
PERFORMANCE_BREAKDOWN_INSIGHTS_FORMAT = """Your return on investment (ROI) helps
you understand how your marketing activities impacted your business objectives.
{lead_roi_channel} drove the highest ROI at {lead_roi_ratio:.1f}. For every $1
you spent on {lead_roi_channel}, you saw ${lead_roi_ratio:.2f} in revenue.
{lead_effectiveness_channel} had the highest effectiveness, which is your
incremental revenue per media unit. {lead_mroi_channel} had the highest marginal
ROI at {lead_mroi_channel_value:.2f}. {lead_cpik_channel} drove the lowest CPIK
at ${lead_cpik_ratio:.2f}. For every KPI unit, you spent ${lead_cpik_ratio:.2f}.
"""

ROI_EFFECTIVENESS_CHART_ID = 'roi-effectiveness-chart'
ROI_EFFECTIVENESS_CHART_TITLE = 'ROI vs. effectiveness'
ROI_EFFECTIVENESS_CHART_DESCRIPTION = """Note: Effectiveness measures the
incremental revenue generated per impression. A low ROI does not necessarily
imply low media effectiveness; it may result from high media cost, as positioned
in the upper-left corner of the chart. Conversely, a high ROI can coexist with
low media effectiveness and low media costs, as indicated in the bottom-right
corner of the chart. The diagonal section of the chart suggests that the ROI is
primarily influenced by media effectiveness. The size of the bubbles represents
the scale of the media spend."""

ROI_MARGINAL_CHART_ID = 'roi-marginal-chart'
ROI_MARGINAL_CHART_TITLE = 'ROI vs. marginal ROI'
ROI_MARGINAL_CHART_DESCRIPTION = """Note: Marginal ROI measures the additional
return generated for every additional dollar spent. It's an indicator of
efficiency of additional spend. Channels with a high ROI but a low marginal ROI
are likely in the saturation phase, where the initial investments have paid off,
but additional investment does not bring in as much return. Conversely, channels
that have a high ROI and a high marginal ROI perform well and continue to yield
high returns with additional spending. The size of the bubbles represents the
scale of the media spend."""

ROI_CHANNEL_CHART_ID = 'roi-channel-chart'
ROI_CHANNEL_CHART_TITLE_FORMAT = 'ROI by channel {ci}'

CPIK_CHANNEL_CHART_ID = 'cpik-channel-chart'
CPIK_CHANNEL_CHART_TITLE_FORMAT = 'CPIK by channel {ci}'
CPIK_CHANNEL_CHART_DESCRIPTION = """Note: CPIK (cost per incremental KPI) point
estimate is determined by the posterior median, whereas ROI point estimate is
determined by the posterior mean."""

RESPONSE_CURVES_CARD_ID = 'response-curves'
RESPONSE_CURVES_CARD_TITLE = 'Response curves'
RESPONSE_CURVES_INSIGHTS_FORMAT = """Your response curves depict the
relationship between marketing spend and the resulting incremental {impact}."""
OPTIMAL_FREQUENCY_INSIGHTS_FORMAT = """Your optimal weekly frequency for
{rf_channel} is {opt_freq} to maximize ROI."""

RESPONSE_CURVES_CHART_ID = 'response-curves-chart'
RESPONSE_CURVES_CHART_TITLE = (
    'Response curves by marketing channel {top_channels}'
)
RESPONSE_CURVES_CHART_DESCRIPTION_FORMAT = """Note: The response curves are
constructed based on the historical flighting pattern and present the cumulative
incremental {impact} from the total media spend over the selected time
period."""

OPTIMAL_FREQUENCY_CHART_ID = 'optimal-frequency-chart'
OPTIMAL_FREQUENCY_CHART_TITLE = '{metric} by weekly average frequency'
OPTIMAL_FREQ_CHART_DESCRIPTION = """Note: Optimal frequency is the recommended
average weekly impressions per user (# impressions / # reached users) that
maximizes ROI. When multiple channels have reach and frequency data, only the
channel with the highest spend will be displayed. The same chart can be viewed
for all other channels as described in "Optimize frequency" in the User
Guide."""


# Budget optimization texts.
OPTIMIZATION_TITLE = 'MMM Optimization Report'

SCENARIO_PLAN_CARD_ID = 'scenario-plan'
SCENARIO_PLAN_CARD_TITLE = 'Optimization scenario'
SCENARIO_PLAN_INSIGHTS_FORMAT = """These are the estimated results from a
{scenario_type} budget scenario with a channel-level spend constraint of
-{lower_bound}% to +{upper_bound}% of the current spend over the time period
from {start_date} to {end_date}."""
SCENARIO_PLAN_BASE_INSIGHTS_FORMAT = """These are the estimated
results from a {scenario_type} budget scenario based on channel-level spend
constraints over the time period from {start_date} to {end_date}."""


CURRENT_BUDGET_LABEL = 'Current budget'
OPTIMIZED_BUDGET_LABEL = 'Optimized budget'
FIXED_BUDGET_LABEL = 'Fixed'
FLEXIBLE_BUDGET_LABEL = 'Flexible'
CURRENT_ROI_LABEL = 'Current ROI'
OPTIMIZED_ROI_LABEL = 'Optimized ROI'
CURRENT_CPIK_LABEL = 'Current CPIK'
OPTIMIZED_CPIK_LABEL = 'Optimized CPIK'
CURRENT_INC_IMPACT_LABEL = 'Current incremental {impact}'
OPTIMIZED_INC_IMPACT_LABEL = 'Optimized incremental {impact}'

BUDGET_ALLOCATION_CARD_ID = 'budget-allocation'
BUDGET_ALLOCATION_CARD_TITLE = 'Recommended budget allocation'
BUDGET_ALLOCATION_INSIGHTS = """You can see the channel-level budget shifts and
the performance lift after optimization."""

SPEND_DELTA_CHART_ID = 'spend-delta-chart'
SPEND_DELTA_CHART_TITLE = 'Change in optimized spend for each channel'
SPEND_DELTA_CHART_INSIGHTS = """Note: Each bar represents the change in
optimized spend for a channel. Negative values indicate a decrease in spend
while positive values indicate an increase from your current spend."""

SPEND_ALLOCATION_CHART_ID = 'spend-allocation-chart'
SPEND_ALLOCATION_CHART_TITLE = 'Optimized budget allocation'

IMPACT_DELTA_CHART_ID = 'impact-delta-chart'
IMPACT_DELTA_CHART_TITLE = 'Optimized incremental {impact} across all channels'
IMPACT_DELTA_CHART_INSIGHTS_FORMAT = """Note: The "current" bar presents the
incremental {impact} before optimization, and the subsequent bars represent the
change in incremental {impact} for each channel based on the optimal spend
level. The final bar labeled "optimized" shows the total incremental {impact}
after optimization."""

SPEND_ALLOCATION_TABLE_ID = 'spend-allocation-table'

OPTIMIZED_RESPONSE_CURVES_CARD_ID = 'optimized-response-curves'
OPTIMIZED_RESPONSE_CURVES_CARD_TITLE = 'Response curves with optimized spend'
OPTIMIZED_RESPONSE_CURVES_INSIGHTS_FORMAT = """These response curves depict the
relationship between marketing spend and the resulting incremental {impact} and
show the optimal spend amount for each channel that maximizes the total
incremental {impact} while staying within the channel-level spend
constraints."""

OPTIMIZED_RESPONSE_CURVES_CHART_ID = 'optimized-response-curves-chart'
OPTIMIZED_RESPONSE_CURVES_CHART_TITLE = 'Optimized response curves'


# Visualizer-only plot titles.
PRIOR_POSTERIOR_DIST_CHART_TITLE = 'Prior vs Posterior Distributions'
RHAT_BOXPLOT_TITLE = 'R-hat Convergence Diagnostic'
ADSTOCK_DECAY_CHART_TITLE = 'Adstock Decay of Effectiveness Over Time'
HILL_SATURATION_CHART_TITLE = 'Hill Saturation Curves'


# Plot labels.
CHANNEL_LABEL = 'Channel'
SPEND_LABEL = 'Spend'
ROI_LABEL = 'ROI'
CPIK_LABEL = 'CPIK'
KPI_LABEL = 'KPI'
REVENUE_LABEL = 'Revenue'
INC_REVENUE_LABEL = 'Incremental revenue'
INC_KPI_LABEL = 'Incremental KPI'
OPTIMIZED_SPEND_LABEL = 'Optimized spend'
NONOPTIMIZED_SPEND_LABEL = 'Non-optimized spend'
CURRENT_SPEND_LABEL = 'Current spend'
RESPONSE_CURVES_LABEL = 'Response curves'
HILL_SHADED_REGION_RF_LABEL = 'Relative Distribution of Average Frequency'
HILL_SHADED_REGION_MEDIA_LABEL = (
    'Relative Distribution of Media Units per Capita'
)
HILL_X_AXIS_MEDIA_LABEL = 'Media Units per Capita'
HILL_X_AXIS_RF_LABEL = 'Average Frequency'
HILL_Y_AXIS_LABEL = 'Hill Saturation Level'
EXPECTED_ROI_LABEL = 'Expected ROI'
OPTIMAL_FREQ_LABEL = 'Optimal Frequency'

# Table contents.
DATASET_LABEL = 'Dataset'
R_SQUARED_LABEL = 'R-squared'
MAPE_LABEL = 'MAPE'
WMAPE_LABEL = 'wMAPE'
TRAINING_DATA_LABEL = 'Training Data'
TESTING_DATA_LABEL = 'Testing Data'
ALL_DATA_LABEL = 'All Data'

# Summary metrics table columns.
PCT_IMPRESSIONS_COL = '% impressions'
PCT_SPEND_COL = '% spend'
PCT_CONTRIBUTION_COL = '% contribution'
INC_REVENUE_COL = 'incremental revenue'
INC_KPI_COL = 'incremental KPI'
