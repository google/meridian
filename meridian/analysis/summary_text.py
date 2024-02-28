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
MODEL_FIT_CARD_TITLE = "Your marketing mix's model fit"
MODEL_FIT_INSIGHTS_FORMAT = """Model fitting is a measure of how well a machine
learning model generalizes to similar data to that on which it was trained.
A well-fitted model produces more accurate outcomes. A model that is overfitted
matches the data too closely, and a model that is underfitted doesn't match
closely enough."""

EXPECTED_ACTUAL_IMPACT_CHART_ID = 'expected-actual-impact-chart'
EXPECTED_ACTUAL_IMPACT_CHART_TITLE = 'Expected {impact} vs. actual {impact}'
EXPECTED_ACTUAL_IMPACT_CHART_DESCRIPTION = """Note: The expected and baseline
are your posterior mean."""

PREDICTIVE_ACCURACY_TABLE_ID = 'model-fit-statistics-table-chart'
PREDICTIVE_ACCURACY_TABLE_TITLE = (
    'Model fit of your predicted and actual {impact}'
)
PREDICTIVE_ACCURACY_TABLE_DESCRIPTION = """Note: Correlation measures the
strength of the relationship between predicted and actual {impact}. R-squared
measures the amount of variation in the data that is explained by the model. The
closer to 1 in both r-squared and correlation the more accurate the model."""

IMPACT_CONTRIB_CARD_ID = 'impact-contrib'
IMPACT_CONTRIB_CARD_TITLE = '{impact} contribution'
IMPACT_CONTRIB_INSIGHTS_FORMAT = """Your revenue contributions help you
understand what drove your revenue. {lead_channels} drove the most
overall revenue. {lead_roi_channel} drove the highest return on investment at
{lead_roi_ratio:.1f}. For every $1 you spent on {lead_roi_channel}, you saw
${lead_roi_ratio:.1f} in revenue."""

CHANNEL_DRIVERS_CHART_ID = 'channel-drivers-chart'
CHANNEL_DRIVERS_CHART_TITLE = 'Contribution by baseline and marketing channels'
CHANNEL_DRIVERS_CHART_DESCRIPTION = """Note: This graphic encompasses all of
your {impact} drivers, but breaks down your marketing actives by channel."""

SPEND_IMPACT_CHART_ID = 'spend-impact-chart'
SPEND_IMPACT_CHART_TITLE = (
    'Spend and {impact} contribution by marketing channel'
)
SPEND_IMPACT_CHART_DESCRIPTION = """Note: Return on investment is calculated by
dividing the {impact} attributed to a channel by marketing costs."""

IMPACT_CONTRIBUTION_CHART_ID = 'impact-contribution-chart'
CONTRIBUTION_CHART_TITLE = 'Contribution by baseline and marketing channels'
IMPACT_CONTRIBUTION_CHART_DESCRIPTION = """Note: This is a percentage
breakdown of all your contributions of {impact}."""

ROI_BREAKDOWN_CARD_ID = 'roi-breakdown'
ROI_BREAKDOWN_CARD_TITLE = 'Return on investment'
ROI_BREAKDOWN_INSIGHTS_FORMAT = """Your return on investment helps you
understand how your marketing activities impacted your business' objectives.
{lead_effectiveness_channel} had the highest effectiveness, which is your
incremental revenue per media unit. {lead_marginal_roi_channel} had the highest
marginal return on investment at {lead_marginal_roi_channel_value:.2f}."""

ROI_EFFECTIVENESS_CHART_ID = 'roi-effectiveness-chart'
ROI_EFFECTIVENESS_CHART_TITLE = 'Return on investment vs. effectiveness'
ROI_EFFECTIVENESS_CHART_DESCRIPTION = """Note: Return on investment by
effectiveness measures the profitability of an investment, taking into account
the effectiveness = incremental revenue / number of impressions."""

ROI_MARGINAL_CHART_ID = 'roi-marginal-chart'
ROI_MARGINAL_CHART_TITLE = 'Return on investment vs. marginal'
ROI_MARGINAL_CHART_DESCRIPTION = """Note: Return on investment by marginal (ROI
by marginal) is a measure of the profitability of an investment, taking into
account the additional revenue generated by the investment. It is calculated by
dividing the additional revenue generated by the investment by the cost of the
investment."""

ROI_CHANNEL_CHART_ID = 'roi-channel-chart'
ROI_CHANNEL_CHART_TITLE_FORMAT = 'Return on investment by channel {ci}'
ROI_CHANNEL_CHART_DESCRIPTION = """Note: This is your return on investment with
a confidence interval for each channel."""

BUDGET_OPTIMIZATION_CARD_ID = 'budget-optimization'
BUDGET_OPTIMIZATION_CARD_TITLE = 'Response curves'
BUDGET_OPTIMIZATION_INSIGHTS_FORMAT = """Your response curves and optimal
frequency for budget planning. Your optimal frequency for {rf_channel} is
{opt_freq} per week."""
BUDGET_OPTIMIZATION_INSIGHTS_NO_RF = 'Your response curves for budget planning.'

RESPONSE_CURVES_CHART_ID = 'response-curves-chart'
RESPONSE_CURVES_CHART_TITLE = (
    'Response curves by marketing channel {top_channels}'
)
RESPONSE_CURVES_CHART_DESCRIPTION = """Note: Response curves display your
estimated relationship between your marketing spend and your {impact} based on
your actual historical data and estimation of marginal performance."""

OPTIMAL_FREQUENCY_CHART_ID = 'optimal-frequency-chart'
OPTIMAL_FREQUENCY_CHART_TITLE = (
    'Return on investment by weekly average frequency'
)
OPTIMAL_FREQUENCY_CHART_DESCRIPTION = """Note: optimal frequency is the
recommended average weekly impressions per user (# impressions / # reached
users)."""


# Budget optimization texts.
OPTIMIZATION_TITLE = 'MMM Optimization Report'

SCENARIO_PLAN_CARD_ID = 'scenario-plan'
SCENARIO_PLAN_CARD_TITLE = 'Optimization scenario plan'
SCENARIO_PLAN_INSIGHTS_FORMAT = """These are the results of your future
marketing budgets with a channel-level spend constraint of {lower_bound}x -
{upper_bound}x current spend over the time period from {start_date} to
{end_date}."""

CURRENT_BUDGET_LABEL = 'Current budget'
OPTIMIZED_BUDGET_LABEL = 'Optimized budget'
FIXED_BUDGET_LABEL = 'Fixed'
FLEXIBLE_BUDGET_LABEL = 'Flexible'
CURRENT_ROI_LABEL = 'Current ROI'
OPTIMIZED_ROI_LABEL = 'Optimized ROI'
CURRENT_INC_IMPACT_LABEL = 'Current incremental {impact}'
OPTIMIZED_INC_IMPACT_LABEL = 'Optimized incremental {impact}'

BUDGET_ALLOCATION_CARD_ID = 'budget-allocation'
BUDGET_ALLOCATION_CARD_TITLE = 'Changes in your marketing budget allocation'
BUDGET_ALLOCATION_INSIGHTS = """You can see how much your channel performance
and spend have affected your {impact}."""

SPEND_DELTA_CHART_ID = 'spend-delta-chart'
SPEND_DELTA_CHART_TITLE = 'Change in optimized spend for each channel'

SPEND_ALLOCATION_CHART_ID = 'spend-allocation-chart'
SPEND_ALLOCATION_CHART_TITLE = 'Optimized spend allocation'

IMPACT_DELTA_CHART_ID = '{impact}-delta-chart'
IMPACT_DELTA_CHART_TITLE = 'Optimized incremental {impact} across all channels'

SPEND_ALLOCATION_TABLE_ID = 'spend-allocation-table'

OPTIMIZED_RESPONSE_CURVES_CARD_ID = 'optimized-response-curves'
OPTIMIZED_RESPONSE_CURVES_CARD_TITLE = 'Optimized response curves by channel'
OPTIMIZED_RESPONSE_CURVES_INSIGHTS = """These response curves show the potential
return on investment on your channel spend and your potential {impact}. You can
use the optimized spend as a recommendation to guide your future marketing
spend. The more bend in your response curve the better the potential return on
investment."""

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
KPI_LABEL = 'KPI'
REVENUE_LABEL = 'Revenue'
INC_REVENUE_LABEL = 'Incremental Revenue'
INC_KPI_LABEL = 'Incremental KPI'
OPTIMIZED_SPEND_LABEL = 'Optimized spend'
NONOPTIMIZED_SPEND_LABEL = 'Non-optimized spend'
RESPONSE_CURVES_LABEL = 'Response curves'
HILL_SHADED_REGION_RF_LABEL = 'Relative Distribution of Average Frequency'
HILL_SHADED_REGION_MEDIA_LABEL = (
    'Relative Distribution of Media Units per Capita'
)
HILL_X_AXIS_MEDIA_LABEL = 'Media Units per Capita'
HILL_X_AXIS_RF_LABEL = 'Average Frequency'
HILL_Y_AXIS_LABEL = 'Hill Saturation Level'

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
