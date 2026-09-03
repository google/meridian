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

"""Constants specific to MeridianEDA."""

import textwrap
from typing import Literal
import altair as alt
import immutabledict
from meridian import constants
from meridian.model.calibration import base as calibration_base
import numpy as np

##### EDA Engine constants #####
DEFAULT_DA_VAR_AGG_FUNCTION = np.sum
COST_PER_MEDIA_UNIT = 'cost_per_media_unit'
ABS_COST_PER_MEDIA_UNIT = 'abs_cost_per_media_unit'
RSQUARED_GEO = 'rsquared_geo'
RSQUARED_TIME = 'rsquared_time'
VARIABLE_1 = 'var1'
VARIABLE_2 = 'var2'
CORRELATION = 'correlation'
ABS_CORRELATION_COL_NAME = 'abs_correlation'
CORRELATION_MATRIX_NAME = 'correlation_matrix'

# Default values for eda_spec.py
OVERALL_PAIRWISE_CORR_THRESHOLD = 0.999
GEO_PAIRWISE_CORR_THRESHOLD = 0.999
NATIONAL_PAIRWISE_CORR_THRESHOLD = 0.999
Q1_THRESHOLD = 0.25
Q3_THRESHOLD = 0.75
IQR_MULTIPLIER = 1.5
STD_WITH_OUTLIERS_VAR_NAME = 'std_with_outliers'
STD_WITHOUT_OUTLIERS_VAR_NAME = 'std_without_outliers'
STD_THRESHOLD = 1e-4
OUTLIERS_COL_NAME = 'outliers'
ABS_OUTLIERS_COL_NAME = 'abs_outliers'
VIF_COL_NAME = 'VIF'
EXTREME_CORRELATION_WITH = 'extreme_correlation_with'
TIME_AND_GEO_AGGREGATION = 'times and geos'
TIME_AGGREGATION = 'times'
PRIOR_CONTRIBUTION = 'prior_contribution'

DEFAULT_PRIOR_N_DRAW = 500
DEFAULT_PRIOR_SEED = 0

##### EDA Plotting properties #####
CORRELATION_RED = '#d73027'
CORRELATION_WHITE = '#f7f7f7'
CORRELATION_BLUE = '#4575b4'
CORRELATION_LEGEND_TITLE = 'correlation (blue=OK, red=bad)'
VARIABLE = 'var'
VALUE = 'value'
NATIONALIZE: Literal['nationalize'] = 'nationalize'
MEDIA_IMPRESSIONS_SCALED = 'media_impressions_scaled'
IMPRESSION_SHARE_SCALED = 'impression_share_scaled'
SPEND_SHARE = 'spend_share'
LABEL = 'label'
DEFAULT_CHART_COLOR = '#4C78A8'
PAIRWISE_CORR_COLOR_SCALE = alt.Scale(
    domain=[-1.0, -0.5, 0.0, 0.5, 1.0],
    range=[
        CORRELATION_RED,
        CORRELATION_WHITE,
        CORRELATION_BLUE,
        CORRELATION_WHITE,
        CORRELATION_RED,
    ],
    type='linear',
)
POPULATION_RAW_MEDIA_CORRELATION_ENCODINGS = immutabledict.immutabledict({
    'x': alt.X(
        f'{VARIABLE}:N',
        sort=None,
        title=constants.CHANNEL,
        axis=alt.Axis(labelAngle=-45),
    ),
    'y': alt.Y(
        f'{VALUE}:Q', title=CORRELATION, scale=alt.Scale(domain=[-1, 1])
    ),
    'color': alt.Color(
        f'{VALUE}:Q',
        scale=alt.Scale(
            domain=[-1, 0, 1],
            range=[CORRELATION_RED, CORRELATION_WHITE, CORRELATION_BLUE],
        ),
        legend=alt.Legend(title=CORRELATION_LEGEND_TITLE, orient='bottom'),
    ),
})
POPULATION_TREATMENT_CORRELATION_ENCODINGS = immutabledict.immutabledict({
    'x': alt.X(
        f'{VARIABLE}:N',
        sort=None,
        title=constants.CHANNEL,
        axis=alt.Axis(labelAngle=-45),
    ),
    'y': alt.Y(
        f'{VALUE}:Q', title=CORRELATION, scale=alt.Scale(domain=[-1, 1])
    ),
    'color': alt.Color(
        f'{VALUE}:Q',
        scale=alt.Scale(
            domain=[-1, -0.5, 0, 0.5, 1],
            range=[
                CORRELATION_RED,
                CORRELATION_WHITE,
                CORRELATION_BLUE,
                CORRELATION_WHITE,
                CORRELATION_RED,
            ],
        ),
        legend=alt.Legend(title=CORRELATION_LEGEND_TITLE, orient='bottom'),
    ),
})
PRIOR_MEAN_ENCODINGS = immutabledict.immutabledict({
    'x': alt.X(
        f'{VARIABLE}:N',
        sort=None,
        title=constants.CHANNEL,
        axis=alt.Axis(labelAngle=-45),
    ),
    'y': alt.Y(f'{VALUE}:Q', title=PRIOR_CONTRIBUTION),
})
CHANNEL_TYPE_TO_COLOR = immutabledict.immutabledict({
    constants.MEDIA_UNITS: '#4285F4',
    constants.MEDIA_CHANNEL: '#4285F4',
    constants.SPEND: '#FBBC04',
    COST_PER_MEDIA_UNIT: '#A142F4',
    constants.ORGANIC_MEDIA_CHANNEL: '#F29900',
    constants.RF_CHANNEL: '#EA4335',
    constants.ORGANIC_RF_CHANNEL: '#FBBC04',
    constants.CONTROL_VARIABLE: '#34A853',
    constants.NON_MEDIA_CHANNEL: '#12939A',
})


EXPERIMENT_SOURCE_TYPE_TO_LABEL_SUFFIX = immutabledict.immutabledict({
    calibration_base.SourceType.MERIDIAN_GEOX: ' (Meridian GeoX)',
    calibration_base.SourceType.GENERIC: ' (Incrementality)',
})
# Prior calibration plotting constants
PRIOR_QUANTILE_THRESHOLD = 0.99
EXPERIMENT_SD_MULTIPLIER = 3
MIN_ROI = 0.01
ROI_GRID_BOUND_MULTIPLIER = 1.5
ROI_GRID_POINTS = 100
HISTOGRAM_BINS = 100
PRIOR_CALIBRATION_BANNER_TEXT_TEMPLATE = (
    'This figure displays your incrementality experiments and their impact on'
    ' your Meridian priors for {channels}.'
    ' Please inspect the plot to ensure the calibrated prior aligns with'
    ' your expectations.'
)
PRIOR_CALIBRATION_REPORT_CHANNEL_LIMIT = 5
PRIOR_CALIBRATION_REPORT_EXPERIMENT_LIMIT = 3
PRIOR_CALIBRATION_LIMIT_MESSAGE = (
    ' (Due to space constraints, this report only displays your {limit}'
    ' highest-spend calibrated channels. Please use'
    ' `MeridianEDA.plot_calibration()` in your notebook to review all'
    ' calibrated channels.)'
)
PRIOR_CALIBRATION_EXPERIMENT_ORDER_INFO = (
    'Experiments within each channel are displayed in the order they were'
    ' input in `CalibrationBuilder`.'
)
PRIOR_CALIBRATION_ADJUSTMENT_GRID_INFO = (
    'Shows step-by-step experiment parameter adjustments for calibrated'
    ' channels.'
)
STAGE = 'stage'
POINT_ESTIMATE = 'point_estimate'
STANDARD_ERROR = 'standard_error'
CI_LOWER = 'ci_lower'
CI_UPPER = 'ci_upper'
LABEL_TEXT = 'label_text'
MEAN_ROI_PLUS_MINUS_SE = 'Mean ROI ± SE'
EXPERIMENT_ADJUSTMENTS_PLOT_TITLE_TEMPLATE = 'Experiment Adjustments: {ch_name}'
EXPERIMENT_LEGEND_TITLE = 'Experiment'
STAGE_UNADJUSTED_RAW = 'Unadjusted (Raw)'
STAGE_SPEND_ADJUSTED = 'Spend Adjusted'
STAGE_SPEND_DURATION_ADJUSTED = 'Spend + Duration\nAdjusted'
STAGE_SPEND_DURATION_RECENCY_ADJUSTED = 'Spend + Duration +\nRecency Adjusted'
STAGE_SPEND_DURATION_RECENCY_USER_ADJUSTED = (
    'Spend + Duration +\nRecency + User Adjusted'
)
STAGE_FINAL_ADJUSTED = 'Adjusted (Final)'
DENSITY = 'density'
INTERMEDIARY_PRIOR = 'Intermediary prior'
BASELINE_PRIOR = 'Baseline prior'
CALIBRATED_PRIOR = 'Calibrated prior'
EXPERIMENT_LABEL_PREFIX = 'Experiment'
CALIBRATION_LEFT_PLOT_TITLE = (
    'Incrementality Experiments and Intermediary Prior'
)
CALIBRATION_RIGHT_PLOT_TITLE = 'Intermediary and Parameterized Priors'
INTERMEDIARY_PRIOR_COLOR = '#bdc1c6'
BASELINE_PRIOR_COLOR = '#70757a'
CALIBRATED_PRIOR_COLOR = '#146c2e'
EXPERIMENT_COLORS = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
)
PRIOR_CALIBRATION_CHART_ID = 'prior-calibration-chart'
EXPERIMENT_ADJUSTMENTS_CHART_ID = 'experiment-adjustments-chart'
EXPERIMENT_ADJUSTMENTS_REPORT_CHANNEL_LIMIT = 5
EXPERIMENT_ADJUSTMENTS_REPORT_EXPERIMENT_LIMIT = 3
EXPERIMENT_ADJUSTMENTS_BANNER_TEXT_TEMPLATE = (
    'These plots display your incrementality experiments and the adjustments'
    " we made for each experiment's spend, duration and recency for {channels}."
    ' They also show the final mean and standard error we used to inform the'
    ' Meridian prior. The Meridian prior for each channel is a combination of'
    ' your adjusted experiments for that channel and your baseline prior, if'
    ' applicable.'
)
EXPERIMENT_ADJUSTMENTS_LIMIT_MESSAGE = (
    ' (Due to space constraints, this report only displays your {limit}'
    ' highest-spend calibrated channels with their 3 smallest standard error'
    ' experiments. Please use `MeridianEDA.plot_experiment_adjustments()` in'
    ' your notebook to review experiment adjustments for all experiments in'
    ' all calibrated channels.)'
)


##### Report constants #####
REPORT_TITLE = 'Meridian Exploratory Data Analysis Report'
DISPLAY_LIMIT_MESSAGE = (
    '<br/>(Due to space constraints, this table only displays the 5 most severe'
    ' cases. Please use {function} to review {to_review}.)'
)
DISPLAY_LIMIT = 5
TIME_SERIES_LIMIT = 2
POPULATION_CORRELATION_BARCHART_LIMIT = 15
PRIOR_MEAN_BARCHART_LIMIT = 15
PAIRWISE_CORRELATION_VAR_NUMBER_LIMIT = 15
BOXPLOT_VAR_NUMBER_LIMIT = 15
# category 1
SPEND_AND_MEDIA_UNIT_CARD_ID = 'spend-and-media-unit'
SPEND_AND_MEDIA_UNIT_CARD_TITLE = 'Spend and Media Unit'
RELATIVE_SPEND_SHARE_CHART_ID = 'relative-spend-share-chart'
SPEND_PER_MEDIA_UNIT_CHART_ID = 'spend-per-media-unit-chart'
DATA_ADEQUACY_TABLE_ID = 'data-adequacy-table'
INCONSISTENT_DATA_TABLE_ID = 'inconsistent-data-table'
COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID = 'cost-per-media-unit-outlier-table'
# category 2
RESPONSE_VARIABLES_CARD_ID = 'response-variables'
RESPONSE_VARIABLES_CARD_TITLE = 'Individual Explanatory/Response Variables'
TREATMENTS_CHART_ID = 'treatments-chart'
CONTROLS_AND_NON_MEDIA_CHART_ID = 'controls-and-non-media-chart'
KPI_CHART_ID = 'kpi-chart'
TREATMENT_CONTROL_VARIABILITY_TABLE_ID = 'treatment-control-variability-table'
TREATMENT_CONTROL_OUTLIER_TABLE_ID = 'treatment-control-outlier-table'
# category 3
POPULATION_SCALING_CARD_ID = 'population-scaling'
POPULATION_SCALING_CARD_TITLE = 'Population Scaling of Explanatory Variables'
POPULATION_RAW_MEDIA_CHART_ID = 'population-raw-media-chart'
POPULATION_TREATMENT_CHART_ID = 'population-treatment-chart'
# category 4
RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID = 'relationship-among-variables'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE = 'Relationship Among the Variables'
PAIRWISE_CORRELATION_CHART_ID = 'pairwise-correlation-chart'
EXTREME_VIF_FAIL_TABLE_ID = 'extreme-vif-fail-table'
EXTREME_VIF_REVIEW_TABLE_ID = 'extreme-vif-review-table'
R_SQUARED_TIME_TABLE_ID = 'r-squared-time-table'
R_SQUARED_GEO_TABLE_ID = 'r-squared-geo-table'
# category 5
PRIOR_SPECIFICATIONS_CARD_ID = 'prior-specifications'
PRIOR_SPECIFICATIONS_CARD_TITLE = 'Prior Specifications'
PRIOR_CHART_ID = 'prior-chart'
PRIOR_QUALITY_TABLE_ID = 'prior-quality-table'
CALIBRATED_PRIOR_PARAMETERS = (constants.ROI_M, constants.ROI_RF)
PRIOR_QUALITY_RELATIVE_WIDTH_RATIO = 'relative_width_ratio'
PRIOR_QUALITY_BIMODAL_STATISTIC = 'bimodal_statistic'
PRIOR_QUALITY_OVERLAP_PERCENTAGE = 'overlap_percentage'
PRIOR_QUALITY_NEGATIVE_EXPERIMENTS = 'negative_experiments'
OVERLAP_PERCENTAGE_THRESHOLD = 0.80
# Empirically calibrated F0.5-maximizing Dip Test statistic cutoff threshold
# across 220 benchmark priors (44 LogNormal baselines x 5 Normal likelihoods).
DIP_STATISTIC_THRESHOLD = 0.00338
# Threshold for prior width ratio.
HIGH_VARIANCE_THRESHOLD = 1.0
# summary
SUMMARY_CARD_ID = 'summary'
SUMMARY_CARD_TITLE = 'Summary'
SUMMARY_TABLE_ID = 'summary-table'
CATEGORY = 'Category'
FINDING = 'Finding'
RECOMMENDED_NEXT_STEP = 'Recommended Next Step'


##### Finding messages #####
SUMMARY_TABLE_SUMMARY_INFO = (
    'Review the full report to investigate the health of your dataset and'
    ' confirm findings align with your expectations.'
)
SUMMARY_TABLE_SUMMARY_FINDING = (
    'Review the health of your dataset below. Resolve all FAILS and investigate'
    ' REVIEW flags in the detailed sections to ensure your data is ready for'
    ' modeling.'
)
SUMMARY_TABLE_SPEND_AND_MEDIA_UNIT_INFO = (
    'No automated issues detected. See <a'
    f' href="#{SPEND_AND_MEDIA_UNIT_CARD_ID}">Spend and Media Units</a> for'
    ' more details.'
)
SUMMARY_TABLE_SPEND_AND_MEDIA_UNIT_FINDING = (
    f'See <a href="#{SPEND_AND_MEDIA_UNIT_CARD_ID}">Spend and Media Units</a>.'
    ' Where applicable, verify that spend and media units align across'
    ' channels, and review outliers in cost per media unit.'
)
SUMMARY_TABLE_RESPONSE_VARIABLES_INFO = (
    'No automated issues detected. See <a'
    f' href="#{RESPONSE_VARIABLES_CARD_ID}">Individual Explanatory/Response'
    ' Variables</a> for more details.'
)
SUMMARY_TABLE_RESPONSE_VARIABLES_FINDING = (
    f'See <a href="#{RESPONSE_VARIABLES_CARD_ID}">Individual'
    ' Explanatory/Response Variables</a>. Where applicable, review any'
    ' variables with low signal or with outliers.'
)
SUMMARY_TABLE_POPULATION_SCALING_INFO = (
    'No automated issues detected. See <a'
    f' href="#{POPULATION_SCALING_CARD_ID}">Population Scaling</a> for more'
    ' details.'
)
SUMMARY_TABLE_RELATIONSHIP_BETWEEN_VARIABLES_INFO = (
    'No automated issues detected. See <a'
    f' href="#{RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID}">Relationship Among the'
    ' Variables</a> for more details.'
)
SUMMARY_TABLE_RELATIONSHIP_BETWEEN_VARIABLES_FINDING = (
    f'See <a href="#{RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID}">Relationship'
    ' Among the Variables</a>. Check for high multicollinearity among the'
    ' variables that could lead to model convergence issues.'
)
SUMMARY_TABLE_PRIOR_SPECIFICATIONS_INFO = (
    'No automated issues detected. See <a'
    f' href="#{PRIOR_SPECIFICATIONS_CARD_ID}">Prior Specifications</a> for more'
    ' details. Assess the likelihood of a negative baseline occurring.'
)
SPEND_PER_MEDIA_UNIT_INFO = (
    'Please review the patterns for spend, media units, and'
    ' cost-per-media unit. Any erratic or unexpected patterns warrant a data'
    ' review.'
)
VARIABILITY_PLOT_INFO = (
    'Please review the variability of the explanatory and response variables'
    ' illustrated by the boxplots. Note that variables with very low'
    ' variability could be difficult to estimate and could hurt model'
    ' convergence. Consider merging or replacing them with other variables,'
    ' dropping them if they are negligibly small, or using a custom prior if'
    ' you have relevant information. If outliers exist, check your data input'
    ' to determine if they are genuine or erroneous.'
)
RELATIVE_SPEND_SHARE_INFO = (
    "Please review the channel's share of spend. Channels with a very small"
    ' share of spend might be difficult to estimate. You might want to combine'
    ' them with other channels.'
)
PAIRWISE_CORRELATION_CHECK_INFO = (
    'Please review the computed pairwise correlations. Note that'
    ' high pairwise correlation may cause model identifiability'
    ' and convergence issues. Consider combining the variables if'
    ' high correlation exists.'
)
MULTICOLLINEARITY_FAIL = (
    'Some variables have extreme multicollinearity (VIF'
    ' > {threshold}) across all {aggregation}. Note that'
    ' a common cause of multicollinearity is perfect pairwise'
    ' correlation. To address multicollinearity, please drop any'
    ' variable that is a linear combination of other variables.'
    ' Otherwise, consider combining variables.{additional_info}'
)
MULTICOLLINEARITY_REVIEW = (
    'Some variables have extreme multicollinearity (VIF >'
    ' {threshold}) in certain geo(s). Note that a common'
    ' cause of multicollinearity is perfect pairwise'
    " correlation. While this geo-level issue isn't necessarily"
    ' problematic due to hierarchical modeling in Meridian, it'
    ' may be a data issue that could lead to poor inference or'
    ' even poor convergence. Consider checking your data or'
    ' combining these variables, especially if they also have'
    ' high VIF in other geos.{additional_info}'
)
R_SQUARED_TIME_INFO = (
    'This check regresses each variable against time as a'
    ' categorical variable. In this case, high R-squared indicates'
    ' low geo variation of a variable. This could lead to a weakly'
    ' identifiable and non-converging model if a large number of'
    ' knots are used. Consider dropping the variable with very high'
    ' R-squared or reducing `knots` argument in `ModelSpec`.'
)
R_SQUARED_GEO_INFO = (
    'This check regresses each variable against geo as a'
    ' categorical variable. In this case, high R-squared indicates'
    ' low time variation of a variable. This could lead to a weakly'
    ' identifiable and non-converging model due to geo main'
    ' effects. Consider dropping the variable with very high'
    ' R-squared.'
)
POPULATION_CORRELATION_SCALED_TREATMENT_CONTROL_INFO = textwrap.dedent("""\
Please review the Spearman correlation between population and scaled treatment \
units or scaled controls.

For controls and non-media channels: Meridian doesn't population-scale these \
variables by default. High correlation indicates that users should \
population-scale these variables using the `control_population_scaling_id` or \
`non_media_population_scaling_id` argument in `ModelSpec`.

For paid and organic media channels: Meridian automatically population-scales \
these media channels by default. High correlation indicates that the variable \
may have been population-scaled before being passed to Meridian. Please check \
your data input.""")
POPULATION_CORRELATION_RAW_MEDIA_INFO = (
    'Please review the Spearman correlation between population and raw paid and'
    ' organic media variables. These raw media variables are expected to have'
    ' positive correlation with population. If there is low or negative'
    ' correlation, please check your data input.'
)
PRIOR_PROBABILITY_INFO = (
    'Negative baseline is equivalent to the treatment effects getting too much'
    ' credit. Please review the prior probability of negative baseline together'
    ' with the channel-level prior mean of contribution. If the prior'
    ' probability of negative baseline is high, consider custom treatment'
    ' priors. In particular, a custom `contribution prior` type may be'
    ' appropriate.'
)
PRIOR_PROBABILITY_REPORT_INFO = (
    'Negative baseline is equivalent to the treatment effects getting too much'
    ' credit. Please review the prior probability of negative baseline together'
    ' with the bar chart for channel-level prior mean of contribution. If the'
    ' prior probability of negative baseline is high, consider custom treatment'
    ' priors. In particular, a custom `contribution prior` type may be'
    ' appropriate.<br/><br/>'
)
PRIOR_QUALITY_TABLE_INFO = (
    'Please review the prior quality checks for calibrated channels. A bimodal'
    ' statistic greater than 0.00338 indicates potential multimodality.'
)
BASELINE_PRIOR_TYPE = 'Baseline Prior Type'
PRIOR_BIMODALITY_STATISTIC = 'Prior Bimodality Statistic'
PRIOR_WIDTH_RATIO = 'Prior Width Ratio'
INTERMEDIARY_AND_PARAMETERIZED_OVERLAP_PERCENTAGE = (
    'Intermediary and Parameterized Overlap Percentage'
)
NEGATIVE_POINT_ESTIMATE_EXPERIMENTS = 'Negative Point Estimate Experiments'
NOT_AVAILABLE = 'Not Available'
LOG_CONCAVITY_EVALUATION_BOUNDS = (0.01, 20.0)
LOG_CONCAVITY_NUM_POINTS = 1000
LOG_CONCAVITY_NUMERICAL_TOLERANCE = 1e-6
BIMODALITY_STATISTIC_THRESHOLD = 0.00338
GRID = 'grid'
PDF = 'pdf'
DX = 'dx'
NEGATIVE_POINT_ESTIMATE_EXPERIMENTS_WARNING = (
    'Channel(s) [{channels}] have a calibrated prior based on experiments with'
    ' negative point estimates. Review these experiment results for accuracy.'
)
UNINFORMATIVE_CALIBRATED_PRIOR_WARNING = (
    'Channel(s) [{channels}] have an uninformative calibrated prior. Consider'
    ' adjusting the baseline prior or user adjustments for the experiments'
    ' associated with these channels to increase the informativeness of this'
    ' prior.'
)
PRIOR_BIMODALITY_WARNING = (
    'Channel(s) [{channels}] have an intermediary prior that is bimodal.'
    ' Inspect the Incrementality Experiments and Intermediary Prior and'
    ' Intermediary Prior and Parameterized Prior plots to determine if the'
    ' parameterized prior is an accurate representation of the intermediary'
    ' prior. To reduce bimodality, consider using a baseline prior that is'
    ' more aligned with your experiments.'
)
BASELINE_PRIOR_TYPE_BIMODALITY_WARNING = (
    'Channel(s) [{channels}] have a baseline prior type that may lead to'
    ' intermediary prior bimodality. Inspect the Incrementality Experiments'
    ' and Intermediary Prior and Intermediary Prior and Parameterized Prior'
    ' plots to determine if the parameterized prior is an accurate'
    ' representation of the intermediary prior. To reduce bimodality,'
    ' consider using a baseline prior that is more aligned with your'
    ' experiments.'
)
LOW_OVERLAP_PRIORS_WARNING = (
    'Channel(s) [{channels}] have intermediary and parameterized priors with'
    ' low overlap. Review the "Intermediary and Parameterized Distribution"'
    ' plot to confirm and consider making user adjustments to experiments.'
)
DATA_ADEQUACY_INFO = textwrap.dedent("""\
As a rough guidance, please review the ratio of data points to
parameters, where
* the number of data points = n_geos * n_times,
* the number of parameters = (n_geos-1) + n_knots + n_controls + n_treatments.

A very small ratio indicates insufficient data for estimation.
In that case, consider dropping or combining channels,
or reducing the number of knots with `knots` argument in `ModelSpec`.
For more details, please refer to this documentation page:
https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed.""")

# The boolean keys indicate whether findings were detected (True) or
# not (False), and the values are the corresponding message that should be
# displayed. Example, if there were fails or reviews in the spend and media
# unit card (True), then we want to display the finding message,
# otherwise (False) we display the info message.
CATEGORY_TO_MESSAGE_BY_STATUS = immutabledict.immutabledict({
    SUMMARY_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_SUMMARY_INFO,
        True: SUMMARY_TABLE_SUMMARY_FINDING,
    }),
    SPEND_AND_MEDIA_UNIT_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_SPEND_AND_MEDIA_UNIT_INFO,
        True: SUMMARY_TABLE_SPEND_AND_MEDIA_UNIT_FINDING,
    }),
    RESPONSE_VARIABLES_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_RESPONSE_VARIABLES_INFO,
        True: SUMMARY_TABLE_RESPONSE_VARIABLES_FINDING,
    }),
    POPULATION_SCALING_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_POPULATION_SCALING_INFO,
        True: '',  # currently there are no findings for this card
    }),
    RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_RELATIONSHIP_BETWEEN_VARIABLES_INFO,
        True: SUMMARY_TABLE_RELATIONSHIP_BETWEEN_VARIABLES_FINDING,
    }),
    PRIOR_SPECIFICATIONS_CARD_TITLE: immutabledict.immutabledict({
        False: SUMMARY_TABLE_PRIOR_SPECIFICATIONS_INFO,
        True: '',  # currently there are no findings for this card
    }),
})
