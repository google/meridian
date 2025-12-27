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

"""Constants specific to MeridianEDA."""
from typing import Literal
import altair as alt
import numpy as np

# EDA Engine constants
DEFAULT_DA_VAR_AGG_FUNCTION = np.sum
COST_PER_MEDIA_UNIT = 'cost_per_media_unit'
RSQUARED_GEO = 'rsquared_geo'
RSQUARED_TIME = 'rsquared_time'
VARIABLE_1 = 'var1'
VARIABLE_2 = 'var2'
CORRELATION = 'correlation'
ABS_CORRELATION_COL_NAME = 'abs_correlation'
CORRELATION_MATRIX_NAME = 'correlation_matrix'
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

# EDA Plotting properties
VARIABLE = 'var'
VALUE = 'value'
NATIONALIZE: Literal['nationalize'] = 'nationalize'
MEDIA_IMPRESSIONS_SCALED = 'media_impressions_scaled'
IMPRESSION_SHARE_SCALED = 'impression_share_scaled'
SPEND_SHARE = 'spend_share'
LABEL = 'label'
PAIRWISE_CORR_COLOR_SCALE = alt.Scale(
    domain=[-1.0, 0.0, 1.0],
    range=['#1f78b4', '#f7f7f7', '#e34a33'],  # Blue-light grey-Orange
    type='linear',
)

# Report constants
REPORT_TITLE = 'Meridian Exploratory Data Analysis Report'
SPEND_AND_MEDIA_UNIT_CARD_ID = 'spend-and-media-unit'
SPEND_AND_MEDIA_UNIT_CARD_TITLE = 'Spend and Media Unit'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID = 'relationship-among-variables'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE = 'Relationship Among the Variables'
RELATIVE_SPEND_SHARE_CHART_ID = 'relative-spend-share-chart'
PAIRWISE_CORRELATION_CHART_ID = 'pairwise-correlation-chart'
R_SQUARED_TIME_TABLE_ID = 'r-squared-time-table'
R_SQUARED_GEO_TABLE_ID = 'r-squared-geo-table'
DISPLAY_LIMIT = 5


# Finding messages
RELATIVE_SPEND_SHARE_INFO = (
    "Please review the channel's share of spend. Channels with a very small"
    ' share of spend might be difficult to estimate. You might want to combine'
    ' them with other channels. Meanwhile, a channel with a huge spend share'
    ' would increase the risk of producing a negative baseline if it also has a'
    ' high ROI.'
)
PAIRWISE_CORRELATION_CHECK_INFO = (
    'Please review the computed pairwise correlations. Note that'
    ' high pairwise correlation may cause model identifiability'
    ' and convergence issues. Consider combining the variables if'
    ' high correlation exists.'
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
