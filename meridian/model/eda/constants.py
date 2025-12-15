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
RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID = 'relationship-among-variables'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE = 'Relationship Among the Variables'
PAIRWISE_CORRELATION_CHART_ID = 'pairwise-correlation-chart'
PERFECT_PAIRWISE_CORRELATION_TEXT_ID = 'perfect-pairwise-correlation-text'
PERFECT_PAIRWISE_CORRELATION_TABLE_ID = 'perfect-pairwise-correlation-table'
EXTREME_VIF_TEXT_ID = 'extreme-vif-text'
EXTREME_VIF_TABLE_ID = 'extreme-vif-table'
R_SQUARED_TIME_TABLE_ID = 'r-squared-time-table'
R_SQUARED_GEO_TABLE_ID = 'r-squared-geo-table'
DISPLAY_LIMIT = 5


# Finding messages
PAIRWISE_CORRELATION_CHECK_INFO = (
    'Please review the computed pairwise correlations. Note that'
    ' high pairwise correlation may cause model identifiability'
    ' and convergence issues. Consider combining the variables if'
    ' high correlation exists.'
)
EXTREME_CORR_VAR_PAIRS_GEO_ATTENTION = (
    'Some variables have perfect pairwise correlation in certain'
    ' geo(s). Consider checking your data, and/or combining these'
    ' variables if they also have high pairwise correlations in'
    ' other geos.'
)
EXTREME_CORR_VAR_PAIRS_OVERALL_ERROR = (
    'Some variables have perfect pairwise correlation across all'
    ' times and geos. For each pair of perfectly-correlated'
    ' variables, please remove one of the variables from the'
    ' model.'
)
EXTREME_CORR_VAR_PAIRS_NATIONAL_ERROR = (
    'Some variables have perfect pairwise correlation across all'
    ' times. For each pair of perfectly-correlated'
    ' variables, please remove one of the variables from the'
    ' model.'
)
PAIRS_WITH_PERFECT_CORRELATION = 'Pairs with perfect correlation: '
MULTICOLLINEARITY_GEO_ATTENTION = (
    'Some variables have extreme multicollinearity (with VIF >'
    ' {geo_threshold}) in certain geo(s). Consider checking your'
    ' data, and/or combining these variables if they also have'
    ' high VIF in other geos.'
)
MULTICOLLINEARITY_OVERALL_ERROR = (
    'Some variables have extreme multicollinearity (VIF'
    ' >{overall_threshold}) across all times and geos. To'
    ' address multicollinearity, please drop any variable that'
    ' is a linear combination of other variables. Otherwise,'
    ' consider combining variables.'
)
MULTICOLLINEARITY_NATIONAL_ERROR = (
    'Some variables have extreme multicollinearity (with VIF >'
    ' {national_threshold}) across all times. To address'
    ' multicollinearity, please drop any variable that is a'
    ' linear combination of other variables. Otherwise, consider'
    ' combining variables.'
)
VARIABLES_WITH_EXTREME_VIF = 'Variables with extreme VIF: '

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
