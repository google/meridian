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

# Report constants
REPORT_TITLE = 'Meridian Exploratory Data Analysis Report'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID = 'relationship-among-variables'
RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE = 'Relationship Among the Variables'
PAIRWISE_CORRELATION_CHART_ID = 'pairwise-correlation-chart'


# Finding messages
PAIRWISE_CORRELATION_CHECK_INFO = (
    'Please review the computed pairwise correlations. Note that'
    ' high pairwise correlation may cause model identifiability'
    ' and convergence issues. Consider combining the variables if'
    ' high correlation exists.'
)
