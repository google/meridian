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

"""Dataframe converter constants."""

SHEET_NAME_DELIMITER = "_"

# Special analysis aggregation tags.
ANALYSIS_TAG_ALL = "ALL"

# ModelFit table column names
MODEL_FIT = "ModelFit"
MODEL_FIT_TIME_COLUMN = "Time"
MODEL_FIT_EXPECTED_CI_LOW_COLUMN = "Expected CI Low"
MODEL_FIT_EXPECTED_CI_HIGH_COLUMN = "Expected CI High"
MODEL_FIT_EXPECTED_COLUMN = "Expected"
MODEL_FIT_BASELINE_COLUMN = "Baseline"
MODEL_FIT_ACTUAL_COLUMN = "Actual"

# ModelDiagnostics table column names
MODEL_DIAGNOSTICS = "ModelDiagnostics"
MODEL_DIAGNOSTICS_DATASET_COLUMN = "Dataset"
MODEL_DIAGNOSTICS_R_SQUARED_COLUMN = "R Squared"
MODEL_DIAGNOSTICS_MAPE_COLUMN = "MAPE"
MODEL_DIAGNOSTICS_WMAPE_COLUMN = "wMAPE"

# Common column names
ANALYSIS_PERIOD_COLUMN = "Analysis Period"
ANALYSIS_DATE_START_COLUMN = "Analysis Date Start"
ANALYSIS_DATE_END_COLUMN = "Analysis Date End"

# MediaOutcome table column names
MEDIA_OUTCOME = "MediaOutcome"
MEDIA_OUTCOME_CHANNEL_INDEX_COLUMN = "Channel Index"
MEDIA_OUTCOME_CHANNEL_COLUMN = "Channel"
MEDIA_OUTCOME_INCREMENTAL_OUTCOME_COLUMN = "Incremental Outcome"
MEDIA_OUTCOME_CONTRIBUTION_SHARE_COLUMN = "Contribution Share"
MEDIA_OUTCOME_BASELINE_PSEUDO_CHANNEL_INDEX = 0
MEDIA_OUTCOME_ALL_CHANNELS_PSEUDO_CHANNEL_INDEX = 1
MEDIA_OUTCOME_CHANNEL_INDEX = 2

# MediaSpend table column names
MEDIA_SPEND = "MediaSpend"
MEDIA_SPEND_CHANNEL_COLUMN = "Channel"
MEDIA_SPEND_SHARE_VALUE_COLUMN = "Share Value"
MEDIA_SPEND_LABEL_COLUMN = "Label"
# The "Label" column enums
MEDIA_SPEND_LABEL_SPEND_SHARE = "Spend Share"
MEDIA_SPEND_LABEL_REVENUE_SHARE = "Revenue Share"
MEDIA_SPEND_LABEL_KPI_SHARE = "KPI Share"

# MediaROI table column names
MEDIA_ROI = "MediaROI"
MEDIA_ROI_CHANNEL_COLUMN = "Channel"
MEDIA_ROI_SPEND_COLUMN = "Spend"
MEDIA_ROI_EFFECTIVENESS_COLUMN = "Effectiveness"
MEDIA_ROI_ROI_COLUMN = "ROI"
MEDIA_ROI_ROI_CI_LOW_COLUMN = "ROI CI Low"
MEDIA_ROI_ROI_CI_HIGH_COLUMN = "ROI CI High"
MEDIA_ROI_MARGINAL_ROI_COLUMN = "Marginal ROI"
MEDIA_ROI_IS_REVENUE_KPI_COLUMN = "Is Revenue KPI"


# Shared column names among Optimization tables
OPTIMIZATION_GROUP_ID_COLUMN = "Group ID"
OPTIMIZATION_CHANNEL_COLUMN = "Channel"

# Optimization grid table column names
# (Table name is user-generated from the spec)
OPTIMIZATION_GRID_SPEND_COLUMN = "Spend"
OPTIMIZATION_GRID_INCREMENTAL_OUTCOME_COLUMN = "Incremental Outcome"

# R&F Optimization grid table column names
# (Table name is user-generated from the spec)
RF_OPTIMIZATION_GRID_FREQ_COLUMN = "Frequency"
RF_OPTIMIZATION_GRID_ROI_OUTCOME_COLUMN = "ROI"

# Budget optimization grid table name
OPTIMIZATION_GRID_NAME_PREFIX = "budget_opt_grid"

# R&F optimization grid table name
RF_OPTIMIZATION_GRID_NAME_PREFIX = "rf_opt_grid"

# Optimization spec table column names and enum values
OPTIMIZATION_SPECS = "budget_opt_specs"
OPTIMIZATION_SPEC_DATE_INTERVAL_START_COLUMN = "Date Interval Start"
OPTIMIZATION_SPEC_DATE_INTERVAL_END_COLUMN = "Date Interval End"
OPTIMIZATION_SPEC_OBJECTIVE_COLUMN = "Objective"
OPTIMIZATION_SPEC_SCENARIO_TYPE_COLUMN = "Scenario Type"
OPTIMIZATION_SPEC_SCENARIO_FIXED = "Fixed"
OPTIMIZATION_SPEC_SCENARIO_FLEXIBLE = "Flexible"
OPTIMIZATION_SPEC_INITIAL_CHANNEL_SPEND_COLUMN = "Initial Channel Spend"
OPTIMIZATION_SPEC_TARGET_METRIC_CONSTRAINT_COLUMN = "Target Metric Constraint"
OPTIMIZATION_SPEC_TARGET_METRIC_KPI = "KPI"
OPTIMIZATION_SPEC_TARGET_METRIC_ROI = "ROI"
OPTIMIZATION_SPEC_TARGET_METRIC_MARGINAL_ROI = "Marginal ROI"
OPTIMIZATION_SPEC_TARGET_METRIC_CPIK = "Cost per Incremental KPI"
OPTIMIZATION_SPEC_TARGET_METRIC_VALUE_COLUMN = "Target Metric Value"
OPTIMIZATION_SPEC_CHANNEL_COLUMN = "Channel"
OPTIMIZATION_SPEC_CHANNEL_SPEND_MIN_COLUMN = "Channel Spend Min"
OPTIMIZATION_SPEC_CHANNEL_SPEND_MAX_COLUMN = "Channel Spend Max"

# R&F Optimization spec table column names and enum values
RF_OPTIMIZATION_SPECS = "rf_opt_specs"
RF_OPTIMIZATION_SPEC_CHANNEL_FREQUENCY_MIN_COLUMN = "Channel Frequency Min"
RF_OPTIMIZATION_SPEC_CHANNEL_FREQUENCY_MAX_COLUMN = "Channel Frequency Max"

# Optimization results table column names
OPTIMIZATION_RESULTS = "budget_opt_results"
OPTIMIZATION_RESULT_SPEND_COLUMN = "Optimal Spend"
OPTIMIZATION_RESULT_SPEND_SHARE_COLUMN = "Optimal Spend Share"
OPTIMIZATION_RESULT_EFFECTIVENESS_COLUMN = "Optimal Impression Effectiveness"
OPTIMIZATION_RESULT_ROI_COLUMN = "Optimal ROI"
OPTIMIZATION_RESULT_MROI_COLUMN = "Optimal mROI"
OPTIMIZATION_RESULT_CPC_COLUMN = "Optimal CPC"
OPTIMIZATION_RESULT_IS_REVENUE_KPI_COLUMN = "Is Revenue KPI"

# R&F Optimization results table column names
RF_OPTIMIZATION_RESULTS = "rf_opt_results"
RF_OPTIMIZATION_RESULT_INITIAL_SPEND_COLUMN = "Initial Spend"
RF_OPTIMIZATION_RESULT_AVG_FREQ_COLUMN = "Optimal Avg Frequency"

# Optimization results' response curves table column names
OPTIMIZATION_RESPONSE_CURVES = "response_curves"
OPTIMIZATION_RESPONSE_CURVE_SPEND_COLUMN = "Spend"
OPTIMIZATION_RESPONSE_CURVE_INCREMENTAL_OUTCOME_COLUMN = "Incremental Outcome"
