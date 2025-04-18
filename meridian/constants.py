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

"""Constants shared across the Meridian library."""

import immutabledict

# HEX color map.
BLACK_100 = '#000000'

BLUE_700 = '#1967D2'
BLUE_600 = '#1A73E8'
BLUE_400 = '#669DF6'
BLUE_500 = '#4285F4'
BLUE_300 = '#8AB4F8'
BLUE_200 = '#AECBFA'

YELLOW_600 = '#F9AB00'
YELLOW_500 = '#FBBC04'

GREEN_700 = '#188038'
GREEN_500 = '#34A853'
GREEN_300 = '#81C995'

PURPLE_500 = '#AF5CF7'

RED_600 = '#D93025'
RED_300 = '#F28B82'
RED_200 = '#FC645B'

CYAN_500 = '#24C1E0'
CYAN_400 = '#4ECDE6'

GREY_900 = '#202124'
GREY_800 = '#3C4043'
GREY_700 = '#5F6368'
GREY_600 = '#C3C7C9'
GREY_300 = '#DADCE0'


# Example: "2024-01-09"
DATE_FORMAT = '%Y-%m-%d'
# Example: "2024 Apr"
QUARTER_FORMAT = '%Y %b'

# Input data variables.
KPI = 'kpi'
REVENUE_PER_KPI = 'revenue_per_kpi'
MEDIA = 'media'
MEDIA_SPEND = 'media_spend'
CONTROLS = 'controls'
POPULATION = 'population'
REACH = 'reach'
FREQUENCY = 'frequency'
RF_SPEND = 'rf_spend'
ORGANIC_MEDIA = 'organic_media'
ORGANIC_REACH = 'organic_reach'
ORGANIC_FREQUENCY = 'organic_frequency'
NON_MEDIA_TREATMENTS = 'non_media_treatments'
REVENUE = 'revenue'
NON_REVENUE = 'non_revenue'
REQUIRED_INPUT_DATA_ARRAY_NAMES = (
    KPI,
    CONTROLS,
    POPULATION,
)
OPTIONAL_INPUT_DATA_ARRAY_NAMES = (
    REVENUE_PER_KPI,
    ORGANIC_MEDIA,
    ORGANIC_REACH,
    ORGANIC_FREQUENCY,
    NON_MEDIA_TREATMENTS,
)
MEDIA_INPUT_DATA_ARRAY_NAMES = (
    MEDIA,
    MEDIA_SPEND,
)
RF_INPUT_DATA_ARRAY_NAMES = (
    REACH,
    FREQUENCY,
    RF_SPEND,
)
POSSIBLE_INPUT_DATA_ARRAY_NAMES = (
    REQUIRED_INPUT_DATA_ARRAY_NAMES
    + OPTIONAL_INPUT_DATA_ARRAY_NAMES
    + MEDIA_INPUT_DATA_ARRAY_NAMES
    + RF_INPUT_DATA_ARRAY_NAMES
)
PAID_CHANNELS = (MEDIA, REACH, FREQUENCY)
PAID_DATA = PAID_CHANNELS + (REVENUE_PER_KPI,)
NON_PAID_DATA = (
    ORGANIC_MEDIA,
    ORGANIC_REACH,
    ORGANIC_FREQUENCY,
    NON_MEDIA_TREATMENTS,
)
SPEND_DATA = (
    MEDIA_SPEND,
    RF_SPEND,
)
PERFORMANCE_DATA = PAID_DATA + SPEND_DATA
IMPRESSIONS_DATA = PAID_CHANNELS + NON_PAID_DATA
RF_DATA = (
    REACH,
    FREQUENCY,
    RF_SPEND,
    REVENUE_PER_KPI,
)
NON_REVENUE_DATA = IMPRESSIONS_DATA + (CONTROLS,)

# Scaled input data variables.
MEDIA_SCALED = 'media_scaled'
REACH_SCALED = 'reach_scaled'
ORGANIC_MEDIA_SCALED = 'organic_media_scaled'
ORGANIC_REACH_SCALED = 'organic_reach_scaled'
NON_MEDIA_TREATMENTS_SCALED = 'non_media_treatments_scaled'
CONTROLS_SCALED = 'controls_scaled'

# Non-media treatments baseline value constants.
NON_MEDIA_BASELINE_MIN = 'min'
NON_MEDIA_BASELINE_MAX = 'max'
NON_MEDIA_BASELINE_VALUES = 'non_media_baseline_values'

# Input data coordinates.
GEO = 'geo'
TIME = 'time'
MEDIA_TIME = 'media_time'
MEDIA_CHANNEL = 'media_channel'
RF_CHANNEL = 'rf_channel'
CHANNEL = 'channel'
RF = 'rf'
ORGANIC_MEDIA_CHANNEL = 'organic_media_channel'
ORGANIC_RF_CHANNEL = 'organic_rf_channel'
NON_MEDIA_CHANNEL = 'non_media_channel'
CONTROL_VARIABLE = 'control_variable'
REQUIRED_INPUT_DATA_COORD_NAMES = (
    GEO,
    TIME,
    MEDIA_TIME,
    CONTROL_VARIABLE,
)
NON_PAID_MEDIA_INPUT_DATA_COORD_NAMES = (
    ORGANIC_MEDIA_CHANNEL,
    ORGANIC_RF_CHANNEL,
    NON_MEDIA_CHANNEL,
)
MEDIA_INPUT_DATA_COORD_NAMES = (MEDIA_CHANNEL,)
RF_INPUT_DATA_COORD_NAMES = (RF_CHANNEL,)
POSSIBLE_INPUT_DATA_COORD_NAMES = (
    REQUIRED_INPUT_DATA_COORD_NAMES
    + NON_PAID_MEDIA_INPUT_DATA_COORD_NAMES
    + MEDIA_INPUT_DATA_COORD_NAMES
    + RF_INPUT_DATA_COORD_NAMES
)
POSSIBLE_INPUT_DATA_COORDS_AND_ARRAYS_SET = frozenset(
    POSSIBLE_INPUT_DATA_COORD_NAMES + POSSIBLE_INPUT_DATA_ARRAY_NAMES
)


# National model constants.
NATIONAL = 'national'
NATIONAL_MODEL_DEFAULT_GEO_NAME = 'national_geo'
NATIONAL_MODEL_DEFAULT_POPULATION_VALUE = 1.0


# Media random effects distributions.
MEDIA_EFFECTS_NORMAL = 'normal'
MEDIA_EFFECTS_LOG_NORMAL = 'log_normal'
MEDIA_EFFECTS_DISTRIBUTIONS = frozenset(
    {MEDIA_EFFECTS_NORMAL, MEDIA_EFFECTS_LOG_NORMAL}
)


# Model spec variables.
PRIOR = 'prior'
MEDIA_EFFECTS_DIST = 'media_effects_dist'
HILL_BEFORE_ADSTOCK = 'hill_before_adstock'
ADSTOCK_MEMORY_OPTIMIZED = 'adstock_memory_optimized'
MAX_LAG = 'max_lag'
UNIQUE_SIGMA_FOR_EACH_GEO = 'unique_sigma_for_each_geo'
ROI_CALIBRATION_PERIOD = 'roi_calibration_period'
RF_ROI_CALIBRATION_PERIOD = 'rf_roi_calibration_period'
KNOTS = 'knots'
BASELINE_GEO = 'baseline_geo'

# Treatment prior types.
TREATMENT_PRIOR_TYPE_ROI = 'roi'
TREATMENT_PRIOR_TYPE_MROI = 'mroi'
TREATMENT_PRIOR_TYPE_COEFFICIENT = 'coefficient'
TREATMENT_PRIOR_TYPE_CONTRIBUTION = 'contribution'
PAID_TREATMENT_PRIOR_TYPES = frozenset({
    TREATMENT_PRIOR_TYPE_ROI,
    TREATMENT_PRIOR_TYPE_MROI,
    TREATMENT_PRIOR_TYPE_COEFFICIENT,
    TREATMENT_PRIOR_TYPE_CONTRIBUTION,
})
NON_PAID_TREATMENT_PRIOR_TYPES = frozenset({
    TREATMENT_PRIOR_TYPE_COEFFICIENT,
    TREATMENT_PRIOR_TYPE_CONTRIBUTION,
})
PAID_MEDIA_ROI_PRIOR_TYPES = frozenset(
    {TREATMENT_PRIOR_TYPE_ROI, TREATMENT_PRIOR_TYPE_MROI}
)
# Represents a 1% increase in spend.
MROI_FACTOR = 1.01

NATIONAL_MODEL_SPEC_ARGS = immutabledict.immutabledict(
    {MEDIA_EFFECTS_DIST: MEDIA_EFFECTS_NORMAL, UNIQUE_SIGMA_FOR_EACH_GEO: False}
)

NATIONAL_ANALYZER_PARAMETERS_DEFAULTS = immutabledict.immutabledict(
    {'aggregate_geos': True, 'geos_to_include': None}
)


# Inference data coordinates.
CHAIN = 'chain'
DRAW = 'draw'
KNOTS = 'knots'
SIGMA_DIM = 'sigma_dim'


# Model parameters.
PARAMETER = 'parameter'
KNOT_VALUES = 'knot_values'
MU_T = 'mu_t'
ROI_M = 'roi_m'
ROI_RF = 'roi_rf'
MROI_M = 'mroi_m'
MROI_RF = 'mroi_rf'
CONTRIBUTION_M = 'contribution_m'
CONTRIBUTION_RF = 'contribution_rf'
CONTRIBUTION_OM = 'contribution_om'
CONTRIBUTION_ORF = 'contribution_orf'
CONTRIBUTION_N = 'contribution_n'
GAMMA_C = 'gamma_c'
GAMMA_N = 'gamma_n'
XI_C = 'xi_c'
XI_N = 'xi_n'
ALPHA_M = 'alpha_m'
ALPHA_RF = 'alpha_rf'
EC_M = 'ec_m'
EC_RF = 'ec_rf'
SLOPE_M = 'slope_m'
SLOPE_RF = 'slope_rf'
ETA_M = 'eta_m'
ETA_RF = 'eta_rf'
BETA_M = 'beta_m'
BETA_RF = 'beta_rf'
BETA_GM = 'beta_gm'
BETA_GRF = 'beta_grf'
BETA_OM = 'beta_om'
BETA_ORF = 'beta_orf'
ETA_OM = 'eta_om'
ETA_ORF = 'eta_orf'
ALPHA_OM = 'alpha_om'
ALPHA_ORF = 'alpha_orf'
EC_OM = 'ec_om'
EC_ORF = 'ec_orf'
SLOPE_OM = 'slope_om'
SLOPE_ORF = 'slope_orf'
BETA_GOM = 'beta_gom'
BETA_GORF = 'beta_gorf'
SIGMA = 'sigma'
TAU_G = 'tau_g'
TAU_G_EXCL_BASELINE = 'tau_g_excl_baseline'
GAMMA_GC = 'gamma_gc'
GAMMA_GN = 'gamma_gn'
BETA_GM_DEV = 'beta_gm_dev'
BETA_GRF_DEV = 'beta_grf_dev'
BETA_GOM_DEV = 'beta_gom_dev'
BETA_GORF_DEV = 'beta_gorf_dev'
GAMMA_GC_DEV = 'gamma_gc_dev'
GAMMA_GN_DEV = 'gamma_gn_dev'
COMMON_PARAMETER_NAMES = (
    KNOT_VALUES,
    MU_T,
    GAMMA_C,
    XI_C,
    SIGMA,
    TAU_G,
    GAMMA_GC,
)
# These constants are only used in unit tests for mocking default inference data
# which doesn't include MROI priors.
MEDIA_PARAMETER_NAMES = (
    ROI_M,
    ALPHA_M,
    EC_M,
    SLOPE_M,
    ETA_M,
    BETA_M,
    BETA_GM,
)
RF_PARAMETER_NAMES = (
    ROI_RF,
    ALPHA_RF,
    EC_RF,
    SLOPE_RF,
    ETA_RF,
    BETA_RF,
    BETA_GRF,
)

MEDIA_PARAMETERS = (
    ROI_M,
    MROI_M,
    CONTRIBUTION_M,
    BETA_M,
    ETA_M,
    ALPHA_M,
    EC_M,
    SLOPE_M,
)
RF_PARAMETERS = (
    ROI_RF,
    MROI_RF,
    CONTRIBUTION_RF,
    BETA_RF,
    ETA_RF,
    ALPHA_RF,
    EC_RF,
    SLOPE_RF,
)
ORGANIC_MEDIA_PARAMETERS = (
    CONTRIBUTION_OM,
    BETA_OM,
    ETA_OM,
    ALPHA_OM,
    EC_OM,
    SLOPE_OM,
)
ORGANIC_RF_PARAMETERS = (
    CONTRIBUTION_ORF,
    BETA_ORF,
    ETA_ORF,
    ALPHA_ORF,
    EC_ORF,
    SLOPE_ORF,
)
NON_MEDIA_PARAMETERS = (
    CONTRIBUTION_N,
    GAMMA_N,
    XI_N,
)

KNOTS_PARAMETERS = (KNOT_VALUES,)
CONTROL_PARAMETERS = (GAMMA_C, XI_C)
SIGMA_PARAMETERS = (SIGMA,)
GEO_PARAMETERS = (
    # TAU_G in InferenceData is derived from TAU_G_EXCL_BASELINE in the priors.
    TAU_G,
)
TIME_PARAMETERS = (
    # MU_T in InferenceData is derived from KNOT_VALUES in the priors.
    MU_T,
)
GEO_MEDIA_PARAMETERS = (BETA_GM,)
GEO_RF_PARAMETERS = (BETA_GRF,)
GEO_CONTROL_PARAMETERS = (GAMMA_GC,)
GEO_NON_MEDIA_PARAMETERS = (GAMMA_GN,)

ALL_PRIOR_DISTRIBUTION_PARAMETERS = (
    *KNOTS_PARAMETERS,
    *MEDIA_PARAMETERS,
    *RF_PARAMETERS,
    *ORGANIC_MEDIA_PARAMETERS,
    *ORGANIC_RF_PARAMETERS,
    *NON_MEDIA_PARAMETERS,
    *CONTROL_PARAMETERS,
    *SIGMA_PARAMETERS,
    TAU_G_EXCL_BASELINE,
    *TIME_PARAMETERS,
)

UNSAVED_PARAMETERS = (
    BETA_GM_DEV,
    BETA_GRF_DEV,
    BETA_GOM_DEV,
    BETA_GORF_DEV,
    GAMMA_GC_DEV,
    GAMMA_GN_DEV,
    TAU_G_EXCL_BASELINE,  # Used to derive TAU_G.
)
IGNORED_PRIORS_MEDIA = immutabledict.immutabledict({
    TREATMENT_PRIOR_TYPE_ROI: (
        BETA_M,
        MROI_M,
        CONTRIBUTION_M,
    ),
    TREATMENT_PRIOR_TYPE_MROI: (
        BETA_M,
        ROI_M,
        CONTRIBUTION_M,
    ),
    TREATMENT_PRIOR_TYPE_CONTRIBUTION: (
        BETA_M,
        ROI_M,
        MROI_M,
    ),
    TREATMENT_PRIOR_TYPE_COEFFICIENT: (
        ROI_M,
        MROI_M,
        CONTRIBUTION_M,
    ),
})
IGNORED_PRIORS_RF = immutabledict.immutabledict({
    TREATMENT_PRIOR_TYPE_ROI: (
        BETA_RF,
        MROI_RF,
        CONTRIBUTION_RF,
    ),
    TREATMENT_PRIOR_TYPE_MROI: (
        BETA_RF,
        ROI_RF,
        CONTRIBUTION_RF,
    ),
    TREATMENT_PRIOR_TYPE_CONTRIBUTION: (
        BETA_RF,
        ROI_RF,
        MROI_RF,
    ),
    TREATMENT_PRIOR_TYPE_COEFFICIENT: (
        ROI_RF,
        MROI_RF,
        CONTRIBUTION_RF,
    ),
})
IGNORED_PRIORS_ORGANIC_MEDIA = immutabledict.immutabledict({
    TREATMENT_PRIOR_TYPE_CONTRIBUTION: (BETA_OM,),
    TREATMENT_PRIOR_TYPE_COEFFICIENT: (CONTRIBUTION_OM,),
})
IGNORED_PRIORS_ORGANIC_RF = immutabledict.immutabledict({
    TREATMENT_PRIOR_TYPE_CONTRIBUTION: (BETA_ORF,),
    TREATMENT_PRIOR_TYPE_COEFFICIENT: (CONTRIBUTION_ORF,),
})
IGNORED_PRIORS_NON_MEDIA_TREATMENTS = immutabledict.immutabledict({
    TREATMENT_PRIOR_TYPE_CONTRIBUTION: (GAMMA_N,),
    TREATMENT_PRIOR_TYPE_COEFFICIENT: (CONTRIBUTION_N,),
})

# Inference data dimensions.
INFERENCE_DIMS = immutabledict.immutabledict(
    {
        MU_T: (TIME,),
        KNOT_VALUES: (KNOTS,),
        TAU_G: (GEO,),
        BETA_GM: (GEO, MEDIA_CHANNEL),
        BETA_GRF: (GEO, RF_CHANNEL),
        BETA_GOM: (GEO, ORGANIC_MEDIA_CHANNEL),
        BETA_GORF: (GEO, ORGANIC_RF_CHANNEL),
        GAMMA_GC: (GEO, CONTROL_VARIABLE),
        GAMMA_GN: (GEO, NON_MEDIA_CHANNEL),
    }
    | {param: (CONTROL_VARIABLE,) for param in CONTROL_PARAMETERS}
    | {param: (NON_MEDIA_CHANNEL,) for param in NON_MEDIA_PARAMETERS}
    | {param: (MEDIA_CHANNEL,) for param in MEDIA_PARAMETERS}
    | {param: (RF_CHANNEL,) for param in RF_PARAMETERS}
    | {param: (ORGANIC_MEDIA_CHANNEL,) for param in ORGANIC_MEDIA_PARAMETERS}
    | {param: (ORGANIC_RF_CHANNEL,) for param in ORGANIC_RF_PARAMETERS}
)

IGNORED_TRACE_METRICS = ('variance_scaling',)

STEP_SIZE = 'step_size'
TUNE = 'tune'
TARGET_LOG_PROBABILITY_TF = 'target_log_prob'
TARGET_LOG_PROBABILITY_ARVIZ = 'lp'
DIVERGING = 'diverging'
ACCEPT_RATIO = 'accept_ratio'
N_STEPS = 'n_steps'
SAMPLE_SHAPE = 'sample_shape'
SEED = 'seed'

SAMPLE_STATS_METRICS = immutabledict.immutabledict({
    STEP_SIZE: STEP_SIZE,
    TARGET_LOG_PROBABILITY_TF: TARGET_LOG_PROBABILITY_ARVIZ,
    DIVERGING: DIVERGING,
    N_STEPS: N_STEPS,
})


# Adstock hill functions.
ADSTOCK_HILL_FUNCTIONS = frozenset({
    'adstock_memory_optimized',
    'adstock_speed_optimized',
    'hill',
})


# Distribution constants.
DISTRIBUTION = 'distribution'
DISTRIBUTION_TYPE = 'distribution_type'
PRIOR = 'prior'
POSTERIOR = 'posterior'
# Prior mean proportion of KPI incremental due to all media.
P_MEAN = 0.4
# Prior standard deviation proportion of KPI incremental to all media.
P_SD = 0.2


# Model metrics.
METRIC = 'metric'
MEAN = 'mean'
MEDIAN = 'median'
CI_LO = 'ci_lo'
CI_HI = 'ci_hi'
RHAT = 'rhat'
PCT = 'pct'
CONFIDENCE_LEVEL = 'confidence_level'


# Model fit types.
TYPE = 'type'
EXPECTED = 'expected'
ACTUAL = 'actual'
BASELINE = 'baseline'

# Model fit filtering.
GEO_GRANULARITY = 'geo_granularity'
EVALUATION_SET_VAR = 'evaluation_set'
VALUE = 'value'

# Predictive accuracy metrics.
R_SQUARED = 'R_Squared'
MAPE = 'MAPE'
WMAPE = 'wMAPE'

# Data splitting types.
TRAIN = 'Train'
TEST = 'Test'
ALL_DATA = 'All Data'

# Model fit sets.
EVALUATION_SET = (
    TRAIN,
    TEST,
    ALL_DATA,
)
GEO_GRANULARITY_SET = (GEO, NATIONAL)
METRICS_SET = (R_SQUARED, MAPE, WMAPE)
COLUMN_VAR_SET = (None, METRIC, GEO_GRANULARITY, EVALUATION_SET)

# Media effects metrics.
SPEND_MULTIPLIER = 'spend_multiplier'
TIME_UNITS = 'time_units'
MEDIA_UNITS = 'media_units'
HILL_SATURATION_LEVEL = 'hill_saturation_level'
OPTIMAL_FREQUENCY = 'optimal_frequency'
EXPECTED_ROI = 'expected_roi'
EFFECT = 'effect'

# Media effects helper values
MEDIA_UNITS_COUNT_HISTOGRAM = 'media_units_count_histogram'
SCALED_COUNT_HISTOGRAM = 'scaled_count_histogram'
START_INTERVAL_HISTOGRAM = 'start_interval_histogram'
END_INTERVAL_HISTOGRAM = 'end_interval_histogram'
COUNT_HISTOGRAM = 'count_histogram'
CHANNEL_TYPE = 'channel_type'
IS_INT_TIME_UNIT = 'is_int_time_unit'
OPTIMAL_FREQ = 'optimal_freq'
CURRENT_SPEND = 'current_spend'

# Media summary metrics.
SPEND = 'spend'
IMPRESSIONS = 'impressions'
ROI = 'roi'
OPTIMIZED_ROI = 'optimized_roi'
MROI = 'mroi'
OPTIMIZED_MROI_BY_REACH = 'optimized_mroi_by_reach'
OPTIMIZED_MROI_BY_FREQUENCY = 'optimized_mroi_by_frequency'
CPIK = 'cpik'
OPTIMIZED_CPIK = 'optimized_cpik'
ROI_SCALED = 'roi_scaled'
OUTCOME = 'outcome'
INCREMENTAL_OUTCOME = 'incremental_outcome'
BASELINE_OUTCOME = 'baseline_outcome'
OPTIMIZED_INCREMENTAL_OUTCOME = 'optimized_incremental_outcome'
EFFECTIVENESS = 'effectiveness'
OPTIMIZED_EFFECTIVENESS = 'optimized_effectiveness'
PCT_OF_IMPRESSIONS = 'pct_of_impressions'
PCT_OF_SPEND = 'pct_of_spend'
CPM = 'cpm'
PCT_OF_CONTRIBUTION = 'pct_of_contribution'
OPTIMIZED_PCT_OF_CONTRIBUTION = 'optimized_pct_of_contribution'
BUDGET = 'budget'
PROFIT = 'profit'
IS_REVENUE_KPI = 'is_revenue_kpi'
TOTAL_INCREMENTAL_OUTCOME = 'total_incremental_outcome'
TOTAL_ROI = 'total_roi'
TOTAL_CPIK = 'total_cpik'
USE_KPI = 'use_kpi'
USE_POSTERIOR = 'use_posterior'

# R-hat summary metrics.
PARAM = 'param'
N_PARAMS = 'n_params'
AVG_RHAT = 'avg_rhat'
MAX_RHAT = 'max_rhat'
PERCENT_BAD_RHAT = 'percent_bad_rhat'
ROW_IDX_BAD_RHAT = 'row_idx_bad_rhat'
COL_IDX_BAD_RHAT = 'col_idx_bad_rhat'


# Analyzer Parameters.
NEW_MEDIA = 'new_media'
NEW_MEDIA_SPEND = 'new_media_spend'
NEW_REACH = 'new_reach'
NEW_FREQUENCY = 'new_frequency'
NEW_RF_SPEND = 'new_rf_spend'
NEW_ORGANIC_MEDIA = 'new_organic_media'
NEW_ORGANIC_REACH = 'new_organic_reach'
NEW_ORGANIC_FREQUENCY = 'new_organic_frequency'
NEW_NON_MEDIA_TREATMENTS = 'new_non_media_treatments'
NEW_CONTROLS = 'new_controls'
NEW_DATA = 'new_data'


# Media types.
ALL_CHANNELS = 'All Channels'

# Optimization constants.
OPTIMIZED = 'optimized'
NON_OPTIMIZED = 'non_optimized'
SPEND_CONSTRAINT = 'spend_constraint'
SPEND_LEVEL = 'spend_level'
LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'
SPEND_GRID = 'spend_grid'
SPEND_STEP_SIZE = 'spend_step_size'
INCREMENTAL_OUTCOME_GRID = 'incremental_outcome_grid'
GRID_SPEND_INDEX = 'grid_spend_index'
USE_HISTORICAL_BUDGET = 'use_historical_budget'


# Optimization constraints.
FIXED_BUDGET = 'fixed_budget'
TARGET_ROI = 'target_roi'
TARGET_MROI = 'target_mroi'
SPEND_CONSTRAINT_DEFAULT_FIXED_BUDGET = 0.3
SPEND_CONSTRAINT_DEFAULT_FLEXIBLE_BUDGET = 1.0
SPEND_CONSTRAINT_DEFAULT = 1.0


# Plot constants.
BAR_SIZE = 42
PADDING_10 = 10
PADDING_20 = 20
AXIS_FONT_SIZE = 12
TEXT_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
SCALED_PADDING = 0.2
CORNER_RADIUS = 2
STROKE_DASH = (4, 2)
POINT_SIZE = 80
INDEPENDENT = 'independent'
RESPONSE_CURVE_STEP_SIZE = 0.01


# Font names.
FONT_ROBOTO = 'Roboto'
FONT_GOOGLE_SANS_DISPLAY = 'Google Sans Display'

# Default confidence level for the analysis.
DEFAULT_CONFIDENCE_LEVEL = 0.9

# Default number of max draws per chain in Analyzer.expected_outcome()
DEFAULT_BATCH_SIZE = 100


# Optimization constants.
CHAINS_DIMENSION = 0
DRAWS_DIMENSION = 1
SPEND_CONSTRAINT_PADDING = 0.5


# Range constants for defining the step_size for the x-axis in select plots.
ADSTOCK_DECAY_DEFAULT_STEPS_PER_TIME_PERIOD = 5
ADSTOCK_DECAY_MAX_TOTAL_STEPS = 1000
HILL_NUM_STEPS = 500

# Summary template params.
START_DATE = 'start_date'
END_DATE = 'end_date'
CARD_INSIGHTS = 'insights'
CARD_CHARTS = 'charts'
CARD_STATS = 'stats'

# VegaLite common params.
VEGALITE_FACET_DEFAULT_WIDTH = 400
VEGALITE_FACET_LARGE_WIDTH = 500
VEGALITE_FACET_EXTRA_LARGE_WIDTH = 700

# Time Granularity Constants
WEEKLY = 'weekly'
QUARTERLY = 'quarterly'
TIME_GRANULARITIES = frozenset({WEEKLY, QUARTERLY})
QUARTERLY_SUMMARY_THRESHOLD_WEEKS = 52
