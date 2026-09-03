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

"""Constants for model review."""

import immutabledict

from meridian.model.eda import constants as eda_constants


RHAT = "rhat"
PARAMETER = "parameter"
CONVERGENCE_THRESHOLD = "convergence_threshold"
CHANNELS_LOW_HIGH = "channels_low_high"
PRIOR_ROI_LO = "prior_roi_lo"
PRIOR_ROI_HI = "prior_roi_hi"
POSTERIOR_ROI_MEAN = "posterior_roi_mean"
QUANTILE_NOT_DEFINED_MSG = "quantile_not_defined_msg"
INF_CHANNELS_MSG = "inf_channels_msg"
LOW_HIGH_CHANNELS_MSG = "low_high_channels_msg"
NEGATIVE_BASELINE_PROB = "negative_baseline_prob"
NEGATIVE_BASELINE_PROB_FAIL_THRESHOLD = "negative_baseline_prob_fail_threshold"
NEGATIVE_BASELINE_PROB_REVIEW_THRESHOLD = (
    "negative_baseline_prob_review_threshold"
)
R_SQUARED = "r_squared"
MAPE = "mape"
WMAPE = "wmape"
ALL_SUFFIX = ""
TRAIN_SUFFIX = "_train"
TEST_SUFFIX = "_test"
NAME = "name"
STATUS = "status"
RECOMMENDATION = "recommendation"
TOTAL_CHANNELS = "total_channels"
PASSED_CHANNELS = "passed_channels"
EVALUATION_SET_SUFFIXES = (ALL_SUFFIX, TRAIN_SUFFIX, TEST_SUFFIX)
MEAN = "mean"
VARIANCE = "variance"
MEDIAN = "median"
Q1 = "q1"
Q3 = "q3"
BAYESIAN_PPP = "bayesian_ppp"
CHANNELS_STR = "channels_str"
SPEND_SHARE = "spend_share"
ROI_MEAN = "roi_mean"
ROI_MEDIAN = "roi_median"
SPEND_WEIGHTED_ROI = "spend_weighted_roi"
CORRELATION_MATRIX = "correlation_matrix"
PRIOR_RELATIVE_HDI_WIDTH_FOR_80_PERCENT = 2.0687

CHECK_RESULT_NAME_MAP = immutabledict.immutabledict({
    "ConvergenceCheckResult": "Convergence",
    "BaselineCheckResult": "Baseline",
    "GoodnessOfFitCheckResult": "Goodness of fit",
    "BayesianPPPCheckResult": "Bayesian p-value",
    "PriorPosteriorShiftCheckResult": "Prior-posterior shift",
    "ROIConsistencyCheckResult": "ROI consistency",
    "ImplausibleROICheckResult": "Implausible ROI",
    "HighVarianceCheckResult": "High-variance ROI",
    "PotentialBiasCheckResult": "Potential bias",
})

# Health score constants
R2_MIDPOINT = 0.5
R2_STEEPNESS = 15
FAIL_RATIO_POWER = 0.4
HEALTH_SCORE_WEIGHT_BASELINE = 0.3
HEALTH_SCORE_WEIGHT_BAYESIAN_PPP = 0.3
HEALTH_SCORE_WEIGHT_GOF = 0.1
HEALTH_SCORE_WEIGHT_PRIOR_POSTERIOR_SHIFT = 0.15
HEALTH_SCORE_WEIGHT_ROI_CONSISTENCY = 0.15

IMPLAUSIBLE_ROI_RECOMMENDATION = (
    "Please review these channels to determine if the ROI estimates are "
    "reasonable within your business context. Consider calibrating with an "
    "incrementality experiment to improve accuracy."
)

HIGH_VARIANCE_ROI_RECOMMENDATION = (
    "We recommend calibrating these channels using an incrementality experiment"
    " to reduce posterior ROI uncertainty."
)

POTENTIAL_BIAS_RECOMMENDATION = (
    "Some channels have low correlation with all controls. These channels might"
    " have biased posterior estimates due to missing potential confounders."
    " We recommend checking if important controls are missing or calibrating"
    " these channels using an incrementality experiment to address this."
)

# Chart and table channel colors
CHANNEL_COLORS: tuple[str, ...] = (
    "#185abc",
    "#b31412",
    "#ea8600",
    "#137333",
    "#c26401",
    "#b80672",
    "#7627bb",
    "#098591",
    "#669df6",
    "#ee675c",
    "#fcc934",
    "#5bb974",
    "#fa903e",
    "#ff63b8",
    "#af5cf7",
    "#4ecde6",
    "#8f4e06",
    "#041e49",
    "#c4cce1",
    "#8d0053",
)

MAX_CHANNELS_FOR_CALIBRATED_DISPLAY = 20
MAX_CHANNELS_FOR_DETAILS_CARD = 5
MAX_EXPERIMENTS_FOR_DETAILS_CARD = 5
MAX_CHANNELS_FOR_OVERVIEW_CARD = 5
MAX_EXPERIMENTS_FOR_OVERVIEW_CARD = 5
CHANNEL_NAME = "channel_name"
IS_CALIBRATED = "is_calibrated"
HIGH_ROI_STATUS = "high_roi_status"
LOW_ROI_STATUS = "low_roi_status"
HIGH_VARIANCE_STATUS = "high_variance_status"
POTENTIAL_BIAS_STATUS = "potential_bias_status"

# Calibration score constants
EPSILON = 1e-9
CALIBRATION_IMPLAUSIBLE_ROI_WEIGHT = 0.5
CALIBRATION_HIGH_VARIANCE_WEIGHT = 0.25
CALIBRATION_POTENTIAL_BIAS_WEIGHT = 0.25
CALIBRATED_CHANNEL_SCORE = 100.0
HIGH_VARIANCE_IDEAL_THRESHOLD = 0.5
CALIBRATION_SCORE_THRESHOLD = 67.5
CALIBRATION_SCORE_YELLOW_COLOR = "#fcc934"
CALIBRATION_SCORE = "calibration_score"

# Channel calibration recommendation message constants
DRIVER = "Driver"
NON_DRIVER = "Non-Driver"
NO_CHANNELS_REQUIRE_CALIBRATION = "No channels require calibration."
SEE_CHANNEL_CALIBRATION_RECOMMENDATION_BELOW = (
    "See Channel calibration recommendation below for more details."
)
REVIEW_BOUNDARIES_INFO_TEXT = (
    "We recommend reviewing the table and plots below to check for channels"
    " near the boundaries that may be good candidates for"
    " calibration via an incrementality experiment such as those run with"
    " Meridian GeoX."
)
NO_CHANNELS_REQUIRE_CALIBRATION_RECOMMENDATION = (
    f"{NO_CHANNELS_REQUIRE_CALIBRATION} {REVIEW_BOUNDARIES_INFO_TEXT}"
)
CALIBRATION_TEXT_METRICS_CHECK = "metrics_check"
CALIBRATION_TEXT_CALIBRATION_SUMMARY = "calibration_summary"
CALIBRATION_TEXT_CHANNEL_RECOMMENDATION = "channel_recommendation"

HIGH_ROI = "high ROI"
LOW_ROI = "low ROI"
IMPLAUSIBLE_ROI = "implausible ROI"
HIGH_VARIANCE = "high variance"
POTENTIAL_BIAS = "potential bias"
# Chart scaling and threshold constants
IMPLAUSIBLE_ROI_THRESHOLD_LOWER = 0.6
IMPLAUSIBLE_ROI_SCALE_FACTOR = 19.0 / IMPLAUSIBLE_ROI_THRESHOLD_LOWER
IMPLAUSIBLE_ROI_GAP_PLOTTED = 19.0
IMPLAUSIBLE_ROI_MAX_PLOTTED = 100.0
HIGH_VARIANCE_RCI_MAX_PLOTTED = 10.0

# Chart color hex codes
IMPLAUSIBLE_ROI_UPPER_COLOR = "#1967d2"
IMPLAUSIBLE_ROI_LOWER_COLOR = "#f2994a"
IMPLAUSIBLE_ROI_LOWER_LINE_COLOR = "#f2994a"
IMPLAUSIBLE_ROI_BREAK_TEXT_COLOR = "#666666"
IMPLAUSIBLE_ROI_BREAK_MARK_COLOR = "#666666"
HIGH_VARIANCE_UPPER_COLOR = "#1967d2"
POTENTIAL_BIAS_RECT_COLOR = "#1967d2"
POTENTIAL_BIAS_RULE_COLOR = "#ea4335"
POTENTIAL_BIAS_THRESHOLD_LINE_COLOR = "#ea4335"
POTENTIAL_BIAS_GEO_POINT_COLOR = "#1976d2"
POTENTIAL_BIAS_FILL_LOW_COLOR = "#fef7e0"
POTENTIAL_BIAS_REVIEW_FILL_COLOR = "#fef7e0"
POTENTIAL_BIAS_FILL_HIGH_COLOR = "#e6f4ea"
POTENTIAL_BIAS_PASS_FILL_COLOR = "#e6f4ea"
POTENTIAL_BIAS_STROKE_LOW_COLOR = "#b06000"
POTENTIAL_BIAS_REVIEW_STROKE_COLOR = "#b06000"
POTENTIAL_BIAS_STROKE_HIGH_COLOR = "#137333"
POTENTIAL_BIAS_PASS_STROKE_COLOR = "#137333"
POTENTIAL_BIAS_MAX_LEGEND_COLOR = "#808080"
POTENTIAL_BIAS_MAX_POINT_COLOR = "#808080"
POSTERIOR_HISTOGRAM_COLOR = "#8ab4f8"

# Chart string keys and encoding field names
CHANNEL_ID = "channel_id"
RELATIVE_WIDTH = "relative_width"
Y_PLOTTED = "y_plotted"
Y2_PLOTTED = "y2_plotted"
LEGEND_LABEL = "legend_label"
REGION = "Region"
Y2 = "y2"
TEXT = "text"
CORRELATION = "correlation"
PAIR = "pair"
ABS_CORRELATION = "abs_correlation"
CHANNEL = "channel"
CONTROL_VARIABLE = "control_variable"
IS_MAX = "is_max"
GEO = "geo"
X1 = "x1"
X2 = "x2"
X = "x"
FILL_COLOR = "fill_color"
STROKE_COLOR = "stroke_color"
LABEL = "label"
CHART_ID = "chart_id"
CHART_JSON = "chart_json"
DETAILS_DESCRIPTION = "details_description"
PLOTTED_CHANNELS = "plotted_channels"

# Plot titles
IMPLAUSIBLE_ROI_PLOT_TITLE = "Spend vs. ROI (Implausible ROI Check)"
HIGH_VARIANCE_PLOT_TITLE = (
    "Spend vs. Relative Credible Interval (High Variance ROI Check)"
)
POTENTIAL_BIAS_PLOT_TITLE = "Correlation with Controls (Potential Bias Check)"
# Chart region labels
HIGH_VARIANCE_ROI = "High-Variance ROI"
IMPLAUSIBLE_HIGH_ROI = "Implausible High ROI"
IMPLAUSIBLE_LOW_ROI = "Implausible Low ROI"

# Chart legend labels and titles
CHANNELS_LEGEND_TITLE = "Channels"
DIAGNOSTIC_THRESHOLDS_TITLE = "Diagnostic Thresholds"
SPEND_PERCENT_TITLE = "Spend %"
ROI_TITLE = "ROI"
RCI_TITLE = "Relative Credible Interval (RCI)"
PEARSON_CORRELATION_TITLE = "Pearson Correlation"
INDIVIDUAL_GEO_CORRELATION = "Individual Geo Correlation"
MAX_ABS_CORRELATION = "Max |Correlation|"
BREAK_MARK_TEXT = "//"

ROI = "roi"
DENSITY = "density"
CALIBRATION_LEFT_PLOT_TITLE = f"1. {eda_constants.CALIBRATION_LEFT_PLOT_TITLE}"
CALIBRATION_MIDDLE_PLOT_TITLE = (
    f"2. {eda_constants.CALIBRATION_RIGHT_PLOT_TITLE}"
)
CALIBRATION_RIGHT_PLOT_TITLE = "3. Calibrated Prior and Meridian Posterior"
CALIBRATION_POSTERIOR_PLOT_TITLE = CALIBRATION_RIGHT_PLOT_TITLE
MERIDIAN_POSTERIOR = "Meridian Posterior"
INTERMEDIARY_PRIOR = eda_constants.INTERMEDIARY_PRIOR
CALIBRATED_MERIDIAN_PRIOR = eda_constants.CALIBRATED_PRIOR
BASELINE_PRIOR = eda_constants.BASELINE_PRIOR
INTERMEDIARY_PRIOR_COLOR = eda_constants.INTERMEDIARY_PRIOR_COLOR
BASELINE_PRIOR_COLOR = eda_constants.BASELINE_PRIOR_COLOR
CALIBRATED_PRIOR_COLOR = eda_constants.CALIBRATED_PRIOR_COLOR
CALIBRATION_EXPERIMENT_COLORS = eda_constants.EXPERIMENT_COLORS
