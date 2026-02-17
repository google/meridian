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
EVALUATION_SET_SUFFIXES = (ALL_SUFFIX, TRAIN_SUFFIX, TEST_SUFFIX)
MEAN = "mean"
VARIANCE = "variance"
MEDIAN = "median"
Q1 = "q1"
Q3 = "q3"
BAYESIAN_PPP = "bayesian_ppp"

# Health score constants
R2_MIDPOINT = 0.5
R2_STEEPNESS = 15
FAIL_RATIO_POWER = 0.3
HEALTH_SCORE_WEIGHT_BASELINE = 0.3
HEALTH_SCORE_WEIGHT_BAYESIAN_PPP = 0.3
HEALTH_SCORE_WEIGHT_GOF = 0.1
HEALTH_SCORE_WEIGHT_PRIOR_POSTERIOR_SHIFT = 0.15
HEALTH_SCORE_WEIGHT_ROI_CONSISTENCY = 0.15
