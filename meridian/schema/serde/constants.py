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

"""Constants shared across the Meridian serde library."""

# Constants for hyperparameters protobuf structure
BASELINE_GEO_ONEOF = 'baseline_geo_oneof'
BASELINE_GEO_INT = 'baseline_geo_int'
BASELINE_GEO_STRING = 'baseline_geo_string'
CONTROL_POPULATION_SCALING_ID = 'control_population_scaling_id'
HOLDOUT_ID = 'holdout_id'
NON_MEDIA_POPULATION_SCALING_ID = 'non_media_population_scaling_id'
ADSTOCK_DECAY_SPEC = 'adstock_decay_spec'
GLOBAL_ADSTOCK_DECAY = 'global_adstock_decay'
ADSTOCK_DECAY_BY_CHANNEL = 'adstock_decay_by_channel'
DEFAULT_DECAY = 'geometric'

# Constants for the declarative `Hyperparameters` config messages.
ROI_CALIBRATION_CONFIG = 'roi_calibration_config'
RF_ROI_CALIBRATION_CONFIG = 'rf_roi_calibration_config'
HOLDOUT_CONFIG = 'holdout_config'
# `CalibrationConfig.spec` and `HoldoutConfig.spec` are both named `spec`.
CONFIG_SPEC_ONEOF = 'spec'
GLOBAL_DATE_RANGES = 'global_date_ranges'
CHANNEL_DATE_RANGES = 'channel_date_ranges'
GEO_DATE_RANGES = 'geo_date_ranges'
RANDOM_HOLDOUT = 'random_holdout'
RESOLVED = 'resolved'
START_DATE = 'start_date'
END_DATE = 'end_date'
SEED = 'seed'

SATURATION_SPEC = 'saturation_spec'
GLOBAL_SATURATION = 'global_saturation'
SATURATION_BY_CHANNEL = 'saturation_by_channel'
DEFAULT_SATURATION = 'hill'

KNOTS_SPEC = 'knots_spec'
N_KNOTS = 'n_knots'
KNOT_LOCATIONS = 'knot_locations'

# Constants for marketing data protobuf structure
GEO_INFO = 'geo_info'
METADATA = 'metadata'
REACH_FREQUENCY = 'reach_frequency'
POPULATION_VALUE = 'population_value'

# Constants for distribution protobuf structure
DISTRIBUTION_TYPE = 'distribution_type'
BATCH_BROADCAST_DISTRIBUTION = 'batch_broadcast'
DETERMINISTIC_DISTRIBUTION = 'deterministic'
HALF_NORMAL_DISTRIBUTION = 'half_normal'
LOG_NORMAL_DISTRIBUTION = 'log_normal'
NORMAL_DISTRIBUTION = 'normal'
TRANSFORMED_DISTRIBUTION = 'transformed'
TRUNCATED_NORMAL_DISTRIBUTION = 'truncated_normal'
UNIFORM_DISTRIBUTION = 'uniform'
BETA_DISTRIBUTION = 'beta'
BIJECTOR_TYPE = 'bijector_type'
SHIFT_BIJECTOR = 'shift'
SCALE_BIJECTOR = 'scale'
RECIPROCAL_BIJECTOR = 'reciprocal'
