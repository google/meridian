# Meridian Parameter Naming Conventions

## Overview

Meridian uses a systematic naming convention that clearly indicates parameter purpose, scope, and relationships. The constants.py file defines all parameter names used throughout the system.

## 1. Naming Convention Patterns

### 1.1 Suffix Conventions

The suffix indicates the **channel type** or **scope**:

- **`_m`**: Media channels (paid media with impressions)
- **`_rf`**: Reach & Frequency channels
- **`_om`**: Organic media channels
- **`_orf`**: Organic reach & frequency channels
- **`_c`**: Control variables
- **`_n`**: Non-media treatment channels
- **`_g`**: Geo-level parameters
- **`_t`**: Time-level parameters

### 1.2 Prefix Conventions

The prefix indicates the **parameter function**:

- **`alpha_`**: Adstock decay parameter
- **`ec_`**: Hill saturation EC50 (half-saturation point)
- **`slope_`**: Hill curve slope parameter
- **`beta_`**: Media effect coefficients
- **`gamma_`**: Control/non-media treatment coefficients
- **`eta_`**: Hierarchical standard deviation (random effects)
- **`xi_`**: Standard deviation for controls/non-media
- **`tau_`**: Geo intercept effects
- **`sigma`**: Residual standard deviation
- **`roi_`**: Return on investment priors
- **`mroi_`**: Marginal ROI priors
- **`contribution_`**: Contribution percentage priors

### 1.3 Geo-Level Parameters

Parameters with geo-level variation use compound naming:

- **`beta_gm`**: Geo-specific media coefficients
- **`beta_grf`**: Geo-specific R&F coefficients
- **`beta_gom`**: Geo-specific organic media coefficients
- **`beta_gorf`**: Geo-specific organic R&F coefficients
- **`gamma_gc`**: Geo-specific control coefficients
- **`gamma_gn`**: Geo-specific non-media treatment coefficients

### 1.4 Special Parameters

- **`knot_values`**: Time spline knot values
- **`mu_t`**: Time effect (derived from knots)
- **`tau_g`**: Geo effects (includes baseline)
- **`tau_g_excl_baseline`**: Geo effects excluding baseline geo

## 2. Parameter Categories

### 2.1 Response Curve Parameters

```python
# Adstock (carryover) parameters
ALPHA_M = 'alpha_m'      # Media adstock
ALPHA_RF = 'alpha_rf'    # R&F adstock
ALPHA_OM = 'alpha_om'    # Organic media adstock
ALPHA_ORF = 'alpha_orf'  # Organic R&F adstock

# Hill saturation parameters
EC_M = 'ec_m'        # Media half-saturation
EC_RF = 'ec_rf'      # R&F half-saturation
EC_OM = 'ec_om'      # Organic media half-saturation
EC_ORF = 'ec_orf'    # Organic R&F half-saturation

SLOPE_M = 'slope_m'      # Media Hill slope
SLOPE_RF = 'slope_rf'    # R&F Hill slope
SLOPE_OM = 'slope_om'    # Organic media Hill slope
SLOPE_ORF = 'slope_orf'  # Organic R&F Hill slope
```

### 2.2 Effect Coefficients

```python
# Hierarchical means
BETA_M = 'beta_m'      # Media coefficient mean
BETA_RF = 'beta_rf'    # R&F coefficient mean
BETA_OM = 'beta_om'    # Organic media mean
BETA_ORF = 'beta_orf'  # Organic R&F mean
GAMMA_C = 'gamma_c'    # Control coefficient mean
GAMMA_N = 'gamma_n'    # Non-media treatment mean

# Hierarchical standard deviations
ETA_M = 'eta_m'      # Media random effect std
ETA_RF = 'eta_rf'    # R&F random effect std
ETA_OM = 'eta_om'    # Organic media std
ETA_ORF = 'eta_orf'  # Organic R&F std
XI_C = 'xi_c'        # Control random effect std
XI_N = 'xi_n'        # Non-media treatment std
```

### 2.3 Calibration Parameters

```python
# ROI-based calibration
ROI_M = 'roi_m'      # Media ROI
ROI_RF = 'roi_rf'    # R&F ROI
MROI_M = 'mroi_m'    # Media marginal ROI
MROI_RF = 'mroi_rf'  # R&F marginal ROI

# Contribution-based calibration
CONTRIBUTION_M = 'contribution_m'      # Media contribution %
CONTRIBUTION_RF = 'contribution_rf'    # R&F contribution %
CONTRIBUTION_OM = 'contribution_om'    # Organic media %
CONTRIBUTION_ORF = 'contribution_orf'  # Organic R&F %
CONTRIBUTION_N = 'contribution_n'      # Non-media treatment %
```

### 2.4 Model Structure Parameters

```python
# Time effects
KNOT_VALUES = 'knot_values'  # Spline knot parameters
MU_T = 'mu_t'                # Time effect (computed)

# Geo effects
TAU_G = 'tau_g'                          # All geo effects
TAU_G_EXCL_BASELINE = 'tau_g_excl_baseline'  # Excluding baseline

# Noise
SIGMA = 'sigma'  # Residual standard deviation
```

## 3. Data Variable Naming

### 3.1 Input Data Arrays

```python
# Core data
KPI = 'kpi'
REVENUE_PER_KPI = 'revenue_per_kpi'
POPULATION = 'population'
CONTROLS = 'controls'

# Media data
MEDIA = 'media'              # Impressions
MEDIA_SPEND = 'media_spend'  # Spend
REACH = 'reach'              # Reach
FREQUENCY = 'frequency'      # Frequency
RF_SPEND = 'rf_spend'        # R&F spend

# Organic data
ORGANIC_MEDIA = 'organic_media'
ORGANIC_REACH = 'organic_reach'
ORGANIC_FREQUENCY = 'organic_frequency'

# Treatments
NON_MEDIA_TREATMENTS = 'non_media_treatments'
```

### 3.2 Coordinate Dimensions

```python
# Dimensions
GEO = 'geo'
TIME = 'time'
MEDIA_TIME = 'media_time'

# Channel dimensions
MEDIA_CHANNEL = 'media_channel'
RF_CHANNEL = 'rf_channel'
ORGANIC_MEDIA_CHANNEL = 'organic_media_channel'
ORGANIC_RF_CHANNEL = 'organic_rf_channel'
NON_MEDIA_CHANNEL = 'non_media_channel'
CONTROL_VARIABLE = 'control_variable'
```

## 4. Scaled/Transformed Data

Variables after preprocessing get `_scaled` suffix:

```python
MEDIA_SCALED = 'media_scaled'
REACH_SCALED = 'reach_scaled'
ORGANIC_MEDIA_SCALED = 'organic_media_scaled'
ORGANIC_REACH_SCALED = 'organic_reach_scaled'
NON_MEDIA_TREATMENTS_SCALED = 'non_media_treatments_scaled'
CONTROLS_SCALED = 'controls_scaled'
```

## 5. Internal/Derived Parameters

### 5.1 Latent Variables (not saved)

```python
# Standard normal deviates for non-centered parameterization
BETA_GM_DEV = 'beta_gm_dev'    # Media geo deviates
BETA_GRF_DEV = 'beta_grf_dev'  # R&F geo deviates
BETA_GOM_DEV = 'beta_gom_dev'  # Organic media deviates
BETA_GORF_DEV = 'beta_gorf_dev'  # Organic R&F deviates
GAMMA_GC_DEV = 'gamma_gc_dev'  # Control geo deviates
GAMMA_GN_DEV = 'gamma_gn_dev'  # Non-media geo deviates
```

### 5.2 Parameter Groupings

```python
# All parameters for each channel type
MEDIA_PARAMETERS = (ROI_M, MROI_M, CONTRIBUTION_M, BETA_M, 
                   ETA_M, ALPHA_M, EC_M, SLOPE_M)

RF_PARAMETERS = (ROI_RF, MROI_RF, CONTRIBUTION_RF, BETA_RF,
                ETA_RF, ALPHA_RF, EC_RF, SLOPE_RF)

ORGANIC_MEDIA_PARAMETERS = (CONTRIBUTION_OM, BETA_OM, ETA_OM,
                           ALPHA_OM, EC_OM, SLOPE_OM)

# Parameters that become deterministic in national models
ALL_NATIONAL_DETERMINISTIC_PARAMETER_NAMES = (
    SLOPE_M, SLOPE_OM, XI_N, XI_C, 
    ETA_M, ETA_RF, ETA_OM, ETA_ORF
)
```

## 6. Analysis Output Metrics

### 6.1 Statistical Metrics

```python
MEAN = 'mean'
MEDIAN = 'median'
CI_LO = 'ci_lo'      # Lower credible interval
CI_HI = 'ci_hi'      # Upper credible interval
RHAT = 'rhat'        # Convergence diagnostic
```

### 6.2 Business Metrics

```python
ROI = 'roi'                    # Return on investment
MROI = 'mroi'                  # Marginal ROI
CPIK = 'cpik'                  # Cost per incremental KPI
EFFECTIVENESS = 'effectiveness'  # Media effectiveness
PCT_OF_CONTRIBUTION = 'pct_of_contribution'  # % contribution
```

### 6.3 Optimization Metrics

```python
OPTIMIZED_ROI = 'optimized_roi'
OPTIMIZED_MROI_BY_REACH = 'optimized_mroi_by_reach'
OPTIMIZED_MROI_BY_FREQUENCY = 'optimized_mroi_by_frequency'
OPTIMIZED_INCREMENTAL_OUTCOME = 'optimized_incremental_outcome'
```

## 7. Special Constants

### 7.1 Prior Types

```python
TREATMENT_PRIOR_TYPE_ROI = 'roi'
TREATMENT_PRIOR_TYPE_MROI = 'mroi'
TREATMENT_PRIOR_TYPE_COEFFICIENT = 'coefficient'
TREATMENT_PRIOR_TYPE_CONTRIBUTION = 'contribution'
```

### 7.2 Model Types

```python
MEDIA_EFFECTS_NORMAL = 'normal'
MEDIA_EFFECTS_LOG_NORMAL = 'log_normal'
```

### 7.3 Decay Functions

```python
GEOMETRIC_DECAY = 'geometric'
BINOMIAL_DECAY = 'binomial'
```

## 8. Key Design Principles

1. **Consistency**: Same suffix pattern across all parameter types
2. **Clarity**: Prefix immediately identifies parameter function
3. **Hierarchy**: Compound names show relationships (e.g., `beta_gm`)
4. **Scalability**: Easy to add new channel types with consistent naming
5. **Traceability**: Parameter names match mathematical notation in papers

## 9. Usage Examples

```python
# Building a parameter name programmatically
channel_type = '_m'  # media
param_type = 'alpha'
param_name = f"{param_type}{channel_type}"  # 'alpha_m'

# Checking parameter dimensions
INFERENCE_DIMS = {
    'beta_m': ('media_channel',),          # 1D: channels
    'beta_gm': ('geo', 'media_channel'),    # 2D: geos × channels
    'mu_t': ('time',),                      # 1D: time periods
}

# Ignored priors based on prior type
if prior_type == 'roi':
    # These are calculated from ROI, so ignored if specified
    ignored = ['beta_m', 'mroi_m', 'contribution_m']
```

## Summary

Meridian's naming conventions provide a clear, systematic approach to parameter identification:
- **Suffixes** identify channel/variable types
- **Prefixes** identify parameter functions
- **Compound names** show hierarchical relationships
- **Consistency** makes the codebase predictable and maintainable

This naming system scales well and makes it easy to understand parameter relationships at a glance.