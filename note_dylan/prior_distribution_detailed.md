# Meridian Prior Distribution System: Complete Analysis

## Overview

The prior distribution system in Meridian is the cornerstone of its Bayesian approach to MMM. It enables:
1. **Business knowledge calibration** through ROI, mROI, and contribution priors
2. **Hierarchical modeling** for geo-level variation
3. **Flexible specification** for different channel types
4. **Automatic broadcasting** to match data dimensions

## 1. Prior Types and Calibration Methods

Meridian supports four distinct prior specification methods for media channels:

### 1.1 ROI Priors (`prior_type='roi'`)
**Purpose**: Calibrate using expected Return on Investment

**Mathematical Relationship**:
- Prior: $\text{ROI}_m \sim \text{LogNormal}(\mu, \sigma)$
- ROI represents total incremental outcome per unit spend
- Beta coefficients are calculated deterministically from ROI

**Default**: `LogNormal(0.2, 0.9)` - implies median ROI ≈ 1.22

**Use Case**: When you have historical ROI estimates or benchmarks

### 1.2 Marginal ROI Priors (`prior_type='mroi'`)
**Purpose**: Calibrate using marginal ROI at current spend levels

**Mathematical Relationship**:
- Prior: $\text{mROI}_m \sim \text{LogNormal}(\mu, \sigma)$
- mROI represents the derivative of response curve at current spend
- Accounts for saturation effects

**Default**: `LogNormal(0.0, 0.5)` - implies median mROI = 1.0

**Use Case**: When you know current efficiency but expect saturation

### 1.3 Contribution Priors (`prior_type='contribution'`)
**Purpose**: Calibrate using expected percentage of total outcome

**Mathematical Relationship**:
- Prior: $\text{Contribution}_m \sim \text{Beta}(\alpha, \beta)$
- Represents fraction of total KPI attributed to channel
- Beta coefficients calculated from total outcome

**Default**: `Beta(1.0, 99.0)` - implies mean contribution ≈ 1%

**Use Case**: When you have attribution studies or market mix estimates

### 1.4 Coefficient Priors (`prior_type='coefficient'`)
**Purpose**: Direct priors on model coefficients

**Mathematical Relationship**:
- Prior: $\beta_m \sim \text{HalfNormal}(5.0)$ (default)
- Direct specification of hierarchical mean parameters
- Most flexible but least interpretable

**Use Case**: Advanced users with specific coefficient knowledge

## 2. Parameter Categories and Defaults

### 2.1 Response Curve Parameters

#### Adstock (Carryover) Parameters
```python
# Alpha: Decay rate [0,1]
alpha_m: Uniform(0.0, 1.0)      # Media channels
alpha_rf: Uniform(0.0, 1.0)     # R&F channels
alpha_om: Uniform(0.0, 1.0)     # Organic media
alpha_orf: Uniform(0.0, 1.0)    # Organic R&F
```

#### Hill Saturation Parameters
```python
# EC (Half-saturation point)
ec_m: TruncatedNormal(0.8, 0.8, 0.1, 10)    # Media
ec_rf: LogNormal(0.7, 0.4) + 0.1            # R&F (shifted)
ec_om: TruncatedNormal(0.8, 0.8, 0.1, 10)   # Organic media
ec_orf: LogNormal(0.7, 0.4) + 0.1           # Organic R&F

# Slope (Hill curve steepness)
slope_m: Deterministic(1.0)      # Fixed for media
slope_rf: LogNormal(0.7, 0.4)    # Variable for R&F
slope_om: Deterministic(1.0)     # Fixed for organic
slope_orf: LogNormal(0.7, 0.4)   # Variable for organic R&F
```

**Note**: Fixed slope=1.0 ensures convex curves for optimization

### 2.2 Hierarchical Effect Parameters

#### Media Effects (when using coefficient priors)
```python
# Hierarchical mean
beta_m: HalfNormal(5.0)    # Media coefficient mean
beta_rf: HalfNormal(5.0)   # R&F coefficient mean
beta_om: HalfNormal(5.0)   # Organic media mean
beta_orf: HalfNormal(5.0)  # Organic R&F mean

# Hierarchical standard deviation
eta_m: HalfNormal(1.0)     # Media variation across geos
eta_rf: HalfNormal(1.0)    # R&F variation
eta_om: HalfNormal(1.0)    # Organic media variation
eta_orf: HalfNormal(1.0)   # Organic R&F variation
```

**Hierarchical Structure**:
- National model: `eta` parameters set to 0 (no geo variation)
- Geo model: `beta_gm ~ Normal(beta_m, eta_m)` or `LogNormal` variant

### 2.3 Control and Non-Media Parameters

```python
# Control variables
gamma_c: Normal(0.0, 5.0)    # Control coefficient mean
xi_c: HalfNormal(5.0)         # Control coefficient std

# Non-media treatments
gamma_n: Normal(0.0, 5.0)     # Treatment coefficient mean
xi_n: HalfNormal(5.0)         # Treatment coefficient std
```

### 2.4 Model Structure Parameters

```python
# Time effects
knot_values: Normal(0.0, 5.0)  # Spline knot values

# Geo effects (excluding baseline)
tau_g_excl_baseline: Normal(0.0, 5.0)  # Geo intercepts

# Noise
sigma: HalfNormal(5.0)  # Residual standard deviation
```

## 3. Special Cases and Advanced Features

### 3.1 Total Media Contribution Prior

When `kpi_type='non_revenue'` and no revenue data provided:

```python
# Automatic ROI prior calculation
roi_mean = p_mean * total_kpi / sum(total_spend)
roi_sd = p_sd * total_kpi / sqrt(sum(spend²))

# Convert to LogNormal parameters
sigma = sqrt(log(roi_sd²/roi_mean² + 1))
mu = log(roi_mean * exp(-sigma²/2))

# Apply same prior to all channels
roi_m ~ LogNormal(mu, sigma)
roi_rf ~ LogNormal(mu, sigma)
```

Default values: `p_mean=0.4`, `p_sd=0.2` (40% ± 20% media contribution)

### 3.2 Independent Multivariate Distribution

For channel-specific priors with different families:

```python
# Example: Different priors per channel
distributions = [
    tfp.distributions.LogNormal(0.5, 0.3),   # Channel 1: Higher ROI
    tfp.distributions.LogNormal(0.1, 0.5),   # Channel 2: Lower ROI
    tfp.distributions.Uniform(0.5, 2.0)      # Channel 3: Bounded ROI
]
roi_prior = IndependentMultivariateDistribution(distributions)
```

### 3.3 Broadcasting System

The `broadcast()` method:
1. Validates custom prior dimensions
2. Expands scalar priors to match channel counts
3. Handles national vs geo-level adjustments
4. Applies special transformations (e.g., total contribution)

```python
# Example: Scalar prior → Vector prior
# Input: roi_m = LogNormal(0.2, 0.9)  # Scalar
# Output: roi_m = BatchBroadcast(LogNormal(0.2, 0.9), n_media_channels)
```

## 4. Prior Validation and Warnings

### 4.1 Automatic Validations

1. **Dimension Matching**: Custom priors must match channel counts
2. **Negative Effects**: ROI/mROI priors must be non-negative for log-normal effects
3. **National Model**: Hierarchical parameters auto-converted to Deterministic(0)
4. **Slope Warning**: Non-unity slopes may cause non-convex optimization

### 4.2 Prior Consistency Rules

```python
# When using ROI priors, these are ignored:
- beta_m, beta_rf (calculated from ROI)

# When using mROI priors, these are ignored:
- beta_m, beta_rf (calculated from mROI)

# When using contribution priors, these are ignored:
- beta_m, beta_rf, beta_om, beta_orf (calculated from contribution)
```

## 5. Practical Prior Setting Guidelines

### 5.1 ROI Prior Selection

**Conservative (Wide) Priors**:
```python
roi_m = LogNormal(0.0, 1.5)  # Very uncertain
```

**Informative Priors**:
```python
roi_m = LogNormal(0.5, 0.3)  # Confident in ~1.6x ROI
```

**Channel-Specific**:
```python
roi_m = IndependentMultivariateDistribution([
    LogNormal(0.7, 0.2),  # TV: Higher ROI
    LogNormal(0.2, 0.4),  # Display: Lower ROI
    LogNormal(0.5, 0.3),  # Search: Medium ROI
])
```

### 5.2 Saturation Prior Guidelines

**Low Saturation** (Linear response):
```python
ec_m = TruncatedNormal(5.0, 2.0, 1.0, 20.0)  # High EC
```

**High Saturation** (Quick diminishing returns):
```python
ec_m = TruncatedNormal(0.3, 0.1, 0.1, 1.0)   # Low EC
```

### 5.3 Carryover Prior Guidelines

**Short Carryover** (Digital media):
```python
alpha_m = Beta(2.0, 5.0)  # Mean ≈ 0.29
```

**Long Carryover** (Brand campaigns):
```python
alpha_m = Beta(5.0, 2.0)  # Mean ≈ 0.71
```

## 6. Implementation Details

### 6.1 Serialization Support

The class includes custom `__getstate__` and `__setstate__` methods for pickling:
- Packs distribution parameters recursively
- Handles nested TransformedDistributions
- Preserves distribution types and parameters

### 6.2 Distribution Equality Testing

`distributions_are_equal()` function:
- Compares distribution types
- Recursively checks nested distributions
- Uses `allclose()` for numerical parameters
- Critical for prior validation logic

## 7. Key Design Principles

1. **Business Interpretability**: ROI/contribution priors are business-friendly
2. **Flexibility**: Multiple prior types for different use cases
3. **Hierarchical Structure**: Captures geo-level variation naturally
4. **Automatic Handling**: Broadcasting and validation reduce errors
5. **Bayesian Calibration**: Priors encode business knowledge systematically

## 8. Common Pitfalls and Solutions

### Pitfall 1: Inconsistent Prior Scales
**Solution**: Ensure EC priors are in same units as media data

### Pitfall 2: Too Narrow Priors
**Solution**: Start wide, use posterior predictive checks

### Pitfall 3: Ignoring Channel Differences
**Solution**: Use IndependentMultivariateDistribution for heterogeneity

### Pitfall 4: Wrong Prior Type
**Solution**: Match prior type to available business knowledge

## Summary

The prior distribution system is sophisticated yet practical:
- **Four calibration methods** match different business contexts
- **Hierarchical structure** handles geo-level variation
- **Automatic broadcasting** simplifies specification
- **Extensive validation** prevents common errors
- **Business-friendly** parameterizations enable stakeholder input

This design makes Meridian accessible to practitioners while maintaining statistical rigor.