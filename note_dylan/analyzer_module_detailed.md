# Meridian Analyzer Module: Detailed Analysis

## Overview

The Analyzer module (`/meridian/analysis/analyzer.py`) is responsible for computing business metrics and insights from the fitted Bayesian model. It transforms posterior samples into actionable metrics like ROI, incremental outcomes, and effectiveness.

## 1. Core Metric Calculations

### 1.1 Incremental Outcome

**Definition**: The causal effect of media on the outcome (KPI or revenue).

$$\text{Incremental Outcome} = E[\text{Outcome}|\text{Treatment}_1, \text{Controls}] - E[\text{Outcome}|\text{Treatment}_0, \text{Controls}]$$

Where:
- Treatment₁: Media at actual/scaled levels
- Treatment₀: Media at baseline (typically 0)

**Implementation** (`incremental_outcome()` method):
```python
# Key calculation flow:
1. Scale media by scaling_factor1 and scaling_factor0
2. Apply response curves (adstock + hill)
3. Calculate expected outcomes under both scenarios
4. Return the difference

# Supports:
- Flexible time periods
- Geo/time aggregation
- KPI vs revenue analysis
- Batch processing for memory efficiency
```

**Key Features**:
- Handles both paid and organic channels
- Supports counterfactual scenarios (what-if analysis)
- Can analyze specific time periods/geos
- Batch processing to handle large posterior samples

### 1.2 Return on Investment (ROI)

**Definition**: Total return per unit of spend.

$$\text{ROI} = \frac{\text{Incremental Outcome}}{\text{Total Spend}}$$

**Implementation** (`roi()` method):
```python
def roi(self, ...):
    # Numerator: Incremental outcome when channel spend = 0
    incremental_outcome = self.incremental_outcome(
        scaling_factor0=0.0,  # Counterfactual: no media
        scaling_factor1=1.0,  # Actual: historical media
        ...
    )
    
    # Denominator: Total channel spend
    spend = filled_data.total_spend()
    
    return tf.math.divide_no_nan(incremental_outcome, spend)
```

**Mathematical Details**:
- ROI represents the total (not marginal) return
- Accounts for saturation and carryover effects
- Can be calculated for subsets of geos/times

### 1.3 Marginal ROI (mROI)

**Definition**: Return from a small increase in spend at current levels.

$$\text{mROI} = \frac{\Delta \text{Outcome}}{\Delta \text{Spend}} \approx \frac{\partial \text{Outcome}}{\partial \text{Spend}}$$

**Implementation** (`marginal_roi()` method):
```python
def marginal_roi(self, incremental_increase=0.01, ...):
    # Numerator: Change in outcome from 1% spend increase
    numerator = self.incremental_outcome(
        scaling_factor0=1.0,
        scaling_factor1=1.0 + incremental_increase,  # 1% increase
        ...
    )
    
    # Denominator: 1% of total spend
    denominator = filled_data.total_spend() * incremental_increase
    
    return tf.math.divide_no_nan(numerator, denominator)
```

**Key Insights**:
- mROI ≤ ROI due to diminishing returns
- Critical for optimization decisions
- Reflects current saturation level

### 1.4 Cost Per Incremental KPI (CPIK)

**Definition**: Inverse of ROI when using KPI units.

$$\text{CPIK} = \frac{\text{Total Spend}}{\text{Incremental KPI}} = \frac{1}{\text{ROI}_{\text{KPI}}}$$

**Implementation** (`cpik()` method):
```python
def cpik(self, ...):
    # Simply 1/ROI with use_kpi=True
    return tf.math.divide_no_nan(
        1.0,
        self.roi(use_kpi=True, ...)
    )
```

## 2. Advanced Analytics

### 2.1 Response Curves

**Purpose**: Visualize the relationship between spend and response.

**Implementation** (`response_curves()` method):
```python
# For each channel:
1. Create spend grid (0 to max_spend)
2. Apply transformations:
   - Population scaling
   - Adstock decay
   - Hill saturation
3. Multiply by channel coefficients
4. Return expected outcomes
```

**Mathematical Flow**:
$$\text{Spend} \xrightarrow{\text{scale}} \text{Media} \xrightarrow{\text{adstock}} \text{Carry} \xrightarrow{\text{hill}} \text{Saturated} \xrightarrow{\beta} \text{Outcome}$$

### 2.2 Media Contribution

**Definition**: Percentage of total outcome attributed to each channel.

$$\text{Contribution}_m = \frac{\text{Incremental Outcome}_m}{\sum_i \text{Incremental Outcome}_i}$$

**Key Calculations**:
- Decompose total outcome into channel contributions
- Account for baseline, seasonality, and controls
- Provide uncertainty estimates

### 2.3 Effectiveness Metrics

**Media Effectiveness**:
$$\text{Effectiveness} = \frac{\text{Response Curve Slope at Current Spend}}{\text{Response Curve Slope at Zero}}$$

**Saturation Level**:
$$\text{Saturation} = \frac{\text{Current Response}}{\text{Maximum Possible Response}}$$

## 3. Data Structures

### 3.1 DataTensors Class

**Purpose**: Container for input data with validation.

```python
@tf.experimental.ExtensionType
class DataTensors:
    media: Optional[tf.Tensor]           # (n_geos, n_times, n_channels)
    media_spend: Optional[tf.Tensor]     # (n_channels,) or (n_geos, n_times, n_channels)
    reach: Optional[tf.Tensor]           # (n_geos, n_times, n_rf_channels)
    frequency: Optional[tf.Tensor]       # (n_geos, n_times, n_rf_channels)
    # ... other media types
```

**Key Methods**:
- `validate_and_fill_missing_data()`: Ensures consistency
- `get_modified_times()`: Detects time dimension changes
- `filter_fields()`: Extracts specific data types

### 3.2 DistributionTensors Class

**Purpose**: Container for parameter samples.

```python
@tf.experimental.ExtensionType
class DistributionTensors:
    # Response curve parameters
    alpha_m: Optional[tf.Tensor]    # Adstock decay
    ec_m: Optional[tf.Tensor]       # Hill EC50
    slope_m: Optional[tf.Tensor]    # Hill slope
    
    # Effect coefficients
    beta_gm: Optional[tf.Tensor]    # Geo-level media effects
    
    # Model structure
    mu_t: Optional[tf.Tensor]       # Time effects
    tau_g: Optional[tf.Tensor]      # Geo effects
```

## 4. Key Implementation Details

### 4.1 Batch Processing

**Problem**: Large posterior samples (10,000+ draws) cause memory issues.

**Solution**:
```python
# Process in batches
for i, start_index in enumerate(batch_starting_indices):
    stop_index = min(n_draws, start_index + batch_size)
    batch_dists = {
        k: params[k][:, start_index:stop_index, ...]
        for k in param_list
    }
    # Calculate for this batch
    batch_result = self._incremental_outcome_impl(...)
```

### 4.2 Flexible Time Analysis

**Challenge**: Support what-if scenarios with different time periods.

**Solution**:
- Allow `new_data` with modified time dimensions
- Validate consistency across tensors
- Handle lagged effects properly

### 4.3 Transformation Handling

**KPI Transformation Pipeline**:
```python
Raw KPI → Population scaling → Centering → Scaling → Model space
```

**Inverse Transformation**:
```python
Model predictions → Unscale → Uncenter → Population unscale → Original KPI
```

## 5. Visualization Components

### 5.1 Model Diagnostics

**R-hat Convergence Plot** (`plot_rhat_boxplot()`):
- Shows convergence diagnostics
- R-hat < 1.2 indicates good convergence
- Boxplot format for parameter groups

**Model Fit Plot** (`plot_model_fit()`):
- Actual vs predicted outcomes
- Credible intervals
- Time series by geo

### 5.2 Response Analysis

**Response Curves** (`plot_response_curves()`):
- Spend vs outcome relationship
- Shows saturation effects
- Includes uncertainty bands

**Adstock Decay** (`plot_adstock_decay()`):
- Carryover effect visualization
- Time lag vs effect strength
- Channel-specific decay patterns

**Hill Saturation** (`plot_hill_curves()`):
- Diminishing returns visualization
- Current operating point
- Saturation percentage

### 5.3 Business Metrics

**ROI Bar Chart** (`plot_roi_bar_chart()`):
- ROI by channel with credible intervals
- Sorted by mean ROI
- Color-coded by channel type

**Contribution Waterfall** (`plot_contribution_waterfall_chart()`):
- Decomposition of total outcome
- Baseline → Controls → Media → Total
- Shows relative importance

**Spend vs Contribution** (`plot_spend_vs_contribution()`):
- Efficiency visualization
- Identifies over/under-invested channels
- Diagonal line shows perfect efficiency

## 6. Statistical Computations

### 6.1 Credible Intervals

```python
def get_central_tendency_and_ci(data, confidence_level=0.9):
    mean = np.mean(data, axis=(0, 1))
    ci_lo = np.quantile(data, (1 - confidence_level) / 2, axis=(0, 1))
    ci_hi = np.quantile(data, (1 + confidence_level) / 2, axis=(0, 1))
    return np.stack([mean, ci_lo, ci_hi], axis=-1)
```

### 6.2 Model Fit Metrics

**R-squared**:
$$R^2 = 1 - \frac{\sum(\text{predicted} - \text{actual})^2}{\text{Var}(\text{actual})}$$

**MAPE** (Mean Absolute Percentage Error):
$$\text{MAPE} = \frac{1}{n}\sum\left|\frac{\text{actual} - \text{predicted}}{\text{actual}}\right|$$

**wMAPE** (Weighted MAPE):
$$\text{wMAPE} = \frac{\sum|\text{actual} - \text{predicted}|}{\sum\text{actual}}$$

## 7. Key Design Principles

1. **Causal Inference**: All metrics based on counterfactual reasoning
2. **Uncertainty Quantification**: Full posterior distributions, not point estimates
3. **Flexibility**: Support for various aggregation levels and scenarios
4. **Memory Efficiency**: Batch processing for large models
5. **Business Alignment**: Metrics directly map to business decisions

## 8. Usage Examples

### Basic ROI Analysis
```python
analyzer = Analyzer(mmm)
roi_posterior = analyzer.roi(
    use_posterior=True,
    aggregate_geos=True,
    use_kpi=False  # Use revenue
)
```

### What-If Scenario
```python
# What if we double TV spend?
new_media = original_media.copy()
new_media[:, :, tv_channel_idx] *= 2.0

incremental = analyzer.incremental_outcome(
    new_data=DataTensors(media=new_media),
    aggregate_geos=True,
    aggregate_times=True
)
```

### Optimization Prep
```python
# Get marginal ROI for optimization
mroi = analyzer.marginal_roi(
    incremental_increase=0.01,  # 1% increase
    use_posterior=True
)
```

## Summary

The Analyzer module bridges the gap between statistical modeling and business decisions by:
- Computing interpretable metrics (ROI, CPIK, effectiveness)
- Supporting flexible what-if analysis
- Providing comprehensive visualization tools
- Handling computational challenges elegantly
- Maintaining full uncertainty quantification

This module is where the Bayesian MMM becomes actionable for marketing teams.