# Meridian MMM Complete Workflow Guide

## Overview
Meridian is Google's Bayesian Media Mix Modeling (MMM) framework designed for geo-level analysis. This guide summarizes the complete workflow from data preparation to optimization results.

## Workflow Architecture

```
Raw Data → InputData → Model Configuration → MCMC Sampling → Analysis → Optimization
```

## Step-by-Step Workflow

### Step 1: Data Preparation

#### Required Data Components
1. **KPI Data**
   - Primary KPI (e.g., conversions, sales)
   - Revenue per KPI (for revenue calculations)
   - Time and geo dimensions

2. **Media Data**
   - Impressions by channel
   - Spend by channel
   - Optional: Reach & Frequency data

3. **Control Variables**
   - External factors (e.g., sentiment, competitor activity)
   - Seasonality indicators

4. **Population Data**
   - Geo-level population for scaling

5. **Non-Media Treatments**
   - Promotions, price changes, etc.

#### Data Format
- **Structure**: Long format with columns for geo, time, and metrics
- **Granularity**: Typically weekly or daily, geo-level
- **Example columns**:
  ```
  geo, time, conversions, revenue_per_conversion, population,
  Channel0_impression, Channel0_spend, ..., 
  sentiment_score_control, competitor_sales_control,
  Promo, Organic_channel0_impression
  ```

### Step 2: Data Loading

```python
# 1. Create builder instance
builder = DataFrameInputDataBuilder(
    kpi_type='non_revenue',  # or 'revenue'
    default_kpi_column='conversions',
    default_revenue_per_kpi_column='revenue_per_conversion',
)

# 2. Add data components
builder = (
    builder.with_kpi(df)
    .with_revenue_per_kpi(df)
    .with_population(df)
    .with_controls(df, control_cols=[...])
    .with_media(df, media_cols=[...], media_spend_cols=[...])
    .with_organic_media(df, organic_media_cols=[...])
    .with_non_media_treatments(df, non_media_treatment_cols=[...])
)

# 3. Build InputData object
data = builder.build()
```

**InputData Structure**:
- Stores all data as xarray DataArrays
- Dimensions: `(geo, time, channel)`
- Validates data consistency and completeness

### Step 3: Model Configuration

#### Prior Specification
```python
# ROI priors for calibration
roi_mu = 0.2
roi_sigma = 0.9
prior = PriorDistribution(
    roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma)
)

# Model specification
model_spec = ModelSpec(prior=prior)

# Initialize model
mmm = Meridian(input_data=data, model_spec=model_spec)
```

#### Key Model Components

1. **Response Curves**
   - **Adstock**: Media effect decay over time
     - Geometric decay: `effect[t] = spend[t] + alpha * effect[t-1]`
     - Binomial decay: Weighted average over past periods
   - **Hill Saturation**: Diminishing returns
     - `response = spend^s / (half_max^s + spend^s)`
   - Order: Can apply Hill→Adstock or Adstock→Hill

2. **Hierarchical Structure**
   - National-level media effects
   - Geo-level random effects
   - Time trends (flexible splines)

3. **Bayesian Framework**
   - Prior distributions on all parameters
   - Posterior sampling via MCMC
   - Uncertainty quantification

### Step 4: Model Fitting

```python
# Sample from prior (for prior predictive checks)
mmm.sample_prior(n_samples=500)

# Sample from posterior
mmm.sample_posterior(
    n_chains=10,        # Number of MCMC chains
    n_adapt=2000,       # Adaptation steps
    n_burnin=500,       # Burn-in samples
    n_keep=1000,        # Samples to keep per chain
    seed=0
)
```

**MCMC Details**:
- Uses No U-Turn Sampler (NUTS)
- GPU-accelerated via TensorFlow
- Total posterior samples: `n_chains × n_keep`

### Step 5: Model Diagnostics

#### Convergence Checks
```python
# R-hat statistics (should be < 1.2)
model_diagnostics = ModelDiagnostics(mmm)
model_diagnostics.plot_rhat_boxplot()
```

#### Model Fit Assessment
```python
# Actual vs predicted comparison
model_fit = ModelFit(mmm)
model_fit.plot_model_fit()
```

**Key Diagnostics**:
- R-hat < 1.2 indicates convergence
- Effective sample size checks
- Posterior predictive checks
- Residual analysis

### Step 6: Results Analysis

```python
# Generate summary report
summarizer = Summarizer(mmm)
summarizer.output_model_results_summary(
    'summary_output.html',
    filepath,
    start_date,
    end_date
)
```

**Results Include**:
1. **Channel Contributions**
   - Percentage of KPI attributed to each channel
   - Uncertainty intervals

2. **ROI Metrics**
   - Return on Ad Spend (ROAS) by channel
   - Marginal ROI at current spend levels

3. **Response Curves**
   - Adstock decay patterns
   - Saturation curves by channel

4. **Time Series Decomposition**
   - Baseline trend
   - Seasonal patterns
   - Media contributions over time

### Step 7: Budget Optimization

```python
# Run optimization
budget_optimizer = BudgetOptimizer(mmm)
optimization_results = budget_optimizer.optimize()

# Generate report
optimization_results.output_optimization_summary(
    'optimization_output.html',
    filepath
)
```

**Optimization Scenarios**:
1. **Fixed Budget**: Reallocate current budget for max ROI
2. **Flexible Budget**: Find optimal budget level
3. **Target ROI**: Achieve specific ROI target
4. **Custom Constraints**: Channel-specific limits

**Optimization Output**:
- Current vs optimal spend allocation
- Expected KPI/ROI improvements
- Channel-specific recommendations
- Sensitivity analysis

### Step 8: Model Persistence

```python
# Save model
model.save_mmm(mmm, 'saved_mmm.pkl')

# Load model
mmm = model.load_mmm('saved_mmm.pkl')
```

## Key Technical Details

### Data Processing Pipeline
1. **Validation**: Check for missing values, data types
2. **Scaling**: Population-based normalization
3. **Transformation**: Log/sqrt transforms for stability
4. **Indexing**: Convert to xarray for efficient computation

### Model Mathematics

#### Media Effect Equation
```
KPI[geo,time] = baseline[geo,time] + 
                Σ(media_effect[channel,geo,time]) +
                Σ(control_effect[control,geo,time]) +
                error[geo,time]
```

#### Media Transformation
```
media_effect = beta * Hill(Adstock(spend))
```

Where:
- `beta`: Channel coefficient
- `Hill()`: Saturation transformation
- `Adstock()`: Carryover transformation

### Prior Types

1. **ROI Priors**: Business knowledge about returns
2. **MROI Priors**: Marginal ROI at current spend
3. **Coefficient Priors**: Direct effect sizes
4. **Contribution Priors**: Percentage of total effect

### Computational Considerations

- **GPU Required**: For MCMC sampling
- **Memory Usage**: Scales with geos × time × channels
- **Runtime**: 10-30 minutes typical for moderate datasets
- **Scalability**: Handles 100s of geos, 100s of time periods

## Best Practices

1. **Data Quality**
   - Ensure consistent geo definitions
   - Handle missing data appropriately
   - Check for outliers and anomalies

2. **Prior Setting**
   - Use business knowledge for ROI priors
   - Start with wider priors if uncertain
   - Validate with prior predictive checks

3. **Model Checking**
   - Always verify convergence (R-hat)
   - Check posterior predictive fit
   - Validate against holdout data

4. **Interpretation**
   - Consider uncertainty in estimates
   - Understand saturation levels
   - Account for carryover effects

5. **Optimization**
   - Test multiple scenarios
   - Consider practical constraints
   - Validate recommendations with tests

## Common Issues and Solutions

1. **Convergence Problems**
   - Increase adaptation steps
   - Simplify model (fewer parameters)
   - Check for data issues

2. **Poor Fit**
   - Add more control variables
   - Adjust transformation order
   - Consider interaction effects

3. **Unrealistic Results**
   - Tighten prior distributions
   - Check data quality
   - Verify spend/impression alignment

4. **Computational Issues**
   - Reduce data granularity
   - Use fewer MCMC chains
   - Upgrade GPU resources