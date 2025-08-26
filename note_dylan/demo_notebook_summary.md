# Meridian Demo Notebook Summary

The `Meridian_Getting_Started.ipynb` demonstrates a complete MMM workflow using simulated data.

## Workflow Steps:

### 1. **Data Loading**
- Uses simulated CSV data with geo-level information
- Builds InputData using `DataFrameInputDataBuilder`
- Includes:
  - **KPI data**: conversions and revenue per conversion
  - **Media channels**: 5 paid channels (Channel0-4) with impressions and spend
  - **Organic media**: 1 organic channel
  - **Control variables**: sentiment score, competitor sales
  - **Non-media treatments**: Promotions
  - **Population data**: for geo-level scaling

### 2. **Model Configuration**
- Sets up Bayesian priors for ROI calibration
- Uses LogNormal(0.2, 0.9) distribution for all channels
- Creates model specification with these priors
- Initializes the Meridian model

### 3. **MCMC Sampling**
- Samples from prior distribution (500 samples)
- Samples from posterior distribution:
  - 10 chains
  - 2000 adaptation steps
  - 500 burn-in samples
  - 1000 kept samples per chain
  - Total: 10,000 posterior samples

### 4. **Model Diagnostics**
- **Convergence check**: R-hat statistics (should be < 1.2)
- **Model fit**: Actual vs predicted sales comparison
- Visual diagnostics for quality assessment

### 5. **Results Generation**
- Creates HTML summary report with:
  - Model performance metrics
  - Channel contributions
  - ROI estimates
  - Response curves
  - Time series decomposition

### 6. **Budget Optimization**
- Runs default "Fixed Budget" scenario
- Maximizes ROI across channels
- Generates optimization report showing:
  - Current vs optimal spend allocation
  - Expected ROI improvements
  - Channel-specific recommendations

### 7. **Model Persistence**
- Saves complete model object as pickle file
- Allows reloading for future analysis

## Key Insights:

1. **Data Structure**: Expects geo-level time series data with clear separation of media, controls, and treatments
2. **Bayesian Calibration**: ROI priors are central to model configuration
3. **GPU Required**: Uses TensorFlow with GPU acceleration for MCMC
4. **Business-Ready Outputs**: Generates stakeholder-friendly HTML reports
5. **Optimization Built-in**: Includes budget allocation recommendations

## Notable Omissions:
- No data preprocessing or EDA shown
- Assumes clean, ready-to-use data
- No reach & frequency data in this example (though supported)
- No custom prior configurations demonstrated