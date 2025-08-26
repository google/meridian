# Meridian Workflow: Function and Script Mapping

This document maps out the specific functions and Python scripts called at each step of the Meridian MMM workflow, providing a technical roadmap through the codebase.

## 1. Data Loading Step

### Primary Entry Point
```python
builder = DataFrameInputDataBuilder(kpi_type='non_revenue', ...)
data = builder.build()
```

### Call Hierarchy

#### DataFrameInputDataBuilder (`/meridian/data/data_frame_input_data_builder.py`)
```
DataFrameInputDataBuilder.__init__()
├── Sets column defaults (kpi, revenue_per_kpi)
└── Initializes empty component storage

builder.with_kpi(df)
├── _validate_cols() - Ensures KPI column exists
├── _extract_data() - Extracts KPI data from DataFrame
└── Stores in self._kpi

builder.with_media(df, media_cols, media_spend_cols)
├── _validate_cols() - Validates media columns exist
├── _extract_data() - Extracts impression/spend data
├── Creates media channel mapping
└── Stores in self._media_execution, self._media_spend

builder.build()
├── _validate_required_components() - Checks all required data present
├── _validate_coords() - Ensures geo/time consistency across components
├── _validate_nas() - Checks for missing values
├── _normalize_coords() - Converts to Meridian coordinate format
├── to_xarray() - Converts pandas → xarray DataArrays
└── Returns InputData instance
```

#### InputData (`/meridian/data/input_data.py`)
```
InputData.__init__()
├── Stores all data as xarray.DataArray objects
├── Validates data types and dimensions
├── Sets coordinate names (geo, time, channel)
└── Freezes data (immutable after creation)
```

### Key Internal Functions
- `_validate_cols()`: Ensures required columns exist in DataFrame
- `_extract_data()`: Safely extracts data with error handling
- `_validate_coords()`: Cross-validates geo/time dimensions
- `to_xarray()`: Converts pandas to xarray with proper dimensions

## 2. Model Configuration Step

### Primary Entry Point
```python
prior = PriorDistribution(roi_m=LogNormal(0.2, 0.9))
model_spec = ModelSpec(prior=prior)
```

### Call Hierarchy

#### PriorDistribution (`/meridian/model/prior_distribution.py`)
```
PriorDistribution.__init__()
├── Accepts distribution objects for each parameter type
├── Validates distribution types (must be TFP distributions)
├── Stores priors for:
│   ├── ROI parameters (roi_m, roi_rf)
│   ├── Coefficient parameters (coef_m, coef_rf)
│   ├── Contribution parameters (contribution_m, contribution_rf)
│   └── Other model parameters
└── No computation, just storage
```

#### ModelSpec (`/meridian/model/spec.py`)
```
ModelSpec.__init__()
├── Sets model configuration parameters
├── __post_init__() validation:
│   ├── _validate_roi_calibration_period()
│   ├── Checks parameter consistency
│   └── Validates prior type settings
├── Properties compute effective settings:
│   ├── effective_media_prior_type
│   └── effective_rf_prior_type
└── Stores all model hyperparameters
```

### Key Configuration Parameters
- `n_time_knots`: Number of time spline knots
- `n_cycles`: Cycles for seasonality
- `hill_after_adstock`: Response curve order
- `paid_media_prior_type`: Prior specification method

## 3. Model Initialization Step

### Primary Entry Point
```python
mmm = Meridian(input_data=data, model_spec=model_spec)
```

### Call Hierarchy

#### Meridian (`/meridian/model/model.py`)
```
Meridian.__init__()
├── Store input_data and model_spec
├── _validate_data_dependent_model_spec()
│   ├── Checks model spec compatibility with data
│   ├── Validates prior dimensions match channels
│   └── Ensures required data for chosen options
├── _warn_setting_ignored_priors()
│   └── Warns if conflicting prior settings
├── Initialize inference data storage:
│   ├── self._prior_inference_data = None
│   └── self._posterior_inference_data = None
├── Build tensor structures:
│   ├── media_tensors property (lazy evaluation)
│   ├── rf_tensors property (lazy evaluation)
│   └── organic_media_tensors property (lazy evaluation)
└── Determine baseline_geo_idx
```

#### Media Tensor Building (`/meridian/model/media.py`)
```
build_media_tensors()
├── Creates MediaTensors dataclass
├── Extracts channel data from InputData
├── Applies transformations:
│   ├── Population scaling
│   ├── Log/sqrt transforms
│   └── Normalization
└── Returns structured tensor data
```

### Key Validation Functions
- `_validate_geo_invariants()`: Ensures geo data consistency
- `_validate_time_invariants()`: Ensures time series alignment
- `_check_for_negative_effects()`: Validates effect directions

## 4. MCMC Sampling Step

### Prior Sampling
```python
mmm.sample_prior(n_draws=500)
```

#### Call Hierarchy
```
Meridian.sample_prior()
├── Creates PriorDistributionSampler instance
├── Calls prior_sampler_callable()
│   ├── _sample_media_priors()
│   ├── _sample_rf_priors()
│   ├── _sample_control_priors()
│   └── _sample_other_priors()
├── Converts samples to InferenceData
└── Stores in self._prior_inference_data
```

### Posterior Sampling
```python
mmm.sample_posterior(n_chains=10, n_adapt=2000, n_burnin=500, n_keep=1000)
```

#### Call Hierarchy
```
Meridian.sample_posterior()
├── Creates PosteriorMCMCSampler instance
├── Calls posterior_sampler_callable()
│   ├── _get_joint_dist_unpinned() - Build probabilistic model
│   ├── _get_initial_state() - Initialize chain states
│   └── _xla_windowed_adaptive_nuts() - Run MCMC
│       ├── backend.experimental.mcmc.windowed_adaptive_nuts()
│       ├── Adaptation phase (n_adapt steps)
│       ├── Burn-in phase (n_burnin steps)
│       └── Sampling phase (n_keep steps)
├── Post-processing:
│   ├── Extract samples
│   ├── Compute diagnostics (R-hat, ESS)
│   └── Transform to original scale
└── Stores in self._posterior_inference_data
```

### Key MCMC Functions
- `_get_joint_dist_unpinned()`: Constructs TFP JointDistribution
- `_get_tau_g()`: Handles baseline geo in hierarchical model
- `_xla_windowed_adaptive_nuts()`: XLA-compiled MCMC kernel

## 5. Diagnostics Step

### Primary Entry Points
```python
diagnostics = ModelDiagnostics(mmm)
model_fit = ModelFit(mmm)
```

### Call Hierarchy

#### ModelDiagnostics (`/meridian/analysis/visualizer.py`)
```
ModelDiagnostics.__init__()
├── Stores model reference
├── Initializes Analyzer instance
└── Sets up diagnostic methods

plot_rhat_boxplot()
├── Extract R-hat values from posterior
├── Create boxplot visualization
└── Add convergence threshold line (1.2)

predictive_accuracy_table()
├── Calls _predictive_accuracy_dataset()
│   ├── analyzer.predictive_accuracy()
│   ├── Computes MAPE, R-squared, etc.
│   └── Aggregates by geo/time
└── Formats as styled pandas table
```

#### ModelFit (`/meridian/analysis/visualizer.py`)
```
ModelFit.__init__()
├── Stores model reference
├── Initializes Analyzer instance
└── Sets confidence level (90%)

plot_model_fit()
├── _validate_times_to_plot()
├── _validate_geos_to_plot()
├── analyzer.expected_vs_actual_data()
│   ├── Compute posterior predictions
│   ├── Calculate credible intervals
│   └── Compare to actual data
├── _transform_data_to_dataframe()
└── Create time series plot with CI bands
```

## 6. Results Analysis Step

### Primary Entry Point
```python
summarizer = Summarizer(mmm)
summarizer.output_model_results_summary(filename, filepath, start_date, end_date)
```

### Call Hierarchy

#### Summarizer (`/meridian/analysis/summarizer.py`)
```
Summarizer.__init__()
├── Store model reference
├── Initialize sub-components:
│   ├── Analyzer (computation engine)
│   ├── ModelFit (visualization)
│   ├── ModelDiagnostics (diagnostics)
│   └── Formatter (output formatting)
└── Set default parameters

output_model_results_summary()
├── _gen_model_results_summary()
│   ├── Generate header section
│   ├── model_fit.plot_model_fit() → Model fit plot
│   ├── model_diagnostics.predictive_accuracy_table() → Accuracy metrics
│   ├── analyzer.contribution() → Channel contributions
│   ├── analyzer.incremental_outcome() → Incremental effects
│   ├── visualizer.plot_response_curves() → Response curves
│   └── visualizer.plot_media_contribution() → Time series decomposition
├── formatter.write_html()
└── Save to file
```

#### Analyzer (`/meridian/analysis/analyzer.py`)
```
Analyzer.__init__()
├── Store model and inference data
└── Cache computed results

contribution()
├── Extract media effects from posterior
├── Compute total contribution per channel
├── Calculate percentage contributions
└── Return with uncertainty intervals

incremental_outcome()
├── Simulate counterfactual (no media)
├── Calculate difference from actual
├── Aggregate by channel
└── Compute ROI metrics
```

## 7. Optimization Step

### Primary Entry Point
```python
optimizer = BudgetOptimizer(mmm)
results = optimizer.optimize()
```

### Call Hierarchy

#### BudgetOptimizer (`/meridian/analysis/optimizer.py`)
```
BudgetOptimizer.__init__()
├── Store model reference
├── Initialize Analyzer
├── Set optimization defaults
└── Prepare optimization bounds

optimize()
├── Determine scenario type:
│   ├── FixedBudgetScenario (default)
│   ├── FlexibleBudgetScenario
│   └── TargetROIScenario
├── _validate_optimization_inputs()
├── get_optimization_bounds()
│   ├── Extract spend ranges from data
│   ├── Apply constraint multipliers
│   └── Handle channel-specific limits
├── Run optimization algorithm:
│   ├── analyzer.response_curves() → Get response functions
│   ├── Build objective function (maximize outcome)
│   ├── Apply constraints (budget, bounds)
│   └── scipy.optimize.minimize() → Find optimal allocation
├── Post-process results:
│   ├── Round to budget precision
│   ├── Calculate expected outcomes
│   └── Compare to current allocation
└── Return OptimizationResults object
```

### Key Optimization Functions
- `get_round_factor()`: Handles budget precision
- `response_curves()`: Extracts response functions from posterior
- Objective function: Maximizes total expected outcome

## 8. Model Persistence Step

### Save Model
```python
model.save_mmm(mmm, 'model.pkl')
```

#### Call Hierarchy
```
save_mmm()
├── Create directory if needed
├── Prepare model for serialization:
│   ├── Convert inference data to serializable format
│   └── Handle backend-specific objects
├── joblib.dump() with compression
└── Confirm successful save
```

### Load Model
```python
mmm = model.load_mmm('model.pkl')
```

#### Call Hierarchy
```
load_mmm()
├── joblib.load() from file
├── Restore model state:
│   ├── Reconstruct inference data
│   └── Reinitialize backend objects
└── Return Meridian instance
```

## Key Supporting Modules

### Backend Abstraction (`/meridian/backend/`)
- Provides unified interface for TensorFlow/JAX
- Handles distribution creation
- Manages computational backend selection

### Transformers (`/meridian/model/transformers.py`)
- Data preprocessing utilities
- Scaling and normalization functions
- Coordinate transformations

### Constants (`/meridian/constants.py`)
- System-wide constants
- Parameter name definitions
- Default values and limits

### Media Processing (`/meridian/model/media.py`)
- Media tensor construction
- Channel data organization
- Transformation pipelines

This mapping provides a complete technical view of how Meridian processes data through each workflow step, showing the specific functions called and their purposes.