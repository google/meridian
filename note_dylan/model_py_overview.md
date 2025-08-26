# Meridian Model.py Overview

## Core Components

### 1. **Meridian Class** (Lines 88-1539)
The main model class that orchestrates the entire MMM workflow.

#### Key Attributes:
- `input_data`: InputData object with all model data
- `model_spec`: ModelSpec with configuration settings
- `inference_data`: ArviZ InferenceData storing MCMC results
- Various tensor properties for different data types (media, RF, controls, etc.)
- Transformers for data preprocessing
- Cached properties for efficient computation

#### Major Methods:

##### Initialization & Validation
- `__init__()`: Sets up model with extensive validation
- `_validate_data_dependent_model_spec()`: Ensures model spec matches data dimensions
- `_validate_geo_invariants()`: Checks for geo variation in data
- `_validate_time_invariants()`: Checks for time variation in data
- `_check_for_negative_effects()`: Validates prior distributions

##### Data Access Properties
- `media_tensors`, `rf_tensors`, `organic_media_tensors`, etc.: Lazy-loaded tensor structures
- `kpi`, `controls`, `non_media_treatments`: Converted to TensorFlow tensors
- `n_geos`, `n_media_channels`, `n_times`, etc.: Dimension properties

##### Transformation Methods
- `adstock_hill_media()` (Lines 1250-1303): Applies adstock and saturation transformations to media
- `adstock_hill_rf()` (Lines 1305-1353): Applies transformations to reach & frequency data
- Order can be configured: Hill→Adstock or Adstock→Hill

##### Calibration Methods
- `calculate_beta_x()` (Lines 1168-1248): Calculates coefficient mean parameters using ROI/contribution priors
- `linear_predictor_counterfactual_difference_*()`: Handles counterfactual calculations for calibration

##### Sampling Methods
- `sample_prior()` (Lines 1435-1446): Draws samples from prior distributions
- `sample_posterior()` (Lines 1448-1539): Runs MCMC sampling with NUTS

### 2. **Key Mathematical Components**

#### Response Curves (Lines 1250-1353)
```python
# Media transformation pipeline
media_out = media
for transformer in [adstock_transformer, hill_transformer]:
    media_out = transformer.forward(media_out)
```

#### Hierarchical Model Structure
- National vs geo-level models
- Baseline geo handling (Lines 397-421)
- Random effects distribution options (normal vs log-normal)

#### Prior Broadcasting (Lines 430-456)
- Adapts prior distributions to match data dimensions
- Handles special case of "total media contribution prior"

### 3. **Validation Logic**

#### Data Quality Checks
- No constant variables across geos/time
- Proper variation for identifiability
- ROI prior validation for non-revenue KPIs
- Dimension matching between data and model spec

#### Model Configuration Validation
- National model adjustments
- Prior type consistency
- Baseline geo selection
- Holdout data handling

### 4. **Cached Properties Pattern**
Uses `@functools.cached_property` extensively for:
- Expensive computations (transformers, tensors)
- One-time initialization (samplers)
- Derived values (baseline_geo_idx)

### 5. **Persistence Functions** (Lines 1542-1581)
- `save_mmm()`: Serializes model to pickle
- `load_mmm()`: Deserializes model from pickle
- Uses joblib for efficient serialization

## Key Design Patterns

1. **Lazy Evaluation**: Heavy computations only done when accessed
2. **Immutable Data**: InputData is frozen after creation
3. **Validation-First**: Extensive checks during initialization
4. **Backend Abstraction**: Works with both TensorFlow and JAX
5. **Probabilistic Programming**: Built on TensorFlow Probability

## Critical Methods for MMM Understanding

1. **Response Curves**: `adstock_hill_media()` and `adstock_hill_rf()`
   - Core transformations that model media effects

2. **Calibration**: `calculate_beta_x()`
   - Implements Bayesian calibration with business priors

3. **Sampling**: `sample_posterior()`
   - MCMC engine for parameter estimation

4. **Validation**: Various `_validate_*()` methods
   - Ensure model identifiability and data quality

This file is the central orchestrator that:
- Validates and prepares data
- Configures the Bayesian model
- Runs MCMC sampling
- Provides clean interfaces for analysis modules