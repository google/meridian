# Meridian MMM: Complete Technical Summary

## Overview

Meridian is Google's state-of-the-art Bayesian Media Mix Modeling (MMM) framework. This document summarizes the key technical insights from our comprehensive exploration of the codebase and provides a suggested reading order for the detailed documentation.

## What is Meridian?

Meridian is a sophisticated MMM implementation that:
- Uses **Bayesian inference** to quantify uncertainty in marketing effectiveness
- Implements **hierarchical geo-level modeling** for granular insights
- Features **advanced response curves** (adstock carryover + Hill saturation)
- Enables **business knowledge calibration** through flexible prior systems
- Provides **actionable optimization** for budget allocation
- Supports both **TensorFlow and JAX** backends

## Core Technical Architecture

### 1. **Data Flow**
```
Raw CSV → InputData (xarray) → Transformations → Model → MCMC → Analysis → Optimization
```

### 2. **Model Structure**
The Bayesian hierarchical model can be expressed as:

$$\text{KPI}_{g,t} = \tau_g + \mu_t + \sum_m \beta_{g,m} \cdot \text{Hill}(\text{Adstock}(\text{media}_{g,t,m})) + \text{controls} + \epsilon$$

Where:
- $\tau_g$: Geo-level intercepts
- $\mu_t$: Time effects (flexible splines)
- $\beta_{g,m}$: Hierarchical media coefficients
- Response curves model carryover and saturation

### 3. **Key Innovations**

1. **Bayesian Calibration**: ROI/mROI/contribution priors encode business knowledge
2. **Flexible Response Curves**: Order matters (Hill→Adstock vs Adstock→Hill)
3. **Reach & Frequency**: Optimal frequency modeling for R&F channels
4. **Grid-Based Optimization**: Efficient budget allocation via hill-climbing
5. **Comprehensive Uncertainty**: Full posterior distributions, not point estimates

## Reading Guide: Suggested Order

### Phase 1: Foundation (Start Here)
1. **`meridian_workflow_complete.md`**
   - Overview of the complete workflow
   - Understand the big picture
   - See how components connect

2. **`demo_notebook_summary.md`**
   - Concrete example of full workflow
   - Practical implementation
   - Expected inputs/outputs

### Phase 2: Core Mechanics
3. **`meridian_function_mapping.md`**
   - Detailed function call hierarchy
   - Understand code organization
   - Navigate the codebase efficiently

4. **`response_curves_detailed.md`**
   - Mathematical heart of MMM
   - Adstock decay formulations
   - Hill saturation mathematics
   - Critical for understanding media effects

5. **`prior_distribution_detailed.md`**
   - Bayesian prior system
   - Calibration methods (ROI, mROI, contribution)
   - Business knowledge integration

### Phase 3: Implementation Details
6. **`model_py_overview.md`**
   - Central Meridian class
   - Model initialization and validation
   - MCMC sampling implementation

7. **`parameter_naming_conventions.md`**
   - Systematic naming patterns
   - Parameter relationships
   - Essential for code comprehension

### Phase 4: Analysis & Optimization
8. **`analyzer_module_detailed.md`**
   - Business metric calculations
   - ROI, mROI, CPIK formulas
   - Causal inference implementation

9. **`optimizer_module_detailed.md`**
   - Budget allocation algorithms
   - Hill-climbing optimization
   - Constraint handling

## Key Technical Takeaways

### 1. **Response Curve Mathematics**

**Adstock (Geometric Decay)**:
$$w_i = \frac{\alpha^i}{\sum_{j=0}^{\text{max\_lag}} \alpha^j}$$

**Hill Saturation**:
$$\text{Hill}(x) = \frac{x^s}{x^s + K^s}$$

### 2. **Prior Calibration System**

Four methods to encode business knowledge:
- **ROI**: Total return expectations
- **mROI**: Marginal return at current spend
- **Contribution**: Percentage of total outcome
- **Coefficient**: Direct parameter priors

### 3. **Optimization Algorithm**

Greedy hill-climbing implementing equimarginal principle:
- At optimum: $\frac{\partial f_i}{\partial s_i} = \frac{\partial f_j}{\partial s_j}$
- Pre-computed grids for efficiency
- Handles complex constraints elegantly

### 4. **Naming Conventions**

Systematic suffixes indicate scope:
- `_m`: Media channels
- `_rf`: Reach & frequency
- `_g`: Geo-level
- `_om`: Organic media

Prefixes indicate function:
- `alpha_`: Adstock decay
- `ec_`: Hill saturation point
- `beta_`: Effect coefficients
- `eta_`: Hierarchical std dev

## Practical Workflow

### 1. **Data Preparation**
```python
builder = DataFrameInputDataBuilder(kpi_type='revenue')
data = builder.with_media(...).with_controls(...).build()
```

### 2. **Model Configuration**
```python
prior = PriorDistribution(roi_m=LogNormal(0.2, 0.9))
model_spec = ModelSpec(prior=prior)
```

### 3. **Fitting**
```python
mmm = Meridian(input_data=data, model_spec=model_spec)
mmm.sample_posterior(n_chains=10, n_keep=1000)
```

### 4. **Analysis**
```python
analyzer = Analyzer(mmm)
roi = analyzer.roi()
incremental = analyzer.incremental_outcome()
```

### 5. **Optimization**
```python
optimizer = BudgetOptimizer(mmm)
results = optimizer.optimize(fixed_budget=True)
```

## Design Philosophy

Meridian embodies several key design principles:

1. **Statistical Rigor + Business Pragmatism**: Sophisticated Bayesian methods with business-friendly interfaces
2. **Flexibility Without Complexity**: Many options, sensible defaults
3. **Uncertainty is Feature**: Full distributions, not just point estimates
4. **Performance at Scale**: GPU acceleration, batch processing
5. **Reproducible Science**: Clear model specification, serialization

## Architecture Highlights

- **Clean Separation**: Data → Model → Analysis → Optimization
- **Type Safety**: Extensive validation and type hints
- **Backend Abstraction**: TensorFlow/JAX agnostic
- **Memory Efficiency**: Batch processing, sliding windows
- **Extensibility**: Clear interfaces for customization

## Key Files for Deep Dives

1. **Model Core**: `/meridian/model/model.py`
2. **Response Curves**: `/meridian/model/adstock_hill.py`
3. **Data Structure**: `/meridian/data/input_data.py`
4. **Prior System**: `/meridian/model/prior_distribution.py`
5. **Analysis Engine**: `/meridian/analysis/analyzer.py`
6. **Optimizer**: `/meridian/analysis/optimizer.py`

## Conclusion

Meridian represents a significant advancement in MMM:
- **Methodologically**: Hierarchical Bayesian modeling with business calibration
- **Practically**: Production-ready with comprehensive tooling
- **Philosophically**: Bridges statistical rigor and business needs

The codebase is exceptionally well-designed, with clear abstractions, comprehensive validation, and thoughtful APIs. It sets a high bar for production ML systems in marketing analytics.

## Next Steps

1. **Run the Demo**: Execute `Meridian_Getting_Started.ipynb`
2. **Experiment**: Modify priors, response curves, optimization constraints
3. **Extend**: Add custom transformations or visualization
4. **Apply**: Use on real marketing data

The documentation in `note_dylan/` provides comprehensive technical details for each component. Follow the suggested reading order for the most coherent learning path.