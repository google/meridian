# Meridian Response Curves: Adstock and Hill Saturation

## Overview

Meridian uses two key transformations to model realistic media effects:
1. **Adstock (Carryover)**: Models how media effects persist over time
2. **Hill Saturation**: Models diminishing returns as spend increases

These can be applied in either order: Hill→Adstock or Adstock→Hill (configurable).

## 1. Adstock Transformation

### Purpose
Adstock models the lagged effect of advertising - the idea that media exposure today continues to influence consumers in future periods.

### Mathematical Formulation

#### Geometric Decay (Default)
For geo $g$, time $t$, and media channel $m$:

$$\text{adstock}_{g,t,m} = \sum_{i=0}^{\text{max\_lag}} w_i \cdot \text{media}_{g,t-i,m}$$

Where the weights are:
$$w_i = \frac{\alpha^i}{\sum_{j=0}^{\text{max\_lag}} \alpha^j}$$

- $\alpha \in [0, 1]$: Decay parameter (0 = no carryover, 1 = full carryover)
- $i$: Lag period
- Weights are normalized to sum to 1

#### Binomial Decay (Alternative)
For binomial decay, the weights are:

$$w_i = \frac{\left(1 - \frac{i}{\text{window\_size}}\right)^{\alpha^*}}{\sum_{j=0}^{\text{max\_lag}} \left(1 - \frac{j}{\text{window\_size}}\right)^{\alpha^*}}$$

Where $\alpha^* = \frac{1}{\alpha} - 1$ (maps $[0,1] \to [0,\infty]$)

### Implementation Details (`adstock_hill.py`)

```python
def _adstock(media, alpha, max_lag, n_times_output, decay_function):
    # 1. Calculate window size
    window_size = min(max_lag + 1, n_media_times)
    
    # 2. Prepare media data (pad if necessary)
    required_n_media_times = n_times_output + window_size - 1
    
    # 3. Create windowed views
    for i in range(window_size):
        window_list[i] = media[..., i : i + n_times_output, :]
    
    # 4. Compute weights
    weights = compute_decay_weights(alpha, l_range, window_size, decay_function)
    
    # 5. Apply weighted sum
    return einsum('...mw,w...gtm->...gtm', weights, windowed)
```

Key features:
- **Efficient windowing**: Uses sliding windows for computation
- **Automatic padding**: Handles edge cases at time series boundaries
- **Normalized weights**: Ensures weights sum to 1
- **Batch processing**: Supports vectorized operations across samples

## 2. Hill Saturation

### Purpose
Hill saturation models diminishing returns - as media spend increases, each additional unit produces less incremental effect.

### Mathematical Formulation

For media input $x$:

$$\text{Hill}(x) = \frac{x^s}{x^s + K^s}$$

Where:
- $s$: Slope parameter (controls steepness of the curve)
- $K$: Half-saturation point (EC50 - where response = 0.5)

In Meridian's implementation:
- $s$ = `slope` parameter
- $K$ = `ec` parameter

### Properties
- When $x = 0$: $\text{Hill}(0) = 0$
- When $x = K$: $\text{Hill}(K) = 0.5$
- When $x \to \infty$: $\text{Hill}(x) \to 1$
- Slope at origin: Proportional to $\frac{s}{K}$

### Implementation Details

```python
def _hill(media, ec, slope):
    t1 = media ** slope[..., newaxis, newaxis, :]
    t2 = (ec ** slope)[..., newaxis, newaxis, :]
    return t1 / (t1 + t2)
```

## 3. Combined Response Curve

### Order of Operations

The final response depends on the order of transformations:

#### Option 1: Adstock → Hill (Default)
$$\text{Response} = \text{Hill}(\text{Adstock}(\text{media}))$$

This means:
1. First apply carryover effects
2. Then apply saturation to the accumulated effect

#### Option 2: Hill → Adstock
$$\text{Response} = \text{Adstock}(\text{Hill}(\text{media}))$$

This means:
1. First apply saturation to each time period
2. Then spread the saturated effects over time

### In Code (`model.py`)

```python
def adstock_hill_media(self, media, alpha, ec, slope, n_times_output=None):
    # Create transformers
    adstock_transformer = AdstockTransformer(alpha, max_lag, n_times_output)
    hill_transformer = HillTransformer(ec, slope)
    
    # Apply in configured order
    transformers_list = (
        [hill_transformer, adstock_transformer]
        if self.model_spec.hill_before_adstock
        else [adstock_transformer, hill_transformer]
    )
    
    media_out = media
    for transformer in transformers_list:
        media_out = transformer.forward(media_out)
    return media_out
```

## 4. Special Case: Reach & Frequency

For R&F channels, the transformation is:

$$\text{RF\_Response} = \text{Adstock}(\text{reach} \times \text{Hill}(\text{frequency}))$$

This models:
1. Optimal frequency via Hill saturation
2. Reach as a multiplier
3. Carryover of the combined effect

```python
def adstock_hill_rf(self, reach, frequency, alpha, ec, slope):
    # Apply Hill to frequency
    adj_frequency = hill_transformer.forward(frequency)
    
    # Multiply by reach and apply adstock
    rf_out = adstock_transformer.forward(reach * adj_frequency)
    return rf_out
```

## 5. Parameter Interpretation

### Adstock Alpha ($\alpha$)
- **Range**: $[0, 1]$
- **Interpretation**:
  - $\alpha = 0$: No carryover (immediate decay)
  - $\alpha = 0.5$: 50% effect carries to next period
  - $\alpha = 0.9$: 90% effect carries over (slow decay)
- **Prior**: Typically `Uniform(0, 1)` or `Beta` distribution

### Hill EC (Half-saturation)
- **Range**: $(0, \infty)$
- **Interpretation**: Media level at which you get 50% of maximum effect
- **Scale**: Should be in same units as media input
- **Prior**: Often `HalfNormal` or `LogNormal`

### Hill Slope ($s$)
- **Range**: $(0, \infty)$
- **Interpretation**:
  - $s < 1$: Rapid initial saturation
  - $s = 1$: Linear at low spend (Michaelis-Menten)
  - $s > 1$: S-shaped curve with threshold effect
- **Prior**: Often fixed at 2.0 or given tight prior

## 6. Practical Implications

### Choosing Transformation Order

**Adstock → Hill** (Default):
- More conservative (saturation applies to accumulated effects)
- Better when media has strong carryover
- Prevents unrealistic buildup from high-frequency campaigns

**Hill → Adstock**:
- Saturation applies immediately to each period
- Better when saturation happens quickly
- Carryover applies to already-saturated effects

### Memory Optimization

The implementation includes:
- Sliding window approach (avoids full convolution)
- Automatic trimming of unnecessary historical data
- Padding only when required
- Efficient Einstein summation for batch operations

### Example Parameter Values

Typical ranges from real MMM applications:
- **Alpha**: 0.2-0.8 (most media shows 20-80% weekly decay)
- **EC**: 0.5-2x average weekly spend
- **Slope**: 1.5-3.0 (S-shaped response common)
- **Max lag**: 3-12 weeks (depends on media type)

## 7. Validation and Constraints

The code includes extensive validation:
- Ensures media is non-negative
- Validates dimension compatibility
- Checks parameter bounds
- Handles edge cases (e.g., `max_lag > n_times`)

## Key Takeaways

1. **Adstock** models temporal dynamics (carryover/decay)
2. **Hill** models cross-sectional dynamics (saturation)
3. **Order matters**: Results differ based on transformation sequence
4. **Efficient implementation**: Optimized for large-scale MCMC
5. **Flexible parameterization**: Supports different decay functions and priors