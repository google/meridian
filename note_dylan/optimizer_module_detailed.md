# Meridian Optimizer Module: Budget Allocation Algorithms

## Overview

The Optimizer module (`/meridian/analysis/optimizer.py`) implements sophisticated budget allocation algorithms that maximize marketing outcomes subject to business constraints. It uses response curves from the fitted model to find optimal spend distributions across channels.

## 1. Core Optimization Concepts

### 1.1 Optimization Problem Formulation

**Objective**: Maximize total incremental outcome

$$\max_{\mathbf{s}} \sum_{c} f_c(s_c)$$

Where:
- $\mathbf{s} = [s_1, ..., s_C]$: Spend allocation vector
- $f_c(s_c)$: Response curve for channel $c$
- $C$: Number of channels

**Constraints**:

1. **Budget Constraint** (Fixed Budget):
   $$\sum_{c} s_c = B$$

2. **Box Constraints** (Per Channel):
   $$s_c^{\text{lower}} \leq s_c \leq s_c^{\text{upper}}$$
   
   Where:
   - $s_c^{\text{lower}} = (1 - \delta^-) \cdot s_c^{\text{hist}}$
   - $s_c^{\text{upper}} = (1 + \delta^+) \cdot s_c^{\text{hist}}$
   - $\delta^{\pm}$: Constraint multipliers (default 0.3)

3. **Target Constraints** (Flexible Budget):
   - ROI constraint: $\frac{\sum_c f_c(s_c)}{\sum_c s_c} = \text{ROI}_{\text{target}}$
   - mROI constraint: $\frac{\partial}{\partial s}\left[\sum_c f_c(s_c)\right] = \text{mROI}_{\text{target}}$

### 1.2 Optimization Scenarios

#### Fixed Budget Scenario
```python
@dataclass
class FixedBudgetScenario:
    total_budget: float | None = None  # Uses historical if None
```

#### Flexible Budget Scenario
```python
@dataclass
class FlexibleBudgetScenario:
    target_metric: str      # 'roi' or 'mroi'
    target_value: float     # Target ROI/mROI value
```

## 2. Grid-Based Optimization Algorithm

### 2.1 Grid Construction

**Purpose**: Pre-compute response curves on a discrete grid for efficiency.

```python
class OptimizationGrid:
    spend_grid: xr.DataArray         # Shape: (grid_points, n_channels)
    incremental_outcome_grid: xr.DataArray  # Same shape
    spend_step_size: float           # Grid resolution
```

**Grid Generation Process**:
1. Determine budget rounding factor based on tolerance (`gtol`)
2. Create spend range for each channel: $[s_c^{\text{lower}}, s_c^{\text{upper}}]$
3. Discretize with step size = round_factor
4. Pre-compute outcomes for all grid points

### 2.2 Hill-Climbing Search Algorithm

The core optimization uses a **greedy hill-climbing algorithm**:

```python
def _grid_search(self, spend_grid, incremental_outcome_grid, scenario):
    # Initialize at minimum spend
    spend = spend_grid[0, :].copy()
    incremental_outcome = incremental_outcome_grid[0, :].copy()
    
    # Compute marginal ROI grid
    iterative_roi_grid = (incremental_outcome_grid - incremental_outcome) / 
                        (spend_grid - spend)
    
    while True:
        # Find highest marginal ROI
        point = np.unravel_index(np.nanargmax(iterative_roi_grid), 
                               iterative_roi_grid.shape)
        row_idx, channel_idx = point
        
        # Update spend allocation
        spend[channel_idx] = spend_grid[row_idx, channel_idx]
        incremental_outcome[channel_idx] = incremental_outcome_grid[row_idx, channel_idx]
        
        # Check stopping criteria
        if _exceeds_optimization_constraints(spend, incremental_outcome, scenario):
            break
            
        # Update marginal ROI for this channel
        iterative_roi_grid[:row_idx+1, channel_idx] = np.nan
        iterative_roi_grid[row_idx+1:, channel_idx] = 
            (incremental_outcome_grid[row_idx+1:, channel_idx] - 
             incremental_outcome_grid[row_idx, channel_idx]) /
            (spend_grid[row_idx+1:, channel_idx] - 
             spend_grid[row_idx, channel_idx])
```

**Algorithm Logic**:
1. Start with minimum spend for all channels
2. Find channel with highest marginal ROI
3. Increment spend for that channel
4. Update marginal ROIs
5. Repeat until constraints are met

### 2.3 Mathematical Intuition

The algorithm implements the **equimarginal principle**:

At optimum: $\frac{\partial f_i}{\partial s_i} = \frac{\partial f_j}{\partial s_j} = \lambda$

Where $\lambda$ is the Lagrange multiplier (shadow price of budget).

## 3. Key Implementation Features

### 3.1 Optimal Frequency Integration

For R&F channels, the optimizer can use optimal frequency:

```python
if use_optimal_frequency:
    # Calculate optimal frequency from model
    optimal_freq = analyzer.optimal_frequency()
    # Use in optimization instead of historical
```

This leverages the R&F model's ability to find frequency sweet spots.

### 3.2 Flighting Pattern Preservation

**Key Assumption**: Optimization preserves relative spend patterns across time/geo.

$$\frac{s_{g,t,c}}{\sum_{g,t} s_{g,t,c}} = \frac{s_{g,t,c}^{\text{hist}}}{\sum_{g,t} s_{g,t,c}^{\text{hist}}}$$

This means:
- Geo allocation proportions stay fixed
- Seasonal patterns are maintained
- Only total channel budgets change

### 3.3 Memory-Efficient Grid Operations

```python
def trim_grids(self, spend_bound_lower, spend_bound_upper):
    # Keep only valid spend ranges
    for ch in range(len(self.channels)):
        valid_indices = np.where(
            (spend_grid[:, ch] >= spend_bound_lower[ch]) &
            (spend_grid[:, ch] <= spend_bound_upper[ch])
        )[0]
        # Efficiently roll and mask invalid regions
```

## 4. Optimization Workflow

### 4.1 Main Entry Point

```python
budget_optimizer = BudgetOptimizer(meridian)
results = budget_optimizer.optimize(
    fixed_budget=True,
    budget=1_000_000,
    spend_constraint_lower=0.5,  # 50% lower bound
    spend_constraint_upper=0.5,  # 50% upper bound
    use_optimal_frequency=True,
    use_kpi=False  # Optimize revenue
)
```

### 4.2 Processing Steps

1. **Validation**:
   - Check model is fitted
   - Validate budget parameters
   - Ensure constraints are feasible

2. **Grid Creation**:
   - Determine spend bounds
   - Calculate grid resolution
   - Pre-compute response curves

3. **Optimization**:
   - Run hill-climbing algorithm
   - Find optimal allocation
   - Validate results meet constraints

4. **Results Packaging**:
   - Calculate metrics (ROI, mROI, effectiveness)
   - Compare to non-optimized baseline
   - Generate visualizations

## 5. Optimization Results

### 5.1 OptimizationResults Class

```python
@dataclass
class OptimizationResults:
    nonoptimized_data: xr.Dataset      # Historical allocation metrics
    optimized_data: xr.Dataset         # Optimal allocation metrics
    optimization_grid: OptimizationGrid # Grid used for optimization
    
    # Key metrics in datasets:
    # - spend: Channel budgets
    # - roi: Return on investment
    # - mroi: Marginal ROI
    # - incremental_outcome: Total effect
    # - effectiveness: Saturation level
```

### 5.2 Key Metrics Computed

**ROI Improvement**:
$$\text{ROI Lift} = \frac{\text{ROI}_{\text{optimal}} - \text{ROI}_{\text{current}}}{\text{ROI}_{\text{current}}} \times 100\%$$

**Budget Reallocation**:
$$\text{Spend Delta}_c = s_c^{\text{optimal}} - s_c^{\text{current}}$$

**Efficiency Gain**:
$$\text{Efficiency} = \frac{\text{Outcome}_{\text{optimal}}}{\text{Outcome}_{\text{current}}}$$

## 6. Visualization Components

### 6.1 Incremental Outcome Waterfall
Shows how outcome changes by channel reallocation:
- Decreases (red) from over-invested channels
- Increases (cyan) to under-invested channels
- Net effect (blue)

### 6.2 Response Curves with Constraints
Displays for each channel:
- Full response curve
- Current operating point
- Optimal operating point
- Feasible region (shaded)

### 6.3 Spend Allocation Comparison
- Pie charts: Current vs Optimal
- Bar chart: Spend deltas by channel
- Efficiency metrics table

## 7. Advanced Features

### 7.1 Multiple Time Period Support

```python
def optimize(self, start_date=None, end_date=None, ...):
    # Can optimize for specific time windows
    # Useful for campaign planning
```

### 7.2 What-If Analysis

```python
# Test different budget levels
new_data = DataTensors(
    media=scaled_media,  # Scaled by some factor
    media_spend=scaled_spend
)
results = optimizer.optimize(new_data=new_data)
```

### 7.3 Custom Constraints

```python
# Channel-specific constraints
spend_constraint_lower = [0.5, 0.3, 0.7, ...]  # Per channel
spend_constraint_upper = [0.3, 0.5, 0.2, ...]
```

## 8. Mathematical Properties

### 8.1 Optimality Conditions

At optimal allocation:
1. **First-Order**: Marginal ROIs equalized (subject to constraints)
2. **Complementary Slackness**: Constraints are active or have zero shadow price
3. **Global Optimum**: Guaranteed for convex response curves

### 8.2 Convergence Guarantees

The hill-climbing algorithm converges because:
1. Finite grid ensures finite iterations
2. Monotonic improvement at each step
3. Bounded objective function

### 8.3 Constraint Handling

**Box Constraints**: Naturally handled by grid bounds
**Budget Constraint**: Maintained throughout search
**Target Constraints**: Used as stopping criteria

## 9. Practical Considerations

### 9.1 Grid Resolution Trade-offs

- **Fine Grid** (`gtol=0.0001`): More accurate but slower
- **Coarse Grid** (`gtol=0.01`): Faster but less precise
- Default balances accuracy and speed

### 9.2 Constraint Setting Guidelines

```python
# Conservative reallocation
spend_constraint_lower = 0.3  # ±30%
spend_constraint_upper = 0.3

# Aggressive reallocation  
spend_constraint_lower = 0.7  # ±70%
spend_constraint_upper = 0.7

# Asymmetric constraints
spend_constraint_lower = 0.5  # Can decrease 50%
spend_constraint_upper = 1.0  # Can increase 100%
```

### 9.3 Scenario Selection

- **Fixed Budget**: Most common, total spend predetermined
- **Target ROI**: Find budget that achieves desired efficiency
- **Target mROI**: Stop when marginal returns hit threshold

## 10. Key Algorithmic Insights

1. **Greedy is Optimal**: For convex problems, greedy hill-climbing finds global optimum
2. **Pre-computation**: Grid approach trades memory for speed
3. **Incremental Updates**: Efficient marginal ROI recalculation
4. **Constraint Projection**: Automatic handling via grid bounds

## Summary

The Optimizer module implements a sophisticated yet practical approach to budget allocation:
- **Grid-based** pre-computation for efficiency
- **Hill-climbing** algorithm leveraging equimarginal principle  
- **Flexible scenarios** supporting various business constraints
- **Rich visualizations** for stakeholder communication
- **Mathematical rigor** with practical considerations

This design makes MMM insights actionable by providing clear, optimal budget recommendations backed by rigorous optimization theory.