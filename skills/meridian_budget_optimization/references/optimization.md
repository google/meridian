# Budget Optimization

Use `optimizer.BudgetOptimizer` to run optimization and generate the summary.

## Basic Usage

```python
from meridian.analysis import optimizer

budget_optimizer = optimizer.BudgetOptimizer(model)

# Run optimization (defaults to Fixed Budget with historical budget)
optimization_results = budget_optimizer.optimize(use_posterior=True)

# Save optimization summary to HTML
optimization_results.output_optimization_summary(
    filename="optimization.html", filepath=output_dir
)
```

## Comprehensive Usage Examples

### 1. Fixed Budget Optimization with Custom Budget and Constraints

```python
optimization_results = budget_optimizer.optimize(
    use_posterior=True,
    fixed_budget=True,
    budget=1000000.0,  # Custom total budget
    spend_constraint_lower=0.5,
    spend_constraint_upper=0.5,
)
```

### 2. Flexible Budget Optimization with Target ROI

```python
optimization_results = budget_optimizer.optimize(
    use_posterior=True,
    fixed_budget=False,
    target_roi=2.0,  # Find budget to achieve ROI of 2.0
)
```

### 3. Reach & Frequency Optimization with Optimal Frequency

```python
optimization_results = budget_optimizer.optimize(
    use_posterior=True,
    use_optimal_frequency=True,  # Use optimal frequency calculated by model
    max_frequency=3.0,  # Upper bound for optimal frequency search
)
```

## Detailed Arguments Reference

The `optimize` method takes the following arguments:

*   **`new_data`**: Optional `DataTensors` container to override data.
*   **`use_posterior`**: Boolean. If `True` (default), uses posterior
    distribution. If `False`, uses prior.
*   **`selected_geos`**: Optional list of geos to include. Defaults to all geos.
*   **`start_date` / `end_date`**: Optional date range selectors (inclusive) in
    'yyyy-mm-dd' format.
*   **`fixed_budget`**: Boolean. `True` for fixed budget (default), `False` for
    flexible budget.
*   **`budget`**: Number indicating total budget for fixed budget scenario.
    Defaults to historical budget.
*   **`pct_of_spend`**: Numeric list of size `n_paid_channels` containing
    percentage allocation.
*   **`spend_constraint_lower` / `spend_constraint_upper`**: Bounds for
    media-level spend. Defaults to `0.3` for fixed, `1.0` for flexible.
*   **`target_roi` / `target_mroi`**: Targets for flexible budget scenario.
*   **`gtol`**: Acceptable relative error for budget in grid setup (default
    `0.0001`).
*   **`use_optimal_frequency`**: If `True` (default), uses optimal frequency
    calculated by model for RF channels.
*   **`max_frequency`**: Frequency upper bound for optimal frequency search
    space.
*   **`use_kpi`**: If `True`, runs optimization on KPI instead of revenue
    (default `False`).
*   **`confidence_level`**: Threshold for computing confidence intervals.
*   **`batch_size`**: Maximum draws per chain in each batch to avoid memory
    exhaustion.
*   **`optimization_grid`**: Reuse an existing `OptimizationGrid` to save time.
