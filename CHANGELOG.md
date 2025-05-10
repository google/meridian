# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google/meridian/compare/v1.0.0...v2.0.0`
* Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

* Change `VEGALITE_FACET_EXTRA_LARGE_WIDTH` from 900 to 700.
* Prevent negative media effect priors when using lognormal distribution upon
  model init.
* Add an optional `optimization_grid` arg to the optimizer.
* Fix `incremental_outcome` to accept unscaled `non_media_treatments_baseline`.
* Add spend allocation per geo and time if per-channel spend is provided.

## [1.0.9] - 2025-04-17

* Add support for optimization with forecasted data.
* Deprecate `get_historical_spend` for `get_aggregated_spend` with `new_data`
  support.
* Raise an error when `kpi_scaled` is all zero and `paid_media_prior_type` is
  anything other than "coefficient".
* Prevent negative media effect priors when using lognormal distribution.
* Add support for weekly time granularity to the contribution area and bump
  charts.
* Update channel contribution over time charts in Summarizer to toggle weekly vs
  quarterly granularity based on the selected time period length.
* Adjust the optimization summary to display dates in the same format as
  "Marketing Mix Modeling Report".
* Increase width of model fit and channel contribution over time charts in the
  Visualizer.

## [1.0.8] - 2025-04-08

* Update contribution calculation methods in `MediaSummary` with
  `aggregate_times` parameter to support granular time.
* Add a `new_data` argument to `analyzer.optimal_freq()`.
* Refactor args in `create_optimization_grid` to be consistent with
  `optimize(...)`.
* Fix response curves for KPI-based optimization.
* Add `plot_channel_contribution_area_chart` method to `MediaSummary` in the
  visualizer.
* Add `plot_channel_contribution_bump_chart` method to `MediaSummary` in the
  visualizer.
* Add organic media support for adstock decay in analyzer.
* Add channel contribution area chart and channel contribution bump chart to
  model results summary report in the summarizer.
* Add an extra check for zeros or negative values in `revenue_per_kpi`.
* Add per-channel constraints parameters to `OptimizationGrid.optimize(...)`.
* Add organic media support for hill curves in analyzer.
* Add organic media support for `plot_hill_curves` in visualizer.

## [1.0.7] - 2025-03-19

* Bump tensorflow to 2.18.
* Bump tensorflow-probability to 0.25.
* Bump numpy to 2.0.2.
* Bump pandas to 2.2.2.
* Bump scipy to 1.13.1.

## [1.0.6] - 2025-03-18

* Fix issue #548: Make time coordinate regularity check less stringent.
* Force `DataTensors` to have all tensors with `dtype=tf.float32`.
* Refactor new data validation and data filling into the `DataTensors` class.
* Fix bug in marginal ROI calculation in `summary_metrics` when new spend data
  is passed in.
* Add `by_reach` param to `incremental_outcome()` to allow scaling `reach` or
  `frequency`.
* Refactor `marginal_roi()` and the mROI calculation in `summary_metrics` to use
  the scaling factors in `incremental_outcome()`, removing duplicate code.
* Allow `new_data` to have any number of time dimensions for `roi()`,
  `marginal_roi()`, `cpik()`, and `summary_metrics()`. This allows the use of
  forecasted data for analysis.
* Add `optimize()` method to the `OptimizationGrid` dataclass.
* Add a warning when the target constraint of flexible budget optimization is
  not met.

## [1.0.5] - 2025-03-06

* Add technical support for python 3.10.
* Align `NaNs` in `spend_grid` and `incremental_outcome_grid` in the optimizer.
* Fix the stopping criteria of target total ROI in flexible budget optimization.
* Separate creation of the grid data and the optimization.

## [1.0.4] - 2025-02-28

* Fix validation on injected inference data when `unique_sigma_for_each_geo` is
  used in the model initialization.
* Fix a divide-by-zero error in spend ratio calculation when historical spend is
  zero, preventing a `ValueError` in `output_optimization_summary`.
* Add `non_media_baseline_values` argument to `MediaSummary` visualizations.
* Refactor prior and posterior sampling logic into separate modules, simplifying
  `model` module.
* Create a helper argument builder construct for API parameters that require
  an ordered list/array of values.
  See, e.g., `InputData.get_paid_channels_argument_builder()`.

## [1.0.3] - 2025-02-07

* Temporarily downgrade `tensorflow` version to 2.16 until the convergence
  issues on L4 and A100 GPU runtimes are resolved.

## [1.0.2] - 2025-02-06

* Bump minimum `tensorflow` version to 2.18.
* Add `[and-cuda]` optional dependencies to install `tensorflow` dependency with
  GPU support.
* Add `non_media_baseline_values` argument to `summary_metrics`,
  `baseline_summary_metrics` and `expected_vs_actual` methods.
* Update `compute_incremental_outcome_aggregate` docstring to match
  `incremental_outcome`.

## [1.0.1] - 2025-02-04

* Bump minimum `pandas` version to 2.2.
* Make `compute_incremental_outcome_aggregate` public.
* Add `new_data` argument to `Analyzer.summary_metrics` method.
* Add `use_kpi` argument to the `optimize()` method.

## [1.0.0] - 2025-01-24

* Bump `tensorflow` version to 2.16 to support Python 3.12.

## [0.17.0] - 2025-01-23

* Define constants for channel constraints in the optimizer.
* Remove `aggregate_times` from `roi`, `marginal_roi`, and `cpik` methods in
  `Analyzer` and do not report these metrics in the `summary_metrics` method
  when `aggregate_times=False` as these metrics do not have a clear
  interpretation by time period.

## [0.16.0] - 2025-01-08

* Organize tensor arguments of `roi`, `mroi`, and `cpik` methods of Analyzer
  into a `DataTensors` container.
* Add warning message when user sets custom priors that will be ignored by the
  `paid_media_prior_type` argument.

## [0.15.0] - 2025-01-07

* Convert `InputData` geo coordinates to strings upon initialization to avoid
  type mismatches with `GeoInfo` proto which expects strings.
* Add `get_historical_spend` method to `Analyzer` class.
* Split up `roi_*` and `mroi_*` parameters.

## [0.14.0] - 2024-12-17

* Remove deprecated `use_roi_prior` attribute from `ModelSpec`.

## [0.13.0] - 2024-12-11

* Add support for marginal ROI priors in Meridian.

## [0.12.0] - 2024-12-09

* Rename `incremental_impact` to `incremental_outcome`.
* Rename `plot_incremental_impact_delta` to `plot_incremental_outcome_delta`.

## [0.11.2] - 2024-11-27

* Remove deprecated `all_channel_names` property from `Meridian` class.

## [0.11.1] - 2024-11-22

* Remove unneeded argument `include_non_paid_channels` from
  `expected_outcome()`.
* Fix a bug in the custom RF prior validation.

## [0.11.0] - 2024-11-19

* Consistent naming for "rhat" methods.

## [0.10.0] - 2024-11-18

* Add support for organic media, organic reach and frequency, and non-media
  treatment variables.
* Rename `Analyzer.media_summary_metrics` method to `Analyzer.summary_metrics`
  with `include_non_paid_channels` argument.

## [0.9.0] - 2024-11-15

* Organize arguments of `incremental_impact` and `expected_outcome` methods into
  a `DataTensors` container.

## [0.8.0] - 2024-11-05

* Expand media summary metrics to return ROI, mROI, and CPIK in all scenarios
  with the addition of the `use_kpi` argument.
* Optimal frequency now calculates the frequency that maximizes the mean ROI in
  all cases such that it is consistent when used in the budget optimization that
  optimizes revenue.
* Fix an error in the data loader that occurs when the geo column is an integer.
* Add a `_check_if_no_time_variation` method to Meridian to raise an error if a
  variable has no time variation.
* Make the performance breakdown section of the model summary report display
  both ROI and CPIK charts for all scenarios.
* Set default ROI priors for non-revenue, no revenue-per-KPI models.
* Do not specify significant digits in the y-axis labels in plot_spend_delta,
  trim insignificant trailing zeros in all charts.
* Rename `ControlsTransformer` to `CenteringAndScalingTransformer`.

## [0.7.0] - 2024-09-20

* Make `get_r_hat` public.
* Add `media_selected_times` parameter to `Analyzer.incremental_impact()`
  method.
  This allows, among other things, to project impact for future media values.
* For `"All Channels"` media summary metrics: `effectiveness` and `mroi` data
  variables are now masked out (`math.nan`).
* Introduce a `data.TimeCoordinates` construct.
* Pin numpy dependency to ">= 1.26, < 2".
* `InputData` now has `[media_]*time_coordinates` properties.
* `InputData` now explicitly checks that time coordinate values are evenly
  spaced.

## [0.6.0] - 2024-08-20

* Add `Analyzer.baseline_summary_metrics()` method.
* Fix a bug where custom priors were sometimes not able to be detected.
* Fix a bug in the controls transformer with mean and stddev computations.

## [0.5.0] - 2024-08-15

* Include `pct_of_contribution` and `effectiveness` data to
  `OptimizationResults` datasets.
* Add `Analyzer.get_aggregated_impressions()` method.
* Add `spend_step_size` to `OptimizationResults.optimization_grid`.
* Add `use_posterior` argument to the budget optimizer.
* Rename `expected_impact` to `expected_outcome`.

## [0.4.0] - 2024-07-19

* Refactor `BudgetOptimizer.optimize()` API: it now returns an
  `OptimizationResults` dataclass.

## [0.3.0] - 2024-07-19

* Rename `tau_t` to `mu_t` throughout.

## [0.2.0] - 2024-07-16

## 0.1.0 - 2022-01-01

* Initial release

<!-- mdlint off(LINK_UNUSED_ID) -->
[0.2.0]: https://github.com/google/meridian/releases/tag/v0.2.0
[0.3.0]: https://github.com/google/meridian/releases/tag/v0.3.0
[0.4.0]: https://github.com/google/meridian/releases/tag/v0.4.0
[0.5.0]: https://github.com/google/meridian/releases/tag/v0.5.0
[0.6.0]: https://github.com/google/meridian/releases/tag/v0.6.0
[0.7.0]: https://github.com/google/meridian/releases/tag/v0.7.0
[0.8.0]: https://github.com/google/meridian/releases/tag/v0.8.0
[0.9.0]: https://github.com/google/meridian/releases/tag/v0.9.0
[0.10.0]: https://github.com/google/meridian/releases/tag/v0.10.0
[0.11.0]: https://github.com/google/meridian/releases/tag/v0.11.0
[0.11.1]: https://github.com/google/meridian/releases/tag/v0.11.1
[0.11.2]: https://github.com/google/meridian/releases/tag/v0.11.2
[0.12.0]: https://github.com/google/meridian/releases/tag/v0.12.0
[0.13.0]: https://github.com/google/meridian/releases/tag/v0.13.0
[0.14.0]: https://github.com/google/meridian/releases/tag/v0.14.0
[0.15.0]: https://github.com/google/meridian/releases/tag/v0.15.0
[0.16.0]: https://github.com/google/meridian/releases/tag/v0.16.0
[0.17.0]: https://github.com/google/meridian/releases/tag/v0.17.0
[1.0.0]: https://github.com/google/meridian/releases/tag/v1.0.0
[1.0.1]: https://github.com/google/meridian/releases/tag/v1.0.1
[1.0.2]: https://github.com/google/meridian/releases/tag/v1.0.2
[1.0.3]: https://github.com/google/meridian/releases/tag/v1.0.3
[1.0.4]: https://github.com/google/meridian/releases/tag/v1.0.4
[1.0.5]: https://github.com/google/meridian/releases/tag/v1.0.5
[1.0.6]: https://github.com/google/meridian/releases/tag/v1.0.6
[1.0.7]: https://github.com/google/meridian/releases/tag/v1.0.7
[1.0.8]: https://github.com/google/meridian/releases/tag/v1.0.8
[1.0.9]: https://github.com/google/meridian/releases/tag/v1.0.9
[Unreleased]: https://github.com/google/meridian/compare/v1.0.9...HEAD
