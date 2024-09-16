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

* Make `get_r_hat` public.
* Add `media_selected_times` parameter to `Analyzer.incremental_impact()`
  method.
  This allows, among other things, to project impact for future media values.
* For `"All Channels"` media summary metrics: `effectiveness` and `mroi` data
  variables are now masked out (`math.nan`).
* Introduce a `data.TimeCoordinates` construct.
* `InputData` now has `[media_]*time_coordinates` properties.
* Pin numpy dependency to ">= 1.26, < 2".

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

[0.2.0]: https://github.com/google/meridian/releases/tag/v0.2.0
[0.3.0]: https://github.com/google/meridian/releases/tag/v0.3.0
[0.4.0]: https://github.com/google/meridian/releases/tag/v0.4.0
[0.5.0]: https://github.com/google/meridian/releases/tag/v0.5.0
[0.6.0]: https://github.com/google/meridian/releases/tag/v0.6.0
[Unreleased]: https://github.com/google/meridian/compare/v0.6.0...HEAD
