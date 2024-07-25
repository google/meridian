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

[Unreleased]: https://github.com/google/meridian/compare/v0.3.1...HEAD

## [0.3.1] - 2024-07-25

* Rollback changes in [0.4.0].

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
[0.3.1]: https://github.com/google/meridian/releases/tag/v0.3.1
