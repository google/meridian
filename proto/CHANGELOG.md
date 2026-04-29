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
  `[2.0.0]: https://github.com/google/meridian/compare/proto-v1.0.0...proto-v1.1.0`
* Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v1.1.0...HEAD`

-->

## [Unreleased]

## [1.1.4] - 2026-04-28

*   Add `saturation_spec` to `Hyperparameters` proto to support selective
    saturation.

## [1.1.3] - 2026-04-22

*   Add `ComputationPrecision` proto for Meridian models.

## [1.1.2] - 2026-03-02

*   Add `population_value` field to `GeoInfo` proto.
*   Add `ComputationBackend` proto for Meridian models.

## [1.1.1] - 2025-12-22

*   Add field to record ArviZ version in meridian_model proto definition.

## [1.1.0] - 2025-12-08

*   Add proto definitions needed for Meridian Scenario Planner.

## [1.0.0] - 2025-11-12

*   Initial release of `mmm-proto-schema` package.
*   Add proto definitions needed for Meridian `serde` package.

<!-- mdlint off(LINK_UNUSED_ID) -->

[1.0.0]: https://github.com/google/meridian/releases/tag/proto-v1.0.0
[1.1.0]: https://github.com/google/meridian/releases/tag/proto-v1.1.0
[1.1.1]: https://github.com/google/meridian/releases/tag/proto-v1.1.1
[1.1.2]: https://github.com/google/meridian/releases/tag/proto-v1.1.2
[1.1.3]: https://github.com/google/meridian/releases/tag/proto-v1.1.3
[1.1.4]: https://github.com/google/meridian/releases/tag/proto-v1.1.4
[Unreleased]: https://github.com/google/meridian/compare/proto-v1.1.4...HEAD
