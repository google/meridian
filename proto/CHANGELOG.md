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

## [1.1.0] - 2025-12-08

* Add proto definitions needed for Meridian Scenario Planner.

## [1.0.0] - 2025-11-12

* Initial release of `mmm-proto-schema` package.
* Add proto definitions needed for Meridian `serde` package.

<!-- mdlint off(LINK_UNUSED_ID) -->
[1.0.0]: https://github.com/google/meridian/releases/tag/proto-v1.0.0
[1.1.0]: https://github.com/google/meridian/releases/tag/proto-v1.1.0
[Unreleased]: https://github.com/google/meridian/compare/proto-v1.1.0...HEAD
