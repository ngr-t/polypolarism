# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `polypolarism` CLI now reports parse / read failures as failing
  `CheckResult`s instead of silently skipping them, so external callers
  (pre-commit, CI) cannot miss a broken file.
- Multi-file `--format json` output: each diagnostic carries its own `file`
  field, allowing pre-commit-style invocations such as
  `polypolarism --format json a.py b.py` to be attributed correctly.
- Project tooling for contributors: `[tool.ruff]` / `[tool.pyright]`
  configuration, `.pre-commit-config.yaml`, ruff / ruff-format / pyright
  in CI without `|| true`, and Codecov upload.
- `py.typed` marker so installs expose polypolarism's own type hints.
- `.pre-commit-hooks.yaml` so downstream projects can register
  polypolarism via `repo: https://github.com/ngr-t/polypolarism`.
- `publish.yml` workflow for PyPI Trusted Publishing (manual dispatch /
  tag-triggered; not yet wired up to a published project).

### Changed

- `FrameType.__init__` now formally accepts
  `Mapping[str, ColumnSpec | DataType]`, matching the runtime
  normalization that was already happening in `__post_init__`.
