# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New warning `PLW007`: a method polypolarism does not model on a
  precisely-known receiver now warns that the dtype degrades to
  `Unknown` (one warning per chain; a `.cast(...)` directly after the
  call retracts it).
- `arr.eval(...)` inference: `as_list=True` yields `List(body dtype)`,
  `as_list=False`/omitted yields `Array(body dtype)`; the eval body is
  now type-checked in all forms (probed on polars 1.41.2).
- `str.to_datetime` resolves `Datetime[UTC]` for every chrono offset
  directive (`%z`, `%:z`, `%::z`, `%:::z`, `%#z`), not just `%z`.
- Int-constant propagation for `rolling_*` arguments: a local `ms = 1`
  or module-level `MIN_SAMPLES = 1` now resolves `min_samples` /
  `window_size` / `ddof` like a literal when deciding nullability.
- README sections: leniency rules ("when 'no error' is not a proof")
  and the previously undocumented `PLW006` row in the warning table.
- Fixture corpus: invalid twins for pivot, partition_by, landmark
  dtypes, frame literals, pl constructors, variable annotations, plural
  col, struct rename_fields and hstack (the ADR-0003 pairing-convention
  backlog), plus the PLW007 pair.
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
