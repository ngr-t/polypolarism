# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Annotated assignments are now checked against the inferred RHS
  (ADR-0005 two-direction rule): `x: DataFrame[A] = expr` where the
  expression provably infers an *unrelated* schema is a new error
  `PLY033`; a pure *narrowing* assertion (e.g. non-null over a
  post-left-join nullable) stays allowed and warns `PLW008` with
  `Schema.validate(...)` as the runtime-backed upgrade. Pivot-style
  annotations over `Unknown` inference remain silent; the annotation
  still wins for downstream typing.
- New warning `PLW007`: a method polypolarism does not model on a
  precisely-known receiver now warns that the dtype degrades to
  `Unknown` (one warning per chain; a `.cast(...)` directly after the
  call retracts it). Frame-level methods probed to return a
  DataFrame/LazyFrame warn too — the variable silently untracks
  otherwise — while terminal methods (`to_dicts`, `write_*`, ...) stay
  silent; wrapping the call in `Schema.validate(...)` retracts the
  frame-level warning.
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

### Fixed

- `Array` widths are now tracked (closing the issue #53 "width ignored"
  gap): `pl.Array(pl.Int64, 3)` declared against an inferred width 5 is
  an error (probed: pandera rejects the mismatch and coerce cannot
  repair it), width-changing casts flag `PLY013` ("cannot cast Array to
  a different width" raises in both strict modes), and the `.arr`
  namespace preserves the receiver's width. A width the analyzer cannot
  resolve (non-literal or multi-dimensional `shape=`) acts as a wildcard
  with a `via:` leniency note.
- Float32 width preservation (probed on polars 1.41.2): the float-family
  reductions (`mean`/`std`/`var`/`median`/`quantile`) keep Float32 on a
  Float32 receiver in rolling, select and `group_by().agg()` contexts,
  and `pl.mean_horizontal` returns Float32 when every operand is
  Float32. Previously all of these claimed Float64, falsely rejecting
  correct Float32 declarations.
