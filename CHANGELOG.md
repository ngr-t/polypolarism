# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Implicit open-frame sources (ADR-0006, backlog C-12a): a parameter
  annotated with bare `pl.DataFrame` / `pl.LazyFrame` now opts the
  function into checking with an empty *open* frame, and
  `pl.read_parquet` / `pl.scan_csv` / the other unconditional IO readers
  infer open frames with the right laziness. Column lookups on an open
  frame are assumed to succeed (absence is never provable — the
  gradual-typing boundary); everything the body itself determines is
  checked: pinned columns carry exact dtypes through every dtype rule,
  shape-determining calls (`select`, `group_by().agg()`, `unpivot`)
  close the frame and make later column misses provable errors, and
  declared `DataFrame[Schema]` returns check pinned columns exactly.
  Join keys, `rename`, `cast`, `drop_nulls` subsets, selectors, and
  `pl.concat` are open-frame aware (no manufactured proofs; provable
  conflicts still fire).
- Schema inference for three frame methods that previously hard-failed
  correct code with "Could not infer return type" (issue #74):
  `null_count()` (same column names, every dtype `UInt32`, eager and
  lazy), `upsample(...)` (identity columns, but non-key columns become
  `Nullable` — gap rows are null-filled; `time_column` and `group_by`
  keys keep their dtype, and the eager-only method now raises `PLY030`
  on a LazyFrame receiver), and `to_dummies(...)` (value-dependent
  output columns now get the pivot-style `PLW005`
  annotate-the-result suggestion instead of silent inference death).
  All rules probed on polars 1.41.2 and 1.37.0 (identical).
- `join_where(...)` no longer hard-fails: polars documents the method
  as experimental, so instead of encoding its schema the result
  degrades to an *open* frame with a dedicated `PLW007` warning —
  correct code passes via the visible open-frame leniency, and
  `Schema.validate(...)` retracts the warning (issue #74).
- New error `PLY041` (issue #69): a schema field whose
  `Annotated[pl.<Dtype>, ...]` metadata arity provably crashes pandera —
  pandera maps the metadata 1:1 onto the dtype class's `__init__`
  parameters and requires exactly all of them, raising a deferred
  `TypeError` the first time the schema is used (`to_schema` /
  `validate` / `@pa.check_types`). Previously such forms (including the
  README's own `Annotated[pl.Array, pl.Int64(), 3]` example) were
  accepted silently because the parse degraded to `Unknown`. Every
  function referencing a broken schema (parameter / return / variable
  annotation, `Schema.validate(...)` calls) now fails with one PLY041
  per schema; the single-argument `Annotated[X]` form (a typing-level
  `TypeError` at import) is flagged too. A child schema re-declaring
  the field with a full-arity annotation repairs it (probed on
  pandera 0.31.1 / polars 1.41.2).
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

- New diagnostic code `PLY040` for the declared-return-type comparison
  family — `Missing column`, `Extra column`, dtype difference, and
  `Could not infer return type` were the only diagnostics without a
  stable code (issue #70). One shared code for the family; the message
  distinguishes the kind.
- `--format json` diagnostics now carry a structured `"code"` field
  (`"PLY040"`, `"PLW001"`, ...) in addition to the `[PLY###]` message
  prefix, so JSON consumers no longer regex the message (issue #70).
  Untagged diagnostics (file read / parse failures) omit the field;
  the schema change is purely additive.

### Changed

- Return-type mismatch messages are now prefixed with their diagnostic
  code, e.g. `[PLY040] Missing column 'name' of type Utf8` (issue #70) —
  in line with every other diagnostic.
- `FrameType.__init__` now formally accepts
  `Mapping[str, ColumnSpec | DataType]`, matching the runtime
  normalization that was already happening in `__post_init__`.

### Fixed

- Undeclared-column references inside functions with non-strict declared
  schemas are reported honestly (issue #83): new code `PLY042` replaces
  `PLY001` there, naming the schema and stating the truth — a
  `strict=False` schema admits caller extras at runtime, so the
  reference is an undeclared dependency against the function's declared
  interface ("checked island"), not a provable runtime failure. The
  declaration remains enforced (the corpus's missing-column detection is
  unchanged in strength); `PLY001`'s runtime-certainty wording is now
  reserved for exact frames (strict schemas, `select` outputs, open-frame
  negative knowledge), and shape-determining calls re-anchor exactness.
  Row-polymorphic helpers keep their two honest spellings: declare the
  columns you touch, or take a bare `pl.DataFrame` (ADR-0006). This
  resolves the layer disagreement the issue identified, on the
  checked-island side; the open-binding alternative was measured to
  invalidate 159 tests/25 fixtures of declared-interface checking and
  rejected by design choice.

- Call results of functions with non-strict (`strict = False`) return
  schemas bind as OPEN frames (issue #81): pandera's `check_types`
  passes the caller's extra columns through such a return, so the
  row-polymorphic helper pattern no longer turns the caller's own
  columns into PLY001/PLY040 errors — they resolve through the rest
  (dtype Unknown, visible leniency). `strict = True` returns stay
  closed, keeping select-style proofs; applies to direct calls, method
  calls and `df.pipe(helper)`.
- Passing a frame with provable extra columns into a `strict = True`
  parameter is flagged at the call site (issue #82): `check_types`
  validates arguments at runtime and rejects undeclared columns, but the
  detailed call-site error generator only reported missing columns and
  dtype conflicts — the strict-extra direction passed silently. Pinned
  extras on open frames are provable and flag too; unknown open-frame
  extras stay lenient.

- Aliased base-class imports resolve regardless of import order (issue
  #80, boundary of #76): the alias registration lived behind the
  `visited` skip in the import merger, so `from m import Base as B0` +
  `class C(B0)` only worked when that was the module's FIRST import
  statement — any earlier import of `m` dropped the alias, leaving the
  subclass unregistered and its functions skipped with PLW006. Each
  file's pass-1 registry is now cached per invocation so repeat import
  statements still bind their aliases.

- Open frames carry negative knowledge (issue #78, amending ADR-0006):
  `drop("a")` / `rename({"a": "b"})` mark `a` as PROVABLY absent — a
  later reference to it (expressions, `select`, a second `drop`, `cast`,
  `drop_nulls` subsets, join keys) is a guaranteed runtime
  ColumnNotFoundError and now errors, instead of being resurrected by
  the open frame's row variable. Reintroducing the name
  (`with_columns(a=...)`, a rename target, the join's other side) clears
  the mark; rename swaps (`{"a": "b", "c": "a"}`) are handled; a
  declared return column that was provably removed is a real
  MissingColumn instead of an open-frame leniency. Selector-based drops
  keep assumption semantics (unenumerable).
- Open-left join pins no longer manufacture proofs (issue #79, amending
  ADR-0006): joining a closed right frame onto an OPEN left frame pins
  each right column's NAME (it provably exists — as the right column,
  or as the left's colliding column after polars suffixes the right one
  away) but degrades its dtype to `Unknown`; downstream diagnostics like
  `.str` on a right `Int64` pin were not proofs (the code succeeds when
  the left rest carries that name as String). Collisions with PINNED
  left columns keep their deterministic suffix and precise dtypes, and
  the closed-left case keeps every pin precise.
- Stdlib `decimal.Decimal` and `datetime.time` field annotations resolve
  instead of silently dropping the field from the schema (issue #77).
  `decimal.Decimal` / bare `Decimal` (`from decimal import Decimal`)
  register pandera's engine default `Decimal(28, 0)` — the same value as a
  bare `pl.Decimal` (issue #75) — and `datetime.time` / `dt.time` / bare
  `time` register `Time`, both probed against pandera 0.31.1 in both
  directions. Nested positions (`pl.List(decimal.Decimal)`) stay runtime
  wildcards and parse to `Unknown`, mirroring nested `pl.Decimal`. The
  bare `time` name is read as `from datetime import time` (annotating with
  the stdlib `time` *module* is never meaningful — pandera rejects it).
- An UNRECOGNIZED field annotation no longer silently drops the field
  (issue #77, ADR-0007): the column registers with `Unknown` dtype — so
  strict schemas no longer reject correct code with phantom
  "Extra column" errors and open schemas no longer pass wrong dtypes
  against a vanished column — and every function referencing the schema
  gets a new `PLW011` warning naming the field and annotation. A warning
  rather than a `PLY041` error because the name may be a runtime alias of
  a real dtype (`MyAlias = pl.Int64` resolves fine in pandera); if it
  genuinely does not resolve, pandera raises TypeError at first schema
  use, which the warning text points at.
- Cross-file schema inheritance resolves (issue #76): a class whose base
  is an IMPORTED schema (`from base import WithId` + `class
  Users(WithId)`) was not recognized as a schema at all — functions
  annotated with it carried no declared type and passed vacuously. A
  second fixpoint pass re-scans every parsed tree against the merged
  registry, so such subclasses parse with the parent's fields merged
  exactly like the same-file path — including in-file chains rooted at
  an imported base, subclasses living inside imported files, aliased
  bases (`from base import WithId as M`), and module-qualified bases
  (`import base` + `class Users(base.WithId)`, via the issue #68 dotted
  keys). `from X import Y as Z` aliases now also resolve in
  `DataFrame[Z]` annotations; a local class shadows a same-named
  imported schema (runtime name-binding precedence). The previously
  vacuous transitive-import test now asserts the inherited columns are
  real and that violations against them fail.
- README documented the runtime-broken `Annotated[pl.Array, pl.Int64(), 3]`
  form; the example now shows the form pandera actually accepts,
  `Annotated[pl.Array, pl.Int64(), 3, None]` (issue #69 — pandera demands
  all of `inner, shape, width`).
- A bare `pl.Decimal` field annotation registers pandera's engine default
  `Decimal(28, 0)` instead of polars' materialized `Decimal(38, 0)`
  (issue #75; probed: `to_schema()` reports 28 and `validate` rejects a
  (38, 0) column) — both the false positive on a correct (28, 0) return
  and the false negative on a (38, 0) one are gone. Call forms
  (`pl.Decimal()`, omitted/`None` args) carry a polars instance and keep
  38; a NESTED bare class (`pl.List(pl.Decimal)`, struct fields) is a
  probed runtime wildcard and parses to `Unknown`; unreadable call
  arguments degrade to `Unknown` instead of claiming the bare default.
- `pct_change()` is no longer typed as dtype-preserving (issue #71): it
  divides, so an int receiver (any width — Boolean, the temporals,
  Decimal and Null too) infers `Float64?` and float receivers keep their
  width (`Float32 -> Float32?`); both the false positive on a correct
  `Float64?` declaration and the false negative on a wrong `Int64?` one
  are gone. Probed-invalid receivers (Binary / Categorical / Enum /
  List / Array / Struct — the Struct cell is a process-aborting rust
  crash) flag `PLY016`.
- `not_()` / `~` is no longer Boolean-returning unconditionally (issue
  #72): per the documented `Expr.not_` contract it negates Booleans but
  operates **bitwise** on integers, preserving the dtype (`~Int64 ->
  Int64`). Every other receiver (floats included) is a probed runtime
  `InvalidOperationError` -> `PLY016`. `~` on a non-Boolean column used
  in a `filter(...)` predicate now correctly flags `PLY008` as a bonus.
- `dt.epoch(...)` return dtype follows its `time_unit` argument (issue
  #73): `"d"` infers `Int32` (probed; days since epoch), the sub-second
  units and the no-arg default stay `Int64`, and a non-literal or
  invalid argument degrades to `Unknown` instead of claiming `Int64`.
- `Annotated[pl.Decimal, 12, 4]` schema annotations register the declared
  precision/scale instead of the bare `Decimal(38, 0)` default, so
  exactly-matching code is no longer rejected with `PLY033` and real
  mismatches report the true declared dtype (issue #65; a `None` literal
  takes the polars default, wrong arity — a pandera runtime TypeError —
  degrades to `Unknown`).
- `Datetime`/`Duration` now carry their `time_unit` (issue #66): a
  declared `Datetime[ns]` over an inferred `Datetime[us]` (or
  `Duration[ms]` over `Duration[us]`) is an error, matching pandera's
  runtime SchemaError — both were silent false negatives while only the
  tz was modeled. Units flow through inference per probed polars 1.41.2
  semantics: mixed-unit arithmetic and `when/then` resolve to the
  coarser unit, `diff` keeps the receiver's unit (Date is us-based, Time
  ns-based), `str.to_datetime` honors `time_unit=` and the `%.3f`/`%.9f`
  format directives, `pl.datetime_range` honors `time_unit=` and a
  ns-bearing interval, and `dt.replace_time_zone`/`convert_time_zone`
  preserve the receiver's unit. Under `Config.coerce` a unit *coarsening*
  (us -> ms, a value-independent division) is tolerated; *refining*
  (us -> ns) stays an error — the cast overflows for extreme values.
  `Annotated[pl.Datetime, "ns", None]`, `Annotated[pl.Duration, "ms"]`
  and the `pl.Duration("ms")` call form parse their unit.
- `Enum` category tuples are compared (issue #67): an inferred
  `Enum['a', 'c']` no longer satisfies a declared `Enum['a', 'b']` —
  polars treats different category *sequences* (sets and reorderings
  alike) as distinct dtypes and pandera rejects them at runtime.
  `pl.Enum([...])` call forms and `Annotated[pl.Enum, [...]]` carry the
  ordered categories; a bare `pl.Enum` or a non-literal category list
  models as "some Enum, categories unknown" and acts as a wildcard with
  a `via:` leniency note (mirroring unknown Array widths), while still
  catching cross-dtype mismatches.
- Numeric aggregations accept every numeric width (probed on polars
  1.41.2): Int8/Int16/UInt8/UInt16/Int128/UInt128/Float16 receivers were
  falsely rejected. Small-int `sum`/`product` infer Int64 (unsigned
  included — polars lands on signed), Float16 keeps its width through
  select-context reductions, and the four probed rust-panic cells
  (grouped `mean`/`median`/`quantile` on Float16, grouped `product` on
  UInt128 — including `.over()` windows) are now `PLY011` errors instead
  of accepted crashes.
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
- Module-qualified schema annotations resolve (issue #68):
  `DataFrame[mod.Schema]` after a project-local `import mod` (or
  `import pkg.mod [as m]`) now type-checks like its `from mod import
  Schema` spelling — previously the return side passed vacuously and a
  param-side annotation silently de-registered the function. The
  registry stays flat: plain imports mount the imported module's schemas
  under their dotted spelling as written at the annotation site. A
  qualified name that still doesn't resolve (third-party module, nested
  class like `DataFrame[Outer.Inner]`) now warns `PLW006` with the full
  dotted name instead of being silently ignored.
