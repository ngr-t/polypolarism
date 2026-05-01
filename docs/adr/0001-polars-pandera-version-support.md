# ADR-0001: Polars and Pandera Version Support Strategy

- **Status**: Proposed
- **Date**: 2026-05-01
- **Deciders**: @negotetsu

## Context

`polypolarism` is a static type checker that infers Polars `DataFrame` /
`LazyFrame` schemas from AST-only analysis. It never imports `polars` or
`pandera` at runtime — every coupling to those libraries is an assumption
encoded in string tables, name sets, and dispatch branches across
`src/polypolarism/`.

Both libraries change. The shape of that change is asymmetric:

**Polars** shipped a substantial *cluster* of breaking changes between 0.19
and 1.0 — all in a tight window, mostly mechanical renames but several
genuine semantic shifts:

- Renames: `groupby` → `group_by`, `apply` → (`map_elements` /
  `map_rows` / `map_groups`, *receiver-dependent*),
  `cumsum/cumprod/cummax/cummin/cumcount` → underscored variants,
  `Utf8` → `String`, `outer` join → `full` join.
- Semantic shifts: outer-join no longer coalesces keys (1.0), `count()`
  ignores nulls (0.20), `replace()` redesigned with `replace_strict()` split
  (1.0), `Array(inner, width)` and `Decimal(precision, scale)` parameter
  reordering (0.20), horizontal aggregation extracted to `sum_horizontal`
  etc. (0.19).

Within Polars 1.x (1.0 → 1.40, July 2024 → April 2026), most minors are
additive, but the cumulative drift is non-trivial. A static checker has
to handle:

- **New dtypes**: `Int128` (1.18), `Enum` stabilized (1.25), `UInt128`
  (1.34), `Decimal` stabilized (1.35), `Float16` (1.36) — all things AST
  callers can write as `pl.SomeDtype`.
- **Selector-as-DSL** (1.32): `pl.selectors.*` now returns `Selector`
  objects, not raw `Expr`. Downstream destructuring may diverge.
- **`Categorical` / `(Frozen)Categories` rework** (1.32): `pl.Categorical`
  surface and dtype-equality story changed.
- **`hist` bin-closure shift** (1.27) and **left-join row-order
  de-guarantee** (1.16): semantics under fixed signatures.

Within any single supported-pair window (e.g. 1.39 ↔ 1.40), the diff is
small. Across 1.0 → 1.40 the diff is meaningful, and the analyzer's
current dtype tables in particular lag the language by several minors.

**Pandera** has one mechanical rename relevant to AST analysis:
`SchemaModel` → `DataFrameModel` (deprecated 0.20). That class-name set is
already encoded in `src/polypolarism/pandera_schema.py:19-21`:

```python
_BASE_NAMES = frozenset({"DataFrameModel", "SchemaModel"})
```

Carrying both names indefinitely costs one entry in a frozenset.

The rest of Pandera's evolution (added `pandera.polars` module, pyarrow dtype
support, pinning `polars >= 1.0`) does not affect AST-level introspection.

A version-by-version inventory of the relevant churn lives at
[`docs/research/polars-pandera-churn.md`](../research/polars-pandera-churn.md).
A file:line map of every coupling site in the current code lives at
[`docs/research/coupling-inventory.md`](../research/coupling-inventory.md).

## Decision

1. **Support Polars 1.x only** — concretely, the latest two 1.x minor
   releases. Pre-1.0 surface is explicitly out of scope.
2. **Continue supporting both `SchemaModel` and `DataFrameModel`** as
   Pandera base classes, silently. Do **not** emit a deprecation warning.
3. **Version is best-effort detected for warning purposes only**, with
   explicit override:
   - Detection sources, in priority order: `--polars-version` /
     `--pandera-version` CLI flag, `[tool.polypolarism]` in the target's
     `pyproject.toml`, the target's `uv.lock` (exact version), the floor
     in `[project.dependencies]` / `[dependency-groups.*]`.
   - **The floor reflects "fully supported only"**: for polars, the lower
     bound of the latest-two-minors window (currently 1.39, tracked in
     `version_check.POLARS_LATEST_KNOWN`). For pandera, 0.19 — pandera's
     AST-relevant surface (class-name matching) is stable across minors,
     so a "latest two minors" window would not differentiate anything we
     actually test against. When the detected version is below the floor,
     `polypolarism` emits a `PLW010` warning to stderr.
     `--no-version-check` suppresses detection entirely.
   - The detected version **does not feed analyzer dispatch today** — it
     only gates the warning. When `PolarsProfile` later grows fields, the
     same detection result will choose the profile.
4. **Centralize all Polars/Pandera surface knowledge** into a new
   `src/polypolarism/compat/` module so that future churn is absorbed in
   one place rather than across `analyzer.py`, `expr_infer.py`,
   `ops/*.py`, and `pandera_*.py`.

## Rationale

### Why drop pre-1.0 Polars

The 0.19 → 1.0 rename cluster is dense and some renames are
receiver-contextual (`apply` → `map_elements` / `map_rows` /
`map_groups`). Supporting them would require either AST receiver-aware
rewriting or a parallel dispatch path per old name. The semantic shifts
(join coalesce, `count()` null handling) cannot be papered over with
aliases at all — they need version-aware behavior. The combined alias
matrix and per-version semantic table is non-trivial to build, test, and
keep correct.

Polars 1.0 shipped in 2024; users have had ample time to migrate. The cost
of supporting a pre-1.0 long tail outweighs the benefit.

### Why keep `SchemaModel` silently

It is a single string in a set. No semantic difference, no signature
divergence, no introspection divergence. Emitting a deprecation warning
would require adding diagnostic plumbing that does not yet exist (warnings
are distinct from type errors), which adds more code than it removes. The
Pandera case is fundamentally simpler than the Polars case — treat it as
such.

### Why detection-with-override (warnings only)

Project pins lie: `polars>=1.0.0` could be a project running 1.0 or one
running 1.40. For *behavior dispatch* that ambiguity is fatal — we'd pick
the wrong profile silently. For *warnings* it's fine: the question is
only "is the project somewhere below the supported floor?", which the
floor in a `>=` spec answers correctly.

So detection feeds the warning channel only. Lockfiles (`uv.lock`) give
exact versions when present; `pyproject.toml` floors are good enough
otherwise. Users who want to assert a specific version (because their
lockfile is elsewhere, or they're auditing CI for a particular target)
get `--polars-version` and `[tool.polypolarism]` as overrides.

### Why centralize `compat/`

Today the dtype name map is literally duplicated between
`src/polypolarism/analyzer.py:210` and
`src/polypolarism/pandera_dtype.py:75`. A future `String`/`Utf8`-style
change has to be applied twice, and the sites that *don't* duplicate
(the aggregation table in `ops/groupby.py`, the join `how` set in
`ops/join.py`, the namespace tables in `analyzer.py`) are spread thinly
enough that a version bump can easily miss one.

## Alternatives considered

### A. Wide window (Polars 0.19 + 0.20 + 1.x)

Rejected. Adds receiver-contextual `apply` rewriting and several semantic
profile fields. Cost grows roughly linearly with the number of supported
minors; benefit accrues only to users who haven't migrated in 1+ years.

### B. Pandera `SchemaModel` deprecation warning

Rejected. Requires building warning infrastructure that doesn't exist yet
(distinct from the existing error-emission path). The whole point of
keeping `SchemaModel` working is that it's free; adding a warning makes
it no longer free.

### C. Auto-sniff Polars version and feed it into analyzer dispatch

Rejected. Pins are floors, not exact versions; using a sniffed value to
*pick a behavior profile* (e.g. "treat join coalesce semantics as 0.20")
would silently mislabel many projects. The compromise actually adopted —
sniff only for the support-floor warning, never for dispatch — keeps the
benefit (catch users on unsupported polars) without the silent-mislabel
risk.

### D. Versioned dispatcher with per-minor profile fields shipped today

Rejected for now. Within any one two-minor window the divergence is
small. The cumulative 1.x drift (selector-as-DSL, `Categorical` rework,
`hist` bin-closure) is real, but it has *not* moved across our
supported-pair window in a way that requires per-minor branching today —
both supported minors are post-1.32. The `PolarsProfile` scaffold ships
but with no fields until a real fixture failure motivates one. New dtypes
go into the (single, version-agnostic) dtype table — they're additive,
not a profile concern.

## Consequences

### Positive

- Eliminates the bulk of the alias maintenance surface (groupby family,
  cumsum family, `apply` receiver-dependent rewrites, `Utf8`,
  `outer`-join).
- Pandera support remains zero-friction for users on either generation
  (`SchemaModel` or `DataFrameModel`).
- Single canonical `compat/` module makes future churn (whenever it
  arrives) a localized edit.
- Existing duplication of `_PL_DTYPE_NAME_MAP` between `analyzer.py` and
  `pandera_dtype.py` goes away.

### Negative

- Users on **anything below the latest two 1.x minors** see a `PLW010`
  warning. Pre-1.0 is the harder case (the analyzer doesn't recognize
  the legacy spellings and will silently misanalyze); 1.0–1.38 are
  best-effort (the analyzer should work but isn't actively tested for
  per-minor quirks). Mitigation: `--polars-version` opts a project back
  in; the warning text says results are best-effort, not wrong.
- A future Polars 2.0 will trigger a similar ADR — this defers, not
  eliminates, the multi-major-version question.

### Neutral / maintenance

- `METHOD_ALIASES` ships empty as scaffolding. A future intra-1.x rename
  (or eventual 2.0) is a one-line addition.
- `PolarsProfile` ships as a `name`-only dataclass. Fields get added when
  fixtures actually diverge between supported minors.
- **`POLARS_LATEST_KNOWN` needs a bump per polars minor release.** The
  floor follows automatically (latest minor − 1). One-line change in
  `version_check.py`. Good candidate for a scheduled cleanup task.

## Implementation outline

| Step | Action | Files |
|---|---|---|
| 1 | **Done.** `compat/{__init__.py, polars_api.py, pandera_api.py}`. | new |
| 2 | **Done.** All dispatch tables (DTYPE_NAME_MAP, AGG name lookups, `IDENTITY_FRAME_METHODS` / `LAZY_ONLY_METHODS` / `EAGER_ONLY_METHODS`, the five sub-namespace tables) moved into `compat/polars_api.py`. analyzer.py keeps thin local aliases for legibility. | `analyzer.py` → `compat/polars_api.py` |
| 3 | **Done.** `pandera_dtype.py:_PL_DTYPE_MAP` re-exports `DTYPE_NAME_MAP` — single source of truth. | `pandera_dtype.py` |
| 4 | **Done.** `JOIN_HOW_VALUES` / `JOIN_HOW_INFERRED` / `join_left_nullable` / `join_right_nullable` in compat; `ops/join.py:infer_join` consumes them. | `ops/join.py` |
| 5 | **Done (adapted).** Polars-surface part of agg table (`agg_function_for(name)`, `AGG_SHORTHAND_NAMES`) in compat. Inference logic (`_AGG_INFER_MAP`, `_infer_*` per-function) stays in `ops/groupby.py` because it's behavior, not surface. | `ops/groupby.py`, `analyzer.py` |
| 6 | **Done.** `METHOD_ALIASES = {}` scaffold + `canonicalize_method()` shim called at frame-method dispatch entry. Empty today; one entry away from supporting any future intra-1.x rename. | `analyzer.py`, `compat/polars_api.py` |
| 7 | **Done.** `SCHEMA_BASE_NAMES`, `FRAME_ANNOTATION_HEADS`, `FIELD_CALLABLE_NAME` in `compat/pandera_api.py`. | `pandera_schema.py`, `pandera_annotation.py` |
| 8 | **Done.** `Int128`, `UInt128`, `Float16`, `Enum`, `Decimal(p, s)` added to `DTYPE_NAME_MAP`. Fixtures per landmark version. New `_parse_decimal_call` extracts precision/scale (the only parametrized dtype where args matter for type identity). | `compat/polars_api.py`, `pandera_dtype.py`, `types.py`, `tests/fixtures/` |
| 9 | **Done.** Audited selector dispatch against five 1.32-affected patterns — all currently infer correctly. Pinned in `tests/fixtures/valid/selector_dsl_1_32.py`. | `analyzer.py`, `tests/fixtures/` |
| 10 | **Done.** `PolarsProfile` scaffold + default `POLARS_1_X` in compat. Name-only; fields get added when a real divergence appears. | `compat/polars_api.py` |
| 11 | **Done** in `src/polypolarism/version_check.py`: detection from CLI flag, `[tool.polypolarism]`, `uv.lock`, dependencies; PLW010 warning on below-floor; `--polars-version` / `--pandera-version` / `--no-version-check` CLI flags. | `version_check.py`, `cli.py`, `diagnostics.py` |
| 12 | **Done.** New "Supported versions" section, `--polars-version` / `--pandera-version` / `--no-version-check` CLI examples, `[tool.polypolarism]` config block, dtype list refreshed for `Int128` / `UInt128` / `Float16` / `Enum` / `Decimal(p, s)`, `cs.integer()` / `cs.float()` lists updated, `PLW010` added to the warning code table. | `README.md` |

All 12 steps complete. Remaining future work is documented in
`docs/research/coupling-inventory.md` under "Known gaps" — none of those
gaps block the supported window.

## Verification

- After steps 1–7: `uv run pytest` green, `uv run polypolarism
  tests/fixtures/valid/` and `tests/fixtures/invalid/` produce identical
  error counts to the pre-refactor baseline.
- After step 8: fixtures using `Int128`, `UInt128`, `Float16`, `Enum`,
  `Decimal` as Pandera schema dtypes type-check correctly.
- After step 9: a fixture using `cs.numeric()` and downstream selector
  algebra still produces the expected projected schema under the 1.32+
  `Selector` semantics.
- Step 11 (shipped): a project pinning `polars>=0.20.0` triggers
  `[PLW010] detected polars 0.20.0 ...` on stderr; `--polars-version
  <floor-or-above>` silences it; `--no-version-check` skips detection
  entirely. Polars 1.0–1.38 also warn under the "fully supported only"
  policy. `tests/test_version_check.py` covers 35 paths including the
  boundary cases.
- A `SchemaModel`-using fixture continues to type-check correctly with
  no new output (silent acceptance preserved).
