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
3. **Version selection** is explicit, not auto-detected:
   - `polypolarism --polars-version <ver>` CLI flag.
   - `[tool.polypolarism] polars_version = "..."` in `pyproject.toml`.
   - No automatic dependency-pin sniffing.
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

### Why explicit version selection (no sniffing)

Project pins lie: `polars>=1.0.0` could mean 1.0 or 1.32. Sniffing creates
flakiness in return for marginal convenience. CLI flag plus
`pyproject.toml` config is predictable and discoverable.

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

### C. Auto-sniff Polars version from `[project.dependencies]`

Rejected. Pins are floors, not exact versions. Reading `uv.lock` /
`poetry.lock` would be more precise but requires per-tool plumbing. Net:
fragile for marginal UX gain.

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

- Users on pre-1.0 Polars cannot use `polypolarism`. This is a hard cut —
  the tool will likely produce confusingly wrong results rather than a
  clean error, since the AST analyzer has no runtime version check.
  Mitigation: document the support floor prominently in `README.md` and
  CLI `--help`.
- A future Polars 2.0 will trigger a similar ADR — this defers, not
  eliminates, the multi-major-version question.

### Neutral / maintenance

- `METHOD_ALIASES` ships empty as scaffolding. A future intra-1.x rename
  (or eventual 2.0) is a one-line addition.
- `PolarsProfile` ships as a `name`-only dataclass. Fields get added when
  fixtures actually diverge between supported minors.

## Implementation outline

| Step | Action | Files |
|---|---|---|
| 1 | Create `src/polypolarism/compat/{__init__.py, polars_api.py, pandera_api.py}` | new |
| 2 | Move dtype name map, agg shorthand, namespace tables, method classifications into `compat/polars_api.py` | `analyzer.py` → `compat/polars_api.py` |
| 3 | Replace `pandera_dtype.py:_PL_DTYPE_MAP` with re-export from `compat/polars_api.py` | `pandera_dtype.py` |
| 4 | Move `JoinHow` + nullability rules into `compat/polars_api.py` | `ops/join.py` |
| 5 | Move `_AGG_INFER_MAP` into `compat/polars_api.py` | `ops/groupby.py` |
| 6 | Add empty `METHOD_ALIASES = {}` scaffold + canonicalize-at-entry shim in dispatch | `analyzer.py`, `compat/polars_api.py` |
| 7 | Move `_BASE_NAMES`, `_HEAD_NAMES`, `Field` detection into `compat/pandera_api.py` | `pandera_schema.py`, `pandera_annotation.py` |
| 8 | **Catch up the dtype table to current 1.x**: add `Int128`, `UInt128`, `Float16`, `Enum`, `Decimal` to the unified dtype map; add fixtures exercising each | `compat/polars_api.py`, `tests/fixtures/` |
| 9 | **Audit selector dispatch against 1.32 `Selector` DSL change**: confirm `analyzer.py:424-496` still produces correct types after `pl.selectors.*` returns `Selector` objects; adjust as needed | `analyzer.py` (post-refactor: `compat/polars_api.py`) |
| 10 | Add `PolarsProfile` scaffold (name-only) + default `POLARS_1_X` | `compat/polars_api.py` |
| 11 | `--polars-version` CLI flag + `[tool.polypolarism]` config reader | `cli.py` |
| 12 | Document support window + new config in `README.md` | `README.md` |

Steps 1–7 are pure refactors — existing 265 tests must stay green and
serve as the regression net. Steps 8–9 close the cumulative-drift gap
identified in the churn survey. Steps 10–12 add the new capability and
documentation.

## Verification

- After steps 1–7: `uv run pytest` green, `uv run polypolarism
  tests/fixtures/valid/` and `tests/fixtures/invalid/` produce identical
  error counts to the pre-refactor baseline.
- After step 8: fixtures using `Int128`, `UInt128`, `Float16`, `Enum`,
  `Decimal` as Pandera schema dtypes type-check correctly.
- After step 9: a fixture using `cs.numeric()` and downstream selector
  algebra still produces the expected projected schema under the 1.32+
  `Selector` semantics.
- After step 11: `polypolarism --polars-version 1.x
  tests/fixtures/valid/` runs; an unknown version produces a clear error
  from the CLI.
- A `SchemaModel`-using fixture continues to type-check correctly with
  no new output (silent acceptance preserved).
