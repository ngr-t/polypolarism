# Diagnostics

## Diagnostic codes

Errors are tagged with a stable `[pple-<slug>]` prefix for IDE/CI consumers.
The `ppl` stem plus the `e` (error) namespace keeps the codes from colliding
with mypy / ruff inside `# type: ignore[...]`:

| Code | Meaning |
|---|---|
| `pple-column-not-found` | a referenced column does not exist on the frame — covers `pl.col("X")` / `cs.*` expression lookups, `drop`, `rename` source, `cast`, `drop_nulls` subset, `sort`, and `unique` subset (the message text distinguishes the kind) |
| `pple-column-name-collision` | `with_row_index`: name collides with existing column |
| `pple-non-boolean-predicate` | `filter` predicate / `when` condition dtype is not Boolean |
| `pple-incompatible-operands` | binary operation between incompatible dtypes (arithmetic, comparison, `is_in`) |
| `pple-invalid-cast` | `cast` between structurally incompatible dtypes |
| `pple-duplicate-column` | duplicate output column name in `select` / `with_columns` / `rename` |
| `pple-non-numeric-operand` | numeric-only operation applied to a non-numeric column |
| `pple-list-literal-misuse` | list literal mixed with other positional expression arguments |
| `pple-join-key` | join key error (missing / dtype mismatch) |
| `pple-groupby` | `group_by` key missing or aggregation type error |
| `pple-wrong-namespace-dtype` | namespace accessor (`.str` / `.dt` / `.list` / `.arr` / `.struct` / `.bin` / `.cat`) used on a wrong dtype |
| `pple-concat-mismatch` | `concat` schema mismatch (vertical / horizontal overlap / diagonal unify) |
| `pple-explode` | `explode`: column not found or not `List[T]` / `Array[T]` |
| `pple-unpivot` | `unpivot`: column not found or `on`-columns dtype mismatch |
| `pple-eager-only-method` | eager-only method called on a `LazyFrame` (e.g. `lf.write_csv(...)`) — suggests `.collect()` |
| `pple-lazy-only-method` | lazy-only method called on a `DataFrame` (e.g. `df.sink_csv(...)`, `df.collect()`) — suggests `.lazy()` or removing the call |
| `pple-eager-lazy-mismatch` | function-call argument or return type mixes up `DataFrame[S]` and `LazyFrame[S]` — suggests the appropriate `.collect()` / `.lazy()` |
| `pple-annotation-conflict` | a variable annotation re-interprets the inferred frame as an unrelated type (neither subtype direction holds, ADR-0005) |
| `pple-return-type` | declared return type does not match the inferred return type — one shared code for the whole family: missing column, extra column, dtype difference, or the return type could not be inferred |
| `pple-broken-schema-annotation` | a schema field's `Annotated[pl.<Dtype>, ...]` metadata arity provably crashes pandera — the TypeError fires the first time the schema is used (`validate` / `@pa.check_types`), so every function referencing the schema is dead on arrival |
| `pple-undeclared-column` | a column referenced inside a function is not declared in its (non-strict) parameter/validated schema — an undeclared dependency on caller extras ("checked island"), not a provable runtime failure; declare the column, or take a bare `pl.DataFrame` for row-polymorphic helpers |
| `pple-rowpoly-not-preserved` | a `@rowpoly` helper body provably drops its row variable — a return point produces a closed frame that loses the caller's extra columns (e.g. a `select` of a fixed column set), breaking the threading promise. Static-only (the property is relative to the caller). Use `with_columns` / `select(pl.all())`, or remove the row variable from `@rowpoly`. See [Row polymorphism](row-polymorphism.md) |

> **`pple-undeclared-column` "declare the column" quick fix (`--format json`)
> is location-only unless the column's use constrains its type.** The `fix`
> object always carries where to insert (`schema`, `column`, `schema_file`,
> `schema_insert_line`); it adds `suggested_dtype` (a ready-to-insert
> annotation, e.g. `"pl.Float64"`) only when the reference statically pins the
> dtype — e.g. `pl.col("amount").cast(pl.Float64)`. A bare
> `pl.col("amount")` on an open/non-strict frame resolves to `Unknown`, so
> `suggested_dtype` is omitted (sound — never a guessed dtype); the editor then
> offers only the "relax the param to bare `pl.DataFrame`" fix (issue #114).

## Apply-style helpers and warning codes

Some patterns are **not statically decidable** without help from the
user. Polypolarism detects them, falls back to a best-effort inference,
and emits a `[pplw-<slug>]` **warning** that names a concrete source change
that would let the analyser check the code precisely. Warnings are
non-fatal: the CLI exits `0` even when warnings are emitted.

| Form | Status | Note |
|---|---|---|
| `df.pipe(typed_helper)` where `typed_helper` is a `DataFrame[A] → DataFrame[B]` defined in the same module | ✅ inferred | uses the helper's declared return type |
| `df.pipe(untyped_helper)` defined in the same module | ✅ inferred | body is re-analysed with the propagated argument types |
| `df.pipe(external_helper)` (imported from another module) | ⚠️ `pplw-unresolved-pipe` | suggests defining the helper locally with a `DataFrame[Schema]` annotation |
| `df.pipe(lambda d: ...)` | ⚠️ `pplw-untyped-callable` | suggests promoting the lambda to a top-level typed function |
| `pl.col("x").map_elements(fn, return_dtype=pl.Float64)` | ✅ inferred | the declared `return_dtype` becomes the result dtype |
| `pl.col("x").map_elements(fn)` (no `return_dtype=`) | ⚠️ `pplw-missing-return-dtype` | falls back to receiver dtype; suggests adding `return_dtype=pl.<DType>` |
| `pl.col("x").map_batches(fn, return_dtype=...)` | ✅ inferred | same rule as `map_elements` |
| `external_helper(df)` (top-level call into an imported helper) | ⚠️ `pplw-unknown-function` | suggests defining the helper locally or inlining the transformation |

Warning codes (the `w` namespace):

| Code | Meaning |
|---|---|
| `pplw-missing-return-dtype` | `map_elements` / `map_batches` without `return_dtype=` |
| `pplw-unresolved-pipe` | `pipe` with a callable that isn't in the analysed module |
| `pplw-unknown-function` | function call to a name that isn't defined in the analysed module |
| `pplw-untyped-callable` | lambda / inline callable used where its return dtype is unknowable |
| `pplw-data-dependent-schema` | `pivot()` output schema is data-dependent; bind to a `DataFrame[Schema]` variable |
| `pplw-unknown-schema` | `DataFrame[X]` / `LazyFrame[X]` annotation references a schema the analyzer cannot resolve |
| `pplw-unmodeled-method` | method not modeled by polypolarism. Expression/namespace methods: the result dtype degrades to `Unknown`; pin it with `.cast(...)` or a schema validation (a `.cast(...)` directly after the call retracts the warning). Frame methods probed to return a DataFrame/LazyFrame: the frame untracks; wrap the call in `Schema.validate(...)` (which retracts the warning). Terminal frame methods (`to_dicts`, `write_*`, `height`, ...) legitimately return non-frames and stay silent |
| `pplw-unbacked-narrowing` | a variable annotation *narrows* the inferred schema without runtime backing (ADR-0005) — e.g. non-null over a post-join nullable; assert it with `Schema.validate(...)` or widen the annotation |
| `pplw-unsupported-version` | detected polars / pandera version is below the supported floor (see [Supported versions](versions.md)) |
| `pplw-unrecognized-annotation` | a schema field annotation polypolarism cannot translate — the column registers as `Unknown` dtype instead of silently vanishing (pandera raises TypeError at first use if the annotation genuinely doesn't resolve) |
| `pplw-all-null-aggregation` | a grouped `std`/`var`/`sum` on a Date/Datetime/Time column is probed to yield an unconditionally all-null column (the same reduction raises in a plain select) — accepted, but probably not what you meant |
| `pplw-ignored-cast` | `typing.cast(DataFrame[Schema], x)` is **not** honored as a schema assertion — polypolarism infers `x`'s real schema and checks that (a lying cast fails at runtime under `@pa.check_types` too). The note flags an inert cast over a known schema (suggesting `# type: ignore[pple-return-type]` to suppress a mismatch) or an unverified assumption over an open/unknown source |
| `pplw-rowpoly-not-threaded` | an imported `@rowpoly` helper does not provably preserve its row variable, so its extras are not threaded at the call site (issue #112) — see [Row polymorphism](row-polymorphism.md) |

JSON output (`--format json`) emits warnings as `severity: "warning"`
diagnostics so editors and CI can route them separately from errors.
Every tagged diagnostic also exposes its slug code as a structured `"code"`
field alongside the `[pple-…]` / `[pplw-…]`-prefixed message, so consumers
(pre-commit hooks, CI annotators, editors) never need to regex the message
text. Untagged diagnostics (file read / parse failures) omit the field.

The JSON payload also carries a `functions` array — one entry per
analyzed function with its source span and schema summaries (parameter
frames, declared/inferred return frames as rendered dtype maps, plus
`open` / `strict` / `lazy` markers). Editor integrations use it to show
hovers without re-running the analysis. A `@rowpoly` helper additionally
carries its bound row variable(s) — `"row_var": "R"` for `@rowpoly("R")`
or `"param_row_vars": {"a": "R1", "b": "R2"}` for the keyword form (added
only when present, so the payload stays backward-compatible). See
[Row polymorphism](row-polymorphism.md).

## Schema diff block

When a single function has at least two column-level mismatches
(`MissingColumn`, `ExtraColumn`, or `TypeDifference` in any combination)
the text formatter appends an aligned diff block under the per-line
errors so the user can scan the whole shape difference at once:

```
  f (line 19): FAIL
    - [pple-return-type] Column 'id' has type Int64, but declared type is Int32
    - [pple-return-type] Missing column 'amount' of type Float64
    - [pple-return-type] Column 'name' has type Utf8, but declared type is Float64
    - [pple-return-type] Missing column 'extra' of type Int64
    schema diff:
      column  declared  inferred   status
      ──────  ────────  ─────────  ────────
      id      Int32     Int64      mismatch
      amount  Float64   (missing)  missing
      name    Float64   Utf8       mismatch
      extra   Int64     (missing)  missing
```

Single-mismatch failures keep the original one-line output. JSON output
is unchanged — each mismatch remains an individual diagnostic.

## Legacy code mapping (one-time scheme change)

Pre-release, the diagnostic codes moved from the old numeric scheme
(error prefix `PLY` + a three-digit number, warning prefix `PLW` + a
three-digit number) to the semantic slugs above. This was a clean break
with no back-compat aliases — the old numeric codes no longer resolve.

The old→new correspondence is mechanical and ordered to match this
document's two tables: the old numbered error codes map, in order, to the
`pple-*` slugs in the error table, and the old numbered warning codes map,
in order, to the `pplw-*` slugs in the warning table. The one structural
change beyond the rename: the seven former column-not-found error codes
(the `pl.col` lookup, `drop`, `rename`-source, `cast`, `drop_nulls`-subset,
`sort`, and `unique`-subset misses) were **merged** into the single
`pple-column-not-found`. The exact line-by-line substitution lives in the
migration commit (see `CHANGELOG.md`); this is the durable reference.
