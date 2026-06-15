# Diagnostics

## Diagnostic codes

Errors are tagged with a stable `[PLY###]` prefix for IDE/CI consumers:

| Code | Meaning |
|---|---|
| `PLY001` | column not found in expression (`pl.col("X")`) |
| `PLY002` | `drop`: column not found |
| `PLY003` | `rename`: source column not found |
| `PLY004` | `cast`: column not found |
| `PLY005` | `drop_nulls`: subset column not found |
| `PLY006` | `with_row_index`: name collides with existing column |
| `PLY010` | join key error (missing / dtype mismatch) |
| `PLY011` | `group_by` key missing or aggregation type error |
| `PLY020` | `concat` schema mismatch (vertical / horizontal overlap / diagonal unify) |
| `PLY021` | `explode`: column not found or not `List[T]` / `Array[T]` |
| `PLY022` | `unpivot`: column not found or `on`-columns dtype mismatch |
| `PLY030` | eager-only method called on a `LazyFrame` (e.g. `lf.write_csv(...)`) — suggests `.collect()` |
| `PLY031` | lazy-only method called on a `DataFrame` (e.g. `df.sink_csv(...)`, `df.collect()`) — suggests `.lazy()` or removing the call |
| `PLY032` | function-call argument or return type mixes up `DataFrame[S]` and `LazyFrame[S]` — suggests the appropriate `.collect()` / `.lazy()` |
| `PLY033` | a variable annotation re-interprets the inferred frame as an unrelated type (neither subtype direction holds, ADR-0005) |
| `PLY040` | declared return type does not match the inferred return type — one shared code for the whole family: missing column, extra column, dtype difference, or the return type could not be inferred |
| `PLY041` | a schema field's `Annotated[pl.<Dtype>, ...]` metadata arity provably crashes pandera — the TypeError fires the first time the schema is used (`validate` / `@pa.check_types`), so every function referencing the schema is dead on arrival |
| `PLY042` | a column referenced inside a function is not declared in its (non-strict) parameter/validated schema — an undeclared dependency on caller extras ("checked island"), not a provable runtime failure; declare the column, or take a bare `pl.DataFrame` for row-polymorphic helpers |

## Apply-style helpers and warning codes

Some patterns are **not statically decidable** without help from the
user. Polypolarism detects them, falls back to a best-effort inference,
and emits a `[PLW###]` **warning** that names a concrete source change
that would let the analyser check the code precisely. Warnings are
non-fatal: the CLI exits `0` even when warnings are emitted.

| Form | Status | Note |
|---|---|---|
| `df.pipe(typed_helper)` where `typed_helper` is a `DataFrame[A] → DataFrame[B]` defined in the same module | ✅ inferred | uses the helper's declared return type |
| `df.pipe(untyped_helper)` defined in the same module | ✅ inferred | body is re-analysed with the propagated argument types |
| `df.pipe(external_helper)` (imported from another module) | ⚠️ `PLW002` | suggests defining the helper locally with a `DataFrame[Schema]` annotation |
| `df.pipe(lambda d: ...)` | ⚠️ `PLW004` | suggests promoting the lambda to a top-level typed function |
| `pl.col("x").map_elements(fn, return_dtype=pl.Float64)` | ✅ inferred | the declared `return_dtype` becomes the result dtype |
| `pl.col("x").map_elements(fn)` (no `return_dtype=`) | ⚠️ `PLW001` | falls back to receiver dtype; suggests adding `return_dtype=pl.<DType>` |
| `pl.col("x").map_batches(fn, return_dtype=...)` | ✅ inferred | same rule as `map_elements` |
| `external_helper(df)` (top-level call into an imported helper) | ⚠️ `PLW003` | suggests defining the helper locally or inlining the transformation |

Warning codes:

| Code | Meaning |
|---|---|
| `PLW001` | `map_elements` / `map_batches` without `return_dtype=` |
| `PLW002` | `pipe` with a callable that isn't in the analysed module |
| `PLW003` | function call to a name that isn't defined in the analysed module |
| `PLW004` | lambda / inline callable used where its return dtype is unknowable |
| `PLW005` | `pivot()` output schema is data-dependent; bind to a `DataFrame[Schema]` variable |
| `PLW006` | `DataFrame[X]` / `LazyFrame[X]` annotation references a schema the analyzer cannot resolve |
| `PLW007` | method not modeled by polypolarism. Expression/namespace methods: the result dtype degrades to `Unknown`; pin it with `.cast(...)` or a schema validation (a `.cast(...)` directly after the call retracts the warning). Frame methods probed to return a DataFrame/LazyFrame: the frame untracks; wrap the call in `Schema.validate(...)` (which retracts the warning). Terminal frame methods (`to_dicts`, `write_*`, `height`, ...) legitimately return non-frames and stay silent |
| `PLW008` | a variable annotation *narrows* the inferred schema without runtime backing (ADR-0005) — e.g. non-null over a post-join nullable; assert it with `Schema.validate(...)` or widen the annotation |
| `PLW010` | detected polars / pandera version is below the supported floor (see [Supported versions](versions.md)) |
| `PLW011` | a schema field annotation polypolarism cannot translate — the column registers as `Unknown` dtype instead of silently vanishing (pandera raises TypeError at first use if the annotation genuinely doesn't resolve) |
| `PLW012` | a grouped `std`/`var`/`sum` on a Date/Datetime/Time column is probed to yield an unconditionally all-null column (the same reduction raises in a plain select) — accepted, but probably not what you meant |
| `PLW013` | `typing.cast(DataFrame[Schema], x)` is **not** honored as a schema assertion — polypolarism infers `x`'s real schema and checks that (a lying cast fails at runtime under `@pa.check_types` too). The note flags an inert cast over a known schema (suggesting `# type: ignore[PLY040]` to suppress a mismatch) or an unverified assumption over an open/unknown source |

JSON output (`--format json`) emits warnings as `severity: "warning"`
diagnostics so editors and CI can route them separately from errors.
Every tagged diagnostic also exposes its `PLY###` / `PLW###` code as a
structured `"code"` field alongside the `[PLY###]`-prefixed message, so
consumers (pre-commit hooks, CI annotators, editors) never need to regex
the message text. Untagged diagnostics (file read / parse failures) omit
the field.

The JSON payload also carries a `functions` array — one entry per
analyzed function with its source span and schema summaries (parameter
frames, declared/inferred return frames as rendered dtype maps, plus
`open` / `strict` / `lazy` markers). Editor integrations use it to show
hovers without re-running the analysis.

## Schema diff block

When a single function has at least two column-level mismatches
(`MissingColumn`, `ExtraColumn`, or `TypeDifference` in any combination)
the text formatter appends an aligned diff block under the per-line
errors so the user can scan the whole shape difference at once:

```
  f (line 19): FAIL
    - [PLY040] Column 'id' has type Int64, but declared type is Int32
    - [PLY040] Missing column 'amount' of type Float64
    - [PLY040] Column 'name' has type Utf8, but declared type is Float64
    - [PLY040] Missing column 'extra' of type Int64
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
