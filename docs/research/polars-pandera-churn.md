# Polars / Pandera API Churn Survey

A version-by-version reference of the API changes most relevant to a static
AST-based type checker. Used as input to
[`docs/adr/0001-polars-pandera-version-support.md`](../adr/0001-polars-pandera-version-support.md).

This document is a living reference: when a new Polars or Pandera release
introduces a relevant change, append a row rather than rewriting history.

## Cost classes

| Class | Meaning | Typical handling |
|---|---|---|
| **rename** | Same behavior, new identifier (method, dtype, kwarg). | One alias entry. |
| **semantics** | Same identifier, different return type / null behavior / column count. | Version-aware dispatch (a `PolarsProfile` field). |
| **constructor** | Positional-arg shape changed for a dtype/expression. | AST-level rewrite — same cost as a rename per call site. |
| **additive** | New API surface; old code keeps working. | Pick up when a fixture needs it. |

## Polars

### 0.19 (Sep 2023)

| Change | Cost | Affects static analysis? |
|---|---|---|
| `groupby` → `group_by` | rename | yes — method dispatch |
| `apply` → `map_elements` (Expr/Series), `map_rows` (DataFrame), `map_groups` (GroupBy) | rename, *receiver-contextual* | yes — same name on different receivers maps to different new names |
| `outer` join → `full` join (kwarg value) | rename | yes — `how=` literal table |
| Horizontal aggregations split out: `pl.sum`/`min`/`max` no longer reduce across columns; use `sum_horizontal`/`min_horizontal`/`max_horizontal` | semantics | yes — return-type changes from row-scalar to column-scalar shape |
| `all()`/`any()`: `drop_nulls=` → `ignore_nulls=`, default behavior changed to ignore nulls | rename + semantics | partially — silent behavior change |

### 0.20 (Jan 2024)

| Change | Cost | Affects static analysis? |
|---|---|---|
| `cumsum`/`cumprod`/`cummax`/`cummin`/`cumcount` → underscored `cum_*` | rename | yes — method dispatch |
| `Utf8` → `String` (canonical name; `Utf8` retained as alias) | rename | yes — dtype name table; already handled at `analyzer.py:222` |
| `count()` ignores nulls; use `len()` for old behavior | semantics | partially — return dtype unchanged (UInt32) but nullability differs |
| `Array(inner, width)` parameter order changed | constructor | yes — AST positional-arg parsing |
| `Decimal(precision, scale)` parameter order changed | constructor | yes — same |
| Dtype objects became instances rather than classes (`is` vs `==`) | runtime | no — we do AST string comparison |
| `Enum` dtype added | additive | only if a fixture uses it |

### 1.0 (Jul 2024)

| Change | Cost | Affects static analysis? |
|---|---|---|
| Outer/full join no longer coalesces keys by default; `coalesce=True` and `how="outer_coalesce"` removed | semantics | yes — output column count and naming changes (`property_name_right` etc.) |
| `replace()` redesigned: `default=` and `return_dtype=` removed; new `replace_strict()` for type-changing replace | rename + signature | yes — `replace()` return type and `replace_strict()` new dispatch entry |
| Many deprecated 0.19/0.20 paths removed | — | yes — closes the "alias surface still works" loophole |

### 1.x (Jul 2024 → present)

| Change | Cost | Affects static analysis? |
|---|---|---|
| `LazyFrame.collect(streaming=True/False)` → `engine="streaming"`/`"auto"` (1.23) | rename | runtime control flow only — no schema impact |
| `Categorical` lexical-comparison change; `ordering=` parameter deprecated (~1.32) | semantics | only if we model Categorical comparison results |

**Within-1.x summary**: no analyzer-relevant divergence between any two
1.x minors observed at the time of this writing. This is what motivates
shipping `PolarsProfile` with no fields.

## Pandera

### 0.18 / 0.19 (early 2024)

| Change | Cost | Affects static analysis? |
|---|---|---|
| `pandera.polars` module added (polars schema validation) | additive | no |

### 0.20 (mid 2024)

| Change | Cost | Affects static analysis? |
|---|---|---|
| `SchemaModel` → `DataFrameModel` (legacy name deprecated) | rename | yes — base-class name set; already handled at `pandera_schema.py:19-21` |
| Pyarrow dtype support | additive | no |

### 0.21+

| Change | Cost | Affects static analysis? |
|---|---|---|
| `polars >= 1.0.0` pinned as runtime dep | additive | no — we don't import pandera at runtime |

**Pandera summary**: the entire AST-relevant churn surface is `SchemaModel`
→ `DataFrameModel`, which is one element in a frozenset. There is no
ongoing maintenance burden here. `Field`, `Config: strict = True`, and
class-annotation introspection are stable across all 0.17–0.21+ versions
we've sampled.

## Characterization

Polars churn is **roughly half mechanical renames, half real semantic
shifts**, all clustered between 0.19 and 1.0. Renames are cheap if a
single dispatch layer canonicalizes them at entry. Semantic shifts cannot
be papered over with aliases; they need version-aware behavior. Together,
the pre-1.0 cluster is the largest single source of compatibility cost an
AST tool would face.

Within 1.x, churn has been quiet enough that a "support the latest two
minors" policy needs zero version-conditional logic in the analyzer
today.

Pandera's class-introspection surface is intentionally stable
(pydantic-style); a single legacy class name is the entire compat
surface.

## Source links

- [Polars 0.19 upgrade guide](https://docs.pola.rs/releases/upgrade/0.19/)
- [Polars 0.20 upgrade guide](https://docs.pola.rs/releases/upgrade/0.20/)
- [Polars 1.0 upgrade guide](https://docs.pola.rs/releases/upgrade/1/)
- [Polars Categorical & Enum guide](https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/)
- [Pandera DataFrame Models docs](https://pandera.readthedocs.io/en/latest/dataframe_models.html)
- [Pandera 0.19 release: Polars support](https://www.union.ai/blog-post/pandera-0-19-0-polars-dataframe-validation)
- [Pandera 0.20 release: Pyarrow support](https://www.union.ai/blog-post/pandera-0-20-0-pyarrow-data-type-support)
