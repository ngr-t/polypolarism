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

### 1.0 (Jul 2024) — big-bang minor

| Change | Cost | Affects static analysis? |
|---|---|---|
| Outer/full join no longer coalesces keys by default; `coalesce=True` and `how="outer_coalesce"` removed | semantics | yes — output column count and naming changes (`property_name_right` etc.) |
| `replace()` redesigned: `default=` and `return_dtype=` removed; new `replace_strict()` for type-changing replace | rename + signature | yes — `replace()` return type and `replace_strict()` new dispatch entry |
| `DataFrame.pivot(columns=…)` → `pivot(on=…)` (now first positional) | rename | yes — kwarg dispatch |
| `pl.nth(idx, *cols)` lost its `cols` parameter | signature | yes — call-site shape |
| `DataFrame.set_sorted(*cols)` reduced to single column | signature | yes |
| `Series.reshape((m, n))` and 2D-numpy `pl.Series` return `Array`, not `List` | semantics | yes — return dtype |
| `Series.rle()` struct fields renamed `lengths`/`values` → `len`/`value`; length dtype `Int32` → `UInt32` | semantics | yes — Struct field dispatch |
| `Series.list.get`/`array.get`/`gather` raise on OOB instead of returning null (param `null_on_oob` added) | semantics | partially — nullability of result |
| `Series.equals` no longer compares names by default | semantics | bool result, low-impact |
| `pl.from_arrow` decimals → `pl.Decimal` (was `Float64`) | semantics | yes — dtype inference of IO |
| `clip` with null bound now keeps original (was null) | semantics | nullability |
| `ewm_*` no longer forward-fills nulls | semantics | nullability |
| `group_by_dynamic(offset=)` default `"-every"` → `"0"` | semantics | low-impact |
| `LazyFrame.schema|dtypes|columns|width` issue `PerformanceWarning` (use `collect_schema()`) | deprecation | runtime-only |
| Many deprecated 0.19/0.20 paths removed | — | yes — closes the "alias surface still works" loophole |

### 1.1 – 1.16 (mid 2024 – late 2024)

Mostly additive. AST-relevant items:

| Version | Change | Cost |
|---|---|---|
| 1.1 | Right-join support added | additive (op table) |
| 1.2 | Scans gain `include_file_paths` (adds a `String` column when set) | additive (return shape conditional on kwarg) |
| 1.4 | `Expr.bin.size` / `Series.bin.size` | additive |
| 1.7 | IEJoin / non-equi `full` join semantics | semantics (new join algorithm) |
| 1.9 | `rename(strict=)`, `read_parquet(allow_missing_columns=)` | additive kwargs |
| 1.10 | `Expr.struct.unnest()` (multi-column projection); `Series.{first,last,approx_n_unique}` | additive |
| 1.11 | `pl.escape_regex` / `str.escape_regex` | additive |
| 1.15 | `pl.concat_arr` (returns `Array`) | additive |
| 1.16 | Left-join row-order **guarantee removed** from docs | semantics — invisible at signature level, breaks `set_sorted` assumptions downstream |

### 1.17 – 1.30 (late 2024 – mid 2025)

| Version | Change | Cost |
|---|---|---|
| 1.17 | `maintain_order` kwarg added to joins; `pl.select` lazy support | additive |
| 1.18 | **`Int128` dtype added** | new dtype |
| 1.19 | `Int128` IO support (CSV/IPC) | additive |
| 1.22 | `pl.linear_spaces`; `Catalog.schema` → `Catalog.namespace`; `DataType.is_object()` | rename + additive |
| 1.23 | `DataFrame.remove` / `LazyFrame.remove` (sibling of `filter`, inverted predicate) | additive — new method dispatch entry |
| 1.24 | `is_in(nulls_equal=)` kwarg | semantics toggle |
| 1.25 | **`Enum` dtype stabilized**; `arr.len` added; `Int128` round-trips from Arrow | new dtype |
| 1.27 | **`hist` bottom interval now closed** — Struct column count can change in some bucketings; `Partition` API renamed | semantics |
| 1.28 | `Series.backward_fill`/`forward_fill`; `rolling_kurtosis`; rolling-min/max on temporals; `sort(nulls_last=True)` extended to bool/categorical/enum | additive |
| 1.29 | `RoundMode` enum for Decimal/Float rounding | additive |
| 1.30 | `list/arr.contains(nulls_equal=)` | semantics toggle |

### 1.31 – 1.40 (mid 2025 – Apr 2026)

| Version | Change | Cost |
|---|---|---|
| 1.31 | **Old streaming engine removed** (#23103); `scan_parquet(allow_missing_columns=)` → `missing_columns=` | rename (deprecation) |
| 1.32 | **`Selector` becomes a concrete DSL type** — `pl.selectors.*` returns `Selector` objects, not raw `Expr` (#23351). **`Categorical`/`(Frozen)Categories` reworked** (#23016) — `pl.Categorical(ordering=)` semantics differ. `arr.mean/slice/head/tail`, `is_close`, `pl.row_index()` (unstable), `to_dummies(drop_nulls=)`, `str.to_integer(dtype=)` added | semantics + new types |
| 1.33 | "Eager `Expr` → lazy compatible" sweep (#24027) — several `Expr` methods that previously eagerly materialized were converted | semantics (eagerness) |
| 1.34 | **`UInt128` dtype**; `LazyFrame.sink_batches`, `collect_batches` (return iterators of `DataFrame`); decimals fixed-scale (#24542); `Array` gains `unique`/`n_unique`/`arg_unique` | new dtype + new methods |
| 1.35 | **`Decimal` stabilized**; `union()` (unordered concat); `name.replace`; `arr.eval`, `list.agg`, `arr.agg`; `rolling_rank`; `Expr.item`; `unnest(separator=)` | dtype stabilization + new methods |
| 1.36 | **`Float16` dtype**; `Schema.to_arrow()` (unstable); `concat(how="horizontal", strict=)`; `bin.slice/head/tail`; `explode(empty_as_null=, keep_nulls=)`; `mode(maintain_order=)` | new dtype + new methods |
| 1.37 | `pl.PartitionBy`; `Expr.min_by`/`max_by`; `Series.arr.mean`; `expr.get(null_on_oob=)`; lazy `collect_all` | additive |
| 1.38 | `retries=n` → `storage_options={"max_retries": n}`; old partition-sink API removed; `scan_lines`; `height=` (unstable) on frame ctor | rename + additive |
| 1.39 | `Schema.contains_dtype()`; `LazyFrame.sink_iceberg` (unstable); Decimal product reduction; `implode(maintain_order=)`; `Expr.truncate` | additive |
| 1.40 | Dataframe interchange protocol deprecated; `pl.merge_sorted` over multiple frames; `{list,arr}.{any,all}(ignore_nulls=)`; `is_unique` on list/array; `group_by()` with no key expressions; `unnest()` defaults to all eligible columns; `scan_lines` column renamed `lines` → `line` | additive + minor renames |

**Within-1.x summary** (revised after release-by-release survey, current
through 1.40.1, 2026-04-22):

The 1.x line is **overwhelmingly additive at the method-name level** —
~30 of 41 minors are additive only. **However**, there are at least five
distinct categories of breakage a static checker must model:

1. **The 1.0 sweep itself** (pivot kwarg, replace split, nth/set_sorted
   varargs, reshape→Array, rle field rename, from_arrow decimal, clip null
   semantics).
2. **New dtypes** that AST callers may write literally as `pl.Int128`
   (1.18), `pl.UInt128` (1.34), `pl.Float16` (1.36), and stabilized
   `Enum` (1.25), `Decimal` (1.35). **Currently absent from
   `_PL_DTYPE_NAME_MAP`** in both `analyzer.py` and `pandera_dtype.py`.
3. **Selector-as-DSL** (1.32) — `pl.selectors.*` returns a `Selector`
   object, not an `Expr`. The current selector dispatch
   (`analyzer.py:424–496`) treats results as expressions; downstream
   destructuring may diverge.
4. **`Categorical`/`(Frozen)Categories` rework** (1.32) — `pl.Categorical`
   surface and dtype-equality story changed.
5. **`hist` bin-closure shift** (1.27) and **left-join row-order
   de-guarantee** (1.16) — semantics under fixed signatures, invisible
   in stubs.

The "two latest minors" window keeps any single supported-pair difference
small (e.g. 1.39 ↔ 1.40 diverges only by `merge_sorted` and a handful of
kwargs), **but** the cumulative gap between *what the analyzer currently
knows* and *what 1.x users write today* (Int128 / UInt128 / Float16 / Enum
/ Decimal as schema dtypes) is significant. This is the gap the implementation
plan needs to close as a separate, pre-refactor step.

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

The pre-1.0 cluster (0.19 → 1.0) is **roughly half mechanical renames,
half real semantic shifts**. Renames are cheap if a single dispatch layer
canonicalizes them at entry; semantic shifts cannot be papered over with
aliases.

Within 1.x, **most minors are additive**, but the cumulative drift across
1.0 → 1.40 (~22 months) is **non-trivial**: five new dtypes
(`Int128`/`UInt128`/`Float16` plus stabilization of `Enum` and `Decimal`),
the 1.32 selector-as-DSL change, the 1.32 `Categorical` rework, the 1.27
`hist` bin-closure shift, and the 1.16 left-join row-order de-guarantee
are all things a static checker needs to know about. A "support the latest
two minors" policy keeps any one supported-pair diff small, but the
analyzer's current dtype tables lag the language by several minors and
need to catch up before that policy becomes meaningful.

Pandera's class-introspection surface is intentionally stable
(pydantic-style); a single legacy class name is the entire compat
surface.

## Source links

- [Polars 0.19 upgrade guide](https://docs.pola.rs/releases/upgrade/0.19/)
- [Polars 0.20 upgrade guide](https://docs.pola.rs/releases/upgrade/0.20/)
- [Polars 1.0 upgrade guide](https://docs.pola.rs/releases/upgrade/1/)
- [Polars changelog index](https://docs.pola.rs/releases/changelog/)
- [Polars GitHub releases](https://github.com/pola-rs/polars/releases)
- [Polars versioning policy](https://docs.pola.rs/development/versioning/)
- [Polars Categorical & Enum guide](https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/)
- [Pandera DataFrame Models docs](https://pandera.readthedocs.io/en/latest/dataframe_models.html)
- [Pandera 0.19 release: Polars support](https://www.union.ai/blog-post/pandera-0-19-0-polars-dataframe-validation)
- [Pandera 0.20 release: Pyarrow support](https://www.union.ai/blog-post/pandera-0-20-0-pyarrow-data-type-support)
