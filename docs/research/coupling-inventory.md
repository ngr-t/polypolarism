# Polars / Pandera Coupling Inventory

A file:line map of every place `polypolarism` encodes assumptions about
the Polars or Pandera surface. Used as the refactor checklist for the
implementation steps in
[`docs/adr/0001-polars-pandera-version-support.md`](../adr/0001-polars-pandera-version-support.md).

The whole codebase is AST/string-based — there is **no runtime
`import polars` or `import pandera`**. Every entry below is a string set,
dispatch branch, or table that needs to migrate to
`src/polypolarism/compat/`.

This inventory is a snapshot. Line numbers will drift as the refactor
proceeds; the section names and table sizes are the durable anchors.

## `src/polypolarism/analyzer.py`

### Method-classification frozensets

| Lines | Symbol | Purpose |
|---|---|---|
| 100–131 | `_IDENTITY_FRAME_METHODS` | Methods that return the receiver shape unchanged (`sort`, `head`, `unique`, sinks, …) |
| 136–154 | `_LAZY_ONLY_METHODS` | LazyFrame-only methods (`collect*`, `explain`, `sink_*`, …) |
| 159–205 | `_EAGER_ONLY_METHODS` | DataFrame-only methods (`to_pandas`, `write_*`, `iter_*`, `partition_by`, `n_unique`, …) |

### Dtype name map

| Lines | Symbol | Notes |
|---|---|---|
| 210–227 | `_PL_DTYPE_NAME_MAP` | Maps `Int8…Int64`, `UInt8…UInt64`, `Float32/64`, `Utf8`/`String`, `Boolean`, `Date`, `Datetime`, `Duration`. **Duplicated** in `pandera_dtype.py:75–94`. The `String` → `Utf8()` alias at line 222 is the only intra-1.x rename currently handled. |

### Top-level `pl.*` shorthand

| Lines | Symbol | Notes |
|---|---|---|
| 235–247 | `_PL_AGG_SHORTHAND` | `sum/mean/min/max/first/last/count/n_unique/median/std/var` — accepts `pl.sum("col")` form |

### Selector-namespace dispatch

| Lines | Symbol | Notes |
|---|---|---|
| 424–496 | selector dispatch | `cs.all`, `cs.numeric`, `cs.integer`, `cs.float`, `cs.string`, `cs.boolean`, `cs.temporal`, `cs.by_name`, `cs.by_dtype`, `cs.starts_with`, `cs.ends_with`, `cs.contains`, `cs.exclude`, `cs.first`, `cs.last`. Selector algebra (`|`, `&`, `-`, `~`) handled inline. |

### Aggregation name → enum

| Lines | Symbol | Notes |
|---|---|---|
| 723–738 | aggregation `func_map` | Maps method names to `AggFunction` enum values (14 entries). |

### Expression method dispatch

| Lines | Group | Examples |
|---|---|---|
| 814–831 | Boolean predicates | `is_null`, `is_not_null`, `is_nan`, `is_finite`, `is_unique`, `is_in`, `is_between`, `has_nulls`, `not_` |
| 834–844 | Float-returning | `log`, `log10`, `log1p`, `exp`, `sqrt`, `cbrt`, `entropy` |
| 847–870 | Dtype-preserving | `abs`, `round`, `clip`, `floor`, `ceil`, `sign`, `neg`, `cum_sum`, `cum_max`, `cum_min`, `cum_prod`, `over`, `rolling_sum/min/max`, … |
| 873 | Shift-like | `shift`, `diff`, `pct_change` |
| 876–883 | Rolling float | `rolling_mean`, `rolling_std`, `rolling_var`, `rolling_median`, `rolling_quantile` |
| 1289 | Special | `cum_count` → `UInt32()` |
| 1308 | Utility | `map_elements`, `map_batches` (with `return_dtype=` kwarg) |

### Sub-namespace tables

| Lines | Symbol | Notes |
|---|---|---|
| 890–930 | `_STR_RETURN` | `str.*` namespace return types — `contains`, `starts_with`, `lower`, `upper`, `replace*`, `split`, `to_date`, `to_datetime`, `len_chars`, `count_matches`, … |
| 935–960 | `_DT_RETURN` | `dt.*` returning Int (`year`, `month`, `weekday`, …), `Date()`, `Int64()` epochs/totals |
| `_DT_PRESERVING` | `dt.*` returning the same Datetime/Date — `truncate`, `round`, `offset_by`, `replace_time_zone`, `convert_time_zone`, … |
| 976–1002 | `_LIST_PRESERVING` | `list.*` returning same List type — `unique`, `sort`, `reverse`, `head`, `tail`, `slice`, `drop_nulls`, `sample`, `shift` |
| | `_LIST_ELEMENT_RETURN` | `list.*` returning element type — `get`, `first`, `last`, `sum`, `mean`, `min`, `max`, `median` |
| 1194–1211 | Struct namespace | `field(name)` extracts field dtype; `rename_fields` preserves struct |

### Frame method dispatch

| Lines | Method | Notes |
|---|---|---|
| 1714 | `pl.concat()` | Top-level fn |
| 1718 | `validate()` | Pandera schema narrowing |
| 1737 | `pipe()` | Custom callable |
| 1746 | `collect`, `collect_async`, `collect_batches` | Lazy → eager transition |
| 1758 | `lazy` | Eager → lazy |
| 1764 | `agg` | Post `.group_by()` |
| 1777 | `join` | |
| 1779 | `group_by`, `group_by_dynamic`, `rolling` | Opaque (returns None) |
| 1782 | `join_asof` | |
| 1786 | `select` | |
| 1788 | `with_columns` | |
| 1792 | `drop` | |
| 1794 | `rename` | |
| 1796 | `cast` | |
| 1798 | `drop_nulls` | |
| 1802 | `with_row_index` | |
| 1806 | `filter` | |
| 1808 | `explode` | |
| 1810 | `vstack` | |
| 1812 | `hstack`, `extend` | |
| 1814 | `unpivot`, `melt` | |
| 1816 | `unnest` | |
| 1818 | `pivot` | |

### Join `how` literal handling

| Lines | Notes |
|---|---|
| 2079–2082 | Validates `how in {"inner", "left", "right", "full"}`. **`outer` is not accepted** — reflects the Polars 1.0 rename. |

### `pl.concat` `how` literal handling

| Lines | Notes |
|---|---|
| 2477–2497 | Accepts `vertical`, `horizontal`, `diagonal`, `diagonal_relaxed`. |

### Group-by spelling

| Lines | Notes |
|---|---|
| 1779, 2144, 2155, 2177 | Recognizes `group_by` only (not the legacy `groupby`). Kwarg form accepts both `by=` and `group_by=`. |

### Misc helpers

| Lines | What |
|---|---|
| 707, 1011 | `.alias()` for column renaming |
| 803, 2227 | `pl.col()` reference |
| 1398 | `pl.lit()` literal |
| 311, 1385 | `.cast()` |
| 1649 | `.partition_by()` (returns frame list) |

## `src/polypolarism/expr_infer.py`

Expression inference; receives the dispatch tables above. No standalone
hardcoded names beyond what `analyzer.py` exposes.

## `src/polypolarism/ops/join.py`

| Lines | What |
|---|---|
| 16 | `JoinHow = Literal["inner", "left", "right", "full"]` |
| 95–96 | Nullability rules: `left_nullable = how in ("right", "full")`, `right_nullable = how in ("left", "full")` |

## `src/polypolarism/ops/groupby.py`

| Lines | What |
|---|---|
| 105–209 | Aggregation signature inference. Per-aggregation functions `_infer_sum/_infer_mean/_infer_count/_infer_n_unique/_infer_list/_infer_first/_infer_last/_infer_min/_infer_max/_infer_float_reduction/_infer_product` and the `_AGG_INFER_MAP: dict[AggFunction, Callable[[DataType], DataType]]` table. |

Aggregation contract:

- `count()` → `UInt32()` (never nullable)
- `n_unique()` → `UInt32()` (never nullable)
- `sum(T)` → `T`
- `mean(T)` → `Float64()`
- `list(T)` → `List(T)`
- `first/last/min/max(T)` → `T`
- `std/var/median/quantile(T)` → `Float64()`
- `product(T)` → `T`

## `src/polypolarism/pandera_dtype.py`

| Lines | What |
|---|---|
| 75–94 | `_PL_DTYPE_MAP` — **near-duplicate** of `analyzer.py:210–227`, plus `Categorical` and `Null`. The duplication is the headline thing the `compat/` refactor eliminates. |
| 209 | `pl.List` / `pl.Array` wrapping |
| 217 | `pl.Struct` field dict |
| 249–262 | `_is_field_with_nullable` — recognizes `Field(nullable=True)` and `pa.Field(nullable=True)` |

## `src/polypolarism/pandera_schema.py`

| Lines | What |
|---|---|
| 19–21 | `_BASE_NAMES = frozenset({"DataFrameModel", "SchemaModel"})` — the entire Pandera-rename compat surface |
| 108 | `base.id in _BASE_NAMES or base.id in known_schemas` |
| 111 | `base.attr in _BASE_NAMES` (qualified `pa.DataFrameModel`) |
| 136–141 | Walks class body for `ast.AnnAssign` (field decls) and `class Config:` block |

## `src/polypolarism/pandera_annotation.py`

| Lines | What |
|---|---|
| 15, 78–82 | `_HEAD_NAMES = frozenset({"DataFrame", "LazyFrame"})` — accepts both bare and qualified (`pa.DataFrame`) |

## `pyproject.toml`

| Lines | What |
|---|---|
| 11 | `dependencies = []` — zero runtime deps |
| 32–40 | `dev` group: `polars>=1.0.0`, plus tooling. **No pandera dep listed** anywhere. |

## Refactor target → file mapping

When the `compat/` module lands, this is which inventory section moves
where:

| Inventory section | New home |
|---|---|
| Method classification frozensets | `compat/polars_api.py` |
| `_PL_DTYPE_NAME_MAP` (single source of truth) | `compat/polars_api.py` |
| `_PL_AGG_SHORTHAND`, `_AGG_INFER_MAP` | `compat/polars_api.py` |
| Selector / namespace tables (`_STR_RETURN` etc.) | `compat/polars_api.py` |
| `JoinHow` + nullability rules | `compat/polars_api.py` |
| `pl.concat` `how` set | `compat/polars_api.py` |
| `METHOD_ALIASES = {}` (new, scaffolding) | `compat/polars_api.py` |
| `PolarsProfile` (new, scaffolding) | `compat/polars_api.py` |
| `_BASE_NAMES`, `_HEAD_NAMES`, Field detection | `compat/pandera_api.py` |
| `pandera_dtype.py:_PL_DTYPE_MAP` | re-export from `compat/polars_api.py` |
