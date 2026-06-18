# Supported Operations

## Join

- Supported join types: `inner`, `left`, `right`, `full`
- Left join makes right columns nullable
- Right join makes left columns nullable
- Full join makes both sides nullable

## Group By + Aggregation

Supported aggregation functions: `sum`, `mean`, `min`, `max`, `count`,
`n_unique`, `first`, `last`, `list`, `std`, `var`, `median`, `quantile`,
`product`. `median`/`quantile` return `Float64` (or `Nullable[Float64]`
if the input is nullable); `std`/`var` return `Nullable[Float64]` even on
non-nullable input — the default `ddof=1` is null for singleton groups
(an explicit literal `ddof=0` keeps plain `Float64` in expression
position); `product` preserves the numeric input dtype.

## Select / With Columns

`select` and `with_columns` are recognised; column dtypes flow through
literal expressions, `pl.col(...)` references, arithmetic, and
`.alias(...)`.

## Frame reshape

| Method | Effect on FrameType |
|---|---|
| `filter(...)` | identity |
| `sort(...)`, `head`, `tail`, `limit`, `slice`, `reverse`, `sample`, `unique`, `clone`, `lazy`, `set_sorted`, `shrink_to_fit`, `rechunk` | identity |
| `drop("a", "b")` / `drop(["a","b"])` | remove the named columns; error on unknown name |
| `rename({"old": "new"})` | rename keys; preserves dtype + nullability + required |
| `cast({"col": pl.Int32})` | per-column dtype change; preserves the receiver's `Nullable[...]` wrapper |
| `drop_nulls(subset=[...])` | strips `Nullable[...]` from `subset` (or every column if omitted) |
| `with_row_index(name="index")` | prepends a `UInt32` column |

`pl.<dtype>` literals recognised by `cast` and Pandera schema fields:
`Int8` / `Int16` / `Int32` / `Int64` / `Int128` (polars 1.18+),
`UInt8` / `UInt16` / `UInt32` / `UInt64` / `UInt128` (polars 1.34+),
`Float16` (polars 1.36+) / `Float32` / `Float64`,
`Utf8` (alias `String`), `Boolean`, `Date`,
`Datetime` (incl. `pl.Datetime("us", "UTC")`), `Duration`,
`Categorical`, `Enum` (polars 1.25+ stabilized),
`Decimal(precision, scale)` (polars 1.35+ stabilized — precision and
scale are preserved, so `Decimal(20, 4)` and `Decimal(20, 2)` are
distinct types), `List(inner)` and `Array(inner, width)` (the width is
tracked — `Array` is a distinct dtype from `List`, two `Array`s with
different known widths are distinct types, and a width the analyzer
cannot resolve compares as a wildcard).

## Expression predicates and narrowing

| Form | Inferred type |
|---|---|
| `pl.col("x") > y`, `==`, `!=`, `<`, `<=`, `>=` | `Boolean` |
| `expr1 & expr2`, `expr1 \| expr2`, `~expr` | `Boolean` |
| `pl.col("x").is_null() / is_not_null() / is_nan() / is_not_nan() / is_finite() / is_infinite() / is_unique() / is_duplicated() / is_first_distinct() / is_last_distinct() / is_in([...]) / is_between(lo, hi) / has_nulls()` | `Boolean` |
| `pl.col("x").fill_null(v)` / `fill_nan(v)` | strips `Nullable[...]` from the receiver |
| `pl.col("x").abs() / round() / clip(...) / floor() / ceil() / sign() / neg()` | preserves the receiver's dtype |
| `pl.col("x").log() / log10() / log1p() / exp() / sqrt() / cbrt() / entropy()` | `Float64` (`Nullable[Float64]` if input is nullable) |
| `pl.col("x").median() / quantile(p)` (in select / agg) | `Float64` |
| `pl.col("x").std() / var()` (in select / agg) | `Nullable[Float64]` (`ddof=1` is null on a single sample); a literal `ddof=0` keeps `Float64` |
| `pl.col("x").sum() / mean() / min() / max() / first() / last() / count() / n_unique() / product()` (in select / agg) | reduction result dtype |

`df.filter(pred)` is identity-typed but the predicate is walked through
the same expression analyser, so referencing a missing column produces a
`Column 'X' not found` error.

## Sub-namespaces

Method chains on `.str` / `.dt` / `.list` / `.arr` / `.struct` / `.bin` /
`.cat` are dispatched to per-namespace return-type tables, and each
namespace validates its receiver dtype (`.list` requires `List`, `.arr`
requires `Array`, `.cat` requires `Categorical` / `Enum` — pple-wrong-namespace-dtype
otherwise). The receiver's `Nullable[...]` wrapper is preserved on the
result.

**`pl.col("s").str.<m>(...)`**

| Methods | Return |
|---|---|
| `contains`, `contains_any`, `starts_with`, `ends_with`, `is_empty` | `Boolean` |
| `lower`, `upper`, `to_lowercase`, `to_uppercase`, `to_titlecase`, `strip*`, `lstrip`, `rstrip`, `replace*`, `replace_many`, `pad_start`, `pad_end`, `zfill`, `slice`, `head`, `tail`, `reverse`, `concat`, `join` | `Utf8` |
| `len_chars`, `len_bytes`, `count_matches` | `UInt32` |
| `split` | `List[Utf8]` |
| `to_date` / `to_datetime` | `Date` / `Datetime` |

**`pl.col("ts").dt.<m>(...)`**

| Methods | Return |
|---|---|
| `year`, `iso_year`, `millisecond`, `microsecond`, `nanosecond` | `Int32` |
| `month`, `day`, `hour`, `minute`, `second`, `weekday`, `quarter`, `week` | `Int8` |
| `ordinal_day` | `Int16` |
| `epoch`, `timestamp`, `total_*` | `Int64` |
| `date()` | `Date` |
| `truncate`, `round`, `offset_by`, `replace_time_zone`, `convert_time_zone`, `month_start`, `month_end` | preserves receiver dtype |

**`pl.col("xs").list.<m>(...)`** (requires a `List[T]` receiver)

| Methods | Return |
|---|---|
| `len` | `UInt32` |
| `unique`, `sort`, `reverse`, `head`, `tail`, `slice`, `drop_nulls`, `sample`, `shift` | preserves receiver dtype |
| `get(i)`, `first`, `last`, `min`, `max`, `explode` | element dtype |
| `sum` | element dtype for the ≥32-bit numeric widths; `Int8/Int16/UInt8/UInt16` widen to `Int64`, `Boolean` to `UInt32` |
| `mean`, `median`, `std`, `var` | `Float64` (`Float32` elements stay `Float32`) |

**`pl.col("q").arr.<m>(...)`** (requires an `Array[T]` receiver)

| Methods | Return |
|---|---|
| `len`, `n_unique`, `arg_min`, `arg_max`, `count_matches` | `UInt32` |
| `contains`, `any`, `all` | `Boolean` |
| `sort`, `reverse`, `shift` | preserves receiver `Array[T]` |
| `unique`, `head`, `tail`, `slice`, `to_list` | `List[T]` (the fixed width is lost) |
| `get(i)`, `first`, `last`, `min`, `max`, `explode` | element dtype |
| `sum` / `mean`, `median`, `std`, `var` | same widening rules as `.list` |
| `eval(body)` | `Array[<body dtype>]` |

**`pl.col("c").cat.<m>(...)`** (requires a `Categorical` / `Enum` receiver)

| Methods | Return |
|---|---|
| `get_categories` | `Utf8` (length-changing; never inherits the receiver's nullability) |
| `len_bytes`, `len_chars` | `UInt32` |
| `starts_with`, `ends_with` | `Boolean` |
| `slice` | `Utf8` |

**`pl.col("s").struct.<m>(...)`**

| Methods | Return |
|---|---|
| `field("name")` | dtype of that field on the receiver `Struct{...}`; errors if the field doesn't exist |
| `rename_fields(...)` | preserves receiver `Struct{...}` |

## Frame restructuring

| Form | Result |
|---|---|
| `df.explode("xs")` / `explode(["a","b"])` | `List[T]` / `Array[T]` columns become `T`; errors if column isn't a list-like container |
| `pl.concat([f1, f2])` (default `how="vertical"`) | column sets must match; per-column dtypes unified, `Nullable[...]` widened |
| `pl.concat([f1, f2], how="horizontal")` | column sets must be disjoint; merged into one frame |
| `pl.concat([f1, f2], how="diagonal")` / `"diagonal_relaxed"` | union of columns; columns absent in any input become `Nullable[T]` |
| `df.vstack(other)` | shorthand for vertical `pl.concat([df, other])` |
| `df.hstack(other)` / `df.extend(other)` | shorthand for horizontal `pl.concat([df, other])` |
| `df.unpivot(index=[...], on=[...], variable_name="variable", value_name="value")` / `df.melt(...)` | output schema `{index..., variable_name: Utf8, value_name: T}` where `T` unifies the dtypes of the `on` columns |
| `df.unnest("s")` / `df.unnest(["a","b"])` | each named `Struct{...}` column is replaced by its fields; receiver `Nullable[Struct]` widens each field to `Nullable[T]`; errors on missing column or non-`Struct` |
| `df.pivot(on=, index=, values=)` | output schema is data-dependent and so cannot be inferred. Polypolarism emits `[pplw-data-dependent-schema]` with a copy-pasteable Pandera schema sketch and trusts the user's `result: DataFrame[Schema]` annotation when the result is bound to a typed variable. |
| **LazyFrame**: `lf.collect()`, `lf.collect_async()`, `lf.collect_batches()`, `lf.cache()`, `lf.first()`, `lf.last()`, `lf.inspect()`, `lf.top_k(...)`, `lf.bottom_k(...)`, `lf.sink_csv(...)`, `lf.sink_parquet(...)`, `lf.sink_ipc(...)`, `lf.sink_ndjson(...)`, `lf.sink_batches(...)`, `df.lazy()` | column shape preserved through the call. `LazyFrame[Schema]` and `DataFrame[Schema]` are *statically distinguished*: `lazy()` flips the receiver to lazy and `collect*()` flips it to eager. Calling a lazy-only method on a `DataFrame` (or vice versa) emits `pple-lazy-only-method` / `pple-eager-only-method`. Crossing eager↔lazy in function-call arguments or return types emits `pple-eager-lazy-mismatch`. |

`df.partition_by("k")` returns a list of frames — assigning it
to a variable binds the variable as a `FrameList(element=...)` whose
element type carries through subscript indexing (`parts[0]`) and
for-loop iteration (`for p in parts:`). With `include_key=False` the
partition keys are dropped from each element schema. Only
`partition_by` produces a `FrameList` today; other operations that
return multiple frames are out of scope.

## Window / time-series

| Form | Inferred type |
|---|---|
| `pl.col("v").cum_sum() / cum_max() / cum_min() / cum_prod()` | preserves receiver dtype |
| `pl.col("v").cum_count()` | `UInt32` |
| `pl.col("v").shift(n) / diff(n) / pct_change(n)` | `Nullable[T]` (head positions become NULL) |
| `pl.col("v").rolling_sum(...) / rolling_min / rolling_max` | `Nullable[T]` of the receiver dtype (`rolling_sum` upcasts `Int8/Int16/UInt8/UInt16` → `Int64`, `Boolean` → `UInt32`); probed-invalid receivers (e.g. String, Decimal — Date/Datetime/Time/Duration too for `rolling_sum`) flag pple-non-numeric-operand |
| `pl.col("v").rolling_mean / rolling_std / rolling_var / rolling_median / rolling_quantile` | `Nullable[Float64]` (windows below `min_samples` are null); an explicit literal `min_samples<=1` / `window_size=1` keeps plain `Float64` (`rolling_std`/`rolling_var` also need `ddof=0`) |
| `pl.col("v").mean().over("k")` and other aggregations followed by `.over(...)` | preserves receiver dtype |
| `df.group_by_dynamic("ts", every="1d").agg(...)` / `df.rolling("ts", period="1d").agg(...)` | same shape as `df.group_by(...).agg(...)` |
| `df.join_asof(other, on=..., left_on=..., right_on=...)` | same column shape as `df.join(other, how="left")` (right side `Nullable`) |

## Plural `pl.col`

`pl.col("a", "b", ...)` and `pl.col(["a", "b"])` inside `select` /
`with_columns` fan out to the named columns (their dtypes flow through
unchanged). Missing names raise `[pple-column-not-found]`.

## `pl.*` expression constructors

| Form | Inferred type |
|---|---|
| `pl.concat_str([...], separator=...)` | `Utf8` |
| `pl.format(template, *exprs)` | `Utf8` |
| `pl.coalesce(*exprs)` | unification of operand dtypes; non-`Nullable` if any operand is non-`Nullable` |
| `pl.struct(pl.col("a"), pl.col("b"), ...)` | `Struct{a: T_a, b: T_b, ...}` (from receiver column names) |

## `polars.selectors`

`cs.*` selectors are expanded to a list of matching columns when used as
positional arguments to `select`, `with_columns`, or `drop`.

| Selector | Matches |
|---|---|
| `cs.all()` | every column in the frame |
| `cs.numeric()` | all integer + float columns |
| `cs.integer()` | `Int8`/`Int16`/`Int32`/`Int64`/`Int128`/`UInt8`/`UInt16`/`UInt32`/`UInt64`/`UInt128` |
| `cs.float()` | `Float16`/`Float32`/`Float64` |
| `cs.string()` | `Utf8` |
| `cs.boolean()` | `Boolean` |
| `cs.temporal()` | `Date`/`Datetime`/`Duration` |
| `cs.starts_with("prefix")` / `ends_with("suffix")` / `contains("part")` | column-name pattern |
| `cs.by_name("a", "b", ...)` | exact-name list |
| `cs.by_dtype(pl.Int64, pl.Float64)` | matching dtypes |
| `cs.first()` / `cs.last()` | first or last column of the frame |
| `cs.exclude(names_or_selector)` | every column not matched by the names or inner selector |
| `~sel` / `sel1 | sel2` / `sel1 & sel2` / `sel1 - sel2` | complement / union / intersection / difference of selectors |
