# polypolarism

Static type checker for Polars DataFrames based on row polymorphism.

## Features

- **Static type checking** for Polars DataFrame operations without running your code
- **Pandera schema declaration** with `pa.DataFrameModel` and `DataFrame[Schema]` annotations
- **Validation-as-narrowing**: `Schema.validate(df)`, `df.pipe(Schema.validate)`, and `Schema.validate(lf).collect()` all narrow the static type downstream
- **Operation support**: join, join_asof, group_by, group_by_dynamic, rolling (time-window), select, with_columns, filter (predicates type-checked), sort, head/tail/limit/slice, unique, drop, rename, cast, drop_nulls, with_row_index, explode, unpivot/melt, vstack/hstack/extend, `pl.concat([...], how=vertical|horizontal|diagonal)`, plus identity passthrough for lazy/collect/clone/reverse/sample/set_sorted
- **Error detection**:
  - Missing columns
  - Type mismatches in join keys
  - Invalid aggregation function applications
  - Declared vs inferred return type differences
  - Strict-mode extra columns (`class Config: strict = True`)
  - `Optional[T]` (column may be absent) vs `pa.Field(nullable=True)` (value may be null)

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ngr-t/polypolarism.git
```

Or clone and install locally:

```bash
git clone https://github.com/ngr-t/polypolarism.git
cd polypolarism
pip install .
```

> **Note**: This package is not yet published to PyPI.

## Quick Start

Declare schemas as Pandera `DataFrameModel` classes and annotate functions
with `DataFrame[Schema]`:

```python
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Users(pa.DataFrameModel):
    user_id: int
    name: str


class Orders(pa.DataFrameModel):
    order_id: int
    user_id: int
    amount: pl.Float64


class Joined(pa.DataFrameModel):
    user_id: int
    name: str
    order_id: int
    amount: pl.Float64


def merge_users_orders(
    users: DataFrame[Users],
    orders: DataFrame[Orders],
) -> DataFrame[Joined]:
    return users.join(orders, on="user_id", how="inner")
```

Then run the type checker:

```bash
polypolarism your_module.py
```

## Schema declaration

Use Pandera class-based schemas. Field annotations accept Python builtins
(`int`, `str`, `float`, `bool`), polars dtype classes (`pl.Int64`,
`pl.Float64`, `pl.UInt32`, ...), `Optional[T]`, and
`Annotated[pl.List, pl.Int64()]` / `Annotated[pl.Array, pl.Int64(), 3]` /
`Annotated[pl.Struct, {...}]` (or the equivalent `pl.List(...)` /
`pl.Array(...)` / `pl.Struct(...)` call forms) for nested types. `List`
and `Array` are distinct dtypes — polars does not interchange them — but
the `Array` width is not tracked.

```python
class Example(pa.DataFrameModel):
    id: int                                          # required, non-null
    name: str = pa.Field(nullable=True)              # required, may be null
    age: Optional[int]                               # column may be absent
    score: pl.Float64
    tags: Annotated[pl.List, pl.Utf8()]
    addr: Annotated[pl.Struct, {"city": pl.Utf8()}]

    class Config:
        strict = True   # reject any column not listed above
```

### Validation as type narrowing

Any of the following bind a downstream variable's type to the schema:

```python
df2 = Schema.validate(df)            # assignment-bound LHS
Schema.validate(df)                  # bare statement narrows df
df.pipe(Schema.validate)             # pipe chain
Schema.validate(lf).collect()        # LazyFrame -> DataFrame
```

Bare-statement narrowing only fires at the function body's top level;
narrowing inside `if`/`for`/`while`/`try`/`with` is intentionally out of
scope.

## CLI Usage

```bash
# Check a single file
polypolarism path/to/file.py

# Check a directory
polypolarism path/to/project/

# JSON output (e.g. for editor integrations)
polypolarism --format json path/to/file.py

# Show version
polypolarism --version

# Disable colored output
polypolarism --no-color path/to/file.py

# Override the assumed Polars / Pandera version (suppresses PLW010)
polypolarism --polars-version 1.40 path/to/file.py
polypolarism --pandera-version 0.20 path/to/file.py

# Skip version detection entirely
polypolarism --no-version-check path/to/file.py
```

## Supported versions

`polypolarism` reasons about your code through AST analysis only — it
does not import `polars` or `pandera` at runtime. Even so, the dispatch
tables encode assumptions about the libraries' surface, so the supported
window is narrow and explicit:

- **Polars**: latest two 1.x minor releases (currently `1.39` / `1.40`).
  Pre-1.0 surface is **out of scope** — the analyzer doesn't recognize
  the legacy method spellings (`groupby`, `cumsum`, `apply`, `Utf8`,
  `outer`-join, …) and will silently misanalyse code written against
  those.
- **Pandera**: `0.19+`. Both `DataFrameModel` (post-0.20) and the
  legacy `SchemaModel` are accepted indefinitely; the difference is
  one entry in a name set and costs nothing.

When the version detected from your project's lockfile (`uv.lock` /
`poetry.lock`) or the installed environment falls below the supported
floor, polypolarism emits a `[PLW010]` warning to stderr — it doesn't
fail the run, just tells you that type-check accuracy is best-effort
below the window. Use `--polars-version <ver>` (or the
`[tool.polypolarism]` config below) to opt back in if you've audited
that your code stays within the analyzer's known surface.

Only exact sources can trigger the warning. When the version is merely
inferred from a `>=` floor in `[project.dependencies]`, no PLW010 is
emitted — `polars>=1.0` says what the project tolerates, not what it
runs.

### Project-level configuration

Persist the assumed versions in your project's `pyproject.toml`:

```toml
[tool.polypolarism]
polars_version = "1.40"
pandera_version = "0.20"
```

Detection priority (first match wins per package):

1. `--polars-version` / `--pandera-version` CLI flag
2. `[tool.polypolarism]` section in the project's `pyproject.toml`
3. The project's `uv.lock` (exact pinned version)
4. The project's `poetry.lock` (exact pinned version)
5. The version installed in the running environment
   (`importlib.metadata`)
6. `[project.dependencies]` / `[dependency-groups.*]` floor
   (never triggers PLW010 — a `>=` floor is not the version in use)

For more on the policy, see
[`docs/adr/0001-polars-pandera-version-support.md`](docs/adr/0001-polars-pandera-version-support.md).

## Example Output

For valid code:

```
  merge_users_orders: OK
  aggregate_sales: OK

All 2 function(s) passed.
```

For invalid code:

```
  bad_join: FAIL
    - Column 'user_id' not found in right frame
    - Could not infer return type

1 function(s) failed, 0 passed.
```

## Supported Operations

### Join

- Supported join types: `inner`, `left`, `right`, `full`
- Left join makes right columns nullable
- Right join makes left columns nullable
- Full join makes both sides nullable

### Group By + Aggregation

Supported aggregation functions: `sum`, `mean`, `min`, `max`, `count`,
`n_unique`, `first`, `last`, `list`, `std`, `var`, `median`, `quantile`,
`product`. `median`/`quantile` return `Float64` (or `Nullable[Float64]`
if the input is nullable); `std`/`var` return `Nullable[Float64]` even on
non-nullable input — the default `ddof=1` is null for singleton groups
(an explicit literal `ddof=0` keeps plain `Float64` in expression
position); `product` preserves the numeric input dtype.

### Select / With Columns

`select` and `with_columns` are recognised; column dtypes flow through
literal expressions, `pl.col(...)` references, arithmetic, and
`.alias(...)`.

### Frame reshape

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
not tracked — `Array` is a distinct dtype from `List` but two `Array`s
of the same element type compare equal regardless of width).

### Expression predicates and narrowing

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

### Sub-namespaces

Method chains on `.str` / `.dt` / `.list` / `.arr` / `.struct` / `.bin` /
`.cat` are dispatched to per-namespace return-type tables, and each
namespace validates its receiver dtype (`.list` requires `List`, `.arr`
requires `Array`, `.cat` requires `Categorical` / `Enum` — PLY012
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

### Frame restructuring

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
| `df.pivot(on=, index=, values=)` | output schema is data-dependent and so cannot be inferred. Polypolarism emits `[PLW005]` with a copy-pasteable Pandera schema sketch and trusts the user's `result: DataFrame[Schema]` annotation when the result is bound to a typed variable. |
| **LazyFrame**: `lf.collect()`, `lf.collect_async()`, `lf.collect_batches()`, `lf.cache()`, `lf.first()`, `lf.last()`, `lf.inspect()`, `lf.top_k(...)`, `lf.bottom_k(...)`, `lf.sink_csv(...)`, `lf.sink_parquet(...)`, `lf.sink_ipc(...)`, `lf.sink_ndjson(...)`, `lf.sink_batches(...)`, `df.lazy()` | column shape preserved through the call. `LazyFrame[Schema]` and `DataFrame[Schema]` are *statically distinguished*: `lazy()` flips the receiver to lazy and `collect*()` flips it to eager. Calling a lazy-only method on a `DataFrame` (or vice versa) emits `PLY031` / `PLY030`. Crossing eager↔lazy in function-call arguments or return types emits `PLY032`. |

`df.partition_by("k")` returns a list of frames — assigning it
to a variable binds the variable as a `FrameList(element=...)` whose
element type carries through subscript indexing (`parts[0]`) and
for-loop iteration (`for p in parts:`). With `include_key=False` the
partition keys are dropped from each element schema. Only
`partition_by` produces a `FrameList` today; other operations that
return multiple frames are out of scope.

### Window / time-series

| Form | Inferred type |
|---|---|
| `pl.col("v").cum_sum() / cum_max() / cum_min() / cum_prod()` | preserves receiver dtype |
| `pl.col("v").cum_count()` | `UInt32` |
| `pl.col("v").shift(n) / diff(n) / pct_change(n)` | `Nullable[T]` (head positions become NULL) |
| `pl.col("v").rolling_sum(...) / rolling_min / rolling_max` | `Nullable[T]` of the receiver dtype (`rolling_sum` upcasts `Int8/Int16/UInt8/UInt16` → `Int64`, `Boolean` → `UInt32`); probed-invalid receivers (e.g. String, Decimal — Date/Datetime/Time/Duration too for `rolling_sum`) flag PLY016 |
| `pl.col("v").rolling_mean / rolling_std / rolling_var / rolling_median / rolling_quantile` | `Nullable[Float64]` (windows below `min_samples` are null); an explicit literal `min_samples<=1` / `window_size=1` keeps plain `Float64` (`rolling_std`/`rolling_var` also need `ddof=0`) |
| `pl.col("v").mean().over("k")` and other aggregations followed by `.over(...)` | preserves receiver dtype |
| `df.group_by_dynamic("ts", every="1d").agg(...)` / `df.rolling("ts", period="1d").agg(...)` | same shape as `df.group_by(...).agg(...)` |
| `df.join_asof(other, on=..., left_on=..., right_on=...)` | same column shape as `df.join(other, how="left")` (right side `Nullable`) |

### Plural `pl.col`

`pl.col("a", "b", ...)` and `pl.col(["a", "b"])` inside `select` /
`with_columns` fan out to the named columns (their dtypes flow through
unchanged). Missing names raise `[PLY001]`.

### `pl.*` expression constructors

| Form | Inferred type |
|---|---|
| `pl.concat_str([...], separator=...)` | `Utf8` |
| `pl.format(template, *exprs)` | `Utf8` |
| `pl.coalesce(*exprs)` | unification of operand dtypes; non-`Nullable` if any operand is non-`Nullable` |
| `pl.struct(pl.col("a"), pl.col("b"), ...)` | `Struct{a: T_a, b: T_b, ...}` (from receiver column names) |

### `polars.selectors`

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

### Diagnostic codes

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

### Apply-style helpers and warning codes

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
| `PLW010` | detected polars / pandera version is below the supported floor (see [Supported versions](#supported-versions)) |

JSON output (`--format json`) emits warnings as `severity: "warning"`
diagnostics so editors and CI can route them separately from errors.

### Schema diff block

When a single function has at least two column-level mismatches
(`MissingColumn`, `ExtraColumn`, or `TypeDifference` in any combination)
the text formatter appends an aligned diff block under the per-line
errors so the user can scan the whole shape difference at once:

```
  f (line 19): FAIL
    - Column 'id' has type Int64, but declared type is Int32
    - Missing column 'amount' of type Float64
    - Column 'name' has type Utf8, but declared type is Float64
    - Missing column 'extra' of type Int64
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

## Development

```bash
git clone https://github.com/ngr-t/polypolarism.git
cd polypolarism

uv sync --dev
uv run pytest
uv run pytest --cov
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
