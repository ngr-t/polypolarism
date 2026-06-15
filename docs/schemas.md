# Schema declaration

Use Pandera class-based schemas (`pa.DataFrameModel`; the legacy
`SchemaModel` name is also accepted). Every supported field-annotation
form:

| Form | Example | Parses to |
|---|---|---|
| Python builtins | `int`, `str`, `float`, `bool`, `bytes` | `Int64`, `Utf8`, `Float64`, `Boolean`, `Binary` |
| stdlib temporal, bare | `date`, `datetime`, `timedelta` (via `from datetime import ...`) | `Date`, `Datetime`, `Duration` |
| stdlib temporal, qualified | `datetime.date`, `dt.datetime`, ... (any non-`pl` prefix) | same as bare |
| polars dtype classes | `pl.Int8`–`pl.Int128`, `pl.UInt8`–`pl.UInt128`, `pl.Float16/32/64`, `pl.String`/`pl.Utf8`, `pl.Boolean`, `pl.Binary`, `pl.Date`, `pl.Datetime`, `pl.Time`, `pl.Duration`, `pl.Categorical`, `pl.Enum`, `pl.Decimal`, `pl.Null` — with or without `()` | the corresponding dtype |
| parametrized `Datetime` | `pl.Datetime("us", "UTC")`, `pl.Datetime(time_zone="UTC")`, `pl.Duration("ms")` | `Datetime[us, UTC]` / `Duration[ms]` — the time unit IS part of dtype identity (issue #66); a non-literal argument degrades to `Unknown` |
| parametrized `Decimal` | `pl.Decimal(20, 4)`, `pl.Decimal(scale=2)` | `Decimal(p, s)`; omitted args take polars' defaults (38, 0) |
| `Enum` with variants | `pl.Enum(["new", "paid"])`, `Annotated[pl.Enum, ["new", "paid"]]` | `Enum['new', 'paid']` — the ordered category tuple is dtype identity (issue #67); a non-literal list is a categories-unknown wildcard |
| `List` call form | `pl.List(pl.Int64)` | `List[Int64]` |
| `Array` call form | `pl.Array(pl.Int64, 3)`, `pl.Array(pl.Int64, shape=3)`, `pl.Array(pl.Int64, (3,))` | `Array[Int64, 3]` — the width is tracked; a multi-dimensional or non-literal `shape` leaves the width a wildcard |
| `Struct` call form | `pl.Struct({"a": pl.Utf8, "b": pl.Float64()})` | `Struct{a: Utf8, b: Float64}` |
| `Annotated` containers | `Annotated[pl.List, pl.Int64()]`, `Annotated[pl.Array, pl.Int64(), 3, None]`, `Annotated[pl.Struct, {...}]` | same as the call forms. pandera requires **exactly all** of the dtype's parameters as metadata (`Array` needs `inner, shape, width` — a `None` literal keeps the polars default); a wrong arity crashes pandera at runtime and is flagged `PLY041` |
| bare containers | `pl.List`, `pl.Array`, `pl.Struct` | `List[Unknown]`, `Array[Unknown]`, `Struct{...}` (an OPEN struct — provably a struct, fields unknown; field lookups are assumed, wrong-namespace accessors are errors) |
| `Series` wrapper | `Series[T]` (bare or qualified: `pa.typing.Series[T]`, ...) | unwraps to `T` |
| optional column | `Optional[T]`, `T \| None` | the **column may be absent** (`required=False`) |
| nullable values | `T = pa.Field(nullable=True)` | `Nullable(T)` — the **values may be null** |

Containers nest arbitrarily (`pl.List(pl.Array(pl.Int8, 4))`,
`pl.Struct({"xs": pl.List(pl.Int64)})`). `Optional[T]` (column may be
missing) and `pa.Field(nullable=True)` (values may be null) are
independent and combine. `List` and `Array` are distinct dtypes — polars
does not interchange them — and the `Array` width is checked: a declared
`pl.Array(pl.Int64, 3)` against an inferred width 5 is an error (pandera
rejects the mismatch and `coerce` cannot repair it).

Frame-level annotations accept `DataFrame[Schema]` and
`LazyFrame[Schema]`, on function parameters and return types as the
checked contract, and on local variables (`x: DataFrame[S] = ...`) where
the annotation is checked against the inferred right-hand side
(ADR-0005: a pure narrowing assertion warns `PLW008`; an unrelated
re-interpretation is a `PLY033` error).

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

## Validation as type narrowing

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
