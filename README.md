# polypolarism

Static type checker for Polars DataFrames based on row polymorphism.

## Features

- **Static type checking** for Polars DataFrame operations without running your code
- **Pandera schema declaration** with `pa.DataFrameModel` and `DataFrame[Schema]` annotations
- **Validation-as-narrowing**: `Schema.validate(df)`, `df.pipe(Schema.validate)`, and `Schema.validate(lf).collect()` all narrow the static type downstream
- **Operation support**: join, group_by, select, with_columns
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
`Annotated[pl.List, pl.Int64()]` / `Annotated[pl.Struct, {...}]` for
nested types.

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
```

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
`n_unique`, `first`, `last`, `list`.

### Select / With Columns

`select` and `with_columns` are recognised; column dtypes flow through
literal expressions, `pl.col(...)` references, arithmetic, and
`.alias(...)`.

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
