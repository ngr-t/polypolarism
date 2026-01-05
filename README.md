# polypolarism

Static type checker for Polars DataFrames based on row polymorphism.

## Features

- **Static type checking** for Polars DataFrame operations without running your code
- **Schema annotation DSL**: `DF["{col: Type, ...}"]` for type hints
- **Operation support**: join, group_by, select, with_columns
- **Error detection**:
  - Missing columns
  - Type mismatches in join keys
  - Invalid aggregation function applications
  - Declared vs inferred return type differences

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

Annotate your DataFrame functions with the `DF` type:

```python
from polypolarism import DF

def merge_users_orders(
    users: DF["{user_id: Int64, name: Utf8}"],
    orders: DF["{order_id: Int64, user_id: Int64, amount: Float64}"],
) -> DF["{user_id: Int64, name: Utf8, order_id: Int64, amount: Float64}"]:
    return users.join(orders, on="user_id", how="inner")
```

Then run the type checker:

```bash
polypolarism your_module.py
```

## Schema DSL

The schema DSL uses a simple syntax to describe DataFrame types:

```python
# Basic types
DF["{id: Int64, name: Utf8, value: Float64}"]

# Nullable types (append ?)
DF["{id: Int64, name: Utf8?}"]

# List types
DF["{tags: List[Utf8]}"]

# Struct types
DF["{address: Struct{city: Utf8, zip: Int64}}"]
```

### Supported Types

| Type | Description |
|------|-------------|
| `Int64`, `Int32` | Integer types |
| `UInt64`, `UInt32` | Unsigned integer types |
| `Float64`, `Float32` | Floating point types |
| `Utf8` | String type |
| `Boolean` | Boolean type |
| `Date`, `Datetime` | Temporal types |
| `List[T]` | List of type T |
| `Struct{...}` | Struct with named fields |

## CLI Usage

```bash
# Check a single file
polypolarism path/to/file.py

# Check a directory
polypolarism path/to/project/

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

```python
def merge(
    left: DF["{id: Int64, value: Utf8}"],
    right: DF["{id: Int64, score: Float64}"],
) -> DF["{id: Int64, value: Utf8, score: Float64}"]:
    return left.join(right, on="id", how="inner")
```

- Supported join types: `inner`, `left`, `right`, `full`
- Left join makes right columns nullable
- Right join makes left columns nullable
- Full join makes both sides nullable

### Group By + Aggregation

```python
import polars as pl

def summarize(
    data: DF["{category: Utf8, amount: Float64}"],
) -> DF["{category: Utf8, total: Float64, count: UInt32}"]:
    return data.group_by("category").agg(
        pl.col("amount").sum().alias("total"),
        pl.col("amount").count().alias("count"),
    )
```

Supported aggregation functions:
- `sum`, `mean`, `min`, `max`
- `count`, `n_unique`
- `first`, `last`
- `list`

### Select

```python
def select_cols(
    data: DF["{id: Int64, name: Utf8, value: Float64}"],
) -> DF["{id: Int64, value: Float64}"]:
    return data.select(pl.col("id"), pl.col("value"))
```

### With Columns

```python
def add_doubled(
    data: DF["{id: Int64, value: Float64}"],
) -> DF["{id: Int64, value: Float64, doubled: Float64}"]:
    return data.with_columns(
        (pl.col("value") * 2).alias("doubled"),
    )
```

## Development

```bash
# Clone the repository
git clone https://github.com/ngr-t/polypolarism.git
cd polypolarism

# Install with uv
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
