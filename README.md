# polypolarism

Static type checker for Polars DataFrames based on row polymorphism.

## Features

- **Static type checking** for Polars DataFrame operations without running your code
- **Pandera schema declaration** with `pa.DataFrameModel` and `DataFrame[Schema]` annotations
- **Validation-as-narrowing**: `Schema.validate(df)`, `df.pipe(Schema.validate)`, and `Schema.validate(lf).collect()` all narrow the static type downstream
- **Broad operation support**: join, group_by/agg, select, with_columns, filter, reshape (explode, unpivot, concat, …), `.str`/`.dt`/`.list`/`.arr`/`.struct`/`.cat` sub-namespaces, window/time-series, and selectors — see [docs/operations.md](docs/operations.md)
- **Error detection**: missing columns, join-key type mismatches, invalid aggregations, declared-vs-inferred return types, strict-mode extra columns, and `Optional[T]` (column may be absent) vs `pa.Field(nullable=True)` (value may be null)

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

### Catching a type error statically

This multi-step pipeline — join → derive column → group-by → aggregate —
shows an error the checker catches without running the code:

```python
class Sales(pa.DataFrameModel):
    order_id: int
    region: str
    product_id: int
    quantity: int


class Products(pa.DataFrameModel):
    product_id: int
    unit_price: pl.Float64


class RevenueByRegion(pa.DataFrameModel):
    region: str
    total_revenue: int  # declared Int64, but sum(Float64) produces Float64


def compute_revenue(
    sales: DataFrame[Sales],
    products: DataFrame[Products],
) -> DataFrame[RevenueByRegion]:
    return (
        sales
        .join(products, on="product_id", how="inner")
        .with_columns(revenue=pl.col("quantity") * pl.col("unit_price"))
        .group_by("region")
        .agg(total_revenue=pl.col("revenue").sum())
    )
```

`polypolarism pipeline.py` reports:

```
  compute_revenue: FAIL
    - [PLY040] Column 'total_revenue' has type Float64, but declared type is Int64

1 function(s) failed, 0 passed.
```

The checker tracks the dtype of `revenue` through `quantity × unit_price`
(Int64 × Float64 → Float64) and catches the mismatch against the `Int64`
declaration — before any data reaches runtime.

## CLI Usage

```bash
# Check a single file or a directory
polypolarism path/to/file.py
polypolarism path/to/project/

# JSON output (e.g. for editor integrations)
polypolarism --format json path/to/file.py

# Other flags
polypolarism --version
polypolarism --no-color path/to/file.py
polypolarism --polars-version 1.40 path/to/file.py   # suppresses PLW010
polypolarism --no-version-check path/to/file.py
```

Example output for valid code:

```
  merge_users_orders: OK
  aggregate_sales: OK

All 2 function(s) passed.
```

For invalid code:

```
  bad_join: FAIL
    - [PLY010] Column 'user_id' not found in right frame
    - [PLY040] Could not infer return type

1 function(s) failed, 0 passed.
```

## Documentation

- [Schema declaration](docs/schemas.md) — every Pandera field-annotation
  form, `Optional[T]` vs `pa.Field(nullable=True)`, validation-as-narrowing.
- [Supported operations](docs/operations.md) — the full operation and
  sub-namespace reference.
- [Diagnostics](docs/diagnostics.md) — `PLY###` / `PLW###` codes, JSON
  output, and the schema diff block.
- [Leniency](docs/leniency.md) — when a clean run is *not* a proof.
- [Supported versions](docs/versions.md) — Polars `1.37+` / Pandera
  `0.19+`, lockfile detection, and `[tool.polypolarism]` config.
- [Design records](docs/) — ADRs and backlog.

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
