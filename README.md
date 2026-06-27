# polypolarism

Static type checker for Polars DataFrames, inspired by row polymorphism (open-record structural subtyping).

## Features

- **Static type checking** for Polars DataFrame operations without running your code
- **Pandera schema declaration** with `pa.DataFrameModel` and `DataFrame[Schema]` annotations
- **Patito schema declaration** (pydantic-for-polars) with `patito.Model` and `pt.DataFrame[Model]` annotations — see [Patito models](#patito-models)
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

It lists each annotated function with its source line, then a summary:

```
your_module.py
  merge_users_orders (line 24): OK

All 1 function(s) passed.
```

### Catching a type error statically

This multi-step pipeline — join → derive column → group-by → aggregate —
shows an error the checker catches without running the code:

```python
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


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
pipeline.py
  compute_revenue (line 23): FAIL
    - [pple-return-type] Column 'total_revenue' has type Float64, but declared type is Int64

1 function(s) failed, 0 passed.
```

The checker tracks the dtype of `revenue` through `quantity × unit_price`
(Int64 × Float64 → Float64) and catches the mismatch against the `Int64`
declaration — before any data reaches runtime.

## Patito models

[Patito](https://github.com/JakobGM/patito) (a pydantic-for-polars layer) is
supported as a second schema frontend alongside Pandera. Subclass
`patito.Model` and annotate with `pt.DataFrame[Model]` / `pt.LazyFrame[Model]`;
`Model.validate(df)` narrows downstream just like Pandera's `validate`:

```python
import patito as pt
import polars as pl


class In(pt.Model):
    id: int
    name: str


class Out(pt.Model):
    id: int
    name: str
    score: float


def add_score(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.with_columns(score=pl.col("id") * 1.0)
```

Patito's semantics differ from Pandera's and are modeled faithfully
(see [ADR-0010](docs/adr/0010-patito-frontend.md)):

- `Optional[T]` / `T | None` makes the **value** nullable (the column stays
  required) — the inverse of Pandera's `Optional[T]` (column may be absent).
- `int` / `float` accept **any** integer / float width and `Literal[...]`
  accepts `String` or its `Enum`, so a `UInt32` column satisfies an `int`
  field without a false positive.
- `pt.Field(dtype=pl.UInt16)` forces an exact dtype; a nested `Model` field
  becomes a `Struct`; Patito models are strict (extra columns rejected).

> **Single-dialect assumption**: a file mixing Patito *and* Pandera schemas is
> not a supported configuration.

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
polypolarism --polars-version 1.40 path/to/file.py   # suppresses pplw-unsupported-version
polypolarism --no-version-check path/to/file.py
```

## Documentation

- [Schema declaration](docs/schemas.md) — every Pandera field-annotation
  form, `Optional[T]` vs `pa.Field(nullable=True)`, validation-as-narrowing.
- [Supported operations](docs/operations.md) — the full operation and
  sub-namespace reference.
- [Diagnostics](docs/diagnostics.md) — `pple-*` / `pplw-*` codes, JSON
  output, and the schema diff block.
- [Row polymorphism](docs/row-polymorphism.md) — the opt-in `@rowpoly`
  dialect for column-preserving helpers (threading + `pple-rowpoly-not-preserved`).
- [Patito frontend](docs/adr/0010-patito-frontend.md) — the pydantic-for-polars
  schema dialect, its semantic differences from Pandera, and the single-dialect
  assumption.
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
