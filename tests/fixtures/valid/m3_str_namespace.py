"""pl.col(...).str methods are dispatched to the correct return type."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    name: str


class Out(pa.DataFrameModel):
    is_admin: bool
    name_upper: str
    name_len: pl.UInt32


class Raw(pa.DataFrameModel):
    qty: str
    price: str
    opens_at: str


class Parsed(pa.DataFrameModel):
    qty: int
    # str.to_decimal requires a kw-only ``scale`` since polars 1.x and yields
    # Decimal(38, scale) — read from the literal argument since issue #61.
    # The parametrized annotations match exactly (pandera reads bare
    # ``pl.Decimal`` as Decimal(28, 0), polars as (38, 0)).
    price: pl.Decimal(38, 0)
    price_cents: pl.Decimal(38, 2)
    opens_at: pl.Time


def normalize(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        pl.col("name").str.starts_with("admin_").alias("is_admin"),
        pl.col("name").str.to_uppercase().alias("name_upper"),
        pl.col("name").str.len_chars().alias("name_len"),
    )


def parse(df: DataFrame[Raw]) -> DataFrame[Parsed]:
    """Issues #19 / #61: str parse helpers infer precise dtypes."""
    return df.select(
        pl.col("qty").str.to_integer(base=10).alias("qty"),
        pl.col("price").str.to_decimal(scale=0).alias("price"),
        pl.col("price").str.to_decimal(scale=2).alias("price_cents"),
        pl.col("opens_at").str.to_time().alias("opens_at"),
    )
