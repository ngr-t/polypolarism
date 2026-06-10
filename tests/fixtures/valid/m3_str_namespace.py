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
    price: pl.Decimal
    opens_at: pl.Time


def normalize(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        pl.col("name").str.starts_with("admin_").alias("is_admin"),
        pl.col("name").str.to_uppercase().alias("name_upper"),
        pl.col("name").str.len_chars().alias("name_len"),
    )


def parse(df: DataFrame[Raw]) -> DataFrame[Parsed]:
    """Issue #19: str parse helpers infer precise dtypes."""
    return df.select(
        pl.col("qty").str.to_integer(base=10).alias("qty"),
        pl.col("price").str.to_decimal().alias("price"),
        pl.col("opens_at").str.to_time().alias("opens_at"),
    )
