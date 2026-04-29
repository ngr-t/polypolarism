"""M3: pl.col(...).str methods are dispatched to the correct return type."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    name: str


class Out(pa.DataFrameModel):
    is_admin: bool
    name_upper: str
    name_len: pl.UInt32


def normalize(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        pl.col("name").str.starts_with("admin_").alias("is_admin"),
        pl.col("name").str.to_uppercase().alias("name_upper"),
        pl.col("name").str.len_chars().alias("name_len"),
    )
