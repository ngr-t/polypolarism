"""partition_by element types flow through subscript and for-loop."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    sales: pl.Float64


class Out(pa.DataFrameModel):
    sales: pl.Float64


def first_partition(df: DataFrame[S]) -> DataFrame[S]:
    parts = df.partition_by("region")
    return parts[0]


def per_partition(df: DataFrame[S]):
    accum: DataFrame[Out] = df.select(pl.col("sales"))
    for part in df.partition_by("region", include_key=False):
        accum = part.select(pl.col("sales"))
    return accum
