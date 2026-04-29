"""Pandera fixture: declared schema differs from inferred output."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class OutSchema(pa.DataFrameModel):
    id: int
    doubled: pl.Int64  # wrong: actual is Float64


def double_value(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )
