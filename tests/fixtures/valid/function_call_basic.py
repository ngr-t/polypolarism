"""Basic function call type inference."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class OutSchema(pa.DataFrameModel):
    id: int
    doubled: pl.Float64


def double_value(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    """Transform: double the value and return as 'doubled'."""
    return df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )


def process_data(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    """Pipeline: call double_value."""
    return double_value(data)
