"""Smoke fixture: Pandera DataFrameModel form is recognised."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InputSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class OutputSchema(pa.DataFrameModel):
    id: int
    doubled: pl.Float64


def double_value(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    return df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )
