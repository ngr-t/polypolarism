"""M7: df.pipe(typed_helper) carries the helper's declared return type."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64


class Out(pa.DataFrameModel):
    id: int
    value: pl.Float64
    doubled: pl.Float64


def double_value(df: DataFrame[S]) -> DataFrame[Out]:
    return df.with_columns((pl.col("value") * 2).alias("doubled"))


def via_pipe(df: DataFrame[S]) -> DataFrame[Out]:
    return df.pipe(double_value)
