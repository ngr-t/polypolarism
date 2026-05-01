"""map_elements with return_dtype= is precisely typed."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    value: pl.Float64


class Out(pa.DataFrameModel):
    id: int
    value: pl.Float64
    doubled: pl.Float64


def f(df: DataFrame[In]) -> DataFrame[Out]:
    return df.with_columns(
        pl.col("value").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64).alias("doubled")
    )
