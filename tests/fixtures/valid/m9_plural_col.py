"""pl.col("a", "b", ...) selects multiple named columns."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    a: int
    b: pl.Float64
    c: str


class Out(pa.DataFrameModel):
    a: int
    b: pl.Float64


def pick(df: DataFrame[S]) -> DataFrame[Out]:
    return df.select(pl.col("a", "b"))
