"""Valid: arithmetic dtype rules (issue #14).

True division ``/`` always yields Float64 — even int / int. Floor
division ``//`` keeps the integer dtype. Mixed int/float arithmetic
promotes to Float64.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Pair(pa.DataFrameModel):
    a: int
    b: int


class Ratios(pa.DataFrameModel):
    ratio: float  # int / int -> Float64
    quotient: int  # int // int -> Int64
    scaled: float  # int + float literal -> Float64


def divide(df: DataFrame[Pair]) -> DataFrame[Ratios]:
    return df.select(
        ratio=pl.col("a") / pl.col("b"),
        quotient=pl.col("a") // pl.col("b"),
        scaled=pl.col("a") + 0.5,
    )
