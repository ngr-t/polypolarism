"""Invalid: true division of integers declared as int (issue #14).

``pl.col("a") / pl.col("b")`` yields Float64 in polars even when both
operands are integers. With ``coerce`` off (the default), declaring the
result as int is a dtype mismatch.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Pair(pa.DataFrameModel):
    a: int
    b: int


class WrongRatio(pa.DataFrameModel):
    r: int  # wrong: int / int is Float64


def divide(df: DataFrame[Pair]) -> DataFrame[WrongRatio]:
    return df.select(r=pl.col("a") / pl.col("b"))
