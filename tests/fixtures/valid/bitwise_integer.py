"""Valid: integer bitwise &/|/^ return the integer promotion result (issue #137).

Bitwise operators are logical (Boolean) on Boolean operands but real bitwise
arithmetic on integers, returning the promoted integer dtype. Declaring the
integer result must type-check; Boolean operands still yield Boolean.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i64: int
    i8: pl.Int8
    i16: pl.Int16
    b: bool


class Out(pa.DataFrameModel):
    a: int  # i64 & i64 -> Int64
    x: pl.Int16  # i8 ^ i16 -> Int16 (promoted width)
    o: int  # i8 | i64 -> Int64
    flag: bool  # b & b -> Boolean (unchanged)


def bitwise(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        a=pl.col("i64") & pl.col("i64"),
        x=pl.col("i8") ^ pl.col("i16"),
        o=pl.col("i8") | pl.col("i64"),
        flag=pl.col("b") & pl.col("b"),
    )
