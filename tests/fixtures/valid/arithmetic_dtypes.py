"""Valid: arithmetic dtype and nullability rules (issues #14 / #18).

True division ``/`` always yields Float64 — even int / int. Floor
division ``//`` keeps the integer dtype. Mixed int/float arithmetic
promotes to Float64. Elementwise ops with a Nullable operand produce a
Nullable result, satisfied by ``pa.Field(nullable=True)`` declarations.
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


class WithNullable(pa.DataFrameModel):
    a: int
    x: int = pa.Field(nullable=True)


class NullablePropagated(pa.DataFrameModel):
    z: int = pa.Field(nullable=True)  # a + nullable x -> nullable
    q: float = pa.Field(nullable=True)  # a / nullable x -> nullable Float64
    pos: bool = pa.Field(nullable=True)  # nullable x > 0 -> nullable Boolean


def combine(df: DataFrame[WithNullable]) -> DataFrame[NullablePropagated]:
    return df.select(
        z=pl.col("a") + pl.col("x"),
        q=pl.col("a") / pl.col("x"),
        pos=pl.col("x") > 0,
    )
