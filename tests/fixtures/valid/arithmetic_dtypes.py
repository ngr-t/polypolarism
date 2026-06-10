"""Valid: arithmetic dtype and nullability rules (issues #14 / #18 / #30).

True division ``/`` always yields Float64 — even int / int. Floor
division ``//`` keeps the integer dtype. Mixed int/float arithmetic
promotes to Float64. Elementwise ops with a Nullable operand produce a
Nullable result, satisfied by ``pa.Field(nullable=True)`` declarations.
String concat (``Utf8 + Utf8``) and temporal arithmetic (date - date,
date + duration, duration * int, duration / int) are allowed binary
operations and must not trip the PLY009 incompatible-dtype check.
"""

from datetime import date, timedelta

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


class Events(pa.DataFrameModel):
    name: str
    suffix: str
    start: date
    end: date
    gap: timedelta


class Derived(pa.DataFrameModel):
    label: str  # str + str -> Utf8 (concat)
    span: timedelta  # date - date -> Duration
    shifted: date  # date + duration -> Date
    doubled: timedelta  # duration * int -> Duration
    halved: pl.Duration  # duration / int -> Duration


def derive(df: DataFrame[Events]) -> DataFrame[Derived]:
    return df.select(
        label=pl.col("name") + pl.col("suffix"),
        span=pl.col("end") - pl.col("start"),
        shifted=pl.col("start") + pl.col("gap"),
        doubled=pl.col("gap") * 2,
        halved=pl.col("gap") / 2,
    )
