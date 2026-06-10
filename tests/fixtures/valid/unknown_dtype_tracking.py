"""Valid: un-inferable expression outputs stay registered as columns (issue #8).

Columns produced by expressions polypolarism cannot type (interpolate,
pl.len) carry an Unknown dtype instead of being dropped from the tracked
schema, so later references resolve cleanly. when/then/otherwise used to be
in that bucket but is inferred precisely since #40 —
``when_then_otherwise_column`` now passes because then(1).otherwise(0)
really is the declared Int64.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: int
    ts: pl.Datetime


class WithFlags(pa.DataFrameModel):
    v: int
    ts: pl.Datetime
    a: int
    b: int


def when_then_otherwise_column(df: DataFrame[In]) -> DataFrame[WithFlags]:
    return df.with_columns(a=pl.when(pl.col("v") > 0).then(1).otherwise(0)).with_columns(
        b=pl.col("a") + 1
    )


def interpolate_with_cast(df: DataFrame[In]) -> DataFrame[WithFlags]:
    return df.with_columns(a=pl.col("v").interpolate().cast(pl.Int64)).with_columns(
        b=pl.col("a") + 1
    )


class Monthly(pa.DataFrameModel):
    ym: str
    n: int  # pl.len() yields UInt32; coerce casts it to Int64 at runtime

    class Config:
        coerce = True


def monthly_counts(df: DataFrame[In]) -> DataFrame[Monthly]:
    return df.with_columns(ym=pl.col("ts").dt.strftime("%Y-%m")).group_by("ym").agg(n=pl.len())
