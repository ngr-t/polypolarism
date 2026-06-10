"""Plural ``pl.col("a", "b")`` nested inside expressions (issue #42).

polars expands the surrounding expression once per column, keeping each
column's name: ``select(pl.col("a", "b") * 10)`` yields columns ``a`` and
``b``; ``agg(pl.col("a", "b").sum())`` aggregates both columns.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    s: str
    a: int
    b: pl.Float64


class ScaledAB(pa.DataFrameModel):
    a: int
    b: pl.Float64

    class Config:
        strict = True


class SummedAB(pa.DataFrameModel):
    s: str
    a: int
    b: pl.Float64

    class Config:
        strict = True


def scale(df: DataFrame[In]) -> DataFrame[ScaledAB]:
    return df.select(pl.col("a", "b") * 10)


def totals(df: DataFrame[In]) -> DataFrame[SummedAB]:
    return df.group_by("s").agg(pl.col("a", "b").sum())
