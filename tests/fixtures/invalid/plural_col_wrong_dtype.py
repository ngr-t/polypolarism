"""Plural ``pl.col("a", "b")`` keeps each column's own dtype.

False-negative twin of ``valid/m9_plural_col.py`` and
``valid/plural_col_exprs.py``: plain selection, nested arithmetic, and the
aggregation form, each declaring the Float64 column 'b' as Int64.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    s: str
    a: int
    b: pl.Float64


class PickedWrong(pa.DataFrameModel):
    a: int
    b: pl.Int64  # WRONG: pl.col("a", "b") keeps 'b' at Float64


def pick(df: DataFrame[S]) -> DataFrame[PickedWrong]:
    return df.select(pl.col("a", "b"))


class ScaledWrong(pa.DataFrameModel):
    a: int
    b: pl.Int64  # WRONG: Float64 * 10 stays Float64 for 'b'


def scale(df: DataFrame[S]) -> DataFrame[ScaledWrong]:
    return df.select(pl.col("a", "b") * 10)


class SummedWrong(pa.DataFrameModel):
    s: str
    a: int
    b: pl.Int64  # WRONG: sum(Float64) is Float64 for 'b'


def totals(df: DataFrame[S]) -> DataFrame[SummedWrong]:
    return df.group_by("s").agg(pl.col("a", "b").sum())
