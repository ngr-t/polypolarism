"""Chained function calls with type propagation."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class A(pa.DataFrameModel):
    a: int


class AB(pa.DataFrameModel):
    a: int
    b: int


class ABC(pa.DataFrameModel):
    a: int
    b: int
    c: int


def add_b(df: DataFrame[A]) -> DataFrame[AB]:
    """Step 1: add column b.

    ``dtype=pl.Int64`` keeps the literal runtime-faithful to ``b: int``:
    bare ``pl.lit(100)`` materializes as Int32 at runtime (probed on
    polars 1.41.2) while polypolarism models int literals as Int64.
    """
    return df.with_columns(pl.lit(100, dtype=pl.Int64).alias("b"))


def add_c(df: DataFrame[AB]) -> DataFrame[ABC]:
    """Step 2: add column c."""
    return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))


def pipeline(data: DataFrame[A]) -> DataFrame[ABC]:
    """Pipeline: chain add_b -> add_c."""
    temp = add_b(data)
    result = add_c(temp)
    return result
