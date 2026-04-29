"""Chained function calls with type propagation."""

import polars as pl
import pandera.polars as pa
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
    """Step 1: add column b."""
    return df.with_columns(pl.lit(100).alias("b"))


def add_c(df: DataFrame[AB]) -> DataFrame[ABC]:
    """Step 2: add column c."""
    return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))


def pipeline(data: DataFrame[A]) -> DataFrame[ABC]:
    """Pipeline: chain add_b -> add_c."""
    temp = add_b(data)
    result = add_c(temp)
    return result
