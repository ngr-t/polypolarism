"""Chained function calls with type propagation."""
import polars as pl
from polypolarism import DF


def add_b(df: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64}"]:
    """Step 1: add column b."""
    return df.with_columns(pl.lit(100).alias("b"))


def add_c(df: DF["{a: Int64, b: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
    """Step 2: add column c."""
    return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))


def pipeline(data: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
    """Pipeline: chain add_b -> add_c."""
    temp = add_b(data)
    result = add_c(temp)
    return result
