"""Chained function calls with type propagation."""
import polars as pl
from polypolarism import DF


def add_b(df: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64}"]:
    """Step 1: b 列を追加."""
    return df.with_columns(pl.lit(100).alias("b"))


def add_c(df: DF["{a: Int64, b: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
    """Step 2: c 列を追加."""
    return df.with_columns((pl.col("a") + pl.col("b")).alias("c"))


def pipeline(data: DF["{a: Int64}"]) -> DF["{a: Int64, b: Int64, c: Int64}"]:
    """Pipeline: add_b -> add_c の連鎖."""
    temp = add_b(data)
    result = add_c(temp)
    return result
