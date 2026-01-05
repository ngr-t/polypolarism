"""Forward reference: call function defined later."""
import polars as pl
from polypolarism import DF


def caller(data: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
    """Caller: call helper (helper is defined later)."""
    return helper(data)


def helper(df: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
    """Helper: add column y."""
    return df.with_columns((pl.col("x") * 2).alias("y"))
