"""Forward reference: call function defined later."""
import polars as pl
from polypolarism import DF


def caller(data: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
    """Caller: helper を呼び出す（helper は後で定義）."""
    return helper(data)


def helper(df: DF["{x: Int64}"]) -> DF["{x: Int64, y: Int64}"]:
    """Helper: y 列を追加."""
    return df.with_columns((pl.col("x") * 2).alias("y"))
