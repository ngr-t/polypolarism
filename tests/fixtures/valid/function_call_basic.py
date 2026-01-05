"""Basic function call type inference."""
import polars as pl
from polypolarism import DF


def double_value(df: DF["{id: Int64, value: Float64}"]) -> DF["{id: Int64, doubled: Float64}"]:
    """Transform: value を2倍にして doubled として返す."""
    return df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )


def process_data(
    data: DF["{id: Int64, value: Float64}"]
) -> DF["{id: Int64, doubled: Float64}"]:
    """Pipeline: double_value を呼び出す."""
    return double_value(data)
