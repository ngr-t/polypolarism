"""Function call combined with Polars method chain."""
import polars as pl
from polypolarism import DF


def normalize(df: DF["{id: Int64, value: Float64}"]) -> DF["{id: Int64, norm: Float64}"]:
    """Normalize value column."""
    return df.select(
        pl.col("id"),
        (pl.col("value") / 100.0).alias("norm"),
    )


def process_and_filter(
    data: DF["{id: Int64, value: Float64}"]
) -> DF["{id: Int64, norm: Float64}"]:
    """Continue Polars method chain after function call."""
    normalized = normalize(data)
    # Further operations on function return value
    return normalized.select(pl.col("id"), pl.col("norm"))
