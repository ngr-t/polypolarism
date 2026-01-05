"""Invalid test case: pl.col references non-existent column."""

import polars as pl

from polypolarism import DF


def reference_missing_column(
    data: DF["{id: Int64, value: Float64}"],
) -> DF["{id: Int64, doubled: Float64}"]:
    """ERROR: 'amount' column does not exist, should be 'value'."""
    return data.select(
        pl.col("id"),
        (pl.col("amount") * 2).alias("doubled"),
    )
