"""Invalid test case: Group by non-existent column."""

import polars as pl

from polypolarism import DF


def bad_groupby(
    sales: DF["{product: Utf8, amount: Float64}"],
) -> DF["{category: Utf8, total: Float64}"]:
    """ERROR: 'category' column does not exist in sales DataFrame."""
    return sales.group_by("category").agg(
        pl.col("amount").sum().alias("total"),
    )
