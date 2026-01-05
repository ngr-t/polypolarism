"""Valid test case: Basic group_by with aggregation."""

import polars as pl

from polypolarism import DF


def sales_by_country(
    sales: DF["{country: Utf8, product: Utf8, amount: Float64}"],
) -> DF["{country: Utf8, total_amount: Float64, count: UInt32}"]:
    """Group by country and aggregate sales."""
    return sales.group_by("country").agg(
        pl.col("amount").sum().alias("total_amount"),
        pl.col("amount").count().alias("count"),
    )
