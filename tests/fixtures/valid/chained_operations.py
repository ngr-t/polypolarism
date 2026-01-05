"""Valid test case: Chained join and groupby operations."""

import polars as pl

from polypolarism import DF


def sales_summary_by_region(
    orders: DF["{order_id: Int64, customer_id: Int64, amount: Float64}"],
    customers: DF["{customer_id: Int64, region: Utf8}"],
) -> DF["{region: Utf8, total_sales: Float64, order_count: UInt32}"]:
    """Join orders with customers, then group by region."""
    return (
        orders.join(customers, on="customer_id", how="inner")
        .group_by("region")
        .agg(
            pl.col("amount").sum().alias("total_sales"),
            pl.col("order_id").count().alias("order_count"),
        )
    )
