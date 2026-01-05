"""Valid test case: Multiple aggregation functions."""

import polars as pl

from polypolarism import DF


def product_stats(
    orders: DF["{product_id: Int64, quantity: Int64, price: Float64}"],
) -> DF["{product_id: Int64, total_qty: Int64, avg_price: Float64, order_count: UInt32}"]:
    """Group by product and compute multiple statistics."""
    return orders.group_by("product_id").agg(
        pl.col("quantity").sum().alias("total_qty"),
        pl.col("price").mean().alias("avg_price"),
        pl.col("price").count().alias("order_count"),
    )
