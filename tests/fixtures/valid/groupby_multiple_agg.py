"""Valid test case: Multiple aggregation functions."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class OrdersSchema(pa.DataFrameModel):
    product_id: int
    quantity: int
    price: pl.Float64


class StatsSchema(pa.DataFrameModel):
    product_id: int
    total_qty: int
    avg_price: pl.Float64
    order_count: pl.UInt32


def product_stats(orders: DataFrame[OrdersSchema]) -> DataFrame[StatsSchema]:
    """Group by product and compute multiple statistics."""
    return orders.group_by("product_id").agg(
        pl.col("quantity").sum().alias("total_qty"),
        pl.col("price").mean().alias("avg_price"),
        pl.col("price").count().alias("order_count"),
    )
