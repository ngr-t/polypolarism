"""Valid test case: Chained join and groupby operations."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class OrdersSchema(pa.DataFrameModel):
    order_id: int
    customer_id: int
    amount: pl.Float64


class CustomersSchema(pa.DataFrameModel):
    customer_id: int
    region: str


class SalesSummarySchema(pa.DataFrameModel):
    region: str
    total_sales: pl.Float64
    order_count: pl.UInt32


def sales_summary_by_region(
    orders: DataFrame[OrdersSchema],
    customers: DataFrame[CustomersSchema],
) -> DataFrame[SalesSummarySchema]:
    """Join orders with customers, then group by region."""
    return (
        orders.join(customers, on="customer_id", how="inner")
        .group_by("region")
        .agg(
            pl.col("amount").sum().alias("total_sales"),
            pl.col("order_id").count().alias("order_count"),
        )
    )
