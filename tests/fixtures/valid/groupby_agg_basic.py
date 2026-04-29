"""Valid test case: Basic group_by with aggregation."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SalesSchema(pa.DataFrameModel):
    country: str
    product: str
    amount: pl.Float64


class SummarySchema(pa.DataFrameModel):
    country: str
    total_amount: pl.Float64
    count: pl.UInt32


def sales_by_country(sales: DataFrame[SalesSchema]) -> DataFrame[SummarySchema]:
    """Group by country and aggregate sales."""
    return sales.group_by("country").agg(
        pl.col("amount").sum().alias("total_amount"),
        pl.col("amount").count().alias("count"),
    )
