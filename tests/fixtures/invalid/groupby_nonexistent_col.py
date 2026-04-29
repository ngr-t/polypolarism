"""Invalid test case: Group by non-existent column."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class SalesSchema(pa.DataFrameModel):
    product: str
    amount: pl.Float64


class CategoryTotalSchema(pa.DataFrameModel):
    category: str
    total: pl.Float64


def bad_groupby(sales: DataFrame[SalesSchema]) -> DataFrame[CategoryTotalSchema]:
    """ERROR: 'category' column does not exist in sales DataFrame."""
    return sales.group_by("category").agg(
        pl.col("amount").sum().alias("total"),
    )
