"""Invalid test case: Aggregation function type error."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    category: str
    label: str


class OutSchema(pa.DataFrameModel):
    category: str
    label_sum: str


def sum_on_string(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    """ERROR: Cannot apply sum() to Utf8 column."""
    return data.group_by("category").agg(
        pl.col("label").sum().alias("label_sum"),
    )
