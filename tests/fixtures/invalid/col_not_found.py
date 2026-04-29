"""Invalid test case: pl.col references non-existent column."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class OutSchema(pa.DataFrameModel):
    id: int
    doubled: pl.Float64


def reference_missing_column(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    """ERROR: 'amount' column does not exist, should be 'value'."""
    return data.select(
        pl.col("id"),
        (pl.col("amount") * 2).alias("doubled"),
    )
