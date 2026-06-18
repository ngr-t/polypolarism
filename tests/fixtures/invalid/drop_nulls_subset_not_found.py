"""Invalid test case: drop_nulls subset references a missing column (pple-column-not-found)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InputSchema(pa.DataFrameModel):
    id: int
    amount: pl.Float64 = pa.Field(nullable=True)


class OutputSchema(pa.DataFrameModel):
    id: int
    amount: pl.Float64 = pa.Field(nullable=True)


def drop_nulls_missing_subset(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    """ERROR: 'price' does not exist in InputSchema."""
    return df.drop_nulls(subset="price")
