"""Invalid test case: cast targets a column that does not exist (pple-column-not-found)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InputSchema(pa.DataFrameModel):
    id: int
    amount: pl.Float64


class OutputSchema(pa.DataFrameModel):
    id: int
    amount: pl.Float64


def cast_missing_column(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    """ERROR: 'price' does not exist in InputSchema."""
    return df.cast({"price": pl.Float64})
