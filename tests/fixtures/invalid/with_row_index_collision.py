"""Invalid test case: with_row_index name collides with an existing column (pple-column-name-collision)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InputSchema(pa.DataFrameModel):
    index: int
    amount: pl.Float64


class OutputSchema(pa.DataFrameModel):
    index: int
    amount: pl.Float64


def add_colliding_row_index(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    """ERROR: 'index' already exists in InputSchema."""
    return df.with_row_index("index")
