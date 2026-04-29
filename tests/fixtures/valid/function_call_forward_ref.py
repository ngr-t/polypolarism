"""Forward reference: call function defined later."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class XSchema(pa.DataFrameModel):
    x: int


class XYSchema(pa.DataFrameModel):
    x: int
    y: int


def caller(data: DataFrame[XSchema]) -> DataFrame[XYSchema]:
    """Caller: call helper (helper is defined later)."""
    return helper(data)


def helper(df: DataFrame[XSchema]) -> DataFrame[XYSchema]:
    """Helper: add column y."""
    return df.with_columns((pl.col("x") * 2).alias("y"))
