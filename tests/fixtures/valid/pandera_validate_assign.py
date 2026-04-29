"""Validation narrowing via assignment: df2 = Schema.validate(df)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


def process(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
    # Assigning the result of validate() narrows df to CleanSchema.
    df = CleanSchema.validate(raw)
    return df.select(pl.col("id"), pl.col("value"))
