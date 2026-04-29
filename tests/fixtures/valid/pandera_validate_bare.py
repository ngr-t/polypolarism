"""Validation narrowing via bare statement: Schema.validate(df) then later use df."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


def process(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
    # Bare validate() call narrows raw's type for subsequent statements.
    CleanSchema.validate(raw)
    return raw.select(pl.col("id"), pl.col("value"))
