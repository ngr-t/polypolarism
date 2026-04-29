"""Validation narrowing via df.pipe(Schema.validate)."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


def process(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
    return raw.pipe(CleanSchema.validate).select(pl.col("id"), pl.col("value"))
