"""Validation narrowing on a LazyFrame: Schema.validate(lf).collect()."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import LazyFrame, DataFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


def process(raw: LazyFrame[RawSchema]) -> DataFrame[CleanSchema]:
    df = CleanSchema.validate(raw).collect()
    return df.select(pl.col("id"), pl.col("value"))
