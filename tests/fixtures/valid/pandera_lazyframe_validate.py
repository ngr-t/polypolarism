"""Validation narrowing on a LazyFrame: Schema.validate(lf).collect()."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, LazyFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


def process(raw: LazyFrame[RawSchema]) -> DataFrame[CleanSchema]:
    df = CleanSchema.validate(raw).collect()
    return df.select(pl.col("id"), pl.col("value"))
