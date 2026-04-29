"""Strict declared schema rejects extra columns from with_columns."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int


class StrictOut(pa.DataFrameModel):
    id: int

    class Config:
        strict = True


def add_extra(df: DataFrame[InSchema]) -> DataFrame[StrictOut]:
    # Adds 'doubled', which violates strict=True on the return type.
    return df.with_columns((pl.col("id") * 2).alias("doubled"))
