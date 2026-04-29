"""Pandera strict=True schema with exact column match passes."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class StrictOut(pa.DataFrameModel):
    id: int
    name: pl.Utf8

    class Config:
        strict = True


def select_exact(df: DataFrame[StrictOut]) -> DataFrame[StrictOut]:
    return df.select(pl.col("id"), pl.col("name"))
