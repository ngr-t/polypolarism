"""M1: drop_nulls strips Nullable; with_row_index adds UInt32 column."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int = pa.Field(nullable=True)
    value: pl.Float64


class Out(pa.DataFrameModel):
    index: pl.UInt32
    id: int
    value: pl.Float64


def assign_index(df: DataFrame[In]) -> DataFrame[Out]:
    return df.drop_nulls(subset=["id"]).with_row_index()
