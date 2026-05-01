"""pl.col(...).fill_null(value) strips Nullable from the receiver."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    value: pl.Float64 = pa.Field(nullable=True)


class Out(pa.DataFrameModel):
    id: int
    value: pl.Float64


def fill(df: DataFrame[In]) -> DataFrame[Out]:
    return df.with_columns(pl.col("value").fill_null(0.0).alias("value"))
