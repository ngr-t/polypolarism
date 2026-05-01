"""pl.col(...).dt methods extract date parts and preserve dtype where appropriate."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    ts: pl.Datetime
    label: str


class Out(pa.DataFrameModel):
    label: str
    year: pl.Int32
    month: pl.Int8
    day_truncated: pl.Datetime


def break_down_date(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        pl.col("label"),
        pl.col("ts").dt.year().alias("year"),
        pl.col("ts").dt.month().alias("month"),
        pl.col("ts").dt.truncate("1d").alias("day_truncated"),
    )
