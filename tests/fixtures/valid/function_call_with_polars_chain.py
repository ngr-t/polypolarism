"""Function call combined with Polars method chain."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class NormSchema(pa.DataFrameModel):
    id: int
    norm: pl.Float64


def normalize(df: DataFrame[InSchema]) -> DataFrame[NormSchema]:
    """Normalize value column."""
    return df.select(
        pl.col("id"),
        (pl.col("value") / 100.0).alias("norm"),
    )


def process_and_filter(data: DataFrame[InSchema]) -> DataFrame[NormSchema]:
    """Continue Polars method chain after function call."""
    normalized = normalize(data)
    return normalized.select(pl.col("id"), pl.col("norm"))
