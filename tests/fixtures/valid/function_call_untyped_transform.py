"""Untyped function with transformation: body analysis infers return type."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class IdSchema(pa.DataFrameModel):
    id: int


class IdNewSchema(pa.DataFrameModel):
    id: int
    new_col: int


def untyped_add_column(df):
    """No type annotation, add column and return.

    ``dtype=pl.Int64`` keeps the literal runtime-faithful to ``new_col: int``
    (bare int literals are Int32 at runtime, modeled as Int64 statically).
    """
    return df.with_columns(pl.lit(100, dtype=pl.Int64).alias("new_col"))


def caller(data: DataFrame[IdSchema]) -> DataFrame[IdNewSchema]:
    """Transformed type is inferred via body analysis even through untyped function."""
    return untyped_add_column(data)
