"""Untyped function with transformation: body analysis infers return type."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class IdSchema(pa.DataFrameModel):
    id: int


class IdNewSchema(pa.DataFrameModel):
    id: int
    new_col: int


def untyped_add_column(df):
    """No type annotation, add column and return."""
    return df.with_columns(pl.lit(100).alias("new_col"))


def caller(data: DataFrame[IdSchema]) -> DataFrame[IdNewSchema]:
    """Transformed type is inferred via body analysis even through untyped function."""
    return untyped_add_column(data)
