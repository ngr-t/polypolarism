"""Untyped function with transformation: body analysis infers return type."""
import polars as pl
from polypolarism import DF


def untyped_add_column(df):
    """No type annotation, add column and return."""
    return df.with_columns(pl.lit(100).alias("new_col"))


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, new_col: Int64}"]:
    """Transformed type is inferred via body analysis even through untyped function."""
    return untyped_add_column(data)
