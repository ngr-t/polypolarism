"""Untyped function passthrough: body analysis infers return type."""
import polars as pl
from polypolarism import DF


def untyped_passthrough(df):
    """No type annotation, return DataFrame as-is."""
    return df


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
    """Type is inferred via body analysis even through untyped function."""
    return untyped_passthrough(data)
