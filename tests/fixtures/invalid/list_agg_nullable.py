"""Invalid: list reductions that null on an empty sub-list, declared non-null (issue #130).

``list.min`` / ``list.max`` / ``list.mean`` / ``list.median`` / ``list.std`` /
``list.var`` all return null for an empty sub-list, and ``List(T)`` always admits
empty sub-lists, so the sound result is unconditionally nullable. Declaring it
non-null is a false negative (pandera rejects the injected null). Aggregation
sibling of #128 — no keyword involved, so it is stronger. ``list.sum`` (empty ->
0) and ``list.len`` stay non-null (see the valid counterpart).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    xs: pl.List(pl.Int64)


class XOut(pa.DataFrameModel):
    x: int  # non-null


class FOut(pa.DataFrameModel):
    x: pl.Float64  # non-null


def list_max_nonnull(df: DataFrame[ListIn]) -> DataFrame[XOut]:
    return df.select(x=pl.col("xs").list.max())


def list_min_nonnull(df: DataFrame[ListIn]) -> DataFrame[XOut]:
    return df.select(x=pl.col("xs").list.min())


def list_mean_nonnull(df: DataFrame[ListIn]) -> DataFrame[FOut]:
    return df.select(x=pl.col("xs").list.mean())


def list_median_nonnull(df: DataFrame[ListIn]) -> DataFrame[FOut]:
    return df.select(x=pl.col("xs").list.median())


def list_std_nonnull(df: DataFrame[ListIn]) -> DataFrame[FOut]:
    return df.select(x=pl.col("xs").list.std())
