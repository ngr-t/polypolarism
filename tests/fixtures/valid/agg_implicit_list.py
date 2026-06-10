"""Valid: implicit list aggregation (issue #27).

A bare column reference inside ``agg`` (no reducing function) collects
each group's values into a list: ``group_by("k").agg(vs=pl.col("v"))``
has runtime schema ``{'k': String, 'vs': List(Int64)}``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Raw(pa.DataFrameModel):
    k: str
    v: int


class Listed(pa.DataFrameModel):
    k: str
    vs: pl.List(pl.Int64) = pa.Field()

    class Config:
        strict = True


@pa.check_types
def agg_to_list(df: DataFrame[Raw]) -> DataFrame[Listed]:
    return df.group_by("k").agg(vs=pl.col("v"))


class ListedDefaultName(pa.DataFrameModel):
    k: str
    v: pl.List(pl.Int64) = pa.Field()


@pa.check_types
def agg_positional(df: DataFrame[Raw]) -> DataFrame[ListedDefaultName]:
    return df.group_by("k").agg(pl.col("v"))


@pa.check_types
def agg_bare_string(df: DataFrame[Raw]) -> DataFrame[ListedDefaultName]:
    return df.group_by("k").agg("v")
