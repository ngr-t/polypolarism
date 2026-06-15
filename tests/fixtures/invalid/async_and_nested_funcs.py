"""async def and def nested in compound statements are discovered (issue #99)."""

from __future__ import annotations

import contextlib

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True


async def bug_async(df: DataFrame[KV]) -> DataFrame[KVa]:
    return df.select(pl.col("k"))


if True:

    def bug_in_if(df: DataFrame[KV]) -> DataFrame[KVa]:
        return df.select(pl.col("k"))


with contextlib.suppress(Exception):

    def bug_in_with(df: DataFrame[KV]) -> DataFrame[KVa]:
        return df.select(pl.col("k"))


def _factory() -> object:
    def bug_closure(df: DataFrame[KV]) -> DataFrame[KVa]:
        return df.select(pl.col("k"))

    return bug_closure
