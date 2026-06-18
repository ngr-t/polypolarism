"""# type: ignore suppresses diagnostics on the def line."""

from __future__ import annotations

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


def blanket(df: DataFrame[KV]) -> DataFrame[KVa]:  # type: ignore
    return df.select(pl.col("k"))


def specific(df: DataFrame[KV]) -> DataFrame[KVa]:  # type: ignore[pple-return-type]
    return df.select(pl.col("k"))


def multi_code(df: DataFrame[KV]) -> DataFrame[KVa]:  # type: ignore[pple-return-type, pple-eager-lazy-mismatch]
    return df.select(pl.col("k"))
