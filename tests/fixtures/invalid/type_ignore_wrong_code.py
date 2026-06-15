"""# type: ignore with a non-matching code does not suppress the diagnostic."""

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


def wrong_code(df: DataFrame[KV]) -> DataFrame[KVa]:  # type: ignore[PLW006]
    return df.select(pl.col("k"))
