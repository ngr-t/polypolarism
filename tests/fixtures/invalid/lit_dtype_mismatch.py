"""pl.lit(value, dtype=T) declared as the wrong type is caught (issue #100)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class K(pa.DataFrameModel):
    k: str

    class Config:
        strict = True


class KaI64(pa.DataFrameModel):
    k: str
    a: pl.Int64

    class Config:
        strict = True


def lit_int32_declared_i64(df: DataFrame[K]) -> DataFrame[KaI64]:
    """pl.lit(5, dtype=Int32) is Int32, but return is declared Int64 — FAIL."""
    return df.with_columns(a=pl.lit(5, dtype=pl.Int32))
