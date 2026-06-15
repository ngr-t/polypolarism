"""pl.lit(value, dtype=T) is inferred as T, not ignored (issue #100)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class K(pa.DataFrameModel):
    k: str

    class Config:
        strict = True


class KaF(pa.DataFrameModel):
    k: str
    a: float = pa.Field(nullable=True)

    class Config:
        strict = True


class KaI32(pa.DataFrameModel):
    k: str
    a: pl.Int32

    class Config:
        strict = True


def lit_none_typed(df: DataFrame[K]) -> DataFrame[KaF]:
    """pl.lit(None, dtype=Float64) -> Nullable(Float64), not Null."""
    return df.with_columns(a=pl.lit(None, dtype=pl.Float64))


def lit_typed_int32(df: DataFrame[K]) -> DataFrame[KaI32]:
    """pl.lit(5, dtype=Int32) -> Int32, not Int64."""
    return df.with_columns(a=pl.lit(5, dtype=pl.Int32))
