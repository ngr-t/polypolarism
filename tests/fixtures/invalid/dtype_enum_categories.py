"""Invalid fixture: Enum category sequences are dtype identity (issue #67).

polars treats Enums with different category sequences as distinct dtypes
and pandera validation rejects both a different *set* and a reordered
one (probed 1.41.2: ``pl.Enum(["a","b"]) != pl.Enum(["b","a"])``), so
the comparison is exact and order-sensitive. Both were static false
negatives while ``Enum`` carried no categories.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SrcAC(pa.DataFrameModel):
    e: pl.Enum(["a", "c"])

    class Config:
        strict = True


class SrcBA(pa.DataFrameModel):
    e: pl.Enum(["b", "a"])

    class Config:
        strict = True


class EnumAB(pa.DataFrameModel):
    e: pl.Enum(["a", "b"])

    class Config:
        strict = True


def enum_category_set_mismatch(df: DataFrame[SrcAC]) -> DataFrame[EnumAB]:
    return df.select(pl.col("e"))


def enum_category_order_mismatch(df: DataFrame[SrcBA]) -> DataFrame[EnumAB]:
    return df.select(pl.col("e"))
