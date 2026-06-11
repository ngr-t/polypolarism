"""``~UInt8`` declared Int64 is a dtype lie (issue #72 boundary).

False-positive twin: ``valid/not_bitwise_unsigned_width``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Bytes(pa.DataFrameModel):
    u: pl.UInt8

    class Config:
        strict = True
        coerce = True


class WrongOut(pa.DataFrameModel):
    x: pl.Int64  # WRONG: ~UInt8 stays UInt8

    class Config:
        strict = True


def invert_declared_i64(df: DataFrame[Bytes]) -> DataFrame[WrongOut]:
    return df.select(x=~pl.col("u"))
