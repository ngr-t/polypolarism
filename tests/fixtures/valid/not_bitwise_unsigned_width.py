"""Bitwise NOT preserves unsigned widths (issue #72 boundary).

``~UInt8`` stays UInt8 (two's-complement within the width: ``~1 == 254``)
— the dtype-preserving rule must not collapse unsigned receivers to
Int64.

False-negative twin: ``invalid/not_unsigned_declared_i64``.
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


class BytesOut(pa.DataFrameModel):
    x: pl.UInt8

    class Config:
        strict = True


def invert_preserves_uint8(df: DataFrame[Bytes]) -> DataFrame[BytesOut]:
    return df.select(x=~pl.col("u"))
