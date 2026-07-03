"""Valid: strict=False adds no nulls to an always-succeeding cast (issue #125).

``strict=False`` only injects nulls for *value-dependent* casts. A cast that
always succeeds (a widening numeric cast, ``Int64 -> Float64``) stays non-null,
and the default ``strict=True`` keeps the receiver's own nullability regardless
of the verdict — so these must still type-check against non-null declarations.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    s: str
    small: pl.Int8
    u: pl.UInt8


class Out(pa.DataFrameModel):
    wide: pl.Float64  # Int64 -> Float64 always succeeds, non-null under strict=False
    strict_default: int  # strict=True (default) keeps receiver non-null
    widen_int: pl.Int64  # Int8 -> Int64 (same sign, wider) never overflows
    unsigned_wider: pl.Int16  # UInt8 -> Int16 (strictly wider signed) holds every value


def casts(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        wide=pl.col("a").cast(pl.Float64, strict=False),
        strict_default=pl.col("s").cast(pl.Int64),
        widen_int=pl.col("small").cast(pl.Int64, strict=False),
        unsigned_wider=pl.col("u").cast(pl.Int16, strict=False),
    )
