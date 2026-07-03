"""Invalid: value-dependent cast(strict=False) declared non-null (issue #125).

``Expr.cast(T, strict=False)`` turns every unconvertible value into null, so a
value-dependent cast (e.g. ``Utf8 -> Int64``) is nullable even from a non-null
receiver. Declaring the result non-null is unsound — pandera (nullable=False,
the default) rejects the injected nulls at validation time. The whole-frame
``DataFrame.cast({...}, strict=False)`` form has the same hole.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    s: str
    big: pl.Int64
    f: pl.Float64


class Out(pa.DataFrameModel):
    v: int  # non-null declared


def non_strict_expr_cast(df: DataFrame[In]) -> DataFrame[Out]:
    # Unparsable strings become null.
    return df.select(v=pl.col("s").cast(pl.Int64, strict=False))


class NarrowOut(pa.DataFrameModel):
    n: pl.Int8  # non-null declared


def non_strict_int_narrowing(df: DataFrame[In]) -> DataFrame[NarrowOut]:
    # Int64 values outside Int8's range overflow to null.
    return df.select(n=pl.col("big").cast(pl.Int8, strict=False))


class FloatToIntOut(pa.DataFrameModel):
    fi: pl.Int64  # non-null declared


def non_strict_float_to_int(df: DataFrame[In]) -> DataFrame[FloatToIntOut]:
    # NaN / inf / out-of-range floats become null.
    return df.select(fi=pl.col("f").cast(pl.Int64, strict=False))


class FrameOut(pa.DataFrameModel):
    s: int  # non-null declared


def non_strict_frame_cast(df: DataFrame[In]) -> DataFrame[FrameOut]:
    return df.cast({"s": pl.Int64}, strict=False)
