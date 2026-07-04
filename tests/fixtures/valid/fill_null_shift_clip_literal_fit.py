"""Valid: literal args of fill_null / shift(fill_value=) / clip fit the column
dtype like binary operands (issue #156, extending #147/#149).

``fill_null`` and ``shift`` WIDEN when the literal does not fit (``Int8`` filled
with ``1000`` -> Int16), fold a unary sign (``-1`` stays Int8, not Int64), and
resolve a negative against an unsigned column to the signed supertype. ``clip``
never widens — its result stays the column dtype — as long as every bound is
representable in that dtype. Declaring the true polars result must type-check.
Probed on polars 1.41.2.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    v8: pl.Int8 = pa.Field(nullable=True)
    u8: pl.UInt8


class Out(pa.DataFrameModel):
    fn_fit: pl.Int8  # v8.fill_null(-1) -> Int8 (fits, sign folded)
    fn_widen: pl.Int16  # v8.fill_null(1000) -> Int16 (minimal widening)
    fn_widen_neg: pl.Int16  # v8.fill_null(-1000) -> Int16
    fn_u8_neg: pl.Int16  # u8.fill_null(-1) -> Int16 (signed supertype)
    fn_float: pl.Float64  # v8.fill_null(1.5) -> Float64 (float literal)
    sh_fit: pl.Int8  # i8.shift(1, fill_value=1) -> Int8
    sh_widen: pl.Int16  # i8.shift(1, fill_value=1000) -> Int16
    sh_neg: pl.Int8  # i8.shift(1, fill_value=-1) -> Int8 (sign folded)
    clip_bounds: pl.Int8  # i8.clip(-128, 127) -> Int8 (exact bounds, no widen)


def fit(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        fn_fit=pl.col("v8").fill_null(-1),
        fn_widen=pl.col("v8").fill_null(1000),
        fn_widen_neg=pl.col("v8").fill_null(-1000),
        fn_u8_neg=pl.col("u8").fill_null(-1),
        fn_float=pl.col("v8").fill_null(1.5),
        sh_fit=pl.col("i8").shift(1, fill_value=1),
        sh_widen=pl.col("i8").shift(1, fill_value=1000),
        sh_neg=pl.col("i8").shift(1, fill_value=-1),
        clip_bounds=pl.col("i8").clip(-128, 127),
    )
