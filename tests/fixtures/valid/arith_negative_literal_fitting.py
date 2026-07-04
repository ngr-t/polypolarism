"""Valid: NEGATIVE numeric literals fit the column dtype too (issue #149).

A negative literal parses as ``UnaryOp(USub, Constant(n))``, not a bare
``Constant``, so it escaped the #147 literal-fit path and fell through to the
generic Int64/Float64 promotion lattice — a false positive (and a regression on
``i8 + (-1)``, previously Int8). Folding the ``USub`` restores the fit: a signed
column keeps its dtype (or widens same-sign), a float column absorbs the
literal, and a negative literal against an UNSIGNED column resolves to the
smallest signed supertype polars picks (``u8+(-1)->Int16``, ``u16+(-1)->Int32``,
``u32+(-1)->Int64``, ``u64+(-1)->Int64``). Probed on polars 1.41.2.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    u8: pl.UInt8
    u16: pl.UInt16
    u32: pl.UInt32
    u64: pl.UInt64
    f32: pl.Float32


class Out(pa.DataFrameModel):
    fits: pl.Int8  # i8 + (-1) -> Int8 (fits, regression guard)
    widened: pl.Int16  # i8 + (-1000) -> Int16 (minimal signed widening)
    minus_neg: pl.Int8  # i8 - (-1) -> Int8
    mult_neg: pl.Int16  # i8 * (-1000) -> Int16
    reversed: pl.Int8  # (-1) + i8 -> Int8 (literal on the left)
    u8_neg: pl.Int16  # u8 + (-1) -> Int16 (smallest signed supertype)
    u16_neg: pl.Int32  # u16 + (-1) -> Int32
    u32_neg: pl.Int64  # u32 + (-1) -> Int64
    u64_neg: pl.Int64  # u64 + (-1) -> Int64 (capped, not Float64)
    f32_neg: pl.Float32  # f32 + (-1.0) -> Float32 (float column absorbs)
    via_lit: pl.Int8  # i8 + pl.lit(-1) -> Int8 (folded inside pl.lit too)


def fit_negative(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        fits=pl.col("i8") + (-1),
        widened=pl.col("i8") + (-1000),
        minus_neg=pl.col("i8") - (-1),
        mult_neg=pl.col("i8") * (-1000),
        reversed=(-1) + pl.col("i8"),
        u8_neg=pl.col("u8") + (-1),
        u16_neg=pl.col("u16") + (-1),
        u32_neg=pl.col("u32") + (-1),
        u64_neg=pl.col("u64") + (-1),
        f32_neg=pl.col("f32") + (-1.0),
        via_lit=pl.col("i8") + pl.lit(-1),
    )
