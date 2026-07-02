"""Valid: full-width integer promotion follows polars' supertype lattice (issue #127).

polars promotes ``+`` / ``-`` / ``*`` / ``//`` / ``%`` operands to their
common supertype across the entire integer/float width matrix — not just the
four widths ``promote_types`` historically modelled. Same-sign pairs take the
wider width; mixed-sign pairs take the signed type one width above the
unsigned operand, and ``UInt64`` + any signed widens to ``Float64``. Declaring
the true polars result dtype must type-check (previously the checker kept the
left operand's dtype and rejected these).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    i16: pl.Int16
    i32: pl.Int32
    i64: pl.Int64
    u8: pl.UInt8
    u16: pl.UInt16
    u32: pl.UInt32
    u64: pl.UInt64


class Out(pa.DataFrameModel):
    a: pl.Int16  # Int8 + Int16 -> Int16 (same sign, wider width wins)
    b: pl.UInt16  # UInt8 + UInt16 -> UInt16 (same sign)
    c: pl.Int16  # UInt8 + Int8 -> Int16 (mixed: signed one width above unsigned)
    d: pl.Int32  # UInt16 + Int16 -> Int32
    e: pl.Int64  # UInt32 + Int64 -> Int64
    f: pl.Float64  # UInt64 + Int64 -> Float64
    g: pl.Int64  # UInt8 * Int64 -> Int64
    h: pl.Int32  # Int8 // UInt16 -> Int32


def promote(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        a=pl.col("i8") + pl.col("i16"),
        b=pl.col("u8") + pl.col("u16"),
        c=pl.col("u8") + pl.col("i8"),
        d=pl.col("u16") + pl.col("i16"),
        e=pl.col("u32") + pl.col("i64"),
        f=pl.col("u64") + pl.col("i64"),
        g=pl.col("u8") * pl.col("i64"),
        h=pl.col("i8") // pl.col("u16"),
    )
