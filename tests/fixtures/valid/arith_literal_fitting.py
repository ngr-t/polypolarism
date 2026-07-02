"""Valid: numeric literals fit the column dtype in arithmetic (issue #147).

polars types a Python literal by the COLUMN it combines with: an int literal
adopts the column's dtype when it fits, else the minimal same-sign widening
(``i8 + 1000`` -> Int16); a float column absorbs any literal into its own width
(``f32 + 1.0`` -> Float32); an int column with a float literal widens to Float64.
Declaring the true polars result must type-check (the literal is NOT promoted to
a uniform Int64/Float64). Distinct from #127 (col-vs-col promotion).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    i32: pl.Int32
    u16: pl.UInt16
    f32: pl.Float32


class Out(pa.DataFrameModel):
    fits: pl.Int8  # i8 + 1 -> Int8 (literal fits)
    widened: pl.Int16  # i8 + 1000 -> Int16 (minimal widening)
    kept32: pl.Int32  # i32 * 2 -> Int32
    unsigned: pl.UInt16  # u16 + 1 -> UInt16
    float32: pl.Float32  # f32 + 1.0 -> Float32 (float column absorbs)
    floordiv: pl.Int8  # i8 // 2 -> Int8
    to_float: pl.Float64  # i32 - 2.5 -> Float64 (int col + float literal)


def fit(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        fits=pl.col("i8") + 1,
        widened=pl.col("i8") + 1000,
        kept32=pl.col("i32") * 2,
        unsigned=pl.col("u16") + 1,
        float32=pl.col("f32") + 1.0,
        floordiv=pl.col("i8") // 2,
        to_float=pl.col("i32") - 2.5,
    )
