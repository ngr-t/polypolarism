"""Valid: fill_null(<expression>) resolves to the supertype of receiver and fill
(issue #158, boundary of #156).

A non-literal fill argument is typed by the usual supertype rule — a wider
column argument widens the result (``Int8`` filled with an ``Int64`` column ->
Int64), exactly like ``shift(fill_value=<expr>)``. A null survives only where
both sides are null, so a nullable fill keeps the result Nullable; a non-null
fill (even on a null-free receiver) removes every null. Declaring the true
polars result must type-check. Probed on polars 1.41.2.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v8: pl.Int8 = pa.Field(nullable=True)
    i8: pl.Int8
    i64: pl.Int64
    v64: pl.Int64 = pa.Field(nullable=True)
    f32: pl.Float32
    vf: pl.Float32 = pa.Field(nullable=True)


class Out(pa.DataFrameModel):
    same: pl.Int8  # v8.fill_null(i8) -> Int8
    narrower: pl.Int64  # v64.fill_null(i8) -> Int64
    wider: pl.Int64  # v8.fill_null(i64) -> Int64 (the FP)
    float_arg: pl.Float32  # v8.fill_null(f32) -> Float32
    nonnull_wider: pl.Int64  # i8.fill_null(i64) -> Int64 (non-null receiver still widens)
    nullable_wider: pl.Int64 = pa.Field(nullable=True)  # v8.fill_null(v64) -> Nullable(Int64)
    nullable_float: pl.Float32 = pa.Field(nullable=True)  # v8.fill_null(vf) -> Nullable(Float32)


def fill_expr(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        same=pl.col("v8").fill_null(pl.col("i8")),
        narrower=pl.col("v64").fill_null(pl.col("i8")),
        wider=pl.col("v8").fill_null(pl.col("i64")),
        float_arg=pl.col("v8").fill_null(pl.col("f32")),
        nonnull_wider=pl.col("i8").fill_null(pl.col("i64")),
        nullable_wider=pl.col("v8").fill_null(pl.col("v64")),
        nullable_float=pl.col("v8").fill_null(pl.col("vf")),
    )
