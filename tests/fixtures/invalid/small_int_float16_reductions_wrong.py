"""False-negative twin of ``valid/small_int_float16_reductions`` (backlog N-5).

The SAME reductions with wrong declared widths: if the widened receiver
matrix ever degrades to Unknown, these declarations would start passing via
leniency and the invalid-category invariant catches it.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Telemetry(pa.DataFrameModel):
    device: str
    raw_i8: pl.Int8
    raw_u16: pl.UInt16
    half: pl.Float16


class SmallIntTotalsWrong(pa.DataFrameModel):
    # WRONG: sum upcasts sub-32-bit ints to Int64 (probed 1.41.2)
    total_i8: pl.Int8
    # WRONG: only Float32/Float16 receivers keep a narrow width; mean of a
    # UInt16 column is Float64 (probed 1.41.2)
    avg_u16: pl.Float32

    class Config:
        strict = True


def select_small_int_reductions_wrong(
    df: DataFrame[Telemetry],
) -> DataFrame[SmallIntTotalsWrong]:
    return df.select(
        pl.col("raw_i8").sum().alias("total_i8"),
        pl.col("raw_u16").mean().alias("avg_u16"),
    )


class HalfStatsWrong(pa.DataFrameModel):
    # WRONG: Float16 keeps its width through select-context reductions
    # (probed 1.41.2); the mean is Float16, not Float64
    avg_half: pl.Float64

    class Config:
        strict = True


def select_float16_mean_wrong(df: DataFrame[Telemetry]) -> DataFrame[HalfStatsWrong]:
    return df.select(pl.col("half").mean().alias("avg_half"))
