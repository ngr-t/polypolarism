"""Small-int, Float16 and 128-bit receivers through numeric reductions (backlog N-5).

Probed (polars 1.41.2):

- ``sum``/``product`` upcast Int8/Int16/UInt8/UInt16 to **Int64** — signed
  Int64 even for the unsigned receivers — identically in ``select`` and
  ``group_by().agg()`` contexts.
- ``mean``/``std``/``var``/``median``/``quantile`` on integer receivers
  return Float64.
- Float16 keeps its width through every whole-frame (``select``) reduction,
  like Float32. The grouped forms of mean/median/quantile on Float16 PANIC
  instead — see ``invalid/float16_uint128_grouped_panic``.
- Int128/UInt128 ``sum``/``min``/``max`` preserve the receiver width.

The false-negative twin is ``invalid/small_int_float16_reductions_wrong``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Telemetry(pa.DataFrameModel):
    device: str
    raw_i8: pl.Int8
    raw_u16: pl.UInt16
    counter_u8: pl.UInt8
    half: pl.Float16
    big: pl.Int128
    big_u: pl.UInt128


class SmallIntTotals(pa.DataFrameModel):
    total_i8: pl.Int64
    avg_u16: pl.Float64

    class Config:
        strict = True


def select_small_int_reductions(df: DataFrame[Telemetry]) -> DataFrame[SmallIntTotals]:
    return df.select(
        pl.col("raw_i8").sum().alias("total_i8"),
        pl.col("raw_u16").mean().alias("avg_u16"),
    )


class PerDevice(pa.DataFrameModel):
    device: str
    total_i8: pl.Int64
    prod_u8: pl.Int64
    spread_u16: pl.Float64 = pa.Field(nullable=True)

    class Config:
        strict = True


def agg_small_int_reductions(df: DataFrame[Telemetry]) -> DataFrame[PerDevice]:
    # The sub-32-bit upcasts apply in grouped context too; product(UInt8)
    # lands on SIGNED Int64 (probed 1.41.2). std stays nullable (ddof=1
    # singleton-group rule, issue #60).
    return df.group_by("device").agg(
        pl.col("raw_i8").sum().alias("total_i8"),
        pl.col("counter_u8").product().alias("prod_u8"),
        pl.col("raw_u16").std().alias("spread_u16"),
    )


class HalfStats(pa.DataFrameModel):
    avg_half: pl.Float16
    q_half: pl.Float16

    class Config:
        strict = True


def select_float16_reductions(df: DataFrame[Telemetry]) -> DataFrame[HalfStats]:
    # Valid ONLY as a whole-frame reduction: the grouped forms of these
    # exact cells panic in rust (probed 1.41.2) and are flagged in the
    # ``invalid/float16_uint128_grouped_panic`` twin.
    return df.select(
        pl.col("half").mean().alias("avg_half"),
        pl.col("half").quantile(0.5).alias("q_half"),
    )


class BigTotals(pa.DataFrameModel):
    total_big: pl.Int128

    class Config:
        strict = True


def select_sum_int128_keeps_width(df: DataFrame[Telemetry]) -> DataFrame[BigTotals]:
    return df.select(pl.col("big").sum().alias("total_big"))


class PerDeviceBig(pa.DataFrameModel):
    device: str
    total_u: pl.UInt128

    class Config:
        strict = True


def agg_sum_uint128_keeps_width(df: DataFrame[Telemetry]) -> DataFrame[PerDeviceBig]:
    # sum on UInt128 is grouped-safe (only product is the panic cell).
    return df.group_by("device").agg(pl.col("big_u").sum().alias("total_u"))
