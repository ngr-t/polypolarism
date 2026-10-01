"""Small-int, Float16 and 128-bit receivers through numeric reductions (backlog N-5).

Probed (polars 1.41.2; Float16 grouped forms re-probed on 1.43.2/1.44.2):

- ``sum``/``product`` upcast Int8/Int16/UInt8/UInt16 to **Int64** — signed
  Int64 even for the unsigned receivers — identically in ``select`` and
  ``group_by().agg()`` contexts.
- ``mean``/``std``/``var``/``median``/``quantile`` on integer receivers
  return Float64.
- Float16 keeps its width through every reduction, like Float32 — in
  ``select`` and, since polars 1.43.2, in grouped contexts too (through
  1.42 grouped mean/median/quantile on Float16 panicked in rust).
- Grouped ``product`` on UInt128 still panics — see
  ``invalid/uint128_grouped_product_panic``.
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
    return df.select(
        pl.col("half").mean().alias("avg_half"),
        pl.col("half").quantile(0.5).alias("q_half"),
    )


class PerDeviceHalf(pa.DataFrameModel):
    device: str
    avg_half: pl.Float16
    med_half: pl.Float16

    class Config:
        strict = True


def agg_float16_reductions(df: DataFrame[Telemetry]) -> DataFrame[PerDeviceHalf]:
    # Grouped mean/median on Float16 keep the width (polars >= 1.43.2;
    # these panicked in rust through 1.42).
    return df.group_by("device").agg(
        pl.col("half").mean().alias("avg_half"),
        pl.col("half").median().alias("med_half"),
    )


class HalfWindow(pa.DataFrameModel):
    q_half: pl.Float16

    class Config:
        strict = True


def over_float16_quantile(df: DataFrame[Telemetry]) -> DataFrame[HalfWindow]:
    # over windows agree with group_by().agg() (polars >= 1.43.2).
    return df.select(pl.col("half").quantile(0.5).over("device").alias("q_half"))


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
