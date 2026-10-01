"""Grouped evaluation PANICS on probed agg cells (backlog N-5).

Probed (polars 1.41.2): ``mean``/``median``/``quantile`` on Float16 and
``product`` on UInt128 panic in rust (pyo3 ``PanicException`` — a
BaseException, not a catchable polars error class) under grouped
evaluation: ``group_by().agg()`` and ``Expr.over`` windows alike. The SAME
reductions are valid as whole-frame ``select`` reductions — that
select-context acceptance lives in ``valid/small_int_float16_reductions``.
A guaranteed crash must not type-check, so each grouped form below is a
pple-groupby error.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Readings(pa.DataFrameModel):
    device: str
    half: pl.Float16
    big_u: pl.UInt128


def agg_mean_float16(df: DataFrame[Readings]):
    # WRONG: grouped mean on Float16 panics at runtime
    # ("not implemented for dtype Float16", probed 1.41.2)
    return df.group_by("device").agg(pl.col("half").mean().alias("avg"))


def over_median_float16(df: DataFrame[Readings]):
    # WRONG: over windows are grouped evaluation — the same panic fires
    return df.select(pl.col("half").median().over("device").alias("med"))


def agg_product_uint128(df: DataFrame[Readings]):
    # WRONG: grouped product on UInt128 panics in rust
    # (SchemaMismatch "Expected list[i64], got u128", probed 1.41.2)
    return df.group_by("device").agg(pl.col("big_u").product().alias("prod"))
