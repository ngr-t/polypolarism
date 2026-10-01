"""Grouped evaluation PANICS on the UInt128 product cell (backlog N-5).

Probed (polars 1.41.2 and 1.44.2): ``product`` on UInt128 panics in rust
(pyo3 ``PanicException`` — a BaseException, not a catchable polars error
class) under grouped evaluation: ``group_by().agg()`` and ``Expr.over``
windows alike. The SAME reduction is valid as a whole-frame ``select``
reduction. A guaranteed crash must not type-check, so each grouped form
below is a pple-groupby error.

mean/median/quantile on Float16 used to be panic cells here too; polars
1.43.2 fixed them, and their grouped forms now live in
``valid/small_int_float16_reductions``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Readings(pa.DataFrameModel):
    device: str
    big_u: pl.UInt128


def agg_product_uint128(df: DataFrame[Readings]):
    # WRONG: grouped product on UInt128 panics in rust
    # (SchemaMismatch "Expected list[i64], got u128")
    return df.group_by("device").agg(pl.col("big_u").product().alias("prod"))


def over_product_uint128(df: DataFrame[Readings]):
    # WRONG: over windows are grouped evaluation — the same panic fires
    return df.select(pl.col("big_u").product().over("device").alias("prod"))
