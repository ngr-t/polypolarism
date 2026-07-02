"""Valid: list reductions whose nullability is correct (issue #130 boundary).

``list.sum`` returns 0 (non-null) for an empty sub-list and ``list.len`` returns
0, so both stay non-null and must still type-check against non-null fields. The
reducing aggregations that CAN null (min/max/mean/...) are fine when declared
nullable. Counterpart to ``invalid/list_agg_nullable``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    xs: pl.List(pl.Int64)


class SumOut(pa.DataFrameModel):
    s: int  # sum -> non-null (empty sub-list -> 0)
    n: pl.UInt32  # len -> non-null


def list_sum_len_nonnull(df: DataFrame[ListIn]) -> DataFrame[SumOut]:
    return df.select(s=pl.col("xs").list.sum(), n=pl.col("xs").list.len())


class NullMax(pa.DataFrameModel):
    x: int = pa.Field(nullable=True)  # max declared nullable -> OK


def list_max_nullable_ok(df: DataFrame[ListIn]) -> DataFrame[NullMax]:
    return df.select(x=pl.col("xs").list.max())
