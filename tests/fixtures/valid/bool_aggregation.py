"""Valid: numeric aggregations on a Boolean column (issue #126).

polars supports ``sum`` / ``mean`` / ``std`` / ``var`` / ``median`` on Boolean
receivers (counting / share-of-trues is a common pattern), in both the grouped
and whole-frame (select) contexts:

- ``sum(Boolean)`` -> ``UInt32`` (non-null; an empty / all-null group sums to 0),
- ``mean`` / ``median`` (Boolean) -> ``Float64`` (non-null for non-null input),
- ``std`` / ``var`` (Boolean) -> ``Float64``, always-nullable (ddof=1 leaves a
  singleton group null, issue #60).

``min`` / ``max`` already pass through as Boolean. (``quantile`` and ``product``
on Boolean behave differently and are out of scope here.)
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    g: str
    b: bool


class GroupedOut(pa.DataFrameModel):
    s: pl.UInt32  # sum -> UInt32 (non-null)
    m: pl.Float64  # mean -> Float64
    md: pl.Float64  # median -> Float64
    sd: pl.Float64 = pa.Field(nullable=True)  # std -> Float64 (singleton -> null)
    v: pl.Float64 = pa.Field(nullable=True)  # var -> Float64 (singleton -> null)


def bool_agg(df: DataFrame[In]) -> DataFrame[GroupedOut]:
    return (
        df.group_by("g")
        .agg(
            s=pl.col("b").sum(),
            m=pl.col("b").mean(),
            md=pl.col("b").median(),
            sd=pl.col("b").std(),
            v=pl.col("b").var(),
        )
        .select("s", "m", "md", "sd", "v")
    )


class SelSum(pa.DataFrameModel):
    b: pl.UInt32  # select sum keeps the column name -> UInt32


def bool_select_sum(df: DataFrame[In]) -> DataFrame[SelSum]:
    return df.select(pl.col("b").sum())


class SelStats(pa.DataFrameModel):
    m: pl.Float64  # select mean -> Float64
    md: pl.Float64  # select median -> Float64


def bool_select_stats(df: DataFrame[In]) -> DataFrame[SelStats]:
    return df.select(m=pl.col("b").mean(), md=pl.col("b").median())
