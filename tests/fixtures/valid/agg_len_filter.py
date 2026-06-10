"""Valid fixture: issue #23 — Expr.len() and filter(...).<agg>() in agg.

``pl.col("v").len()`` infers UInt32 (count-including-nulls), bridged to
the declared ``int`` by ``coerce = True``. The conditional aggregation
``filter(pred).sum()`` is row-subsetting and dtype-preserving, so the
sum stays Int64. ``strict = True`` ensures the inferred columns are
exactly the declared ones — nothing is dropped to Unknown.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    g: str
    v: int

    class Config:
        strict = True
        coerce = True


class OSum(pa.DataFrameModel):
    g: str
    s: int

    class Config:
        strict = True
        coerce = True


class OLen(pa.DataFrameModel):
    g: str
    n: int

    class Config:
        strict = True
        coerce = True


class OFSum(pa.DataFrameModel):
    g: str
    fs: int

    class Config:
        strict = True
        coerce = True


def agg_sum(df: DataFrame[In]) -> DataFrame[OSum]:
    """Regression guard: plain sum aggregation already worked."""
    return df.group_by("g").agg(pl.col("v").sum().alias("s"))


def agg_len(df: DataFrame[In]) -> DataFrame[OLen]:
    """Group size via Expr.len() — UInt32, coerced to Int64."""
    return df.group_by("g").agg(pl.col("v").len().alias("n"))


def agg_filter_sum(df: DataFrame[In]) -> DataFrame[OFSum]:
    """Conditional aggregation: sum of the positive values per group."""
    return df.group_by("g").agg(pl.col("v").filter(pl.col("v") > 0).sum().alias("fs"))
