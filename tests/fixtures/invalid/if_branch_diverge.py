"""Variable assigned different schemas in if vs else branches is caught (issue #95)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True


def if_branch_wrong(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    """if-branch drops 'a', else-branch is correct — if-path is a bug."""
    if flag:  # noqa: SIM108
        x = df.filter(pl.col("v") > 0)  # {k,v} — missing a
    else:
        x = df.with_columns(a=pl.col("v"))  # {k,v,a}
    return x


def else_branch_wrong(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    """else-branch drops 'a', if-branch is correct — else-path is a bug."""
    if flag:  # noqa: SIM108
        x = df.with_columns(a=pl.col("v"))  # {k,v,a}
    else:
        x = df.filter(pl.col("v") > 0)  # {k,v} — missing a
    return x


def if_strict_extra(df: DataFrame[KV], flag: bool = True) -> DataFrame[KVa]:
    """Both branches add a column not in strict KVa — each path has an extra."""
    if flag:  # noqa: SIM108
        x = df.with_columns(a=pl.col("v") * 2, b=pl.col("v"))  # {k,v,a,b}
    else:
        x = df.with_columns(a=pl.col("v") + 1, c=pl.col("v"))  # {k,v,a,c}
    return x


def if_only_missing(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    """if-only: column 'a' is added in the if-branch but absent on the no-else path."""
    x = df.filter(pl.col("v") > 0)  # {k,v}
    if flag:
        x = df.with_columns(a=pl.col("v"))  # {k,v,a}
    return x
