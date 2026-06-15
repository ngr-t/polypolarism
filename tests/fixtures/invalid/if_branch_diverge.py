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
