"""Variable assigned same schema in both if/else branches is not a false positive (issue #95)."""

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


def both_branches_ok(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    """Both branches produce the correct schema — no false positive."""
    if flag:  # noqa: SIM108
        x = df.with_columns(a=pl.col("v") * 2.0)
    else:
        x = df.with_columns(a=pl.col("v"))
    return x
