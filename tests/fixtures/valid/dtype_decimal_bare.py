"""Valid fixture: bare ``pl.Decimal`` annotation is pandera's (28, 0).

Issue #75: pandera's engine resolves the bare class-name form through its
own Decimal default — precision 28 — not polars' materialized (38, 0).
``validate`` passes a (28, 0) column, so the exactly-matching cast is OK.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class DecBare(pa.DataFrameModel):
    d: pl.Decimal  # pandera: Decimal(28, 0)

    class Config:
        strict = True


def correct_28(df: DataFrame[Src]) -> DataFrame[DecBare]:
    return df.select(d=pl.col("a").cast(pl.Decimal(28, 0)))
