"""Invalid fixture: polars' (38, 0) does NOT satisfy a bare ``pl.Decimal``.

Issue #75 (the false-negative direction): pandera enforces (28, 0) for
the bare class-name annotation, so returning the polars-default
``Decimal(38, 0)`` fails ``validate`` at runtime — and must fail
statically too.
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


def wrong_38(df: DataFrame[Src]) -> DataFrame[DecBare]:
    return df.select(d=pl.col("a").cast(pl.Decimal(38, 0)))
