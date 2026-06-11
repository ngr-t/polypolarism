"""Invalid fixture: provably incompatible validate() inputs (issue #89).

``Schema.validate(arg)`` runs the same validation as ``check_types`` —
a genuinely exact argument frame with a non-coercible dtype conflict or
a provably missing required column raises SchemaError on every call.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True
        coerce = True


class StrictStr(pa.DataFrameModel):
    a: str  # the input is provably Int64; coerce is off

    class Config:
        strict = True


class NeedsB(pa.DataFrameModel):
    b: int


def dtype_conflict(df: DataFrame[Src]) -> pl.DataFrame:
    return StrictStr.validate(df.select(pl.col("a")))


def provably_missing(df: DataFrame[Src]) -> pl.DataFrame:
    picked = df.select(pl.col("a"))
    return NeedsB.validate(picked)
