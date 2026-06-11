"""Warning fixture: grouped all-null temporal reductions (issue #91).

std/var/sum on Date/Datetime/Time SUCCEED in grouped contexts — all-null
with the receiver dtype (probed identical on polars 1.37.0 through
1.41.2) — while raising as whole-frame reductions. The grouped form is
accepted (Nullable receiver dtype) with a PLW012 "provably all-null"
advisory: it runs, but it is almost certainly not what the author meant.
"""

from __future__ import annotations

import typing

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    g: str
    t: typing.Annotated[pl.Datetime, "us", None]

    class Config:
        strict = True
        coerce = True


class Out(pa.DataFrameModel):
    g: str
    x: typing.Annotated[pl.Datetime, "us", None] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


@pa.check_types
def grouped_std_datetime(df: DataFrame[Src]) -> DataFrame[Out]:
    return df.group_by("g").agg(x=pl.col("t").std())
