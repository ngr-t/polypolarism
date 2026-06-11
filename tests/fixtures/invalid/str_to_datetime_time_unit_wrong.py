"""Declaring the default unit for a ``time_unit="ns"`` parse is wrong (issue #66).

The keyword changes the result to ``Datetime[ns]``; a stale ``us``
declaration must fail.

False-positive twin: ``valid/str_to_datetime_time_unit``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Annotated


class Raw(pa.DataFrameModel):
    s: str

    class Config:
        strict = True
        coerce = True


class ParsedWrong(pa.DataFrameModel):
    t: Annotated[pl.Datetime, "us", None]  # WRONG: time_unit="ns" yields Datetime[ns]

    class Config:
        strict = True


def to_datetime_stale_unit(df: DataFrame[Raw]) -> DataFrame[ParsedWrong]:
    return df.select(t=pl.col("s").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ns"))
