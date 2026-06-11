"""``str.to_datetime`` follows its ``time_unit=`` argument and chrono offsets.

The ``time_unit="ns"`` keyword changes the result unit (issue #66
boundary), and the extended chrono offset directives (``%::z``) resolve
``Datetime[UTC]`` like ``%z``.

False-negative twin: ``invalid/str_to_datetime_time_unit_wrong``.
"""

from __future__ import annotations

from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Raw(pa.DataFrameModel):
    s: str
    s_off: str  # offset-bearing strings for the %::z parse

    class Config:
        strict = True
        coerce = True


class Parsed(pa.DataFrameModel):
    ns_naive: Annotated[pl.Datetime, "ns", None]
    us_utc: Annotated[pl.Datetime, "us", "UTC"]

    class Config:
        strict = True
        coerce = True


def to_datetime_units(df: DataFrame[Raw]) -> DataFrame[Parsed]:
    # One source column cannot satisfy both formats at runtime (the
    # offset-less format rejects offset suffixes and vice versa) — each
    # parse reads its own column.
    return df.select(
        ns_naive=pl.col("s").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ns"),
        us_utc=pl.col("s_off").str.to_datetime("%Y-%m-%d %H:%M:%S %::z"),
    )
