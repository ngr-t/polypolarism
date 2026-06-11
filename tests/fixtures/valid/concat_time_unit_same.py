"""Vertical concat with matching Datetime time units unifies cleanly.

Same-unit frames concatenate; the result keeps the shared unit.

False-negative twin: ``invalid/concat_time_unit_mixing``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Annotated


class EventsUs(pa.DataFrameModel):
    t: Annotated[pl.Datetime, "us", None]

    class Config:
        strict = True
        coerce = True


def concat_same_unit(a: DataFrame[EventsUs], b: DataFrame[EventsUs]) -> DataFrame[EventsUs]:
    return pl.concat([a, b])
