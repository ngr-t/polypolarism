"""Vertical concat cannot unify mixed Datetime time units (issue #66 boundary).

polars raises ``SchemaError: type Datetime('ns') is incompatible with
expected type Datetime('us')`` — the time-unit analogue of
``invalid/tz_mixing``.

False-positive twin: ``valid/concat_time_unit_same``.
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


def concat_mixed_units(df: DataFrame[EventsUs]) -> DataFrame[EventsUs]:
    ns_part = df.select(t=pl.col("t").cast(pl.Datetime("ns")))
    return pl.concat([df.select(pl.col("t")), ns_part])  # WRONG: us vs ns cannot unify
