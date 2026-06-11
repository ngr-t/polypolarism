"""Invalid fixture: Datetime/Duration time_unit mismatches (issue #66).

The time unit is part of dtype identity: pandera validation raises
SchemaError for a declared ``Datetime[ns]`` over an actual
``Datetime[us]`` column, and for ``Duration[ms]`` over ``Duration[us]``.
Both were static false negatives while the dtype model only carried the
tz. Neither schema sets ``coerce`` — and even under coerce a us -> ns
*refinement* stays an error (the cast multiplies and overflows for
extreme values: value-dependent, probed); only coarsening is repaired
(see ``valid/dtype_annotated_params.py``).
"""

from __future__ import annotations

import typing

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    t: pl.Datetime  # bare form — polars' "us" default

    class Config:
        strict = True


class DtNs(pa.DataFrameModel):
    t: typing.Annotated[pl.Datetime, "ns", None]

    class Config:
        strict = True


def datetime_unit_mismatch(df: DataFrame[Src]) -> DataFrame[DtNs]:
    return df.select(pl.col("t"))  # actual Datetime[us]


class DurMs(pa.DataFrameModel):
    d: typing.Annotated[pl.Duration, "ms"] = pa.Field(nullable=True)

    class Config:
        strict = True


def duration_unit_mismatch(df: DataFrame[Src]) -> DataFrame[DurMs]:
    return df.select(d=pl.col("t").diff())  # actual Duration[us]
