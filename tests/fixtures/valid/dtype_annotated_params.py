"""Valid fixture: parametrized dtypes via ``typing.Annotated`` annotations.

Issues #65/#66/#67: pandera passes the ``Annotated[pl.<Dtype>, ...]``
metadata as the dtype's positional arguments, so the declared side must
carry the parameters:

- ``Annotated[pl.Decimal, 12, 4]`` is ``Decimal(12, 4)`` — NOT the bare
  default ``Decimal(38, 0)`` (the issue #65 false positive);
- ``Annotated[pl.Datetime, "ns", None]`` is ``Datetime[ns]`` and
  ``Annotated[pl.Duration, "ms"]`` is ``Duration[ms]`` (issue #66);
- ``Annotated[pl.Enum, ["lo", "hi"]]`` carries the ordered categories
  (issue #67).

Exactly-matching expressions therefore pass. The last function pins the
coerce boundary for units: a unit *coarsening* (us -> ms) is a
value-independent cast, so ``Config.coerce`` repairs it (leniency note);
refining stays an error — see ``invalid/dtype_time_unit.py``.
"""

from __future__ import annotations

import typing

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int
    t: typing.Annotated[pl.Datetime, "ms", None]
    e: pl.Enum(["lo", "hi"])

    class Config:
        strict = True


class DecAnnotated(pa.DataFrameModel):
    d: typing.Annotated[pl.Decimal, 12, 4]

    class Config:
        strict = True


def decimal_annotated_exact(df: DataFrame[Src]) -> DataFrame[DecAnnotated]:
    return df.select(d=pl.col("a").cast(pl.Decimal(12, 4)))


class DtNs(pa.DataFrameModel):
    t: typing.Annotated[pl.Datetime, "ns", None]

    class Config:
        strict = True


def datetime_unit_cast(df: DataFrame[Src]) -> DataFrame[DtNs]:
    return df.select(pl.col("t").cast(pl.Datetime("ns")))


class DurMs(pa.DataFrameModel):
    d: typing.Annotated[pl.Duration, "ms"] = pa.Field(nullable=True)

    class Config:
        strict = True


def duration_unit_from_diff(df: DataFrame[Src]) -> DataFrame[DurMs]:
    # ``diff`` keeps the receiver's time unit (probed): Datetime[ms] ->
    # Duration[ms], nullable (head slot).
    return df.select(d=pl.col("t").diff())


class EnumAnnotated(pa.DataFrameModel):
    e: typing.Annotated[pl.Enum, ["lo", "hi"]]

    class Config:
        strict = True


def enum_annotated_passthrough(df: DataFrame[Src]) -> DataFrame[EnumAnnotated]:
    return df.select(pl.col("e"))


class DtMsCoerce(pa.DataFrameModel):
    t: typing.Annotated[pl.Datetime, "ms", None]

    class Config:
        strict = True
        coerce = True


def datetime_unit_coarsened_by_coerce(df: DataFrame[Src]) -> DataFrame[DtMsCoerce]:
    # Inferred Datetime[us] against declared Datetime[ms]: us -> ms is a
    # coarsening cast (a division — value-independent, probed), so
    # pandera's coerce repairs it at validation time.
    return df.select(t=pl.col("t").cast(pl.Datetime("us")))
