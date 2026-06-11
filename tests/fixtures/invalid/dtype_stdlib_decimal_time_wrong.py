"""Invalid fixture: wrong dtypes against stdlib ``decimal.Decimal`` / ``datetime.time``.

Issue #77 FN repro: the unrecognized annotations silently dropped the fields,
so an Int64 column against a declared ``decimal.Decimal`` (pandera:
``Decimal(28, 0)``) passed statically on a non-strict schema and failed at
runtime. Probed (pandera 0.31.1): ``DecOpen.validate`` rejects the Int64
frame ("The return is expected to be of Decimal class") and
``TimeOpen.validate`` raises SchemaError ("expected column 't' to have type
Time, got Int64").
"""

from __future__ import annotations

import datetime
import decimal

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class DecOpen(pa.DataFrameModel):
    d: decimal.Decimal  # pandera: Decimal(28, 0)

    class Config:
        strict = False


class TimeOpen(pa.DataFrameModel):
    t: datetime.time  # pandera: Time

    class Config:
        strict = False


def decimal_wrong_dtype(df: DataFrame[Src]) -> DataFrame[DecOpen]:
    return df.select(d=pl.col("a"))  # Int64, runtime validate rejects


def time_wrong_dtype(df: DataFrame[Src]) -> DataFrame[TimeOpen]:
    return df.select(t=pl.col("a"))  # Int64, runtime SchemaError
