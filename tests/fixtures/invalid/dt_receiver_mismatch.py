"""Invalid: dt-namespace method on the wrong temporal type (issue #139).

The ``.dt`` namespace validated only that the receiver is *some* temporal dtype,
then handed out the per-method return dtype regardless of WHICH temporal it is.
These six method-receiver combinations always raise at runtime
(InvalidOperationError / SchemaError) but passed silently — human-error-shaped
mistakes (calendar/time-of-day fields on a Duration, ``total_*`` on a Datetime/
Date, confusing ``minute()`` with ``total_minutes()``).
"""

from datetime import date, datetime, timedelta

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class DurIn(pa.DataFrameModel):
    d: timedelta


class DtIn(pa.DataFrameModel):
    t: datetime


class DateIn(pa.DataFrameModel):
    t: date


class IntOut(pa.DataFrameModel):
    r: int


class DtOut(pa.DataFrameModel):
    r: datetime


def duration_year(df: DataFrame[DurIn]) -> DataFrame[IntOut]:
    return df.select(r=pl.col("d").dt.year())


def duration_hour(df: DataFrame[DurIn]) -> DataFrame[IntOut]:
    return df.select(r=pl.col("d").dt.hour())


def duration_truncate(df: DataFrame[DurIn]) -> DataFrame[DtOut]:
    return df.select(r=pl.col("d").dt.truncate("1h"))


def datetime_total_minutes(df: DataFrame[DtIn]) -> DataFrame[IntOut]:
    return df.select(r=pl.col("t").dt.total_minutes())


def date_total_days(df: DataFrame[DateIn]) -> DataFrame[IntOut]:
    return df.select(r=pl.col("t").dt.total_days())


def date_hour(df: DataFrame[DateIn]) -> DataFrame[IntOut]:
    return df.select(r=pl.col("t").dt.hour())
