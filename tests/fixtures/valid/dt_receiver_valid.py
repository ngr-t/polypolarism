"""Valid: dt-namespace methods on their correct temporal type (issue #139 guards).

The per-method receiver matrix must NOT flag the correct pairings: ``total_*`` on
Duration, calendar accessors on Datetime, time-of-day on Time. These stay OK.
"""

from datetime import date, datetime, time, timedelta

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    d: timedelta
    dt: datetime
    dd: date
    tt: time


class Out(pa.DataFrameModel):
    tot: int  # duration.total_minutes -> Int64
    yr: pl.Int32  # datetime.year -> Int32
    wk: pl.Int8  # date.week -> Int8
    hr: pl.Int8  # time.hour -> Int8


def dt_ok(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        tot=pl.col("d").dt.total_minutes(),
        yr=pl.col("dt").dt.year(),
        wk=pl.col("dd").dt.week(),
        hr=pl.col("tt").dt.hour(),
    )
