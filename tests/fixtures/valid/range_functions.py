"""Valid: module-level range constructors are modeled precisely (issue #140).

``pl.int_range`` -> Int64 (or an explicit ``dtype=``), ``pl.int_ranges`` -> a List
of that, ``pl.datetime_ranges`` -> List(Datetime) — the plurals of the already
modeled ``pl.datetime_range``. Declaring the true dtype must type-check.
"""

from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    x: int
    a: datetime
    b: datetime


class Out(pa.DataFrameModel):
    r: int  # int_range -> Int64
    u: pl.UInt32  # int_range(dtype=UInt32) -> UInt32
    lst: pl.List(pl.Int64)  # int_ranges -> List(Int64)
    dts: pl.List(pl.Datetime)  # datetime_ranges -> List(Datetime)


def ranges(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        r=pl.int_range(0, pl.len()),
        u=pl.int_range(0, pl.len(), dtype=pl.UInt32),
        lst=pl.int_ranges(0, pl.col("x")),
        dts=pl.datetime_ranges(pl.col("a"), pl.col("b")),
    )
