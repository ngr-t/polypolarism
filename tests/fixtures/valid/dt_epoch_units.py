"""``dt.epoch`` return dtype follows its ``time_unit`` argument (issue #73).

Probed (polars 1.41.2): ``epoch("ns"/"us"/"ms"/"s")`` and the no-arg
default return Int64, but ``epoch("d")`` (days since epoch) returns
**Int32**. Before #73 the fixed table entry claimed Int64 for every
argument and the correct Int32 declaration was falsely rejected.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Events(pa.DataFrameModel):
    ts: pl.Datetime


class EpochOut(pa.DataFrameModel):
    day: pl.Int32
    second: pl.Int64
    default: pl.Int64


def epoch_units(df: DataFrame[Events]) -> DataFrame[EpochOut]:
    return df.select(
        day=pl.col("ts").dt.epoch("d"),
        second=pl.col("ts").dt.epoch("s"),
        default=pl.col("ts").dt.epoch(),
    )
