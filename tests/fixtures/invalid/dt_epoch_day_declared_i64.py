"""Invalid: ``dt.epoch("d")`` declared Int64 (issue #73).

False-negative twin of ``valid/dt_epoch_units.py``: ``epoch("d")`` returns
Int32 at runtime (probed, polars 1.41.2) — before #73 the fixed Int64
table entry let this wrong declaration pass statically while pandera
rejects the runtime Int32.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Events(pa.DataFrameModel):
    ts: pl.Datetime


class EpochWrong(pa.DataFrameModel):
    day: pl.Int64  # WRONG: epoch("d") yields Int32


def bug_day_declared_i64(df: DataFrame[Events]) -> DataFrame[EpochWrong]:
    return df.select(day=pl.col("ts").dt.epoch("d"))
