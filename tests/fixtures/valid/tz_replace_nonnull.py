"""Valid: dt.replace_time_zone without a null DST policy stays non-null (issue #152).

Only ``ambiguous="null"`` / ``non_existent="null"`` inject nulls; the default
policies (``"raise"``) and the other resolving policies (``"earliest"`` /
``"latest"``) keep every row, so the tz-aware result stays non-null and a
non-null declaration must still type-check. Counterpart to
``invalid/tz_replace_null_injection`` — guards against over-nullifying.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TsIn(pa.DataFrameModel):
    ts: pl.Datetime


class TzOut(pa.DataFrameModel):
    r: pl.Datetime(time_zone="America/New_York")  # non-null


def replace_default(df: DataFrame[TsIn]) -> DataFrame[TzOut]:
    return df.select(r=pl.col("ts").dt.replace_time_zone("America/New_York"))


def ambiguous_earliest(df: DataFrame[TsIn]) -> DataFrame[TzOut]:
    return df.select(r=pl.col("ts").dt.replace_time_zone("America/New_York", ambiguous="earliest"))
