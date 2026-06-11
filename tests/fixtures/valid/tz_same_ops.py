"""Valid contrasts for issue #50: same-tz Datetime operations keep working.

Probed on polars 1.41.2: ``Datetime[UTC] - Datetime[UTC]`` -> Duration,
``Datetime[UTC] + Duration`` keeps the tz, same-tz concat and
``dt.replace_time_zone`` aligning a naive column with an aware one all
succeed. ``str.to_datetime`` with a format literal containing an offset
directive (``%z`` or its ``%:z`` / ``%#z`` variants) resolves to
``Datetime[UTC]``. False-negative twin: ``invalid/tz_mixing``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Events(pa.DataFrameModel):
    started_at: pl.Datetime(time_zone="UTC")
    ended_at: pl.Datetime(time_zone="UTC")
    grace: pl.Duration


class LocalEvents(pa.DataFrameModel):
    started_at: pl.Datetime


class Report(pa.DataFrameModel):
    elapsed: pl.Duration
    deadline: pl.Datetime(time_zone="UTC")


def summarize(df: DataFrame[Events]) -> DataFrame[Report]:
    return df.select(
        elapsed=pl.col("ended_at") - pl.col("started_at"),
        deadline=pl.col("ended_at") + pl.col("grace"),
    )


def stack(a: DataFrame[Events], b: DataFrame[Events]) -> DataFrame[Events]:
    return pl.concat([a, b], how="vertical")


def localize(df: DataFrame[LocalEvents]) -> DataFrame[LocalEvents]:
    utc = df.with_columns(pl.col("started_at").dt.replace_time_zone("UTC"))
    return utc.with_columns(pl.col("started_at").dt.replace_time_zone(None))


class RawEvents(pa.DataFrameModel):
    stamp: str


class ParsedEvents(pa.DataFrameModel):
    stamp: pl.Datetime(time_zone="UTC")


def parse(df: DataFrame[RawEvents]) -> DataFrame[ParsedEvents]:
    return df.with_columns(pl.col("stamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%:z"))
