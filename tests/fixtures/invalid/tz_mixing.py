"""Invalid: mixing tz-aware and tz-naive Datetime columns (issue #50).

Probed on polars 1.41.2:
- ``pl.concat([naive, utc], how="vertical")`` raises SchemaError
  ("type Datetime('μs', 'UTC') is incompatible with expected type
  Datetime('μs')") -> PLY020.
- ``aware - naive`` raises SchemaError ("failed to determine supertype
  of datetime[μs, UTC] and datetime[μs]") -> PLY009.
- Comparing two different time zones raises SchemaError too -> PLY009.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TsNaive(pa.DataFrameModel):
    t: pl.Datetime


class TsUTC(pa.DataFrameModel):
    t: pl.Datetime(time_zone="UTC")


class Mixed(pa.DataFrameModel):
    a: pl.Datetime
    b: pl.Datetime(time_zone="UTC")
    c: pl.Datetime(time_zone="Asia/Tokyo")


class Span(pa.DataFrameModel):
    span: pl.Duration


class Flag(pa.DataFrameModel):
    same: bool


def stack(naive: DataFrame[TsNaive], utc: DataFrame[TsUTC]) -> DataFrame[TsNaive]:
    return pl.concat([naive, utc], how="vertical")


def elapsed(df: DataFrame[Mixed]) -> DataFrame[Span]:
    return df.select(span=pl.col("b") - pl.col("a"))


def compare_zones(df: DataFrame[Mixed]) -> DataFrame[Flag]:
    return df.select(same=pl.col("b") == pl.col("c"))
