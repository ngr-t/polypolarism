"""Invalid: mixing tz-aware and tz-naive Datetime columns (issue #50).

Probed on polars 1.41.2:
- ``pl.concat([naive, utc], how="vertical")`` raises SchemaError
  ("type Datetime('μs', 'UTC') is incompatible with expected type
  Datetime('μs')") -> PLY020.
- ``aware - naive`` raises SchemaError ("failed to determine supertype
  of datetime[μs, UTC] and datetime[μs]") -> PLY009.
- Comparing two different time zones raises SchemaError too -> PLY009.
- ``str.to_datetime`` with an offset directive in the format (here the
  ``%:z`` variant) yields ``Datetime[UTC]``, never a naive Datetime —
  false-negative twin of ``valid/tz_same_ops`` (the ``parse`` function).
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


class Raw(pa.DataFrameModel):
    stamp: str


class ParsedNaive(pa.DataFrameModel):
    stamp: pl.Datetime  # WRONG: an offset directive in the format yields Datetime[UTC]


def parse_offset(df: DataFrame[Raw]) -> DataFrame[ParsedNaive]:
    return df.with_columns(pl.col("stamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%:z"))
