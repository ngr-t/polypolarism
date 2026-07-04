"""Invalid: str parsing with strict=False declared non-null (issues #129/#151, sibling of #125).

``Expr.str.to_integer(strict=False)``, ``str.to_datetime(..., strict=False)`` and
its two remaining siblings ``str.to_date(strict=False)`` / ``str.to_time(strict=False)``
map every unparseable string to null. The String source is always
value-dependent, so the result is nullable — declaring it non-null is unsound
(pandera rejects the injected nulls at validation time). Same unread-``strict``
family as ``Expr.cast`` / ``DataFrame.cast`` (issue #125).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class StrIn(pa.DataFrameModel):
    raw: str


class IntOut(pa.DataFrameModel):
    val: int  # non-null declared


class DtOut(pa.DataFrameModel):
    val: pl.Datetime  # non-null declared


class DateOut(pa.DataFrameModel):
    val: pl.Date  # non-null declared


class TimeOut(pa.DataFrameModel):
    val: pl.Time  # non-null declared


def to_integer_non_strict(df: DataFrame[StrIn]) -> DataFrame[IntOut]:
    return df.select(val=pl.col("raw").str.to_integer(strict=False))


def to_datetime_non_strict(df: DataFrame[StrIn]) -> DataFrame[DtOut]:
    return df.select(val=pl.col("raw").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))


def to_date_non_strict(df: DataFrame[StrIn]) -> DataFrame[DateOut]:
    return df.select(val=pl.col("raw").str.to_date("%Y-%m-%d", strict=False))


def to_time_non_strict(df: DataFrame[StrIn]) -> DataFrame[TimeOut]:
    return df.select(val=pl.col("raw").str.to_time("%H:%M:%S", strict=False))
