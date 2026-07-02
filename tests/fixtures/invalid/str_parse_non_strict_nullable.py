"""Invalid: str parsing with strict=False declared non-null (issue #129, sibling of #125).

``Expr.str.to_integer(strict=False)`` and ``Expr.str.to_datetime(..., strict=False)``
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


def to_integer_non_strict(df: DataFrame[StrIn]) -> DataFrame[IntOut]:
    return df.select(val=pl.col("raw").str.to_integer(strict=False))


def to_datetime_non_strict(df: DataFrame[StrIn]) -> DataFrame[DtOut]:
    return df.select(val=pl.col("raw").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
