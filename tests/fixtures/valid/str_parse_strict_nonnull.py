"""Valid: str parsing with the default strict=True stays non-null (issue #129).

Only a literal ``strict=False`` injects nulls; ``str.to_integer()`` /
``str.to_datetime()`` with the default ``strict=True`` keep the current non-null
result (they raise on an unparseable string rather than nulling it), so a
non-null declaration must still type-check. Counterpart to
``invalid/str_parse_non_strict_nullable`` — guards against over-nullifying the
default form.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class StrIn(pa.DataFrameModel):
    num: str
    stamp: str


class IntOut(pa.DataFrameModel):
    val: int  # non-null


class DtOut(pa.DataFrameModel):
    val: pl.Datetime  # non-null


def to_integer_strict_default(df: DataFrame[StrIn]) -> DataFrame[IntOut]:
    return df.select(val=pl.col("num").str.to_integer())


def to_datetime_strict_default(df: DataFrame[StrIn]) -> DataFrame[DtOut]:
    return df.select(val=pl.col("stamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))
