"""Invalid: dt.replace_time_zone DST null policies declared non-null (issue #152).

``replace_time_zone(ambiguous="null")`` maps a fall-back time that occurs twice
to null, and ``replace_time_zone(non_existent="null")`` maps a spring-forward
gap time (which never happens) to null. Both are value-dependent null injection,
like ``cast(strict=False)`` — declaring the result non-null is unsound (pandera
rejects the injected nulls at validation time). The other policy values
(``"raise"`` / ``"earliest"`` / ``"latest"``) keep the result non-null.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TsIn(pa.DataFrameModel):
    amb: pl.Datetime  # holds a DST fall-back (ambiguous) time
    gap: pl.Datetime  # holds a DST spring-forward (non-existent) time


class TzOut(pa.DataFrameModel):
    r: pl.Datetime(time_zone="America/New_York")  # non-null declared


def ambiguous_null(df: DataFrame[TsIn]) -> DataFrame[TzOut]:
    return df.select(r=pl.col("amb").dt.replace_time_zone("America/New_York", ambiguous="null"))


def non_existent_null(df: DataFrame[TsIn]) -> DataFrame[TzOut]:
    return df.select(r=pl.col("gap").dt.replace_time_zone("America/New_York", non_existent="null"))
