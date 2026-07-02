"""Invalid: int_range dtype now tracked, wrong declarations rejected (issue #140).

``pl.int_range(...)`` is Int64 (or the explicit ``dtype=``); the dtype was
untracked, so a wrong declaration passed silently (false negative). It is now
modeled precisely and the mismatch is a ``pple-return-type`` error.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class XIn(pa.DataFrameModel):
    x: int


class StrOut(pa.DataFrameModel):
    r: str  # wrong: int_range is Int64


class I64Out(pa.DataFrameModel):
    r: int  # wrong: int_range(dtype=UInt32) is UInt32


def int_range_declared_str(df: DataFrame[XIn]) -> DataFrame[StrOut]:
    return df.select(r=pl.int_range(0, pl.len()))


def int_range_dtype_mismatch(df: DataFrame[XIn]) -> DataFrame[I64Out]:
    return df.select(r=pl.int_range(0, pl.len(), dtype=pl.UInt32))
