"""null_count — same column names, every dtype UInt32, one row (issue #74).

Probed (polars 1.41.2, identical on 1.37.0): each column maps to a
non-null UInt32 tally of its nulls, on both DataFrame and LazyFrame.
False-negative twin: ``invalid/null_count_wrong_dtype``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, LazyFrame


class Events(pa.DataFrameModel):
    id: int
    label: str = pa.Field(nullable=True)
    at: pl.Datetime


class NullTally(pa.DataFrameModel):
    id: pl.UInt32
    label: pl.UInt32
    at: pl.UInt32

    class Config:
        strict = True


def tally(data: DataFrame[Events]) -> DataFrame[NullTally]:
    """Eager: every column becomes a non-null UInt32 of the same name."""
    return data.null_count()


def tally_lazy(data: LazyFrame[Events]) -> LazyFrame[NullTally]:
    """Lazy: same mapping, laziness preserved."""
    return data.null_count()
