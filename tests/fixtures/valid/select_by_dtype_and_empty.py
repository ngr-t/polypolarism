"""Valid: select(pl.col(<dtype>)) and zero-arg select() (issue #142).

``df.select(pl.col(pl.Int64))`` is dtype-based selection — the matching column
set is derivable from the schema, like ``cs.by_dtype``. ``df.select()`` is the
documented zero-column select, yielding the provably empty frame. Both used to
hard-fail ``Could not infer return type`` with no diagnostic.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TwoInts(pa.DataFrameModel):
    a: int
    b: int
    s: str


class IntsOnly(pa.DataFrameModel):
    a: int
    b: int

    class Config:
        strict = True


class Empty(pa.DataFrameModel):
    class Config:
        strict = True


def select_by_dtype(df: DataFrame[TwoInts]) -> DataFrame[IntsOnly]:
    return df.select(pl.col(pl.Int64))


def zero_column_select(df: DataFrame[TwoInts]) -> DataFrame[Empty]:
    return df.select()
