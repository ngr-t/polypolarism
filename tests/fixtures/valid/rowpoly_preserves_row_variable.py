"""A @rowpoly helper that correctly preserves the row variable (C-14 Tier 4).

``with_columns`` keeps every existing column, so the caller's extra columns
flow through unchanged and the row variable is preserved. The preservation
check (which skolemizes the row variable and looks for a provable drop) finds
none, so the decorator is accepted.

Invalid twin: ``invalid/rowpoly_drops_row_variable.py``, where an explicit
``select`` drops the extras and the same decorator is rejected (PLY043).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False


@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))
