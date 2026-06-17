"""An all-columns selector preserves the @rowpoly row variable (C-14 Tier 5).

``select(pl.all())`` (and the ``cs.all()`` spelling) selects every column,
including the caller's arbitrary extras, so the row variable survives. This is
the contrast to ``invalid/rowpoly_drops_row_variable.py``, where an explicit
``select("id")`` of a fixed column set drops them and fires PLY043.

The preservation check skolemizes the row variable by injecting a sentinel
column; an all-columns selector resolves to the full column set (sentinel
included), so the check finds no provable drop and accepts the decorator. This
pins the audited absence of a false positive for all-columns selectors (a fixed
select of named columns is still rejected — see the invalid twin).
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
    return df.select(pl.all()).with_columns(score=pl.col("id").cast(pl.Float64))
