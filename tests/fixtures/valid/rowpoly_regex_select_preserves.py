"""A regex select(pl.col("^.*$")) preserves the @rowpoly row variable (#111).

``select(pl.col("^.*$"))`` is a row-variable-preserving idiom (an alternative
to ``pl.all()``): the regex matches every column, including the caller's
arbitrary extras, so the row variable survives. The preservation check
skolemizes the row variable by injecting a sentinel column; the ``^.*$`` regex
resolves to the full column set (sentinel included), so no provable drop is
found and the decorator is accepted — no false PLY043. Contrast the
``^id$``-style fixed regex, which would drop the extras and IS rejected.
"""

from __future__ import annotations

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
    return df.select(pl.col("^.*$")).with_columns(score=pl.col("id").cast(pl.Float64))
