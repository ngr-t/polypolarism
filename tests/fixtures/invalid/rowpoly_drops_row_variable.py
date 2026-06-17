"""A @rowpoly helper whose body drops the row variable (issue C-14 Tier 4, PLY043).

``@rowpoly("R")`` promises the helper preserves the caller's extra columns,
so a call site threads them into the result (Tier 3). This body breaks that
promise: the explicit ``select("id")`` produces a closed frame that keeps
only ``id``, dropping any caller extras before ``score`` is added. The
preservation check skolemizes the row variable and catches the provable drop
as PLY043.

Static-only: the property is relative to the caller (whether the helper kept
columns the *caller* supplied), so Pandera cannot check it at runtime — an
input synthesized from ``InId`` alone has no extras to drop. The
runtime-differential harness SKIPs this fixture for that reason.

Contrast ``valid/rowpoly_preserves_row_variable.py``, where ``with_columns``
keeps every input column and the same decorator is accepted.
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
    # select("id") drops the caller's extras -> the row variable is not preserved.
    return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))
