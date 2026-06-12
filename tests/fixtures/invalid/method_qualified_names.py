"""Methods report class-qualified names (user report 2026-06-12).

Two classes carry a same-named ``process`` method; the failing one must
be attributed to ``Pipeline.process`` with its own line number — bare
names used to render both as ``process`` and the flat name->line table
let the last definition win, pointing the FAIL at the wrong class.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int

    class Config:
        strict = True
        coerce = True


class Pipeline:
    def process(self, df: DataFrame[S]) -> DataFrame[S]:
        return df.with_columns(pl.col("missing") * 2)  # WRONG: no such column


class Other:
    def process(self, df: DataFrame[S]) -> DataFrame[S]:
        return df
