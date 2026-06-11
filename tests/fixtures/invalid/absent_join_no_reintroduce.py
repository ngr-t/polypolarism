"""A join that does not carry the dropped name keeps it absent (issue #78).

``drop("a")`` then a join whose other side provably lacks ``a`` cannot
resurrect the name — the reference is still a guaranteed
ColumnNotFoundError.

False-positive twin: ``valid/absent_reintroduce_join`` (the other side
pinning ``a`` does clear the mark).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KeyedOnly(pa.DataFrameModel):
    k: pl.Int64
    z: str

    class Config:
        strict = True
        coerce = True


def join_does_not_reintroduce(df: pl.DataFrame, g: DataFrame[KeyedOnly]) -> pl.DataFrame:
    return df.drop("a").join(g, on="k").select(pl.col("a"))  # WRONG: 'a' is still absent
