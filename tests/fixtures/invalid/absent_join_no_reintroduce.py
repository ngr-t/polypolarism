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


# The default is the runtime witness: a frame that really carries 'a'
# and the join key, so the drop -> join -> select chain executes for real
# in the differential harness (bare params are not synthesizable).
_LEFT = pl.DataFrame({"k": [1, 2], "a": [10, 20]})


def join_does_not_reintroduce(g: DataFrame[KeyedOnly], df: pl.DataFrame = _LEFT) -> pl.DataFrame:
    return df.drop("a").join(g, on="k").select(pl.col("a"))  # WRONG: 'a' is still absent
