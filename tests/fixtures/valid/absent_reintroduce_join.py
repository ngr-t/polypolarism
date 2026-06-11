"""A join's other side clears the absence mark (issue #78 amendment).

``drop("a")`` records ``a`` as provably absent, but joining a frame
whose schema pins ``a`` reintroduces the name — the amendment lists the
join's other side as one of the three reintroduction paths
(``with_columns``, a rename target, a join).

False-negative twin: ``invalid/absent_join_no_reintroduce``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KeyedA(pa.DataFrameModel):
    k: pl.Int64
    a: pl.Int64

    class Config:
        strict = True
        coerce = True


# Executable default for the differential harness (bare params are not
# synthesizable): carries 'a' (to be dropped) and the join key.
_LEFT = pl.DataFrame({"k": [1, 2], "a": [10, 20]})


def join_reintroduces_dropped(g: DataFrame[KeyedA], df: pl.DataFrame = _LEFT) -> pl.DataFrame:
    return df.drop("a").join(g, on="k").select(pl.col("a"))
