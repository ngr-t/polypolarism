"""Valid fixture: open-left join pins are collision-aware (issue #79).

Joining a closed right frame onto an OPEN left frame: a right column's
unsuffixed pin is conditional — if the left rest happens to carry the
same name, polars suffixes the RIGHT column away and the unsuffixed name
is the left column, dtype unknown. So the pin's dtype is Unknown and a
downstream ``.str`` on it is NOT a provable error (there are runtime
inputs on which it succeeds). Negative knowledge still flows: the name
itself provably exists, and reintroduction clears drop marks.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KZ(pa.DataFrameModel):
    k: pl.Int64
    z: pl.Int64

    class Config:
        strict = True


# The default IS the issue's runtime counterexample: an open-side frame
# that happens to carry z as String. The runtime differential harness
# executes the function with it, demonstrating the code genuinely
# succeeds (the static error would have been a false positive).
_COLLIDING_LEFT = pl.DataFrame({"k": [1, 2, 3], "z": ["x", "y", "w"]})


def join_then_str_is_not_disprovable(
    g: DataFrame[KZ], df: pl.DataFrame = _COLLIDING_LEFT
) -> pl.DataFrame:
    # df carries z: String -> polars suffixes the right z to z_right and
    # pl.col("z") is the LEFT column: this code SUCCEEDS at runtime.
    return df.join(g, on="k").select(pl.col("z").str.to_uppercase())


def reintroduction_after_drop(df: pl.DataFrame) -> pl.DataFrame:
    # with_columns clears the provable absence created by drop (issue #78).
    return df.drop("a").with_columns(a=pl.lit(1)).select(pl.col("a") + 1)
