"""Branch merge unifies a column's differing nullability instead of dropping it (issue #107).

When a variable is assigned in both branches of an `if` and a column is present
in both but with different nullability (`Float64` vs `Float64?`), the merge
unifies it to the common supertype (`Float64?`) and keeps the column — rather
than dropping it and raising a false `column not found` downstream. The `sum`
of that nullable column is non-nullable (sum drops nulls), so the result
matches the strict, non-nullable schema.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


class W:
    @pa.check_types
    def merge_diff_nullability(self, flag: bool, src: pl.DataFrame) -> DataFrame[KV]:
        if flag:
            acc = src.select([pl.col("k"), pl.col("v").cast(pl.Float64)])
        else:
            acc = src.select([pl.col("k"), pl.col("v").cast(pl.Float64).shift(1)])
        return acc.sort("k").group_by("k").agg(pl.col("v").sum())

    @pa.check_types
    def merge_same_dtype(self, flag: bool, src: pl.DataFrame) -> DataFrame[KV]:
        if flag:
            acc = src.select([pl.col("k"), pl.col("v").cast(pl.Float64)])
        else:
            acc = src.select([pl.col("k"), pl.col("v").cast(pl.Float64)])
        return acc.sort("k").group_by("k").agg(pl.col("v").sum())
