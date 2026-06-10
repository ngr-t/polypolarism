"""Invalid: cast to a structurally impossible target dtype (issue #34).

``pl.col("v").cast(pl.Int64)`` where ``v: List(Int64)`` — polars raises
``InvalidOperationError: cannot cast List type`` at runtime, even with
``strict=False``. polypolarism flags it statically as PLY013, and the
output degrades to Unknown instead of silently matching the declared
``v: int``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    v: pl.List(pl.Int64) = pa.Field()


class VInt(pa.DataFrameModel):
    v: int

    class Config:
        strict = True


def bug_cast_list_to_int(df: DataFrame[ListIn]) -> DataFrame[VInt]:
    return df.select(v=pl.col("v").cast(pl.Int64))
