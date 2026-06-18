"""``list.sum()`` over string elements raises at runtime (issue #55, pple-non-numeric-operand).

Probed (polars 1.41.2): ``pl.col(List(str)).list.sum()`` raises
``InvalidOperationError: `sum` operation not supported for dtype `str```.
Regression guard for the #53 rework, which let this degrade silently to
Unknown — BOTH wrong declarations below (int and str) must fail.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Tags(pa.DataFrameModel):
    id: int
    tags: pl.List(pl.Utf8) = pa.Field()


class WrongInt(pa.DataFrameModel):
    id: int
    total: int


class WrongStr(pa.DataFrameModel):
    id: int
    total: str


@pa.check_types
def bug_sum_tags_as_int(df: DataFrame[Tags]) -> DataFrame[WrongInt]:
    return df.select("id", total=pl.col("tags").list.sum())


@pa.check_types
def bug_sum_tags_as_str(df: DataFrame[Tags]) -> DataFrame[WrongStr]:
    return df.select("id", total=pl.col("tags").list.sum())
