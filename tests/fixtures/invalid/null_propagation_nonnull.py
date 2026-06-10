"""Invalid: expression over a nullable column declared non-nullable (issue #18).

``x`` is ``pa.Field(nullable=True)``; nulls propagate through
``pl.col("a") + pl.col("x")``, so the non-nullable ``z`` declaration is
wrong — pandera would reject the nulls at runtime.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class WithNullable(pa.DataFrameModel):
    a: int
    x: int = pa.Field(nullable=True)


class WrongSum(pa.DataFrameModel):
    z: int  # wrong: a + nullable x is Nullable[Int64]


def combine(df: DataFrame[WithNullable]) -> DataFrame[WrongSum]:
    return df.select(z=pl.col("a") + pl.col("x"))
