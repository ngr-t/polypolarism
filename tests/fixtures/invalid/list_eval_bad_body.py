"""Invalid: incompatible expression inside ``list.eval()`` (issue #44).

``pl.col("v").list.eval(pl.element() + pl.lit("x"))`` on a List(Int64)
column adds a String to each Int64 element — polars raises
``InvalidOperationError`` at runtime. polypolarism binds ``pl.element()``
to the list's inner dtype and flags the body statically as pple-incompatible-operands.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: pl.List(pl.Int64) = pa.Field()


class Out(pa.DataFrameModel):
    v: pl.List(pl.Int64) = pa.Field()


def bump(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(pl.col("v").list.eval(pl.element() + pl.lit("x")))
