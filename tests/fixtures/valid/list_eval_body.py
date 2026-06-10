"""``list.eval(...)`` bodies type-check with ``pl.element()`` bound to the
list's inner dtype (issue #44): ``eval(pl.element() * 2)`` on List(Int64)
stays List(Int64)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: pl.List(pl.Int64) = pa.Field()


class Out(pa.DataFrameModel):
    v: pl.List(pl.Int64) = pa.Field()

    class Config:
        strict = True


def double(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(pl.col("v").list.eval(pl.element() * 2))
