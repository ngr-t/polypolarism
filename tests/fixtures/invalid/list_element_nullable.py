"""Invalid: list element accessors that can yield null, declared non-null (issue #128).

Under ``List(T)`` an empty (or too-short) sub-list produces a null element, so:

- ``list.first()`` / ``list.last()`` — an empty sub-list is always possible,
- ``list.get(i, null_on_oob=True)`` — an out-of-bounds index becomes null.

Returning any of these into a non-null declared field is unsound (pandera
rejects the null at validation time). ``arr.get(i, null_on_oob=True)`` on a
fixed-width array has the same out-of-bounds hole.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    xs: pl.List(pl.Int64)


class Out(pa.DataFrameModel):
    x: int  # non-null declared


def get_null_on_oob(df: DataFrame[ListIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("xs").list.get(0, null_on_oob=True))


def first_elem(df: DataFrame[ListIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("xs").list.first())


def last_elem(df: DataFrame[ListIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("xs").list.last())


class ArrIn(pa.DataFrameModel):
    ys: pl.Array(pl.Int64, 3)


def arr_get_null_on_oob(df: DataFrame[ArrIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("ys").arr.get(5, null_on_oob=True))
