"""Valid: list/arr element accessors that stay non-null (issue #128).

- ``list.get(i)`` with the default ``null_on_oob=False`` raises on OOB rather
  than injecting a null, so the bare element dtype is sound.
- Fixed-width ``arr.first()`` / ``arr.last()`` / ``arr.get(i)`` always hit a
  real element (a width-``n`` array is never empty), so they stay non-null.

Counterpart to ``invalid/list_element_nullable`` — only the empty / OOB forms
must carry ``Nullable``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    xs: pl.List(pl.Int64)


class Out(pa.DataFrameModel):
    x: int  # non-null declared


def get_default_raises(df: DataFrame[ListIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("xs").list.get(0))


class ArrIn(pa.DataFrameModel):
    ys: pl.Array(pl.Int64, 3)


def arr_first(df: DataFrame[ArrIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("ys").arr.first())


def arr_last(df: DataFrame[ArrIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("ys").arr.last())


def arr_get_in_bounds(df: DataFrame[ArrIn]) -> DataFrame[Out]:
    return df.select(x=pl.col("ys").arr.get(0))
