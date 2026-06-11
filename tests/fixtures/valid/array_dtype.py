"""Valid contrast for issue #53: ``.arr`` methods on a real Array column.

Probed return dtypes on polars 1.41.2: ``arr.sum()`` on Array(Int64, n)
-> Int64, ``arr.len()`` -> UInt32, ``arr.unique()`` -> List(Int64) (the
fixed width is lost), ``arr.sort()`` keeps the Array dtype, ``arr.mean()``
-> Float64. ``explode`` on an Array column yields its element dtype, and
casting Array -> List is always allowed.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    q: pl.Array(pl.Int64, 4) = pa.Field()


class Stats(pa.DataFrameModel):
    id: int
    total: int
    width: pl.UInt32
    distinct: pl.List(pl.Int64) = pa.Field()
    ordered: pl.Array(pl.Int64, 4) = pa.Field()
    avg: float


def array_stats(df: DataFrame[In]) -> DataFrame[Stats]:
    return df.select(
        "id",
        total=pl.col("q").arr.sum(),
        width=pl.col("q").arr.len(),
        distinct=pl.col("q").arr.unique(),
        ordered=pl.col("q").arr.sort(),
        avg=pl.col("q").arr.mean(),
    )


class Exploded(pa.DataFrameModel):
    id: int
    q: int


def explode_array(df: DataFrame[In]) -> DataFrame[Exploded]:
    return df.explode("q")


class AsList(pa.DataFrameModel):
    id: int
    q: pl.List(pl.Int64) = pa.Field()


def array_to_list(df: DataFrame[In]) -> DataFrame[AsList]:
    return df.with_columns(pl.col("q").cast(pl.List(pl.Int64)))
