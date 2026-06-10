"""Valid: probed dtypes for ``over(mapping_strategy=...)`` and temporal
``diff()`` (issues #45, #46).

Ground truth (polars 1.41.2):
- ``pl.col("a").over("g", mapping_strategy="join")`` -> ``List(Int64)``
  (each partition's values gathered into a list per row); an aggregation
  under "join" broadcasts its scalar unchanged (``sum()`` -> Int64).
- ``mapping_strategy="explode"`` and the default "group_to_rows" keep the
  element dtype.
- ``pl.col(date).diff()`` -> ``Duration`` with a null head slot;
  ``UInt32.diff()`` widens to ``Int64``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    g: str
    d: pl.Date
    u: pl.UInt32


class OverJoinOut(pa.DataFrameModel):
    o: pl.List(pl.Int64) = pa.Field()

    class Config:
        strict = True


def over_join_is_list(df: DataFrame[In]) -> DataFrame[OverJoinOut]:
    return df.select(o=pl.col("a").over("g", mapping_strategy="join"))


class OverScalarOut(pa.DataFrameModel):
    o: int

    class Config:
        strict = True


def over_join_aggregation_broadcasts(df: DataFrame[In]) -> DataFrame[OverScalarOut]:
    return df.select(o=pl.col("a").sum().over("g", mapping_strategy="join"))


def over_explode_keeps_element_dtype(df: DataFrame[In]) -> DataFrame[OverScalarOut]:
    return df.select(o=pl.col("a").over("g", mapping_strategy="explode"))


class DiffOut(pa.DataFrameModel):
    dd: pl.Duration = pa.Field(nullable=True)
    du: pl.Int64 = pa.Field(nullable=True)

    class Config:
        strict = True


def temporal_and_unsigned_diff(df: DataFrame[In]) -> DataFrame[DiffOut]:
    return df.select(
        dd=pl.col("d").diff(),
        du=pl.col("u").diff(),
    )
