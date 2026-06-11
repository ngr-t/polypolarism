"""Valid: UDF expressions inside ``group_by().agg()`` (issue #86).

In grouped context a non-aggregating UDF result is implicitly
list-aggregated (probed identical on polars 1.41.2 and 1.37.0):

- ``map_elements(f, return_dtype=T)`` -> ``List(T)``;
- ``map_batches(f, return_dtype=T)`` -> ``List(T)``, but
  ``returns_scalar=True`` keeps the scalar ``T`` (the
  custom-aggregation-function pattern);
- ``pl.map_groups(..., return_dtype=T)`` follows the same matrix;
- a native aggregation chained after the UDF reduces as usual.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


def double(v: float) -> float:
    return v * 2.0


def doubled_series(s: pl.Series) -> pl.Series:
    return s * 2.0


def series_mean(s: pl.Series) -> float:
    return s.mean()


def first_series_doubled(series_list: list[pl.Series]) -> pl.Series:
    return series_list[0] * 2.0


class Src(pa.DataFrameModel):
    g: str
    v: int


class ListedOut(pa.DataFrameModel):
    g: str
    x: pl.List(pl.Float64) = pa.Field()

    class Config:
        strict = True


@pa.check_types
def map_elements_listed(df: DataFrame[Src]) -> DataFrame[ListedOut]:
    return df.group_by("g").agg(x=pl.col("v").map_elements(double, return_dtype=pl.Float64))


@pa.check_types
def map_batches_listed(df: DataFrame[Src]) -> DataFrame[ListedOut]:
    return df.group_by("g").agg(x=pl.col("v").map_batches(doubled_series, return_dtype=pl.Float64))


@pa.check_types
def pl_map_groups_listed(df: DataFrame[Src]) -> DataFrame[ListedOut]:
    return df.group_by("g").agg(
        x=pl.map_groups(exprs=["v"], function=first_series_doubled, return_dtype=pl.Float64)
    )


class ScalarOut(pa.DataFrameModel):
    g: str
    x: pl.Float64

    class Config:
        strict = True


@pa.check_types
def map_batches_custom_aggregation(df: DataFrame[Src]) -> DataFrame[ScalarOut]:
    return df.group_by("g").agg(
        x=pl.col("v").map_batches(series_mean, return_dtype=pl.Float64, returns_scalar=True)
    )


@pa.check_types
def native_agg_after_udf(df: DataFrame[Src]) -> DataFrame[ScalarOut]:
    return df.group_by("g").agg(x=pl.col("v").map_elements(double, return_dtype=pl.Float64).sum())
