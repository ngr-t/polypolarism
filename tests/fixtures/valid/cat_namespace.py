"""Valid contrast for issue #54: ``.cat`` methods on Categorical / Enum columns.

Probed return dtypes on polars 1.41.2 (identical for Categorical and
Enum receivers): ``cat.get_categories()`` -> String (length-changing, no
nulls even for a nullable receiver), ``cat.len_chars()`` -> UInt32,
``cat.starts_with(...)`` -> Boolean.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    color: pl.Categorical
    size: pl.Enum


class Cats(pa.DataFrameModel):
    color: str


def color_categories(df: DataFrame[In]) -> DataFrame[Cats]:
    return df.select(pl.col("color").cat.get_categories())


class Derived(pa.DataFrameModel):
    name_len: pl.UInt32
    is_xl: bool


def derived(df: DataFrame[In]) -> DataFrame[Derived]:
    return df.select(
        name_len=pl.col("color").cat.len_chars(),
        is_xl=pl.col("size").cat.starts_with("x"),
    )
