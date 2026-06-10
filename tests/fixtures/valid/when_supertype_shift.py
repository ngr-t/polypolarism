"""Valid: probed supertype inference for when/then/otherwise, unpivot and
shift(fill_value=...) (issues #40, #41, #43).

Ground truth (polars 1.41.2):
- ``when(a > 0).then(pl.lit(1)).otherwise(pl.lit("x"))`` -> String
- ``unpivot(index="id", on=["a", "s"])`` with Int64 + String value
  columns -> ``value: String``
- ``shift(1, fill_value=0)`` on non-null Int64 -> non-null Int64
  (``[0, 1, 2]``, null_count 0); ``shift(1)`` without a fill stays
  nullable.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    a: int
    s: str


class WithLabel(pa.DataFrameModel):
    id: int
    a: int
    s: str
    label: str


def when_mixed_branches(df: DataFrame[In]) -> DataFrame[WithLabel]:
    return df.with_columns(label=pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x")))


class Long(pa.DataFrameModel):
    id: int
    variable: str
    value: str


def unpivot_mixed_values(df: DataFrame[In]) -> DataFrame[Long]:
    return df.unpivot(index="id", on=["a", "s"])


def shift_with_fill_stays_nonnull(df: DataFrame[In]) -> DataFrame[In]:
    return df.with_columns(pl.col("a").shift(1, fill_value=0))


class WithNullableShift(pa.DataFrameModel):
    id: int
    a: int = pa.Field(nullable=True)
    s: str


def shift_without_fill_is_nullable(df: DataFrame[In]) -> DataFrame[WithNullableShift]:
    return df.with_columns(pl.col("a").shift(1))
