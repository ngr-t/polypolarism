"""Invalid: wrong dtypes declared for cumulative / over / rolling outputs.

False-negative twin of ``valid/m5_window_and_rolling.py``: the same three
window-style expressions with wrong declared dtypes must fail — this
guards the window output-dtype rules against degrading to Unknown.

Probed (polars 1.41 + pandera): the actual output dtypes are
``cum_sum(Int64)`` -> Int64, ``mean(Int64).over()`` -> Float64,
``rolling_mean(Float64)`` -> Float64; every wrong declaration below is
rejected by pandera at runtime.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int


class CumWrong(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int
    running: str  # WRONG: cum_sum on Int64 stays Int64


def cum_wrong_dtype(df: DataFrame[S]) -> DataFrame[CumWrong]:
    return df.with_columns(running=pl.col("qty").cum_sum())


class OverWrong(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int
    region_mean: int  # WRONG: mean() over Int64 yields Float64


def over_wrong_dtype(df: DataFrame[S]) -> DataFrame[OverWrong]:
    return df.with_columns(region_mean=pl.col("qty").mean().over("region"))


class RollWrong(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int
    roll: int = pa.Field(nullable=True)  # WRONG: rolling_mean yields Float64


def rolling_wrong_dtype(df: DataFrame[S]) -> DataFrame[RollWrong]:
    return df.with_columns(roll=pl.col("sales").rolling_mean(window_size=3))
