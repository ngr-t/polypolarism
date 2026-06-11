"""cumulative + over + rolling — window-style expressions are typed.

The return annotation pins the precise output dtypes (probed, polars 1.41):
``cum_sum(Float64)`` -> Float64, ``mean().over()`` -> Float64, and
``rolling_mean`` -> Float64. ``sales_rolling_mean`` is declared nullable
because rolling windows yield leading nulls at runtime (rows before
``window_size`` is reached), which is the runtime-correct declaration.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    ts: pl.Datetime
    sales: pl.Float64


class Windowed(pa.DataFrameModel):
    region: str
    ts: pl.Datetime
    sales: pl.Float64
    sales_running: pl.Float64
    sales_region_mean: pl.Float64
    sales_rolling_mean: pl.Float64 = pa.Field(nullable=True)

    class Config:
        strict = True


def windowed(df: DataFrame[S]) -> DataFrame[Windowed]:
    return df.with_columns(
        pl.col("sales").cum_sum().alias("sales_running"),
        pl.col("sales").mean().over("region").alias("sales_region_mean"),
        pl.col("sales").rolling_mean(window_size=3).alias("sales_rolling_mean"),
    )
