"""cumulative + over + rolling — window-style expressions are typed."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    ts: pl.Datetime
    sales: pl.Float64


def windowed(df: DataFrame[S]):
    return df.with_columns(
        pl.col("sales").cum_sum().alias("sales_running"),
        pl.col("sales").mean().over("region").alias("sales_region_mean"),
        pl.col("sales").rolling_mean(window_size=3).alias("sales_rolling_mean"),
    )
