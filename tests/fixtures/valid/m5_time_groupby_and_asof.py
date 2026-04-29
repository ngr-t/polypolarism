"""M5: group_by_dynamic + join_asof — time-window groupby and asof join."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Trade(pa.DataFrameModel):
    ts: pl.Datetime
    symbol: str
    px: pl.Float64


class Quote(pa.DataFrameModel):
    ts: pl.Datetime
    bid: pl.Float64
    ask: pl.Float64


def hourly_with_quote(trades: DataFrame[Trade], quotes: DataFrame[Quote]):
    hourly = trades.group_by_dynamic("ts", every="1h").agg(
        pl.col("px").mean().alias("avg_px"),
    )
    return hourly.join_asof(quotes, on="ts")
