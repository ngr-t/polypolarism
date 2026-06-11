"""group_by_dynamic + join_asof — time-window groupby and asof join.

The return annotation pins the precise output (probed, polars 1.41):
``group_by_dynamic`` keeps the time axis and ``mean(Float64)`` stays
Float64; ``join_asof`` behaves like a left join, so the right-side quote
columns are nullable (a bucket earlier than every quote matches nothing).
"""

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


class HourlyQuote(pa.DataFrameModel):
    ts: pl.Datetime
    avg_px: pl.Float64
    bid: pl.Float64 = pa.Field(nullable=True)
    ask: pl.Float64 = pa.Field(nullable=True)

    class Config:
        strict = True


def hourly_with_quote(
    trades: DataFrame[Trade],
    quotes: DataFrame[Quote],
) -> DataFrame[HourlyQuote]:
    hourly = trades.group_by_dynamic("ts", every="1h").agg(
        pl.col("px").mean().alias("avg_px"),
    )
    return hourly.join_asof(quotes, on="ts")
