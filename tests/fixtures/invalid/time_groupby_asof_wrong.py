"""Invalid: wrong declarations for group_by_dynamic and join_asof.

False-negative twin of ``valid/m5_time_groupby_and_asof.py``: the same
time-window pipeline with wrong declarations must fail.

Probed (polars 1.41 + pandera):

- ``mean(Float64)`` in a ``group_by_dynamic`` agg yields Float64, so
  ``avg_px: int`` fails at runtime;
- ``join_asof`` is left-join-like: a bucket earlier than every quote gets
  null ``bid``/``ask``, so a non-nullable ``bid`` fails at runtime;
- asof keys must share a dtype: joining Datetime onto Date raises
  SchemaError inside polars itself.
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


class QuoteByDate(pa.DataFrameModel):
    ts: pl.Date
    bid: pl.Float64
    ask: pl.Float64


class HourlyWrong(pa.DataFrameModel):
    ts: pl.Datetime
    avg_px: int  # WRONG: mean() over Float64 stays Float64


def dynamic_agg_wrong_dtype(trades: DataFrame[Trade]) -> DataFrame[HourlyWrong]:
    return trades.group_by_dynamic("ts", every="1h").agg(
        pl.col("px").mean().alias("avg_px"),
    )


class AsofNonNull(pa.DataFrameModel):
    ts: pl.Datetime
    avg_px: pl.Float64
    bid: pl.Float64  # WRONG: join_asof makes right-side columns nullable
    ask: pl.Float64 = pa.Field(nullable=True)


def asof_nonnullable_declared(
    trades: DataFrame[Trade],
    quotes: DataFrame[Quote],
) -> DataFrame[AsofNonNull]:
    hourly = trades.group_by_dynamic("ts", every="1h").agg(
        pl.col("px").mean().alias("avg_px"),
    )
    return hourly.join_asof(quotes, on="ts")


class HourlyQuote(pa.DataFrameModel):
    ts: pl.Datetime
    avg_px: pl.Float64
    bid: pl.Float64 = pa.Field(nullable=True)
    ask: pl.Float64 = pa.Field(nullable=True)


def asof_key_dtype_mismatch(
    trades: DataFrame[Trade],
    quotes: DataFrame[QuoteByDate],
) -> DataFrame[HourlyQuote]:
    hourly = trades.group_by_dynamic("ts", every="1h").agg(
        pl.col("px").mean().alias("avg_px"),
    )
    # WRONG: asof key is Datetime on the left but Date on the right
    return hourly.join_asof(quotes, on="ts")
