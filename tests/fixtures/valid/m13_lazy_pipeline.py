"""M13: LazyFrame end-to-end pipeline carries the FrameType through."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, LazyFrame


class Event(pa.DataFrameModel):
    user_id: int
    ts: pl.Datetime
    amount: pl.Float64 = pa.Field(nullable=True)


class UserTotal(pa.DataFrameModel):
    user_id: int
    total: pl.Float64
    n_events: pl.UInt32


def per_user_totals(events: LazyFrame[Event]) -> DataFrame[UserTotal]:
    return (
        events.cache()                                            # LF identity
              .filter(pl.col("amount").is_not_null())             # LF identity
              .with_columns(pl.col("amount").fill_null(0.0))      # LF identity
              .group_by("user_id")
              .agg(
                  pl.col("amount").sum().alias("total"),
                  pl.col("amount").count().alias("n_events"),
              )
              .sort("user_id")
              .collect()                                          # LF → DF
    )


def via_collect_async(events: LazyFrame[Event]) -> DataFrame[Event]:
    return events.collect_async()


def streaming_sink(events: LazyFrame[Event]) -> LazyFrame[Event]:
    """sink_* methods terminate the plan at runtime; statically identity."""
    return events.filter(pl.col("user_id") > 0).sink_csv("out.csv", lazy=True)
