"""Temporal receivers through mean/median/quantile/sum/std (issue #85).

Probed (polars 1.41.2), identical in select and group_by().agg() contexts:

- ``mean``/``median``/``quantile`` preserve the receiver dtype EXACTLY —
  Datetime keeps its time unit AND tz, Duration its unit, Time stays Time —
  and return a naive ``Datetime[us]`` for a Date receiver.
- ``sum``/``std`` on Duration preserve the unit; std is null on singleton
  groups (ddof=1, issue #60), so it must be declared nullable.

The false-negative twin is ``invalid/temporal_reductions_wrong`` (which also
pins the genuinely-invalid temporal cells: var on Duration, sum/std on
Datetime).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Sessions(pa.DataFrameModel):
    device: str
    started_at: pl.Datetime("ms", time_zone="UTC")
    seen_on: pl.Date
    wake_at: pl.Time
    duration: pl.Duration("ns")


class PerDevice(pa.DataFrameModel):
    device: str
    mean_started: pl.Datetime("ms", time_zone="UTC")
    median_wake: pl.Time
    q_duration: pl.Duration("ns")

    class Config:
        strict = True


def agg_temporal_mean_like(df: DataFrame[Sessions]) -> DataFrame[PerDevice]:
    # mean/median/quantile preserve the receiver instance (probed 1.41.2):
    # the ms unit and the UTC tz survive the grouped mean, Time stays Time,
    # Duration keeps ns. Singleton groups yield a value, not null.
    return df.group_by("device").agg(
        pl.col("started_at").mean().alias("mean_started"),
        pl.col("wake_at").median().alias("median_wake"),
        pl.col("duration").quantile(0.5).alias("q_duration"),
    )


class Midpoints(pa.DataFrameModel):
    mid_seen: pl.Datetime  # bare form — polars' "us" default
    mean_started: pl.Datetime("ms", time_zone="UTC")

    class Config:
        strict = True


def select_temporal_means(df: DataFrame[Sessions]) -> DataFrame[Midpoints]:
    # Date is the one transforming receiver: its mean/median/quantile
    # return a naive Datetime[us] (probed 1.41.2 — the midpoint of dates
    # is not a date).
    return df.select(
        pl.col("seen_on").mean().alias("mid_seen"),
        pl.col("started_at").mean().alias("mean_started"),
    )


class DurationStats(pa.DataFrameModel):
    device: str
    total: pl.Duration("ns")
    spread: pl.Duration("ns") = pa.Field(nullable=True)

    class Config:
        strict = True


def agg_duration_sum_std(df: DataFrame[Sessions]) -> DataFrame[DurationStats]:
    # Duration is the only temporal accepted by sum/std (probed 1.41.2);
    # both preserve the unit. std keeps the ddof=1 singleton-group null
    # (issue #60), hence the nullable declaration.
    return df.group_by("device").agg(
        pl.col("duration").sum().alias("total"),
        pl.col("duration").std().alias("spread"),
    )
