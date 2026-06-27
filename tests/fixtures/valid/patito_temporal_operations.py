"""Regression for #120: a Patito ``datetime``/``timedelta`` acceptance group
(introduced in #119) must be recognized by the ``.dt`` accessor and temporal
arithmetic, not just the return-type check.

The group collapses to its canonical dtype the moment a column reference feeds
expression inference (``infer_col``), so ``.dt.*`` and ``datetime - datetime``
type-check on group columns exactly like on concrete temporal columns.

- ``a.dt.year()`` -> Int32 (satisfies ``int``)
- ``a - b`` -> Duration (satisfies ``timedelta``)
- ``dur.dt.total_seconds()`` -> Int64 (satisfies ``int``)
"""

from __future__ import annotations

import datetime as dt

import patito as pt
import polars as pl


class In(pt.Model):
    a: dt.datetime
    b: dt.datetime
    dur: dt.timedelta


class Out(pt.Model):
    y: int
    delta: dt.timedelta
    secs: int


def temporal_ops(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.select(
        y=pl.col("a").dt.year(),
        delta=pl.col("a") - pl.col("b"),
        secs=pl.col("dur").dt.total_seconds(),
    )
