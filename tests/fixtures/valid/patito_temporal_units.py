"""Regression for #119: Patito ``datetime.datetime`` / ``datetime.timedelta``
fields accept ANY time unit (and any/no time zone), modeled as acceptance
groups like ``int`` / ``float`` (ADR-0010).

A ``Datetime[ns]`` / ``Duration[ms]`` column satisfies the slot — mapping them
to an exact ``Datetime[us]`` / ``Duration[us]`` would be a false positive.

False-positive twin: ``invalid/patito_datetime_rejects_date`` (a ``Date`` is
not a ``Datetime``).
"""

from __future__ import annotations

import datetime as dt

import patito as pt
import polars as pl


class In(pt.Model):
    ts: dt.datetime
    d: dt.timedelta


class Out(pt.Model):
    ts: dt.datetime
    d: dt.timedelta


def cast_units(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.with_columns(
        ts=pl.col("ts").cast(pl.Datetime("ns")),
        d=pl.col("d").cast(pl.Duration("ms")),
    )
