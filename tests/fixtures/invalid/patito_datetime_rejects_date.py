"""Patito's datetime acceptance group is class-scoped (#119): a ``Date`` column
does NOT satisfy a ``datetime.datetime`` field — only a ``Datetime`` of any
unit / time zone does (probed: patito rejects a Date column there).

False-positive twin: ``valid/patito_temporal_units``.
"""

from __future__ import annotations

import datetime as dt

import patito as pt
import polars as pl


class In(pt.Model):
    ts: dt.datetime


class Out(pt.Model):
    ts: dt.datetime


def to_date(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.select(ts=pl.col("ts").cast(pl.Date))
