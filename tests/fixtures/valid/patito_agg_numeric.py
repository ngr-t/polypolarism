"""Regression for #116: aggregations over Patito ``int`` / ``float`` columns
must type-check.

``int`` / ``float`` map to a ``DataTypeGroup`` (ADR-0010); the aggregation
numeric guard must recognise a numeric group (via its canonical
representative) so ``sum`` / ``mean`` / ``min`` / ``max`` / ``std`` /
``product`` over them are not falsely rejected.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class S(pt.Model):
    g: str
    i: int
    f: float


class Out(pt.Model):
    g: str
    si: int
    sf: float


def agg_sum(df: pt.DataFrame[S]) -> pt.DataFrame[Out]:
    return df.group_by("g").agg(si=pl.col("i").sum(), sf=pl.col("f").sum())
