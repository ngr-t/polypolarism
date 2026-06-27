"""Patito's bare model-qualified frame annotations (ADR-0010 follow-up #2):
``Model.DataFrame`` / ``Model.LazyFrame`` (no subscript) resolve to the
model's FrameType, the same as ``pt.DataFrame[Model]``.

``In.LazyFrame`` also carries the laziness, so ``.lazy()`` on a
``DataFrame``-annotated input satisfies a ``LazyFrame``-annotated return.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class In(pt.Model):
    id: int
    name: str


class Out(pt.Model):
    id: int
    name: str
    score: float


def add_score(df: In.DataFrame) -> Out.DataFrame:
    return df.with_columns(score=pl.col("id") * 1.0)


def to_lazy(df: In.DataFrame) -> In.LazyFrame:
    return df.lazy()
