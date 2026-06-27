"""Regression for #117: a non-null column satisfies an ``Optional[T]`` field.

Patito maps ``Optional[T]`` to nullable-but-required (ADR-0010). A provably
non-null ``T`` column is a subtype of the nullable slot (one-way
``T <: Nullable[T]`` widening), so returning a non-null ``a`` into a declared
``Optional[int]`` must pass. The reverse (nullable inferred vs non-null
declared) stays rejected.
"""

from __future__ import annotations

from typing import Optional

import patito as pt
import polars as pl


class S(pt.Model):
    a: int


class Out(pt.Model):
    a: Optional[int]  # noqa: UP045 — #117 repro deliberately uses the Optional spelling


def keep(df: pt.DataFrame[S]) -> pt.DataFrame[Out]:
    return df.select(pl.col("a"))
