"""Regression for #118: accessing a non-existent field of a nested-model
``Struct`` column must be a provable error.

A nested Patito model maps to a CLOSED ``Struct`` of its declared fields
(ADR-0010), so ``.struct.field("nope")`` is a guaranteed runtime miss — the
same ``pple-column-not-found`` the Pandera frontend raises.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Inner(pt.Model):
    x: int
    name: str


class S(pt.Model):
    s: Inner  # -> Struct({"x": ..., "name": ...})


class Out(pt.Model):
    x: int


def pick_missing(df: pt.DataFrame[S]) -> pt.DataFrame[Out]:
    return df.select(x=pl.col("s").struct.field("nope"))
