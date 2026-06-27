"""Patito frontend (ADR-0010): ``pt.DataFrame[Model]`` params and returns
type-check like Pandera's ``DataFrame[Schema]``.

The annotation resolves through the SAME detector (the trailing ``DataFrame``
attribute + the subscript model name), once the Patito model is registered.
The body must produce exactly the declared output columns of a strict Patito
model.
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


def add_score(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.with_columns(score=pl.col("id") * 1.0)
