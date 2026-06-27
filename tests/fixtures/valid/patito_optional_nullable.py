"""Patito ``Optional[T]`` / ``T | None`` makes the VALUE nullable while the
column stays required — the inverse of Pandera's ``Optional`` (ADR-0010).

Probed: ``Model.nullable_columns`` contains exactly the ``Optional`` fields,
and every column is required. A non-null ``T`` column satisfies a nullable
slot (the one-way ``T <: Nullable[T]`` widening), so passing a non-null
``name`` into the nullable output ``note`` slot is valid.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class In(pt.Model):
    id: int
    name: str  # non-null


class Out(pt.Model):
    id: int
    note: str | None  # nullable value, column still required


def widen(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.select("id", pl.col("name").alias("note"))
