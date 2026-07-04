"""Patito: ``Model.validate(arg)`` normalizes abstract int/float families in the
argument comparison, exactly like the return boundary (issue #150).

A recomputed numeric column carries a CONCRETE dtype (``pl.col("x") * 2.0`` ->
Float64), but a patito ``x: float`` / ``n: int`` field stays an abstract
acceptance family (``float`` / ``integer``). The return-boundary checker already
normalizes these; the ``validate()`` argument check must too, or it reports a
``SchemaError on every call`` that never happens at runtime (patito ``float`` IS
Float64). Only a recomputed numeric column triggers it — a pass-through column
keeps the model's group and compares equal.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class FModel(pt.Model):
    x: float


class IModel(pt.Model):
    n: int


def ok_validate_float(df: pt.DataFrame[FModel]) -> pt.DataFrame[FModel]:
    return FModel.validate(df.with_columns(x=pl.col("x") * 2.0))  # x -> Float64


def ok_validate_int(df: pt.DataFrame[IModel]) -> pt.DataFrame[IModel]:
    return IModel.validate(df.with_columns(n=pl.col("n") + 1))  # n -> Int64
