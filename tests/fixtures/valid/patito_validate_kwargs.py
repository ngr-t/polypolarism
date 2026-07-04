"""Patito: ``Model.validate(...)`` honours the official semantics-relaxing
kwargs instead of assuming plain strict validation (issue #150).

- ``allow_superfluous_columns=True`` — extras are legal and survive in the
  result (the frame stays open), so a provable extra is not a false SchemaError
  and a later access of the surviving extra is not a false column-not-found.
- ``drop_superfluous_columns=True`` — extras are legal and the result is exactly
  the model's columns (extras dropped), so the extra is not a false SchemaError.
- ``allow_missing_columns=True`` — a passing validate does NOT prove the
  schema's missing columns exist, so it must not narrow the argument to the full
  model shape (which would fabricate a return-type error).

The plain-call proof still fires (``Ks.validate(x)`` with a genuine extra is a
real ``DataFrameValidationError``); only the kwarg-modified semantics change.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Ks(pt.Model):
    k: str


class KV(pt.Model):
    k: str
    v: str


def ok_superfluous_allowed(df: pt.DataFrame[Ks]) -> pl.DataFrame:
    x = df.with_columns(extra=pl.col("k").str.to_uppercase())
    kept = Ks.validate(x, allow_superfluous_columns=True)  # 'extra' survives
    return kept.select("k", "extra")  # accessing the surviving extra is OK


def ok_superfluous_dropped(df: pt.DataFrame[Ks]) -> pt.DataFrame[Ks]:
    x = df.with_columns(extra=pl.col("k").str.to_uppercase())
    return Ks.validate(x, drop_superfluous_columns=True)  # drops 'extra' -> exactly Ks


def ok_missing_allowed(df: pt.DataFrame[Ks]) -> pt.DataFrame[Ks]:
    KV.validate(df, allow_missing_columns=True)  # PASS; df still has only 'k'
    return df
