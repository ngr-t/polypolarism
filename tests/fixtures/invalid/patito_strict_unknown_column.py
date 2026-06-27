"""Patito models bind ``strict=True`` (ADR-0010): extra columns are rejected
at validate time, so a closed Patito frame makes an undeclared-column lookup a
provable ``pple-column-not-found`` error.

Unlike a non-strict Pandera schema (which binds as a checked island and emits
the softer ``pple-undeclared-column`` interface lint), the strict Patito frame
holds exactly its declared columns.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class In(pt.Model):
    id: int


def lookup_missing(df: pt.DataFrame[In]) -> pl.DataFrame:
    return df.select("missing")
