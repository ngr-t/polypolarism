"""Backward narrowing stays scoped to its variable (ADR-0006 future-work).

The pins a successful ``select`` writes back onto its source must not
leak across variables, and rebinding the name discards them — otherwise
assumption pins would manufacture false positives on unrelated frames.

Intentionally unpaired: these functions pin the *absence* of narrowing
(leniency must hold); there is no wrong-declaration twin to write.
"""

from __future__ import annotations

import polars as pl


def narrowing_does_not_cross_variables(df: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    _probe = df.select(pl.col("a").str.to_uppercase())
    return df2.select(pl.col("a").dt.year())


def narrowing_cleared_by_rebinding(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    _probe = df.select(pl.col("a").str.to_uppercase())
    df = other
    return df.select(pl.col("a").dt.year())
