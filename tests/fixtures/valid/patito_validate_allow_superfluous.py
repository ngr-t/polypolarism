"""Patito ``Model.validate(df, allow_superfluous_columns=True)`` passes the
input's extra columns THROUGH (ADR-0010 #4, probed).

So the narrowed frame is OPEN — accessing a passed-through extra column is not
a false ``pple-column-not-found`` (ADR-0009). With the default (strict)
validate the same access IS a provable miss; here the kwarg opts into extras.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class S(pt.Model):
    a: int


def allow_extras(df: pl.DataFrame) -> pl.DataFrame:
    S.validate(df, allow_superfluous_columns=True)
    return df.select("a", "extra")
