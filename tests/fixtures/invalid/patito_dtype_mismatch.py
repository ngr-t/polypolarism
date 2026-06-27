"""Patito group acceptance is family-scoped (ADR-0010): a String column does
NOT satisfy an ``int`` field — the integer and non-integer families are
disjoint (probed: an ``int`` field rejects a String column).

False-positive twin: ``valid/patito_group_widths`` (a UInt32 column DOES
satisfy ``int``).
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Out(pt.Model):
    a: int


def stringify(df: pl.DataFrame) -> pt.DataFrame[Out]:
    return df.select(a=pl.col("x").cast(pl.Utf8))
