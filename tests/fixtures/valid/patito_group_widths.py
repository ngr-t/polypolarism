"""Patito ``int`` / ``float`` fields accept ANY integer / float width
(ADR-0010 group acceptance; ADR-0009 no-false-positive).

Probed: ``Model.valid_dtypes`` for an ``int`` field is the full set of integer
dtypes and for ``float`` the float widths. So producing a ``UInt32`` column
for an ``int`` field or ``Float32`` for a ``float`` field must NOT be flagged
— mapping ``int`` to a single concrete dtype would wrongly reject these.

False-positive twin: ``invalid/patito_dtype_mismatch`` (a String column does
NOT satisfy ``int`` — the families are disjoint).
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Out(pt.Model):
    a: int
    b: float


def widths(df: pl.DataFrame) -> pt.DataFrame[Out]:
    return df.select(
        a=pl.col("x").cast(pl.UInt32),
        b=pl.col("y").cast(pl.Float32),
    )
