"""``Model.validate(df)`` narrows the argument variable to the model's frame
at the function-body top level (ADR-0010).

This is the same bare-statement narrowing Pandera's ``Schema.validate(df)``
gets — the narrowing keys on the method name ``validate`` plus a schema
registry lookup, which is dialect-neutral. Probed: patito's ``validate`` also
returns the validated frame, so the narrowed columns are sound.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class S(pt.Model):
    a: int
    b: str


def narrowed(df: pl.DataFrame) -> pl.DataFrame:
    S.validate(df)
    # After validate, ``df`` is known to carry exactly S's columns.
    return df.select("a", "b")
