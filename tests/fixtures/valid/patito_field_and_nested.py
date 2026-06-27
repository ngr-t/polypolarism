"""``pt.Field(dtype=...)`` forces an exact polars dtype and a nested Patito
model becomes a ``Struct`` column (ADR-0010).

``rank`` is pinned to ``UInt16`` by ``Field(dtype=...)`` — an exact dtype, not
the ``int`` acceptance group — and ``inner`` is the ``Inner`` model rendered
as a struct of its columns. The passthrough preserves both exactly.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Inner(pt.Model):
    a: int
    b: str


class Outer(pt.Model):
    rank: int = pt.Field(dtype=pl.UInt16)
    inner: Inner


def passthrough(df: pt.DataFrame[Outer]) -> pt.DataFrame[Outer]:
    return df.select("rank", "inner")
