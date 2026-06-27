"""A strict Patito output model rejects a body that fails to produce a
declared column (ADR-0010).

Patito validates strictly (probed: a missing declared column raises
``DataFrameValidationError``), so the missing ``label`` is a provable
return-type error.
"""

from __future__ import annotations

import patito as pt
import polars as pl


class In(pt.Model):
    id: int


class Out(pt.Model):
    id: int
    label: str


def drop_label(df: pt.DataFrame[In]) -> pt.DataFrame[Out]:
    return df.select(pl.col("id"))  # 'label' is never produced
