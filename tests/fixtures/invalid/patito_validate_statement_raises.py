"""Invalid: a statement-position patito validate that provably raises (issue #154).

The always-raises argument check ran only in return / assigned position; a
bare-statement (assertion-style) validate whose argument can never satisfy the
schema — here an extra column against a strict model — passed silently
(false negative). It must be flagged exactly like the return-position call: the
call raises ``DataFrameValidationError`` on every input, so the function can
never return. Residual of #150 (the return-position and narrowing paths were
fixed there; this is the statement-position raise-check).
"""

from __future__ import annotations

import patito as pt
import polars as pl


class Ks(pt.Model):
    k: str


def statement_plain_raises(df: pt.DataFrame[Ks]) -> pt.DataFrame[Ks]:
    x = df.with_columns(extra=pl.col("k").str.to_uppercase())
    Ks.validate(x)  # DataFrameValidationError every call — 'extra' is superfluous
    return df
