"""Simultaneous rename swap keeps both names alive (issue #78 boundary).

``rename({"a": "b", "b": "a"})`` is a swap — polars applies the mapping
simultaneously, so both names still exist afterwards. A naive
"old names become absent" implementation of ``FrameType.absent`` would
mark both as provably removed and reject this valid code.

False-negative twin: ``invalid/absent_rename_chain``.
"""

from __future__ import annotations

import polars as pl


def rename_swap_keeps_both(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({"a": "b", "b": "a"}).select(pl.col("a"), pl.col("b"))
