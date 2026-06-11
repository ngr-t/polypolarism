"""Sequential renames accumulate absence (issue #78 boundary).

``rename({"a": "b"})`` then ``rename({"b": "c"})`` leaves only ``c``;
both old names are provably gone and referencing either must fail.

False-positive twin: ``valid/absent_rename_swap`` (a *simultaneous*
swap keeps both names — the two must not be conflated).
"""

from __future__ import annotations

import polars as pl


def chained_rename_old_name(df: pl.DataFrame) -> pl.DataFrame:
    renamed = df.rename({"a": "b"}).rename({"b": "c"})
    return renamed.select(pl.col("a"))  # WRONG: 'a' was renamed away in step 1


def chained_rename_intermediate_name(df: pl.DataFrame) -> pl.DataFrame:
    renamed = df.rename({"a": "b"}).rename({"b": "c"})
    return renamed.select(pl.col("b"))  # WRONG: 'b' was renamed away in step 2
