"""Valid fixture: ADR-0006 amendment happy paths.

``pl.DataFrame(non_literal)`` binds open (checking continues instead of
untracking); matching bare-return laziness passes; backward-narrowed
pins flow without disturbing shape-determined outputs.
"""

from __future__ import annotations

import polars as pl


def constructor_open_frame(rows) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    return df.with_columns(x=pl.col("a") + 1)


def matching_laziness(df: pl.DataFrame) -> pl.LazyFrame:
    return df.lazy()


def narrowed_pin_does_not_leak(df: pl.DataFrame) -> pl.DataFrame:
    checked = df.filter(pl.col("extra").is_not_null())
    return checked.select(total=pl.col("amount").cast(pl.Float64))
