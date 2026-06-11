"""Invalid fixture: provable contradictions on open frames (ADR-0006).

An open frame claims nothing about its source — but everything the code
itself determines is a proof obligation. ``select`` closes the frame
(its output shape is the call's own choice), so a later reference to a
column it did not keep is a guaranteed runtime ColumnNotFoundError; a
column pinned by ``with_columns`` carries an exact dtype, so invalid
arithmetic on it is a guaranteed runtime error; a declared return schema
is contradicted by a pinned dtype.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


def select_closed_the_frame(df: pl.DataFrame) -> pl.DataFrame:
    picked = df.select("a", "b")
    return picked.select(pl.col("c"))  # 'c' is provably not in {a, b}


def pinned_dtype_misused(df: pl.DataFrame) -> pl.DataFrame:
    tagged = df.with_columns(label=pl.lit("x"))
    return tagged.select(out=pl.col("label") - 1)  # Utf8 - Int64 errors


class Report(pa.DataFrameModel):
    total: int

    class Config:
        strict = True


def pinned_dtype_contradicts_declared(df: pl.DataFrame) -> DataFrame[Report]:
    # ``total`` is pinned Utf8 by the cast — provably not the declared int.
    return df.select(total=pl.col("amount").cast(pl.Utf8))
