"""Invalid fixture: negative knowledge on open frames (issue #78).

``drop`` / ``rename`` create PROVABLE absence — conditional on reaching
the next line, exactly the ADR-0006 standard. A later reference to the
removed/old name is a guaranteed runtime ColumnNotFoundError on every
execution that reaches it, open frame or not.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


def use_after_drop(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop("a").select(pl.col("a"))


def use_old_name_after_rename(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({"a": "b"}).select(pl.col("a"))


class Out(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


def declared_column_provably_removed(df: pl.DataFrame) -> DataFrame[Out]:
    # The open frame would normally get the not-provably-absent leniency,
    # but drop() proved the absence.
    return df.drop("a")
