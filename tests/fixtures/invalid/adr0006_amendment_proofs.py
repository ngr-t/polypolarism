"""Invalid fixture: ADR-0006 amendment proofs.

Backward narrowing: an assumed lookup pins the column into the open
frame, so a later strict-parameter call provably carries the extra.
Bare return annotations check the eager/lazy bit (pple-eager-lazy-mismatch). The no-args
``pl.DataFrame()`` constructor is the provably empty frame.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class StrictPrice(pa.DataFrameModel):
    price: float

    class Config:
        strict = True
        coerce = True


def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
    return df.filter(pl.col("price") > 0)


def backward_narrowed_extra(df: pl.DataFrame) -> pl.DataFrame:
    checked = df.filter(pl.col("region") != "x")  # pins 'region'
    return strict_helper(checked)


def lazy_returned_as_bare_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df.lazy()


def empty_constructor_is_exact() -> pl.DataFrame:
    return pl.DataFrame().select(pl.col("a"))
