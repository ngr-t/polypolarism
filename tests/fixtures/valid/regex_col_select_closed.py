"""Regex pl.col / cs.matches on a CLOSED frame expands precisely (issue #111).

When every column name is known (``strict = True``), the regex matching set is
exact — ``pl.col("^.*$")`` is all columns and ``pl.col("^a_.*$")`` /
``cs.matches("^a_")`` are exactly ``a_x``, ``a_y`` with their declared dtypes
flowing through unchanged. This is the precise twin of the open-frame
``regex_col_select.py``; here no degradation to an open frame is needed.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    a_x: int
    a_y: int
    b: str

    class Config:
        strict = True


class Prefixed(pa.DataFrameModel):
    a_x: int
    a_y: int

    class Config:
        strict = True


def all_columns(df: DataFrame[Wide]) -> DataFrame[Wide]:
    return df.select(pl.col("^.*$"))


def regex_prefix(df: DataFrame[Wide]) -> DataFrame[Prefixed]:
    return df.select(pl.col("^a_.*$"))


def cs_matches_prefix(df: DataFrame[Wide]) -> DataFrame[Prefixed]:
    return df.select(cs.matches("^a_"))
