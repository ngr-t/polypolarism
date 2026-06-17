"""pl.col("^regex$") and cs.matches(...) select columns whose name matches.

polars treats a single ``pl.col`` string that starts with ``^`` AND ends with
``$`` as a regex over the input column names, and ``cs.matches(pat)`` likewise
selects columns whose name matches the pattern (issue #111). polypolarism must
expand these to the matching set rather than treating the pattern as a literal
column name (which produced a false PLY042 / PLY040). On the (open) frame here
the regex may also match unknown extras, so the result degrades to an open
frame — never a hard error on valid runtime code. The ``r_literal`` control
keeps the explicit-name path behaving as before.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class T(pa.DataFrameModel):
    a_x: int
    a_y: int
    b: str

    class Config:
        strict = False
        coerce = True


@pa.check_types
def r_all(df: DataFrame[T]) -> DataFrame[T]:
    return df.select(pl.col("^.*$"))  # regex: all columns


@pa.check_types
def r_prefix(df: DataFrame[T]) -> DataFrame[T]:
    return df.select(pl.col("^a_.*$"), pl.col("b"))  # regex: a_x, a_y + b


@pa.check_types
def r_cs_matches(df: DataFrame[T]) -> DataFrame[T]:
    return df.select(cs.matches("^a_"), pl.col("b"))  # selector: a_x, a_y + b


@pa.check_types
def r_literal(df: DataFrame[T]) -> DataFrame[T]:  # control
    return df.select(pl.col("a_x"), pl.col("a_y"), pl.col("b"))
