"""Valid: unary minus is dtype-preserving on signed numerics / Duration (issue #136).

``-pl.col(...)`` preserves the dtype for signed ints, floats and Duration (and
Decimal), so declaring the (correct, now-tracked) result dtype must type-check.
Previously the result was untracked, so ANY declaration passed — including wrong
ones; the tracked dtype is exercised here.
"""

from datetime import timedelta

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    f64: pl.Float64
    d: timedelta


class Out(pa.DataFrameModel):
    ni: pl.Int8  # -Int8 -> Int8
    nf: pl.Float64  # -Float64 -> Float64
    nd: timedelta  # -Duration -> Duration


def negate(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        ni=-pl.col("i8"),
        nf=-pl.col("f64"),
        nd=-pl.col("d"),
    )
