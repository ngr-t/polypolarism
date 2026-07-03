"""Valid: fill forms that provably remove every null still narrow (issue #124).

A literal fill (``fill_null(0)``) and a provably non-null expression fill
(``fill_null(pl.col("b"))`` where ``b`` is non-null) cover every null row, so
the receiver's ``Nullable`` wrapper is correctly stripped and a non-null
declaration type-checks. This is the counterpart to
``invalid/fill_null_keeps_nullable`` — only the strategy / nullable-expr /
fill_nan forms must keep ``Nullable``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int = pa.Field(nullable=True)
    b: int  # non-null


class Out(pa.DataFrameModel):
    a: int  # non-null


def fill_literal(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(a=pl.col("a").fill_null(0))


def fill_nonnull_expr(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(a=pl.col("a").fill_null(pl.col("b")))
