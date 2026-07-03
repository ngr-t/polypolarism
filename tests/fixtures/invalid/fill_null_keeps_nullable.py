"""Invalid: fill forms that do NOT remove every null, declared non-null (issue #124).

Three fill forms provably leave nulls behind, so returning them into a
non-null declared field is unsound (pandera rejects the surviving null at
validation time):

1. ``fill_null(strategy="forward")`` — a leading null has nothing to fill
   from and survives (``"backward"`` leaves a trailing null).
2. ``fill_null(<nullable expr>)`` — rows where the fill expression is itself
   null stay null.
3. ``fill_nan(...)`` — replaces NaN only; nulls are untouched entirely.

Contrast ``fill_null(0)`` (a literal fill), which genuinely removes every null
and correctly narrows — covered by the valid fixture.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class NullableIn(pa.DataFrameModel):
    a: int = pa.Field(nullable=True)
    b: int = pa.Field(nullable=True)


class NonNullOut(pa.DataFrameModel):
    a: int  # non-null declared


def forward_fill(df: DataFrame[NullableIn]) -> DataFrame[NonNullOut]:
    return df.select(a=pl.col("a").fill_null(strategy="forward"))


def fill_with_nullable_expr(df: DataFrame[NullableIn]) -> DataFrame[NonNullOut]:
    return df.select(a=pl.col("a").fill_null(pl.col("b")))


class NullableFloat(pa.DataFrameModel):
    f: pl.Float64 = pa.Field(nullable=True)


class NonNullFloat(pa.DataFrameModel):
    f: pl.Float64


def fill_nan_only(df: DataFrame[NullableFloat]) -> DataFrame[NonNullFloat]:
    return df.select(f=pl.col("f").fill_nan(0.0))
