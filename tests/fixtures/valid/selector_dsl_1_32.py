"""Valid fixture: post-1.32 selector-as-DSL patterns.

Polars 1.32 changed ``pl.selectors.*`` to return ``Selector`` objects
rather than thin ``Expr`` wrappers (#23351). For an AST-only analyzer
the distinction is mostly invisible — what matters is whether the
existing dispatch in ``analyzer.py:424-496`` still infers the right
result schema for the patterns users actually write.

This fixture pins the patterns we audited:

- Selector composition (algebra) — already covered by `m10_selector_algebra.py`.
- Selector with chained aggregation: ``cs.numeric().sum()``.
- Selector under arithmetic: ``cs.numeric() * 2``.
- Selector with chained method that returns Expr/Selector: ``cs.numeric().fill_null(0)``.
- Selector inside ``with_columns`` (columns are replaced in-place).
- ``cs.exclude`` and ``cs.by_dtype`` flowing through ``select``.

If polars 2.x ever breaks one of these, the failure surfaces here rather
than as a silent miscalculation downstream.
"""

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    a: pl.Float64
    b: pl.Float64
    name: str


def select_chained_agg(df: DataFrame[S]):
    return df.select(cs.numeric().sum())


def select_with_arithmetic(df: DataFrame[S]):
    return df.select(cs.numeric() * 2)


def with_columns_selector_fill_null(df: DataFrame[S]) -> DataFrame[S]:
    return df.with_columns(cs.numeric().fill_null(0))


def select_by_dtype(df: DataFrame[S]):
    return df.select(cs.by_dtype(pl.Float64))


def drop_via_exclude(df: DataFrame[S]):
    return df.drop(cs.exclude(cs.numeric()))
