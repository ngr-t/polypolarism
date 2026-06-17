"""Soundness check: a @rowpoly helper body must preserve the row variable (C-14 Tier 4).

Tier 3 trusts the @rowpoly contract when threading caller extras. Tier 4
verifies the body actually preserves arbitrary extra columns: it skolemizes
the row variable (injects a sentinel extra column into the parameter frame),
re-analyzes the body, and flags PLY043 when a return point provably drops it
(e.g. an explicit ``select`` of named columns, or a ``group_by().agg()`` that
collapses the schema). This property is relative to the caller, so Pandera
cannot check it at runtime — it is static-only.
"""

import textwrap

from polypolarism.checker import check_source

_PRELUDE = """
import polars as pl
import polars.selectors as cs
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False
"""


def _check(body: str) -> dict:
    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return {r.function_name: r for r in check_source(src)}


def _has_ply043(result) -> bool:
    return any("PLY043" in str(e) for e in result.errors)


def test_with_columns_preserves_row_variable() -> None:
    # with_columns keeps every existing column => the row variable survives.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert results["add_score"].passed
    assert not _has_ply043(results["add_score"])


def test_explicit_select_drops_row_variable() -> None:
    # select of named columns produces a closed frame without the caller's
    # extras => the @rowpoly contract is violated.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not results["add_score"].passed
    assert _has_ply043(results["add_score"])


def test_group_by_agg_drops_row_variable() -> None:
    results = _check("""
        class Grouped(pa.DataFrameModel):
            id: int
            total: float

            class Config:
                strict = False

        @rowpoly("R")
        def summarize(df: DataFrame[InId]) -> DataFrame[Grouped]:
            return df.group_by("id").agg(total=pl.col("id").cast(pl.Float64).sum())
    """)
    assert _has_ply043(results["summarize"])


def test_non_rowpoly_helper_is_not_preservation_checked() -> None:
    # Without the decorator there is no preservation claim, so dropping
    # columns is fine (it just returns the declared schema).
    results = _check("""
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_multi_frame_param_helper_is_not_preservation_checked() -> None:
    # With two frame parameters the single row variable is ill-defined
    # (whose extras does R capture?) — that is the R1 # R2 case deferred to
    # Tier 5. Tier 4 only checks single-frame-parameter helpers, so a
    # two-param helper is never flagged (sound: we just don't check).
    results = _check("""
        class Other(pa.DataFrameModel):
            id: int
            tag: str

            class Config:
                strict = False

        @rowpoly("R")
        def joined(df: DataFrame[InId], o: DataFrame[Other]) -> DataFrame[InId]:
            return o.select("id")
    """)
    assert not _has_ply043(results["joined"])


# --- C-14 Tier 5 remainder (1a): no false positive for legitimately-preserving
# bodies. The skolem sentinel must survive each of these forms. These pin the
# CURRENT behavior: an audit (2026-06-17) found no false positives among them —
# the all-columns selectors already include the sentinel in the resolved column
# set, drop/rename only touch named real columns, branch merge keeps shared
# columns, and pl.concat unions identical inputs. No new machinery was needed.


def test_select_all_pl_preserves_row_variable() -> None:
    # ``select(pl.all())`` selects every column including the skolem sentinel,
    # so the all-columns selector preserves the row variable (no PLY043).
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(pl.all()).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert results["add_score"].passed
    assert not _has_ply043(results["add_score"])


def test_select_all_cs_preserves_row_variable() -> None:
    # ``select(cs.all())`` is the selectors-namespace spelling of the same
    # all-columns selection; it likewise keeps the sentinel.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(cs.all()).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert results["add_score"].passed
    assert not _has_ply043(results["add_score"])


def test_drop_real_column_preserves_row_variable() -> None:
    # Dropping a NAMED real column leaves the sentinel untouched — the helper
    # still preserves the caller's arbitrary extras.
    results = _check("""
        class OutDropped(pa.DataFrameModel):
            score: float

            class Config:
                strict = False

        @rowpoly("R")
        def reshape(df: DataFrame[InId]) -> DataFrame[OutDropped]:
            return df.drop("id").with_columns(score=pl.lit(1.0))
    """)
    assert not _has_ply043(results["reshape"])


def test_rename_real_column_preserves_row_variable() -> None:
    # Renaming a NAMED real column does not touch the sentinel.
    results = _check("""
        class Renamed(pa.DataFrameModel):
            key: int

            class Config:
                strict = False

        @rowpoly("R")
        def reshape(df: DataFrame[InId]) -> DataFrame[Renamed]:
            return df.rename({"id": "key"})
    """)
    assert not _has_ply043(results["reshape"])


def test_conditional_early_return_preserves_row_variable() -> None:
    # Both return points preserve every column, so neither branch drops the
    # sentinel.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            if df.is_empty():
                return df.with_columns(score=pl.lit(0.0))
            return df.with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_pl_concat_preserves_row_variable() -> None:
    # Concatenating the (skolemized) input with itself keeps the sentinel in
    # the union, so the row variable survives.
    results = _check("""
        @rowpoly("R")
        def doubled(df: DataFrame[InId]) -> DataFrame[InId]:
            return pl.concat([df, df])
    """)
    assert not _has_ply043(results["doubled"])
