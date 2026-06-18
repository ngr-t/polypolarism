"""Soundness check: a @rowpoly helper body must preserve the row variable (C-14 Tier 4).

Tier 3 trusts the @rowpoly contract when threading caller extras. Tier 4
verifies the body actually preserves arbitrary extra columns: it skolemizes
the row variable (injects a sentinel extra column into the parameter frame),
re-analyzes the body, and flags pple-rowpoly-not-preserved when a return point provably drops it
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
    return any("pple-rowpoly-not-preserved" in str(e) for e in result.errors)


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
    # so the all-columns selector preserves the row variable (no pple-rowpoly-not-preserved).
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


# --- C-14 follow-up #3: pattern/selector reductions that DROP the row variable.
# The sentinel probe is blind to a reduction keyed by a PREDICATE (regex / dtype
# / name-pattern selector) that doesn't happen to match the sentinel name: the
# sentinel survives, so the probe says "preserved", but a real caller extra that
# DOES match the predicate would be dropped at runtime. The structural
# pattern-drop guard catches these; the boundary below pins exactly which forms
# flag (predicate-based reductions over the open frame) and which stay OK
# (explicit known-name reductions / all-columns forms). A false positive here is
# worse than the residual false negative, so the OK guards are load-bearing.


def test_select_exclude_regex_drops_row_variable() -> None:
    # select(pl.exclude("^tmp_.*$")) keeps every column NOT matching the regex.
    # A caller extra named tmp_junk would be excluded -> not preserved. The
    # sentinel \x00__rowvar_0__ does not match ^tmp_.*$, so it survives the
    # exclude -> the probe alone would miss this; the pattern guard catches it.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(
                pl.exclude("^tmp_.*$")
            ).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert _has_ply043(results["add_score"])


def test_select_cs_starts_with_drops_row_variable() -> None:
    # select(cs.starts_with("id")) keeps only id-prefixed columns; a caller
    # extra with a different prefix is dropped -> not preserved.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(
                cs.starts_with("id")
            ).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert _has_ply043(results["add_score"])


def test_select_cs_numeric_drops_row_variable() -> None:
    # select(cs.numeric()) keeps only numeric columns; a caller extra of a
    # non-numeric dtype is dropped -> not preserved.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(
                cs.numeric()
            ).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert _has_ply043(results["add_score"])


def test_select_narrowing_regex_drops_row_variable() -> None:
    # select(pl.col("^id$")) is an anchored regex that is NOT match-all; a
    # caller extra not matching ^id$ is dropped -> not preserved.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(
                pl.col("^id$")
            ).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert _has_ply043(results["add_score"])


def test_drop_cs_predicate_drops_row_variable() -> None:
    # drop(cs.numeric()) removes every numeric column; a caller extra of a
    # numeric dtype would be removed -> not preserved.
    results = _check("""
        class OutTag(pa.DataFrameModel):
            tag: str

            class Config:
                strict = False

        @rowpoly("R")
        def reshape(df: DataFrame[InId]) -> DataFrame[OutTag]:
            return df.drop(cs.numeric()).with_columns(tag=pl.lit("x"))
    """)
    assert _has_ply043(results["reshape"])


def test_drop_regex_drops_row_variable() -> None:
    # drop(pl.col("^tmp_.*$")) removes every column matching the regex; a
    # caller extra named tmp_junk would be removed -> not preserved.
    results = _check("""
        class OutTag(pa.DataFrameModel):
            id: int

            class Config:
                strict = False

        @rowpoly("R")
        def reshape(df: DataFrame[InId]) -> DataFrame[OutTag]:
            return df.drop(pl.col("^tmp_.*$"))
    """)
    assert _has_ply043(results["reshape"])


# --- Boundary: PRESERVING forms below must stay silent (no false pple-rowpoly-not-preserved). ---


def test_select_match_all_regex_preserves_row_variable() -> None:
    # select(pl.col("^.*$")) is the match-everything regex (#111): it keeps
    # every column, including the caller's extras -> preserved.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(
                pl.col("^.*$")
            ).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_select_exclude_literal_name_preserves_row_variable() -> None:
    # select(pl.exclude("id")) excludes a single KNOWN name; every OTHER column
    # (including unknown extras) is kept -> preserved.
    results = _check("""
        class OutDropped(pa.DataFrameModel):
            score: float

            class Config:
                strict = False

        @rowpoly("R")
        def reshape(df: DataFrame[InId]) -> DataFrame[OutDropped]:
            return df.select(pl.exclude("id")).with_columns(score=pl.lit(1.0))
    """)
    assert not _has_ply043(results["reshape"])


def test_drop_literal_name_preserves_row_variable() -> None:
    # drop("id") removes a single KNOWN name; unknown extras survive -> OK.
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


def test_select_pl_all_still_preserves_row_variable() -> None:
    # Re-pin the all-columns forms now that the pattern guard exists: pl.all()
    # must not be misclassified as a predicate reduction.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(pl.all()).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_select_cs_all_still_preserves_row_variable() -> None:
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select(cs.all()).with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_with_columns_still_preserves_row_variable() -> None:
    # with_columns only ADDS — it is never a reduction.
    results = _check("""
        @rowpoly("R")
        def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert not _has_ply043(results["add_score"])


def test_rename_still_preserves_row_variable() -> None:
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


# --- C-14 follow-up: @rowpoly("R", drops=<selector>) declares an INTENTIONAL
# pattern-restriction of the row variable. A body whose only reduction matches
# the declared selector is accepted (the user declared it); a broader/other
# drop still flags. ``drops=`` never becomes a blanket "preserve nothing"
# escape — only the declared pattern is exempt.


def test_drops_select_complement_of_declared_is_accepted() -> None:
    # The canonical sanitizer: declare drops=cs.starts_with("_internal_") and
    # the body select(~cs.starts_with("_internal_")) — drops EXACTLY the
    # declared pattern, keeps everything else => accepted (no pple-rowpoly-not-preserved).
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_internal_"))
    """)
    assert not _has_ply043(results["sanitize"])


def test_drops_drop_declared_selector_is_accepted() -> None:
    # The ``drop`` spelling of the same declaration: drop(cs.starts_with(...))
    # removes exactly the declared pattern => accepted.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.drop(cs.starts_with("_internal_"))
    """)
    assert not _has_ply043(results["sanitize"])


def test_drops_with_trailing_with_columns_is_accepted() -> None:
    # The reduction is followed by an add-only with_columns; the drop node must
    # still be attributable to the declared selector on the RETURN frame.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[OutScore]) -> DataFrame[OutScore]:
            return df.select(~cs.starts_with("_internal_")).with_columns(
                score=pl.col("id").cast(pl.Float64)
            )
    """)
    assert not _has_ply043(results["sanitize"])


def test_drops_regex_matches_matching_body_is_accepted() -> None:
    # Declared drops as a cs.matches regex (the columns REMOVED match it); body
    # keeps the complement via select(~cs.matches(...)) — drops exactly the
    # declared regex pattern => accepted.
    results = _check("""
        @rowpoly("R", drops=cs.matches("^tmp_.*$"))
        def sanitize(df: DataFrame[OutScore]) -> DataFrame[OutScore]:
            return df.select(~cs.matches("^tmp_.*$")).with_columns(
                score=pl.col("id").cast(pl.Float64)
            )
    """)
    assert not _has_ply043(results["sanitize"])


def test_drops_narrower_than_declared_is_accepted() -> None:
    # Body drops a STRICTER subset (cs.starts_with("_internal_secret")) of the
    # declared pattern (cs.starts_with("_internal_")) => still within the
    # declaration, accepted.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_internal_secret"))
    """)
    assert not _has_ply043(results["sanitize"])


def test_drops_broader_than_declared_still_ply043() -> None:
    # Declared drops=cs.starts_with("_internal_") but the body drops the BROADER
    # cs.starts_with("_") (every underscore-prefixed column). That removes
    # caller extras OUTSIDE the declared pattern (e.g. "_other") => still
    # pple-rowpoly-not-preserved.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_"))
    """)
    assert _has_ply043(results["sanitize"])


def test_drops_different_pattern_still_ply043() -> None:
    # Declared drops a name pattern; the body drops by DTYPE (cs.numeric()).
    # A non-numeric caller extra outside the declared name pattern is dropped
    # => still pple-rowpoly-not-preserved.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.drop(cs.numeric())
    """)
    assert _has_ply043(results["sanitize"])


def test_drops_does_not_relax_fixed_name_select() -> None:
    # ``drops=`` declares a PATTERN, not a licence to drop everything: a fixed
    # select("id") closes the frame and drops every caller extra by name =>
    # still pple-rowpoly-not-preserved (not a pattern drop the declaration covers).
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[OutScore]:
            return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))
    """)
    assert _has_ply043(results["sanitize"])


def test_without_drops_pattern_drop_still_ply043_no_regression() -> None:
    # The SAME body as the accepted sanitizer, but WITHOUT drops= => no
    # declaration, so the pattern drop is rejected exactly as today.
    results = _check("""
        @rowpoly("R")
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_internal_"))
    """)
    assert _has_ply043(results["sanitize"])


def test_drops_non_selector_value_is_ignored() -> None:
    # A non-selector drops= value (a string literal) is unrecognised and
    # ignored — behaves as if drops= were absent => the pattern drop still
    # flags (lenient: a typo cannot silence a real drop).
    results = _check("""
        @rowpoly("R", drops="_internal_")
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_internal_"))
    """)
    assert _has_ply043(results["sanitize"])


def test_drops_stacked_reductions_still_ply043() -> None:
    # Two stacked predicate reductions: the drop is no longer attributable to a
    # single selector, so containment cannot be proven => still pple-rowpoly-not-preserved even
    # though the first reduction alone is within the declaration.
    results = _check("""
        @rowpoly("R", drops=cs.starts_with("_internal_"))
        def sanitize(df: DataFrame[InId]) -> DataFrame[InId]:
            return df.select(~cs.starts_with("_internal_")).select(~cs.ends_with("_tmp"))
    """)
    assert _has_ply043(results["sanitize"])
