"""Per-parameter row variables for multi-frame-param helpers (C-14 Tier 5).

``@rowpoly(a="R1", b="R2")`` names one row variable per frame parameter, so a
join / concat helper preserves BOTH sides' extra columns. This generalizes
the single-parameter threading (Tier 3) and preservation (Tier 4) that
``@rowpoly("R")`` provides. Disjointness *diagnostics* (R1 # R2) and explicit
row add/drop/rename tracking are out of scope here.
"""

import textwrap

from polypolarism.checker import check_source

_PRELUDE = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from polypolarism import rowpoly


class A(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class B(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


class Joined(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


@rowpoly(a="R1", b="R2")
def merge(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Joined]:
    return a.join(b, on="id", how="inner")


# Caller frames carry side-specific extras beyond A / B.
class LeftCaller(pa.DataFrameModel):
    id: int
    left_extra: int


class RightCaller(pa.DataFrameModel):
    id: int
    tag: str
    right_extra: float
"""


def _check(body: str) -> dict:
    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return {r.function_name: r for r in check_source(src)}


def _has_rowpoly_not_preserved(result) -> bool:
    return any("pple-rowpoly-not-preserved" in str(e) for e in result.errors)


def test_both_sides_extras_are_threaded() -> None:
    # The result should carry left_extra (Int64) AND right_extra (Float64).
    results = _check("""
        class Result(pa.DataFrameModel):
            id: int
            tag: str
            left_extra: int
            right_extra: float

        def use(l: DataFrame[LeftCaller], r: DataFrame[RightCaller]) -> DataFrame[Result]:
            out = merge(l, r)
            return out.select("id", "tag", "left_extra", "right_extra")
    """)
    assert results["use"].passed, [str(e) for e in results["use"].errors]


def test_threaded_extra_keeps_real_dtype() -> None:
    # left_extra is really Int64; declaring it Utf8 must FAIL (precise dtype).
    results = _check("""
        class ResultWrong(pa.DataFrameModel):
            id: int
            tag: str
            left_extra: str

        def use_wrong(l: DataFrame[LeftCaller], r: DataFrame[RightCaller]) -> DataFrame[ResultWrong]:
            out = merge(l, r)
            return out.select("id", "tag", "left_extra")
    """)
    assert not results["use_wrong"].passed
    assert any("pple-return-type" in str(e) for e in results["use_wrong"].errors)


def test_merge_body_preserving_both_is_accepted() -> None:
    results = _check("""
        def noop() -> int:
            return 0
    """)
    # `merge` itself (in the prelude) joins both sides -> both row vars preserved.
    assert not _has_rowpoly_not_preserved(results["merge"])


def test_dropping_one_side_is_flagged_for_that_row_var() -> None:
    results = _check("""
        @rowpoly(a="R1", b="R2")
        def merge_bad(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Joined]:
            # Drops b entirely: B's row variable R2 is not preserved.
            return a.join(b, on="id", how="inner").select("id")
    """)
    assert _has_rowpoly_not_preserved(results["merge_bad"])
    assert any("R2" in str(e) for e in results["merge_bad"].errors)


def test_keyword_for_unknown_param_is_ignored() -> None:
    # A keyword naming a non-existent parameter binds nothing; no crash.
    results = _check("""
        @rowpoly(nonexistent="R")
        def passthrough(a: DataFrame[A]) -> DataFrame[A]:
            return a
    """)
    assert results["passthrough"].passed
    assert not _has_rowpoly_not_preserved(results["passthrough"])
