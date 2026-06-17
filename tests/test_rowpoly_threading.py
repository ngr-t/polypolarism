"""Call-site threading of caller extras through a @rowpoly helper (C-14 Tier 3).

When a ``@rowpoly("R")`` helper is called, the caller's columns beyond the
declared parameter schema are preserved into the result WITH THEIR REAL
DTYPES, instead of degrading to ``Unknown`` (or vanishing) through the call.
This is the precision half: downstream reads of the caller's extras stay
precisely typed. Soundness (the helper body must actually preserve them) is a
later tier; here we trust the ``@rowpoly`` contract like any other annotation.
"""

import textwrap

from polypolarism.analyzer import analyze_source
from polypolarism.checker import check_source

_PRELUDE = """
import polars as pl
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


@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))


# Same shape WITHOUT the decorator — the baseline that drops caller extras.
def add_score_plain(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))


class Caller(pa.DataFrameModel):
    id: int
    region: str
"""


def _check(body: str) -> dict:
    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return {r.function_name: r for r in check_source(src)}


def _analyze(body: str) -> dict:
    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return {a.name: a for a in analyze_source(src)}


def test_caller_extra_is_preserved_through_rowpoly_call() -> None:
    # Result declares the caller's extra `region` — preserved by the helper.
    results = _check("""
        class Result(pa.DataFrameModel):
            id: int
            score: float
            region: str

        def use(c: DataFrame[Caller]) -> DataFrame[Result]:
            out = add_score(c)
            return out.select("id", "score", "region")
    """)
    assert results["use"].passed, [str(e) for e in results["use"].errors]


def test_preserved_extra_keeps_its_real_dtype_not_unknown() -> None:
    # region is really Utf8; declaring it Int64 must FAIL — proving the dtype
    # was preserved precisely (not degraded to Unknown, which would pass).
    results = _check("""
        class ResultWrong(pa.DataFrameModel):
            id: int
            score: float
            region: int

        def use_wrong(c: DataFrame[Caller]) -> DataFrame[ResultWrong]:
            out = add_score(c)
            return out.select("id", "score", "region")
    """)
    assert not results["use_wrong"].passed
    assert any("PLY040" in str(e) for e in results["use_wrong"].errors)


def test_without_decorator_extra_degrades_to_unknown() -> None:
    # Contrast: the plain (non-@rowpoly) helper returns a non-strict =>
    # open frame, so reading `region` off its result degrades to Unknown.
    # The decorator is what upgrades that to the precise Utf8 (see the dtype
    # test above). Here we pin the baseline: a WRONG-dtype declaration of
    # region still passes via Unknown leniency without the decorator.
    results = _check("""
        class ResultWrong(pa.DataFrameModel):
            id: int
            score: float
            region: int

        def use_plain_wrong(c: DataFrame[Caller]) -> DataFrame[ResultWrong]:
            out = add_score_plain(c)
            return out.select("id", "score", "region")
    """)
    assert results["use_plain_wrong"].passed, [str(e) for e in results["use_plain_wrong"].errors]


def test_threaded_result_carries_extra_column_in_inferred_type() -> None:
    analyses = _analyze("""
        def use(c: DataFrame[Caller]) -> DataFrame[Caller]:
            out = add_score(c)
            return out.select("id", "region")
    """)
    # The intermediate `out` frame should expose `region` as Utf8 precisely;
    # observe it through the final inferred return after select.
    rt = analyses["use"].inferred_return_type
    assert "region" in rt.columns
    assert str(rt.columns["region"].dtype) == "Utf8"
