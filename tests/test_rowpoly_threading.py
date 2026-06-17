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


def test_strict_return_does_not_thread_extras() -> None:
    # Soundness: a strict return schema rejects extras at runtime
    # (@pa.check_types), so threading them would be a false claim. A strict
    # return is left untouched; a downstream read of a non-declared caller
    # extra is correctly rejected rather than silently accepted.
    results = _check("""
        class OutStrict(pa.DataFrameModel):
            id: int
            score: float

            class Config:
                strict = True

        @rowpoly("R")
        def add_score_strict(df: DataFrame[InId]) -> DataFrame[OutStrict]:
            return df.with_columns(score=pl.col("id").cast(pl.Float64))

        class Result(pa.DataFrameModel):
            id: int
            score: float
            region: str

        def use_strict(c: DataFrame[Caller]) -> DataFrame[Result]:
            return add_score_strict(c).select("id", "score", "region")
    """)
    assert not results["use_strict"].passed


def test_multi_frame_positional_rowpoly_does_not_thread_precisely() -> None:
    # Soundness: positional @rowpoly("R") only threads (and is only
    # preservation-checked) for a SINGLE frame parameter. A multi-frame
    # positional helper must not precisely claim a caller extra (the body may
    # drop a side, unchecked). It degrades to the open-frame Unknown leniency
    # instead, so a wrong-dtype declaration passes rather than being precisely
    # — and falsely — accepted/rejected.
    results = _check("""
        class Other(pa.DataFrameModel):
            id: int

            class Config:
                strict = False

        @rowpoly("R")
        def two(a: DataFrame[InId], b: DataFrame[Other]) -> DataFrame[OutScore]:
            return a.with_columns(score=pl.col("id").cast(pl.Float64))

        class CallerB(pa.DataFrameModel):
            id: int
            extra: int

        class ResultWrong(pa.DataFrameModel):
            id: int
            score: float
            extra: str

            class Config:
                strict = False

        def use_two(c: DataFrame[Caller], d: DataFrame[CallerB]) -> DataFrame[ResultWrong]:
            return two(c, d).select("id", "score", "extra")
    """)
    # Not precisely threaded -> `extra` is Unknown -> wrong dtype passes
    # (gradual leniency), NOT a precise false dtype claim.
    assert results["use_two"].passed, [str(e) for e in results["use_two"].errors]


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
