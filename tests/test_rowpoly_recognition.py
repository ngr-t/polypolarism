"""Static recognition of the ``@rowpoly("R")`` decorator (C-14 Tier 2).

The analyzer records the row-variable name on ``FunctionAnalysis.row_var``.
This tier is metadata-only: capturing the name has no effect on the
verdict yet (the threading that uses it is a later tier), so these tests
pin *recognition* and *non-regression*, not behavior change.
"""

import textwrap

from polypolarism.analyzer import analyze_source
from polypolarism.checker import check_source

_PRELUDE = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from polypolarism import rowpoly
import polypolarism as pp

class S(pa.DataFrameModel):
    id: int

ROWNAME = "dynamic"
"""


def _analyze(body: str) -> dict:
    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return {a.name: a for a in analyze_source(src)}


def test_bare_call_form_is_recognized() -> None:
    funcs = _analyze("""
        @rowpoly("R")
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var == "R"


def test_attribute_call_form_is_recognized() -> None:
    funcs = _analyze("""
        @pp.rowpoly("Rest")
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var == "Rest"


def test_absent_decorator_leaves_row_var_none() -> None:
    funcs = _analyze("""
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var is None


def test_bare_decorator_without_call_is_ignored() -> None:
    # ``@rowpoly`` with no argument carries no name -> not a row variable.
    funcs = _analyze("""
        @rowpoly
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var is None


def test_non_literal_argument_is_ignored() -> None:
    funcs = _analyze("""
        @rowpoly(ROWNAME)
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var is None


def test_coexists_with_other_decorators_first_match_wins() -> None:
    funcs = _analyze("""
        import functools

        @functools.cache
        @rowpoly("A")
        @rowpoly("B")
        def f(df: DataFrame[S]) -> DataFrame[S]:
            return df
    """)
    assert funcs["f"].row_var == "A"


def test_recognition_does_not_change_the_verdict() -> None:
    # A return-type mismatch must still FAIL with the decorator present
    # (recognition is metadata-only). The PLY040 verdict comes from the
    # checker layer, so assert through check_source, not the raw analysis.
    body = """
        class T(pa.DataFrameModel):
            other: int

        @rowpoly("R")
        def f(df: DataFrame[S]) -> DataFrame[T]:
            return df
    """
    funcs = _analyze(body)
    assert funcs["f"].row_var == "R"

    src = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    result = next(r for r in check_source(src) if r.function_name == "f")
    assert not result.passed
    assert any("PLY040" in str(e) for e in result.errors)
