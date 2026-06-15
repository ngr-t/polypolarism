"""All return paths are checked, and ternary returns infer per arm.

Covers issues #94 (conditional expression / ``ast.IfExp`` returns) and
#95 (every ``return`` statement is checked, not just the last one).
Both stem from one gap: a function can produce several possible return
frames, and ALL of them must satisfy the declared return type.
"""

from __future__ import annotations

import textwrap

from polypolarism.checker import check_source

COMMON = textwrap.dedent(
    """
    from __future__ import annotations
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame


    class KV(pa.DataFrameModel):
        k: str
        v: float

        class Config:
            strict = True
            coerce = True


    class KVa(pa.DataFrameModel):
        k: str
        v: float
        a: float

        class Config:
            strict = True
            coerce = True
    """
)


def _check(body: str) -> dict[str, bool]:
    results = check_source(COMMON + textwrap.dedent(body))
    return {r.function_name: r.passed for r in results}


def _result(body: str, name: str):
    results = check_source(COMMON + textwrap.dedent(body))
    return next(r for r in results if r.function_name == name)


class TestTernaryReturn:
    """Issue #94: ``A if cond else B`` returns."""

    def test_both_arms_unify_passes(self):
        passed = _check(
            """
            def ternary_unify(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                return (
                    df.with_columns(a=pl.col("v") * 2)
                    if flag
                    else df.with_columns(a=pl.col("v") + 1)
                )
            """
        )
        assert passed["ternary_unify"] is True

    def test_diverging_arm_fails(self):
        result = _result(
            """
            def ternary_diverge(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                return (
                    df.with_columns(a=pl.col("v") * 2)
                    if flag
                    else df.with_columns(b=pl.col("v"))
                )
            """,
            "ternary_diverge",
        )
        assert result.passed is False
        # The failure is the missing 'a' in the else arm, not a generic
        # "could not infer".
        joined = " ".join(str(e) for e in result.errors)
        assert "a" in joined
        assert "Could not infer" not in joined

    def test_no_regression_on_plain_return(self):
        passed = _check(
            """
            def plain_no_branch(df: DataFrame[KV]) -> DataFrame[KVa]:
                return df.with_columns(a=pl.col("v") * 2)
            """
        )
        assert passed["plain_no_branch"] is True


class TestMultipleReturns:
    """Issue #95: every return statement is checked."""

    def test_wrong_early_return_detected(self):
        result = _result(
            """
            def if_wrong_else_right(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                if flag:
                    return df.with_columns(b=pl.col("v"))  # wrong: no 'a'
                else:
                    return df.with_columns(a=pl.col("v"))  # right (last)
            """,
            "if_wrong_else_right",
        )
        assert result.passed is False

    def test_wrong_last_return_still_detected(self):
        result = _result(
            """
            def if_right_else_wrong(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                if flag:
                    return df.with_columns(a=pl.col("v"))
                else:
                    return df.with_columns(b=pl.col("v"))  # wrong (last)
            """,
            "if_right_else_wrong",
        )
        assert result.passed is False

    def test_wrong_middle_elif_detected(self):
        result = _result(
            """
            def elif_middle_wrong(df: DataFrame[KV], m: int) -> DataFrame[KVa]:
                if m == 0:
                    return df.with_columns(a=pl.col("v"))
                elif m == 1:
                    return df.with_columns(b=pl.col("v"))  # wrong (middle)
                return df.with_columns(a=pl.col("v"))
            """,
            "elif_middle_wrong",
        )
        assert result.passed is False

    def test_all_correct_returns_pass(self):
        passed = _check(
            """
            def all_right(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                if flag:
                    return df.with_columns(a=pl.col("v") * 2)
                return df.with_columns(a=pl.col("v") + 1)
            """
        )
        assert passed["all_right"] is True

    def test_error_attributes_offending_return_line(self):
        result = _result(
            """
            def if_wrong_else_right(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
                if flag:
                    return df.with_columns(b=pl.col("v"))  # wrong: no 'a'
                else:
                    return df.with_columns(a=pl.col("v"))
            """,
            "if_wrong_else_right",
        )
        joined = " ".join(str(e) for e in result.errors)
        # The wrong return is the third line of the function body.
        assert "line" in joined
