"""Tests for Pandera Schema.validate(df) narrowing in the analyzer."""

from __future__ import annotations

import textwrap

from polypolarism.checker import check_source

COMMON_SCHEMAS = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame, LazyFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64
"""


def _check(body: str) -> bool:
    """Run the full source through analyze + check, return True iff all pass."""
    src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(body)
    results = check_source(src)
    return all(r.passed for r in results)


class TestAssignNarrowing:
    def test_assign_validate_narrows_lhs(self):
        # df = Schema.validate(raw) should bind df to CleanSchema's frame type.
        assert _check(
            """
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                df = CleanSchema.validate(raw)
                return df.select(pl.col("id"), pl.col("value"))
            """
        )


class TestBareStatementNarrowing:
    def test_bare_validate_narrows_var(self):
        # Schema.validate(raw) on its own should retype raw downstream.
        assert _check(
            """
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                CleanSchema.validate(raw)
                return raw.select(pl.col("id"), pl.col("value"))
            """
        )

    def test_bare_validate_for_unknown_schema_is_noop(self):
        # Unknown schema name should not crash and should not narrow.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                NotASchema.validate(raw)
                return raw.select(pl.col("id"), pl.col("value"))
            """
        )
        # Should fail because 'value' is not in raw's schema.
        results = check_source(src)
        assert any(not r.passed for r in results)


class TestPipeValidate:
    def test_pipe_validate_narrows_chain(self):
        assert _check(
            """
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                return raw.pipe(CleanSchema.validate).select(pl.col("id"), pl.col("value"))
            """
        )

    def test_pipe_unknown_callable_acts_as_identity(self):
        # pipe with a non-validate callable should pass receiver type through.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            def f(raw: DataFrame[RawSchema]) -> DataFrame[RawSchema]:
                return raw.pipe(some_other_func)
            """
        )
        results = check_source(src)
        # Not crashing is the main thing; declared = inferred should pass.
        assert all(r.passed for r in results)


class TestCollectAfterValidate:
    def test_collect_preserves_frame_type(self):
        assert _check(
            """
            def f(raw: LazyFrame[RawSchema]) -> DataFrame[CleanSchema]:
                df = CleanSchema.validate(raw).collect()
                return df.select(pl.col("id"), pl.col("value"))
            """
        )


class TestTypingCastPassthrough:
    """``typing.cast(T, expr)`` is a static-typing no-op — polypolarism should
    look through it and infer from the inner expression."""

    def test_return_cast_validate(self):
        # ``return cast(DataFrame[Schema], Schema.validate(df))`` — the cast
        # should not defeat narrowing; the inferred type comes from the
        # inner ``Schema.validate(df)``.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            from typing import cast
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                return cast(DataFrame[CleanSchema], CleanSchema.validate(raw))
            """
        )
        results = check_source(src)
        assert all(r.passed for r in results), [r.errors for r in results]

    def test_qualified_typing_cast(self):
        # ``import typing; typing.cast(...)`` form should also pass through.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            import typing
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                return typing.cast(DataFrame[CleanSchema], CleanSchema.validate(raw))
            """
        )
        results = check_source(src)
        assert all(r.passed for r in results), [r.errors for r in results]

    def test_assign_cast_narrows(self):
        # ``df = cast(T, Schema.validate(df))`` — assignment narrowing.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            from typing import cast
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                df = cast(DataFrame[CleanSchema], CleanSchema.validate(raw))
                return df.select(pl.col("id"), pl.col("value"))
            """
        )
        results = check_source(src)
        assert all(r.passed for r in results), [r.errors for r in results]

    def test_cast_does_not_lie(self):
        # The cast claim should NOT be trusted blindly — we still infer from
        # the inner expression and report a mismatch.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            from typing import cast
            def f(raw: DataFrame[RawSchema]) -> DataFrame[CleanSchema]:
                return cast(DataFrame[CleanSchema], raw)
            """
        )
        results = check_source(src)
        # raw is RawSchema, declared CleanSchema — must surface a mismatch
        # rather than passing because of the cast.
        assert any(not r.passed for r in results)


class TestNoNarrowingAcrossInvalidPattern:
    def test_validate_call_buried_in_if_does_not_narrow(self):
        # This is the documented limitation: only top-level statements narrow.
        # Using raw inside the function w/o narrowing means 'value' should not exist.
        src = textwrap.dedent(COMMON_SCHEMAS) + textwrap.dedent(
            """
            def f(raw: DataFrame[RawSchema], cond: bool) -> DataFrame[CleanSchema]:
                if cond:
                    CleanSchema.validate(raw)
                return raw.select(pl.col("id"), pl.col("value"))
            """
        )
        results = check_source(src)
        # Expect a failure: 'value' is unknown because narrowing was nested.
        assert any(not r.passed for r in results)


PATITO_SCHEMAS = """
import polars as pl
import patito as pt


class S(pt.Model):
    a: int
"""


def _check_patito(body: str) -> list:
    src = textwrap.dedent(PATITO_SCHEMAS) + textwrap.dedent(body)
    return check_source(src)


class TestPatitoValidateSuperfluous:
    """Patito Model.validate(df, allow_superfluous_columns=...) (ADR-0010 #4)."""

    def test_allow_superfluous_opens_frame(self):
        # The kwarg passes extras through, so accessing one is not a miss.
        results = _check_patito(
            """
            def f(df: pl.DataFrame) -> pl.DataFrame:
                S.validate(df, allow_superfluous_columns=True)
                return df.select("a", "extra")
            """
        )
        assert all(r.passed for r in results), [r.errors for r in results]

    def test_default_validate_stays_strict(self):
        # Without the kwarg the narrowed frame is closed: 'extra' is a miss.
        results = _check_patito(
            """
            def f(df: pl.DataFrame) -> pl.DataFrame:
                S.validate(df)
                return df.select("a", "extra")
            """
        )
        assert any(not r.passed for r in results)
        assert any("pple-column-not-found" in e for r in results for e in r.errors)
