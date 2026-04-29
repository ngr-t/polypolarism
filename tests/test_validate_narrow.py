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
