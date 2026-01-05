"""Tests for checker."""

import pytest
import textwrap

from polypolarism.types import (
    FrameType,
    Int64,
    Float64,
    Utf8,
    UInt32,
    Nullable,
)
from polypolarism.analyzer import FunctionAnalysis
from polypolarism.checker import (
    check_function,
    check_source,
    CheckResult,
    TypeMismatch,
    MissingColumn,
    ExtraColumn,
    TypeDifference,
)


class TestCheckFunctionBasic:
    """Test basic function checking."""

    def test_matching_types_pass(self):
        """Function with matching declared and inferred types passes."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is True
        assert len(result.errors) == 0

    def test_analysis_errors_propagate(self):
        """Analysis errors are included in check result."""
        analysis = FunctionAnalysis(
            name="bad_func",
            lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=None,
            errors=["Column 'missing' not found in DataFrame"],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any("missing" in str(e) for e in result.errors)


class TestCheckMissingColumn:
    """Test detection of missing columns."""

    def test_detects_missing_column(self):
        """Detect when declared column is missing from inferred type."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64(), "name": Utf8()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, MissingColumn) for e in result.errors)
        missing_errors = [e for e in result.errors if isinstance(e, MissingColumn)]
        assert any(e.column == "name" for e in missing_errors)

    def test_missing_column_error_message(self):
        """Missing column error has helpful message."""
        error = MissingColumn("name", Utf8())

        message = str(error)

        assert "name" in message
        assert "Utf8" in message


class TestCheckExtraColumn:
    """Test detection of extra columns."""

    def test_detects_extra_column(self):
        """Detect when inferred type has column not in declared type."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"id": Int64(), "name": Utf8()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64(), "name": Utf8()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, ExtraColumn) for e in result.errors)
        extra_errors = [e for e in result.errors if isinstance(e, ExtraColumn)]
        assert any(e.column == "name" for e in extra_errors)

    def test_extra_column_error_message(self):
        """Extra column error has helpful message."""
        error = ExtraColumn("extra", Float64())

        message = str(error)

        assert "extra" in message


class TestCheckTypeDifference:
    """Test detection of type differences."""

    def test_detects_type_mismatch(self):
        """Detect when column has different type than declared."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Float64()}),
            inferred_return_type=FrameType({"value": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)
        type_errors = [e for e in result.errors if isinstance(e, TypeDifference)]
        assert any(
            e.column == "value" and e.declared == Float64() and e.inferred == Int64()
            for e in type_errors
        )

    def test_type_difference_error_message(self):
        """Type difference error has helpful message."""
        error = TypeDifference("value", declared=Float64(), inferred=Int64())

        message = str(error)

        assert "value" in message
        assert "Float64" in message
        assert "Int64" in message


class TestCheckNullability:
    """Test nullability checking."""

    def test_nullable_inferred_matches_nullable_declared(self):
        """Nullable inferred type matches nullable declared type."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"value": Nullable(Int64())})},
            declared_return_type=FrameType({"value": Nullable(Int64())}),
            inferred_return_type=FrameType({"value": Nullable(Int64())}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is True

    def test_inferred_nullable_declared_non_nullable_fails(self):
        """Inferred Nullable when declared non-nullable fails."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Int64()}),
            inferred_return_type=FrameType({"value": Nullable(Int64())}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any(isinstance(e, TypeDifference) for e in result.errors)

    def test_inferred_non_nullable_declared_nullable_passes(self):
        """Inferred non-nullable is compatible with declared nullable."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"value": Int64()})},
            declared_return_type=FrameType({"value": Nullable(Int64())}),
            inferred_return_type=FrameType({"value": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        # Non-nullable is a subtype of nullable
        assert result.passed is True


class TestCheckSource:
    """Test source code checking."""

    def test_check_valid_source(self):
        """Check source with valid function."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def identity(
                data: DF["{id: Int64, name: Utf8}"],
            ) -> DF["{id: Int64, name: Utf8}"]:
                return data
        ''')

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True

    def test_check_invalid_source_detects_mismatch(self):
        """Check source detects type mismatch."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def wrong_return(
                data: DF["{id: Int64}"],
            ) -> DF["{id: Int64, extra: Utf8}"]:
                return data
        ''')

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is False
        assert any(isinstance(e, MissingColumn) for e in results[0].errors)

    def test_check_source_with_join(self):
        """Check source with join operation."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def merge(
                left: DF["{id: Int64, name: Utf8}"],
                right: DF["{id: Int64, value: Float64}"],
            ) -> DF["{id: Int64, name: Utf8, value: Float64}"]:
                return left.join(right, on="id", how="inner")
        ''')

        results = check_source(source)

        assert len(results) == 1
        assert results[0].passed is True


class TestCheckResult:
    """Test CheckResult data class."""

    def test_check_result_function_name(self):
        """CheckResult includes function name."""
        analysis = FunctionAnalysis(
            name="my_function",
            lineno=1,
            input_types={},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert result.function_name == "my_function"

    def test_check_result_repr(self):
        """CheckResult has readable repr."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=FrameType({"id": Int64()}),
            errors=[],
        )

        result = check_function(analysis)

        assert "process" in repr(result)


class TestNoReturnTypeInferred:
    """Test cases where return type cannot be inferred."""

    def test_no_inferred_type_fails(self):
        """If no return type is inferred, check fails."""
        analysis = FunctionAnalysis(
            name="process",
            lineno=1,
            input_types={"data": FrameType({"id": Int64()})},
            declared_return_type=FrameType({"id": Int64()}),
            inferred_return_type=None,  # Could not infer
            errors=[],
        )

        result = check_function(analysis)

        assert result.passed is False
        assert any("infer" in str(e).lower() for e in result.errors)
