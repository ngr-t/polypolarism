"""Declaration vs inference result checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from polypolarism.types import DataType, FrameType, Nullable
from polypolarism.analyzer import analyze_source, FunctionAnalysis


class TypeMismatch:
    """Base class for type mismatch errors."""

    pass


@dataclass
class MissingColumn(TypeMismatch):
    """Error when declared column is missing from inferred type."""

    column: str
    expected_type: DataType

    def __str__(self) -> str:
        return f"Missing column '{self.column}' of type {self.expected_type}"


@dataclass
class ExtraColumn(TypeMismatch):
    """Error when inferred type has column not in declared type."""

    column: str
    inferred_type: DataType

    def __str__(self) -> str:
        return f"Extra column '{self.column}' of type {self.inferred_type} not in declared return type"


@dataclass
class TypeDifference(TypeMismatch):
    """Error when column has different type than declared."""

    column: str
    declared: DataType
    inferred: DataType

    def __str__(self) -> str:
        return (
            f"Column '{self.column}' has type {self.inferred}, "
            f"but declared type is {self.declared}"
        )


@dataclass
class InferenceFailure:
    """Error when return type could not be inferred."""

    message: str

    def __str__(self) -> str:
        return self.message


CheckError = Union[TypeMismatch, InferenceFailure, str]


@dataclass
class CheckResult:
    """Result of type checking a single function."""

    function_name: str
    passed: bool
    errors: list[CheckError] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"CheckResult({self.function_name}: {status}, errors={len(self.errors)})"


def _get_base_type(dtype: DataType) -> DataType:
    """Get base type, unwrapping Nullable if present."""
    if isinstance(dtype, Nullable):
        return dtype.inner
    return dtype


def _is_subtype(inferred: DataType, declared: DataType) -> bool:
    """Check if inferred type is a subtype of declared type.

    Rules:
    - T is a subtype of T
    - T is a subtype of Nullable[T] (non-nullable can be used where nullable is expected)
    - Nullable[T] is NOT a subtype of T (nullable cannot be used where non-nullable is expected)
    """
    # Exact match
    if inferred == declared:
        return True

    # Non-nullable is subtype of nullable with same base type
    if isinstance(declared, Nullable) and not isinstance(inferred, Nullable):
        return _get_base_type(inferred) == _get_base_type(declared)

    return False


def check_function(analysis: FunctionAnalysis) -> CheckResult:
    """
    Check a single function's declared return type against its inferred return type.

    Args:
        analysis: The function analysis result from analyzer

    Returns:
        CheckResult with pass/fail status and any errors
    """
    errors: list[CheckError] = []

    # Include any analysis errors
    for err in analysis.errors:
        errors.append(err)

    # Check if we have both declared and inferred types
    if analysis.declared_return_type is None:
        # No declared type to check against
        return CheckResult(
            function_name=analysis.name,
            passed=len(errors) == 0,
            errors=errors,
        )

    if analysis.inferred_return_type is None:
        errors.append(InferenceFailure("Could not infer return type"))
        return CheckResult(
            function_name=analysis.name,
            passed=False,
            errors=errors,
        )

    declared = analysis.declared_return_type
    inferred = analysis.inferred_return_type

    # Check for missing columns (in declared but not in inferred)
    for col_name, col_type in declared.columns.items():
        if col_name not in inferred.columns:
            errors.append(MissingColumn(col_name, col_type))

    # Check for extra columns (in inferred but not in declared)
    for col_name, col_type in inferred.columns.items():
        if col_name not in declared.columns:
            errors.append(ExtraColumn(col_name, col_type))

    # Check for type differences in common columns
    for col_name in declared.columns:
        if col_name in inferred.columns:
            declared_type = declared.columns[col_name]
            inferred_type = inferred.columns[col_name]

            if not _is_subtype(inferred_type, declared_type):
                errors.append(TypeDifference(col_name, declared_type, inferred_type))

    return CheckResult(
        function_name=analysis.name,
        passed=len(errors) == 0,
        errors=errors,
    )


def check_source(source: str) -> list[CheckResult]:
    """
    Check all functions with DF annotations in source code.

    Args:
        source: Python source code as a string

    Returns:
        List of CheckResult for each function with DF annotations
    """
    analyses = analyze_source(source)
    return [check_function(analysis) for analysis in analyses]
