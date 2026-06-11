"""Declaration vs inference result checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polypolarism.analyzer import FunctionAnalysis, analyze_source
from polypolarism.types import NUMERIC_DTYPES, Array, DataType, Datetime, List, Nullable, Unknown


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
        return (
            f"Extra column '{self.column}' of type {self.inferred_type} not in declared return type"
        )


@dataclass
class TypeDifference(TypeMismatch):
    """Error when column has different type than declared."""

    column: str
    declared: DataType
    inferred: DataType

    def __str__(self) -> str:
        return (
            f"Column '{self.column}' has type {self.inferred}, but declared type is {self.declared}"
        )


@dataclass
class InferenceFailure:
    """Error when return type could not be inferred."""

    message: str

    def __str__(self) -> str:
        return self.message


CheckError = TypeMismatch | InferenceFailure | str


@dataclass
class CheckResult:
    """Result of type checking a single function."""

    function_name: str
    passed: bool
    errors: list[CheckError] = field(default_factory=list)
    # Non-fatal advisories. ``passed`` ignores them — they exist to nudge
    # the user toward source changes that would let polypolarism check the
    # code precisely (e.g. adding ``return_dtype=`` to ``map_elements``).
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"CheckResult({self.function_name}: {status}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


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
    - Unknown is compatible with everything in both directions (gradual
      typing: uncertainty must not error), even ``Nullable[Unknown]`` vs a
      non-nullable declared type. The rule recurses into ``List`` so that
      ``List[Unknown]`` (e.g. an un-inferable ``list.eval`` body) satisfies
      a declared ``List[T]`` — but the column's own nullability is still
      enforced.
    """
    # Unknown on either side (after Nullable unwrap) always passes.
    if isinstance(_get_base_type(inferred), Unknown) or isinstance(
        _get_base_type(declared), Unknown
    ):
        return True

    # Exact match
    if inferred == declared:
        return True

    # Nullable inferred cannot fill a non-nullable declared slot.
    if isinstance(inferred, Nullable) and not isinstance(declared, Nullable):
        return False

    inferred_base = _get_base_type(inferred)
    declared_base = _get_base_type(declared)

    # List / Array containers: compare element types with the same rules so
    # the Unknown leniency reaches nested dtypes. Array vs List falls
    # through to False — probed (issue #53): pandera rejects a List column
    # where ``pl.Array(...)`` is declared and vice versa.
    if isinstance(inferred_base, List) and isinstance(declared_base, List):
        return _is_subtype(inferred_base.inner, declared_base.inner)
    if isinstance(inferred_base, Array) and isinstance(declared_base, Array):
        return _is_subtype(inferred_base.inner, declared_base.inner)

    # Non-nullable is subtype of nullable with same base type
    if isinstance(declared, Nullable) and not isinstance(inferred, Nullable):
        return inferred_base == declared_base

    return False


def _is_coercible_difference(inferred: DataType, declared: DataType) -> bool:
    """True when Pandera ``coerce=True`` would resolve this dtype difference.

    Coercion casts values between numeric dtypes, so both bases must be
    numeric — non-numeric mismatches (e.g. Utf8 vs Int64) still error
    under coerce. Datetime-vs-Datetime tz differences are also coercible
    (probed: pandera ``coerce=True`` casts tz-naive into a tz-aware schema
    and vice versa; issue #50). It does not remove nulls: a ``Nullable``
    inferred side can only coerce into a ``Nullable`` declared side.

    Reused by ``analyzer._is_frame_subtype`` for the function-argument
    position (``pa.check_types`` coerces input frames too).
    """
    if isinstance(inferred, Nullable) and not isinstance(declared, Nullable):
        return False
    inferred_base = _get_base_type(inferred)
    declared_base = _get_base_type(declared)
    if isinstance(inferred_base, Datetime) and isinstance(declared_base, Datetime):
        return True
    # Container differences involving an Array side are coercible (issue
    # #53). Probed: pandera coerce casts List -> declared Array (valid
    # whenever the widths line up — value-dependent, so flagging would be
    # a false positive) and Array -> declared List (always valid); every
    # probed polars cast between list-like containers is structurally
    # permissive (even Array(Date) -> List(Duration)). List-vs-List
    # differences keep the pre-existing behavior (not coercible).
    if isinstance(inferred_base, (List, Array)) and isinstance(declared_base, (List, Array)):
        return isinstance(inferred_base, Array) or isinstance(declared_base, Array)
    return type(inferred_base) in NUMERIC_DTYPES and type(declared_base) in NUMERIC_DTYPES


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
            warnings=list(analysis.warnings),
        )

    if analysis.inferred_return_type is None:
        errors.append(InferenceFailure("Could not infer return type"))
        return CheckResult(
            function_name=analysis.name,
            passed=False,
            errors=errors,
            warnings=list(analysis.warnings),
        )

    declared = analysis.declared_return_type
    inferred = analysis.inferred_return_type

    # Eager/lazy mismatch on the return type.
    if declared.is_lazy != inferred.is_lazy:
        expected_kind = "LazyFrame" if declared.is_lazy else "DataFrame"
        actual_kind = "LazyFrame" if inferred.is_lazy else "DataFrame"
        fix = ".collect() before returning" if inferred.is_lazy else ".lazy() before returning"
        errors.append(
            f"[PLY032] Return type expected {expected_kind}[...] but inferred "
            f"{actual_kind}[...]; {fix}."
        )

    # Required/optional + dtype check for declared columns. An open inferred
    # frame (``rest`` is not None) may hold extra unknown columns, so a
    # declared column missing from it is not provably absent — no error.
    for col_name, declared_spec in declared.columns.items():
        inferred_spec = inferred.columns.get(col_name)
        if inferred_spec is None:
            if declared_spec.required and inferred.rest is None:
                errors.append(MissingColumn(col_name, declared_spec.dtype))
            continue
        # Inferred frame may have the column as optional (may be absent);
        # if declared expects it always-present, that's a mismatch.
        if declared_spec.required and not inferred_spec.required:
            errors.append(MissingColumn(col_name, declared_spec.dtype))
            continue
        if not _is_subtype(inferred_spec.dtype, declared_spec.dtype):
            # Pandera ``Config.coerce`` casts coercible dtypes at
            # validation time — those differences are not errors.
            if declared.coerce and _is_coercible_difference(
                inferred_spec.dtype, declared_spec.dtype
            ):
                continue
            errors.append(TypeDifference(col_name, declared_spec.dtype, inferred_spec.dtype))

    # Extra columns (only flagged for strict declared schemas)
    if declared.strict:
        for col_name, inferred_spec in inferred.columns.items():
            if col_name not in declared.columns:
                errors.append(ExtraColumn(col_name, inferred_spec.dtype))

    return CheckResult(
        function_name=analysis.name,
        passed=len(errors) == 0,
        errors=errors,
        warnings=list(analysis.warnings),
    )


def check_source(source: str, file_path: Path | None = None) -> list[CheckResult]:
    """
    Check all functions with ``DataFrame[Schema]`` annotations in source code.

    Args:
        source: Python source code as a string
        file_path: Optional path of the file ``source`` came from.
            Forwarded to :func:`analyze_source` so cross-module
            ``from X import Schema`` references resolve on disk.

    Returns:
        List of CheckResult for each function with DataFrame[Schema] annotations
    """
    analyses = analyze_source(source, file_path=file_path)
    return [check_function(analysis) for analysis in analyses]
