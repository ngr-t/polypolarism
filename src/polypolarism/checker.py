"""Declaration vs inference result checker."""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from polypolarism.analyzer import FunctionAnalysis, _cast_verdict, analyze_source
from polypolarism.diagnostics import PLY040, tag
from polypolarism.types import (
    NUMERIC_DTYPES,
    Array,
    DataType,
    Enum,
    FrameType,
    List,
    Nullable,
    Span,
    Struct,
    Unknown,
)


class TypeMismatch:
    """Base class for type mismatch errors.

    The whole declared-vs-inferred return-type family shares one diagnostic
    code, ``PLY040`` (issue #70); the message distinguishes the kind.

    ``primary`` / ``declared_span`` are per-column source spans (issue #110),
    excluded from equality so they never perturb the error comparisons the
    test suite relies on. ``primary`` is the body expression that produced
    the offending column (with_columns / agg / alias keyword, else ``None``
    → the diagnostic layer falls back to the return-statement line);
    ``declared_span`` is the schema field it was checked against. Set by
    ``_check_one_frame`` from the two frames' ``column_spans`` side maps.
    """

    code = PLY040

    # Declared here (not as dataclass fields) so the subclasses can attach
    # them with ``compare=False`` without each restating the contract.


@dataclass
class MissingColumn(TypeMismatch):
    """Error when declared column is missing from inferred type."""

    column: str
    expected_type: DataType
    primary: Span | None = field(default=None, compare=False)
    declared_span: Span | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return tag(self.code, f"Missing column '{self.column}' of type {self.expected_type}")


@dataclass
class ExtraColumn(TypeMismatch):
    """Error when inferred type has column not in declared type."""

    column: str
    inferred_type: DataType
    primary: Span | None = field(default=None, compare=False)
    declared_span: Span | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return tag(
            self.code,
            f"Extra column '{self.column}' of type {self.inferred_type} "
            "not in declared return type",
        )


@dataclass
class TypeDifference(TypeMismatch):
    """Error when column has different type than declared."""

    column: str
    declared: DataType
    inferred: DataType
    primary: Span | None = field(default=None, compare=False)
    declared_span: Span | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return tag(
            self.code,
            f"Column '{self.column}' has type {self.inferred}, "
            f"but declared type is {self.declared}",
        )


@dataclass
class InferenceFailure:
    """Error when return type could not be inferred."""

    code = PLY040

    message: str

    def __str__(self) -> str:
        return tag(self.code, self.message)


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
    # Leniency notes: per-column records of checks that passed only via a
    # leniency rule (Unknown compatibility, open-frame missing-column skip,
    # coerce-tolerated dtype difference). ``passed`` ignores them — they
    # exist to make leniency-mediated passes *visible* (golden files render
    # them), so a fixture that passes only because inference degraded to
    # Unknown can't silently mask a false negative (issue #47).
    leniency: list[str] = field(default_factory=list)
    # Inference trace, pre-rendered (populated only under --verbose).
    trace: list[str] = field(default_factory=list)
    # Inferred vs declared frame context for each failing return point.
    # Each entry: (lineno_or_None, inferred_render, declared_render).
    # ``lineno`` is None for single-return functions (no "at line N" prefix).
    # Populated whenever there are schema-level return errors so the CLI
    # can show the full frame shapes alongside the column-level error list.
    mismatch_frames: list[tuple[int | None, str, str]] = field(default_factory=list)

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


class Verdict(NamedTuple):
    """Outcome of a compatibility check, carrying *why* a pass happened.

    ``reason`` is ``None`` for sound passes (exact match, Nullable
    widening) and a short human-readable note when the pass relied on a
    leniency rule. Failures always carry ``reason=None``.
    """

    ok: bool
    reason: str | None = None


_VIA_UNKNOWN = "passed via Unknown"


def _subtype_verdict(inferred: DataType, declared: DataType) -> Verdict:
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

    A pass that relied on the Unknown rule (at any nesting depth) carries
    ``reason=_VIA_UNKNOWN`` so callers can surface the leniency.
    """
    # Unknown on either side (after Nullable unwrap) always passes.
    if isinstance(_get_base_type(inferred), Unknown) or isinstance(
        _get_base_type(declared), Unknown
    ):
        return Verdict(True, _VIA_UNKNOWN)

    # Exact match
    if inferred == declared:
        return Verdict(True)

    # Nullable inferred cannot fill a non-nullable declared slot.
    if isinstance(inferred, Nullable) and not isinstance(declared, Nullable):
        return Verdict(False)

    inferred_base = _get_base_type(inferred)
    declared_base = _get_base_type(declared)

    # Enum categories are part of dtype identity (issue #67): polars and
    # pandera reject a category-sequence mismatch at runtime, so two
    # concrete category tuples must be exactly equal (order included).
    # A statically-unknown side (``categories=None`` — a bare ``pl.Enum``
    # reference or a non-literal list) is a wildcard, surfaced as a
    # leniency note like the unknown Array width below.
    if isinstance(inferred_base, Enum) and isinstance(declared_base, Enum):
        if inferred_base.categories is None or declared_base.categories is None:
            if inferred_base.categories == declared_base.categories:
                return Verdict(True)  # both unknown — nothing to compare
            return Verdict(True, "passed via unknown Enum categories")
        return Verdict(inferred_base.categories == declared_base.categories)

    # Struct fields are dtype identity at runtime (polars struct dtypes
    # compare exactly). An OPEN side (backlog C-9 — bare ``pl.Struct`` /
    # unreadable construction) is a wildcard over its UNKNOWN fields, but
    # its pinned fields still prove: an overlapping-pin conflict fails,
    # and a pin provably absent from a CLOSED other side fails (struct
    # dtypes are exact, so a definite extra field cannot match).
    if isinstance(inferred_base, Struct) and isinstance(declared_base, Struct):
        if not inferred_base.open and not declared_base.open:
            return Verdict(inferred_base.fields == declared_base.fields)
        for name, inferred_field in inferred_base.fields.items():
            declared_field = declared_base.fields.get(name)
            if declared_field is None:
                if not declared_base.open:
                    return Verdict(False)
                continue
            field_verdict = _subtype_verdict(inferred_field, declared_field)
            if not field_verdict.ok:
                return Verdict(False)
        if not inferred_base.open:
            for name in declared_base.fields:
                if name not in inferred_base.fields:
                    return Verdict(False)
        return Verdict(True, "passed via open Struct fields")

    # List / Array containers: compare element types with the same rules so
    # the Unknown leniency reaches nested dtypes. Array vs List falls
    # through to False — probed (issue #53): pandera rejects a List column
    # where ``pl.Array(...)`` is declared and vice versa.
    if isinstance(inferred_base, List) and isinstance(declared_base, List):
        return _subtype_verdict(inferred_base.inner, declared_base.inner)
    if isinstance(inferred_base, Array) and isinstance(declared_base, Array):
        if (
            inferred_base.width is not None
            and declared_base.width is not None
            and inferred_base.width != declared_base.width
        ):
            # Probed (backlog C-7): pandera rejects an Array width mismatch
            # and coerce cannot repair it ("cannot cast Array to a
            # different width").
            return Verdict(False)
        verdict = _subtype_verdict(inferred_base.inner, declared_base.inner)
        if verdict.ok and verdict.reason is None and inferred_base.width != declared_base.width:
            # One side's width is statically unknown — surface the leniency
            # like the Unknown rule (ADR-0003 visibility).
            return Verdict(True, "passed via unknown Array width")
        return verdict

    # Non-nullable is subtype of nullable with same base type
    if isinstance(declared, Nullable) and not isinstance(inferred, Nullable):
        return Verdict(inferred_base == declared_base)

    return Verdict(False)


def _is_subtype(inferred: DataType, declared: DataType) -> bool:
    """Boolean wrapper around :func:`_subtype_verdict` (see its docstring).

    Kept for call sites that only need the yes/no answer (property tests,
    ops modules); :func:`check_function` uses the verdict directly to
    record leniency notes.
    """
    return _subtype_verdict(inferred, declared).ok


def _is_coercible_difference(inferred: DataType, declared: DataType) -> bool:
    """True when Pandera ``coerce=True`` would resolve this dtype difference.

    Direction-aware (issue #58): coerce casts the INFERRED dtype into the
    DECLARED dtype, so the difference is tolerated when that cast is
    probed value-independent (``analyzer._cast_verdict`` == "always" —
    e.g. anything-formattable -> String, bool <-> numeric, Datetime tz
    changes). Value-DEPENDENT casts (Utf8 -> Int64, Enum targets, ...)
    stay errors — the boundary pin is
    ``tests/fixtures/invalid/coerce_limits.py``.

    Two deliberate policy exceptions on top of the verdict:

    - numeric -> numeric is always tolerated, including value-dependent
      narrowing (pre-#58 behavior, kept by mandate);
    - container differences involving an Array side are tolerated (issue
      #53): pandera coerce casts List -> declared Array (valid whenever
      the widths line up — value-dependent, so flagging would be a false
      positive) and Array -> declared List (always valid). List-vs-List
      differences recurse on the element dtypes instead (probed:
      coerce casts elements; element nullability is preserved).

    Coercion does not remove nulls: a ``Nullable`` inferred side can only
    coerce into a ``Nullable`` declared side.

    Reused by ``analyzer._is_frame_subtype`` for the function-argument
    position (``pa.check_types`` coerces input frames too).
    """
    if isinstance(inferred, Nullable) and not isinstance(declared, Nullable):
        return False
    inferred_base = _get_base_type(inferred)
    declared_base = _get_base_type(declared)
    if isinstance(inferred_base, (List, Array)) and isinstance(declared_base, (List, Array)):
        if (
            isinstance(inferred_base, Array)
            and isinstance(declared_base, Array)
            and inferred_base.width is not None
            and declared_base.width is not None
            and inferred_base.width != declared_base.width
        ):
            # Probed (backlog C-7): coerce's underlying cast raises
            # "cannot cast Array to a different width".
            return False
        if isinstance(inferred_base, Array) or isinstance(declared_base, Array):
            return True
        return _is_coercible_difference(inferred_base.inner, declared_base.inner)
    if type(inferred_base) in NUMERIC_DTYPES and type(declared_base) in NUMERIC_DTYPES:
        return True
    return _cast_verdict(inferred_base, declared_base) == "always"


def _check_one_frame(
    declared: FrameType,
    inferred: FrameType,
    return_line: int | None = None,
) -> tuple[list[CheckError], list[str]]:
    """Check one return frame against the declared type.

    Returns ``(errors, leniency)`` for that single frame.

    ``return_line`` is the source line of the return statement this frame
    came from; it is the PRIMARY-span fallback for any column whose
    producing expression we did not stamp (issue #110) — so even a
    pass-through column underlines the ``return`` line rather than the whole
    function.
    """
    errors: list[CheckError] = []
    leniency: list[str] = []

    def primary_span(col_name: str) -> Span | None:
        """The inferred-side expression span for ``col_name`` (the column's
        producing ``with_columns`` / ``agg`` / ``alias`` keyword), falling
        back to the return statement line when the column was not stamped."""
        stamped = inferred.column_spans.get(col_name)
        if stamped is not None:
            return stamped
        if return_line is not None:
            return Span(line=return_line, column=0)
        return None

    def declared_span(col_name: str) -> Span | None:
        """The declared schema-field span for ``col_name`` (SECONDARY)."""
        return declared.column_spans.get(col_name)

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
    # declared column missing from it is not provably absent — no error,
    # but the skip is recorded as a leniency note.
    for col_name, declared_spec in declared.columns.items():
        inferred_spec = inferred.columns.get(col_name)
        if inferred_spec is None:
            if declared_spec.required:
                # ``lacks`` is provable absence: a closed frame without
                # the column, or an open frame that removed it via
                # drop/rename (negative knowledge, issue #78).
                if inferred.lacks(col_name):
                    errors.append(
                        MissingColumn(
                            col_name,
                            declared_spec.dtype,
                            primary=primary_span(col_name),
                            declared_span=declared_span(col_name),
                        )
                    )
                else:
                    leniency.append(f"column '{col_name}': not provably absent (open frame)")
            continue
        # Inferred frame may have the column as optional (may be absent);
        # if declared expects it always-present, that's a mismatch.
        if declared_spec.required and not inferred_spec.required:
            errors.append(
                MissingColumn(
                    col_name,
                    declared_spec.dtype,
                    primary=primary_span(col_name),
                    declared_span=declared_span(col_name),
                )
            )
            continue
        # Presence-only column (issue #109): a guard proved the column EXISTS
        # but not its dtype. The blanket ``Unknown`` leniency would wrongly
        # accept it against any declared dtype; withhold it unless the
        # declared side is itself ``Unknown`` or the schema coerces (pandera
        # casts at validation time). Otherwise the dtype is unproven and the
        # presence guard alone cannot satisfy a concrete, non-coerce slot.
        if inferred_spec.presence_only and not isinstance(
            _get_base_type(declared_spec.dtype), Unknown
        ):
            if declared.coerce:
                leniency.append(
                    f"column '{col_name}': presence proven by guard, dtype coerced "
                    f"to {declared_spec.dtype}"
                )
            else:
                errors.append(
                    MissingColumn(
                        col_name,
                        declared_spec.dtype,
                        primary=primary_span(col_name),
                        declared_span=declared_span(col_name),
                    )
                )
            continue
        verdict = _subtype_verdict(inferred_spec.dtype, declared_spec.dtype)
        if verdict.ok:
            if verdict.reason is not None:
                leniency.append(f"column '{col_name}': {verdict.reason}")
        elif declared.coerce and _is_coercible_difference(inferred_spec.dtype, declared_spec.dtype):
            # Pandera ``Config.coerce`` casts coercible dtypes at
            # validation time — those differences are not errors.
            leniency.append(
                f"column '{col_name}': {inferred_spec.dtype} -> {declared_spec.dtype} via coerce"
            )
        else:
            errors.append(
                TypeDifference(
                    col_name,
                    declared_spec.dtype,
                    inferred_spec.dtype,
                    primary=primary_span(col_name),
                    declared_span=declared_span(col_name),
                )
            )

    # Extra columns (only flagged for strict declared schemas). An
    # OPTIONAL (required=False) inferred column may be absent at runtime —
    # value-dependent, not provable (issue #84) — recorded as leniency.
    if declared.strict:
        for col_name, inferred_spec in inferred.columns.items():
            if col_name not in declared.columns:
                if inferred_spec.required:
                    # An extra column has no declared field; its primary span
                    # is still the producing expression (else return line).
                    errors.append(
                        ExtraColumn(
                            col_name,
                            inferred_spec.dtype,
                            primary=primary_span(col_name),
                        )
                    )
                else:
                    leniency.append(
                        f"column '{col_name}': optional extra vs strict schema "
                        f"(absent inputs pass; present ones fail at runtime)"
                    )

    return errors, leniency


_ANY_CODE = _re.compile(r"\[(PL[YW]\d{3})\]")


def _code_of(err: CheckError | str) -> str | None:
    """Extract the PLY/PLW code from any error value.

    Handles typed errors (``TypeMismatch``/``InferenceFailure`` with a
    ``.code`` attribute), tagged strings (``"[PLY040] ..."``), and the
    multi-return prefix form (``"at line N: [PLY040] ..."``).
    """
    if isinstance(err, str):
        m = _ANY_CODE.search(err)
        return m.group(1) if m else None
    return getattr(err, "code", None)


def _apply_suppression(
    items: list,
    suppressed: frozenset[str] | None,
) -> list:
    """Filter ``items`` according to a ``# type: ignore`` suppression spec.

    ``suppressed=frozenset()``  — no suppression; return items unchanged.
    ``suppressed=None``         — blanket suppress; return empty list.
    ``suppressed={...}``        — remove items whose code is in the set.
    """
    if suppressed is not None and not suppressed:
        return items  # empty frozenset → nothing to suppress
    return [item for item in items if not (suppressed is None or _code_of(item) in suppressed)]


def check_function(analysis: FunctionAnalysis) -> CheckResult:
    """
    Check a single function's declared return type against its inferred return type.

    Args:
        analysis: The function analysis result from analyzer

    Returns:
        CheckResult with pass/fail status and any errors
    """
    errors: list[CheckError] = []
    leniency: list[str] = []
    mismatch_frames: list[tuple[int | None, str, str]] = []
    trace = [f"L{e.lineno:<4} {e.label:<28} {e.result}" for e in analysis.trace]

    # Include any analysis errors
    for err in analysis.errors:
        errors.append(err)

    # Check if we have both declared and inferred types
    if analysis.declared_return_type is None:
        # No declared type to check against
        errors = _apply_suppression(errors, analysis.suppressed_codes)
        warnings = _apply_suppression(list(analysis.warnings), analysis.suppressed_codes)
        return CheckResult(
            function_name=analysis.name,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            trace=trace,
        )

    declared = analysis.declared_return_type
    # When multiple return points exist (multi-branch / ternary), prefix each
    # error with its source line so the user can pinpoint the offending return.
    multi = len(analysis.return_frames) > 1

    if analysis.return_frames:
        # Validate every recorded return point (issues #94, #95).
        for lineno, inferred in analysis.return_frames:
            frame_errors, frame_leniency = _check_one_frame(declared, inferred, return_line=lineno)
            if frame_errors:
                mismatch_frames.append(
                    (lineno if multi else None, inferred.render(), declared.render())
                )
                if multi:
                    for e in frame_errors:
                        errors.append(f"at line {lineno}: {e}")
                else:
                    errors.extend(frame_errors)
            leniency.extend(frame_leniency)
    elif analysis.inferred_return_type is None:
        errors.append(InferenceFailure("Could not infer return type"))
    else:
        frame_errors, frame_leniency = _check_one_frame(declared, analysis.inferred_return_type)
        if frame_errors:
            mismatch_frames.append(
                (None, analysis.inferred_return_type.render(), declared.render())
            )
        errors.extend(frame_errors)
        leniency.extend(frame_leniency)

    errors = _apply_suppression(errors, analysis.suppressed_codes)
    warnings = _apply_suppression(list(analysis.warnings), analysis.suppressed_codes)
    return CheckResult(
        function_name=analysis.name,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        leniency=leniency,
        trace=trace,
        mismatch_frames=mismatch_frames if errors else [],
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
