"""Output formatting for diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from polypolarism.checker import (
    CheckResult,
    ExtraColumn,
    MissingColumn,
    TypeDifference,
    TypeMismatch,
)
from polypolarism.diagnostics import extract_code
from polypolarism.types import Span, render_dtype_annotation


class DiagnosticSeverity(Enum):
    """Severity level for diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    """A single diagnostic message with location information."""

    file: str
    line: int
    column: int
    message: str
    severity: DiagnosticSeverity
    end_line: int | None = None
    end_column: int | None = None
    source: str = "polypolarism"
    # Stable ``PLY###`` / ``PLW###`` code, exposed structurally so JSON
    # consumers don't have to regex the message prefix (issue #70). ``None``
    # for untagged diagnostics (parse / read failures).
    code: str | None = None
    # Secondary "declared here" location(s) for column mismatches (issue
    # #110): each entry is ``{line, column, end_line?, end_column?,
    # message}`` (1-indexed lines / 0-indexed columns, like the primary
    # range), shaped to map onto LSP ``relatedInformation``. Empty list →
    # the ``related`` key is omitted, keeping the JSON additive.
    related: list[dict] = field(default_factory=list)
    # Structured operands of the diagnostic, lifted from the data the
    # checker / analyzer already hold so consumers (the VS Code extension)
    # don't regex the message text (Batch A). All optional; each is emitted
    # only when applicable, keeping the JSON additive and backward-compatible
    # — the ``message`` text is unchanged.
    #
    # ``column_name`` is the DataFrame column the diagnostic is about (named
    # to avoid colliding with ``column``, the 0-indexed *source* position).
    # ``schema`` is the declaring schema (pple-undeclared-column). ``declared_type`` /
    # ``inferred_type`` are the canonical ``str`` renders of the dtypes the
    # pple-return-type family already carries (the same text the message shows).
    column_name: str | None = None
    schema: str | None = None
    declared_type: str | None = None
    inferred_type: str | None = None
    # Fix metadata for a pple-return-type "retype the schema field" quick fix (Batch B,
    # Request 3). ``suggested_annotation`` is a ready-to-insert pandera
    # annotation STRING for the INFERRED dtype (e.g. ``"pl.Float64"``);
    # ``None`` when the dtype is unrenderable → the key is omitted (never a
    # guess). ``declared_annotation_range`` is the {line, column, end_line,
    # end_column} of JUST the declared field's ANNOTATION node (issue #113 —
    # ``int`` / ``pl.Int64``, NOT the whole ``name: ann = pa.Field(...)``
    # line), so ``replace(range, suggested_annotation)`` rewrites only the
    # annotation; ``None`` when the span is unknown (e.g. a string forward-ref).
    suggested_annotation: str | None = None
    declared_annotation_range: dict | None = None
    # Fix object for a pple-undeclared-column "declare the column on the schema" quick fix
    # (Batch B, Request 2): ``{schema, column, schema_file, schema_insert_line,
    # suggested_dtype?}``. ``None`` when the fix cannot be determined soundly
    # (unknown schema file) → the key is omitted.
    fix: dict | None = None
    # "Relax the param" helper fields (Batch B, Request 4): the parameter
    # whose annotation a pple-undeclared-column suggests loosening and its annotation
    # ``{line, column, end_line, end_column}`` range. Both ``None`` when not
    # cleanly determinable → the keys are omitted.
    param_name: str | None = None
    param_annotation_range: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
        }
        if self.code is not None:
            result["code"] = self.code
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.end_column is not None:
            result["end_column"] = self.end_column
        if self.related:
            result["related"] = self.related
        # ``column`` (the int source position above) is unchanged; the
        # DataFrame column NAME goes under its own key to avoid clobbering it.
        if self.column_name is not None:
            result["column_name"] = self.column_name
        if self.schema is not None:
            result["schema"] = self.schema
        if self.declared_type is not None:
            result["declared_type"] = self.declared_type
        if self.inferred_type is not None:
            result["inferred_type"] = self.inferred_type
        if self.suggested_annotation is not None:
            result["suggested_annotation"] = self.suggested_annotation
        if self.declared_annotation_range is not None:
            result["declared_annotation_range"] = self.declared_annotation_range
        if self.fix is not None:
            result["fix"] = self.fix
        if self.param_name is not None:
            result["param_name"] = self.param_name
        if self.param_annotation_range is not None:
            result["param_annotation_range"] = self.param_annotation_range
        return result


def _error_to_message(error) -> str:
    """Convert a CheckError to a message string."""
    return str(error)


@dataclass
class FileResults:
    """Per-file group of check results, used by format_json for multi-file invocations."""

    file_path: str
    results: list[CheckResult]
    function_lines: dict[str, int] = field(default_factory=dict)
    function_end_lines: dict[str, int] = field(default_factory=dict)
    # Per-function schema summaries (``function_summaries``) — consumed by
    # editor integrations (hover) via ``--format json``.
    functions: list[dict] = field(default_factory=list)


def _frame_summary(frame) -> dict | None:
    """JSON-ready summary of a FrameType: rendered column dtypes plus the
    openness marker (an open frame may hold extra unknown columns —
    ADR-0006). ``None`` passes through for absent sides."""
    if frame is None:
        return None
    return {
        "columns": {name: str(spec) for name, spec in frame.columns.items()},
        "open": frame.rest is not None,
        "strict": frame.strict,
        "lazy": frame.is_lazy,
    }


def _function_summary(analysis) -> dict:
    """One ``function_summaries`` entry for a single analyzed function.

    The row-polymorphism fields (C-14 Tier 6) are added only when present so
    the JSON stays backward-compatible for pandera-only code: a positional
    ``@rowpoly("R")`` adds ``"row_var": "R"``; a keyword
    ``@rowpoly(a="R1", b="R2")`` adds ``"param_row_vars": {"a": "R1", ...}``;
    a function with neither carries neither key.
    """
    summary = {
        "name": analysis.name,
        "line": analysis.lineno,
        "end_line": analysis.end_lineno,
        "params": {name: _frame_summary(ft) for name, ft in analysis.input_types.items()},
        "declared_return": _frame_summary(analysis.declared_return_type),
        "inferred_return": _frame_summary(analysis.inferred_return_type),
    }
    row_var = getattr(analysis, "row_var", None)
    if row_var is not None:
        summary["row_var"] = row_var
    param_row_vars = getattr(analysis, "param_row_vars", None)
    if param_row_vars:
        summary["param_row_vars"] = dict(param_row_vars)
    return summary


def function_summaries(analyses) -> list[dict]:
    """Per-function schema summaries for editor hovers (D-11).

    One entry per analyzed function: source span, parameter frames, the
    declared / inferred return frames (dtypes rendered through their canonical
    ``str`` forms), and any bound row variable(s) from ``@rowpoly`` (C-14).
    """
    return [_function_summary(analysis) for analysis in analyses]


def _span_range(span: Span) -> tuple[int, int, int | None, int | None]:
    """Unpack a :class:`Span` into ``(line, column, end_line, end_column)``."""
    return (span.line, span.column, span.end_line, span.end_column)


def _span_dict(span: Span) -> dict:
    """A :class:`Span` as a ``{line, column, end_line?, end_column?}`` dict
    (1-indexed line, 0-indexed col — the same convention ``related`` uses).
    ``end_line`` / ``end_column`` are included only when known."""
    out: dict = {"line": span.line, "column": span.column}
    if span.end_line is not None:
        out["end_line"] = span.end_line
    if span.end_column is not None:
        out["end_column"] = span.end_column
    return out


def _related_from_declared(span: Span) -> dict:
    """LSP-``relatedInformation``-shaped secondary location for a declared
    schema field (issue #110)."""
    related = {
        "line": span.line,
        "column": span.column,
        "message": "declared here",
    }
    if span.end_line is not None:
        related["end_line"] = span.end_line
    if span.end_column is not None:
        related["end_column"] = span.end_column
    return related


def _structured_fields(error) -> dict[str, str]:
    """Structured operands of a check error, lifted from data the error
    object already carries (Batch A) — never parsed from the message.

    Returns a dict with any of ``column_name`` / ``schema`` /
    ``declared_type`` / ``inferred_type`` that apply; absent fields are left
    out so the JSON stays minimal.

    - pple-return-type family (typed errors): ``TypeDifference`` carries the column
      plus both dtypes; ``MissingColumn`` the column + its declared
      ``expected_type``; ``ExtraColumn`` the column + its ``inferred_type``.
      Each dtype is rendered via canonical ``str(...)`` — the same text the
      message already shows.
    - pple-undeclared-column / pple-column-not-found (analyzer ``TaggedError`` strings): carry ``column``
      and, for the non-strict-island pple-undeclared-column, the ``schema`` name.
    """
    fields: dict[str, str] = {}
    if isinstance(error, TypeDifference):
        fields["column_name"] = error.column
        fields["declared_type"] = str(error.declared)
        fields["inferred_type"] = str(error.inferred)
    elif isinstance(error, MissingColumn):
        fields["column_name"] = error.column
        fields["declared_type"] = str(error.expected_type)
    elif isinstance(error, ExtraColumn):
        fields["column_name"] = error.column
        fields["inferred_type"] = str(error.inferred_type)
    else:
        # Analyzer string errors: a ``TaggedError`` carries ``.column`` /
        # ``.schema`` attributes (plain ``str`` errors have neither).
        column = getattr(error, "column", None)
        if column is not None:
            fields["column_name"] = column
        schema = getattr(error, "schema", None)
        if schema is not None:
            fields["schema"] = schema
    return fields


def _retype_fix_fields(error) -> tuple[str | None, dict | None]:
    """Fix metadata for a pple-return-type retype quick fix (Batch B, Request 3):
    ``(suggested_annotation, declared_annotation_range)``.

    Only a :class:`TypeDifference` is a "retype the field" mismatch (the
    inferred dtype differs from the declared one). ``suggested_annotation``
    renders the INFERRED dtype to a pandera annotation string — ``None`` when
    unrenderable, so the caller omits the key rather than guess.
    ``declared_annotation_range`` comes from ``declared_annotation_span`` — the
    span of JUST the field's ANNOTATION node (``AnnAssign.annotation``), NOT
    the whole-field ``declared_span`` (issue #113), so a consumer doing
    ``replace(declared_annotation_range, suggested_annotation)`` turns
    ``total: int`` into ``total: pl.Float64`` while preserving the field name
    and any ``= pa.Field(...)``. ``None`` when the annotation span is unknown
    (e.g. a string forward-ref annotation) → the caller omits the key."""
    if not isinstance(error, TypeDifference):
        return None, None
    suggested = render_dtype_annotation(error.inferred)
    span = getattr(error, "declared_annotation_span", None)
    declared_range = _span_dict(span) if span is not None else None
    return suggested, declared_range


def _build_diagnostics(group: FileResults) -> list[dict]:
    diagnostics: list[dict] = []
    for result in group.results:
        line = group.function_lines.get(result.function_name, 1)
        end_line = group.function_end_lines.get(result.function_name)
        if not result.passed:
            for error in result.errors:
                message = _error_to_message(error)
                # Per-column / per-expression span (issue #110): a typed
                # column mismatch carries the inferred-side PRIMARY span and
                # the declared-side SECONDARY span. Fall back to the whole-
                # function range for untyped / span-less errors.
                diag_line, diag_col = line, 0
                diag_end_line, diag_end_col = end_line, 0
                related: list[dict] = []
                if isinstance(error, TypeMismatch):
                    primary = getattr(error, "primary", None)
                    if primary is not None:
                        diag_line, diag_col, diag_end_line, diag_end_col = _span_range(primary)
                    declared = getattr(error, "declared_span", None)
                    if declared is not None:
                        related.append(_related_from_declared(declared))
                fields = _structured_fields(error)
                suggested_annotation, declared_annotation_range = _retype_fix_fields(error)
                diag = Diagnostic(
                    file=group.file_path,
                    line=diag_line,
                    column=diag_col,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    end_line=diag_end_line,
                    end_column=diag_end_col,
                    code=extract_code(message),
                    related=related,
                    column_name=fields.get("column_name"),
                    schema=fields.get("schema"),
                    declared_type=fields.get("declared_type"),
                    inferred_type=fields.get("inferred_type"),
                    suggested_annotation=suggested_annotation,
                    declared_annotation_range=declared_annotation_range,
                    fix=getattr(error, "fix", None),
                    param_name=getattr(error, "param_name", None),
                    param_annotation_range=getattr(error, "param_annotation_range", None),
                )
                diagnostics.append(diag.to_dict())
        for warning in result.warnings:
            diag = Diagnostic(
                file=group.file_path,
                line=line,
                column=0,
                message=warning,
                severity=DiagnosticSeverity.WARNING,
                end_line=end_line,
                end_column=0,
                code=extract_code(warning),
            )
            diagnostics.append(diag.to_dict())
    return diagnostics


def format_json(
    results: list[CheckResult],
    file_path: str,
    function_lines: dict[str, int] | None = None,
    function_end_lines: dict[str, int] | None = None,
    functions: list[dict] | None = None,
) -> str:
    """
    Format check results as JSON for a single source file.

    Args:
        results: List of CheckResult objects
        file_path: Path to the source file
        function_lines: Optional mapping of function name to line number
        function_end_lines: Optional mapping of function name to end line number

    Returns:
        JSON string with diagnostics
    """
    group = FileResults(
        file_path=file_path,
        results=results,
        function_lines=function_lines or {},
        function_end_lines=function_end_lines or {},
        functions=functions or [],
    )
    output = {
        "file": file_path,
        "diagnostics": _build_diagnostics(group),
        "functions": _attributed_functions(group),
    }
    return json.dumps(output, indent=2)


def format_json_files(groups: list[FileResults]) -> str:
    """
    Format check results as JSON across multiple source files.

    Each diagnostic carries its own ``file`` field, so callers reading the
    output can attribute errors to the correct source path. Used by the CLI
    when it is invoked with more than one file (e.g. by pre-commit).
    """
    diagnostics: list[dict] = []
    functions: list[dict] = []
    for group in groups:
        diagnostics.extend(_build_diagnostics(group))
        functions.extend(_attributed_functions(group))
    return json.dumps({"diagnostics": diagnostics, "functions": functions}, indent=2)


def _attributed_functions(group: FileResults) -> list[dict]:
    """The group's function summaries with the owning file stamped in."""
    return [{**fn, "file": group.file_path} for fn in group.functions]
