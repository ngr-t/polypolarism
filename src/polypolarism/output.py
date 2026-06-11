"""Output formatting for diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from polypolarism.checker import CheckResult
from polypolarism.diagnostics import extract_code


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


def function_summaries(analyses) -> list[dict]:
    """Per-function schema summaries for editor hovers (D-11).

    One entry per analyzed function: source span, parameter frames, and
    the declared / inferred return frames, with dtypes rendered through
    their canonical ``str`` forms.
    """
    return [
        {
            "name": analysis.name,
            "line": analysis.lineno,
            "end_line": analysis.end_lineno,
            "params": {name: _frame_summary(ft) for name, ft in analysis.input_types.items()},
            "declared_return": _frame_summary(analysis.declared_return_type),
            "inferred_return": _frame_summary(analysis.inferred_return_type),
        }
        for analysis in analyses
    ]


def _build_diagnostics(group: FileResults) -> list[dict]:
    diagnostics: list[dict] = []
    for result in group.results:
        line = group.function_lines.get(result.function_name, 1)
        end_line = group.function_end_lines.get(result.function_name)
        if not result.passed:
            for error in result.errors:
                message = _error_to_message(error)
                diag = Diagnostic(
                    file=group.file_path,
                    line=line,
                    column=0,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    end_line=end_line,
                    end_column=0,
                    code=extract_code(message),
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
