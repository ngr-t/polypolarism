"""Output formatting for diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from polypolarism.checker import CheckResult


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


def _build_diagnostics(group: FileResults) -> list[dict]:
    diagnostics: list[dict] = []
    for result in group.results:
        if result.passed:
            continue
        line = group.function_lines.get(result.function_name, 1)
        end_line = group.function_end_lines.get(result.function_name)
        for error in result.errors:
            diag = Diagnostic(
                file=group.file_path,
                line=line,
                column=0,
                message=_error_to_message(error),
                severity=DiagnosticSeverity.ERROR,
                end_line=end_line,
                end_column=0,
            )
            diagnostics.append(diag.to_dict())
    return diagnostics


def format_json(
    results: list[CheckResult],
    file_path: str,
    function_lines: dict[str, int] | None = None,
    function_end_lines: dict[str, int] | None = None,
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
    )
    output = {
        "file": file_path,
        "diagnostics": _build_diagnostics(group),
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
    for group in groups:
        diagnostics.extend(_build_diagnostics(group))
    return json.dumps({"diagnostics": diagnostics}, indent=2)
