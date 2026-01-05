"""Output formatting for diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from polypolarism.checker import CheckResult, TypeMismatch, InferenceFailure


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
    end_line: Optional[int] = None
    end_column: Optional[int] = None
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


def format_json(
    results: list[CheckResult],
    file_path: str,
    function_lines: Optional[dict[str, int]] = None,
) -> str:
    """
    Format check results as JSON.

    Args:
        results: List of CheckResult objects
        file_path: Path to the source file
        function_lines: Optional mapping of function name to line number

    Returns:
        JSON string with diagnostics
    """
    if function_lines is None:
        function_lines = {}

    diagnostics: list[dict] = []

    for result in results:
        if result.passed:
            continue

        line = function_lines.get(result.function_name, 1)

        for error in result.errors:
            diag = Diagnostic(
                file=file_path,
                line=line,
                column=0,
                message=_error_to_message(error),
                severity=DiagnosticSeverity.ERROR,
            )
            diagnostics.append(diag.to_dict())

    output = {
        "file": file_path,
        "diagnostics": diagnostics,
    }

    return json.dumps(output, indent=2)
