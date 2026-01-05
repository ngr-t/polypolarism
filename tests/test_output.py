"""Tests for output formatting."""

import json
import textwrap

import pytest

from polypolarism.types import FrameType, Int64, Utf8
from polypolarism.analyzer import FunctionAnalysis
from polypolarism.checker import (
    CheckResult,
    MissingColumn,
    ExtraColumn,
    TypeDifference,
    InferenceFailure,
)
from polypolarism.output import format_json, Diagnostic, DiagnosticSeverity


class TestDiagnostic:
    """Test Diagnostic data class."""

    def test_diagnostic_has_required_fields(self):
        """Diagnostic should have file, line, message, severity."""
        diag = Diagnostic(
            file="test.py",
            line=10,
            column=0,
            message="Missing column 'name'",
            severity=DiagnosticSeverity.ERROR,
        )

        assert diag.file == "test.py"
        assert diag.line == 10
        assert diag.column == 0
        assert diag.message == "Missing column 'name'"
        assert diag.severity == DiagnosticSeverity.ERROR

    def test_diagnostic_to_dict(self):
        """Diagnostic should be convertible to dict for JSON serialization."""
        diag = Diagnostic(
            file="test.py",
            line=10,
            column=0,
            message="Test error",
            severity=DiagnosticSeverity.ERROR,
        )

        d = diag.to_dict()

        assert d["file"] == "test.py"
        assert d["line"] == 10
        assert d["column"] == 0
        assert d["message"] == "Test error"
        assert d["severity"] == "error"


class TestFormatJson:
    """Test JSON output formatting."""

    def test_empty_results_returns_empty_list(self):
        """No results produces empty diagnostics list."""
        output = format_json([], "test.py")
        data = json.loads(output)

        assert data["diagnostics"] == []
        assert data["file"] == "test.py"

    def test_passed_function_has_no_diagnostics(self):
        """Passed function produces no diagnostics."""
        result = CheckResult(
            function_name="process",
            passed=True,
            errors=[],
        )

        output = format_json([result], "test.py")
        data = json.loads(output)

        assert data["diagnostics"] == []

    def test_missing_column_produces_diagnostic(self):
        """MissingColumn error produces diagnostic with location."""
        result = CheckResult(
            function_name="process",
            passed=False,
            errors=[MissingColumn("name", Utf8())],
        )

        output = format_json([result], "test.py", function_lines={"process": 10})
        data = json.loads(output)

        assert len(data["diagnostics"]) == 1
        diag = data["diagnostics"][0]
        assert diag["line"] == 10
        assert "name" in diag["message"]
        assert diag["severity"] == "error"

    def test_multiple_errors_produce_multiple_diagnostics(self):
        """Multiple errors produce multiple diagnostics."""
        result = CheckResult(
            function_name="process",
            passed=False,
            errors=[
                MissingColumn("name", Utf8()),
                ExtraColumn("extra", Int64()),
            ],
        )

        output = format_json([result], "test.py", function_lines={"process": 5})
        data = json.loads(output)

        assert len(data["diagnostics"]) == 2

    def test_analysis_error_string_produces_diagnostic(self):
        """Analysis error string produces diagnostic."""
        result = CheckResult(
            function_name="bad_func",
            passed=False,
            errors=["Column 'missing' not found"],
        )

        output = format_json([result], "test.py", function_lines={"bad_func": 15})
        data = json.loads(output)

        assert len(data["diagnostics"]) == 1
        assert "missing" in data["diagnostics"][0]["message"]

    def test_inference_failure_produces_diagnostic(self):
        """InferenceFailure produces diagnostic."""
        result = CheckResult(
            function_name="no_return",
            passed=False,
            errors=[InferenceFailure("Could not infer return type")],
        )

        output = format_json([result], "test.py", function_lines={"no_return": 20})
        data = json.loads(output)

        assert len(data["diagnostics"]) == 1
        assert "infer" in data["diagnostics"][0]["message"].lower()

    def test_json_output_is_valid_json(self):
        """Output should be valid JSON."""
        result = CheckResult(
            function_name="test",
            passed=False,
            errors=[MissingColumn("col", Int64())],
        )

        output = format_json([result], "test.py", function_lines={"test": 1})

        # Should not raise
        parsed = json.loads(output)
        assert isinstance(parsed, dict)


class TestFormatJsonIntegration:
    """Integration tests for JSON output with real analysis."""

    def test_format_json_with_check_results(self):
        """format_json works with actual CheckResult objects."""
        from polypolarism.checker import check_source

        source = textwrap.dedent('''
            from polypolarism import DF

            def wrong_return(
                data: DF["{id: Int64}"],
            ) -> DF["{id: Int64, missing: Utf8}"]:
                return data
        ''')

        results = check_source(source)
        # Get line numbers from analysis
        from polypolarism.analyzer import analyze_source
        analyses = analyze_source(source)
        function_lines = {a.name: a.lineno for a in analyses}

        output = format_json(results, "test.py", function_lines=function_lines)
        data = json.loads(output)

        assert len(data["diagnostics"]) >= 1
        # Should have error about missing column
        assert any("missing" in d["message"].lower() for d in data["diagnostics"])
