"""Tests for output formatting."""

import json
import textwrap

from polypolarism.checker import (
    CheckResult,
    ExtraColumn,
    InferenceFailure,
    MissingColumn,
)
from polypolarism.output import Diagnostic, DiagnosticSeverity, format_json
from polypolarism.types import Int64, Utf8


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

    def test_error_diagnostic_carries_structured_code(self):
        """Tagged errors expose their diagnostic code as a structured field
        (issue #70), not only as a message prefix."""
        result = CheckResult(
            function_name="process",
            passed=False,
            errors=[
                MissingColumn("name", Utf8()),
                "[PLY032] Return type expected DataFrame[...] but inferred LazyFrame[...]",
            ],
        )

        output = format_json([result], "test.py", function_lines={"process": 10})
        data = json.loads(output)

        codes = [d["code"] for d in data["diagnostics"]]
        assert codes == ["PLY040", "PLY032"]
        # The message prefix stays for backward compatibility.
        assert data["diagnostics"][0]["message"].startswith("[PLY040] ")

    def test_warning_diagnostic_carries_structured_code(self):
        """Tagged warnings expose their PLW code structurally too."""
        result = CheckResult(
            function_name="process",
            passed=True,
            warnings=["[PLW001] map_elements without return_dtype="],
        )

        output = format_json([result], "test.py", function_lines={"process": 3})
        data = json.loads(output)

        assert len(data["diagnostics"]) == 1
        assert data["diagnostics"][0]["code"] == "PLW001"
        assert data["diagnostics"][0]["severity"] == "warning"

    def test_untagged_diagnostic_has_no_code_field(self):
        """Untagged errors (e.g. parse failures) omit the code key entirely,
        keeping the schema additive."""
        result = CheckResult(
            function_name="<broken.py>",
            passed=False,
            errors=["SyntaxError: invalid syntax"],
        )

        output = format_json([result], "broken.py")
        data = json.loads(output)

        assert len(data["diagnostics"]) == 1
        assert "code" not in data["diagnostics"][0]

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

        source = textwrap.dedent("""
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class InSchema(pa.DataFrameModel):
                id: int

            class OutSchema(pa.DataFrameModel):
                id: int
                missing: str

            def wrong_return(
                data: DataFrame[InSchema],
            ) -> DataFrame[OutSchema]:
                return data
        """)

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


class TestJsonFunctionsArray:
    """D-11 hover support: ``--format json`` exposes per-function schema
    summaries (params, declared/inferred returns, openness) so editor
    integrations can render hovers without re-analyzing."""

    def _functions(self, source: str):
        import json
        import textwrap

        from polypolarism.analyzer import analyze_source
        from polypolarism.checker import check_function
        from polypolarism.output import FileResults, format_json_files, function_summaries

        analyses = analyze_source(textwrap.dedent(source))
        results = [check_function(a) for a in analyses]
        group = FileResults(
            file_path="x.py",
            results=results,
            function_lines={a.name: a.lineno for a in analyses},
            function_end_lines={a.name: a.end_lineno for a in analyses},
            functions=function_summaries(analyses),
        )
        payload = json.loads(format_json_files([group]))
        return payload["functions"]

    def test_summary_carries_schemas_and_span(self):
        functions = self._functions(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Src(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True

            class Out(pa.DataFrameModel):
                doubled: int

                class Config:
                    strict = True

            def f(df: DataFrame[Src]) -> DataFrame[Out]:
                return df.select(doubled=pl.col("a") * 2)
            """
        )
        assert len(functions) == 1
        fn = functions[0]
        assert fn["name"] == "f"
        assert fn["file"] == "x.py"
        assert fn["line"] >= 1 and fn["end_line"] >= fn["line"]
        assert fn["params"]["df"]["columns"] == {"a": "Int64"}
        assert fn["declared_return"]["columns"] == {"doubled": "Int64"}
        assert fn["inferred_return"]["columns"] == {"doubled": "Int64"}
        assert fn["inferred_return"]["open"] is False

    def test_open_frame_and_missing_sides_render(self):
        functions = self._functions(
            """
            import polars as pl

            def g(df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(x=pl.lit(1))
            """
        )
        fn = functions[0]
        assert fn["declared_return"] is None
        assert fn["inferred_return"]["open"] is True
        assert fn["inferred_return"]["columns"]["x"] == "Int64"
        assert fn["params"]["df"]["open"] is True
