"""Tests for CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from polypolarism.cli import check_directory, check_file, format_results, main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCheckFile:
    """Test file checking functionality."""

    def test_check_valid_file(self):
        """Check a valid Python file."""
        valid_file = FIXTURES_DIR / "valid" / "basic_join.py"
        results = check_file(valid_file)

        assert len(results) == 1
        assert results[0].passed is True

    def test_check_invalid_file(self):
        """Check an invalid Python file."""
        invalid_file = FIXTURES_DIR / "invalid" / "join_missing_column.py"
        results = check_file(invalid_file)

        assert len(results) == 1
        assert results[0].passed is False

    def test_check_file_not_found(self):
        """Handle non-existent file."""
        with pytest.raises(FileNotFoundError):
            check_file(Path("/nonexistent/file.py"))

    def test_check_file_with_multiple_functions(self):
        """Check file with multiple annotated functions."""
        # The chained_operations.py has one function
        file_path = FIXTURES_DIR / "valid" / "chained_operations.py"
        results = check_file(file_path)

        assert len(results) >= 1


class TestCheckDirectory:
    """Test directory checking functionality."""

    def test_check_valid_directory(self):
        """Check all files in valid directory."""
        valid_dir = FIXTURES_DIR / "valid"
        results = check_directory(valid_dir)

        # Should have results for all 5 valid test files
        assert len(results) >= 5
        # All should pass
        assert all(r.passed for r in results)

    def test_check_invalid_directory(self):
        """Check all files in invalid directory."""
        invalid_dir = FIXTURES_DIR / "invalid"
        results = check_directory(invalid_dir)

        # Should have results for invalid test files
        assert len(results) >= 5
        # At least some should fail (helper functions may pass)
        assert any(not r.passed for r in results)

    def test_check_empty_directory(self, tmp_path):
        """Check an empty directory."""
        results = check_directory(tmp_path)
        assert results == []

    def test_filter_nonbool_fixture_fails_with_ply008(self):
        """Issue #28 fixture: non-boolean filter predicate."""
        results = check_file(FIXTURES_DIR / "invalid" / "filter_nonbool_predicate.py")

        assert len(results) == 1
        assert results[0].passed is False
        assert any("PLY008" in str(e) for e in results[0].errors)

    def test_sort_missing_column_fixture_fails_with_ply007(self):
        """Issue #29 fixture: sort key column doesn't exist."""
        results = check_file(FIXTURES_DIR / "invalid" / "sort_missing_column.py")

        assert len(results) == 1
        assert results[0].passed is False
        assert any("PLY007" in str(e) for e in results[0].errors)

    def test_namespace_wrong_dtype_fixture_fails_with_ply012(self):
        """Issue #31 fixture: namespace accessors on wrong-dtype columns."""
        results = check_file(FIXTURES_DIR / "invalid" / "namespace_wrong_dtype.py")

        assert len(results) == 3
        assert all(r.passed is False for r in results)
        assert all(any("PLY012" in str(e) for e in r.errors) for r in results)

    def test_over_missing_column_fixture_fails_with_ply001(self):
        """Issue #32 fixture: over() partition column doesn't exist."""
        results = check_file(FIXTURES_DIR / "invalid" / "over_missing_column.py")

        assert len(results) == 1
        assert results[0].passed is False
        assert any("PLY001" in str(e) and "ghost" in str(e) for e in results[0].errors)

    def test_compare_incompatible_fixture_fails_with_ply009(self):
        """Issue #33 fixture: String == Int64 comparison and Int64.is_in(list-of-str)."""
        results = check_file(FIXTURES_DIR / "invalid" / "compare_incompatible.py")

        assert len(results) == 2
        for result in results:
            assert result.passed is False
            assert any("PLY009" in str(e) for e in result.errors)

    def test_cast_impossible_fixture_fails_with_ply013(self):
        """Issue #34 fixture: List(Int64) -> Int64 cast is structurally impossible."""
        results = check_file(FIXTURES_DIR / "invalid" / "cast_impossible.py")

        assert len(results) == 1
        assert results[0].passed is False
        assert any("PLY013" in str(e) for e in results[0].errors)

    def test_duplicate_output_fixture_fails_with_ply015(self):
        """Issue #36 fixture: two select outputs share the name 'a'."""
        results = check_file(FIXTURES_DIR / "invalid" / "duplicate_output.py")

        assert len(results) == 1
        assert results[0].passed is False
        assert any("PLY015" in str(e) for e in results[0].errors)


class TestCheckWarningFixtures:
    """Files in fixtures/warning produce warnings but still pass type-check."""

    def test_warning_fixtures_pass_with_warnings(self):
        warning_dir = FIXTURES_DIR / "warning"
        results = check_directory(warning_dir)
        assert len(results) >= 3
        # Every function passes (warnings don't fail)
        assert all(r.passed for r in results)
        # Every function carries at least one warning
        assert all(r.warnings for r in results)
        # PLW codes show up
        all_warnings = [w for r in results for w in r.warnings]
        assert any("PLW001" in w for w in all_warnings)
        assert any("PLW003" in w for w in all_warnings)
        assert any("PLW004" in w for w in all_warnings)

    def test_cli_exits_zero_with_only_warnings(self, tmp_path):
        warning_dir = FIXTURES_DIR / "warning"
        rc = main([str(warning_dir)])
        # passed=True for every function so the CLI returns 0.
        assert rc == 0

    def test_check_directory_not_found(self):
        """Handle non-existent directory."""
        with pytest.raises(FileNotFoundError):
            check_directory(Path("/nonexistent/directory"))


class TestFormatResults:
    """Test result formatting."""

    def test_format_passed_result(self):
        """Format a passed check result."""
        from polypolarism.checker import CheckResult

        result = CheckResult(function_name="my_func", passed=True, errors=[])
        output = format_results([result])

        assert "my_func" in output
        assert "pass" in output.lower() or "ok" in output.lower()

    def test_format_failed_result_with_errors(self):
        """Format a failed check result with errors."""
        from polypolarism.checker import CheckError, CheckResult, MissingColumn
        from polypolarism.types import Utf8

        errors: list[CheckError] = [MissingColumn("name", Utf8())]
        result = CheckResult(function_name="bad_func", passed=False, errors=errors)
        output = format_results([result])

        assert "bad_func" in output
        assert "name" in output

    def test_format_multiple_results(self):
        """Format multiple results."""
        from polypolarism.checker import CheckResult

        results = [
            CheckResult(function_name="func1", passed=True, errors=[]),
            CheckResult(function_name="func2", passed=False, errors=["error"]),
        ]
        output = format_results(results)

        assert "func1" in output
        assert "func2" in output


class TestSchemaDiffBlock:
    """When 2+ schema mismatches occur, the formatter appends a diff block."""

    def test_single_mismatch_has_no_diff_block(self):
        from polypolarism.checker import CheckError, CheckResult, TypeDifference
        from polypolarism.types import Float64, Int64

        errors: list[CheckError] = [TypeDifference("a", Int64(), Float64())]
        result = CheckResult(function_name="f", passed=False, errors=errors)
        output = format_results([result])
        # Per-line error still shown
        assert "Int64" in output and "Float64" in output
        # No diff block heading
        assert "schema diff" not in output

    def test_multiple_mismatches_render_diff_block(self):
        from polypolarism.checker import (
            CheckError,
            CheckResult,
            ExtraColumn,
            MissingColumn,
            TypeDifference,
        )
        from polypolarism.types import Float64, Int64, Utf8

        errors: list[CheckError] = [
            TypeDifference("a", Int64(), Float64()),
            TypeDifference("b", Float64(), Utf8()),
            MissingColumn("c", Int64()),
            ExtraColumn("d", Utf8()),
        ]
        result = CheckResult(function_name="f", passed=False, errors=errors)
        output = format_results([result])
        # Per-line errors still present
        assert "Column 'a'" in output or "a" in output
        # Diff block appears
        assert "schema diff" in output
        # Each affected column shows up in the block
        for col in ("a", "b", "c", "d"):
            assert col in output
        # Status keywords are present
        assert "mismatch" in output
        assert "missing" in output
        assert "extra" in output


class TestMainExitCode:
    """Test CLI exit codes."""

    def test_main_with_valid_file_exits_zero(self, tmp_path, monkeypatch):
        """CLI exits with 0 when all checks pass."""
        # Create a valid test file
        test_file = tmp_path / "valid.py"
        test_file.write_text("""
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class IdSchema(pa.DataFrameModel):
    id: int

def identity(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    return data
""")
        monkeypatch.setattr(sys, "argv", ["polypolarism", str(test_file)])

        exit_code = main()
        assert exit_code == 0

    def test_main_with_invalid_file_exits_nonzero(self, tmp_path, monkeypatch):
        """CLI exits with non-zero when checks fail."""
        # Create an invalid test file
        test_file = tmp_path / "invalid.py"
        test_file.write_text("""
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class InSchema(pa.DataFrameModel):
    id: int

class OutSchema(pa.DataFrameModel):
    id: int
    extra: str

def bad(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return data
""")
        monkeypatch.setattr(sys, "argv", ["polypolarism", str(test_file)])

        exit_code = main()
        assert exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI using subprocess."""

    def test_cli_help(self):
        """CLI shows help message."""
        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", "--help"],
            capture_output=True,
            text=True,
        )
        # argparse returns 0 for --help
        assert result.returncode == 0
        assert "polypolarism" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_version(self):
        """CLI shows version."""
        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_check_valid_file(self):
        """CLI checks a valid file successfully."""
        valid_file = FIXTURES_DIR / "valid" / "basic_join.py"
        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", str(valid_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_check_invalid_file(self):
        """CLI fails for invalid file."""
        invalid_file = FIXTURES_DIR / "invalid" / "join_missing_column.py"
        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", str(invalid_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_check_directory(self):
        """CLI checks all files in a directory."""
        valid_dir = FIXTURES_DIR / "valid"
        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", str(valid_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestFormatOption:
    """Test --format option for different output formats."""

    def test_format_json_outputs_valid_json(self, tmp_path):
        """--format json outputs valid JSON."""
        import json

        test_file = tmp_path / "test.py"
        test_file.write_text("""
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class IdSchema(pa.DataFrameModel):
    id: int

def identity(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    return data
""")

        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", "--format", "json", str(test_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Output should be valid JSON
        data = json.loads(result.stdout)
        assert "diagnostics" in data

    def test_format_json_with_errors(self, tmp_path):
        """--format json includes errors in output."""
        import json

        test_file = tmp_path / "test.py"
        test_file.write_text("""
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class InSchema(pa.DataFrameModel):
    id: int

class OutSchema(pa.DataFrameModel):
    id: int
    missing: str

def bad(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return data
""")

        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", "--format", "json", str(test_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0  # Errors should cause non-zero exit
        data = json.loads(result.stdout)
        assert len(data["diagnostics"]) >= 1
        assert any("missing" in d["message"].lower() for d in data["diagnostics"])

    def test_format_text_is_default(self, tmp_path):
        """--format text is the default format."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class IdSchema(pa.DataFrameModel):
    id: int

def identity(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    return data
""")

        result_default = subprocess.run(
            [sys.executable, "-m", "polypolarism", str(test_file)],
            capture_output=True,
            text=True,
        )

        result_text = subprocess.run(
            [sys.executable, "-m", "polypolarism", "--format", "text", str(test_file)],
            capture_output=True,
            text=True,
        )

        # Both should exit with 0
        assert result_default.returncode == 0
        assert result_text.returncode == 0
        # Both should produce similar output (not JSON)
        assert "identity" in result_default.stdout
        assert "identity" in result_text.stdout


# ---------------------------------------------------------------------------
# Pre-commit-style invocation: callers pass an explicit list of file paths.
# These tests lock the CLI's behaviour for that shape.
# ---------------------------------------------------------------------------

VALID_SOURCE = """
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class IdSchema(pa.DataFrameModel):
    id: int

def identity(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    return data
"""


INVALID_SOURCE = """
import pandera.polars as pa
from pandera.typing.polars import DataFrame

class InSchema(pa.DataFrameModel):
    id: int

class OutSchema(pa.DataFrameModel):
    id: int
    missing: str

def bad(data: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return data
"""


SYNTAX_ERROR_SOURCE = "def broken(:\n"


class TestSyntaxErrorHandling:
    """Files that fail to parse must be reported as failures, not silently skipped."""

    def test_check_file_returns_failure_for_syntax_error(self, tmp_path):
        broken = tmp_path / "broken.py"
        broken.write_text(SYNTAX_ERROR_SOURCE)

        results = check_file(broken)

        assert len(results) == 1
        assert results[0].passed is False
        assert any("SyntaxError" in str(e) for e in results[0].errors)

    def test_check_directory_includes_syntax_error_failures(self, tmp_path):
        (tmp_path / "good.py").write_text(VALID_SOURCE)
        (tmp_path / "broken.py").write_text(SYNTAX_ERROR_SOURCE)

        results = check_directory(tmp_path)

        assert any(not r.passed and "SyntaxError" in str(r.errors) for r in results), (
            "directory scan must surface parse failures, not silently swallow them"
        )

    def test_main_exit_code_nonzero_when_syntax_error_in_directory(self, tmp_path, monkeypatch):
        (tmp_path / "good.py").write_text(VALID_SOURCE)
        (tmp_path / "broken.py").write_text(SYNTAX_ERROR_SOURCE)
        monkeypatch.setattr(sys, "argv", ["polypolarism", str(tmp_path)])

        assert main() != 0


class TestMultiFileInvocation:
    """pre-commit invokes the CLI as `polypolarism a.py b.py c.py ...`."""

    def test_main_accepts_multiple_files(self, tmp_path, monkeypatch):
        valid = tmp_path / "valid.py"
        valid.write_text(VALID_SOURCE)
        invalid = tmp_path / "invalid.py"
        invalid.write_text(INVALID_SOURCE)

        monkeypatch.setattr(sys, "argv", ["polypolarism", str(valid), str(invalid)])

        assert main() != 0

    def test_main_all_valid_multi_file_exits_zero(self, tmp_path, monkeypatch):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text(VALID_SOURCE)
        b.write_text(VALID_SOURCE)

        monkeypatch.setattr(sys, "argv", ["polypolarism", str(a), str(b)])

        assert main() == 0

    def test_cli_multi_file_text_output_lists_each_function(self, tmp_path):
        valid = tmp_path / "valid.py"
        valid.write_text(VALID_SOURCE)
        invalid = tmp_path / "invalid.py"
        invalid.write_text(INVALID_SOURCE)

        result = subprocess.run(
            [sys.executable, "-m", "polypolarism", str(valid), str(invalid)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "identity" in result.stdout
        assert "bad" in result.stdout

    def test_cli_multi_file_json_attributes_diagnostics_per_file(self, tmp_path):
        import json

        good = tmp_path / "good.py"
        good.write_text(VALID_SOURCE)
        bad_a = tmp_path / "bad_a.py"
        bad_a.write_text(INVALID_SOURCE)
        bad_b = tmp_path / "bad_b.py"
        bad_b.write_text(INVALID_SOURCE)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "polypolarism",
                "--format",
                "json",
                str(good),
                str(bad_a),
                str(bad_b),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "diagnostics" in data

        files_in_output = {d["file"] for d in data["diagnostics"]}
        # Every diagnostic should know which file produced it. The bad files
        # should both appear; the good file should not generate diagnostics.
        assert str(bad_a) in files_in_output
        assert str(bad_b) in files_in_output
        assert str(good) not in files_in_output

    def test_cli_multi_file_json_includes_syntax_errors(self, tmp_path):
        import json

        valid = tmp_path / "valid.py"
        valid.write_text(VALID_SOURCE)
        broken = tmp_path / "broken.py"
        broken.write_text(SYNTAX_ERROR_SOURCE)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "polypolarism",
                "--format",
                "json",
                str(valid),
                str(broken),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert any(
            d.get("file") == str(broken) and "SyntaxError" in d["message"]
            for d in data["diagnostics"]
        )
