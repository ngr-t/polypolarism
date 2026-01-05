"""Tests for CLI."""

import pytest
import subprocess
import sys
from pathlib import Path

from polypolarism.cli import main, check_file, check_directory, format_results


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

        # Should have results for all 5 invalid test files
        assert len(results) >= 5
        # All should fail
        assert all(not r.passed for r in results)

    def test_check_empty_directory(self, tmp_path):
        """Check an empty directory."""
        results = check_directory(tmp_path)
        assert results == []

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
        from polypolarism.checker import CheckResult, MissingColumn
        from polypolarism.types import Utf8

        errors = [MissingColumn("name", Utf8())]
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


class TestMainExitCode:
    """Test CLI exit codes."""

    def test_main_with_valid_file_exits_zero(self, tmp_path, monkeypatch):
        """CLI exits with 0 when all checks pass."""
        # Create a valid test file
        test_file = tmp_path / "valid.py"
        test_file.write_text('''
from polypolarism import DF

def identity(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
    return data
''')
        monkeypatch.setattr(sys, "argv", ["polypolarism", str(test_file)])

        exit_code = main()
        assert exit_code == 0

    def test_main_with_invalid_file_exits_nonzero(self, tmp_path, monkeypatch):
        """CLI exits with non-zero when checks fail."""
        # Create an invalid test file
        test_file = tmp_path / "invalid.py"
        test_file.write_text('''
from polypolarism import DF

def bad(data: DF["{id: Int64}"]) -> DF["{id: Int64, extra: Utf8}"]:
    return data
''')
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
