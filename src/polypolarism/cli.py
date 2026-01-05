"""Command-line interface for polypolarism."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from polypolarism.checker import check_source, CheckResult
from polypolarism.analyzer import analyze_source
from polypolarism.output import format_json


__version__ = "0.1.0"


def check_file(file_path: Path) -> list[CheckResult]:
    """
    Check a single Python file for DataFrame type errors.

    Args:
        file_path: Path to the Python file

    Returns:
        List of CheckResult for each function with DF annotations

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = file_path.read_text()
    return check_source(source)


def check_directory(dir_path: Path) -> list[CheckResult]:
    """
    Check all Python files in a directory for DataFrame type errors.

    Args:
        dir_path: Path to the directory

    Returns:
        List of CheckResult for all functions in all files

    Raises:
        FileNotFoundError: If the directory does not exist
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    results: list[CheckResult] = []
    for py_file in dir_path.glob("**/*.py"):
        try:
            file_results = check_file(py_file)
            results.extend(file_results)
        except Exception:
            # Skip files that can't be parsed
            pass

    return results


def format_results(results: list[CheckResult], verbose: bool = False) -> str:
    """
    Format check results for display.

    Args:
        results: List of check results
        verbose: If True, show more details

    Returns:
        Formatted string for output
    """
    if not results:
        return "No functions with DF annotations found.\n"

    lines: list[str] = []
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    for result in results:
        if result.passed:
            status = "\033[32mOK\033[0m"  # Green
        else:
            status = "\033[31mFAIL\033[0m"  # Red

        lines.append(f"  {result.function_name}: {status}")

        if not result.passed:
            for error in result.errors:
                lines.append(f"    - {error}")

    # Summary
    lines.append("")
    if failed_count == 0:
        lines.append(f"\033[32mAll {passed_count} function(s) passed.\033[0m")
    else:
        lines.append(
            f"\033[31m{failed_count} function(s) failed, {passed_count} passed.\033[0m"
        )

    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="polypolarism",
        description="Static type checker for Polars DataFrames",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to check",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return parser


def _check_file_with_locations(file_path: Path) -> tuple[list[CheckResult], dict[str, int]]:
    """
    Check a file and return results with function line numbers.

    Returns:
        Tuple of (results, function_lines mapping)
    """
    source = file_path.read_text()
    analyses = analyze_source(source)
    results = check_source(source)
    function_lines = {a.name: a.lineno for a in analyses}
    return results, function_lines


def main(args: Optional[list[str]] = None) -> int:
    """
    Entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.paths:
        parser.print_help()
        return 0

    all_results: list[CheckResult] = []
    all_function_lines: dict[str, int] = {}
    current_file: Optional[Path] = None

    for path in parsed.paths:
        if not path.exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            return 1

        if path.is_file():
            try:
                results, function_lines = _check_file_with_locations(path)
                all_results.extend(results)
                all_function_lines.update(function_lines)
                current_file = path
            except Exception as e:
                print(f"Error checking {path}: {e}", file=sys.stderr)
                return 1
        elif path.is_dir():
            try:
                results = check_directory(path)
                all_results.extend(results)
            except Exception as e:
                print(f"Error checking {path}: {e}", file=sys.stderr)
                return 1

    # Format output based on --format option
    if parsed.format == "json":
        file_path_str = str(current_file) if current_file else "unknown"
        output = format_json(all_results, file_path_str, all_function_lines)
    else:
        output = format_results(all_results, verbose=parsed.verbose)
        # Strip color codes if requested
        if parsed.no_color:
            import re
            output = re.sub(r"\033\[[0-9;]*m", "", output)

    print(output)

    # Return non-zero if any check failed
    if any(not r.passed for r in all_results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
