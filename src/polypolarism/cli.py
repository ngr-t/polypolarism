"""Command-line interface for polypolarism."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polypolarism.analyzer import analyze_source
from polypolarism.checker import (
    CheckError,
    CheckResult,
    ExtraColumn,
    MissingColumn,
    TypeDifference,
    check_source,
)
from polypolarism.output import FileResults, format_json, format_json_files

__version__ = "0.1.0"


def _format_schema_diff(errors: list[CheckError]) -> list[str]:
    """Build an aligned per-column diff block from an error list.

    Returns ``[]`` when fewer than two schema mismatches are present — for
    a single error the per-line message is already the clearest output.
    Mismatches that aren't column-level (e.g. ``InferenceFailure``, raw
    strings) are skipped.
    """
    rows: list[tuple[str, str, str, str]] = []
    for err in errors:
        if isinstance(err, TypeDifference):
            rows.append((err.column, str(err.declared), str(err.inferred), "mismatch"))
        elif isinstance(err, MissingColumn):
            rows.append((err.column, str(err.expected_type), "(missing)", "missing"))
        elif isinstance(err, ExtraColumn):
            rows.append((err.column, "(absent)", str(err.inferred_type), "extra"))

    if len(rows) < 2:
        return []

    headers = ("column", "declared", "inferred", "status")
    col_w = max(len(headers[0]), *(len(r[0]) for r in rows))
    decl_w = max(len(headers[1]), *(len(r[1]) for r in rows))
    inf_w = max(len(headers[2]), *(len(r[2]) for r in rows))
    sep = "─"

    lines: list[str] = ["    schema diff:"]
    lines.append(
        f"      {headers[0]:<{col_w}}  {headers[1]:<{decl_w}}  "
        f"{headers[2]:<{inf_w}}  {headers[3]}"
    )
    lines.append(f"      {sep * col_w}  {sep * decl_w}  {sep * inf_w}  {sep * 8}")
    for col, declared, inferred, status in rows:
        lines.append(
            f"      {col:<{col_w}}  {declared:<{decl_w}}  "
            f"{inferred:<{inf_w}}  {status}"
        )
    return lines


def _parse_error_result(file_path: Path, err: Exception) -> CheckResult:
    """Build a CheckResult representing a file-level read or parse failure."""
    return CheckResult(
        function_name=f"<{file_path}>",
        passed=False,
        errors=[f"{type(err).__name__}: {err}"],
    )


def check_file(file_path: Path) -> list[CheckResult]:
    """
    Check a single Python file for DataFrame type errors.

    On read or parse failures, returns a single failing CheckResult instead of
    raising or silently passing — so external callers (pre-commit, CI) cannot
    miss a broken file.

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        source = file_path.read_text()
    except (UnicodeDecodeError, OSError) as err:
        return [_parse_error_result(file_path, err)]
    try:
        return check_source(source)
    except SyntaxError as err:
        return [_parse_error_result(file_path, err)]


def check_directory(dir_path: Path) -> list[CheckResult]:
    """
    Check all Python files in a directory for DataFrame type errors.

    Files that fail to read or parse contribute a synthetic failing
    CheckResult rather than being silently skipped.

    Raises:
        FileNotFoundError: If the directory does not exist
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    results: list[CheckResult] = []
    for py_file in sorted(dir_path.glob("**/*.py")):
        results.extend(check_file(py_file))
    return results


def format_results(
    results: list[CheckResult],
    verbose: bool = False,
    function_lines: dict[str, int] | None = None,
) -> str:
    """Format check results for human-readable display."""
    if function_lines is None:
        function_lines = {}

    if not results:
        return "No functions with DataFrame[Schema] annotations found.\n"

    lines: list[str] = []
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    warn_count = sum(1 for r in results if r.warnings)

    for result in results:
        if result.passed and result.warnings:
            status = "\033[33mWARN\033[0m"
        elif result.passed:
            status = "\033[32mOK\033[0m"
        else:
            status = "\033[31mFAIL\033[0m"

        lineno = function_lines.get(result.function_name)
        if lineno is not None:
            lines.append(f"  {result.function_name} (line {lineno}): {status}")
        else:
            lines.append(f"  {result.function_name}: {status}")

        if not result.passed:
            for error in result.errors:
                lines.append(f"    - {error}")
            diff_block = _format_schema_diff(result.errors)
            if diff_block:
                lines.extend(diff_block)
        for warning in result.warnings:
            lines.append(f"    \033[33m! {warning}\033[0m")

    lines.append("")
    if failed_count == 0:
        summary = f"\033[32mAll {passed_count} function(s) passed.\033[0m"
        if warn_count:
            summary += f" \033[33m({warn_count} with warnings)\033[0m"
        lines.append(summary)
    else:
        summary = (
            f"\033[31m{failed_count} function(s) failed, {passed_count} passed.\033[0m"
        )
        if warn_count:
            summary += f" \033[33m({warn_count} with warnings)\033[0m"
        lines.append(summary)

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


def _check_file_with_locations(
    file_path: Path,
) -> tuple[list[CheckResult], dict[str, int], dict[str, int]]:
    """
    Check a file and return results plus per-function source line numbers.

    On read or parse failures, returns a single failing CheckResult and empty
    line-number maps. Never raises for those cases.
    """
    try:
        source = file_path.read_text()
    except (UnicodeDecodeError, OSError) as err:
        return [_parse_error_result(file_path, err)], {}, {}
    try:
        analyses = analyze_source(source)
        results = check_source(source)
    except SyntaxError as err:
        return [_parse_error_result(file_path, err)], {}, {}
    function_lines = {a.name: a.lineno for a in analyses}
    function_end_lines = {a.name: a.end_lineno for a in analyses}
    return results, function_lines, function_end_lines


def _expand_directory_groups(dir_path: Path) -> list[FileResults]:
    """Per-file FileResults for every .py under dir_path."""
    groups: list[FileResults] = []
    for py_file in sorted(dir_path.glob("**/*.py")):
        results, function_lines, function_end_lines = _check_file_with_locations(py_file)
        groups.append(
            FileResults(
                file_path=str(py_file),
                results=results,
                function_lines=function_lines,
                function_end_lines=function_end_lines,
            )
        )
    return groups


def main(args: list[str] | None = None) -> int:
    """Entry point for the CLI. Returns 0 on success, 1 on any failure."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.paths:
        parser.print_help()
        return 0

    file_groups: list[FileResults] = []

    for path in parsed.paths:
        if not path.exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            return 1

        if path.is_file():
            results, function_lines, function_end_lines = _check_file_with_locations(path)
            file_groups.append(
                FileResults(
                    file_path=str(path),
                    results=results,
                    function_lines=function_lines,
                    function_end_lines=function_end_lines,
                )
            )
        elif path.is_dir():
            file_groups.extend(_expand_directory_groups(path))

    all_results: list[CheckResult] = []
    all_function_lines: dict[str, int] = {}
    for group in file_groups:
        all_results.extend(group.results)
        all_function_lines.update(group.function_lines)

    if parsed.format == "json":
        if len(file_groups) == 1:
            single = file_groups[0]
            output = format_json(
                single.results,
                single.file_path,
                single.function_lines,
                single.function_end_lines,
            )
        else:
            output = format_json_files(file_groups)
    else:
        output = format_results(
            all_results, verbose=parsed.verbose, function_lines=all_function_lines
        )
        if parsed.no_color:
            import re

            output = re.sub(r"\033\[[0-9;]*m", "", output)

    print(output)

    if any(not r.passed for r in all_results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
