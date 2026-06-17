"""Command-line interface for polypolarism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polypolarism.analyzer import analyze_source
from polypolarism.checker import (
    CheckError,
    CheckResult,
    ExtraColumn,
    MissingColumn,
    TypeDifference,
    check_function,
    check_source,
)
from polypolarism.output import (
    FileResults,
    format_json,
    format_json_files,
    function_summaries,
)
from polypolarism.version_check import (
    check_versions,
    detect_versions,
    find_project_root,
)

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
        f"      {headers[0]:<{col_w}}  {headers[1]:<{decl_w}}  {headers[2]:<{inf_w}}  {headers[3]}"
    )
    lines.append(f"      {sep * col_w}  {sep * decl_w}  {sep * inf_w}  {sep * 8}")
    for col, declared, inferred, status in rows:
        lines.append(f"      {col:<{col_w}}  {declared:<{decl_w}}  {inferred:<{inf_w}}  {status}")
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
        return check_source(source, file_path=file_path)
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


def _format_mismatch_context(
    mismatch_frames: list[tuple[int | None, str, str]],
) -> list[str]:
    """Render the inferred-vs-declared frame block shown in verbose mode.

    Each failing return point is listed as ``returned (line N): <render>``
    (or just ``returned: <render>`` for single-return functions), followed
    by ``declared: <render>`` once — the declared type is the same for all
    return points.
    """
    lines: list[str] = ["    return type context:"]
    declared_r: str = ""
    for lineno, inferred_r, decl_r in mismatch_frames:
        declared_r = decl_r
        if lineno is not None:
            lines.append(f"      returned (line {lineno}): {inferred_r}")
        else:
            lines.append(f"      returned: {inferred_r}")
    lines.append(f"      declared: {declared_r}")
    return lines


def _format_result_block(result: CheckResult, function_lines: dict[str, int]) -> list[str]:
    """Render one function's status line plus its errors/warnings."""
    lines: list[str] = []
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
    if result.trace and result.mismatch_frames:
        lines.extend(_format_mismatch_context(result.mismatch_frames))
    if result.trace:
        lines.append("    trace:")
        for step in result.trace:
            lines.append(f"      \033[2m{step}\033[0m")
    return lines


def _format_summary(results: list[CheckResult]) -> str:
    """Aggregate pass/fail/warning counts into the closing summary line."""
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    warn_count = sum(1 for r in results if r.warnings)

    if failed_count == 0:
        summary = f"\033[32mAll {passed_count} function(s) passed.\033[0m"
    else:
        summary = f"\033[31m{failed_count} function(s) failed, {passed_count} passed.\033[0m"
    if warn_count:
        summary += f" \033[33m({warn_count} with warnings)\033[0m"
    return summary


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
    for result in results:
        lines.extend(_format_result_block(result, function_lines))
    lines.append("")
    lines.append(_format_summary(results))
    return "\n".join(lines)


def format_results_files(groups: list[FileResults], verbose: bool = False) -> str:
    """Format check results grouped per file, each under its path header.

    Every checked file is listed — files without annotated functions get
    an explicit note — so multi-file runs show exactly what was targeted
    and which file each function came from. Line numbers come from each
    group's own table, so same-named functions in different files cannot
    shadow each other.
    """
    lines: list[str] = []
    for group in groups:
        lines.append(group.file_path)
        if not group.results:
            lines.append("  (no functions with DataFrame[Schema] annotations)")
        for result in group.results:
            lines.extend(_format_result_block(result, group.function_lines))
        lines.append("")

    all_results = [r for g in groups for r in g.results]
    if not all_results:
        lines.append("No functions with DataFrame[Schema] annotations found.")
    else:
        lines.append(_format_summary(all_results))
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
    parser.add_argument(
        "--polars-version",
        metavar="VERSION",
        default=None,
        help=("Override the detected polars version assumed by the analyzer (e.g. 1.0, 1.32.1)"),
    )
    parser.add_argument(
        "--pandera-version",
        metavar="VERSION",
        default=None,
        help="Override the detected pandera version (e.g. 0.20)",
    )
    parser.add_argument(
        "--no-version-check",
        action="store_true",
        help="Skip the polars / pandera version detection and warning",
    )
    parser.add_argument(
        "--rename-targets",
        metavar="FILE:LINE:COL",
        default=None,
        help=(
            "Read-only query mode (Batch C): given a source position on a "
            "column-name token, emit JSON listing every source occurrence "
            "that PROVABLY refers to the same column (declaration + references "
            "sharing a (schema, field) origin). LINE is 1-indexed, COL is "
            "0-indexed. No files are modified."
        ),
    )
    return parser


def _check_file_with_locations(
    file_path: Path,
    collect_trace: bool = False,
) -> tuple[list[CheckResult], dict[str, int], dict[str, int], list[dict]]:
    """
    Check a file and return results plus per-function source line numbers
    and schema summaries (the ``functions`` JSON array, D-11 hover data).

    On read or parse failures, returns a single failing CheckResult and empty
    maps. Never raises for those cases. The analyses are produced once and
    fed through ``check_function`` directly (``check_source`` would
    re-analyze the file a second time).
    """
    try:
        source = file_path.read_text()
    except (UnicodeDecodeError, OSError) as err:
        return [_parse_error_result(file_path, err)], {}, {}, []
    try:
        analyses = analyze_source(source, file_path=file_path, collect_trace=collect_trace)
    except SyntaxError as err:
        return [_parse_error_result(file_path, err)], {}, {}, []
    results = [check_function(a) for a in analyses]
    function_lines = {a.name: a.lineno for a in analyses}
    function_end_lines = {a.name: a.end_lineno for a in analyses}
    return results, function_lines, function_end_lines, function_summaries(analyses)


def _expand_directory_groups(dir_path: Path, collect_trace: bool = False) -> list[FileResults]:
    """Per-file FileResults for every .py under dir_path."""
    groups: list[FileResults] = []
    for py_file in sorted(dir_path.glob("**/*.py")):
        results, function_lines, function_end_lines, functions = _check_file_with_locations(
            py_file, collect_trace=collect_trace
        )
        groups.append(
            FileResults(
                file_path=str(py_file),
                results=results,
                function_lines=function_lines,
                function_end_lines=function_end_lines,
                functions=functions,
            )
        )
    return groups


def _emit_version_warnings(
    paths: list[Path],
    polars_override: str | None,
    pandera_override: str | None,
    no_color: bool,
) -> None:
    """Detect polars/pandera versions from the target project and warn on
    out-of-range versions. Stderr-only — never affects the exit code."""
    project_root = None
    for p in paths:
        if not p.exists():
            continue
        project_root = find_project_root(p)
        if project_root is not None:
            break
    info = detect_versions(
        project_root,
        polars_override=polars_override,
        pandera_override=pandera_override,
    )
    for w in check_versions(info):
        if no_color:
            print(f"! {w.message}", file=sys.stderr)
        else:
            print(f"\033[33m! {w.message}\033[0m", file=sys.stderr)


def _parse_position_arg(spec: str) -> tuple[Path, int, int] | None:
    """Parse a ``FILE:LINE:COL`` position spec for ``--rename-targets``.

    ``LINE`` is 1-indexed, ``COL`` is 0-indexed (the shape ``ast`` exposes).
    The FILE part may itself contain colons (Windows drive letters), so the
    spec is split from the RIGHT: the last two ``:``-separated fields are
    LINE and COL, everything before is the path. Returns ``None`` when the
    trailing fields are not integers.
    """
    parts = spec.rsplit(":", 2)
    if len(parts) != 3:
        return None
    file_str, line_str, col_str = parts
    try:
        line = int(line_str)
        col = int(col_str)
    except ValueError:
        return None
    return Path(file_str), line, col


def _run_rename_targets(spec: str) -> int:
    """Handle the read-only ``--rename-targets FILE:LINE:COL`` query mode.

    Emits the rename-target JSON payload on stdout and returns 0. A malformed
    position spec or a missing file is reported on stderr with exit code 1.
    """
    from polypolarism.column_index import rename_targets

    parsed = _parse_position_arg(spec)
    if parsed is None:
        print(
            f"Error: invalid --rename-targets position '{spec}' (expected FILE:LINE:COL)",
            file=sys.stderr,
        )
        return 1
    file_path, line, col = parsed
    if not file_path.exists():
        print(f"Error: Path not found: {file_path}", file=sys.stderr)
        return 1
    result = rename_targets(file_path, line=line, col=col)
    print(json.dumps(result))
    return 0


def main(args: list[str] | None = None) -> int:
    """Entry point for the CLI. Returns 0 on success, 1 on any failure."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.rename_targets is not None:
        return _run_rename_targets(parsed.rename_targets)

    if not parsed.paths:
        parser.print_help()
        return 0

    if not parsed.no_version_check:
        _emit_version_warnings(
            paths=parsed.paths,
            polars_override=parsed.polars_version,
            pandera_override=parsed.pandera_version,
            no_color=parsed.no_color,
        )

    file_groups: list[FileResults] = []

    for path in parsed.paths:
        if not path.exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            return 1

        if path.is_file():
            results, function_lines, function_end_lines, functions = _check_file_with_locations(
                path, collect_trace=parsed.verbose
            )
            file_groups.append(
                FileResults(
                    file_path=str(path),
                    results=results,
                    function_lines=function_lines,
                    function_end_lines=function_end_lines,
                    functions=functions,
                )
            )
        elif path.is_dir():
            file_groups.extend(_expand_directory_groups(path, collect_trace=parsed.verbose))

    all_results: list[CheckResult] = [r for g in file_groups for r in g.results]

    if parsed.format == "json":
        if len(file_groups) == 1:
            single = file_groups[0]
            output = format_json(
                single.results,
                single.file_path,
                single.function_lines,
                single.function_end_lines,
                functions=single.functions,
            )
        else:
            output = format_json_files(file_groups)
    else:
        output = format_results_files(file_groups, verbose=parsed.verbose)
        if parsed.no_color:
            import re

            output = re.sub(r"\033\[[0-9;]*m", "", output)

    print(output)

    if any(not r.passed for r in all_results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
