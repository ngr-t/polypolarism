"""Golden-file fixture harness — compiler-style "UI tests".

Modeled on rustc's UI test suite and LLVM lit/FileCheck: every fixture under
``tests/fixtures/{valid,invalid,warning}/`` is discovered automatically, run
through the real ``check_file()`` entry point, and its full diagnostic output
(status, errors, warnings per function) is compared against the ``.expected``
golden file sitting next to the fixture.

Category invariants, enforced independently of the golden contents so a
stale golden can never bless a category violation:

* ``valid/``   — every function passes
* ``invalid/`` — at least one function fails
* ``warning/`` — every function passes and at least one emits a warning

When diagnostics change intentionally, regenerate the goldens and review the
diff in git like any other code change:

    POLYPOLARISM_UPDATE_EXPECTED=1 uv run pytest tests/test_fixtures.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import polypolarism.diagnostics as diagnostics
from polypolarism.checker import CheckResult
from polypolarism.cli import check_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATEGORIES = ("valid", "invalid", "warning")
UPDATE = os.environ.get("POLYPOLARISM_UPDATE_EXPECTED") == "1"

DIAGNOSTIC_CODE = re.compile(r"PL[YW]\d{3}")

# Diagnostic codes that cannot be triggered from a self-contained fixture
# file. Each entry must name the unit test that covers it instead.
FIXTURE_EXEMPT_CODES = {
    "PLW010",  # environment version floor — covered by test_version_check.py
}


def render_report(results: list[CheckResult]) -> str:
    """Render check results as a stable, human-reviewable diagnostic report."""
    lines: list[str] = []
    for result in results:
        if not result.passed:
            status = "FAIL"
        elif result.warnings:
            status = "WARN"
        else:
            status = "OK"
        lines.append(f"{result.function_name}: {status}")
        lines.extend(f"  error: {error}" for error in result.errors)
        lines.extend(f"  warning: {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def _discover() -> list:
    params = []
    for category in CATEGORIES:
        for path in sorted((FIXTURES_DIR / category).glob("*.py")):
            params.append(pytest.param(category, path, id=f"{category}/{path.name}"))
    return params


def _assert_category_invariant(category: str, results: list[CheckResult], fixture: Path) -> None:
    if category == "valid":
        failed = [r.function_name for r in results if not r.passed]
        assert not failed, f"valid fixture {fixture.name} has failing functions: {failed}"
    elif category == "invalid":
        assert any(not r.passed for r in results), (
            f"invalid fixture {fixture.name} produced no errors — "
            "move it to valid/ or fix the checker"
        )
    elif category == "warning":
        failed = [r.function_name for r in results if not r.passed]
        assert not failed, f"warning fixture {fixture.name} has failing functions: {failed}"
        assert any(r.warnings for r in results), (
            f"warning fixture {fixture.name} produced no warnings — "
            "move it to valid/ or fix the analyzer"
        )


@pytest.mark.parametrize(("category", "fixture"), _discover())
def test_fixture_against_golden(category: str, fixture: Path) -> None:
    results = check_file(fixture)
    assert results, f"fixture {fixture.name} contains no checkable functions"
    _assert_category_invariant(category, results, fixture)

    report = render_report(results)
    expected_path = fixture.with_suffix(".expected")

    if UPDATE:
        expected_path.write_text(report)

    assert expected_path.exists(), (
        f"missing golden file {expected_path.name} — generate it with "
        "POLYPOLARISM_UPDATE_EXPECTED=1 uv run pytest tests/test_fixtures.py "
        "and review the result before committing"
    )
    assert report == expected_path.read_text(), (
        f"diagnostics for {fixture.name} diverge from {expected_path.name}. "
        "If the change is intentional, regenerate with "
        "POLYPOLARISM_UPDATE_EXPECTED=1 and review the golden diff."
    )


def test_no_orphaned_golden_files() -> None:
    """Every .expected file must belong to a still-existing fixture."""
    orphans = [
        expected
        for category in CATEGORIES
        for expected in (FIXTURES_DIR / category).glob("*.expected")
        if not expected.with_suffix(".py").exists()
    ]
    assert not orphans, f"golden files without fixtures: {[o.name for o in orphans]}"


def test_every_diagnostic_code_is_exercised_by_a_fixture() -> None:
    """Each PLY/PLW code defined in diagnostics.py must appear in at least
    one golden file, so no diagnostic can silently lose its end-to-end test.
    """
    defined = {
        value
        for name, value in vars(diagnostics).items()
        if isinstance(value, str) and DIAGNOSTIC_CODE.fullmatch(name)
    }
    covered: set[str] = set()
    for category in CATEGORIES:
        for expected in (FIXTURES_DIR / category).glob("*.expected"):
            covered.update(DIAGNOSTIC_CODE.findall(expected.read_text()))

    missing = defined - covered - FIXTURE_EXEMPT_CODES
    assert not missing, (
        f"diagnostic codes without a covering fixture: {sorted(missing)} — "
        "add a fixture under tests/fixtures/ or add the code to "
        "FIXTURE_EXEMPT_CODES with a pointer to its unit test"
    )

    stale_exemptions = FIXTURE_EXEMPT_CODES & covered
    assert not stale_exemptions, (
        f"codes in FIXTURE_EXEMPT_CODES are now fixture-covered: {sorted(stale_exemptions)}"
    )
