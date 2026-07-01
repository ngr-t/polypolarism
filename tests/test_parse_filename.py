"""Issue #122: every ``ast.parse`` of a scanned file must carry the real
filename, so any ``SyntaxWarning`` / ``SyntaxError`` raised while parsing it is
attributed to the offending file instead of the useless ``<unknown>:N``.

On Python 3.12+ a legacy invalid escape sequence (e.g. a non-raw ``"\\s+"``)
emits a ``SyntaxWarning`` at parse time; the warning's ``filename`` is whatever
was passed to ``ast.parse(..., filename=...)`` — or ``"<unknown>"`` when omitted.
We assert the real path is threaded through the file-reading parse paths.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from polypolarism.analyzer import analyze_source
from polypolarism.column_index import build_column_index

# A module-level non-raw string literal with an invalid escape sequence.
# Parsing it triggers the SyntaxWarning this issue is about.
_BAD_ESCAPE = 'PATTERN = "\\s+"\n'


def _syntax_warning_filenames(records: list[warnings.WarningMessage]) -> list[str]:
    return [w.filename for w in records if issubclass(w.category, SyntaxWarning)]


def _project_marker(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')


class TestAstParseFilename:
    def test_analyze_source_attributes_warning_to_file_path(self):
        path = Path("/proj/schemas.py")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            analyze_source(_BAD_ESCAPE, file_path=path)
        files = _syntax_warning_filenames(rec)
        assert files, "expected a SyntaxWarning from the invalid escape"
        assert "<unknown>" not in files
        assert all(f == str(path) for f in files), files

    def test_analyze_source_without_path_does_not_fabricate_filename(self):
        # Legacy callers pass raw source with no file_path — must not crash and
        # must never fabricate a bogus ``"None"`` filename.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            analyze_source(_BAD_ESCAPE)  # file_path=None
        files = _syntax_warning_filenames(rec)
        assert files
        assert "None" not in files

    def test_build_column_index_attributes_main_file(self, tmp_path: Path):
        _project_marker(tmp_path)
        target = tmp_path / "schemas.py"
        target.write_text(
            "import pandera.polars as pa\n"
            'PATTERN = "\\s+"\n'
            "class S(pa.DataFrameModel):\n    a: int\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            build_column_index(target)
        files = _syntax_warning_filenames(rec)
        assert files, "expected a SyntaxWarning from the scanned file"
        assert "<unknown>" not in files
        assert all(f == str(target) for f in files), files

    def test_build_column_index_attributes_imported_file(self, tmp_path: Path):
        # The issue's core symptom: a bad escape in an IMPORTED module surfaced
        # as an untraceable ``<unknown>:N`` because the import-following parse
        # paths dropped the filename.
        _project_marker(tmp_path)
        helper = tmp_path / "helper.py"
        helper.write_text(
            "import pandera.polars as pa\n"
            'PATTERN = "\\s+"\n'
            "class Base(pa.DataFrameModel):\n    a: int\n"
        )
        main = tmp_path / "main.py"
        main.write_text(
            "from helper import Base\nimport pandera.polars as pa\nclass S(Base):\n    b: int\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            build_column_index(main)
        files = _syntax_warning_filenames(rec)
        assert files, "expected a SyntaxWarning from the imported file"
        assert "<unknown>" not in files
        assert str(helper) in files
