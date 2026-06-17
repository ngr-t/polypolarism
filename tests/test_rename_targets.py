"""Tests for the ``--rename-targets`` capability (Batch C).

The capability returns the set of source occurrences that PROVABLY refer to
the SAME column, so the extension can offer a safe rename. Soundness is
paramount: a returned occurrence is rewritten by the editor, so a wrong
target corrupts the user's code. Two references are "the same column" ONLY
when they share a provable ``(schema, field)`` origin; everything else falls
back to single-occurrence.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from polypolarism.cli import main
from polypolarism.column_index import (
    build_column_index,
    rename_targets,
)


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip("\n"))


def _project_marker(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "demo"\n')


# --------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------


class TestIndexDeclarationsAndRefs:
    def test_schema_field_declaration_recorded_with_origin(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "schemas.py"
        _write(
            f,
            """
            import pandera.polars as pa


            class Input(pa.DataFrameModel):
                region: str
                amount: int
            """,
        )
        index = build_column_index(f)
        decls = [r for r in index if r.column_name == "region"]
        assert decls, "region declaration not indexed"
        decl = decls[0]
        assert decl.origin == ("Input", "region")
        # The AnnAssign target name token, 1-indexed line.
        assert decl.span.line == 5

    def test_pl_col_ref_resolves_to_field_origin(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str
                amount: int


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"))
            """,
        )
        index = build_column_index(f)
        refs = [r for r in index if r.column_name == "region"]
        origins = {r.origin for r in refs}
        assert ("Input", "region") in origins
        ref_lines = sorted(r.span.line for r in refs if r.origin == ("Input", "region"))
        assert 7 in ref_lines  # declaration line
        assert 12 in ref_lines  # pl.col ref line

    def test_regex_col_is_not_indexed(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("^reg.*$"))
            """,
        )
        index = build_column_index(f)
        assert not [r for r in index if r.column_name == "^reg.*$"]

    def test_open_frame_ref_has_no_origin(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl


            def f(df: pl.DataFrame) -> pl.DataFrame:
                return df.select(pl.col("region"))
            """,
        )
        index = build_column_index(f)
        refs = [r for r in index if r.column_name == "region"]
        assert len(refs) == 1
        assert refs[0].origin is None


# --------------------------------------------------------------------------
# rename_targets resolution
# --------------------------------------------------------------------------


class TestRenameTargetsResolution:
    def test_query_on_field_declaration_returns_decl_plus_refs(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str
                amount: int


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"), pl.col("amount"))
            """,
        )
        result = rename_targets(f, line=7, col=4)
        assert result["column"] == "region"
        assert result["schema"] == "Input"
        lines = sorted(t["line"] for t in result["targets"])
        assert 7 in lines  # declaration
        assert 12 in lines  # pl.col("region") ref

    def test_query_on_pl_col_resolves_full_set(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.filter(pl.col("region") == "x")
            """,
        )
        col = _col_inside_literal(f, 11, "region")
        result = rename_targets(f, line=11, col=col)
        assert result["column"] == "region"
        assert result["schema"] == "Input"
        lines = sorted(t["line"] for t in result["targets"])
        assert lines == [7, 11]

    def test_renamed_column_does_not_merge_old_and_new(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str


            def f(df: DataFrame[Input]):
                renamed = df.rename({"region": "area"})
                return renamed.select(pl.col("area"))
            """,
        )
        col = _col_inside_literal(f, 12, "area")
        result = rename_targets(f, line=12, col=col)
        # `area` has no provable schema origin -> single-occurrence fallback.
        assert len(result["targets"]) == 1
        assert result["targets"][0]["line"] == 12

    def test_query_with_no_column_token_returns_empty(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Input(pa.DataFrameModel):
                region: str


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"))
            """,
        )
        result = rename_targets(f, line=1, col=0)
        assert result["targets"] == []

    def test_open_frame_query_is_single_occurrence(self, tmp_path: Path):
        _project_marker(tmp_path)
        f = tmp_path / "app.py"
        _write(
            f,
            """
            import polars as pl


            def f(df: pl.DataFrame) -> pl.DataFrame:
                a = df.select(pl.col("region"))
                b = df.select(pl.col("region"))
                return a
            """,
        )
        col = _col_inside_literal(f, 5, "region")
        result = rename_targets(f, line=5, col=col)
        assert len(result["targets"]) == 1
        assert result["targets"][0]["line"] == 5


# --------------------------------------------------------------------------
# Cross-file
# --------------------------------------------------------------------------


class TestCrossFile:
    def test_field_decl_query_includes_cross_file_refs(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Input(pa.DataFrameModel):
                region: str
                amount: int
            """,
        )
        app = tmp_path / "app.py"
        _write(
            app,
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Input


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"))
            """,
        )
        result = rename_targets(tmp_path / "schemas.py", line=5, col=4)
        assert result["column"] == "region"
        assert result["schema"] == "Input"
        files = {t["file"] for t in result["targets"]}
        assert str((tmp_path / "schemas.py").resolve()) in files
        assert str((tmp_path / "app.py").resolve()) in files

    def test_pl_col_query_finds_cross_file_declaration(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Input(pa.DataFrameModel):
                region: str
            """,
        )
        app = tmp_path / "app.py"
        _write(
            app,
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Input


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"))
            """,
        )
        col = _col_inside_literal(app, 7, "region")
        result = rename_targets(app, line=7, col=col)
        files = {t["file"] for t in result["targets"]}
        assert str((tmp_path / "schemas.py").resolve()) in files
        assert str((tmp_path / "app.py").resolve()) in files


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


class TestCli:
    def test_cli_rename_targets_emits_json(self, tmp_path: Path, capsys):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Input(pa.DataFrameModel):
                region: str
            """,
        )
        app = tmp_path / "app.py"
        _write(
            app,
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Input


            def f(df: DataFrame[Input]) -> DataFrame[Input]:
                return df.select(pl.col("region"))
            """,
        )
        rc = main(["--rename-targets", f"{tmp_path / 'schemas.py'}:5:4"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["column"] == "region"
        assert payload["schema"] == "Input"
        files = {t["file"] for t in payload["targets"]}
        assert str((tmp_path / "schemas.py").resolve()) in files
        assert str((tmp_path / "app.py").resolve()) in files
        for t in payload["targets"]:
            assert {"file", "line", "column", "end_line", "end_column"} <= set(t)

    def test_cli_no_token_returns_empty_targets(self, tmp_path: Path, capsys):
        _project_marker(tmp_path)
        app = tmp_path / "app.py"
        _write(
            app,
            """
            import polars as pl


            def f(df: pl.DataFrame) -> pl.DataFrame:
                return df
            """,
        )
        rc = main(["--rename-targets", f"{app}:1:0"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["targets"] == []

    def test_cli_unprovable_ref_returns_single_occurrence(self, tmp_path: Path, capsys):
        _project_marker(tmp_path)
        app = tmp_path / "app.py"
        _write(
            app,
            """
            import polars as pl


            def f(df: pl.DataFrame) -> pl.DataFrame:
                return df.select(pl.col("region"))
            """,
        )
        col = _col_inside_literal(app, 5, "region")
        rc = main(["--rename-targets", f"{app}:5:{col}"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert len(payload["targets"]) == 1
        assert payload["targets"][0]["line"] == 5


def _col_inside_literal(path: Path, line: int, name: str) -> int:
    """Return a 0-indexed column INSIDE the ``"name"`` string literal on
    ``line`` so a position query lands on that column token."""
    text = path.read_text().splitlines()
    src_line = text[line - 1]
    needle = f'"{name}"'
    idx = src_line.index(needle)
    return idx + 1  # land inside the quotes, on the name's first char
