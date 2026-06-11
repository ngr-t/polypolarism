"""Tests for cross-module schema import resolution (issue #1)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from polypolarism.checker import check_source
from polypolarism.cli import check_file
from polypolarism.pandera_schema import collect_schemas_with_imports


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip("\n"))


def _project_marker(tmp_path: Path) -> None:
    """Drop a minimal pyproject.toml so the import resolver bounds its
    upward walk at ``tmp_path`` instead of climbing into the actual repo."""
    _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "demo"
        """,
    )


class TestSiblingFileImport:
    """``schemas.py`` next to ``app.py``."""

    def test_resolves_sibling_schema(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Users(pa.DataFrameModel):
                user_id: int
                name: str
            """,
        )
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Users


            def take_users(df: DataFrame[Users]) -> DataFrame[Users]:
                return df.select(pl.col("user_id"), pl.col("name"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        # Function is no longer silently skipped — it shows up and passes.
        assert len(results) == 1, [r.errors for r in results]
        assert results[0].function_name == "take_users"
        assert results[0].passed, results[0].errors

    def test_string_check_source_without_path_warns_on_unresolved(self):
        # When no file_path is provided (and no in-source schema), the
        # function should still surface so the user knows nothing was
        # silently skipped.
        src = textwrap.dedent(
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Users


            def take_users(df: DataFrame[Users]) -> DataFrame[Users]:
                return df.select(pl.col("user_id"), pl.col("name"))
            """
        )
        results = check_source(src)
        assert len(results) == 1
        assert any("PLW006" in w for w in results[0].warnings)
        assert any("Users" in w for w in results[0].warnings)


class TestPackageImport:
    """``from package.module import Schema`` walking up to project root."""

    def test_resolves_package_module(self, tmp_path: Path):
        _project_marker(tmp_path)
        pkg = tmp_path / "demo"
        pkg.mkdir()
        _write(pkg / "__init__.py", "")
        _write(
            pkg / "schemas.py",
            """
            import pandera.polars as pa


            class Orders(pa.DataFrameModel):
                order_id: int
                amount: int
            """,
        )
        _write(
            pkg / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from demo.schemas import Orders


            def take(df: DataFrame[Orders]) -> DataFrame[Orders]:
                return df.select(pl.col("order_id"), pl.col("amount"))
            """,
        )
        results = check_file(pkg / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors


class TestRelativeImport:
    """``from .schemas import Users`` relative imports."""

    def test_resolves_relative_import(self, tmp_path: Path):
        _project_marker(tmp_path)
        pkg = tmp_path / "demo"
        pkg.mkdir()
        _write(pkg / "__init__.py", "")
        _write(
            pkg / "schemas.py",
            """
            import pandera.polars as pa


            class Users(pa.DataFrameModel):
                user_id: int
            """,
        )
        _write(
            pkg / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from .schemas import Users


            def take(df: DataFrame[Users]) -> DataFrame[Users]:
                return df.select(pl.col("user_id"))
            """,
        )
        results = check_file(pkg / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors


class TestTransitiveImport:
    """app -> schemas -> base — chain of imports."""

    def test_resolves_transitively(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "base.py",
            """
            import pandera.polars as pa


            class WithId(pa.DataFrameModel):
                user_id: int
            """,
        )
        _write(
            tmp_path / "schemas.py",
            """
            from base import WithId


            class Users(WithId):
                name: str
            """,
        )
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Users


            def take(df: DataFrame[Users]) -> DataFrame[Users]:
                return df.select(pl.col("user_id"), pl.col("name"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors


class TestUnresolvedSchemaWarning:
    """When the schema name can't be resolved, surface a PLW006 warning
    so the user knows the file isn't being silently treated as empty."""

    def test_unresolved_schema_warns(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            from pandera.typing.polars import DataFrame
            from third_party_module import NonExistent


            def take(df: DataFrame[NonExistent]) -> DataFrame[NonExistent]:
                return df
            """,
        )
        results = check_file(tmp_path / "app.py")
        # Function shows up rather than being silently skipped.
        assert len(results) == 1
        assert any("PLW006" in w for w in results[0].warnings)
        assert any("NonExistent" in w for w in results[0].warnings)


class TestModuleQualifiedImport:
    """``import schemas_mod`` + ``DataFrame[schemas_mod.Schema]`` (issue #68)."""

    def _write_remote_schema(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "schemas_mod.py",
            """
            import pandera.polars as pa


            class RemoteSchema(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True
                    coerce = True
            """,
        )

    def test_qualified_return_violation_detected(self, tmp_path: Path):
        # The issue #68 repro: RemoteSchema is strict {a: int}; returning
        # an extra column must FAIL instead of passing vacuously.
        _project_marker(tmp_path)
        self._write_remote_schema(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            import schemas_mod


            class Local(pa.DataFrameModel):
                a: int
                b: str


            def qualified_contradiction(df: DataFrame[Local]) -> DataFrame[schemas_mod.RemoteSchema]:
                return df.select(pl.col("a"), pl.col("b"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert not results[0].passed
        assert any("'b'" in str(e) for e in results[0].errors), results[0].errors

    def test_qualified_param_registers_function(self, tmp_path: Path):
        # Param-side qualified annotation: the function must not be
        # silently de-registered.
        _project_marker(tmp_path)
        self._write_remote_schema(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame

            import schemas_mod


            def take(df: DataFrame[schemas_mod.RemoteSchema]) -> DataFrame[schemas_mod.RemoteSchema]:
                return df.select(pl.col("a"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert results[0].function_name == "take"
        assert results[0].passed, results[0].errors
        assert not results[0].warnings, results[0].warnings

    def test_aliased_package_import(self, tmp_path: Path):
        # ``import pkg.schemas as s`` registers under the alias ``s``.
        _project_marker(tmp_path)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        _write(pkg / "__init__.py", "")
        _write(
            pkg / "schemas.py",
            """
            import pandera.polars as pa


            class Out(pa.DataFrameModel):
                a: int
            """,
        )
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame

            import pkg.schemas as s


            def take(df: DataFrame[s.Out]) -> DataFrame[s.Out]:
                return df.select(pl.col("a"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors
        assert not results[0].warnings, results[0].warnings

    def test_dotted_package_import(self, tmp_path: Path):
        # ``import pkg.schemas`` binds the full dotted path at use sites.
        _project_marker(tmp_path)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        _write(pkg / "__init__.py", "")
        _write(
            pkg / "schemas.py",
            """
            import pandera.polars as pa


            class Out(pa.DataFrameModel):
                a: int
            """,
        )
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame

            import pkg.schemas


            def take(df: DataFrame[pkg.schemas.Out]) -> DataFrame[pkg.schemas.Out]:
                return df.select(pl.col("a"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors
        assert not results[0].warnings, results[0].warnings

    def test_registry_holds_dotted_keys(self, tmp_path: Path):
        # Design pin: the registry stays flat; qualified imports register
        # schemas under their dotted spelling as written at use sites.
        _project_marker(tmp_path)
        self._write_remote_schema(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import schemas_mod
            """,
        )
        tree = ast.parse((tmp_path / "app.py").read_text())
        registry = collect_schemas_with_imports(tree, tmp_path / "app.py")
        assert "schemas_mod.RemoteSchema" in registry
        assert "RemoteSchema" not in registry  # plain import binds only the module name


class TestQualifiedUnresolvedWarning:
    """Qualified names that don't resolve must warn PLW006, never stay silent."""

    def test_unresolved_qualified_module_warns(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            from pandera.typing.polars import DataFrame
            import third_party_module


            def take(df: DataFrame[third_party_module.Remote]) -> DataFrame[third_party_module.Remote]:
                return df
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert any("PLW006" in w for w in results[0].warnings)
        # The warning names the full dotted reference.
        assert any("third_party_module.Remote" in w for w in results[0].warnings)

    def test_nested_class_reference_warns(self, tmp_path: Path):
        # Same-module nested classes (DataFrame[Outer.Inner]) are out of
        # scope for resolution but must warn instead of passing silently.
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame


            class Schemas:
                class Inner(pa.DataFrameModel):
                    a: int


            def take(df: DataFrame[Schemas.Inner]) -> DataFrame[Schemas.Inner]:
                return df
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert any("PLW006" in w for w in results[0].warnings)
        assert any("Schemas.Inner" in w for w in results[0].warnings)


class TestStdlibImportNotFollowed:
    """Stdlib / third-party imports (e.g. ``from typing import cast``)
    should not be picked up as schema sources, just silently skipped."""

    def test_stdlib_import_does_not_break(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Users(pa.DataFrameModel):
                user_id: int
            """,
        )
        _write(
            tmp_path / "app.py",
            """
            from typing import cast
            import polars as pl
            from pandera.typing.polars import DataFrame
            from schemas import Users


            def take(df: DataFrame[Users]) -> DataFrame[Users]:
                return cast(DataFrame[Users], Users.validate(df))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert results[0].passed, results[0].errors
