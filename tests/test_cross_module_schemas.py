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
        # Issue #76: this test used to pass VACUOUSLY — ``Users(WithId)``
        # was not recognized as a schema (its base is imported), so the
        # function carried no declared type and "passed" with a PLW006
        # the assertions never inspected. It now checks the inherited
        # column is genuinely resolved.
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
        # The resolution must be real, not a PLW006-mediated skip.
        assert not any("PLW006" in str(w) for w in results[0].warnings), results[0].warnings

    def test_inherited_column_violation_fails(self, tmp_path: Path):
        # The other half of the vacuity (issue #76): dropping an
        # INHERITED column must be a provable miss, not a silent pass.
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
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from base import WithId


            class Users(WithId):
                name: str


            def drops_inherited(df: DataFrame[Users]) -> DataFrame[Users]:
                return df.select(pl.col("name"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        assert not results[0].passed
        assert any("user_id" in str(e) for e in results[0].errors), results[0].errors


class TestCrossFileInheritance:
    """Issue #76: subclasses of IMPORTED schema bases resolve with the
    parent's fields merged — in the analyzed file, in imported files, and
    through alias / dotted-base spellings."""

    def _base_file(self, tmp_path: Path) -> None:
        _project_marker(tmp_path)
        _write(
            tmp_path / "base.py",
            """
            import pandera.polars as pa


            class WithId(pa.DataFrameModel):
                user_id: int

                class Config:
                    strict = True
            """,
        )

    def test_subclass_in_analyzed_file_merges_imported_parent(self, tmp_path: Path):
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            from base import WithId


            class Users(WithId):
                name: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        assert set(users.columns) == {"user_id", "name"}
        assert users.strict is True  # Config inherited like the same-file path

    def test_chain_rooted_at_imported_base(self, tmp_path: Path):
        # class A(Imported); class B(A) — the in-file chain only becomes
        # recognizable once the imported root is known (fixpoint).
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            from base import WithId


            class Accounts(WithId):
                balance: float


            class PremiumAccounts(Accounts):
                tier: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        premium = registry.get("PremiumAccounts")
        assert premium is not None
        assert set(premium.columns) == {"user_id", "balance", "tier"}

    def test_aliased_import_base(self, tmp_path: Path):
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            from base import WithId as IdMixin


            class Users(IdMixin):
                name: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        assert set(users.columns) == {"user_id", "name"}

    def test_dotted_module_qualified_base(self, tmp_path: Path):
        # ``import base`` + ``class Users(base.WithId)`` — the dotted key
        # machinery from issue #68 must serve base-class lookup too.
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            import base


            class Users(base.WithId):
                name: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        assert set(users.columns) == {"user_id", "name"}

    def test_inheritance_inside_imported_file(self, tmp_path: Path):
        # The issue's exact shape: the subclass lives in an imported file
        # whose own base import chains one file further.
        self._base_file(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            from base import WithId


            class Users(WithId):
                name: str
            """,
        )
        source = "from schemas import Users\n"
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        assert set(users.columns) == {"user_id", "name"}

    def test_child_override_wins_over_imported_parent(self, tmp_path: Path):
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            from base import WithId


            class Users(WithId):
                user_id: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        from polypolarism.types import Utf8

        assert users.columns["user_id"].dtype == Utf8()

    def test_local_class_shadows_same_named_import(self, tmp_path: Path):
        # ``from schemas import ...`` merges schemas.py's Users into the
        # registry, but the analyzed file defines its OWN Users — at
        # runtime the local class wins, so the local definition must take
        # precedence over the merged import.
        self._base_file(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa


            class Users(pa.DataFrameModel):
                wrong: float
            """,
        )
        source = textwrap.dedent(
            """
            from base import WithId
            from schemas import Users as _ImportedUsers


            class Users(WithId):
                name: str
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        users = registry.get("Users")
        assert users is not None
        assert set(users.columns) == {"user_id", "name"}, users.columns


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


class TestAliasedBaseRepeatImport:
    """Issue #80: the alias registration lived behind the ``visited``
    skip in ``_merge_imports`` — a module imported by an EARLIER import
    statement never got its later statements' aliases registered, so
    ``from m import Base as B0`` + ``class C(B0)`` resolved only when it
    was the module's first import statement."""

    def _base_file(self, tmp_path: Path) -> None:
        _project_marker(tmp_path)
        _write(
            tmp_path / "_mod_base.py",
            """
            import pandera.polars as pa


            class Base(pa.DataFrameModel):
                id: int
                x: str

                class Config:
                    strict = True
            """,
        )

    def test_alias_in_second_import_statement_of_same_module(self, tmp_path: Path):
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            from _mod_base import Base
            from _mod_base import Base as B0


            class FromAlias(B0):
                w: int
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        from_alias = registry.get("FromAlias")
        assert from_alias is not None
        assert set(from_alias.columns) == {"id", "x", "w"}

    def test_alias_after_plain_module_import(self, tmp_path: Path):
        # ``import _mod_base`` (the #68 dotted-key path) does not mark the
        # file visited for _merge_imports, but mixed forms must not
        # interfere either way.
        self._base_file(tmp_path)
        source = textwrap.dedent(
            """
            import _mod_base
            from _mod_base import Base as B0


            class FromAlias(B0):
                w: int
            """
        )
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        from_alias = registry.get("FromAlias")
        assert from_alias is not None
        assert set(from_alias.columns) == {"id", "x", "w"}

    def test_issue_80_end_to_end_violation_detected(self, tmp_path: Path):
        self._base_file(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
            from _mod_base import Base
            from _mod_base import Base as B0


            class Src(pa.DataFrameModel):
                a: int
                b: str


            class FromAlias(B0):
                w: int


            def t4_aliased_base(df: DataFrame[Src]) -> DataFrame[FromAlias]:
                return df.select(id=pl.col("b"), x=pl.col("b"), w=pl.col("a"))
            """,
        )
        results = check_file(tmp_path / "app.py")
        target = [r for r in results if r.function_name == "t4_aliased_base"]
        assert len(target) == 1
        assert not target[0].passed
        assert any("id" in str(e) for e in target[0].errors), target[0].errors
        assert not any("PLW006" in str(w) for w in target[0].warnings), target[0].warnings


class TestObjectApiCrossModule:
    """Backlog C-11: object-API schemas merge through imports like class
    schemas (they live in the same registry)."""

    def test_imported_object_schema_resolves(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "schemas.py",
            """
            import pandera.polars as pa

            order_schema = pa.DataFrameSchema({"order_id": pa.Column(int)}, strict=True)
            """,
        )
        source = "from schemas import order_schema\n"
        registry = collect_schemas_with_imports(ast.parse(source), tmp_path / "app.py")
        schema = registry.get("order_schema")
        assert schema is not None
        assert set(schema.columns) == {"order_id"}
        assert schema.strict is True


class TestProjectRootDiscovery:
    """Layout coverage for the resolver's upward walk (user report
    2026-06-12): a dotted ``from module1.module2 import (...)`` failed
    to resolve in every tree whose root had no packaging marker —
    ``.git``-only repos — and in the ``src/`` layout even with one."""

    APP = """
    import polars as pl
    from pandera.typing.polars import DataFrame
    from module1.module2 import (
        InputSchema,
    )


    def f(df: DataFrame[InputSchema]) -> DataFrame[InputSchema]:
        return df.select(pl.col("id"))
    """

    SCHEMA = """
    import pandera.polars as pa


    class InputSchema(pa.DataFrameModel):
        id: int
    """

    def _package(self, base: Path) -> None:
        pkg = base / "module1"
        pkg.mkdir(parents=True)
        _write(pkg / "__init__.py", "")
        _write(pkg / "module2.py", self.SCHEMA)

    def _assert_resolves(self, app: Path) -> None:
        results = check_file(app)
        assert len(results) == 1
        assert results[0].passed, results[0].errors
        assert not results[0].warnings, results[0].warnings

    def test_git_dir_marks_root_file_inside_package(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        self._package(tmp_path)
        _write(tmp_path / "module1" / "app.py", self.APP)
        self._assert_resolves(tmp_path / "module1" / "app.py")

    def test_git_dir_marks_root_sibling_scripts_dir(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        self._package(tmp_path)
        (tmp_path / "scripts").mkdir()
        _write(tmp_path / "scripts" / "app.py", self.APP)
        self._assert_resolves(tmp_path / "scripts" / "app.py")

    def test_src_layout_resolves_through_src_dir(self, tmp_path: Path):
        _project_marker(tmp_path)
        self._package(tmp_path / "src")
        (tmp_path / "scripts").mkdir()
        _write(tmp_path / "scripts" / "app.py", self.APP)
        self._assert_resolves(tmp_path / "scripts" / "app.py")

    def test_markerless_tree_falls_back_to_cwd(self, tmp_path: Path, monkeypatch):
        self._package(tmp_path)
        (tmp_path / "scripts").mkdir()
        _write(tmp_path / "scripts" / "app.py", self.APP)
        monkeypatch.chdir(tmp_path)
        self._assert_resolves(tmp_path / "scripts" / "app.py")

    def test_cwd_fallback_requires_file_under_cwd(self, tmp_path: Path, monkeypatch):
        """The cwd fallback must not fire for files outside the
        invocation tree — resolution stays bounded."""
        self._package(tmp_path / "elsewhere")
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "app.py", self.APP)
        monkeypatch.chdir(tmp_path / "elsewhere")
        results = check_file(outside / "app.py")
        assert len(results) == 1
        assert any("PLW006" in str(w) for w in results[0].warnings)


class TestUnresolvedImportDiagnostic:
    """When the schema name WAS imported but the module didn't resolve,
    PLW006 must say that instead of suggesting an import the user
    already wrote (user report 2026-06-12)."""

    def test_plw006_names_the_unresolved_import(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            import polars as pl
            from pandera.typing.polars import DataFrame
            from module1.module2 import (
                InputSchema,
            )


            def f(df: DataFrame[InputSchema]) -> DataFrame[InputSchema]:
                return df
            """,
        )
        results = check_file(tmp_path / "app.py")
        assert len(results) == 1
        joined = "\n".join(str(w) for w in results[0].warnings)
        assert "PLW006" in joined
        assert "`from module1.module2 import InputSchema`" in joined
        assert "did not resolve" in joined
        # The generic "import it" suggestion would be confusing here.
        assert "import it from a project-local module" not in joined

    def test_generic_hint_kept_when_name_never_imported(self, tmp_path: Path):
        _project_marker(tmp_path)
        _write(
            tmp_path / "app.py",
            """
            from pandera.typing.polars import DataFrame


            def f(df: DataFrame[Ghost]) -> DataFrame[Ghost]:
                return df
            """,
        )
        results = check_file(tmp_path / "app.py")
        joined = "\n".join(str(w) for w in results[0].warnings)
        assert "import it from a project-local module" in joined
