"""Tests for pandera_schema.py — schema class registry."""

from __future__ import annotations

import ast
import textwrap

from polypolarism.pandera_schema import collect_schemas
from polypolarism.types import (
    ColumnSpec,
    Datetime,
    Decimal,
    Enum,
    Float64,
    Int64,
    List,
    Nullable,
    Unknown,
    Utf8,
)


def _collect(src: str):
    tree = ast.parse(textwrap.dedent(src))
    return collect_schemas(tree)


class TestBasicSchema:
    def test_simple_schema(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int
                name: str
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns == {
            "id": ColumnSpec(Int64(), required=True),
            "name": ColumnSpec(Utf8(), required=True),
        }
        assert schema.strict is False

    def test_unqualified_base(self):
        registry = _collect(
            """
            from pandera.polars import DataFrameModel

            class S(DataFrameModel):
                value: float
            """
        )
        assert "S" in registry

    def test_non_schema_class_ignored(self):
        registry = _collect(
            """
            class Plain:
                x: int
            """
        )
        assert registry.get("Plain") is None


class TestFieldOptions:
    def test_field_nullable(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                name: str = pa.Field(nullable=True)
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["name"] == ColumnSpec(Nullable(Utf8()), required=True)

    def test_optional_column(self):
        registry = _collect(
            """
            from typing import Optional
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                age: Optional[int]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["age"] == ColumnSpec(Int64(), required=False)

    def test_optional_and_nullable(self):
        registry = _collect(
            """
            from typing import Optional
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                age: Optional[int] = pa.Field(nullable=True)
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["age"] == ColumnSpec(Nullable(Int64()), required=False)

    def test_polars_dtype_field(self):
        registry = _collect(
            """
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                price: pl.Float64
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["price"] == ColumnSpec(Float64(), required=True)

    def test_annotated_list(self):
        registry = _collect(
            """
            from typing import Annotated
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                items: Annotated[pl.List, pl.Int64()]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["items"] == ColumnSpec(List(Int64()), required=True)


class TestConfig:
    def test_strict_true(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    strict = True
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.strict is True

    def test_strict_false_explicit(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    strict = False
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.strict is False

    def test_no_config(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int
            """
        )
        schema_s = registry.get("S")
        assert schema_s is not None
        assert schema_s.strict is False
        assert schema_s.coerce is False

    def test_coerce_true(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    coerce = True
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.coerce is True

    def test_coerce_false_explicit(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    coerce = False
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.coerce is False

    def test_strict_and_coerce_together(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    strict = True
                    coerce = True
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.strict is True
        assert schema.coerce is True


class TestInheritance:
    def test_child_inherits_parent_columns(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                id: int

            class Child(Parent):
                name: str
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert child.columns == {
            "id": ColumnSpec(Int64(), required=True),
            "name": ColumnSpec(Utf8(), required=True),
        }
        assert child.bases == ["Parent"]

    def test_child_overrides_parent_field(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                value: int

            class Child(Parent):
                value: float
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert child.columns["value"] == ColumnSpec(Float64(), required=True)

    def test_strict_inherited(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                id: int

                class Config:
                    strict = True

            class Child(Parent):
                name: str
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert child.strict is True

    def test_coerce_inherited(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                id: int

                class Config:
                    coerce = True

            class Child(Parent):
                name: str
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert child.coerce is True

    def test_grandchild_chain(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class A(pa.DataFrameModel):
                a: int

            class B(A):
                b: str

            class C(B):
                c: float
            """
        )
        c = registry.get("C")
        assert c is not None
        assert set(c.columns.keys()) == {"a", "b", "c"}


class TestToFrameType:
    def test_to_frame_type(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    strict = True
            """
        )
        ft = registry.to_frame_type("S")
        assert ft is not None
        assert ft.strict is True
        assert ft.columns["id"] == ColumnSpec(Int64(), required=True)

    def test_to_frame_type_passes_coerce(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int

                class Config:
                    coerce = True
            """
        )
        ft = registry.to_frame_type("S")
        assert ft is not None
        assert ft.coerce is True

    def test_unknown_schema_returns_none(self):
        registry = _collect("")
        assert registry.to_frame_type("Unknown") is None


class TestDefinitionErrors:
    """Issue #69: a field annotation with the wrong ``Annotated`` metadata
    arity makes pandera raise TypeError the first time the schema is used.
    The registry records these per field so the analyzer can flag every
    function referencing the schema."""

    def test_broken_annotated_field_is_recorded(self):
        registry = _collect(
            """
            import typing
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                v: typing.Annotated[pl.Array, pl.Int64(), 2]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert "v" in schema.definition_errors
        assert "inner, shape, width" in schema.definition_errors["v"]

    def test_legal_schema_has_no_definition_errors(self):
        registry = _collect(
            """
            import typing
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                a: int
                v: typing.Annotated[pl.Array, pl.Int64(), 2, None]
                t: typing.Annotated[pl.Datetime, "us", None]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.definition_errors == {}

    def test_child_inherits_broken_field(self):
        # Probed: a child class inheriting the broken annotation crashes
        # exactly like the parent.
        registry = _collect(
            """
            import typing
            import polars as pl
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                v: typing.Annotated[pl.Array, pl.Int64(), 2]

            class Child(Parent):
                extra: int
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert "v" in child.definition_errors

    def test_child_override_with_legal_annotation_repairs(self):
        # Probed: re-declaring the field with the full-arity form shadows
        # the broken parent annotation — the child schema builds fine.
        registry = _collect(
            """
            import typing
            import polars as pl
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                v: typing.Annotated[pl.Array, pl.Int64(), 2]

            class Child(Parent):
                v: typing.Annotated[pl.Array, pl.Int64(), 2, None]
            """
        )
        parent = registry.get("Parent")
        child = registry.get("Child")
        assert parent is not None and "v" in parent.definition_errors
        assert child is not None
        assert child.definition_errors == {}


class TestUnrecognizedAnnotationDegrade:
    """Issue #77: a field annotation the parser cannot translate must NOT
    silently vanish from the schema (phantom "extra column" FPs on strict
    schemas, vanished-column FNs on open ones). The column registers with
    Unknown dtype and the schema records a per-field definition warning
    that the analyzer surfaces as pplw-unrecognized-annotation on every referencing function.

    Probed (pandera 0.31.1): a genuinely-unresolvable annotation makes
    pandera raise TypeError at FIRST USE (to_schema/validate), not at the
    class statement — but a bare name may equally be a runtime alias of a
    real dtype (``MyAlias = pl.Int64`` resolves fine), so this is a
    warning, not a provably-broken pple-broken-schema-annotation error.
    """

    def test_unrecognized_field_registers_unknown_column(self):
        from polypolarism.types import Unknown

        registry = _collect(
            """
            import pandera.polars as pa

            class CustomThing:
                pass

            class S(pa.DataFrameModel):
                a: int
                mystery: CustomThing
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["a"] == ColumnSpec(Int64(), required=True)
        assert schema.columns["mystery"] == ColumnSpec(Unknown(), required=True)
        assert "mystery" in schema.definition_warnings
        assert "CustomThing" in schema.definition_warnings["mystery"]

    def test_optional_unrecognized_field_is_not_required(self):
        from polypolarism.types import Unknown

        registry = _collect(
            """
            import typing
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                mystery: typing.Optional[CustomThing]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.columns["mystery"] == ColumnSpec(Unknown(), required=False)
        assert "mystery" in schema.definition_warnings

    def test_legal_schema_has_no_definition_warnings(self):
        registry = _collect(
            """
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                a: int
                b: pl.Utf8
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.definition_warnings == {}

    def test_child_inherits_definition_warning(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                mystery: CustomThing

            class Child(Parent):
                extra: int
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert "mystery" in child.definition_warnings
        assert "mystery" in child.columns

    def test_child_override_with_recognized_annotation_clears_warning(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class Parent(pa.DataFrameModel):
                mystery: CustomThing

            class Child(Parent):
                mystery: int
            """
        )
        child = registry.get("Child")
        assert child is not None
        assert child.definition_warnings == {}
        assert child.columns["mystery"] == ColumnSpec(Int64(), required=True)

    def test_arity_broken_field_is_not_double_reported(self):
        # A wrong-arity ``Annotated`` form already carries the pple-broken-schema-annotation
        # verdict (issue #69) — it must not ALSO get a pplw-unrecognized-annotation warning.
        registry = _collect(
            """
            import typing
            import polars as pl
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                v: typing.Annotated[pl.Array, pl.Int64(), 2]
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert "v" in schema.definition_errors
        assert schema.definition_warnings == {}


class TestObjectApiSchemas:
    """Backlog C-11 tier 1: module-level ``pa.DataFrameSchema({...})``
    object-API assignments register like class schemas, keyed by the
    variable name."""

    def test_basic_registration(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            order_schema = pa.DataFrameSchema(
                {
                    "order_id": pa.Column(int),
                    "amount": pa.Column(pl.Float64, nullable=True),
                    "note": pa.Column(str, required=False),
                }
            )
            """
        )
        schema = registry.get("order_schema")
        assert schema is not None
        assert schema.columns["order_id"] == ColumnSpec(Int64(), required=True)
        assert schema.columns["amount"] == ColumnSpec(Nullable(Float64()), required=True)
        assert schema.columns["note"] == ColumnSpec(Utf8(), required=False)
        assert schema.strict is False and schema.coerce is False

    def test_strict_and_coerce_kwargs(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int)}, strict=True, coerce=True)
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert schema.strict is True and schema.coerce is True
        # strict=True object schema binds closed (no checked-island mark).
        assert schema.to_frame_type().nonstrict_schema is None

    def test_nonstrict_object_schema_is_checked_island(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int)})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert schema.to_frame_type().nonstrict_schema == "s"

    def test_parametrized_dtypes_match_class_form(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            s = pa.DataFrameSchema(
                {
                    "t": pa.Column(pl.Datetime("ms", "UTC")),
                    "d": pa.Column(pl.Decimal),
                    "e": pa.Column(pl.Enum(["a", "b"])),
                }
            )
            """
        )
        schema = registry.get("s")
        assert schema is not None
        cols = schema.columns
        assert cols["t"].dtype == Datetime(tz="UTC", unit="ms")
        # Probed: pandera resolves the bare class through its engine
        # default (28, 0) in Column dtype position too.
        assert cols["d"].dtype == Decimal(28, 0)
        assert cols["e"].dtype == Enum(categories=("a", "b"))

    def test_string_dtype_degrades_loudly(self):
        # Probed: pandera accepts "int64" string aliases; polypolarism
        # does not model them — the column registers as Unknown with a
        # definition warning (pplw-unrecognized-annotation channel), not silently dropped.
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column("int64")})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert isinstance(schema.columns["a"].dtype, Unknown)
        assert "a" in schema.definition_warnings

    def test_non_literal_kwarg_degrades_loudly(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int, nullable=flag)})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert isinstance(schema.columns["a"].dtype, Unknown)
        assert "a" in schema.definition_warnings

    def test_rebinding_to_non_schema_unregisters(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int)})
            s = 42
            """
        )
        assert registry.get("s") is None


class TestObjectApiConstruction:
    """Backlog C-11 tier 2: constant-foldable construction — dict
    comprehensions over literal/const string lists, ``**`` spreads of
    module-level column dicts, and add_columns/remove_columns
    derivation."""

    def test_dict_comprehension_over_literal_list(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            s = pa.DataFrameSchema({c: pa.Column(pl.Float64) for c in ["x", "y", "z"]})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        cols = schema.columns
        assert set(cols) == {"x", "y", "z"}
        assert all(spec.dtype == Float64() for spec in cols.values())

    def test_dict_comprehension_over_module_const(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            METRICS = ["clicks", "views"]

            s = pa.DataFrameSchema({c: pa.Column(int) for c in METRICS})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert set(schema.columns) == {"clicks", "views"}

    def test_comprehension_value_referencing_loopvar_degrades(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema({c: pa.Column(int, alias=c) for c in ["x", "y"]})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert set(schema.columns) == {"x", "y"}
        assert all(isinstance(spec.dtype, Unknown) for spec in schema.columns.values())
        assert "x" in schema.definition_warnings

    def test_spread_of_module_column_dict(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            COMMON = {"id": pa.Column(int)}

            s = pa.DataFrameSchema({**COMMON, "price": pa.Column(pl.Float64)})
            """
        )
        schema = registry.get("s")
        assert schema is not None
        cols = schema.columns
        assert set(cols) == {"id", "price"}
        assert cols["id"].dtype == Int64()

    def test_direct_name_argument(self):
        registry = _collect(
            """
            import pandera.polars as pa

            COLS = {"id": pa.Column(int)}

            s = pa.DataFrameSchema(COLS, strict=True)
            """
        )
        schema = registry.get("s")
        assert schema is not None
        assert set(schema.columns) == {"id"}
        assert schema.strict is True

    def test_add_columns_derivation(self):
        registry = _collect(
            """
            import pandera.polars as pa
            import polars as pl

            base = pa.DataFrameSchema({"a": pa.Column(int)}, strict=True, coerce=True)
            wide = base.add_columns({"b": pa.Column(pl.Float64)})
            """
        )
        base = registry.get("base")
        wide = registry.get("wide")
        assert base is not None and wide is not None
        assert set(base.columns) == {"a"}  # immutable derivation
        assert set(wide.columns) == {"a", "b"}
        assert wide.strict is True and wide.coerce is True

    def test_remove_columns_derivation(self):
        registry = _collect(
            """
            import pandera.polars as pa

            base = pa.DataFrameSchema({"a": pa.Column(int), "b": pa.Column(str)})
            narrow = base.remove_columns(["a"])
            """
        )
        narrow = registry.get("narrow")
        assert narrow is not None
        assert set(narrow.columns) == {"b"}


class TestUnresolvedObjectSchemaDerivations:
    """Issue #90: a derivation polypolarism cannot fold (non-literal
    remove_columns, update_columns/rename_columns, unreadable
    DataFrameSchema columns) must not silently unregister the schema —
    it registers as UNRESOLVED: validate still narrows (to a fully open
    assumption frame) and pplw-unrecognized-annotation surfaces the degrade."""

    def test_nonliteral_remove_columns_registers_unresolved(self):
        registry = _collect(
            """
            import pandera.polars as pa

            base = pa.DataFrameSchema({"a": pa.Column(int), "b": pa.Column(str)})
            narrow = base.remove_columns(cols_var)
            """
        )
        schema = registry.get("narrow")
        assert schema is not None
        assert schema.unresolved
        assert schema.definition_warnings
        ft = schema.to_frame_type()
        assert ft.rest is not None and ft.columns == {}
        assert ft.nonstrict_schema is None  # pure assumption, no island lint

    def test_update_columns_registers_unresolved(self):
        registry = _collect(
            """
            import pandera.polars as pa

            base = pa.DataFrameSchema({"a": pa.Column(int)})
            tweaked = base.update_columns({"a": {"nullable": True}})
            """
        )
        schema = registry.get("tweaked")
        assert schema is not None and schema.unresolved

    def test_unreadable_dataframeschema_columns_registers_unresolved(self):
        registry = _collect(
            """
            import pandera.polars as pa

            s = pa.DataFrameSchema(make_columns())
            """
        )
        schema = registry.get("s")
        assert schema is not None and schema.unresolved


class TestSchemaSourceProvenance:
    """Batch B, Request 2: each Schema records the absolute path of the file
    that DEFINES its class (``source_file``) and the class-header line
    (``header_line``), for the pple-undeclared-column "declare the column" quick fix.

    ``collect_schemas`` takes an optional ``source_file``; the cross-file
    import path (``collect_schemas_with_imports``) populates each schema with
    the path of the module it was actually parsed from."""

    def test_collect_schemas_records_source_file_and_header(self, tmp_path):
        import ast as _ast

        from polypolarism.pandera_schema import collect_schemas

        src = textwrap.dedent(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int
                name: str
            """
        ).lstrip("\n")
        path = tmp_path / "mod.py"
        path.write_text(src)
        registry = collect_schemas(_ast.parse(src), source_file=path)
        schema = registry.get("S")
        assert schema is not None
        assert schema.source_file == str(path)
        # ``class S`` is on line 3 (1-indexed).
        assert schema.header_line == 3

    def test_no_source_file_leaves_provenance_none(self):
        registry = _collect(
            """
            import pandera.polars as pa

            class S(pa.DataFrameModel):
                id: int
            """
        )
        schema = registry.get("S")
        assert schema is not None
        assert schema.source_file is None

    def test_cross_file_import_records_defining_module(self, tmp_path):
        import ast as _ast

        from polypolarism.pandera_schema import collect_schemas_with_imports

        (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
        schemas_py = tmp_path / "schemas.py"
        schemas_py.write_text(
            textwrap.dedent(
                """
                import pandera.polars as pa


                class Users(pa.DataFrameModel):
                    user_id: int

                    class Config:
                        strict = False
                """
            ).lstrip("\n")
        )
        app_py = tmp_path / "app.py"
        app_src = textwrap.dedent(
            """
            from pandera.typing.polars import DataFrame
            from schemas import Users
            """
        ).lstrip("\n")
        app_py.write_text(app_src)
        registry = collect_schemas_with_imports(_ast.parse(app_src), app_py)
        users = registry.get("Users")
        assert users is not None
        # The defining file is schemas.py, NOT app.py.
        assert users.source_file == str(schemas_py.resolve())
