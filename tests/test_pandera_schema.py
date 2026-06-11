"""Tests for pandera_schema.py — schema class registry."""

from __future__ import annotations

import ast
import textwrap

from polypolarism.pandera_schema import collect_schemas
from polypolarism.types import (
    ColumnSpec,
    Float64,
    Int64,
    List,
    Nullable,
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
    that the analyzer surfaces as PLW011 on every referencing function.

    Probed (pandera 0.31.1): a genuinely-unresolvable annotation makes
    pandera raise TypeError at FIRST USE (to_schema/validate), not at the
    class statement — but a bare name may equally be a runtime alias of a
    real dtype (``MyAlias = pl.Int64`` resolves fine), so this is a
    warning, not a provably-broken PLY041 error.
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
        # A wrong-arity ``Annotated`` form already carries the PLY041
        # verdict (issue #69) — it must not ALSO get a PLW011 warning.
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
