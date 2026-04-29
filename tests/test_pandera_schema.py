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

    def test_unknown_schema_returns_none(self):
        registry = _collect("")
        assert registry.to_frame_type("Unknown") is None
