"""Unit tests for the Patito schema collector (ADR-0010)."""

from __future__ import annotations

import ast

from polypolarism.pandera_schema import SchemaRegistry, collect_schemas
from polypolarism.patito_schema import collect_patito_schemas, scan_patito_imports
from polypolarism.types import (
    DataTypeGroup,
    Nullable,
    Struct,
    UInt16,
    Utf8,
)


def _registry(src: str) -> SchemaRegistry:
    return collect_schemas(ast.parse(src))


class TestScanImports:
    def test_import_patito_as_alias(self):
        tree = ast.parse("import patito as pt")
        module_aliases, model_names = scan_patito_imports(tree)
        assert module_aliases == frozenset({"pt"})
        assert model_names == frozenset()

    def test_plain_import_patito(self):
        module_aliases, _ = scan_patito_imports(ast.parse("import patito"))
        assert module_aliases == frozenset({"patito"})

    def test_from_patito_import_model(self):
        _, model_names = scan_patito_imports(ast.parse("from patito import Model"))
        assert model_names == frozenset({"Model"})

    def test_from_patito_import_model_aliased(self):
        _, model_names = scan_patito_imports(ast.parse("from patito import Model as M"))
        assert model_names == frozenset({"M"})

    def test_non_patito_import_is_ignored(self):
        module_aliases, model_names = scan_patito_imports(
            ast.parse("from pydantic import BaseModel as Model")
        )
        assert module_aliases == frozenset()
        assert model_names == frozenset()


class TestDetection:
    def test_pt_model_base_detected(self):
        reg = _registry("import patito as pt\nclass S(pt.Model):\n    a: int\n")
        assert reg.get("S") is not None

    def test_from_import_model_base_detected(self):
        reg = _registry("from patito import Model\nclass S(Model):\n    a: int\n")
        assert reg.get("S") is not None

    def test_class_model_without_patito_import_not_detected(self):
        # ADR-0009 safeguard: a bare ``class X(Model)`` with no patito import
        # must NOT be treated as a Patito schema (collision-prone name).
        reg = _registry("class Model:\n    pass\nclass S(Model):\n    a: int\n")
        assert reg.get("S") is None

    def test_no_op_when_patito_not_imported(self):
        registry = SchemaRegistry()
        collect_patito_schemas(ast.parse("class S:\n    a: int\n"), registry)
        assert registry.schemas == {}


class TestSchemaShape:
    def test_models_bind_strict(self):
        reg = _registry("import patito as pt\nclass S(pt.Model):\n    a: int\n")
        schema = reg.get("S")
        assert schema is not None
        assert schema.strict is True

    def test_field_dtype_override(self):
        src = (
            "import patito as pt\nimport polars as pl\n"
            "class S(pt.Model):\n    rank: int = pt.Field(dtype=pl.UInt16)\n"
        )
        schema = _registry(src).get("S")
        assert schema is not None
        assert schema.columns["rank"].dtype == UInt16()

    def test_optional_is_nullable_and_required(self):
        src = "import patito as pt\nclass S(pt.Model):\n    note: str | None\n"
        schema = _registry(src).get("S")
        assert schema is not None
        spec = schema.columns["note"]
        assert spec.dtype == Nullable(Utf8())
        assert spec.required is True

    def test_int_field_is_acceptance_group(self):
        reg = _registry("import patito as pt\nclass S(pt.Model):\n    a: int\n")
        schema = reg.get("S")
        assert schema is not None
        assert isinstance(schema.columns["a"].dtype, DataTypeGroup)


class TestInheritance:
    def test_child_merges_parent_columns(self):
        src = (
            "import patito as pt\n"
            "class Base(pt.Model):\n    id: int\n"
            "class Child(Base):\n    name: str\n"
        )
        child = _registry(src).get("Child")
        assert child is not None
        assert set(child.columns) == {"id", "name"}


class TestNestedStruct:
    def test_nested_model_resolves_to_struct(self):
        src = (
            "import patito as pt\n"
            "class Inner(pt.Model):\n    a: int\n    b: str\n"
            "class Outer(pt.Model):\n    id: int\n    inner: Inner\n"
        )
        outer = _registry(src).get("Outer")
        assert outer is not None
        dtype = outer.columns["inner"].dtype
        assert isinstance(dtype, Struct)
        assert set(dtype.fields) == {"a", "b"}
        # #118: a nested model is a CLOSED struct so a missing field access is
        # a provable miss.
        assert dtype.open is False

    def test_unknown_nested_model_keeps_open_struct(self):
        src = "import patito as pt\nclass Outer(pt.Model):\n    inner: Missing\n"
        outer = _registry(src).get("Outer")
        assert outer is not None
        dtype = outer.columns["inner"].dtype
        # ``Missing`` is not a patito model name, so it is not a nested ref —
        # it degrades through the leaf parser to Unknown, not a struct.
        assert not isinstance(dtype, Struct)
