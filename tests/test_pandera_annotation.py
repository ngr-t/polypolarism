"""Tests for pandera_annotation.py — DataFrame[Schema] detection."""

from __future__ import annotations

import ast
import textwrap

from polypolarism.pandera_annotation import (
    extract_dataframe_annotation,
    frame_annotation_schema_name,
)
from polypolarism.pandera_schema import collect_schemas
from polypolarism.types import ColumnSpec, Int64, Utf8


def _registry_with_schema():
    src = textwrap.dedent(
        """
        import pandera.polars as pa

        class MySchema(pa.DataFrameModel):
            id: int
            name: str
        """
    )
    return collect_schemas(ast.parse(src))


def _annotation_node(src: str) -> ast.expr:
    """Parse ``x: <src>`` and return the annotation AST node."""
    tree = ast.parse(f"x: {src}")
    stmt = tree.body[0]
    assert isinstance(stmt, ast.AnnAssign)
    return stmt.annotation


class TestSimpleAnnotations:
    def test_dataframe_schema(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("DataFrame[MySchema]"), registry)
        assert ft is not None
        assert ft.columns == {
            "id": ColumnSpec(Int64(), required=True),
            "name": ColumnSpec(Utf8(), required=True),
        }

    def test_lazyframe_schema(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("LazyFrame[MySchema]"), registry)
        assert ft is not None
        assert "id" in ft.columns

    def test_qualified_pa_dataframe(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("pa.DataFrame[MySchema]"), registry)
        assert ft is not None
        assert "id" in ft.columns

    def test_deeply_qualified_dataframe(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(
            _annotation_node("pandera.typing.polars.DataFrame[MySchema]"),
            registry,
        )
        assert ft is not None
        assert "id" in ft.columns


class TestModelAttributeAnnotation:
    """Patito's bare ``Model.DataFrame`` / ``Model.LazyFrame`` form (ADR-0010 #2)."""

    def test_model_dataframe_attribute_resolves(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("MySchema.DataFrame"), registry)
        assert ft is not None
        assert "id" in ft.columns
        assert ft.is_lazy is False

    def test_model_lazyframe_attribute_resolves(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("MySchema.LazyFrame"), registry)
        assert ft is not None
        assert ft.is_lazy is True

    def test_pl_dataframe_attribute_is_not_a_schema_frame(self):
        # ``pl.DataFrame`` (qualifier not a registered schema) must NOT resolve
        # here — it is a bare-frame annotation handled elsewhere.
        registry = _registry_with_schema()
        assert extract_dataframe_annotation(_annotation_node("pl.DataFrame"), registry) is None

    def test_unknown_model_attribute_does_not_resolve(self):
        registry = _registry_with_schema()
        assert extract_dataframe_annotation(_annotation_node("Unknown.DataFrame"), registry) is None


class TestForwardReferences:
    def test_string_forward_ref(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(
            _annotation_node('DataFrame["MySchema"]'),
            registry,
        )
        assert ft is not None
        assert "id" in ft.columns


class TestModuleQualifiedSchema:
    """``DataFrame[mod.Schema]`` — module-qualified schema references (issue #68)."""

    def test_schema_name_of_qualified_reference(self):
        name = frame_annotation_schema_name(_annotation_node("DataFrame[mod.MySchema]"))
        assert name == "mod.MySchema"

    def test_schema_name_of_deeply_qualified_reference(self):
        name = frame_annotation_schema_name(_annotation_node("LazyFrame[pkg.schemas.Out]"))
        assert name == "pkg.schemas.Out"

    def test_schema_name_with_qualified_head_and_slice(self):
        name = frame_annotation_schema_name(_annotation_node("pa.DataFrame[mod.MySchema]"))
        assert name == "mod.MySchema"

    def test_schema_name_of_call_slice_is_none(self):
        # Only Name/Attribute chains qualify — a call in the chain does not.
        name = frame_annotation_schema_name(_annotation_node("DataFrame[get_mod().MySchema]"))
        assert name is None

    def test_extract_resolves_dotted_registry_key(self):
        # The registry is flat; qualified imports register schemas under
        # their dotted spelling (see pandera_schema._merge_module_imports).
        registry = _registry_with_schema()
        registry.schemas["mod.MySchema"] = registry.schemas["MySchema"]
        ft = extract_dataframe_annotation(_annotation_node("DataFrame[mod.MySchema]"), registry)
        assert ft is not None
        assert "id" in ft.columns

    def test_extract_unknown_qualified_returns_none(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("DataFrame[mod.Unknown]"), registry)
        assert ft is None


class TestNonMatching:
    def test_unknown_schema_returns_none(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(
            _annotation_node("DataFrame[OtherSchema]"),
            registry,
        )
        assert ft is None

    def test_non_subscript_returns_none(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("int"), registry)
        assert ft is None

    def test_wrong_head_returns_none(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(_annotation_node("List[int]"), registry)
        assert ft is None
