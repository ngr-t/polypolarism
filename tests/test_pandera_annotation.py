"""Tests for pandera_annotation.py — DataFrame[Schema] detection."""

from __future__ import annotations

import ast
import textwrap

from polypolarism.pandera_annotation import extract_dataframe_annotation
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


class TestForwardReferences:
    def test_string_forward_ref(self):
        registry = _registry_with_schema()
        ft = extract_dataframe_annotation(
            _annotation_node('DataFrame["MySchema"]'),
            registry,
        )
        assert ft is not None
        assert "id" in ft.columns


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
