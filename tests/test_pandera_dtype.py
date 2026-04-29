"""Tests for pandera_dtype.py — AST → ColumnSpec translator."""

from __future__ import annotations

import ast

from polypolarism.pandera_dtype import parse_field_annotation
from polypolarism.types import (
    Boolean,
    ColumnSpec,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    Utf8,
)


def _parse(annotation_src: str, value_src: str | None = None) -> ColumnSpec | None:
    """Helper: build a class with a single field and parse it."""
    class_src = f"class S:\n    x: {annotation_src}"
    if value_src is not None:
        class_src += f" = {value_src}"
    tree = ast.parse(class_src)
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)
    stmt = cls.body[0]
    assert isinstance(stmt, ast.AnnAssign)
    return parse_field_annotation(stmt.annotation, stmt.value)


class TestPythonBuiltins:
    def test_int(self):
        assert _parse("int") == ColumnSpec(Int64(), required=True)

    def test_str(self):
        assert _parse("str") == ColumnSpec(Utf8(), required=True)

    def test_float(self):
        assert _parse("float") == ColumnSpec(Float64(), required=True)

    def test_bool(self):
        assert _parse("bool") == ColumnSpec(Boolean(), required=True)


class TestPolarsDtype:
    def test_pl_int64_bare(self):
        assert _parse("pl.Int64") == ColumnSpec(Int64(), required=True)

    def test_pl_int64_call(self):
        assert _parse("pl.Int64()") == ColumnSpec(Int64(), required=True)

    def test_pl_utf8(self):
        assert _parse("pl.Utf8") == ColumnSpec(Utf8(), required=True)

    def test_pl_string(self):
        assert _parse("pl.String") == ColumnSpec(Utf8(), required=True)

    def test_pl_float64(self):
        assert _parse("pl.Float64") == ColumnSpec(Float64(), required=True)


class TestOptional:
    def test_optional_int(self):
        assert _parse("Optional[int]") == ColumnSpec(Int64(), required=False)

    def test_optional_pl_int64(self):
        assert _parse("Optional[pl.Int64]") == ColumnSpec(Int64(), required=False)

    def test_typing_optional_qualified(self):
        assert _parse("typing.Optional[int]") == ColumnSpec(Int64(), required=False)

    def test_pep604_union_none(self):
        assert _parse("int | None") == ColumnSpec(Int64(), required=False)

    def test_pep604_union_none_reversed(self):
        assert _parse("None | int") == ColumnSpec(Int64(), required=False)


class TestFieldNullable:
    def test_field_nullable_true(self):
        assert _parse("int", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Int64()), required=True
        )

    def test_field_nullable_unqualified(self):
        assert _parse("str", "Field(nullable=True)") == ColumnSpec(Nullable(Utf8()), required=True)

    def test_field_without_nullable(self):
        assert _parse("int", "pa.Field(unique=True)") == ColumnSpec(Int64(), required=True)

    def test_optional_with_nullable_field(self):
        # Optional[T] (column may be absent) AND nullable values
        assert _parse("Optional[int]", "pa.Field(nullable=True)") == ColumnSpec(
            Nullable(Int64()), required=False
        )

    def test_field_with_value_constraints_ignored(self):
        # Value constraints are runtime-only, ignored statically
        assert _parse("int", "pa.Field(gt=0, lt=100, unique=True)") == ColumnSpec(
            Int64(), required=True
        )


class TestAnnotatedList:
    def test_annotated_list_int(self):
        assert _parse("Annotated[pl.List, pl.Int64()]") == ColumnSpec(List(Int64()), required=True)

    def test_annotated_list_utf8(self):
        assert _parse("Annotated[pl.List, pl.Utf8()]") == ColumnSpec(List(Utf8()), required=True)

    def test_annotated_array_treated_as_list(self):
        # Width is ignored for our purposes
        assert _parse("Annotated[pl.Array, pl.Int64(), 3]") == ColumnSpec(
            List(Int64()), required=True
        )


class TestAnnotatedStruct:
    def test_annotated_struct(self):
        result = _parse('Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]')
        assert result == ColumnSpec(Struct({"a": Utf8(), "b": Float64()}), required=True)

    def test_annotated_struct_optional(self):
        result = _parse('Optional[Annotated[pl.Struct, {"a": pl.Utf8()}]]')
        assert result == ColumnSpec(Struct({"a": Utf8()}), required=False)


class TestUnknown:
    def test_unknown_name_returns_none(self):
        assert _parse("MyCustom") is None

    def test_unknown_pl_attr_returns_none(self):
        assert _parse("pl.NotARealType") is None

    def test_other_module_returns_none(self):
        assert _parse("np.int64") is None
