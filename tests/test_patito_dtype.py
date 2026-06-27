"""Unit tests for the Patito field translator (ADR-0010)."""

from __future__ import annotations

import ast

from polypolarism.patito_dtype import parse_patito_field
from polypolarism.types import (
    Binary,
    Boolean,
    DataTypeGroup,
    Date,
    Enum,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    UInt16,
    Unknown,
    Utf8,
)


def _parse(annotation: str, value: str | None = None, models: frozenset[str] = frozenset()):
    ann = ast.parse(annotation, mode="eval").body
    val = ast.parse(value, mode="eval").body if value is not None else None
    return parse_patito_field(ann, val, models)


class TestScalarMappings:
    def test_int_is_integer_group_with_int64_canonical(self):
        spec, nested = _parse("int")
        assert nested is None
        assert isinstance(spec.dtype, DataTypeGroup)
        assert spec.dtype.label == "integer"
        assert spec.dtype.representative() == Int64()
        assert spec.required is True

    def test_float_is_float_group_with_float64_canonical(self):
        spec, _ = _parse("float")
        assert isinstance(spec.dtype, DataTypeGroup)
        assert spec.dtype.representative() == Float64()

    def test_str_bool_bytes_are_exact(self):
        assert _parse("str")[0].dtype == Utf8()
        assert _parse("bool")[0].dtype == Boolean()
        assert _parse("bytes")[0].dtype == Binary()

    def test_explicit_polars_dtype_is_exact(self):
        # An explicit polars dtype is precise — no acceptance group.
        assert _parse("pl.UInt16")[0].dtype == UInt16()

    def test_stdlib_temporal(self):
        assert _parse("datetime.date")[0].dtype == Date()


class TestNullability:
    def test_optional_makes_value_nullable_column_required(self):
        spec, _ = _parse("Optional[str]")
        assert spec.dtype == Nullable(Utf8())
        assert spec.required is True  # inverse of Pandera's Optional

    def test_pipe_none_makes_value_nullable(self):
        spec, _ = _parse("int | None")
        assert isinstance(spec.dtype, Nullable)
        assert isinstance(spec.dtype.inner, DataTypeGroup)
        assert spec.required is True


class TestFieldDtypeOverride:
    def test_field_dtype_forces_exact_dtype(self):
        spec, _ = _parse("int", "pt.Field(dtype=pl.UInt16)")
        assert spec.dtype == UInt16()

    def test_field_without_dtype_keeps_annotation_mapping(self):
        spec, _ = _parse("int", "pt.Field(unique=True)")
        assert isinstance(spec.dtype, DataTypeGroup)

    def test_field_dtype_override_under_optional_stays_nullable(self):
        spec, _ = _parse("Optional[int]", "pt.Field(dtype=pl.UInt16)")
        assert spec.dtype == Nullable(UInt16())


class TestLiteral:
    def test_string_literal_accepts_string_or_enum(self):
        spec, _ = _parse('Literal["dry", "cold"]')
        group = spec.dtype
        assert isinstance(group, DataTypeGroup)
        assert group.members == frozenset({Utf8(), Enum(categories=("dry", "cold"))})
        assert group.representative() == Utf8()

    def test_int_literal_is_integer_group(self):
        spec, _ = _parse("Literal[1, 2, 3]")
        assert isinstance(spec.dtype, DataTypeGroup)
        assert spec.dtype.representative() == Int64()


class TestContainersAndNested:
    def test_list_of_str(self):
        assert _parse("list[str]")[0].dtype == List(Utf8())

    def test_list_of_int_inner_is_group(self):
        spec, _ = _parse("list[int]")
        assert isinstance(spec.dtype, List)
        assert isinstance(spec.dtype.inner, DataTypeGroup)

    def test_nested_model_returns_ref_and_open_struct_placeholder(self):
        spec, nested = _parse("Inner", models=frozenset({"Inner"}))
        assert nested == "Inner"
        assert spec.dtype == Struct(open=True)

    def test_optional_nested_model_keeps_nullable_and_ref(self):
        spec, nested = _parse("Optional[Inner]", models=frozenset({"Inner"}))
        assert nested == "Inner"
        assert isinstance(spec.dtype, Nullable)


class TestUnrecognized:
    def test_unknown_annotation_degrades_to_unknown_required(self):
        spec, nested = _parse("SomeRandomAlias")
        assert spec.dtype == Unknown()
        assert spec.required is True
        assert nested is None
