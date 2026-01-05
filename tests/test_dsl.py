"""Tests for DSL parser."""

import pytest

from polypolarism.dsl import parse_schema, parse_type, ParseError
from polypolarism.types import (
    Int64,
    Int32,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Utf8,
    Boolean,
    Date,
    Datetime,
    Duration,
    Decimal,
    Categorical,
    List,
    Struct,
    Nullable,
    FrameType,
)


class TestParseType:
    """Test parsing individual type expressions."""

    def test_parse_int64(self):
        assert parse_type("Int64") == Int64()

    def test_parse_int32(self):
        assert parse_type("Int32") == Int32()

    def test_parse_uint32(self):
        assert parse_type("UInt32") == UInt32()

    def test_parse_uint64(self):
        assert parse_type("UInt64") == UInt64()

    def test_parse_float32(self):
        assert parse_type("Float32") == Float32()

    def test_parse_float64(self):
        assert parse_type("Float64") == Float64()

    def test_parse_utf8(self):
        assert parse_type("Utf8") == Utf8()

    def test_parse_string_as_utf8_alias(self):
        """String is an alias for Utf8 (Polars 1.x compatibility)."""
        assert parse_type("String") == Utf8()

    def test_parse_boolean(self):
        assert parse_type("Boolean") == Boolean()

    def test_parse_date(self):
        assert parse_type("Date") == Date()

    def test_parse_datetime(self):
        assert parse_type("Datetime") == Datetime()

    def test_parse_duration(self):
        assert parse_type("Duration") == Duration()

    def test_parse_categorical(self):
        assert parse_type("Categorical") == Categorical()


class TestParseNullable:
    """Test parsing nullable types."""

    def test_parse_nullable_int64(self):
        assert parse_type("Int64?") == Nullable(Int64())

    def test_parse_nullable_utf8(self):
        assert parse_type("Utf8?") == Nullable(Utf8())

    def test_parse_nullable_float64(self):
        assert parse_type("Float64?") == Nullable(Float64())


class TestParseList:
    """Test parsing List types."""

    def test_parse_list_int64(self):
        assert parse_type("List[Int64]") == List(Int64())

    def test_parse_list_utf8(self):
        assert parse_type("List[Utf8]") == List(Utf8())

    def test_parse_nested_list(self):
        assert parse_type("List[List[Int64]]") == List(List(Int64()))

    def test_parse_nullable_list(self):
        assert parse_type("List[Int64]?") == Nullable(List(Int64()))

    def test_parse_list_of_nullable(self):
        assert parse_type("List[Int64?]") == List(Nullable(Int64()))


class TestParseStruct:
    """Test parsing Struct types."""

    def test_parse_simple_struct(self):
        result = parse_type("Struct{a: Int64, b: Utf8}")
        expected = Struct({"a": Int64(), "b": Utf8()})
        assert result == expected

    def test_parse_struct_single_field(self):
        result = parse_type("Struct{name: Utf8}")
        expected = Struct({"name": Utf8()})
        assert result == expected

    def test_parse_struct_with_nullable_field(self):
        result = parse_type("Struct{id: Int64, value: Float64?}")
        expected = Struct({"id": Int64(), "value": Nullable(Float64())})
        assert result == expected

    def test_parse_nullable_struct(self):
        result = parse_type("Struct{a: Int64}?")
        expected = Nullable(Struct({"a": Int64()}))
        assert result == expected


class TestParseSchema:
    """Test parsing full schema strings."""

    def test_parse_empty_schema(self):
        result = parse_schema("{}")
        assert result == FrameType({})

    def test_parse_single_column(self):
        result = parse_schema("{id: Int64}")
        expected = FrameType({"id": Int64()})
        assert result == expected

    def test_parse_multiple_columns(self):
        result = parse_schema("{id: Int64, name: Utf8, score: Float64}")
        expected = FrameType({
            "id": Int64(),
            "name": Utf8(),
            "score": Float64(),
        })
        assert result == expected

    def test_parse_schema_with_nullable(self):
        result = parse_schema("{id: Int64, value: Float64?}")
        expected = FrameType({
            "id": Int64(),
            "value": Nullable(Float64()),
        })
        assert result == expected

    def test_parse_schema_with_list(self):
        result = parse_schema("{id: Int64, tags: List[Utf8]}")
        expected = FrameType({
            "id": Int64(),
            "tags": List(Utf8()),
        })
        assert result == expected

    def test_parse_schema_with_struct(self):
        result = parse_schema("{id: Int64, address: Struct{city: Utf8, zip: Int64}}")
        expected = FrameType({
            "id": Int64(),
            "address": Struct({"city": Utf8(), "zip": Int64()}),
        })
        assert result == expected

    def test_parse_schema_whitespace_tolerance(self):
        result = parse_schema("{  id :  Int64 ,  name : Utf8  }")
        expected = FrameType({"id": Int64(), "name": Utf8()})
        assert result == expected


class TestParseErrors:
    """Test error handling in parser."""

    def test_unknown_type_raises_error(self):
        with pytest.raises(ParseError, match="Unknown type"):
            parse_type("UnknownType")

    def test_invalid_schema_format(self):
        with pytest.raises(ParseError):
            parse_schema("not a schema")

    def test_unclosed_brace(self):
        with pytest.raises(ParseError):
            parse_schema("{id: Int64")
