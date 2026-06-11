"""Tests for types module."""

from polypolarism.types import (
    Array,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float64,
    FrameType,
    Int32,
    Int64,
    List,
    Nullable,
    Struct,
    Time,
    Unknown,
    Utf8,
    unwrap_nullable,
    wrap_nullable,
)


class TestDataTypes:
    """Test basic DataType classes."""

    def test_int64_is_datatype(self):
        assert isinstance(Int64(), DataType)

    def test_int64_equality(self):
        assert Int64() == Int64()

    def test_int64_not_equal_to_int32(self):
        assert Int64() != Int32()

    def test_float64_is_datatype(self):
        assert isinstance(Float64(), DataType)

    def test_utf8_is_datatype(self):
        assert isinstance(Utf8(), DataType)

    def test_boolean_is_datatype(self):
        assert isinstance(Boolean(), DataType)

    def test_date_is_datatype(self):
        assert isinstance(Date(), DataType)

    def test_datetime_is_datatype(self):
        assert isinstance(Datetime(), DataType)

    def test_datetime_with_timezone(self):
        dt = Datetime(tz="UTC")
        assert dt.tz == "UTC"

    def test_duration_is_datatype(self):
        assert isinstance(Duration(), DataType)

    def test_decimal_is_datatype(self):
        dec = Decimal(precision=10, scale=2)
        assert isinstance(dec, DataType)
        assert dec.precision == 10
        assert dec.scale == 2

    def test_categorical_is_datatype(self):
        assert isinstance(Categorical(), DataType)

    def test_time_is_datatype(self):
        assert isinstance(Time(), DataType)

    def test_time_equality(self):
        assert Time() == Time()

    def test_time_not_equal_to_date_or_datetime(self):
        assert Time() != Date()
        assert Time() != Datetime()

    def test_time_str(self):
        assert str(Time()) == "Time"

    def test_time_hashable(self):
        assert hash(Time()) == hash(Time())
        assert len({Time(), Time()}) == 1


class TestNullable:
    """Test Nullable wrapper."""

    def test_nullable_wraps_type(self):
        nullable_int = Nullable(Int64())
        assert isinstance(nullable_int, DataType)
        assert nullable_int.inner == Int64()

    def test_nullable_equality(self):
        assert Nullable(Int64()) == Nullable(Int64())
        assert Nullable(Int64()) != Nullable(Utf8())

    def test_nullable_not_equal_to_non_nullable(self):
        assert Nullable(Int64()) != Int64()


class TestListType:
    """Test List type."""

    def test_list_of_int64(self):
        list_type = List(Int64())
        assert isinstance(list_type, DataType)
        assert list_type.inner == Int64()

    def test_list_equality(self):
        assert List(Int64()) == List(Int64())
        assert List(Int64()) != List(Utf8())

    def test_nested_list(self):
        nested = List(List(Int64()))
        assert nested.inner == List(Int64())


class TestArrayType:
    """Test Array type (issue #53: distinct from List, width not tracked)."""

    def test_array_of_int64(self):
        array_type = Array(Int64())
        assert isinstance(array_type, DataType)
        assert array_type.inner == Int64()

    def test_array_equality(self):
        assert Array(Int64()) == Array(Int64())
        assert Array(Int64()) != Array(Utf8())

    def test_array_is_not_list(self):
        assert Array(Int64()) != List(Int64())
        assert List(Int64()) != Array(Int64())

    def test_array_str(self):
        assert str(Array(Int64())) == "Array[Int64]"

    def test_array_hashable(self):
        assert hash(Array(Int64())) == hash(Array(Int64()))
        assert hash(Array(Int64())) != hash(List(Int64()))

    def test_nested_containers(self):
        assert List(Array(Int64())).inner == Array(Int64())
        assert Array(List(Int64())).inner == List(Int64())


class TestUnknownType:
    """Test Unknown type (gradual-typing escape hatch)."""

    def test_unknown_is_datatype(self):
        assert isinstance(Unknown(), DataType)

    def test_unknown_equality(self):
        assert Unknown() == Unknown()

    def test_unknown_not_equal_to_other_types(self):
        assert Unknown() != Int64()
        assert Unknown() != Utf8()

    def test_unknown_str(self):
        assert str(Unknown()) == "Unknown"

    def test_unknown_hashable(self):
        assert hash(Unknown()) == hash(Unknown())


class TestStructType:
    """Test Struct type."""

    def test_struct_with_fields(self):
        struct = Struct({"name": Utf8(), "age": Int64()})
        assert isinstance(struct, DataType)
        assert struct.fields["name"] == Utf8()
        assert struct.fields["age"] == Int64()

    def test_struct_equality(self):
        s1 = Struct({"a": Int64(), "b": Utf8()})
        s2 = Struct({"a": Int64(), "b": Utf8()})
        assert s1 == s2

    def test_struct_inequality_different_fields(self):
        s1 = Struct({"a": Int64()})
        s2 = Struct({"b": Int64()})
        assert s1 != s2


class TestFrameType:
    """Test FrameType class."""

    def test_empty_frame_type(self):
        ft = FrameType({})
        assert ft.columns == {}

    def test_frame_type_with_columns(self):
        ft = FrameType(
            {
                "id": Int64(),
                "name": Utf8(),
                "score": Float64(),
            }
        )
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["name"].dtype == Utf8()
        assert ft.columns["score"].dtype == Float64()

    def test_frame_type_has_column(self):
        ft = FrameType({"id": Int64(), "name": Utf8()})
        assert ft.has_column("id")
        assert ft.has_column("name")
        assert not ft.has_column("unknown")

    def test_frame_type_get_column_type(self):
        ft = FrameType({"id": Int64(), "name": Utf8()})
        assert ft.get_column_type("id") == Int64()
        assert ft.get_column_type("name") == Utf8()
        assert ft.get_column_type("unknown") is None

    def test_frame_type_equality(self):
        ft1 = FrameType({"id": Int64(), "name": Utf8()})
        ft2 = FrameType({"id": Int64(), "name": Utf8()})
        assert ft1 == ft2

    def test_frame_type_with_nullable(self):
        ft = FrameType(
            {
                "id": Int64(),
                "score": Nullable(Float64()),
            }
        )
        assert ft.columns["score"].dtype == Nullable(Float64())

    def test_frame_type_coerce_defaults_false(self):
        ft = FrameType({"id": Int64()})
        assert ft.coerce is False

    def test_frame_type_coerce_stored(self):
        ft = FrameType({"id": Int64()}, coerce=True)
        assert ft.coerce is True

    def test_frame_type_coerce_excluded_from_eq(self):
        # Like ``is_lazy``, ``coerce`` must not participate in __eq__ so
        # shape-equality assertions keep working.
        coercing = FrameType({"id": Int64()}, coerce=True)
        plain = FrameType({"id": Int64()})
        assert coercing == plain


class TestNullableHelpers:
    """Shared unwrap/wrap helpers for the Nullable wrapper (backlog A-2)."""

    def test_unwrap_nullable_on_nullable(self):
        assert unwrap_nullable(Nullable(Int64())) == (Int64(), True)

    def test_unwrap_nullable_on_plain(self):
        assert unwrap_nullable(Int64()) == (Int64(), False)

    def test_unwrap_nullable_does_not_recurse(self):
        # Nested Nullable should not occur, but unwrap only one layer.
        assert unwrap_nullable(Nullable(Nullable(Utf8()))) == (Nullable(Utf8()), True)

    def test_wrap_nullable_true(self):
        assert wrap_nullable(Int64(), True) == Nullable(Int64())

    def test_wrap_nullable_false(self):
        assert wrap_nullable(Int64(), False) == Int64()

    def test_wrap_unwrap_roundtrip(self):
        for dtype in (Int64(), Nullable(Utf8())):
            inner, is_nullable = unwrap_nullable(dtype)
            assert wrap_nullable(inner, is_nullable) == dtype


class TestDataTypeStr:
    """Test string representation of types."""

    def test_int64_str(self):
        assert str(Int64()) == "Int64"

    def test_nullable_str(self):
        assert str(Nullable(Int64())) == "Int64?"

    def test_list_str(self):
        assert str(List(Int64())) == "List[Int64]"

    def test_struct_str(self):
        s = Struct({"a": Int64(), "b": Utf8()})
        # Order may vary, so check contains
        s_str = str(s)
        assert "Struct{" in s_str
        assert "a: Int64" in s_str
        assert "b: Utf8" in s_str
