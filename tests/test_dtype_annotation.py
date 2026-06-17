"""Tests for ``render_dtype_annotation`` — the dtype -> pandera annotation
string renderer (Batch B, Request 3).

The renderer produces a ready-to-insert pandera class-based annotation string
for a ``DataType`` (e.g. ``"pl.Float64"``, ``"pl.List(pl.Int64)"``). It returns
``None`` for dtypes that cannot be rendered soundly (Unknown / exotic) so the
JSON layer can OMIT the field rather than emit a guess.

Nullability is NOT part of the dtype annotation (pandera declares it via
``pa.Field(nullable=True)``), so ``Nullable(inner)`` renders the inner.
"""

from __future__ import annotations

from polypolarism.types import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Null,
    Nullable,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    Utf8,
    render_dtype_annotation,
)


class TestNumericWidths:
    def test_all_integer_widths(self):
        assert render_dtype_annotation(Int8()) == "pl.Int8"
        assert render_dtype_annotation(Int16()) == "pl.Int16"
        assert render_dtype_annotation(Int32()) == "pl.Int32"
        assert render_dtype_annotation(Int64()) == "pl.Int64"
        assert render_dtype_annotation(Int128()) == "pl.Int128"
        assert render_dtype_annotation(UInt8()) == "pl.UInt8"
        assert render_dtype_annotation(UInt16()) == "pl.UInt16"
        assert render_dtype_annotation(UInt32()) == "pl.UInt32"
        assert render_dtype_annotation(UInt64()) == "pl.UInt64"
        assert render_dtype_annotation(UInt128()) == "pl.UInt128"

    def test_float_widths(self):
        assert render_dtype_annotation(Float16()) == "pl.Float16"
        assert render_dtype_annotation(Float32()) == "pl.Float32"
        assert render_dtype_annotation(Float64()) == "pl.Float64"


class TestScalars:
    def test_string_and_bool_and_binary(self):
        assert render_dtype_annotation(Utf8()) == "pl.Utf8"
        assert render_dtype_annotation(Boolean()) == "pl.Boolean"
        assert render_dtype_annotation(Binary()) == "pl.Binary"

    def test_temporal_simple(self):
        assert render_dtype_annotation(Date()) == "pl.Date"
        assert render_dtype_annotation(Time()) == "pl.Time"

    def test_categorical(self):
        assert render_dtype_annotation(Categorical()) == "pl.Categorical"


class TestParametrizedScalars:
    def test_datetime_unit_only(self):
        assert render_dtype_annotation(Datetime(unit="us")) == 'pl.Datetime("us")'
        assert render_dtype_annotation(Datetime(unit="ns")) == 'pl.Datetime("ns")'

    def test_datetime_unit_and_tz(self):
        assert render_dtype_annotation(Datetime(tz="UTC", unit="us")) == 'pl.Datetime("us", "UTC")'

    def test_duration(self):
        assert render_dtype_annotation(Duration(unit="ms")) == 'pl.Duration("ms")'

    def test_decimal(self):
        assert render_dtype_annotation(Decimal(12, 4)) == "pl.Decimal(12, 4)"


class TestContainers:
    def test_list(self):
        assert render_dtype_annotation(List(Int64())) == "pl.List(pl.Int64)"

    def test_nested_list(self):
        assert render_dtype_annotation(List(List(Utf8()))) == "pl.List(pl.List(pl.Utf8))"

    def test_array_with_width(self):
        assert render_dtype_annotation(Array(Int64(), 3)) == "pl.Array(pl.Int64, 3)"

    def test_array_without_width_is_unrenderable(self):
        # A width-less Array has no faithful call form (pandera requires the
        # width); omit rather than guess.
        assert render_dtype_annotation(Array(Int64(), None)) is None

    def test_struct(self):
        rendered = render_dtype_annotation(Struct({"a": Utf8(), "b": Float64()}))
        assert rendered == 'pl.Struct({"a": pl.Utf8, "b": pl.Float64})'

    def test_empty_closed_struct(self):
        assert render_dtype_annotation(Struct({})) == "pl.Struct({})"


class TestEnum:
    def test_enum_with_categories(self):
        assert render_dtype_annotation(Enum(("a", "b"))) == 'pl.Enum(["a", "b"])'

    def test_bare_enum_unrenderable(self):
        # Unknown categories — cannot render a faithful call form.
        assert render_dtype_annotation(Enum(None)) is None


class TestNullableRendersInner:
    def test_nullable_scalar(self):
        assert render_dtype_annotation(Nullable(Int64())) == "pl.Int64"

    def test_nullable_container(self):
        assert render_dtype_annotation(Nullable(List(Int64()))) == "pl.List(pl.Int64)"


class TestUnrenderable:
    def test_unknown_returns_none(self):
        assert render_dtype_annotation(Unknown()) is None

    def test_null_returns_none(self):
        # The Null dtype is a null-literal sentinel, not a declarable column
        # dtype — omit rather than emit ``pl.Null``.
        assert render_dtype_annotation(Null()) is None

    def test_open_struct_returns_none(self):
        # An open struct (unknown extra fields) has no faithful closed form.
        assert render_dtype_annotation(Struct({"a": Int64()}, open=True)) is None

    def test_container_with_unrenderable_inner_returns_none(self):
        assert render_dtype_annotation(List(Unknown())) is None
        assert render_dtype_annotation(Array(Unknown(), 3)) is None

    def test_struct_with_unrenderable_field_returns_none(self):
        assert render_dtype_annotation(Struct({"a": Unknown()})) is None
